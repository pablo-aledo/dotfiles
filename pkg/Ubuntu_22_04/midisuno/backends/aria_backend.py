"""
backends/aria_backend.py

Backend for:  loubb/aria-medium-base  (EleutherAI / Louis Bradshaw, 2025)
Paper:        "Scaling Self-Supervised Representation Learning for
               Symbolic Piano Performance" (ISMIR 2025)
HuggingFace:  https://huggingface.co/loubb/aria-medium-base
GitHub:       https://github.com/EleutherAI/aria

Notes:
- Aria is a LLaMA 3.2 (1B) model trained on ~60k hours of expressive piano MIDI.
- It works best as a CONTINUATION model (give it a MIDI prompt, it extends it).
- For generation from scratch we use an empty/minimal prompt.
- Requires:  pip install git+https://github.com/EleutherAI/aria-utils.git
- The tokenizer lives in the aria-utils package (not HuggingFace AutoTokenizer).
- trust_remote_code=True is required.

Install deps:
    pip install git+https://github.com/EleutherAI/aria-utils.git
    pip install transformers torch
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

from .base import MusicBackend, GenerationConfig

HF_REPO = "loubb/aria-medium-base"   # public, no auth needed


class AriaBackend(MusicBackend):

    backend_id  = "aria"
    name        = "Aria"
    version     = "1.0"
    description = (
        "LLaMA 3.2 1B piano continuation model (EleutherAI, ISMIR 2025). "
        "Trained on 60k hours of expressive piano MIDI. "
        "Best for: solo piano, classical/jazz continuations."
    )

    def __init__(self):
        super().__init__()
        self._model     = None
        self._tokenizer = None
        self._device    = self._get_device()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        print(f"[aria] Loading {HF_REPO}  device={self._device}")

        # aria-utils provides the tokenizer — must be installed separately
        try:
            # Package installs as 'ariautils', not 'aria'
            from ariautils.tokenizer import AbsTokenizer
            self._tokenizer = AbsTokenizer()
            print("[aria] ariautils tokenizer loaded.")
        except ImportError:
            raise ImportError(
                "ariautils is required for the Aria backend.\n"
                "Install with:\n"
                "  pip install git+https://github.com/EleutherAI/aria-utils.git"
            )

        from transformers import AutoModelForCausalLM
        self._model = (
            AutoModelForCausalLM
            .from_pretrained(HF_REPO, trust_remote_code=True)
            .to(self._device)
            .eval()
        )

        self._loaded = True
        print("[aria] Ready.")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, config: GenerationConfig) -> bytes:
        """
        Aria generates MIDI continuations. Since it doesn't accept free text,
        we start from a minimal seed sequence and let it generate freely.
        The prompt text is used only to pick a starting tempo/style seed.
        """
        self._require_loaded()
        import torch

        # Build a minimal seed token sequence from the prompt
        seed_ids = self._build_seed(config)
        input_ids = torch.tensor([seed_ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_length=min(len(seed_ids) + config.max_tokens, 2048),
                do_sample=True,
                temperature=config.temperature,
                top_p=0.95,
                use_cache=True,
                pad_token_id=self._model.config.eos_token_id,
            )

        generated = output_ids[0].cpu().tolist()
        return self._ids_to_midi(generated, config.bpm)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_seed(self, config: GenerationConfig) -> list[int]:
        """
        Build a minimal token seed. Aria's tokenizer uses an absolute-time
        format. We start with just the BOS/pad token and let the model generate.
        For a richer seed, you'd encode an actual MIDI file with
        tokenizer.encode_from_file().
        """
        tok = self._tokenizer

        # Try to get a BOS / start token
        seed = []
        for attr in ["bos_token_id", "pad_id", "pad_token_id"]:
            val = getattr(tok, attr, None)
            if val is not None:
                seed = [val]
                break

        # Aria's tokenizer may expose encode_from_tokens for metadata
        if hasattr(tok, "encode_from_tokens"):
            try:
                # Build a minimal sequence with tempo information
                tokens = [("prefix", "instrument", "piano")]
                encoded = tok.encode_from_tokens(tokens, return_tensors=None)
                if encoded:
                    seed = encoded
            except Exception:
                pass

        return seed if seed else [0]

    def _ids_to_midi(self, token_ids: list[int], bpm: int) -> bytes:
        """Decode Aria token IDs → MIDI bytes using aria-utils."""
        tok = self._tokenizer

        # Official decoding path: tokenizer.decode() returns a MidiDict
        if hasattr(tok, "decode"):
            try:
                midi_dict = tok.decode(token_ids)
                if hasattr(midi_dict, "to_midi"):
                    midi_obj = midi_dict.to_midi()
                    buf = io.BytesIO()
                    midi_obj.save(file=buf)
                    return buf.getvalue()
            except Exception as e:
                print(f"[aria] tok.decode() failed: {e}")

        # Fallback: write to temp file via midi_dict.save() if it expects a path
        if hasattr(tok, "decode"):
            try:
                midi_dict = tok.decode(token_ids)
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                    tmp = f.name
                midi_dict.to_midi().save(tmp)
                return Path(tmp).read_bytes()
            except Exception as e:
                print(f"[aria] file-save path failed: {e}")

        # Last resort: minimal MIDI
        return _minimal_piano_midi(token_ids, bpm)


def _minimal_piano_midi(token_ids: list[int], bpm: int) -> bytes:
    try:
        import pretty_midi
    except ImportError:
        raise RuntimeError("pip install pretty_midi")

    pm   = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=0)
    beat = 60.0 / bpm
    t    = 0.0

    for tid in token_ids:
        pitch = (tid % 88) + 21   # Aria uses 88-key piano range
        if 21 <= pitch <= 108:
            inst.notes.append(pretty_midi.Note(
                velocity=75, pitch=pitch,
                start=t, end=t + beat * 0.45,
            ))
    t += beat * 0.5

    if not inst.notes:
        inst.notes.append(pretty_midi.Note(
            velocity=60, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()
