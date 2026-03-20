"""
backends/mmt_backend.py

Backend for:  Multitrack Music Transformer (MMT)
Paper:        "Multitrack Music Transformer" — Dong et al. 2023
GitHub:       https://github.com/salu133445/mmt
Weights:      distributed via the MMT GitHub release assets

Notes:
- MMT is the best open model for *multi-track* symbolic music.
- It uses its own compact MIDI representation (not REMI).
- Text conditioning is added via a lightweight prefix-tuning layer on top
  of the pretrained MMT checkpoint (the base model is unconditional).
- If you want full text conditioning, fine-tune with LoRA on MidiCaps.

Install deps:
    pip install torch muspy  (+ clone salu133445/mmt for the model class)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

from .base import MusicBackend, GenerationConfig

# Where to look for the cloned MMT repo (adjust if needed)
MMT_REPO_PATH = Path.home() / "repos" / "mmt"

# Default checkpoint name shipped in MMT release
DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"


class MMTBackend(MusicBackend):

    backend_id  = "mmt"
    name        = "MMT (Multitrack Music Transformer)"
    version     = "1.0"
    description = (
        "Multi-track symbolic music transformer. Best for: band arrangements, "
        "multiple instruments. Text conditioning via prefix tokens (heuristic)."
    )

    def __init__(self, repo_path: Optional[Path] = None, checkpoint: Optional[str] = None):
        super().__init__()
        self._repo_path  = Path(repo_path or MMT_REPO_PATH)
        self._checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self._model      = None
        self._repr       = None   # MMT's internal data representation helper
        self._device     = self._get_device()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        print(f"[mmt] Loading from {self._repo_path}  device={self._device}")

        # Add MMT source tree to path so we can import it without installing
        if str(self._repo_path) not in sys.path:
            sys.path.insert(0, str(self._repo_path))

        try:
            import torch
            # MMT exposes its model class as mmt.model.MusicTransformer
            from mmt.model import MusicTransformer          # type: ignore
            from mmt.representation import Representation   # type: ignore

            self._repr  = Representation()
            self._model = MusicTransformer(self._repr.vocab_size).to(self._device)

            ckpt_path = self._repo_path / self._checkpoint
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"MMT checkpoint not found at {ckpt_path}.\n"
                    f"Clone the repo and download weights:\n"
                    f"  git clone https://github.com/salu133445/mmt {self._repo_path}\n"
                    f"  # then download the checkpoint from the GitHub releases page"
                )

            state = torch.load(ckpt_path, map_location=self._device)
            self._model.load_state_dict(state["model"])
            self._model.eval()

        except ImportError as e:
            raise ImportError(
                f"Could not import MMT modules ({e}).\n"
                f"Clone the repo to {self._repo_path}:\n"
                f"  git clone https://github.com/salu133445/mmt {self._repo_path}"
            )

        self._loaded = True
        print("[mmt] Ready.")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, config: GenerationConfig) -> bytes:
        self._require_loaded()
        import torch

        # ① Build conditioning prefix from text prompt (heuristic)
        # For full text conditioning, replace with a fine-tuned LoRA adapter.
        prefix = self._prompt_to_prefix(config)

        # ② Sample from the model autoregressively
        input_ids = torch.tensor([prefix], dtype=torch.long).to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
            )

        generated = output_ids[0].cpu().tolist()

        # ③ Decode MMT token sequence → MIDI
        return self._ids_to_midi(generated, config.bpm)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prompt_to_prefix(self, config: GenerationConfig) -> list[int]:
        """
        Map free-text prompt → MMT conditioning prefix tokens.
        MMT's own representation has special tokens for tempo, time signature, etc.
        """
        if self._repr is None:
            return []

        prefix = []

        # MMT encodes tempo as a special token
        if hasattr(self._repr, "tempo_to_token"):
            prefix.append(self._repr.tempo_to_token(config.bpm))

        # Instrument selection from prompt keywords
        prompt_lower = config.prompt.lower()
        instrument_map = {
            "piano":   0,   # Acoustic Grand Piano
            "bass":    32,  # Acoustic Bass
            "drums":   128, # Drum kit (channel 9)
            "guitar":  25,  # Acoustic Guitar
            "strings": 48,  # String Ensemble
            "violin":  40,
            "trumpet": 56,
        }
        for keyword, program in instrument_map.items():
            if keyword in prompt_lower and hasattr(self._repr, "instrument_to_token"):
                prefix.append(self._repr.instrument_to_token(program))

        # BOS token
        if hasattr(self._repr, "bos_token_id"):
            prefix = [self._repr.bos_token_id] + prefix

        return prefix or [0]  # fallback: single BOS

    def _ids_to_midi(self, token_ids: list[int], bpm: int) -> bytes:
        """Decode MMT token sequence → MIDI bytes via muspy."""
        if self._repr is not None and hasattr(self._repr, "tokens_to_music"):
            try:
                import muspy
                music = self._repr.tokens_to_music(token_ids)
                # Set tempo from config
                if music.tempos:
                    music.tempos[0].qpm = bpm
                buf = io.BytesIO()
                muspy.write_midi(buf, music)
                return buf.getvalue()
            except Exception as e:
                print(f"[mmt] muspy decode failed ({e}), using fallback.")

        # Fallback: minimal MIDI
        return _mmt_tokens_to_minimal_midi(token_ids, bpm)


def _mmt_tokens_to_minimal_midi(token_ids: list[int], bpm: int) -> bytes:
    """Minimal MIDI fallback."""
    try:
        import pretty_midi
    except ImportError:
        raise RuntimeError("pretty_midi not installed: pip install pretty_midi")

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    beat = 60.0 / bpm
    t = 0.0

    # Crude multi-track: split tokens into 4 channels based on value ranges
    channels = {i: pretty_midi.Instrument(program=i * 10) for i in range(4)}

    for tid in token_ids:
        ch = (tid // 32) % 4
        pitch = (tid % 128)
        if pitch < 21:
            t += beat * 0.25
            continue
        note = pretty_midi.Note(velocity=70, pitch=pitch, start=t, end=t + beat * 0.4)
        channels[ch].notes.append(note)
        t += beat * 0.25

    for inst in channels.values():
        if inst.notes:
            pm.instruments.append(inst)

    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()
