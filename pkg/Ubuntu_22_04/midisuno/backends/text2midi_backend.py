"""
backends/text2midi_backend.py  —  amaai-lab/text2midi (AAAI 2025)
"""
from __future__ import annotations
import io, pickle, importlib.util
from pathlib import Path
from typing import Optional
from .base import MusicBackend, GenerationConfig

HF_REPO = "amaai-lab/text2midi"


class Text2MidiBackend(MusicBackend):
    backend_id  = "text2midi"
    name        = "text2midi"
    version     = "1.0"
    description = "End-to-end text-to-MIDI (AAAI 2025). Custom Transformer + MidiTok REMI."

    def __init__(self):
        super().__init__()
        self._model     = None
        self._tokenizer = None
        self._device    = self._get_device()
        self.soundfont_path: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        print(f"[text2midi] Downloading from {HF_REPO} ...")
        from huggingface_hub import hf_hub_download
        import torch, inspect

        model_file = hf_hub_download(HF_REPO, "pytorch_model.bin")
        vocab_file = hf_hub_download(HF_REPO, "vocab_remi.pkl")
        arch_file  = hf_hub_download(HF_REPO, "transformer_model.py")
        try:
            self.soundfont_path = hf_hub_download(HF_REPO, "soundfont.sf2")
            print(f"[text2midi] Soundfont: {self.soundfont_path}")
        except Exception:
            pass

        # ① Tokenizer
        with open(vocab_file, "rb") as f:
            self._tokenizer = pickle.load(f)
        vocab_size = len(self._tokenizer)
        print(f"[text2midi] Tokenizer: {type(self._tokenizer).__name__}, vocab={vocab_size}")

        # ② Load state dict first — infer hyperparams from tensor shapes
        state = torch.load(model_file, map_location=self._device, weights_only=False)
        sd = state.get("model", state.get("state_dict", state))

        # Read d_model from embedding weight shape (always present)
        d_model = sd["input_emb.weight"].shape[1]
        # Read n_layers from decoder keys
        layer_ids = set()
        for k in sd:
            if k.startswith("decoder.layers."):
                try: layer_ids.add(int(k.split(".")[2]))
                except ValueError: pass
        n_layers = max(layer_ids) + 1 if layer_ids else 6
        # Read ffn_dim from linear1 weight
        ffn_dim = sd.get("decoder.layers.0.linear1.weight", None)
        ffn_dim = ffn_dim.shape[0] if ffn_dim is not None else d_model * 4
        # Sequence length from positional encoding
        max_seq = sd.get("pos_encoder.pe", None)
        max_seq = max_seq.shape[0] if max_seq is not None else 2048

        print(f"[text2midi] Inferred: d_model={d_model}, n_layers={n_layers}, "
              f"ffn_dim={ffn_dim}, max_seq={max_seq}")

        # ③ Import model class
        spec = importlib.util.spec_from_file_location("transformer_model", arch_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_cls = None
        for cname in ["Text2MidiTransformer", "MusicTransformer", "Transformer", "Model"]:
            if hasattr(mod, cname):
                model_cls = getattr(mod, cname)
                print(f"[text2midi] Model class: {cname}")
                break
        if model_cls is None:
            raise AttributeError(f"No model class in transformer_model.py. "
                                  f"Found: {[x for x in dir(mod) if not x.startswith('_')]}")

        # ④ Instantiate with correct hyperparams
        # Try different argument patterns based on what the constructor accepts
        sig    = inspect.signature(model_cls.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        print(f"[text2midi] Constructor params: {params}")

        # Map inferred values to the actual constructor parameter names
        param_map = {
            # vocab
            "n_vocab":            vocab_size,
            "vocab_size":         vocab_size,
            # d_model
            "d_model":            d_model,
            "dim":                d_model,
            # layers
            "num_decoder_layers": n_layers,
            "n_layers":           n_layers,
            "num_layers":         n_layers,
            # feedforward
            "dim_feedforward":    ffn_dim,
            "ffn_dim":            ffn_dim,
            "dim_ff":             ffn_dim,
            # sequence length
            "max_len":            max_seq,
            "max_seq":            max_seq,
            "seq_len":            max_seq,
        }
        kwargs = {k: v for k, v in param_map.items() if k in params}

        try:
            self._model = model_cls(**kwargs)
        except TypeError as e:
            print(f"[text2midi] Kwargs {kwargs} failed ({e}), trying positional...")
            self._model = model_cls(vocab_size, d_model, n_layers)

        # ⑤ Load weights
        missing, unexpected = self._model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[text2midi] Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[text2midi] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

        self._model = self._model.to(self._device).eval()
        self._loaded = True
        print("[text2midi] Ready.")

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, config: GenerationConfig) -> bytes:
        self._require_loaded()
        import torch

        self._last_prompt = config.prompt   # stored for T5 encoding in the loop
        prompt_ids = self._encode_prompt(config)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            token_ids = self._autoregressive_loop(
                input_ids, config.max_tokens, config.temperature, config.top_k
            )

        return self._ids_to_midi(token_ids, config.bpm)

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode_prompt(self, config: GenerationConfig) -> list[int]:
        vocab = self._tokenizer.vocab
        ids   = []
        for bos in ["BOS", "BOS_None", "<s>"]:
            if bos in vocab:
                ids.append(vocab[bos]); break

        words = config.prompt.lower().replace(",", " ").replace("-", " ").split()
        for word in words:
            for pat in [f"TEXT_{word}", f"Genre_{word}", f"genre_{word}", word]:
                if pat in vocab:
                    ids.append(vocab[pat]); break

        for t in [f"Tempo_{config.bpm}", f"TEMPO_{config.bpm}"]:
            if t in vocab:
                ids.append(vocab[t]); break

        return ids or [0]

    def _get_src_ids(self, prompt: str):
        """
        Tokenize the text prompt with flan-t5-base tokenizer.
        Returns (src_ids, src_mask) both as (1, S) LongTensors.
        The model's forward() calls self.encoder(src, attention_mask=src_mask)
        internally, so we just pass token IDs — not embeddings.
        """
        import torch
        from transformers import AutoTokenizer
        t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        enc = t5_tok(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self._device)
        return enc["input_ids"], enc["attention_mask"]   # both (1, S)

    def _autoregressive_loop(self, input_ids, max_new_tokens, temperature, top_k):
        import torch, torch.nn.functional as F
        vocab   = self._tokenizer.vocab
        eos_ids = {vocab[e] for e in ["EOS", "EOS_None", "</s>"] if e in vocab}

        # Encode text prompt once — src_ids/src_mask reused every step
        prompt   = getattr(self, "_last_prompt", "")
        src_ids, src_mask = self._get_src_ids(prompt)
        S = src_ids.shape[1]

        # src_mask for forward() must be (S, S) square causal/padding mask
        # Convert padding mask (1, S) → (S, S) additive float mask
        # positions where attention_mask==0 should be -inf (ignored)
        pad_mask = (src_mask[0] == 0)                    # (S,) bool
        src_sq   = torch.zeros(S, S, device=self._device)
        src_sq[:, pad_mask] = float("-inf")               # (S, S)

        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            tgt = torch.tensor([generated], dtype=torch.long).to(self._device)

            # forward(src, src_mask, tgt, ..., tgt_is_causal=True)
            # src and tgt must have same ndim — both are (N, S)/(N, T) LongTensors
            out = self._model(
                src=src_ids,
                src_mask=src_sq,
                tgt=tgt,
                tgt_is_causal=True,
            )
            # out shape: (N, T, vocab) or tuple (out, aux_loss) if use_moe
            if isinstance(out, tuple):
                out = out[0]

            logits  = out[0, -1, :]                       # (vocab,)
            logits  = logits / max(temperature, 1e-8)
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[-1]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            generated.append(next_id)
            if next_id in eos_ids:
                break

        return generated

    def _ids_to_midi(self, token_ids: list[int], bpm: int) -> bytes:
        for method in ["tokens_to_midi", "decode"]:
            if hasattr(self._tokenizer, method):
                try:
                    midi = getattr(self._tokenizer, method)([token_ids])
                    buf  = io.BytesIO()
                    midi.dump(buf)
                    return buf.getvalue()
                except Exception as e:
                    print(f"[text2midi] {method}() failed: {e}")

        id2tok = {v: k for k, v in self._tokenizer.vocab.items()}
        return _remi_tokens_to_midi([id2tok.get(i, "") for i in token_ids], bpm)


def _remi_tokens_to_midi(tokens: list[str], bpm: int) -> bytes:
    try:
        import pretty_midi
    except ImportError:
        raise RuntimeError("pip install pretty_midi")

    pm   = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=0)
    beat = 60.0 / bpm
    bar = 0; pos = 0; pitch = None; vel = 80; dur = 0.25

    for tok in tokens:
        if tok in ("Bar_None", "BAR"):
            bar += 1; pos = 0
        elif tok.startswith(("Position_", "POSITION_")):
            try: pos = int(tok.split("_")[1])
            except ValueError: pass
        elif tok.startswith(("Pitch_", "PITCH_")):
            try: pitch = int(tok.split("_")[1])
            except ValueError: pitch = None
        elif tok.startswith(("Velocity_", "VELOCITY_")):
            try: vel = int(tok.split("_")[1])
            except ValueError: pass
        elif tok.startswith(("Duration_", "DURATION_")):
            try: dur = int(tok.split("_")[1]) / 16.0
            except ValueError: pass
            if pitch is not None and 0 <= pitch <= 127:
                t0 = (bar * 4 + pos / 4.0) * beat
                inst.notes.append(pretty_midi.Note(
                    velocity=max(1, min(127, vel)), pitch=pitch,
                    start=t0, end=max(t0 + 0.05, t0 + dur * beat),
                ))
            pitch = None

    if not inst.notes:
        inst.notes.append(pretty_midi.Note(velocity=60, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()
