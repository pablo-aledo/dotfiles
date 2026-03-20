"""
backends/base.py
Abstract interface that every backend must implement.
Adding a new model = subclass MusicBackend, implement load() and generate().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    """
    Backend-agnostic generation parameters.
    Each backend maps these to its own API internally.
    """
    prompt: str
    temperature: float = 0.95      # randomness (0=deterministic, 2=chaos)
    top_k: int = 50                # top-k sampling filter
    top_p: float = 0.95            # nucleus sampling (used by some backends)
    max_tokens: int = 512          # max MIDI tokens to generate
    bpm: int = 120                 # tempo hint (not all backends honour it)
    extra: dict = field(default_factory=dict)  # backend-specific overrides


class MusicBackend(ABC):
    """
    All backends share this interface.

    Lifecycle:
        backend = MyBackend()   # lightweight init
        backend.load()          # heavy: download weights, load to GPU
        midi_bytes = backend.generate(config)  # returns raw MIDI bytes
        backend.unload()        # optional: free VRAM
    """

    # Subclasses must set these class attributes
    backend_id: str = "base"       # key used in BACKENDS dict
    name: str = "Base Backend"     # human-readable display name
    version: str = "0.0"
    description: str = ""

    def __init__(self):
        self._loaded = False

    # ── Required ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """
        Download weights if needed, load model to device.
        Must set self._loaded = True on success.
        """

    @abstractmethod
    def generate(self, config: GenerationConfig) -> bytes:
        """
        Generate music from config.prompt.
        Must return raw MIDI file bytes (ready to write to .mid).
        """

    # ── Optional hooks ────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Free GPU memory. Override if your backend holds tensors."""
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(f"Backend '{self.name}' is not loaded. Call .load() first.")

    @staticmethod
    def _get_device() -> str:
        """Return 'cuda', 'mps', or 'cpu'."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
