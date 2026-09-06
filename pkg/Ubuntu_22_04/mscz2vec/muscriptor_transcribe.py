#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MUSCRIPTOR_TRANSCRIBE.PY                                             ║
# ║  Transcripción polifónica multi-instrumento: audio → MIDI / JSON      ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
muscriptor_transcribe.py — transcripción de audio a MIDI/eventos con un
transformer decoder-only (arquitectura y pesos: MuScriptor, Kyutai x Mirelo,
CC BY-NC 4.0 — https://github.com/kyutai-labs/muscriptor).

Toma una grabación (wav/mp3/flac/…) y decodifica un stream de tokens estilo
MT3 en notas por instrumento, chunk a chunk, con soporte de streaming
(tie-prologue entre chunks) y detección opcional de tempo/compás para
cuantizar el MIDI resultante.

Requiere autenticarse una vez en HuggingFace para descargar los pesos:
    uvx hf auth login
    (o) export HF_TOKEN=hf_...

USO
---
    muscriptor_transcribe.py transcribe audio.wav
    muscriptor_transcribe.py transcribe audio.wav -f json -o eventos.json
    muscriptor_transcribe.py transcribe audio.wav -f jsonl -o - > eventos.jsonl
    muscriptor_transcribe.py transcribe audio.wav -m large -d cuda
    muscriptor_transcribe.py transcribe audio.wav --instruments "piano,cello"
    muscriptor_transcribe.py list-instruments

MODELOS
-------
    small   103M params · 14 capas · dim 768   — el más rápido en CPU
    medium  307M params · 24 capas · dim 1024  — por defecto
    large   1.4B params · 48 capas · dim 1536  — el más preciso, pide GPU

Alcance de esta versión de un solo fichero: solo transcripción a MIDI/JSON.
El repo original de MuScriptor añade además exportación a partitura vía
MuseScore y un servidor FastAPI + web con piano roll en vivo; ambos se
dejaron fuera aquí a propósito por no encajar con el estilo mutopia (un
único fichero, CLI simple, sin servidor).
"""

import argparse
import contextlib
import dataclasses
import difflib
import hashlib
import io
import json
import logging
import math
import os
import platform
import re
import sys
import time
import urllib.request
import warnings
import wave
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, Any, Literal, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import EntryNotFoundError
from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick
from packaging.version import Version
from safetensors.torch import load_file

PROG = "muscriptor_transcribe.py"

# --- ANSI color output (estilo mutopia) -------------------------------------
_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stderr.isatty()
RESET = "" if _NO_COLOR else "\033[0m"
DIM = "" if _NO_COLOR else "\033[2m"
CYAN = "" if _NO_COLOR else "\033[36m"
GREEN = "" if _NO_COLOR else "\033[32m"
YELLOW = "" if _NO_COLOR else "\033[33m"
RED = "" if _NO_COLOR else "\033[31m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if color else text


def _err(text: str) -> None:
    print(text, file=sys.stderr)


logger = logging.getLogger("muscriptor")


_HAS_TORCH_ACCELERATOR = Version(torch.__version__.split("+")[0]) >= Version("2.6")


def _mps_available() -> bool:
    """Whether MPS is available and worth auto-selecting.

    torch <= 2.2 also reports MPS as available on Intel Macs with AMD GPUs, a
    backend that was never solid and has since been abandoned. Passing
    ``device="mps"`` explicitly still works there for those who want to try
    it; this only affects auto-detection.
    """
    return torch.backends.mps.is_available() and platform.machine() == "arm64"


def is_available() -> bool:
    """Whether an accelerator (GPU) is available."""
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.is_available()
    return torch.cuda.is_available() or _mps_available()


def current_accelerator() -> torch.device:
    """The current accelerator device.

    Raises ``RuntimeError`` if no accelerator is available; check
    :func:`is_available` first.
    """
    if _HAS_TORCH_ACCELERATOR:
        return torch.accelerator.current_accelerator()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    raise RuntimeError("No available accelerator detected.")


def synchronize() -> None:
    """Wait for all kernels on the current accelerator to complete.

    No-op if no accelerator is available.
    """
    if _HAS_TORCH_ACCELERATOR:
        # torch.accelerator.synchronize() still tries to init CUDA even on CPU-only systems
        # Only call it if we actually have a non-CPU accelerator
        try:
            current_device = torch.accelerator.current_device_index()
            if current_device >= 0:
                torch.accelerator.synchronize()
        except RuntimeError:
            # No accelerator available (CUDA not found, etc.)
            pass
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif _mps_available():
        torch.mps.synchronize()


# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020




def sinc(x: torch.Tensor) -> torch.Tensor:
    """sin(x) / x, with the limit value 1 at x == 0.

    __Warning__: the input is not multiplied by `pi`!
    """
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(x) / x,
    )


class ResampleFrac(torch.nn.Module):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """

    def __init__(
        self, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945
    ):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import torch
        >>> resample = ResampleFrac(4, 5)
        >>> x = torch.randn(1000)
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        # rolloff will perform antialiasing filtering by removing the highest frequencies.
        # At first I thought I only needed this when downsampling, but when upsampling
        # you will get edge artifacts without this, the edge is equivalent to zero padding,
        # which will add high freq artifacts.
        sr *= self.rolloff

        # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
        # using the sinc interpolation formula:
        #   x(t) = sum_i x[i] sinc(pi * old_sr * (i / old_sr - t))
        # We can then sample the function x(t) with a different sample rate:
        #    y[j] = x(j / new_sr)
        # or,
        #    y[j] = sum_i x[i] sinc(pi * old_sr * (i / old_sr - j / new_sr))

        # We see here that y[j] is the convolution of x[i] with a specific filter, for which
        # we take an FIR approximation, stopping when we see at least `zeros` zeros crossing.
        # But y[j+1] is going to have a different set of weights and so on, until y[j + new_sr].
        # Indeed:
        # y[j + new_sr] = sum_i x[i] sinc(pi * old_sr * ((i / old_sr - (j + new_sr) / new_sr))
        #               = sum_i x[i] sinc(pi * old_sr * ((i - old_sr) / old_sr - j / new_sr))
        #               = sum_i x[i + old_sr] sinc(pi * old_sr * (i / old_sr - j / new_sr))
        # so y[j+new_sr] uses the same filter as y[j], but on a shifted version of x by `old_sr`.
        # This will explain the F.conv1d after, with a stride of old_sr.
        self._width = math.ceil(self.zeros * self.old_sr / sr)
        # If old_sr is still big after GCD reduction, most filters will be very unbalanced, i.e.,
        # they will have a lot of almost zero values to the left or to the right...
        # There is probably a way to evaluate those filters more efficiently, but this is kept for
        # future work.
        idx = torch.arange(-self._width, self._width + self.old_sr).float()
        for i in range(self.new_sr):
            t = (-i / self.new_sr + idx / self.old_sr) * sr
            t = t.clamp_(-self.zeros, self.zeros)
            t *= math.pi
            window = torch.cos(t / self.zeros / 2) ** 2
            kernel = sinc(t) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel.div_(kernel.sum())
            kernels.append(kernel)

        self.register_buffer("kernel", torch.stack(kernels).view(self.new_sr, 1, -1))

    def forward(
        self, x: torch.Tensor, output_length: Optional[int] = None, full: bool = False
    ):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        if self.old_sr == self.new_sr:
            return x
        shape = x.shape
        length = x.shape[-1]
        x = x.reshape(-1, length)
        x = F.pad(
            x[:, None], (self._width, self._width + self.old_sr), mode="replicate"
        )
        ys = F.conv1d(x, self.kernel, stride=self.old_sr)  # type: ignore
        y = ys.transpose(1, 2).reshape(list(shape[:-1]) + [-1])

        float_output_length = torch.as_tensor(self.new_sr * length / self.old_sr)
        max_output_length = torch.ceil(float_output_length).long()
        default_output_length = torch.floor(float_output_length).long()

        if output_length is None:
            applied_output_length = max_output_length if full else default_output_length
        elif output_length < 0 or output_length > max_output_length:
            raise ValueError(f"output_length must be between 0 and {max_output_length}")
        else:
            applied_output_length = torch.tensor(output_length)
            if full:
                raise ValueError("You cannot pass both full=True and output_length")
        return y[..., :applied_output_length]  # type: ignore

    def __repr__(self):
        return (
            f"ResampleFrac(old_sr={self.old_sr}, new_sr={self.new_sr}, "
            f"zeros={self.zeros}, rolloff={self.rolloff})"
        )


def resample_frac(
    x: torch.Tensor,
    old_sr: int,
    new_sr: int,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    return ResampleFrac(old_sr, new_sr, zeros, rolloff).to(x)(x, output_length, full)


def _read_wav_file(source) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV file using the stdlib `wave` module.

    `source` may be a filesystem path or a binary file-like object.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    if hasattr(source, "read"):
        opened = wave.open(source, "rb")
    else:
        opened = wave.open(str(source), "rb")
    with opened as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        as_int32 = np.where(as_int32 >= (1 << 23), as_int32 - (1 << 24), as_int32)
        data = as_int32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    data = data.reshape(-1, n_channels)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def _read_non_wav_file(source: str | Path | IO[bytes]) -> tuple[torch.Tensor, int]:
    """Load a non-WAV audio file using `soundfile`.

    `source` may be a filesystem path or a binary file-like object (e.g. an
    ``io.BytesIO`` of an uploaded file), since libsndfile reads either.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    target = str(source) if isinstance(source, (str, Path)) else source
    data, sample_rate = sf.read(target, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sample_rate


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    """Sinc resampler via julius `resample_frac`. Operates along the last dim."""
    if orig_freq == new_freq:
        return waveform
    return resample_frac(waveform, int(orig_freq), int(new_freq))


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return a mono float32 tensor at target_sr.

    PCM WAV files are read with the stdlib `wave` module. Other formats (mp3,
    flac, ogg, m4a, …) are decoded via `soundfile`. Dispatch is by content, not
    file extension, so misnamed files (e.g. an MP3 upload saved as .wav) still
    load.

    Returns:
        Tensor of shape [1, T] at target_sr.
    """
    filepath = Path(path)
    try:
        wav, sr = _read_wav_file(str(filepath))
    except (wave.Error, EOFError):
        wav, sr = _read_non_wav_file(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav


logger = logging.getLogger(__name__)

# Used to detect songs that don't have a constant tempo (don't use a metronome)
MAX_TEMPO_RESIDUAL = 0.05

# Fraction of bars that must agree on a beats-per-bar count to write a time
# signature. Trackers that lose the meter spread their downbeats across several
# spacings, and a wrong time signature is worse than none.
MIN_METER_AGREEMENT = 0.9

MIN_BEATS = 8

# Marker text prefix recording how far notes were delayed to align bar lines,
# so `/auralize` can line the synthesis back up with the original audio.
BAR_OFFSET_MARKER = "muscriptor:bar_offset="

### Constants for estimate_onset_delay()

# Candidate grids, binary and triplet divisions of the beat, simplest first.
ONSET_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24)

# Discard detections with a concentration lower than this. On our test set, 92% songs
# are above 0.5
MIN_ONSET_CONCENTRATION = 0.5

# |R| for n random angles is about 1/sqrt(n). With MIN_ONSETS = 40 the chance of
# passing the bar with a random distribution is about 1 in 5,000.
MIN_ONSETS = 40

# The largest delay this is meant to correct.
# Trade-off: we want to correct larger offsets if they happen, but a smaller value
# allows us to use smaller grids like 1/16th notes (depends also on the tempo).
# This is because if a grid has 20ms, we can't distinguish between a +15ms and -5ms
# offset since things are cyclical, unless we have a max offset of <15ms.
# With 0.04 we can do 16th notes up to 187 BPM.
MAX_ONSET_DELAY_S = 0.04

# Note onset times in seconds, however the caller happens to hold them.
Onsets = Sequence[float] | np.ndarray


class BeatDetectionError(RuntimeError):
    """No usable beat grid in the audio (too short, or no constant tempo)."""


# What to do when the tempo can't be detected: True raises, False doesn't even
# try (the escape hatch for songs the detector gets wrong), and "best-effort"
# warns and falls back to the placeholder tempo.
TempoDetection = bool | Literal["best-effort"]


def read_bar_offset(midi) -> float:
    """Seconds of bar-alignment delay recorded in a MidiFile, 0.0 if absent."""
    for track in midi.tracks:
        for msg in track:
            if msg.type == "marker" and msg.text.startswith(BAR_OFFSET_MARKER):
                try:
                    return float(msg.text.removeprefix(BAR_OFFSET_MARKER))
                except ValueError:
                    return 0.0
    return 0.0


@dataclasses.dataclass
class BeatGrid:
    """A constant-tempo beat grid detected from audio."""

    bpm: float
    # None when the meter could not be determined; write no time signature.
    beats_per_bar: int | None
    # Time of the first detected bar line, in seconds.
    first_downbeat: float
    # The individual beat times the grid was fitted to, kept for
    # `with_onset_delay`, which needs them rather than the fitted tempo: the
    # tracked beats follow the recording's small tempo wobbles, and those are the
    # same size as the offset being measured. None for a hand-built grid; kept
    # out of equality and repr, being an array and an implementation detail.
    beats: np.ndarray | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    # Seconds by which the transcribed onsets are late against these beats, or None
    # until some notes have been measured against the grid (see `with_onset_delay`).
    # Whoever writes the notes out subtracts it from them.
    onset_delay: float | None = None
    # Subdivisions of the beat to use for quantizing the onsets.
    beat_subdivision: int | None = None

    @property
    def bar_seconds(self) -> float | None:
        if self.beats_per_bar is None:
            return None
        return self.beats_per_bar * 60.0 / self.bpm

    def bar_offset(self, min_shift: float = 0.0) -> float:
        """Seconds to delay every note so bar lines land on downbeats.

        MIDI has no pickup measure: bar 1 starts at tick 0, so the only way to
        put a bar line on the first downbeat is to shift the music later. Always
        a forward shift, keeping ticks non-negative and dropping no notes; the
        leading partial bar holds whatever preceded the first downbeat.

        Whole bars are added until the shift reaches `min_shift`, so a caller
        that moved the notes earlier (by `onset_delay`) still gets non-negative
        ticks without having to squash the notes near the start.
        """
        step = self.bar_seconds
        if step is None:
            # No meter to align to, so the only reason to shift is the headroom;
            # do it in whole beats, which are all this grid has.
            step, offset = 60.0 / self.bpm, 0.0
        else:
            offset = (step - self.first_downbeat % step) % step
        if offset < min_shift:
            offset += step * math.ceil((min_shift - offset) / step)
        return offset

    def with_onset_delay(self, onsets: Onsets) -> "BeatGrid":
        """This grid with `onsets`' lag against it measured and filled in.

        Subtracting that lag from the note times moves them onto the beats. We
        trust the beats tracked by beat_this over the transcription - they're a bit
        more accurate based on our measurements. 0.0 when there is nothing to
        measure from; see estimate_onset_delay() for details.

        Measuring once and carrying the answer around is what keeps the MIDI file
        and the UI (which is told the same number) shifting by the same amount.
        """
        measured = None
        if self.beats is not None:
            measured = estimate_onset_delay(onsets, self)
        if measured is None:
            return dataclasses.replace(self, onset_delay=0.0, beat_subdivision=None)
        logger.info(
            "onsets sit %+.1f ms off a 1/%d-beat grid (|R| = %.2f over %d "
            "onsets); moving the notes back by that much",
            1000 * measured.seconds,
            measured.subdivision,
            measured.concentration,
            measured.n_onsets,
        )
        return dataclasses.replace(
            self, onset_delay=measured.seconds, beat_subdivision=measured.subdivision
        )


@dataclasses.dataclass(frozen=True)
class OnsetDelay:
    """How late a set of note onsets sits against the beat subdivision it is on."""

    # Signed seconds, positive when the onsets are late.
    seconds: float
    # Resultant length in [0, 1]: how tightly the onsets sit on the grid.
    concentration: float
    # Subdivisions per beat of the grid the delay is measured against.
    subdivision: int
    # Distinct onset times that went into it.
    n_onsets: int


def get_onsets_phase(onsets: Onsets, beats: np.ndarray) -> np.ndarray:
    """Onset times as positions in continuous beats.

    Onsets outside the tracked span drop out, since there is no beat to place
    them in. One entry per distinct onset time (to the millisecond) rather than
    per note, so a six-note chord does not outvote six single notes elsewhere.
    """
    times = np.unique(np.round(np.asarray(onsets, dtype=float), 3))
    inside = (times >= beats[0]) & (times <= beats[-1])
    return np.interp(times[inside], beats, np.arange(len(beats)))


def get_phase_with_subdivision(
    phase_beats: np.ndarray, subdivision: int
) -> tuple[float, float]:
    """Compute the average of unit vectors (phase_beats) on a circle.

    The angle tells us the mean offset, and the magnitude tells us how well they align
    (the concentration).
    """
    angles = 2 * np.pi * np.mod(phase_beats * subdivision, 1.0)
    mean = np.exp(1j * angles).mean()
    turns = 1 / (2 * np.pi * subdivision)  # radians on the fine grid → beats
    return float(np.abs(mean)), float(np.angle(mean)) * turns


def estimate_onset_delay(onsets: Onsets, grid: "BeatGrid") -> OnsetDelay | None:
    """How late `onsets` sit against the beat subdivision they are on.

    This is to correct for the fact that we observed that the detected onsets don't
    always align well with the beats.

    Algorithm: For each onset, plot it as a unit vector on a circle where the angle is
    its relative position in the beat. Then take the average of these vectors: the
    angle tells you the average offset and the magnitude says how well they align.
    In practice, the onsets are on a subdivision of the beat (e.g. 8th notes) so
    also repeat this for different subdivisions and pick the one where the notes align
    the best to read the alignment from.
    """
    if grid.beats is None or len(grid.beats) < 2:
        return None
    period_s = 60.0 / grid.bpm

    # The phases relative to full beats
    phase_beats = get_onsets_phase(onsets, grid.beats)
    if len(phase_beats) < MIN_ONSETS:
        logger.info(
            "not correcting the downbeat: %d onset time(s) on the tracked span, "
            "need %d",
            len(phase_beats),
            MIN_ONSETS,
        )
        return None

    candidates = [
        s for s in ONSET_SUBDIVISIONS if period_s / (2 * s) >= MAX_ONSET_DELAY_S
    ]
    if not candidates:
        # Should not happen with reasonable tempos but guard it anyway
        logger.info("not correcting the downbeat: no grid coarse enough to fit")
        return None

    scored = {s: get_phase_with_subdivision(phase_beats, s) for s in candidates}

    # Keep the subdivision with the highest concentration
    subdivision = max(scored, key=lambda s: scored[s][0])
    concentration, delay_beats = scored[subdivision]

    if concentration < MIN_ONSET_CONCENTRATION:
        logger.info(
            "not correcting the downbeat: onsets sit on no subdivision of the beat "
            "(best |R| = %.2f, need %.2f)",
            concentration,
            MIN_ONSET_CONCENTRATION,
        )
        return None

    delay_s = delay_beats * period_s
    if abs(delay_s) > MAX_ONSET_DELAY_S:
        logger.warning(
            "not correcting the downbeat: onsets measured %+.0f ms off a "
            "1/%d-beat grid, further than the %+.0f ms this can plausibly be",
            1000 * delay_s,
            subdivision,
            1000 * MAX_ONSET_DELAY_S,
        )
        return None

    return OnsetDelay(
        seconds=delay_s,
        concentration=concentration,
        subdivision=subdivision,
        n_onsets=len(phase_beats),
    )


def fit_tempo(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares tempo over the beat sequence.

    Returns (bpm, residual RMS in seconds). Fitting a line through beat index
    against time beats taking the median inter-beat interval: trackers quantise
    beats to a frame grid (50 Hz for beat_this), which alone limits median-IBI
    tempo resolution to a few BPM.
    """
    index = np.arange(len(beats))
    slope, intercept = np.polyfit(index, beats, 1)
    residual = beats - (intercept + slope * index)
    return 60.0 / float(slope), float(residual.std())


def infer_beats_per_bar(
    beats: np.ndarray,
    downbeats: np.ndarray,
    min_agreement: float = MIN_METER_AGREEMENT,
) -> int | None:
    """Beats per bar from downbeat spacing, or None if the bars disagree.

    Only measures how far apart the downbeats are; it cannot tell whether the
    downbeats themselves are on the right beat. Note that a tracker that
    subdivides the bar wrongly (reporting two beats per bar for music in 3/4)
    can still be self-consistent here, which is why this stays conservative.
    """
    if len(downbeats) < 3 or len(beats) < 2:
        return None
    beat = float(np.median(np.diff(beats)))
    counts = np.round(np.diff(downbeats) / beat).astype(int)
    counts = counts[counts >= 2]
    if not len(counts):
        return None
    values, tally = np.unique(counts, return_counts=True)
    best = int(tally.argmax())
    if tally[best] / len(counts) < min_agreement:
        return None
    return int(values[best])


def detect_grid(
    wav: torch.Tensor, sr: int, checkpoint: str = "final0", device: str = "cpu"
) -> BeatGrid:
    """Detect a constant-tempo beat grid.

    Args:
        wav: Audio as [C, T] (this repo's convention), float32.
        sr: Sample rate of `wav`; beat_this resamples internally.
        checkpoint: beat_this checkpoint name. "final0" over "small0": the small
            model emits spurious beats before the first downbeat, which shifts
            the bar offset by a beat or two.
        device: Torch device for the beat model.

    Raises BeatDetectionError when the audio is too short or the beats do not
    fit a constant tempo. An unclear meter is not fatal: the BeatGrid comes back
    with beats_per_bar=None, since tempo alone is worth writing.
    """
    # Imported here, not at module scope: beat_this pulls in torchaudio and soxr,
    # which would slow every CLI invocation that never transcribes anything.
    from beat_this.inference import Audio2Beats

    # This triggers an error in beat_this so report as BeatDetectionError directly
    min_duration_s = 1.0
    if wav.shape[-1] < min_duration_s * sr:
        raise BeatDetectionError(
            f"Audio is {wav.shape[-1] / sr:.2f}s long, too short to detect a tempo"
        )

    signal = wav.mean(dim=0).detach().cpu().numpy()  # beat_this wants mono, 1-D
    # Returns (beats, downbeats) despite beat_this's own File2File unpacking
    # them the other way round.
    beats, downbeats = Audio2Beats(
        checkpoint_path=checkpoint, device=device, dbn=False
    )(signal, sr)

    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < MIN_BEATS:
        raise BeatDetectionError(
            f"Only {len(beats)} beats detected, need at least {MIN_BEATS}"
        )

    bpm, residual = fit_tempo(beats)
    beat_seconds = 60.0 / bpm
    if residual > MAX_TEMPO_RESIDUAL * beat_seconds:
        raise BeatDetectionError(
            f"The recording has no fixed tempo (beats deviate {residual * 1000:.0f} ms "
            f"RMS from a constant {bpm:.1f} BPM)"
        )

    beats_per_bar = infer_beats_per_bar(beats, downbeats)
    first_downbeat = float(downbeats[0]) if len(downbeats) else float(beats[0])
    logger.info(
        "detected %.3f BPM, %s, first downbeat %.3fs (beat residual %.1f ms)",
        bpm,
        f"{beats_per_bar}/4" if beats_per_bar else "meter unknown",
        first_downbeat,
        residual * 1000,
    )
    return BeatGrid(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        first_downbeat=first_downbeat,
        beats=beats,
    )


_CACHE_DIR = Path.home() / ".cache" / "muscriptor"


class ModelDownloadError(RuntimeError):
    """Downloading the model weights failed for a reason the user must fix
    (typically missing HuggingFace authentication). The message is meant to
    be shown to the user as-is, without a traceback."""


def _auth_help(repo_id: str) -> str:
    return (
        f"cannot download '{repo_id}' from HuggingFace: the MuScriptor model "
        "weights are gated and require a (free) HuggingFace account.\n\n"
        f"  1. Accept the model license at https://huggingface.co/{repo_id}\n"
        "     (access is granted automatically).\n"
        "  2. Authenticate on this machine, either:\n"
        "       - run: uvx hf auth login\n"
        "       - or set the HF_TOKEN environment variable to a read token\n"
        "         from https://huggingface.co/settings/tokens\n\n"
        "See the 'HuggingFace login' section of the README for details."
    )


def download_if_necessary(url: str | Path) -> Path:
    """Resolve a weights location to a local file, downloading if necessary.

    Args:
        url: Where to find the weights:
            - ``hf://<repo_id>/<path/in/repo>`` — downloaded via huggingface_hub.
            - ``http(s)://…`` — fetched with a plain HTTP GET and cached under
              the cache dir (filename prefixed with a hash of the URL).
            - anything else (a local path, as ``str`` or ``Path``) — used as-is;
              nothing is downloaded, but the file must already exist.

    Returns:
        Path to the local file.
    """
    if isinstance(url, str) and url.startswith("hf://"):
        org, name, hf_filename = url[len("hf://") :].split("/", 2)
        repo_id = f"{org}/{name}"
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=hf_filename)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            # What an unauthenticated (or not-yet-approved) client gets back
            # from a gated repo, depending on hub version and repo state.
            raise ModelDownloadError(_auth_help(repo_id)) from e
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) in (401, 403):
                raise ModelDownloadError(_auth_help(repo_id)) from e
            raise
        return Path(cached)

    if isinstance(url, str) and url.startswith(("http://", "https://")):
        # Prefix the cache filename with a hash of the URL so two different URLs
        # that share a filename don't map to the same file.
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1].split("?")[0]
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        dest = _CACHE_DIR / f"{url_hash}_{filename}"
        if dest.exists():
            return dest
        print(f"Downloading {filename} …")
        # Download to a per-process temp file, then rename: an interrupted or
        # concurrent download must never leave a partial file at `dest`, where
        # it would be mistaken for a complete one forever after.
        tmp = dest.with_name(f"{dest.name}.part{os.getpid()}")
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    # Local file — nothing to download, just check it's there.
    path = Path(url)
    if not path.exists():
        raise FileNotFoundError(f"weights file not found: {path}")
    return path


def download_companion(url: str | Path, filename: str) -> Path | None:
    """Best-effort fetch of a sibling file from the same ``hf://`` repo.

    Used to grab a model's ``config.json`` next to its weights. Returns the
    local path, or ``None`` if ``url`` isn't an ``hf://`` URL or the file can't
    be fetched — repo/file missing, gated, or offline (so callers can fall back
    to other detection schemes rather than failing the whole load).
    """
    if not (isinstance(url, str) and url.startswith("hf://")):
        return None
    org, name, _ = url[len("hf://") :].split("/", 2)
    try:
        cached = hf_hub_download(repo_id=f"{org}/{name}", filename=filename)
    except (EntryNotFoundError, HfHubHTTPError):
        return None
    return Path(cached)


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Convert a tensor of sequence lengths to a boolean mask."""
    assert len(lengths.shape) == 1
    final_length = int(lengths.max().item()) if not max_len else max_len
    final_length = max(final_length, 1)
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def multinomial(
    input: torch.Tensor, num_samples: int, replacement: bool = False, *, generator=None
) -> torch.Tensor:
    """torch.multinomial with arbitrary number of dimensions."""
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(
        input_, num_samples=num_samples, replacement=replacement, generator=generator
    )
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int, num_samples: int = 1) -> torch.Tensor:
    """Sample from top-k probabilities."""
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs = probs * (probs >= min_value_top_k).float()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return multinomial(probs, num_samples=num_samples)


def sample_top_p(probs: torch.Tensor, p: float, num_samples: int = 1) -> torch.Tensor:
    """Sample from nucleus (top-p) distribution."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort * (~mask).float()
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    next_token = multinomial(probs_sort, num_samples=num_samples)
    return torch.gather(probs_idx, -1, next_token)


def sample_from_probs(
    probs: torch.Tensor, top_p: float = 0.0, top_k: int = 0
) -> torch.Tensor:
    """Sample one token from probs, optionally filtered by top-p or top-k."""
    if top_p > 0.0:
        return sample_top_p(probs, top_p)
    if top_k > 0:
        return sample_top_k(probs, top_k)
    return multinomial(probs, num_samples=1)


def sample_stratified(
    probs: torch.Tensor,
    special_token: int,
    first_temp: float,
    second_temp: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Stratified sampling: first decide special vs. non-special, then sample among non-special."""
    eps = 1e-12
    probs_special = probs[..., special_token : special_token + 1].clamp(
        min=eps, max=1 - eps
    )
    logits_two = torch.cat(
        [torch.log(probs_special), torch.log(1 - probs_special)], dim=-1
    )
    logits_two = logits_two / max(first_temp, eps)
    probs_two = torch.softmax(logits_two, dim=-1)
    probs_special_temp = probs_two[..., 0:1]
    next_token_is_special = torch.rand_like(probs_special_temp).lt(probs_special_temp)

    denom = (1 - probs_special).clamp(min=eps)
    new_probs = probs.clone() / denom
    new_probs[..., special_token] = 0.0
    if second_temp > 0:
        log_new = torch.log(new_probs.clamp(min=eps)) / second_temp
        new_probs = torch.softmax(log_new, dim=-1)

    next_token = sample_from_probs(new_probs, top_p=top_p, top_k=top_k)

    return torch.where(
        next_token_is_special, torch.full_like(next_token, special_token), next_token
    )


def _hz_to_mel_htk(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
) -> torch.Tensor:
    """Triangular mel filterbank matching torchaudio.functional.melscale_fbanks
    with mel_scale='htk' and norm=None. Returns a tensor of shape [n_freqs, n_mels]."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel_htk(torch.tensor(float(f_min)))
    m_max = _hz_to_mel_htk(torch.tensor(float(f_max)))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
    f_pts = _mel_to_hz_htk(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.maximum(torch.zeros(()), torch.minimum(down_slopes, up_slopes))


class _Spectrogram(nn.Module):
    """Holds the STFT window so the safetensors key
    `...mel_spec_transform.spectrogram.window` round-trips."""

    def __init__(self, n_fft: int):
        super().__init__()
        self.register_buffer("window", torch.hann_window(n_fft))


class _MelScale(nn.Module):
    """Holds the mel filterbank so the safetensors key
    `...mel_spec_transform.mel_scale.fb` round-trips."""

    def __init__(self, fb: torch.Tensor):
        super().__init__()
        self.register_buffer("fb", fb)


class _MelSpectrogram(nn.Module):
    """Pure-torch equivalent of torchaudio.transforms.MelSpectrogram
    (htk mel scale, no Slaney norm, win_length == n_fft)."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.center = center
        self.pad_mode = pad_mode

        self.spectrogram = _Spectrogram(n_fft)
        fb = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.mel_scale = _MelScale(fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.spectrogram.window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        spec = spec.abs() ** self.power
        mel = torch.matmul(spec.transpose(-1, -2), self.mel_scale.fb).transpose(-1, -2)
        return mel.reshape(*leading, *mel.shape[-2:])


State = dict[str, Any]
ModelState = dict[str, State]


class StatefulModule(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_absolute_name: str | None = None

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int) -> State:
        raise NotImplementedError

    def increment_step(self, state: State, increment: int = 1) -> None:
        pass

    def get_state(self, model_state: ModelState | None) -> State | None:
        if model_state is None or self._module_absolute_name is None:
            return None
        return model_state.get(self._module_absolute_name)


def init_states(model: nn.Module, batch_size: int, sequence_length: int) -> ModelState:
    """Allocate state for every :class:`StatefulModule` reachable from ``model``.

    Side effect: each stateful submodule has its ``_module_absolute_name`` set
    so subsequent ``get_state`` calls can find its slot.
    """
    result: ModelState = {}
    for module_name, module in model.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = module_name
            result[module_name] = module.init_state(batch_size, sequence_length)
    return result


def increment_steps(
    model: nn.Module, model_state: ModelState, increment: int = 1
) -> None:
    """Bump the step counter for every stateful submodule of ``model``.

    Uses each module's ``_module_absolute_name`` (set by :func:`init_states`)
    to look up its slot, so this works on subtrees even when ``init_states``
    was called on a different root.
    """
    for _, module in model.named_modules():
        if (
            isinstance(module, StatefulModule)
            and module._module_absolute_name is not None
        ):
            module.increment_step(model_state[module._module_absolute_name], increment)


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class StreamingMultiheadAttention(StatefulModule):
    """Causal multi-head self-attention with a preallocated KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory_kwargs)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        weight = self.in_proj_weight
        return {
            "cache": torch.full(
                (2, batch_size, sequence_length, self.num_heads, self.dim_per_head),
                float("nan"),
                device=weight.device,
                dtype=weight.dtype,
            ),
            # Kept as a plain host int: it is advanced deterministically by the
            # host-side generate loop, and reading it from a device tensor
            # (`.item()`) would force a GPU sync per layer per decode step.
            "offset": 0,
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offset"] = state["offset"] + increment

    def _complete_kv(self, k, v, state: State | None):
        if state is None:
            return k, v
        cache = state["cache"]
        end = state["offset"]
        T = k.shape[1]
        cache[0, :, end : end + T] = k
        cache[1, :, end : end + T] = v
        return cache[0, :, : end + T], cache[1, :, : end + T]

    def forward(
        self,
        query: torch.Tensor,
        model_state: ModelState | None = None,
    ):
        state = self.get_state(model_state)
        projected = nn.functional.linear(query, self.in_proj_weight)
        packed = rearrange(projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads)
        q, k, v = packed.unbind(dim=2)

        k, v = self._complete_kv(k, v, state)
        dtype = q.dtype

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Causality must be bottom-right aligned so streaming decode steps
        # (T_q=1, T_k=cache_len) attend to all past tokens; PyTorch's
        # is_causal=True is top-left aligned and would mask out all cached
        # tokens except position 0 when T_q < T_k. An explicit attn_mask
        # forces SDPA onto the unfused math fallback, so only build one in
        # the rectangular case that actually needs it — the two shapes this
        # model hits (single-token decode and square prefill) stay mask-free
        # and dispatch to the fused (flash) CPU/CUDA kernels.
        T_q, T_k = q_t.shape[2], k_t.shape[2]
        if T_q == 1:
            # One query row, bottom-right aligned: nothing is masked.
            x = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
        elif T_q == T_k:
            # Square: bottom-right and top-left alignment coincide.
            x = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, dropout_p=0.0
            )
        else:
            # Unused in practice
            raise NotImplementedError(
                f"Streaming attention with T_q={T_q} and T_k={T_k} is not supported; use T_q=1 or T_q=T_k."
            )
        x = x.transpose(1, 2).to(dtype)

        x = rearrange(x, "b t h d -> b t (h d)")
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    """Pre-norm transformer block: self-attention + GELU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, **factory_kwargs
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        model_state: ModelState | None = None,
    ):
        x = x + self.self_attn(self.norm1(x), model_state=model_state)
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x


class StreamingTransformer(StatefulModule):
    """Stack of causal streaming transformer layers with sinusoidal positions."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        max_period: float = 10_000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period
        self.layers = nn.ModuleList(
            [
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        device = self.layers[0].norm2.weight.device
        return {
            "offsets": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offsets"] = state["offsets"] + increment

    def forward(
        self,
        x: torch.Tensor,
        prepend_length: int = 0,
        model_state: ModelState | None = None,
    ):
        del prepend_length  # unused; positions come from state['offsets']
        B, T, C = x.shape
        state = self.get_state(model_state)
        offsets = (
            state["offsets"]
            if state is not None
            else torch.zeros(B, dtype=torch.long, device=x.device)
        )

        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offsets.view(-1, 1, 1)
        # Always compute the sinusoidal embedding in fp32: fp16 cannot even
        # represent odd integers above 2048, so half-precision positions would
        # collapse neighbouring timesteps to the same embedding.
        pos_emb = create_sin_embedding(
            positions, C, max_period=self.max_period, dtype=torch.float32
        )
        x = x + (pos_emb * (positions >= 0).float()).to(x.dtype)

        for layer in self.layers:
            x = layer(x, model_state=model_state)
        return x


ConditionType = tuple[torch.Tensor, torch.Tensor]  # (embedding [B, T, D], mask [B, T])


class WavCondition(NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: list[int]
    path: list[str | None] = []
    seek_time: list[float | None] = []


@dataclass
class ConditioningAttributes:
    text: dict[str, str | None] = field(default_factory=dict)
    wav: dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: dict[str, Any] = field(default_factory=dict)
    symbolic: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def symbolic_attributes(self):
        return self.symbolic.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
            "symbolic": self.symbolic_attributes,
        }

    @classmethod
    def condition_types(cls) -> list:
        return ["text", "wav", "joint_embed", "symbolic"]


def nullify_wav(cond: WavCondition) -> WavCondition:
    B = cond.wav.shape[0]
    return WavCondition(
        wav=torch.zeros(*cond.wav.shape[:-1], 1, device=cond.wav.device),
        length=torch.zeros(B, dtype=cond.length.dtype, device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * B,
        seek_time=[None] * B,
    )


def nullify_all_conditions(
    samples: list[ConditioningAttributes],
) -> list[ConditioningAttributes]:
    """Return a copy of ``samples`` with every wav/text condition nulled out.

    Used to build the unconditional batch for classifier-free guidance.
    """
    samples = deepcopy(samples)
    for sample in samples:
        for k in list(sample.wav):
            sample.wav[k] = nullify_wav(sample.wav[k])
        for k in list(sample.text):
            sample.text[k] = None
    return samples


class MelSpectrogramConditioner(nn.Module):
    """Log mel spectrogram conditioner. Projects mel bins to transformer dim."""

    def __init__(
        self,
        output_dim: int,
        device: torch.device | str,
        sample_rate: int,
        n_fft: int = 2048,
        frame_rate: int = 100,
        n_mel_bins: int = 512,
        normalize_audio: bool = False,
        log_scale: bool = True,
        eps: float = 1e-6,
        # unused arg kept for config compatibility
        fine_frame_rate: int = None,
    ):
        self.fine_frame_rate_ratio = 1
        if fine_frame_rate is not None:
            assert fine_frame_rate % frame_rate == 0
            self.fine_frame_rate_ratio = fine_frame_rate // frame_rate

        super().__init__()
        self.dim = n_mel_bins * self.fine_frame_rate_ratio
        self.output_dim = output_dim
        self.output_proj = nn.Linear(self.dim, output_dim)
        self.device = device
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.normalize_audio = normalize_audio
        self.log_scale = log_scale
        self.eps = eps

        if self.fine_frame_rate_ratio == 1:
            assert sample_rate % frame_rate == 0
            self.hop_length = sample_rate // frame_rate
        else:
            assert sample_rate % fine_frame_rate == 0
            self.hop_length = sample_rate // fine_frame_rate

        self.mel_spec_transform = _MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mel_bins,
            power=1.0,
            center=True,
            pad_mode="reflect",
        ).to(device)

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(
            wav.to(self.device), length.to(self.device), sample_rate, path, seek_time
        )

    def _mel_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = x.wav
            if self.normalize_audio:
                wav = wav / (wav.abs().max(dim=-1, keepdim=True).values + 1e-8)
            mel = self.mel_spec_transform(wav)
            mel = rearrange(mel, "b 1 d t -> b t d")
            if self.fine_frame_rate_ratio > 1:
                mel = rearrange(
                    mel[:, :-1], "b (t f) d -> b t (f d)", f=self.fine_frame_rate_ratio
                )
            if self.log_scale:
                mel = torch.log(mel + self.eps)
        synchronize()
        print(
            f"[muscriptor] mel-spec ({wav.shape[0]} × {wav.shape[-1]} samples): "
            f"{time.perf_counter() - t0:.3f}s"
        )
        return mel

    def forward(self, x: WavCondition) -> ConditionType:
        _, lengths, *_ = x
        with torch.no_grad():
            embeds = self._mel_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None:
            lengths = lengths / (self.sample_rate // self.frame_rate)
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()
        else:
            mask = torch.ones_like(embeds[..., 0])
        mask_f = mask.float().unsqueeze(-1).to(embeds.device)
        embeds = embeds * mask_f
        return embeds, mask


class ClassConditioner(nn.Module):
    """Conditioner that embeds class indices (e.g., instrument group, dataset name)."""

    def __init__(
        self,
        num_classes: int,
        output_dim: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.embed = nn.Embedding(num_classes + 1, output_dim).to(device)
        self.pad_idx = 0

    def tokenize(self, x: list[str | None]) -> torch.Tensor:
        int_x = [list(map(int, s.split())) if s is not None else [-1] for s in x]
        max_len = max(len(xi) for xi in int_x)
        int_x = [xi + [-1] * (max_len - len(xi)) for xi in int_x]
        int_x = 1 + torch.LongTensor(int_x).to(self.device)
        return int_x

    def forward(self, inputs: torch.Tensor) -> ConditionType:
        embeds = self.embed(inputs + 1)
        mask = torch.ones_like(embeds[..., 0])
        return embeds, mask


def collate_wavs(
    samples: list[ConditioningAttributes], wav_conditions: list[str]
) -> dict[str, WavCondition]:
    """Collate wav conditions from a list of ConditioningAttributes."""
    wavs = defaultdict(list)
    lengths = defaultdict(list)
    sample_rates: dict[str, list] = defaultdict(list)
    paths: dict[str, list] = defaultdict(list)
    seek_times: dict[str, list] = defaultdict(list)
    out: dict[str, WavCondition] = {}

    for sample in samples:
        for attribute in wav_conditions:
            wav, length, sample_rate, path, seek_time = sample.wav[attribute]
            assert wav.dim() == 3
            B, K, T = wav.shape
            assert B == 1
            if K == 2:
                wav = wav.mean(1, keepdim=True)
            wavs[attribute].append(wav)
            lengths[attribute].append(length)
            sample_rates[attribute].extend(sample_rate)
            paths[attribute].extend(path)
            seek_times[attribute].extend(seek_time)

    for attribute in wav_conditions:
        # Stack along batch dim
        all_wavs = wavs[attribute]
        max_len = max(w.shape[-1] for w in all_wavs)
        padded = torch.cat(
            [F.pad(w, (0, max_len - w.shape[-1])) for w in all_wavs], dim=0
        )
        out[attribute] = WavCondition(
            padded,
            torch.cat(lengths[attribute]),
            sample_rates[attribute],
            paths[attribute],
            seek_times[attribute],
        )
    return out


class ConditioningProvider(nn.Module):
    """Runs all conditioners and returns a dict of condition tensors."""

    def __init__(
        self,
        conditioners: dict[str, nn.Module],
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [
            k for k, v in self.conditioners.items() if isinstance(v, ClassConditioner)
        ]

    @property
    def wav_conditions(self):
        return [
            k
            for k, v in self.conditioners.items()
            if isinstance(v, MelSpectrogramConditioner)
        ]

    def tokenize(self, inputs: list[ConditioningAttributes]) -> dict[str, Any]:
        output = {}
        # Collate text conditions
        text_batch: dict[str, list[str | None]] = defaultdict(list)
        for sample in inputs:
            for cond in self.text_conditions:
                text_batch[cond].append(sample.text.get(cond))
        for attr, batch in text_batch.items():
            output[attr] = self.conditioners[attr].tokenize(batch)

        # Collate wav conditions
        if self.wav_conditions:
            wav_batch = collate_wavs(inputs, self.wav_conditions)
            for attr, wav_cond in wav_batch.items():
                output[attr] = self.conditioners[attr].tokenize(wav_cond)

        return output

    def forward(self, tokenized: dict[str, Any]) -> dict[str, ConditionType]:
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output


DRUM_PROGRAM = 128
MINIMUM_NOTE_DURATION_SEC = 0.01


@dataclass
class Note:
    is_drum: bool
    program: int  # MIDI program number (0-127); 128 for drum
    onset: float  # onset time in seconds
    offset: float  # offset time in seconds (== onset for drums)
    pitch: int  # MIDI note number (0-127)


@dataclass
class NoteEvent:
    is_drum: bool
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    time: float  # absolute time in seconds
    velocity: int  # 1 for onset, 0 for offset; drum has no offset
    pitch: int  # MIDI pitch


@dataclass
class TieNoteEvent:
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    pitch: int  # MIDI pitch


@dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int  # inclusive


@dataclass
class Event:
    type: str
    value: int


def sort_notes(notes: list[Note]):
    if len(notes) > 0:
        notes.sort(key=lambda n: (n.onset, n.is_drum, n.program, n.pitch, n.offset))


def sort_note_events(note_events: list[NoteEvent]):
    if len(note_events) > 0:
        note_events.sort(
            key=lambda n: (n.time, n.is_drum, n.program, n.velocity, n.pitch)
        )


def sort_tie_note_events(tie_note_events: list[TieNoteEvent]):
    if len(tie_note_events) > 0:
        tie_note_events.sort(key=lambda n: (n.program, n.pitch))


def validate_notes(
    notes: list[Note],
    minimum_offset: float | None = MINIMUM_NOTE_DURATION_SEC,
    fix: bool = True,
) -> list[Note]:
    if len(notes) > 0:
        for note in list(notes):
            if note.onset is None and fix:
                notes.remove(note)
            elif note.offset is None and fix:
                note.offset = note.onset + minimum_offset
            elif note.onset > note.offset:
                if fix:
                    note.offset = max(note.offset, note.onset + minimum_offset)
            elif note.is_drum is False and note.offset - note.onset < 0.01 and fix:
                note.offset = note.onset + minimum_offset
    return notes


def trim_overlapping_notes(notes: list[Note], sort: bool = True) -> list[Note]:
    if len(notes) <= 1:
        return notes
    trimmed_notes = []
    channels = set((note.program, note.pitch, note.is_drum) for note in notes)
    for program, pitch, is_drum in channels:
        channel_notes = [
            n
            for n in notes
            if n.pitch == pitch and n.program == program and n.is_drum == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda n: n.onset)
        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].offset > sorted_notes[i].onset:
                sorted_notes[i - 1].offset = sorted_notes[i].onset
        valid_notes = [n for n in sorted_notes if n.onset < n.offset]
        trimmed_notes.extend(valid_notes)
    if sort:
        sort_notes(trimmed_notes)
    return trimmed_notes


# Special tokens occupy the first indices of the vocabulary, in this order.
SPECIAL_TOKENS = ("PAD", "EOS", "UNK")


def build_event_vocab(max_shift_steps: int) -> list[Event]:
    """Return the token-index → :class:`Event` decode table.

    Index ``i`` maps to the event the model emits at token ``i``. The layout
    is fixed: special tokens, then ``shift``, then the note-event ranges.
    """
    ranges = (
        [EventRange(token, 0, 0) for token in SPECIAL_TOKENS]
        + [EventRange("shift", 0, max_shift_steps - 1)]
        + [
            EventRange("pitch", 0, 127),
            EventRange("velocity", 0, 1),
            EventRange("tie", 0, 0),
            EventRange("program", 0, 129),
            EventRange("drum", 0, 127),
        ]
    )
    vocab: list[Event] = []
    for er in ranges:
        for value in range(er.min_value, er.max_value + 1):
            vocab.append(Event(type=er.type, value=value))
    return vocab


def note_event2note(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    shorten_notes_above_n_sec: int = 10,
    fix_broken_notes: bool = True,
    trim_overlap: bool = True,
    force_offset_past_segment_end: float | None = None,
    force_onset_before_segment_start: float | None = None,
) -> tuple[list[Note], Counter]:
    notes: list[Note] = []
    active_note_events: dict[tuple[int, int], NoteEvent | TieNoteEvent] = {}
    err_cnt: Counter = Counter()

    if tie_note_events is not None:
        for ne in tie_note_events:
            active_note_events[(ne.program, ne.pitch)] = ne

    sort_note_events(note_events)
    for ne in note_events:
        try:
            if ne.time is None:
                continue
            elif ne.is_drum:
                if ne.velocity == 1:
                    notes.append(
                        Note(
                            is_drum=True,
                            program=DRUM_PROGRAM,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC,
                            pitch=ne.pitch,
                        )
                    )
                else:
                    continue
            else:
                active_ne = active_note_events.pop((ne.program, ne.pitch), None)
                if ne.velocity == 0 and active_ne is None:
                    raise ValueError("Err/onset not found")
                if active_ne is not None:
                    if type(active_ne) is NoteEvent:
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=active_ne.time,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                    else:  # TieNoteEvent
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=force_onset_before_segment_start,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                if ne.velocity == 1:
                    active_note_events[(ne.program, ne.pitch)] = ne
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    for ne in active_note_events.values():
        try:
            if type(ne) is NoteEvent and ne.velocity == 1:
                if ne.program is None or ne.pitch is None:
                    raise ValueError("Err/active ne incomplete")
                elif ne.time is None:
                    continue
                else:
                    notes.append(
                        Note(
                            is_drum=False,
                            program=ne.program,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC
                            if force_offset_past_segment_end is None
                            else force_offset_past_segment_end,
                            pitch=ne.pitch,
                        )
                    )
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    if shorten_notes_above_n_sec > 0:
        for n in list(notes):
            try:
                if n.offset - n.onset > shorten_notes_above_n_sec:
                    n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
                    raise ValueError(f"Err/long note > {shorten_notes_above_n_sec}s")
            except ValueError as ve:
                err_cnt[str(ve)] += 1
    if fix_broken_notes:
        notes = validate_notes(notes, fix=True)
    if trim_overlap:
        notes = trim_overlapping_notes(notes, sort=True)
    else:
        sort_notes(notes)
    return notes, err_cnt


def note2note_event(notes: list[Note]) -> list[NoteEvent]:
    note_events = []
    for note in notes:
        if note.program == 1024:
            note.is_drum = True
        note_events.append(
            NoteEvent(note.is_drum, note.program, note.onset, 1, note.pitch)
        )
        if not note.is_drum:
            note_events.append(
                NoteEvent(note.is_drum, note.program, note.offset, 0, note.pitch)
            )
    sort_note_events(note_events)
    return note_events


def note_event2midi(
    note_events: list[NoteEvent],
    output_file: str | os.PathLike | None = None,
    velocity: int = 100,
    ticks_per_beat: int = 480,
    tempo: int = 500000,
    program_names: dict[int, str] | None = None,
    beats_per_bar: int | None = None,
    offset_s: float = 0.0,
) -> MidiFile:
    """Convert NoteEvent list to a type-1 (multi-track) MIDI file.

    Each program gets its own named track so DAWs that split imports by
    track (e.g. Ableton, which ignores channels/programs) keep the
    instruments apart. Channel assignments match the earlier type-0 layout:
    programs claim channels 0-8 then 10-15 in order of first appearance
    (sharing 15 on overflow), drums live on channel 9.

    `program_names` maps a program number (DRUM_PROGRAM for drums) to the
    track name; unmapped programs fall back to "program <n>" / "drums".

    `beats_per_bar` writes a time signature (denominator 4).
    `offset_s` delays every event so bar lines land on real downbeats.
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    meta_track = MidiTrack()
    meta_track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    if beats_per_bar is not None:
        meta_track.append(
            MetaMessage(
                "time_signature", numerator=beats_per_bar, denominator=4, time=0
            )
        )
    if offset_s:
        # Doesn't do anything for the MIDI but we just mark "we had to shift by this
        # much to align the bars". Read by /auralize later
        meta_track.append(
            MetaMessage("marker", text=f"{BAR_OFFSET_MARKER}{offset_s:.4f}", time=0)
        )
    midi.tracks.append(meta_track)

    drum_offset_events = []
    for ne in note_events:
        if ne.is_drum:
            drum_offset_events.append(
                NoteEvent(
                    is_drum=True,
                    program=ne.program,
                    time=ne.time + 0.01,
                    pitch=ne.pitch,
                    velocity=0,
                )
            )
    note_events = list(note_events) + drum_offset_events
    sort_note_events(note_events)

    program_names = program_names or {}
    program_to_channel: dict[int, int] = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    tracks: dict[int, MidiTrack] = {}
    track_ticks: dict[int, int] = {}
    current_tick = 0
    for ne in note_events:
        absolute_tick = round(second2tick(ne.time + offset_s, ticks_per_beat, tempo))
        if absolute_tick < current_tick:
            raise ValueError(
                f"at ne.time {ne.time}, absolute_tick {absolute_tick} < current_tick {current_tick}"
            )
        current_tick = absolute_tick

        key = DRUM_PROGRAM if (ne.is_drum or ne.program == DRUM_PROGRAM) else ne.program
        if key not in tracks:
            track = MidiTrack()
            midi.tracks.append(track)
            tracks[key] = track
            track_ticks[key] = 0
            if key == DRUM_PROGRAM:
                ne_channel = 9
                name = program_names.get(key, "drums")
                gm_program = 0
            else:
                try:
                    ne_channel = available_channels.pop(0)
                except IndexError:
                    ne_channel = 15
                name = program_names.get(key, f"program {key}")
                gm_program = ne.program
            program_to_channel[key] = ne_channel
            track.append(MetaMessage("track_name", name=name, time=0))
            # MuseScore ignores set_tempo in a conductor track that has no notes,
            # so repeat it here. Harmless for hosts that read the meta track.
            track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
            track.append(
                Message(
                    "program_change", program=gm_program, time=0, channel=ne_channel
                )
            )
        track = tracks[key]
        ne_channel = program_to_channel[key]
        delta_tick = absolute_tick - track_ticks[key]
        track_ticks[key] = absolute_tick

        msg_note = "note_on" if ne.velocity > 0 else "note_off"
        msg_velocity = velocity if ne.velocity > 0 else 0
        track.append(
            Message(
                msg_note,
                note=ne.pitch,
                velocity=msg_velocity,
                time=delta_tick,
                channel=ne_channel,
            )
        )

    if output_file is not None:
        midi.save(output_file)
    return midi


logger = logging.getLogger(__name__)


def get_group_program_map(
    instrument_vocabulary: str,
    misc_programs: str,
    is_mt3: bool = False,
    include_drums: bool = False,
) -> dict[int, list[int]]:
    if instrument_vocabulary == "ONLY_PIANO":
        ret = {0: list(range(128))}
    elif instrument_vocabulary == "FULL":
        ret = {i: [i] for i in range(128)}
    elif instrument_vocabulary == "MT3_MIDI_PLUS":
        ret = {
            0: list(range(8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
            4: list(range(32, 40)),
            5: list(range(40, 56)),
            6: list(range(56, 64)),
            7: list(range(64, 72)),
            8: list(range(72, 80)),
            9: list(range(80, 88)),
            10: list(range(88, 96)),
            11: list(range(100, 102)),
        }
    elif instrument_vocabulary == "MT3_FULL_PLUS":
        ret = {
            0: [0, 1, 3, 6, 7],
            1: [2, 4, 5],
            2: list(range(8, 16)),
            3: list(range(16, 24)),
            4: [24, 25],
            5: [26, 27, 28],
            6: [29, 30, 31],
            7: [32, 35],
            8: [33, 34, 36, 37, 38, 39],
            9: [40],
            10: [41],
            11: [42],
            12: [43],
            13: [46],
            14: [47],
            15: [48, 49, 44, 45],
            16: [50, 51],
            17: [52, 53, 54],
            18: [55],
            19: [56, 59],
            20: [57],
            21: [58],
            22: [60],
            23: [61, 62, 63],
            24: [64, 65],
            25: [66],
            26: [67],
            27: [68],
            28: [69],
            29: [70],
            30: [71],
            31: list(range(72, 80)),
            32: list(range(80, 88)),
            33: list(range(88, 96)),
            34: [100],
            35: [101],
        }
    elif instrument_vocabulary == "OURS_INSTRUMENT_GROUPS":
        ret = {
            0: list(range(8)),
            1: list(range(24, 32)),
            2: list(range(32, 40)),
            3: list(range(40, 56)),
            4: list(range(56, 64)),
            5: list(range(16, 24)) + list(range(64, 80)),
            6: list(range(80, 96)),
            7: list(range(8, 16)) + list(range(112, 119)),
        }
    else:
        assert False, instrument_vocabulary

    if instrument_vocabulary == "MT3_FULL_PLUS" and not is_mt3:
        not_assigned = set(range(130)) - set([v for vs in ret.values() for v in vs])
    else:
        not_assigned = set(range(128)) - set([v for vs in ret.values() for v in vs])
    if include_drums:
        not_assigned = not_assigned.union({DRUM_PROGRAM})
    if misc_programs == "ONE_GROUP":
        ret[len(ret)] = list(not_assigned)
    elif misc_programs == "SINGLETON_GROUPS":
        for p in not_assigned:
            ret[len(ret)] = [p]
    else:
        assert misc_programs == "OMIT", misc_programs
    return ret


# Human-readable names for the MT3_FULL_PLUS instrument groups (see
# get_group_program_map). Used by the CLI's --instruments option and the
# web app's /instruments endpoint. The group IDs index the model's learned
# program groups and must not change; only the user-facing names do.
# Notes the model still decodes into an omitted group surface as "program_<n>".
MT3_FULL_PLUS_GROUP_NAMES: dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 1,
    "chromatic_percussion": 2,
    "organ": 3,
    "acoustic_guitar": 4,
    "clean_electric_guitar": 5,
    "distorted_electric_guitar": 6,
    "acoustic_bass": 7,
    "electric_bass": 8,
    "violin": 9,
    "viola": 10,
    "cello": 11,
    "contrabass": 12,
    "orchestral_harp": 13,
    "timpani": 14,
    "string_ensemble": 15,
    "synth_strings": 16,
    "voice": 17,
    "orchestra_hit": 18,
    "trumpet": 19,
    "trombone": 20,
    "tuba": 21,
    "french_horn": 22,
    "brass_section": 23,
    "soprano_and_alto_sax": 24,
    "tenor_sax": 25,
    "baritone_sax": 26,
    "oboe": 27,
    "english_horn": 28,
    "bassoon": 29,
    "clarinet": 30,
    "flutes": 31,
    "synth_lead": 32,
    "synth_pad": 33,
    "drums": 36,
}


def instrument_group_from_names(names: Iterable[str]) -> str:
    """Map exact instrument group names to the model's conditioning string.

    The strict counterpart of :func:`resolve_instrument_names`: every name
    must appear verbatim in ``MT3_FULL_PLUS_GROUP_NAMES``. Raises ValueError
    listing the unknown names otherwise.
    """
    names = list(names)
    unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
    if unknown:
        raise ValueError(
            f"unknown instrument name(s): {', '.join(map(repr, unknown))}; "
            f"valid names: {', '.join(MT3_FULL_PLUS_GROUP_NAMES)}"
        )
    return " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in names)


def resolve_instrument_names(tokens: Iterable[str]) -> list[str]:
    """Resolve loosely-typed instrument tokens to canonical group names.

    Matching is case-insensitive; a token that is not an exact name may be
    any substring that matches exactly one group name (``"timp"`` →
    ``"timpani"``). Raises ValueError when a token is ambiguous (listing the
    candidates) or matches nothing (suggesting close spellings).
    """
    resolved = []
    for token in tokens:
        t = token.strip().lower()
        if t in MT3_FULL_PLUS_GROUP_NAMES:
            resolved.append(t)
            continue
        hits = [n for n in MT3_FULL_PLUS_GROUP_NAMES if t in n]
        if len(hits) == 1:
            resolved.append(hits[0])
        elif hits:
            raise ValueError(
                f"ambiguous instrument name {token!r}: "
                f"matches {', '.join(hits)}"
            )
        else:
            # Compare against each name AND its underscore-separated words,
            # so a typo like "pinao" still surfaces "acoustic_piano".
            def closeness(name: str) -> float:
                return max(
                    difflib.SequenceMatcher(None, t, part).ratio()
                    for part in (name, *name.split("_"))
                )

            ranked = sorted(MT3_FULL_PLUS_GROUP_NAMES, key=closeness, reverse=True)
            suggestions = [n for n in ranked[:3] if closeness(n) >= 0.6]
            hint = (
                f" — did you mean {', '.join(suggestions)}?"
                if suggestions
                else ""
            )
            raise ValueError(f"unknown instrument name {token!r}{hint}")
    return resolved


class MT3Tokenizer:
    def __init__(
        self,
        instrument_vocabulary: str = "FULL",
        max_shift_steps: int = 1001,
        frame_rate: int = 100,
    ):
        self.group_program_map = get_group_program_map(
            instrument_vocabulary, misc_programs="SINGLETON_GROUPS", is_mt3=True
        )
        self.frame_rate = frame_rate
        self._vocab = build_event_vocab(max_shift_steps)
        self._token_index = {(e.type, e.value): i for i, e in enumerate(self._vocab)}
        self.num_tokens = len(self._vocab)
        self.eos_id = SPECIAL_TOKENS.index("EOS")

        logger.info(f"MT3Tokenizer: {self.num_tokens} tokens")

    def tie_section_token_ids(
        self, open_note_keys: Iterable[tuple[int, int]]
    ) -> list[int]:
        """Encode a tie prologue declaring ``open_note_keys`` as sustained.

        ``open_note_keys`` are the ``(program, pitch)`` pairs of notes still
        sounding at a chunk boundary. The layout matches the training encoder
        (``note_event2event``): pairs sorted by (program, pitch), each program
        token emitted once for its run of pitches, terminated by the ``tie``
        token. Teacher-forcing these as the start of a chunk pins the model's
        tie section to the notes actually sustained from the previous chunk.
        """
        tokens: list[int] = []
        program_state: int | None = None
        for program, pitch in sorted(open_note_keys):
            if program != program_state:
                tokens.append(self._token_index[("program", program)])
                program_state = program
            tokens.append(self._token_index[("pitch", pitch)])
        tokens.append(self._token_index[("tie", 0)])
        return tokens

    def forbidden_token_ids(self, instruments: Iterable[str]) -> list[int]:
        """Token ids that must never be sampled when only ``instruments`` may
        appear in the transcription (the hard counterpart of the advisory
        instrument_group conditioning).

        ``instruments`` are exact MT3_FULL_PLUS group names (so this only makes
        sense on a tokenizer built with that vocabulary). A ``program`` token is
        forbidden unless it decodes to one of the given groups — i.e. it is the
        representative (first) program of an allowed group; ``drum`` tokens are
        forbidden unless "drums" is listed. Timing, pitch, velocity, tie and
        special tokens are never forbidden. Raises ValueError on unknown names.
        """
        names = list(instruments)
        unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise ValueError(
                f"unknown instrument name(s): {', '.join(map(repr, unknown))}; "
                f"valid names: {', '.join(MT3_FULL_PLUS_GROUP_NAMES)}"
            )
        allow_drums = "drums" in names
        # Same representative-program convention as decoding
        # (transcription_model._build_instrument_for_program): the model emits
        # the first program of a group, so only that program is allowed.
        allowed_programs = set()
        for name in names:
            if name == "drums":
                continue
            gid = MT3_FULL_PLUS_GROUP_NAMES[name]
            if gid in self.group_program_map and self.group_program_map[gid]:
                allowed_programs.add(self.group_program_map[gid][0])
        forbidden = []
        for token_id, event in enumerate(self._vocab):
            if event.type == "program" and event.value not in allowed_programs:
                forbidden.append(token_id)
            elif event.type == "drum" and not allow_drums:
                forbidden.append(token_id)
        return forbidden


# Written when no grid was detected: 120 BPM and no time signature, leaving the
# meter for notation software to guess.
PLACEHOLDER_GRID = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0)


def shifted_notes(notes: list[Note], delay_s: float) -> list[Note]:
    """`notes` moved by `delay_s`. May go negative; the bar offset covers that."""
    if not delay_s:
        return notes
    return [
        dataclasses.replace(
            note, onset=note.onset + delay_s, offset=note.offset + delay_s
        )
        for note in notes
    ]


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
    quantize: bool = False,
):
    """Convert a list of Note objects to a mido MidiFile.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).

    `grid` is a detected beat grid (ver detect_grid más arriba en este mismo fichero): it supplies the
    tempo, the time signature and a delay that puts bar lines on real downbeats.
    Defaults to PLACEHOLDER_GRID.

    The notes are moved onto the beat grid based on its `onset_delay` to better match
    the grid.

    `quantize` additionally snaps every note onto the beat subdivision, if present
    in the passed `beat_grid`.
    """
    beat_grid = beat_grid or PLACEHOLDER_GRID
    if beat_grid.onset_delay is None:
        beat_grid = beat_grid.with_onset_delay([n.onset for n in notes])
    delay = beat_grid.onset_delay
    offset = beat_grid.bar_offset(min_shift=delay)
    notes = shifted_notes(notes, -delay)

    if quantize and beat_grid.beat_subdivision is not None:
        step = 60.0 / beat_grid.bpm / beat_grid.beat_subdivision
        notes = quantized_notes(notes, step, offset)

    return note_event2midi(
        note2note_event(notes),
        output_file=None,
        velocity=velocity,
        tempo=round(60_000_000 / beat_grid.bpm),
        program_names=program_names,
        beats_per_bar=beat_grid.beats_per_bar,
        offset_s=offset,
    )


def quantized_notes(notes: list[Note], step: float, offset_s: float) -> list[Note]:
    """`notes` with every onset and offset moved onto a `step`-second grid.

    Snapped in the timeline the MIDI file will have — `offset_s` is the shift
    that puts bar 1 at tick 0 — so the notes land on whole grid steps from the
    start of the score rather than a constant fraction off it.
    """

    def snap(t: float) -> float:
        return round((t + offset_s) / step) * step - offset_s

    snapped = []
    for note in notes:
        onset = snap(note.onset)
        snapped.append(
            dataclasses.replace(
                note,
                onset=onset,
                # A note shorter than half a step rounds to nothing and would
                # vanish from the score; give it the shortest length the grid has.
                offset=max(snap(note.offset), onset + step),
            )
        )
    # Bumping the short ones can make two notes of the same pitch overlap.
    return trim_overlapping_notes(snapped)


_DRUM_INSTRUMENT = "drums"


@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


@dataclass
class ProgressEvent:
    """A coarse transcription-progress signal, woven into the event stream.

    Marks that ``completed`` of ``total`` fixed-size audio chunks have been
    transcribed (``completed == 0`` is emitted once up front so consumers learn
    ``total`` and get a timing baseline; ``completed == total`` marks the end).
    These are deliberately coarse anchors — the frontend smooths between them
    and derives an ETA, since wall-clock time per chunk is only observable
    there. Advisory only: consumers that build notes/MIDI ignore them.
    """

    completed: int
    total: int


@dataclass
class ChunkBoundary:
    """Marks the start of a new model-output chunk in the token stream.

    ``seek_time`` is the chunk's start time in seconds; ``next_seek_time`` is
    the following chunk's start (``None`` for the last chunk), used to drop
    events the model emits past its window.
    """

    seek_time: float
    next_seek_time: float | None


@dataclass
class _StartNote:
    """A note opens: (program, pitch) starts sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _EndNote:
    """An open note closes: (program, pitch) stops sounding at `time`."""

    program: int
    pitch: int
    time: float


@dataclass
class _DrumHit:
    """An instantaneous drum hit at `time`; never enters the open set."""

    pitch: int
    time: float


_NoteAction = _StartNote | _EndNote | _DrumHit


class OpenNoteTracker:
    """The chunk-decoding state machine for the model's token stream.

    :meth:`feed` consumes the interleaved :class:`ChunkBoundary` markers and
    token indices and returns the note actions they imply; :meth:`finish`
    flushes the end-of-stream closes. All decode rules live here — the tie
    prologue (open notes absent from the tie set close at the boundary),
    malformed chunks (a shift before the ``tie`` token closes everything and
    drops the rest), the next_seek_time window, and retriggers.

    Two consumers share it: :func:`decode_model_tokens` turns the actions into
    indexed NoteStart/NoteEnd events, and the prelude-forcing path
    (``TranscriptionModel._generate_token_stream``) ignores the actions and
    reads :meth:`open_keys` at chunk boundaries — the ``(program, pitch)``
    pairs the next chunk's tie prologue must declare as sustained (see
    ``MT3Tokenizer.tie_section_token_ids``). One state machine serving both
    keeps decoding and forcing consistent by construction.
    """

    def __init__(self, vocab: list[Event], frame_rate: int = 100):
        self._vocab = vocab
        self._frame_rate = frame_rate
        # (program, pitch) -> onset time. Insertion-ordered: end-of-stream
        # closes replay in onset order.
        self._open: dict[tuple[int, int], float] = {}
        # Per-chunk state, reset at every ChunkBoundary.
        self._seek_time = 0.0
        self._next_seek_time: float | None = None
        self._start_tick = 0
        self._tick_state = 0
        self._program: int | None = None
        self._velocity: int | None = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()
        self._chunk_started = False

    def feed(self, item: "int | ChunkBoundary") -> list[_NoteAction]:
        if isinstance(item, ChunkBoundary):
            actions: list[_NoteAction] = []
            # If the previous chunk never closed its tie prologue (malformed:
            # no `tie` token before it ended), treat its tie set as empty so
            # every still-open note ends at that chunk's boundary.
            if self._chunk_started and self._in_prologue:
                actions = self._end_all(self._seek_time)
            self._seek_time = item.seek_time
            self._next_seek_time = item.next_seek_time
            self._start_tick = round(item.seek_time * self._frame_rate)
            self._tick_state = self._start_tick
            self._program = None
            self._velocity = None
            self._in_prologue = True
            self._skip_rest = False
            self._tie_set = set()
            self._chunk_started = True
            return actions

        event = self._vocab[item]
        etype = event.type

        if self._in_prologue:
            if etype == "tie":
                # End of the tie section: close prior notes not sustained here.
                self._in_prologue = False
                self._velocity = None
                ended = [k for k in self._open if k not in self._tie_set]
                for key in ended:
                    del self._open[key]
                return [_EndNote(*key, self._seek_time) for key in ended]
            if etype == "shift":
                # No tie token: the chunk is malformed. Close all open notes at
                # the boundary and drop the rest of the chunk.
                self._in_prologue = False
                self._skip_rest = True
                return self._end_all(self._seek_time)
            if etype == "program":
                self._program = event.value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, event.value))
            return []

        if self._skip_rest:
            return []

        if etype == "shift":
            if event.value > 0:
                self._tick_state = self._start_tick + event.value
        elif etype == "program":
            self._program = event.value
        elif etype == "velocity":
            self._velocity = event.value
        elif etype == "drum":
            time = self._tick_state / self._frame_rate
            if self._next_seek_time is None or time < self._next_seek_time:
                return [_DrumHit(event.value, time)]
        elif etype == "pitch":
            if self._program is None or self._velocity is None:
                return []
            time = self._tick_state / self._frame_rate
            if self._next_seek_time is not None and time >= self._next_seek_time:
                return []
            key = (self._program, event.value)
            actions = []
            if key in self._open:
                del self._open[key]
                actions.append(_EndNote(*key, time))
            if self._velocity > 0:
                self._open[key] = time
                actions.append(_StartNote(*key, time))
            return actions
        return []

    def finish(self) -> list[_NoteAction]:
        """End of stream: close anything still open.

        A well-formed final chunk uses the minimum-duration fallback; a chunk
        that ended mid-prologue closes at its boundary (matching the
        malformed-chunk rule in :meth:`feed`).
        """
        if self._chunk_started and self._in_prologue:
            return self._end_all(self._seek_time)
        actions = [
            _EndNote(*key, onset + MINIMUM_NOTE_DURATION_SEC)
            for key, onset in self._open.items()
        ]
        self._open.clear()
        return actions

    def _end_all(self, time: float) -> list[_NoteAction]:
        actions: list[_NoteAction] = [_EndNote(*key, time) for key in self._open]
        self._open.clear()
        return actions

    def open_keys(self) -> list[tuple[int, int]]:
        """Sorted ``(program, pitch)`` pairs currently held open."""
        return sorted(self._open)


def decode_model_tokens(
    stream: Iterator[int | ChunkBoundary | ProgressEvent],
    vocab: list[Event],
    instrument_for_program: Callable[[int], str],
    frame_rate: int = 100,
) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
    """Stream model token indices straight into NoteStart/NoteEnd events.

    ``stream`` interleaves :class:`ChunkBoundary` markers with token indices:
    each boundary starts a new chunk, followed by that chunk's tokens (EOS and
    anything after it already stripped). Tokens are consumed strictly in
    order: no buffering, no end-of-chunk sort. Each chunk begins with a *tie
    prologue* — ``(program, pitch)`` pairs for notes sustained from the
    previous chunk, terminated by a ``tie`` token — after which any prior open
    note not in that tie set is closed at the chunk boundary. The rest of the
    chunk drives note onsets/offsets directly.

    The decode rules themselves live in :class:`OpenNoteTracker`; this
    generator only turns its actions into events — minting indices, naming
    instruments, and pairing every NoteEndEvent with its NoteStartEvent.
    """
    tracker = OpenNoteTracker(vocab, frame_rate)
    open_notes: dict[tuple[int, int], NoteStartEvent] = {}
    next_index = 0

    def mint(pitch: int, start_time: float, instrument: str) -> NoteStartEvent:
        nonlocal next_index
        ev = NoteStartEvent(
            pitch=pitch, start_time=start_time, index=next_index, instrument=instrument
        )
        next_index += 1
        return ev

    def events_for(
        actions: list[_NoteAction],
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        for action in actions:
            if isinstance(action, _EndNote):
                start = open_notes.pop((action.program, action.pitch))
                yield NoteEndEvent(end_time=action.time, start_event=start)
            elif isinstance(action, _StartNote):
                start = mint(
                    action.pitch, action.time, instrument_for_program(action.program)
                )
                open_notes[(action.program, action.pitch)] = start
                yield start
            else:  # _DrumHit: an instantaneous start/end pair
                start = mint(action.pitch, action.time, _DRUM_INSTRUMENT)
                yield start
                yield NoteEndEvent(
                    end_time=action.time + MINIMUM_NOTE_DURATION_SEC, start_event=start
                )

    for item in stream:
        if isinstance(item, ProgressEvent):
            # Advisory progress signal — pass straight through, untouched by
            # the decode state machine.
            yield item
            continue
        yield from events_for(tracker.feed(item))
    yield from events_for(tracker.finish())


logger = logging.getLogger(__name__)
ConditionTensors = dict[str, ConditionType]


# ---------------------------------------------------------------------------
# ScaledEmbedding  (used for token embeddings, keeps weight compatible with ckpt)
# ---------------------------------------------------------------------------


class ScaledEmbedding(nn.Embedding):
    """Embedding that maps zero_idx (a negative index) to a zero vector."""

    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        return torch.where(is_zero[..., None], torch.zeros_like(y), y)


# ---------------------------------------------------------------------------
# TorchAutocast
# ---------------------------------------------------------------------------


class TorchAutocast:
    """Minimal autocast context manager (matches the audiocraft interface)."""

    def __init__(
        self,
        enabled: bool = False,
        device_type: str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.enabled = enabled
        self.device_type = device_type
        self.dtype = dtype
        self._ctx = None

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled and self._ctx is not None:
            self._ctx.__exit__(*args)


# ---------------------------------------------------------------------------
# LMModel
# ---------------------------------------------------------------------------


class LMModel(nn.Module):
    """Causal transformer LM for MIDI token generation.

    Single-stream
    Supports classifier-free guidance at inference time.
    """

    def __init__(
        self,
        condition_provider: ConditioningProvider,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        cfg_coef: float = 1.0,
        autocast: TorchAutocast | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_provider = condition_provider
        self.card = card
        self.dim = dim
        self.cfg_coef = cfg_coef
        self.autocast = (
            autocast if autocast is not None else TorchAutocast(enabled=False)
        )

        self.emb = ScaledEmbedding(
            self.card + 1,
            dim,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )

        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.linear = nn.Linear(dim, card, bias=False)

    # ------------------------------------------------------------------
    # Token ID properties
    # ------------------------------------------------------------------

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence: torch.Tensor,  # [B, S]
        condition_tensors: ConditionTensors,
        first_step: bool = False,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:  # [B, S, card]
        B, S = sequence.shape

        input_ = self.emb(sequence)  # [B, S, D]

        prepend_length = 0
        if first_step:
            for cond, _ in condition_tensors.values():
                # Conditioners run in fp32 even when the transformer runs in
                # half precision (mel numerics degrade in fp16) — cast at the
                # seam.
                input_ = torch.cat([cond.to(input_.dtype), input_], dim=1)
            prepend_length = input_.shape[1] - S

        transformer_out = self.transformer(
            input_,
            prepend_length=prepend_length,
            model_state=model_state,
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)

        # Remove prepended conditioning tokens
        if prepend_length > 0:
            transformer_out = transformer_out[:, -S:]

        logits = self.linear(transformer_out)
        return logits  # [B, S, card]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _compute_logits(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, card]
        """Run the forward pass and return masked logits at the last timestep."""
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        if cfg_coef == 1.0:
            logits = self(
                sequence,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
        else:
            doubled = torch.cat([sequence, sequence], dim=0)
            all_logits = self(
                doubled,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        logits = logits[:, -1, :].float()  # [B, card] — last timestep
        logits[:, 1393:] = -torch.inf      # mask reserved / OOV tokens
        if forbidden_tokens is not None:
            logits[:, forbidden_tokens] = -torch.inf
        return logits

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B]
        logits = self._compute_logits(
            sequence, cfg_conditions, model_state, first_step, cfg_coef,
            forbidden_tokens=forbidden_tokens,
        )
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            next_tokens = sample_from_probs(probs, top_p=top_p, top_k=top_k)[:, 0]
        else:
            next_tokens = torch.argmax(logits, dim=-1)  # [B]
        return next_tokens  # [B]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.Tensor | None = None,
        conditions: list[ConditioningAttributes] = [],
        num_samples: int | None = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        early_stop_on_token: int | None = None,
        beam_size: int = 1,
        beam_length_score_alpha: float = 0.75,
        forbidden_tokens: torch.Tensor | list[int] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Autoregressively generate tokens, yielding one timestep at a time.

        Each yield is a ``[num_samples]`` tensor. For beam_size == 1 (default),
        tokens are yielded as they are generated. For beam_size > 1, beam search
        is run non-streamingly and all tokens are yielded at the end.

        ``forbidden_tokens`` are token ids whose logits are forced to -inf at
        every step, so they can never be sampled (greedy, sampling or beam).
        """
        assert not self.training
        if beam_size > 1:
            assert early_stop_on_token is not None, "beam search requires early_stop_on_token"
        device = self.emb.weight.device

        if forbidden_tokens is not None and not isinstance(
            forbidden_tokens, torch.Tensor
        ):
            forbidden_tokens = torch.tensor(
                forbidden_tokens, device=device, dtype=torch.long
            )

        if num_samples is None:
            num_samples = (
                len(conditions)
                if conditions
                else (prompt.shape[0] if prompt is not None else 1)
            )

        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        # Build condition tensors (with null conditions appended for CFG)
        if conditions:
            if cfg_coef == 1.0:
                prepared = self.condition_provider.tokenize(conditions)
                synchronize()
                _t = time.perf_counter()
                cfg_conditions: ConditionTensors = self.condition_provider(prepared)
                synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
                )
            else:
                null_conditions = nullify_all_conditions(conditions)
                all_conditions = conditions + null_conditions
                prepared = self.condition_provider.tokenize(all_conditions)
                print(
                    "[muscriptor] instrument_group tokens:",
                    prepared.get("instrument_group"),
                )
                print(
                    "[muscriptor] dataset_name tokens:    ",
                    prepared.get("dataset_name"),
                )
                synchronize()
                _t = time.perf_counter()
                cfg_conditions = self.condition_provider(prepared)
                synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
                )
        else:
            cfg_conditions = {}

        eff_batch = num_samples * beam_size

        # Expand conditions so each beam gets its own copy (interleaved for CFG).
        if beam_size > 1 and cfg_conditions:
            cfg_conditions = {
                k: (
                    torch.repeat_interleave(cond, beam_size, dim=0),
                    torch.repeat_interleave(mask, beam_size, dim=0),
                )
                for k, (cond, mask) in cfg_conditions.items()
            }

        # Initialise generation buffer (eff_batch rows = num_samples × beam_size)
        ungenerated = self.ungenerated_token_id
        gen_sequence = torch.full(
            (eff_batch, max_gen_len + 1),
            ungenerated,
            device=device,
            dtype=torch.long,
        )
        gen_sequence[:, 0] = self.initial_token_id

        start_offset = 0
        if prompt is not None:
            PT = prompt.shape[-1]
            if beam_size > 1:
                prompt = torch.repeat_interleave(prompt, beam_size, dim=0)
            gen_sequence[:, 1 : 1 + PT] = prompt
            ungenerated_steps = (gen_sequence == ungenerated).nonzero()[:, 1]
            start_offset = max(0, int(ungenerated_steps.amin()) - 1)

        prepend_length = sum(cond.shape[1] for cond, _ in cfg_conditions.values())
        cache_batch_size = eff_batch * (1 if cfg_coef == 1.0 else 2)
        cache_seq_len = prepend_length + max_gen_len
        model_state = init_states(
            self, batch_size=cache_batch_size, sequence_length=cache_seq_len
        )

        # Accumulated log-prob scores, one per beam row.
        beam_scores = torch.zeros(eff_batch, device=device, dtype=torch.float)

        # For greedy/sampling emit prompt steps now; beam search emits at the end.
        if beam_size == 1:
            for t in range(start_offset):
                yield gen_sequence[:, t + 1]

        last_offset = start_offset - 1
        with self.autocast:
            for offset in range(start_offset, max_gen_len):
                last_offset = offset
                first_iter = offset == start_offset
                input_ = (
                    gen_sequence[:, : offset + 1]
                    if first_iter
                    else gen_sequence[:, offset : offset + 1]
                )

                if beam_size == 1:
                    # ── Standard greedy / sampling path ──────────────────
                    if early_stop_on_token is not None:
                        done = (gen_sequence == early_stop_on_token).any(dim=1).all()
                        if done:
                            break

                    next_token = self._sample_next_token(
                        input_,
                        cfg_conditions,
                        model_state,
                        first_step=first_iter,
                        use_sampling=use_sampling,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                    )  # [B]

                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    this_gen_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(
                        this_gen_step == ungenerated, next_token, this_gen_step
                    )
                    gen_sequence[:, offset + 1] = next_token

                    yield gen_sequence[:, offset + 1]  # [num_samples]

                else:
                    # ── Beam search step ──────────────────────────────────
                    logits = self._compute_logits(
                        input_, cfg_conditions, model_state,
                        first_step=first_iter, cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                    )  # [eff_batch, card]
                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    log_probs = torch.log_softmax(logits.float(), dim=-1)

                    # Top beam_size candidate tokens per current beam
                    topk_scores, topk_tokens = torch.topk(log_probs, k=beam_size, dim=-1)

                    # Track which beams have already emitted EOS
                    eos_mask = gen_sequence == early_stop_on_token
                    beam_has_ended = eos_mask.any(dim=-1)
                    eos_pos = eos_mask.int().argmax(dim=-1).clamp(min=1)
                    beam_lengths = torch.where(
                        beam_has_ended, eos_pos,
                        torch.full_like(eos_pos, offset + 1),
                    )

                    # Finished beams: don't expand further
                    topk_scores = torch.where(
                        beam_has_ended.unsqueeze(-1),
                        torch.zeros_like(topk_scores),
                        topk_scores,
                    )

                    # Length-normalized candidate scores: [eff_batch, beam_size]
                    lp = 1.0 / (beam_lengths.float() ** beam_length_score_alpha)
                    cand = (beam_scores.unsqueeze(-1) + topk_scores) * lp.unsqueeze(-1)

                    # Reshape to [num_samples, beam_size²] for cross-beam selection
                    cand_2d = cand.reshape(num_samples, beam_size * beam_size)

                    if offset == start_offset:
                        # All beams identical at start — take first beam_size tokens
                        new_scores = cand_2d[:, :beam_size]
                        best_idx = (
                            torch.arange(beam_size, device=device)
                            .unsqueeze(0)
                            .expand(num_samples, -1)
                        )
                    else:
                        new_scores, best_idx = torch.topk(cand_2d, k=beam_size, dim=-1)

                    # Decode flat index → (prev_beam_within_sample, token_rank)
                    prev_local = (best_idx // beam_size).reshape(-1)
                    tok_rank = (best_idx % beam_size).reshape(-1)

                    # Map to global row indices in [eff_batch, …] tensors
                    sample_base = (
                        torch.arange(num_samples, device=device)
                        .repeat_interleave(beam_size) * beam_size
                    )
                    prev_global = sample_base + prev_local

                    # Token for each new beam
                    next_token = topk_tokens[prev_global, tok_rank]

                    # Update beam scores (store un-normalized for the next step)
                    beam_scores = new_scores.reshape(-1) / lp[prev_global]

                    # Reorder generation sequences to match winning beams
                    gen_sequence = gen_sequence[prev_global]

                    # Reorder KV caches — shape is [2, batch, T, heads, head_dim]
                    for state in model_state.values():
                        if "cache" in state:
                            cache = state["cache"]
                            if cache.shape[1] == 2 * eff_batch:  # CFG-doubled cache
                                reorder = torch.cat([prev_global, prev_global + eff_batch])
                            else:
                                reorder = prev_global
                            state["cache"] = cache[:, reorder, :, :, :]

                    # Write next token (respecting pre-filled prompt positions)
                    this_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(this_step == ungenerated, next_token, this_step)
                    gen_sequence[:, offset + 1] = next_token

                    # Early stop when every beam in every sample has emitted EOS
                    if (gen_sequence == early_stop_on_token).any(dim=-1).all():
                        break

        # Beam search: select best beam per sample and yield all tokens at once
        if beam_size > 1:
            best_beam = beam_scores.reshape(num_samples, beam_size).argmax(dim=-1)
            best_global = (
                torch.arange(num_samples, device=device) * beam_size + best_beam
            )
            best_sequence = gen_sequence[best_global]  # [num_samples, T]
            for t in range(last_offset + 1):
                yield best_sequence[:, t + 1]


@contextlib.contextmanager
def _timed(label: str, store: list[tuple[str, float]] | None = None):
    """Print and (optionally) record how long a block of work takes."""
    synchronize()
    t0 = time.perf_counter()
    yield
    synchronize()
    dt = time.perf_counter() - t0
    print(f"[muscriptor] {label}: {dt:.2f}s", file=sys.stderr)
    if store is not None:
        store.append((label, dt))


# Published model variants live at hf://MuScriptor/muscriptor-<size>. A bare
# size keyword ("small"/"medium"/"large") resolves to the matching repo; the
# architecture is then read from that repo's config.json (see _resolve_config).
_HF_REPO_TEMPLATE = "hf://MuScriptor/muscriptor-{size}/model.safetensors"
_MODEL_SIZES = ("small", "medium", "large")
_DEFAULT_SIZE = "medium"


def _resolve_source(weights_path: str | Path | None) -> str | Path:
    """Map a --model value to a weights location.

    A size keyword ("small"/"medium"/"large") — or None, which defaults to
    ``medium`` — becomes the corresponding HuggingFace repo URL. Anything else
    (a local path, an ``hf://`` or ``http(s)://`` URL) is passed through as-is.
    """
    if weights_path is None:
        weights_path = _DEFAULT_SIZE
    if isinstance(weights_path, str) and weights_path in _MODEL_SIZES:
        return _HF_REPO_TEMPLATE.format(size=weights_path)
    return weights_path


_SAMPLE_RATE = 16000
# Must match the segment duration used during training / evaluation.
_SEGMENT_DURATION = 5.0


@dataclass
class _ModelConfig:
    dim: int
    num_heads: int
    num_layers: int
    card: int


# Per-variant configs, keyed by the size that appears in the HF repo name
# (muscriptor-<size>). Each published repo also ships these values in its
# config.json; this table is the fallback when no config.json is present.
_CONFIGS: dict[str, _ModelConfig] = {
    "large": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
    "medium": _ModelConfig(dim=1024, num_heads=16, num_layers=24, card=1395),
    "small": _ModelConfig(dim=768, num_heads=12, num_layers=14, card=1393),
}

_DEFAULT_CONFIG = _CONFIGS["large"]

# Legacy local checkpoints identified by the 8-hex tag in their filename,
# mapped to the equivalent variant config.
_LEGACY_CONFIGS: dict[str, _ModelConfig] = {
    "01684fbb": _CONFIGS["large"],
    "0ac4ce03": _CONFIGS["small"],
    "8f59580c": _CONFIGS["medium"],
    "e84904c4": _CONFIGS["large"],
}

_CONFIG_FILENAME = "config.json"
_CONFIG_FIELDS = ("dim", "num_heads", "num_layers", "card")


def _config_from_json(path: Path) -> _ModelConfig:
    """Read a _ModelConfig from a HuggingFace-style config.json."""
    data = json.loads(path.read_text())
    return _ModelConfig(**{field: data[field] for field in _CONFIG_FIELDS})


def _resolve_config(source: str | Path, weights_path: Path) -> _ModelConfig:
    """Determine the model architecture for a set of weights.

    Resolution order, most to least authoritative:
      1. ``config.json`` sitting next to the weights — the self-describing,
         HuggingFace-idiomatic source of truth (local dir or hf:// repo).
      2. the ``muscriptor-<size>`` segment of an ``hf://`` repo name.
      3. the legacy 8-hex tag embedded in a local checkpoint filename.
    """
    config_path = weights_path.parent / _CONFIG_FILENAME
    if not config_path.exists():
        fetched = download_companion(source, _CONFIG_FILENAME)
        if fetched is not None:
            config_path = fetched
    if config_path.exists():
        return _config_from_json(config_path)

    m = re.search(r"muscriptor-(large|medium|small)", str(source))
    if m:
        return _CONFIGS[m.group(1)]

    m = re.search(r"_([0-9a-f]{8})_", weights_path.name)
    if m and m.group(1) in _LEGACY_CONFIGS:
        return _LEGACY_CONFIGS[m.group(1)]
    return _DEFAULT_CONFIG


def _remap_single_codebook_keys(state_dict: dict) -> dict:
    """Adapt legacy multi-codebook checkpoints to the single-stream LMModel.

    Older checkpoints store the token embedding and output head as the first
    entry of an ``nn.ModuleList`` (``emb.0.*`` / ``linears.0.*``). LMModel is
    single-stream, so those map to ``emb.*`` / ``linear.*``. Checkpoints with a
    second codebook (``emb.1.*`` etc.) are unsupported and rejected.
    """
    if any(k.startswith(("emb.1.", "linears.1.")) for k in state_dict):
        raise ValueError(
            "Checkpoint has more than one codebook (n_q > 1); "
            "only single-stream models are supported."
        )
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("emb.0."):
            key = "emb." + key[len("emb.0.") :]
        elif key.startswith("linears.0."):
            key = "linear." + key[len("linears.0.") :]
        remapped[key] = value
    return remapped


def _build_model(device: torch.device, cfg: _ModelConfig = _DEFAULT_CONFIG) -> LMModel:
    mel_cond = MelSpectrogramConditioner(
        output_dim=cfg.dim,
        device=device,
        sample_rate=_SAMPLE_RATE,
        n_fft=2048,
        frame_rate=100,
        n_mel_bins=512,
        log_scale=True,
        eps=1e-6,
        normalize_audio=False,
    )
    inst_cond = ClassConditioner(num_classes=1000, output_dim=cfg.dim, device=device)
    ds_cond = ClassConditioner(num_classes=4, output_dim=cfg.dim, device=device)

    condition_provider = ConditioningProvider(
        conditioners={
            "self_wav": mel_cond,
            "instrument_group": inst_cond,
            "dataset_name": ds_cond,
        },
        device=device,
    )

    # Disabled off-CUDA: on MPS half precision comes from native fp16 weights
    # (see load_model) — autocast there is measurably slower than fp32.
    autocast = TorchAutocast(enabled=False)
    if device.type == "cuda":
        autocast = TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)

    model = LMModel(
        condition_provider=condition_provider,
        card=cfg.card,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        hidden_scale=4,
        cfg_coef=1.0,
        autocast=autocast,
        # StreamingTransformer kwargs (forwarded via **kwargs)
        num_layers=cfg.num_layers,
        max_period=10000,
        device=device,
    )
    return model


def _build_instrument_for_program(tokenizer: MT3Tokenizer) -> Callable[[int], str]:
    """Map a decoded program int → human-readable instrument name.

    MT3_FULL_PLUS groups multiple GM programs together; the decoded program
    is always the first program of the group. We map that representative
    back to the readable group name.
    """
    group_map = tokenizer.group_program_map
    program_to_name: dict[int, str] = {}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if gid in group_map and group_map[gid]:
            program_to_name[group_map[gid][0]] = name

    def lookup(program: int) -> str:
        if program == DRUM_PROGRAM:
            return "drums"
        return program_to_name.get(program, f"program_{program}")

    return lookup


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        from pathlib import Path

        model = TranscriptionModel.load_model()
        for event in model.transcribe("audio.wav"):
            print(event)

        Path("out.mid").write_bytes(model.transcribe_and_postprocess("audio.wav")[0])
    """

    def __init__(self, model: LMModel, tokenizer: MT3Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._instrument_for_program = _build_instrument_for_program(tokenizer)

    @classmethod
    def load_model(
        cls,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "TranscriptionModel":
        """Load model weights and return a ready-to-use TranscriptionModel.

        Args:
            weights_path: A size keyword (``"small"``/``"medium"``/``"large"``)
                selecting a published HuggingFace variant, a local safetensors
                path, an ``hf://`` or ``https://`` URL, or None.  If None, the
                default ``medium`` variant is downloaded from HuggingFace.
                Remote URLs are cached under ~/.cache/muscriptor/.
            device: Torch device to use.  Defaults to the current accelerator
                (CUDA, MPS, ...) if one is available, else CPU.
            dtype: Transformer weight/compute dtype: ``"float32"``,
                ``"float16"``, ``"bfloat16"`` (or the torch dtypes). ``None``
                picks per device: float16 on MPS (halves memory traffic —
                decode is bandwidth-bound), float32 elsewhere (CUDA gets fp16
                compute via autocast instead). The conditioning pipeline
                (mel-spectrogram/class embeddings) always stays in fp32; its
                outputs are cast at the transformer boundary.
        """
        if device is None:
            device = (
                current_accelerator()
                if is_available()
                else torch.device("cpu")
            )
        elif isinstance(device, str):
            device = torch.device(device)

        if dtype is None:
            dtype = torch.float16 if device.type == "mps" else torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        source = _resolve_source(weights_path)
        weights_path = download_if_necessary(source)
        model = _build_model(device, _resolve_config(source, weights_path))
        model.eval()

        state_dict = load_file(weights_path, device=str(device))
        state_dict = _remap_single_codebook_keys(state_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        if dtype != torch.float32:
            model.to(dtype)
            # Conditioners keep fp32 numerics (log-mel of quiet passages
            # underflows in fp16); LMModel.forward casts their outputs.
            model.condition_provider.float()

        tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            max_shift_steps=1001,
        )

        return cls(model=model, tokenizer=tokenizer, device=device)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        instruments: list[str] | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = True,
        beam_size: int = 1,
        prelude_forcing: bool = True,
    ) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
        """Transcribe audio into a stream of note events.

        See the README for full argument documentation and the streaming /
        chunk-ordering guarantees. The audio is split into 5-second chunks;
        within each chunk events arrive in temporal order, and all events
        from chunk N are yielded before any event from chunk N+1.

        ``instruments``, when given, is a hard constraint: every program/drum
        token outside the listed groups is masked out during generation, so
        no other instrument can appear in the output. Leave it unset to let
        the model decode whatever instruments it detects.

        ``prelude_forcing`` (default True) teacher-forces each chunk's tie
        prologue — the tokens declaring which notes are sustained from the
        previous chunk — from the previous chunk's actually-unfinished notes,
        instead of letting the model guess (and occasionally re-enter with
        the wrong instruments). It requires chunks to be generated strictly
        in order, so while it is on the batch size defaults to (and must be)
        1; combining it with ``batch_size > 1`` raises ValueError — pass
        ``prelude_forcing=False`` explicitly to trade chunk-boundary quality
        for batched throughput.

        The event times may all carry the same small lag (up to ~25 ms) due to model
        bias. Taking it out needs the beat grid and every onset in the transcription,
        which only exist once the stream has finished, so prefer
        :meth:`transcribe_and_postprocess` when precise note timing matters.

        Interleaved with the note events are coarse :class:`ProgressEvent`
        anchors (``completed`` of ``total`` chunks): one up front with
        ``completed == 0``, then one as each chunk finishes. Consumers that
        only care about notes can ignore them.
        """
        batch_size = self._resolve_batch_size(batch_size, prelude_forcing)

        # Exact names only here — the CLI resolves abbreviations before
        # calling in (resolve_instrument_names).
        instrument_group = (
            instrument_group_from_names(instruments) if instruments else None
        )
        forbidden_tokens = None
        if instruments:
            forbidden_tokens = torch.tensor(
                self._tokenizer.forbidden_token_ids(instruments),
                device=self._device,
                dtype=torch.long,
            )

        timings: list[tuple[str, float]] = []
        t_total = time.perf_counter()

        if isinstance(audio, tuple):
            tensor, sample_rate = audio
            with _timed("load audio", timings):
                wav = self._load_wav(tensor, sample_rate)
        else:
            with _timed("load audio", timings):
                wav = self._load_wav(audio, None)

        total_samples = wav.shape[-1]
        total_duration = total_samples / _SAMPLE_RATE

        segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        num_chunks = math.ceil(total_samples / segment_samples)
        max_gen_len = 2000
        print(
            f"[muscriptor] audio: {total_duration:.1f}s → {num_chunks} chunk(s) of {_SEGMENT_DURATION}s",
            file=sys.stderr,
        )

        with _timed("build conditions", timings):
            all_conditions: list[ConditioningAttributes] = []
            seek_times: list[float] = []
            for i in range(num_chunks):
                start = i * segment_samples
                chunk = wav[:, start : start + segment_samples]
                if chunk.shape[-1] < segment_samples:
                    chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                all_conditions.append(
                    self._build_conditions(chunk, instrument_group)[0]
                )
                seek_times.append(i * _SEGMENT_DURATION)

        t_gen = time.perf_counter()

        # Up-front anchor: tells consumers the total chunk count and gives them a
        # timing baseline (t0) for the first chunk, before any tokens are gen'd.
        yield ProgressEvent(completed=0, total=num_chunks)

        yield from decode_model_tokens(
            self._generate_token_stream(
                all_conditions,
                seek_times,
                batch_size,
                max_gen_len,
                use_sampling,
                temperature,
                cfg_coef,
                no_eos_is_ok,
                prelude_forcing,
                beam_size,
                forbidden_tokens,
            ),
            self._tokenizer._vocab,
            self._instrument_for_program,
            frame_rate=self._tokenizer.frame_rate,
        )

        synchronize()
        print(
            f"[muscriptor] generate total: {time.perf_counter() - t_gen:.2f}s",
            file=sys.stderr,
        )
        print(
            f"[muscriptor] transcribe total: {time.perf_counter() - t_total:.2f}s "
            f"({total_duration:.1f}s audio)",
            file=sys.stderr,
        )

    def _resolve_batch_size(self, batch_size: int | None, prelude_forcing: bool) -> int:
        """Default the batch size, favouring transcription quality.

        Prelude forcing needs chunks generated strictly in order, so while it
        is on (the default) the batch size defaults to — and must be — 1.
        Batching is an explicit quality trade-off: asking for both raises
        instead of silently dropping the forcing.
        """
        if batch_size is None:
            if prelude_forcing:
                return 1
            return 4 if self._device.type in ("cuda", "mps") else 1
        if prelude_forcing and batch_size > 1:
            raise ValueError(
                f"batch_size={batch_size} disables prelude forcing, which lowers "
                "quality at chunk boundaries; pass prelude_forcing=False to "
                "accept that trade-off"
            )
        return batch_size

    # ------------------------------------------------------------------
    def _generate_token_stream(
        self,
        all_conditions: list[ConditioningAttributes],
        seek_times: list[float],
        batch_size: int,
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        no_eos_is_ok: bool,
        prelude_forcing: bool = True,
        beam_size: int = 1,
        forbidden_tokens: torch.Tensor | None = None,
    ) -> Iterator[int | ChunkBoundary | ProgressEvent]:
        """Generate tokens and yield them per chunk, as soon as they are ready.

        The model emits one token per chunk per timestep across the batch, but
        the decoder consumes whole chunks in order. So within each batch we
        stream the first chunk's tokens live as they are generated and buffer
        the others; once the first chunk hits EOS we flush the next chunk's
        buffered tokens and stream it live, and so on. EOS (and anything after
        it) is dropped.

        With ``prelude_forcing`` (and ``batch_size == 1``, so chunks generate
        strictly in order), every chunk after the first has its tie prologue
        teacher-forced: the notes left open by the previous chunk are encoded
        as ``(program, pitch)…tie`` tokens and passed to ``generate`` as a
        prompt, so the model can't restate them with the wrong instruments.
        The forced tokens flow through this stream like generated ones, which
        keeps the downstream decoder's view consistent by construction.
        """
        eos_id = self._tokenizer.eos_id
        num_chunks = len(seek_times)

        # Chunks in a batch generate concurrently, so with batch_size > 1 the
        # previous chunk's open notes aren't known when the next one starts —
        # forcing is only possible chunk-by-chunk. transcribe() rejects that
        # combination up front (_resolve_batch_size); this guard keeps the
        # invariant for direct callers too.
        tracker = None
        if prelude_forcing and batch_size == 1:
            tracker = OpenNoteTracker(
                self._tokenizer._vocab, self._tokenizer.frame_rate
            )

        def boundary(chunk_index: int) -> ChunkBoundary:
            next_seek_time = (
                seek_times[chunk_index + 1] if chunk_index + 1 < num_chunks else None
            )
            return ChunkBoundary(seek_times[chunk_index], next_seek_time)

        for batch_start in range(0, num_chunks, batch_size):
            batch_conditions = all_conditions[batch_start : batch_start + batch_size]
            n = len(batch_conditions)
            buffers: list[list[int]] = [[] for _ in range(n)]
            done = [False] * n
            active = 0  # within-batch index of the chunk streaming live

            # The first chunk in the batch streams live from the start.
            bnd = boundary(batch_start)
            prompt = None
            if tracker is not None:
                # Feed the boundary first: it settles the tracker (e.g. a
                # previous chunk that never emitted its tie token drops all
                # open notes) so open_keys() is exactly the decoder's view.
                tracker.feed(bnd)
                if batch_start > 0:
                    prompt = torch.tensor(
                        [self._tokenizer.tie_section_token_ids(tracker.open_keys())],
                        device=self._device,
                        dtype=torch.long,
                    )
            yield bnd

            for step in self._model.generate(
                prompt=prompt,
                conditions=batch_conditions,
                max_gen_len=max_gen_len,
                use_sampling=use_sampling,
                temp=temperature,
                top_k=0,
                top_p=0.0,
                cfg_coef=cfg_coef,
                early_stop_on_token=eos_id,
                beam_size=beam_size,
                forbidden_tokens=forbidden_tokens,
            ):
                row = step.tolist()  # one token per chunk: [n]
                for j in range(n):
                    if done[j]:
                        continue
                    tok = row[j]
                    if tok == eos_id:
                        done[j] = True
                    else:
                        if tracker is not None:
                            tracker.feed(tok)
                        if j == active:
                            yield tok
                        else:
                            buffers[j].append(tok)
                # When the live chunk finishes, flush and stream the next one(s).
                while active < n and done[active]:
                    active += 1
                    if active < n:
                        yield boundary(batch_start + active)
                        yield from buffers[active]
                        buffers[active] = []

            # Any chunk still open never emitted EOS within max_gen_len.
            for j in range(active, n):
                if not done[j]:
                    chunk_index = batch_start + j
                    msg = (
                        f"chunk {chunk_index} (seek={seek_times[chunk_index]:.1f}s) "
                        f"did not emit EOS within {max_gen_len} tokens"
                    )
                    if no_eos_is_ok:
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    else:
                        raise RuntimeError(
                            msg + " (this is only raised under --strict-eos)"
                        )
                # The live (active) chunk has already streamed; emit the rest.
                if j != active:
                    yield boundary(batch_start + j)
                    yield from buffers[j]

            # This batch's chunks are fully generated: emit a completion anchor.
            # (batch_size=1 on the web path => one event per chunk.) The event
            # trails the chunk's tokens, so by the time it surfaces from
            # decode_model_tokens all of that chunk's notes have been yielded.
            yield ProgressEvent(completed=batch_start + n, total=num_chunks)

    # ------------------------------------------------------------------
    def transcribe_and_postprocess(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        instruments: list[str] | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = True,
        beam_size: int = 1,
        prelude_forcing: bool = True,
        detect_tempo: TempoDetection = "best-effort",
        quantize: bool = False,
    ) -> tuple[bytes, BeatGrid | None]:
        """Same as :meth:`transcribe`, but as a MIDI file plus the grid it used.

        The grid comes back measured against the transcription's own onsets, so
        its `onset_delay` and `beat_subdivision` are filled in; it is None when
        no tempo was detected.

        `quantize` snaps the notes onto that subdivision, which is what sheet
        music has to be engraved from (see :meth:`events_to_midi_bytes`) and not
        what anyone wants to listen to. The returned grid says whether there was
        a subdivision to snap to at all.
        """
        beat_grid = self.detect_beat_grid_for(audio, detect_tempo)
        events = list(
            self.transcribe(
                audio,
                use_sampling=use_sampling,
                temperature=temperature,
                cfg_coef=cfg_coef,
                instruments=instruments,
                batch_size=batch_size,
                no_eos_is_ok=no_eos_is_ok,
                beam_size=beam_size,
                prelude_forcing=prelude_forcing,
            )
        )
        if beat_grid is not None:
            beat_grid = beat_grid.with_onset_delay(
                [ev.start_time for ev in events if isinstance(ev, NoteStartEvent)]
            )
        midi_bytes = self.events_to_midi_bytes(
            iter(events), beat_grid=beat_grid, quantize=quantize
        )
        return midi_bytes, beat_grid

    def detect_beat_grid_for(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        mode: TempoDetection = "best-effort",
    ) -> BeatGrid | None:
        """Detect the beat grid of `audio`, or None if there isn't a usable one.

        Accepts the same input forms as :meth:`transcribe`.
        `mode` decides what a failed detection means: raise (True), skip detection
        entirely (False), or warn and fall back to the placeholder tempo
        ("best-effort").

        The grid carries the tracked beat times along, so that writing the MIDI
        can move the transcription's onsets onto them (see BeatGrid.onset_delay).
        """
        if mode is False:
            return None
        tensor, sample_rate = audio if isinstance(audio, tuple) else (audio, None)
        try:
            return detect_grid(self._load_wav(tensor, sample_rate), _SAMPLE_RATE)
        except BeatDetectionError as e:
            if mode is True:
                raise
            print(
                f"Warning: {e}; falling back to the placeholder tempo",
                file=sys.stderr,
            )
            return None

    def events_to_midi_bytes(
        self,
        events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent],
        beat_grid: BeatGrid | None = None,
        quantize: bool = False,
    ) -> bytes:
        """Reassemble Notes from a NoteStart/NoteEnd stream and serialize MIDI.

        Shared by :meth:`transcribe_and_postprocess` and the HTTP server, so the MIDI
        bytes are identical regardless of how the events were obtained.

        `quantize` snaps the notes onto `beat_grid.beat_subdivision` first,
        which is what sheet music has to be engraved from (see
        `quantized_notes` (misma sección utils/midi)) and not what anyone wants to
        listen to. Call twice for both versions of the same transcription.
        """
        notes: list[Note] = []
        open_notes: dict[int, Note] = {}
        program_names: dict[int, str] = {}
        for ev in events:
            if isinstance(ev, ProgressEvent):
                continue
            if isinstance(ev, NoteStartEvent):
                is_drum = ev.instrument == "drums"
                program = (
                    DRUM_PROGRAM
                    if is_drum
                    else self._program_for_instrument(ev.instrument)
                )
                program_names[program] = ev.instrument.replace("_", " ")
                note = Note(
                    is_drum=is_drum,
                    program=program,
                    onset=ev.start_time,
                    offset=ev.start_time,  # patched on NoteEndEvent
                    pitch=ev.pitch,
                )
                open_notes[ev.index] = note
            else:  # NoteEndEvent
                note = open_notes.pop(ev.start_event_index)
                note.offset = ev.end_time
                notes.append(note)

        # Match the legacy decoder's note-cleanup pass so the MIDI bytes
        # don't drift from earlier reference outputs.
        notes = validate_notes(notes, fix=True)
        notes = trim_overlapping_notes(notes, sort=True)
        midi = notes_to_midi(
            notes, program_names=program_names, beat_grid=beat_grid, quantize=quantize
        )
        buf = io.BytesIO()
        midi.save(file=buf)
        return buf.getvalue()

    def _program_for_instrument(self, instrument: str) -> int:
        """Inverse of `_instrument_for_program` for non-drum instruments."""
        if not hasattr(self, "_inst_to_program"):
            group_map = self._tokenizer.group_program_map
            self._inst_to_program = {
                name: group_map[gid][0]
                for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items()
                if gid in group_map and group_map[gid]
            }
        if instrument in self._inst_to_program:
            return self._inst_to_program[instrument]
        # fallback for unknown names like "program_42"
        if instrument.startswith("program_"):
            return int(instrument.removeprefix("program_"))
        raise ValueError(f"Unknown instrument name: {instrument!r}")

    # ------------------------------------------------------------------
    def _load_wav(
        self, audio: str | Path | torch.Tensor, sample_rate: int | None
    ) -> torch.Tensor:
        """Return mono float32 waveform at 16 kHz, shape [1, T]."""
        if isinstance(audio, (str, Path)):
            wav = load_audio(audio, target_sr=_SAMPLE_RATE)
        else:
            wav = audio.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sample_rate is not None and sample_rate != _SAMPLE_RATE:
                wav = resample(wav, sample_rate, _SAMPLE_RATE)
        return wav.to(self._device)

    def _build_conditions(
        self,
        wav: torch.Tensor,
        instrument_group: str | None = None,
    ) -> list[ConditioningAttributes]:
        """Build a single-element list of ConditioningAttributes for one 5-second chunk."""
        T = wav.shape[-1]
        wav_3d = wav.unsqueeze(0)  # [1, 1, T]
        length = torch.tensor([T], device=self._device)
        wav_cond = WavCondition(
            wav=wav_3d,
            length=length,
            sample_rate=[_SAMPLE_RATE],
            path=[None],
            seek_time=[0.0],
        )
        return [
            ConditioningAttributes(
                wav={"self_wav": wav_cond},
                text={
                    "instrument_group": instrument_group,
                    # Always unconditional on dataset: the null/pad class.
                    "dataset_name": None,
                },
            )
        ]
# =============================================================================
# CLI
# =============================================================================


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"
    jsonl = "jsonl"


def _load_model(model_path: str | None, device: str | None, dtype: str | None = None) -> "TranscriptionModel":
    """load_model with CLI-friendly failure: known download problems (missing
    HuggingFace authentication, …) print a plain message instead of a traceback."""
    try:
        return TranscriptionModel.load_model(weights_path=model_path, device=device, dtype=dtype)
    except ModelDownloadError as e:
        _err(f"Error: {e}")
        sys.exit(1)


def _transcribe(model, kwargs: dict, detect_tempo: str, quantize: bool = False):
    """transcribe_and_postprocess, with the CLI's --detect-tempo spelling and errors."""
    try:
        mode: TempoDetection = {
            "true": True,
            "false": False,
            "best-effort": "best-effort",
        }[detect_tempo]
        return model.transcribe_and_postprocess(**kwargs, detect_tempo=mode, quantize=quantize)
    except BeatDetectionError as e:
        _err(f"Error: {e}")
        _err("Pass --detect-tempo best-effort or false to continue.")
        sys.exit(1)


def _event_to_dict(ev: "NoteStartEvent | NoteEndEvent") -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe an audio file to MIDI/JSON/JSONL."""
    audio_file = Path(args.audio_file)

    instrument_names: list[str] | None = None
    if args.instruments is not None:
        tokens = [n for n in args.instruments.split(",") if n.strip()]
        try:
            instrument_names = resolve_instrument_names(tokens)
        except ValueError as e:
            _err(f"Error: {e}. Run '{PROG} list-instruments' to see available names.")
            sys.exit(1)
        _err(f"{_c('Instruments', CYAN)}: {', '.join(instrument_names)}")

    if not audio_file.exists():
        _err(f"Error: file not found: {audio_file}")
        sys.exit(1)

    if args.prelude_forcing and args.batch_size is not None and args.batch_size != 1:
        _err(
            f"Error: --batch-size {args.batch_size} requires --no-prelude-forcing: "
            "batching disables prelude forcing, which lowers transcription "
            "quality at chunk boundaries."
        )
        sys.exit(1)

    output = Path(args.output) if args.output is not None else None
    is_stdout = output is not None and str(output) == "-"
    fmt = OutputFormat(args.format)

    if output is None:
        suffix = {
            OutputFormat.midi: ".mid",
            OutputFormat.json: ".json",
            OutputFormat.jsonl: ".jsonl",
        }[fmt]
        output = audio_file.with_suffix(suffix)

    _device = None if args.device == "auto" else args.device

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    _err(_c("Loading model…", DIM))
    model = _load_model(args.model, _device, args.dtype)

    _err(_c(f"Transcribing {audio_file} …", DIM))

    kwargs = dict(
        audio=audio_file,
        use_sampling=args.sampling,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
        instruments=instrument_names,
        batch_size=args.batch_size,
        no_eos_is_ok=not args.strict_eos,
        beam_size=args.beam_size,
        prelude_forcing=args.prelude_forcing,
    )

    if fmt == OutputFormat.midi:
        midi_bytes, _ = _transcribe(model, kwargs, args.detect_tempo)
        if is_stdout:
            sys.stdout.buffer.write(midi_bytes)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(midi_bytes)
            _err(_c(f"✓ Saved MIDI to {output}", GREEN))
        if args.notes:
            _err("Re-run with --format json to inspect the event stream.")
    elif fmt == OutputFormat.jsonl:
        # Stream one JSON object per line, flushing after each event so the
        # file (or stdout pipe) can be consumed live.
        if is_stdout:
            sink = sys.stdout
            close_after = False
        else:
            sink = output.open("w")
            close_after = True
        try:
            for e in model.transcribe(**kwargs):
                if isinstance(e, ProgressEvent):
                    continue
                sink.write(json.dumps(_event_to_dict(e)) + "\n")
                sink.flush()
                if args.notes:
                    _err(str(e))
        finally:
            if close_after:
                sink.close()
        if not is_stdout:
            _err(_c(f"✓ Saved JSONL to {output}", GREEN))
    else:  # json
        events = [e for e in model.transcribe(**kwargs) if not isinstance(e, ProgressEvent)]
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            _err(_c(f"✓ Saved JSON to {output}", GREEN))
        if args.notes:
            for e in events:
                _err(str(e))


def cmd_list_instruments(args: argparse.Namespace) -> None:
    """List the instrument group names accepted by --instruments."""
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        print(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="Transcribe an audio file to MIDI/JSON/JSONL")
    p.add_argument("audio_file", help="Input audio file (wav, mp3, flac, …)")
    p.add_argument(
        "-o", "--output",
        help="Output path ('-' for stdout, all progress goes to stderr). "
             "Default: <audio_file>.<ext> matching --format.",
    )
    p.add_argument(
        "-f", "--format", choices=[f.value for f in OutputFormat], default=OutputFormat.midi.value,
        help="midi (default): quantized MIDI file. json: one array of events. "
             "jsonl: one event per line, streamed live.",
    )
    p.add_argument("--notes", action="store_true", help="Print decoded events to stderr")
    p.add_argument("--sampling", action="store_true", help="Temperature sampling instead of greedy decoding")
    p.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature (only with --sampling)")
    p.add_argument("--cfg-coef", type=float, default=1.0, help="Classifier-free guidance coefficient")
    p.add_argument(
        "-m", "--model", default=None,
        help="Model size ('small', 'medium', 'large'; default: medium), a local "
             "safetensors path, or an hf:// / http(s):// URL",
    )
    p.add_argument("-d", "--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:0', 'mps', …")
    p.add_argument(
        "--dtype", default=None,
        help="Transformer dtype: 'float32', 'float16' or 'bfloat16'. "
             "Default: float16 on MPS, float32 elsewhere.",
    )
    p.add_argument(
        "-b", "--batch-size", type=int, default=None,
        help="Chunks generated per forward pass (default: 1; with --no-prelude-forcing: "
             "4 on GPU, 1 on CPU). Values > 1 lower quality at chunk boundaries and "
             "require --no-prelude-forcing.",
    )
    p.add_argument(
        "--strict-eos", action="store_true",
        help="Raise an error if a chunk fails to emit EOS within budget "
             "(default: downgrade to a warning)",
    )
    p.add_argument("--beam-size", type=int, default=1, help="Beam search width (1 = greedy/sampling, ≥2 enables beam search)")
    p.add_argument(
        "--prelude-forcing", dest="prelude_forcing", action="store_true", default=True,
        help="Teacher-force each chunk's tie prologue (default: on, needs --batch-size 1)",
    )
    p.add_argument("--no-prelude-forcing", dest="prelude_forcing", action="store_false")
    p.add_argument(
        "--instruments", default=None,
        help="Comma-separated expected instrument group names (unambiguous "
             "abbreviations accepted, e.g. 'timp,cello,dist'). Run "
             f"'{PROG} list-instruments' for the full list.",
    )
    p.add_argument(
        "--detect-tempo", choices=["true", "false", "best-effort"], default="best-effort",
        help="Detect tempo/time-signature and write them into the MIDI. 'true' fails "
             "if no steady tempo is found, 'best-effort' warns and falls back to a "
             "placeholder 120 BPM, 'false' skips detection.",
    )
    p.set_defaults(func=cmd_transcribe)

    p2 = sub.add_parser("list-instruments", help="List instrument group names accepted by --instruments")
    p2.set_defaults(func=cmd_list_instruments)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
