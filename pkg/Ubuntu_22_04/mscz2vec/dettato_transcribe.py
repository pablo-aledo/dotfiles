#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DETTATO_TRANSCRIBE.PY                                                ║
# ║  Transcripción multi-instrumento + batería: audio → MIDI / JSON       ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
dettato_transcribe.py — Transcripción de audio a MIDI con dettato
(KobiMusic; código y pesos Apache-2.0, https://github.com/kobimusic/dettato).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Toma una grabación (wav/mp3/flac/ogg/m4a/…) y escribe un MIDI con una pista
por instrumento (con su nombre y programa General MIDI) más una pista de
batería. Son cuatro redes pequeñas (28,1M parámetros en total) encadenadas:

    1. transcriber  15,3M  notas desde el audio (88 teclas, 2 voces por tecla)
    2. evidence      4,9M  repara las notas con la evidencia del transcriber:
                           quita falsas, corrige onsets/finales, añade perdidas
    3. grouper       2,9M  asigna cada nota a una pista y nombra el instrumento
    4. drums         5,0M  golpes de batería GM, kits y percusión orquestal

Antes de todo, se estima la afinación y el audio se reescala a A440 si se
desvía claramente. Una sola pasada del transcriber alimenta al repairer y al
grouper. La batería solo se transcribe con fuentes grabadas a 40 kHz o más.

El MIDI resultante es de interpretación: conserva el tiempo de la grabación
(120 BPM fijos, sin cuantizar ni mapa de tempo). El punto débil conocido es
el agrupado por instrumento (en jazz, ~1/4 de las notas de saxo puede caer en
la pista de piano).

PESOS:
    Se descargan de HuggingFace Hub en el primer uso (sin autenticación):
        kobimusic/dettato  dettato-evidence  dettato-grouper  dettato-drums
    Sin red, apunta --weights (o la variable DETTATO_WEIGHTS) a una carpeta
    con los cuatro repos (dettato/, dettato-evidence/, dettato-grouper/,
    dettato-drums/), cada uno con config.json y model.safetensors.

RENDIMIENTO:
    GPU CUDA recomendada: ~15 s para 4 min de audio (RTX 5060, <4 GB de VRAM).
    En CPU funciona pero es lento (un clip de 15 s tardó 87 s, carga incluida).
    Por encima de 250 s los buffers del transcriber pasan a RAM (~10 MB por
    segundo de audio). Los formatos que soundfile no lee (m4a/aac, algunos
    mp3) pasan por ffmpeg, que debe estar en el PATH.

LICENCIA:
    El código y los pesos son Apache-2.0, pero algunos datasets de
    entrenamiento del grouper y de la batería tienen términos no comerciales
    (ver las model cards en HuggingFace antes de un uso comercial).

Alcance de esta versión de un solo fichero: transcripción a MIDI/JSON, sin
importar el paquete `dettato` (el código de las cuatro etapas va incluido).
Se dejan fuera a propósito el procesado de carpetas completas y la API de
Python; para eso se pueden encadenar llamadas a este script.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  transcribe        Transcribe un fichero de audio a MIDI/JSON
  list-instruments  Lista los instrumentos que el grouper sabe nombrar
  info              Carga los modelos y muestra dispositivo, pesos y tamaños

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  transcribe
    audio_file AUDIO_FILE       Fichero de audio de entrada (wav, mp3, flac, …)
    -o, --output OUTPUT         Ruta de salida ('-' para stdout; el progreso
                                va a stderr). Por defecto: <audio_file>.mid o
                                .json según --format.
    -f, --format FORMAT         midi (defecto): un .mid con una pista por
                                instrumento y una de batería. json: resumen
                                (pistas, conteos, afinación) con todas las
                                notas [inicio s, fin s, pitch, velocidad].
    -d, --device DEVICE         'auto', 'cpu', 'cuda', 'cuda:0', … (defecto
                                auto: cuda si hay GPU, si no cpu)
    -w, --weights WEIGHTS       Carpeta con los cuatro repos de pesos (en vez
                                de descargarlos del Hub). Equivale a la
                                variable DETTATO_WEIGHTS.
    --no-drums                  No cargar ni ejecutar el modelo de batería
    --notes                     Imprime las pistas y su número de notas en
                                stderr

  list-instruments
    (sin opciones)

  info
    -d, --device DEVICE         'auto', 'cpu', 'cuda', 'cuda:0', … (defecto
                                auto)
    -w, --weights WEIGHTS       Carpeta con los cuatro repos de pesos

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[transcribe]
audio_file     file  io=in ext=wav required  # Fichero de audio de entrada (wav, mp3, flac, …)
--output       file  io=out ext=mid alias=-o  # Ruta de salida ('-' para stdout; el progreso va a stderr)
--format       enum  default=midi values=midi|json alias=-f  # midi: .mid multipista. json: resumen con todas las notas
--device       str   default=auto alias=-d  # 'auto', 'cpu', 'cuda', 'cuda:0', …
--weights      dir   io=in alias=-w  # Carpeta con los cuatro repos de pesos (en vez del Hub)
--no-drums     bool  default=false  # No cargar ni ejecutar el modelo de batería
--notes        bool  default=false  # Imprime las pistas y su número de notas en stderr
[list-instruments]
[info]
--device       str   default=auto alias=-d  # 'auto', 'cpu', 'cuda', 'cuda:0', …
--weights      dir   io=in alias=-w  # Carpeta con los cuatro repos de pesos (en vez del Hub)
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  ---
      dettato_transcribe.py transcribe cancion.mp3
      dettato_transcribe.py transcribe cancion.mp3 -o salida/cancion.mid
      dettato_transcribe.py transcribe cancion.wav -f json -o notas.json
      dettato_transcribe.py transcribe cancion.wav -f json -o - | jq '.tracks[].name'
      dettato_transcribe.py transcribe cancion.wav -d cuda --no-drums
      dettato_transcribe.py transcribe cancion.wav -w ~/modelos/dettato
      dettato_transcribe.py list-instruments
      dettato_transcribe.py info -d cpu

Dependencias: torch>=2.2, numpy, scipy, soundfile, soxr, pretty_midi,
huggingface_hub, safetensors (ffmpeg opcional, para m4a/aac).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROG = "dettato_transcribe.py"

# --- ANSI color output (estilo mutopia) -------------------------------------
_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stderr.isatty()
RESET = "" if _NO_COLOR else "\033[0m"
DIM = "" if _NO_COLOR else "\033[2m"
CYAN = "" if _NO_COLOR else "\033[36m"
GREEN = "" if _NO_COLOR else "\033[32m"
YELLOW = "" if _NO_COLOR else "\033[33m"
RED = "" if _NO_COLOR else "\033[31m"


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if color else text


def _err(text: str) -> None:
    print(text, file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════
# hub
# ══════════════════════════════════════════════════════════════════════

# Model weights: downloaded from the Hugging Face Hub on first use and cached there.
#
# Each stage has its own model repo:
#
#     transcriber  kobimusic/dettato
#     evidence     kobimusic/dettato-evidence
#     grouper      kobimusic/dettato-grouper
#     drums        kobimusic/dettato-drums
#
# Set ``DETTATO_WEIGHTS=/some/dir`` to load from local copies instead (``/some/dir/dettato/config.json``,
# ``/some/dir/dettato-grouper/model.safetensors``, ...), for example on a machine without network access.


HUB_REPOS = {
    "transcriber": "kobimusic/dettato",
    "evidence": "kobimusic/dettato-evidence",
    "grouper": "kobimusic/dettato-grouper",
    "drums": "kobimusic/dettato-drums",
}


def hub_fetch(stage: str, filename: str, revision: str | None = None) -> str:
    """Local path of ``filename`` in the stage's model repo (downloaded once, then read from the cache)."""
    repo = HUB_REPOS[stage]
    local = os.environ.get("DETTATO_WEIGHTS")
    if local:
        path = Path(local).expanduser() / repo.split("/", 1)[1] / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} (DETTATO_WEIGHTS is set, so the Hub is not used)")
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision)


def hub_load_config(stage: str, filename: str = "config.json") -> dict:
    with open(hub_fetch(stage, filename)) as fh:
        return json.load(fh)


def hub_load_state(stage: str, filename: str = "model.safetensors") -> dict:
    from safetensors.torch import load_file

    return load_file(hub_fetch(stage, filename))


# ══════════════════════════════════════════════════════════════════════
# audio
# ══════════════════════════════════════════════════════════════════════

# Reading audio files: one decode at the file's own rate, mono, then resampled for each model.
#
# The transcriber, the evidence model and the grouper work on 16 kHz audio; the drum model on 44.1 kHz audio and only when the
# source itself was recorded at 40 kHz or more.


EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".opus")


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono at `rate`
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate

    def at(self, rate: int) -> np.ndarray:
        """The audio resampled to ``rate`` (float32 mono)."""
        if rate == self.rate:
            return self.samples
        import soxr

        return soxr.resample(self.samples, self.rate, rate, quality="HQ").astype(np.float32)


def _ffmpeg(path: str) -> Audio:
    rate = _probe_rate(path) or 44100
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
                         capture_output=True, check=True).stdout
    return Audio(np.frombuffer(raw, np.float32).copy(), rate)


def _probe_rate(path: str) -> int | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                              "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
        return int(out.strip().split(",")[0].split()[0])
    except Exception:  # noqa: BLE001
        return None


def audio_load(path: str) -> Audio:
    """Any audio file -> mono float32 at its own sample rate (soundfile first, ffmpeg for anything it cannot read)."""
    try:
        import soundfile as sf

        x, rate = sf.read(path, dtype="float32", always_2d=True)
        return Audio(x.mean(1).astype(np.float32), int(rate))
    except Exception:  # noqa: BLE001
        return _ffmpeg(path)


# ══════════════════════════════════════════════════════════════════════
# transcriber/frontend
# ══════════════════════════════════════════════════════════════════════

# Harmonic constant-Q-like front end: 16 kHz audio -> log power on a log-frequency axis, one channel per harmonic.


SR = 16000
HOP = 160            # 10 ms spectrogram hop
N_KEYS = 88          # lanes: lane i = MIDI pitch 21 + i
PITCH_LO = 21


def logfreq_filterbank(n_bins, n_fft, fmin=27.5, bps=3, min_width_bins=2.0):
    """Triangular filters on a log-frequency grid (``bps`` bins per semitone from ``fmin``), each normalised to sum 1."""
    fft_freqs = np.linspace(0, SR / 2, n_fft // 2 + 1)
    centers = fmin * 2.0 ** (np.arange(n_bins) / (12.0 * bps))
    fb = np.zeros((n_bins, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_bins):
        ctr = centers[i]
        lo, hi = ctr * 2 ** (-1 / (12 * bps)), ctr * 2 ** (1 / (12 * bps))
        width = max(hi - lo, min_width_bins * SR / n_fft)
        lo, hi = ctr - width / 2, ctr + width / 2
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        v = np.maximum(0.0, np.minimum(up, down))
        if v.sum() > 0:
            fb[i] = v / v.sum()
    return fb


class HCQT(nn.Module):
    """audio (B, n) -> (B, C, T, F) log power, T = n // HOP, F = n_bins (3 per semitone from A0).

    Channels: one per harmonic h (the spectrum shifted so that harmonic h of a note lands on the note's own bin) from the long
    window, plus plain channels from shorter windows (sharper onsets)."""

    def __init__(self, n_fft=2048, n_bins=288, bps=3, fmin=27.5, harmonics=(0.5, 1, 2, 3, 4, 5, 6, 7), extra_nffts=(512,),
                 log_eps=1e-5):
        super().__init__()
        self.n_fft, self.n_bins, self.bps, self.log_eps = n_fft, n_bins, bps, log_eps
        self.harmonics = tuple(harmonics)
        self.shifts = [int(round(12 * bps * math.log2(h))) for h in self.harmonics]
        lo, hi = min(0, min(self.shifts)), max(0, max(self.shifts))
        self.lo, self.hi = lo, hi
        fb = logfreq_filterbank(n_bins + hi - lo, n_fft, fmin * 2 ** (lo / (12 * bps)), bps)
        self.register_buffer("fb", torch.from_numpy(fb), persistent=False)
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.extra = tuple(extra_nffts)
        for i, nf in enumerate(self.extra):
            self.register_buffer(f"fb_x{i}", torch.from_numpy(logfreq_filterbank(n_bins, nf, fmin, bps)), persistent=False)
            self.register_buffer(f"win_x{i}", torch.hann_window(nf), persistent=False)
        self.n_channels = len(self.harmonics) + len(self.extra)

    def _pow(self, audio, n_fft, win):
        spec = torch.stft(audio, n_fft, hop_length=HOP, win_length=n_fft, window=win, center=True, return_complex=True,
                          pad_mode="constant")
        return spec.real ** 2 + spec.imag ** 2

    @torch.autocast("cuda", enabled=False)
    def forward(self, audio):
        audio = audio.float()
        T = audio.shape[-1] // HOP
        p = self._pow(audio, self.n_fft, self.win)[:, :, :T]
        lf = torch.log(torch.matmul(self.fb, p) + self.log_eps)
        chans = []
        for s in self.shifts:
            a = s - self.lo
            x = lf[:, a:a + self.n_bins]
            if x.shape[1] < self.n_bins:
                x = F.pad(x, (0, 0, 0, self.n_bins - x.shape[1]), value=math.log(self.log_eps))
            chans.append(x)
        for i, nf in enumerate(self.extra):
            px = self._pow(audio, nf, getattr(self, f"win_x{i}"))[:, :, :T]
            chans.append(torch.log(torch.matmul(getattr(self, f"fb_x{i}"), px) + self.log_eps))
        return torch.stack(chans, 1).transpose(2, 3)


# ══════════════════════════════════════════════════════════════════════
# transcriber/decode
# ══════════════════════════════════════════════════════════════════════

# Probability maps -> notes (peak picking, no learned parameters).



def _peaks(p, th):
    """(T, L) -> bool mask of local maxima >= th (plateaus: first frame)."""
    left = np.vstack([np.full((1, p.shape[1]), -1.0), p[:-1]])
    right = np.vstack([p[1:], np.full((1, p.shape[1]), -1.0)])
    return (p >= th) & (p > left) & (p >= right)


def _refine(p, t, l):
    """Sub-frame position of a peak at (t, l) from its two neighbours, clipped to +-0.5 frame."""
    T = p.shape[0]
    C = p[t, l]
    A = p[t - 1, l] if t > 0 else C
    Bv = p[t + 1, l] if t + 1 < T else C
    den = 2 * (C - min(A, Bv))
    if den <= 1e-6:
        return 0.0
    return float(np.clip((Bv - A) / den, -0.5, 0.5))


def decode(onset, frame, offset, hop, on_th=0.25, off_th=0.3, fr_th=0.4, fr_low_frames=2, on_nms=3, min_dur=0.02,
           off_min_frames=2):
    """Maps (T, 88) of one voice -> notes [(onset s, offset s, MIDI pitch)], sorted.

    A note starts at every onset peak >= on_th (peaks within on_nms frames of a higher one are dropped). It ends at the first of:
    an offset peak >= off_th (from onset + off_min_frames), the frame probability staying below fr_th for fr_low_frames frames,
    the next onset on the same key."""
    on, fr, off = onset, frame, offset
    T, L = on.shape
    pk_on = _peaks(on, on_th)
    pk_off = _peaks(off, off_th)
    low = fr < fr_th
    run = low.copy()
    for k in range(1, fr_low_frames):
        run[:T - k] &= low[k:]
    notes = []
    for l in range(L):
        ons = np.nonzero(pk_on[:, l])[0]
        if len(ons) == 0:
            continue
        if on_nms > 0 and len(ons) > 1:
            keep = np.ones(len(ons), bool)
            for oi in np.argsort(-on[ons, l]):
                if not keep[oi]:
                    continue
                close = np.abs(ons - ons[oi]) <= on_nms
                close[oi] = False
                keep &= ~close
            ons = ons[keep]
        offs = np.nonzero(pk_off[:, l])[0]
        lows = np.nonzero(run[:, l])[0]
        for i, t0 in enumerate(ons):
            u0 = t0 + _refine(on, t0, l)
            nxt = ons[i + 1] if i + 1 < len(ons) else T
            j = np.searchsorted(lows, t0 + 1)
            e_fr = lows[j] if j < len(lows) else T
            j = np.searchsorted(offs, t0 + off_min_frames)
            e_off = offs[j] if j < len(offs) else T
            if e_off < T and e_off <= min(e_fr, nxt):
                u1 = e_off + _refine(off, e_off, l)
            elif e_fr <= nxt:
                u1 = e_fr - 0.5
            else:
                u1 = nxt + (_refine(on, nxt, l) if nxt < T else 0.0)
            on_s, off_s = u0 * hop, max(u1 * hop, u0 * hop + min_dur)
            notes.append((on_s, off_s, PITCH_LO + l))
    notes.sort()
    return notes


# ══════════════════════════════════════════════════════════════════════
# transcriber/model
# ══════════════════════════════════════════════════════════════════════

# The transcriber network: a pitch-equivariant lane model.
#
# audio (B, n) 16 kHz -> HCQT -> stem (20 ms frames x 1 semitone, 96 lanes from A0) -> lane blocks at 40 ms (ConvNeXt-style: depthwise
# time conv + sparse pitch mixing at local and harmonic intervals), with axial attention (across pitch, across time) and a bidirectional
# lane GRU after blocks 5, 10 and 15 -> back to 20 ms + refine blocks -> 1x1 heads on lanes 0-87.
#
# Heads (logits, each (B, T, 88)): onset / frame / offset for the first voice of each key, onset2 / frame2 / offset2 for a second
# voice on the same key (two instruments playing the same pitch).



class PitchMix(nn.Module):
    """Per-channel taps along the pitch axis at fixed semitone offsets, applied as one banded (P x P) matrix per channel."""

    def __init__(self, c, n_lanes, offsets):
        super().__init__()
        self.offsets = tuple(offsets)
        self.taps = nn.Parameter(torch.zeros(c, len(self.offsets)))
        S = torch.zeros(len(self.offsets), n_lanes, n_lanes)
        i = torch.arange(n_lanes)
        for k, o in enumerate(self.offsets):
            j = i + o
            ok = (j >= 0) & (j < n_lanes)
            S[k, i[ok], j[ok]] = 1.0
        self.register_buffer("S", S, persistent=False)

    def forward(self, x):
        W = torch.einsum("ck,kij->cij", self.taps.to(x.dtype), self.S.to(x.dtype))
        return torch.einsum("cij,bctj->bcti", W, x)


class LaneBlock(nn.Module):
    """Depthwise conv along time (kernel kt, dilation dt) + pitch mixing, BatchNorm, pointwise MLP, layer-scaled residual."""

    def __init__(self, c, kt=7, dt=1, exp=2, pmix=(), n_lanes=96):
        super().__init__()
        self.pmix = PitchMix(c, n_lanes, pmix)
        self.dw_t = nn.Conv2d(c, c, (kt, 1), padding=((kt // 2) * dt, 0), dilation=(dt, 1), groups=c)
        self.norm = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, c * exp, 1)
        self.pw2 = nn.Conv2d(c * exp, c, 1)
        self.gamma = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        y = self.dw_t(x)
        y = y + self.pmix(y)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + y * self.gamma[:, None, None]


class AxialAttn(nn.Module):
    """Self-attention along one axis of (B, C, T, P). axis="time": each lane over its frames (rotary positions); axis="pitch":
    the lanes of each frame (a learned bias per head for every pitch distance, so the layer is transposition-equivariant)."""

    def __init__(self, c, axis, heads=8, n_pos=128):
        super().__init__()
        self.axis, self.heads = axis, heads
        self.ln = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.out = nn.Linear(c, c)
        self.gamma = nn.Parameter(torch.zeros(c))
        if axis == "pitch":
            self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * n_pos - 1))
            idx = torch.arange(n_pos)
            self.register_buffer("rel_idx", (idx[None, :] - idx[:, None]) + n_pos - 1, persistent=False)

    @staticmethod
    def rope(x):
        N, H, L, D = x.shape
        half = D // 2
        freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        t = torch.arange(L, device=x.device, dtype=torch.float32)
        ang = t[:, None] * freqs[None]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:2 * half]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half:]], -1)

    def forward(self, x):
        B, C, T, P = x.shape
        if self.axis == "time":
            s = x.permute(0, 3, 2, 1).reshape(B * P, T, C)
        else:
            s = x.permute(0, 2, 3, 1).reshape(B * T, P, C)
        h = self.ln(s)
        N, L, _ = h.shape
        qkv = self.qkv(h).view(N, L, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = None
        if self.axis == "time":
            q, k = self.rope(q), self.rope(k)
        else:
            mask = self.rel_bias[:, self.rel_idx[:L, :L]].to(q.dtype)[None]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = self.out(a.transpose(1, 2).reshape(N, L, C)) * self.gamma
        if self.axis == "time":
            o = o.view(B, P, T, C).permute(0, 3, 2, 1)
        else:
            o = o.view(B, T, P, C).permute(0, 3, 1, 2)
        return x + o


class LaneGRU(nn.Module):
    """Bidirectional GRU along time for every lane (shared weights), projected back to C channels. Residual.
    On CUDA it runs in bfloat16 (float32 parameters cast per call), as it was trained."""

    def __init__(self, c, hidden=192, bf16=True):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.gru = nn.GRU(c, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, c)
        self.bf16 = bf16

    def forward(self, x):
        B, C, T, P = x.shape
        s = x.permute(0, 3, 2, 1).reshape(B * P, T, C)
        if self.bf16 and x.is_cuda:
            inp = self.ln(s).to(torch.bfloat16)
            w = [p.to(torch.bfloat16) for p in self.gru._flat_weights]
            hx = inp.new_zeros(2, inp.shape[0], self.gru.hidden_size)
            h, _ = torch._VF.gru(inp, hx, w, True, 1, 0.0, False, True, True)
        else:
            with torch.autocast("cuda", enabled=False):
                h, _ = self.gru(self.ln(s).float())
        o = self.out(h.to(s.dtype))
        return x + o.view(B, P, T, C).permute(0, 3, 2, 1)


class Stem(nn.Module):
    """HCQT (B, Cin, T10, 3 P) -> (B, C, T, P): two 3x3 convs, then a centred stride-(2, 3) conv to 20 ms x 1 bin per semitone."""

    def __init__(self, cin, c, mid=64, t_stride=2):
        super().__init__()
        self.inorm = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, mid, 3, padding=1)
        self.n1 = nn.BatchNorm2d(mid)
        self.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.n2 = nn.BatchNorm2d(mid)
        kt = 3 if t_stride == 2 else 1
        self.down = nn.Conv2d(mid, c, (kt, 3), stride=(t_stride, 3), padding=(kt // 2, 1))
        self.n3 = nn.BatchNorm2d(c)

    def forward(self, x):
        x = self.inorm(x)
        x = F.gelu(self.n1(self.c1(x)))
        x = F.gelu(self.n2(self.c2(x)))
        return self.n3(self.down(x))


class LaneNet(nn.Module):
    def __init__(self, c=368, depth=15, exp=2, kt=7, dilations=(1, 2, 4, 8), mid=64, t_stride=2, attn_at=(5, 10, 15),
                 attn_axes=("pitch", "time"), attn_heads=8, gru_at=(5, 10, 15), gru_hidden=192, gru_bf16=True,
                 heads=("onset", "frame", "offset", "onset2", "frame2", "offset2"), harmonics=(0.5, 1, 2, 3, 4, 5, 6, 7),
                 extra_nffts=(512,), n_fft=2048, refine=2, n_lanes=88,
                 pmix=(-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31)):
        super().__init__()
        self.n_lanes = n_lanes
        self.front = HCQT(n_fft=n_fft, harmonics=harmonics, extra_nffts=extra_nffts)
        self.hop_out = 0.01 * t_stride
        self.stem = Stem(self.front.n_channels, c, mid, t_stride)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(LaneBlock(c, kt=kt, dt=dilations[i % len(dilations)], exp=exp, pmix=pmix, n_lanes=n_lanes))
            if (i + 1) in attn_at:
                for ax in attn_axes:
                    self.layers.append(AxialAttn(c, ax, heads=attn_heads))
            if (i + 1) in gru_at:
                self.layers.append(LaneGRU(c, gru_hidden, bf16=gru_bf16))
        self.down = nn.Conv2d(c, c, (3, 1), stride=(2, 1), padding=(1, 0), groups=c)
        self.up_norm = nn.BatchNorm2d(c)
        self.refine = nn.ModuleList([LaneBlock(c, kt=kt, dt=1, exp=exp, pmix=pmix, n_lanes=n_lanes) for _ in range(refine)])
        self.head_names = tuple(heads)
        self.head = nn.Conv2d(c, len(self.head_names), 1)

    def forward(self, audio, taps: dict | None = None):
        """audio (B, n) -> {head: logits (B, T, 88)}. With ``taps`` (a dict), also stores taps["stem"] = the stem output
        (B, C, T, 96) and taps["hidden"] = the heads' input (B, C, T, 88)."""
        x = self.front(audio)
        h = self.stem(x)
        if taps is not None:
            taps["stem"] = h
        if self.n_lanes < h.shape[3]:
            h = h[:, :, :, :self.n_lanes]
        skip, T = h, h.shape[2]
        h = self.down(h)
        for layer in self.layers:
            h = layer(h)
        h = self.up_norm(h.repeat_interleave(2, dim=2)[:, :, :T] + skip)
        for layer in self.refine:
            h = layer(h)
        h = h[:, :, :, :N_KEYS]
        if taps is not None:
            taps["hidden"] = h
        o = self.head(h)
        return {n: o[:, i] for i, n in enumerate(self.head_names)}


# ══════════════════════════════════════════════════════════════════════
# transcriber/core
# ══════════════════════════════════════════════════════════════════════

# Running the transcriber over a whole recording.
#
# The network sees 8 s windows. A song is covered by overlapping windows (every 2 s); each window's outputs are weighted by a Hann
# taper and averaged per 20 ms frame. Besides the six probability maps the same pass can average two internal states for the later
# stages: ``hidden`` (the 368-d input of the output heads, per frame and key) and ``stem`` (the stem output, the least processed
# state). The same pass can also return a second average over every other window (every 4 s, the grouper's features), so the
# network runs once per recording.



HEADS = ("onset", "frame", "offset", "onset2", "frame2", "offset2")
WIN_S = 8.0


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


class _HostBuffer:
    """A (T, 88, C) float16 sum kept in host memory while the windows are added on the device.

    Windows arrive in order of their start, so a frame that lies before the current window's first frame is final: only the
    frames the current and the next windows touch (``cap`` frames, ~50 MB) live on the device; everything before them is divided
    by its weight sum (the same float16 division as the on-device path) and copied to the host once. The host array is allocated
    at its exact size (a pinned allocation would be rounded up to a power of two)."""

    def __init__(self, T, C, device, cap):
        self.T, self.device, self.cap = T, device, cap
        self.host = torch.zeros(T, N_KEYS, C, dtype=torch.float16)
        self.act = torch.zeros(cap, N_KEYS, C, dtype=torch.float16, device=device)
        self.base = 0                                  # song frame of act[0]

    def add(self, a, b, x, ws):
        if b > self.base + self.cap:
            self.flush(a, ws)
        self.act[a - self.base:b - self.base] += x

    def flush(self, upto, ws):
        """Finish and move out the frames [base, upto)."""
        upto = min(upto, self.T)
        while self.base < upto:
            n = min(upto - self.base, self.cap)
            self.host[self.base:self.base + n] = (self.act[:n] / ws[self.base:self.base + n, None, None].half()).cpu()
            self.act[:self.cap - n] = self.act[n:].clone()
            self.act[self.cap - n:] = 0
            self.base += n


class _Average:
    """Running Hann-weighted sums of the six maps and some taps over the windows added to it (see ``Transcriber.run``)."""

    def __init__(self, T, device, taps, host_taps=False, cap=0):
        self.T, self.device, self.taps, self.host_taps, self.cap = T, device, tuple(taps), host_taps, cap
        self.P = {k: torch.zeros(T, N_KEYS, device=device) for k in HEADS}
        self.buf = {}
        self.wsum = torch.zeros(T, device=device)

    def add(self, a, b, g0, k, wt, pr, tp):
        for hd in HEADS:
            self.P[hd][a:b] += pr[hd][k, a - g0:b - g0] * wt[:, None]
        for name in self.taps:
            v = tp[name]
            x = (v[k, :, a - g0:b - g0, :N_KEYS].permute(1, 2, 0).float() * wt[:, None, None]).half()
            if self.host_taps:
                if name not in self.buf:
                    self.buf[name] = _HostBuffer(self.T, v.shape[1], self.device, self.cap)
                self.buf[name].add(a, b, x, self.wsum.clamp(min=1e-6))
            else:
                if name not in self.buf:
                    self.buf[name] = torch.zeros(self.T, N_KEYS, v.shape[1], device=self.device, dtype=torch.float16)
                self.buf[name][a:b] += x
        self.wsum[a:b] += wt

    def finish(self):
        ws = self.wsum.clamp(min=1e-6)
        maps = {hd: (self.P[hd] / ws[:, None]).cpu().numpy() for hd in HEADS}
        out = {}
        for name, buf in self.buf.items():
            if self.host_taps:
                buf.flush(self.T, ws)
                out[name] = buf.host
            else:
                buf /= ws[:, None, None].half()
                out[name] = buf
        return maps, out


class Transcriber:
    """Audio (16 kHz mono) -> notes, two voices per key, each with a confidence.

    >>> t = Transcriber.from_pretrained("cuda")
    >>> maps, _ = t.run(audio)
    >>> notes = t.notes(maps)          # dict of arrays: onset, offset, pitch, conf, slot
    """

    def __init__(self, model: LaneNet, thresholds: dict, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.th = dict(thresholds)
        self.hop = model.hop_out

    @classmethod
    def from_pretrained(cls, device="cpu"):
        cfg = hub_load_config("transcriber")
        model = LaneNet(**cfg["model"])
        model.load_state_dict(hub_load_state("transcriber"))
        return cls(model, cfg["thresholds"], device)

    @property
    def channels(self) -> int:
        return self.model.head.in_channels

    # ------------------------------------------------------------------ windowed pass
    def _layout(self, n, hop_frac):
        W, Fw = int(round(WIN_S * SR)), int(round(WIN_S / self.hop))
        T = int(math.ceil(n / SR / self.hop))
        step = max(1, int(round(W * hop_frac)))
        pad = W // 2
        starts = list(range(0, n + pad, step))
        g_pad = int(round(pad / SR / self.hop))
        return W, Fw, T, starts, pad, g_pad

    def frame_offset(self, s0, g_pad):
        """Song frame of window frame 0 for a window starting at padded sample s0."""
        return int(round(s0 / SR / self.hop)) - g_pad

    @torch.no_grad()
    def windows(self, audio, hop_frac=0.25, batch=2, taps=(), pad_batch=False):
        """Yield (window starts, {head: sigmoid probs (b, Fw, 88)}, {tap: (b, C, Fw, lanes)}) batch by batch.
        pad_batch: fill the last batch with silent windows (one static batch shape)."""
        n = len(audio)
        W, Fw, T, starts, pad, g_pad = self._layout(n, hop_frac)
        x_all = np.concatenate([np.zeros(pad, np.float32), audio.astype(np.float32), np.zeros(pad + W, np.float32)])
        for bi in range(0, len(starts), batch):
            ss = starts[bi:bi + batch]
            xs = np.stack([x_all[s0:s0 + W] for s0 in ss])
            if pad_batch and len(ss) < batch:
                xs = np.concatenate([xs, np.zeros((batch - len(ss), W), np.float32)])
            x = torch.from_numpy(xs).to(self.device)
            box = {} if taps else None
            with _autocast(self.device):
                res = self.model(x, box)
            probs = {k: torch.sigmoid(res[k].float()) for k in HEADS}
            yield ss, probs, {k: box[k] for k in taps}

    @torch.no_grad()
    def run(self, audio, hop_frac=0.25, batch=2, taps=(), pad_batch=False, sub_hop_frac=None, sub_taps=(), buf_device=None):
        """-> (maps {head: (T, 88) float32 numpy}, {tap: (T, 88, C) float16 tensor}); with ``sub_hop_frac`` also a third
        element, the same pair for the sub-run (below).

        Each map / tap is the Hann-weighted mean over the windows covering the frame.

        sub_hop_frac: a second, coarser average taken from the SAME network outputs, over only the windows whose start (in
            padded samples) is a multiple of ``round(8 s * sub_hop_frac)``. Those are exactly the windows a separate run with
            ``hop_frac=sub_hop_frac`` would use, and they are accumulated in the same order, in the same precision and with their
            own weight sum, so the sub-average equals what that separate run returns (up to the batch size the windows ran in).
            It must be a multiple of ``hop_frac``. ``sub_taps`` names the taps it returns (it may hold taps ``taps`` lacks).
        buf_device: where the float16 tap buffers live (default: the model's device). "cpu" keeps them in host memory so a long
            recording does not need ~10 MB of GPU memory per second of audio: the windows are added on the device in a sliding
            ~50 MB working region and each frame is moved to the host once, when no later window can touch it (the result is the
            same, bit for bit)."""
        n = len(audio)
        W, Fw, T, starts, pad, g_pad = self._layout(n, hop_frac)
        w = torch.from_numpy(np.hanning(Fw + 2)[1:-1].astype(np.float32)).to(self.device)
        host = str(buf_device) == "cpu" and self.device.type != "cpu"
        cap = 2 * Fw                      # on-device working frames of a host-resident tap buffer
        acc = _Average(T, self.device, taps, host, cap)
        sub = None
        if sub_hop_frac is not None:
            sub_step = max(1, int(round(W * sub_hop_frac)))
            step = max(1, int(round(W * hop_frac)))
            if sub_step % step:
                raise ValueError("sub_hop_frac must be a multiple of hop_frac")
            sub = _Average(T, self.device, sub_taps, host, cap)
        want = tuple(dict.fromkeys(tuple(taps) + tuple(sub_taps)))
        for ss, pr, tp in self.windows(audio, hop_frac, batch, want, pad_batch):
            for k, s0 in enumerate(ss):
                g0 = self.frame_offset(s0, g_pad)
                a, b = max(0, g0), min(T, g0 + Fw)
                if b <= a:
                    continue
                wt = w[a - g0:b - g0]
                acc.add(a, b, g0, k, wt, pr, tp)
                if sub is not None and s0 % sub_step == 0:
                    sub.add(a, b, g0, k, wt, pr, tp)
        out = acc.finish()
        return out if sub is None else (*out, sub.finish())

    # ------------------------------------------------------------------ notes
    def notes(self, maps):
        """Decode both voices -> dict of arrays sorted by (onset, offset, pitch):
        onset / offset (s), pitch (MIDI), conf (the voice's onset probability at the onset frame), slot (1 / 2)."""
        th = self.th
        args = dict(off_th=th["off_th"], fr_th=th["fr_th"], fr_low_frames=th["fr_low_frames"], on_nms=th.get("on_nms", 0))
        rows = []
        for slot, keys, on_th in ((1, ("onset", "frame", "offset"), th["on_th"]), (2, ("onset2", "frame2", "offset2"), th.get("v2_th"))):
            if on_th is None:
                continue
            on = maps[keys[0]]
            for u0, u1, p in decode(maps[keys[0]], maps[keys[1]], maps[keys[2]], self.hop, on_th=on_th, **args):
                t = int(min(max(round(u0 / self.hop), 0), on.shape[0] - 1))
                rows.append((u0, u1, p, float(on[t, p - PITCH_LO]), slot))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        a = np.array([r[:3] for r in rows], np.float64).reshape(-1, 3)
        return dict(onset=a[:, 0], offset=a[:, 1], pitch=a[:, 2].astype(np.int64), conf=np.array([r[3] for r in rows], np.float32),
                    slot=np.array([r[4] for r in rows], np.int8))


# ══════════════════════════════════════════════════════════════════════
# evidence/features
# ══════════════════════════════════════════════════════════════════════

# The evidence vector the repairer reads for every transcribed note (769 numbers).
#
# For a note on key l from frame a to frame b (20 ms frames):
#
# * 33 map numbers from the transcriber's six probability maps: its own voice's onset probability at a-2..a+2, frame probability at
#   8 points over the note and at b-1+{1,3,6,12}, the highest frame probability in the 5 frames before a, the highest offset probability
#   over [a+1, b+12) with its relative position, the offset probability at b-1; the other voice's onset (max a-1..a+1), frame at a and
#   mean frame over the note; a second-voice flag; the first voice's onset probability (max a-1..a+1) on keys l-1, l+1, l-12, l+12,
#   l+19, l+24; the log spread of the two hidden-state vectors below.
# * 2 x 368 hidden-state numbers: the transcriber's last hidden state (the input of its output heads) at (a, l) and its mean over
#   [a, b) on l, overlap-averaged like the maps; each vector standardised over its 368 dims and quantised to int8 steps of 1/32.


C_H = 368
N_MAP = 33
N_EV = N_MAP + 2 * C_H          # 769
Q = 32.0
MAX_S_1PASS = 250.0             # longer recordings: the tap buffers go to host memory (dettato.pipeline), or, in the standalone
                                # transcribe_with_evidence below, the hidden state is gathered in a second pass (bounded memory)


def _map_feats(own, other, L, A, B):
    on, fr, off = own
    T_ = on.shape[0]
    c = lambda x: np.clip(x, 0, T_ - 1)  # noqa: E731
    span = np.maximum(B - A, 1)
    f = [on[c(A + d), L] for d in (-2, -1, 0, 1, 2)]
    f += [fr[c(A + (j * span) // 8), L] for j in range(8)]
    f += [fr[c(B - 1 + k), L] for k in (1, 3, 6, 12)]
    f.append(np.stack([np.where(A - k >= 0, fr[c(A - k), L], 0.0) for k in range(1, 6)]).max(0))
    om, opos = np.zeros(len(L), np.float32), np.zeros(len(L), np.float32)
    for i in range(len(L)):
        seg = off[A[i] + 1:min(B[i] + 12, T_), L[i]]
        if len(seg):
            j = int(seg.argmax())
            om[i], opos[i] = seg[j], min((j + 1) / span[i], 2.0)
    f += [om, opos, off[c(B - 1), L]]
    o_on, o_fr, _ = other
    f.append(np.stack([o_on[c(A + d), L] for d in (-1, 0, 1)]).max(0))
    f.append(o_fr[c(A), L])
    f.append(np.stack([o_fr[c(A + (j * span) // 4), L] for j in range(4)]).mean(0))
    return f


def map_evidence(maps, L, A, B, S):
    """-> (N, 31) float32: every map number (the two log-spread numbers are appended by the caller)."""
    out = np.zeros((len(L), N_MAP - 2), np.float32)
    s1 = (maps["onset"], maps["frame"], maps["offset"])
    s2 = (maps["onset2"], maps["frame2"], maps["offset2"])
    on1 = maps["onset"]
    T_ = on1.shape[0]
    for slot, own, other in ((1, s1, s2), (2, s2, s1)):
        m = S == slot
        if not m.any():
            continue
        f = _map_feats(own, other, L[m], A[m], B[m])
        f.append(np.full(int(m.sum()), float(slot == 2), np.float32))
        Lm, Am = L[m], A[m]
        for dl in (-1, 1, -12, 12, 19, 24):
            l2 = Lm + dl
            ok = (l2 >= 0) & (l2 < N_KEYS)
            v = np.stack([on1[np.clip(Am + d, 0, T_ - 1), np.clip(l2, 0, N_KEYS - 1)] for d in (-1, 0, 1)]).max(0)
            f.append(np.where(ok, v, 0.0))
        out[m] = np.stack(f, 1)
    return out


@torch.no_grad()
def gather_hidden(H, L, A, B, device, chunk=4):
    """H (T, 88, C), on the device or in host memory -> per note H[A, L] and mean(H[A:B, L]) (float32 numpy), a few lanes at a time."""
    N = len(L)
    at = np.zeros((N, C_H), np.float32)
    mean = np.zeros((N, C_H), np.float32)
    if not N:
        return at, mean
    lanes = np.unique(L)
    for i in range(0, len(lanes), chunk):
        ls = lanes[i:i + chunk]
        hl = H[:, torch.as_tensor(ls, device=H.device)].to(device).float()   # H may live in host memory
        cs = torch.cat([torch.zeros(1, len(ls), C_H, device=device), hl.cumsum(0)], 0)
        for u, l in enumerate(ls):
            sel = np.nonzero(L == l)[0]
            a_, b_ = torch.as_tensor(A[sel], device=device), torch.as_tensor(B[sel], device=device)
            at[sel] = hl[a_, u].cpu().numpy()
            mean[sel] = ((cs[b_, u] - cs[a_, u]) / (b_ - a_).clamp(min=1)[:, None].float()).cpu().numpy()
    return at, mean


@torch.no_grad()
def gather_hidden_2pass(transcriber, audio, L, A, B, batch=2, hop_frac=0.25):
    """The same two vectors without holding the whole song's hidden state: a second pass over the windows that adds each window's
    weighted share at the notes it touches."""
    device = transcriber.device
    hop = transcriber.hop
    W, Fw, Tsong, starts, pad, g_pad = transcriber._layout(len(audio), hop_frac)
    w = np.hanning(Fw + 2)[1:-1].astype(np.float32)
    wsum = np.zeros(Tsong, np.float32)
    for s0 in starts:
        g0 = transcriber.frame_offset(s0, g_pad)
        a, b = max(0, g0), min(Tsong, g0 + Fw)
        if b > a:
            wsum[a:b] += w[a - g0:b - g0]
    wsum = np.maximum(wsum, 1e-6)
    N = len(L)
    at = torch.zeros(N, C_H, device=device)
    mean = torch.zeros(N, C_H, device=device)
    if not N:
        return at.cpu().numpy(), mean.cpu().numpy()
    Lt, At, Bt = (torch.as_tensor(v, device=device) for v in (L, A, B))
    wsum_t = torch.as_tensor(wsum, device=device)
    for ss, _, taps in transcriber.windows(audio, hop_frac, batch, taps=("hidden",)):
        H = taps["hidden"][:, :, :, :N_KEYS].float()
        for k, s0 in enumerate(ss):
            g0 = transcriber.frame_offset(s0, g_pad)
            lo, hi = max(0, g0), min(Tsong, g0 + Fw)
            if hi <= lo:
                continue
            sel = torch.nonzero((Bt > lo) & (At < hi)).flatten()
            if not len(sel):
                continue
            ul, inv = torch.unique(Lt[sel], return_inverse=True)
            wt = torch.as_tensor(w[lo - g0:hi - g0], device=device) / wsum_t[lo:hi]
            hs = H[k][:, lo - g0:hi - g0, :][:, :, ul].permute(1, 2, 0) * wt[:, None, None]
            cs = torch.cat([torch.zeros(1, len(ul), C_H, device=device), hs.cumsum(0)], 0)
            a_, b_ = At[sel], Bt[sel]
            in_at = (a_ >= lo) & (a_ < hi)
            if in_at.any():
                at[sel[in_at]] += hs[(a_[in_at] - lo), inv[in_at]]
            s_lo, s_hi = (torch.clamp(a_, lo, hi) - lo), (torch.clamp(b_, lo, hi) - lo)
            mean[sel] += cs[s_hi, inv] - cs[s_lo, inv]
    mean /= (Bt - At).clamp(min=1)[:, None].float()
    return at.cpu().numpy(), mean.cpu().numpy()


def _std_q(x):
    mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True) + 1e-6
    z = np.clip((x - mu) / sd, -3.96, 3.96)
    return np.round(z * Q).astype(np.int8), np.log(sd[:, 0])


def _decode_maps(transcriber, maps):
    """-> notes, their (key, onset frame, end frame, voice) indices and the 31 map numbers of each."""
    hop = transcriber.hop
    res = transcriber.notes(maps)
    T_ = maps["onset"].shape[0]
    L = (res["pitch"] - PITCH_LO).astype(np.int64)
    A = np.clip(np.round(res["onset"] / hop).astype(np.int64), 0, T_ - 1)
    B = np.clip(np.round(res["offset"] / hop).astype(np.int64), A + 1, T_)
    S = res["slot"].astype(np.int64)
    return res, L, A, B, map_evidence(maps, L, A, B, S)


def _assemble(mf, at, mean):
    qa, la = _std_q(at)
    qm, lmn = _std_q(mean)
    evf = np.concatenate([mf, la[:, None], lmn[:, None]], 1).astype(np.float16)
    evh = np.concatenate([qa, qm], 1)
    return np.concatenate([evf.astype(np.float32), evh.astype(np.float32) / Q], 1)


def evidence_from_run(transcriber, maps, H):
    """The transcriber's maps and hidden state H (T, 88, C float16, on the device or in host memory), both averaged over all
    windows -> (notes dict as Transcriber.notes, ev (N, 769) float32 row-aligned with the notes)."""
    res, L, A, B, mf = _decode_maps(transcriber, maps)
    at, mean = gather_hidden(H, L, A, B, transcriber.device)
    return res, _assemble(mf, at, mean)


def transcribe_with_evidence(transcriber, audio, batch=2):
    """Run the transcriber (its own pass) and compute every note's evidence.
    -> (notes dict as Transcriber.notes, maps, ev (N, 769) float32 row-aligned with the notes).
    ``Dettato.transcribe`` does not call this: it runs the transcriber once for the repairer and the grouper together and uses
    ``evidence_from_run``."""
    one_pass = len(audio) <= MAX_S_1PASS * SR
    maps, buf = transcriber.run(audio, hop_frac=0.25, batch=batch, taps=("hidden",) if one_pass else ())
    if one_pass:
        res, ev = evidence_from_run(transcriber, maps, buf.pop("hidden"))
        return res, maps, ev
    res, L, A, B, mf = _decode_maps(transcriber, maps)
    at, mean = gather_hidden_2pass(transcriber, audio, L, A, B, batch)
    return res, maps, _assemble(mf, at, mean)


# ══════════════════════════════════════════════════════════════════════
# evidence/model
# ══════════════════════════════════════════════════════════════════════

# The repairer: a note-level transformer that edits the transcriber's notes.
#
# One token per transcribed note (pitch, a confidence-derived velocity, onset and duration within an 8 s window, a few context
# features, and the note's 769-number evidence vector). A 9-layer transformer (rotary positions on onset time and pitch) predicts per
# note: P(keep), an onset correction (5 ms bins over +-0.5 s), a duration ratio (log steps of 0.05 over +-4) and, in two insertion slots,
# P(a note is missing here) with its pitch offset, onset offset and duration.


L_WIN = 8.0                                     # window length (s)
DON_STEP, DON_MAX = 0.005, 0.5                  # onset correction bins
N_DON = int(round(2 * DON_MAX / DON_STEP)) + 1  # 201
N_DUR = 128                                     # absolute duration bins, log-spaced 0.01 .. 16 s
DUR_EDGES = np.exp(np.linspace(np.log(0.01), np.log(16.0), N_DUR + 1))
DUR_CENT = np.sqrt(DUR_EDGES[:-1] * DUR_EDGES[1:]).astype(np.float32)
DR_STEP, DR_MAX = 0.05, 4.0                     # duration-ratio bins: log(new / old)
N_DR = int(round(2 * DR_MAX / DR_STEP)) + 1     # 161
INS_T_STEP, INS_T_MIN, INS_T_MAX = 0.01, -1.0, 3.0
N_INS_T = int(round((INS_T_MAX - INS_T_MIN) / INS_T_STEP)) + 1   # 401
INS_P_MAX = 24
N_INS_P = 2 * INS_P_MAX + 1                     # 49
N_SLOT = 2

CAND_DT = np.dtype([("inst", "i2"), ("pitch", "i2"), ("on", "f4"), ("off", "f4"), ("score", "f4"), ("kind", "i1")])


# ----------------------------------------------------------------------------- building blocks
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def fourier(x, n, lo, hi):
    """x [...] -> [..., 2n] sin / cos features with wavelengths geometric in [lo, hi]."""
    lam = torch.exp(torch.linspace(math.log(lo), math.log(hi), n, device=x.device))
    a = 2 * math.pi * x.float()[..., None] / lam
    return torch.cat([a.sin(), a.cos()], -1)


class MultiAxisRope(nn.Module):
    """Rotary embedding over continuous axes (onset seconds, pitch)."""

    def __init__(self, head_dim, axes):
        super().__init__()
        self.axes = axes
        self.used = 2 * sum(a[0] for a in axes)
        assert self.used <= head_dim
        for i, (n, lo, hi) in enumerate(axes):
            lam = torch.exp(torch.linspace(math.log(lo), math.log(hi), n))
            self.register_buffer(f"w{i}", 2 * math.pi / lam, persistent=False)

    def angles(self, positions):
        angs = [p.float()[..., None] * getattr(self, f"w{i}") for i, p in enumerate(positions)]
        a = torch.cat(angs, -1)[:, None]
        return a.cos(), a.sin()

    def apply(self, x, cs):
        cos, sin = cs
        u = self.used
        xr, xp = x[..., :u].float(), x[..., u:]
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        xr = torch.stack([r1, r2], -1).flatten(-2)
        return torch.cat([xr.to(x.dtype), xp], -1)


class Attention(nn.Module):
    def __init__(self, d, heads, rope=None):
        super().__init__()
        self.h = heads
        self.hd = d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = rope

    def forward(self, x, mask=None, cs=None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, N, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope is not None and cs is not None:
            q = self.rope.apply(q, cs)
            k = self.rope.apply(k, cs)
        am = mask[:, None, None, :] if mask is not None else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(o.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, -1)
        return self.w2(F.silu(a) * b)


class EvBlock(nn.Module):
    def __init__(self, d, heads, rope=None, ff_mult=3):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.att = Attention(d, heads, rope)
        self.n2 = RMSNorm(d)
        self.ff = SwiGLU(d, int(round(d * ff_mult / 32)) * 32)

    def forward(self, x, mask=None, cs=None):
        x = x + self.att(self.n1(x), mask=mask, cs=cs)
        return x + self.ff(self.n2(x))


class Backbone(nn.Module):
    def __init__(self, d, layers, heads, rope_t=10, rope_p=4, ff_mult=3):
        super().__init__()
        self.rope = MultiAxisRope(d // heads, [(rope_t, 0.02, 40.0), (rope_p, 2.0, 96.0)])
        self.blocks = nn.ModuleList([EvBlock(d, heads, self.rope, ff_mult) for _ in range(layers)])
        self.norm = RMSNorm(d)

    def forward(self, x, b):
        cs = self.rope.angles([b["on"].float(), b["pitch"].float()])
        for blk in self.blocks:
            x = blk(x, mask=b["mask"], cs=cs)
        return self.norm(x)


# ----------------------------------------------------------------------------- note embedding
def context_features(b, block=256):
    """Per-note context from its neighbours [B, N, 8]: gaps to the previous / next note on the same key, overlap with the next one,
    chord size, note density within 0.5 s, same-key notes within 0.25 s."""
    on, dur, mask = b["on"], b["dur"], b["mask"]
    off = on + dur
    key = b["inst"] * 128 + b["pitch"]
    big = 10.0
    N = on.shape[1]
    outs = []
    for a in range(0, N, block):
        z = slice(a, min(a + block, N))
        mq = mask[:, z]
        same_key = (key[:, z, None] == key[:, None, :]) & mask[:, None, :] & mq[:, :, None]
        same_inst = (b["inst"][:, z, None] == b["inst"][:, None, :]) & mask[:, None, :] & mq[:, :, None]
        dt = on[:, None, :] - on[:, z, None]
        ar = torch.arange(N, device=on.device)
        eye = (ar[None, :] == ar[z][:, None])[None]
        sk = same_key & ~eye
        prev_dt = torch.where(sk & (dt <= 0), -dt, torch.full_like(dt, big)).min(-1)
        prev_gap_on = prev_dt.values
        prev_off = torch.gather(off, 1, prev_dt.indices)
        onq = on[:, z]
        prev_gap_off = torch.where(prev_gap_on < big, onq - prev_off, torch.full_like(onq, big))
        nxt = torch.where(sk & (dt > 0), dt, torch.full_like(dt, big)).min(-1)
        next_gap_on = nxt.values
        next_overlap = torch.where(next_gap_on < big, off[:, z] - (onq + next_gap_on), torch.full_like(onq, -big))
        adt = dt.abs()
        chord = ((adt < 0.035) & same_inst).float().sum(-1)
        dens = ((adt < 0.5) & mask[:, None, :]).float().sum(-1)
        dens_inst = ((adt < 0.5) & same_inst).float().sum(-1)
        same_p_near = ((adt < 0.25) & sk).float().sum(-1)

        def sl(x):
            return torch.sign(x) * torch.log1p(x.abs() * 20)
        outs.append(torch.stack([sl(prev_gap_on), sl(prev_gap_off), sl(next_gap_on), sl(next_overlap),
                                 torch.log1p(chord), torch.log1p(dens), torch.log1p(dens_inst),
                                 torch.log1p(same_p_near)], -1))
    f = torch.cat(outs, 1)
    return f * mask[..., None]


def grid_features(b, n_per=2, subdiv=(1, 2, 3, 4, 6, 8)):
    """Parameter-free beat-grid estimate per window (comb filter over 72 periods 0.18-1.5 s) -> per-note phase features [B, N, 28]."""
    on, mask = b["on"].float(), b["mask"]
    dev = on.device
    P = torch.exp(torch.linspace(math.log(0.18), math.log(1.5), 72, device=dev))
    w = mask.float()
    ang = 2 * math.pi * on[:, :, None] / P[None, None, :]
    zr = (torch.cos(ang) * w[..., None]).sum(1)
    zi = (torch.sin(ang) * w[..., None]).sum(1)
    n = w.sum(1, keepdim=True).clamp(min=1)
    R = torch.sqrt(zr ** 2 + zi ** 2) / n
    phase = torch.atan2(zi, zr)
    feats = []
    Rm = R.clone()
    for _ in range(n_per):
        k = Rm.argmax(1)
        Pk = P[k][:, None]
        ph = phase.gather(1, k[:, None])
        rk = R.gather(1, k[:, None])
        base = 2 * math.pi * on / Pk - ph
        for m in subdiv:
            feats += [torch.sin(m * base), torch.cos(m * base)]
        feats += [rk.expand_as(on), torch.log(Pk).expand_as(on)]
        kk = torch.arange(len(P), device=dev)[None]
        Rm = Rm.masked_fill((kk - k[:, None]).abs() <= 3, -1.0)
    return torch.stack(feats, -1) * mask[..., None]


class NoteEmbed(nn.Module):
    def __init__(self, d, n_ev):
        super().__init__()
        self.inst = nn.Embedding(130, d)
        self.pitch = nn.Embedding(256, d)
        self.vel = nn.Embedding(16, d)
        self.cont = nn.Sequential(nn.Linear(24 + 24 + 24 + 2 + 8 + 28, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = RMSNorm(d)
        self.evm = nn.Sequential(nn.Linear(n_ev, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, b):
        drum = (b["inst"] == 128).long()
        x = self.inst(b["inst"]) + self.pitch(b["pitch"] + 128 * drum) + self.vel(b["vel"] // 8)
        ld = torch.log(b["dur"].clamp(min=1e-4))
        c = [fourier(b["on"], 12, 0.01, 32.0), fourier(ld, 12, 0.04, 40.0), fourier(b["on"] + b["dur"], 12, 0.01, 32.0),
             (ld[..., None] + 3) / 3, b["dur"][..., None].clamp(max=8) / 2, context_features(b), grid_features(b)]
        x = x + self.cont(torch.cat(c, -1)) + self.evm(b["ev"].to(x.dtype))
        return self.norm(x)


# ----------------------------------------------------------------------------- the model
def _box_argmax(p, half):
    shp = p.shape
    k = torch.ones(1, 1, 2 * half + 1, device=p.device)
    s = F.conv1d(p.reshape(-1, 1, shp[-1]).float(), k, padding=half)[:, 0].view(shp)
    return s.argmax(-1), s.max(-1).values


class Repairer(nn.Module):
    def __init__(self, d=192, layers=9, heads=6, ff_mult=3, n_ev=769):
        super().__init__()
        self.emb = NoteEmbed(d, n_ev)
        self.bb = Backbone(d, layers, heads, ff_mult=ff_mult)
        self.keep = nn.Linear(d, 1)
        self.don = nn.Linear(d, N_DON)
        self.dur = nn.Linear(d, N_DR)
        self.ins = nn.ModuleList([nn.Linear(d, 1 + N_INS_P + N_INS_T + N_DUR) for _ in range(N_SLOT)])
        self._W = None

    def forward(self, b):
        h = self.bb(self.emb(b), b)
        return dict(keep=self.keep(h).squeeze(-1), don=self.don(h), dur=self.dur(h), ins=torch.stack([m(h) for m in self.ins], 2))

    def _dur_matrix(self, dev):
        if self._W is None or self._W.device != dev:
            c = torch.tensor(DUR_CENT, device=dev)
            tol = torch.clamp(0.2 * c[:, None], min=0.05)
            self._W = ((c[None, :] - c[:, None]).abs() <= tol).float()
        return self._W

    @torch.no_grad()
    def candidates(self, b, tol_on=0.05):
        """-> per window a CAND_DT array: the n input notes edited (kind 0: score = P(keep), corrected onset / end), then 2 n
        insertion proposals (kind 1: score = P(missing note))."""
        o = self(b)
        W = self._dur_matrix(b["on"].device)
        pk = torch.sigmoid(o["keep"].float())
        don_b, _ = _box_argmax(torch.softmax(o["don"].float(), -1), int(round(tol_on / DON_STEP)))
        cent = torch.tensor(DUR_CENT, device=pk.device)
        rb, _ = _box_argmax(torch.softmax(o["dur"].float(), -1), 3)
        dur_k = b["dur"].clamp(min=1e-4) * torch.exp(rb.float() * DR_STEP - DR_MAX)
        on_k = b["on"] + (don_b.float() * DON_STEP - DON_MAX)
        ins = o["ins"].float()
        P, T = N_INS_P, N_INS_T
        pe = torch.sigmoid(ins[..., 0])
        ip = ins[..., 1:1 + P].argmax(-1) - INS_P_MAX
        it, _ = _box_argmax(torch.softmax(ins[..., 1 + P:1 + P + T], -1), int(round(tol_on / INS_T_STEP)))
        idd = (torch.softmax(ins[..., 1 + P + T:], -1) @ W).argmax(-1)
        on_i = b["on"][..., None] + (it.float() * INS_T_STEP + INS_T_MIN)
        dur_i = cent[idd]
        pitch_i = (b["pitch"][..., None] + ip).clamp(0, 127)
        res = []
        mask = b["mask"].cpu().numpy()
        arrs = {k: v.cpu().numpy() for k, v in dict(pk=pk, on_k=on_k, dur_k=dur_k, pe=pe, on_i=on_i, dur_i=dur_i,
                                                     pitch_i=pitch_i, inst=b["inst"], pitch=b["pitch"]).items()}
        for i in range(mask.shape[0]):
            n = int(mask[i].sum())
            S = N_SLOT
            c = np.zeros(n + n * S, CAND_DT)
            c["inst"][:n] = arrs["inst"][i, :n]
            c["pitch"][:n] = arrs["pitch"][i, :n]
            c["on"][:n] = arrs["on_k"][i, :n]
            c["off"][:n] = arrs["on_k"][i, :n] + arrs["dur_k"][i, :n]
            c["score"][:n] = arrs["pk"][i, :n]
            c["kind"][:n] = 0
            c["inst"][n:] = np.repeat(arrs["inst"][i, :n], S)
            c["pitch"][n:] = arrs["pitch_i"][i, :n].reshape(-1)
            c["on"][n:] = arrs["on_i"][i, :n].reshape(-1)
            c["off"][n:] = (arrs["on_i"][i, :n] + arrs["dur_i"][i, :n]).reshape(-1)
            c["score"][n:] = arrs["pe"][i, :n].reshape(-1)
            c["kind"][n:] = 1
            res.append(c)
        return res


# ══════════════════════════════════════════════════════════════════════
# evidence/repair
# ══════════════════════════════════════════════════════════════════════

# Applying the repairer to a whole recording: 8 s windows of transcribed notes -> kept / moved / dropped notes + inserted notes.


MAX_WIN = 4096          # windows with more notes than this pass through unchanged
ONSET_TH = 0.25         # the transcriber's onset threshold: confidence [.25, 1] -> velocity 1..127
DEDUP_S = 0.03

NOTE_DT = np.dtype([("inst", "u1"), ("pitch", "u1"), ("vel", "u1"), ("on", "f4"), ("off", "f4")])


def conf_vel(conf):
    x = (np.asarray(conf, np.float64) - ONSET_TH) / (1.0 - ONSET_TH)
    return np.clip(np.round(1 + 126 * x), 1, 127).astype(np.uint8)


def input_notes(res):
    """Transcriber notes -> the repairer's note array sorted by (onset, pitch), and that sort order."""
    t = np.round(res["onset"], 4)
    off = np.round(res["offset"], 4)
    n = len(t)
    k = np.zeros(n, NOTE_DT)
    if not n:
        return k, np.zeros(0, np.int64)
    k["inst"] = 0
    k["pitch"] = np.asarray(res["pitch"], np.int64)
    k["vel"] = conf_vel(res["conf"])
    on = np.asarray(t, np.float64)
    k["on"], k["off"] = on, np.maximum(np.asarray(off, np.float64), on + 1e-3)
    o = np.lexsort((k["pitch"], k["inst"], k["on"]))
    return k[o], o


def _window_batch(k, ev, t0, device):
    n = len(k)
    b = dict(inst=k["inst"].astype(np.int64), pitch=k["pitch"].astype(np.int64), vel=k["vel"].astype(np.int64),
             on=(k["on"] - t0).astype(np.float32), dur=np.maximum(k["off"] - k["on"], 0).astype(np.float32))
    out = {f: torch.from_numpy(v[None]).to(device) for f, v in b.items()}
    out["mask"] = torch.ones(1, max(1, n), dtype=torch.bool, device=device)
    out["ev"] = torch.from_numpy(np.asarray(ev, np.float32)[None]).to(device)
    return out


def _dedup_idx(c, tol=DEDUP_S):
    """Same key within tol -> keep the higher score. Returns kept indices (sorted)."""
    if len(c) < 2:
        return np.arange(len(c))
    key = c["inst"].astype(np.int64) * 128 + c["pitch"]
    o = np.lexsort((c["on"], key))
    cs, ks = c[o], key[o]
    keep = np.ones(len(cs), bool)
    close = (ks[1:] == ks[:-1]) & (cs["on"][1:] - cs["on"][:-1] < tol)
    for j in np.nonzero(close)[0]:
        a, b = j, j + 1
        if not keep[a]:
            continue
        if cs["score"][a] >= cs["score"][b]:
            keep[b] = False
        else:
            keep[a] = False
    return np.sort(o[keep])


def _safe_dedup_idx(c, tol=DEDUP_S):
    """A kept note is never removed: an insertion is dropped when a kept note of the same pitch starts within tol, and the
    insertions among themselves go through _dedup_idx. (The transcriber's two voices can hold two notes on one key.)"""
    keep_m = c["kind"] == 0
    ins_i = np.nonzero(~keep_m)[0]
    if not len(ins_i):
        return np.arange(len(c))
    ok = np.ones(len(ins_i), bool)
    kp = {}
    for q in np.nonzero(keep_m)[0]:
        kp.setdefault(int(c["pitch"][q]), []).append(float(c["on"][q]))
    kp = {p: np.sort(v) for p, v in kp.items()}
    for j, q in enumerate(ins_i):
        v = kp.get(int(c["pitch"][q]))
        if v is not None:
            i = np.searchsorted(v, c["on"][q])
            ok[j] = not any(0 <= r < len(v) and abs(v[r] - c["on"][q]) < tol for r in (i - 1, i))
    ins_ok = ins_i[ok]
    ins_ok = ins_ok[_dedup_idx(c[ins_ok])]
    return np.sort(np.concatenate([np.nonzero(keep_m)[0], ins_ok]))


class EvidenceRepairer:
    """Edits the transcriber's notes using its own evidence: drops notes it judges wrong, corrects onsets and ends, inserts
    missing notes.

    >>> rep = EvidenceRepairer.from_pretrained("cuda")
    >>> notes = rep(transcriber, audio16k)       # dict: onset, offset, pitch, inserted (bool), score
    """

    def __init__(self, model: Repairer, keep_th=0.4, insert_th=0.6, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.keep_th, self.insert_th = float(keep_th), float(insert_th)

    @classmethod
    def from_pretrained(cls, device="cpu"):
        cfg = hub_load_config("evidence")
        model = Repairer(**cfg["model"])
        model.load_state_dict(hub_load_state("evidence"))
        return cls(model, cfg["keep_th"], cfg["insert_th"], device)

    @torch.no_grad()
    def edits(self, k, ev):
        """Per input note: P(keep), repaired onset / end (NaN = its window passed through unchanged); insertion proposals
        [(onset, end, pitch, score)], all in seconds from the start of the recording."""
        n = len(k)
        pk, on_r, off_r = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        ins = []
        if not n:
            return pk, on_r, off_r, ins
        end, on = float(k["off"].max()), k["on"]
        for t0 in np.arange(0, end + L_WIN, L_WIN):
            lo, hi = int(np.searchsorted(on, t0)), int(np.searchsorted(on, t0 + L_WIN))
            if hi <= lo or hi - lo > MAX_WIN:
                continue
            c = self.model.candidates(_window_batch(k[lo:hi], ev[lo:hi], float(t0), self.device))[0]
            m = hi - lo
            kc = c[:m]
            ok = (kc["on"] >= 0) & (kc["on"] < L_WIN)      # a note whose corrected onset leaves the window is dropped
            pk[lo:hi] = np.where(ok, kc["score"], -1.0)
            on_r[lo:hi], off_r[lo:hi] = kc["on"] + t0, kc["off"] + t0
            for i in range(m, len(c)):
                x = c[i]
                if x["kind"] != 0 and 0 <= x["on"] < L_WIN:
                    ins.append((float(x["on"]) + t0, float(x["off"]) + t0, int(x["pitch"]), float(x["score"])))
        return pk, on_r, off_r, ins

    def apply(self, k, e):
        """Input notes + edits -> the repaired notes (dict of arrays sorted by onset, pitch)."""
        pk, on_r, off_r, ins = e
        has = ~np.isnan(pk)
        rows = []                                               # (on, off, pitch, score, kind)
        for j in range(len(k)):
            x = k[j]
            if not has[j]:
                rows.append((float(x["on"]), float(x["off"]), int(x["pitch"]), 1.0, 0))
            elif pk[j] >= self.keep_th:
                rows.append((on_r[j], off_r[j], int(x["pitch"]), pk[j], 0))
        rows += [(a, b, p, s, 1) for a, b, p, s in ins if s >= self.insert_th]
        c = np.zeros(len(rows), [("inst", "i2"), ("pitch", "i2"), ("on", "f8"), ("off", "f8"), ("score", "f4"), ("kind", "i1")])
        for i, r in enumerate(rows):
            c[i] = (0, r[2], r[0], r[1], r[3], r[4])
        if len(c):
            c = c[_safe_dedup_idx(c)]
        on = np.array([round(float(a), 4) for a in c["on"]], np.float64)
        off = np.array([round(max(float(b), float(a) + 0.01), 4) for a, b in zip(c["on"], c["off"])], np.float64)
        return dict(onset=on, offset=off, pitch=c["pitch"].astype(np.int64), inserted=c["kind"] == 1,
                    score=c["score"].astype(np.float32))

    def _run(self, res, ev):
        k, o = input_notes(res)
        e = self.edits(k, ev[o])
        rep = self.apply(k, e)
        info = dict(transcribed=len(k), kept=int((~rep["inserted"]).sum()), inserted=int(rep["inserted"].sum()))
        info["dropped"] = info["transcribed"] - info["kept"]
        return rep, res, info

    def __call__(self, transcriber, audio, batch=2):
        """audio: 16 kHz mono float32 -> (repaired notes, transcriber notes, info). Runs the transcriber itself."""
        res, _, ev = transcribe_with_evidence(transcriber, audio, batch=batch)
        return self._run(res, ev)

    def from_run(self, transcriber, maps, hidden):
        """The same from a transcriber pass already made: its maps and its hidden state (T, 88, 368 float16, on the device or in
        host memory), both averaged over every window (``Transcriber.run(audio, hop_frac=0.25, taps=("hidden",))``)."""
        res, ev = evidence_from_run(transcriber, maps, hidden)
        return self._run(res, ev)


# ══════════════════════════════════════════════════════════════════════
# grouper/instruments
# ══════════════════════════════════════════════════════════════════════

# The grouper's instrument vocabulary: 73 instruments, each with its General MIDI programs and one or more families.
#
# General MIDI programs that are the same real instrument share one name (Acoustic Grand / Bright / Electric Grand / Honky-tonk
# Piano are all "piano"; the 8 Synth Lead programs are one "synth_lead"). A track is written with the first program of its
# instrument. Families are used when the decode picks a track's name: it maximises the expected credit (1 for the right
# instrument, 0.5 for another instrument of a shared family), not the raw probability.


_TABLE = [
    # name               GM programs (0-based)              families
    ("piano",            [0, 1, 2, 3],                       ["keys"]),
    ("electric_piano",   [4, 5],                             ["keys"]),
    ("harpsichord",      [6],                                ["keys"]),
    ("clavinet",         [7],                                ["keys"]),
    ("harp",             [46],                               ["keys", "plucked"]),
    ("celesta",          [8],                                ["mallets", "keys"]),
    ("glockenspiel",     [9],                                ["mallets"]),
    ("music_box",        [10],                               ["mallets"]),
    ("vibraphone",       [11],                               ["mallets"]),
    ("marimba",          [12],                               ["mallets"]),
    ("xylophone",        [13],                               ["mallets"]),
    ("tubular_bells",    [14],                               ["mallets"]),
    ("dulcimer",         [15],                               ["plucked", "mallets"]),
    ("kalimba",          [108],                              ["mallets"]),
    ("tinkle_bell",      [112],                              ["mallets"]),
    ("steel_drums",      [114],                              ["mallets"]),
    ("organ",            [16, 17, 18],                       ["organ"]),
    ("church_organ",     [19],                               ["organ"]),
    ("reed_organ",       [20],                               ["organ"]),
    ("accordion",        [21, 23],                           ["organ"]),
    ("harmonica",        [22],                               ["organ", "reeds"]),
    ("acoustic_guitar",  [24, 25],                           ["plucked"]),
    ("electric_guitar",  [26, 27, 28, 29, 30, 31],           ["plucked"]),
    ("sitar",            [104],                              ["plucked"]),
    ("banjo",            [105],                              ["plucked"]),
    ("shamisen",         [106],                              ["plucked"]),
    ("koto",             [107],                              ["plucked"]),
    ("double_bass",      [32, 43],                           ["bass", "strings"]),
    ("electric_bass",    [33, 34, 35, 36, 37],               ["bass"]),
    ("synth_bass",       [38, 39],                           ["bass", "synth"]),
    ("violin",           [40, 110],                          ["strings"]),
    ("viola",            [41],                               ["strings"]),
    ("cello",            [42],                               ["strings"]),
    ("string_ensemble",  [44, 45, 48, 49],                   ["strings"]),
    ("synth_strings",    [50, 51],                           ["strings", "synth"]),
    ("voice",            [52, 53],                           ["voice"]),
    ("synth_voice",      [54],                               ["voice", "synth"]),
    ("trumpet",          [56, 59],                           ["brass"]),
    ("trombone",         [57],                               ["brass"]),
    ("tuba",             [58],                               ["brass"]),
    ("french_horn",      [60],                               ["brass"]),
    ("brass_section",    [61],                               ["brass"]),
    ("synth_brass",      [62, 63],                           ["brass", "synth"]),
    ("soprano_sax",      [64],                               ["reeds"]),
    ("alto_sax",         [65],                               ["reeds"]),
    ("tenor_sax",        [66],                               ["reeds"]),
    ("baritone_sax",     [67],                               ["reeds"]),
    ("oboe",             [68],                               ["reeds"]),
    ("english_horn",     [69],                               ["reeds"]),
    ("bassoon",          [70],                               ["reeds"]),
    ("clarinet",         [71],                               ["reeds"]),
    ("bagpipe",          [109],                              ["reeds"]),
    ("shanai",           [111],                              ["reeds"]),
    ("piccolo",          [72],                               ["flutes"]),
    ("flute",            [73],                               ["flutes"]),
    ("recorder",         [74],                               ["flutes"]),
    ("pan_flute",        [75],                               ["flutes"]),
    ("blown_bottle",     [76],                               ["flutes"]),
    ("shakuhachi",       [77],                               ["flutes"]),
    ("whistle",          [78],                               ["flutes"]),
    ("ocarina",          [79],                               ["flutes"]),
    ("orchestra_hit",    [55],                               ["synth"]),
    ("synth_lead",       list(range(80, 88)),                ["synth"]),
    ("synth_pad",        list(range(88, 96)),                ["synth"]),
    ("synth_fx",         list(range(96, 104)),               ["synth"]),
    ("timpani",          [47],                               ["percussion"]),
    ("agogo",            [113],                              ["percussion"]),
    ("woodblock",        [115],                              ["percussion"]),
    ("taiko",            [116],                              ["percussion"]),
    ("melodic_tom",      [117],                              ["percussion"]),
    ("synth_drum",       [118],                              ["percussion", "synth"]),
    ("reverse_cymbal",   [119],                              ["percussion"]),
    ("sound_effect",     list(range(120, 128)),              ["sfx"]),
]

INSTRUMENTS = [name for name, _, _ in _TABLE]
PROGRAMS = {name: tuple(progs) for name, progs, _ in _TABLE}
FAMILIES = {name: tuple(fam) for name, _, fam in _TABLE}
K = len(INSTRUMENTS)
PROGRAM = np.array([progs[0] for _, progs, _ in _TABLE], np.int64)      # the GM program a track of each instrument is written with


def _credit(a, b):
    if a == b:
        return 1.0
    return 0.5 if set(FAMILIES[a]) & set(FAMILIES[b]) else 0.0


CREDIT = np.array([[_credit(a, b) for b in INSTRUMENTS] for a in INSTRUMENTS], np.float64)    # (predicted, true)


# ══════════════════════════════════════════════════════════════════════
# grouper/retune
# ══════════════════════════════════════════════════════════════════════

# Tuning normalisation for the grouper's features.
#
# A recording that sits clearly off A440 is resampled so that it does (pitch and time move together); the grouper reads the
# transcriber's features on that audio and every time is mapped back afterwards (t_source = t_retuned x r).


RETUNE_SR = 16000
MIN_CENTS = 10.0          # retune only when the estimate is at least this far from A440 ...
MIN_STRENGTH = 0.12       # ... and the estimate is consistent enough (resultant length of the circular mean)


def retune_tuning(y, n_fft=8192, hop=1024, H=6, fmin=65.0, fmax=1000.0, step_c=5.0, rel=0.5):
    """-> (offset from A440 in cents, in [-50, 50), strength in [0, 1]).

    Harmonic salience (harmonics 1..6, weights .85^(h-1)) of the square-root magnitude STFT on a 5-cent grid from 65 Hz to 1 kHz;
    per frame the local maxima above half the frame maximum, parabolically refined; their weighted circular mean modulo 100 c."""
    if len(y) < n_fft:
        return 0.0, 0.0
    win = np.hanning(n_fft).astype(np.float32)
    nfr = 1 + (len(y) - n_fft) // hop
    cents = np.arange(1200 * np.log2(fmin / 440), 1200 * np.log2(fmax / 440), step_c)
    f0 = 440 * 2 ** (cents / 1200)
    cs, ws = [], []
    for b0 in range(0, nfr, 512):
        idx = np.arange(b0, min(nfr, b0 + 512))
        fr = np.stack([y[i * hop:i * hop + n_fft] for i in idx]) * win
        S = np.abs(np.fft.rfft(fr, axis=1)).T.astype(np.float32) ** 0.5
        nb = S.shape[0]
        sal = np.zeros((len(cents), S.shape[1]), np.float32)
        for h in range(1, H + 1):
            fb = h * f0 * n_fft / RETUNE_SR
            ok = fb < nb - 2
            i0 = np.floor(fb[ok]).astype(int)
            w_ = (fb[ok] - i0)[:, None].astype(np.float32)
            sal[ok] += (0.85 ** (h - 1)) * ((1 - w_) * S[i0] + w_ * S[i0 + 1])
        a, b, c = sal[:-2], sal[1:-1], sal[2:]
        mx = sal.max(0, keepdims=True)
        pk = (b > a) & (b >= c) & (b > rel * mx) & (mx > 0)
        j, t = np.nonzero(pk)
        A, B, Cc = a[j, t], b[j, t], c[j, t]
        d = 0.5 * (A - Cc) / (A - 2 * B + Cc - 1e-12)
        cs.append((cents[j + 1] + np.clip(d, -1, 1) * step_c + 50) % 100 - 50)
        ws.append(B)
    c = np.concatenate(cs) if cs else np.zeros(0)
    w = np.concatenate(ws) if ws else np.zeros(0)
    if w.sum() <= 0:
        return 0.0, 0.0
    z = (w * np.exp(2j * np.pi * c / 100)).sum() / w.sum()
    return float(np.angle(z) * 100 / (2 * np.pi)), float(np.abs(z))


def retune_ratio(y):
    """-> (offset in cents, strength, shift applied in cents, r). r = 1 when the recording is left as it is."""
    delta, strength = retune_tuning(y)
    shift = -delta if (abs(delta) >= MIN_CENTS and strength >= MIN_STRENGTH) else 0.0
    return delta, strength, shift, 2.0 ** (shift / 1200.0)


def retune_resample(y, r):
    """Play the audio r times faster (pitch x r): 16 kHz -> 16 kHz / r samples."""
    import soxr

    return soxr.resample(y, RETUNE_SR, RETUNE_SR / r, quality="HQ").astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# grouper/front
# ══════════════════════════════════════════════════════════════════════

# The grouper's own audio front end (no learned weights) and the per-note audio readers built on it.
#
# Front:  audio -> X: the harmonic CQT (B, ch, T, 288), 10 ms frames, 3 bins per semitone from A0, with PCEN (a time-local gain
#                    control per frequency channel) and a per-window standardisation, so recording level and EQ matter less;
#                 P: linear power on the extended log-frequency axis (B, NE, T), for harmonic demixing.
# HDemix: every note lays a comb of H harmonics over its sounding frames; the combs of a window add up to an occupancy map and a
#         note takes its share of the power in each comb bin (energy under chord mates is split, not counted twice). Per note:
#         the log share-power of each harmonic at attack offsets and sustain points, relative to the note's own peak.
# Prof:   mean / std of every channel at the note's bin (and the next bin up) over the note.
# Note times are relative to the window; a window's audio starts PAD0 seconds before it.



GFRONT_SR, GFRONT_HOP = 16000, 160
GFRONT_PAD0 = 0.5
HD_K = (-2, 0, 2, 4, 8, 12, 16, 24, 32, 48)                       # attack sample offsets (10 ms frames)


def pcen(X, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, K=200):
    """PCEN on log-power maps (B, C, T, F): every frequency channel divided by its own causal slow level."""
    B, Cc, T, Fq = X.shape
    E = torch.exp(X.float()).permute(0, 1, 3, 2)
    k = s * (1 - s) ** torch.arange(K, device=X.device, dtype=torch.float32)
    n = 1 << int(math.ceil(math.log2(T + K)))
    M = torch.fft.irfft(torch.fft.rfft(E, n) * torch.fft.rfft(k, n), n)[..., :T].clamp(min=0)
    out = (E / (eps + M) ** alpha + delta) ** r - delta ** r
    return torch.log(out + 1e-4).permute(0, 1, 3, 2)


def standardise(X):
    """Per window and channel: standardise over the active region (the top 12 log units)."""
    act = X > (X.amax((2, 3), keepdim=True) - 12.0)
    cnt = act.float().sum((2, 3), keepdim=True).clamp(min=1.0)
    mu = (X * act).sum((2, 3), keepdim=True) / cnt
    sd = (((X - mu) ** 2) * act).sum((2, 3), keepdim=True).div(cnt).sqrt().clamp(min=0.5)
    return (X - mu) / sd


class Front(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = HCQT()
        self.ch = self.h.n_channels
        self.lo, self.NE = self.h.lo, self.h.fb.shape[0]

    @torch.no_grad()
    def forward(self, audio):
        fr = self.h
        with torch.autocast("cuda", enabled=False):
            audio = audio.float()
            T = audio.shape[-1] // GFRONT_HOP
            P = torch.matmul(fr.fb, fr._pow(audio, fr.n_fft, fr.win)[:, :, :T])
            lf = torch.log(P + fr.log_eps)
            chans = []
            for s in fr.shifts:
                a = s - fr.lo
                x = lf[:, a:a + fr.n_bins]
                if x.shape[1] < fr.n_bins:
                    x = F.pad(x, (0, 0, 0, fr.n_bins - x.shape[1]), value=math.log(fr.log_eps))
                chans.append(x)
            for i, nf in enumerate(fr.extra):
                px = fr._pow(audio, nf, getattr(fr, f"win_x{i}"))[:, :, :T]
                chans.append(torch.log(torch.matmul(getattr(fr, f"fb_x{i}"), px) + fr.log_eps))
            X = torch.stack(chans, 1).transpose(2, 3)
            X = standardise(pcen(X))
        return X, P


def note_frames(notes, T, pad0=GFRONT_PAD0):
    f_on = ((notes[..., 0] + pad0) * 100).round().long().clamp(0, T - 1)
    f_off = torch.maximum(((notes[..., 1] + pad0) * 100).round().long().clamp(max=T), f_on + 1)
    lane = (notes[..., 2].round().long() - 21).clamp(0, 87)
    return f_on, f_off, lane


class HDemix(nn.Module):
    def __init__(self, d, lo, H=10, sus=4, hid=160):
        super().__init__()
        self.H, self.sus, self.lo = H, sus, lo
        hs = [int(round(36 * math.log2(h))) for h in range(1, H + 1)]
        self.register_buffer("shift", torch.tensor(hs, dtype=torch.long), persistent=False)
        self.prior = nn.Parameter(torch.log(torch.expm1(1.0 / torch.arange(1, H + 1, dtype=torch.float32))))
        self.nfeat = H * (len(HD_K) + sus) + H
        self.enc = nn.Sequential(nn.Linear(self.nfeat, hid), nn.GELU(), nn.Linear(hid, d))

    def raw(self, P, notes, mask):
        B, NE, T = P.shape
        N = notes.shape[1]
        H = self.H
        f_on, f_off, lane = note_frames(notes, T)
        bins = (3 * lane - self.lo)[..., None] + self.shift                             # (B, N, H) on the extended axis
        valid_b = (bins < NE).float()
        bins = bins.clamp(0, NE - 1)
        pi = F.softplus(self.prior)
        T1 = T + 1
        flat_on = (bins * T1 + f_on[..., None]).reshape(B, -1)
        flat_off = (bins * T1 + f_off[..., None]).reshape(B, -1)
        dur = (f_off - f_on).float()
        kf = torch.tensor(HD_K, device=P.device)
        fr_ = torch.tensor([0.25, 0.5, 0.75, 0.95], device=P.device) if self.sus == 4 else torch.linspace(0.25, 0.95, self.sus, device=P.device)
        tk = torch.cat([f_on[..., None] + kf, f_on[..., None] + (dur[..., None] * fr_).long()], -1).clamp(0, T - 1)
        Kt = tk.shape[-1]
        act = ((tk >= f_on[..., None]) & (tk < f_off[..., None])).float()
        idx = (bins[..., :, None] * T + tk[..., None, :]).reshape(B, -1)
        Pg = torch.gather(P.reshape(B, -1), 1, idx).view(B, N, H, Kt)
        mk = mask[..., None].float() * valid_b
        own = pi[None, None, :].expand(B, N, H)                                          # each note's claim on its comb bins
        w = own * mk
        O = torch.zeros(B, NE * T1, device=P.device).scatter_add(1, flat_on, w.reshape(B, -1))
        O = O.scatter_add(1, flat_off, -w.reshape(B, -1)).view(B, NE, T1).cumsum(2)[:, :, :T]       # occupancy
        Og = torch.gather(O.reshape(B, -1), 1, idx).view(B, N, H, Kt)
        o4 = own[..., None]
        share = o4 / (Og + o4 * (1 - act[:, :, None, :])).clamp(min=1e-6)
        lE = torch.log(Pg * share.clamp(max=1.0) + 1e-7) * valid_b[..., None]
        lE = lE - lE.amax((2, 3), keepdim=True)
        return torch.cat([lE.clamp(min=-20.0).reshape(B, N, -1) / 10.0, valid_b], -1)

    def forward(self, P, notes, mask):
        return self.enc(self.raw(P, notes, mask))


class Prof(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * ch, 64), nn.GELU(), nn.Linear(64, d))

    def forward(self, X, notes):
        B, ch, T, Fq = X.shape
        f_on, f_off, lane = note_frames(notes, T)
        fb = 3 * lane
        cs = torch.cumsum(F.pad(X, (0, 0, 1, 0)), 2).permute(0, 2, 3, 1)
        cs2 = torch.cumsum(F.pad(X * X, (0, 0, 1, 0)), 2).permute(0, 2, 3, 1)
        bi = torch.arange(B, device=X.device)[:, None]
        Ln = (f_off - f_on).clamp(min=1).float()[..., None]
        prof = []
        for off in (0, 1):
            fbb = (fb + off).clamp(0, Fq - 1)
            m1 = (cs[bi, f_off, fbb] - cs[bi, f_on, fbb]) / Ln
            m2 = (cs2[bi, f_off, fbb] - cs2[bi, f_on, fbb]) / Ln
            prof += [m1, ((m2 - m1 * m1).clamp(min=0) + 1e-4).sqrt()]
        return self.net(torch.cat(prof, -1))


def note_conf(notes):
    """The onset probability of the note's own voice."""
    return torch.where(notes[..., 3] > 0.5, notes[..., 5], notes[..., 4])


def rasters(notes, mask, Tq, P, fps, pad0=GFRONT_PAD0):
    """Note occupancy on a (Tq frames at fps) x (P lanes from A0) grid: onsets, sustain, onset confidence, second-voice onsets
    -> (B, 4, Tq, P), plus each note's onset / end frame and lane."""
    B, N = mask.shape
    on = ((notes[..., 0] + pad0) * fps).long().clamp(0, Tq - 1)
    off = torch.maximum(((notes[..., 1] + pad0) * fps).long().clamp(max=Tq), on + 1)
    lane = (notes[..., 2].round().long() - 21).clamp(0, P - 1)
    mk = mask.float()
    R = torch.zeros(B, 4, (Tq + 1) * P, device=notes.device)
    io, ie = on * P + lane, off * P + lane
    R[:, 0].scatter_add_(1, io, mk)
    R[:, 1].scatter_add_(1, io, mk)
    R[:, 1].scatter_add_(1, ie, -mk)
    R[:, 2].scatter_add_(1, io, mk * note_conf(notes))
    R[:, 3].scatter_add_(1, io, mk * (notes[..., 3] > 0.5).float())
    R = R.view(B, 4, Tq + 1, P)
    return torch.stack([R[:, 0], R[:, 1].cumsum(1), R[:, 2], R[:, 3]], 1)[:, :, :Tq], on, off, lane


# ══════════════════════════════════════════════════════════════════════
# grouper/model
# ══════════════════════════════════════════════════════════════════════

# The grouper network: one token per transcribed note, a transformer over the notes of a 24 s window.
#
# Each token adds up
#   - the note itself: pitch embedding + 11 scalars (log duration, the transcriber's onset / frame / offset probabilities of both
#     voices around the note, voice slot, register);
#   - three audio readers on the grouper's own front end: harmonic demixing (HDemix), the channel profile at the note's bin (Prof)
#     and a lane trunk over the spectrogram plus the window's note rasters, read at each note (LaneFront);
#   - the transcriber's own internal state at the note (three 368-d taps: its last hidden state at the onset and over the note,
#     and its stem output over the first 200 ms), standardised over the window's notes.
# Seven transformer blocks (rotary positions on onset time) follow. Heads per note: instrument logits (73 instruments), a 32-d
# track embedding (unit vector; notes of one track point the same way), a keep score, and the note state that a small track head
# turns into the track's instrument once the decode has formed the tracks.



def rope_time(x, pos, base=10000.0):
    """x (B, H, N, D), pos (B, N) continuous positions -> rotary embedding along N."""
    D = x.shape[-1]
    half = D // 2
    freqs = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float()[:, None, :, None] * freqs[None, None, None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half:]], -1)


class GroupBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, pos, mask):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope_time(q, pos), rope_time(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.ln2(x))


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w, self.b, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) * torch.rsqrt(s + self.eps) * self.w[:, None, None] + self.b[:, None, None]


class GroupPitchMix(nn.Module):
    """Per-channel taps along the lanes at local and harmonic semitone offsets."""

    def __init__(self, c, n_lanes, offsets=(-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31)):
        super().__init__()
        self.taps = nn.Parameter(torch.zeros(c, len(offsets)))
        S = torch.zeros(len(offsets), n_lanes, n_lanes)
        i = torch.arange(n_lanes)
        for k, o in enumerate(offsets):
            j = i + o
            ok = (j >= 0) & (j < n_lanes)
            S[k, i[ok], j[ok]] = 1.0
        self.register_buffer("S", S, persistent=False)

    def forward(self, x):
        W = torch.einsum("ck,kij->cij", self.taps.to(x.dtype), self.S.to(x.dtype))
        return torch.einsum("cij,bctj->bcti", W, x)


class LBlock(nn.Module):
    """Dilated depthwise time conv, harmonic lane mixing, LayerNorm, pointwise MLP, scaled residual."""

    def __init__(self, c, dil, n_lanes, exp=2):
        super().__init__()
        self.dw_t = nn.Conv2d(c, c, (7, 1), padding=(3 * dil, 0), dilation=(dil, 1), groups=c)
        self.pm = GroupPitchMix(c, n_lanes)
        self.norm = LayerNorm2d(c)
        self.pw1, self.pw2 = nn.Conv2d(c, exp * c, 1), nn.Conv2d(exp * c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        y = self.dw_t(x)
        y = y + self.pm(y)
        return x + self.pw2(F.gelu(self.pw1(self.norm(y)))) * self.gamma[:, None, None]


class LanePitchAttn(nn.Module):
    """At every frame the lanes attend to each other, with a learned bias per pitch distance (transposition-equivariant)."""

    def __init__(self, c, n_lanes, heads=4):
        super().__init__()
        self.h = heads
        self.ln, self.qkv, self.out = nn.LayerNorm(c), nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.gamma = nn.Parameter(torch.zeros(c))
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * n_lanes - 1))
        idx = torch.arange(n_lanes)
        self.register_buffer("rel_idx", (idx[None, :] - idx[:, None]) + n_lanes - 1, persistent=False)

    def forward(self, x):
        B, C, T, P = x.shape
        s = self.ln(x.permute(0, 2, 3, 1).reshape(B * T, P, C))
        q, k, v = self.qkv(s).view(B * T, P, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.rel_bias[:, self.rel_idx].to(q.dtype)[None])
        o = self.out(a.transpose(1, 2).reshape(B * T, P, C)) * self.gamma
        return x + o.view(B, T, P, C).permute(0, 3, 1, 2)


class LaneFront(nn.Module):
    """The normalised HCQT -> n_lanes semitone lanes at 80 ms (stem conv) + the notes' occupancy rasters -> lane blocks
    (dilations 1 2 4 8 ...) with lane attention after blocks `attn_at` -> each note reads its lane at its onset frame and
    averaged over its frames -> (B, N, d)."""

    def __init__(self, ch, d, c=128, depth=6, n_lanes=96, ts=8, n_attn=2, exp=2):
        super().__init__()
        self.P, self.ts = n_lanes, ts
        self.stem = nn.Conv2d(ch, c, (ts, 3), stride=(ts, 3), padding=(0, 1))
        self.occ = nn.Conv2d(4, c, 1)
        dils = [(1, 2, 4, 8)[i % 4] for i in range(depth)]
        self.blocks = nn.ModuleList([LBlock(c, dl, n_lanes, exp) for dl in dils])
        self.attn_at = set(range(depth - 1, -1, -max(1, depth // max(n_attn, 1)))[:n_attn]) if n_attn else set()
        self.attn = nn.ModuleDict({str(i): LanePitchAttn(c, n_lanes) for i in sorted(self.attn_at)})
        self.read = nn.Linear(2 * c, d)

    def forward(self, X, notes, mask):
        B = X.shape[0]
        Y = self.stem(X.to(next(self.stem.parameters()).dtype))
        Tq = Y.shape[2]
        R, on, off, lane = rasters(notes, mask, Tq, self.P, 100.0 / self.ts)
        Y = Y + self.occ(R.to(Y.dtype))
        for i, blk in enumerate(self.blocks):
            Y = blk(Y)
            if i in self.attn_at:
                Y = self.attn[str(i)](Y)
        bi = torch.arange(B, device=X.device)[:, None]
        Yp = Y.permute(0, 2, 3, 1)
        at_on = Yp[bi, on, lane]
        cs = F.pad(Y.float(), (0, 0, 1, 0)).cumsum(2).permute(0, 2, 3, 1)
        mean = (cs[bi, off, lane] - cs[bi, on, lane]) / (off - on).clamp(min=1).float()[..., None]
        return self.read(torch.cat([at_on.float(), mean], -1))


def note_scalars(notes):
    """(B, N, 12) per-note scalars -> (B, N, 11) token features."""
    dur = (notes[..., 1] - notes[..., 0]).clamp(min=0.01)
    return torch.cat([torch.log(dur)[..., None], notes[..., 3:12], (notes[..., 2:3] / 64.0 - 1.0)], -1)


class Heads(nn.Module):
    def __init__(self, d, emb=32):
        super().__init__()
        self.inst = nn.Linear(d, K)
        self.emb = nn.Linear(d, emb)
        self.keep = nn.Linear(d, 1)
        # track head: (mean note state of a track + 6 register statistics) -> instrument logits; applied by the decode
        self.th = nn.Sequential(nn.Linear(d + 6, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, K))

    def forward(self, x):
        return {"inst": self.inst(x), "emb": F.normalize(self.emb(x).float(), dim=-1), "keep": self.keep(x)[..., 0], "h": x}


class NoteNet(nn.Module):
    def __init__(self, d=176, depth=7, heads=4, ff=352, emb=32, hd=10, hd_sus=4, lanes=128, lane_depth=6, lane_ts=8, lane_attn=2,
                 lane_exp=2, n_lanes=96, s15_taps=(0, 1, 2), s15_hid=256, s15_dim=368, pos_scale=10.0):
        super().__init__()
        self.pos_scale, self.s15_taps = pos_scale, tuple(s15_taps)
        self.front = Front()
        ch = self.front.ch
        self.hd = HDemix(d, self.front.lo, H=hd, sus=hd_sus)
        self.prof = Prof(ch, d)
        self.lanes = LaneFront(ch, d, c=lanes, depth=lane_depth, n_lanes=n_lanes, ts=lane_ts, n_attn=lane_attn, exp=lane_exp)
        self.s15_ln = nn.LayerNorm(s15_dim)
        self.s15_net = nn.Sequential(nn.Linear(s15_dim * len(self.s15_taps), s15_hid), nn.GELU(), nn.Linear(s15_hid, d))
        self.pitch_emb = nn.Embedding(128, d)
        self.feat = nn.Sequential(nn.Linear(11, 64), nn.GELU(), nn.Linear(64, d))
        self.ln_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([GroupBlock(d, heads, ff) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(d)
        self.heads = Heads(d, emb)

    def tokens(self, batch):
        notes, mask = batch["notes"], batch["mask"]
        x = self.pitch_emb(notes[..., 2].round().long().clamp(0, 127)) + self.feat(note_scalars(notes))
        X, P = self.front(batch["audio"])
        xa = 0
        xa = xa + self.hd(P, notes, mask)
        xa = xa + self.prof(X, notes)
        xa = xa + self.lanes(X, notes, mask).to(x.dtype)
        x = x + xa
        f = batch["feat"][:, :, list(self.s15_taps)].float()
        # standardise every tap channel over the window's notes (what is constant over a recording carries no grouping cue)
        mk = mask[:, :, None, None].float()
        n = mk.sum(1, keepdim=True).clamp(min=1.0)
        mu = (f * mk).sum(1, keepdim=True) / n
        sd = (((f - mu) ** 2) * mk).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-3)
        f = self.s15_net(self.s15_ln((f - mu) / sd).flatten(2))
        return x + f

    def forward(self, batch):
        """batch: audio (B, L), notes (B, N, 12), mask (B, N), feat (B, N, 3, 368) -> per-note inst / emb / keep / h."""
        notes, mask = batch["notes"], batch["mask"]
        x = self.ln_in(self.tokens(batch))
        pos = notes[..., 0] * self.pos_scale
        for blk in self.blocks:
            x = blk(x, pos, mask)
        return self.heads(self.ln_out(x))


# ══════════════════════════════════════════════════════════════════════
# grouper/decode
# ══════════════════════════════════════════════════════════════════════

# From per-note predictions to tracks ("cluster, then name").
#
# 1. Tracklets: in every window, average-linkage clusters of the notes on (1 - track-embedding cosine) + lam (1 - instrument
#    distribution cosine), cut at tau; the clusters are chained across overlapping windows by their shared notes (Hungarian
#    matching on the overlap); every note keeps the cluster of the window it sits most centrally in.
# 2. The tracklets of the whole song are merged by size-weighted average linkage on their mean embeddings and instrument
#    distributions until the closest pair is farther than tau2.
# 3. `refine` k-means passes (every note to the nearest centre of a track with >= min_notes notes); tracks still under
#    min_notes are folded into the big track with the most similar instrument distribution.
# 4. Distinct tracks: a track with less than `absorb_frac` of the song's notes survives only if it is clearly its own instrument;
#    otherwise its notes are handed, one by one, to the nearest larger track.
# 5. Naming: alpha log p(track head) + (1 - alpha) log (summed note probabilities); the name maximises the expected credit
#    (1 = right instrument, 0.5 = same family). With `merge`, tracks that get the same name are one track.



def _cluster(e, p, tau, lam):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    m = len(e)
    if m == 1:
        return np.zeros(1, np.int64)
    e = e.astype(np.float32)
    Dm = 1 - e @ e.T
    if lam > 0:
        pn = p.astype(np.float32)
        pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)
        Dm = Dm + lam * (1 - pn @ pn.T)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), "average"), tau, "distance") - 1


def group_chain(P, on, tau, lam):
    """Per-window clusters chained across windows by shared notes -> tracklet id per note (-1 = none)."""
    from scipy.optimize import linear_sum_assignment
    n = len(on)
    track = -np.ones(n, np.int64)
    best_w = np.full(n, -1.0)
    owner = -np.ones(n, np.int64)
    n_glob = 0
    W = P["W"]
    for t0, sel, e, p in sorted(P["wins"], key=lambda w: w[0]):
        if len(sel) == 0:
            continue
        lab = _cluster(e, p, tau, lam)
        nl = int(lab.max()) + 1
        glob_of = -np.ones(nl, np.int64)
        prev = owner[sel]
        if (prev >= 0).any():
            gids = np.unique(prev[prev >= 0])
            ov = np.zeros((nl, len(gids)))
            np.add.at(ov, (lab[prev >= 0], np.searchsorted(gids, prev[prev >= 0])), 1)
            r, c = linear_sum_assignment(-ov)
            for a_, b_ in zip(r, c):
                if ov[a_, b_] > 0:
                    glob_of[a_] = gids[b_]
        for li in range(nl):
            if glob_of[li] < 0:
                glob_of[li] = n_glob
                n_glob += 1
        g = glob_of[lab]
        owner[sel] = g
        w = np.clip(1.0 - np.abs((on[sel] - t0) / W - 0.5) * 2, 0.05, 1.0)
        better = w > best_w[sel]
        track[sel[better]] = g[better]
        best_w[sel[better]] = w[better]
    return track


def _agglo(spaces, size, tau2, cap=600):
    """Size-weighted average linkage of M groups. spaces = [(means (M, D), weight)]: the distance of two groups is
    sum_w w (1 - m_A . m_B), with m the size-weighted mean of their members' unit vectors (= the mean pairwise cosine distance
    over all member pairs). Merges until the closest pair is farther than tau2 -> a label per group."""
    spaces = [(m.astype(np.float64).copy(), w) for m, w in spaces if w > 0]
    size = size.astype(np.float64).copy()
    M = len(size)
    lab = np.arange(M)
    if M < 2:
        return lab
    Dm = sum(w * (1 - m @ m.T) for m, w in spaces)
    np.fill_diagonal(Dm, np.inf)
    alive = np.ones(M, bool)
    if M > cap:                                                   # very long pieces: only the largest groups merge freely
        small = np.argsort(-size)[cap:]
        Dm[small] = np.inf
        Dm[:, small] = np.inf
    while True:
        k = int(np.argmin(Dm))
        i, j = divmod(k, M)
        if not np.isfinite(Dm[i, j]) or Dm[i, j] > tau2:
            break
        si, sj = size[i], size[j]
        for m, _ in spaces:
            m[i] = (si * m[i] + sj * m[j]) / (si + sj)
        size[i] = si + sj
        alive[j] = False
        lab[lab == j] = i
        Dm[j] = np.inf
        Dm[:, j] = np.inf
        d = sum(w * (1 - m @ m[i]) for m, w in spaces)
        d[~alive] = np.inf
        d[i] = np.inf
        Dm[i] = d
        Dm[:, i] = d
    return lab


def group_tracklet(P, on, tau1, tau2, lam):
    """Tracklets (step 1), then the whole song's tracklets merged (step 2) -> track id per note."""
    tr = group_chain(P, on, tau1, lam)
    idx = np.nonzero(tr >= 0)[0]
    if len(idx) == 0:
        return tr
    tids, inv = np.unique(tr[idx], return_inverse=True)
    M = len(tids)
    if M == 1:
        return tr
    E_, Pm = P["emb"][idx].astype(np.float64), P["probs"][idx].astype(np.float64)
    Pm = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-9)
    size = np.bincount(inv, minlength=M).astype(np.float64)
    me = np.zeros((M, E_.shape[1]))
    np.add.at(me, inv, E_)
    me /= size[:, None]
    mp = np.zeros((M, Pm.shape[1]))
    np.add.at(mp, inv, Pm)
    mp /= size[:, None]
    lab = _agglo([(me, 1.0), (mp, lam)], size, tau2)
    out = -np.ones(len(tr), np.int64)
    out[idx] = lab[inv]
    return out


def track_stats(on, off, pitch):
    p = pitch / 64.0 - 1.0
    n = len(p)
    if n <= 3000:
        ov = np.minimum(off[:, None], off[None, :]) > np.maximum(on[:, None], on[None, :]) + 0.03
        poly = float(np.mean(ov.sum(1) > 1.5))
    else:
        poly = 0.0
    return np.array([p.mean(), p.std(), p.min(), p.max(), np.log1p(n) / 5, poly], np.float64)


def absorb_small(inv, emb, on, off, frac, tau_a):
    """Stray tracks: a track holding less than `frac` of the song's notes survives only if it is clearly its own instrument.
    With s = its share of the notes and d = the smallest distance (1 - mean pairwise cosine) to a LARGER track, it is dissolved
    when d < tau_a (1 - s / frac): the smaller the track, the more distinct it has to be. Its notes then go, one by one, to
    the larger track whose centre each is nearest to. inv = track id per note -> new inv."""
    inv = inv.copy()
    n = len(inv)
    if n == 0 or frac <= 0 or tau_a <= 0:
        return inv
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    while True:
        tids, lab = np.unique(inv, return_inverse=True)
        M = len(tids)
        if M < 2:
            break
        size = np.bincount(lab, minlength=M).astype(np.float64)
        cen = np.zeros((M, e.shape[1]))
        np.add.at(cen, lab, e)
        cen /= size[:, None]
        done = True
        for t in np.argsort(size):
            s = size[t] / n
            if s >= frac:
                break
            big = np.nonzero(size > size[t])[0]
            if len(big) == 0:
                continue
            d = 1.0 - cen[big] @ cen[t]
            thr = np.full(len(big), tau_a * (1.0 - s / frac))
            if (d - thr).min() < 0:
                mine = np.nonzero(lab == t)[0]
                cu = cen[big] / np.maximum(np.linalg.norm(cen[big], axis=1, keepdims=True), 1e-9)
                inv[mine] = tids[big[np.argmax(e[mine] @ cu.T, 1)]]
                done = False
                break                                             # sizes / centres changed: start over
        if done:
            break
    return inv


def track_logits(ids, P, notes, th, alpha):
    """Log instrument distribution of one track: alpha log p(track head) + (1 - alpha) log (summed note probabilities)."""
    from scipy.special import erf
    pn = P["probs"][ids].astype(np.float64).sum(0)
    ln = np.log(np.maximum(pn / pn.sum(), 1e-9))
    if th is None or alpha <= 0 or P["h"] is None:
        return ln
    on, off, pitch = notes
    x = np.concatenate([P["h"][ids].astype(np.float64).mean(0), track_stats(on[ids], np.maximum(off[ids], on[ids] + 0.02), pitch[ids])])
    for li, (Wm, b) in enumerate(th):
        x = Wm @ x + b
        if li < len(th) - 1:
            x = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    lt = x - x.max()
    lt = lt - np.log(np.exp(lt).sum())
    return alpha * lt + (1 - alpha) * ln


def grouper_decode(P, notes, th, tau=0.1, tau2=0.4, lam=0.5, min_notes=30, refine=2, alpha=0.5, ec=True, merge=True, absorb=0.3,
           absorb_frac=0.25):
    """P = the song's predictions (Grouper.predict), notes = (onset, offset, pitch) arrays, th = the track head's layers
    [(W, b)] -> [(instrument index, sorted note ids)]."""
    on, off, pitch = notes
    if len(on) == 0:
        return []
    track = group_tracklet(P, on, tau, tau2, lam)
    idx = np.nonzero(track >= 0)[0]
    if len(idx) == 0:
        return []
    tids, inv = np.unique(track[idx], return_inverse=True)
    T = len(tids)
    probs = P["probs"]
    size = np.bincount(inv, minlength=T)
    mn = min_notes
    emb = P["emb"][idx].astype(np.float64)
    for _ in range(refine):                                       # every note to the nearest big-track centre
        keepc = np.nonzero(size >= mn)[0]
        if len(keepc) < 2:
            break
        cen = np.zeros((T, emb.shape[1]))
        np.add.at(cen, inv, emb)
        cen = cen[keepc] / np.maximum(np.linalg.norm(cen[keepc], axis=1, keepdims=True), 1e-9)
        sc = emb @ cen.T
        if lam > 0:
            pm = np.zeros((T, probs.shape[1]))
            np.add.at(pm, inv, probs[idx].astype(np.float64))
            pm = pm[keepc] / np.maximum(np.linalg.norm(pm[keepc], axis=1, keepdims=True), 1e-9)
            pn = probs[idx].astype(np.float64)
            sc = sc + lam * (pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)) @ pm.T
        inv = keepc[np.argmax(sc, 1)]
        size = np.bincount(inv, minlength=T)
    mass = np.zeros((T, probs.shape[1]))
    np.add.at(mass, inv, probs[idx])
    big = np.nonzero(size >= mn)[0]
    if len(big) == 0:
        big = np.array([int(np.argmax(size))])
    isbig = np.zeros(T, bool)
    isbig[big] = True
    bign = mass[big] / np.maximum(np.linalg.norm(mass[big], axis=1, keepdims=True), 1e-9)
    src_v = mass / np.maximum(np.linalg.norm(mass, axis=1, keepdims=True), 1e-9)
    for t in range(T):                                            # fold small tracks into the best-matching big one
        if not isbig[t] and size[t] > 0:
            inv[inv == t] = big[int(np.argmax(bign @ src_v[t]))]
    if absorb > 0 and absorb_frac > 0:
        inv = absorb_small(inv, emb, on[idx], off[idx], absorb_frac, absorb)
    out = {}
    for t in np.unique(inv):
        ids = idx[inv == t]
        z = track_logits(ids, P, notes, th, alpha)
        if ec:
            q = np.exp(z - z.max())
            k = int(np.argmax(CREDIT @ (q / q.sum())))
        else:
            k = int(np.argmax(z))
        if merge and k in out:
            out[k] = np.concatenate([out[k], ids])
        else:
            out[k if merge else (k, int(t))] = ids
    return [((k if merge else k[0]), np.sort(ids)) for k, ids in out.items()]


# ══════════════════════════════════════════════════════════════════════
# grouper/core
# ══════════════════════════════════════════════════════════════════════

# Grouping a song's notes into instrument tracks.
#
#     notes (onset, offset, pitch) + the audio
#       -> tuning normalisation (the audio is resampled onto A440 when it sits clearly off it)
#       -> the transcriber's maps and internal states on that audio (taken from the one transcriber pass that also fed the evidence
#          repairer: the average over every other of its windows), gathered at every note (12 scalars + three 368-d taps)
#       -> the grouper network over 24 s windows (every 12 s): per-note instrument probabilities and track embeddings,
#          averaged over the windows that hold the note (weighted towards the window centre)
#       -> the decode (decode.py): tracks, each named by one of 73 instruments



GROUPER_SR = 16000
GROUPER_PITCH_LO, GROUPER_N_KEYS = 21, 88
GROUPER_PAD0, PAD1 = 0.5, 1.0             # a window's audio starts PAD0 before it and ends PAD1 after it
SPAN_CAP, STEM_SPAN = 50, 10      # frames averaged for tap 1 (the note, <= 1 s) and tap 2 (the attack, 200 ms)
SLOT_TOL = 0.1                    # s: an input note takes the voice slot of the nearest same-pitch transcriber note within this
DEDUP = 0.03                      # s: same-pitch notes this close inside one track are one note
FEAT_DIM = 368


def _grouper_autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


@contextmanager
def _deterministic(device):
    """On CUDA, use PyTorch's deterministic kernels while the grouper runs: its front end accumulates with scatter_add, whose
    default CUDA kernel sums in a varying order, and on some songs those last-bit differences change which track a note joins.
    Costs ~20% of the grouper's own (small) run time; CPU results are unaffected."""
    if not str(device).startswith("cuda"):
        yield
        return
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


# ----------------------------------------------------------------------------- note features from the transcriber
def assign_slots(notes, tnotes):
    """Input notes (N, 3) + the transcriber's own notes (M, 4) [onset, offset, pitch, slot 0 / 1] -> (N, 4) sorted by
    (onset, pitch): end at least 20 ms after the onset, voice slot of the nearest same-pitch transcriber note (else 0)."""
    ov = np.asarray(notes, np.float64).reshape(-1, 3)
    slot = np.zeros(len(ov))
    for p in np.unique(ov[:, 2]):
        a = np.where(ov[:, 2] == p)[0]
        b = np.where(tnotes[:, 2] == p)[0]
        if len(b):
            d = np.abs(ov[a, 0][:, None] - tnotes[b, 0][None, :])
            j = d.argmin(1)
            slot[a] = np.where(d[np.arange(len(a)), j] <= SLOT_TOL, tnotes[b[j], 3], 0.0)
    o = np.lexsort((ov[:, 2], ov[:, 0]))
    return np.stack([ov[o, 0], np.maximum(ov[o, 1], ov[o, 0] + 0.02), ov[o, 2], slot[o]], 1)


@torch.no_grad()
def gather(P, H, S, notes, hop):
    """Maps P {head: (T, 88)}, hidden state H and stem output S (T, 88, C; on the device or in host memory) at the notes (N, 4) ->
    meta (N, 12) float32: onset, offset, pitch, slot, onset prob of each voice (max over +-1 frame), frame prob of each voice
         at the onset and averaged over the note, offset prob of each voice (max over the last frame +-1);
    feat (N, 3, C) int8 + fsc (N, 3) float16 per-row scales: H at the onset, H over the note (<= 1 s), S over the first 200 ms."""
    dev = P["onset"].device
    T, L = P["onset"].shape
    N = len(notes)
    Cc = H.shape[2]
    if N == 0:
        return np.zeros((0, 12), np.float32), np.zeros((0, 3, Cc), np.int8), np.zeros((0, 3), np.float16)
    nt = torch.from_numpy(notes).to(dev)
    lane = (nt[:, 2].round().long() - GROUPER_PITCH_LO).clamp(0, L - 1)
    t0 = (nt[:, 0] / hop).round().long().clamp(0, T - 1)
    te = torch.maximum((nt[:, 1] / hop).ceil().long(), t0 + 1).clamp(max=T)

    def at(m, t):
        return m[t.clamp(0, T - 1), lane]

    def mx3(m, t):
        return torch.stack([at(m, t - 1), at(m, t), at(m, t + 1)]).amax(0)

    def span_mean(m):
        cs = torch.cat([torch.zeros(1, L, device=dev), m.cumsum(0)])
        return (cs[te, lane] - cs[t0, lane]) / (te - t0).clamp(min=1).float()

    meta = torch.stack([nt[:, 0].float(), nt[:, 1].float(), nt[:, 2].float(), nt[:, 3].float(),
                        mx3(P["onset"], t0), mx3(P["onset2"], t0), at(P["frame"], t0), at(P["frame2"], t0),
                        span_mean(P["frame"]), span_mean(P["frame2"]), mx3(P["offset"], te - 1), mx3(P["offset2"], te - 1)], 1)
    meta = meta.cpu().numpy().astype(np.float32)

    def rows(M, t):                 # M[t, lane] on the device, whichever device M lives on
        return M[t.to(M.device), lane.to(M.device)].to(dev)

    def span(M, cap):
        acc = torch.zeros(N, Cc, device=dev)
        cnt = torch.zeros(N, device=dev)
        lim = torch.minimum(te, t0 + cap)
        for k in range(cap):
            t = t0 + k
            m = (t < lim).float()
            acc += rows(M, t.clamp(max=T - 1)).float() * m[:, None]
            cnt += m
        return acc / cnt.clamp(min=1)[:, None]

    X = torch.stack([rows(H, t0).float(), span(H, SPAN_CAP), span(S, STEM_SPAN)], 1)
    sc = X.abs().amax(-1).clamp(min=1e-6) / 127.0
    feat = (X / sc[..., None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    return meta, feat, sc.cpu().numpy().astype(np.float16)


# ----------------------------------------------------------------------------- windows
def song_windows(on, W, hop, nmax):
    """Windows covering every note (onsets `on`, sorted): a fixed time hop; a window holding more than nmax notes is followed
    by one that starts at its first uncovered note. -> [(t0, note indices)]"""
    out, t0 = [], 0.0
    n = len(on)
    if n == 0:
        return out
    end = float(on[-1])
    while True:
        a, b = np.searchsorted(on, t0), np.searchsorted(on, t0 + W)
        sel = np.arange(a, min(b, a + nmax))
        if len(sel) >= 1:
            out.append((t0, sel))
        if b >= n and (a + nmax >= b):
            break
        if b - a > nmax:
            t0 = max(t0 + 0.25 * hop, float(on[a + nmax // 2]))
        else:
            t0 += hop
        if t0 > end:
            break
    return out


class _Song:
    """One song's model inputs: 16-bit audio, note scalars, taps."""

    def __init__(self, y, meta, feat, fsc):
        self.audio = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
        self.meta, self.feat, self.fsc = meta, feat, fsc

    def audio_slice(self, t_a, L):
        x = np.zeros(L, np.float32)
        s0 = int(round(t_a * GROUPER_SR))
        lo, hi = max(s0, 0), min(s0 + L, len(self.audio))
        if hi > lo:
            x[lo - s0:hi - s0] = np.asarray(self.audio[lo:hi], np.float32) / 32768.0
        return x

    def window(self, t0, W, sel):
        """One window's inputs, or None if it holds fewer than 4 notes."""
        if len(sel) < 4:
            return None
        notes = self.meta[sel].copy()
        notes[:, 0] -= t0
        notes[:, 1] = np.minimum(notes[:, 1] - t0, W + PAD1)
        feat = np.asarray(self.feat[sel], np.float32) * np.asarray(self.fsc[sel], np.float32)[..., None]
        return dict(audio=self.audio_slice(t0 - GROUPER_PAD0, int(round((GROUPER_PAD0 + W + PAD1) * GROUPER_SR))), notes=notes, feat=feat)


def collate(items, nmax, device):
    B = len(items)
    N = max(8, max(len(it["notes"]) for it in items))
    N = min(nmax, int(np.ceil(N / 64) * 64))
    L = len(items[0]["audio"])
    out = dict(audio=np.zeros((B, L), np.float32), notes=np.zeros((B, N, 12), np.float32), mask=np.zeros((B, N), bool),
               feat=np.zeros((B, N, 3, FEAT_DIM), np.float16))
    for b, it in enumerate(items):
        n = min(N, len(it["notes"]))
        out["audio"][b] = it["audio"]
        out["notes"][b, :n] = it["notes"][:n]
        out["notes"][b, n:, 2] = 60.0
        out["mask"][b, :n] = True
        out["feat"][b, :n] = it["feat"][:n]
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


# ----------------------------------------------------------------------------- the grouper
class Grouper:
    """Notes + audio -> instrument-named tracks.

    >>> g = Grouper.from_pretrained("cuda")
    >>> tracks, info = g.group(audio16k, notes, transcriber)   # notes (N, 3): onset s, offset s, MIDI pitch
    >>> tracks[0]["instrument"], tracks[0]["program"], tracks[0]["notes"]   # notes (M, 4): onset, offset, pitch, velocity
    """

    def __init__(self, model: NoteNet, cfg: dict, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.window_s = float(cfg["window_s"])
        self.window_hop = float(cfg["window_hop"])
        self.max_notes = int(cfg["max_notes"])
        self.batch = int(cfg.get("batch", 8))
        self.params = dict(cfg["decode"])
        # the track head, applied in float64 by the decode
        self.th = [(l.weight.detach().float().cpu().numpy().astype(np.float64), l.bias.detach().float().cpu().numpy().astype(np.float64))
                   for l in self.model.heads.th if isinstance(l, torch.nn.Linear)]

    @classmethod
    def from_pretrained(cls, device="cpu"):
        cfg = hub_load_config("grouper")
        assert cfg["instruments"] == INSTRUMENTS, "instrument list of the weights does not match the package"
        model = NoteNet(**cfg["model"])
        model.load_state_dict(hub_load_state("grouper"))
        return cls(model, cfg, device)

    # ------------------------------------------------------------------ steps
    def features(self, y, notes, transcriber):
        """The transcriber's maps / taps on audio y (its own pass, a window every 4 s), gathered at the notes (N, 3) (already on
        y's time axis). ``group`` does not call this: it takes the same averages from the shared pass (``features_from_run``)."""
        maps, taps = transcriber.run(y, hop_frac=0.5, batch=4, taps=("hidden", "stem"), pad_batch=True)
        return self.features_from_run(maps, taps, notes, transcriber)

    def features_from_run(self, maps, taps, notes, transcriber):
        """The same from a transcriber pass already made: ``maps`` and ``taps`` {"hidden", "stem"} averaged over the windows that
        start every 4 s (the sub-run of ``Transcriber.run(y, hop_frac=0.25, sub_hop_frac=0.5, sub_taps=("hidden", "stem"))``)."""
        tn = transcriber.notes(maps)
        own = np.stack([tn["onset"], tn["offset"], tn["pitch"].astype(np.float64), tn["slot"] - 1.0], 1).reshape(-1, 4)
        nt = assign_slots(notes, own)
        P = {k: torch.from_numpy(v).to(transcriber.device) for k, v in maps.items()}
        out = gather(P, taps["hidden"], taps["stem"], nt, transcriber.hop)
        del P, taps
        return out

    @torch.no_grad()
    def predict(self, song: _Song):
        """-> per-note predictions: probs (n, K), emb (n, E) unit vectors, h (n, d) float16, wins [(t0, sel, emb, probs)]."""
        W, nmax = self.window_s, self.max_notes
        on_all = song.meta[:, 0]
        n = len(on_all)
        acc, wsum, best = np.zeros((n, K), np.float64), np.zeros(n), np.full(n, -1.0)
        eacc = h = None
        wins = []
        jobs = song_windows(on_all, W, W * self.window_hop, nmax)
        for bi in range(0, len(jobs), self.batch):
            chunk = jobs[bi:bi + self.batch]
            items = [song.window(t0, W, sel) or song.window(t0, W, np.concatenate([sel] * 4)[:4]) for t0, sel in chunk]
            batch = collate(items, nmax, self.device)
            with _grouper_autocast(self.device), _deterministic(self.device):
                o = self.model(batch)
            p = torch.softmax(o["inst"].float(), -1).cpu().numpy()
            e = o["emb"].float().cpu().numpy()
            hh = o["h"].float().cpu().numpy()
            for k, (t0, sel) in enumerate(chunk):
                m = len(sel)
                on = on_all[sel]
                w = np.clip(1.0 - np.abs((on - t0) / W - 0.5) * 2, 0.05, 1.0)
                acc[sel] += p[k, :m] * w[:, None]
                wsum[sel] += w
                if eacc is None:
                    eacc = np.zeros((n, e.shape[-1]), np.float64)
                    h = np.zeros((n, hh.shape[-1]), np.float16)
                eacc[sel] += e[k, :m] * w[:, None]
                better = w > best[sel]
                h[sel[better]] = hh[k, :m][better]
                best[sel[better]] = w[better]
                wins.append((float(t0), sel, e[k, :m].astype(np.float16), p[k, :m].astype(np.float16)))
        ws = np.maximum(wsum, 1e-9)
        emb = eacc if eacc is not None else np.zeros((n, 1))
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        return dict(probs=(acc / ws[:, None]).astype(np.float32), emb=emb.astype(np.float32), h=h, wins=wins, W=W)

    # ------------------------------------------------------------------ all of it
    def group(self, audio16k: np.ndarray, notes: np.ndarray, transcriber, tuning=None, run=None):
        """audio16k: the recording (16 kHz mono float32); notes (N, 3): onset s, offset s, MIDI pitch (any order), in the
        recording's time axis; transcriber: a dettato.transcriber.Transcriber (its maps and internal states are features).

        Without ``run`` the tuning is estimated here and the transcriber runs on the retuned audio (a window every 4 s). To share
        one transcriber pass with the evidence repairer, estimate the tuning first (``retune.ratio``), run the transcriber on the
        retuned audio y, and pass ``tuning`` = that ``retune.ratio`` result and ``run`` = (maps, taps) of the sub-average of that
        pass (``features_from_run``). Then ``audio16k`` is y and ``notes`` are on y's time axis; the tracks come back on the
        original axis (times x r) either way.

        -> (tracks, info). tracks: [{"instrument", "program", "notes": (M, 4) [onset, offset, pitch, velocity 1-127]}], the
        largest first; notes sorted by onset."""
        x = np.asarray(audio16k, np.float32)
        nin = np.array(notes, np.float64).reshape(-1, 3)
        if run is None:
            delta, strength, shift, r = retune_ratio(x)
            y = retune_resample(x, r) if shift else x
            nin[:, :2] /= r                                        # onto the retuned time axis
            dur = len(x) / GROUPER_SR
            meta, feat, fsc = self.features(y, nin, transcriber)
        else:
            if tuning is None:
                raise ValueError("pass the retune.ratio result with a precomputed run")
            delta, strength, shift, r = tuning
            dur = len(x) * r / GROUPER_SR
            y = x                                                  # already the retuned audio, notes already on its axis
            meta, feat, fsc = self.features_from_run(*run, nin, transcriber)
        song = _Song(y, meta, feat, fsc)
        on, off, pitch = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64), meta[:, 2].round().astype(np.int64)
        conf = np.where(meta[:, 3] > 0.5, meta[:, 5], meta[:, 4])
        groups = grouper_decode(self.predict(song), (on, off, pitch.astype(np.float64)), self.th, **self.params) if len(on) else []
        tracks = self._tracks(on * r, off * r, pitch, conf, groups)
        info = dict(tuning_cents=round(float(delta), 2), tuning_strength=round(float(strength), 3), retune_cents=round(float(shift), 2),
                    retune_ratio=float(r), duration_s=round(dur, 2), notes_in=len(on),
                    notes_out=int(sum(len(t["notes"]) for t in tracks)))
        return tracks, info

    @staticmethod
    def _tracks(on, off, pitch, conf, groups):
        """Decoded groups -> track dicts (same-pitch notes within 30 ms in one track become one note)."""
        out = []
        for k, ids in sorted(groups, key=lambda t: -len(t[1])):
            ids = np.asarray(ids)
            o = ids[np.lexsort((on[ids], pitch[ids]))]
            keep, last = [], None
            for i in o:
                if last is not None and pitch[i] == pitch[last] and on[i] - on[last] <= DEDUP:
                    continue
                keep.append(i)
                last = i
            rows = [(float(on[i]), float(max(off[i], on[i] + 0.02)), int(pitch[i]), int(np.clip(20 + 107 * conf[i], 1, 127)))
                    for i in sorted(keep, key=lambda q: on[q])]
            out.append({"instrument": INSTRUMENTS[k], "program": int(PROGRAM[k]),
                        "notes": np.array(rows, np.float64).reshape(-1, 4), "merged_duplicates": len(o) - len(keep)})
        return out


# ══════════════════════════════════════════════════════════════════════
# drums/features
# ══════════════════════════════════════════════════════════════════════

# Input features of the drum model: 280 log-power bands per 10 ms frame, from 44.1 kHz mono audio.
#
#     lo (128)  mel bands 20 Hz - 7.8 kHz      (16 kHz signal, 512-point STFT)
#     hi (32)   mel bands 7.8 - 20 kHz         (44.1 kHz signal, 1411-point STFT)
#     cq (120)  log-frequency bands, 24 per octave from 36.7 Hz (D1) to 1.14 kHz (16 kHz signal, 6144-point STFT):
#               fine enough in the low register to tell timpani pitches a semitone apart
#
# All three come from one level normalisation (the recording brought to an RMS of -18 dBFS) and use log(power + 1e-5).
# The high band is why the model needs a source recorded at 40 kHz or more.


DRUMS_SR = 44100
FLOOR = 1e-5
N_LO, N_HI, DRUMFEAT_N_CQ = 128, 32, 120
N_BANDS = N_LO + N_HI + DRUMFEAT_N_CQ          # 280
CQ_BPO, CQ_FMIN, CQ_NFFT = 24, 36.708, 6144


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-normalised triangular mel filters, (n_mels, n_fft // 2 + 1) float32."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    fb *= (2.0 / (mel_pts[2:] - mel_pts[:-2]))[:, None].astype(np.float32)
    return fb


def drums_logfreq_filterbank(sr: int = 16000, n_fft: int = CQ_NFFT, fmin: float = CQ_FMIN, bpo: int = CQ_BPO, n: int = DRUMFEAT_N_CQ) -> np.ndarray:
    """Triangular log-frequency filters (each sums to 1), (n, n_fft // 2 + 1) float32. A band narrower than about two FFT bins
    reads the spectrum by linear interpolation at its centre instead."""
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    c = fmin * 2.0 ** (np.arange(-1, n + 1) / bpo)
    fb = np.zeros((n, len(freqs)), np.float32)
    for k in range(n):
        lo, ce, hi = c[k], c[k + 1], c[k + 2]
        w = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)))
        if w.sum() < 1.0:
            j = ce * n_fft / sr
            j0 = int(np.floor(j))
            a = j - j0
            w = np.zeros_like(w)
            w[j0] = 1 - a
            w[j0 + 1] = a
        fb[k] = w / w.sum()
    return fb


_FILTERS: dict[str, torch.Tensor] = {}


def _filters(name: str) -> torch.Tensor:
    if not _FILTERS:
        _FILTERS["lo"] = torch.from_numpy(mel_filterbank(sr=16000, n_fft=512, n_mels=N_LO, fmin=20.0, fmax=7800.0))
        _FILTERS["hi"] = torch.from_numpy(mel_filterbank(sr=DRUMS_SR, n_fft=1411, n_mels=N_HI, fmin=7800.0, fmax=20000.0))
        _FILTERS["cq"] = torch.from_numpy(drums_logfreq_filterbank())
    return _FILTERS[name]


def _logspec(x: np.ndarray, n_fft: int, hop: int, fb: torch.Tensor) -> np.ndarray:
    s = torch.stft(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), n_fft, hop_length=hop, win_length=n_fft,
                   window=torch.hann_window(n_fft), center=True, return_complex=True, pad_mode="constant")
    return torch.log(fb @ (s.abs() ** 2) + FLOOR).T.numpy()


def level_gain(x: np.ndarray) -> float:
    """The gain that brings x to an RMS of -18 dBFS."""
    rms = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-9
    return 10 ** (-18 / 20) / rms


def features(x44: np.ndarray) -> np.ndarray:
    """Mono audio at 44.1 kHz -> (T, 280) float16 features [lo | hi | cq] at 100 frames per second."""
    import soxr

    x44 = np.asarray(x44, np.float32)
    x = (x44 * level_gain(x44)).astype(np.float32)
    x16 = soxr.resample(x, DRUMS_SR, 16000).astype(np.float32)
    lo = _logspec(x16, 512, 160, _filters("lo"))
    hi = _logspec(x, 1411, DRUMS_SR // 100, _filters("hi"))
    cq = _logspec(x16, CQ_NFFT, 160, _filters("cq"))
    T = min(len(x16) // 160, len(lo), len(hi), len(cq))
    return np.concatenate([lo[:T], hi[:T], cq[:T]], 1).astype(np.float16)


# ══════════════════════════════════════════════════════════════════════
# drums/model
# ══════════════════════════════════════════════════════════════════════

# The drum networks.
#
# ``DrumNet`` (188,764 parameters) reads the 160 mel bands of ``features.py`` (lo + hi) at 100 fps and writes, at 50 fps, one
# onset logit and one velocity logit per drum key of the GM range 27-87:
#
#     stem      2-D CNN over (time, band); time is halved once, the bands are pooled 8x
#     collapse  a bidirectional GRU across the bands of every frame -> one 88-d vector per frame
#     temporal  depthwise temporal conv block, residual bidirectional GRU, song-level content attention
#     heads     one small bidirectional GRU per drum family (kick / snare / hats + ride / toms + crashes / shakers), each predicting
#               only its own keys; every other key comes from a shared linear head; velocity from a low-rank linear head
#
# ``OrchestralExpert`` = a ``DrumNet`` + a pitch-resolving low band (the 120 ``cq`` bands) + an adapter (bidirectional GRU and
# self-attention) that corrects the drum logits for concert percussion and adds 13 timpani logits, one per key 41-53 (the key of a
# timpani hit is its pitch). Its router reads the adapter's states pooled over the whole recording and gives one number per
# recording: the probability that it is orchestral music, which decides whether the expert or the plain ``DrumNet`` transcribes it.


N_MEL = 160                              # lo + hi bands
DRUMNET_N_CQ = 120
KEYS = list(range(27, 88))               # drum keys (GM numbering) of the drum logits
TIMPANI_KEYS = list(range(41, 54))       # keys of the timpani logits (key = pitch)


def _act(name):
    return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[name]()


class Res2d(nn.Module):
    """Residual separable block: depthwise 3x3 -> BN -> act -> pointwise -> BN, added to the input."""

    def __init__(self, c, act="relu"):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, c, 1)
        self.act = _act(act)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return x + self.bn2(self.pw1(self.act(self.bn1(self.dw(x)))))


class DrumStem(nn.Module):
    """A comma-separated layer spec: ``cN`` 3x3 conv to N channels (+ BN, act), ``r`` residual block, ``pTF`` max-pool (T, F)."""

    def __init__(self, cin, spec, act="relu"):
        super().__init__()
        layers, c, self.tdiv, self.fdiv = [], cin, 1, 1
        for tok in spec.split(","):
            kind, arg = tok.strip()[0], tok.strip()[1:]
            if kind == "c":
                n = int(arg)
                layers += [nn.Conv2d(c, n, 3, padding=1), nn.BatchNorm2d(n), _act(act)]
                c = n
            elif kind == "r":
                layers.append(Res2d(c, act))
            elif kind == "p":
                t, f = int(arg[0]), int(arg[1:])
                layers.append(nn.MaxPool2d((t, f)))
                self.tdiv *= t
                self.fdiv *= f
            else:
                raise ValueError(f"unknown stem layer {tok!r}")
        self.net = nn.Sequential(*layers)
        self.cout = c

    def forward(self, x):
        return self.net(x)


class BandGRU(nn.Module):
    """(B, C, T, F) -> (B, T, W): a bidirectional GRU scans the F bands of every frame; its two final states and its mean output
    are projected to W."""

    def __init__(self, c, w, m):
        super().__init__()
        self.rnn = nn.GRU(c, m, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(4 * m, w)

    def forward(self, h):
        B, C, T, Fq = h.shape
        with torch.autocast(h.device.type, enabled=False):
            z = h.permute(0, 2, 3, 1).reshape(B * T, Fq, C).float()
            out, hn = self.rnn(z)
            z = torch.cat([hn[0], hn[1], out.mean(1)], -1)
        return self.proj(z).view(B, T, -1)


class ConvBlock(nn.Module):
    """Depthwise temporal conv (kernel k) -> BN -> SiLU -> pointwise, residual. On (B, T, W)."""

    def __init__(self, w, k=15, d=1):
        super().__init__()
        self.dw = nn.Conv1d(w, w, k, padding=d * (k // 2), dilation=d, groups=w)
        self.bn = nn.BatchNorm1d(w)
        self.pw = nn.Conv1d(w, w, 1)

    def forward(self, x):
        h = x.transpose(1, 2)
        return x + self.pw(F.silu(self.bn(self.dw(h)))).transpose(1, 2)


class GRUBlock(nn.Module):
    """LayerNorm -> bidirectional GRU (h per direction) -> linear back to W, residual."""

    def __init__(self, w, h=48):
        super().__init__()
        self.rnn = nn.GRU(w, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, w)
        self.norm = nn.LayerNorm(w)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            y, _ = self.rnn(self.norm(x.float()))
        return x + self.out(y)


class SongAttention(nn.Module):
    """Self-attention over the whole input, content only (no positions), residual."""

    def __init__(self, w, da=32, heads=2):
        super().__init__()
        self.n = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * da)
        self.o = nn.Linear(da, w)
        self.da, self.heads = da, heads

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n(x)).view(B, T, 3, self.heads, self.da // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, self.da)
        return x + self.o(a)


def _temporal_block(spec, w):
    t = spec["t"]
    if t == "tb":
        return ConvBlock(w, spec.get("k", 15), spec.get("d", 1))
    if t == "gru":
        return GRUBlock(w, spec.get("h", 48))
    if t == "attn":
        return SongAttention(w, spec.get("da", 32), spec.get("heads", 2))
    raise ValueError(f"unknown temporal block {t!r}")


class FamilyHeads(nn.Module):
    """Onset logits: one bidirectional GRU + linear head per key family (each predicts only its family's keys); the other keys
    come from a shared linear head."""

    def __init__(self, w, K, groups, h=24):
        super().__init__()
        self.groups = groups
        self.towers = nn.ModuleList([nn.GRU(w, h, batch_first=True, bidirectional=True) for _ in groups])
        self.heads = nn.ModuleList([nn.Linear(2 * h + w, len(g)) for g in groups])
        self.rest = nn.Linear(w, K)
        self.register_buffer("idx", torch.tensor([l for g in groups for l in g]))

    def forward(self, h):
        z = self.rest(h).float()
        with torch.autocast(h.device.type, enabled=False):
            hf = h.float()
            parts = [hd(torch.cat([tw(hf)[0], hf], -1)) for tw, hd in zip(self.towers, self.heads)]
        return z.index_copy(-1, self.idx, torch.cat(parts, -1))


class DrumNet(nn.Module):
    """(B, T, 160) mel features at 100 fps -> (onset logits, velocity logits), each (B, T/2, 61) at 50 fps."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        K = len(KEYS)
        self.stem = DrumStem(1, cfg["stem"], cfg.get("act", "relu"))
        assert self.stem.tdiv == 2, "the stem must halve time once"
        w = cfg["w"]
        assert cfg.get("collapse", "fgru") == "fgru"
        self.collapse = BandGRU(self.stem.cout, w, cfg["collapse_m"])
        blocks = []
        for spec in cfg["temporal"]:
            blocks += [_temporal_block(spec, w) for _ in range(spec.get("n", 1))]
        self.temporal = nn.ModuleList(blocks)
        groups = [[KEYS.index(k) for k in g] for g in cfg["towers"]]
        self.on = FamilyHeads(w, K, groups, cfg.get("tower_h", 24))
        self.vel = nn.Sequential(nn.Linear(w, 16), nn.Linear(16, K))

    def embed(self, x):
        """(B, T, 160) -> frame embeddings (B, T/2, W)."""
        return self.collapse(self.stem(((x + 5.0) / 3.0)[:, None]))

    def trunk(self, c):
        """frame embeddings -> (trunk states, onset logits, velocity logits)"""
        h = c
        for b in self.temporal:
            h = b(h)
        return h, self.on(h), self.vel(h)

    def embed_long(self, x, chunk=8000, margin=200):
        """``embed`` on a long input in windows of ``chunk`` frames with ``margin`` frames of context each side (more than the
        stem's receptive field): bounded memory, the same result up to float rounding."""
        T = x.shape[1]
        parts, s = [], 0
        while s < T:
            a, b = max(0, s - margin), min(T, s + chunk + margin)
            h = self.embed(x[:, a:b])
            lo = (s - a) // 2
            parts.append(h[:, lo:lo + (min(s + chunk, T) - s) // 2])
            s += chunk
        return torch.cat(parts, 1)

    def forward(self, x, long_above=9000):
        c = self.embed_long(x) if x.shape[1] > long_above else self.embed(x)
        _, z, v = self.trunk(c)
        return z, v


class CQBranch(nn.Module):
    """(B, T, 120) log-frequency bands at 100 fps -> (B, T/2, d). The input is stacked with itself shifted up a fifth (14 bands)
    and an octave (24 bands), so a timpani's principal tone and its strongest overtones meet in one column."""

    def __init__(self, d=128, c=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, 24, (5, 7), padding=(2, 3))
        self.c2 = nn.Conv2d(24, c, (3, 5), stride=(2, 1), padding=(1, 2))
        self.c3 = nn.Conv2d(c, c, (3, 5), padding=(1, 2))
        self.n1, self.n2, self.n3 = nn.GroupNorm(4, 24), nn.GroupNorm(4, c), nn.GroupNorm(4, c)
        self.proj = nn.Linear(c * DRUMNET_N_CQ, d)

    def forward(self, xc):
        xc = xc.float()
        xc = (xc - xc.mean(dim=(1, 2), keepdim=True)) / 4.0          # level-free
        lo = xc.amin(dim=(1, 2), keepdim=True)

        def shift(k):
            return torch.cat([xc[..., k:], lo.expand(-1, xc.shape[1], k)], -1)

        z = torch.stack([xc, shift(14), shift(24)], 1)                # (B, 3, T, F)
        z = F.silu(self.n1(self.c1(z)))
        z = F.silu(self.n2(self.c2(z)))
        z = F.silu(self.n3(self.c3(z))) + z
        B, C, T, Fq = z.shape
        return F.silu(self.proj(z.permute(0, 2, 1, 3).reshape(B, T, C * Fq)))


class OrchestralExpert(nn.Module):
    """A ``DrumNet`` plus the cq branch and an adapter. (B, T, 280) -> per frame at 50 fps: the DrumNet's own onset logits, the
    adapter's correction to them, timpani logits (13), velocity logits, and one router logit per input."""

    def __init__(self, base_cfg, d=192, h=128, layers=2, heads=4, cq_d=128):
        super().__init__()
        self.base = DrumNet(base_cfg)
        K = len(KEYS)
        self.cq = CQBranch(cq_d)
        self.inp = nn.Linear(base_cfg["w"] + cq_d + K, d)
        self.rnn = nn.GRU(d, h, num_layers=layers, batch_first=True, bidirectional=True)
        self.heads = heads
        self.qkv = nn.Linear(2 * h, 3 * 2 * h)
        self.ao = nn.Linear(2 * h, 2 * h)
        self.ln = nn.LayerNorm(2 * h)
        self.delta = nn.Linear(2 * h, K)
        self.timp = nn.Linear(2 * h, len(TIMPANI_KEYS))
        self.gate = nn.Sequential(nn.Linear(2 * h, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, chunk=8000, margin=200):
        T = x.shape[1] // 2 * 2
        x = x[:, :T]
        xm = x[..., :N_MEL]
        c = self.base.embed_long(xm, chunk, margin) if T > chunk + 2 * margin else self.base.embed(xm)
        h0, z0, v0 = self.base.trunk(c)
        cq = self.cq(x[..., N_MEL:])
        n = min(h0.shape[1], cq.shape[1])
        h0, z0, v0, cq = h0[:, :n], z0[:, :n], v0[:, :n], cq[:, :n]
        a = F.silu(self.inp(torch.cat([h0.float(), cq, torch.sigmoid(z0.float())], -1)))
        r, _ = self.rnn(a)
        B, Tn, D = r.shape
        q, k, v = self.qkv(r).view(B, Tn, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Tn, D)
        r = self.ln(r + self.ao(att))
        router = self.gate(r.float().mean(1))[:, 0]                   # one decision per input (a whole recording)
        return {"base": z0.float(), "drum": z0.float() + self.delta(r).float(), "timpani": self.timp(r).float(),
                "velocity": v0.float(), "router": router}


# ══════════════════════════════════════════════════════════════════════
# drums/infer
# ══════════════════════════════════════════════════════════════════════

# Drum transcription: 44.1 kHz audio -> drum hits on GM drum keys (timpani on keys 41-53, the key being the drum's pitch).
#
#     from dettato.drums import DrumTranscriber
#     drums = DrumTranscriber.from_pretrained("cuda")      # or "cpu"
#     hits = drums.transcribe(audio_44k)                   # [{"onset", "offset", "pitch", "is_drum", "velocity"}, ...]
#     drums.last_info                                      # {"router": 0.03, "model": "drum-kit"}
#
# Two models and a router:
#
#     drum-kit     ``DrumNet``: pop, rock, jazz, electronic music with a drum kit
#     orchestral   three ``OrchestralExpert`` models, averaged: timpani by pitch and concert percussion
#
# The router (the experts' mean, one value per recording, computed from the whole recording) picks the orchestral model when it
# is at least 0.5, else the drum-kit model. Hits are the local peaks of each key's onset probability at or above the threshold
# at 50 frames per second, at least 3 frames apart per key; the orchestral model adds its timpani hits, leaving out one that
# repeats a key already hit within 40 ms. Every hit is written 100 ms long.


MIN_SOURCE_RATE = 40000      # the drum model reads the 7.8-20 kHz band: sources recorded below this rate are not transcribed
FPS = 50                     # output frame rate


def peaks(prob: np.ndarray, th: float, gap: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(T, L) onset probabilities -> (frames, lanes) of the local maxima >= th, at least ``gap`` frames apart per lane."""
    T, L = prob.shape
    left = np.vstack([prob[:1], prob[:-1]])
    right = np.vstack([prob[1:], prob[-1:]])
    cand = (prob >= th) & (prob >= left) & (prob >= right)
    last = np.full(L, -10, np.int64)
    fr, ln = [], []
    for t in np.nonzero(cand.any(1))[0]:
        m = cand[t] & (t - last > gap - 1)
        for lane in np.nonzero(m)[0]:
            fr.append(int(t))
            ln.append(int(lane))
        last = np.where(m, t, last)
    return np.array(fr, int), np.array(ln, int)


def _split(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


class DrumTranscriber:
    """The drum stage. ``last_info`` holds the router value and the model that ran for the last ``transcribe`` call."""

    def __init__(self, config: dict, state: dict, device: str | torch.device = "cpu"):
        assert config.get("model_type") == "dettato-drums", config.get("model_type")
        assert config["keys"] == KEYS and config["timpani_keys"] == TIMPANI_KEYS and config["sample_rate"] == DRUMS_SR
        self.config = config
        self.device = torch.device(device)
        th = config["thresholds"]
        self.th_kit, self.th_drum, self.th_timpani, self.th_router = th["drum_kit"], th["orchestral_drum"], th["timpani"], th["router"]
        self.kit = DrumNet(config["drum_kit"])
        self.kit.load_state_dict(_split(state, "drum_kit."))
        self.kit.to(self.device).eval()
        self.experts = []
        for i in range(config["orchestral"]["n_models"]):
            ex = OrchestralExpert(config["drum_kit"], **config["orchestral"]["adapter"])
            ex.load_state_dict(_split(state, f"orchestral.{i}."))
            self.experts.append(ex.to(self.device).eval())
        self.last_info: dict = {}

    @classmethod
    def from_pretrained(cls, device: str | torch.device = "cpu") -> "DrumTranscriber":
        """The released weights (``kobimusic/dettato-drums``)."""
        return cls(hub_load_config("drums"), hub_load_state("drums"), device)

    def n_params(self) -> int:
        return sum(p.numel() for m in [self.kit, *self.experts] for p in m.parameters())

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    @torch.no_grad()
    def probabilities(self, feats: np.ndarray) -> dict:
        """(T, 280) features -> router value, the drum-kit model's onset / velocity probabilities and the orchestral models'
        (averaged) drum / timpani / velocity probabilities, all at 50 fps."""
        T = feats.shape[0] // 2 * 2
        x = torch.from_numpy(np.ascontiguousarray(feats[:T], dtype=np.float32))[None].to(self.device)
        with self._autocast():
            zk, vk = self.kit(x[..., :160])
        outs = []
        for ex in self.experts:
            with self._autocast():
                o = ex(x)
            base = o["base"][0].float()
            # the corrected drum logits are formed as the base logits plus the correction, as during training
            drum = torch.sigmoid(base + (o["drum"][0].float() - base))
            outs.append((float(torch.sigmoid(o["router"][0].float())), drum.cpu().numpy(),
                         torch.sigmoid(o["timpani"][0].float()).cpu().numpy(), torch.sigmoid(o["velocity"][0].float()).cpu().numpy()))
        n = min(o[1].shape[0] for o in outs)
        return {"router": float(np.mean([o[0] for o in outs])),
                "kit": torch.sigmoid(zk[0].float()).cpu().numpy(), "kit_vel": torch.sigmoid(vk[0].float()).cpu().numpy(),
                "drum": np.mean([o[1][:n] for o in outs], 0), "timpani": np.mean([o[2][:n] for o in outs], 0),
                "vel": np.mean([o[3][:n] for o in outs], 0)}

    def decode(self, P: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """-> (onset seconds, key, velocity 0-1, which model: "drum-kit" / "orchestral")"""
        if P["router"] < self.th_router:
            fr, ln = peaks(P["kit"], self.th_kit)
            return fr / FPS, np.array([KEYS[lane] for lane in ln], np.int64), P["kit_vel"][fr, ln], "drum-kit"
        fd, ld = peaks(P["drum"], self.th_drum)
        on, ks, vs = list(fd / FPS), [KEYS[lane] for lane in ld], list(P["vel"][fd, ld])
        seen: dict[int, list] = {}
        for o, k in zip(on, ks):
            seen.setdefault(k, []).append(o)
        ft, lt = peaks(P["timpani"], self.th_timpani)
        for f, lane in zip(ft, lt):
            o, k = f / FPS, TIMPANI_KEYS[int(lane)]
            if any(abs(o - q) <= 0.04 for q in seen.get(k, [])):
                continue
            on.append(o)
            ks.append(k)
            vs.append(float(np.clip(P["timpani"][f, lane], 0.3, 1.0)))
        order = np.argsort(on, kind="stable")
        return np.array(on)[order], np.array(ks, np.int64)[order], np.array(vs)[order], "orchestral"

    def transcribe(self, audio: np.ndarray, sr: int = DRUMS_SR) -> list[dict]:
        """Mono audio (from a source recorded at >= 40 kHz) -> drum hits sorted by onset. Audio at another rate than 44.1 kHz is
        resampled first."""
        x = np.asarray(audio, np.float32)
        if sr != DRUMS_SR:
            import soxr

            x = soxr.resample(x, sr, DRUMS_SR, quality="HQ").astype(np.float32)
        if len(x) < DRUMS_SR // 10:
            self.last_info = {"router": None, "model": None}
            return []
        P = self.probabilities(features(x).astype(np.float32))
        on, keys, vel, which = self.decode(P)
        self.last_info = {"router": P["router"], "model": which}
        return [{"onset": round(float(a), 4), "offset": round(float(a) + 0.1, 4), "pitch": int(k), "is_drum": True,
                 "velocity": round(float(np.clip(v, 0.05, 1.0)), 3)} for a, k, v in zip(on, keys, vel)]


# ══════════════════════════════════════════════════════════════════════
# pipeline
# ══════════════════════════════════════════════════════════════════════

# The whole system: audio file -> instrument-named MIDI tracks + a drum track.
#
#     tuning (estimated first; the audio is resampled onto A440 when it sits clearly off it)
#       -> transcriber (notes, two voices per key). ONE network pass over the audio feeds both stages below: every window's output
#          is averaged for the repairer, and every other window's for the grouper's features
#       -> evidence repairer (drops wrong notes, fixes onsets / ends, inserts missed notes, reading the transcriber's own evidence)
#       -> grouper (assigns every note to an instrument track and names the track)
#     drum model (drum-kit and orchestral percussion hits; only for sources recorded at >= 40 kHz)



MIN_END_S = 0.03            # every pitched note lasts at least 30 ms before grouping
HOST_BUF_AFTER_S = MAX_S_1PASS   # above this length the transcriber's tap buffers (3 float16 arrays of T x 88 x 368, ~10 MB per
                                 # second of audio: 2.4 GB at 250 s, 3.5 GB at 6 min) live in host memory instead of on the GPU.
                                 # Windows are added on the device in a ~50 MB sliding region and every frame is copied out once
                                 # it is final, so GPU memory stays bounded (a 6-minute song fits the 8 GB card), the cost is
                                 # a few GB of host RAM and one device-to-host copy, and there is still only one network pass.


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Track:
    name: str               # instrument name ("Acoustic Grand Piano", ..., "Drums")
    program: int            # General MIDI program 0-127 (ignored for drums)
    notes: np.ndarray       # (N, 4): onset s, offset s, MIDI pitch, velocity 1-127
    is_drum: bool = False


@dataclass
class Transcription:
    source: str
    seconds: float
    tracks: list[Track]
    info: dict = field(default_factory=dict)

    def to_midi(self):
        """-> pretty_midi.PrettyMIDI (120 BPM, times in seconds)."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for t in self.tracks:
            ins = pretty_midi.Instrument(program=0 if t.is_drum else int(t.program), is_drum=t.is_drum, name=t.name)
            for on, off, pitch, vel in t.notes:
                ins.notes.append(pretty_midi.Note(velocity=int(vel), pitch=int(pitch), start=float(on), end=float(off)))
            pm.instruments.append(ins)
        return pm

    def save_midi(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_midi().write(path)

    def summary(self) -> dict:
        return {"source": self.source, "seconds": round(self.seconds, 2), **self.info,
                "tracks": [{"name": t.name, "program": None if t.is_drum else int(t.program), "is_drum": t.is_drum,
                            "notes": int(len(t.notes))} for t in self.tracks]}


class Dettato:
    """All four models, loaded once.

    >>> model = Dettato.from_pretrained()          # downloads the weights from the Hugging Face Hub on first use
    >>> result = model.transcribe("song.mp3")
    >>> result.save_midi("song.mid")
    """

    def __init__(self, transcriber, repairer, grouper, drums, device):
        self.transcriber, self.repairer, self.grouper, self.drums = transcriber, repairer, grouper, drums
        self.device = device

    @classmethod
    def from_pretrained(cls, device: str | None = None, drums: bool = True):

        device = device or default_device()
        return cls(Transcriber.from_pretrained(device), EvidenceRepairer.from_pretrained(device), Grouper.from_pretrained(device),
                   (DrumTranscriber.from_pretrained(device) if drums else None), device)

    @torch.no_grad()
    def pitched(self, x, batch: int = 2):
        """The pitched part of the pipeline on 16 kHz mono audio x: tuning -> one transcriber pass -> evidence repairer -> grouper.
        -> (repaired notes, transcriber notes, repairer info, tracks, grouper info), all times on x's axis."""
        delta, strength, shift, r = retune_ratio(x)
        y = retune_resample(x, r) if shift else x
        host = len(y) > HOST_BUF_AFTER_S * 16000 and str(self.device).startswith("cuda")
        # every 2 s: all windows (the repairer: maps + hidden); every 4 s: the windows the grouper reads (maps + hidden + stem)
        maps, bufs, (smaps, sbufs) = self.transcriber.run(
            y, hop_frac=0.25, batch=batch, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"),
            buf_device="cpu" if host else None)
        rep, res, rinfo = self.repairer.from_run(self.transcriber, maps, bufs.pop("hidden"))   # notes on y's time axis
        del maps, bufs
        notes = np.stack([rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1)
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        groups, ginfo = self.grouper.group(y, notes, self.transcriber, tuning=(delta, strength, shift, r), run=(smaps, sbufs))
        if r != 1.0:                                    # the groups come back on x's axis already; so do these two
            rep, res = ({**d, "onset": d["onset"] * r, "offset": d["offset"] * r} for d in (rep, res))
        return rep, res, rinfo, groups, ginfo

    @torch.no_grad()
    def transcribe(self, source, rate: int | None = None) -> Transcription:
        """source: a path to an audio file, or a mono float32 array with its sample ``rate``."""

        if isinstance(source, (str, os.PathLike)):
            name, a = str(source), audio_load(str(source))
        else:
            if rate is None:
                raise ValueError("pass the sample rate with an array")
            name, a = "<array>", Audio(np.asarray(source, np.float32), int(rate))
        x = a.at(16000)
        _, _, rinfo, groups, ginfo = self.pitched(x)
        tracks = [Track(g["instrument"].replace("_", " "), int(g["program"]), np.asarray(g["notes"], np.float64)) for g in groups]
        info = {"notes": {"transcribed": rinfo["transcribed"], "dropped": rinfo["dropped"], "inserted": rinfo["inserted"],
                          "grouped": int(sum(len(t.notes) for t in tracks))},
                "tuning_cents": ginfo.get("tuning_cents")}
        if self.drums is None:
            info["drums"] = "disabled (--no-drums)"
        elif a.rate >= MIN_SOURCE_RATE:
            hits = self.drums.transcribe(a.at(44100))
            if hits:
                d = np.array([[h["onset"], h["offset"], h["pitch"], max(1, min(127, int(round(h["velocity"] * 127))))] for h in hits],
                             np.float64)
                tracks.append(Track("Drums", 0, d, is_drum=True))
            info["drums"] = {"hits": len(hits), **getattr(self.drums, "last_info", {})}
        else:
            info["drums"] = f"skipped: source recorded at {a.rate} Hz (the drum model needs >= {MIN_SOURCE_RATE} Hz)"
        return Transcription(name, a.seconds, tracks, info)


_DEFAULT: dict = {}


def transcribe(source, rate: int | None = None, device: str | None = None) -> Transcription:
    """One-call API: loads the models once per device and transcribes ``source`` (path or array + rate)."""
    device = device or default_device()
    if device not in _DEFAULT:
        _DEFAULT[device] = Dettato.from_pretrained(device)
    return _DEFAULT[device].transcribe(source, rate)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════


def _use_weights(weights: str | None) -> None:
    """--weights DIR equivale a DETTATO_WEIGHTS=DIR (lo lee hub_fetch)."""
    if weights is None:
        return
    path = Path(weights).expanduser()
    if not path.is_dir():
        _err(f"Error: --weights no es una carpeta: {path}")
        sys.exit(1)
    os.environ["DETTATO_WEIGHTS"] = str(path)


def _load_models(device: str, drums: bool = True) -> "Dettato":
    dev = None if device == "auto" else device
    try:
        return Dettato.from_pretrained(dev, drums=drums)
    except FileNotFoundError as e:
        _err(f"Error: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 — descarga/carga de pesos: mensaje limpio en vez de traceback
        _err(f"Error: no se pudieron cargar los modelos ({type(e).__name__}: {e}).")
        _err("Sin red, usa --weights DIR (o DETTATO_WEIGHTS=DIR) con los cuatro repos de pesos.")
        sys.exit(1)


def _to_json(res: "Transcription") -> dict:
    out = res.summary()
    for t_out, t in zip(out["tracks"], res.tracks):
        t_out["notes"] = [[round(float(a), 4), round(float(b), 4), int(p), int(v)] for a, b, p, v in t.notes]
    return out


MIN_AUDIO_S = 0.01  # por debajo (p. ej. un fichero de 0-1 muestras) las etapas no tienen ventanas que procesar


def _load_audio(path: Path) -> "Audio":
    """Decodifica el audio con mensajes de error claros y lo sanea (NaN/Inf → 0)."""
    try:
        a = audio_load(str(path))
    except FileNotFoundError:
        _err(f"Error: {path.name} necesita ffmpeg para decodificarse y no está en el PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        _err(f"Error: no se pudo decodificar {path}: ¿formato no soportado o fichero corrupto?")
        sys.exit(1)
    if a.seconds < MIN_AUDIO_S:
        _err(f"Error: el audio está vacío o dura menos de {MIN_AUDIO_S} s: {path}")
        sys.exit(1)
    bad = ~np.isfinite(a.samples)
    if bad.any():
        _err(_c(f"Aviso: {int(bad.sum())} muestras NaN/Inf sustituidas por 0.", YELLOW))
        a.samples = np.nan_to_num(a.samples, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe un fichero de audio a MIDI/JSON."""
    audio_file = Path(args.audio_file)
    if not audio_file.is_file():
        _err(f"Error: no existe el fichero de audio: {audio_file}")
        sys.exit(1)
    _use_weights(args.weights)

    output = Path(args.output) if args.output is not None else None
    is_stdout = output is not None and str(output) == "-"
    if output is None:
        output = audio_file.with_suffix(".mid" if args.format == "midi" else ".json")

    if not is_stdout:  # fallar antes de gastar minutos de cómputo si no se puede escribir
        try:
            if output.is_dir():
                raise IsADirectoryError(f"la salida es un directorio: {output}")
            _ensure_parent(output)
            if not os.access(output.resolve().parent, os.W_OK):
                raise PermissionError(f"sin permiso de escritura en {output.resolve().parent}")
        except OSError as e:
            _err(f"Error: no se puede escribir la salida: {e}")
            sys.exit(1)

    # Todo el progreso va a stderr: stdout queda reservado para '-o -'.
    _err(_c("Cargando modelos…", DIM))
    model = _load_models(args.device, drums=not args.no_drums)
    _err(_c(f"Transcribiendo {audio_file} en {model.device} …", DIM))
    audio = _load_audio(audio_file)
    t0 = time.time()
    try:
        res = model.transcribe(audio.samples, rate=audio.rate)
        res.source = str(audio_file)
    except Exception as e:  # noqa: BLE001
        _err(f"Error: fallo al transcribir {audio_file}: {type(e).__name__}: {e}")
        sys.exit(1)
    _err(_c(f"{res.seconds:.0f} s de audio en {time.time() - t0:.0f} s", DIM))

    if args.notes:
        for t in res.tracks:
            _err(f"  {t.name:<28} {len(t.notes):>6} notas" + ("  (batería)" if t.is_drum else f"  (programa {t.program})"))
    if str(res.info.get("drums", "")).startswith("skipped"):
        _err(_c(f"Aviso: batería {res.info['drums']}", YELLOW))

    try:
        if args.format == "midi":
            if is_stdout:
                buf = io.BytesIO()
                res.to_midi().write(buf)
                sys.stdout.buffer.write(buf.getvalue())
                sys.stdout.buffer.flush()
            else:
                res.save_midi(str(output))
                _err(_c(f"✓ MIDI guardado en {output}", GREEN))
        else:
            payload = json.dumps(_to_json(res), indent=1)
            if is_stdout:
                sys.stdout.write(payload + "\n")
                sys.stdout.flush()
            else:
                output.write_text(payload)
                _err(_c(f"✓ JSON guardado en {output}", GREEN))
    except OSError as e:
        _err(f"Error: no se pudo escribir la salida: {e}")
        sys.exit(1)


def cmd_list_instruments(args: argparse.Namespace) -> None:
    """Lista los instrumentos que el grouper sabe nombrar (con su programa GM)."""
    for name, prog in zip(INSTRUMENTS, PROGRAM):
        print(f"{name:<24} {int(prog)}")


def cmd_info(args: argparse.Namespace) -> None:
    """Carga los modelos y muestra dispositivo, origen de pesos y tamaños."""
    _use_weights(args.weights)
    print(f"{_c('dettato_transcribe', CYAN)} — transcriber + evidence + grouper + drums")
    print(f"  torch                  : {torch.__version__}")
    print(f"  dispositivo por defecto: {default_device()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  pesos                  : {os.environ.get('DETTATO_WEIGHTS') or 'HuggingFace Hub (' + ', '.join(HUB_REPOS.values()) + ')'}")
    model = _load_models(args.device)
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
    sizes = [("transcriber", n(model.transcriber.model)), ("evidence", n(model.repairer.model)),
             ("grouper", n(model.grouper.model)), ("drums", model.drums.n_params() / 1e6)]
    print(f"  cargados en            : {model.device}")
    for name, m in sizes:
        print(f"  {name:<12} {m:6.1f}M parámetros")
    print(f"  {'total':<12} {sum(m for _, m in sizes):6.1f}M parámetros · {len(INSTRUMENTS)} instrumentos")
    print(_c("✓ Modelos cargados correctamente.", GREEN))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="Transcribe un fichero de audio a MIDI/JSON")
    p.add_argument("audio_file", help="Fichero de audio de entrada (wav, mp3, flac, …)")
    p.add_argument(
        "-o", "--output",
        help="Ruta de salida ('-' para stdout; el progreso va a stderr). "
             "Por defecto: <audio_file>.mid o .json según --format.",
    )
    p.add_argument(
        "-f", "--format", choices=["midi", "json"], default="midi",
        help="midi (defecto): .mid con una pista por instrumento y una de batería. "
             "json: resumen (pistas, conteos, afinación) con todas las notas.",
    )
    p.add_argument("-d", "--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    p.add_argument("-w", "--weights", default=None,
                   help="Carpeta con los cuatro repos de pesos (en vez de descargarlos del Hub)")
    p.add_argument("--no-drums", action="store_true", help="No cargar ni ejecutar el modelo de batería")
    p.add_argument("--notes", action="store_true", help="Imprime las pistas y su número de notas en stderr")
    p.set_defaults(func=cmd_transcribe)

    p2 = sub.add_parser("list-instruments", help="Lista los instrumentos que el grouper sabe nombrar")
    p2.set_defaults(func=cmd_list_instruments)

    p3 = sub.add_parser("info", help="Carga los modelos y muestra dispositivo, pesos y tamaños")
    p3.add_argument("-d", "--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    p3.add_argument("-w", "--weights", default=None, help="Carpeta con los cuatro repos de pesos")
    p3.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except BrokenPipeError:  # p. ej. `list-instruments | head -1`: salir en silencio
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)


if __name__ == "__main__":
    main()
