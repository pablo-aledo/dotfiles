#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AUDIO LAB  v1.1                                      ║
║  Síntesis aditiva, análisis DSP y renderizado MIDI — fichero único           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUBCOMANDOS                                                                 ║
║    analyse       WAV/MIDI → espectro FFT/DFT  (amplitudes, fases, freqs)    ║
║    spectrogram   WAV → STFT → .npz [+ --plot PNG]                           ║
║    edit-spectrum .npz → filtros/stretch/pitch/reverse → .npz [+ --wav]      ║
║    npz-to-png    .npz → PNG escala de grises (metadatos embebidos en tEXt)  ║
║    png-to-npz    PNG → .npz con fase cero  (editar espectrograma en imagen) ║
║    reconstruct   .npz / espectro.json → WAV  (síntesis inversa)             ║
║    synth         Síntesis aditiva: nota(s) + timbre.json + env.json → WAV   ║
║    play-midi     MIDI + instrument-map.json → WAV  (sin soundfont externo)  ║
║    roundtrip     WAV → FFT → síntesis → WAV  (diagnóstico del pipeline)     ║
║    info          Diagnóstico rápido de WAV, MIDI, NPZ o JSON                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS  numpy  soundfile  mido  Pillow                               ║
║                scipy (opcional, --plot)   matplotlib (opcional, --plot)     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE NPZ ↔ PNG                                                          ║
║                                                                              ║
║  El PNG es una imagen en escala de grises donde:                            ║
║    · Eje X  = tiempo  (frame 0 a N-1, izquierda → derecha)                 ║
║    · Eje Y  = frecuencia  (0 Hz abajo, Nyquist arriba)                      ║
║    · Valor  = magnitud STFT  (lineal o log según --log)                     ║
║    · Fase   NO se guarda — al reconvertir se usa fase cero                  ║
║                                                                              ║
║  Los metadatos necesarios para la reconversión (sr, window_size, hop_ratio, ║
║  escala, piso dB, orientación) se guardan en chunks tEXt del PNG y se       ║
║  recuperan automáticamente con png-to-npz. Si el PNG viene de un editor     ║
║  externo (GIMP, Photoshop…) que elimina los metadatos, pasa sr/window/hop   ║
║  manualmente con --sr --window --hop.                                        ║
║                                                                              ║
║  Flujo típico de edición creativa:                                           ║
║    spectrogram audio.wav --window 4096 -o audio.npz                         ║
║    npz-to-png audio.npz --log -o audio.png                                  ║
║    [editar audio.png en GIMP/Photoshop: borrar, pintar, filtrar…]           ║
║    png-to-npz audio_editado.png -o audio_editado.npz                        ║
║    reconstruct audio_editado.npz -o resultado.wav                           ║
║                                                                              ║
║  Nota: la reconstrucción desde PNG usa Griffin-Lim iterativo (32 iter por   ║
║  defecto) para estimar fases coherentes. Produce audio limpio y con las      ║
║  frecuencias correctas. Ajusta con reconstruct --gl-iters N (más=mejor).    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  JSON DE REFERENCIA — TIMBRE  (--timbre timbre.json)                        ║
║  Define el perfil espectral de un instrumento: amplitud relativa de cada    ║
║  armónico (1=fundamental) en distintos registros (key frames por nota MIDI).║
║  Se interpola linealmente entre key frames.                                  ║
║                                                                              ║
║  {                                                                           ║
║    "name": "piano_approx",                                                   ║
║    "key_frames": {                                                           ║
║      "36": [1.0, 0.6, 0.4, 0.25, 0.15, 0.10, 0.06, 0.04],  // C2         ║
║      "60": [1.0, 0.5, 0.3, 0.18, 0.10, 0.06, 0.03, 0.01],  // C4         ║
║      "84": [1.0, 0.3, 0.1, 0.04, 0.01, 0.00, 0.00, 0.00]   // C6         ║
║    },                                                                        ║
║    "n_harmonics": 8,                                                         ║
║    "normalize": true                                                         ║
║  }                                                                           ║
║                                                                              ║
║  JSON DE REFERENCIA — ENVOLVENTE  (--envelope env.json)                     ║
║  Parámetros completos ADR con tipo de curva independiente por segmento.      ║
║  Tipos de curva: "linear" | "exponential" | "logarithmic" | "cosine"        ║
║                                                                              ║
║  {                                                                           ║
║    "attack_time":   0.010,                                                   ║
║    "attack_curve":  "cosine",                                                ║
║    "decay_time":    1.800,                                                   ║
║    "decay_curve":   "exponential",                                           ║
║    "decay_level":   0.30,                                                    ║
║    "release_time":  0.250,                                                   ║
║    "release_curve": "exponential",                                           ║
║    "harmonic_decay": [1.0, 1.8, 2.5, 3.2, 4.0, 5.0, 6.0, 7.0]            ║
║  }                                                                           ║
║                                                                              ║
║  harmonic_decay: factor de decay relativo por armónico (>1 = decae antes).  ║
║                                                                              ║
║  JSON DE REFERENCIA — INSTRUMENT MAP  (--instrument-map map.json)           ║
║  Asignación de canales MIDI → instrumento. Cada canal puede usar síntesis   ║
║  aditiva (timbre+envelope JSONs) o samples WAV externos con pitch shift.    ║
║                                                                              ║
║  {                                                                           ║
║    "default": {                                                              ║
║      "mode": "additive",                                                     ║
║      "timbre": "timbres/piano.json",                                         ║
║      "envelope": "envelopes/piano.json"                                      ║
║    },                                                                        ║
║    "channels": {                                                             ║
║      "0": {                                                                  ║
║        "mode": "additive",                                                   ║
║        "timbre": "timbres/piano.json",                                       ║
║        "envelope": "envelopes/piano.json"                                    ║
║      },                                                                      ║
║      "1": {                                                                  ║
║        "mode": "sample",                                                     ║
║        "sample": "samples/flute_c4.wav",                                     ║
║        "root_note": 60,                                                      ║
║        "envelope": "envelopes/flute.json"                                    ║
║      },                                                                      ║
║      "9": {                                                                  ║
║        "mode": "additive",                                                   ║
║        "timbre": "timbres/marimba.json",                                     ║
║        "envelope": "envelopes/perc.json"                                     ║
║      }                                                                       ║
║    }                                                                         ║
║  }                                                                           ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║    python audio_lab.py analyse piano.wav --method fft --output spec.json   ║
║    python audio_lab.py spectrogram piano.wav --window 4096 --log           ║
║    python audio_lab.py npz-to-png piano.npz --log -o piano.png             ║
║    python audio_lab.py png-to-npz piano_edited.png -o piano_edited.npz     ║
║    python audio_lab.py reconstruct piano_edited.npz -o resultado.wav       ║
║    python audio_lab.py synth --notes C4 E4 G4 --timbre timbres/piano.json  ║
║    python audio_lab.py play-midi song.mid --instrument-map map.json         ║
║    python audio_lab.py roundtrip input.wav --method stft                   ║
║    python audio_lab.py info piano.wav                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import sys
import wave as _wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── lazy imports ──────────────────────────────────────────────────────────────
def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. Instala con: pip install soundfile")

def _import_mido():
    try:
        import mido
        return mido
    except ImportError:
        sys.exit("✗  mido no encontrado. Instala con: pip install mido")

def _import_scipy_signal():
    try:
        from scipy import signal
        return signal
    except ImportError:
        sys.exit("✗  scipy no encontrado (necesario para --plot). pip install scipy")

def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.exit("✗  matplotlib no encontrado. pip install matplotlib")

def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        sys.exit("✗  Pillow no encontrado. Instala con: pip install Pillow")

# ── constantes ────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 44100
A4_HZ        = 440.0
A4_MIDI      = 69
NOTE_NAMES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

DEFAULT_TIMBRE = {
    "name": "sine",
    "key_frames": {"0": [1.0], "127": [1.0]},
    "n_harmonics": 1,
    "normalize": True,
}

DEFAULT_ENVELOPE = {
    "attack_time":   0.010,
    "attack_curve":  "cosine",
    "decay_time":    1.500,
    "decay_curve":   "exponential",
    "decay_level":   0.30,
    "release_time":  0.200,
    "release_curve": "exponential",
    "harmonic_decay": None,
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE PITCH
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_hz(note: int) -> float:
    return A4_HZ * 2 ** ((note - A4_MIDI) / 12.0)

def note_name_to_midi(name: str) -> int:
    """'C4', 'F#3', 'Bb5' → número MIDI."""
    name = name.strip().replace("♭", "b").replace("♯", "#")
    octave = int(name[-1])
    pitch  = name[:-1].upper().replace("BB","A#").replace("EB","D#") \
                               .replace("AB","G#").replace("DB","C#") \
                               .replace("GB","F#")
    if pitch not in NOTE_NAMES:
        raise ValueError(f"Nota no reconocida: {name!r}")
    return (octave + 1) * 12 + NOTE_NAMES.index(pitch)

def midi_to_note_name(n: int) -> str:
    return NOTE_NAMES[n % 12] + str(n // 12 - 1)

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Devuelve (audio_mono_float32, sample_rate)."""
    sf = _import_soundfile()
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sr

def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf = _import_soundfile()
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_24")
    print(f"  ✓  WAV escrito → {path}  ({len(audio)/sr:.2f}s, {sr}Hz)")

# ══════════════════════════════════════════════════════════════════════════════
# ENVOLVENTE ADR
# ══════════════════════════════════════════════════════════════════════════════

def _apply_curve(t: np.ndarray, curve: str) -> np.ndarray:
    """Mapea [0,1]→[0,1] según tipo de curva."""
    if curve == "linear":
        return t
    elif curve == "exponential":
        eps = 1e-6
        return (np.exp(t * 6) - 1) / (np.exp(6) - 1 + eps)
    elif curve == "logarithmic":
        eps = 1e-6
        return np.log1p(t * (math.e - 1))
    elif curve == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        return t

@dataclass
class EnvelopeADR:
    attack_time:   float = 0.010
    attack_curve:  str   = "cosine"
    decay_time:    float = 1.500
    decay_curve:   str   = "exponential"
    decay_level:   float = 0.30
    release_time:  float = 0.200
    release_curve: str   = "exponential"
    harmonic_decay: Optional[List[float]] = None  # factor/armónico

    @classmethod
    def from_dict(cls, d: dict) -> "EnvelopeADR":
        return cls(
            attack_time=d.get("attack_time", DEFAULT_ENVELOPE["attack_time"]),
            attack_curve=d.get("attack_curve", DEFAULT_ENVELOPE["attack_curve"]),
            decay_time=d.get("decay_time", DEFAULT_ENVELOPE["decay_time"]),
            decay_curve=d.get("decay_curve", DEFAULT_ENVELOPE["decay_curve"]),
            decay_level=d.get("decay_level", DEFAULT_ENVELOPE["decay_level"]),
            release_time=d.get("release_time", DEFAULT_ENVELOPE["release_time"]),
            release_curve=d.get("release_curve", DEFAULT_ENVELOPE["release_curve"]),
            harmonic_decay=d.get("harmonic_decay"),
        )

    @classmethod
    def from_file(cls, path: Optional[str]) -> "EnvelopeADR":
        if path is None:
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def render(self, duration: float, sr: int = SAMPLE_RATE,
               harmonic_idx: int = 0) -> np.ndarray:
        """
        Genera la envolvente para una nota de `duration` segundos.
        harmonic_idx ajusta el decay según harmonic_decay[i].
        """
        n_total   = int(duration * sr)
        n_attack  = max(1, int(self.attack_time * sr))
        n_release = max(1, int(self.release_time * sr))

        # factor de decay para este armónico
        hd_factor = 1.0
        if self.harmonic_decay and harmonic_idx < len(self.harmonic_decay):
            hd_factor = float(self.harmonic_decay[harmonic_idx])

        decay_time_h = self.decay_time / hd_factor
        n_decay  = max(1, int(decay_time_h * sr))
        n_sustain = max(0, n_total - n_attack - n_decay - n_release)

        env = np.zeros(n_total, dtype=np.float32)
        pos = 0

        # Attack
        end = min(pos + n_attack, n_total)
        t   = np.linspace(0, 1, end - pos)
        env[pos:end] = _apply_curve(t, self.attack_curve)
        pos = end

        # Decay → decay_level
        end = min(pos + n_decay, n_total)
        if end > pos:
            t = np.linspace(0, 1, end - pos)
            env[pos:end] = 1.0 - (1.0 - self.decay_level) * _apply_curve(t, self.decay_curve)
        pos = end

        # Sustain (nivel constante)
        end = min(pos + n_sustain, n_total)
        env[pos:end] = self.decay_level
        pos = end

        # Release → 0
        end = min(pos + n_release, n_total)
        if end > pos:
            t = np.linspace(0, 1, end - pos)
            env[pos:end] = self.decay_level * (1.0 - _apply_curve(t, self.release_curve))

        return env

# ══════════════════════════════════════════════════════════════════════════════
# TIMBRE (perfil de armónicos)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Timbre:
    name: str
    key_frames: Dict[int, List[float]]   # nota_midi → [amp_arm1, amp_arm2, ...]
    n_harmonics: int
    normalize: bool

    @classmethod
    def from_dict(cls, d: dict) -> "Timbre":
        kf = {int(k): v for k, v in d["key_frames"].items()}
        return cls(
            name=d.get("name", "unnamed"),
            key_frames=kf,
            n_harmonics=d.get("n_harmonics", 8),
            normalize=d.get("normalize", True),
        )

    @classmethod
    def from_file(cls, path: Optional[str]) -> "Timbre":
        if path is None:
            return cls.from_dict(DEFAULT_TIMBRE)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def get_harmonics(self, midi_note: int) -> np.ndarray:
        """Interpola entre key frames y devuelve array de amplitudes normalizadas."""
        keys = sorted(self.key_frames.keys())
        if len(keys) == 0:
            return np.ones(self.n_harmonics, dtype=np.float32)

        if midi_note <= keys[0]:
            amps = np.array(self.key_frames[keys[0]], dtype=np.float32)
        elif midi_note >= keys[-1]:
            amps = np.array(self.key_frames[keys[-1]], dtype=np.float32)
        else:
            lo = max(k for k in keys if k <= midi_note)
            hi = min(k for k in keys if k > midi_note)
            t  = (midi_note - lo) / (hi - lo)
            a  = np.array(self.key_frames[lo], dtype=np.float32)
            b  = np.array(self.key_frames[hi], dtype=np.float32)
            # Pad al máximo de los dos
            n = max(len(a), len(b))
            a = np.pad(a, (0, n - len(a)))
            b = np.pad(b, (0, n - len(b)))
            amps = a + t * (b - a)

        # Pad/truncar a n_harmonics
        n = self.n_harmonics
        if len(amps) < n:
            amps = np.pad(amps, (0, n - len(amps)))
        else:
            amps = amps[:n]

        if self.normalize and amps.max() > 0:
            amps = amps / amps.max()

        return amps

# ══════════════════════════════════════════════════════════════════════════════
# SÍNTESIS ADITIVA
# ══════════════════════════════════════════════════════════════════════════════

def synth_note(midi_note: int, duration: float, velocity: float,
               timbre: Timbre, envelope: EnvelopeADR,
               sr: int = SAMPLE_RATE,
               phase_offset: float = 0.0) -> np.ndarray:
    """
    Genera samples float32 de una nota mediante síntesis aditiva.
    velocity ∈ [0, 1].
    phase_offset: fase inicial en radianes (evita coherencia de fase entre notas).
    NO normaliza internamente: devuelve amplitud proporcional a velocity y
    al perfil de armónicos, para que la mezcla de varias notas conserve
    las relaciones de dinámica correctas.
    """
    n_samples  = int(duration * sr)
    t          = np.arange(n_samples, dtype=np.float64) / sr
    fund_hz    = midi_to_hz(midi_note)
    harmonics  = timbre.get_harmonics(midi_note)
    out        = np.zeros(n_samples, dtype=np.float64)

    # Fase dorada por armónico: distribuye la energía en el tiempo y evita
    # que todos los armónicos sumen constructivamente en t=0 (impulso inicial).
    GOLDEN = 2.399963229728653   # 2π × (1 - 1/φ)

    for i, amp in enumerate(harmonics):
        if amp < 1e-6:
            continue
        freq = fund_hz * (i + 1)
        if freq >= sr / 2:
            break
        env   = envelope.render(duration, sr, harmonic_idx=i)
        phase = phase_offset + i * GOLDEN
        wave  = np.sin(2 * np.pi * freq * t + phase)
        out  += wave * env * float(amp)

    # Escala al nº de armónicos activos para que el peak teórico ≈ velocity
    n_active = max(1, sum(1 for a in harmonics if a >= 1e-6))
    out *= velocity / n_active
    return out.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# PITCH SHIFTING DE SAMPLES (resampling)
# ══════════════════════════════════════════════════════════════════════════════

def pitch_shift_resample(audio: np.ndarray, semitones: float,
                          sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Transpone `audio` en `semitones` semitones por resampling.
    Altera la duración (rápido, sin artefactos de fase).
    Funciona bien para ±7 semitones; para más rango considera WSOLA.
    """
    ratio     = 2 ** (semitones / 12.0)
    old_len   = len(audio)
    new_len   = max(1, int(old_len / ratio))
    old_idx   = np.arange(old_len, dtype=np.float64)
    new_idx   = np.linspace(0, old_len - 1, new_len)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# DSP: DFT / FFT / STFT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FreqBin:
    freq_hz:   float
    amplitude: float
    phase_rad: float

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def dft_analyse(audio: np.ndarray, sr: int, max_bins: int = 2048) -> List[FreqBin]:
    """DFT naive (lenta, didáctica). Devuelve bins hasta Nyquist."""
    n      = min(len(audio), max_bins)
    audio  = audio[:n]
    result = []
    with ThreadPoolExecutor() as ex:
        def _bin(k):
            tw = np.exp(-2j * np.pi * k * np.arange(n) / n)
            X  = float(np.dot(audio, tw.real)) - 1j * float(np.dot(audio, tw.imag))
            amp   = abs(X) * 2 / n
            phase = math.atan2(X.imag, X.real)
            freq  = k * sr / n
            return FreqBin(freq, amp, phase)
        bins = list(ex.map(_bin, range(n // 2)))
    return bins

def fft_analyse(audio: np.ndarray, sr: int) -> List[FreqBin]:
    """FFT via numpy. Rápida y precisa."""
    n    = _next_pow2(len(audio))
    buf  = np.zeros(n, dtype=np.float32)
    buf[:len(audio)] = audio
    X    = np.fft.rfft(buf)
    amps = np.abs(X) * 2 / n
    phases = np.angle(X)
    freqs  = np.fft.rfftfreq(n, 1.0 / sr)
    return [FreqBin(float(freqs[k]), float(amps[k]), float(phases[k]))
            for k in range(len(freqs))]

def stft_analyse(audio: np.ndarray, sr: int,
                 window_size: int = 2048,
                 hop_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT con ventana Hann.
    Devuelve (magnitudes [frames×bins], fases [frames×bins], freqs_hz [bins]).
    """
    hop    = max(1, int(window_size * hop_ratio))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / window_size)
    n_pad  = window_size // 2
    padded = np.pad(audio, (n_pad, n_pad + window_size), mode="constant")
    frames = (len(padded) - window_size) // hop + 1

    n_bins = window_size // 2 + 1
    mags   = np.zeros((frames, n_bins), dtype=np.float32)
    phases = np.zeros((frames, n_bins), dtype=np.float32)

    for i in range(frames):
        start = i * hop
        chunk = padded[start : start + window_size] * window
        X     = np.fft.rfft(chunk, n=window_size)
        mags[i]   = np.abs(X)
        phases[i] = np.angle(X)

    freqs = np.fft.rfftfreq(window_size, 1.0 / sr)
    return mags, phases, freqs

def griffin_lim(mags: np.ndarray, hop_ratio: float,
                n_iter: int = 32, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Griffin-Lim iterativo: estima fases coherentes a partir de magnitudes STFT.
    Produce audio mucho más limpio que fase cero cuando no se dispone de las
    fases originales (ej: reconstrucción desde PNG).

    mags: [frames × bins]  magnitudes STFT
    n_iter: iteraciones (16-32 suelen bastar; más = más limpio pero más lento)
    """
    n_bins  = mags.shape[1]
    n_fft   = (n_bins - 1) * 2
    hop     = max(1, int(n_fft * hop_ratio))
    frames  = mags.shape[0]
    win     = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)

    # Inicializar con fase aleatoria reproducible
    rng    = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, mags.shape)

    def _istft(ph: np.ndarray) -> np.ndarray:
        out_len = (frames - 1) * hop + n_fft
        out     = np.zeros(out_len, dtype=np.float64)
        norm    = np.zeros(out_len, dtype=np.float64)
        for i in range(frames):
            X     = mags[i] * np.exp(1j * ph[i])
            chunk = np.fft.irfft(X, n=n_fft)[:n_fft] * win
            start = i * hop
            out[start:start + n_fft]  += chunk
            norm[start:start + n_fft] += win ** 2
        nz = norm > 1e-10
        out[nz] /= norm[nz]
        return out

    def _stft_phases(audio: np.ndarray) -> np.ndarray:
        n_pad  = n_fft // 2
        padded = np.pad(audio, (n_pad, n_pad + n_fft))
        ph     = np.zeros(mags.shape, dtype=np.float64)
        for i in range(frames):
            start = i * hop
            chunk = padded[start:start + n_fft] * win
            X     = np.fft.rfft(chunk, n=n_fft)
            ph[i] = np.angle(X)
        return ph

    for _ in range(n_iter):
        audio  = _istft(phases)
        phases = _stft_phases(audio)

    # ISTFT final con fases estimadas y magnitudes originales
    audio  = _istft(phases)
    n_pad  = n_fft // 2
    return audio[n_pad : n_pad + len(audio) - n_fft].astype(np.float32)


def stft_reconstruct(mags: np.ndarray, phases: np.ndarray,
                     window_size: int, hop_ratio: float,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Síntesis inversa desde magnitudes + fases STFT (overlap-add con ventana Hann).
    El tamaño de irfft se deriva de n_bins (columnas de mags) para ser robusto
    ante discrepancias entre window_size y el NPZ real.
    """
    n_bins  = mags.shape[1]
    n_fft   = (n_bins - 1) * 2
    hop     = max(1, int(n_fft * hop_ratio))
    frames  = mags.shape[0]
    window  = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    out_len = (frames - 1) * hop + n_fft
    out     = np.zeros(out_len, dtype=np.float64)
    norm    = np.zeros(out_len, dtype=np.float64)

    for i in range(frames):
        X     = mags[i] * np.exp(1j * phases[i])
        chunk = np.fft.irfft(X, n=n_fft)[:n_fft]
        chunk *= window
        start  = i * hop
        out[start : start + n_fft]  += chunk
        norm[start : start + n_fft] += window ** 2

    nz       = norm > 1e-10
    out[nz] /= norm[nz]
    n_pad    = n_fft // 2
    out      = out[n_pad : n_pad + len(out) - n_fft]
    return out.astype(np.float32)

def synth_from_bins(bins: List[FreqBin], duration: float,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Síntesis aditiva desde lista de FreqBin (para reconstruct desde FFT/DFT)."""
    n   = int(duration * sr)
    t   = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for b in bins:
        if b.freq_hz >= sr / 2 or b.amplitude < 1e-6:
            continue
        out += b.amplitude * np.sin(2 * np.pi * b.freq_hz * t + b.phase_rad)
    peak = np.abs(out).max()
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT MAP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelInstrument:
    mode:      str              # "additive" | "sample"
    timbre:    Optional[Timbre]
    envelope:  EnvelopeADR
    sample:    Optional[np.ndarray] = None
    sample_sr: int = SAMPLE_RATE
    root_note: int = 60

def load_instrument_map(path: str) -> Dict[int, ChannelInstrument]:
    """Carga el JSON de asignación y devuelve dict canal→ChannelInstrument."""
    with open(path) as f:
        data = json.load(f)

    base_dir = Path(path).parent

    def _resolve(p: str) -> str:
        return str(base_dir / p) if not Path(p).is_absolute() else p

    def _load_channel(cfg: dict) -> ChannelInstrument:
        mode = cfg.get("mode", "additive")
        env_path = cfg.get("envelope")
        envelope = EnvelopeADR.from_file(_resolve(env_path) if env_path else None)

        if mode == "additive":
            timbre_path = cfg.get("timbre")
            timbre = Timbre.from_file(_resolve(timbre_path) if timbre_path else None)
            return ChannelInstrument(mode="additive", timbre=timbre, envelope=envelope)
        elif mode == "sample":
            sample_path = _resolve(cfg["sample"])
            audio, sr   = read_wav(sample_path)
            root        = int(cfg.get("root_note", 60))
            return ChannelInstrument(mode="sample", timbre=None, envelope=envelope,
                                     sample=audio, sample_sr=sr, root_note=root)
        else:
            raise ValueError(f"Modo de instrumento desconocido: {mode!r}")

    default_cfg = data.get("default", {"mode": "additive"})
    default_inst = _load_channel(default_cfg)

    result: Dict[int, ChannelInstrument] = {}
    for ch_str, cfg in data.get("channels", {}).items():
        result[int(ch_str)] = _load_channel(cfg)

    # canal sentinel -1 = default
    result[-1] = default_inst
    return result

# ══════════════════════════════════════════════════════════════════════════════
# RENDER DE NOTA CON INSTRUMENTO
# ══════════════════════════════════════════════════════════════════════════════

def render_note(inst: ChannelInstrument, midi_note: int,
                duration: float, velocity: float,
                sr: int = SAMPLE_RATE,
                phase_offset: float = 0.0) -> np.ndarray:
    if inst.mode == "additive":
        return synth_note(midi_note, duration, velocity,
                          inst.timbre, inst.envelope, sr,
                          phase_offset=phase_offset)
    elif inst.mode == "sample":
        semitones = midi_note - inst.root_note
        shifted   = pitch_shift_resample(inst.sample, semitones, inst.sample_sr)
        # Resample a sr destino si difieren
        if inst.sample_sr != sr:
            n_out   = int(len(shifted) * sr / inst.sample_sr)
            idx_old = np.arange(len(shifted), dtype=np.float64)
            idx_new = np.linspace(0, len(shifted) - 1, n_out)
            shifted = np.interp(idx_new, idx_old, shifted).astype(np.float32)
        # Truncar/pad a duration
        n_target = int(duration * sr)
        if len(shifted) >= n_target:
            out = shifted[:n_target].copy()
        else:
            out = np.pad(shifted, (0, n_target - len(shifted)))
        env = inst.envelope.render(duration, sr, harmonic_idx=0)
        out *= env * velocity
        return out
    return np.zeros(int(duration * sr), dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# MIDI → AUDIO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _MidiEvent:
    time_s:   float
    channel:  int
    note:     int
    velocity: float
    on:       bool

def _parse_midi_events(path: str, bpm_override: Optional[float] = None,
                        transpose: int = 0) -> Tuple[List[_MidiEvent], float]:
    """Extrae eventos note_on/off con timestamps en segundos."""
    mido = _import_mido()
    mid  = mido.MidiFile(path)
    tpb  = mid.ticks_per_beat

    tempo   = 500000  # 120 BPM por defecto
    if bpm_override:
        tempo = int(60_000_000 / bpm_override)

    events: List[_MidiEvent] = []
    for track in mid.tracks:
        tick_abs = 0
        time_s   = 0.0
        local_tempo = tempo
        for msg in track:
            tick_abs += msg.time
            time_s   += mido.tick2second(msg.time, tpb, local_tempo)
            if not bpm_override and msg.type == "set_tempo":
                local_tempo = msg.tempo
            if msg.type == "note_on":
                events.append(_MidiEvent(
                    time_s   = time_s,
                    channel  = msg.channel,
                    note     = msg.note + transpose,
                    velocity = msg.velocity / 127.0,
                    on       = msg.velocity > 0,
                ))
            elif msg.type == "note_off":
                events.append(_MidiEvent(
                    time_s   = time_s,
                    channel  = msg.channel,
                    note     = msg.note + transpose,
                    velocity = 0.0,
                    on       = False,
                ))

    events.sort(key=lambda e: e.time_s)
    total = max((e.time_s for e in events), default=0.0) + 2.0
    return events, total

def render_midi(midi_path: str, instrument_map: Dict[int, ChannelInstrument],
                bpm_override: Optional[float] = None, transpose: int = 0,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Renderiza un MIDI completo a float32 mezclando todos los canales.
    Usa instrument_map[canal] o instrument_map[-1] como default.
    """
    events, total_s = _parse_midi_events(midi_path, bpm_override, transpose)
    n_total = int(total_s * sr) + sr  # +1s margen
    out     = np.zeros(n_total, dtype=np.float32)

    # Rastrear notas activas: (channel, note) → time_on, velocity
    active: Dict[Tuple[int,int], Tuple[float,float]] = {}

    print(f"  [render] {len(events)} eventos MIDI | {total_s:.1f}s | sr={sr}Hz")

    for ev in events:
        key = (ev.channel, ev.note)
        if ev.on and ev.velocity > 0:
            active[key] = (ev.time_s, ev.velocity)
        else:
            if key not in active:
                continue
            t_on, vel = active.pop(key)
            duration  = ev.time_s - t_on
            if duration < 0.01:
                continue

            inst = instrument_map.get(ev.channel, instrument_map.get(-1))
            if inst is None:
                continue

            # Fase basada en nota+canal: notas distintas no suman en fase
            phase = (ev.note * 0.7 + ev.channel * 1.3) % (2 * math.pi)
            note_audio = render_note(inst, ev.note, duration, vel, sr,
                                     phase_offset=phase)
            start = int(t_on * sr)
            end   = start + len(note_audio)
            if end > len(out):
                pad = end - len(out)
                out = np.pad(out, (0, pad))
            out[start:end] += note_audio

    # Notas sin note_off al final
    for (ch, note), (t_on, vel) in active.items():
        duration = total_s - t_on
        if duration < 0.01:
            continue
        inst = instrument_map.get(ch, instrument_map.get(-1))
        if inst is None:
            continue
        phase = (note * 0.7 + ch * 1.3) % (2 * math.pi)
        note_audio = render_note(inst, note, duration, vel, sr,
                                 phase_offset=phase)
        start = int(t_on * sr)
        end   = start + len(note_audio)
        if end > len(out):
            out = np.pad(out, (0, end - len(out)))
        out[start:end] += note_audio

    # Normalizar mezcla
    peak = np.abs(out).max()
    if peak > 1e-6:
        out /= peak * 1.05   # headroom leve
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_analyse(args):
    """WAV o MIDI → lista de frecuencias/amplitudes/fases → JSON."""
    path = Path(args.input)
    sr   = SAMPLE_RATE

    if path.suffix.lower() in (".mid", ".midi"):
        # Para MIDI: calcular espectro teórico de las notas activas
        mido = _import_mido()
        mid  = mido.MidiFile(str(path))
        notes = set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    notes.add(msg.note)
        bins = []
        for n in sorted(notes):
            hz = midi_to_hz(n)
            bins.append({"note": midi_to_note_name(n), "midi": n,
                         "freq_hz": round(hz, 3), "amplitude": 1.0, "phase_rad": 0.0})
        print(f"  [analyse] MIDI: {len(notes)} notas únicas detectadas")
    else:
        audio, sr = read_wav(str(path))
        method    = args.method.lower()
        if method == "dft":
            max_b = getattr(args, "max_bins", 2048)
            print(f"  [analyse] DFT naive ({max_b} muestras)...")
            raw  = dft_analyse(audio, sr, max_b)
        else:
            print(f"  [analyse] FFT ({_next_pow2(len(audio))} puntos)...")
            raw  = fft_analyse(audio, sr)

        # Filtrar bins por umbral de amplitud
        threshold = getattr(args, "threshold", 0.001)
        raw       = [b for b in raw if b.amplitude >= threshold]
        raw.sort(key=lambda b: b.amplitude, reverse=True)
        if hasattr(args, "top_n") and args.top_n:
            raw = raw[:args.top_n]

        bins = [{"freq_hz": round(b.freq_hz, 4),
                 "amplitude": round(b.amplitude, 6),
                 "phase_rad": round(b.phase_rad, 6)} for b in raw]
        print(f"  [analyse] {len(bins)} bins (amp ≥ {threshold})")

    out = {
        "source": str(path),
        "method": getattr(args, "method", "theoretical"),
        "sample_rate": sr,
        "bins": bins,
    }
    out_path = args.output or path.stem + "_spectrum.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓  Espectro → {out_path}  ({len(bins)} bins)")


def cmd_spectrogram(args):
    """WAV → STFT → .npz (+ --plot PNG)."""
    audio, sr = read_wav(args.input)
    window    = args.window
    hop       = args.hop

    print(f"  [spectrogram] STFT window={window} hop={hop} → {len(audio)/sr:.2f}s")
    mags, phases, freqs = stft_analyse(audio, sr, window_size=window, hop_ratio=hop)

    out_path = args.output or Path(args.input).stem + "_stft.npz"
    np.savez_compressed(out_path,
                        magnitudes=mags, phases=phases, freqs=freqs,
                        sample_rate=np.array(sr),
                        window_size=np.array(window),
                        hop_ratio=np.array(hop))
    print(f"  ✓  STFT → {out_path}  ({mags.shape[0]} frames × {mags.shape[1]} bins)")

    if args.plot:
        plt = _import_matplotlib()
        _  = _import_scipy_signal()   # solo para verificar disponibilidad
        db_floor = -80
        mag_db   = 20 * np.log10(mags.T + 1e-10)
        mag_db   = np.clip(mag_db, db_floor, 0)

        fig, ax = plt.subplots(figsize=(14, 5))
        duration = len(audio) / sr
        ax.imshow(mag_db, origin="lower", aspect="auto",
                  extent=[0, duration, 0, sr / 2],
                  cmap="inferno", vmin=db_floor, vmax=0)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_title(f"Espectrograma STFT — {Path(args.input).name}")
        plt.colorbar(ax.images[0], ax=ax, label="dB")
        png_path = str(Path(out_path).with_suffix(".png"))
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"  ✓  PNG → {png_path}")


def cmd_reconstruct(args):
    """NPZ o JSON de espectro → WAV por síntesis inversa."""
    src = Path(args.input)

    if src.suffix == ".npz":
        data      = np.load(str(src))
        mags      = data["magnitudes"]
        phases    = data["phases"]
        sr        = int(data["sample_rate"])
        window    = int(data["window_size"])
        hop       = float(data["hop_ratio"])
        zero_phase = np.abs(phases).max() < 1e-6
        gl_iters   = getattr(args, "gl_iters", 32)

        if zero_phase or args.normalize:
            print(f"  [reconstruct] STFT inversa  {mags.shape[0]} frames × {mags.shape[1]} bins"
                  f"  [Griffin-Lim {gl_iters} iter]")
            audio = griffin_lim(mags, hop, n_iter=gl_iters, sr=sr)
        else:
            print(f"  [reconstruct] STFT inversa  {mags.shape[0]} frames × {mags.shape[1]} bins")
            audio = stft_reconstruct(mags, phases, window, hop, sr)

        peak = np.abs(audio).max()
        if (zero_phase or args.normalize) and peak > 1e-9:
            audio = (audio / peak * 0.85).astype(np.float32)
            print(f"  [reconstruct] Normalizado a 0.85 peak")
    elif src.suffix == ".json":
        with open(str(src)) as f:
            spec = json.load(f)
        sr       = spec.get("sample_rate", SAMPLE_RATE)
        duration = args.duration or 2.0
        bins     = [FreqBin(b["freq_hz"], b["amplitude"], b["phase_rad"])
                    for b in spec["bins"]]
        print(f"  [reconstruct] Síntesis aditiva  {len(bins)} bins  {duration}s")
        audio = synth_from_bins(bins, duration, sr)
    else:
        sys.exit(f"✗  Formato no soportado: {src.suffix!r}  (usa .npz o .json)")

    out_path = args.output or src.stem + "_recon.wav"
    write_wav(out_path, audio, sr)


def cmd_edit_spectrum(args):
    """
    Edita un NPZ de espectrograma (STFT) antes de reconstruirlo como WAV.

    Operaciones disponibles (se aplican en este orden):
      1. highpass / lowpass  — filtro de paso en frecuencia
      2. isolate             — dejar solo bandas alrededor de frecuencias concretas
      3. suppress            — atenuar bandas alrededor de frecuencias concretas
      4. gain                — ganancia global en dB sobre las magnitudes
      5. denoise             — zeroing de bins por debajo de un umbral de magnitud
      6. timestretch         — estirar/comprimir en el tiempo (sin cambiar pitch)
      7. pitchshift          — transponer en semitones (sin cambiar duración)
      8. reverse             — invertir el espectrograma en el tiempo
    """
    src  = Path(args.input)
    data = np.load(str(src))
    mags   = data["magnitudes"].copy().astype(np.float64)   # [frames × bins]
    phases = data["phases"].copy().astype(np.float64)
    freqs  = data["freqs"]                                   # [bins] Hz
    sr     = int(data["sample_rate"])
    window = int(data["window_size"])
    hop    = float(data["hop_ratio"])

    n_frames, n_bins = mags.shape
    print(f"  [edit-spectrum] {n_frames} frames × {n_bins} bins  "
          f"sr={sr}  window={window}  hop={hop}")

    # ── 1. highpass ──────────────────────────────────────────────────────────
    if args.highpass and args.highpass > 0:
        mask = freqs < args.highpass
        mags[:, mask] = 0.0
        print(f"  ✓  highpass  < {args.highpass} Hz  ({mask.sum()} bins zeroed)")

    # ── 2. lowpass ───────────────────────────────────────────────────────────
    if args.lowpass and args.lowpass > 0:
        mask = freqs > args.lowpass
        mags[:, mask] = 0.0
        print(f"  ✓  lowpass   > {args.lowpass} Hz  ({mask.sum()} bins zeroed)")

    # ── 3. isolate ───────────────────────────────────────────────────────────
    if args.isolate:
        margin  = args.isolate_margin
        keep    = np.zeros(n_bins, dtype=bool)
        for hz in args.isolate:
            keep |= (freqs >= hz - margin) & (freqs <= hz + margin)
        mags[:, ~keep] = 0.0
        n_kept = keep.sum()
        print(f"  ✓  isolate   {args.isolate} Hz ±{margin}Hz  ({n_kept}/{n_bins} bins mantenidos)")

    # ── 4. suppress ──────────────────────────────────────────────────────────
    if args.suppress:
        margin = args.suppress_margin
        for hz in args.suppress:
            mask = (freqs >= hz - margin) & (freqs <= hz + margin)
            mags[:, mask] *= args.suppress_gain
        print(f"  ✓  suppress  {args.suppress} Hz ±{margin}Hz  "
              f"(ganancia residual={args.suppress_gain})")

    # ── 5. gain ──────────────────────────────────────────────────────────────
    if args.gain and args.gain != 0.0:
        factor = 10 ** (args.gain / 20.0)
        mags  *= factor
        print(f"  ✓  gain      {args.gain:+.1f} dB  (×{factor:.3f})")

    # ── 6. denoise ───────────────────────────────────────────────────────────
    if args.denoise and args.denoise > 0.0:
        # Umbral sobre los bins con energía real (excluye los ya zeroed)
        nonzero = mags[mags > 0]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, args.denoise)
            zeroed    = ((mags > 0) & (mags < threshold)).sum()
            mags[mags < threshold] = 0.0
            print(f"  ✓  denoise   percentil={args.denoise:.1f} (sobre bins activos)  "
                  f"umbral={threshold:.6f}  ({zeroed} celdas zeroed)")
        else:
            print(f"  ⚠  denoise   sin bins activos, operación omitida")

    # ── 7. timestretch ───────────────────────────────────────────────────────
    if args.timestretch and args.timestretch != 1.0:
        ratio     = args.timestretch          # >1 = más lento, <1 = más rápido
        new_frames = max(1, int(n_frames * ratio))
        old_idx   = np.arange(n_frames, dtype=np.float64)
        new_idx   = np.linspace(0, n_frames - 1, new_frames)
        # Interpolar magnitudes y fases por separado
        new_mags   = np.zeros((new_frames, n_bins), dtype=np.float64)
        new_phases = np.zeros((new_frames, n_bins), dtype=np.float64)
        for b in range(n_bins):
            new_mags[:, b]   = np.interp(new_idx, old_idx, mags[:, b])
            # Interpolación de fase con unwrap para evitar discontinuidades
            uw = np.unwrap(phases[:, b])
            new_phases[:, b] = np.interp(new_idx, old_idx, uw)
        mags, phases = new_mags, new_phases
        print(f"  ✓  timestretch  ×{ratio:.2f}  {n_frames}→{new_frames} frames  "
              f"({new_frames * hop * window / sr:.2f}s)")

    # ── 8. pitchshift ────────────────────────────────────────────────────────
    if args.pitchshift and args.pitchshift != 0:
        semitones = args.pitchshift
        ratio     = 2 ** (semitones / 12.0)
        # Para cada bin destino b, coger el contenido del bin fuente b/ratio.
        # Equivalente a re-etiquetar freqs[b] → freqs[b]*ratio.
        new_mags   = np.zeros_like(mags)
        new_phases = np.zeros_like(phases)
        bin_indices = np.arange(n_bins, dtype=np.float64)
        for b in range(n_bins):
            # bin fuente (posición fraccionaria)
            src_b = b / ratio
            if src_b < 0 or src_b >= n_bins - 1:
                continue
            lo = int(src_b)
            hi = lo + 1
            t  = src_b - lo
            new_mags[:, b]   = mags[:, lo] * (1 - t) + mags[:, hi] * t
            uw_lo = np.unwrap(phases[:, lo])
            uw_hi = np.unwrap(phases[:, hi])
            new_phases[:, b] = uw_lo * (1 - t) + uw_hi * t
        mags, phases = new_mags, new_phases
        print(f"  ✓  pitchshift  {semitones:+d} semitones  (×{ratio:.4f})")

    # ── 9. reverse ───────────────────────────────────────────────────────────
    if args.reverse:
        mags   = mags[::-1].copy()
        phases = phases[::-1].copy()
        print(f"  ✓  reverse")

    # ── guardar NPZ editado ──────────────────────────────────────────────────
    out_npz = args.output or src.stem + "_edited.npz"
    np.savez_compressed(out_npz,
                        magnitudes=mags.astype(np.float32),
                        phases=phases.astype(np.float32),
                        freqs=freqs,
                        sample_rate=np.array(sr),
                        window_size=np.array(window),
                        hop_ratio=np.array(hop))
    n_frames_out = mags.shape[0]
    dur_out = n_frames_out * hop * window / sr
    print(f"  ✓  NPZ editado → {out_npz}  ({n_frames_out} frames, ~{dur_out:.2f}s)")

    # ── reconstruir WAV si se pide ───────────────────────────────────────────
    if args.wav:
        wav_path = args.wav
        audio    = stft_reconstruct(mags, phases, window, hop, sr)
        write_wav(wav_path, audio, sr)


def cmd_npz_to_png(args):
    """
    NPZ de espectrograma → PNG en escala de grises.

    Cada píxel representa la magnitud de un bin frecuencial en un frame temporal:
      - Eje X: tiempo (frame 0 → frame N-1, izquierda → derecha)
      - Eje Y: frecuencia (bin 0=DC arriba, Nyquist abajo) — o invertido con --flip-y
      - Valor: magnitud en escala lineal o logarítmica (dB), normalizada a [0, 255]

    La fase NO se guarda en el PNG. Al reconvertir PNG → NPZ se usa fase cero,
    lo que produce reconstrucciones coherentes en amplitud pero con artefactos de fase
    (similares a Griffin-Lim iteración 0). Para reconstrucciones de calidad, conserva
    siempre el NPZ original o usa edit-spectrum sobre él.

    El PNG guarda los metadatos del NPZ (sr, window, hop) en los campos de texto
    del chunk tEXt de PNG para que png-to-npz pueda recuperarlos automáticamente.
    """
    Image = _import_pil()

    data   = np.load(args.input)
    mags   = data["magnitudes"].astype(np.float64)   # [frames × bins]
    sr     = int(data["sample_rate"])
    window = int(data["window_size"])
    hop    = float(data["hop_ratio"])
    freqs  = data["freqs"]

    n_frames, n_bins = mags.shape
    print(f"  [npz-to-png] {n_frames} frames × {n_bins} bins  sr={sr}  window={window}")

    # ── escala ────────────────────────────────────────────────────────────────
    if args.log:
        db_floor = args.db_floor
        # Normalizar primero a [0,1] para que el máximo sea exactamente 0dB
        # y el rango dinámico útil caiga dentro de [db_floor, 0].
        # Sin normalizar, las magnitudes STFT absolutas tienen valores >1
        # que producen dB positivos → saturación masiva de pixels a 255.
        mags_norm = mags / (mags.max() + 1e-10)
        img_data  = 20 * np.log10(mags_norm + 1e-10)
        img_data  = np.clip(img_data, db_floor, 0.0)
        img_data  = (img_data - db_floor) / (-db_floor)   # → [0, 1]
        scale_desc = f"log (piso={db_floor}dB)"
    else:
        peak = mags.max()
        img_data = mags / (peak + 1e-10)                  # → [0, 1]
        scale_desc = "lineal"

    # ── orientación ───────────────────────────────────────────────────────────
    # img_data es [frames × bins]: transponer para que Y=bins, X=frames
    img_data = img_data.T   # → [bins × frames]
    if not args.no_flip_y:
        img_data = img_data[::-1, :]   # frecuencia 0 abajo, Nyquist arriba

    # Convertir a uint8
    pixels = (img_data * 255).clip(0, 255).astype(np.uint8)
    img    = Image.fromarray(pixels, mode="L")

    # ── metadatos en tEXt PNG ─────────────────────────────────────────────────
    from PIL import PngImagePlugin
    meta = PngImagePlugin.PngInfo()
    meta.add_text("audio_lab_sr",       str(sr))
    meta.add_text("audio_lab_window",   str(window))
    meta.add_text("audio_lab_hop",      str(hop))
    meta.add_text("audio_lab_scale",    "log" if args.log else "linear")
    meta.add_text("audio_lab_db_floor", str(args.db_floor))
    meta.add_text("audio_lab_flip_y",   "0" if args.no_flip_y else "1")
    meta.add_text("audio_lab_n_frames", str(n_frames))
    meta.add_text("audio_lab_n_bins",   str(n_bins))
    meta.add_text("audio_lab_mag_max",  str(float(mags.max())))

    out_path = args.output or Path(args.input).stem + ".png"

    # ── fallback: codificar metadatos críticos en el nombre del fichero ───────
    # GIMP y otros editores eliminan los chunks tEXt propios al exportar.
    # Si el nombre de salida no tiene sufijo _alXXXX, añadir un sidecar .json.
    stem = Path(out_path).stem
    sidecar = Path(out_path).with_suffix(".al.json")
    sidecar_data = {
        "audio_lab_sr":       sr,
        "audio_lab_window":   window,
        "audio_lab_hop":      hop,
        "audio_lab_scale":    "log" if args.log else "linear",
        "audio_lab_db_floor": float(args.db_floor),
        "audio_lab_flip_y":   not args.no_flip_y,
        "audio_lab_n_frames": n_frames,
        "audio_lab_n_bins":   n_bins,
        "audio_lab_mag_max":  float(mags.max()),
    }
    with open(sidecar, "w") as f:
        json.dump(sidecar_data, f, indent=2)

    img.save(out_path, format="PNG", pnginfo=meta)
    h, w = pixels.shape
    print(f"  ✓  PNG → {out_path}  ({w}×{h}px, escala {scale_desc})")
    print(f"     Metadatos embebidos: sr={sr} window={window} hop={hop}")
    print(f"     Sidecar JSON → {sidecar}  (fallback si el editor elimina metadatos tEXt)")


def cmd_png_to_npz(args):
    """
    PNG en escala de grises → NPZ de espectrograma.

    Lee los metadatos embebidos por npz-to-png (sr, window_size, hop_ratio, escala)
    para reconstruir el NPZ con magnitudes correctamente escaladas.
    Si el PNG no tiene metadatos de audio_lab, se usan los valores de --sr, --window
    y --hop que se pasen por línea de comandos (o los defaults).

    IMPORTANTE sobre la fase: el PNG solo contiene magnitudes. El NPZ resultante
    tendrá fase cero en todos los bins. Al reconstruir con 'reconstruct' se obtendrá
    audio coherente en amplitud pero con artefactos de fase típicos de magnitud-only.
    Para uso creativo (editar el PNG en un editor de imagen y volver a audio) esto
    es completamente intencional y útil; para preservación de audio, usa el NPZ.
    """
    Image = _import_pil()

    img = Image.open(args.input).convert("L")   # fuerza escala de grises
    pixels = np.array(img, dtype=np.float64)    # [H × W]

    # ── leer metadatos: tEXt del PNG, luego sidecar .al.json, luego args ─────
    meta = dict(img.info)   # chunks tEXt (GIMP los elimina al exportar)

    sidecar_path = Path(args.input).with_suffix(".al.json")
    if "audio_lab_sr" not in meta and sidecar_path.exists():
        with open(sidecar_path) as f:
            sc = json.load(f)
        # Convertir booleanos del sidecar al formato de tEXt (string)
        for k, v in sc.items():
            meta[k] = str(v) if not isinstance(v, bool) else ("1" if v else "0")
        print(f"  [png-to-npz] Metadatos leídos desde sidecar: {sidecar_path.name}")

    sr       = int(float(meta.get("audio_lab_sr",      args.sr)))
    window   = int(float(meta.get("audio_lab_window",  args.window)))
    hop      = float(meta.get("audio_lab_hop",         args.hop))
    scale    = meta.get("audio_lab_scale",             "linear")
    db_floor = float(meta.get("audio_lab_db_floor",    -80.0))
    flip_y   = str(meta.get("audio_lab_flip_y",        "1")) in ("1", "True", "true")
    mag_max  = float(meta.get("audio_lab_mag_max",     1.0))
    # --mag-max explícito sobreescribe siempre (útil cuando no hay metadatos ni sidecar)
    if hasattr(args, "mag_max") and args.mag_max is not None:
        mag_max = args.mag_max
        print(f"  [png-to-npz] mag_max forzado a {mag_max} (--mag-max)")

    n_bins_meta   = int(float(meta.get("audio_lab_n_bins",   0)))
    n_frames_meta = int(float(meta.get("audio_lab_n_frames", 0)))

    has_meta = "audio_lab_sr" in meta
    H, W     = pixels.shape   # altura × anchura de la imagen

    if has_meta:
        print(f"  [png-to-npz] Metadatos: sr={sr} window={window} hop={hop} "
              f"escala={scale} mag_max={mag_max:.2f}")
    else:
        print(f"  [png-to-npz] Sin metadatos ni sidecar — usando defaults/args")

    # ── determinar orientación (frames × bins) ────────────────────────────────
    # npz-to-png escribe la imagen como [bins(H) × frames(W)].
    # Algunos visores/editores la guardan transpuesta. Detectar usando el sidecar.
    if n_bins_meta and n_frames_meta:
        if H == n_frames_meta and W == n_bins_meta:
            # PNG rotado 90°: transponer para restaurar [bins(H) × frames(W)]
            pixels = pixels.T
            H, W   = pixels.shape
            print(f"  ⚠  PNG rotado detectado (H={H} coincide con n_frames={n_frames_meta}) "
                  f"— transpuesto automáticamente")
        # else: orientación correcta H=n_bins, W=n_frames

    print(f"  [png-to-npz] {W}×{H}px → {W} frames × {H} bins")

    # ── derivar window correcto desde n_bins (más robusto que el metadato) ───
    # rfftfreq(window) da window//2 + 1 bins. Invertir: window = (n_bins-1)*2
    window_from_bins = (H - 1) * 2
    if window_from_bins != window:
        print(f"  [png-to-npz] window ajustado: {window} → {window_from_bins} "
              f"(derivado de n_bins={H})")
        window = window_from_bins

    # ── deshacer orientación Y ────────────────────────────────────────────────
    if flip_y:
        pixels = pixels[::-1, :]   # volver a bin-0 arriba (DC en fila 0)

    # pixels es [bins(H) × frames(W)]; transponer a [frames × bins]
    img_data = pixels.T / 255.0   # → [frames × bins] ∈ [0, 1]

    # ── desescalar magnitudes ─────────────────────────────────────────────────
    mag_max = float(meta.get("audio_lab_mag_max", 1.0))

    if scale == "log":
        # [0,1] → [db_floor, 0] dB → lineal → escala original
        db    = img_data * (-db_floor) + db_floor
        mags  = 10 ** (db / 20.0)
        mags[pixels.T < 1] = 0.0    # píxeles negros (valor 0) → silencio exacto
        mags *= mag_max              # restaurar escala absoluta original
    else:
        # lineal: [0,1] → escala original
        mags = (img_data * mag_max).astype(np.float32)

    mags = mags.astype(np.float32)

    # ── fase cero ─────────────────────────────────────────────────────────────
    phases = np.zeros_like(mags)

    # ── frecuencias por bin ───────────────────────────────────────────────────
    n_bins = mags.shape[1]
    freqs  = np.fft.rfftfreq(window, 1.0 / sr)
    if len(freqs) != n_bins:
        # Caso genuinamente anómalo (resize manual del PNG que cambió n_bins)
        freqs = np.linspace(0, sr / 2, n_bins)
        print(f"  ⚠  n_bins imagen ({n_bins}) ≠ rfftfreq({window}) ({len(freqs)}): "
              f"frecuencias interpoladas linealmente")

    out_path = args.output or Path(args.input).stem + ".npz"
    np.savez_compressed(out_path,
                        magnitudes=mags,
                        phases=phases,
                        freqs=freqs.astype(np.float32),
                        sample_rate=np.array(sr),
                        window_size=np.array(window),
                        hop_ratio=np.array(hop))

    n_frames, n_bins_out = mags.shape
    dur = n_frames * hop * window / sr
    print(f"  ✓  NPZ → {out_path}  ({n_frames} frames × {n_bins_out} bins, ~{dur:.2f}s)")
    print(f"     Fase: cero (reconstrucción por magnitudes)")


def cmd_synth(args):
    """Síntesis aditiva de una o varias notas simultáneas o en secuencia."""
    timbre   = Timbre.from_file(args.timbre)
    envelope = EnvelopeADR.from_file(args.envelope)
    sr       = SAMPLE_RATE
    mode     = args.chord_mode   # "chord" | "sequence"

    notes_midi = []
    for n in args.notes:
        try:
            notes_midi.append(int(n))
        except ValueError:
            notes_midi.append(note_name_to_midi(n))

    N        = len(notes_midi)
    duration = args.duration
    GOLDEN   = 2.399963229728653

    # ── velocidades por nota ──────────────────────────────────────────────────
    if args.velocities:
        vels = [float(v) for v in args.velocities]
        if len(vels) < N:
            vels += [vels[-1]] * (N - len(vels))
    else:
        vels = [args.velocity] * N

    # ── detuning por nota (cents) ─────────────────────────────────────────────
    if args.detune and args.detune > 0:
        # Distribuye ±detune/2 cents entre las notas de forma determinista
        if N == 1:
            detune_offsets = [0.0]
        else:
            detune_offsets = [args.detune * (k / (N - 1) - 0.5) for k in range(N)]
    else:
        detune_offsets = [0.0] * N

    # ── función interna que sintetiza una nota con detuning ───────────────────
    def _synth_detuned(note: int, dur: float, vel: float,
                       detune_cents: float, phase_offset: float) -> np.ndarray:
        """Como synth_note pero aplica detuning en cents a la fundamental."""
        n_samples = int(dur * sr)
        t         = np.arange(n_samples, dtype=np.float64) / sr
        fund_hz   = midi_to_hz(note) * (2 ** (detune_cents / 1200))
        harmonics = timbre.get_harmonics(note)
        out       = np.zeros(n_samples, dtype=np.float64)
        n_active  = max(1, sum(1 for a in harmonics if a >= 1e-6))
        for i, amp in enumerate(harmonics):
            if amp < 1e-6:
                continue
            freq = fund_hz * (i + 1)
            if freq >= sr / 2:
                break
            env  = envelope.render(dur, sr, harmonic_idx=i)
            ph   = phase_offset + i * GOLDEN
            out += np.sin(2 * np.pi * freq * t + ph) * env * float(amp)
        out *= vel / n_active
        return out.astype(np.float32)

    # ── strum: retraso en ms por nota ─────────────────────────────────────────
    strum_ms = args.strum or 0.0   # ms entre nota y nota

    if mode == "chord":
        n_total = int((duration + strum_ms * N / 1000 + 0.1) * sr)
        out     = np.zeros(n_total, dtype=np.float32)
        for k, note in enumerate(notes_midi):
            delay_s = k * strum_ms / 1000
            chunk   = _synth_detuned(note, duration, vels[k],
                                     detune_offsets[k], k * GOLDEN)
            start = int(delay_s * sr)
            out[start : start + len(chunk)] += chunk

        peak = np.abs(out).max()
        if peak > 1e-6:
            out = (out / peak * max(vels) * 0.95).astype(np.float32)

    else:  # sequence
        gap     = args.gap
        chunks  = []
        silence = np.zeros(int(gap * sr), dtype=np.float32)
        for k, note in enumerate(notes_midi):
            chunk = _synth_detuned(note, duration, vels[k],
                                   detune_offsets[k], k * GOLDEN)
            peak = np.abs(chunk).max()
            if peak > 1e-6:
                chunk = (chunk / peak * vels[k] * 0.95).astype(np.float32)
            chunks.append(chunk)
            chunks.append(silence)
        out = np.concatenate(chunks)

    out_path = args.output or "synth_out.wav"
    write_wav(out_path, out, sr)
    note_names = [midi_to_note_name(n) for n in notes_midi]
    det_str    = f"  detuning={args.detune}¢" if args.detune else ""
    strum_str  = f"  strum={strum_ms}ms" if strum_ms else ""
    print(f"  [synth] {note_names}{det_str}{strum_str}")


def cmd_play_midi(args):
    """MIDI + instrument-map.json → WAV renderizado."""
    if not args.instrument_map:
        print("  [play-midi] Sin --instrument-map: usando síntesis sinusoidal simple")
        default_inst = ChannelInstrument(
            mode="additive",
            timbre=Timbre.from_file(None),
            envelope=EnvelopeADR(),
        )
        instrument_map = {-1: default_inst}
    else:
        print(f"  [play-midi] Cargando instrument map: {args.instrument_map}")
        instrument_map = load_instrument_map(args.instrument_map)

    bpm       = getattr(args, "bpm", None)
    transpose = getattr(args, "transpose", 0)
    audio     = render_midi(args.input, instrument_map, bpm, transpose)

    out_path = args.output or Path(args.input).stem + "_render.wav"
    write_wav(out_path, audio)


def cmd_roundtrip(args):
    """WAV → análisis → síntesis → WAV (mide pérdida del pipeline)."""
    audio, sr = read_wav(args.input)
    method    = args.method.lower()

    if method == "stft":
        window = args.window
        hop    = args.hop
        print(f"  [roundtrip] STFT window={window} hop={hop}")
        mags, phases, freqs = stft_analyse(audio, sr, window, hop)
        recon = stft_reconstruct(mags, phases, window, hop, sr)
    else:
        print(f"  [roundtrip] FFT → síntesis aditiva")
        bins  = fft_analyse(audio, sr)
        recon = synth_from_bins(bins, len(audio) / sr, sr)

    # Pad/truncar a la misma longitud
    n = min(len(audio), len(recon))
    err     = audio[:n] - recon[:n]
    rms_in  = float(np.sqrt(np.mean(audio[:n] ** 2)))
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    snr     = 20 * math.log10(rms_in / (rms_err + 1e-12))
    print(f"  [roundtrip] SNR estimado: {snr:.1f} dB")

    out_path = args.output or Path(args.input).stem + "_roundtrip.wav"
    write_wav(out_path, recon, sr)


def cmd_info(args):
    """Diagnóstico rápido de un WAV o MIDI."""
    path = Path(args.input)
    ext  = path.suffix.lower()

    if ext in (".mid", ".midi"):
        mido = _import_mido()
        mid  = mido.MidiFile(str(path))
        n_notes, n_tracks = 0, len(mid.tracks)
        tempos, programs  = set(), set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    n_notes += 1
                elif msg.type == "set_tempo":
                    tempos.add(round(60_000_000 / msg.tempo, 1))
                elif msg.type == "program_change":
                    programs.add(msg.program)
        print(f"\n  MIDI: {path.name}")
        print(f"  Tipo         : {mid.type}")
        print(f"  Pistas       : {n_tracks}")
        print(f"  Ticks/beat   : {mid.ticks_per_beat}")
        print(f"  Duración     : {mid.length:.2f}s")
        print(f"  Notas        : {n_notes}")
        print(f"  BPMs         : {tempos or {'120.0 (default)'}}")
        print(f"  Programas GM : {programs or 'N/A'}")
    elif ext == ".wav":
        audio, sr = read_wav(str(path))
        rms  = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        print(f"\n  WAV: {path.name}")
        print(f"  Duración     : {len(audio)/sr:.3f}s")
        print(f"  Sample rate  : {sr}Hz")
        print(f"  Muestras     : {len(audio)}")
        print(f"  RMS          : {rms:.5f}  ({20*math.log10(rms+1e-12):.1f} dBFS)")
        print(f"  Peak         : {peak:.5f}  ({20*math.log10(peak+1e-12):.1f} dBFS)")
        print(f"  Clipping     : {'⚠ SÍ' if peak > 0.999 else '✓ no'}")
    elif ext == ".npz":
        data = np.load(str(path))
        print(f"\n  NPZ (STFT): {path.name}")
        for k in data.files:
            v = data[k]
            print(f"  {k:<16}: shape={v.shape}  dtype={v.dtype}")
    elif ext == ".json":
        with open(str(path)) as f:
            d = json.load(f)
        print(f"\n  JSON: {path.name}")
        for k, v in d.items():
            if isinstance(v, list):
                print(f"  {k:<16}: [{len(v)} elementos]")
            else:
                print(f"  {k:<16}: {v}")
    else:
        print(f"  ✗  Formato no reconocido: {ext}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="audio_lab",
        description="Síntesis aditiva, análisis DSP y renderizado MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyse ──────────────────────────────────────────────────────────────
    p = sub.add_parser("analyse", help="WAV/MIDI → espectro JSON")
    p.add_argument("input",  help="Fichero WAV o MIDI")
    p.add_argument("--method", default="fft", choices=["fft","dft"],
                   help="Método de análisis (default: fft)")
    p.add_argument("--threshold", type=float, default=0.001,
                   help="Amplitud mínima para incluir un bin (default: 0.001)")
    p.add_argument("--top-n", type=int, default=None,
                   help="Mantener solo los N bins de mayor amplitud")
    p.add_argument("--max-bins", type=int, default=2048,
                   help="Muestras máximas para DFT (default: 2048)")
    p.add_argument("--output", "-o", default=None, help="Fichero JSON de salida")
    p.set_defaults(func=cmd_analyse)

    # ── spectrogram ──────────────────────────────────────────────────────────
    p = sub.add_parser("spectrogram", help="WAV → STFT → .npz [+ PNG]")
    p.add_argument("input",  help="Fichero WAV")
    p.add_argument("--window", type=int, default=2048,
                   help="Tamaño de ventana STFT en muestras (default: 2048)")
    p.add_argument("--hop", type=float, default=0.25,
                   help="Hop ratio ∈ (0,1] (default: 0.25)")
    p.add_argument("--plot", action="store_true",
                   help="Genera también imagen PNG (requiere matplotlib+scipy)")
    p.add_argument("--output", "-o", default=None, help="Fichero NPZ de salida")
    p.set_defaults(func=cmd_spectrogram)

    # ── reconstruct ──────────────────────────────────────────────────────────
    p = sub.add_parser("reconstruct", help="NPZ / JSON espectro → WAV")
    p.add_argument("input",  help="Fichero .npz (STFT) o .json (espectro FFT/DFT)")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Duración en segundos (solo para JSON, default: 2.0)")
    p.add_argument("--normalize", action="store_true",
                   help="Normalizar el WAV de salida a 0.85 peak (automático para NPZ con fase cero)")
    p.add_argument("--gl-iters", type=int, default=32,
                   help="Iteraciones de Griffin-Lim para NPZ con fase cero (default: 32)")
    p.add_argument("--output", "-o", default=None, help="WAV de salida")
    p.set_defaults(func=cmd_reconstruct)

    # ── npz-to-png ───────────────────────────────────────────────────────────
    p = sub.add_parser("npz-to-png",
                       help="Espectrograma NPZ → PNG escala de grises")
    p.add_argument("input", help="Fichero .npz (de spectrogram o edit-spectrum)")
    p.add_argument("--log", action="store_true",
                   help="Escala logarítmica en dB en vez de lineal (más perceptivo)")
    p.add_argument("--db-floor", type=float, default=-80.0,
                   help="Piso en dB para escala log (default: -80)")
    p.add_argument("--no-flip-y", action="store_true",
                   help="No invertir eje Y (bin 0=DC queda arriba; por defecto va abajo)")
    p.add_argument("--output", "-o", default=None, help="Fichero PNG de salida")
    p.set_defaults(func=cmd_npz_to_png)

    # ── png-to-npz ───────────────────────────────────────────────────────────
    p = sub.add_parser("png-to-npz",
                       help="PNG escala de grises → espectrograma NPZ (fase cero)")
    p.add_argument("input", help="Fichero PNG (idealmente generado con npz-to-png)")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help=f"Sample rate si el PNG no tiene metadatos (default: {SAMPLE_RATE})")
    p.add_argument("--window", type=int, default=2048,
                   help="Tamaño de ventana STFT si el PNG no tiene metadatos (default: 2048)")
    p.add_argument("--hop", type=float, default=0.25,
                   help="Hop ratio si el PNG no tiene metadatos (default: 0.25)")
    p.add_argument("--mag-max", type=float, default=None,
                   help="Factor de escala de magnitudes (recupera nivel original). "
                        "Valor típico: 50-200. Si no se pasa, reconstruct normaliza automáticamente.")
    p.add_argument("--output", "-o", default=None, help="Fichero NPZ de salida")
    p.set_defaults(func=cmd_png_to_npz)

    # ── edit-spectrum ─────────────────────────────────────────────────────────
    p = sub.add_parser("edit-spectrum",
                       help="Edita un NPZ de espectrograma → NPZ [+ WAV]",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=
                       "Operaciones (se aplican en orden):\n"
                       "  highpass/lowpass  filtro frecuencial\n"
                       "  isolate           dejar solo bandas concretas\n"
                       "  suppress          atenuar bandas concretas\n"
                       "  gain              ganancia global en dB\n"
                       "  denoise           zeroing por umbral de percentil\n"
                       "  timestretch       estirar tiempo sin cambiar pitch\n"
                       "  pitchshift        transponer sin cambiar duración\n"
                       "  reverse           invertir en el tiempo")
    p.add_argument("input", help="Fichero .npz de entrada (de spectrogram)")
    p.add_argument("--highpass", type=float, default=None,
                   help="Frecuencia de corte paso-alto en Hz (elimina < valor)")
    p.add_argument("--lowpass", type=float, default=None,
                   help="Frecuencia de corte paso-bajo en Hz (elimina > valor)")
    p.add_argument("--isolate", nargs="+", type=float, default=None,
                   help="Mantener solo estas frecuencias Hz (ej: 261.6 329.6 392)")
    p.add_argument("--isolate-margin", type=float, default=30.0,
                   help="Margen en Hz a cada lado de --isolate (default: 30)")
    p.add_argument("--suppress", nargs="+", type=float, default=None,
                   help="Atenuar estas frecuencias Hz (ej: 50 100 150)")
    p.add_argument("--suppress-margin", type=float, default=20.0,
                   help="Margen en Hz a cada lado de --suppress (default: 20)")
    p.add_argument("--suppress-gain", type=float, default=0.0,
                   help="Ganancia residual en zona suprimida ∈ [0,1] (default: 0)")
    p.add_argument("--gain", type=float, default=None,
                   help="Ganancia global en dB aplicada a magnitudes (ej: +6, -3)")
    p.add_argument("--denoise", type=float, default=None,
                   help="Zeroing de bins bajo el percentil N de magnitud (ej: 60)")
    p.add_argument("--timestretch", type=float, default=None,
                   help="Factor de estiramiento temporal >1=más lento (ej: 1.5, 0.75)")
    p.add_argument("--pitchshift", type=int, default=None,
                   help="Transposición en semitones enteros (ej: +3, -5)")
    p.add_argument("--reverse", action="store_true",
                   help="Invertir el espectrograma en el tiempo")
    p.add_argument("--output", "-o", default=None,
                   help="NPZ de salida (default: <input>_edited.npz)")
    p.add_argument("--wav", default=None,
                   help="Si se especifica, reconstruye también un WAV con este nombre")
    p.set_defaults(func=cmd_edit_spectrum)

    # ── synth ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("synth", help="Síntesis aditiva de notas → WAV")
    p.add_argument("--notes", nargs="+", required=True,
                   help="Notas: nombre (C4, F#3) o número MIDI (60)")
    p.add_argument("--timbre", default=None,
                   help="JSON de timbre (default: sinusoide pura)")
    p.add_argument("--envelope", default=None,
                   help="JSON de envolvente ADR (default: built-in)")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Duración de cada nota en segundos (default: 2.0)")
    p.add_argument("--velocity", type=float, default=0.8,
                   help="Velocidad/dinámica global ∈ [0,1] (default: 0.8)")
    p.add_argument("--velocities", nargs="+", type=float, default=None,
                   help="Velocidad por nota (ej: 0.7 0.6 0.9). Sobreescribe --velocity")
    p.add_argument("--detune", type=float, default=0.0,
                   help="Detuning total en cents distribuido entre notas (ej: 8.0). "
                        "Separa las notas del acorde ±detune/2 ¢ para evitar fusión tímbrica")
    p.add_argument("--strum", type=float, default=0.0,
                   help="Retraso en ms entre notas del acorde (ej: 20). "
                        "0=simultáneo, >0=arpeggiato. Solo en modo chord")
    p.add_argument("--chord-mode", default="chord",
                   choices=["chord","sequence"],
                   help="chord=simultáneo/strum, sequence=en secuencia (default: chord)")
    p.add_argument("--gap", type=float, default=0.1,
                   help="Silencio entre notas en modo sequence (default: 0.1)")
    p.add_argument("--output", "-o", default="synth_out.wav")
    p.set_defaults(func=cmd_synth)

    # ── play-midi ─────────────────────────────────────────────────────────────
    p = sub.add_parser("play-midi", help="MIDI + instrument map → WAV")
    p.add_argument("input", help="Fichero MIDI (.mid)")
    p.add_argument("--instrument-map", default=None,
                   help="JSON de asignación de instrumentos por canal")
    p.add_argument("--bpm", type=float, default=None,
                   help="Override de tempo en BPM")
    p.add_argument("--transpose", type=int, default=0,
                   help="Transposición en semitones (default: 0)")
    p.add_argument("--output", "-o", default=None, help="WAV de salida")
    p.set_defaults(func=cmd_play_midi)

    # ── roundtrip ─────────────────────────────────────────────────────────────
    p = sub.add_parser("roundtrip", help="WAV → análisis → síntesis → WAV")
    p.add_argument("input", help="Fichero WAV de entrada")
    p.add_argument("--method", default="stft", choices=["stft","fft"],
                   help="Pipeline de análisis/síntesis (default: stft)")
    p.add_argument("--window", type=int, default=2048,
                   help="Tamaño de ventana STFT (default: 2048)")
    p.add_argument("--hop", type=float, default=0.25,
                   help="Hop ratio STFT (default: 0.25)")
    p.add_argument("--output", "-o", default=None, help="WAV de salida")
    p.set_defaults(func=cmd_roundtrip)

    # ── info ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("info", help="Diagnóstico rápido de WAV / MIDI / NPZ / JSON")
    p.add_argument("input", help="Fichero a inspeccionar")
    p.set_defaults(func=cmd_info)

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    print(f"\n● audio_lab  →  {args.command}")
    args.func(args)
    print()


if __name__ == "__main__":
    main()
