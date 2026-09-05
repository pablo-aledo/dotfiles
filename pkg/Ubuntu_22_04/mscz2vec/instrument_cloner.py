#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        INSTRUMENT_CLONER  v1.0                               ║
║  Clonación neuronal de timbres instrumentales a partir de ~16s de audio      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Puerto autocontenido, en un único fichero, de "Neural Instrument          ║
║    Cloning" (Jonason/Erlj, DAFx 2022 · magenta/ddsp). Reimplementa la idea   ║
║    original en PyTorch puro -- sin depender de tensorflow ni de la          ║
║    librería ddsp -- conservando la arquitectura central: un decoder GRU     ║
║    COMPARTIDO entre todos los instrumentos de entrenamiento, más un         ║
║    embedding "z" pequeño y libre POR INSTRUMENTO. Clonar un instrumento     ║
║    nuevo no reentrena la red: solo se ajustan sus pocos parámetros libres   ║
║    (z + respuesta de reverb + mezcla wet/dry) sobre unos segundos de        ║
║    audio, con el decoder compartido congelado.                              ║
║                                                                              ║
║  ARQUITECTURA                                                               ║
║    f0 (Hz) + loudness (dB) + z[instrumento] --(GRU + MLPs)--> distribución  ║
║    armónica + amplitud + magnitudes de ruido --(síntesis armónico+ruido,    ║
║    estilo DDSP)--> audio seco --(reverb con IR aprendida por instrumento,   ║
║    mezcla wet/dry aprendida)--> audio final. Extracción de f0 y loudness    ║
║    100% propia (YIN + ponderación A), sin CREPE ni librosa.                 ║
║                                                                              ║
║  SUBCOMANDOS                                                                ║
║    prepare-data   carpeta de audio (una subcarpeta por instrumento) ->      ║
║                    dataset de ventanas con f0/loudness ya extraídos         ║
║    train           dataset -> entrena el decoder COMPARTIDO + un banco de   ║
║                    z/reverb/gain por cada instrumento del dataset           ║
║    clone           wav objetivo (10-20s) + checkpoint compartido -> voz     ║
║                    clonada (decoder congelado, solo se ajusta esa voz)      ║
║    synthesize       checkpoint + voz clonada + (wav de control | nota fija) ║
║                    -> audio renderizado con ese timbre                      ║
║    info             inspecciona un dataset / checkpoint / voz clonada       ║
║    list-instruments lista los instrumentos de entrenamiento de un checkpoint║
║                                                                              ║
║  USO                                                                        ║
║    instrument_cloner.py prepare-data data/instrumentos --out dataset.pt     ║
║    instrument_cloner.py train dataset.pt --out shared.pt --epochs 60        ║
║    instrument_cloner.py clone violin_16s.wav --checkpoint shared.pt \\      ║
║                          --out voces/violin.pt --name "mi violín"           ║
║    instrument_cloner.py synthesize --checkpoint shared.pt \\                ║
║                          --voice voces/violin.pt --control-wav melodia.wav \\║
║                          --out violin_clonado.wav                           ║
║    instrument_cloner.py synthesize --checkpoint shared.pt \\                ║
║                          --voice voces/violin.pt --note 69 --duration 2.0 \\ ║
║                          --out la_central.wav                               ║
║    instrument_cloner.py info shared.pt                                      ║
║    instrument_cloner.py list-instruments shared.pt                          ║
║                                                                              ║
║  FORMATOS (todos son diccionarios guardados con torch.save)                 ║
║    dataset.pt   ventanas de audio + f0_hz/f0_confidence/loudness_db ya      ║
║                 extraídas, separadas en train/val/test, más el              ║
║                 "fingerprint" de los parámetros de extracción usados.       ║
║    shared.pt    pesos del decoder compartido + banco de z/ir/gain de cada   ║
║                 instrumento visto en entrenamiento + hiperparámetros +      ║
║                 "fingerprint" de arquitectura (para detectar checkpoints    ║
║                 incompatibles con una voz clonada o viceversa).             ║
║    voice.pt     z + ir + dry_gain + wet_gain de UN instrumento clonado,     ║
║                 más el fingerprint del checkpoint compartido con el que se  ║
║                 clonó -- `synthesize` verifica que coincidan antes de usar  ║
║                 la voz sobre un checkpoint distinto.                        ║
║                                                                              ║
║  DEPENDENCIAS  numpy  scipy  soundfile  torch (CPU, no requiere GPU)        ║
║                                                                              ║
║  LIMITACIONES                                                               ║
║    · No es el código del paper: la extracción de f0/loudness (YIN propio   ║
║      en vez de CRePE) y el ruido filtrado (conformado en dominio STFT en   ║
║      vez del banco de filtros FIR de ddsp) son reimplementaciones          ║
║      funcionalmente equivalentes, no bit-exactas.                          ║
║    · `prepare-data` carga TODAS las ventanas en un único fichero en RAM;   ║
║      para datasets grandes (NSynth completo, AIR completo) esto puede      ║
║      agotar la memoria -- pensado para colecciones de tamaño moderado, no  ║
║      para reproducir el dataset íntegro del paper original.                ║
║    · `clone` funciona mejor entre ~10 y ~30s de audio limpio y             ║
║      monofónico de un único instrumento; con menos de ~5s el ajuste de z   ║
║      tiende a sobreajustar el timbre a muy pocas notas.                    ║
║    · El decoder compartido necesita haberse entrenado sobre varios         ║
║      instrumentos variados para que clonar generalice bien -- clonar       ║
║      sobre un checkpoint entrenado con 1-2 instrumentos apenas mejora un   ║
║      ajuste desde cero.                                                    ║
║                                                                              ║
║  Módulo importable:                                                        ║
║    from instrument_cloner import (SynthModel, train, clone, synthesize,    ║
║        prepare_dataset, estimate_f0_contour, estimate_loudness_contour,    ║
║        multiscale_spectral_loss, harmonic_synth, filtered_noise_synth)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np


# =============================================================================
# §0  imports perezosos (torch/soundfile/scipy solo se cargan si hacen falta)
# =============================================================================

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        sys.exit("torch no encontrado. Instala con: pip install torch --break-system-packages")


def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("soundfile no encontrado. Instala con: pip install soundfile --break-system-packages")


def _import_scipy_signal():
    try:
        from scipy import signal
        return signal
    except ImportError:
        sys.exit("scipy no encontrado. Instala con: pip install scipy --break-system-packages")


# =============================================================================
# §1  constantes / hiperparámetros por defecto
# =============================================================================

FORMAT_DATASET = "instrument_cloner.dataset.v1"
FORMAT_CHECKPOINT = "instrument_cloner.checkpoint.v1"
FORMAT_VOICE = "instrument_cloner.voice.v1"

DEFAULTS = dict(
    sample_rate=16000,
    frame_rate=250,          # frames/seg -> hop = sample_rate // frame_rate muestras
    window_s=4.0,            # duración de cada ventana de entrenamiento
    z_size=16,
    hidden=128,
    n_harmonics=60,
    n_noise_fft=256,         # -> n_noise_mags = n_noise_fft//2 + 1 = 129
    ir_seconds=1.0,          # duración de la respuesta al impulso de reverb
    bidirectional=False,
    use_f0_confidence=True,
    f0_min_hz=50.0,
    f0_max_hz=2000.0,
    loud_min_db=-80.0,
    loud_max_db=20.0,
    yin_threshold=0.15,
)

# claves de DEFAULTS que definen si un checkpoint/voz/dataset son compatibles
# entre sí (ver _fingerprint). Cambiar cualquiera de estas invalida checkpoints
# y voces clonadas anteriores.
_FINGERPRINT_KEYS = (
    "sample_rate", "frame_rate", "window_s", "z_size", "hidden",
    "n_harmonics", "n_noise_fft", "ir_seconds", "bidirectional",
    "use_f0_confidence", "f0_min_hz", "f0_max_hz", "loud_min_db", "loud_max_db",
)


# =============================================================================
# §2  DSP: extracción de f0 y loudness framewise (autocontenido)
#
#     Basado en el YIN de genopatch_v2.py (De Cheveigné & Kawahara 2002),
#     pero llamado una vez por frame (hop = sample_rate/frame_rate) en vez
#     de una sola vez por clip -- necesitamos un CONTORNO, no una nota.
# =============================================================================

def _yin_frame(seg: np.ndarray, sr: int, fmin: float, fmax: float,
                threshold: float) -> Tuple[float, float]:
    """Un único frame de YIN. Devuelve (f0_hz, confianza 0-1); f0_hz=0.0 si
    no hay periodicidad clara (silencio, ruido, transitorio)."""
    seg = seg.astype(np.float64)
    seg = seg - seg.mean()
    if seg.size < 8 or np.max(np.abs(seg)) < 1e-9:
        return 0.0, 0.0
    n = len(seg)
    lag_min = max(1, int(sr / fmax))
    lag_max = min(int(sr / fmin), n - 1)
    if lag_max <= lag_min:
        return 0.0, 0.0

    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    spec = np.fft.rfft(seg, n=n_fft)
    autocorr = np.fft.irfft(spec * np.conj(spec))[:n]

    energy = seg * seg
    cum_energy = np.concatenate(([0.0], np.cumsum(energy)))
    taus = np.arange(0, lag_max + 1)
    total_energy = cum_energy[n]
    energy_head = cum_energy[np.clip(n - taus, 0, n)]
    energy_tail = total_energy - cum_energy[np.clip(taus, 0, n)]
    d = energy_head + energy_tail - 2.0 * autocorr[:lag_max + 1]
    d = np.maximum(d, 0.0)

    cmndf = np.ones_like(d)
    running_sum = 0.0
    for tau in range(1, lag_max + 1):
        running_sum += d[tau]
        cmndf[tau] = d[tau] * tau / running_sum if running_sum > 0 else 1.0

    window = cmndf[lag_min:lag_max + 1]
    below = np.where(window < threshold)[0]
    if len(below) > 0:
        idx = below[0]
        while idx + 1 < len(window) and window[idx + 1] < window[idx]:
            idx += 1
    else:
        idx = int(np.argmin(window))

    lag = lag_min + idx
    confidence = float(max(0.0, min(1.0, 1.0 - window[idx])))
    if confidence <= 0.0 or lag <= 0:
        return 0.0, 0.0
    return sr / lag, confidence


def estimate_f0_contour(audio: np.ndarray, sr: int, frame_rate: int,
                         fmin: float = 50.0, fmax: float = 2000.0,
                         threshold: float = 0.15,
                         win_periods: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Contorno de f0 (Hz) y confianza (0-1), un valor cada 1/frame_rate
    segundos, corriendo YIN en una ventana centrada en cada frame. La
    ventana cubre `win_periods` periodos del fmin dado (por defecto 3),
    para tener suficiente contexto incluso en registros graves."""
    hop = sr // frame_rate
    n_frames = len(audio) // hop
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    win = max(int(win_periods * sr / fmin), hop * 2)
    win = min(win, max(len(audio), hop * 2))
    half = win // 2
    padded = np.pad(audio, (half, half), mode="constant")

    f0 = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        center = i * hop + half
        start = center - half
        seg = padded[start:start + win]
        freq, c = _yin_frame(seg, sr, fmin, fmax, threshold)
        f0[i] = freq
        conf[i] = c
    return f0, conf


def _a_weighting_db(freqs: np.ndarray) -> np.ndarray:
    """Curva de ponderación A estándar (IEC 61672), en dB."""
    f2 = freqs.astype(np.float64) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = (12194.0 ** 2 * f2 ** 2) / (
            (f2 + 20.6 ** 2)
            * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
            * (f2 + 12194.0 ** 2)
            + 1e-20
        )
    a_db = 20 * np.log10(np.maximum(ra, 1e-20)) + 2.00
    a_db[freqs <= 0] = -100.0
    return a_db


def estimate_loudness_contour(audio: np.ndarray, sr: int, frame_rate: int,
                               n_fft: int = 1024) -> np.ndarray:
    """Loudness perceptual aproximada: potencia espectral ponderada-A, en
    dB, una vez por frame. Análogo al preprocesador de loudness de ddsp,
    sin depender de él."""
    hop = sr // frame_rate
    n_frames = len(audio) // hop
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    a_weight_lin = 10 ** (_a_weighting_db(freqs) / 10.0)
    half = n_fft // 2
    padded = np.pad(audio, (half, half), mode="constant")

    loud = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        center = i * hop + half
        seg = padded[center - half:center - half + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.fft.rfft(seg * window)
        power = (np.abs(spec) ** 2) * a_weight_lin
        total = power.sum() / n_fft
        loud[i] = 10 * np.log10(total + 1e-10)
    return loud


def load_audio_mono(path: str, sample_rate: int) -> np.ndarray:
    """Carga un wav/flac/ogg cualquiera, lo pasa a mono y lo resamplea a
    `sample_rate` si hace falta (resample_poly, sin librosa)."""
    sf = _import_soundfile()
    audio, sr_in = sf.read(str(path), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)  # mono
    if sr_in != sample_rate:
        signal = _import_scipy_signal()
        g = np.gcd(sr_in, sample_rate)
        audio = signal.resample_poly(audio, sample_rate // g, sr_in // g).astype(np.float32)
    return audio


def extract_features(audio: np.ndarray, sample_rate: int, frame_rate: int,
                      fmin: float, fmax: float, threshold: float) -> Dict[str, np.ndarray]:
    """f0_hz/f0_confidence/loudness_db alineados al mismo número de frames,
    más el propio audio recortado a un múltiplo exacto del hop."""
    hop = sample_rate // frame_rate
    f0_hz, f0_conf = estimate_f0_contour(audio, sample_rate, frame_rate, fmin, fmax, threshold)
    loudness_db = estimate_loudness_contour(audio, sample_rate, frame_rate)
    n_frames = min(len(f0_hz), len(loudness_db))
    audio = audio[:n_frames * hop]
    return dict(
        audio=audio.astype(np.float32),
        f0_hz=f0_hz[:n_frames].astype(np.float32),
        f0_confidence=f0_conf[:n_frames].astype(np.float32),
        loudness_db=loudness_db[:n_frames].astype(np.float32),
    )


# =============================================================================
# §3  ventaneo y preparación del dataset
# =============================================================================

def _fingerprint(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Subconjunto de hiperparámetros que debe coincidir para que un
    dataset/checkpoint/voz sean compatibles entre sí. Se guarda tal cual
    (no como hash) para que un mismatch se pueda imprimir y entender."""
    return {k: hparams[k] for k in _FINGERPRINT_KEYS}


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _window_file(path: Path, hparams: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Extrae f0/loudness/audio de un fichero completo y lo trocea en
    ventanas de `window_s` segundos con solape `hop_s`. Ventanas con
    confianza de f0 media muy baja (silencio/ruido) se descartan."""
    sr = hparams["sample_rate"]
    frame_rate = hparams["frame_rate"]
    hop_samples = sr // frame_rate
    win_frames = int(round(hparams["window_s"] * frame_rate))
    win_samples = win_frames * hop_samples

    try:
        audio = load_audio_mono(path, sr)
    except Exception as e:
        print(f"  ! no se pudo leer {path}: {e}")
        return None
    if len(audio) < win_samples:
        return None

    feats = extract_features(audio, sr, frame_rate, hparams["f0_min_hz"],
                              hparams["f0_max_hz"], hparams["yin_threshold"])
    n_frames_total = len(feats["f0_hz"])
    if n_frames_total < win_frames:
        return None

    return dict(audio=feats["audio"], f0_hz=feats["f0_hz"],
                f0_confidence=feats["f0_confidence"], loudness_db=feats["loudness_db"],
                hop_samples=hop_samples, win_frames=win_frames, win_samples=win_samples)


def _slice_windows(feats: Dict[str, np.ndarray], hop_frames: int, min_confidence: float
                    ) -> List[Dict[str, np.ndarray]]:
    win_frames = feats["win_frames"]
    win_samples = feats["win_samples"]
    hop_samples = feats["hop_samples"]
    n_frames_total = len(feats["f0_hz"])
    out = []
    start_frame = 0
    while start_frame + win_frames <= n_frames_total:
        f0 = feats["f0_hz"][start_frame:start_frame + win_frames]
        conf = feats["f0_confidence"][start_frame:start_frame + win_frames]
        loud = feats["loudness_db"][start_frame:start_frame + win_frames]
        a0 = start_frame * hop_samples
        audio = feats["audio"][a0:a0 + win_samples]
        if conf.mean() >= min_confidence and len(audio) == win_samples:
            out.append(dict(audio=audio, f0_hz=f0, f0_confidence=conf, loudness_db=loud))
        start_frame += hop_frames
    return out


def prepare_dataset(input_dir: str, out_path: str, sample_rate: int = DEFAULTS["sample_rate"],
                     frame_rate: int = DEFAULTS["frame_rate"], window_s: float = DEFAULTS["window_s"],
                     hop_s: float = 1.0, val_fraction: float = 0.1, test_fraction: float = 0.1,
                     fmin: float = DEFAULTS["f0_min_hz"], fmax: float = DEFAULTS["f0_max_hz"],
                     yin_threshold: float = DEFAULTS["yin_threshold"], min_confidence: float = 0.5,
                     seed: int = 0, verbose: bool = True) -> None:
    """input_dir/<instrumento>/*.wav (una subcarpeta por instrumento) ->
    dataset de ventanas train/val/test en `out_path`. Los ficheros (no las
    ventanas) se reparten completos entre splits, para que val/test midan
    generalización a grabaciones nuevas y no solo a otro trozo del mismo
    fichero."""
    torch, nn, F = _import_torch()
    rng = np.random.RandomState(seed)

    root = Path(input_dir)
    instrument_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not instrument_dirs:
        sys.exit(f"no se encontraron subcarpetas de instrumento en {input_dir}")

    hparams = dict(DEFAULTS)
    hparams.update(sample_rate=sample_rate, frame_rate=frame_rate, window_s=window_s,
                    f0_min_hz=fmin, f0_max_hz=fmax, yin_threshold=yin_threshold)
    hop_frames_train = max(1, int(round(hop_s * frame_rate)))
    hop_frames_eval = int(round(window_s * frame_rate))  # sin solape en val/test

    instrument_names = [d.name for d in instrument_dirs]
    splits: Dict[str, List[Dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    splits_idx: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    for inst_idx, inst_dir in enumerate(instrument_dirs):
        wavs = sorted([p for p in inst_dir.iterdir()
                       if p.suffix.lower() in (".wav", ".flac", ".ogg", ".aiff", ".aif")])
        if not wavs:
            print(f"  ! {inst_dir.name}: sin audio, se omite")
            continue
        order = rng.permutation(len(wavs))
        n_val = max(1, int(len(wavs) * val_fraction)) if len(wavs) > 2 else 0
        n_test = max(1, int(len(wavs) * test_fraction)) if len(wavs) > 3 else 0
        file_split = ["train"] * len(wavs)
        for i in order[:n_val]:
            file_split[i] = "val"
        for i in order[n_val:n_val + n_test]:
            file_split[i] = "test"

        n_windows = 0
        for wav_path, split in zip(wavs, file_split):
            feats = _window_file(wav_path, hparams)
            if feats is None:
                continue
            hop_frames = hop_frames_train if split == "train" else hop_frames_eval
            windows = _slice_windows(feats, hop_frames, min_confidence)
            splits[split].extend(windows)
            splits_idx[split].extend([inst_idx] * len(windows))
            n_windows += len(windows)
        if verbose:
            print(f"  {inst_dir.name:20s}  {len(wavs)} ficheros -> {n_windows} ventanas")

    out = dict(format=FORMAT_DATASET, hparams=hparams, fingerprint=_fingerprint(hparams),
               instrument_names=instrument_names)
    for split in ("train", "val", "test"):
        rows = splits[split]
        if not rows:
            out[split] = None
            continue
        out[split] = dict(
            audio=torch.tensor(np.stack([r["audio"] for r in rows])),
            f0_hz=torch.tensor(np.stack([r["f0_hz"] for r in rows])),
            f0_confidence=torch.tensor(np.stack([r["f0_confidence"] for r in rows])),
            loudness_db=torch.tensor(np.stack([r["loudness_db"] for r in rows])),
            instrument_idx=torch.tensor(np.array(splits_idx[split], dtype=np.int64)),
        )
    torch.save(out, out_path)
    n_train = 0 if out["train"] is None else len(out["train"]["instrument_idx"])
    n_val = 0 if out["val"] is None else len(out["val"]["instrument_idx"])
    n_test = 0 if out["test"] is None else len(out["test"]["instrument_idx"])
    print(f"  OK {len(instrument_names)} instrumento(s), "
          f"{n_train} train / {n_val} val / {n_test} test ventanas -> {out_path}")


# =============================================================================
# §4  síntesis diferenciable (armónico + ruido filtrado + reverb)
#
#     Reimplementación funcional del paradigma harmonic-plus-noise de ddsp,
#     sin usar ddsp: todo en tensores de torch, todo diferenciable.
# =============================================================================

def _exp_sigmoid(x, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7):
    """La no-linealidad estándar de ddsp para forzar salidas positivas y
    acotadas sin matar el gradiente cerca de cero (sigmoid puro satura)."""
    torch, nn, F = _import_torch()
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold


def hz_to_unit(f0_hz, f0_min_hz: float, f0_max_hz: float):
    """f0 en Hz -> escala ~[0,1.2] vía nota MIDI/127, recortada al rango de
    entrenamiento. No es más que una normalización de entrada a la red."""
    torch, nn, F = _import_torch()
    midi = 69.0 + 12.0 * torch.log2(torch.clamp(f0_hz, min=1e-5) / 440.0)
    return torch.clamp(midi / 127.0, 0.0, 1.5)


def db_to_unit(loudness_db, loud_min_db: float, loud_max_db: float):
    torch, nn, F = _import_torch()
    return torch.clamp((loudness_db - loud_min_db) / (loud_max_db - loud_min_db), 0.0, 1.0)


def harmonic_synth(f0_hz, harmonic_distribution, amplitudes, sample_rate: int, hop_samples: int):
    """Suma de sinusoides en f0*k (k=1..H), pesadas por harmonic_distribution
    (normalizada a que sume 1 por frame) y escaladas por `amplitudes`.
    f0_hz: [B,T]  harmonic_distribution: [B,T,H]  amplitudes: [B,T,1]
    -> audio [B, T*hop_samples]
    """
    torch, nn, F = _import_torch()
    B, T, H = harmonic_distribution.shape
    n_samples = T * hop_samples

    f0_audio = F.interpolate(f0_hz.unsqueeze(1), size=n_samples, mode="linear",
                              align_corners=False).squeeze(1)                       # [B,N]
    harm_idx = torch.arange(1, H + 1, device=f0_hz.device, dtype=f0_hz.dtype)        # [H]
    freqs = f0_audio.unsqueeze(-1) * harm_idx                                        # [B,N,H]

    nyquist = sample_rate / 2.0
    alias_mask = (freqs < nyquist).to(freqs.dtype)

    hd_audio = F.interpolate(harmonic_distribution.transpose(1, 2), size=n_samples,
                              mode="linear", align_corners=False).transpose(1, 2)     # [B,N,H]
    amp_audio = F.interpolate(amplitudes.transpose(1, 2), size=n_samples,
                               mode="linear", align_corners=False).transpose(1, 2)    # [B,N,1]

    hd_audio = hd_audio * alias_mask
    hd_audio = hd_audio / (hd_audio.sum(-1, keepdim=True) + 1e-7)

    omega = 2.0 * np.pi * freqs / sample_rate
    phase = torch.cumsum(omega, dim=1)
    signal = (torch.sin(phase) * hd_audio).sum(-1) * amp_audio.squeeze(-1)           # [B,N]
    return signal


def filtered_noise_synth(magnitudes, n_samples: int, n_fft: int, hop_samples: int):
    """Ruido blanco conformado espectralmente por un filtro que varía en el
    tiempo (magnitudes[B,T,M], M=n_fft//2+1): se genera ruido, se pasa a
    STFT, se re-escala cada bin por la envolvente deseada (conservando la
    fase aleatoria del propio ruido) y se reconstruye con ISTFT. No es el
    banco de filtros FIR de ddsp, pero cumple el mismo papel: ruido con
    envolvente espectral controlable frame a frame.
    """
    torch, nn, F = _import_torch()
    B, T, M = magnitudes.shape
    assert M == n_fft // 2 + 1, f"n_noise_fft={n_fft} implica {n_fft//2+1} magnitudes, no {M}"
    device = magnitudes.device
    noise = torch.randn(B, n_samples, device=device, dtype=magnitudes.dtype)
    window = torch.hann_window(n_fft, device=device, dtype=magnitudes.dtype)
    stft_noise = torch.stft(noise, n_fft=n_fft, hop_length=hop_samples, win_length=n_fft,
                             window=window, return_complex=True, center=True, pad_mode="reflect")
    Tf = stft_noise.shape[-1]
    mag_i = F.interpolate(magnitudes.transpose(1, 2), size=Tf, mode="linear",
                           align_corners=False)                                       # [B,M,Tf]
    unit_phase = stft_noise / (stft_noise.abs() + 1e-7)
    shaped = unit_phase * mag_i
    audio = torch.istft(shaped, n_fft=n_fft, hop_length=hop_samples, win_length=n_fft,
                         window=window, center=True, length=n_samples)
    return audio


def apply_reverb(signal, ir):
    """Convolución vía FFT de `signal` [B,N] con una respuesta al impulso
    `ir` [B,IR] aprendida (una por instrumento). Devuelve la señal 'wet',
    recortada a la longitud original."""
    torch, nn, F = _import_torch()
    B, N = signal.shape
    IR = ir.shape[-1]
    n_fft = 1
    while n_fft < N + IR - 1:
        n_fft *= 2
    S = torch.fft.rfft(signal, n=n_fft)
    Hf = torch.fft.rfft(ir, n=n_fft)
    wet = torch.fft.irfft(S * Hf, n=n_fft)[:, :N]
    return wet


def multiscale_spectral_loss(x, y, fft_sizes: Tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)):
    """Pérdida espectral multi-escala estándar de ddsp: L1 sobre magnitud
    lineal + L1 sobre log-magnitud, promediado sobre varios tamaños de
    ventana. Es la señal de entrenamiento principal de todo el sistema."""
    torch, nn, F = _import_torch()
    total = 0.0
    for n_fft in fft_sizes:
        n_fft = min(n_fft, x.shape[-1])
        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                        return_complex=True, center=True, pad_mode="reflect").abs()
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                        return_complex=True, center=True, pad_mode="reflect").abs()
        total = total + F.l1_loss(X, Y) + F.l1_loss(torch.log(X + 1e-5), torch.log(Y + 1e-5))
    return total / len(fft_sizes)


# =============================================================================
# §5  decoder compartido + banco de parámetros por instrumento
# =============================================================================

def _build_decoder_class():
    """Construye la clase Decoder en el primer uso (necesita nn.Module de
    torch, que se importa perezosamente)."""
    torch, nn, F = _import_torch()

    class FcStack(nn.Module):
        def __init__(self, in_dim, hidden, n_layers=2):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)]
                d = hidden
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        """Equivalente al CustomRnnFcDecoder del proyecto original: MLPs de
        entrada por-feature -> GRU (temporal) -> MLP de salida -> 3
        cabezas (amplitud global, distribución armónica, magnitudes de
        ruido). Es la única parte de la red que se COMPARTE entre todos
        los instrumentos."""

        def __init__(self, z_size, hidden, n_harmonics, n_noise_mags,
                     bidirectional=False, use_f0_confidence=True):
            super().__init__()
            self.use_f0_confidence = use_f0_confidence
            self.in_f0 = FcStack(1, hidden)
            self.in_loud = FcStack(1, hidden)
            self.in_z = FcStack(z_size, hidden)
            n_inputs = 3
            if use_f0_confidence:
                self.in_conf = FcStack(1, hidden)
                n_inputs = 4
            self.gru = nn.GRU(hidden * n_inputs, hidden, batch_first=True,
                               bidirectional=bidirectional)
            gru_out_dim = hidden * (2 if bidirectional else 1)
            self.out_stack = FcStack(gru_out_dim + hidden * n_inputs, hidden)
            self.out_amp = nn.Linear(hidden, 1)
            self.out_harm = nn.Linear(hidden, n_harmonics)
            self.out_noise = nn.Linear(hidden, n_noise_mags)

        def forward(self, f0_scaled, loud_scaled, z, conf_scaled=None):
            feats = [self.in_f0(f0_scaled), self.in_loud(loud_scaled), self.in_z(z)]
            if self.use_f0_confidence:
                feats.append(self.in_conf(conf_scaled))
            x = torch.cat(feats, dim=-1)
            gru_out, _ = self.gru(x)
            h = self.out_stack(torch.cat([gru_out, x], dim=-1))
            amp = _exp_sigmoid(self.out_amp(h))
            harm = _exp_sigmoid(self.out_harm(h))
            harm = harm / (harm.sum(-1, keepdim=True) + 1e-7)
            noise = _exp_sigmoid(self.out_noise(h) - 5.0)  # sesgo inicial hacia poco ruido
            return amp, harm, noise

    return Decoder


def _build_instrument_bank_class():
    torch, nn, F = _import_torch()

    class InstrumentBank(nn.Module):
        """Todo lo que hace único a un instrumento: su embedding z, la
        respuesta al impulso de su reverb, y la mezcla wet/dry. Clonar un
        instrumento nuevo == añadir una fila aquí y optimizarla con el
        decoder congelado."""

        def __init__(self, n_instruments, z_size, ir_samples):
            super().__init__()
            self.z = nn.Parameter(torch.randn(n_instruments, z_size) * 0.1)
            self.ir_raw = nn.Parameter(torch.zeros(n_instruments, ir_samples))
            self.dry_gain_raw = nn.Parameter(torch.full((n_instruments, 1), 2.0))
            self.wet_gain_raw = nn.Parameter(torch.full((n_instruments, 1), -2.0))

        def get(self, idx):
            z = torch.tanh(self.z[idx])
            ir = torch.tanh(self.ir_raw[idx]) * 0.1
            dry = torch.sigmoid(self.dry_gain_raw[idx])
            wet = torch.sigmoid(self.wet_gain_raw[idx])
            return z, ir, dry, wet

    return InstrumentBank


class SynthModel:
    """Envoltorio de alto nivel: decoder compartido + banco de instrumentos
    + toda la cadena de síntesis. No hereda de nn.Module directamente para
    poder construirse sin tener torch importado a nivel de módulo; internamente
    sí es un nn.Module real (`self.torch_module`)."""

    def __init__(self, hparams: Dict[str, Any], n_instruments: int):
        torch, nn, F = _import_torch()
        self.torch, self.nn, self.F = torch, nn, F
        self.hparams = dict(hparams)
        Decoder = _build_decoder_class()
        InstrumentBank = _build_instrument_bank_class()

        class _Module(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.decoder = Decoder(
                    hparams["z_size"], hparams["hidden"], hparams["n_harmonics"],
                    hparams["n_noise_fft"] // 2 + 1, hparams["bidirectional"],
                    hparams["use_f0_confidence"],
                )
                ir_samples = int(hparams["ir_seconds"] * hparams["sample_rate"])
                self_inner.bank = InstrumentBank(n_instruments, hparams["z_size"], ir_samples)

        self.module = _Module()
        self.n_instruments = n_instruments

    def to(self, device):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def decoder_parameters(self):
        return self.module.decoder.parameters()

    def bank_parameters(self):
        return self.module.bank.parameters()

    def set_shared_trainable(self, trainable: bool):
        for p in self.module.decoder.parameters():
            p.requires_grad_(trainable)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def forward(self, f0_hz, loudness_db, instrument_idx, f0_confidence=None):
        torch, nn, F = self.torch, self.nn, self.F
        hp = self.hparams
        B, T = f0_hz.shape
        z_i, ir_i, dry_i, wet_i = self.module.bank.get(instrument_idx)
        z = z_i.unsqueeze(1).expand(B, T, -1)
        f0s = hz_to_unit(f0_hz, hp["f0_min_hz"], hp["f0_max_hz"]).unsqueeze(-1)
        louds = db_to_unit(loudness_db, hp["loud_min_db"], hp["loud_max_db"]).unsqueeze(-1)
        confs = None
        if hp["use_f0_confidence"]:
            confs = (f0_confidence if f0_confidence is not None
                     else torch.ones_like(f0_hz)).unsqueeze(-1)
        amp, harm, noise = self.module.decoder(f0s, louds, z, confs)

        hop_samples = hp["sample_rate"] // hp["frame_rate"]
        harmonic_audio = harmonic_synth(f0_hz, harm, amp, hp["sample_rate"], hop_samples)
        n_samples = harmonic_audio.shape[-1]
        noise_audio = filtered_noise_synth(noise, n_samples, hp["n_noise_fft"], hop_samples)
        dry_audio = harmonic_audio + noise_audio
        wet_audio = apply_reverb(dry_audio, ir_i)
        return dry_audio * dry_i + wet_audio * wet_i

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    def state_dicts(self):
        return dict(decoder=self.module.decoder.state_dict(), bank=self.module.bank.state_dict())

    def load_state_dicts(self, state):
        self.module.decoder.load_state_dict(state["decoder"])
        self.module.bank.load_state_dict(state["bank"])


# =============================================================================
# §6  guardado / carga de checkpoints y voces clonadas
# =============================================================================

def save_checkpoint(path: str, model: "SynthModel", instrument_names: List[str],
                     epochs_trained: int, final_train_loss: float, final_val_loss: Optional[float]) -> None:
    torch, nn, F = _import_torch()
    out = dict(
        format=FORMAT_CHECKPOINT,
        hparams=model.hparams,
        fingerprint=_fingerprint(model.hparams),
        n_instruments=model.n_instruments,
        instrument_names=instrument_names,
        state=model.state_dicts(),
        epochs_trained=epochs_trained,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )
    torch.save(out, path)


def _load_raw(path: str) -> Dict[str, Any]:
    torch, nn, F = _import_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # torch viejo sin el argumento weights_only
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str) -> Tuple["SynthModel", Dict[str, Any]]:
    ckpt = _load_raw(path)
    if ckpt.get("format") != FORMAT_CHECKPOINT:
        sys.exit(f"{path} no es un checkpoint de instrument_cloner (formato: {ckpt.get('format')})")
    model = SynthModel(ckpt["hparams"], ckpt["n_instruments"])
    model.load_state_dicts(ckpt["state"])
    model.eval()
    return model, ckpt


def save_voice(path: str, name: str, checkpoint_fingerprint: Dict[str, Any],
               z_raw, ir_raw, dry_gain_raw, wet_gain_raw,
               steps_trained: int, final_loss: float, source_wav: str, source_duration_s: float) -> None:
    torch, nn, F = _import_torch()
    out = dict(
        format=FORMAT_VOICE,
        name=name,
        checkpoint_fingerprint=checkpoint_fingerprint,
        z_raw=z_raw.detach().cpu(), ir_raw=ir_raw.detach().cpu(),
        dry_gain_raw=dry_gain_raw.detach().cpu(), wet_gain_raw=wet_gain_raw.detach().cpu(),
        steps_trained=steps_trained, final_loss=final_loss,
        source_wav=source_wav, source_duration_s=source_duration_s,
    )
    torch.save(out, path)


def load_voice(path: str) -> Dict[str, Any]:
    voice = _load_raw(path)
    if voice.get("format") != FORMAT_VOICE:
        sys.exit(f"{path} no es una voz clonada de instrument_cloner (formato: {voice.get('format')})")
    return voice


def _check_fingerprint(checkpoint_fp: Dict[str, Any], other_fp: Dict[str, Any],
                        what: str, path_hint: str) -> None:
    if checkpoint_fp != other_fp:
        diffs = [k for k in checkpoint_fp if checkpoint_fp.get(k) != other_fp.get(k)]
        sys.exit(
            f"El checkpoint no es compatible con {what} ({path_hint}): "
            f"difieren en {', '.join(diffs)}.\n"
            f"  checkpoint: {checkpoint_fp}\n"
            f"  {what}:     {other_fp}"
        )


# =============================================================================
# §7  entrenamiento del modelo compartido
# =============================================================================

def _iterate_batches(split: Dict[str, "torch.Tensor"], batch_size: int, shuffle: bool, rng):
    n = len(split["instrument_idx"])
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield {k: v[idx] for k, v in split.items()}


def _run_epoch(model: "SynthModel", split, batch_size, device, rng, optimizer=None):
    torch = model.torch
    total_loss, n_batches = 0.0, 0
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    with torch.set_grad_enabled(is_train):
        for batch in _iterate_batches(split, batch_size, shuffle=is_train, rng=rng):
            f0 = batch["f0_hz"].to(device)
            loud = batch["loudness_db"].to(device)
            conf = batch["f0_confidence"].to(device)
            idx = batch["instrument_idx"].to(device)
            target = batch["audio"].to(device)
            pred = model.forward(f0, loud, idx, conf)
            loss = multiscale_spectral_loss(pred, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)


def train(dataset_path: str, out_path: str, epochs: int = 60, batch_size: int = 16,
          lr: float = 3e-4, seed: int = 0, resume: Optional[str] = None,
          save_every: Optional[int] = None, verbose: bool = True) -> None:
    torch, nn, F = _import_torch()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    device = "cpu"

    data = _load_raw(dataset_path)
    if data.get("format") != FORMAT_DATASET:
        sys.exit(f"{dataset_path} no es un dataset de instrument_cloner (formato: {data.get('format')})")
    if data["train"] is None:
        sys.exit("el dataset no tiene ventanas de train")

    hparams = data["hparams"]
    instrument_names = data["instrument_names"]
    n_instruments = len(instrument_names)

    if resume:
        model, ckpt = load_checkpoint(resume)
        _check_fingerprint(ckpt["fingerprint"], data["fingerprint"], "el dataset", dataset_path)
        if ckpt["n_instruments"] != n_instruments:
            sys.exit(f"--resume tiene {ckpt['n_instruments']} instrumentos, "
                      f"el dataset tiene {n_instruments}: no se puede continuar el mismo banco")
        epochs_done = ckpt["epochs_trained"]
    else:
        model = SynthModel(hparams, n_instruments)
        epochs_done = 0
    model.to(device)
    model.set_shared_trainable(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_loss = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, data["train"], batch_size, device, rng, optimizer)
        if data["val"] is not None:
            val_loss = _run_epoch(model, data["val"], batch_size, device, rng, optimizer=None)
        if verbose:
            msg = f"  epoch {epochs_done + epoch:4d}  train_loss {train_loss:.4f}"
            if val_loss is not None:
                msg += f"  val_loss {val_loss:.4f}"
            msg += f"  ({time.time() - t0:.0f}s)"
            print(msg)
        if save_every and epoch % save_every == 0:
            save_checkpoint(out_path, model, instrument_names, epochs_done + epoch, train_loss, val_loss)

    save_checkpoint(out_path, model, instrument_names, epochs_done + epochs, train_loss, val_loss)
    print(f"  OK {epochs} época(s) -> {out_path}")


# =============================================================================
# §8  clonación: ajustar una voz nueva sobre el decoder compartido y congelado
# =============================================================================

def _windows_from_wav(path: str, hparams: Dict[str, Any], hop_s: float,
                       min_confidence: float = 0.0) -> Tuple[Dict[str, np.ndarray], float]:
    """Trocea un wav en ventanas de hparams['window_s'] segundos. Si el wav
    es más corto que una ventana, se repite (tile) hasta llenarla y se
    avisa -- mejor eso que fallar, pero el resultado será peor cuanto más
    haya que rellenar."""
    sr = hparams["sample_rate"]
    audio_raw = load_audio_mono(path, sr)
    duration_s = len(audio_raw) / sr
    win_samples = int(round(hparams["window_s"] * sr))
    if len(audio_raw) < win_samples:
        reps = int(np.ceil(win_samples / max(1, len(audio_raw))))
        print(f"  ! {path} dura {duration_s:.1f}s, menos que la ventana de "
              f"{hparams['window_s']:.1f}s del checkpoint -- se repite el audio para rellenar")
        audio_raw = np.tile(audio_raw, reps)

    feats = extract_features(audio_raw, sr, hparams["frame_rate"], hparams["f0_min_hz"],
                              hparams["f0_max_hz"], hparams["yin_threshold"])
    feats["hop_samples"] = sr // hparams["frame_rate"]
    feats["win_frames"] = int(round(hparams["window_s"] * hparams["frame_rate"]))
    feats["win_samples"] = feats["win_frames"] * feats["hop_samples"]
    hop_frames = max(1, int(round(hop_s * hparams["frame_rate"])))
    windows = _slice_windows(feats, hop_frames, min_confidence)
    if not windows:
        sys.exit(f"no se pudo extraer ninguna ventana usable de {path} "
                 f"(¿silencio, o f0 poco claro?) -- prueba con --min-confidence 0")
    torch, nn, F = _import_torch()
    batch = dict(
        audio=torch.tensor(np.stack([w["audio"] for w in windows])),
        f0_hz=torch.tensor(np.stack([w["f0_hz"] for w in windows])),
        f0_confidence=torch.tensor(np.stack([w["f0_confidence"] for w in windows])),
        loudness_db=torch.tensor(np.stack([w["loudness_db"] for w in windows])),
    )
    return batch, duration_s


def clone(target_wav: str, checkpoint_path: str, out_path: str, name: Optional[str] = None,
          steps: int = 1500, lr: float = 5e-3, batch_size: int = 8, hop_s: float = 1.0,
          seed: int = 0, verbose: bool = True) -> None:
    torch, nn, F = _import_torch()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    shared_model, ckpt = load_checkpoint(checkpoint_path)
    hp = shared_model.hparams
    batch, duration_s = _windows_from_wav(target_wav, hp, hop_s)
    if verbose:
        print(f"  {target_wav}: {duration_s:.1f}s -> {batch['audio'].shape[0]} ventana(s) de entrenamiento")

    voice_model = SynthModel(hp, n_instruments=1)
    voice_model.module.decoder.load_state_dict(shared_model.module.decoder.state_dict())
    voice_model.set_shared_trainable(False)
    voice_model.train()
    optimizer = torch.optim.Adam(voice_model.bank_parameters(), lr=lr)

    n = batch["audio"].shape[0]
    bs = min(batch_size, n)
    log_every = max(1, steps // 10)
    loss_val = None
    t0 = time.time()
    for step in range(1, steps + 1):
        idx_sel = rng.randint(0, n, size=bs)
        f0 = batch["f0_hz"][idx_sel]
        loud = batch["loudness_db"][idx_sel]
        conf = batch["f0_confidence"][idx_sel]
        target = batch["audio"][idx_sel]
        inst_idx = torch.zeros(bs, dtype=torch.long)
        pred = voice_model.forward(f0, loud, inst_idx, conf)
        loss = multiscale_spectral_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        if verbose and (step % log_every == 0 or step == steps):
            print(f"  paso {step:5d}/{steps}  loss {loss_val:.4f}  ({time.time() - t0:.0f}s)")

    bank = voice_model.module.bank
    save_voice(out_path, name or Path(target_wav).stem, _fingerprint(hp),
               bank.z[0], bank.ir_raw[0], bank.dry_gain_raw[0], bank.wet_gain_raw[0],
               steps, loss_val, Path(target_wav).name, duration_s)
    print(f"  OK voz clonada ({loss_val:.4f} loss final) -> {out_path}")


# =============================================================================
# §9  síntesis: renderizar audio con una voz (clonada o de entrenamiento)
# =============================================================================

def _adsr_envelope(n_frames: int, frame_rate: int, attack_s: float = 0.05,
                    release_s: float = 0.3, sustain_db: float = -12.0,
                    floor_db: float = -80.0) -> np.ndarray:
    """Envolvente de loudness simple (attack lineal, sustain, release) para
    sintetizar una nota "seca" sin necesidad de un wav de control."""
    attack_n = max(1, int(attack_s * frame_rate))
    release_n = max(1, int(release_s * frame_rate))
    env = np.full(n_frames, sustain_db, dtype=np.float32)
    a = np.linspace(floor_db, sustain_db, min(attack_n, n_frames))
    env[:len(a)] = a
    if release_n > 0 and n_frames > release_n:
        r = np.linspace(sustain_db, floor_db, release_n)
        env[-release_n:] = r
    return env


def _resolve_voice_params(shared_model: "SynthModel", ckpt: Dict[str, Any],
                           voice_path: Optional[str], instrument_idx: Optional[int]):
    torch, nn, F = shared_model.torch, shared_model.nn, shared_model.F
    if voice_path is not None:
        voice = load_voice(voice_path)
        _check_fingerprint(ckpt["fingerprint"], voice["checkpoint_fingerprint"], "la voz", voice_path)
        return voice["z_raw"], voice["ir_raw"], voice["dry_gain_raw"], voice["wet_gain_raw"], voice["name"]
    if instrument_idx is not None:
        names = ckpt["instrument_names"]
        if not (0 <= instrument_idx < len(names)):
            sys.exit(f"--instrument-idx debe estar entre 0 y {len(names) - 1} "
                      f"(instrumentos: {', '.join(f'{i}:{n}' for i, n in enumerate(names))})")
        bank = shared_model.module.bank
        return (bank.z[instrument_idx], bank.ir_raw[instrument_idx],
                bank.dry_gain_raw[instrument_idx], bank.wet_gain_raw[instrument_idx],
                names[instrument_idx])
    sys.exit("hace falta --voice o --instrument-idx para saber qué timbre usar")


def synthesize(checkpoint_path: str, out_path: str, voice_path: Optional[str] = None,
               instrument_idx: Optional[int] = None, control_wav: Optional[str] = None,
               note: Optional[int] = None, duration: Optional[float] = None,
               velocity: float = 0.8, seed: int = 0) -> None:
    torch, nn, F = _import_torch()
    torch.manual_seed(seed)
    shared_model, ckpt = load_checkpoint(checkpoint_path)
    hp = shared_model.hparams
    z_raw, ir_raw, dry_raw, wet_raw, label = _resolve_voice_params(
        shared_model, ckpt, voice_path, instrument_idx)

    if control_wav is not None:
        audio = load_audio_mono(control_wav, hp["sample_rate"])
        feats = extract_features(audio, hp["sample_rate"], hp["frame_rate"],
                                  hp["f0_min_hz"], hp["f0_max_hz"], hp["yin_threshold"])
        f0_hz = feats["f0_hz"]
        loudness_db = feats["loudness_db"]
        f0_conf = feats["f0_confidence"]
    elif note is not None and duration is not None:
        n_frames = int(round(duration * hp["frame_rate"]))
        f0_hz = np.full(n_frames, 440.0 * 2 ** ((note - 69) / 12.0), dtype=np.float32)
        sustain_db = hp["loud_min_db"] + velocity * (hp["loud_max_db"] - hp["loud_min_db"])
        loudness_db = _adsr_envelope(n_frames, hp["frame_rate"], sustain_db=sustain_db,
                                      floor_db=hp["loud_min_db"])
        f0_conf = np.ones(n_frames, dtype=np.float32)
    else:
        sys.exit("hace falta --control-wav, o --note y --duration, para saber qué tocar")

    model = SynthModel(hp, n_instruments=1)
    model.module.decoder.load_state_dict(shared_model.module.decoder.state_dict())
    model.module.bank.z.data[0] = z_raw
    model.module.bank.ir_raw.data[0] = ir_raw
    model.module.bank.dry_gain_raw.data[0] = dry_raw
    model.module.bank.wet_gain_raw.data[0] = wet_raw
    model.eval()

    with torch.no_grad():
        f0_t = torch.tensor(f0_hz).unsqueeze(0)
        loud_t = torch.tensor(loudness_db).unsqueeze(0)
        conf_t = torch.tensor(f0_conf).unsqueeze(0)
        idx_t = torch.zeros(1, dtype=torch.long)
        audio_out = model.forward(f0_t, loud_t, idx_t, conf_t)[0].cpu().numpy()

    peak = np.max(np.abs(audio_out)) + 1e-9
    if peak > 0.999:
        audio_out = audio_out / peak * 0.999
    sf = _import_soundfile()
    sf.write(out_path, audio_out, hp["sample_rate"], subtype="FLOAT")
    print(f"  OK voz '{label}' -> {out_path} ({len(audio_out) / hp['sample_rate']:.2f}s)")


# =============================================================================
# §10  CLI
# =============================================================================

def _print_hparams(hp: Dict[str, Any]) -> None:
    for k in _FINGERPRINT_KEYS:
        print(f"  {k:18s} {hp[k]}")


def cmd_prepare_data(args):
    prepare_dataset(args.input_dir, args.out, sample_rate=args.sample_rate,
                     frame_rate=args.frame_rate, window_s=args.window_s, hop_s=args.hop_s,
                     val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                     fmin=args.fmin, fmax=args.fmax, yin_threshold=args.yin_threshold,
                     min_confidence=args.min_confidence, seed=args.seed, verbose=not args.quiet)


def cmd_train(args):
    train(args.dataset, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, seed=args.seed, resume=args.resume, save_every=args.save_every,
          verbose=not args.quiet)


def cmd_clone(args):
    clone(args.target_wav, args.checkpoint, args.out, name=args.name, steps=args.steps,
          lr=args.lr, batch_size=args.batch_size, hop_s=args.hop_s, seed=args.seed,
          verbose=not args.quiet)


def cmd_synthesize(args):
    if not args.control_wav and (args.note is None or args.duration is None):
        sys.exit("especifica --control-wav, o --note y --duration, para saber que tocar")
    synthesize(args.checkpoint, args.out, voice_path=args.voice, instrument_idx=args.instrument_idx,
               control_wav=args.control_wav, note=args.note, duration=args.duration,
               velocity=args.velocity, seed=args.seed)


def cmd_info(args):
    obj = _load_raw(args.path)
    fmt = obj.get("format")
    if fmt == FORMAT_DATASET:
        print(f"formato          dataset")
        print(f"instrumentos     {len(obj['instrument_names'])}: {', '.join(obj['instrument_names'])}")
        for split in ("train", "val", "test"):
            n = 0 if obj[split] is None else len(obj[split]["instrument_idx"])
            print(f"ventanas {split:6s} {n}")
        print("hiperparámetros")
        _print_hparams(obj["hparams"])
    elif fmt == FORMAT_CHECKPOINT:
        print(f"formato          checkpoint compartido")
        print(f"instrumentos     {obj['n_instruments']}: {', '.join(obj['instrument_names'])}")
        print(f"épocas entrenado {obj['epochs_trained']}")
        print(f"train_loss final {obj['final_train_loss']:.4f}")
        if obj.get("final_val_loss") is not None:
            print(f"val_loss final   {obj['final_val_loss']:.4f}")
        print("hiperparámetros")
        _print_hparams(obj["hparams"])
    elif fmt == FORMAT_VOICE:
        print(f"formato            voz clonada")
        print(f"nombre             {obj['name']}")
        print(f"wav origen         {obj['source_wav']} ({obj['source_duration_s']:.1f}s)")
        print(f"pasos entrenamiento {obj['steps_trained']}")
        print(f"loss final         {obj['final_loss']:.4f}")
        print(f"fingerprint checkpoint requerido")
        for k, v in obj["checkpoint_fingerprint"].items():
            print(f"  {k:18s} {v}")
    else:
        sys.exit(f"{args.path}: formato desconocido ({fmt}) -- ¿es un fichero de instrument_cloner.py?")


def cmd_list_instruments(args):
    obj = _load_raw(args.checkpoint)
    if obj.get("format") != FORMAT_CHECKPOINT:
        sys.exit(f"{args.checkpoint} no es un checkpoint de instrument_cloner")
    for i, name in enumerate(obj["instrument_names"]):
        print(f"  {i:3d}  {name}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instrument_cloner",
        description="Clonacion neuronal de timbres instrumentales a partir de "
                     "unos segundos de audio (puerto autocontenido en PyTorch, "
                     "sin tensorflow/ddsp).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-data", help="Carpeta de audio -> dataset de ventanas")
    p.add_argument("input_dir", help="una subcarpeta por instrumento, wavs dentro")
    p.add_argument("--out", required=True)
    p.add_argument("--sample-rate", type=int, default=DEFAULTS["sample_rate"])
    p.add_argument("--frame-rate", type=int, default=DEFAULTS["frame_rate"])
    p.add_argument("--window-s", type=float, default=DEFAULTS["window_s"])
    p.add_argument("--hop-s", type=float, default=1.0, help="solape entre ventanas de train")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--fmin", type=float, default=DEFAULTS["f0_min_hz"])
    p.add_argument("--fmax", type=float, default=DEFAULTS["f0_max_hz"])
    p.add_argument("--yin-threshold", type=float, default=DEFAULTS["yin_threshold"])
    p.add_argument("--min-confidence", type=float, default=0.5,
                    help="descarta ventanas con confianza de f0 media por debajo de esto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train", help="Entrena el decoder compartido + banco de instrumentos")
    p.add_argument("dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="continuar el entrenamiento de un checkpoint")
    p.add_argument("--save-every", type=int, default=None, help="guardar cada N epocas (ademas del final)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("clone", help="Ajusta una voz nueva sobre un checkpoint compartido")
    p.add_argument("target_wav")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("synthesize", help="Renderiza audio con una voz (clonada o de entrenamiento)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--voice", default=None, help="fichero de voz clonada (ver `clone`)")
    p.add_argument("--instrument-idx", type=int, default=None,
                    help="alternativa a --voice: usar un instrumento del propio checkpoint")
    p.add_argument("--control-wav", default=None,
                    help="extrae f0/loudness de este wav y los toca con la voz elegida")
    p.add_argument("--note", type=int, default=None, help="nota MIDI (alternativa a --control-wav)")
    p.add_argument("--duration", type=float, default=None, help="segundos (con --note)")
    p.add_argument("--velocity", type=float, default=0.8, help="0-1, con --note")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("info", help="Inspecciona un dataset / checkpoint / voz clonada")
    p.add_argument("path")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("list-instruments", help="Lista los instrumentos de un checkpoint")
    p.add_argument("checkpoint")
    p.set_defaults(func=cmd_list_instruments)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
