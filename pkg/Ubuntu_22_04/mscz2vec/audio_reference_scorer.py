#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AUDIO REFERENCE SCORER  v1.0                              ║
║  Score 0-1 de un WAV candidato contra un corpus de referencia — fichero único║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Compara un WAV candidato contra un corpus de WAV de referencia (obras     ║
║    que le gustan al usuario) usando distancia de Mahalanobis en un espacio   ║
║    de features de audio, y devuelve un score 0-1 de afinidad/calidad.        ║
║                                                                              ║
║  AVISO DE DOMINIO (leer antes de usar en serio, no es opcional)             ║
║    Si el corpus de referencia son grabaciones reales y los candidatos son    ║
║    renders de soundfont, la distancia mide sobre todo "¿esto es un          ║
║    soundfont o una grabación real?", no "¿esto es buena música?". Esta      ║
║    herramienta lo mitiga con --domain-tag (avisa en el informe) y con       ║
║    --floor-corpus (calibración por suelo, ver §4.5 de la especificación):   ║
║    separa "qué tan lejos de mis referencias" de "qué tan lejos comparado    ║
║    con mis propios renders anteriores" — la segunda es la que sirve para    ║
║    guiar RL cuando el corpus es de dominio distinto ('cross').              ║
║                                                                              ║
║  USO                                                                        ║
║    --build-cache DIR --cache F --domain-tag {same,cross}  construye caché  ║
║    candidato.wav --cache F                                 puntúa 1 WAV    ║
║    cand1.wav cand2.wav ... --cache F --mode rank            rankea varios  ║
║    cand1.wav cand2.wav ... --cache F --mode fad-batch        FAD de lote   ║
║                                                                              ║
║  BACKENDS DE FEATURES (--backend)                                           ║
║    spectral (default)  mel-espectrograma + descriptores, 100% numpy/scipy, ║
║                         sin modelo entrenado, sensible a timbre/producción ║
║    latent               tokens semánticos de audiolm.py (requiere codec ya  ║
║                         entrenado), reduce (no elimina) el hueco de dominio ║
║                                                                              ║
║  DEPENDENCIAS  numpy  scipy  soundfile        (obligatorias, backend       ║
║                                                 spectral, el del default)   ║
║                audiolm.py (mismo ecosistema)   solo si --backend latent    ║
║                mido                            solo para EvalProvider,     ║
║                                                 al puntuar MIDI intermedio  ║
║                matplotlib                      solo si --plot             ║
║                                                                              ║
║  LIMITACIONES DE FONDO (no son bugs, ver especificación §10)               ║
║    · El backend spectral no distingue "peor música" de "instrumento/mezcla ║
║      distinta" — mide distancia tímbrica/de producción, no calidad per se. ║
║    · La calibración por suelo mitiga, no elimina, el hueco de dominio.     ║
║    · El backend latent depende de qué vio el codec de audiolm.py en su     ║
║      propio entrenamiento — no asumir que resuelve el hueco de dominio sin ║
║      verificarlo empíricamente.                                            ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║    python audio_reference_scorer.py --build-cache refs/ --cache ref.npz \\ ║
║           --domain-tag cross                                                ║
║    python audio_reference_scorer.py mi_obra.wav --cache ref.npz            ║
║    python audio_reference_scorer.py mi_obra.wav --cache ref.npz \\         ║
║           --floor-corpus mis_renders/ --floor-cache floor.npz              ║
║    python audio_reference_scorer.py a.wav b.wav c.wav --cache ref.npz \\   ║
║           --mode rank                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Módulo importable:
    from audio_reference_scorer import score_against_corpus, build_reference_cache
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ── lazy imports (mismo patrón que el resto del ecosistema) ───────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. Instala con: pip install soundfile")

def _import_scipy_signal():
    try:
        from scipy import signal
        return signal
    except ImportError:
        sys.exit("✗  scipy no encontrado (obligatorio para el backend spectral). "
                  "pip install scipy")

def _import_scipy_linalg():
    try:
        from scipy import linalg
        return linalg
    except ImportError:
        sys.exit("✗  scipy no encontrado (obligatorio para --mode fad-batch). "
                  "pip install scipy")

def _import_mido():
    try:
        import mido
        return mido
    except ImportError:
        sys.exit("✗  mido no encontrado (necesario para puntuar candidatos MIDI vía "
                  "AudioReferenceProvider). pip install mido")

def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.exit("✗  matplotlib no encontrado (necesario para --plot). "
                  "pip install matplotlib")

def _try_import_audiolm():
    """Import no-fatal de audiolm.py (módulo del mismo ecosistema, no un paquete pip).
    Se busca en el mismo directorio que este script y en sys.path. Devuelve None si no
    está disponible — nunca sys.exit aquí, porque is_available() necesita poder
    responder False sin abortar el proceso."""
    if importlib.util.find_spec("audiolm") is not None:
        import audiolm
        return audiolm
    local = Path(__file__).resolve().parent / "audiolm.py"
    if local.exists():
        spec = importlib.util.spec_from_file_location("audiolm", local)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ── constantes (puntos de partida razonables, no validados empíricamente) ─────
# ══════════════════════════════════════════════════════════════════════════════

TARGET_SR = 22050          # §3.1 — tasa de remuestreo del backend spectral
N_FFT = 2048                # §3.1 — tamaño de ventana STFT
HOP_LENGTH = 512             # §3.1 — salto entre frames STFT
N_MELS = 40                  # §3.1 — bandas del banco de filtros mel
ROLLOFF_PCT = 0.85           # §3.1 — percentil del roll-off espectral
SPECTRAL_EMBEDDING_DIM = (N_MELS + 4) * 2   # 44 por frame (mel+4 descriptores) × (media+std) = 88

COV_DIAG_THRESHOLD_FACTOR = 10   # §4.2 — si n_ref < FACTOR * dim, covarianza diagonal
COV_SHRINKAGE = 1e-6              # §4.2 — shrinkage mínimo de la covarianza completa

CACHE_FORMAT_VERSION = 1

# Timbre "neutro" fijo usado por AudioReferenceProvider al renderizar MIDI intermedio
# (§6) — armónicos con caída 1/n, sin vibrato ni variación entre notas, para aislar
# la decisión compositiva del timbre real. No pretende sonar "bien", solo ser estable
# y reproducible entre steps de RL.
_NEUTRAL_N_HARMONICS = 6
_NEUTRAL_ATTACK_S = 0.012
_NEUTRAL_RELEASE_S = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# ── excepciones ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class BackendUnavailableError(RuntimeError):
    """El backend de features solicitado no está disponible (dependencia opcional
    ausente, o checkpoint/codec no encontrado). Nunca se debe caer en silencio a
    otro backend — el usuario debe verlo y decidir."""

class CacheBackendMismatchError(RuntimeError):
    """El caché de corpus fue construido con un backend distinto al solicitado en
    tiempo de scoring — comparar embeddings de espacios distintos daría un número
    sin sentido, así que esto se trata como error duro, nunca como resultado
    silenciosamente incorrecto."""

class InsufficientCorpusError(RuntimeError):
    """El corpus de referencia o de floor no tiene suficientes ficheros .wav para
    calcular estadísticas mínimamente razonables (mínimo 2)."""


# ══════════════════════════════════════════════════════════════════════════════
# ── §3.1 backend "spectral" — DSP a mano, sin dependencias pesadas ────────────
# ══════════════════════════════════════════════════════════════════════════════

def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    """Banco de filtros mel triangulares hecho a mano (fórmula estándar). Devuelve
    una matriz (n_mels, n_fft//2 + 1) para multiplicar directamente por un
    espectrograma de magnitud de STFT en escala lineal de frecuencia."""
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_min, mel_max = hz_to_mel(0), hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_prev, f_curr, f_next = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        fb[m - 1, f_prev:f_curr] = np.linspace(0, 1, max(f_curr - f_prev, 1))
        fb[m - 1, f_curr:f_next] = np.linspace(1, 0, max(f_next - f_curr, 1))
    return fb


def _load_wav_mono(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Lee un WAV con soundfile, lo pasa a mono (media de canales) y lo remuestrea
    a target_sr con scipy.signal.resample_poly si hace falta."""
    sf = _import_soundfile()
    audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        signal = _import_scipy_signal()
        g = gcd(int(target_sr), int(sr))
        up, down = target_sr // g, sr // g
        audio = signal.resample_poly(audio, up, down)
        sr = target_sr
    return np.ascontiguousarray(audio, dtype=np.float64), sr


def _stft_magnitude(audio: np.ndarray, sr: int, n_fft: int = N_FFT,
                     hop: int = HOP_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """STFT con ventana Hann → (freqs, magnitud) con magnitud de forma
    (n_fft//2 + 1, n_frames). Solo magnitud, la fase se descarta."""
    signal = _import_scipy_signal()
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    freqs, _times, Zxx = signal.stft(audio, fs=sr, window="hann", nperseg=n_fft,
                                      noverlap=n_fft - hop, boundary=None, padded=False)
    return freqs, np.abs(Zxx)


def _spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    total = mag.sum(axis=0) + 1e-12
    return (freqs[:, None] * mag).sum(axis=0) / total


def _spectral_flatness(mag: np.ndarray) -> np.ndarray:
    eps = 1e-12
    gmean = np.exp(np.mean(np.log(mag + eps), axis=0))
    amean = mag.mean(axis=0) + eps
    return gmean / amean


def _spectral_rolloff(mag: np.ndarray, freqs: np.ndarray, pct: float = ROLLOFF_PCT) -> np.ndarray:
    cumsum = np.cumsum(mag, axis=0)
    total = cumsum[-1, :] + 1e-12
    thresh = pct * total
    idx = np.argmax(cumsum >= thresh[None, :], axis=0)
    return freqs[idx]


def _zero_crossing_rate(audio: np.ndarray, n_fft: int, hop: int, n_frames: int) -> np.ndarray:
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + n_fft]
        if len(frame) < 2:
            continue
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr[i] = np.mean(signs[:-1] != signs[1:])
    return zcr


def extract_spectral_embedding(wav_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Devuelve un vector fijo de 88 dimensiones. Determinista, sin estado
    entrenado. Pipeline: carga+mono+remuestreo → STFT → mel-log + 4 descriptores
    por frame (centroide, flatness, roll-off, ZCR) → agregación media+std sobre
    el tiempo (invariante a duración, ver §3.1 y plan de pruebas #3)."""
    audio, sr = _load_wav_mono(wav_path, target_sr)
    freqs, mag = _stft_magnitude(audio, sr)
    n_frames = mag.shape[1]
    mel_fb = mel_filterbank(sr, N_FFT, N_MELS)
    log_mel = np.log1p(mel_fb @ mag)                                  # (n_mels, n_frames)
    centroid = _spectral_centroid(mag, freqs)
    flatness = _spectral_flatness(mag)
    rolloff = _spectral_rolloff(mag, freqs)
    zcr = _zero_crossing_rate(audio, N_FFT, HOP_LENGTH, n_frames)
    frame_vectors = np.vstack([log_mel, centroid[None, :], flatness[None, :],
                                rolloff[None, :], zcr[None, :]])       # (44, n_frames)
    mean_vec = frame_vectors.mean(axis=1)
    std_vec = frame_vectors.std(axis=1)
    return np.concatenate([mean_vec, std_vec])                        # (88,)


# ══════════════════════════════════════════════════════════════════════════════
# ── §3.2 backend "latent" (opcional) ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#
# Usa los tokens SEMÁNTICOS (prosodia/contenido) de audiolm.py, no los acústicos
# (Coarse/Fine, que capturan timbre — justo lo que se quiere evitar aquí). Esto
# reduce, pero no elimina, la sensibilidad a timbre/producción frente a
# "spectral": si el codec de audiolm.py se entrenó solo con soundfont, sus tokens
# sobre grabaciones reales pueden no ser fiables (mismo aviso de dominio de §1,
# aplicado ahora al propio codec en vez de al backend spectral).
#
# La API exacta de audiolm.py no está fijada por esta herramienta: se asume un
# método de carga y uno de extracción de tokens semánticos (ver LatentBackend
# más abajo). Si tu audiolm.py expone nombres distintos, ajusta esos dos puntos.

class SpectralBackend:
    name = "spectral"

    def is_available(self) -> bool:
        return (importlib.util.find_spec("scipy") is not None
                and importlib.util.find_spec("soundfile") is not None)

    def extract(self, wav_path: str) -> np.ndarray:
        return extract_spectral_embedding(wav_path)


class LatentBackend:
    name = "latent"

    def __init__(self, codec_checkpoint: Optional[str]):
        self.codec_checkpoint = codec_checkpoint
        self._model = None

    def is_available(self) -> bool:
        if not self.codec_checkpoint:
            return False
        return _try_import_audiolm() is not None and Path(self.codec_checkpoint).exists()

    def _load(self):
        if self._model is not None:
            return self._model
        audiolm = _try_import_audiolm()
        if audiolm is None:
            raise BackendUnavailableError(
                "audiolm.py no es importable (debe estar en el mismo directorio o en "
                "PYTHONPATH) — el backend 'latent' requiere el mismo ecosistema.")
        if not self.codec_checkpoint or not Path(self.codec_checkpoint).exists():
            raise BackendUnavailableError(
                f"Checkpoint de audiolm.py no encontrado: {self.codec_checkpoint}")
        try:
            self._model = audiolm.load_semantic_model(self.codec_checkpoint)
        except AttributeError as e:
            raise BackendUnavailableError(
                "audiolm.py no expone 'load_semantic_model(checkpoint)' — la API "
                "asumida por este adaptador no coincide con tu versión. Ajusta "
                "LatentBackend._load()/extract() en audio_reference_scorer.py a la "
                f"API real de tu audiolm.py. Detalle: {e}") from e
        return self._model

    def extract(self, wav_path: str) -> np.ndarray:
        """Carga el codec+semantic transformer de audiolm.py, extrae la secuencia
        de tokens semánticos, agrega (media+std sobre la secuencia) a un vector
        fijo. Lanza BackendUnavailableError si el checkpoint no existe o
        audiolm.py no es importable — nunca cae en silencio a 'spectral'."""
        model = self._load()
        try:
            tokens = model.encode_semantic(str(wav_path))
        except AttributeError as e:
            raise BackendUnavailableError(
                "audiolm.py no expone 'encode_semantic(wav_path)' en el modelo "
                "cargado — ajusta LatentBackend.extract() a la API real. "
                f"Detalle: {e}") from e
        tokens = np.asarray(tokens, dtype=np.float64)
        if tokens.ndim == 1:
            tokens = tokens[:, None]
        return np.concatenate([tokens.mean(axis=0), tokens.std(axis=0)])


def get_backend(name: str, audiolm_codec: Optional[str] = None):
    """Factoría: devuelve una instancia de FeatureBackend por nombre. Ambos
    backends implementan is_available() / extract(wav_path) -> np.ndarray."""
    if name == "spectral":
        return SpectralBackend()
    if name == "latent":
        return LatentBackend(audiolm_codec)
    raise ValueError(f"Backend desconocido: {name!r} (usa 'spectral' o 'latent')")


# ══════════════════════════════════════════════════════════════════════════════
# ── §4.1-4.3 distancia de Mahalanobis, regularización, calibración ────────────
# ══════════════════════════════════════════════════════════════════════════════

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
    diff = x - mean
    return float(np.sqrt(max(diff @ cov_inv @ diff, 0.0)))


def compute_reference_stats(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """§4.2 — regularización obligatoria. Con n_ref archivos y dimensión d, si
    n_ref no es sustancialmente mayor que d la covarianza completa es singular o
    casi singular. Si n < COV_DIAG_THRESHOLD_FACTOR * d, se usa covarianza
    diagonal (solo varianzas) en vez de covarianza completa."""
    n, d = embeddings.shape
    mean = embeddings.mean(axis=0)
    if n < COV_DIAG_THRESHOLD_FACTOR * d:
        var = embeddings.var(axis=0) + COV_SHRINKAGE
        cov = np.diag(var)
    else:
        cov = np.cov(embeddings, rowvar=False) + COV_SHRINKAGE * np.eye(d)
    return mean, cov


def calibrate_scale(embeddings: np.ndarray) -> Tuple[float, float]:
    """§4.3 — Devuelve (mu, sigma) de las distancias leave-one-out dentro del
    corpus de referencia: para cada fichero, su distancia de Mahalanobis al
    resto del corpus. Define 'cuánto se dispersan típicamente las obras que le
    gustan al usuario entre sí'."""
    n = len(embeddings)
    distances = []
    for i in range(n):
        rest = np.delete(embeddings, i, axis=0)
        mean_i, cov_i = compute_reference_stats(rest)
        cov_inv_i = np.linalg.pinv(cov_i)   # pinv, no inv — nunca debe lanzar aquí
        distances.append(mahalanobis_distance(embeddings[i], mean_i, cov_inv_i))
    return float(np.mean(distances)), float(np.std(distances) + 1e-6)


def normalize_score(distance: float, mu: float, sigma: float) -> float:
    """Transforma distancia cruda a score 0-1: 1.0 = mucho más cerca de lo
    típico, 0.5 = tan típico como una obra de referencia cualquiera (distance ==
    mu), decreciendo suavemente cuanto más se aleja. Sigmoide centrada en mu."""
    z = (distance - mu) / sigma
    return float(1.0 / (1.0 + np.exp(z)))


def frechet_distance(mean1: np.ndarray, cov1: np.ndarray,
                      mean2: np.ndarray, cov2: np.ndarray) -> float:
    """§4.1 — Fréchet Distance completa entre dos distribuciones (media+
    covarianza a cada lado). Solo tiene sentido cuando hay más de un candidato
    del lado 'generado' (--mode fad-batch) — con un único candidato no hay
    distribución de ese lado y la métrica correcta es Mahalanobis (arriba)."""
    linalg = _import_scipy_linalg()
    diff = mean1 - mean2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


# ══════════════════════════════════════════════════════════════════════════════
# ── §5 estructuras de datos ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReferenceCorpusStats:
    backend: str                 # "spectral" | "latent"
    domain_tag: str              # "same" | "cross"
    mean: np.ndarray
    cov: np.ndarray              # covarianza regularizada (§4.2) — necesaria para fad-batch
    cov_inv: np.ndarray
    calibration_mu: float
    calibration_sigma: float
    n_files: int
    embedding_dim: int


@dataclass
class FloorBaseline:
    mu: float
    sigma: float
    n_files: int


@dataclass
class ScoreResult:
    candidate_path: str
    backend: str
    domain_tag: str
    raw_distance: float
    score_absolute: float
    score_relative_to_floor: Optional[float]
    warning: Optional[str]

    def to_json_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# ── §4.4 caché del corpus de referencia ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _list_wavs(directory: str) -> List[Path]:
    wavs = sorted(Path(directory).glob("*.wav")) + sorted(Path(directory).glob("*.WAV"))
    return sorted(set(wavs))


def build_reference_cache(corpus_dir: str, backend, domain_tag: str,
                           cache_path: str, verbose: bool = True) -> ReferenceCorpusStats:
    """Extrae embeddings de todos los .wav de corpus_dir, calcula mean/cov (§4.2)
    y mu/sigma de calibración (§4.3), y lo guarda todo en un .npz. El corpus se
    procesa una sola vez — obligatorio por coste (más aún dentro de un loop de
    RL): puntuar candidatos nunca vuelve a tocar el corpus."""
    wavs = _list_wavs(corpus_dir)
    if len(wavs) < 2:
        raise InsufficientCorpusError(
            f"Corpus de referencia en {corpus_dir!r} tiene {len(wavs)} .wav — "
            "se necesitan al menos 2 para calcular estadísticas.")
    embeddings = []
    for i, w in enumerate(wavs, 1):
        if verbose:
            print(f"  [{i}/{len(wavs)}] {w.name}")
        embeddings.append(backend.extract(str(w)))
    embeddings = np.array(embeddings)
    mean, cov = compute_reference_stats(embeddings)
    cov_inv = np.linalg.pinv(cov)
    mu, sigma = calibrate_scale(embeddings)
    np.savez(cache_path, mean=mean, cov=cov, cov_inv=cov_inv,
              mu=np.float64(mu), sigma=np.float64(sigma),
              backend=backend.name, domain_tag=domain_tag,
              n_files=len(wavs), dim=embeddings.shape[1],
              timestamp=time.time(), version=CACHE_FORMAT_VERSION)
    return ReferenceCorpusStats(backend=backend.name, domain_tag=domain_tag, mean=mean,
                                 cov=cov, cov_inv=cov_inv, calibration_mu=mu,
                                 calibration_sigma=sigma, n_files=len(wavs),
                                 embedding_dim=embeddings.shape[1])


def load_reference_cache(cache_path: str, expected_backend: Optional[str] = None) -> ReferenceCorpusStats:
    """Carga un caché ya construido. Si expected_backend no coincide con el
    backend guardado en el caché, lanza CacheBackendMismatchError explícito —
    nunca un resultado numérico silenciosamente sin sentido (§4.4)."""
    if not Path(cache_path).exists():
        sys.exit(f"✗  Caché de corpus no encontrado: {cache_path} "
                  f"(constrúyelo primero con --build-cache)")
    data = np.load(cache_path, allow_pickle=False)
    backend_name = str(data["backend"])
    if expected_backend is not None and backend_name != expected_backend:
        raise CacheBackendMismatchError(
            f"El caché {cache_path!r} fue construido con backend {backend_name!r}, "
            f"pero se solicitó {expected_backend!r}. Los embeddings de ambos "
            "espacios no son comparables — reconstruye el caché con --build-cache "
            f"--backend {expected_backend}, o puntúa con --backend {backend_name}.")
    return ReferenceCorpusStats(
        backend=backend_name, domain_tag=str(data["domain_tag"]),
        mean=data["mean"], cov=data["cov"], cov_inv=data["cov_inv"],
        calibration_mu=float(data["mu"]), calibration_sigma=float(data["sigma"]),
        n_files=int(data["n_files"]), embedding_dim=int(data["dim"]))


# ══════════════════════════════════════════════════════════════════════════════
# ── §4.5 calibración por suelo (mitigación del hueco de dominio) ─────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_floor_baseline(floor_corpus_dir: str, ref_mean: np.ndarray, ref_cov_inv: np.ndarray,
                            backend, verbose: bool = True) -> Tuple[float, float, int]:
    """floor_corpus_dir contiene ejemplos representativos de renders PROPIOS
    pasados (mismo dominio que los candidatos futuros). Se calcula la distancia
    de cada uno al corpus de referencia real, y se devuelve (mu_floor,
    sigma_floor, n) — 'qué tan lejos caen normalmente mis propios renders,
    aunque sean buenos, solo por ser soundfont'."""
    wavs = _list_wavs(floor_corpus_dir)
    if len(wavs) < 2:
        raise InsufficientCorpusError(
            f"Floor corpus en {floor_corpus_dir!r} tiene {len(wavs)} .wav — "
            "se necesitan al menos 2.")
    distances = []
    for i, w in enumerate(wavs, 1):
        if verbose:
            print(f"  [floor {i}/{len(wavs)}] {w.name}")
        emb = backend.extract(str(w))
        distances.append(mahalanobis_distance(emb, ref_mean, ref_cov_inv))
    return float(np.mean(distances)), float(np.std(distances) + 1e-6), len(wavs)


def build_floor_cache(floor_corpus_dir: str, ref: ReferenceCorpusStats, backend,
                       floor_cache_path: Optional[str], verbose: bool = True) -> FloorBaseline:
    mu, sigma, n = compute_floor_baseline(floor_corpus_dir, ref.mean, ref.cov_inv, backend, verbose)
    floor = FloorBaseline(mu=mu, sigma=sigma, n_files=n)
    if floor_cache_path:
        np.savez(floor_cache_path, mu=np.float64(mu), sigma=np.float64(sigma),
                  n_files=n, backend=backend.name, timestamp=time.time(),
                  version=CACHE_FORMAT_VERSION)
    return floor


def load_floor_cache(floor_cache_path: str) -> FloorBaseline:
    data = np.load(floor_cache_path, allow_pickle=False)
    return FloorBaseline(mu=float(data["mu"]), sigma=float(data["sigma"]),
                          n_files=int(data["n_files"]))


# ══════════════════════════════════════════════════════════════════════════════
# ── scoring ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_WARNING = ("domain 'cross' sin floor calibrado — ver S1/S4.5 de la especificacion")


def score_against_corpus(candidate_path: str, cache_path: str, backend: str = "spectral",
                          floor_cache: Optional[str] = None,
                          audiolm_codec: Optional[str] = None) -> ScoreResult:
    """Punto de entrada principal para consumo programático (§7, módulo
    importable). Puntúa un único candidato contra un caché de corpus ya
    construido. Si floor_cache existe, añade score_relative_to_floor."""
    backend_obj = get_backend(backend, audiolm_codec=audiolm_codec)
    if not backend_obj.is_available():
        raise BackendUnavailableError(
            f"Backend {backend!r} no disponible (dependencia opcional ausente o "
            "checkpoint no encontrado).")
    ref = load_reference_cache(cache_path, expected_backend=backend)
    embedding = backend_obj.extract(candidate_path)
    distance = mahalanobis_distance(embedding, ref.mean, ref.cov_inv)
    score_abs = normalize_score(distance, ref.calibration_mu, ref.calibration_sigma)

    score_floor = None
    warning = None
    if floor_cache and Path(floor_cache).exists():
        floor = load_floor_cache(floor_cache)
        score_floor = normalize_score(distance, floor.mu, floor.sigma)
    elif ref.domain_tag == "cross":
        warning = _DOMAIN_WARNING

    return ScoreResult(candidate_path=str(candidate_path), backend=ref.backend,
                        domain_tag=ref.domain_tag, raw_distance=distance,
                        score_absolute=score_abs, score_relative_to_floor=score_floor,
                        warning=warning)


def rank_candidates(candidate_paths: List[str], cache_path: str, backend: str = "spectral",
                     floor_cache: Optional[str] = None,
                     audiolm_codec: Optional[str] = None) -> List[ScoreResult]:
    """--mode rank: puntúa varios candidatos y los ordena de mejor a peor (score
    relativo al floor si hay uno calibrado, si no score absoluto)."""
    results = [score_against_corpus(p, cache_path, backend, floor_cache, audiolm_codec)
               for p in candidate_paths]

    def _key(r: ScoreResult) -> float:
        return r.score_relative_to_floor if r.score_relative_to_floor is not None else r.score_absolute

    return sorted(results, key=_key, reverse=True)


def score_fad_batch(candidate_paths: List[str], cache_path: str, backend: str = "spectral",
                     audiolm_codec: Optional[str] = None) -> dict:
    """--mode fad-batch: Fréchet Distance real entre la distribución de varios
    candidatos y la distribución del corpus de referencia (§4.1) — distinto del
    modo por defecto de un único candidato, que usa Mahalanobis contra un punto."""
    if len(candidate_paths) < 2:
        raise InsufficientCorpusError(
            "--mode fad-batch necesita al menos 2 candidatos para formar una "
            "distribución del lado generado.")
    backend_obj = get_backend(backend, audiolm_codec=audiolm_codec)
    if not backend_obj.is_available():
        raise BackendUnavailableError(f"Backend {backend!r} no disponible.")
    ref = load_reference_cache(cache_path, expected_backend=backend)
    embeddings = np.array([backend_obj.extract(p) for p in candidate_paths])
    cand_mean, cand_cov = compute_reference_stats(embeddings)
    fad = frechet_distance(cand_mean, cand_cov, ref.mean, ref.cov)
    return {
        "candidates": [str(p) for p in candidate_paths],
        "backend": ref.backend,
        "domain_tag": ref.domain_tag,
        "n_candidates": len(candidate_paths),
        "frechet_distance": fad,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── §6 integración como EvalProvider (sistema RL de rl_pipeline_agent_spec.md) ─
# ══════════════════════════════════════════════════════════════════════════════

class AudioReferenceProvider:
    """Cuarto EvalProvider (§5.4.1 de rl_pipeline_agent_spec.md), junto a
    quality_scorer/preference_trainer/human — opera sobre WAV en vez de MIDI.
    cfg["audio_reference"] espera: cache_file, backend (opcional, default
    spectral), render_soundfont (timbre neutro para renderizar MIDI intermedio),
    floor_cache_file (opcional)."""
    name = "audio_reference"
    blocking = False

    def __init__(self, cfg: dict):
        section = cfg["audio_reference"]
        self.cache_path = section["cache_file"]
        self.backend_name = section.get("backend", "spectral")
        self.neutral_soundfont = section.get("render_soundfont")
        self.floor_cache_path = section.get("floor_cache_file")
        self.audiolm_codec = section.get("audiolm_codec")
        self._render_cache: dict = {}   # hash(MIDI) -> ruta WAV, ver nota de coste §6

    def is_available(self, ctx: dict) -> bool:
        return os.path.exists(self.cache_path)   # nunca intenta construir el caché sobre la marcha

    def score(self, candidate_path: str, ctx: dict) -> float:
        wav_path = self._ensure_audio(candidate_path, ctx)
        result = score_against_corpus(wav_path, self.cache_path, backend=self.backend_name,
                                       floor_cache=self.floor_cache_path,
                                       audiolm_codec=self.audiolm_codec)
        return (result.score_relative_to_floor if result.score_relative_to_floor is not None
                else result.score_absolute)

    def _ensure_audio(self, candidate_path: str, ctx: dict) -> str:
        """Si candidate_path es MIDI (etapas intermedias del pipeline), renderiza
        con timbre neutro fijo a un WAV temporal (aísla la decisión compositiva
        del timbre, ver §6). Si ya es WAV (etapa final de render), se usa tal
        cual. El render se cachea por hash del MIDI de entrada si se repite entre
        steps (nota de coste §6)."""
        suffix = Path(candidate_path).suffix.lower()
        if suffix in (".wav",):
            return candidate_path
        if suffix not in (".mid", ".midi"):
            raise ValueError(f"AudioReferenceProvider: extensión no soportada: {candidate_path}")

        import hashlib
        digest = hashlib.sha1(Path(candidate_path).read_bytes()).hexdigest()
        if digest in self._render_cache and Path(self._render_cache[digest]).exists():
            return self._render_cache[digest]

        out_path = str(Path(candidate_path).with_suffix("")) + f".{digest[:10]}.neutral.wav"
        _render_midi_neutral(candidate_path, out_path)
        self._render_cache[digest] = out_path
        return out_path


def _render_midi_neutral(midi_path: str, out_wav_path: str, sr: int = TARGET_SR) -> None:
    """Renderiza un MIDI a WAV con timbre neutro y fijo: síntesis aditiva mínima
    (armónicos con caída 1/n) + envolvente ADR simple, sin sample libraries ni
    soundfont externo — la instancia autocontenida más pequeña que sirve para
    aislar la decisión compositiva del timbre al puntuar etapas intermedias del
    pipeline de RL (§6). No pretende sonar bien, solo ser estable/reproducible
    entre steps."""
    mido = _import_mido()
    sf = _import_soundfile()
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat or 480

    # ── recolectar eventos note-on/off con tiempo absoluto en segundos, manejando
    # cambios de tempo (set_tempo) por pista, fusionando todas las pistas ──
    events = []   # (time_s, type, note, velocity)
    for track in mid.tracks:
        t_ticks = 0
        tempo = 500000   # microsegundos por beat, default MIDI (120 BPM)
        for msg in track:
            t_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            t_s = mido.tick2second(t_ticks, ticks_per_beat, tempo)
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((t_s, "on", msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                events.append((t_s, "off", msg.note, 0))
    if not events:
        # MIDI vacío o ilegible: escribe un WAV de silencio corto para no romper el pipeline.
        sf.write(out_wav_path, np.zeros(int(0.5 * sr)), sr)
        return

    events.sort(key=lambda e: e[0])
    open_notes: dict = {}
    notes = []   # (start_s, end_s, pitch, velocity)
    for t_s, kind, pitch, vel in events:
        if kind == "on":
            open_notes.setdefault(pitch, []).append((t_s, vel))
        else:
            if pitch in open_notes and open_notes[pitch]:
                start_s, vel0 = open_notes[pitch].pop(0)
                notes.append((start_s, max(t_s, start_s + 0.02), pitch, vel0))

    total_dur = max(n[1] for n in notes) + _NEUTRAL_RELEASE_S + 0.1
    buf = np.zeros(int(total_dur * sr) + sr)

    for start_s, end_s, pitch, vel in notes:
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        dur = max(end_s - start_s, 0.02)
        n_samples = int((dur + _NEUTRAL_RELEASE_S) * sr)
        t = np.arange(n_samples) / sr
        tone = np.zeros(n_samples)
        for h in range(1, _NEUTRAL_N_HARMONICS + 1):
            tone += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
        # envolvente ADR: attack lineal, sustain plano, release exponencial
        env = np.ones(n_samples)
        a_n = min(int(_NEUTRAL_ATTACK_S * sr), n_samples)
        r_n = min(int(_NEUTRAL_RELEASE_S * sr), n_samples)
        if a_n > 0:
            env[:a_n] = np.linspace(0, 1, a_n)
        if r_n > 0:
            env[-r_n:] *= np.exp(-3.0 * np.linspace(0, 1, r_n))
        amp = (vel / 127.0) * 0.25
        tone = tone * env * amp

        start_i = int(start_s * sr)
        end_i = start_i + n_samples
        if end_i > len(buf):
            buf = np.pad(buf, (0, end_i - len(buf)))
        buf[start_i:end_i] += tone

    peak = np.max(np.abs(buf)) + 1e-9
    if peak > 0.95:
        buf = buf * (0.95 / peak)
    sf.write(out_wav_path, buf, sr)


# ══════════════════════════════════════════════════════════════════════════════
# ── informes (texto / JSON) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_BAR = "═" * 55


def _c(text: str, code: str, no_color: bool) -> str:
    return text if no_color else f"\033[{code}m{text}\033[0m"


def print_single_report(result: ScoreResult, cache_path: str, embedding_dim: int,
                         n_files: int, no_color: bool = False) -> None:
    print(_BAR)
    print(_c("  AUDIO REFERENCE SCORER", "1;36", no_color))
    print(_BAR)
    print(f"  Candidato       : {result.candidate_path}")
    print(f"  Backend         : {result.backend} ({embedding_dim} dim)")
    print(f"  Corpus          : {cache_path} ({n_files} archivos, dominio: {result.domain_tag})")
    if result.warning:
        print()
        print(_c(f"  ⚠ Corpus de dominio '{result.domain_tag}' sin --floor-corpus calibrado.", "33", no_color))
        print("    El score absoluto puede estar dominado por diferencias de")
        print("    produccion/timbre entre soundfont y grabacion real, no por")
        print("    calidad musical. Considera anadir --floor-corpus.")
    print()
    print(f"  Distancia Mahalanobis : {result.raw_distance:.2f}")
    print(f"  Score absoluto         : {result.score_absolute:.2f}")
    if result.score_relative_to_floor is not None:
        print(f"  Score relativo a floor : {result.score_relative_to_floor:.2f}")
    print(_BAR)


def print_rank_report(results: List[ScoreResult], no_color: bool = False) -> None:
    print(_BAR)
    print(_c("  AUDIO REFERENCE SCORER — RANK", "1;36", no_color))
    print(_BAR)
    for i, r in enumerate(results, 1):
        key = r.score_relative_to_floor if r.score_relative_to_floor is not None else r.score_absolute
        label = "floor" if r.score_relative_to_floor is not None else "absoluto"
        print(f"  {i}. {Path(r.candidate_path).name:<30} score({label})={key:.3f}  dist={r.raw_distance:.2f}")
    print(_BAR)


def print_fad_report(fad_result: dict, no_color: bool = False) -> None:
    print(_BAR)
    print(_c("  AUDIO REFERENCE SCORER — FAD BATCH", "1;36", no_color))
    print(_BAR)
    print(f"  Candidatos      : {fad_result['n_candidates']}")
    print(f"  Backend         : {fad_result['backend']}")
    print(f"  Corpus dominio  : {fad_result['domain_tag']}")
    print(f"  Frechet distance: {fad_result['frechet_distance']:.4f}")
    print(_BAR)


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_build_cache(args) -> None:
    if not args.domain_tag:
        sys.exit("✗  --domain-tag {same,cross} es obligatorio con --build-cache (ver §1/§7).")
    if not args.cache:
        sys.exit("✗  --cache FILE es obligatorio con --build-cache.")
    backend = get_backend(args.backend, audiolm_codec=args.audiolm_codec)
    if not backend.is_available():
        sys.exit(f"✗  Backend {args.backend!r} no disponible — revisa dependencias opcionales.")
    if not args.quiet:
        print(f"  [build-cache] {args.build_cache} → {args.cache}  (backend={args.backend}, "
              f"domain={args.domain_tag})")
    ref = build_reference_cache(args.build_cache, backend, args.domain_tag, args.cache,
                                 verbose=not args.quiet)
    print(f"  ✓  Caché de corpus construido → {args.cache}  "
          f"({ref.n_files} archivos, {ref.embedding_dim} dim, mu={ref.calibration_mu:.2f}, "
          f"sigma={ref.calibration_sigma:.2f})")

    if args.floor_corpus:
        floor = build_floor_cache(args.floor_corpus, ref, backend, args.floor_cache,
                                   verbose=not args.quiet)
        dest = args.floor_cache or "(no guardado — pasa --floor-cache para persistirlo)"
        print(f"  ✓  Calibración de suelo construida → {dest}  "
              f"({floor.n_files} archivos, mu={floor.mu:.2f}, sigma={floor.sigma:.2f})")

    if args.plot:
        _plot_corpus(args.build_cache, backend, args.cache.rsplit(".", 1)[0] + "_pca.png")


def _plot_corpus(corpus_dir: str, backend, out_png: str) -> None:
    """Proyección PCA 2D del corpus (solo diagnóstico visual, --plot). PCA manual
    vía SVD — no añade dependencia de scikit-learn."""
    plt = _import_matplotlib()
    wavs = _list_wavs(corpus_dir)
    embeddings = np.array([backend.extract(str(w)) for w in wavs])
    centered = embeddings - embeddings.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:, 0], proj[:, 1])
    for i, w in enumerate(wavs):
        plt.annotate(w.stem, proj[i], fontsize=6)
    plt.title("Corpus de referencia — proyección PCA 2D")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"  ✓  Proyección PCA → {out_png}")


def _cmd_score(args) -> None:
    if not args.cache:
        sys.exit("✗  --cache FILE es obligatorio en modo scoring.")
    if not args.candidates:
        sys.exit("✗  Falta al menos un WAV candidato.")
    if args.backend == "latent" and not args.audiolm_codec:
        sys.exit("✗  --backend latent requiere --audiolm-codec FILE.")

    ref_meta = load_reference_cache(args.cache, expected_backend=args.backend)

    # Si se pidió --floor-corpus y aún no hay --floor-cache construido, se construye
    # ahora (una sola vez) y se reutiliza su ruta en el resto del scoring — ver §4.5/§7.
    if args.floor_corpus and not (args.floor_cache and Path(args.floor_cache).exists()):
        backend_obj = get_backend(args.backend, audiolm_codec=args.audiolm_codec)
        if not backend_obj.is_available():
            sys.exit(f"✗  Backend {args.backend!r} no disponible — revisa dependencias opcionales.")
        if not args.quiet:
            print(f"  [floor] construyendo calibración de suelo desde {args.floor_corpus}")
        floor = build_floor_cache(args.floor_corpus, ref_meta, backend_obj, args.floor_cache,
                                   verbose=not args.quiet)
        if not args.floor_cache:
            # Sin ruta persistente: se escribe a un fichero temporal junto al candidato
            # para que score_against_corpus/rank_candidates (que trabajan por ruta de
            # fichero, no por objeto en memoria) puedan cargarlo igualmente.
            tmp_floor = str(Path(args.cache).with_suffix("")) + ".floor_tmp.npz"
            np.savez(tmp_floor, mu=np.float64(floor.mu), sigma=np.float64(floor.sigma),
                      n_files=floor.n_files, backend=args.backend, timestamp=time.time(),
                      version=CACHE_FORMAT_VERSION)
            args.floor_cache = tmp_floor
        if not args.quiet:
            print(f"  ✓  Calibración de suelo lista (mu={floor.mu:.2f}, sigma={floor.sigma:.2f})")

    if args.mode == "fad-batch":
        result = score_fad_batch(args.candidates, args.cache, backend=args.backend,
                                  audiolm_codec=args.audiolm_codec)
        if not args.quiet:
            print_fad_report(result, no_color=args.no_color)
        else:
            print(f"{result['frechet_distance']:.4f}")
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.mode == "rank":
        if len(args.candidates) < 2:
            sys.exit("✗  --mode rank necesita al menos 2 candidatos.")
        results = rank_candidates(args.candidates, args.cache, backend=args.backend,
                                   floor_cache=args.floor_cache, audiolm_codec=args.audiolm_codec)
        if not args.quiet:
            print_rank_report(results, no_color=args.no_color)
        else:
            for r in results:
                key = r.score_relative_to_floor if r.score_relative_to_floor is not None else r.score_absolute
                print(f"{key:.4f}  {r.candidate_path}")
        if args.json:
            Path(args.json).write_text(json.dumps([r.to_json_dict() for r in results],
                                                    indent=2, ensure_ascii=False))
        return

    # mode == "single"
    if len(args.candidates) != 1:
        sys.exit("✗  --mode single (default) espera exactamente 1 candidato — "
                  "usa --mode rank o --mode fad-batch para varios.")
    result = score_against_corpus(args.candidates[0], args.cache, backend=args.backend,
                                   floor_cache=args.floor_cache, audiolm_codec=args.audiolm_codec)
    if args.quiet:
        key = result.score_relative_to_floor if result.score_relative_to_floor is not None else result.score_absolute
        print(f"{key:.4f}")
    else:
        print_single_report(result, args.cache, ref_meta.embedding_dim, ref_meta.n_files,
                             no_color=args.no_color)
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        prog="audio_reference_scorer",
        description="Score 0-1 de un WAV candidato contra un corpus de referencia "
                     "(distancia de Mahalanobis en espacio de features de audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("candidates", nargs="*",
                         help="WAV(s) candidato(s) a puntuar (modo scoring)")
    parser.add_argument("--build-cache", metavar="DIR", default=None,
                         help="Construye el caché de corpus de referencia desde todos los "
                              ".wav en DIR (modo separado, no requiere candidate)")
    parser.add_argument("--cache", metavar="FILE", default=None,
                         help="Ruta al caché de corpus de referencia (.npz)")
    parser.add_argument("--backend", choices=["spectral", "latent"], default="spectral",
                         help="Backend de features (default: spectral)")
    parser.add_argument("--audiolm-codec", metavar="FILE", default=None,
                         help="Checkpoint de audiolm.py, requerido solo si --backend latent")
    parser.add_argument("--domain-tag", choices=["same", "cross"], default=None,
                         help="Obligatorio en --build-cache: relación de dominio entre el "
                              "corpus de referencia y los candidatos futuros")
    parser.add_argument("--floor-corpus", metavar="DIR", default=None,
                         help="Directorio de renders propios pasados para calibración por "
                              "suelo (§4.5, mitigación del hueco de dominio)")
    parser.add_argument("--floor-cache", metavar="FILE", default=None,
                         help="Caché de la calibración por suelo — se construye o se "
                              "reutiliza según exista o no")
    parser.add_argument("--mode", choices=["single", "rank", "fad-batch"], default="single",
                         help="single: 1 candidato, 1 score. rank: varios candidatos "
                              "ordenados. fad-batch: Fréchet Distance real entre la "
                              "distribución de candidatos y la referencia")
    parser.add_argument("--json", metavar="FILE", default=None,
                         help="Sidecar JSON con el/los ScoreResult completo(s)")
    parser.add_argument("--quiet", action="store_true",
                         help="Solo imprime el/los score(s) final(es) (consumo por script/RL)")
    parser.add_argument("--no-color", action="store_true",
                         help="Desactiva colores ANSI en el informe de texto")
    parser.add_argument("--plot", action="store_true",
                         help="(solo con --build-cache) guarda una proyección PCA 2D del "
                              "corpus como PNG — requiere matplotlib")
    args = parser.parse_args()

    try:
        if args.build_cache:
            _cmd_build_cache(args)
        else:
            _cmd_score(args)
    except SystemExit:
        raise
    except (BackendUnavailableError, CacheBackendMismatchError, InsufficientCorpusError) as e:
        sys.exit(f"✗  {e}")
    except FileNotFoundError as e:
        sys.exit(f"✗  Fichero no encontrado: {e.filename or e}")
    except (OSError, IOError) as e:
        sys.exit(f"✗  Error de E/S accediendo a {getattr(e, 'filename', None) or 'un fichero'}: {e}")
    except Exception as e:
        if type(e).__name__ in ("LibsndfileError", "SoundFileError", "SoundFileRuntimeError"):
            sys.exit(f"✗  Error leyendo/escribiendo audio: {e}")
        raise


if __name__ == "__main__":
    main()
