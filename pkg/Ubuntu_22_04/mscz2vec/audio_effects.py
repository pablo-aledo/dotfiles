#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          AUDIO EFFECTS  v1.1                                 ║
║  Librería DSP + CLI: aplica una cadena de efectos YA DADA a un WAV           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Aplica una cadena de efectos — EQ paramétrico, compresor, reverb,        ║
║    limitador, delay, saturación, chorus, trémolo, puerta de ruido,          ║
║    de-esser — conocida de antemano: de un JSON, de flags de línea de       ║
║    comandos, o de una llamada programática desde otro programa (p.ej.       ║
║    mix_evolver.py) — a un WAV de entrada. No genera, no evalúa, no busca:   ║
║    solo aplica. Ver mix_evolver.py para la búsqueda evolutiva de cadenas.   ║
║                                                                              ║
║  ALCANCE                                                                     ║
║    Opera sobre la mezcla completa ya renderizada (mono o estéreo,           ║
║    post-render), no sobre pistas individuales por instrumento.              ║
║                                                                              ║
║  USO                                                                        ║
║    audio_effects.py dry.wav --chain chain.json --out processed.wav          ║
║    audio_effects.py dry.wav --eq 1000:3.0:1.2 --compressor -18:3.0:10:120 \\║
║                     --out processed.wav                                     ║
║    audio_effects.py dry.wav --saturation 8:0.5 --delay 250:0.35:0.25 \\    ║
║                     --de-esser 6500:-22:4:2.5 --out processed.wav           ║
║    audio_effects.py dry.wav --chain chain.json --out processed.wav \\      ║
║                     --json report.json                                      ║
║                                                                              ║
║  DEPENDENCIAS  numpy  scipy  soundfile   (obligatorias)                     ║
║                                                                              ║
║  LIMITACIONES (ver especificación §5)                                       ║
║    · Solo opera sobre la mezcla ya renderizada, nunca a nivel de pista.     ║
║    · Los cuatro efectos son implementaciones simples y funcionales, no de   ║
║      calidad de producción profesional — suficientes para dar a un GA un    ║
║      espacio de búsqueda razonable, no pensadas como reemplazo de un DAW.   ║
║    · Estéreo: cada canal se procesa de forma independiente (sin linking     ║
║      de compresor/limitador entre canales); el reverb usa la misma IR       ║
║      (misma semilla) en ambos canales.                                      ║
║                                                                              ║
║  Módulo importable:                                                         ║
║    from audio_effects import apply_chain, EFFECT_FUNCTIONS, load_manifest, \\║
║                               EffectGene, Chain                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

EJEMPLOS DE chain.json PARA SITUACIONES HABITUALES
────────────────────────────────────────────────────
Cada bloque es un chain.json completo, listo para `--chain FILE`. Todos los
parámetros están dentro de los rangos del manifest por defecto (§1.3), así
que también son puntos de partida razonables para inicializar/acotar una
búsqueda con mix_evolver.py. Los `gain_db`/`threshold_db`/etc. son de
partida, no verdad revelada — ajustar oído en mano.

1) Mastering suave / transparente — realce sutil de agudos, compresión
   ligera para controlar picos ocasionales, limitador solo de seguridad:
   [
     {"effect": "eq_peaking", "params": {"freq_hz": 9000, "gain_db": 2.0, "q": 0.7}},
     {"effect": "compressor", "params": {"threshold_db": -12, "ratio": 2.0,
                                          "attack_ms": 20, "release_ms": 250}},
     {"effect": "limiter", "params": {"ceiling_db": -1.0}}
   ]

2) Punch pop/rock — scoop de medios para dejar hueco a voz/caja, compresor
   rápido para pegada, limitador cercano al techo para sonar competitivo:
   [
     {"effect": "eq_peaking", "params": {"freq_hz": 400, "gain_db": -3.5, "q": 1.2}},
     {"effect": "eq_peaking", "params": {"freq_hz": 3000, "gain_db": 2.5, "q": 1.0}},
     {"effect": "compressor", "params": {"threshold_db": -18, "ratio": 4.0,
                                          "attack_ms": 5, "release_ms": 80}},
     {"effect": "limiter", "params": {"ceiling_db": -0.3}}
   ]

3) Ambient espacioso — calidez en graves, reverb grande y prominente,
   sin apenas compresión para preservar la dinámica natural:
   [
     {"effect": "eq_peaking", "params": {"freq_hz": 200, "gain_db": 2.0, "q": 0.8}},
     {"effect": "reverb", "params": {"size": 2.5, "decay": 3.0, "mix": 0.45}},
     {"effect": "limiter", "params": {"ceiling_db": -1.5}}
   ]

4) Lo-fi / "cinta vieja" — agudos cortados, compresión pegajosa y lenta,
   un toque de reverb corta para simular ambiente cerrado:
   [
     {"effect": "eq_peaking", "params": {"freq_hz": 8000, "gain_db": -9.0, "q": 0.5}},
     {"effect": "compressor", "params": {"threshold_db": -24, "ratio": 6.0,
                                          "attack_ms": 40, "release_ms": 400}},
     {"effect": "reverb", "params": {"size": 0.4, "decay": 5.0, "mix": 0.15}}
   ]

5) Brickwall loud (masterización estilo streaming, muy comprimida) —
   compresión agresiva en dos etapas + limitador pegado al techo:
   [
     {"effect": "compressor", "params": {"threshold_db": -28, "ratio": 3.0,
                                          "attack_ms": 10, "release_ms": 120}},
     {"effect": "eq_peaking", "params": {"freq_hz": 90, "gain_db": -4.0, "q": 0.9}},
     {"effect": "compressor", "params": {"threshold_db": -10, "ratio": 8.0,
                                          "attack_ms": 2, "release_ms": 60}},
     {"effect": "limiter", "params": {"ceiling_db": -0.1}}
   ]

6) Identidad (control/baseline) — cadena vacía, útil como referencia de
   "sin procesar" al comparar contra cualquiera de las anteriores:
   []
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ── lazy imports (mismo patrón que el resto del ecosistema) ───────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _import_scipy_signal():
    try:
        from scipy import signal
        return signal
    except ImportError:
        sys.exit("✗  scipy no encontrado. Instala con: pip install scipy")


def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. Instala con: pip install soundfile")


# ══════════════════════════════════════════════════════════════════════════════
# ── §1.2 efectos DSP (autocontenidos, numpy/scipy) ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada función pública opera sobre audio mono (1D) o estéreo (2D, forma
# (n_samples, n_channels)). El núcleo matemático de cada efecto (fórmulas RBJ
# Audio EQ Cookbook para el EQ, envelope follower para el compresor,
# convolución con IR sintética para el reverb, lookahead brick-wall para el
# limitador) es exactamente el dado por la especificación; para estéreo se
# aplica canal a canal de forma independiente (ver limitaciones arriba).

def _per_channel(audio: np.ndarray, mono_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Aplica mono_fn a cada canal si audio es 2D (n_samples, n_channels);
    si es 1D, aplica directamente. No asume nada del resto de la cadena."""
    if audio.ndim == 2:
        return np.stack([mono_fn(audio[:, ch]) for ch in range(audio.shape[1])], axis=1)
    return mono_fn(audio)


def eq_peaking(audio: np.ndarray, sr: int, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
    """EQ paramétrico peaking, filtro biquad, fórmulas RBJ Audio EQ Cookbook."""
    signal = _import_scipy_signal()
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    def _mono(x):
        return signal.lfilter(b, a, x)

    return _per_channel(audio, _mono)


def _compressor_mono(audio: np.ndarray, sr: int, threshold_db: float, ratio: float,
                      attack_ms: float, release_ms: float) -> np.ndarray:
    eps = 1e-8
    env_db = 20 * np.log10(np.abs(audio) + eps)
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000))
    smoothed = np.zeros_like(env_db)
    for i in range(1, len(env_db)):
        coef = attack_coef if env_db[i] > smoothed[i - 1] else release_coef
        smoothed[i] = coef * smoothed[i - 1] + (1 - coef) * env_db[i]
    over = np.maximum(smoothed - threshold_db, 0)
    gain_reduction_db = over * (1 - 1 / ratio)
    gain_linear = 10 ** (-gain_reduction_db / 20)
    return audio * gain_linear


def compressor(audio: np.ndarray, sr: int, threshold_db: float, ratio: float,
                attack_ms: float, release_ms: float) -> np.ndarray:
    """Compresor de rango dinámico: envelope follower (bucle secuencial
    intencional, depende de la muestra anterior — no vectorizar salvo que el
    rendimiento resulte insuficiente en la práctica) + reducción de ganancia."""
    return _per_channel(audio, lambda x: _compressor_mono(
        x, sr, threshold_db, ratio, attack_ms, release_ms))


def reverb(audio: np.ndarray, sr: int, size: float, decay: float, mix: float,
           seed: int = 42) -> np.ndarray:
    """Reverb por convolución con un impulso sintético (ruido con envolvente
    exponencial) — sin depender de bancos de IR reales. `seed` con valor por
    defecto fijo y determinista es obligatorio (§1.5 de la especificación):
    es lo que permite a mix_evolver.py evaluar fitness de forma reproducible
    sin tener que gestionar semillas él mismo."""
    signal = _import_scipy_signal()
    rng = np.random.default_rng(seed)
    ir_len = int(size * sr)
    noise = rng.standard_normal(ir_len)
    envelope = np.exp(-decay * np.linspace(0, size, ir_len))
    ir = noise * envelope
    ir /= np.max(np.abs(ir)) + 1e-8

    def _mono(x):
        wet = signal.fftconvolve(x, ir)[: len(x)]
        return (1 - mix) * x + mix * wet

    return _per_channel(audio, _mono)


def _limiter_mono(audio: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20)
    lookahead = int(sr * lookahead_ms / 1000)
    gain = np.ones_like(audio)
    abs_audio = np.abs(audio)
    for i in range(len(audio)):
        window = abs_audio[i: i + lookahead]
        peak = window.max() if len(window) else abs_audio[i]
        if peak > ceiling:
            gain[i] = ceiling / (peak + 1e-8)
    return audio * gain


def limiter(audio: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float = 5.0) -> np.ndarray:
    """Limitador brick-wall con lookahead simple."""
    return _per_channel(audio, lambda x: _limiter_mono(x, sr, ceiling_db, lookahead_ms))


# ── §1.2b efectos nuevos (Fase 1) ──────────────────────────────────────────────
# Mismo contrato que los cuatro anteriores: función mono cerrada sobre los
# coeficientes/parámetros + _per_channel para estéreo. Ninguno de estos toca
# la arquitectura "cada canal independiente" — eso es Fase 2.

def delay(audio: np.ndarray, sr: int, delay_ms: float, feedback: float, mix: float) -> np.ndarray:
    """Eco con feedback: línea de retardo realimentada, implementada como
    filtro IIR de comb (b=[1], a con un único polo en delay_ms con
    coeficiente -feedback). Determinista, sin estado más allá del propio
    filtro. `feedback` debe mantenerse < 1 para estabilidad (ver manifest)."""
    signal = _import_scipy_signal()
    delay_samples = max(1, int(sr * delay_ms / 1000))
    b = np.array([1.0])
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[-1] = -feedback

    def _mono(x):
        wet = signal.lfilter(b, a, x)
        return (1 - mix) * x + mix * wet

    return _per_channel(audio, _mono)


def saturation(audio: np.ndarray, sr: int, drive_db: float, mix: float) -> np.ndarray:
    """Saturación armónica: waveshaper tanh con `drive_db` controlando cuánto
    se empuja la señal hacia la zona no lineal, y normalización de nivel
    (dividir por tanh(drive)) para que a poco drive el efecto sea ~unitario
    en vez de simplemente atenuar. `mix` permite saturación paralela."""
    drive = max(10 ** (drive_db / 20), 1e-6)
    norm = np.tanh(drive)

    def _mono(x):
        wet = np.tanh(x * drive) / norm
        return (1 - mix) * x + mix * wet

    return _per_channel(audio, _mono)


def _modulated_delay_mono(x: np.ndarray, sr: int, rate_hz: float, depth_ms: float,
                           mix: float, center_ms: float) -> np.ndarray:
    """Línea de retardo de longitud variable (LFO senoidal) con interpolación
    lineal — núcleo compartido de chorus (y reutilizable por flanger/phaser
    si se añaden más adelante). No usa lfilter porque el retardo no es fijo."""
    n = len(x)
    t = np.arange(n) / sr
    lfo = np.sin(2 * np.pi * rate_hz * t)
    delay_samples = (center_ms + depth_ms * lfo) / 1000 * sr
    src_idx = np.clip(np.arange(n) - delay_samples, 0, n - 1)
    wet = np.interp(src_idx, np.arange(n), x)
    return (1 - mix) * x + mix * wet


def chorus(audio: np.ndarray, sr: int, rate_hz: float, depth_ms: float, mix: float) -> np.ndarray:
    """Chorus: retardo modulado por LFO en torno a un centro fijo de 15ms
    (rango típico de chorus, por debajo de la percepción de eco discreto)."""
    return _per_channel(audio, lambda x: _modulated_delay_mono(
        x, sr, rate_hz, depth_ms, mix, center_ms=15.0))


def tremolo(audio: np.ndarray, sr: int, rate_hz: float, depth: float, mix: float) -> np.ndarray:
    """Trémolo: modulación de amplitud por LFO senoidal. `depth` 0=sin efecto,
    1=silencio total en los valles del LFO."""
    def _mono(x):
        n = len(x)
        t = np.arange(n) / sr
        lfo = 1 - depth * (0.5 - 0.5 * np.cos(2 * np.pi * rate_hz * t))
        wet = x * lfo
        return (1 - mix) * x + mix * wet

    return _per_channel(audio, _mono)


def _noise_gate_mono(audio: np.ndarray, sr: int, threshold_db: float, attack_ms: float,
                      release_ms: float, range_db: float) -> np.ndarray:
    """Mismo envelope follower que _compressor_mono (bucle secuencial
    intencional, ver nota de compressor), pero la reducción de ganancia es
    proporcional a cuánto cae la envolvente POR DEBAJO del threshold, no por
    encima — es el inverso conceptual del compresor. `range_db` acota la
    atenuación máxima (gate real, no silencio absoluto salvo range_db alto)."""
    eps = 1e-8
    env_db = 20 * np.log10(np.abs(audio) + eps)
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000))
    smoothed = np.zeros_like(env_db)
    for i in range(1, len(env_db)):
        coef = attack_coef if env_db[i] > smoothed[i - 1] else release_coef
        smoothed[i] = coef * smoothed[i - 1] + (1 - coef) * env_db[i]
    under = np.maximum(threshold_db - smoothed, 0)
    gain_reduction_db = np.minimum(under, range_db)
    gain_linear = 10 ** (-gain_reduction_db / 20)
    return audio * gain_linear


def noise_gate(audio: np.ndarray, sr: int, threshold_db: float, attack_ms: float,
               release_ms: float, range_db: float) -> np.ndarray:
    """Puerta de ruido / expansor descendente según range_db (bajo = expansor
    suave, alto ~80dB = gate casi total)."""
    return _per_channel(audio, lambda x: _noise_gate_mono(
        x, sr, threshold_db, attack_ms, release_ms, range_db))


def de_esser(audio: np.ndarray, sr: int, freq_hz: float, threshold_db: float,
             ratio: float, q: float) -> np.ndarray:
    """De-esser: compresor de banda. Filtra la señal a una banda estrecha
    centrada en freq_hz (biquad bandpass RBJ, ganancia de pico constante 0dB)
    para aislar la energía de sibilantes, calcula la reducción de ganancia
    SOLO sobre esa banda (envelope follower con attack/release fijos y
    rápidos: 5ms/80ms, típico de-esser — no expuestos como parámetro para no
    inflar el flag CLI) y aplica esa reducción a la señal completa, no solo
    a la banda filtrada."""
    signal = _import_scipy_signal()
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = alpha, 0.0, -alpha
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    attack_coef = np.exp(-1.0 / (sr * 5 / 1000))
    release_coef = np.exp(-1.0 / (sr * 80 / 1000))

    def _mono(x):
        band = signal.lfilter(b, a, x)
        eps = 1e-8
        env_db = 20 * np.log10(np.abs(band) + eps)
        smoothed = np.zeros_like(env_db)
        for i in range(1, len(env_db)):
            coef = attack_coef if env_db[i] > smoothed[i - 1] else release_coef
            smoothed[i] = coef * smoothed[i - 1] + (1 - coef) * env_db[i]
        over = np.maximum(smoothed - threshold_db, 0)
        gain_reduction_db = over * (1 - 1 / ratio)
        gain_linear = 10 ** (-gain_reduction_db / 20)
        return x * gain_linear

    return _per_channel(audio, _mono)


EFFECT_FUNCTIONS: Dict[str, Callable] = {
    "eq_peaking": eq_peaking,
    "compressor": compressor,
    "reverb": reverb,
    "limiter": limiter,
    "delay": delay,
    "saturation": saturation,
    "chorus": chorus,
    "tremolo": tremolo,
    "noise_gate": noise_gate,
    "de_esser": de_esser,
}

# Flags CLI de aplicación manual → (nombre de efecto, orden de parámetros
# posicionales tal como se pasan separados por ':').
_CLI_FLAG_TO_EFFECT = {
    "--eq": ("eq_peaking", ("freq_hz", "gain_db", "q")),
    "--compressor": ("compressor", ("threshold_db", "ratio", "attack_ms", "release_ms")),
    "--reverb": ("reverb", ("size", "decay", "mix")),
    "--limiter": ("limiter", ("ceiling_db",)),
    "--delay": ("delay", ("delay_ms", "feedback", "mix")),
    "--saturation": ("saturation", ("drive_db", "mix")),
    "--chorus": ("chorus", ("rate_hz", "depth_ms", "mix")),
    "--tremolo": ("tremolo", ("rate_hz", "depth", "mix")),
    "--noise-gate": ("noise_gate", ("threshold_db", "attack_ms", "release_ms", "range_db")),
    "--de-esser": ("de_esser", ("freq_hz", "threshold_db", "ratio", "q")),
}


# ══════════════════════════════════════════════════════════════════════════════
# ── §1.3 manifest de efectos (declarativo, extensible) ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MANIFEST: dict = {
    "effects": [
        {"name": "eq_peaking", "params": {
            "freq_hz": [80, 12000], "gain_db": [-12, 12], "q": [0.3, 5.0]}},
        {"name": "compressor", "params": {
            "threshold_db": [-40, 0], "ratio": [1.0, 10.0],
            "attack_ms": [1, 100], "release_ms": [20, 500]}},
        {"name": "reverb", "params": {
            "size": [0.1, 3.0], "decay": [0.5, 8.0], "mix": [0.0, 0.6]}},
        {"name": "limiter", "params": {"ceiling_db": [-3.0, -0.1]}},
        {"name": "delay", "params": {
            "delay_ms": [10, 800], "feedback": [0.0, 0.85], "mix": [0.0, 0.7]}},
        {"name": "saturation", "params": {
            "drive_db": [0.0, 24.0], "mix": [0.0, 1.0]}},
        {"name": "chorus", "params": {
            "rate_hz": [0.1, 5.0], "depth_ms": [1.0, 8.0], "mix": [0.0, 0.6]}},
        {"name": "tremolo", "params": {
            "rate_hz": [0.5, 12.0], "depth": [0.0, 1.0], "mix": [0.0, 1.0]}},
        {"name": "noise_gate", "params": {
            "threshold_db": [-60, -10], "attack_ms": [0.1, 50], "release_ms": [20, 500],
            "range_db": [6, 80]}},
        {"name": "de_esser", "params": {
            "freq_hz": [3000, 10000], "threshold_db": [-40, -5], "ratio": [1.5, 10.0],
            "q": [0.7, 6.0]}},
    ],
    "max_chain_length": 5,
    "min_chain_length": 1,
}


class ManifestError(ValueError):
    """El manifest de efectos (por defecto o cargado de un JSON) no tiene una
    estructura válida — error duro, nunca se corrige silenciosamente porque
    determina el espacio de búsqueda del GA en mix_evolver.py."""


def _validate_manifest(manifest: dict, source: str) -> dict:
    if "effects" not in manifest or not isinstance(manifest["effects"], list):
        raise ManifestError(f"Manifest inválido ({source}): falta 'effects' (lista).")
    if not manifest["effects"]:
        raise ManifestError(f"Manifest inválido ({source}): 'effects' está vacío.")
    for entry in manifest["effects"]:
        name = entry.get("name")
        if name not in EFFECT_FUNCTIONS:
            raise ManifestError(
                f"Manifest inválido ({source}): efecto {name!r} no está en "
                f"EFFECT_FUNCTIONS ({sorted(EFFECT_FUNCTIONS)}).")
        params = entry.get("params")
        if not isinstance(params, dict) or not params:
            raise ManifestError(
                f"Manifest inválido ({source}): efecto {name!r} sin 'params' (dict).")
        for pname, prange in params.items():
            if (not isinstance(prange, (list, tuple)) or len(prange) != 2
                    or prange[0] > prange[1]):
                raise ManifestError(
                    f"Manifest inválido ({source}): rango de {name}.{pname} debe ser "
                    f"[lo, hi] con lo <= hi, recibido {prange!r}.")
    min_len = manifest.get("min_chain_length")
    max_len = manifest.get("max_chain_length")
    if not isinstance(min_len, int) or not isinstance(max_len, int) or min_len < 1 or min_len > max_len:
        raise ManifestError(
            f"Manifest inválido ({source}): min_chain_length/max_chain_length "
            f"deben ser enteros con 1 <= min <= max (recibido min={min_len!r}, max={max_len!r}).")
    return manifest


def load_manifest(path: Optional[str] = None) -> dict:
    """Carga y valida un manifest de efectos. Sin `path`, devuelve el manifest
    por defecto de §1.3. Este manifest vive y se mantiene aquí, no en
    mix_evolver.py: cuando el GA lo necesita, lo importa desde este módulo."""
    if path is None:
        import copy
        return copy.deepcopy(DEFAULT_MANIFEST)
    data = json.loads(Path(path).read_text())
    return _validate_manifest(data, source=str(path))


# ══════════════════════════════════════════════════════════════════════════════
# ── §1.4 representación de una cadena de efectos (contrato compartido) ────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EffectGene:
    effect: str
    params: Dict[str, float]


Chain = List[EffectGene]   # una cadena concreta, ya decidida — no un genoma "en evolución"


def chain_to_dicts(chain: Chain) -> list:
    """Serializa una Chain al formato JSON compartido (§1.4/1.6): lista de
    {"effect": ..., "params": {...}}. Formato consumido tanto por la CLI de
    este programa (--chain) como escrito por mix_evolver.py en best_chain.json."""
    return [asdict(gene) for gene in chain]


def chain_from_dicts(data: list) -> Chain:
    """Deserializa una Chain desde el formato JSON compartido. Único punto de
    parseo — tanto la CLI de este módulo como mix_evolver.py deben pasar por
    aquí para no divergir en el formato."""
    chain = []
    for i, entry in enumerate(data):
        if "effect" not in entry or "params" not in entry:
            raise ValueError(
                f"Entrada {i} de la cadena inválida: falta 'effect' o 'params' ({entry!r}).")
        if entry["effect"] not in EFFECT_FUNCTIONS:
            raise ValueError(
                f"Entrada {i}: efecto {entry['effect']!r} desconocido "
                f"(disponibles: {sorted(EFFECT_FUNCTIONS)}).")
        chain.append(EffectGene(effect=entry["effect"], params=dict(entry["params"])))
    return chain


def load_chain(path: str) -> Chain:
    """Carga una Chain desde un fichero JSON (formato de §1.4)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: se esperaba una lista de EffectGene en la raíz del JSON.")
    return chain_from_dicts(data)


def save_chain(chain: Chain, path: str) -> None:
    """Escribe una Chain a JSON en el formato de §1.4/1.6 — el mismo que
    espera `--chain` de la CLI de este programa."""
    Path(path).write_text(json.dumps(chain_to_dicts(chain), indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
# ── §1.5 aplicación de una cadena — función central, reutilizada por ambos ────
#         programas (audio_effects.py y mix_evolver.py)
# ══════════════════════════════════════════════════════════════════════════════

def apply_chain(chain: Chain, audio: np.ndarray, sr: int) -> np.ndarray:
    """Aplica cada efecto de la cadena en orden secuencial. Determinista: para
    el mismo `chain` y el mismo audio de entrada, produce siempre el mismo
    resultado (el reverb usa la semilla fija de §1.2, no hay ninguna otra
    fuente de aleatoriedad). Cadena vacía = identidad."""
    result = audio.copy()
    for gene in chain:
        fn = EFFECT_FUNCTIONS[gene.effect]
        result = fn(result, sr, **gene.params)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI (§1.6) ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class _EffectFlagAction(argparse.Action):
    """Registra cada ocurrencia de --eq/--compressor/--reverb/--limiter, en el
    orden exacto en que aparecen en la línea de comandos, para poder construir
    una Chain que respete ese orden (el orden de los efectos importa: EQ antes
    que compresor no es lo mismo que al revés)."""

    def __call__(self, parser, namespace, values, option_string=None):
        order = getattr(namespace, "_effect_flag_order", None)
        if order is None:
            order = []
            namespace._effect_flag_order = order
        order.append((option_string, values))


def _parse_flag_value(option_string: str, raw: str) -> EffectGene:
    effect_name, param_names = _CLI_FLAG_TO_EFFECT[option_string]
    parts = raw.split(":")
    if len(parts) != len(param_names):
        raise ValueError(
            f"{option_string} espera {len(param_names)} valores separados por ':' "
            f"({':'.join(param_names)}), recibido {raw!r}.")
    params = {name: float(val) for name, val in zip(param_names, parts)}
    return EffectGene(effect=effect_name, params=params)


def _build_chain_from_args(args) -> Chain:
    if args.chain:
        return load_chain(args.chain)
    order = getattr(args, "_effect_flag_order", [])
    if not order:
        raise ValueError(
            "No se ha dado ninguna cadena: usa --chain FICHERO.json o al menos un "
            "flag de efecto (--eq/--compressor/--reverb/--limiter).")
    return [_parse_flag_value(opt, val) for opt, val in order]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio_effects",
        description="Aplica una cadena de efectos (EQ, compresor, reverb, limitador) "
                     "ya dada a un WAV. No evoluciona ni busca nada — para eso, "
                     "mix_evolver.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_wav", help="WAV de entrada (dry, sin procesar)")
    parser.add_argument("--chain", metavar="FILE", default=None,
                         help="JSON con una lista de EffectGene (§1.4) a aplicar en orden")
    parser.add_argument("--eq", metavar="FREQ:GAIN:Q", action=_EffectFlagAction)
    parser.add_argument("--compressor", metavar="THRESH:RATIO:ATTACK:RELEASE",
                         action=_EffectFlagAction)
    parser.add_argument("--reverb", metavar="SIZE:DECAY:MIX", action=_EffectFlagAction)
    parser.add_argument("--limiter", metavar="CEILING", action=_EffectFlagAction)
    parser.add_argument("--delay", metavar="DELAY_MS:FEEDBACK:MIX", action=_EffectFlagAction)
    parser.add_argument("--saturation", metavar="DRIVE_DB:MIX", action=_EffectFlagAction)
    parser.add_argument("--chorus", metavar="RATE_HZ:DEPTH_MS:MIX", action=_EffectFlagAction)
    parser.add_argument("--tremolo", metavar="RATE_HZ:DEPTH:MIX", action=_EffectFlagAction)
    parser.add_argument("--noise-gate", metavar="THRESHOLD_DB:ATTACK_MS:RELEASE_MS:RANGE_DB",
                         action=_EffectFlagAction)
    parser.add_argument("--de-esser", metavar="FREQ_HZ:THRESHOLD_DB:RATIO:Q",
                         action=_EffectFlagAction)
    parser.add_argument("--out", metavar="FILE", required=True,
                         help="WAV procesado de salida")
    parser.add_argument("--json", metavar="FILE", default=None,
                         help="Informe opcional (cadena aplicada, duración, picos antes/después)")
    return parser


def _preprocess_argv(argv: List[str]) -> List[str]:
    """Convierte '--flag value' en '--flag=value' para los flags de efecto,
    de forma que argparse no confunda un valor que empieza por '-' (p.ej.
    '--compressor -18:3.0:10:120', con threshold negativo) con otra opción."""
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _CLI_FLAG_TO_EFFECT and i + 1 < len(argv):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def main():
    parser = _build_arg_parser()
    args = parser.parse_args(_preprocess_argv(sys.argv[1:]))

    try:
        chain = _build_chain_from_args(args)
    except (ValueError, json.JSONDecodeError) as e:
        sys.exit(f"✗  {e}")

    sf = _import_soundfile()
    try:
        audio, sr = sf.read(args.input_wav, dtype="float64", always_2d=False)
    except FileNotFoundError:
        sys.exit(f"✗  Fichero no encontrado: {args.input_wav}")
    except Exception as e:
        if type(e).__name__ in ("LibsndfileError", "SoundFileError", "SoundFileRuntimeError"):
            sys.exit(f"✗  Error leyendo audio: {e}")
        raise

    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    t0 = time.time()
    processed = apply_chain(chain, audio, sr)
    elapsed = time.time() - t0
    peak_after = float(np.max(np.abs(processed))) if processed.size else 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, processed, sr, subtype="FLOAT")
    print(f"  ✓  {len(chain)} efecto(s) aplicados → {args.out}  "
          f"(peak {peak_before:.3f} → {peak_after:.3f}, {elapsed:.2f}s)")

    if args.json:
        report = {
            "input_wav": str(args.input_wav),
            "out_wav": str(args.out),
            "sample_rate": sr,
            "duration_s": len(audio) / sr if sr else 0.0,
            "chain": chain_to_dicts(chain),
            "peak_before": peak_before,
            "peak_after": peak_after,
            "processing_time_s": elapsed,
        }
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  ✓  Informe → {args.json}")


if __name__ == "__main__":
    main()
