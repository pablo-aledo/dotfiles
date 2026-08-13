#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            GENOPATCH  v2.0                                   ║
║  Matching de patches de síntesis por búsqueda evolutiva — fichero único      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Dado un WAV objetivo, busca los parámetros de un motor de síntesis        ║
║    paramétrico que mejor lo reproducen, mediante un algoritmo genético       ║
║    (selección por torneo + crossover aritmético + mutación gaussiana +       ║
║    elitismo). Inspirado en el "Genopatch" de Synplant 2 (Sonic Charge),      ║
║    pero sin red neuronal: aquí la búsqueda evolutiva ES el modelo, en la     ║
║    línea de mix_evolver.py (ver audio_effects.py) y del Patch Mutator de     ║
║    Nord — puro CPU, sin GPU, determinista dado un --seed.                    ║
║                                                                              ║
║  MOTORES DE SÍNTESIS — genéricos, seleccionables con --engine                ║
║    fm2           2 op. FM + filtro resonante CON envolvente propia.         ║
║    additive      envuelve Timbre/EnvelopeADR/synth_note de audio_lab.py.    ║
║    subtractive   2 osc (saw/pulso+PWM) + sub-osc + ruido + filtro           ║
║                  resonante con envolvente — el "analógico" clásico.         ║
║    karplus       cuerda pulsada (Karplus-Strong), comb filter vectorizado   ║
║                  vía scipy.lfilter — plucks/percusión afinada.              ║
║    wavetable     4 tablas de un ciclo (seno/triángulo/sierra/formante)      ║
║                  escaneadas a lo largo de la nota + filtro.                 ║
║    noise         ruido coloreado + resonancia de "cuerpo" (bandpass) —      ║
║                  percusión/texturas no tonales.                             ║
║    layered_fm_add  fm2 + additive mezclados (layer_mix) — wrapper genérico  ║
║                  de "capas", demuestra que EngineSpec compone sin tocar     ║
║                  GA/fitness/CLI. Prefijos a_*/b_* en sus parámetros.        ║
║    Añadir un motor nuevo = registrar un EngineSpec más en RAW_ENGINES; ni   ║
║    el GA ni el CLI necesitan cambios.  Ver subcomando `list-engines`.       ║
║                                                                              ║
║    TODOS los motores llevan además, automáticamente, un wrapper de FX       ║
║    genérico (post-proceso sobre el audio YA renderizado, no cableado en    ║
║    ningún motor):                                                          ║
║      vibrato_rate/depth/delay   LFO de pitch (remuestreo a tasa variable)   ║
║      unison_voices/detune       hasta 7 voces desafinadas sumadas (mono)    ║
║      drive_amount/mix           waveshaping tanh mezclado con la señal seca ║
║    Todos con default = sin efecto → retrocompatible con patch.json          ║
║    guardados en versiones anteriores.                                      ║
║                                                                              ║
║  FITNESS — genérico, seleccionable con --fitness                            ║
║    spectral   autocontenido: log-magnitud STFT (n_fft=2048, ~21.5Hz/bin —   ║
║               antes 1024/43Hz), agregada media+std sobre el tiempo          ║
║               (invariante a duración exacta) + descriptor explícito de      ║
║               vibrato (profundidad+tasa vía frecuencia instantánea/Hilbert, ║
║               ponderado). Sin este descriptor, la resolución de la STFT     ║
║               por sí sola sigue sin distinguir vibratos sutiles — es lo que ║
║               le da al GA señal real para ajustar vibrato_depth/rate.       ║
║               El paisaje de vibrato tiene varios óptimos locales: usa       ║
║               --strands 4+ en `match` si el objetivo tiene vibrato notorio. ║
║               Rápido — recomendado para el bucle del GA (miles de evals).   ║
║    reference  reutiliza el extractor de audio_reference_scorer.py           ║
║               (--fitness-backend spectral|latent) como espacio de           ║
║               features, con distancia euclídea simple al objetivo. NO usa   ║
║               la maquinaria de Mahalanobis/corpus de ese script — está      ║
║               diseñada para comparar contra un corpus de referencia         ║
║               (media+covarianza, mínimo 2 WAVs), y aquí el objetivo es una  ║
║               única muestra. Cada evaluación escribe el candidato a un WAV  ║
║               temporal (extract() lee de disco) — más lento que 'spectral', ║
║               y mucho más lento aún con --fitness-backend latent (requiere  ║
║               audiolm.py + checkpoint, inferencia por candidato). No lleva  ║
║               descriptor de vibrato propio (es ajeno a este script).        ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    match          target.wav → patch.json [+ --preview .wav] [+ --json]     ║
║    render         patch.json → WAV (motor + nota + duración del patch,      ║
║                    todo sobreescribible por flags)                          ║
║    mutate          patch.json → N variantes (nueva semilla desde un patch,   ║
║                    igual que "Plant Seed" en Synplant)                      ║
║    explore         N patches ALEATORIOS por motor, renderizados sin GA —     ║
║                    para descubrir sonidos por accidente, no por matching.   ║
║    morph           interpola linealmente entre dos patch.json en N pasos —   ║
║                    viajar deliberadamente por el espacio de un timbre a     ║
║                    otro, en vez de mutar al azar.                           ║
║    list-engines    lista motores registrados y sus parámetros               ║
║    info            imprime los parámetros de un patch.json                  ║
║    pitch           detecta la nota fundamental de un WAV (diagnóstico)      ║
║                                                                              ║
║  DETECCIÓN DE PITCH (YIN — De Cheveigné & Kawahara 2002, autocontenido)      ║
║    Si `match` no recibe --note, estima la fundamental del WAV objetivo con   ║
║    YIN: función de diferencia normalizada acumulada (más robusta frente a    ║
║    errores de octava que la autocorrelación simple — confirmado en           ║
║    testing: un grave profundo que antes se detectaba ~36 armónicos por       ║
║    encima ahora sale casi exacto; una campana inarmónica que saltaba a un    ║
║    subarmónico ahora falla por solo 1 semitono). Salta ~20ms de ataque,      ║
║    ventana de hasta 2s, rango [--pitch-fmin, --pitch-fmax] Hz. Si la         ║
║    confianza cae por debajo de --pitch-confidence (sonidos no                ║
║    tonales/percusivos/ruido), cae a --note 60 (C4) con un aviso — nunca      ║
║    falla silenciosamente con una nota inventada de baja confianza.           ║
║    Contenido MUY inarmónico (ratios de armónicos no enteros) sigue siendo    ║
║    ambiguo para cualquier método basado en periodicidad — usa --note         ║
║    explícito si sabes que el timbre es de ese tipo.                          ║
║    `genopatch.py pitch archivo.wav` expone el mismo detector suelto, para    ║
║    inspeccionar cualquier WAV sin lanzar un match completo.                  ║
║                                                                              ║
║                                                                              ║
║  USO                                                                        ║
║    genopatch.py match target.wav --engine fm2 --note C4 --out patch.json \\ ║
║                 --preview preview.wav                                       ║
║    genopatch.py match target.wav --engine subtractive --strands 4 \\        ║
║                 --out patch.json                                            ║
║    genopatch.py render patch.json --note E4 --out variant.wav               ║
║    genopatch.py mutate patch.json -n 8 --amount 0.15 --out-dir variants/    ║
║    genopatch.py explore --engine karplus -n 12 --out-dir explore/           ║
║    genopatch.py morph a.json b.json --steps 8 --render --out-dir morph/     ║
║    genopatch.py list-engines                                                ║
║    genopatch.py pitch target.wav                                            ║
║                                                                              ║
║  FORMATO patch.json                                                         ║
║    {"engine": "fm2", "note": 60, "duration": 1.5, "velocity": 1.0,          ║
║     "params": {"carrier_ratio": 1.0, "mod_ratio": 2.0, ...}}                ║
║                                                                              ║
║  DEPENDENCIAS  numpy  scipy  soundfile   (obligatorias)                     ║
║                audio_lab.py               (solo --engine additive/layered)  ║
║                audio_reference_scorer.py  (solo --fitness reference)        ║
║                                                                              ║
║  LIMITACIONES                                                               ║
║    · Más parámetros = espacio de búsqueda más grande: con el mismo          ║
║      presupuesto de --generations/--pop, motores con más parámetros         ║
║      (fm2/subtractive/layered_fm_add, con FX incluido, rondan 20-34         ║
║      dimensiones) convergen más despacio que motores simples (karplus,      ║
║      10). Subir --strands ayuda más que subir --pop a igual coste.          ║
║    · wavetable/subtractive no son banda-limitados con precisión (saw/       ║
║      square vía scipy.signal / suma de armónicos truncada) — puede haber    ║
║      aliasing leve en notas muy agudas.                                     ║
║    · unison/vibrato son efectos MONO (remuestreo, no líneas estéreo) —      ║
║      dan grosor, no anchura estéreo real.                                   ║
║    · La detección de pitch es monofónica y por autocorrelación (no hay     ║
║      DNN de pitch): fiable para notas sostenidas de fundamental clara,      ║
║      menos para acordes, ruido de banda ancha o transitorios muy cortos     ║
║      (donde de todas formas --note apenas importa para el resultado).       ║
║    · fm2/subtractive/noise usan semilla de ruido fija (0) para que el       ║
║      fitness sea determinista entre generaciones.                          ║
║    · additive y layered_fm_add requieren audio_lab.py junto a este          ║
║      fichero; reference requiere audio_reference_scorer.py (y               ║
║      opcionalmente audiolm.py).                                             ║
║                                                                              ║
║  Módulo importable:                                                        ║
║    from genopatch_v2 import ENGINES, run_ga, Patch, load_patch, \\          ║
║                             save_patch, detect_pitch, hz_to_midi            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
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


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _import_audio_lab():
    """Requerido solo por --engine additive. audio_lab.py debe vivir junto a
    este fichero o estar en PYTHONPATH — no se duplica su código aquí."""
    try:
        import audio_lab
        return audio_lab
    except ImportError:
        sys.path.insert(0, str(_script_dir()))
        try:
            import audio_lab
            return audio_lab
        except ImportError:
            sys.exit(
                "✗  audio_lab.py no encontrado — el motor 'additive' lo necesita "
                "en el mismo directorio o en PYTHONPATH.")


def _import_audio_reference_scorer():
    """Requerido solo por --fitness reference."""
    try:
        import audio_reference_scorer
        return audio_reference_scorer
    except ImportError:
        sys.path.insert(0, str(_script_dir()))
        try:
            import audio_reference_scorer
            return audio_reference_scorer
        except ImportError:
            sys.exit(
                "✗  audio_reference_scorer.py no encontrado — --fitness reference "
                "lo necesita en el mismo directorio o en PYTHONPATH.")


# ══════════════════════════════════════════════════════════════════════════════
# ── §1 motores de síntesis — interfaz genérica ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    name: str
    lo: float
    hi: float
    default: float


@dataclass
class EngineSpec:
    name: str
    params: List[ParamSpec]
    render: Callable[[Dict[str, float], int, float, float, int], np.ndarray]
    description: str = ""

    def default_params(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}


def midi_to_hz(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12.0)


def _adsr_envelope(attack: float, decay: float, sustain: float, release: float,
                    duration: float, sr: int) -> np.ndarray:
    """ADSR lineal simple, autocontenida (sin dependencia de audio_lab).
    sustain es un NIVEL (0-1), no una duración."""
    n_total = max(1, int(duration * sr))
    n_a = max(1, int(attack * sr))
    n_d = max(1, int(decay * sr))
    n_r = max(1, int(release * sr))
    n_s = max(0, n_total - n_a - n_d - n_r)

    env = np.zeros(n_total, dtype=np.float64)
    pos = 0
    end = min(pos + n_a, n_total)
    env[pos:end] = np.linspace(0.0, 1.0, end - pos, endpoint=True)
    pos = end
    end = min(pos + n_d, n_total)
    if end > pos:
        env[pos:end] = np.linspace(1.0, sustain, end - pos, endpoint=True)
    pos = end
    end = min(pos + n_s, n_total)
    env[pos:end] = sustain
    pos = end
    end = min(pos + n_r, n_total)
    if end > pos:
        env[pos:end] = np.linspace(sustain, 0.0, end - pos, endpoint=True)
    return env


def _biquad_lowpass(audio: np.ndarray, sr: int, freq_hz: float, q: float) -> np.ndarray:
    """Lowpass resonante, fórmulas RBJ Audio EQ Cookbook (mismo origen que
    eq_peaking en audio_effects.py)."""
    signal = _import_scipy_signal()
    freq_hz = min(max(freq_hz, 20.0), sr / 2 * 0.99)
    q = max(q, 0.1)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, audio)


# ── fm2 — ahora con envolvente de filtro propia (AD sobre el cutoff) ─────────

FM2_PARAMS = [
    ParamSpec("carrier_ratio",    0.25, 8.0,    1.0),
    ParamSpec("mod_ratio",        0.25, 32.0,   2.0),    # antes hasta 16
    ParamSpec("mod_index",        0.0,  24.0,   3.0),    # antes hasta 16
    ParamSpec("mod_index_decay",  0.05, 4.0,    0.3),
    ParamSpec("amp_attack",       0.001, 1.0,   0.01),
    ParamSpec("amp_decay",        0.005, 2.0,   0.3),
    ParamSpec("amp_sustain",      0.0,  1.0,    0.6),
    ParamSpec("amp_release",      0.005, 2.0,   0.2),
    ParamSpec("filter_cutoff",    80.0, 16000.0, 6000.0),   # antes 200-12000
    ParamSpec("filter_q",         0.5,  8.0,    0.71),
    ParamSpec("filter_env_amount", -1.0, 3.0,   0.0),    # 0 = filtro estático (compat v1)
    ParamSpec("filter_attack",    0.001, 1.0,   0.01),
    ParamSpec("filter_decay",     0.005, 2.0,   0.2),
    ParamSpec("noise_mix",        0.0,  0.5,    0.0),
]


def _biquad_lowpass_varying(audio: np.ndarray, sr: int, cutoff_env: np.ndarray,
                              q: float, block_size: int = 128) -> np.ndarray:
    """Lowpass RBJ con cutoff variable en el tiempo, por bloques (los
    coeficientes se recalculan cada `block_size` muestras, con el estado del
    filtro (`zi`) llevado de un bloque a otro para no introducir clics). No
    es sample-accurate, pero es más que suficiente para una envolvente de
    filtro (que cambia en decenas/cientos de ms, no muestra a muestra)."""
    signal = _import_scipy_signal()
    n = len(audio)
    if n == 0:
        return audio
    out = np.zeros(n, dtype=np.float64)
    zi = None
    q = max(q, 0.1)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        cutoff = float(np.clip(cutoff_env[start], 20.0, sr / 2 * 0.99))
        w0 = 2 * np.pi * cutoff / sr
        alpha = np.sin(w0) / (2 * q)
        b0 = (1 - np.cos(w0)) / 2
        b1 = 1 - np.cos(w0)
        b2 = (1 - np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        if zi is None:
            zi = signal.lfilter_zi(b, a) * audio[start]
        block_out, zi = signal.lfilter(b, a, audio[start:end], zi=zi)
        out[start:end] = block_out
    return out


def _biquad_bandpass(audio: np.ndarray, sr: int, freq_hz: float, q: float) -> np.ndarray:
    """Bandpass RBJ (constant peak gain) — usado por el motor 'noise' para
    simular la resonancia de un "cuerpo" (parche de tambor, caja, etc)."""
    signal = _import_scipy_signal()
    freq_hz = min(max(freq_hz, 20.0), sr / 2 * 0.99)
    q = max(q, 0.1)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, audio)


def _render_fm2(params: Dict[str, float], note_midi: int, duration: float,
                 velocity: float, sr: int) -> np.ndarray:
    n = max(1, int(duration * sr))
    t = np.arange(n) / sr
    f0 = midi_to_hz(note_midi)
    fc = f0 * params["carrier_ratio"]
    fm = f0 * params["mod_ratio"]

    mod_env = params["mod_index"] * np.exp(-t / max(params["mod_index_decay"], 1e-3))
    phase = 2 * np.pi * fc * t + mod_env * np.sin(2 * np.pi * fm * t)
    osc = np.sin(phase)

    if params["noise_mix"] > 1e-6:
        rng = np.random.default_rng(0)   # semilla fija: fitness determinista
        noise = rng.standard_normal(n)
        mix = params["noise_mix"]
        osc = (1.0 - mix) * osc + mix * noise

    env = _adsr_envelope(params["amp_attack"], params["amp_decay"],
                          params["amp_sustain"], params["amp_release"], duration, sr)
    audio = osc * env * velocity

    env_amount = params.get("filter_env_amount", 0.0)
    if abs(env_amount) < 1e-6:
        audio = _biquad_lowpass(audio, sr, params["filter_cutoff"], params["filter_q"])
    else:
        # envolvente AD (0→1→0) sobre el cutoff — sweep clásico de sintetizador
        filt_env = _adsr_envelope(params.get("filter_attack", 0.01),
                                    params.get("filter_decay", 0.2), 0.0, 0.0, duration, sr)
        cutoff_env = params["filter_cutoff"] * (1.0 + env_amount * filt_env)
        audio = _biquad_lowpass_varying(audio, sr, cutoff_env, params["filter_q"])
    return audio.astype(np.float32)


# ── additive (envuelve audio_lab.py) ─────────────────────────────────────────

N_ADDITIVE_HARMONICS = 8

ADDITIVE_PARAMS = [
    ParamSpec(f"amp_h{i+1}", 0.0, 1.0, max(0.05, 1.0 / (i + 1)))
    for i in range(N_ADDITIVE_HARMONICS)
] + [
    ParamSpec("attack_time",  0.001, 1.0, 0.01),
    ParamSpec("decay_time",   0.01,  3.0, 0.3),
    ParamSpec("decay_level",  0.0,   1.0, 0.5),
    ParamSpec("release_time", 0.005, 2.0, 0.2),
]


def _render_additive(params: Dict[str, float], note_midi: int, duration: float,
                       velocity: float, sr: int) -> np.ndarray:
    al = _import_audio_lab()
    harmonics = [params[f"amp_h{i+1}"] for i in range(N_ADDITIVE_HARMONICS)]
    timbre = al.Timbre(name="genopatch", key_frames={0: harmonics, 127: harmonics},
                        n_harmonics=N_ADDITIVE_HARMONICS, normalize=True)
    envelope = al.EnvelopeADR(
        attack_time=params["attack_time"], attack_curve="cosine",
        decay_time=params["decay_time"], decay_curve="exponential",
        decay_level=params["decay_level"],
        release_time=params["release_time"], release_curve="exponential",
    )
    audio = al.synth_note(midi_note=note_midi, duration=duration, velocity=velocity,
                           timbre=timbre, envelope=envelope, sr=sr)
    return audio.astype(np.float32)


# ── subtractive — 2 osciladores (blend saw/pulso con PWM) + ruido + filtro
#    resonante CON envolvente propia (el clásico "sintetizador analógico") ──

SUBTRACTIVE_PARAMS = [
    ParamSpec("osc_mix",         0.0,  1.0,   0.5),    # 0=solo osc1, 1=solo osc2
    ParamSpec("osc2_detune",     -1.0, 1.0,   0.0),     # semitonos
    ParamSpec("pulse_width",     0.05, 0.95,  0.5),    # forma osc1: 0.5≈saw-ish sim., extremos=pulso fino
    ParamSpec("sub_osc_mix",     0.0,  0.6,   0.0),    # osc grave a -1 octava
    ParamSpec("noise_mix",       0.0,  0.5,   0.0),
    ParamSpec("amp_attack",      0.001, 1.0,  0.01),
    ParamSpec("amp_decay",       0.005, 2.0,  0.3),
    ParamSpec("amp_sustain",     0.0,  1.0,   0.7),
    ParamSpec("amp_release",     0.005, 2.0,  0.2),
    ParamSpec("filter_cutoff",   80.0, 16000.0, 4000.0),
    ParamSpec("filter_q",        0.5,  10.0,  1.0),
    ParamSpec("filter_env_amount", -1.0, 3.0, 1.0),    # aquí por defecto SÍ hay envolvente (el "wow" clásico)
    ParamSpec("filter_attack",   0.001, 1.0,  0.005),
    ParamSpec("filter_decay",    0.005, 2.0,  0.25),
]


def _render_subtractive(params: Dict[str, float], note_midi: int, duration: float,
                          velocity: float, sr: int) -> np.ndarray:
    signal = _import_scipy_signal()
    n = max(1, int(duration * sr))
    t = np.arange(n) / sr
    f0 = midi_to_hz(note_midi)

    osc1 = signal.sawtooth(2 * np.pi * f0 * t, width=params["pulse_width"])
    f2 = f0 * (2 ** (params["osc2_detune"] / 12.0))
    osc2 = signal.square(2 * np.pi * f2 * t, duty=0.5)
    mix = params["osc_mix"]
    osc = (1.0 - mix) * osc1 + mix * osc2

    if params["sub_osc_mix"] > 1e-6:
        sub = np.sign(np.sin(2 * np.pi * (f0 / 2.0) * t))
        osc = (1.0 - params["sub_osc_mix"]) * osc + params["sub_osc_mix"] * sub

    if params["noise_mix"] > 1e-6:
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(n)
        nm = params["noise_mix"]
        osc = (1.0 - nm) * osc + nm * noise

    env = _adsr_envelope(params["amp_attack"], params["amp_decay"],
                          params["amp_sustain"], params["amp_release"], duration, sr)
    audio = osc * env * velocity

    env_amount = params.get("filter_env_amount", 0.0)
    if abs(env_amount) < 1e-6:
        audio = _biquad_lowpass(audio, sr, params["filter_cutoff"], params["filter_q"])
    else:
        filt_env = _adsr_envelope(params.get("filter_attack", 0.005),
                                    params.get("filter_decay", 0.25), 0.0, 0.0, duration, sr)
        cutoff_env = params["filter_cutoff"] * (1.0 + env_amount * filt_env)
        audio = _biquad_lowpass_varying(audio, sr, cutoff_env, params["filter_q"])
    return audio.astype(np.float32)


# ── karplus — cuerda pulsada (Karplus-Strong), vectorizado vía scipy.lfilter
#    (el algoritmo es una recurrencia de comb filter — se puede expresar
#    como un único IIR y dejar que lfilter haga el bucle en C, no en Python) ─

KARPLUS_PARAMS = [
    ParamSpec("decay",        0.90, 0.999, 0.996),   # realimentación del lazo → sustain
    ParamSpec("brightness",   0.0,  1.0,   0.5),      # filtrado del burst inicial de ruido
    ParamSpec("out_release",  0.005, 1.0,  0.05),     # fade final para evitar clic de corte
]


def _render_karplus(params: Dict[str, float], note_midi: int, duration: float,
                      velocity: float, sr: int) -> np.ndarray:
    signal = _import_scipy_signal()
    f0 = max(midi_to_hz(note_midi), 20.0)
    period_n = max(2, int(round(sr / f0)))
    n_total = max(period_n + 2, int(duration * sr))

    rng = np.random.default_rng(0)
    burst = rng.uniform(-1.0, 1.0, period_n)
    brightness = params["brightness"]
    if brightness < 0.999:
        alpha = 0.05 + 0.9 * (1.0 - brightness)
        burst = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], burst)

    x = np.zeros(n_total, dtype=np.float64)
    x[:period_n] = burst
    fb = params["decay"]
    a = np.zeros(period_n + 2, dtype=np.float64)
    a[0] = 1.0
    a[period_n] = -fb * 0.5
    a[period_n + 1] = -fb * 0.5
    y = signal.lfilter([1.0], a, x)

    audio = y * velocity
    release_n = max(1, int(params["out_release"] * sr))
    if release_n < n_total:
        fade = np.ones(n_total)
        fade[-release_n:] = np.linspace(1.0, 0.0, release_n)
        audio = audio * fade
    return audio.astype(np.float32)


# ── wavetable — 4 tablas de un ciclo fijas, "escaneadas" a lo largo de la
#    nota (posición inicial→final), con filtro + ADSR ─────────────────────

_WAVETABLE_LEN = 256


def _build_wavetables() -> np.ndarray:
    ph = np.linspace(0.0, 2 * np.pi, _WAVETABLE_LEN, endpoint=False)
    sine = np.sin(ph)
    # triangular vía suma de armónicos impares con signo alterno (rápida de generar, sin bordes duros)
    tri = np.zeros(_WAVETABLE_LEN)
    for k in range(1, 8, 2):
        tri += ((-1) ** ((k - 1) // 2)) * np.sin(k * ph) / (k ** 2)
    tri = tri / np.max(np.abs(tri))
    # diente de sierra vía suma de armónicos (banda-limitada a mano, sin aliasing agresivo)
    saw = np.zeros(_WAVETABLE_LEN)
    for k in range(1, 16):
        saw += np.sin(k * ph) / k
    saw = saw / np.max(np.abs(saw))
    # tabla "formante": un par de picos espectrales, tipo vocal/nasal
    formant = np.sin(ph) + 0.6 * np.sin(5 * ph) + 0.3 * np.sin(9 * ph)
    formant = formant / np.max(np.abs(formant))
    return np.stack([sine, tri, saw, formant])   # shape (4, _WAVETABLE_LEN)


_WAVETABLES = _build_wavetables()
_N_WAVETABLES = _WAVETABLES.shape[0]

WAVETABLE_PARAMS = [
    ParamSpec("pos_start",     0.0, float(_N_WAVETABLES - 1), 0.0),
    ParamSpec("pos_end",       0.0, float(_N_WAVETABLES - 1), 0.0),
    ParamSpec("amp_attack",    0.001, 1.0, 0.01),
    ParamSpec("amp_decay",     0.005, 2.0, 0.3),
    ParamSpec("amp_sustain",   0.0,  1.0,  0.7),
    ParamSpec("amp_release",   0.005, 2.0, 0.2),
    ParamSpec("filter_cutoff", 80.0, 16000.0, 8000.0),
    ParamSpec("filter_q",      0.5,  8.0,   0.71),
]


def _render_wavetable(params: Dict[str, float], note_midi: int, duration: float,
                        velocity: float, sr: int) -> np.ndarray:
    n = max(1, int(duration * sr))
    f0 = midi_to_hz(note_midi)
    t = np.arange(n) / sr
    phase = np.mod(f0 * t, 1.0)   # 0..1

    position = np.linspace(params["pos_start"], params["pos_end"], n)
    position = np.clip(position, 0.0, _N_WAVETABLES - 1)
    t_idx0 = np.floor(position).astype(int)
    t_idx1 = np.clip(t_idx0 + 1, 0, _N_WAVETABLES - 1)
    t_frac = position - t_idx0

    p_idx_f = phase * _WAVETABLE_LEN
    p_idx0 = np.floor(p_idx_f).astype(int) % _WAVETABLE_LEN
    p_idx1 = (p_idx0 + 1) % _WAVETABLE_LEN
    p_frac = p_idx_f - np.floor(p_idx_f)

    val_t0 = (_WAVETABLES[t_idx0, p_idx0] * (1 - p_frac) + _WAVETABLES[t_idx0, p_idx1] * p_frac)
    val_t1 = (_WAVETABLES[t_idx1, p_idx0] * (1 - p_frac) + _WAVETABLES[t_idx1, p_idx1] * p_frac)
    osc = val_t0 * (1 - t_frac) + val_t1 * t_frac

    env = _adsr_envelope(params["amp_attack"], params["amp_decay"],
                          params["amp_sustain"], params["amp_release"], duration, sr)
    audio = osc * env * velocity
    audio = _biquad_lowpass(audio, sr, params["filter_cutoff"], params["filter_q"])
    return audio.astype(np.float32)


# ── noise — ruido coloreado + resonancia de "cuerpo" + ADSR, para
#    percusión/texturas no tonales (el hueco que dejaban fm2/additive) ─────

NOISE_PARAMS = [
    ParamSpec("noise_tone",   200.0, 14000.0, 6000.0),   # lowpass de color sobre el ruido base
    ParamSpec("body_freq",    80.0,  8000.0,  400.0),     # centro del bandpass "cuerpo"
    ParamSpec("body_q",       0.5,   15.0,    3.0),
    ParamSpec("body_mix",     0.0,   1.0,     0.3),
    ParamSpec("amp_attack",   0.001, 0.5,     0.001),
    ParamSpec("amp_decay",    0.005, 2.0,     0.15),
    ParamSpec("amp_sustain",  0.0,   1.0,     0.0),
    ParamSpec("amp_release",  0.005, 1.0,     0.1),
]


def _render_noise(params: Dict[str, float], note_midi: int, duration: float,
                    velocity: float, sr: int) -> np.ndarray:
    n = max(1, int(duration * sr))
    rng = np.random.default_rng(0)
    base = rng.standard_normal(n)
    colored = _biquad_lowpass(base, sr, params["noise_tone"], 0.71)
    body = _biquad_bandpass(base, sr, params["body_freq"], params["body_q"])
    mix = params["body_mix"]
    osc = (1.0 - mix) * colored + mix * body

    env = _adsr_envelope(params["amp_attack"], params["amp_decay"],
                          params["amp_sustain"], params["amp_release"], duration, sr)
    audio = osc * env * velocity
    return audio.astype(np.float32)


RAW_ENGINES: Dict[str, EngineSpec] = {
    "fm2": EngineSpec(
        "fm2", FM2_PARAMS, _render_fm2,
        "2 operadores FM (portadora+moduladora) + filtro resonante con envolvente propia"),
    "additive": EngineSpec(
        "additive", ADDITIVE_PARAMS, _render_additive,
        "síntesis aditiva — envuelve Timbre/EnvelopeADR/synth_note de audio_lab.py"),
    "subtractive": EngineSpec(
        "subtractive", SUBTRACTIVE_PARAMS, _render_subtractive,
        "2 osciladores (saw/pulso+PWM) + sub-osc + ruido + filtro resonante con envolvente — el 'analógico' clásico"),
    "karplus": EngineSpec(
        "karplus", KARPLUS_PARAMS, _render_karplus,
        "cuerda pulsada (Karplus-Strong) — comb filter vectorizado, plucks/percusión afinada"),
    "wavetable": EngineSpec(
        "wavetable", WAVETABLE_PARAMS, _render_wavetable,
        "4 tablas de un ciclo (seno/triángulo/sierra/formante) escaneadas a lo largo de la nota"),
    "noise": EngineSpec(
        "noise", NOISE_PARAMS, _render_noise,
        "ruido coloreado + resonancia de cuerpo (bandpass) — percusión/texturas no tonales"),
}


# ── motor "en capas" — combina dos motores base y los mezcla; wrapper
#    genérico igual que vibrato/unison/drive más abajo, demuestra que la
#    interfaz EngineSpec compone sin tocar GA/fitness/CLI ──────────────────

def _make_layered_engine(name: str, engine_a: EngineSpec, engine_b: EngineSpec) -> EngineSpec:
    prefix_a, prefix_b = "a_", "b_"
    params = ([ParamSpec(prefix_a + p.name, p.lo, p.hi, p.default) for p in engine_a.params]
              + [ParamSpec(prefix_b + p.name, p.lo, p.hi, p.default) for p in engine_b.params]
              + [ParamSpec("layer_mix", 0.0, 1.0, 0.5)])

    def render(params_dict: Dict[str, float], note_midi: int, duration: float,
               velocity: float, sr: int) -> np.ndarray:
        params_a = {k[len(prefix_a):]: v for k, v in params_dict.items() if k.startswith(prefix_a)}
        params_b = {k[len(prefix_b):]: v for k, v in params_dict.items() if k.startswith(prefix_b)}
        audio_a = engine_a.render(params_a, note_midi, duration, velocity, sr)
        audio_b = engine_b.render(params_b, note_midi, duration, velocity, sr)
        n = min(len(audio_a), len(audio_b))
        mix = params_dict.get("layer_mix", 0.5)
        return ((1.0 - mix) * audio_a[:n] + mix * audio_b[:n]).astype(np.float32)

    return EngineSpec(name, params, render,
                       f"capas: {engine_a.name} + {engine_b.name}, mezclables con layer_mix")


RAW_ENGINES["layered_fm_add"] = _make_layered_engine(
    "layered_fm_add", RAW_ENGINES["fm2"], RAW_ENGINES["additive"])


# ── FX genérico — vibrato + unison + drive, wrapper único aplicado a TODOS
#    los motores de RAW_ENGINES (incluido el en capas). Consolidar los tres
#    en un solo wrapper evita anidar varias capas de post-proceso y su coste
#    de overhead acumulado. depth=0/voices=1/drive=0 = idéntico al motor
#    base (retrocompatible con patches guardados antes de este cambio). ────

VIBRATO_PARAMS = [
    ParamSpec("vibrato_rate",  0.5, 12.0, 5.0),
    ParamSpec("vibrato_depth", 0.0, 0.08, 0.0),
    ParamSpec("vibrato_delay", 0.0, 1.0,  0.1),
]
UNISON_PARAMS = [
    ParamSpec("unison_voices",  1.0, 7.0,  1.0),    # se redondea al renderizar
    ParamSpec("unison_detune",  0.0, 50.0, 0.0),    # cents, máxima separación entre voces
]
DRIVE_PARAMS = [
    ParamSpec("drive_amount", 0.0, 1.0, 0.0),
    ParamSpec("drive_mix",    0.0, 1.0, 0.0),
]
FX_PARAMS = VIBRATO_PARAMS + UNISON_PARAMS + DRIVE_PARAMS
_FX_NAMES = {p.name for p in FX_PARAMS}


def _apply_vibrato(audio: np.ndarray, sr: int, rate_hz: float, depth: float,
                    delay_s: float) -> np.ndarray:
    """Modula el pitch por remuestreo de tasa variable con interpolación
    lineal: la posición de lectura avanza más rápido/despacio según un LFO
    senoidal. depth=0 deja el audio intacto."""
    if depth <= 1e-6 or audio.size < 4:
        return audio
    n = len(audio)
    t = np.arange(n) / sr
    onset = np.clip((t - delay_s) / 0.05, 0.0, 1.0)
    lfo = onset * depth * np.sin(2 * np.pi * rate_hz * t)
    read_pos = np.concatenate(([0.0], np.cumsum(1.0 + lfo)[:-1]))
    read_pos = np.clip(read_pos, 0.0, n - 1.0001)
    idx0 = np.floor(read_pos).astype(int)
    idx1 = np.clip(idx0 + 1, 0, n - 1)
    frac = read_pos - idx0
    return (audio[idx0] * (1.0 - frac) + audio[idx1] * frac).astype(np.float32)


def _pitch_shift_ratio(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Como _apply_vibrato pero con una tasa CONSTANTE (no oscilante) —
    remuestreo simple para desafinar una copia una cantidad fija. Usado por
    el wrapper de unison. La duración efectiva varía ligeramente con
    detune grandes; a los pocos cents que usa unison es despreciable."""
    n = len(audio)
    if n < 4 or abs(ratio - 1.0) < 1e-6:
        return audio
    read_pos = np.arange(n) * ratio
    read_pos = np.clip(read_pos, 0.0, n - 1.0001)
    idx0 = np.floor(read_pos).astype(int)
    idx1 = np.clip(idx0 + 1, 0, n - 1)
    frac = read_pos - idx0
    return (audio[idx0] * (1.0 - frac) + audio[idx1] * frac).astype(np.float32)


def _apply_unison(audio: np.ndarray, sr: int, voices: float, detune_cents: float) -> np.ndarray:
    """Suma N-1 copias desafinadas (remuestreo a tasa constante) sobre la
    original — engrosa/ensancha el sonido. voices=1 (default) deja el audio
    intacto. Mono: es un "unison de grosor", no estéreo."""
    n_voices = max(1, int(round(voices)))
    if n_voices <= 1 or detune_cents <= 1e-6:
        return audio
    out = audio.astype(np.float64).copy()
    for i in range(1, n_voices):
        frac = (i / n_voices) - 0.5
        cents = frac * 2.0 * detune_cents
        ratio = 2.0 ** (cents / 1200.0)
        out = out + _pitch_shift_ratio(audio, sr, ratio)
    return (out / n_voices).astype(np.float32)


def _apply_drive(audio: np.ndarray, amount: float, mix: float) -> np.ndarray:
    """Waveshaping simple (tanh) mezclado con la señal seca. amount=0 o
    mix=0 (default) deja el audio intacto."""
    if amount <= 1e-6 or mix <= 1e-6:
        return audio
    driven = np.tanh(audio * (1.0 + amount * 15.0))
    return ((1.0 - mix) * audio + mix * driven).astype(np.float32)


def _make_fx_engine(base: EngineSpec) -> EngineSpec:
    combined_params = base.params + FX_PARAMS

    def render(params: Dict[str, float], note_midi: int, duration: float,
               velocity: float, sr: int) -> np.ndarray:
        base_params = {k: v for k, v in params.items() if k not in _FX_NAMES}
        audio = base.render(base_params, note_midi, duration, velocity, sr)
        audio = _apply_unison(audio, sr,
                               params.get("unison_voices", 1.0),
                               params.get("unison_detune", 0.0))
        audio = _apply_vibrato(audio, sr,
                                params.get("vibrato_rate", 5.0),
                                params.get("vibrato_depth", 0.0),
                                params.get("vibrato_delay", 0.1))
        audio = _apply_drive(audio,
                              params.get("drive_amount", 0.0),
                              params.get("drive_mix", 0.0))
        return audio

    return EngineSpec(base.name, combined_params, render,
                       base.description + " · + unison/vibrato/drive")


ENGINES: Dict[str, EngineSpec] = {name: _make_fx_engine(spec) for name, spec in RAW_ENGINES.items()}


# ══════════════════════════════════════════════════════════════════════════════
# ── §2 fitness — interfaz genérica: Callable[[np.ndarray], float] ─────────────
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada fábrica cierra sobre las features del objetivo (calculadas UNA vez) y
# devuelve una función que solo necesita el audio candidato — más bajo es
# mejor (distancia, no score).

_FEAT_N_FFT = 2048    # antes 1024 — más resolución en frecuencia (era 43Hz/bin, ahora ~21.5Hz/bin)
_FEAT_HOP = 512

# Aun con más resolución, un bin de ~21.5Hz sigue sin distinguir una
# profundidad de vibrato típica (unos pocos Hz de pico) — el test exhaustivo
# anterior mostró que la STFT agregada es prácticamente ciega a eso. Se añade
# un descriptor EXPLÍCITO de modulación de pitch (profundidad + tasa) vía
# frecuencia instantánea (transformada de Hilbert), para que el fitness
# tenga señal real con la que ajustar vibrato_depth/vibrato_rate del wrapper.
_VIBRATO_DEPTH_WEIGHT = 60.0   # calibrado empíricamente — ver notas de testing;
                                 # con un solo strand la profundidad recuperada
                                 # tiene varianza alta entre semillas (paisaje
                                 # con múltiples óptimos locales) — usar
                                 # --strands 4+ en `match` cuando el objetivo
                                 # tenga vibrato notorio.
_VIBRATO_RATE_WEIGHT = 0.05
_VIBRATO_DEPTH_CLIP = 0.3   # tope de saturación — ver _extract_internal_features.
                             # Forma parte del fingerprint de features: un
                             # checkpoint de genopatch_flow.py entrenado antes
                             # de este tope calculaba features distintas para
                             # cualquier audio que saturase, así que debe
                             # invalidarse, no solo advertir.


def _vibrato_descriptor(audio: np.ndarray, sr: int) -> "tuple[float, float]":
    """Profundidad relativa y tasa dominante de la modulación de pitch, vía
    frecuencia instantánea (derivada de la fase de la señal analítica de
    Hilbert). Barato: un FFT de longitud N para el Hilbert, uno pequeño para
    la tasa dominante — nada de tracking de pitch por ventanas."""
    signal = _import_scipy_signal()
    if audio.size < 64 or np.max(np.abs(audio)) < 1e-6:
        return 0.0, 0.0
    analytic = signal.hilbert(audio)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi) * sr
    inst_freq = np.clip(inst_freq, 0.0, sr / 2.0)
    if len(inst_freq) > 64:
        kernel = np.ones(51) / 51.0
        inst_freq = np.convolve(inst_freq, kernel, mode="valid")   # suaviza el ruido de fase
    mean_f = float(np.mean(inst_freq))
    if mean_f < 1e-6 or len(inst_freq) < 8:
        return 0.0, 0.0
    deviation = inst_freq - mean_f
    depth = float(np.std(deviation) / mean_f)

    n = len(deviation)
    spec = np.abs(np.fft.rfft(deviation * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= 1.0) & (freqs <= 12.0)   # rango razonable de vibrato musical
    if np.any(mask) and spec[mask].sum() > 0:
        rate_hz = float(freqs[mask][np.argmax(spec[mask])])
    else:
        rate_hz = 0.0
    return depth, rate_hz


def _extract_internal_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """log-magnitud STFT agregada media+std sobre el tiempo (invariante a la
    duración exacta, mismo truco que extract_spectral_embedding de
    audio_reference_scorer.py) + descriptor explícito de vibrato al final,
    ponderado para que sea comparable en magnitud al resto del vector pese a
    ser solo 2 dimensiones contra ~2000."""
    signal = _import_scipy_signal()
    if audio.size < _FEAT_N_FFT:
        audio = np.pad(audio, (0, _FEAT_N_FFT - audio.size))
    _, _, Zxx = signal.stft(audio, fs=sr, nperseg=_FEAT_N_FFT,
                             noverlap=_FEAT_N_FFT - _FEAT_HOP, boundary=None)
    log_mag = np.log1p(np.abs(Zxx))
    if log_mag.shape[1] == 0:
        spectral_feat = np.zeros(2 * log_mag.shape[0], dtype=np.float64)
    else:
        mean_v = log_mag.mean(axis=1)
        std_v = log_mag.std(axis=1)
        spectral_feat = np.concatenate([mean_v, std_v]).astype(np.float64)

    depth, rate_hz = _vibrato_descriptor(audio, sr)
    # tope de seguridad: contenido ruidoso/muy resonante/con pitch muy
    # desajustado puede hacer que el estimador de frecuencia instantánea
    # (Hilbert) se vuelva inestable y devuelva profundidades absurdas
    # (>1.0 — un vibrato real nunca pasa de ~0.08). Sin este tope, un único
    # valor saturado (con peso ×60) puede dominar por completo el vector de
    # ~2050 dimensiones, dejando al GA ciego a las diferencias reales entre
    # candidatos — confirmado en testing con subtractive a nota desajustada:
    # dos patches con parámetros radicalmente distintos daban fitness
    # bit-a-bit idéntico porque solo esta dimensión saturada importaba.
    depth = min(depth, _VIBRATO_DEPTH_CLIP)
    vibrato_feat = np.array([depth * _VIBRATO_DEPTH_WEIGHT, rate_hz * _VIBRATO_RATE_WEIGHT])
    return np.concatenate([spectral_feat, vibrato_feat])


def make_fitness_spectral(target_audio: np.ndarray, sr: int) -> Callable[[np.ndarray], float]:
    target_feat = _extract_internal_features(target_audio, sr)

    def fitness(candidate_audio: np.ndarray) -> float:
        cand_feat = _extract_internal_features(candidate_audio, sr)
        n = min(len(target_feat), len(cand_feat))
        return float(np.linalg.norm(target_feat[:n] - cand_feat[:n]))

    return fitness


def make_fitness_reference(target_wav_path: str, sr: int,
                             backend_name: str = "spectral") -> Callable[[np.ndarray], float]:
    ars = _import_audio_reference_scorer()
    sf = _import_soundfile()
    backend = ars.get_backend(backend_name)
    if not backend.is_available():
        sys.exit(f"✗  Backend {backend_name!r} de audio_reference_scorer.py no disponible.")
    target_feat = backend.extract(target_wav_path)
    tmp_dir = tempfile.mkdtemp(prefix="genopatch_")
    tmp_path = os.path.join(tmp_dir, "candidate.wav")

    def fitness(candidate_audio: np.ndarray) -> float:
        sf.write(tmp_path, _normalize_peak(candidate_audio), sr, subtype="FLOAT")
        cand_feat = backend.extract(tmp_path)
        n = min(len(target_feat), len(cand_feat))
        return float(np.linalg.norm(target_feat[:n] - cand_feat[:n]))

    return fitness


# ══════════════════════════════════════════════════════════════════════════════
# ── §3 algoritmo genético ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Individual:
    params: Dict[str, float]
    fitness: float = field(default=float("inf"))


def _clip_params(params: Dict[str, float], specs: List[ParamSpec]) -> Dict[str, float]:
    by_name = {p.name: p for p in specs}
    return {k: float(np.clip(v, by_name[k].lo, by_name[k].hi)) for k, v in params.items()}


def _random_params(specs: List[ParamSpec]) -> Dict[str, float]:
    """Inicialización mixta por gen: la mitad de las veces explora el rango
    completo; la otra mitad muestrea alrededor del default (gaussiana al 15%
    del rango). Sin esto, parámetros cuyo default representa un estado
    "neutro" (p.ej. vibrato_depth=0, noise_mix=0) arrancan la mayoría de
    individuos ya "contaminados" — con vibrato_depth en concreto eso además
    difumina el espectro vía el remuestreo, quemando presupuesto evolutivo en
    des-aprenderlo incluso cuando el objetivo no tiene vibrato."""
    out = {}
    for p in specs:
        if random.random() < 0.5:
            out[p.name] = random.uniform(p.lo, p.hi)
        else:
            span = p.hi - p.lo
            out[p.name] = random.gauss(p.default, 0.15 * span)
    return out



def _tournament_select(population: List[Individual], k: int = 3) -> Individual:
    contestants = random.sample(population, min(k, len(population)))
    return min(contestants, key=lambda ind: ind.fitness)


def _crossover(a: Individual, b: Individual, specs: List[ParamSpec]) -> Dict[str, float]:
    """Crossover aritmético tipo BLX: interpola (y a veces extrapola un poco,
    t fuera de [0,1]) entre cada gen de los dos padres."""
    child = {}
    for p in specs:
        t = random.uniform(-0.15, 1.15)
        child[p.name] = a.params[p.name] + t * (b.params[p.name] - a.params[p.name])
    return _clip_params(child, specs)


def _mutate(params: Dict[str, float], specs: List[ParamSpec],
            rate: float, amount: float) -> Dict[str, float]:
    """Tolera patches con menos claves que las del motor completo (p.ej.
    los que produce genopatch_flow.py, solo con los parámetros base, sin
    FX) — usa el default de cada ParamSpec como punto de partida para
    cualquier clave ausente, en vez de asumir que ya está en `params`."""
    out = {p.name: params.get(p.name, p.default) for p in specs}
    for p in specs:
        if random.random() < rate:
            span = p.hi - p.lo
            out[p.name] = out[p.name] + random.gauss(0.0, amount * span)
    return _clip_params(out, specs)


def run_ga(engine: EngineSpec, fitness_fn: Callable[[np.ndarray], float],
           note_midi: int, duration: float, velocity: float, sr: int,
           generations: int, pop_size: int, elite: int,
           mutation_rate: float, mutation_amount: float,
           seed: int, verbose: bool = True) -> List[Individual]:
    """Devuelve la población final, ordenada de mejor (fitness más bajo) a
    peor. population[0] es el ganador de esta tanda ('strand')."""
    random.seed(seed)
    np.random.seed(seed)
    specs = engine.params

    population = [Individual(_clip_params(_random_params(specs), specs))
                  for _ in range(pop_size)]
    population[0] = Individual(engine.default_params())   # una semilla "de fábrica"

    for gen in range(generations):
        for ind in population:
            audio = engine.render(ind.params, note_midi, duration, velocity, sr)
            ind.fitness = fitness_fn(audio)
        population.sort(key=lambda ind: ind.fitness)
        if verbose:
            print(f"    gen {gen + 1:>3}/{generations}  best={population[0].fitness:.4f}")
        if gen == generations - 1:
            break
        next_pop = population[:elite]
        while len(next_pop) < pop_size:
            a = _tournament_select(population)
            b = _tournament_select(population)
            child = _crossover(a, b, specs)
            child = _mutate(child, specs, mutation_rate, mutation_amount)
            next_pop.append(Individual(child))
        population = next_pop

    population.sort(key=lambda ind: ind.fitness)
    return population


# ══════════════════════════════════════════════════════════════════════════════
# ── §4 formato patch.json ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Patch:
    engine: str
    note: int
    duration: float
    velocity: float
    params: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        return cls(engine=d["engine"], note=int(d["note"]), duration=float(d["duration"]),
                    velocity=float(d.get("velocity", 1.0)), params=dict(d["params"]))


def load_patch(path: str) -> Patch:
    if not Path(path).exists():
        sys.exit(f"✗  patch.json no encontrado: {path}")
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"✗  {path}: JSON inválido — {e}")
    try:
        return Patch.from_dict(data)
    except KeyError as e:
        sys.exit(f"✗  {path}: falta la clave {e} en el patch.")


def save_patch(patch: Patch, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(patch.to_dict(), indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
# ── notas MIDI (parseo autocontenido, sin depender de audio_lab) ──────────────
# ══════════════════════════════════════════════════════════════════════════════

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_note_name(n: int) -> str:
    return _NOTE_NAMES[n % 12] + str(n // 12 - 1)


# ══════════════════════════════════════════════════════════════════════════════
# ── §5 detección de pitch (autocorrelación, autocontenida) ────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def detect_pitch(audio: np.ndarray, sr: int, fmin: float = 40.0, fmax: float = 2000.0,
                  skip_s: float = 0.02, max_dur_s: float = 2.0,
                  threshold: float = 0.15) -> "tuple[Optional[float], float]":
    """Estima la frecuencia fundamental de un fragmento monofónico con el
    método YIN (De Cheveigné & Kawahara, 2002): función de diferencia
    normalizada acumulada (CMNDF), en vez de un pico simple de
    autocorrelación. Salta los primeros `skip_s` segundos (transitorio de
    ataque) y usa como mucho `max_dur_s` segundos centrales.

    Por qué YIN y no autocorrelación simple: un pico de autocorrelación
    puede caer, con más energía que el de la fundamental, en un armónico
    (período más corto → octava de más) o en un subarmónico (período más
    largo → octava de menos) — y ambos fallos ocurrían con ALTA confianza
    en testing (grave profundo: ~36 armónicos hacia arriba; campana
    inarmónica: bloqueo en 1/8 del período real). La normalización
    acumulada de YIN penaliza sistemáticamente los lags largos y el umbral
    de "primer mínimo suficientemente bueno" evita saltar a armónicos
    cortos — ambos sesgos, actuando junto, corrigen las dos direcciones
    de error a la vez.

    Devuelve (freq_hz, confianza 0-1). freq_hz es None si el audio es
    silencio o demasiado corto — la ausencia de un mínimo claro
    (percusión, ruido) se señaliza con una confianza baja, no con None,
    para que quien llama decida el umbral."""
    n_skip = int(skip_s * sr)
    seg = audio[n_skip: n_skip + int(max_dur_s * sr)]
    if len(seg) < int(sr / fmin) * 2:
        seg = audio[: int(max_dur_s * sr)]   # muy corto tras el skip: usa desde el inicio
    if len(seg) < 64:
        return None, 0.0

    seg = seg.astype(np.float64) - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-6:
        return None, 0.0

    n = len(seg)
    lag_min = max(1, int(sr / fmax))
    lag_max = min(int(sr / fmin), n - 1)
    if lag_max <= lag_min:
        return None, 0.0

    # autocorrelación vía FFT (igual que antes) para acelerar el cómputo
    # de la función de diferencia — d(tau) = E[0:n-tau] + E[tau:n] - 2*autocorr(tau)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    spec = np.fft.rfft(seg, n=n_fft)
    autocorr = np.fft.irfft(spec * np.conj(spec))[:n]

    energy = seg * seg
    cum_energy = np.concatenate(([0.0], np.cumsum(energy)))   # cum_energy[k] = sum(seg[:k]**2)
    taus = np.arange(0, lag_max + 1)
    total_energy = cum_energy[n]
    # sum(seg[:n-tau]**2) + sum(seg[tau:n]**2), válido para tau in [0, n]
    energy_head = cum_energy[np.clip(n - taus, 0, n)]
    energy_tail = total_energy - cum_energy[np.clip(taus, 0, n)]
    d = energy_head + energy_tail - 2.0 * autocorr[:lag_max + 1]
    d = np.maximum(d, 0.0)   # errores de redondeo pueden dar negativos minúsculos

    # función de diferencia normalizada acumulada (CMNDF); d'(0) := 1 por convención
    cmndf = np.ones_like(d)
    running_sum = 0.0
    for tau in range(1, lag_max + 1):
        running_sum += d[tau]
        cmndf[tau] = d[tau] * tau / running_sum if running_sum > 0 else 1.0

    window = cmndf[lag_min:lag_max + 1]
    below = np.where(window < threshold)[0]
    if len(below) > 0:
        # primer mínimo local por debajo del umbral (heurística estándar de
        # YIN) — evita saltar a un armónico más corto aunque tenga un valor
        # de CMNDF ligeramente menor
        idx = below[0]
        while idx + 1 < len(window) and window[idx + 1] < window[idx]:
            idx += 1
    else:
        idx = int(np.argmin(window))

    lag = lag_min + idx
    cmndf_val = float(window[idx])
    confidence = float(max(0.0, min(1.0, 1.0 - cmndf_val)))
    if confidence <= 0.0:
        return None, 0.0
    freq_hz = sr / lag
    return freq_hz, confidence


def hz_to_midi(freq_hz: float) -> int:
    return int(round(69 + 12 * np.log2(freq_hz / 440.0)))


def _resolve_note(args_note: Optional[str], target_audio: np.ndarray, target_sr: int,
                   fmin: float, fmax: float, confidence_threshold: float,
                   verbose: bool = True) -> int:
    """Nota MIDI explícita si se dio --note; si no, detección automática con
    fallback a 60 (C4) cuando la confianza no alcanza el umbral."""
    if args_note:
        return _parse_note(args_note)
    freq_hz, confidence = detect_pitch(target_audio, target_sr, fmin=fmin, fmax=fmax)
    if freq_hz is not None and confidence >= confidence_threshold:
        note_midi = hz_to_midi(freq_hz)
        if verbose:
            print(f"  ♪  pitch detectado: {freq_hz:.1f}Hz ≈ {_midi_to_note_name(note_midi)} "
                  f"(nota MIDI {note_midi}, confianza {confidence:.2f})")
        return note_midi
    if verbose:
        conf_str = f"{confidence:.2f}" if freq_hz is not None else "0.00"
        print(f"  ⚠  sin pitch claro (confianza {conf_str} < {confidence_threshold}) — "
              f"usando --note por defecto: 60 (C4)")
    return 60


def _parse_note(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        pass
    name = s.strip().replace("♭", "b").replace("♯", "#")
    octave = int(name[-1])
    pitch = (name[:-1].upper()
             .replace("BB", "A#").replace("EB", "D#").replace("AB", "G#")
             .replace("DB", "C#").replace("GB", "F#"))
    if pitch not in _NOTE_NAMES:
        raise ValueError(f"Nota no reconocida: {s!r}")
    return (octave + 1) * 12 + _NOTE_NAMES.index(pitch)


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_peak(audio: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Normaliza por ganancia (no recorta) si el pico supera `headroom`.
    Filtros resonantes (filter_q alto en fm2) pueden dar overshoot >0dB de
    forma perfectamente física — recortar con np.clip ahí metería distorsión
    audible real, no un aviso inofensivo."""
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > headroom:
        audio = audio * (headroom / peak)
    return audio.astype(np.float32)


def _read_mono_wav(path: str):
    if not Path(path).exists():
        sys.exit(f"✗  Fichero no encontrado: {path}")
    sf = _import_soundfile()
    try:
        audio, sr = sf.read(path, dtype="float64", always_2d=False)
    except Exception as e:
        # soundfile lanza distintos tipos según la causa (LibsndfileError,
        # RuntimeError...) — el fichero ya sabemos que existe (arriba), así
        # que esto es "existe pero no se puede leer" (formato no soportado,
        # corrupto, etc.), no "no encontrado".
        sys.exit(f"✗  No se pudo leer {path} como WAV: {e}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def cmd_match(args):
    engine = ENGINES.get(args.engine)
    if engine is None:
        sys.exit(f"✗  Motor desconocido: {args.engine!r} (ver `list-engines`)")

    target_audio, target_sr = _read_mono_wav(args.target_wav)
    sr = args.sr or target_sr
    duration = args.duration if args.duration else len(target_audio) / target_sr
    note_midi = _resolve_note(args.note, target_audio, target_sr,
                               args.pitch_fmin, args.pitch_fmax, args.pitch_confidence,
                               verbose=not args.quiet)

    if args.fitness == "spectral":
        fitness_fn = make_fitness_spectral(target_audio, target_sr)
    else:
        fitness_fn = make_fitness_reference(args.target_wav, sr,
                                             backend_name=args.fitness_backend)

    t0 = time.time()
    strand_winners: List[Individual] = []
    for strand in range(args.strands):
        seed = args.seed + strand
        print(f"  ── strand {strand + 1}/{args.strands} (seed={seed}) ──")
        pop = run_ga(engine, fitness_fn, note_midi, duration, args.velocity, sr,
                     args.generations, args.pop, args.elite,
                     args.mutation_rate, args.mutation_amount, seed,
                     verbose=not args.quiet)
        strand_winners.append(pop[0])
    strand_winners.sort(key=lambda ind: ind.fitness)
    best = strand_winners[0]
    elapsed = time.time() - t0

    patch = Patch(engine=args.engine, note=note_midi, duration=duration,
                  velocity=args.velocity, params=best.params)
    save_patch(patch, args.out)
    print(f"  ✓  mejor patch (fitness={best.fitness:.4f}, {elapsed:.1f}s) → {args.out}")

    if args.preview:
        audio = engine.render(best.params, note_midi, duration, args.velocity, sr)
        sf = _import_soundfile()
        sf.write(args.preview, _normalize_peak(audio), sr, subtype="FLOAT")
        print(f"  ✓  preview → {args.preview}")

    if args.strand_out_dir and args.strands > 1:
        out_dir = Path(args.strand_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, ind in enumerate(strand_winners, 1):
            p = Patch(engine=args.engine, note=note_midi, duration=duration,
                       velocity=args.velocity, params=ind.params)
            save_patch(p, str(out_dir / f"strand_{i}.json"))
        print(f"  ✓  {len(strand_winners)} strand(s) → {out_dir}/")

    if args.json:
        report = {
            "target_wav": args.target_wav,
            "engine": args.engine,
            "fitness": args.fitness,
            "note": note_midi,
            "duration": duration,
            "generations": args.generations,
            "pop": args.pop,
            "strands": args.strands,
            "best_fitness": best.fitness,
            "strand_fitness": [ind.fitness for ind in strand_winners],
            "elapsed_s": elapsed,
            "params": best.params,
        }
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  ✓  Informe → {args.json}")


def cmd_render(args):
    patch = load_patch(args.patch_json)
    engine = ENGINES.get(patch.engine)
    if engine is None:
        sys.exit(f"✗  Motor desconocido en patch: {patch.engine!r}")
    note_midi = _parse_note(args.note) if args.note else patch.note
    duration = args.duration if args.duration else patch.duration
    velocity = args.velocity if args.velocity is not None else patch.velocity
    sr = args.sr

    audio = engine.render(patch.params, note_midi, duration, velocity, sr)
    sf = _import_soundfile()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, _normalize_peak(audio), sr, subtype="FLOAT")
    print(f"  ✓  {patch.engine} · nota {note_midi} · {duration:.2f}s → {args.out}")


def cmd_mutate(args):
    patch = load_patch(args.patch_json)
    engine = ENGINES.get(patch.engine)
    if engine is None:
        sys.exit(f"✗  Motor desconocido en patch: {patch.engine!r}")
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = _import_soundfile() if args.render else None

    for i in range(1, args.n + 1):
        mutated = _mutate(patch.params, engine.params, rate=1.0, amount=args.amount)
        variant = Patch(engine=patch.engine, note=patch.note, duration=patch.duration,
                          velocity=patch.velocity, params=mutated)
        json_path = out_dir / f"variant_{i:02d}.json"
        save_patch(variant, str(json_path))
        if args.render:
            audio = engine.render(mutated, patch.note, patch.duration, patch.velocity, args.sr)
            wav_path = out_dir / f"variant_{i:02d}.wav"
            sf.write(str(wav_path), _normalize_peak(audio), args.sr, subtype="FLOAT")

    suffix = " (+ WAV)" if args.render else ""
    print(f"  ✓  {args.n} variante(s){suffix} → {out_dir}/")


def cmd_list_engines(args):
    for name, engine in ENGINES.items():
        print(f"\n{name}  —  {engine.description}")
        for p in engine.params:
            print(f"    {p.name:<16} [{p.lo:g} .. {p.hi:g}]  default={p.default:g}")


def cmd_info(args):
    patch = load_patch(args.patch_json)
    print(f"engine    {patch.engine}")
    print(f"note      {patch.note}")
    print(f"duration  {patch.duration:.3f}s")
    print(f"velocity  {patch.velocity:.3f}")
    print("params")
    for k, v in patch.params.items():
        print(f"    {k:<16} {v:g}")


def cmd_pitch(args):
    audio, sr = _read_mono_wav(args.wav)
    freq_hz, confidence = detect_pitch(audio, sr, fmin=args.pitch_fmin, fmax=args.pitch_fmax)
    if freq_hz is None:
        print("  ⚠  sin señal detectable (silencio o fragmento demasiado corto)")
        return
    note_midi = hz_to_midi(freq_hz)
    print(f"  freq        {freq_hz:.2f} Hz")
    print(f"  nota MIDI   {note_midi}  ({_midi_to_note_name(note_midi)})")
    print(f"  confianza   {confidence:.3f}")
    if confidence < args.pitch_confidence:
        print(f"  ⚠  por debajo del umbral por defecto de `match` ({args.pitch_confidence}) "
              f"— probablemente no tonal/percusivo; `match` caería a C4 salvo que uses --note.")


def cmd_explore(args):
    """Samplea N patches aleatorios (sin GA) por motor y los renderiza —
    para navegar/descubrir sonidos por accidente, no para converger a un
    objetivo. Reusa _random_params (la misma inicialización sesgada hacia
    el default que usa el GA), así que la mayoría de patches no salen
    "contaminados" con vibrato/unison/drive espurios a menos que toque."""
    engines = [args.engine] if args.engine else list(ENGINES.keys())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    sf = _import_soundfile()
    note_midi = _parse_note(args.note) if args.note else 60
    count = 0
    for ename in engines:
        engine = ENGINES[ename]
        for i in range(args.n):
            params = _clip_params(_random_params(engine.params), engine.params)
            audio = engine.render(params, note_midi, args.duration, args.velocity, args.sr)
            patch = Patch(engine=ename, note=note_midi, duration=args.duration,
                           velocity=args.velocity, params=params)
            base_name = f"{ename}_{i+1:02d}"
            save_patch(patch, str(out_dir / f"{base_name}.json"))
            sf.write(str(out_dir / f"{base_name}.wav"), _normalize_peak(audio),
                     args.sr, subtype="FLOAT")
            count += 1
    print(f"  ✓  {count} patch(es) explorados ({len(engines)} motor(es) × {args.n}) → {out_dir}/")


def cmd_morph(args):
    """Interpola linealmente entre dos patch.json (mismo motor) en N pasos
    — para viajar deliberadamente por el espacio de un timbre a otro, en
    vez de mutar al azar."""
    patch_a = load_patch(args.patch_a)
    patch_b = load_patch(args.patch_b)
    if patch_a.engine != patch_b.engine:
        sys.exit(f"✗  Los dos patches deben usar el mismo motor "
                  f"({patch_a.engine!r} vs {patch_b.engine!r}).")
    engine = ENGINES.get(patch_a.engine)
    if engine is None:
        sys.exit(f"✗  Motor desconocido: {patch_a.engine!r}")
    specs = {p.name: p for p in engine.params}
    keys = sorted(set(patch_a.params) | set(patch_b.params))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = _import_soundfile() if args.render else None
    note_midi = patch_a.note
    duration = patch_a.duration
    velocity = patch_a.velocity

    for i in range(args.steps):
        t = i / (args.steps - 1) if args.steps > 1 else 0.0
        params = {}
        for k in keys:
            va = patch_a.params.get(k, specs[k].default if k in specs else 0.0)
            vb = patch_b.params.get(k, specs[k].default if k in specs else 0.0)
            params[k] = va + t * (vb - va)
        if any(k not in specs for k in params):
            params = {k: v for k, v in params.items() if k in specs}
        variant = Patch(engine=patch_a.engine, note=note_midi, duration=duration,
                          velocity=velocity, params=params)
        step_name = f"morph_{i+1:02d}_t{t:.2f}"
        save_patch(variant, str(out_dir / f"{step_name}.json"))
        if args.render:
            audio = engine.render(params, note_midi, duration, velocity, args.sr)
            sf.write(str(out_dir / f"{step_name}.wav"), _normalize_peak(audio),
                     args.sr, subtype="FLOAT")

    suffix = " (+ WAV)" if args.render else ""
    print(f"  ✓  {args.steps} paso(s){suffix} → {out_dir}/")


def _add_common_synth_args(p, require_out=True):
    p.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genopatch",
        description="Matching de patches de síntesis por búsqueda evolutiva "
                     "(inspirado en Genopatch de Synplant 2, sin red neuronal).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── match ─────────────────────────────────────────────────────────────
    p = sub.add_parser("match", help="target.wav → patch.json vía algoritmo genético")
    p.add_argument("target_wav", help="WAV objetivo a reproducir")
    p.add_argument("--engine", default="fm2", choices=list(ENGINES.keys()),
                   help="Motor de síntesis (default: fm2)")
    p.add_argument("--fitness", default="spectral", choices=["spectral", "reference"],
                   help="Función de distancia (default: spectral)")
    p.add_argument("--fitness-backend", default="spectral", choices=["spectral", "latent"],
                   help="Backend de audio_reference_scorer.py si --fitness reference "
                        "(default: spectral)")
    p.add_argument("--note", default=None,
                   help="Nota MIDI o nombre (p.ej. 60, C4). Default: detección automática "
                        "de pitch del WAV objetivo (fallback a 60/C4 si no hay pitch claro)")
    p.add_argument("--pitch-fmin", type=float, default=40.0,
                   help="Frecuencia mínima buscada en la detección de pitch (default: 40Hz)")
    p.add_argument("--pitch-fmax", type=float, default=2000.0,
                   help="Frecuencia máxima buscada en la detección de pitch (default: 2000Hz)")
    p.add_argument("--pitch-confidence", type=float, default=0.4,
                   help="Confianza mínima del pico de autocorrelación para aceptar el pitch "
                        "detectado; por debajo, cae a --note 60/C4 (default: 0.4)")
    p.add_argument("--duration", type=float, default=None,
                   help="Duración en segundos (default: duración del WAV objetivo)")
    p.add_argument("--velocity", type=float, default=1.0, help="Velocity 0-1 (default: 1.0)")
    p.add_argument("--generations", type=int, default=60, help="Generaciones (default: 60)")
    p.add_argument("--pop", type=int, default=40, help="Tamaño de población (default: 40)")
    p.add_argument("--elite", type=int, default=4, help="Individuos élite por generación (default: 4)")
    p.add_argument("--mutation-rate", type=float, default=0.25,
                   help="Probabilidad de mutar cada gen (default: 0.25)")
    p.add_argument("--mutation-amount", type=float, default=0.12,
                   help="Amplitud de mutación, fracción del rango del parámetro (default: 0.12)")
    p.add_argument("--strands", type=int, default=1,
                   help="Nº de poblaciones independientes, cada una con su semilla "
                        "(default: 1; homenaje a las 4 ramas de Genopatch)")
    p.add_argument("--strand-out-dir", default=None,
                   help="Si --strands > 1, guarda el ganador de cada strand aquí")
    p.add_argument("--seed", type=int, default=0, help="Semilla base (default: 0)")
    p.add_argument("--sr", type=int, default=None,
                   help="Sample rate de trabajo (default: el del WAV objetivo)")
    p.add_argument("--out", required=True, help="patch.json de salida (mejor de todos los strands)")
    p.add_argument("--preview", default=None, help="WAV de previsualización del mejor patch")
    p.add_argument("--json", default=None, help="Informe opcional (fitness, params, tiempos)")
    p.add_argument("--quiet", action="store_true", help="No imprimir progreso por generación")
    p.set_defaults(func=cmd_match)

    # ── render ────────────────────────────────────────────────────────────
    p = sub.add_parser("render", help="patch.json → WAV")
    p.add_argument("patch_json")
    p.add_argument("--note", default=None, help="Sobreescribe la nota del patch")
    p.add_argument("--duration", type=float, default=None, help="Sobreescribe la duración")
    p.add_argument("--velocity", type=float, default=None, help="Sobreescribe la velocity")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_render)

    # ── mutate ────────────────────────────────────────────────────────────
    p = sub.add_parser("mutate", help="patch.json → N variantes (nueva semilla desde un patch)")
    p.add_argument("patch_json")
    p.add_argument("-n", type=int, default=8, help="Nº de variantes (default: 8)")
    p.add_argument("--amount", type=float, default=0.15,
                   help="Amplitud de mutación, fracción del rango (default: 0.15)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="También renderiza cada variante a WAV")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_mutate)

    # ── list-engines ──────────────────────────────────────────────────────
    p = sub.add_parser("list-engines", help="Lista motores registrados y sus parámetros")
    p.set_defaults(func=cmd_list_engines)

    # ── info ──────────────────────────────────────────────────────────────
    p = sub.add_parser("info", help="Imprime los parámetros de un patch.json")
    p.add_argument("patch_json")
    p.set_defaults(func=cmd_info)

    # ── pitch ─────────────────────────────────────────────────────────────
    p = sub.add_parser("pitch", help="Detecta la nota fundamental de un WAV (diagnóstico)")
    p.add_argument("wav")
    p.add_argument("--pitch-fmin", type=float, default=40.0)
    p.add_argument("--pitch-fmax", type=float, default=2000.0)
    p.add_argument("--pitch-confidence", type=float, default=0.4,
                   help="Solo para el aviso de umbral (default: 0.4, igual que en `match`)")
    p.set_defaults(func=cmd_pitch)

    # ── explore ───────────────────────────────────────────────────────────
    p = sub.add_parser("explore", help="Samplea N patches aleatorios por motor (sin GA) para navegar")
    p.add_argument("--engine", default=None, choices=list(ENGINES.keys()),
                   help="Motor a explorar (default: todos)")
    p.add_argument("-n", type=int, default=8, help="Patches por motor (default: 8)")
    p.add_argument("--note", default=None, help="Nota MIDI o nombre (default: 60/C4)")
    p.add_argument("--duration", type=float, default=1.5, help="Duración en segundos (default: 1.5)")
    p.add_argument("--velocity", type=float, default=0.85, help="Velocity 0-1 (default: 0.85)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_explore)

    # ── morph ─────────────────────────────────────────────────────────────
    p = sub.add_parser("morph", help="Interpola linealmente entre dos patch.json en N pasos")
    p.add_argument("patch_a")
    p.add_argument("patch_b")
    p.add_argument("--steps", type=int, default=8, help="Nº de pasos, incluyendo extremos (default: 8)")
    p.add_argument("--render", action="store_true", help="También renderiza cada paso a WAV")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_morph)

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
