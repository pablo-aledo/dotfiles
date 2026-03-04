#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COMPLETER  v1.0                                      ║
║         Continuación inteligente de obras musicales incompletas              ║
║                                                                              ║
║  Dado un fragmento MIDI (boceto, intro, frase suelta), genera una           ║
║  continuación coherente que respeta el ADN musical del original:             ║
║  tonalidad, ritmo, contorno melódico, curvas emocionales, densidad,         ║
║  registro, carácter y posición narrativa dentro de una forma mayor.         ║
║                                                                              ║
║  ESTRATEGIAS DE COMPLETADO:                                                  ║
║  [S1] markov      — Cadena de Markov entrenada sobre el fragmento fuente    ║
║                     Continuación estilísticamente fiel, orgánica            ║
║  [S2] contour     — Extrapola el contorno melódico (arco, dirección)        ║
║                     Continuación que sigue la "inercia" del fragmento       ║
║  [S3] phrase      — Sintaxis antecedente/consecuente (phrase_builder)       ║
║                     Completa la frase con la respuesta gramatical correcta  ║
║  [S4] variation   — Genera variaciones del material existente               ║
║                     (variation_engine) y las encadena como continuación     ║
║  [S5] harvest     — Busca en una colección de MIDIs fragmentos compatibles  ║
║                     y los ensambla con coherencia (harvester + stitcher)    ║
║  [S6] narrative   — Infiere la posición narrativa del fragmento y diseña   ║
║                     la continuación según el arco dramático (narrator)      ║
║  [S7] evolve      — Evolución genética guiada por similitud al fragmento   ║
║                     fuente (evolver) — resultados más creativos/sorpresa    ║
║  [S8] blend       — Combina todas las estrategias y elige la mejor por     ║
║                     score de coherencia, compatibilidad y calidad           ║
║                                                                              ║
║  MODOS DE COMPLETADO:                                                        ║
║  [M1] extend      — Continúa la obra a partir del último compás             ║
║  [M2] fill        — Rellena un hueco entre dos fragmentos (A → ? → B)      ║
║  [M3] develop     — Desarrolla el material en una sección de desarrollo     ║
║  [M4] recapitulate— Genera una reexposición modificada del material inicial ║
║  [M5] coda        — Compone una coda/final desde el estado armónico actual  ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │ midi_dna_unified  → ADN completo del fragmento fuente              │    ║
║  │ harvester         → extrae motivos del fragmento fuente            │    ║
║  │ phrase_builder    → expande motivos en frases completas            │    ║
║  │ variation_engine  → genera variaciones del material                │    ║
║  │ evolver           → evolución genética guiada                      │    ║
║  │ analyzer          → vectorización y similitud para elegir mejor    │    ║
║  │ stitcher          → ensamblaje coherente de fragmentos             │    ║
║  │ narrator          → posición narrativa y arco dramático            │    ║
║  │ tension_designer  → curvas emocionales para la continuación        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  USO:                                                                        ║
║    python completer.py fragmento.mid                                         ║
║    python completer.py fragmento.mid --bars 16                               ║
║    python completer.py fragmento.mid --strategy markov --bars 8             ║
║    python completer.py fragmento.mid --strategy phrase --mode develop       ║
║    python completer.py fragmento.mid --strategy blend --candidates 5        ║
║    python completer.py fragmento.mid --end-with coda                        ║
║    python completer.py fragmento.mid --fill-to final.mid                    ║
║    python completer.py fragmento.mid --collection ./midis/ --strategy harvest║
║    python completer.py fragmento.mid --arc hero --position 0.3              ║
║    python completer.py fragmento.mid --strategy blend --verbose             ║
║    python completer.py fragmento.mid --no-stitch  (solo genera, no pega)   ║
║    python completer.py fragmento.mid --append (pega continuación al fuente) ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    fragmento.mid          MIDI de entrada (fragmento a completar)           ║
║    --bars N               Compases de continuación a generar (default: auto)║
║    --strategy S           Estrategia: markov|contour|phrase|variation|      ║
║                           harvest|narrative|evolve|blend (default: blend)   ║
║    --mode M               Modo: extend|fill|develop|recapitulate|coda       ║
║                           (default: extend)                                  ║
║    --candidates N         Candidatos a generar y evaluar (default: 3)       ║
║    --fill-to FILE         MIDI destino para modo fill (A → ? → B)          ║
║    --collection DIR       Carpeta de MIDIs para estrategia harvest          ║
║    --arc NAME             Arco narrativo: hero|tragedy|romance|sonata|…     ║
║    --position F           Posición en la obra 0.0-1.0 (default: auto)      ║
║    --end-with TYPE        Terminar la continuación con: coda|cadence|fade   ║
║    --key KEY              Forzar tonalidad destino (default: auto)          ║
║    --tempo BPM            Forzar tempo (default: heredado del fuente)       ║
║    --append               Concatenar resultado al MIDI fuente               ║
║    --no-stitch            No usar stitcher: exportar solo la continuación   ║
║    --export-fingerprint   Exportar fingerprint de la continuación           ║
║    --output FILE          Archivo de salida (default: <fuente>_completed)   ║
║    --seed N               Semilla aleatoria (default: 42)                   ║
║    --verbose              Informe detallado del proceso                      ║
║    --report               Informe de decisiones + score de calidad          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Dependencias opcionales ──────────────────────────────────────────────────
try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False

try:
    import music21 as m21
    M21_OK = True
except ImportError:
    M21_OK = False

# ── Importaciones del ecosistema (opcionales, con fallback) ──────────────────
_SCRIPT_DIR = Path(__file__).parent

def _import_ecosystem(module_name: str):
    """Importa un módulo del ecosistema si está en el mismo directorio."""
    path = _SCRIPT_DIR / f"{module_name}.py"
    if not path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        return None

# ── Constantes ───────────────────────────────────────────────────────────────
CHROMATIC_SCALE = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
DORIAN_INTERVALS = [0, 2, 3, 5, 7, 9, 10]
PHRYGIAN_INTERVALS = [0, 1, 3, 5, 7, 8, 10]
LYDIAN_INTERVALS = [0, 2, 4, 6, 7, 9, 11]
MIXOLYDIAN_INTERVALS = [0, 2, 4, 5, 7, 9, 10]

MODE_INTERVALS = {
    'major': MAJOR_INTERVALS,
    'minor': MINOR_INTERVALS,
    'dorian': DORIAN_INTERVALS,
    'phrygian': PHRYGIAN_INTERVALS,
    'lydian': LYDIAN_INTERVALS,
    'mixolydian': MIXOLYDIAN_INTERVALS,
}

# Umbrales para inferir posición narrativa
NARRATIVE_POSITIONS = {
    (0.0, 0.15): 'intro',
    (0.15, 0.35): 'exposition',
    (0.35, 0.55): 'development',
    (0.55, 0.75): 'climax',
    (0.75, 0.90): 'resolution',
    (0.90, 1.0): 'coda',
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASES DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    """Nota MIDI con tiempo absoluto."""
    pitch: int       # MIDI pitch 0-127
    velocity: int    # 0-127
    start: float     # beats (float)
    duration: float  # beats (float)
    channel: int = 0


@dataclass
class SourceDNA:
    """ADN extraído del fragmento fuente."""
    path: str
    key_root: int = 60          # MIDI pitch de la tónica
    key_mode: str = 'major'
    tempo_bpm: float = 120.0
    beats_per_bar: int = 4
    n_bars: int = 8
    total_beats: float = 32.0

    # Melodía
    notes: List[Note] = field(default_factory=list)
    pitches: List[int] = field(default_factory=list)
    intervals: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    velocities: List[int] = field(default_factory=list)

    # Estadísticas
    pitch_mean: float = 60.0
    pitch_std: float = 3.0
    pitch_range: int = 12
    pitch_min: int = 55
    pitch_max: int = 75
    interval_mean: float = 0.0
    density: float = 1.0          # notas por beat
    velocity_mean: float = 70.0
    velocity_std: float = 10.0

    # Contorno
    contour: List[int] = field(default_factory=list)   # +1, 0, -1
    contour_tendency: float = 0.0  # >0 ascendente, <0 descendente
    last_pitch: int = 60
    last_interval: int = 0

    # Armonía
    scale_pitches: List[int] = field(default_factory=list)
    detected_chords: List[str] = field(default_factory=list)
    harmonic_rhythm: float = 2.0  # beats por acorde

    # Emocional
    tension_curve: List[float] = field(default_factory=list)
    arousal_curve: List[float] = field(default_factory=list)
    tension_mean: float = 0.5
    tension_direction: str = 'flat'  # rising, falling, arch, flat

    # Ritmo
    rhythm_pattern: List[float] = field(default_factory=list)
    syncopation: float = 0.0
    groove_offsets: List[float] = field(default_factory=list)

    # Markov
    markov_transitions: Dict = field(default_factory=dict)
    markov_starts: List = field(default_factory=list)

    # Fingerprint (para stitcher)
    fingerprint: Optional[Dict] = None


@dataclass
class CompletionResult:
    """Resultado de una continuación generada."""
    strategy: str
    mode: str
    notes: List[Note]
    score: float
    midi_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_lines: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ADN
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_key_simple(pitches: List[int]) -> Tuple[int, str]:
    """Detección de tonalidad simple por correlación con perfiles de Krumhansl."""
    if not pitches:
        return (0, 'major')  # C mayor por defecto

    # Perfil de Krumhansl-Schmuckler
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    pc_counts = Counter(p % 12 for p in pitches)
    pc_vector = [pc_counts.get(i, 0) for i in range(12)]
    total = sum(pc_vector) or 1
    pc_norm = [x / total for x in pc_vector]

    def correlate(profile, shift):
        shifted = profile[shift:] + profile[:shift]
        mean_p = sum(profile) / 12
        mean_v = sum(pc_norm) / 12
        num = sum((p - mean_p) * (v - mean_v) for p, v in zip(shifted, pc_norm))
        den_p = math.sqrt(sum((p - mean_p)**2 for p in shifted))
        den_v = math.sqrt(sum((v - mean_v)**2 for v in pc_norm))
        return num / (den_p * den_v + 1e-9)

    best_score = -999
    best_root = 0
    best_mode = 'major'
    for root in range(12):
        s_major = correlate(major_profile, root)
        s_minor = correlate(minor_profile, root)
        if s_major > best_score:
            best_score = s_major
            best_root = root
            best_mode = 'major'
        if s_minor > best_score:
            best_score = s_minor
            best_root = root
            best_mode = 'minor'

    # Convertir root PC a MIDI en octava 4-5 (60-71)
    root_midi = best_root + 60
    return (root_midi, best_mode)


def _get_scale_pitches(root_midi: int, mode: str) -> List[int]:
    """Genera los pitch classes de la escala."""
    root_pc = root_midi % 12
    intervals = MODE_INTERVALS.get(mode, MAJOR_INTERVALS)
    return [(root_pc + iv) % 12 for iv in intervals]


def _snap_to_scale(pitch: int, scale_pcs: List[int]) -> int:
    """Ajusta un pitch al scale pitch class más cercano."""
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    # Buscar el más cercano
    best = min(scale_pcs, key=lambda s: min(abs(s - pc), 12 - abs(s - pc)))
    diff = best - pc
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12
    return pitch + diff


def extract_dna_from_midi(midi_path: str, verbose: bool = False) -> SourceDNA:
    """
    Extrae el ADN musical completo de un MIDI.
    Usa midi_dna_unified.UnifiedDNA si está disponible; si no, extracción propia.
    """
    dna = SourceDNA(path=midi_path)

    # ── Intentar usar UnifiedDNA del ecosistema ───────────────────────────────
    unified = _import_ecosystem('midi_dna_unified')
    if unified and M21_OK:
        try:
            udna = unified.UnifiedDNA(midi_path)
            udna.extract(verbose=False)

            # Trasvasar datos al SourceDNA
            dna.tempo_bpm = udna.tempo_bpm or 120.0
            dna.beats_per_bar = udna.beats_per_bar or 4
            dna.n_bars = udna.n_bars or 8
            dna.total_beats = dna.n_bars * dna.beats_per_bar

            if udna.key_obj:
                tonic = udna.key_obj.tonic
                if tonic:
                    pc = CHROMATIC_SCALE.index(str(tonic.name).replace('-','b').replace('##','').rstrip('#'))
                    dna.key_root = 60 + pc
                mode_str = str(getattr(udna.key_obj, 'mode', 'major')).lower()
                dna.key_mode = mode_str if mode_str in MODE_INTERVALS else 'major'

            if udna.pitches:
                dna.pitches = list(udna.pitches)
                dna.intervals = list(udna.intervals) if udna.intervals else []
                dna.durations = list(udna.durations) if udna.durations else []

            if udna.tension_curve:
                dna.tension_curve = list(udna.tension_curve)
            if udna.arousal_curve:
                dna.arousal_curve = list(udna.arousal_curve)

            # Markov del ecosistema
            if udna.markov and udna.markov.transitions:
                dna.markov_transitions = dict(udna.markov.transitions)
                dna.markov_starts = list(udna.markov.start_states.keys())

            if verbose:
                print(f"  [DNA] Extraído con UnifiedDNA: {len(dna.pitches)} notas, "
                      f"tonalidad {dna.key_root%12}={CHROMATIC_SCALE[dna.key_root%12]} {dna.key_mode}, "
                      f"tempo {dna.tempo_bpm:.0f} BPM")
        except Exception as e:
            if verbose:
                print(f"  [DNA] UnifiedDNA falló ({e}), usando extracción propia")
            _extract_dna_own(midi_path, dna, verbose)
    else:
        _extract_dna_own(midi_path, dna, verbose)

    # ── Post-procesar ADN ─────────────────────────────────────────────────────
    _compute_dna_statistics(dna)

    # ── Fingerprint (para stitcher) ───────────────────────────────────────────
    fp_path = midi_path.replace('.mid', '.fingerprint.json')
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            dna.fingerprint = json.load(f)

    return dna


def _extract_dna_own(midi_path: str, dna: SourceDNA, verbose: bool = False):
    """Extracción propia de ADN usando mido."""
    if not MIDO_OK:
        if verbose:
            print("  [DNA] mido no disponible; usando valores por defecto")
        return

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    # Detectar tempo
    tempo_us = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
                break

    dna.tempo_bpm = 60_000_000 / tempo_us

    # Extraer notas
    all_notes = []
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start_tick, vel = active.pop(key)
                    dur_ticks = abs_tick - start_tick
                    start_beats = start_tick / tpb
                    dur_beats = dur_ticks / tpb
                    if dur_beats > 0.05:
                        all_notes.append(Note(
                            pitch=msg.note,
                            velocity=vel,
                            start=start_beats,
                            duration=dur_beats,
                            channel=msg.channel,
                        ))

    all_notes.sort(key=lambda n: n.start)
    dna.notes = all_notes

    if all_notes:
        dna.pitches = [n.pitch for n in all_notes]
        dna.durations = [n.duration for n in all_notes]
        dna.velocities = [n.velocity for n in all_notes]
        dna.intervals = [dna.pitches[i+1] - dna.pitches[i]
                         for i in range(len(dna.pitches)-1)]
        dna.total_beats = max(n.start + n.duration for n in all_notes)
        dna.n_bars = max(1, int(dna.total_beats / dna.beats_per_bar))

    # Detectar tonalidad
    if dna.pitches:
        dna.key_root, dna.key_mode = _detect_key_simple(dna.pitches)

    # Detectar compás (simplificado 4/4)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                dna.beats_per_bar = msg.numerator
                break

    if verbose:
        print(f"  [DNA] Extracción propia: {len(all_notes)} notas, "
              f"tonalidad {CHROMATIC_SCALE[dna.key_root%12]} {dna.key_mode}, "
              f"{dna.n_bars} compases, {dna.tempo_bpm:.0f} BPM")


def _compute_dna_statistics(dna: SourceDNA):
    """Calcula estadísticas derivadas del ADN."""
    if not dna.pitches:
        return

    p = dna.pitches
    dna.pitch_mean = sum(p) / len(p)
    dna.pitch_min = min(p)
    dna.pitch_max = max(p)
    dna.pitch_range = dna.pitch_max - dna.pitch_min
    dna.last_pitch = p[-1]

    if NP_OK:
        arr = np.array(p, dtype=float)
        dna.pitch_std = float(np.std(arr))
    else:
        mean = dna.pitch_mean
        dna.pitch_std = math.sqrt(sum((x - mean)**2 for x in p) / len(p))

    if dna.intervals:
        dna.interval_mean = sum(dna.intervals) / len(dna.intervals)
        dna.last_interval = dna.intervals[-1]
        dna.contour = [1 if iv > 0 else (-1 if iv < 0 else 0) for iv in dna.intervals]
        dna.contour_tendency = sum(dna.contour) / len(dna.contour)

    total_beats = dna.total_beats or 1.0
    dna.density = len(dna.pitches) / total_beats

    if dna.velocities:
        dna.velocity_mean = sum(dna.velocities) / len(dna.velocities)
        mean_v = dna.velocity_mean
        dna.velocity_std = math.sqrt(sum((v - mean_v)**2 for v in dna.velocities) / len(dna.velocities))

    dna.scale_pitches = _get_scale_pitches(dna.key_root, dna.key_mode)

    # Tension curve simple si no viene del ecosistema
    if not dna.tension_curve and dna.n_bars > 0:
        dna.tension_curve = _estimate_tension_curve(dna)

    if dna.tension_curve:
        dna.tension_mean = sum(dna.tension_curve) / len(dna.tension_curve)
        if len(dna.tension_curve) > 1:
            first_half = dna.tension_curve[:len(dna.tension_curve)//2]
            second_half = dna.tension_curve[len(dna.tension_curve)//2:]
            mean_first = sum(first_half) / len(first_half)
            mean_second = sum(second_half) / len(second_half)
            if mean_second > mean_first + 0.1:
                dna.tension_direction = 'rising'
            elif mean_second < mean_first - 0.1:
                dna.tension_direction = 'falling'
            elif max(dna.tension_curve) > 0.65 and dna.tension_curve[0] < 0.5 and dna.tension_curve[-1] < 0.5:
                dna.tension_direction = 'arch'
            else:
                dna.tension_direction = 'flat'

    # Construir Markov propio si no viene del ecosistema
    if not dna.markov_transitions and len(dna.intervals) >= 4:
        _build_markov(dna)


def _estimate_tension_curve(dna: SourceDNA) -> List[float]:
    """Estima la curva de tensión a partir del contorno melódico y la densidad."""
    if not dna.pitches or not dna.notes:
        return [0.5] * max(dna.n_bars, 1)

    bpb = dna.beats_per_bar
    curve = []
    for bar in range(dna.n_bars):
        bar_start = bar * bpb
        bar_end = bar_start + bpb
        bar_notes = [n for n in dna.notes if bar_start <= n.start < bar_end]
        if bar_notes:
            mean_p = sum(n.pitch for n in bar_notes) / len(bar_notes)
            normalized_p = (mean_p - dna.pitch_min) / max(dna.pitch_range, 1)
            density_factor = min(1.0, len(bar_notes) / (bpb * 2))
            tension = 0.4 * normalized_p + 0.6 * density_factor
            curve.append(min(1.0, max(0.0, tension)))
        else:
            curve.append(0.2)
    return curve


def _build_markov(dna: SourceDNA):
    """Construye tabla de Markov de 2º orden sobre (intervalo, duración)."""
    def q_dur(d):
        buckets = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(buckets, key=lambda x: abs(x - d))

    def q_int(i):
        return max(-12, min(12, round(i)))

    ivals = dna.intervals
    durs = dna.durations

    n = min(len(ivals), len(durs) - 1)
    if n < 3:
        return

    states = [(q_int(ivals[i]), q_dur(durs[i])) for i in range(n)]
    transitions = defaultdict(Counter)
    starts = Counter()

    for i in range(min(4, len(states))):
        starts[states[i]] += 1

    for i in range(len(states) - 2):
        ctx = (states[i], states[i+1])
        transitions[ctx][states[i+2]] += 1
    for i in range(len(states) - 1):
        ctx1 = (states[i],)
        transitions[ctx1][states[i+1]] += 1

    dna.markov_transitions = dict(transitions)
    dna.markov_starts = list(starts.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTRATEGIAS DE COMPLETADO
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_markov(dna: SourceDNA, n_bars: int, mode: str,
                    position: float, seed: int, verbose: bool) -> List[Note]:
    """
    [S1] Continuación por cadena de Markov.
    Genera melodía siguiendo las transiciones estadísticas del ADN del fuente.
    Incluye sesgo por tensión para dirigir el desarrollo emocional.
    """
    random.seed(seed)
    if NP_OK:
        np.random.seed(seed)

    bpb = dna.beats_per_bar
    n_steps = int(n_bars * bpb * dna.density)
    n_steps = max(n_steps, n_bars * 2)

    # Curva de tensión para la continuación
    target_tension = _design_continuation_tension(dna, n_bars, mode, position)

    if verbose:
        print(f"  [Markov] Generando {n_steps} notas en {n_bars} compases")
        print(f"  [Markov] Tensión objetivo: {[f'{t:.2f}' for t in target_tension[:4]]}...")

    # Usar Markov del ecosistema si disponible
    unified = _import_ecosystem('midi_dna_unified')
    if unified and dna.markov_transitions:
        try:
            # Recrear el objeto MarkovMelody del ecosistema con los datos extraídos
            mm = unified.MarkovMelody(order=2)
            mm.transitions = defaultdict(Counter, dna.markov_transitions)
            mm.start_states = Counter({s: 1 for s in dna.markov_starts})

            # Determinar key_obj para el ecosistema
            key_str = f"{CHROMATIC_SCALE[dna.key_root % 12]} {dna.key_mode}"
            if M21_OK:
                try:
                    key_obj = m21.key.Key(
                        CHROMATIC_SCALE[dna.key_root % 12],
                        dna.key_mode
                    )
                    pitches_and_durs = mm.generate(
                        n_steps=n_steps,
                        start_pitch=dna.last_pitch,
                        key_obj=key_obj,
                        seed=seed,
                        tension_curve=target_tension
                    )
                    return _pitches_durs_to_notes(
                        pitches_and_durs, dna, bpb, target_tension, mode
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [Markov] Ecosistema falló ({e}), usando Markov propio")
        except Exception:
            pass

    # Markov propio
    return _markov_generate_own(dna, n_steps, n_bars, target_tension, seed, verbose)


def _markov_generate_own(dna: SourceDNA, n_steps: int, n_bars: int,
                          target_tension: List[float], seed: int,
                          verbose: bool) -> List[Note]:
    """Generación Markov propia, sin dependencias del ecosistema."""
    random.seed(seed)
    bpb = dna.beats_per_bar
    scale_pcs = dna.scale_pitches

    def tension_at(step, n):
        if not target_tension:
            return 0.5
        frac = step / max(n - 1, 1)
        pos = frac * (len(target_tension) - 1)
        lo_i = int(pos)
        hi_i = min(lo_i + 1, len(target_tension) - 1)
        t = pos - lo_i
        return target_tension[lo_i] * (1 - t) + target_tension[hi_i] * t

    def sample_interval(tension):
        """Muestrea un intervalo sesgado por tensión."""
        if tension > 0.65:
            weights = {-5:1, -4:2, -3:3, -2:4, -1:5, 0:3,
                       1:6, 2:8, 3:7, 4:5, 5:4, 7:3}
        elif tension < 0.35:
            weights = {-7:3, -5:4, -4:5, -3:7, -2:8, -1:6,
                       0:3, 1:5, 2:4, 3:3, 4:2, 5:1}
        else:
            weights = {-5:2, -4:3, -3:5, -2:7, -1:8, 0:4,
                       1:8, 2:7, 3:5, 4:3, 5:2}
        # Añadir sesgo de los intervalos reales del fuente
        if dna.intervals:
            for iv in dna.intervals:
                iv_c = max(-7, min(7, round(iv)))
                if iv_c in weights:
                    weights[iv_c] = int(weights[iv_c] * 1.5)
        choices = list(weights.keys())
        wts = [weights[c] for c in choices]
        total = sum(wts)
        r = random.random() * total
        cumsum = 0
        for c, w in zip(choices, wts):
            cumsum += w
            if r <= cumsum:
                return c
        return 0

    def sample_duration(tension):
        """Muestrea duración sesgada por tensión y densidad del fuente."""
        base_durs = dna.durations if dna.durations else [0.5, 1.0]
        dur_counts = Counter(round(d * 4) / 4 for d in base_durs)
        candidates = list(dur_counts.keys())
        weights = [dur_counts[c] for c in candidates]
        if tension > 0.65:
            weights = [w * (2.0 if c <= 0.5 else 0.5) for c, w in zip(candidates, weights)]
        elif tension < 0.35:
            weights = [w * (0.5 if c <= 0.25 else 1.5) for c, w in zip(candidates, weights)]
        total = sum(weights)
        if total == 0:
            return random.choice([0.5, 1.0])
        r = random.random() * total
        cumsum = 0
        for c, w in zip(candidates, weights):
            cumsum += w
            if r <= cumsum:
                return max(0.125, c)
        return 0.5

    # Generar secuencia
    current_pitch = dna.last_pitch
    notes = []
    current_beat = 0.0
    max_beats = n_bars * bpb

    step = 0
    while current_beat < max_beats and step < n_steps * 2:
        tension = tension_at(step, n_steps)
        interval = sample_interval(tension)
        dur = sample_duration(tension)

        new_pitch = current_pitch + interval
        # Respetar rango vocal razonable (C3-C6)
        while new_pitch > 84:
            new_pitch -= 12
        while new_pitch < 48:
            new_pitch += 12
        new_pitch = _snap_to_scale(new_pitch, scale_pcs)

        # Velocidad dinámica
        base_vel = dna.velocity_mean
        vel_variation = random.gauss(0, dna.velocity_std * 0.5)
        if tension > 0.65:
            vel_variation += 10
        elif tension < 0.35:
            vel_variation -= 5
        velocity = int(max(40, min(110, base_vel + vel_variation)))

        notes.append(Note(
            pitch=new_pitch,
            velocity=velocity,
            start=current_beat,
            duration=min(dur, max_beats - current_beat),
            channel=0,
        ))

        current_beat += dur
        current_pitch = new_pitch
        step += 1

    if verbose:
        print(f"  [Markov] Generadas {len(notes)} notas, "
              f"{current_beat:.1f}/{max_beats:.1f} beats cubiertos")

    return notes


def _pitches_durs_to_notes(pitches_durs: List[Tuple[int, float]],
                            dna: SourceDNA, bpb: int,
                            tension: List[float], mode: str) -> List[Note]:
    """Convierte lista de (pitch, dur) a lista de Note."""
    notes = []
    current_beat = 0.0
    for i, (pitch, dur) in enumerate(pitches_durs):
        t = tension[min(i, len(tension)-1)] if tension else 0.5
        base_vel = dna.velocity_mean
        vel = int(max(40, min(110, base_vel + random.gauss(0, 8) + t * 15)))
        notes.append(Note(
            pitch=pitch, velocity=vel,
            start=current_beat, duration=max(0.1, dur),
            channel=0
        ))
        current_beat += dur
    return notes


def strategy_contour(dna: SourceDNA, n_bars: int, mode: str,
                     position: float, seed: int, verbose: bool) -> List[Note]:
    """
    [S2] Continuación por extrapolación de contorno.
    Analiza la dirección y curvatura del contorno melódico y lo extrapola.
    Respeta la "inercia" del fragmento: si subía, sigue subiendo; etc.
    """
    random.seed(seed)
    bpb = dna.beats_per_bar
    scale_pcs = dna.scale_pitches
    n_steps = int(n_bars * bpb * dna.density)
    n_steps = max(n_steps, n_bars * 2)

    if verbose:
        print(f"  [Contour] Tendencia: {dna.contour_tendency:+.2f}, "
              f"último intervalo: {dna.last_interval:+d}")

    target_tension = _design_continuation_tension(dna, n_bars, mode, position)
    contour_tendency = dna.contour_tendency

    # Ajustar tendencia según modo
    if mode == 'develop':
        # Desarrollo: amplificar la tendencia con más variabilidad
        contour_tendency *= 1.3
    elif mode == 'coda':
        # Coda: siempre descender hacia la resolución
        contour_tendency = -0.4
    elif mode == 'recapitulate':
        # Reexposición: volver al rango original (tender hacia pitch_mean)
        pass

    notes = []
    current_pitch = dna.last_pitch
    current_beat = 0.0
    max_beats = n_bars * bpb

    # Estimar dirección del contorno basado en el origen del fragmento
    tendency_weight = abs(contour_tendency)
    dominant_dir = 1 if contour_tendency > 0 else (-1 if contour_tendency < 0 else 0)

    step = 0
    while current_beat < max_beats:
        tension = target_tension[min(int(step * len(target_tension) / n_steps),
                                      len(target_tension) - 1)] if target_tension else 0.5

        # Decidir intervalo basado en la tendencia del contorno
        r = random.random()
        if r < tendency_weight * 0.6:
            # Seguir la tendencia dominante
            base_interval = dominant_dir * random.choice([1, 2, 2, 3])
        elif r < tendency_weight * 0.6 + 0.25:
            # Corrección de rango (volver hacia el centro si nos alejamos)
            center = dna.pitch_mean
            dist = current_pitch - center
            base_interval = -1 if dist > 4 else (1 if dist < -4 else 0)
        else:
            # Movimiento por grado conjunto neutro
            base_interval = random.choice([-2, -1, 0, 1, 2])

        # Matiz de tensión
        if tension > 0.7 and base_interval <= 0:
            base_interval = abs(base_interval) + 1
        elif tension < 0.3 and base_interval > 0:
            base_interval = -abs(base_interval)

        new_pitch = current_pitch + base_interval
        # Mantener dentro del rango del fuente (±octava)
        lo = max(36, dna.pitch_min - 5)
        hi = min(96, dna.pitch_max + 5)
        if new_pitch > hi:
            new_pitch -= 12
        if new_pitch < lo:
            new_pitch += 12
        new_pitch = _snap_to_scale(max(lo, min(hi, new_pitch)), scale_pcs)

        # Duración: heredar del patrón original
        if dna.durations:
            dur_idx = step % len(dna.durations)
            dur = dna.durations[dur_idx] * random.uniform(0.85, 1.15)
        else:
            dur = random.choice([0.5, 0.5, 1.0, 1.0, 2.0])
        dur = max(0.125, dur)

        velocity = int(max(40, min(110,
            dna.velocity_mean + random.gauss(0, dna.velocity_std * 0.4)
            + tension * 12
        )))

        notes.append(Note(
            pitch=new_pitch, velocity=velocity,
            start=current_beat, duration=min(dur, max_beats - current_beat),
            channel=0
        ))
        current_beat += dur
        current_pitch = new_pitch
        step += 1

    if verbose:
        print(f"  [Contour] Generadas {len(notes)} notas")

    return notes


def strategy_phrase(dna: SourceDNA, n_bars: int, mode: str,
                    position: float, seed: int, verbose: bool) -> List[Note]:
    """
    [S3] Completado por sintaxis de frase (antecedente/consecuente).
    Usa phrase_builder si está disponible; si no, genera una respuesta sintáctica.
    El fragmento fuente se trata como el antecedente: se genera el consecuente.
    """
    phrase_builder = _import_ecosystem('phrase_builder')

    if phrase_builder:
        try:
            # Intentar usar phrase_builder del ecosistema
            # Exportar notas del fuente como motivo
            motif = _notes_to_phrase_motif(dna, phrase_builder)
            if motif:
                key_str = f"{CHROMATIC_SCALE[dna.key_root % 12]} {dna.key_mode}"
                root_pc = dna.key_root % 12
                form_type = _mode_to_phrase_form(mode)
                try:
                    result = phrase_builder.build_phrase(
                        motif=motif,
                        root_pc=root_pc,
                        mode=dna.key_mode,
                        form=form_type,
                        bars=n_bars,
                        seed=seed,
                    )
                    notes = _phrase_result_to_notes(result, dna)
                    if notes:
                        if verbose:
                            print(f"  [Phrase] Generado con phrase_builder "
                                  f"({form_type}): {len(notes)} notas")
                        return notes
                except Exception as e:
                    if verbose:
                        print(f"  [Phrase] phrase_builder falló ({e}), "
                              "usando frase sintáctica propia")
        except Exception:
            pass

    # Frase sintáctica propia: consecuente que responde al antecedente
    return _generate_consequent(dna, n_bars, mode, seed, verbose)


def _notes_to_phrase_motif(dna: SourceDNA, pb_module) -> Optional[List]:
    """Convierte las primeras notas del ADN a motivo de phrase_builder."""
    if not dna.notes:
        return None
    try:
        # Tomar las primeras 4 notas como motivo
        motif_notes = dna.notes[:min(8, len(dna.notes))]
        motif = []
        for n in motif_notes:
            mn = pb_module.MotifNote(
                pitch=n.pitch,
                duration=n.duration,
                velocity=n.velocity,
                offset=n.start,
            )
            motif.append(mn)
        return motif
    except Exception:
        return None


def _phrase_result_to_notes(result, dna: SourceDNA) -> List[Note]:
    """Convierte resultado de phrase_builder a lista de Note."""
    notes = []
    try:
        all_notes = result.all_notes() if hasattr(result, 'all_notes') else []
        for mn in all_notes:
            notes.append(Note(
                pitch=mn.pitch,
                velocity=getattr(mn, 'velocity', int(dna.velocity_mean)),
                start=mn.offset,
                duration=mn.duration,
                channel=0,
            ))
    except Exception:
        pass
    return notes


def _mode_to_phrase_form(mode: str) -> str:
    mapping = {
        'extend': 'period',
        'develop': 'sentence',
        'recapitulate': 'period',
        'coda': 'bar_form',
        'fill': 'period',
    }
    return mapping.get(mode, 'period')


def _generate_consequent(dna: SourceDNA, n_bars: int, mode: str,
                          seed: int, verbose: bool) -> List[Note]:
    """
    Genera un consecuente sintáctico propio.
    El consecuente es esencialmente el antecedente con:
    - Cadencia de reposo en lugar de semicadencia
    - Posiblemente terminando en la tónica
    """
    random.seed(seed)
    bpb = dna.beats_per_bar
    scale_pcs = dna.scale_pitches
    tonic = dna.key_root

    notes_out = []
    # Copiar y transformar las notas del fuente como base del consecuente
    source_notes = dna.notes[:] if dna.notes else []
    if not source_notes:
        return strategy_markov(dna, n_bars, mode, 0.5, seed, verbose)

    max_beats = n_bars * bpb
    scale_up = max_beats / max(source_notes[-1].start + source_notes[-1].duration, 1.0)

    # Parallel consequent: igual que el antecedente pero con cadencia distinta
    for i, n in enumerate(source_notes):
        new_start = n.start * scale_up
        if new_start >= max_beats:
            break

        # Cadencia: hacia el final, ajustar hacia la tónica
        progress = new_start / max_beats
        if progress > 0.8:
            # Hacia la tónica
            nearest_tonic = tonic
            while nearest_tonic < n.pitch - 6:
                nearest_tonic += 12
            while nearest_tonic > n.pitch + 6:
                nearest_tonic -= 12
            target = nearest_tonic
            # Blend hacia la tónica según proximidad al final
            blend = (progress - 0.8) / 0.2
            new_pitch = int(n.pitch * (1 - blend) + target * blend)
            new_pitch = _snap_to_scale(new_pitch, scale_pcs)
        else:
            # Variación leve del original
            new_pitch = _snap_to_scale(
                n.pitch + random.choice([-1, 0, 0, 0, 1]),
                scale_pcs
            )

        dur = n.duration * scale_up
        notes_out.append(Note(
            pitch=new_pitch,
            velocity=n.velocity,
            start=new_start,
            duration=min(dur, max_beats - new_start),
            channel=0,
        ))

    if verbose:
        print(f"  [Phrase] Consecuente propio: {len(notes_out)} notas, "
              f"cadencia hacia tónica {CHROMATIC_SCALE[tonic % 12]}")

    return notes_out


def strategy_variation(dna: SourceDNA, n_bars: int, mode: str,
                        position: float, seed: int, verbose: bool) -> List[Note]:
    """
    [S4] Continuación por variación del material existente.
    Llama a variation_engine.py para generar variaciones y usa la más adecuada
    según el modo (develop → V13 emocional + V01 inversión, etc.)
    """
    variation_engine = _import_ecosystem('variation_engine')

    if variation_engine:
        try:
            # Seleccionar variaciones según el modo
            var_ids = _select_variations_for_mode(mode)
            # Usar la API interna de variation_engine si está disponible
            udna_mod = _import_ecosystem('midi_dna_unified')
            if udna_mod and M21_OK:
                udna = udna_mod.UnifiedDNA(dna.path)
                udna.extract(verbose=False)
                melody = variation_engine.extract_melody_from_midi(dna.path)
                if melody:
                    best_var = _pick_best_variation(
                        variation_engine, udna, melody, n_bars,
                        var_ids, mode, seed, verbose
                    )
                    if best_var:
                        return best_var
        except Exception as e:
            if verbose:
                print(f"  [Variation] variation_engine falló ({e}), "
                      "usando variación propia")

    # Variación propia: aplicar transformaciones básicas a las notas del fuente
    return _apply_variation_own(dna, n_bars, mode, seed, verbose)


def _select_variations_for_mode(mode: str) -> List[str]:
    """Selecciona las variaciones más apropiadas para cada modo."""
    mapping = {
        'extend':       ['V06', 'V10', 'V07'],   # ornamentación, rítmica, acompañamiento
        'develop':      ['V01', 'V09', 'V13'],   # inversión, modal, emocional
        'recapitulate': ['V06', 'V08', 'V12'],   # ornamentación, transpuesta, textural
        'coda':         ['V04', 'V11', 'V13'],   # aumentación, armónica, emocional
        'fill':         ['V05', 'V10', 'V06'],   # diminución, rítmica, ornamentación
    }
    return mapping.get(mode, ['V06', 'V10'])


def _pick_best_variation(ve_mod, udna, melody, n_bars, var_ids, mode, seed, verbose):
    """Intenta generar variaciones y devuelve las notas de la mejor."""
    var_map = {
        'V01': ve_mod.variation_inversion,
        'V02': ve_mod.variation_retrograde,
        'V03': ve_mod.variation_retrograde_inversion,
        'V04': ve_mod.variation_augmentation,
        'V05': ve_mod.variation_diminution,
        'V06': ve_mod.variation_ornamentation,
        'V07': ve_mod.variation_accompaniment,
        'V08': ve_mod.variation_transposed,
        'V09': ve_mod.variation_modal,
        'V10': ve_mod.variation_rhythmic,
        'V11': ve_mod.variation_harmonic,
        'V12': ve_mod.variation_textural,
        'V13': ve_mod.variation_emotional,
        'V14': ve_mod.variation_stochastic,
        'V15': ve_mod.variation_counterpoint,
    }

    for var_id in var_ids:
        fn = var_map.get(var_id)
        if fn:
            try:
                result = fn(udna, melody, n_bars, seed=seed)
                if result and result.get('melody'):
                    raw = result['melody']
                    notes = []
                    beat = 0.0
                    for pitch, vel, start, dur, ch in raw:
                        notes.append(Note(pitch=pitch, velocity=vel,
                                         start=start, duration=dur, channel=ch))
                    if verbose:
                        print(f"  [Variation] Usando {var_id}: {len(notes)} notas")
                    return notes
            except Exception:
                continue
    return None


def _apply_variation_own(dna: SourceDNA, n_bars: int, mode: str,
                          seed: int, verbose: bool) -> List[Note]:
    """Variación propia básica."""
    random.seed(seed)
    if not dna.notes:
        return strategy_markov(dna, n_bars, mode, 0.5, seed, verbose)

    bpb = dna.beats_per_bar
    max_beats = n_bars * bpb
    scale_pcs = dna.scale_pitches

    # Escalar temporalmente el fuente para llenar n_bars
    src_dur = dna.total_beats or (len(dna.notes) * 0.5)
    time_scale = max_beats / src_dur

    notes_out = []
    for n in dna.notes:
        new_start = n.start * time_scale
        if new_start >= max_beats:
            break

        # Transformación según modo
        if mode == 'develop':
            # Inversión parcial + variación rítmica
            interval_from_last = n.pitch - dna.pitch_mean
            new_pitch = int(dna.pitch_mean - interval_from_last * 0.7)
            new_pitch += random.choice([-1, 0, 0, 1])
        elif mode == 'coda':
            # Aumentación: estirar las duraciones, tender hacia tónica
            progress = new_start / max_beats
            tonic_pitch = dna.key_root
            while tonic_pitch < n.pitch - 6:
                tonic_pitch += 12
            blend = progress * 0.5
            new_pitch = int(n.pitch * (1 - blend) + tonic_pitch * blend)
            time_scale *= 1.3
        elif mode == 'recapitulate':
            # Leve ornamentación
            new_pitch = n.pitch
            if random.random() < 0.2:
                new_pitch += random.choice([-1, 1])
        else:
            new_pitch = n.pitch + random.choice([-2, -1, 0, 0, 1, 2])

        new_pitch = _snap_to_scale(max(36, min(96, new_pitch)), scale_pcs)
        dur = n.duration * time_scale * random.uniform(0.9, 1.1)

        notes_out.append(Note(
            pitch=new_pitch,
            velocity=n.velocity,
            start=new_start,
            duration=min(max(0.1, dur), max_beats - new_start),
            channel=0,
        ))

    if verbose:
        print(f"  [Variation] Variación propia ({mode}): {len(notes_out)} notas")

    return notes_out


def strategy_harvest(dna: SourceDNA, n_bars: int, mode: str,
                     position: float, seed: int, verbose: bool,
                     collection_dir: Optional[str] = None) -> List[Note]:
    """
    [S5] Continuación buscando fragmentos compatibles en una colección.
    Usa harvester para extraer fragmentos y analyzer para encontrar los más
    similares al contexto actual.
    """
    if not collection_dir or not os.path.isdir(collection_dir):
        if verbose:
            print("  [Harvest] No hay colección disponible; fallback a Markov")
        return strategy_markov(dna, n_bars, mode, position, seed, verbose)

    analyzer = _import_ecosystem('analyzer')
    harvester = _import_ecosystem('harvester')

    if not (analyzer and harvester):
        if verbose:
            print("  [Harvest] Módulos no disponibles; fallback a Markov")
        return strategy_markov(dna, n_bars, mode, position, seed, verbose)

    # Buscar MIDIs en la colección
    midi_files = list(Path(collection_dir).glob('*.mid'))
    midi_files += list(Path(collection_dir).glob('*.midi'))
    if not midi_files:
        return strategy_markov(dna, n_bars, mode, position, seed, verbose)

    if verbose:
        print(f"  [Harvest] Buscando en {len(midi_files)} MIDIs de la colección")

    try:
        # Usar analyzer para encontrar los más similares al fuente
        result = analyzer.mode_nearest(
            query_path=dna.path,
            collection_paths=[str(f) for f in midi_files[:20]],
            args=type('args', (), {
                'top': 3, 'level': 'auto', 'metric': 'euclidean',
                'dimensions': None, 'cache': None, 'use_cache': None,
                'verbose': False, 'plot': False
            })()
        )

        if result and result.get('nearest'):
            # Tomar el más similar y extraer fragmentos
            nearest_path = result['nearest'][0]['path']
            if verbose:
                print(f"  [Harvest] MIDI más similar: {Path(nearest_path).name} "
                      f"(sim={result['nearest'][0].get('similarity', 0):.2f})")

            # Extraer ADN del más similar
            similar_dna = extract_dna_from_midi(nearest_path, verbose=False)
            # Usar sus notas como continuación adaptada
            return _adapt_foreign_notes(similar_dna, dna, n_bars, seed, verbose)

    except Exception as e:
        if verbose:
            print(f"  [Harvest] Error ({e}); fallback a Markov")

    return strategy_markov(dna, n_bars, mode, position, seed, verbose)


def _adapt_foreign_notes(foreign: SourceDNA, target: SourceDNA,
                          n_bars: int, seed: int, verbose: bool) -> List[Note]:
    """Adapta notas de un MIDI extraño a la tonalidad y rango del fuente."""
    random.seed(seed)
    bpb = target.beats_per_bar
    max_beats = n_bars * bpb
    scale_pcs = target.scale_pitches

    if not foreign.notes:
        return []

    # Calcular transposición necesaria
    key_diff = (target.key_root % 12) - (foreign.key_root % 12)
    pitch_center_diff = int(target.pitch_mean - foreign.pitch_mean)
    transpose = key_diff + round(pitch_center_diff / 12) * 12

    # Escalar temporalmente
    src_dur = foreign.total_beats or 1.0
    time_scale = max_beats / src_dur

    notes_out = []
    for n in foreign.notes:
        new_start = n.start * time_scale
        if new_start >= max_beats:
            break
        new_pitch = _snap_to_scale(
            max(36, min(96, n.pitch + transpose)),
            scale_pcs
        )
        dur = n.duration * time_scale
        notes_out.append(Note(
            pitch=new_pitch,
            velocity=n.velocity,
            start=new_start,
            duration=min(max(0.1, dur), max_beats - new_start),
            channel=0,
        ))

    if verbose:
        print(f"  [Harvest] Adaptadas {len(notes_out)} notas "
              f"(transposición: {transpose:+d} semitonos)")
    return notes_out


def strategy_narrative(dna: SourceDNA, n_bars: int, mode: str,
                        position: float, seed: int, verbose: bool,
                        arc: str = 'hero') -> List[Note]:
    """
    [S6] Continuación narrativa: infiere la posición en el arco dramático
    y diseña la continuación con la función dramática apropiada.
    Usa narrator.py si disponible para diseñar curvas.
    """
    narrative_pos = _infer_narrative_position(position)

    if verbose:
        print(f"  [Narrative] Arco: {arc}, posición: {position:.2f} "
              f"→ sección: {narrative_pos}")

    # Diseñar curvas de tensión según la posición narrativa
    target_tension = _narrative_tension_curve(narrative_pos, n_bars, arc)

    # Generar con Markov pero guiado por las curvas narrativas
    notes = _markov_generate_own(dna,
                                  int(n_bars * dna.beats_per_bar * dna.density),
                                  n_bars, target_tension, seed, verbose)

    # Post-procesar: ajustar dinámicas según posición narrativa
    if notes:
        _apply_narrative_dynamics(notes, narrative_pos, arc)

    if verbose:
        print(f"  [Narrative] Función dramática: {narrative_pos} "
              f"en arco {arc}, tensión media: "
              f"{sum(target_tension)/len(target_tension):.2f}")

    return notes


def _infer_narrative_position(position: float) -> str:
    for (lo, hi), name in NARRATIVE_POSITIONS.items():
        if lo <= position < hi:
            return name
    return 'resolution'


def _narrative_tension_curve(position_name: str, n_bars: int, arc: str) -> List[float]:
    """Diseña la curva de tensión según la posición narrativa y el arco."""
    curves = {
        'intro':        lambda i, n: 0.2 + 0.3 * (i / n),
        'exposition':   lambda i, n: 0.4 + 0.2 * math.sin(math.pi * i / n),
        'development':  lambda i, n: 0.5 + 0.4 * (i / n),
        'climax':       lambda i, n: 0.7 + 0.25 * math.sin(math.pi * i / n),
        'resolution':   lambda i, n: 0.7 - 0.4 * (i / n),
        'coda':         lambda i, n: 0.3 - 0.15 * (i / n),
    }
    curve_fn = curves.get(position_name, lambda i, n: 0.5)
    return [max(0.0, min(1.0, curve_fn(i, max(n_bars-1, 1)))) for i in range(n_bars)]


def _apply_narrative_dynamics(notes: List[Note], position: str, arc: str):
    """Ajusta velocidades según la posición narrativa."""
    vel_centers = {
        'intro': 55, 'exposition': 65, 'development': 75,
        'climax': 90, 'resolution': 65, 'coda': 50
    }
    center = vel_centers.get(position, 65)
    for n in notes:
        n.velocity = int(max(35, min(110, center + random.gauss(0, 8))))


def strategy_evolve(dna: SourceDNA, n_bars: int, mode: str,
                    position: float, seed: int, verbose: bool) -> List[Note]:
    """
    [S7] Continuación por evolución genética.
    Usa evolver.py con el fuente como semilla, guiado hacia el estado
    emocional objetivo de la continuación.
    """
    evolver = _import_ecosystem('evolver')
    if not evolver or not M21_OK:
        if verbose:
            print("  [Evolve] evolver no disponible; fallback a Markov")
        return strategy_markov(dna, n_bars, mode, position, seed, verbose)

    try:
        # Crear MIDI temporal con el fuente para evolver
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = dna.path

            # Ejecutar evolver via subprocess (es más seguro que importar)
            out_dir = Path(tmpdir) / 'evolved'
            out_dir.mkdir()

            cmd = [
                sys.executable, str(_SCRIPT_DIR / 'evolver.py'),
                src_path,
                '--generations', '3',
                '--population', '6',
                '--elite', '2',
                '--mutation-rate', '0.4',
                '--output-dir', str(out_dir),
                '--seed', str(seed),
                '--bars', str(n_bars),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                evolved_midis = list(out_dir.glob('*.mid'))
                if evolved_midis:
                    # Tomar el mejor (primer resultado)
                    best = evolved_midis[0]
                    evolved_dna = extract_dna_from_midi(str(best), verbose=False)
                    if verbose:
                        print(f"  [Evolve] Evolucionado: {len(evolved_dna.notes)} notas")
                    return _adapt_foreign_notes(evolved_dna, dna, n_bars, seed, verbose)

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        if verbose:
            print(f"  [Evolve] evolver.py falló ({e}); fallback a Markov")

    return strategy_markov(dna, n_bars, mode, position, seed, verbose)


def strategy_blend(dna: SourceDNA, n_bars: int, mode: str,
                   position: float, seed: int, verbose: bool,
                   collection_dir: Optional[str] = None,
                   arc: str = 'hero',
                   n_candidates: int = 3) -> List[Note]:
    """
    [S8] Combina todas las estrategias disponibles y elige la mejor
    según score de coherencia musical y compatibilidad con el fuente.
    """
    strategies = [
        ('markov',    lambda s: strategy_markov(dna, n_bars, mode, position, s, verbose)),
        ('contour',   lambda s: strategy_contour(dna, n_bars, mode, position, s, verbose)),
        ('phrase',    lambda s: strategy_phrase(dna, n_bars, mode, position, s, verbose)),
        ('variation', lambda s: strategy_variation(dna, n_bars, mode, position, s, verbose)),
        ('narrative', lambda s: strategy_narrative(dna, n_bars, mode, position, s, verbose, arc)),
    ]

    if verbose:
        print(f"  [Blend] Probando {min(n_candidates, len(strategies))} estrategias...")

    candidates = []
    used_strategies = strategies[:n_candidates]

    for i, (name, fn) in enumerate(used_strategies):
        try:
            notes = fn(seed + i * 137)
            if notes:
                score = score_continuation(notes, dna, n_bars)
                candidates.append((name, notes, score))
                if verbose:
                    print(f"  [Blend] {name}: score={score:.3f}, "
                          f"{len(notes)} notas")
        except Exception as e:
            if verbose:
                print(f"  [Blend] {name} falló: {e}")

    if not candidates:
        return strategy_markov(dna, n_bars, mode, position, seed, verbose)

    # Elegir el mejor
    best_name, best_notes, best_score = max(candidates, key=lambda x: x[2])
    if verbose:
        print(f"  [Blend] ✓ Mejor estrategia: {best_name} (score={best_score:.3f})")

    return best_notes


# ═══════════════════════════════════════════════════════════════════════════════
#  DISEÑO DE CURVAS DE TENSIÓN PARA LA CONTINUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _design_continuation_tension(dna: SourceDNA, n_bars: int,
                                   mode: str, position: float) -> List[float]:
    """
    Diseña la curva de tensión para la continuación según:
    - El estado emocional al final del fragmento fuente
    - El modo de completado solicitado
    - La posición narrativa inferida
    """
    # Estado final del fragmento
    end_tension = dna.tension_curve[-1] if dna.tension_curve else dna.tension_mean

    if mode == 'extend':
        # Continúa la tendencia actual con una pequeña inflexión
        if dna.tension_direction == 'rising':
            return [min(1.0, end_tension + (i / n_bars) * 0.3) for i in range(n_bars)]
        elif dna.tension_direction == 'falling':
            return [max(0.1, end_tension - (i / n_bars) * 0.3) for i in range(n_bars)]
        else:
            # Arco suave desde el estado actual
            return [end_tension + 0.2 * math.sin(math.pi * i / n_bars)
                    for i in range(n_bars)]

    elif mode == 'develop':
        # Desarrollo: tensión ascendente, clímax en 2/3, bajada final
        return [
            end_tension + (0.4 * i / n_bars) if i < int(n_bars * 0.66)
            else end_tension + 0.4 - (0.3 * (i - int(n_bars * 0.66)) / (n_bars / 3))
            for i in range(n_bars)
        ]

    elif mode == 'recapitulate':
        # Reexposición: tensión similar al inicio del fuente
        start_tension = dna.tension_curve[0] if dna.tension_curve else 0.4
        return [end_tension + (start_tension - end_tension) * (i / n_bars)
                for i in range(n_bars)]

    elif mode == 'coda':
        # Coda: descenso gradual hacia el reposo
        return [max(0.05, end_tension * (1 - i / n_bars) * 0.8)
                for i in range(n_bars)]

    elif mode == 'fill':
        # Relleno: arco entre el estado final del A y el inicio del B
        return [end_tension + 0.15 * math.sin(math.pi * i / n_bars)
                for i in range(n_bars)]

    return [dna.tension_mean] * n_bars


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING DE CALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def score_continuation(notes: List[Note], dna: SourceDNA, n_bars: int) -> float:
    """
    Puntúa la calidad de una continuación.
    Combina múltiples métricas de coherencia musical.
    """
    if not notes:
        return 0.0

    score = 0.0
    weights = 0.0

    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    velocities = [n.velocity for n in notes]

    # 1. Compatibilidad tonal (% de notas en la escala)
    scale_pcs = set(dna.scale_pitches)
    in_scale = sum(1 for p in pitches if p % 12 in scale_pcs)
    tonal_score = in_scale / len(pitches)
    score += tonal_score * 0.25
    weights += 0.25

    # 2. Coherencia de rango (diferencia de pitch_mean con el fuente)
    cont_mean = sum(pitches) / len(pitches)
    range_diff = abs(cont_mean - dna.pitch_mean) / max(dna.pitch_range, 1)
    range_score = max(0.0, 1.0 - range_diff)
    score += range_score * 0.15
    weights += 0.15

    # 3. Suavidad del contorno (evitar saltos excesivos)
    if len(pitches) > 1:
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        large_leaps = sum(1 for iv in intervals if iv > 7)
        smoothness = 1.0 - (large_leaps / len(intervals))
        score += smoothness * 0.15
        weights += 0.15

    # 4. Variedad rítmica
    if durations:
        dur_set = set(round(d * 4) / 4 for d in durations)
        rhythm_variety = min(1.0, len(dur_set) / 4)
        score += rhythm_variety * 0.10
        weights += 0.10

    # 5. Coherencia de densidad con el fuente
    total_beats = n_bars * dna.beats_per_bar
    cont_density = len(notes) / total_beats
    density_ratio = min(cont_density, dna.density) / max(cont_density, dna.density, 0.01)
    score += density_ratio * 0.15
    weights += 0.15

    # 6. Conexión con el final del fuente (primera nota de la continuación)
    if pitches and dna.last_pitch:
        first_interval = abs(pitches[0] - dna.last_pitch)
        connection_score = 1.0 if first_interval <= 2 else (
            0.7 if first_interval <= 5 else 0.4 if first_interval <= 7 else 0.1
        )
        score += connection_score * 0.20
        weights += 0.20

    # Normalizar
    if weights > 0:
        score = score / weights
    else:
        score = 0.5

    # Usar score_candidate del ecosistema si disponible (más sofisticado)
    unified = _import_ecosystem('midi_dna_unified')
    if unified and M21_OK and dna.key_mode in ('major', 'minor'):
        try:
            key_str = f"{CHROMATIC_SCALE[dna.key_root % 12]} {dna.key_mode}"
            key_obj = m21.key.Key(CHROMATIC_SCALE[dna.key_root % 12], dna.key_mode)
            notes_fmt = [(n.pitch, n.velocity, n.start, n.duration, n.channel)
                         for n in notes]
            eco_score = unified.score_candidate(notes_fmt, [], key_obj)
            if eco_score and eco_score > 0:
                # Blend entre nuestra métrica y la del ecosistema
                score = 0.4 * score + 0.6 * min(1.0, eco_score / 10.0)
        except Exception:
            pass

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def notes_to_midi(notes: List[Note], tempo_bpm: float, ticks_per_beat: int = 480,
                  time_sig: Tuple[int, int] = (4, 4),
                  program: int = 0) -> 'mido.MidiFile':
    """Convierte lista de Note a MidiFile."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.MetaMessage('time_signature',
                                   numerator=time_sig[0],
                                   denominator=time_sig[1],
                                   time=0))
    track.append(mido.Message('program_change', program=program, channel=0, time=0))

    # Construir eventos
    events = []
    for n in notes:
        start_ticks = int(n.start * ticks_per_beat)
        end_ticks = int((n.start + n.duration) * ticks_per_beat)
        events.append((start_ticks, 'note_on', n.pitch, n.velocity, n.channel))
        events.append((end_ticks, 'note_off', n.pitch, 0, n.channel))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

    current_tick = 0
    for tick, msg_type, pitch, vel, ch in events:
        delta = tick - current_tick
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=pitch, velocity=vel,
                                       channel=ch, time=max(0, delta)))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=vel,
                                       channel=ch, time=max(0, delta)))
        current_tick = tick

    return mid


def save_midi(notes: List[Note], output_path: str, tempo_bpm: float,
              time_sig: Tuple[int, int] = (4, 4)):
    """Guarda notas como archivo MIDI."""
    mid = notes_to_midi(notes, tempo_bpm, time_sig=time_sig)
    mid.save(output_path)


def concatenate_midis(source_path: str, continuation_path: str,
                      output_path: str, verbose: bool = False):
    """
    Concatena dos archivos MIDI en uno.
    Usa stitcher.py si está disponible para una transición coherente;
    si no, concatenación directa.
    """
    stitcher = _import_ecosystem('stitcher')

    if stitcher:
        try:
            # Intentar usar stitcher para la concatenación coherente
            # Primero generar fingerprints
            unified = _import_ecosystem('midi_dna_unified')
            if unified:
                fp_src = unified.fingerprint_from_midi(source_path, verbose=False)
                fp_cont = unified.fingerprint_from_midi(continuation_path, verbose=False)

                fp_src_path = source_path.replace('.mid', '.fingerprint.json')
                fp_cont_path = continuation_path.replace('.mid', '.fingerprint.json')

                with open(fp_src_path, 'w') as f:
                    json.dump(fp_src, f)
                with open(fp_cont_path, 'w') as f:
                    json.dump(fp_cont, f)

                # Llamar a stitcher
                result = subprocess.run([
                    sys.executable, str(_SCRIPT_DIR / 'stitcher.py'),
                    fp_src_path, fp_cont_path,
                    '--fixed-order',
                    '--output', output_path,
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and os.path.exists(output_path):
                    if verbose:
                        print(f"  [Stitch] Concatenación coherente via stitcher.py ✓")
                    return

        except Exception as e:
            if verbose:
                print(f"  [Stitch] stitcher.py falló ({e}), concatenación directa")

    # Concatenación directa con mido
    _concatenate_direct(source_path, continuation_path, output_path, verbose)


def _concatenate_direct(src_path: str, cont_path: str, out_path: str, verbose: bool):
    """Concatenación MIDI directa."""
    if not MIDO_OK:
        if verbose:
            print("  [Concat] mido no disponible; copiando continuación")
        import shutil
        shutil.copy2(cont_path, out_path)
        return

    src_mid = mido.MidiFile(src_path)
    cont_mid = mido.MidiFile(cont_path)

    out_mid = mido.MidiFile(ticks_per_beat=src_mid.ticks_per_beat)

    # Calcular duración del source en ticks
    src_duration = 0
    for track in src_mid.tracks:
        total = 0
        for msg in track:
            total += msg.time
        src_duration = max(src_duration, total)

    # Para cada pista del source, añadir las notas del continuación
    n_tracks = max(len(src_mid.tracks), len(cont_mid.tracks))

    for i in range(max(len(src_mid.tracks), 1)):
        out_track = mido.MidiTrack()
        out_mid.tracks.append(out_track)

        # Pista source
        if i < len(src_mid.tracks):
            for msg in src_mid.tracks[i]:
                out_track.append(msg.copy())

        # Calcular duración acumulada de esta pista
        track_dur = 0
        if i < len(src_mid.tracks):
            for msg in src_mid.tracks[i]:
                track_dur += msg.time

        gap = src_duration - track_dur
        first = True

        # Añadir notas de la continuación (solo pista 0 si hay múltiples)
        cont_track_idx = min(i, len(cont_mid.tracks) - 1)
        if cont_track_idx >= 0 and cont_track_idx < len(cont_mid.tracks):
            for msg in cont_mid.tracks[cont_track_idx]:
                if first and gap > 0:
                    msg = msg.copy(time=msg.time + gap)
                    first = False
                out_track.append(msg)

    out_mid.save(out_path)
    if verbose:
        print(f"  [Concat] Concatenación directa: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERIR PARÁMETROS AUTOMÁTICOS
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_n_bars(dna: SourceDNA, mode: str) -> int:
    """Infiere el número de compases a generar según el fuente y el modo."""
    src_bars = dna.n_bars

    if mode == 'coda':
        # Coda: la mitad del fuente, mínimo 4
        return max(4, src_bars // 2)
    elif mode == 'develop':
        # Desarrollo: doble del fuente, máximo 32
        return min(32, src_bars * 2)
    elif mode == 'recapitulate':
        # Reexposición: igual al fuente
        return src_bars
    elif mode == 'fill':
        # Relleno: la mitad
        return max(4, src_bars // 2)
    else:
        # extend: igual al fuente
        return src_bars


def _infer_position(dna: SourceDNA) -> float:
    """
    Infiere la posición narrativa del fragmento a partir de su ADN.
    Un fragmento que termina con alta tensión probablemente está en desarrollo.
    Un fragmento con tensión media ascendente está en exposición.
    """
    if not dna.tension_curve:
        return 0.0

    end_tension = dna.tension_curve[-1]
    mean_tension = dna.tension_mean

    if end_tension > 0.75:
        return 0.6  # clímax o cerca del clímax
    elif end_tension > 0.55 and dna.tension_direction == 'rising':
        return 0.4  # desarrollo
    elif end_tension < 0.35:
        if mean_tension < 0.35:
            return 0.0  # intro
        else:
            return 0.85  # resolución / coda
    else:
        return 0.2  # exposición


# ═══════════════════════════════════════════════════════════════════════════════
#  INFORME DE ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(dna: SourceDNA, result: CompletionResult,
                  output_path: str, args):
    """Imprime un informe detallado del proceso de completado."""
    B = '\033[1m'  # bold
    G = '\033[32m'  # green
    Y = '\033[33m'  # yellow
    C = '\033[36m'  # cyan
    R = '\033[0m'   # reset

    print(f"\n{B}╔══════════════════════════════════════════════════════════════╗{R}")
    print(f"{B}║                    COMPLETER  — INFORME                      ║{R}")
    print(f"{B}╚══════════════════════════════════════════════════════════════╝{R}\n")

    # Fuente
    print(f"{C}── FRAGMENTO FUENTE{R}")
    print(f"   Archivo:       {dna.path}")
    print(f"   Tonalidad:     {CHROMATIC_SCALE[dna.key_root % 12]} {dna.key_mode}")
    print(f"   Tempo:         {dna.tempo_bpm:.0f} BPM")
    print(f"   Compases:      {dna.n_bars} × {dna.beats_per_bar}/4")
    print(f"   Notas:         {len(dna.pitches)}")
    print(f"   Rango:         {dna.pitch_min} → {dna.pitch_max} "
          f"(MIDI, rango={dna.pitch_range} st)")
    print(f"   Densidad:      {dna.density:.2f} notas/beat")
    print(f"   Vel. media:    {dna.velocity_mean:.0f}")
    print(f"   Tensión:       {dna.tension_mean:.2f} ({dna.tension_direction})")
    if dna.tension_curve:
        tc = dna.tension_curve
        bar = ''.join('█' if t > 0.66 else ('▒' if t > 0.33 else '░')
                       for t in tc[:20])
        print(f"   Curva tensión: [{bar}]")
    print()

    # Decisiones
    print(f"{C}── DECISIONES DE COMPLETADO{R}")
    print(f"   Estrategia:    {Y}{result.strategy}{R}")
    print(f"   Modo:          {result.mode}")
    print(f"   Posición:      {result.metadata.get('position', 'auto'):.2f} "
          f"→ {_infer_narrative_position(result.metadata.get('position', 0.5))}")
    print(f"   Compases gen.: {result.metadata.get('n_bars', '?')}")
    print()

    # Resultado
    print(f"{C}── CONTINUACIÓN GENERADA{R}")
    print(f"   Notas:         {len(result.notes)}")
    if result.notes:
        pitches = [n.pitch for n in result.notes]
        print(f"   Rango:         {min(pitches)} → {max(pitches)}")
        vels = [n.velocity for n in result.notes]
        print(f"   Vel. media:    {sum(vels)/len(vels):.0f}")
    print(f"   Score:         {G}{result.score:.3f}/1.000{R}")
    print(f"   Archivo:       {output_path}")
    print()

    # Integración con el ecosistema
    if result.report_lines:
        print(f"{C}── LOG DEL PROCESO{R}")
        for line in result.report_lines:
            print(f"   {line}")
        print()

    print(f"{B}── PRÓXIMOS PASOS SUGERIDOS{R}")
    _suggest_next_steps(dna, result, output_path)
    print()


def _suggest_next_steps(dna: SourceDNA, result: CompletionResult, output_path: str):
    """Sugiere comandos del ecosistema como siguiente paso."""
    base = Path(output_path).stem
    src_name = Path(dna.path).name

    suggestions = []

    if result.mode == 'extend':
        suggestions.append(
            f"  → Reharmonizar:  python reharmonizer.py {output_path} --strategy all"
        )
        suggestions.append(
            f"  → Orquestar:     python orchestrator.py {output_path} --auto-fingerprints"
        )

    elif result.mode == 'develop':
        suggestions.append(
            f"  → Añadir contrapunto: python counterpoint.py {output_path} --species 5"
        )
        suggestions.append(
            f"  → Evolucionar:   python evolver.py {dna.path} {output_path}"
        )

    elif result.mode == 'recapitulate':
        suggestions.append(
            f"  → Unir todo:  python stitcher.py {dna.path} {output_path}"
        )

    elif result.mode == 'coda':
        suggestions.append(
            f"  → Añadir percusión:  python drummer.py add {output_path}"
        )

    suggestions.append(
        f"  → Analizar:      python analyzer.py compare {src_name} {output_path}"
    )
    suggestions.append(
        f"  → Transferir estilo: python style_transfer.py {output_path} <estilo.mid>"
    )

    for s in suggestions:
        print(s)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORQUESTADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def complete(
    source_path: str,
    strategy: str = 'blend',
    mode: str = 'extend',
    n_bars: Optional[int] = None,
    candidates: int = 3,
    fill_to: Optional[str] = None,
    collection_dir: Optional[str] = None,
    arc: str = 'hero',
    position: Optional[float] = None,
    end_with: Optional[str] = None,
    key_override: Optional[str] = None,
    tempo_override: Optional[float] = None,
    append: bool = True,
    no_stitch: bool = False,
    export_fingerprint: bool = False,
    output_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    report: bool = False,
) -> CompletionResult:
    """
    Función principal de completado. Orquesta todo el pipeline.
    """
    random.seed(seed)

    # ── Validar entrada ───────────────────────────────────────────────────────
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"No se encontró el archivo: {source_path}")
    if not MIDO_OK:
        raise ImportError("mido es necesario: pip install mido")

    print(f"\n[Completer] Analizando {Path(source_path).name}...")
    t0 = time.time()

    # ── Extraer ADN del fuente ────────────────────────────────────────────────
    dna = extract_dna_from_midi(source_path, verbose=verbose)

    # ── Override de parámetros opcionales ────────────────────────────────────
    if tempo_override:
        dna.tempo_bpm = tempo_override
    if key_override:
        # Parsear "Dm", "G major", etc.
        parts = key_override.strip().split()
        note_name = parts[0].replace('b', '#')  # simplificación
        if note_name in CHROMATIC_SCALE:
            dna.key_root = 60 + CHROMATIC_SCALE.index(note_name)
        mode_name = parts[1].lower() if len(parts) > 1 else dna.key_mode
        dna.key_mode = mode_name if mode_name in MODE_INTERVALS else dna.key_mode
        dna.scale_pitches = _get_scale_pitches(dna.key_root, dna.key_mode)

    # ── Inferir n_bars si no se especifica ───────────────────────────────────
    if n_bars is None:
        n_bars = _infer_n_bars(dna, mode)

    # ── Inferir posición narrativa ────────────────────────────────────────────
    if position is None:
        position = _infer_position(dna)

    if verbose:
        print(f"[Completer] ADN extraído en {time.time()-t0:.1f}s")
        print(f"[Completer] Configuración: estrategia={strategy}, modo={mode}, "
              f"bars={n_bars}, posición={position:.2f}")

    # ── Seleccionar y ejecutar estrategia ─────────────────────────────────────
    strategy_fns = {
        'markov':    lambda: strategy_markov(dna, n_bars, mode, position, seed, verbose),
        'contour':   lambda: strategy_contour(dna, n_bars, mode, position, seed, verbose),
        'phrase':    lambda: strategy_phrase(dna, n_bars, mode, position, seed, verbose),
        'variation': lambda: strategy_variation(dna, n_bars, mode, position, seed, verbose),
        'harvest':   lambda: strategy_harvest(dna, n_bars, mode, position, seed, verbose, collection_dir),
        'narrative': lambda: strategy_narrative(dna, n_bars, mode, position, seed, verbose, arc),
        'evolve':    lambda: strategy_evolve(dna, n_bars, mode, position, seed, verbose),
        'blend':     lambda: strategy_blend(dna, n_bars, mode, position, seed, verbose,
                                             collection_dir, arc, candidates),
    }

    fn = strategy_fns.get(strategy, strategy_fns['blend'])
    print(f"[Completer] Generando continuación ({strategy})...")
    notes = fn()

    if not notes:
        print("[Completer] ⚠ Estrategia no produjo notas; usando Markov de fallback")
        notes = strategy_markov(dna, n_bars, mode, position, seed + 999, verbose)

    # ── Añadir coda/cadencia final si se solicita ─────────────────────────────
    if end_with:
        notes = _append_ending(notes, dna, end_with, n_bars, seed)

    # ── Calcular score ────────────────────────────────────────────────────────
    score = score_continuation(notes, dna, n_bars)

    # ── Determinar ruta de salida ─────────────────────────────────────────────
    if output_path is None:
        base = source_path.replace('.mid', '').replace('.midi', '')
        suffix = f'_{mode}' if mode != 'extend' else '_completed'
        output_path = f"{base}{suffix}.mid"

    # ── Guardar la continuación ───────────────────────────────────────────────
    cont_path = output_path
    if append and not no_stitch:
        # Guardar continuación en temporal y luego concatenar
        cont_path = output_path.replace('.mid', '_cont_only.mid')

    save_midi(notes, cont_path, dna.tempo_bpm,
              time_sig=(dna.beats_per_bar, 4))
    print(f"[Completer] Continuación guardada: {cont_path}")

    # ── Concatenar con el fuente si --append ──────────────────────────────────
    if append and not no_stitch:
        print(f"[Completer] Uniendo fuente + continuación...")
        concatenate_midis(source_path, cont_path, output_path, verbose)
        # Limpiar temporal
        if os.path.exists(cont_path) and cont_path != output_path:
            os.remove(cont_path)
        print(f"[Completer] ✓ Obra completa guardada: {output_path}")
    elif no_stitch:
        output_path = cont_path
        print(f"[Completer] ✓ Solo continuación: {output_path}")

    # ── Exportar fingerprint si se solicita ───────────────────────────────────
    if export_fingerprint:
        unified = _import_ecosystem('midi_dna_unified')
        if unified:
            try:
                fp = unified.fingerprint_from_midi(output_path, verbose=False)
                fp_path = unified.export_fingerprint(fp, output_path)
                print(f"[Completer] Fingerprint exportado: {fp_path}")
            except Exception as e:
                if verbose:
                    print(f"  [FP] No se pudo exportar fingerprint: {e}")

    # ── Construir resultado ───────────────────────────────────────────────────
    result = CompletionResult(
        strategy=strategy,
        mode=mode,
        notes=notes,
        score=score,
        midi_path=output_path,
        metadata={
            'n_bars': n_bars,
            'position': position,
            'arc': arc,
            'tempo_bpm': dna.tempo_bpm,
            'key': f"{CHROMATIC_SCALE[dna.key_root % 12]} {dna.key_mode}",
            'elapsed_s': time.time() - t0,
        }
    )

    # ── Informe ───────────────────────────────────────────────────────────────
    if report:
        print_report(dna, result, output_path, None)
    else:
        print(f"[Completer] Score de coherencia: {score:.3f}/1.000")
        print(f"[Completer] Tiempo total: {time.time()-t0:.1f}s")

    return result


def _append_ending(notes: List[Note], dna: SourceDNA,
                    end_type: str, n_bars: int, seed: int) -> List[Note]:
    """Añade un final (coda/cadencia/fade) a las notas generadas."""
    random.seed(seed)
    if not notes:
        return notes

    last_beat = max(n.start + n.duration for n in notes)
    scale_pcs = dna.scale_pitches
    tonic_pc = dna.key_root % 12

    if end_type == 'cadence':
        # Añadir V→I al final
        last_pitch = notes[-1].pitch if notes else dna.last_pitch
        # Dominante (5º grado)
        dominant_pc = (tonic_pc + 7) % 12
        dominant_pitch = last_pitch
        while dominant_pitch % 12 != dominant_pc:
            dominant_pitch += 1
        # Tónica
        tonic_pitch = dominant_pitch
        while tonic_pitch % 12 != tonic_pc:
            tonic_pitch += 1

        notes.append(Note(dominant_pitch, int(dna.velocity_mean),
                           last_beat, 1.0, 0))
        notes.append(Note(tonic_pitch, int(dna.velocity_mean * 0.85),
                           last_beat + 1.0, 2.0, 0))

    elif end_type == 'fade':
        # Reducir velocidades progresivamente en las últimas notas
        fade_notes = sorted(notes, key=lambda n: n.start)
        n_fade = max(3, len(fade_notes) // 4)
        for i, n in enumerate(fade_notes[-n_fade:]):
            progress = i / n_fade
            n.velocity = int(n.velocity * (1.0 - progress * 0.7))

    elif end_type == 'coda':
        # Generar mini-coda descendente
        last_pitch = notes[-1].pitch if notes else dna.last_pitch
        tonic_pitch = last_pitch
        while tonic_pitch % 12 != tonic_pc:
            tonic_pitch -= 1
        if abs(tonic_pitch - last_pitch) > 12:
            tonic_pitch = last_pitch - (last_pitch % 12 - tonic_pc)

        beat = last_beat
        for i in range(4):
            p = _snap_to_scale(last_pitch - i * 2, scale_pcs)
            vel = int(dna.velocity_mean * (0.8 - i * 0.15))
            notes.append(Note(max(36, p), max(30, vel), beat, 1.0, 0))
            beat += 1.0
        notes.append(Note(tonic_pitch, int(dna.velocity_mean * 0.6),
                           beat, 3.0, 0))

    return notes


# ═══════════════════════════════════════════════════════════════════════════════
#  MODO FILL (A → ? → B)
# ═══════════════════════════════════════════════════════════════════════════════

def complete_fill(source_a: str, source_b: str,
                   n_bars: Optional[int] = None,
                   strategy: str = 'blend',
                   seed: int = 42,
                   verbose: bool = False,
                   output_path: Optional[str] = None) -> CompletionResult:
    """
    Genera un puente entre dos fragmentos A y B.
    La continuación debe conectar el final de A con el inicio de B.
    """
    print(f"\n[Completer] Modo FILL: {Path(source_a).name} → ? → {Path(source_b).name}")

    dna_a = extract_dna_from_midi(source_a, verbose=verbose)
    dna_b = extract_dna_from_midi(source_b, verbose=verbose)

    if n_bars is None:
        n_bars = max(4, (dna_a.n_bars + dna_b.n_bars) // 4)

    if verbose:
        print(f"  [Fill] A termina en tensión {dna_a.tension_curve[-1]:.2f}, "
              f"B empieza en {dna_b.tension_curve[0]:.2f}")
        print(f"  [Fill] A: {CHROMATIC_SCALE[dna_a.key_root%12]} {dna_a.key_mode}, "
              f"B: {CHROMATIC_SCALE[dna_b.key_root%12]} {dna_b.key_mode}")

    # Diseñar curva de tensión puente
    end_a = dna_a.tension_curve[-1] if dna_a.tension_curve else 0.5
    start_b = dna_b.tension_curve[0] if dna_b.tension_curve else 0.5
    bridge_tension = [
        end_a + (start_b - end_a) * (i / n_bars) for i in range(n_bars)
    ]

    # Modificar dna_a para que la continuación apunte hacia B
    dna_a.tension_curve = bridge_tension

    # Usar blend pero con orientación hacia B
    notes = strategy_blend(
        dna_a, n_bars, 'fill', 0.5, seed, verbose,
        n_candidates=3
    )

    # Ajustar el final hacia la tonalidad de B
    if notes:
        scale_pcs_b = _get_scale_pitches(dna_b.key_root, dna_b.key_mode)
        for i, n in enumerate(notes[-max(3, len(notes)//5):]):
            progress = i / max(1, len(notes) // 5)
            n.pitch = int(n.pitch * (1-progress) +
                          _snap_to_scale(n.pitch, scale_pcs_b) * progress)

    score = score_continuation(notes, dna_a, n_bars)

    if output_path is None:
        base_a = source_a.replace('.mid', '').replace('.midi', '')
        output_path = f"{base_a}_fill.mid"

    save_midi(notes, output_path, dna_a.tempo_bpm,
              time_sig=(dna_a.beats_per_bar, 4))

    # Si hay stitcher, ensamblar A + fill + B
    full_path = output_path.replace('_fill.mid', '_full.mid')
    if os.path.exists(source_a):
        try:
            ab_path = output_path.replace('_fill.mid', '_AB.mid')
            concatenate_midis(source_a, output_path, ab_path, verbose)
            concatenate_midis(ab_path, source_b, full_path, verbose)
            if os.path.exists(ab_path):
                os.remove(ab_path)
            print(f"[Completer] ✓ Obra completa (A+fill+B): {full_path}")
        except Exception as e:
            if verbose:
                print(f"  [Fill] No se pudo ensamblar completo: {e}")

    print(f"[Completer] ✓ Puente guardado: {output_path} (score={score:.3f})")

    return CompletionResult(
        strategy=strategy, mode='fill',
        notes=notes, score=score, midi_path=output_path,
        metadata={'n_bars': n_bars, 'fill_to': source_b}
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Completer v1.0 — Continuación inteligente de obras MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('source', help='MIDI de entrada (fragmento a completar)')

    p.add_argument('--bars', '-b', type=int, default=None,
                   help='Compases a generar (default: auto según fuente y modo)')
    p.add_argument('--strategy', '-s',
                   choices=['markov','contour','phrase','variation',
                            'harvest','narrative','evolve','blend'],
                   default='blend',
                   help='Estrategia de completado (default: blend)')
    p.add_argument('--mode', '-m',
                   choices=['extend','fill','develop','recapitulate','coda'],
                   default='extend',
                   help='Modo de completado (default: extend)')
    p.add_argument('--candidates', '-c', type=int, default=3,
                   help='Candidatos a evaluar en modo blend (default: 3)')

    p.add_argument('--fill-to', metavar='FILE',
                   help='MIDI destino para modo fill (A → ? → B)')
    p.add_argument('--collection', metavar='DIR',
                   help='Directorio de MIDIs para estrategia harvest')
    p.add_argument('--arc',
                   choices=['hero','tragedy','romance','mystery',
                            'meditation','rondo','sonata','custom'],
                   default='hero',
                   help='Arco narrativo (default: hero)')
    p.add_argument('--position', type=float, default=None,
                   help='Posición en la obra 0.0-1.0 (default: auto)')
    p.add_argument('--end-with',
                   choices=['coda','cadence','fade'],
                   default=None,
                   help='Tipo de final a añadir')

    p.add_argument('--key', default=None,
                   help='Tonalidad forzada, ej: "Dm" "G major" (default: auto)')
    p.add_argument('--tempo', type=float, default=None,
                   help='Tempo forzado en BPM (default: heredado del fuente)')
    p.add_argument('--append', action='store_true', default=True,
                   help='Concatenar continuación al fuente (default: True)')
    p.add_argument('--no-append', action='store_true',
                   help='No concatenar: exportar solo la continuación')
    p.add_argument('--no-stitch', action='store_true',
                   help='No usar stitcher para la concatenación')
    p.add_argument('--export-fingerprint', action='store_true',
                   help='Exportar fingerprint de la continuación')
    p.add_argument('--output', '-o', default=None,
                   help='Archivo de salida (default: <fuente>_completed.mid)')
    p.add_argument('--seed', type=int, default=42,
                   help='Semilla aleatoria (default: 42)')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Informe detallado del proceso')
    p.add_argument('--report', '-r', action='store_true',
                   help='Informe completo de decisiones y score')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not MIDO_OK:
        print("ERROR: mido no está instalado.\n  pip install mido")
        sys.exit(1)

    # Modo fill especial
    if args.fill_to:
        result = complete_fill(
            source_a=args.source,
            source_b=args.fill_to,
            n_bars=args.bars,
            strategy=args.strategy,
            seed=args.seed,
            verbose=args.verbose,
            output_path=args.output,
        )
        sys.exit(0)

    # Modo estándar
    append = args.append and not args.no_append
    result = complete(
        source_path=args.source,
        strategy=args.strategy,
        mode=args.mode,
        n_bars=args.bars,
        candidates=args.candidates,
        fill_to=args.fill_to,
        collection_dir=args.collection,
        arc=args.arc,
        position=args.position,
        end_with=args.end_with,
        key_override=args.key,
        tempo_override=args.tempo,
        append=append,
        no_stitch=args.no_stitch,
        export_fingerprint=args.export_fingerprint,
        output_path=args.output,
        seed=args.seed,
        verbose=args.verbose,
        report=args.report,
    )

    sys.exit(0 if result.notes else 1)


if __name__ == '__main__':
    main()
