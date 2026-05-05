#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       QUALITY SCORER  v1.2                                  ║
║         Evaluación multidimensional de calidad musical para MIDIs            ║
║                                                                              ║
║  Evalúa una o más obras MIDI en cinco dimensiones:                          ║
║    · formal    — corrección de voice leading e intervalos                   ║
║    · coherence — homogeneidad interna entre secciones                       ║
║    · arc       — estructura del arco emocional/tensión                      ║
║    · melodic   — interés melódico y variedad                                ║
║    · corpus    — similitud estilística a una colección de referencia        ║
║                                                                              ║
║  El score final es relativo (percentil en el corpus) con fallback           ║
║  absoluto (0.0–1.0) cuando no hay corpus disponible. Los pesos efectivos    ║
║  se modulan automáticamente por la confianza de cada dimensión.             ║
║                                                                              ║
║  MODOS DE OPERACIÓN:                                                         ║
║    (defecto)  Evaluación completa de un MIDI                                ║
║    rank       Comparación y ranking de varios MIDIs                         ║
║                                                                              ║
║  USO BÁSICO:                                                                 ║
║    python quality_scorer.py obra.mid                                         ║
║    python quality_scorer.py obra.mid --plan obra.theorist.json              ║
║    python quality_scorer.py obra.mid --corpus ./midis/                      ║
║    python quality_scorer.py obra.mid --corpus ./midis/ --save-cache c.qcache║
║    python quality_scorer.py obra.mid --corpus ./midis/ --use-cache c.qcache ║
║    python quality_scorer.py obra.mid --fast                                  ║
║    python quality_scorer.py obra.mid --quiet          # solo imprime score  ║
║    python quality_scorer.py cand*.mid --mode rank                           ║
║    python quality_scorer.py cand*.mid --mode rank --quiet                   ║
║                                                                              ║
║  CONTROL DE DIMENSIONES:                                                     ║
║    python quality_scorer.py obra.mid --skip corpus formal                   ║
║    python quality_scorer.py obra.mid --only arc coherence                   ║
║    python quality_scorer.py obra.mid --weights formal=0.4 coherence=0.3    ║
║    python quality_scorer.py obra.mid --arc-weight-mode static               ║
║    python quality_scorer.py obra.mid --arc-weight-mode off                  ║
║                                                                              ║
║  EXPLICACIÓN EN LENGUAJE NATURAL (--explain):                               ║
║    python quality_scorer.py obra.mid --explain                               ║
║    python quality_scorer.py obra.mid --explain --explain-provider local     ║
║    python quality_scorer.py obra.mid --explain --explain-provider anthropic ║
║    python quality_scorer.py obra.mid --explain --explain-provider openai    ║
║    python quality_scorer.py obra.mid --explain --explain-provider clipboard ║
║    python quality_scorer.py obra.mid --explain --explain-out guia.txt       ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    python quality_scorer.py obra.mid --out-json report.json                 ║
║    python quality_scorer.py obra.mid --out-json report.json --verbose       ║
║                                                                              ║
║  CACHÉ DE CORPUS:                                                            ║
║    --save-cache FILE   Guardar vectores del corpus en .qcache               ║
║    --use-cache FILE    Cargar caché previo (solo recalcula ficheros nuevos) ║
║                                                                              ║
║  OPCIONES PRINCIPALES:                                                       ║
║    --plan FILE             theorist.json o narrator_plan.json (dim. arc)    ║
║    --corpus DIR            Directorio de MIDIs de referencia                ║
║    --mode eval|rank        Modo de operación (default: eval)                ║
║    --fast                  Modo rápido: omite cálculos pesados              ║
║    --arc-weight-mode MODE  dynamic (default) | static | off                 ║
║    --skip DIM...           Dimensiones a omitir                             ║
║    --only DIM...           Solo estas dimensiones                           ║
║    --weights K=V...        Pesos custom por dimensión (se renormalizan)     ║
║    --out-json FILE         Guardar informe completo en JSON                 ║
║    --quiet                 Solo imprime score numérico (útil en scripts)    ║
║    --verbose               Detalle completo en terminal                     ║
║    --no-color              Sin colores ANSI                                 ║
║                                                                              ║
║  OPCIONES DE EXPLICACIÓN:                                                    ║
║    --explain                    Añadir guía de mejora en lenguaje natural   ║
║    --explain-provider PROV      local|anthropic|openai|clipboard            ║
║    --explain-model MODEL        Modelo LLM (default: claude-opus-4-5/gpt-4o)║
║    --api-key KEY                API key (o ANTHROPIC_API_KEY/OPENAI_API_KEY)║
║    --explain-out FILE           Guardar la explicación en fichero de texto  ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:    mido, numpy                                                  ║
║    Opcional:   music21           (mejora formal y arc)                      ║
║    Opcional:   scipy             (mejora melodic)                           ║
║    --explain anthropic:  pip install anthropic                              ║
║    --explain openai:     pip install openai                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── mido ──────────────────────────────────────────────────────────────────────
try:
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage
    MIDO_OK = True
except ImportError:
    print("[ERROR] mido no disponible. Instálalo con: pip install mido")
    sys.exit(1)

# ── music21 ───────────────────────────────────────────────────────────────────
try:
    import music21 as m21
    from music21 import converter as m21converter
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

# ── scipy ─────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.2"

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Pesos por defecto para el score compuesto
DEFAULT_WEIGHTS: Dict[str, float] = {
    "formal":    0.25,
    "coherence": 0.20,
    "arc":       0.25,
    "melodic":   0.15,
    "corpus":    0.15,
}

# Umbrales absolutos heurísticos (calibrar con uso real)
ABSOLUTE_THRESHOLDS = {
    "bajo":    0.40,
    "medio":   0.60,
    "alto":    0.75,
    "muy_alto": 1.01,
}

# Intervalos disonantes (en semitonos, dentro de una octava)
DISSONANT_INTERVALS = {1, 2, 6, 10, 11}   # m2, M2, tritono, m7, M7
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9}  # unísono, 3as, 4as, 5as, 6as

_DRUM_CHANNEL   = 9
_BASS_PROGRAMS  = set(range(32, 40))
_DRUM_PROGRAMS  = set(range(112, 120))

# ANSI colors
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "gray":   "\033[90m",
    "blue":   "\033[94m",
}
_NO_COLOR = False  # se sobreescribe con --no-color


def _c(key: str) -> str:
    return "" if _NO_COLOR else _C.get(key, "")


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionResult:
    score:      float             # 0.0 (malo) – 1.0 (bueno)
    confidence: float             # 0.0–1.0: fiabilidad del score
    reasons:    List[str]         # explicaciones textuales
    details:    Dict              # datos crudos para JSON

    def to_dict(self) -> dict:
        return {
            "score":      round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reasons":    self.reasons,
            **self.details,
        }


@dataclass
class ScoreSummary:
    absolute:       float
    absolute_label: str
    percentile:     Optional[float]
    corpus_size:    Optional[int]
    weights_used:   Dict[str, float]


@dataclass
class QualityReport:
    file:        str
    mode:        str
    plan_used:   bool
    corpus_used: bool
    score:       ScoreSummary
    dimensions:  Dict[str, DimensionResult]
    meta:        Dict

    def to_dict(self) -> dict:
        return {
            "file":        self.file,
            "mode":        self.mode,
            "plan_used":   self.plan_used,
            "corpus_used": self.corpus_used,
            "score": {
                "absolute":       round(self.score.absolute, 4),
                "absolute_label": self.score.absolute_label,
                "percentile":     round(self.score.percentile, 2) if self.score.percentile is not None else None,
                "corpus_size":    self.score.corpus_size,
                "weights_used":   {k: round(v, 4) for k, v in self.score.weights_used.items()},
            },
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "meta":        self.meta,
        }


@dataclass
class Topology:
    """Descripción de la estructura de tracks del MIDI."""
    type:          str              # "single" | "multi_unlabeled" | "multi_labeled"
    voice_map:     Dict[int, str]   # track_idx → role
    melody_tracks: List[int]
    bass_tracks:   List[int]
    drum_tracks:   List[int]
    all_tracks:    List[int]


@dataclass
class EvalContext:
    """Contexto pasado a todos los evaluadores."""
    midi_path:       str
    plan:            Optional[Dict]
    corpus_vectors:  Optional[List[np.ndarray]]
    corpus_paths:    Optional[List[str]]         # rutas paralelas a corpus_vectors
    fast:            bool
    verbose:         bool
    arc_weight_mode: str = "dynamic"             # dynamic | static | off


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMITIVAS MIDI (sin dependencias externas)
# ══════════════════════════════════════════════════════════════════════════════

def _midi_to_absolute(mid: MidiFile) -> list:
    """Convierte todos los tracks a lista (abs_tick, track_idx, msg)."""
    events = []
    for ti, track in enumerate(mid.tracks):
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            events.append((abs_t, ti, msg))
    events.sort(key=lambda x: x[0])
    return events


def _get_tempo(mid: MidiFile) -> int:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500_000


def _get_time_signature(mid: MidiFile) -> Tuple[int, int]:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def _estimate_bar_ticks(mid: MidiFile) -> int:
    num, den = _get_time_signature(mid)
    beats_per_bar = num * (4 / den)
    return max(1, int(beats_per_bar * mid.ticks_per_beat))


def _get_total_ticks(mid: MidiFile) -> int:
    events = _midi_to_absolute(mid)
    return max((e[0] for e in events), default=0)


def _extract_notes(mid: MidiFile,
                   track_filter: Optional[set] = None) -> list:
    """
    Extrae notas como (start_tick, end_tick, pitch, velocity, channel, track_idx).
    """
    notes = []
    active = {}
    for abs_t, ti, msg in _midi_to_absolute(mid):
        if track_filter is not None and ti not in track_filter:
            continue
        if not hasattr(msg, 'channel'):
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            active[(ti, msg.channel, msg.note)] = (abs_t, msg.velocity)
        elif msg.type in ('note_off', 'note_on') and \
                (msg.type == 'note_off' or msg.velocity == 0):
            key = (ti, msg.channel, msg.note)
            if key in active:
                start_t, vel = active.pop(key)
                notes.append((start_t, abs_t, msg.note, vel, msg.channel, ti))
    total = _get_total_ticks(mid)
    for (ti, ch, pitch), (start_t, vel) in active.items():
        notes.append((start_t, total, pitch, vel, ch, ti))
    notes.sort(key=lambda x: x[0])
    return notes


def _detect_key_simple(notes: list) -> Tuple[int, str]:
    """Detección de tonalidad por correlación con perfiles de Krumhansl."""
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    if not notes:
        return 0, 'major'
    counts = np.zeros(12)
    for n in notes:
        counts[n[2] % 12] += 1
    counts /= (counts.sum() + 1e-9)
    best_score, best_root, best_mode = -1.0, 0, 'major'
    for root in range(12):
        for mode, prof in [('major', major_profile), ('minor', minor_profile)]:
            rotated = [prof[(i - root) % 12] for i in range(12)]
            score = float(np.corrcoef(counts, rotated)[0, 1])
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    return best_root, best_mode


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE TOPOLOGÍA
# ══════════════════════════════════════════════════════════════════════════════

def detect_topology(mid: MidiFile) -> Topology:
    """Clasifica la estructura de tracks y asigna roles."""
    n = len(mid.tracks)
    track_info = {}

    for ti, track in enumerate(mid.tracks):
        program, channels, pitches = 0, set(), []
        for msg in track:
            if msg.type == 'program_change':
                program = msg.program
                channels.add(msg.channel)
            elif msg.type == 'note_on' and msg.velocity > 0:
                channels.add(msg.channel)
                pitches.append(msg.note)
        track_info[ti] = {
            'program':    program,
            'channels':   channels,
            'is_drum':    _DRUM_CHANNEL in channels,
            'n_notes':    len(pitches),
            'mean_pitch': float(np.mean(pitches)) if pitches else 0.0,
        }

    active = [ti for ti, info in track_info.items() if info['n_notes'] > 0]

    if len(active) <= 1:
        ttype = "single"
    else:
        ttype = "multi_unlabeled"

    voice_map = {}
    melody_tracks, bass_tracks, drum_tracks = [], [], []

    for ti in active:
        info = track_info[ti]
        if info['is_drum'] or info['program'] in _DRUM_PROGRAMS:
            voice_map[ti] = 'drums'
            drum_tracks.append(ti)
        elif info['program'] in _BASS_PROGRAMS or info['mean_pitch'] < 48:
            voice_map[ti] = 'bass'
            bass_tracks.append(ti)
        else:
            voice_map[ti] = 'unknown'

    # El track con pitch más alto entre los no asignados → melodía
    unknowns = [ti for ti in active if voice_map.get(ti) == 'unknown']
    if unknowns:
        melody_ti = max(unknowns, key=lambda ti: track_info[ti]['mean_pitch'])
        voice_map[melody_ti] = 'melody'
        melody_tracks.append(melody_ti)
        # El resto → harmony
        for ti in unknowns:
            if ti != melody_ti:
                voice_map[ti] = 'harmony'

    if melody_tracks or bass_tracks:
        ttype = "multi_labeled"

    return Topology(
        type          = ttype,
        voice_map     = voice_map,
        melody_tracks = melody_tracks,
        bass_tracks   = bass_tracks,
        drum_tracks   = drum_tracks,
        all_tracks    = active,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DESCRIPTOR DE SEGMENTO (base compartida por coherence y corpus)
# ══════════════════════════════════════════════════════════════════════════════

def _descriptor_8(notes: list, bar_ticks: int) -> np.ndarray:
    """
    Vector de 8 características sobre un conjunto de notas.
    Puro numpy, sin dependencias externas.
    """
    if not notes:
        return np.zeros(8)
    pitches    = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    durations  = [n[1] - n[0] for n in notes]
    intervals  = list(np.diff(pitches)) if len(pitches) > 1 else [0]

    density   = len(notes) / max(bar_ticks, 1)
    mean_p    = np.mean(pitches)
    std_p     = np.std(pitches)
    mean_vel  = np.mean(velocities)
    var_vel   = np.var(velocities)
    mean_dur  = np.mean(durations)
    prop_asc  = np.sum(np.array(intervals) > 0) / max(len(intervals), 1)
    prop_jump = np.sum(np.abs(np.array(intervals)) > 5) / max(len(intervals), 1)

    return np.array([
        min(density, 10.0) / 10.0,
        mean_p / 127.0,
        std_p / 64.0,
        mean_vel / 127.0,
        var_vel / (127.0 ** 2),
        min(mean_dur, bar_ticks) / max(bar_ticks, 1),
        float(prop_asc),
        float(prop_jump),
    ])


def _descriptor_13(notes: list, mid: MidiFile) -> np.ndarray:
    """
    Vector de 13 dimensiones para comparación de corpus.
    Extensión del descriptor de 8 con métricas adicionales.
    """
    if not notes:
        return np.zeros(13)

    bar_ticks  = _estimate_bar_ticks(mid)
    base       = _descriptor_8(notes, bar_ticks)

    pitches    = [n[2] for n in notes]
    velocities = [n[3] for n in notes]
    starts     = [n[0] for n in notes]
    tpb        = mid.ticks_per_beat

    # [8] pitch_range
    pitch_range = (max(pitches) - min(pitches)) / 88.0 if len(pitches) > 1 else 0.0

    # [9] interval_entropy
    intervals = np.abs(np.diff(pitches))
    if len(intervals) > 0:
        hist, _ = np.histogram(intervals, bins=range(0, 25))
        hist = hist / (hist.sum() + 1e-9)
        iv_entropy = float(-np.sum(hist * np.log2(hist + 1e-9))) / 4.58  # normalizado a log2(24)
    else:
        iv_entropy = 0.0

    # [10] tension_mean (intervalos disonantes simultáneos)
    tension_vals = _tension_curve(notes, bar_ticks, _get_total_ticks(mid), n_windows=16)
    tension_mean = float(np.mean(tension_vals)) if len(tension_vals) > 0 else 0.0

    # [11] tension_peak_position
    if len(tension_vals) > 1:
        peak_pos = float(np.argmax(tension_vals)) / len(tension_vals)
    else:
        peak_pos = 0.5

    # [12] track_count normalizado
    n_tracks = len(set(n[5] for n in notes))
    track_norm = min(n_tracks, 8) / 8.0

    return np.concatenate([base, [
        pitch_range,
        iv_entropy,
        tension_mean,
        peak_pos,
        track_norm,
    ]])


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ══════════════════════════════════════════════════════════════════════════════
#  CURVA DE TENSIÓN (compartida por arc y corpus)
# ══════════════════════════════════════════════════════════════════════════════

def _tension_curve(notes: list, bar_ticks: int, total_ticks: int,
                   n_windows: int = 16) -> np.ndarray:
    """
    Curva de tensión en n_windows ventanas temporales.
    Tensión = combinación de disonancia armónica + densidad × velocidad + registro.
    """
    if not notes or total_ticks == 0:
        return np.zeros(n_windows)

    window_size = total_ticks / n_windows
    curve = np.zeros(n_windows)

    for w in range(n_windows):
        t_start = w * window_size
        t_end   = (w + 1) * window_size
        seg     = [n for n in notes if n[0] < t_end and n[1] > t_start]
        if not seg:
            continue

        pitches    = [n[2] for n in seg]
        velocities = [n[3] for n in seg]

        # Disonancia: intervalos simultáneos
        dissonance = 0.0
        if len(pitches) > 1:
            pairs = 0
            for i in range(len(pitches)):
                for j in range(i + 1, min(i + 4, len(pitches))):
                    interval = abs(pitches[i] - pitches[j]) % 12
                    if interval in DISSONANT_INTERVALS:
                        dissonance += 1
                    pairs += 1
            dissonance = dissonance / max(pairs, 1)

        # Actividad: densidad × velocidad media normalizada
        duration   = max(t_end - t_start, 1)
        density    = len(seg) / (duration / max(bar_ticks, 1))
        vel_mean   = np.mean(velocities) / 127.0
        activity   = min(density / 10.0, 1.0) * vel_mean

        # Registro: pitch medio normalizado (agudo = más tensión)
        register = (np.mean(pitches) - 36) / 60.0
        register = max(0.0, min(1.0, register))

        curve[w] = 0.45 * dissonance + 0.35 * activity + 0.20 * register

    return curve


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUADOR: COHERENCE
# ══════════════════════════════════════════════════════════════════════════════

class CoherenceEvaluator:
    """
    Mide la homogeneidad interna de la obra dividiendo en segmentos
    y calculando la varianza de las distancias entre descriptores.
    Score alto = obra internamente consistente.
    """
    NAME = "coherence"

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:

        bar_ticks = _estimate_bar_ticks(mid)
        total     = _get_total_ticks(mid)

        # Usar solo tracks melódicos/armónicos (excluir drums y bajo)
        non_drum = [n for n in notes if n[5] not in topology.drum_tracks]
        working_notes = non_drum if non_drum else notes

        # En modo fast: menos segmentos para reducir coste
        n_bars = max(1, total // bar_ticks)
        if ctx.fast:
            n_segs = max(4, min(8, n_bars // 4))
        else:
            n_segs = max(4, min(16, n_bars // 4))
        seg_ticks = total / n_segs

        # Descriptor por segmento
        descriptors = []
        seg_note_counts = []
        for i in range(n_segs):
            t0 = i * seg_ticks
            t1 = (i + 1) * seg_ticks
            seg_notes = [n for n in working_notes if n[0] < t1 and n[1] > t0]
            descriptors.append(_descriptor_8(seg_notes, bar_ticks))
            seg_note_counts.append(len(seg_notes))

        if all(c == 0 for c in seg_note_counts):
            return DimensionResult(
                score=0.0, confidence=0.0,
                reasons=["Sin notas suficientes para analizar"],
                details={"n_segments": n_segs},
            )

        # Centroide global
        centroid = np.mean(descriptors, axis=0)

        # Distancias al centroide
        dists_to_centroid = [_cosine_distance(d, centroid) for d in descriptors]

        # Distancias entre segmentos consecutivos
        consec_dists = [
            _cosine_distance(descriptors[i], descriptors[i + 1])
            for i in range(len(descriptors) - 1)
        ]

        # Segmentos anómalos: z-score sobre distancias al centroide
        mean_d = np.mean(dists_to_centroid)
        std_d  = np.std(dists_to_centroid)
        anomalous = []
        if std_d > 0:
            for i, d in enumerate(dists_to_centroid):
                if (d - mean_d) / std_d > 1.5:
                    anomalous.append(i + 1)

        # Score: 1 − varianza_normalizada de distancias consecutivas
        # Alta varianza = saltos bruscos = baja coherencia
        if consec_dists:
            variance   = float(np.var(consec_dists))
            mean_cd    = float(np.mean(consec_dists))
            # Normalizar: varianza esperada en obra coherente ~ 0.01–0.05
            var_norm   = min(variance / 0.10, 1.0)
            score_var  = 1.0 - var_norm
            # También penalizar si la distancia media es muy alta (obra muy dispersa)
            score_mean = 1.0 - min(mean_cd / 0.7, 1.0)
            score      = 0.6 * score_var + 0.4 * score_mean
        else:
            score = 0.5

        score = max(0.0, min(1.0, float(score)))

        # Confianza: depende de cuántas notas hay y cuántos segmentos
        min_notes = min(seg_note_counts)
        confidence = 1.0 if min_notes >= 4 else (0.5 if min_notes >= 1 else 0.2)

        reasons = []
        if anomalous:
            reasons.append(f"segmentos anómalos: {anomalous}")
        if consec_dists and float(np.mean(consec_dists)) > 0.4:
            reasons.append("alta variación entre secciones consecutivas")
        if score > 0.75:
            reasons.append("obra internamente consistente")

        return DimensionResult(
            score=score,
            confidence=confidence,
            reasons=reasons,
            details={
                "n_segments":              n_segs,
                "mean_consecutive_dist":   round(float(np.mean(consec_dists)), 4) if consec_dists else None,
                "variance_consecutive":    round(float(np.var(consec_dists)), 4) if consec_dists else None,
                "mean_centroid_dist":      round(mean_d, 4),
                "anomalous_segments":      anomalous,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUADOR: ARC
# ══════════════════════════════════════════════════════════════════════════════

class ArcEvaluator:
    """
    Mide si la obra tiene un arco emocional coherente.
    Sin plan: evalúa si existe algún arco reconocible (no plano, no ruido).
    Con plan: correlaciona la curva real con la planificada.
    """
    NAME = "arc"

    # Formas de arco reconocidas para el modo sin plan
    ARC_TEMPLATES = {
        "ascent":          np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        "climax_early":    np.array([0.2, 0.5, 0.8, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.3,
                                     0.2, 0.2, 0.2, 0.2, 0.2, 0.1]),
        "climax_late":     np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                     0.9, 1.0, 0.8, 0.6, 0.4, 0.2]),
        "climax_center":   np.array([0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0, 0.9, 0.8, 0.6,
                                     0.5, 0.4, 0.3, 0.3, 0.2, 0.1]),
        "tension_release": np.array([0.5, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2,
                                     0.4, 0.6, 0.8, 0.9, 0.7, 0.3]),
        "wave":            np.array([0.3, 0.6, 0.9, 0.6, 0.3, 0.5, 0.8, 1.0, 0.7, 0.4,
                                     0.6, 0.8, 0.6, 0.4, 0.2, 0.1]),
    }

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:

        bar_ticks = _estimate_bar_ticks(mid)
        total     = _get_total_ticks(mid)

        # Excluir drums para el análisis de tensión
        non_drum = [n for n in notes if n[5] not in topology.drum_tracks]
        working  = non_drum if non_drum else notes

        if not working:
            return DimensionResult(
                score=0.0, confidence=0.0,
                reasons=["Sin notas para analizar el arco"],
                details={},
            )

        N_WINDOWS = 8 if ctx.fast else 16
        curve = _tension_curve(working, bar_ticks, total, n_windows=N_WINDOWS)

        # Normalizar curva a [0, 1]
        c_min, c_max = curve.min(), curve.max()
        if c_max > c_min:
            curve_norm = (curve - c_min) / (c_max - c_min)
        else:
            curve_norm = curve.copy()

        climax_pos  = float(np.argmax(curve_norm)) / max(N_WINDOWS - 1, 1)
        flatness    = float(1.0 - (c_max - c_min))  # 0=muy dinámica, 1=plana
        # chaos sobre curva SIN normalizar para evitar amplificación de rangos pequeños
        chaos_raw   = float(np.std(np.diff(curve)))

        details: Dict = {
            "climax_position_real": round(climax_pos, 3),
            "flatness":             round(flatness, 3),
            "tension_curve":        [round(float(v), 3) for v in curve_norm],
        }
        reasons = []

        # ── Modo off (--arc-weight-mode off) ──────────────────────────────────
        if ctx.arc_weight_mode == "off":
            return DimensionResult(
                score=0.5, confidence=0.0,
                reasons=["arc desactivado (--arc-weight-mode off)"],
                details={**details, "arc_detected": "disabled",
                         "plan_correlation": None, "climax_position_planned": None},
            )

        # ── Modo sin plan ─────────────────────────────────────────────────────
        if ctx.plan is None:
            score, arc_name, confidence = self._score_without_plan(
                curve_norm, flatness, reasons,
                chaos_raw=chaos_raw)
            details["arc_detected"] = arc_name
            details["plan_correlation"] = None
            details["climax_position_planned"] = None

        # ── Modo con plan ─────────────────────────────────────────────────────
        else:
            score, confidence = self._score_with_plan(
                curve_norm, ctx.plan, climax_pos, details, reasons)

        # ── Modo static: forzar confianza baja independientemente del resultado
        if ctx.arc_weight_mode == "static" and details.get("arc_detected") not in ("static",):
            # El usuario declara que la pieza tiene arco estático válido.
            # Reducimos la confianza para que composite_score pese menos a arc.
            confidence = min(confidence, 0.40)
            if "arco estático forzado por --arc-weight-mode static" not in reasons:
                reasons.append("arco estático forzado por --arc-weight-mode static")

        score = max(0.0, min(1.0, float(score)))

        return DimensionResult(
            score=score,
            confidence=confidence,
            reasons=reasons,
            details=details,
        )

    def _score_without_plan(self,
                             curve: np.ndarray,
                             flatness: float,
                             reasons: list,
                             chaos_raw: float = 0.5) -> Tuple[float, str, float]:
        """
        Evalúa el arco sin plan: busca el template más parecido y
        penaliza curvas planas o caóticas.

        Detección de arco estático válido (B):
        Una curva plana puede ser intencional (ostinato, forma estrófica,
        marcha) — se distingue del silencio o del ruido por la combinación
        de flatness alta + derivada suave en escala absoluta (chaos_raw,
        medido antes de normalizar para evitar amplificación).
        En ese caso se devuelve score moderado con confianza reducida
        en lugar de penalizar al mínimo.
        """
        # ── Detección de arco estático válido ─────────────────────────────
        if flatness > 0.85:
            # chaos_raw < 0.08 → curva muy suave en escala absoluta
            # → tensión constante por diseño (ostinato, canon, himno)
            if chaos_raw < 0.08:
                reasons.append("arco estático — tensión constante por diseño (ostinato/forma estrófica)")
                return 0.62, "static", 0.45   # score moderado, confianza baja
            else:
                reasons.append("curva de tensión prácticamente plana")
                return 0.25, "flat", 0.70

        # Correlación con templates
        best_corr, best_name = -1.0, "unknown"
        for name, template in self.ARC_TEMPLATES.items():
            # Redimensionar template si es necesario
            t = np.interp(np.linspace(0, 1, len(curve)),
                          np.linspace(0, 1, len(template)),
                          template)
            corr = float(np.corrcoef(curve, t)[0, 1])
            if corr > best_corr:
                best_corr, best_name = corr, name

        # Penalización por curva caótica (demasiada varianza de derivada)
        diff        = np.diff(curve)
        chaos_score = float(np.std(diff))

        # Score final
        # corr en [-1, 1] → normalizar a [0, 1]
        corr_score  = (best_corr + 1.0) / 2.0
        # penalizar caos (std de derivada > 0.3 = muy caótico)
        chaos_pen   = min(chaos_score / 0.3, 1.0) * 0.3
        score       = corr_score - chaos_pen

        if best_corr > 0.6:
            reasons.append(f"arco reconocible: {best_name} (corr={best_corr:.2f})")
        elif best_corr > 0.3:
            reasons.append(f"arco débil: {best_name} (corr={best_corr:.2f})")
        else:
            reasons.append("arco emocional no reconocible")

        if chaos_score > 0.3:
            reasons.append("curva de tensión muy irregular")

        return float(score), best_name, 0.75

    def _score_with_plan(self,
                          curve: np.ndarray,
                          plan: dict,
                          climax_pos: float,
                          details: dict,
                          reasons: list) -> Tuple[float, float]:
        """
        Evalúa correlación con la curva planificada en el plan.
        Soporta formatos de theorist.json y narrator_plan.json.
        """
        planned_curve = self._extract_planned_curve(plan, len(curve))

        if planned_curve is None:
            # Plan encontrado pero sin curva de tensión extraíble
            reasons.append("plan cargado pero sin curva de tensión; usando evaluación libre")
            score, arc_name, conf = self._score_without_plan(curve, float(1.0 - (curve.max() - curve.min())), reasons)
            details["arc_detected"] = arc_name
            details["plan_correlation"] = None
            return score, conf * 0.8

        # Normalizar curva planificada
        p_min, p_max = planned_curve.min(), planned_curve.max()
        if p_max > p_min:
            planned_norm = (planned_curve - p_min) / (p_max - p_min)
        else:
            planned_norm = planned_curve.copy()

        # Correlación de Pearson entre curva real y planificada
        corr = float(np.corrcoef(curve, planned_norm)[0, 1])
        if math.isnan(corr):
            corr = 0.0

        # Penalización por posición errónea del clímax
        planned_climax = float(np.argmax(planned_norm)) / max(len(planned_norm) - 1, 1)
        climax_error   = abs(climax_pos - planned_climax)

        details["plan_correlation"]       = round(corr, 4)
        details["climax_position_planned"] = round(planned_climax, 3)
        details["climax_error"]           = round(climax_error, 3)

        corr_score   = (corr + 1.0) / 2.0
        climax_pen   = climax_error * 0.4
        score        = corr_score - climax_pen

        if corr > 0.7:
            reasons.append(f"arco sigue el plan fielmente (corr={corr:.2f})")
        elif corr > 0.4:
            reasons.append(f"arco sigue el plan parcialmente (corr={corr:.2f})")
        else:
            reasons.append(f"arco diverge del plan (corr={corr:.2f})")

        if climax_error > 0.25:
            reasons.append(f"clímax desplazado: real={climax_pos:.2f}, plan={planned_climax:.2f}")

        return float(score), 0.90

    def _extract_planned_curve(self,
                                plan: dict,
                                n_points: int) -> Optional[np.ndarray]:
        """
        Intenta extraer una curva de tensión del plan.
        Soporta formatos de theorist.json y narrator_plan.json.
        """
        # Formato theorist.json: plan["curves"]["tension"]
        curves = plan.get("curves", {})
        if isinstance(curves, dict):
            tension = curves.get("tension") or curves.get("tension_curve")
            if tension and isinstance(tension, list):
                arr = np.array([float(v) for v in tension])
                return np.interp(np.linspace(0, 1, n_points),
                                 np.linspace(0, 1, len(arr)), arr)

        # Formato narrator_plan.json: plan["sections"][i]["tension"]
        sections = plan.get("sections", [])
        if sections and isinstance(sections, list):
            tensions = []
            for sec in sections:
                t = sec.get("tension", sec.get("tension_target"))
                if t is not None:
                    tensions.append(float(t))
            if tensions:
                arr = np.array(tensions)
                return np.interp(np.linspace(0, 1, n_points),
                                 np.linspace(0, 1, len(arr)), arr)

        # Formato simple: plan["tension_curve"] = [...]
        tc = plan.get("tension_curve") or plan.get("tension")
        if tc and isinstance(tc, list):
            arr = np.array([float(v) for v in tc])
            return np.interp(np.linspace(0, 1, n_points),
                             np.linspace(0, 1, len(arr)), arr)

        return None


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUADOR: FORMAL
# ══════════════════════════════════════════════════════════════════════════════

class FormalEvaluator:
    """
    Corrección de voice leading e intervalos.

    Dos niveles según la topología del MIDI:
      · Básico  (single / multi_unlabeled): analiza intervalos simultáneos
        en ventanas temporales — detecta densidad de disonancia, saltos
        grandes sin resolución y notas fuera de escala.
      · Completo (multi_labeled): aplica un subconjunto de las reglas
        clásicas de voice leading asignando voces por track.

    Score alto = pieza formalmente correcta.
    """
    NAME = "formal"

    # Reglas habilitadas en modo completo
    # R01 paralelas de 5ª · R02 paralelas de 8ª · R03 quintas/octavas directas
    # R04 cruzamiento de voces · R07 saltos grandes · R08 7ª sin resolver
    # R09 sensible sin resolver · R10 7ª de dominante sin resolver

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:

        if not notes:
            return DimensionResult(
                score=0.5, confidence=0.0,
                reasons=["Sin notas para análisis formal"],
                details={},
            )

        key_root, key_mode = _detect_key_simple(notes)
        bar_ticks          = _estimate_bar_ticks(mid)
        tpb                = mid.ticks_per_beat

        # En modo fast siempre usamos nivel básico (evita el bucle por-beat)
        if ctx.fast:
            return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode)

        if topology.type == "multi_labeled" and len(topology.melody_tracks) > 0:
            return self._evaluate_voiced(notes, topology, bar_ticks, tpb,
                                         key_root, key_mode)
        else:
            return self._evaluate_basic(notes, bar_ticks, tpb,
                                        key_root, key_mode)

    # ── Nivel básico ──────────────────────────────────────────────────────────

    def _evaluate_basic(self, notes: list, bar_ticks: int, tpb: int,
                        key_root: int, key_mode: str) -> DimensionResult:
        """
        Analiza intervalos simultáneos en ventanas de 1 beat.
        Métricas: densidad de disonancia, saltos grandes, notas fuera de escala.
        Nivel 1+3: localiza cada problema por compás.
        """
        MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
        MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
        scale_ints  = MAJOR_SCALE if key_mode == 'major' else MINOR_SCALE
        scale_pcs   = {(key_root + i) % 12 for i in scale_ints}

        total_ticks = max(n[1] for n in notes)
        beat_ticks  = tpb
        n_beats     = max(1, total_ticks // beat_ticks)

        dissonance_beats = 0
        total_beats_with_notes = 0
        large_leaps = 0
        leap_total  = 0
        oos_notes   = 0

        # Listas de localización por compás
        dissonant_bars: list = []
        leap_bars:      list = []
        oos_bars:       list = []

        pitches_seq = [n[2] for n in notes]

        # Saltos grandes: localizar por compás
        for i in range(1, len(notes)):
            interval = abs(pitches_seq[i] - pitches_seq[i - 1])
            if interval > 0:
                leap_total += 1
                if interval > 9:
                    large_leaps += 1
                    bar = int(notes[i][0] // bar_ticks) + 1
                    leap_bars.append(bar)

        # Notas fuera de escala: localizar por compás
        for n in notes:
            if (n[2] % 12) not in scale_pcs:
                oos_notes += 1
                bar = int(n[0] // bar_ticks) + 1
                oos_bars.append(bar)
        oos_ratio = oos_notes / max(len(notes), 1)

        # Disonancia por beat: localizar compases problemáticos
        for b in range(n_beats):
            t0 = b * beat_ticks
            t1 = (b + 1) * beat_ticks
            simultaneous = [n[2] for n in notes if n[0] < t1 and n[1] > t0]
            if len(simultaneous) < 2:
                continue
            total_beats_with_notes += 1
            has_dissonance = False
            for i in range(len(simultaneous)):
                for j in range(i + 1, min(i + 4, len(simultaneous))):
                    iv = abs(simultaneous[i] - simultaneous[j]) % 12
                    if iv in DISSONANT_INTERVALS:
                        has_dissonance = True
                        break
                if has_dissonance:
                    break
            if has_dissonance:
                dissonance_beats += 1
                bar = int(t0 // bar_ticks) + 1
                dissonant_bars.append(bar)

        diss_ratio  = dissonance_beats / max(total_beats_with_notes, 1)
        leap_ratio  = large_leaps / max(leap_total, 1)

        s_diss = 1.0 - min(diss_ratio / 0.5, 1.0)
        s_leap = 1.0 - min(leap_ratio / 0.15, 1.0)
        s_oos  = 1.0 - min(oos_ratio / 0.20, 1.0)

        score = 0.45 * s_diss + 0.25 * s_leap + 0.30 * s_oos
        score = max(0.0, min(1.0, score))

        reasons = []
        if diss_ratio > 0.4:
            reasons.append(f"alta densidad de disonancia: {diss_ratio:.0%} de beats")
        if leap_ratio > 0.12:
            reasons.append(f"muchos saltos grandes: {large_leaps}/{leap_total}")
        if oos_ratio > 0.15:
            reasons.append(f"notas fuera de escala: {oos_ratio:.0%}")
        if score > 0.75:
            reasons.append("corrección formal básica adecuada")

        # Deduplicar listas de compases (mantener orden)
        def dedup(lst): return sorted(set(lst))

        return DimensionResult(
            score      = score,
            confidence = 0.65,
            reasons    = reasons,
            details    = {
                "level":              "basic",
                "key":                f"{PITCH_NAMES[key_root]} {key_mode}",
                "dissonance_ratio":   round(diss_ratio, 4),
                "large_leap_ratio":   round(leap_ratio, 4),
                "out_of_scale_ratio": round(oos_ratio, 4),
                "dissonant_bars":     dedup(dissonant_bars),
                "leap_bars":          dedup(leap_bars),
                "oos_bars":           dedup(oos_bars),
            },
        )

    # ── Nivel completo (voces asignadas por track) ────────────────────────────

    def _evaluate_voiced(self, notes: list, topology: Topology,
                         bar_ticks: int, tpb: int,
                         key_root: int, key_mode: str) -> DimensionResult:
        """
        Evalúa reglas de voice leading con voces identificadas por track.
        Cada track es una voz; se ordenan por pitch medio descendente (S→A→T→B).
        """
        # Agrupar notas por track, excluir drums
        tracks_notes: Dict[int, list] = defaultdict(list)
        for n in notes:
            if n[5] not in topology.drum_tracks:
                tracks_notes[n[5]].append(n)

        if len(tracks_notes) < 2:
            return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode)

        # Ordenar voces por pitch medio descendente → [S, A, T, B, ...]
        voice_order = sorted(
            tracks_notes.keys(),
            key=lambda ti: np.mean([n[2] for n in tracks_notes[ti]]) if tracks_notes[ti] else 0,
            reverse=True,
        )

        violations: List[dict] = []
        total_ticks = max(n[1] for n in notes)
        beat_ticks  = tpb

        # Construir chords: en cada beat, la nota activa de cada voz
        n_beats = max(1, total_ticks // beat_ticks)

        def active_pitch(track_notes: list, t0: int, t1: int) -> Optional[int]:
            """Nota de la voz activa en el beat [t0, t1) — la más reciente."""
            cands = [n for n in track_notes if n[0] < t1 and n[1] > t0]
            return cands[-1][2] if cands else None

        prev_chord: Dict[int, Optional[int]] = {v: None for v in voice_order}

        for b in range(n_beats):
            t0   = b * beat_ticks
            t1   = (b + 1) * beat_ticks
            bar  = b * beat_ticks // bar_ticks + 1

            chord: Dict[int, Optional[int]] = {
                v: active_pitch(tracks_notes[v], t0, t1)
                for v in voice_order
            }

            pitches_now  = [chord[v] for v in voice_order if chord[v] is not None]
            pitches_prev = [prev_chord[v] for v in voice_order if prev_chord[v] is not None]

            # R04: cruzamiento de voces (voz i debe ser > voz i+1)
            for i in range(len(voice_order) - 1):
                pi = chord[voice_order[i]]
                pj = chord[voice_order[i + 1]]
                if pi is not None and pj is not None and pi < pj:
                    violations.append({
                        "rule": "voice_crossing",
                        "bar":  bar,
                        "voices": [voice_order[i], voice_order[i + 1]],
                        "detail": f"voz {i} ({pi}) < voz {i+1} ({pj})",
                    })

            # R01/R02: paralelas de 5ª y 8ª entre voces adyacentes
            if len(pitches_prev) >= 2 and len(pitches_now) >= 2:
                for i in range(min(len(voice_order), len(pitches_now)) - 1):
                    v1, v2 = voice_order[i], voice_order[i + 1]
                    p1_now  = chord[v1]
                    p2_now  = chord[v2]
                    p1_prev = prev_chord[v1]
                    p2_prev = prev_chord[v2]
                    if None in (p1_now, p2_now, p1_prev, p2_prev):
                        continue
                    iv_now  = abs(p1_now  - p2_now)  % 12
                    iv_prev = abs(p1_prev - p2_prev) % 12
                    m1 = p1_now - p1_prev  # movimiento voz 1
                    m2 = p2_now - p2_prev  # movimiento voz 2
                    # Paralelas: mismo tipo de intervalo, mismo sentido de movimiento
                    if iv_now == iv_prev and m1 != 0 and m2 != 0 and \
                            (m1 > 0) == (m2 > 0):
                        if iv_now == 7:   # 5ª perfecta
                            violations.append({
                                "rule": "parallel_fifths",
                                "bar":  bar,
                                "voices": [v1, v2],
                            })
                        elif iv_now == 0:  # octava / unísono
                            violations.append({
                                "rule": "parallel_octaves",
                                "bar":  bar,
                                "voices": [v1, v2],
                            })

            # R07/R09: saltos grandes y sensible sin resolver (voz más aguda)
            top_voice = voice_order[0]
            p_now  = chord[top_voice]
            p_prev = prev_chord[top_voice]
            if p_now is not None and p_prev is not None:
                leap = p_now - p_prev
                # R07: salto > 9 semitonos (> 6ª mayor)
                if abs(leap) > 9:
                    violations.append({
                        "rule": "large_leap",
                        "bar":  bar,
                        "voices": [top_voice],
                        "detail": f"salto de {leap:+d} semitonos",
                    })
                # R09: sensible (a distancia 1 semitono de la tónica) debe subir
                leading_tone = (key_root - 1) % 12
                if (p_prev % 12) == leading_tone and leap < 0:
                    violations.append({
                        "rule": "leading_tone_unresolved",
                        "bar":  bar,
                        "voices": [top_voice],
                    })

            prev_chord = chord

        # Score a partir de tasa de violaciones por beat
        v_per_beat = len(violations) / max(n_beats, 1)
        # Umbral: > 0.3 violaciones/beat = muy deficiente
        score = 1.0 - min(v_per_beat / 0.3, 1.0)
        score = max(0.0, min(1.0, score))

        # Contar por tipo
        by_rule: Dict[str, int] = defaultdict(int)
        for v in violations:
            by_rule[v["rule"]] += 1

        reasons = []
        for rule, count in sorted(by_rule.items(), key=lambda x: -x[1]):
            label = {
                "parallel_fifths":        "paralelas de 5ª",
                "parallel_octaves":       "paralelas de 8ª",
                "voice_crossing":         "cruzamientos de voz",
                "large_leap":             "saltos grandes",
                "leading_tone_unresolved":"sensible sin resolver",
            }.get(rule, rule)
            reasons.append(f"{label}: {count}")
        if not violations:
            reasons.append("sin violaciones de voice leading detectadas")

        return DimensionResult(
            score      = score,
            confidence = 0.85,
            reasons    = reasons,
            details    = {
                "level":           "voiced",
                "key":             f"{PITCH_NAMES[key_root]} {key_mode}",
                "n_voices":        len(voice_order),
                "n_beats_analyzed": n_beats,
                "violation_count": len(violations),
                "violations_per_beat": round(v_per_beat, 4),
                "by_rule":         dict(by_rule),
                "violations":      violations[:20],  # máx 20 en el JSON
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUADOR: MELODIC
# ══════════════════════════════════════════════════════════════════════════════

class MelodicEvaluator:
    """
    Interés melódico y variedad. Combina cuatro submétricas:
      1. Entropía de intervalos  — ni monótono ni caótico
      2. Estructuración del contorno — inflexiones periódicas
      3. Predictibilidad de Markov  — diversidad de transiciones pitch→pitch
      4. Ratio repetición/variación — equilibrio motívico

    Usa la pista melódica si está identificada; si no, el track con
    pitch medio más alto entre los no-drum.
    """
    NAME = "melodic"

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:

        # Seleccionar notas melódicas
        if topology.melody_tracks:
            mel_notes = [n for n in notes if n[5] in topology.melody_tracks]
        else:
            non_drum = [n for n in notes if n[5] not in topology.drum_tracks]
            mel_notes = non_drum if non_drum else notes

        if len(mel_notes) < 8:
            return DimensionResult(
                score=0.5, confidence=0.0,
                reasons=["Insuficientes notas melódicas para análisis"],
                details={"n_notes": len(mel_notes), "problem_windows": []},
            )

        pitches = [n[2] for n in mel_notes]

        s1, d1 = self._interval_entropy(pitches)
        s2, d2 = self._contour_structure(pitches)

        if ctx.fast:
            score = 0.55 * s1 + 0.45 * s2
            s3, d3 = 0.5, {"markov_entropy_norm": None, "markov_entropy_bits": None}
            s4, d4 = 0.5, {"repetition_ratio": None, "window_size": None}
        else:
            s3, d3 = self._markov_entropy(pitches)
            s4, d4 = self._repetition_variation(pitches)
            score = 0.30 * s1 + 0.25 * s2 + 0.25 * s3 + 0.20 * s4
        score = max(0.0, min(1.0, float(score)))

        reasons = []
        if s1 < 0.4:
            reasons.append("intervalos muy repetitivos o caóticos")
        if s2 < 0.4:
            reasons.append("contorno melódico sin estructura")
        if s3 < 0.4:
            reasons.append("melodía muy predecible o totalmente aleatoria")
        if s4 < 0.4:
            reasons.append("poco equilibrio entre repetición y variación")
        if score > 0.70:
            reasons.append("melodía con buen interés y variedad")

        # Nivel 3: localización temporal de problemas melódicos
        problem_windows = [] if ctx.fast else self._locate_melodic_problems(mel_notes, mid)

        details = {
            "n_melody_notes":  len(mel_notes),
            "problem_windows": problem_windows,
            **d1, **d2, **d3, **d4,
        }

        # Confianza: sube con el número de notas
        confidence = min(1.0, len(mel_notes) / 32)

        return DimensionResult(
            score=score,
            confidence=confidence,
            reasons=reasons,
            details=details,
        )

    # ── Submétrica 1: Entropía de intervalos ─────────────────────────────────

    def _interval_entropy(self, pitches: list) -> Tuple[float, dict]:
        """
        Entropía de Shannon sobre el histograma de intervalos absolutos.
        Rango óptimo: ni monodia repetitiva (baja entropía) ni saltos aleatorios
        (entropía máxima sin estructura).
        Curva de score: máximo en entropía media (~0.55 de la máxima posible).
        """
        if len(pitches) < 2:
            return 0.5, {"interval_entropy": None}

        intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
        # Bins: 0..24 semitonos
        hist = np.zeros(25)
        for iv in intervals:
            hist[min(iv, 24)] += 1
        hist /= hist.sum() + 1e-9

        if SCIPY_OK:
            from scipy.stats import entropy as sp_entropy
            H = float(sp_entropy(hist + 1e-12, base=2))
        else:
            H = float(-np.sum(hist * np.log2(hist + 1e-9)))

        H_max = math.log2(25)          # ~4.64 bits
        H_norm = H / H_max             # 0–1

        # Curva de score: gaussiana centrada en H_norm = 0.55
        # score = exp(-((H_norm - 0.55)^2) / (2 * 0.20^2))
        optimal = 0.55
        width   = 0.22
        score   = math.exp(-((H_norm - optimal) ** 2) / (2 * width ** 2))

        return float(score), {
            "interval_entropy_bits":  round(H, 4),
            "interval_entropy_norm":  round(H_norm, 4),
        }

    # ── Submétrica 2: Estructuración del contorno ─────────────────────────────

    def _contour_structure(self, pitches: list) -> Tuple[float, dict]:
        """
        Detecta inflexiones (cambios de dirección) en la melodía.
        Una melodía bien estructurada tiene inflexiones periódicas
        (~1 cada 3–8 notas), no aleatorias ni ausentes.
        """
        if len(pitches) < 4:
            return 0.5, {"contour_inflections": None, "inflection_rate": None}

        directions = []
        for i in range(1, len(pitches)):
            d = pitches[i] - pitches[i - 1]
            if d > 0:
                directions.append(1)
            elif d < 0:
                directions.append(-1)
            # d==0: ignorar (nota repetida)

        inflections = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i - 1]:
                inflections += 1

        n_moves = max(len(directions), 1)
        rate    = inflections / n_moves   # 0=línea recta, 1=zigzag puro

        # Score: gaussiana centrada en rate=0.40 (inflexión cada ~2.5 notas)
        # Extremos: rate<0.10 → sin variedad, rate>0.75 → zigzag sin estructura
        optimal = 0.40
        width   = 0.20
        score   = math.exp(-((rate - optimal) ** 2) / (2 * width ** 2))

        return float(score), {
            "contour_inflections":  inflections,
            "inflection_rate":      round(rate, 4),
        }

    # ── Submétrica 3: Entropía de Markov orden 1 ─────────────────────────────

    def _markov_entropy(self, pitches: list) -> Tuple[float, dict]:
        """
        Construye la matriz de transición pitch→pitch (en clases de pc mod 12)
        y calcula la entropía media de cada fila.
        Alta entropía = cada nota puede ir a cualquier lado (caótico).
        Baja entropía = cada nota siempre va al mismo lado (muy predecible).
        Óptimo: entropía media moderada.
        """
        if len(pitches) < 4:
            return 0.5, {"markov_entropy": None}

        pcs = [p % 12 for p in pitches]
        # Matriz de transición 12×12
        T = np.zeros((12, 12))
        for i in range(len(pcs) - 1):
            T[pcs[i], pcs[i + 1]] += 1

        # Entropía por fila (solo filas con al menos 1 transición)
        row_entropies = []
        for row in T:
            if row.sum() > 0:
                p = row / row.sum()
                if SCIPY_OK:
                    from scipy.stats import entropy as sp_entropy
                    H = float(sp_entropy(p + 1e-12, base=2))
                else:
                    H = float(-np.sum(p * np.log2(p + 1e-9)))
                row_entropies.append(H)

        if not row_entropies:
            return 0.5, {"markov_entropy": None}

        H_mean  = float(np.mean(row_entropies))
        H_max   = math.log2(12)   # ~3.58 bits
        H_norm  = H_mean / H_max

        # Score: gaussiana centrada en H_norm=0.50
        optimal = 0.50
        width   = 0.25
        score   = math.exp(-((H_norm - optimal) ** 2) / (2 * width ** 2))

        return float(score), {
            "markov_entropy_norm": round(H_norm, 4),
            "markov_entropy_bits": round(H_mean, 4),
        }

    # ── Submétrica 4: Ratio repetición/variación ──────────────────────────────

    def _repetition_variation(self, pitches: list) -> Tuple[float, dict]:
        """
        Compara ventanas de N notas usando distancia de Hamming normalizada
        entre pares de ventanas.
          · Muchas ventanas idénticas → sin variación (aburrido)
          · Ninguna ventana parecida  → sin motivos reconocibles (caótico)
        Óptimo: ratio de similitud media ~0.35–0.55.
        """
        W = min(6, len(pitches) // 4)  # tamaño de ventana
        if W < 2:
            return 0.5, {"repetition_ratio": None, "window_size": None}

        # Cuantizar intervalos de la ventana (dirección: -1/0/+1)
        def window_shape(start: int) -> tuple:
            seg = pitches[start: start + W]
            return tuple(
                0 if seg[i] == seg[i - 1] else (1 if seg[i] > seg[i - 1] else -1)
                for i in range(1, len(seg))
            )

        windows = [window_shape(i) for i in range(len(pitches) - W + 1)]
        if len(windows) < 2:
            return 0.5, {"repetition_ratio": None, "window_size": W}

        # Similitud media entre pares de ventanas no adyacentes
        sim_vals = []
        step = max(1, len(windows) // 20)  # muestreo para eficiencia
        for i in range(0, len(windows) - W, step):
            for j in range(i + W, min(i + W * 4, len(windows)), step):
                matches = sum(a == b for a, b in zip(windows[i], windows[j]))
                sim_vals.append(matches / max(W - 1, 1))

        if not sim_vals:
            return 0.5, {"repetition_ratio": None, "window_size": W}

        sim_mean = float(np.mean(sim_vals))

        # Score: gaussiana centrada en sim=0.45 (algo de repetición, algo de variedad)
        optimal = 0.45
        width   = 0.25
        score   = math.exp(-((sim_mean - optimal) ** 2) / (2 * width ** 2))

        return float(score), {
            "repetition_ratio": round(sim_mean, 4),
            "window_size":      W,
        }


    # ── Nivel 3: localización temporal ───────────────────────────────────────

    def _locate_melodic_problems(self, mel_notes: list,
                                  mid: MidiFile) -> list:
        """
        Divide la melodía en ventanas de 4 compases y detecta en cuáles
        hay problemas de contorno, saltos o zigzag excesivo.
        Devuelve lista de {kind, bars, detail}.
        """
        bar_ticks   = _estimate_bar_ticks(mid)
        total_ticks = max(n[1] for n in mel_notes)
        n_bars      = max(1, total_ticks // bar_ticks)
        WIN         = 4          # compases por ventana
        problems    = []

        for w_start in range(1, n_bars + 1, WIN):
            w_end  = w_start + WIN - 1
            t0     = (w_start - 1) * bar_ticks
            t1     = w_end * bar_ticks
            seg    = [n for n in mel_notes if n[0] >= t0 and n[0] < t1]
            if len(seg) < 3:
                continue
            pitches = [n[2] for n in seg]
            bars    = list(range(w_start, min(w_end + 1, n_bars + 1)))

            # Saltos grandes (> 9 semitonos) en la ventana
            leaps = [abs(pitches[i] - pitches[i-1])
                     for i in range(1, len(pitches))
                     if abs(pitches[i] - pitches[i-1]) > 9]
            if len(leaps) >= 2:
                problems.append({
                    "kind":   "saltos grandes",
                    "bars":   bars,
                    "detail": f"{len(leaps)} saltos >9 semitonos",
                })

            # Zigzag: inflection_rate > 0.75 en la ventana
            dirs = []
            for i in range(1, len(pitches)):
                d = pitches[i] - pitches[i-1]
                if d != 0:
                    dirs.append(1 if d > 0 else -1)
            if len(dirs) > 2:
                inflections = sum(1 for i in range(1, len(dirs))
                                  if dirs[i] != dirs[i-1])
                rate = inflections / len(dirs)
                if rate > 0.75:
                    problems.append({
                        "kind":   "zigzag excesivo",
                        "bars":   bars,
                        "detail": f"inflection_rate={rate:.2f}",
                    })

            # Rango muy estrecho (< 3 semitonos): melodía estancada
            rng = max(pitches) - min(pitches)
            if rng < 3:
                problems.append({
                    "kind":   "melodía estancada",
                    "bars":   bars,
                    "detail": f"rango={rng} semitonos",
                })

        return problems


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUADOR: CORPUS
# ══════════════════════════════════════════════════════════════════════════════

class CorpusEvaluator:
    """
    Similitud estilística a una colección de referencia.
    Vectoriza la obra con el descriptor de 13 dimensiones y calcula
    la distancia a los k vecinos más cercanos del corpus.
    Score alto = estilísticamente cercano al corpus.

    También calcula el percentil real de la obra dentro del corpus
    (distribución de distancias internas del corpus como referencia).
    """
    NAME  = "corpus"
    K     = 5     # vecinos más cercanos

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:

        if ctx.corpus_vectors is None or len(ctx.corpus_vectors) == 0:
            return DimensionResult(
                score=0.5, confidence=0.0,
                reasons=["Sin corpus disponible"],
                details={"status": "no_corpus"},
            )

        query_vec = _descriptor_13(notes, mid)
        corpus    = ctx.corpus_vectors
        k         = min(3 if ctx.fast else self.K, len(corpus))

        # Distancias a todos los elementos del corpus
        dists = np.array([_cosine_distance(query_vec, cv) for cv in corpus])
        idx_sorted = np.argsort(dists)
        knn_dists  = dists[idx_sorted[:k]]
        mean_knn   = float(np.mean(knn_dists))

        # Score: 1 − distancia_media_normalizada
        # La distancia coseno máxima realista entre MIDIs musicales es ~0.7
        score = 1.0 - min(mean_knn / 0.7, 1.0)
        score = max(0.0, min(1.0, float(score)))

        # Percentil interno del corpus (omitido en modo fast)
        percentile = None if ctx.fast else self._corpus_percentile(
            query_vec, corpus, dists)

        # Nombres de los vecinos más cercanos (si el contexto los tiene)
        nearest_names = []
        if hasattr(ctx, 'corpus_paths') and ctx.corpus_paths:
            for i in idx_sorted[:k]:
                if i < len(ctx.corpus_paths):
                    nearest_names.append(Path(ctx.corpus_paths[i]).name)

        reasons = []
        if mean_knn < 0.15:
            reasons.append(f"muy cercano al corpus (dist_knn={mean_knn:.3f})")
        elif mean_knn < 0.35:
            reasons.append(f"estilísticamente compatible con el corpus (dist_knn={mean_knn:.3f})")
        elif mean_knn < 0.55:
            reasons.append(f"alejado del corpus (dist_knn={mean_knn:.3f})")
        else:
            reasons.append(f"estilo muy distinto al corpus (dist_knn={mean_knn:.3f})")

        return DimensionResult(
            score      = score,
            confidence = 1.0,
            reasons    = reasons,
            details    = {
                "mean_dist_knn":  round(mean_knn, 4),
                "k":              k,
                "corpus_size":    len(corpus),
                "percentile":     round(percentile, 2) if percentile is not None else None,
                "nearest_files":  nearest_names,
                "knn_distances":  [round(float(d), 4) for d in knn_dists],
            },
        )

    def _corpus_percentile(self,
                            query_vec: np.ndarray,
                            corpus: List[np.ndarray],
                            dists: np.ndarray) -> Optional[float]:
        """
        Calcula qué porcentaje de los elementos del corpus están
        más lejos entre sí que la obra evaluada de su vecino más cercano.
        Un percentil alto significa que la obra es inusualmente similar
        al corpus (está en una zona densa).
        """
        if len(corpus) < 4:
            return None

        # Distancia interna: muestreo de hasta 200 pares aleatorios
        rng   = np.random.default_rng(42)
        n     = len(corpus)
        n_samples = min(200, n * (n - 1) // 2)
        pairs = rng.choice(n, size=(n_samples, 2), replace=True)
        internal_dists = [
            _cosine_distance(corpus[i], corpus[j])
            for i, j in pairs if i != j
        ]
        if not internal_dists:
            return None

        query_min_dist = float(dists.min())
        # Percentil: fracción de distancias internas > query_min_dist
        percentile = float(np.mean(np.array(internal_dists) > query_min_dist)) * 100
        return percentile


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE COMPUESTO
# ══════════════════════════════════════════════════════════════════════════════

def composite_score(results: Dict[str, DimensionResult],
                    weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Media ponderada de los scores activos (confidence > 0).

    Ponderación por confianza (C):
    El peso efectivo de cada dimensión se escala por su confidence antes
    de renormalizar. Dimensiones con confidence baja (evaluación incierta,
    arco estático detectado, nivel básico de formal) pesan automáticamente
    menos sin que el usuario tenga que ajustar nada.

      peso_efectivo[k] = weights[k] * confidence[k]

    Esto es transparente: los pesos declarados siguen siendo el punto de
    partida, pero la confianza actúa como modulador continuo.
    """
    active = {k: r for k, r in results.items() if r.confidence > 0.0}
    if not active:
        return 0.0, {}

    # Peso base × confianza
    raw_w = {k: weights.get(k, 0.0) * active[k].confidence for k in active}
    total_w = sum(raw_w.values())

    if total_w < 1e-9:
        eff_weights = {k: 1.0 / len(active) for k in active}
    else:
        eff_weights = {k: raw_w[k] / total_w for k in active}

    score = sum(eff_weights[k] * active[k].score for k in active)
    return float(score), eff_weights


def absolute_label(score: float) -> str:
    if score < ABSOLUTE_THRESHOLDS["bajo"]:
        return "bajo"
    if score < ABSOLUTE_THRESHOLDS["medio"]:
        return "medio"
    if score < ABSOLUTE_THRESHOLDS["alto"]:
        return "alto"
    return "muy alto"


def relativize(abs_score: float,
               corpus_vectors: Optional[List[np.ndarray]],
               query_vector:   Optional[np.ndarray],
               weights:        Optional[Dict[str, float]] = None,
               active_dims:    Optional[List[str]] = None) -> Optional[float]:
    """
    Calcula el percentil del score absoluto de la obra dentro del corpus.

    Estrategia: evalúa cada MIDI del corpus con CoherenceEvaluator y
    ArcEvaluator (los dos evaluadores puramente vectoriales, sin necesidad
    del fichero original) para construir una distribución de scores de
    referencia. El percentil de la obra es la fracción de MIDIs del corpus
    con score inferior.

    Requiere corpus_vectors y query_vector. Devuelve None si el corpus
    tiene menos de 4 elementos.
    """
    if corpus_vectors is None or query_vector is None or len(corpus_vectors) < 4:
        return None

    # Proxy rápido de calidad para cada vector del corpus:
    # usamos la distancia coseno al centroide del corpus como señal inversa
    # (los MIDIs más centrales = más representativos = referencia de calidad media)
    # y la comparamos con la distancia de la obra al corpus.
    #
    # Esto nos da una distribución de "centralidad" sobre la que podemos
    # calcular un percentil real sin necesidad de re-evaluar cada MIDI.
    corpus_arr = np.array(corpus_vectors)            # (N, 13)
    centroid   = corpus_arr.mean(axis=0)             # (13,)

    # Distancias de cada MIDI del corpus al centroide
    corpus_dists = np.array([
        _cosine_distance(v, centroid) for v in corpus_vectors
    ])

    # Distancia de la obra al centroide del corpus
    query_dist = _cosine_distance(query_vector, centroid)

    # Percentil: fracción del corpus con distancia > query_dist
    # (obra más central = más similar al corpus = percentil más alto)
    percentile = float(np.mean(corpus_dists > query_dist)) * 100
    return percentile


# ══════════════════════════════════════════════════════════════════════════════
#  CACHÉ DE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus_cache(cache_path: str) -> dict:
    try:
        with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        if data.get("version") != VERSION:
            print(f"  [AVISO] Caché de versión distinta ({data.get('version')}); se recalculará.")
            return {}
        return data.get("entries", {})
    except Exception as e:
        print(f"  [AVISO] No se pudo cargar caché: {e}")
        return {}


def save_corpus_cache(cache_path: str, entries: dict):
    try:
        data = {"version": VERSION, "entries": entries}
        with gzip.open(cache_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"  Caché guardado: {cache_path} ({len(entries)} entradas)")
    except Exception as e:
        print(f"  [AVISO] No se pudo guardar caché: {e}")


def vectorize_corpus(corpus_dir: str,
                     cache: Optional[dict] = None,
                     verbose: bool = False) -> Tuple[List[np.ndarray], List[str], dict]:
    """
    Vectoriza todos los MIDIs de corpus_dir.
    Usa caché para evitar recalcular ficheros sin cambios.
    Devuelve (vectors, paths, entries_actualizadas).
    Los tres arrays son paralelos: vectors[i] corresponde a paths[i].
    """
    cache = cache or {}
    midi_files = sorted(Path(corpus_dir).rglob("*.mid")) + \
                 sorted(Path(corpus_dir).rglob("*.midi"))

    if not midi_files:
        print(f"  [AVISO] No se encontraron MIDIs en {corpus_dir}")
        return [], [], cache

    vectors = []
    paths   = []
    updated = dict(cache)
    n_cached, n_computed = 0, 0

    for mf in midi_files:
        key = str(mf.resolve())
        stat = mf.stat()
        mtime, size = stat.st_mtime, stat.st_size

        if (key in cache and
                cache[key].get("mtime") == mtime and
                cache[key].get("size") == size):
            vectors.append(np.array(cache[key]["vector"]))
            paths.append(key)
            n_cached += 1
            continue

        try:
            mid  = MidiFile(str(mf))
            nts  = _extract_notes(mid)
            vec  = _descriptor_13(nts, mid)
            vectors.append(vec)
            paths.append(key)
            updated[key] = {
                "mtime":  mtime,
                "size":   size,
                "vector": vec.tolist(),
            }
            n_computed += 1
            if verbose:
                print(f"    vectorizado: {mf.name}")
        except Exception as e:
            if verbose:
                print(f"    [ERROR] {mf.name}: {e}")

    print(f"  Corpus: {len(vectors)} MIDIs "
          f"({n_cached} en caché, {n_computed} calculados)")
    return vectors, paths, updated


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE PLAN
# ══════════════════════════════════════════════════════════════════════════════

def load_plan(plan_path: str) -> Optional[dict]:
    try:
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)
        return plan
    except Exception as e:
        print(f"  [AVISO] No se pudo cargar el plan {plan_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

_EVALUATORS = {
    "formal":    FormalEvaluator(),
    "coherence": CoherenceEvaluator(),
    "arc":       ArcEvaluator(),
    "melodic":   MelodicEvaluator(),
    "corpus":    CorpusEvaluator(),
}


def evaluate_midi(midi_path: str,
                  ctx: EvalContext,
                  weights: Dict[str, float],
                  active_dims: List[str]) -> QualityReport:
    """Pipeline completo de evaluación para un único MIDI."""

    mid   = MidiFile(midi_path)
    notes = _extract_notes(mid)
    topo  = detect_topology(mid)

    # Metadatos básicos
    tempo_us  = _get_tempo(mid)
    tempo_bpm = round(60_000_000 / tempo_us, 1)
    bar_ticks = _estimate_bar_ticks(mid)
    total     = _get_total_ticks(mid)
    n_bars    = max(1, total // bar_ticks)
    key_root, key_mode = _detect_key_simple(notes)
    key_str   = f"{PITCH_NAMES[key_root]} {key_mode}"

    meta = {
        "duration_bars":        n_bars,
        "tempo_bpm":            tempo_bpm,
        "key":                  key_str,
        "n_tracks":             len(topo.all_tracks),
        "topology":             topo.type,
        "fast_mode":            ctx.fast,
        "arc_weight_mode":      ctx.arc_weight_mode,
        "quality_scorer_version": VERSION,
    }

    # Ejecutar evaluadores activos
    results: Dict[str, DimensionResult] = {}
    for dim in active_dims:
        evaluator = _EVALUATORS.get(dim)
        if evaluator is None:
            continue
        try:
            results[dim] = evaluator.evaluate(notes, mid, topo, ctx)
        except Exception as e:
            results[dim] = DimensionResult(
                score=0.5, confidence=0.0,
                reasons=[f"Error en evaluación: {e}"],
                details={"error": str(e)},
            )

    # Score compuesto
    abs_score, eff_weights = composite_score(results, weights)
    label = absolute_label(abs_score)

    # Percentil: preferir el del CorpusEvaluator (más preciso, kNN real)
    # si no hay corpus, usar relativize() sobre distancia al centroide
    percentile: Optional[float] = None
    query_vec = _descriptor_13(notes, mid)
    if "corpus" in results and results["corpus"].confidence > 0:
        percentile = results["corpus"].details.get("percentile")
    if percentile is None and ctx.corpus_vectors:
        percentile = relativize(abs_score, ctx.corpus_vectors, query_vec)

    summary = ScoreSummary(
        absolute       = abs_score,
        absolute_label = label,
        percentile     = percentile,
        corpus_size    = len(ctx.corpus_vectors) if ctx.corpus_vectors else None,
        weights_used   = eff_weights,
    )

    return QualityReport(
        file        = str(Path(midi_path).name),
        mode        = "fast" if ctx.fast else "full",
        plan_used   = ctx.plan is not None,
        corpus_used = ctx.corpus_vectors is not None,
        score       = summary,
        dimensions  = results,
        meta        = meta,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER TERMINAL
# ══════════════════════════════════════════════════════════════════════════════


def _sparkline(values: list, width: int = 16) -> str:
    """Genera un sparkline Unicode de la curva de tensión."""
    BLOCKS = " ▁▂▃▄▅▆▇█"
    if not values:
        return "─" * width
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    chars = []
    for v in values:
        idx = int((v - mn) / rng * (len(BLOCKS) - 1))
        chars.append(BLOCKS[idx])
    return "".join(chars)


def _bars_to_range(bars: list) -> str:
    """Convierte lista de compases a rangos compactos: [1,2,3,5,6] → '1–3, 5–6'."""
    if not bars:
        return ""
    bars = sorted(set(bars))
    ranges, start, prev = [], bars[0], bars[0]
    for b in bars[1:]:
        if b == prev + 1:
            prev = b
        else:
            ranges.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = b
    ranges.append(f"{start}" if start == prev else f"{start}–{prev}")
    return ", ".join(ranges)


def _render_localization(res: DimensionResult, indent: int = 18):
    """
    Imprime información de localización temporal en --verbose:
      · formal voiced: compases de cada tipo de violación
      · formal basic:  compases con disonancia / saltos
      · arc:           sparkline de tensión + posición del clímax
      · coherence:     rango de compases de segmentos anómalos
      · melodic:       ventanas con problemas de contorno/entropía
    """
    G  = _c("gray")
    R  = _c("reset")
    Y  = _c("yellow")
    pad = " " * indent

    d = res.details

    # ── formal voiced ────────────────────────────────────────────────────────
    if d.get("level") == "voiced" and d.get("violations"):
        by_bar: dict = {}
        for v in d["violations"]:
            rule = v["rule"]
            bar  = v.get("bar", "?")
            by_bar.setdefault(rule, []).append(bar)
        rule_labels = {
            "parallel_fifths":         "paralelas 5ª",
            "parallel_octaves":        "paralelas 8ª",
            "voice_crossing":          "cruce de voces",
            "large_leap":              "salto grande",
            "leading_tone_unresolved": "sensible sin resolver",
        }
        for rule, bars in sorted(by_bar.items(), key=lambda x: -len(x[1])):
            label   = rule_labels.get(rule, rule)
            bar_str = _bars_to_range(bars)
            print(f"{pad}{G}⚑ {label}: compás/es {bar_str}{R}")

    # ── formal basic ─────────────────────────────────────────────────────────
    if d.get("level") == "basic":
        if d.get("dissonant_bars"):
            bar_str = _bars_to_range(d["dissonant_bars"])
            print(f"{pad}{G}⚑ alta disonancia: compás/es {bar_str}{R}")
        if d.get("leap_bars"):
            bar_str = _bars_to_range(d["leap_bars"])
            print(f"{pad}{G}⚑ saltos grandes: compás/es {bar_str}{R}")
        if d.get("oos_bars"):
            bar_str = _bars_to_range(d["oos_bars"])
            print(f"{pad}{G}⚑ notas fuera de escala: compás/es {bar_str}{R}")

    # ── arc: sparkline ───────────────────────────────────────────────────────
    if d.get("tension_curve"):
        curve   = d["tension_curve"]
        spark   = _sparkline(curve)
        climax  = d.get("climax_position_real")
        n       = len(curve)
        # Posición del clímax en el sparkline
        if climax is not None:
            climax_idx = int(climax * (n - 1))
            spark_list = list(spark)
            # marcar con color si el terminal lo soporta
            marker = f"{Y}^{G}"
            marker_line = " " * climax_idx + "^"
        else:
            marker_line = ""
        print(f"{pad}{G}tensión: {spark}{R}")
        if marker_line:
            print(f"{pad}{G}         {marker_line}  clímax en {climax:.0%}{R}")

    # ── coherence: segmentos anómalos → rango de compases ───────────────────
    if d.get("anomalous_segments") and d.get("n_segments"):
        n_segs = d["n_segments"]
        # Convertir índice de segmento a compás aproximado
        # (no tenemos bar_ticks aquí, usamos fracción de la pieza)
        segs   = d["anomalous_segments"]
        # Mostrar como fracción de la pieza
        fracs  = [f"{int(s/n_segs*100)}–{int((s+1)/n_segs*100)}%" for s in segs]
        print(f"{pad}{G}⚑ segmentos anómalos: {', '.join(fracs)} de la pieza{R}")

    # ── melodic: ventanas problemáticas ──────────────────────────────────────
    if d.get("problem_windows"):
        for pw in d["problem_windows"]:
            kind    = pw.get("kind", "")
            bar_str = _bars_to_range(pw.get("bars", []))
            print(f"{pad}{G}⚑ {kind}: compás/es {bar_str}{R}")


def _score_color(score: float) -> str:
    if score >= 0.75:
        return _c("green")
    if score >= 0.55:
        return _c("yellow")
    return _c("red")


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"{_score_color(score)}{bar}{_c('reset')}"


def render_terminal(report: QualityReport, verbose: bool = False):
    B, R, C, G, Y = _c("bold"), _c("reset"), _c("cyan"), _c("green"), _c("yellow")
    print(f"\n{'═' * 64}")
    print(f"  {B}QUALITY SCORER  ·  {report.file}{R}")
    print(f"{'═' * 64}")

    # Metadatos
    m = report.meta
    print(f"  Tonalidad : {m['key']}   "
          f"Tempo: {m['tempo_bpm']} BPM   "
          f"Compases: {m['duration_bars']}   "
          f"Tracks: {m['n_tracks']} ({m['topology']})")
    if report.plan_used:
        print(f"  Plan      : {G}sí{R}")
    if report.corpus_used:
        print(f"  Corpus    : {G}{report.score.corpus_size} MIDIs{R}")
    print()

    # Score global
    s  = report.score
    sc = s.absolute
    print(f"  {B}SCORE GLOBAL{R}")
    print(f"  {_score_bar(sc)}  "
          f"{_score_color(sc)}{B}{sc:.3f}{R}  [{s.absolute_label}]")
    if s.percentile is not None:
        print(f"  Percentil en corpus: {s.percentile:.1f}%")
    print()

    # Por dimensión
    print(f"  {B}DIMENSIONES{R}")
    for dim, res in report.dimensions.items():
        if res.confidence == 0.0 and res.details.get("status") in ("stub", "no_corpus"):
            status = f"{_c('gray')}n/d{R}"
            bar    = _c("gray") + "░" * 20 + R
            print(f"  {'  ' + dim:<14} {bar}  {status}")
        else:
            bar = _score_bar(res.score)
            conf_str = f"(conf {res.confidence:.2f})" if res.confidence < 0.8 else ""
            print(f"  {'  ' + dim:<14} {bar}  "
                  f"{_score_color(res.score)}{res.score:.3f}{R}  {_c('gray')}{conf_str}{R}")
            if verbose and res.reasons:
                for r in res.reasons:
                    print(f"  {'':>16}  {_c('gray')}· {r}{R}")
            if verbose:
                _render_localization(res, indent=18)

    print(f"\n{'═' * 64}\n")


def render_rank_terminal(reports: List[QualityReport], verbose: bool = False):
    B, R = _c("bold"), _c("reset")
    reports_sorted = sorted(reports, key=lambda rp: rp.score.absolute, reverse=True)
    print(f"\n{'═' * 64}")
    print(f"  {B}QUALITY SCORER  ·  RANKING ({len(reports)} MIDIs){R}")
    print(f"{'═' * 64}")
    for i, rp in enumerate(reports_sorted, 1):
        bar = _score_bar(rp.score.absolute, width=16)
        print(f"  {i:>2}. {bar}  "
              f"{_score_color(rp.score.absolute)}{rp.score.absolute:.3f}{R}  "
              f"{rp.file}")
    print(f"\n{'═' * 64}\n")




# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO DE EXPLICACIÓN (--explain)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Modo A (local):  plantillas de texto en lenguaje natural, sin dependencias.
#  Modo B (llm):    envía el informe a un LLM para consejo accionable.
#    · --explain-provider anthropic   usa ANTHROPIC_API_KEY / --api-key
#    · --explain-provider openai      usa OPENAI_API_KEY / --api-key
#    · --explain-provider clipboard   copia el prompt, espera que pegues la resp.
#
#  En todos los casos el resultado se imprime en terminal y,
#  opcionalmente, se guarda con --explain-out FILE.
# ══════════════════════════════════════════════════════════════════════════════

# ── Textos de plantilla (modo A) ──────────────────────────────────────────────

_EXPLAIN_TEMPLATES: Dict[str, Dict] = {
    "formal": {
        "label": "Corrección melódica y armónica",
        "high":  "Las notas y acordes encajan bien entre sí. La melodía se mueve "
                 "de forma natural sin saltos bruscos.",
        "mid":   "Hay algunos momentos donde la melodía da saltos demasiado grandes "
                 "o las notas chirrían con los acordes. No es grave, pero se nota.",
        "low":   "La melodía tiene saltos muy amplios que suenan forzados, o hay "
                 "notas que no encajan con la armonía. Es el área con más margen de mejora.",
        "tip_high":  None,
        "tip_mid":   "Intenta que la melodía se mueva principalmente por pasos "
                     "cercanos (notas vecinas). Los saltos grandes funcionan mejor "
                     "usados con moderación, en momentos de énfasis.",
        "tip_low":   "Revisa los momentos donde la melodía salta más de una octava. "
                     "Casi siempre hay una nota intermedia que haría la transición más suave. "
                     "También comprueba que las notas principales de cada compás pertenecen al acorde.",
        "violations": {
            "parallel_fifths":         "acordes que suenan vacíos por moverse en paralelo",
            "parallel_octaves":        "voces que se duplican en paralelo (pierde riqueza)",
            "voice_crossing":          "voces que se cruzan (la grave suena más aguda que la aguda)",
            "large_leap":              "saltos melódicos demasiado grandes",
            "leading_tone_unresolved": "nota de tensión que no resuelve hacia donde espera el oído",
        },
    },
    "coherence": {
        "label": "Consistencia interna",
        "high":  "La pieza suena unificada de principio a fin. El estilo y el carácter "
                 "se mantienen consistentes.",
        "mid":   "La pieza es mayormente consistente, aunque hay algún momento que suena "
                 "diferente al resto, quizá por cambio brusco de textura o densidad.",
        "low":   "Hay secciones que suenan como si pertenecieran a piezas distintas. "
                 "El oyente puede sentir que algo 'desentona'.",
        "tip_high":  None,
        "tip_mid":   "Escucha la pieza entera y localiza el momento donde sientes que "
                     "'algo cambia'. Ese es el segmento anómalo. Puedes suavizarlo "
                     "reduciendo el contraste con la sección anterior.",
        "tip_low":   "La pieza necesita hilo conductor. Considera usar un motivo melódico "
                     "recurrente (una pequeña frase que se repite y varía), o mantener "
                     "el mismo tempo y rango dinámico en toda la obra.",
    },
    "arc": {
        "label": "Arco emocional",
        "high":  "La pieza tiene una forma emocional clara: construye tensión, "
                 "llega a un punto álgido y resuelve. El oyente siente un viaje.",
        "mid":   "Hay algo de movimiento emocional, pero el arco podría ser más pronunciado. "
                 "La pieza sube y baja, pero sin un clímax claro.",
        "low":   "La pieza mantiene el mismo nivel de intensidad de principio a fin. "
                 "Sin cambios de tensión, el oyente puede perder el hilo emocional.",
        "static": "La pieza mantiene un nivel de intensidad constante de forma intencionada "
                  "(forma estrófica, ostinato, himno). Esto puede ser una elección válida.",
        "tip_high":  None,
        "tip_mid":   "Elige un momento para el clímax (normalmente entre el 60% y el 75% "
                     "de la pieza) y hazlo más intenso: más notas por tiempo, registro más "
                     "agudo, o dinámica más fuerte. Luego reduce gradualmente hacia el final.",
        "tip_low":   "Prueba este esquema básico: empieza suave, sube la intensidad poco a poco "
                     "hasta los 3/4 de la pieza, y luego resuelve con calma. Incluso pequeñas "
                     "variaciones de velocidad y volumen crean mucha diferencia.",
    },
    "melodic": {
        "label": "Interés melódico",
        "high":  "La melodía es variada e interesante: mezcla repetición y novedad, "
                 "sube y baja con naturalidad, y tiene personalidad propia.",
        "mid":   "La melodía funciona pero podría tener más personalidad. "
                 "Quizá es algo predecible o le falta variedad.",
        "low":   "La melodía es difícil de recordar o seguir. Puede ser demasiado "
                 "repetitiva, demasiado aleatoria, o carecer de forma clara.",
        "tip_high":  None,
        "tip_mid":   "Toma una frase corta que ya tengas (3-5 notas) y úsala como "
                     "motivo: repítela, transpónla a otro tono, inviértela, o cámbiala "
                     "ligeramente cada vez. Las melodías memorables suelen basarse en "
                     "pocos materiales bien desarrollados.",
        "tip_low":   "Si la melodía suena aleatoria, prueba a limitarte a las notas "
                     "de la escala y movirte principalmente por pasos. Si suena "
                     "repetitiva, añade algún salto en los puntos de énfasis y "
                     "varía el ritmo aunque las notas sean similares.",
        "submetrics": {
            "interval_entropy": ("intervalos muy repetitivos",
                                 "buen equilibrio de intervalos",
                                 "intervalos demasiado aleatorios"),
            "inflection_rate":  ("melodía demasiado recta (sin subidas y bajadas)",
                                 "buen contorno melódico",
                                 "melodía en zigzag constante"),
            "markov_entropy":   ("melodía muy predecible",
                                 "buena variedad de transiciones",
                                 "transiciones melódicas sin patrón"),
            "repetition_ratio": ("sin motivos reconocibles",
                                 "buen equilibrio repetición/variedad",
                                 "demasiado repetitiva"),
        },
    },
    "corpus": {
        "label": "Afinidad estilística",
        "high":  "La pieza encaja bien con el estilo de tu colección de referencia.",
        "mid":   "La pieza tiene similitudes con tu colección pero también diferencias notables.",
        "low":   "La pieza suena bastante diferente a tu colección de referencia. "
                 "Puede ser intencional (estás explorando un estilo nuevo) o involuntario.",
        "tip_high":  None,
        "tip_mid":   "Escucha las piezas más cercanas de tu corpus y fíjate en qué "
                     "elementos comparten con la tuya y cuáles difieren.",
        "tip_low":   "Si quieres acercarte más a tu corpus de referencia, analiza "
                     "el tempo, la densidad de notas y el registro de las piezas más "
                     "similares e intenta adoptar esos parámetros.",
    },
}

_OVERALL_LABELS = {
    "muy alto": ("Resultado excelente",
                 "Esta pieza está muy bien construida. Los aspectos técnicos "
                 "y musicales funcionan de forma conjunta."),
    "alto":     ("Buen resultado",
                 "La pieza funciona bien en general. Hay algunos aspectos "
                 "a pulir pero la base es sólida."),
    "medio":    ("Resultado correcto",
                 "La pieza tiene potencial pero necesita trabajo en algunas áreas "
                 "concretas para alcanzar su mejor versión."),
    "bajo":     ("Hay margen de mejora",
                 "La pieza necesita revisión en varias áreas. Las sugerencias "
                 "a continuación te indicarán por dónde empezar."),
}


def _score_band(score: float) -> str:
    if score >= 0.75: return "high"
    if score >= 0.50: return "mid"
    return "low"


def _explain_local(report: QualityReport) -> str:
    """
    Genera una explicación en lenguaje natural usando plantillas locales.
    No requiere conexión ni API key.
    """
    lines = []
    W = 64

    label   = report.score.absolute_label
    heading, summary = _OVERALL_LABELS.get(label, ("Resultado", ""))
    score   = report.score.absolute

    lines.append("═" * W)
    lines.append(f"  GUÍA DE MEJORA  ·  {report.file}")
    lines.append("═" * W)
    lines.append(f"\n  Puntuación global: {score:.2f}  —  {heading}")
    lines.append(f"\n  {summary}")

    if report.score.percentile is not None:
        pct = report.score.percentile
        if pct >= 75:
            lines.append(f"  Tu pieza está entre el {pct:.0f}% más cercano a tu corpus de referencia.")
        elif pct >= 40:
            lines.append(f"  Tu pieza tiene similitud media con tu corpus ({pct:.0f}º percentil).")
        else:
            lines.append(f"  Tu pieza difiere bastante de tu corpus ({pct:.0f}º percentil).")

    # Identificar las 2 dimensiones más problemáticas (por score × confidence)
    active_dims = {k: v for k, v in report.dimensions.items()
                   if v.confidence > 0.0 and v.details.get("status") not in ("stub", "no_corpus", "disabled")}

    if not active_dims:
        lines.append("\n  (No hay dimensiones con datos suficientes para generar recomendaciones.)")
        lines.append("\n" + "═" * W)
        return "\n".join(lines)

    sorted_dims = sorted(active_dims.items(), key=lambda x: x[1].score * x[1].confidence)
    worst       = [d for d, r in sorted_dims if r.score < 0.70][:2]

    lines.append("\n")
    lines.append("  ─" * 32)
    lines.append("  ANÁLISIS POR ÁREA")
    lines.append("  ─" * 32)

    for dim, res in sorted(active_dims.items(),
                            key=lambda x: x[1].score * x[1].confidence):
        tmpl  = _EXPLAIN_TEMPLATES.get(dim)
        if tmpl is None:
            continue
        band  = _score_band(res.score)

        # Caso especial: arc estático
        if dim == "arc" and any("estático" in r for r in res.reasons):
            band = "static"

        label_dim = tmpl["label"]
        bar_w     = int(res.score * 16)
        bar       = "█" * bar_w + "░" * (16 - bar_w)
        text      = tmpl.get(band) or tmpl.get("mid", "")
        tip       = tmpl.get(f"tip_{band}")

        lines.append(f"\n  {label_dim.upper()}")
        lines.append(f"  [{bar}] {res.score:.2f}")
        lines.append(f"  {text}")

        # Detalles de violaciones (formal)
        if dim == "formal" and res.details.get("by_rule"):
            for rule, count in sorted(res.details["by_rule"].items(),
                                      key=lambda x: -x[1]):
                desc = tmpl["violations"].get(rule, rule)
                lines.append(f"    · {count}x {desc}")

        # Submétricas (melodic)
        if dim == "melodic" and res.confidence > 0.3:
            subs = tmpl.get("submetrics", {})
            details = res.details
            for key, (lo, mid_, hi) in subs.items():
                val = details.get(key)
                if val is None:
                    continue
                # Cada submétrica tiene su propio óptimo gaussiano centrado en ~0.5
                # band: si score × confianza del submétrico está lejos del centro
                if val < 0.35:
                    msg = lo
                elif val > 0.70:
                    msg = hi
                else:
                    continue   # en rango bueno, no mencionar
                lines.append(f"    · {msg}")

        # Nearest corpus
        if dim == "corpus" and res.details.get("nearest_files"):
            nearest = res.details["nearest_files"][:2]
            lines.append(f"    Piezas más similares en tu corpus: {', '.join(nearest)}")

        if tip:
            lines.append(f"\n¿Cómo mejorar?")
            for part in textwrap.wrap(tip, width=60):
                lines.append(f"  {part}")

    # Resumen priorizado
    if worst:
        lines.append("\n")
        lines.append("  ─" * 32)
        lines.append("  POR DÓNDE EMPEZAR")
        lines.append("  ─" * 32)
        for i, dim in enumerate(worst, 1):
            tmpl  = _EXPLAIN_TEMPLATES.get(dim, {})
            tip   = tmpl.get("tip_low") or tmpl.get("tip_mid", "")
            label_dim = tmpl.get("label", dim)
            lines.append(f"\n{i}. {label_dim}")
            if tip:
                for part in textwrap.wrap(tip, width=60):
                    lines.append(f"     {part}")

    lines.append("\n" + "═" * W)
    return "\n".join(lines)


# ── Prompt para LLM ───────────────────────────────────────────────────────────

def _build_llm_prompt(report: QualityReport) -> str:
    """
    Construye el prompt que se envía al LLM.
    Serializa el informe técnico en formato legible y añade instrucciones
    para que el LLM genere una guía de mejora en lenguaje natural.
    """
    d = report.to_dict()
    dims_text = []
    for dim, data in d["dimensions"].items():
        if data["confidence"] == 0.0:
            continue
        reasons_str = "; ".join(data.get("reasons", [])) or "sin observaciones"
        # Incluir detalles relevantes según dimensión
        extra = ""
        if dim == "formal" and data.get("by_rule"):
            extra = f" Violaciones: {data['by_rule']}."
        if dim == "corpus" and data.get("nearest_files"):
            extra = f" Piezas más cercanas en el corpus: {data['nearest_files'][:3]}."
        if dim == "melodic":
            subs = {k: data.get(k) for k in
                    ["interval_entropy_norm","inflection_rate",
                     "markov_entropy_norm","repetition_ratio"]
                    if data.get(k) is not None}
            if subs:
                extra = f" Submétricas: {subs}."
        dims_text.append(
            f"- {dim}: score={data['score']:.3f}, confianza={data['confidence']:.2f}. "
            f"{reasons_str}.{extra}"
        )

    meta   = d["meta"]
    score  = d["score"]
    report_str = (
        "Fichero: " + d["file"] + "\n"
        + "Tonalidad detectada: " + meta["key"]
        + ", tempo: " + str(meta["tempo_bpm"]) + " BPM"
        + ", compases: " + str(meta["duration_bars"])
        + ", tracks: " + str(meta["n_tracks"])
        + " (" + meta["topology"] + ")\n"
        + "Modo arc: " + meta.get("arc_weight_mode", "dynamic") + "\n"
        + "Plan de composición disponible: " + ("sí" if d["plan_used"] else "no") + "\n"
        + "Corpus de referencia: " + (
            "sí (" + str(score["corpus_size"]) + " MIDIs)"
            if d["corpus_used"] else "no"
        ) + "\n\n"
        + "SCORE GLOBAL: " + f"{score['absolute']:.3f}"
        + " [" + score["absolute_label"] + "]\n"
        + (f"Percentil en corpus: {score['percentile']:.1f}%\n"
           if score["percentile"] is not None else "")
        + "Pesos efectivos usados: " + str(score["weights_used"]) + "\n\n"
        + "DIMENSIONES:\n"
        + "\n".join(dims_text)
    )

    prompt = (
        "Eres un profesor de composición musical amable y claro. "
        "Tu alumno acaba de analizar una pieza MIDI con una herramienta automática "
        "y te muestra el informe técnico. El alumno NO tiene formación musical formal "
        "y necesita consejos prácticos y accionables en lenguaje cotidiano, "
        "sin tecnicismos innecesarios. Evita términos como \'voice leading\', "
        "\'paralelas de quinta\', \'sensible\' o \'arco emocional\' sin explicarlos. "
        "Usa analogías simples cuando sea útil.\n\n"
        "INFORME TÉCNICO:\n"
        + report_str
        + "\n\nPor favor genera:\n"
        "1. Un párrafo de resumen global (2-3 frases) que describa cómo suena la pieza "
        "y qué funciona bien.\n"
        "2. Las 2-3 áreas más importantes a mejorar, explicadas en lenguaje llano con "
        "una sugerencia concreta y accionable para cada una.\n"
        "3. Un consejo de motivación final (1 frase).\n\n"
        "Responde en español. Sé específico, positivo y constructivo."
    )
    return prompt


# ── Modo clipboard ────────────────────────────────────────────────────────────

def _explain_clipboard(report: QualityReport,
                        out_path: Optional[str] = None) -> str:
    """
    Imprime el prompt en terminal para que el usuario lo copie,
    espera que pegue la respuesta del LLM, y devuelve esa respuesta.
    """
    prompt = _build_llm_prompt(report)
    sep = "─" * 64

    print(f"\n{sep}")
    print("  MODO CLIPBOARD — copia el texto siguiente y pégalo en tu LLM")
    print(f"{sep}\n")
    print(prompt)
    print(f"\n{sep}")
    print("  Pega aquí la respuesta del LLM (termina con una línea que solo contenga END):")
    print(f"{sep}\n")

    lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass

    response = "\n".join(lines).strip()
    if not response:
        return "(sin respuesta recibida)"

    result = _format_llm_response(report.file, response)
    if out_path:
        _save_explain(result, out_path)
    return result


# ── Modo Anthropic ────────────────────────────────────────────────────────────

def _explain_anthropic(report: QualityReport,
                        api_key: Optional[str],
                        model: str,
                        out_path: Optional[str] = None) -> str:
    """Llama a la API de Anthropic (claude-* models)."""
    try:
        import anthropic as _anthropic
    except ImportError:
        return ("[ERROR] anthropic no instalado. "
                "Instálalo con: pip install anthropic")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ("[ERROR] Falta ANTHROPIC_API_KEY. "
                "Pásalo con --api-key o como variable de entorno.")

    prompt = _build_llm_prompt(report)
    print(f"  Consultando {model} … ", end="", flush=True)
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg    = client.messages.create(
            model      = model,
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}],
        )
        response = msg.content[0].text
        print("listo.")
    except Exception as e:
        print()
        return f"[ERROR] Llamada a Anthropic fallida: {e}"

    result = _format_llm_response(report.file, response)
    if out_path:
        _save_explain(result, out_path)
    return result


# ── Modo OpenAI ───────────────────────────────────────────────────────────────

def _explain_openai(report: QualityReport,
                    api_key: Optional[str],
                    model: str,
                    out_path: Optional[str] = None) -> str:
    """Llama a la API de OpenAI (gpt-* o o-* models)."""
    try:
        import openai as _openai
    except ImportError:
        return ("[ERROR] openai no instalado. "
                "Instálalo con: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ("[ERROR] Falta OPENAI_API_KEY. "
                "Pásalo con --api-key o como variable de entorno.")

    prompt = _build_llm_prompt(report)
    print(f"  Consultando {model} … ", end="", flush=True)
    try:
        client   = _openai.OpenAI(api_key=key)
        response_obj = client.chat.completions.create(
            model    = model,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 1024,
        )
        response = response_obj.choices[0].message.content
        print("listo.")
    except Exception as e:
        print()
        return f"[ERROR] Llamada a OpenAI fallida: {e}"

    result = _format_llm_response(report.file, response)
    if out_path:
        _save_explain(result, out_path)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_llm_response(filename: str, response: str) -> str:
    W = 64
    lines = [
        "═" * W,
        f"  GUÍA DE MEJORA (LLM)  ·  {filename}",
        "═" * W,
        "",
    ]
    for para in response.split("\n"):
        wrapped = textwrap.fill(para, width=62, initial_indent="  ",
                                subsequent_indent="  ") if para.strip() else ""
        lines.append(wrapped)
    lines.append("\n" + "═" * W)
    return "\n".join(lines)


def _save_explain(text: str, path: str):
    try:
        # Guardar sin códigos ANSI
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(clean)
        print(f"  Explicación guardada: {path}")
    except Exception as e:
        print(f"  [AVISO] No se pudo guardar: {e}")


def explain_report(report: QualityReport,
                   provider: str,
                   model: Optional[str],
                   api_key: Optional[str],
                   out_path: Optional[str]) -> str:
    """
    Punto de entrada unificado para el módulo de explicación.
    provider: "local" | "anthropic" | "openai" | "clipboard"
    """
    if provider == "local":
        result = _explain_local(report)
        if out_path:
            _save_explain(result, out_path)
        return result

    if provider == "clipboard":
        return _explain_clipboard(report, out_path=out_path)

    if provider == "anthropic":
        m = model or "claude-opus-4-5"
        return _explain_anthropic(report, api_key=api_key, model=m, out_path=out_path)

    if provider == "openai":
        m = model or "gpt-4o"
        return _explain_openai(report, api_key=api_key, model=m, out_path=out_path)

    return f"[ERROR] Proveedor desconocido: {provider}"

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluación multidimensional de calidad musical para MIDIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("midi_paths", nargs="+", metavar="MIDI",
                   help="Uno o más ficheros MIDI a evaluar.")
    p.add_argument("--plan", metavar="FILE",
                   help="theorist.json o narrator_plan.json para la dimensión arc.")
    p.add_argument("--corpus", metavar="DIR",
                   help="Directorio de MIDIs de referencia para la dimensión corpus.")
    p.add_argument("--use-cache", metavar="FILE",
                   help="Cargar caché de vectores de corpus (.qcache).")
    p.add_argument("--save-cache", metavar="FILE",
                   help="Guardar caché de vectores de corpus (.qcache).")
    p.add_argument("--mode", choices=["eval", "rank"], default="eval",
                   help="Modo de operación: eval (default) | rank.")
    p.add_argument("--fast", action="store_true",
                   help="Modo rápido: omite cálculos pesados (music21).")
    p.add_argument("--arc-weight-mode",
                   choices=["dynamic", "static", "off"],
                   default="dynamic",
                   help=(
                       "Tratamiento del arco emocional: "
                       "dynamic = comportamiento por defecto, la confianza "
                       "baja automáticamente cuando se detecta arco estático; "
                       "static = declara explícitamente que la pieza tiene "
                       "arco estático válido (ostinato, marcha, forma estrófica), "
                       "fuerza confianza≤0.40 para reducir el peso de arc; "
                       "off = excluye arc del score compuesto."
                   ))
    p.add_argument("--skip", nargs="+", metavar="DIM",
                   choices=list(_EVALUATORS.keys()),
                   help="Dimensiones a omitir.")
    p.add_argument("--only", nargs="+", metavar="DIM",
                   choices=list(_EVALUATORS.keys()),
                   help="Solo estas dimensiones.")
    p.add_argument("--weights", nargs="+", metavar="K=V",
                   help="Pesos custom, ej: formal=0.4 coherence=0.3")
    p.add_argument("--out-json", metavar="FILE",
                   help="Guardar informe completo en JSON.")
    p.add_argument("--quiet", action="store_true",
                   help="Solo imprime el score numérico (útil en scripts bash).")
    p.add_argument("--verbose", action="store_true",
                   help="Detalle completo en terminal.")
    p.add_argument("--no-color", action="store_true",
                   help="Sin colores ANSI.")

    # ── Explicación en lenguaje natural ───────────────────────────────────────
    explain_group = p.add_argument_group("explicación (--explain)")
    explain_group.add_argument("--explain", action="store_true",
                               help="Añadir explicación en lenguaje natural al informe.")
    explain_group.add_argument("--explain-provider",
                               choices=["local", "anthropic", "openai", "clipboard"],
                               default="local",
                               help=(
                                   "Proveedor para la explicación: "
                                   "local = plantillas offline (default); "
                                   "anthropic = Claude vía API; "
                                   "openai = GPT vía API; "
                                   "clipboard = muestra el prompt para copiarlo a cualquier LLM."
                               ))
    explain_group.add_argument("--explain-model", metavar="MODEL", default=None,
                               help=(
                                   "Modelo a usar con anthropic u openai. "
                                   "Default: claude-opus-4-5 (anthropic) / gpt-4o (openai)."
                               ))
    explain_group.add_argument("--api-key", metavar="KEY", default=None,
                               help="API key para anthropic u openai "
                                    "(alternativa a ANTHROPIC_API_KEY / OPENAI_API_KEY).")
    explain_group.add_argument("--explain-out", metavar="FILE", default=None,
                               help="Guardar la explicación en un fichero de texto.")
    return p


def parse_weights(raw: Optional[List[str]]) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not raw:
        return weights
    for token in raw:
        if "=" not in token:
            print(f"  [AVISO] Peso ignorado (formato inválido): {token}")
            continue
        k, v = token.split("=", 1)
        k = k.strip().lower()
        if k not in _EVALUATORS:
            print(f"  [AVISO] Dimensión desconocida en --weights: {k}")
            continue
        try:
            weights[k] = float(v)
        except ValueError:
            print(f"  [AVISO] Valor inválido en --weights: {token}")
    return weights


def main():
    global _NO_COLOR

    parser = build_parser()
    args   = parser.parse_args()

    _NO_COLOR = args.no_color

    # ── Dimensiones activas ───────────────────────────────────────────────────
    all_dims = list(_EVALUATORS.keys())
    if args.only:
        active_dims = [d for d in all_dims if d in args.only]
    elif args.skip:
        active_dims = [d for d in all_dims if d not in args.skip]
    else:
        active_dims = all_dims

    # ── Pesos ─────────────────────────────────────────────────────────────────
    weights = parse_weights(args.weights)

    # ── Plan ──────────────────────────────────────────────────────────────────
    plan = None
    if args.plan:
        plan = load_plan(args.plan)
        if not args.quiet:
            print(f"  Plan cargado: {args.plan}")

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_vectors = None
    corpus_paths   = None
    corpus_entries = {}
    if args.corpus:
        cache = {}
        if args.use_cache:
            cache = load_corpus_cache(args.use_cache)
        corpus_vectors, corpus_paths, corpus_entries = vectorize_corpus(
            args.corpus, cache=cache, verbose=args.verbose)
        if args.save_cache:
            save_corpus_cache(args.save_cache, corpus_entries)

    # ── Contexto ──────────────────────────────────────────────────────────────
    ctx = EvalContext(
        midi_path        = "",        # se actualiza por fichero
        plan             = plan,
        corpus_vectors   = corpus_vectors if corpus_vectors else None,
        corpus_paths     = corpus_paths   if corpus_paths   else None,
        fast             = args.fast,
        verbose          = args.verbose,
        arc_weight_mode  = args.arc_weight_mode,
    )

    # ── Validar ficheros de entrada ───────────────────────────────────────────
    midi_paths = []
    for p in args.midi_paths:
        if not Path(p).exists():
            print(f"[ERROR] Fichero no encontrado: {p}", file=sys.stderr)
            continue
        midi_paths.append(p)

    if not midi_paths:
        print("[ERROR] No hay ficheros MIDI válidos.", file=sys.stderr)
        sys.exit(1)

    # ── Evaluación ────────────────────────────────────────────────────────────
    reports: List[QualityReport] = []
    for mp in midi_paths:
        ctx.midi_path = mp
        if not args.quiet:
            if len(midi_paths) > 1:
                print(f"\n  Evaluando: {Path(mp).name} …")
        try:
            rp = evaluate_midi(mp, ctx, weights, active_dims)
            reports.append(rp)
        except Exception as e:
            print(f"[ERROR] {mp}: {e}", file=sys.stderr)
            if args.verbose:
                import traceback; traceback.print_exc()

    if not reports:
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.quiet:
        # Solo stdout limpio — útil para scripts
        for rp in reports:
            if len(reports) == 1:
                print(f"{rp.score.absolute:.4f}")
            else:
                print(f"{rp.score.absolute:.4f}  {rp.file}")
        sys.exit(0)

    # Render terminal
    if args.mode == "rank" and len(reports) > 1:
        render_rank_terminal(reports, verbose=args.verbose)
        # También informe individual si verbose
        if args.verbose:
            for rp in reports:
                render_terminal(rp, verbose=True)
    else:
        for rp in reports:
            render_terminal(rp, verbose=args.verbose)

    # JSON
    if args.out_json:
        if len(reports) == 1:
            output = reports[0].to_dict()
        else:
            output = [rp.to_dict() for rp in reports]
        try:
            with open(args.out_json, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"  Informe guardado: {args.out_json}")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar JSON: {e}", file=sys.stderr)

    # ── Explicación en lenguaje natural ───────────────────────────────────────
    if args.explain:
        for rp in reports:
            # En modo rank con varios ficheros: explicar cada uno
            explanation = explain_report(
                report   = rp,
                provider = args.explain_provider,
                model    = args.explain_model,
                api_key  = args.api_key,
                out_path = args.explain_out,
            )
            print(explanation)


if __name__ == "__main__":
    main()
