#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       QUALITY SCORER  v1.3                                  ║
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
║  ROLES Y CONTEXTO (v1.3):                                                    ║
║  python quality_scorer.py obra.mid --roles '1=melody,ch10=drums'             ║
║  python quality_scorer.py obra.mid --scale whole_tone   # colección          ║
║    python quality_scorer.py obra.mid --context obra.whole_tone_report.json   ║
║  python quality_scorer.py obra.mid --voices satb        # forzar SATB        ║
║  python quality_scorer.py obra.mid --compat             # como v1.2          ║
║    La percusión (canal 10) se excluye de formal/melodic/harmony; los nombres ║
║    de pista (Melody, Bass, Harmony, Drums…) son evidencia de rol si no       ║
║    contradicen canal/programa. El informe lista roles, fuente y avisos.      ║
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
║    --roles MAP              N=rol|chN=rol (melody|bass|harmony|drums)        ║
║    --scale [RAÍZ:]NOMBRE    Colección declarada (whole_tone, D:dorian…)      ║
║    --context JSON           Informe de theme_transformer (un fichero)        ║
║    --voices auto|satb|texture  Reglas de voces (auto: solo si declaradas)    ║
║    --compat                 Reproduce v1.2                                   ║
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

VERSION = "1.3"

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Pesos por defecto para el score compuesto
DEFAULT_WEIGHTS: Dict[str, float] = {
    "formal":    0.20,
    "coherence": 0.13,
    "arc":       0.22,
    "melodic":   0.18,
    "harmony":   0.20,   # NUEVO (P3): dimensión armónica explícita
    "corpus":    0.07,
}

# Umbrales absolutos heurísticos (recalibrados en P5 a la nueva composición
# del score tras añadir `harmony` y corregir `formal`/`coherence`)
ABSOLUTE_THRESHOLDS = {
    "bajo":    0.47,
    "medio":   0.58,
    "alto":    0.66,
    "muy_alto": 1.01,
}

# Intervalos disonantes (en semitonos, dentro de una octava)
DISSONANT_INTERVALS = {1, 2, 6, 10, 11}   # m2, M2, tritono, m7, M7
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9}  # unísono, 3as, 4as, 5as, 6as

def _is_tertian_sonority(pitches) -> bool:
    """True si las clases de altura forman una tríada terciana (con o sin 7ª).

    Evita marcar como disonancia un acorde de séptima normal (p. ej. un V7
    contiene un tritono y una 7ª menor) cuando los intervalos son
    constitutivos del acorde, no fricción contrapuntística. (P1)
    """
    if len(pitches) < 3:
        return False
    pcs = {p % 12 for p in pitches}
    allowed = {0, 3, 4, 6, 7, 8, 9, 10, 11}   # 3ª, 5ª y 7ª diatónicas
    for root in range(12):
        stack = {(root + i) % 12 for i in allowed}
        if not pcs.issubset(stack):
            continue
        rel = {(p - root) % 12 for p in pcs}
        if (rel & {3, 4}) and (rel & {6, 7, 8}):   # tiene 3ª y 5ª
            return True
    return False

# Colecciones declarables (--scale / --context). Misma tabla que theme_transformer.
DECLARABLE_SCALES = {
    'major':             (0, 2, 4, 5, 7, 9, 11),
    'natural_minor':     (0, 2, 3, 5, 7, 8, 10),
    'harmonic_minor':    (0, 2, 3, 5, 7, 8, 11),
    'melodic_minor':     (0, 2, 3, 5, 7, 9, 11),
    'dorian':            (0, 2, 3, 5, 7, 9, 10),
    'phrygian':          (0, 1, 3, 5, 7, 8, 10),
    'lydian':            (0, 2, 4, 6, 7, 9, 11),
    'mixolydian':        (0, 2, 4, 5, 7, 9, 10),
    'locrian':           (0, 1, 3, 5, 6, 8, 10),
    'acoustic':          (0, 2, 4, 6, 7, 9, 10),
    'melodic_minor_m5':  (0, 2, 4, 5, 7, 8, 10),
    'double_harmonic':   (0, 1, 4, 5, 7, 8, 11),
    'whole_tone':        (0, 2, 4, 6, 8, 10),
    'octatonic_wh':      (0, 2, 3, 5, 6, 8, 9, 11),
    'octatonic_hw':      (0, 1, 3, 4, 6, 7, 9, 10),
    'major_pentatonic':  (0, 2, 4, 7, 9),
    'minor_pentatonic':  (0, 3, 5, 7, 10),
    'blues':             (0, 3, 5, 6, 7, 10),
}
# Colecciones para las que la lectura funcional T/S/D + cadencia V→I tiene sentido
_FUNCTIONAL_SCALES = {'major', 'natural_minor', 'harmonic_minor', 'melodic_minor'}

# Evidencia de rol por nombre de pista (v1.3). Orden = prioridad.
_NAME_ROLES = [
    ('drums',   ('drum', 'perc', 'bater', 'kit')),
    ('melody',  ('melod', 'melody', 'lead', 'vocal', 'voz', 'tune')),
    ('bass',    ('bass', 'bajo')),
    ('harmony', ('harmon', 'armon', 'chord', 'acord', 'accomp', 'acomp')),
]
_VOICE_NAMES = ('soprano', 'contralto', 'alto', 'tenor')
_VALID_ROLES = {'melody', 'bass', 'harmony', 'drums'}

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
    name:        str = ""      # v1.3: nombre legible (basename); `file` guarda la ruta dada

    def to_dict(self) -> dict:
        return {
            "file":        self.file,
            "name":        self.name or Path(self.file).name,
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
    # v1.3: trazabilidad de roles
    roles:          Dict[int, Dict] = field(default_factory=dict)   # ti → {role, source, name}
    warnings:       List[str]       = field(default_factory=list)
    declared_voices: bool           = False   # nombres soprano/alto/tenor… presentes


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
    # ── v1.3 ──────────────────────────────────────────────────────────────
    compat:          bool = False                # True = comportamiento v1.2
    roles:           Optional[Dict] = None       # {'track': {ti: rol}, 'channel': {ch: rol}}
    scale_name:      Optional[str] = None        # colección declarada (--scale/--context)
    scale_root:      Optional[int] = None        # tónica declarada (None = detectada)
    scale_pcs:       Optional[set] = None        # se calcula por fichero
    scale_source:    str = ""
    voices:          str = "auto"                # auto | satb | texture


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


_KS_MAJOR = {'Cb': 11, 'Gb': 6, 'Db': 1, 'Ab': 8, 'Eb': 3, 'Bb': 10, 'F': 5, 'C': 0,
             'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6, 'C#': 1}


def _signature_hint(mid) -> Optional[Tuple[int, int]]:
    """(tónica mayor, tónica menor relativa) según el primer key_signature del MIDI."""
    if mid is None:
        return None
    for trk in mid.tracks:
        for msg in trk:
            if msg.type == 'key_signature':
                k = msg.key
                if k.endswith('m'):
                    root = _KS_MAJOR.get(k[:-1])
                    return None if root is None else ((root + 3) % 12, root)
                root = _KS_MAJOR.get(k)
                return None if root is None else (root, (root + 9) % 12)
    return None


def _detect_key_simple(notes: list, mid=None) -> Tuple[int, str]:
    """Detección de tonalidad por correlación con perfiles de Krumhansl.

    Si el MIDI trae `key_signature`, se restringe a esa tonalidad y su relativa
    (mayor/menor) siempre que la correlación no quede muy por debajo de la
    mejor global: evita contradecir la armadura (p.ej. A mayor vs D mayor)
    cuando ambas son casi igual de plausibles por histograma."""
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    if not notes:
        return 0, 'major'
    # P0: histograma PONDERADO POR DURACIÓN (como harmonic_analyzer).
    # Contar notas hacía que la percusión (muchos eventos de duración corta)
    # dominara el histograma y colapsara la detección a F#M/D#m.
    counts = np.zeros(12)
    for n in notes:
        counts[n[2] % 12] += max(1, n[1] - n[0])
    counts /= (counts.sum() + 1e-9)
    scores = {}
    for root in range(12):
        for mode, prof in [('major', major_profile), ('minor', minor_profile)]:
            rotated = [prof[(i - root) % 12] for i in range(12)]
            sc = float(np.corrcoef(counts, rotated)[0, 1])
            scores[(root, mode)] = 0.0 if sc != sc else sc   # NaN (histograma plano) → 0
    best = max(scores, key=scores.get)
    hint = _signature_hint(mid)
    if hint:
        cands = [(hint[0], 'major'), (hint[1], 'minor')]
        in_sig = max(cands, key=lambda k: scores[k])
        if scores[in_sig] >= scores[best] - 0.05:
            return in_sig
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE TOPOLOGÍA
# ══════════════════════════════════════════════════════════════════════════════

def _role_from_name(name: str) -> Optional[str]:
    n = (name or '').casefold()
    for role, keys in _NAME_ROLES:
        if any(k in n for k in keys):
            return role
    return None


def detect_topology(mid: MidiFile, ctx: Optional["EvalContext"] = None) -> Topology:
    """Clasifica la estructura de tracks y asigna roles.

    Precedencia (v1.3): 1) --roles explícito; 2) nombre de pista SI es compatible
    con canal/programa; 3) heurística (batería por canal/programa, bajo por
    programa/registro, melodía = pitch medio más alto). Si nombre y canal se
    contradicen se emite un aviso y se decide por canal/programa. Con
    ctx.compat=True se reproduce el comportamiento v1.2."""
    compat = bool(ctx and ctx.compat)
    override = (ctx.roles if ctx else None) or {}
    warnings: List[str] = []
    track_info = {}

    for ti, track in enumerate(mid.tracks):
        program, channels, pitches, n_drum = 0, set(), [], 0
        for msg in track:
            if msg.type == 'program_change':
                program = msg.program
                channels.add(msg.channel)
            elif msg.type == 'note_on' and msg.velocity > 0:
                channels.add(msg.channel)
                pitches.append(msg.note)
                if msg.channel == _DRUM_CHANNEL:
                    n_drum += 1
        if compat:
            is_drum = _DRUM_CHANNEL in channels
        else:
            # pista de batería = mayoría de sus notas en el canal 10 (una pista
            # tipo 0 con voces + batería no es "batería")
            is_drum = bool(pitches) and n_drum / len(pitches) > 0.5
        track_info[ti] = {
            'program':    program,
            'channels':   channels,
            'is_drum':    is_drum,
            'n_notes':    len(pitches),
            'mean_pitch': float(np.mean(pitches)) if pitches else 0.0,
            'name':       (track.name or '').strip(),
        }

    active = [ti for ti, info in track_info.items() if info['n_notes'] > 0]
    ttype = "single" if len(active) <= 1 else "multi_unlabeled"

    voice_map, roles = {}, {}
    melody_tracks, bass_tracks, drum_tracks = [], [], []

    def _set(ti, role, source):
        voice_map[ti] = role
        roles[ti] = {'role': role, 'source': source, 'name': track_info[ti]['name'],
                     'channels': sorted(track_info[ti]['channels'])}

    def _heur_class(info):
        if info['is_drum'] or info['program'] in _DRUM_PROGRAMS:
            return 'drums'
        if info['program'] in _BASS_PROGRAMS or info['mean_pitch'] < 48:
            return 'bass'
        return 'unknown'

    for ti in active:
        info = track_info[ti]
        heur = _heur_class(info)
        explicit = None if compat else (
            override.get('track', {}).get(ti)
            or next((r for ch, r in override.get('channel', {}).items()
                     if (ch - 1) in info['channels']), None))
        named = None if compat else _role_from_name(info['name'])
        if explicit:
            _set(ti, explicit, 'roles')
        elif named:
            is_perc_like = info['is_drum'] or info['program'] in _DRUM_PROGRAMS
            compatible = (named == 'drums') == is_perc_like
            if compatible:
                _set(ti, named, 'name')
                if heur != 'unknown' and heur != named and named != 'harmony' and named != 'melody':
                    warnings.append(f"pista {ti} «{info['name']}»: el nombre dice {named} "
                                    f"pero la heurística de registro/programa dice {heur}")
            else:
                warnings.append(
                    f"pista {ti} «{info['name']}»: el nombre sugiere {named} pero "
                    f"{'es percusión (canal 10)' if is_perc_like else 'no es percusión'}; "
                    f"se decide por canal/programa ({heur if heur != 'unknown' else 'sin rol fijo'}) "
                    f"")
                if heur != 'unknown':
                    _set(ti, heur, 'heuristic')
                else:
                    voice_map[ti] = 'unknown'
        else:
            if heur != 'unknown':
                _set(ti, heur, 'heuristic')
            else:
                voice_map[ti] = 'unknown'

    # melodía: primero las pistas con rol melody ya asignado (roles/nombre);
    # si no hay, el pitch medio más alto entre las no asignadas (heurística v1.2)
    declared_mel = [ti for ti in active if voice_map.get(ti) == 'melody']
    unknowns = [ti for ti in active if voice_map.get(ti) == 'unknown']
    if declared_mel:
        melody_ti = max(declared_mel, key=lambda ti: track_info[ti]['mean_pitch'])
        for ti in declared_mel:
            if ti != melody_ti:
                voice_map[ti] = 'harmony'
                roles[ti]['role'] = 'harmony'
        heur_ti = max(unknowns, key=lambda ti: track_info[ti]['mean_pitch']) if unknowns else None
        if heur_ti is not None and track_info[heur_ti]['mean_pitch'] > track_info[melody_ti]['mean_pitch'] + 12:
            warnings.append(f"melodía = pista {melody_ti} por {roles[melody_ti]['source']}; la heurística "
                            f"de registro habría elegido la pista {heur_ti}")
        melody_tracks.append(melody_ti)
    elif unknowns:
        melody_ti = max(unknowns, key=lambda ti: track_info[ti]['mean_pitch'])
        _set(melody_ti, 'melody', 'heuristic')
        melody_tracks.append(melody_ti)
    for ti in unknowns:
        if voice_map.get(ti) == 'unknown':
            _set(ti, 'harmony', 'heuristic')

    # ninguna pista quedó como melodía (p.ej. todas llevan nombre «Accompaniment» o la
    # pista «Melody» está en el canal 10): se promociona la de pitch medio más alto
    # entre las armónicas no fijadas por --roles, avisando de que es un respaldo.
    if not melody_tracks and not compat:
        cand = [ti for ti in active
                if voice_map.get(ti) == 'harmony' and roles.get(ti, {}).get('source') != 'roles']
        if cand:
            melody_ti = max(cand, key=lambda ti: track_info[ti]['mean_pitch'])
            _set(melody_ti, 'melody', 'fallback')
            melody_tracks.append(melody_ti)
            warnings.append(f"ninguna pista identificada como melodía: se usa la pista {melody_ti} "
                            f"(pitch medio más alto) como respaldo; use --roles para fijarla")

    for ti in active:
        r = voice_map.get(ti)
        if r == 'drums':
            drum_tracks.append(ti)
        elif r == 'bass':
            bass_tracks.append(ti)

    if melody_tracks or bass_tracks:
        ttype = "multi_labeled"

    n_voice = sum(1 for ti in active
                  if any(k in track_info[ti]['name'].casefold() for k in _VOICE_NAMES))
    declared_voices = (not compat) and n_voice >= 2

    return Topology(
        type          = ttype,
        voice_map     = voice_map,
        melody_tracks = melody_tracks,
        bass_tracks   = bass_tracks,
        drum_tracks   = drum_tracks,
        all_tracks    = active,
        roles         = roles,
        warnings      = warnings,
        declared_voices = declared_voices,
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
    """Curva de tensión ESTRUCTURAL en n_windows ventanas.

    P6: cada componente (disonancia, densidad, registro) se normaliza de forma
    relativa a su propia variación a lo largo de la obra. Así una transcripción
    MIDI con velocidad plana pero con arco real de densidad/registro se mide
    por su estructura, y no se confunde con "estática". La velocidad se excluye
    (es interpretación, no estructura).
    """
    if not notes or total_ticks == 0:
        return np.zeros(n_windows)

    window_size = total_ticks / n_windows
    diss = np.zeros(n_windows)
    act  = np.zeros(n_windows)
    reg  = np.zeros(n_windows)

    for w in range(n_windows):
        t_start = w * window_size
        t_end   = (w + 1) * window_size
        seg     = [n for n in notes if n[0] < t_end and n[1] > t_start]
        if not seg:
            continue

        pitches = [n[2] for n in seg]

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

        duration = max(t_end - t_start, 1)
        diss[w] = dissonance
        # P6: densidad en escala logarítmica (no satura en piezas densas) y
        # SIN multiplicar por velocidad (la velocidad es interpretación).
        density = len(seg) / (duration / max(bar_ticks, 1))
        act[w]  = min(math.log1p(density) / math.log1p(24.0), 1.0)
        reg[w]  = min(1.0, max(0.0, (float(np.mean(pitches)) - 36) / 60.0))

    return 0.45 * diss + 0.35 * act + 0.20 * reg


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

        # P2/P9: coherencia = identidad temática + retorno + estabilidad.
        # La identidad (autosimilitud, i.e. distancia pequeña al centroide) es
        # la que caracteriza a la música tonal real; el contraste bruto no debe
        # competir con ella. Se premia, pues, identidad y reprisse.
        if consec_dists:
            variance = float(np.var(consec_dists))
            mean_cd  = float(np.mean(consec_dists))
            identity = 1.0 - min(mean_d / 0.30, 1.0)          # autosimilitud
            s_stable = 1.0 - min(variance / 0.10, 1.0)
            ret = 0.0
            if len(descriptors) >= 3:
                ret = max(0.0, 1.0 - _cosine_distance(descriptors[-1], descriptors[0]))
            s_ret = ret if ret > 0.85 else 0.5 * ret + 0.3   # retorno temático
            score = 0.55 * identity + 0.25 * s_ret + 0.20 * s_stable
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
                return 0.40, "static", 0.35   # P4: antes 0.62/0.45 (premiaba lo plano)   # score moderado, confianza baja
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

        key_root, key_mode = _detect_key_simple(notes, mid)
        bar_ticks          = _estimate_bar_ticks(mid)
        tpb                = mid.ticks_per_beat

        # En modo fast siempre usamos nivel básico (evita el bucle por-beat)
        sp = ctx.scale_pcs
        if ctx.fast:
            return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode, sp)

        # v1.3: las reglas SATB (cruces, paralelas) solo se aplican si las voces
        # están DECLARADAS (nombres soprano/alto/tenor…, o --voices satb); un
        # arpegio o un acompañamiento monofónico no son partes de un coral.
        # --compat reproduce v1.2 (SATB siempre que la topología sea multi_labeled).
        use_voiced = (topology.type == "multi_labeled" and len(topology.melody_tracks) > 0
                      and sp is None
                      and (ctx.compat or ctx.voices == "satb"
                           or (ctx.voices == "auto" and topology.declared_voices)))
        if use_voiced:
            return self._evaluate_voiced(notes, topology, bar_ticks, tpb,
                                         key_root, key_mode)
        return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode, sp)

    # ── Nivel básico ──────────────────────────────────────────────────────────

    def _evaluate_basic(self, notes: list, bar_ticks: int, tpb: int,
                        key_root: int, key_mode: str,
                        scale_pcs: Optional[set] = None) -> DimensionResult:
        """
        Analiza intervalos simultáneos en ventanas de 1 beat.
        Métricas: densidad de disonancia, saltos grandes, notas fuera de escala.
        Nivel 1+3: localiza cada problema por compás.
        """
        MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
        MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
        scale_ints  = MAJOR_SCALE if key_mode == 'major' else MINOR_SCALE
        declared    = scale_pcs is not None
        scale_pcs   = scale_pcs if declared else {(key_root + i) % 12 for i in scale_ints}

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

        # P1: saltos grandes POR PISTA (antes se medía a través de pistas
        # distintas, lo que producía "saltos" que eran simples cambios de voz).
        by_track: Dict[int, list] = defaultdict(list)
        for n in notes:
            if n[4] != 9:
                by_track[n[5]].append(n)
        for tr, ns in by_track.items():
            ns = sorted(ns, key=lambda n: n[0])
            for i in range(1, len(ns)):
                interval = abs(ns[i][2] - ns[i - 1][2])
                if interval > 0:
                    leap_total += 1
                    if interval > 9:
                        large_leaps += 1
                        bar = int(ns[i][0] // bar_ticks) + 1
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
            # P1: un acorde terciano (tríada ± 7ª) es consonante por
            # construcción; sólo se aplica la regla de intervalos a
            # sonoridades no tercianas (clusters, contrapunto disonante).
            if _is_tertian_sonority(simultaneous):
                has_dissonance = False
            else:
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
                "scale_declared":     declared,
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
        raw_tracks: Dict[int, list] = defaultdict(list)
        for n in notes:
            if n[5] not in topology.drum_tracks:
                raw_tracks[n[5]].append(n)

        if len(raw_tracks) < 2:
            return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode)

        # Polifonía de cada pista (fracción de onsets con >1 nota)
        polys = []
        for ns in raw_tracks.values():
            onsets_p: Dict[int, list] = {}
            for n in ns:
                onsets_p.setdefault(n[0], []).append(n)
            polys.append(sum(1 for v in onsets_p.values() if len(v) > 1)
                         / max(len(onsets_p), 1))

        # P1: las reglas SATB sólo se aplican a material realmente
        # separado en voces (cada pista casi monofónica). Si CUALQUIER pista
        # es polifónica (textura de piano/acompañamiento), se usa el nivel
        # básico, que ya no penaliza acordes tercianos ni mide saltos a
        # través de pistas.
        if polys and max(polys) >= 0.15:
            return self._evaluate_basic(notes, bar_ticks, tpb, key_root, key_mode)

        # ── P1: separación de voces ─────────────────────────────────────────
        # Una pista puede ser polifónica (piano reducido, coral aplanado…).
        # Comparar pistas polifónicas como si fueran voces generaba cruces
        # falsos. Extraemos voces monofónicas sintéticas: superior (skyline)
        # e inferior (baseline) de cada pista polifónica; las monofónicas se
        # usan tal cual.
        tracks_notes: Dict[str, list] = {}
        for ti, ns in raw_tracks.items():
            onsets: Dict[int, list] = {}
            for n in ns:
                onsets.setdefault(n[0], []).append(n)
            poly = sum(1 for v in onsets.values() if len(v) > 1) / max(len(onsets), 1)
            if poly < 0.35:
                tracks_notes[f"t{ti}"] = ns
            else:
                top = {s: max(g, key=lambda n: n[2]) for s, g in onsets.items()}
                bot = {s: min(g, key=lambda n: n[2]) for s, g in onsets.items()}
                def _mono(d, ti=ti):
                    return [(s, d[s][1], d[s][2], d[s][3], d[s][4], ti) for s in sorted(d)]
                tracks_notes[f"t{ti}^"] = _mono(top)
                tracks_notes[f"t{ti}_"] = _mono(bot)

        # Ordenar voces por pitch medio descendente → [S, A, T, B, ...]
        voice_order = sorted(
            tracks_notes.keys(),
            key=lambda v: np.mean([n[2] for n in tracks_notes[v]]) if tracks_notes[v] else 0,
            reverse=True,
        )

        def _base(v: str) -> str:
            return v.split("^")[0].split("_")[0]

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
        n_voiced_beats = 0

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
            if len(pitches_now) >= 2:
                n_voiced_beats += 1

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
                    # P1: no aplicar reglas de paralelas entre las voces
                    # sintéticas de una MISMA pista polifónica.
                    if _base(v1) == _base(v2):
                        continue
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

        # Score a partir de tasa de violaciones por beat CON voces activas
        # (P1): normalizar por los beats polifónicos, no por el total, evita
        # que el silencio diluya la tasa artificialmente.
        denom = max(n_voiced_beats, 1)
        v_per_beat = len(violations) / denom
        # Umbral: 0.5 violaciones por beat con voces = muy deficiente
        score = 1.0 - min(v_per_beat / 0.5, 1.0)
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
                "n_voiced_beats":  n_voiced_beats,
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

        # P4: ámbito de la línea melódica mono (skyline)
        onsets: Dict[int, int] = {}
        for n in mel_notes:
            if n[0] not in onsets or n[2] > onsets[n[0]]:
                onsets[n[0]] = n[2]
        mono = [onsets[k] for k in sorted(onsets)]
        s5, d5 = self._range_score(mono)

        if ctx.fast:
            score = 0.45 * s1 + 0.30 * s2 + 0.25 * s5
            s3, d3 = 0.5, {"markov_entropy_norm": None, "markov_entropy_bits": None}
            s4, d4 = 0.5, {"repetition_ratio": None, "window_size": None}
        else:
            s3, d3 = self._markov_entropy(pitches)
            s4, d4 = self._repetition_variation(pitches)
            score = (0.25 * s1 + 0.20 * s2 + 0.20 * s3 +
                     0.15 * s4 + 0.20 * s5)
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
            **d1, **d2, **d3, **d4, **d5,
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

        # Curva de score: gaussiana centrada en H_norm = 0.65 (P7).
        # El óptimo 0.55 estaba calibrado a las melodías *generadas* (muy
        # regulares); la música real tiene más variedad interválica (~0.6–0.8).
        optimal = 0.62
        width   = 0.18
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

        # Score: gaussiana centrada en rate=0.55 (P7). El óptimo 0.40 estaba
        # calibrado a las melodías generadas; la música real inflexiona más.
        optimal = 0.48
        width   = 0.18
        score   = math.exp(-((rate - optimal) ** 2) / (2 * width ** 2))

        return float(score), {
            "contour_inflections":  inflections,
            "inflection_rate":      round(rate, 4),
        }

    # ── Submétrica 2b: Ámbito melódico (P4) ────────────────────────────────

    def _range_score(self, pitches: list) -> Tuple[float, dict]:
        """Premia un ámbito idiomatic (≈12–22 semitonos) y penaliza líneas
        planas (pocos semitonos) o desmesuradas (más de ~2'5 octavas, señal
        de que se está midiendo acompañamiento y no melodía)."""
        if len(pitches) < 4:
            return 0.5, {"melodic_range": None}
        rng = max(pitches) - min(pitches)
        optimal, width = 16.0, 7.0
        score = math.exp(-((rng - optimal) ** 2) / (2 * width ** 2))
        if rng < 7 or rng > 34:
            score *= 0.4
        return float(score), {"melodic_range": int(rng)}

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

        # Score: gaussiana centrada en H_norm=0.68 (P7). El óptimo 0.50
        # penalizaba la impredecibilidad propia de la música real (más
        # transiciones posibles entre clases de altura).
        optimal = 0.62
        width   = 0.24
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
#  EVALUADOR: HARMONY  (P3 — nueva dimensión)
# ══════════════════════════════════════════════════════════════════════════════

class HarmonyEvaluator:
    """Oficio armónico: lo que ninguna otra dimensión medía.

    Calcula, por compás, la tríada que mejor explica el croma (M/m/dim/aug) y
    evalúa diatonicidad, funcionalidad T/S/D, diversidad de acordes, cadencias
    V→I / IV→I y ritmo armónico.
    """
    NAME = "harmony"
    MAJOR = {0, 2, 4, 5, 7, 9, 11}
    MINOR = {0, 2, 3, 5, 7, 8, 10}
    TRIADS = {"M": {0, 4, 7}, "m": {0, 3, 7}, "d": {0, 3, 6}, "a": {0, 4, 8}}

    def _best_triad(self, pcs: set):
        """Tríada (M/m/dim/aug) que mejor explica un conjunto de clases de altura."""
        best, best_cov = None, -1e9
        for root in range(12):
            for qual, ivs in self.TRIADS.items():
                tones = {(root + i) % 12 for i in ivs}
                cov = len(pcs & tones) - 0.4 * len(pcs - tones)
                if cov > best_cov:
                    best_cov, best = cov, (root, qual, tones)
        return best

    def _functional_key(self, bar_pcs: list) -> Tuple[int, str]:
        """Tonalidad que mejor explica las tríadas por compás (T/S/D).

        El histograma global de alturas es ambiguo en texturas de piano
        (A#m y F#M comparten 6 de 7 notas). Maximizar la diatonicidad de las
        tríadas y el énfasis en la tónica es más fiable y coincide mejor con
        el análisis funcional de harmonic_analyzer.
        """
        best, best_sc = (0, "major"), -1e9
        for root in range(12):
            for mode in ("major", "minor"):
                scale = self.MAJOR if mode == "major" else self.MINOR
                diat = {(root + i) % 12 for i in scale}
                if mode == "minor":
                    diat |= {(root + 11) % 12}
                hits = tonic_bars = 0
                for pcs in bar_pcs:
                    r, q, tones = self._best_triad(pcs)
                    if (r in diat) and tones.issubset(diat):
                        hits += 1
                    if r == root:
                        tonic_bars += 1
                n = max(len(bar_pcs), 1)
                sc = hits / n + 0.30 * tonic_bars / n
                if sc > best_sc:
                    best_sc, best = sc, (root, mode)
        return best

    def evaluate(self, notes: list, mid: MidiFile,
                 topology: Topology, ctx: EvalContext) -> DimensionResult:
        if not notes:
            return DimensionResult(0.5, 0.0,
                ["Sin notas para análisis armónico"], {})

        bar_ticks = _estimate_bar_ticks(mid)
        total     = _get_total_ticks(mid)
        n_bars    = max(1, total // bar_ticks)

        def best_triad(pcs: set):
            return self._best_triad(pcs)

        # Croma por compás
        bar_pcs = []
        for b in range(n_bars):
            t0, t1 = b * bar_ticks, (b + 1) * bar_ticks
            pcs = {n[2] % 12 for n in notes
                   if n[4] != 9 and n[0] < t1 and n[1] > t0}
            if len(pcs) >= 2:
                bar_pcs.append(pcs)
        if not bar_pcs:
            return DimensionResult(0.5, 0.0, ["Sin armonía vertical detectada"], {})

        # P0: tonalidad del detector global (ponderado por duración), que
        # coincide con harmonic_analyzer en ~3/4 de las piezas. El ajuste
        # funcional sesgaba la tonalidad en texturas polifónicas.
        key_root, key_mode = _detect_key_simple(notes, mid)
        scale = self.MAJOR if key_mode == "major" else self.MINOR
        diat_pcs = {(key_root + i) % 12 for i in scale}
        if key_mode == "minor":
            diat_pcs = diat_pcs | {(key_root + 11) % 12}   # sensible (armónica)
        # v1.3: colección declarada → diatonicidad respecto a ella; si no es
        # funcional (whole-tone, octatónica…) no se exigen T/S/D ni cadencia V→I.
        functional = True
        if ctx.scale_pcs is not None:
            diat_pcs = set(ctx.scale_pcs)
            functional = ctx.scale_name in _FUNCTIONAL_SCALES

        chords = []
        for pcs in bar_pcs:
            root, qual, tones = best_triad(pcs)
            diat = (root in diat_pcs) and tones.issubset(diat_pcs)
            chords.append((root, qual, diat))

        diat_ratio = sum(1 for c in chords if c[2]) / len(chords)
        roots      = {c[0] for c in chords}
        roman_div  = min(len(roots) / 5.0, 1.0)
        quals      = {(c[0], c[1]) for c in chords}
        qual_div   = min(len(quals) / 8.0, 1.0)   # P8: riqueza de vocabulario

        # funcionalidad T / S / D
        I     = key_root
        V     = (key_root + 7) % 12
        s_pcs = {(key_root + (5 if key_mode == "major" else 3)) % 12,
                 (key_root + 9) % 12,
                 (key_root + 5) % 12}
        funcs = set()
        for r, _, _ in chords:
            if r == I:     funcs.add("T")
            elif r == V:   funcs.add("D")
            elif r in s_pcs: funcs.add("S")
        func_div = len(funcs) / 3.0

        # cadencias V→I / IV→I
        cad = 0.0
        for i in range(1, len(chords)):
            pr, cr = chords[i-1][0], chords[i][0]
            if pr == V and cr == I:
                cad += 1.0
            elif pr == (key_root + 5) % 12 and cr == I:
                cad += 0.5
        s_cad = min(1.0, 0.35 + 0.30 * (1 if cad > 0 else 0) + 0.35 * min(cad, 2) / 2)

        # ritmo armónico: proporción de cambios de acorde entre compases
        changes = sum(1 for i in range(1, len(chords)) if chords[i][0] != chords[i-1][0])
        hr = changes / max(len(chords) - 1, 1)
        s_hr = float(math.exp(-((hr - 0.5) ** 2) / (2 * 0.3 ** 2)))

        # P8: se reparte el peso para premiar la RIQUEZA de vocabulario
        # (raíces+calidades) y no tanto la diatonicidad bruta, que penalizaba
        # al material real (más cromático y con extensiones).
        if functional:
            score = (0.30 * diat_ratio + 0.18 * func_div + 0.14 * roman_div +
                     0.14 * s_cad + 0.08 * s_hr + 0.16 * qual_div)
        else:
            # sin función tonal ni cadencia: se renormalizan los demás pesos
            score = (0.30 * diat_ratio + 0.14 * roman_div +
                     0.08 * s_hr + 0.16 * qual_div) / 0.68
        score = max(0.0, min(1.0, float(score)))

        reasons = []
        if diat_ratio < 0.6:
            reasons.append(f"diatonicidad baja: {diat_ratio:.0%} compases")
        if not functional:
            reasons.append(f"colección declarada «{ctx.scale_name}»: no se evalúan T/S/D ni cadencia V→I")
        elif func_div < 0.6:
            reasons.append("poca variedad funcional (T/S/D)")
        if qual_div > 0.6:
            reasons.append("vocabulario armónico rico")
        if functional and cad == 0:
            reasons.append("sin cadencia V→I ni IV→I")
        if score > 0.75:
            reasons.append("armonía diatónica, funcional y con cadencias")

        conf = min(1.0, len(chords) / 8) * (1.0 if key_mode in ("major", "minor") else 0.6)
        if not functional:
            conf *= 0.8   # menos reglas aplicables → menos confianza
        return DimensionResult(
            score=score, confidence=conf, reasons=reasons,
            details={
                "key":             f"{PITCH_NAMES[key_root]} {key_mode}",
                "scale_declared":  ctx.scale_name if ctx.scale_pcs is not None else None,
                "functional":      functional,
                "n_bars":          len(chords),
                "diatonic_ratio":  round(diat_ratio, 4),
                "func_div":        round(func_div, 4),
                "roman_div":       round(roman_div, 4),
                "qual_div":        round(qual_div, 4),
                "cadences":        int(cad),
                "harmonic_rhythm": round(hr, 4),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _parse_roles_arg(txt: str) -> Dict:
    """'1=melody,2=harmony,ch10=drums' → {'track': {1:'melody',2:'harmony'}, 'channel': {10:'drums'}}"""
    out = {'track': {}, 'channel': {}}
    for part in txt.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"--roles: «{part}» no tiene la forma N=rol")
        k, v = (x.strip() for x in part.split('=', 1))
        v = v.casefold()
        if v not in _VALID_ROLES:
            raise ValueError(f"--roles: rol «{v}» desconocido (válidos: {sorted(_VALID_ROLES)})")
        if k.casefold().startswith('ch'):
            ch = int(k[2:])
            if not 1 <= ch <= 16:
                raise ValueError("--roles: el canal debe estar entre 1 y 16")
            out['channel'][ch] = v
        else:
            out['track'][int(k)] = v
    return out


def _parse_scale_arg(txt: str) -> Tuple[str, Optional[int]]:
    root = None
    name = txt.strip()
    if ':' in name:
        r, name = name.split(':', 1)
        r = r.strip()
        if r not in PITCH_NAMES:
            raise ValueError(f"--scale: raíz «{r}» desconocida (use {', '.join(PITCH_NAMES)})")
        root = PITCH_NAMES.index(r)
    name = name.strip()
    if name not in DECLARABLE_SCALES:
        raise ValueError(f"--scale: colección «{name}» desconocida. Válidas: "
                         f"{', '.join(sorted(DECLARABLE_SCALES))}")
    return name, root


def _load_context_report(path: str) -> Tuple[str, Optional[int]]:
    """Lee un informe de theme_transformer: target_scale + tónica de source_key."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        raise ValueError(f"--context: no se pudo leer {path}: {e}")
    name = d.get('target_scale')
    if name not in DECLARABLE_SCALES:
        raise ValueError(f"--context: target_scale «{name}» no es una colección declarable "
                         f"(¿escala dinámica 'octatonic_auto'/'pentatonic_auto'? use --scale)")
    root = None
    sk = str(d.get('source_key', '')).split(' ')[0]
    if sk in PITCH_NAMES:
        root = PITCH_NAMES.index(sk)
    return name, root


_EVALUATORS = {
    "formal":    FormalEvaluator(),
    "coherence": CoherenceEvaluator(),
    "arc":       ArcEvaluator(),
    "melodic":   MelodicEvaluator(),
    "harmony":   HarmonyEvaluator(),
    "corpus":    CorpusEvaluator(),
}


def evaluate_midi(midi_path: str,
                  ctx: EvalContext,
                  weights: Dict[str, float],
                  active_dims: List[str]) -> QualityReport:
    """Pipeline completo de evaluación para un único MIDI."""

    mid   = MidiFile(midi_path)
    notes = _extract_notes(mid)
    topo  = detect_topology(mid, ctx)

    # v1.3: filtro ÚNICO de percusión para los análisis tonales (formal, melodic,
    # harmony): las alturas GM del canal 10 no son notas armónicas. coherence/arc/
    # corpus siguen usando todas las notas (densidad y energía sí incluyen la batería).
    if ctx.compat:
        tonal_notes = notes
    else:
        tonal_notes = [n for n in notes if n[4] != _DRUM_CHANNEL] or notes
    n_drum_excl = len(notes) - len(tonal_notes)

    # colección declarada: tónica explícita o la tonalidad detectada
    ctx.scale_pcs = None
    if ctx.scale_name:
        root = ctx.scale_root
        if root is None:
            root = _detect_key_simple(tonal_notes, mid)[0]
        ctx.scale_pcs = {(root + i) % 12 for i in DECLARABLE_SCALES[ctx.scale_name]}
        ctx._scale_root_used = root

    # Metadatos básicos
    tempo_us  = _get_tempo(mid)
    tempo_bpm = round(60_000_000 / tempo_us, 1)
    bar_ticks = _estimate_bar_ticks(mid)
    total     = _get_total_ticks(mid)
    n_bars    = max(1, total // bar_ticks)
    key_root, key_mode = _detect_key_simple(tonal_notes, mid)
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
        # v1.3: trazabilidad
        "roles":                {str(k): v for k, v in topo.roles.items()},
        "role_warnings":        list(topo.warnings),
        "notes_total":          len(notes),
        "notes_tonal_used":     len(tonal_notes),
        "drum_notes_excluded":  n_drum_excl,
        "declared_scale":       ({"name": ctx.scale_name,
                                  "root": PITCH_NAMES[getattr(ctx, '_scale_root_used', 0)],
                                  "source": ctx.scale_source}
                                 if ctx.scale_pcs is not None else None),
        "compat":               ctx.compat,
    }

    # Ejecutar evaluadores activos
    results: Dict[str, DimensionResult] = {}
    for dim in active_dims:
        evaluator = _EVALUATORS.get(dim)
        if evaluator is None:
            continue
        try:
            dim_notes = tonal_notes if dim in ("formal", "melodic", "harmony") else notes
            results[dim] = evaluator.evaluate(dim_notes, mid, topo, ctx)
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
        file        = str(midi_path),
        name        = Path(midi_path).name,
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
    if m.get('roles'):
        _rs = ", ".join(f"{k}:{v['role']}({v['source'][0]})" for k, v in m['roles'].items())
        print(f"  Roles     : {_rs}   (r=roles n=nombre h=heurística f=respaldo)")
    if m.get('drum_notes_excluded'):
        print(f"  Percusión : {m['drum_notes_excluded']} notas excluidas del análisis tonal")
    if m.get('declared_scale'):
        _d = m['declared_scale']
        print(f"  Colección : {_d['root']} {_d['name']} (declarada: {_d['source']})")
    for _w in m.get('role_warnings', []):
        print(f"  {Y}AVISO{R}     : {_w}")
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

    # ── P5: avisos de interpretabilidad ────────────────────────────────────
    warns = []
    cohe = report.dimensions.get("coherence")
    if cohe is not None and cohe.score > 0.97:
        warns.append("coherence saturada (>0.97): no discrimina entre obras")
    fm = report.dimensions.get("formal")
    if fm is not None and fm.details.get("level") == "basic" and \
            report.meta.get("topology") != "single":
        warns.append("formal evaluado en nivel básico (textura polifónica): "
                     "las reglas SATB no se aplican")
    arc = report.dimensions.get("arc")
    if arc is not None and arc.details.get("arc_detected") == "static":
        warns.append("arco estático: tensión constante (score penalizado)")
    if warns:
        print(f"  {_c('yellow')}Avisos:{R}")
        for w in warns:
            print(f"    {_c('yellow')}· {w}{R}")

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
    p.add_argument("--roles", metavar="MAP", default=None,
                   help="roles explícitos: '1=melody,2=harmony,ch10=drums' "
                        "(N = índice de pista; chN = canal MIDI 1-16). Roles: "
                        "melody|bass|harmony|drums. Tiene precedencia sobre nombres y heurística")
    p.add_argument("--scale", metavar="[RAÍZ:]NOMBRE", default=None,
                   help="colección declarada, p.ej. whole_tone, D:dorian, C:double_harmonic. "
                        "Sin raíz usa la tonalidad detectada. Sustituye mayor/menor en formal "
                        "y harmony; si no es funcional no se exige V→I")
    p.add_argument("--context", metavar="JSON", default=None,
                   help="informe de theme_transformer (--report): toma target_scale y la "
                        "tónica de source_key. Pensado para UN fichero (en lote, un informe por pieza)")
    p.add_argument("--voices", choices=["auto", "satb", "texture"], default="auto",
                   help="reglas de voces: auto = SATB solo si las pistas se llaman "
                        "soprano/alto/tenor…; satb = forzar; texture = nunca")
    p.add_argument("--compat", action="store_true",
                   help="comportamiento de v1.2 (sin filtro de percusión, sin nombres, "
                        "SATB heurístico)")
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
        compat           = args.compat,
        voices           = args.voices,
    )
    try:
        _roles = _parse_roles_arg(args.roles) if args.roles else None
        if _roles:
            ctx.roles = _roles
        if args.scale:
            ctx.scale_name, ctx.scale_root = _parse_scale_arg(args.scale)
            ctx.scale_source = "--scale"
        if args.context:
            _n, _r = _load_context_report(args.context)
            if args.scale and _n != ctx.scale_name:
                print(f"[AVISO] --scale ({ctx.scale_name}) y --context ({_n}) difieren; "
                      f"se usa --scale")
            elif not args.scale:
                ctx.scale_name, ctx.scale_root = _n, _r
                ctx.scale_source = f"--context {Path(args.context).name}"
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

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
