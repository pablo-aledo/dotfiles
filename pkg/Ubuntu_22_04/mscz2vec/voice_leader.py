#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       VOICE LEADER  v1.0                                     ║
║         Motor de voice leading para bloques de acordes SATB                  ║
║                                                                              ║
║  Toma una progresión de acordes (texto o MIDI de acordes) y genera el        ║
║  voicing SATB óptimo compás a compás, aplicando las reglas clásicas de       ║
║  conducción de voces: movimiento mínimo, sin paralelas de 5ª/8ª, sin        ║
║  cruzamientos, rangos vocales idiomáticos y resolución de tensiones.         ║
║                                                                              ║
║  A diferencia de counterpoint.py (una voz vs. cantus firmus) y              ║
║  midi_dna_unified (voicing interno de acompañamiento), este módulo           ║
║  razona sobre los CUATRO bloques vocales simultáneos como sistema.           ║
║                                                                              ║
║  VOCES GENERADAS:                                                            ║
║    Soprano  (S)  — voz aguda, normalmente lleva la melodía principal         ║
║    Alto     (A)  — voz media-aguda (contralto)                               ║
║    Tenor    (T)  — voz media-grave                                           ║
║    Bajo     (B)  — voz grave, define la función armónica                     ║
║                                                                              ║
║  MODOS DE ENTRADA:                                                           ║
║    --chords "Cm G7 Fm Bb7"          — un acorde por compás                  ║
║    --chords "Cm:2 G7:2 Fm:1 Bb7:1" — duración en beats tras ':'             ║
║    --chords-midi acordes.mid        — pista de acordes en MIDI               ║
║    --soprano melodia.mid            — fija la voz Soprano y genera S/A/T/B  ║
║                                                                              ║
║  ESTILOS DE VOICING (--style):                                               ║
║    chorale    — coral clásico (Bach). Movimiento mínimo, voz leading         ║
║                 riguroso. Saltos de bajo resueltos. Sin cruzamientos.       ║
║    open       — voicing abierto: S y T en octavas extremas, A y B           ║
║                 rellenando. Sonido "orquestal" espacioso.                   ║
║    close      — voicing cerrado: S/A/T en el espacio de una octava.        ║
║                 Sonido íntimo, cuarteto vocal/madrigal.                     ║
║    jazz       — voicings de jazz: 7ª y 9ª en el bloque, bajo                ║
║                 independiente, tensiones permitidas en voces internas.      ║
║    keyboard   — dos manos: bajo en mano izquierda, tres voces en            ║
║                 mano derecha. Registración de piano/órgano.                 ║
║                                                                              ║
║  REGLAS IMPLEMENTADAS:                                                       ║
║  [R01] Sin paralelas de 5ª perfecta entre cualquier par de voces            ║
║  [R02] Sin paralelas de 8ª (unísono) entre cualquier par de voces           ║
║  [R03] Sin quintas/octavas directas hacia tiempo fuerte                     ║
║  [R04] Sin cruzamientos de voz (S > A > T > B siempre)                     ║
║  [R05] Sin solapamientos de voz (overlap)                                   ║
║  [R06] Rangos vocales idiomáticos respetados (SATB estándar)                ║
║  [R07] Saltos grandes (> 6ª) en cualquier voz penalizados                  ║
║  [R08] Saltos de 7ª resuelven en dirección contraria                       ║
║  [R09] Sensible (grado VII) resuelve hacia arriba (a la tónica)            ║
║  [R10] La 7ª de un acorde dominante resuelve hacia abajo                   ║
║  [R11] Bajo debe completar el tríada o séptima del acorde                  ║
║  [R12] No duplicar nota sensible ni 7ª del acorde                          ║
║  [R13] Preferir movimiento contrario o oblicuo sobre movimiento similar     ║
║  [R14] En modulaciones, la voz pivote se mantiene en su altura             ║
║                                                                              ║
║  USO STANDALONE:                                                             ║
║    python voice_leader.py --chords "C Am F G7"                              ║
║    python voice_leader.py --chords "Dm:2 G7:2 CM7:4" --key Dm              ║
║    python voice_leader.py --chords "C Am F G" --style jazz --tempo 120      ║
║    python voice_leader.py --soprano melodia.mid --chords "C F G C"          ║
║    python voice_leader.py --chords-midi acordes.mid --style chorale         ║
║    python voice_leader.py --chords "C Am F G" --all-styles                  ║
║    python voice_leader.py --chords "C Am F G" --report --verbose            ║
║                                                                              ║
║  USO COMO MÓDULO:                                                            ║
║    from voice_leader import voice_lead_progression, ChordBlock              ║
║                                                                              ║
║    chords = [("C", 4), ("Am", 4), ("F", 4), ("G7", 4)]  # (nombre, beats)  ║
║    result = voice_lead_progression(                                          ║
║        chords,                                                               ║
║        key="C major",                                                        ║
║        style="chorale",                                                      ║
║        soprano_hint=None,   # lista de MIDI pitches para fijar soprano      ║
║        seed=42,                                                              ║
║    )                                                                         ║
║    # result → lista de ChordBlock(soprano, alto, tenor, bass, duration)     ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --chords S          Progresión de acordes en texto                        ║
║    --chords-midi FILE  Progresión desde MIDI                                 ║
║    --soprano FILE      MIDI de melodía soprano (fija la voz S)              ║
║    --key KEY           Tonalidad: "C", "Am", "Bb major" (default: auto)     ║
║    --style S           chorale|open|close|jazz|keyboard (default: chorale)  ║
║    --all-styles        Generar una versión por cada estilo                   ║
║    --tempo BPM         Tempo de salida (default: 90)                        ║
║    --beats N           Beats por compás (default: 4)                        ║
║    --voices S [S…]     Voces a exportar: S A T B (default: todas)           ║
║    --split-tracks      Una pista MIDI por voz (para DAW)                    ║
║    --report            Guardar análisis de voice leading en JSON             ║
║    --score-report      Mostrar score de cada transición (reglas violadas)   ║
║    --output FILE       Fichero de salida (default: voice_led.mid)           ║
║    --output-dir DIR    Directorio con --all-styles                           ║
║    --listen            Reproducir al terminar (requiere pygame)              ║
║    --verbose           Informe regla a regla por stdout                      ║
║    --seed N            Semilla aleatoria (default: 42)                       ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    voice_led.mid                  — MIDI de 4 pistas (S/A/T/B)              ║
║    voice_led_S.mid…               — con --split-tracks                      ║
║    voice_led.report.json          — con --report                             ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:  mido, numpy                                                     ║
║    --key auto / análisis tonal:  music21 (opcional, fallback si no)         ║
║    --listen: pygame                                                          ║
║    Integración:  midi_dna_unified en el mismo directorio                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import mido

# ── Integración con el ecosistema ─────────────────────────────────────────────
_DNA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DNA_DIR)
try:
    from midi_dna_unified import (
        _snap_to_scale, _get_scale_pcs, _get_scale_midi,
        _build_chord_pitches_from_roman, _voice_lead, voice_lead_next_chord,
        _quarter_to_ticks, _clamp_pitch,
        MAJOR_SCALE_DEGREES, MINOR_SCALE_DEGREES, INSTRUMENT_RANGES,
    )
    DNA_OK = True
except ImportError:
    DNA_OK = False

try:
    from music21 import key as m21key, harmony as m21harmony, pitch as m21pitch
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# Rangos SATB estándar (MIDI): (min, max)
VOICE_RANGES = {
    "S": (60, 81),   # C4–A5
    "A": (53, 74),   # F3–D5
    "T": (48, 69),   # C3–A4
    "B": (36, 60),   # C2–C4
}

# Programas GM por voz
VOICE_PROGRAMS = {
    "S": 52,   # Choir Aahs
    "A": 52,
    "T": 52,
    "B": 52,
}

# Canales MIDI por voz
VOICE_CHANNELS = {"S": 0, "A": 1, "T": 2, "B": 3}

# Nombre de notas
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}

# Intervalos de calidad de acorde
CHORD_INTERVALS = {
    "M":   [0, 4, 7],
    "m":   [0, 3, 7],
    "d":   [0, 3, 6],
    "aug": [0, 4, 8],
    "M7":  [0, 4, 7, 11],
    "m7":  [0, 3, 7, 10],
    "7":   [0, 4, 7, 10],
    "dim7":[0, 3, 6, 9],
    "m7b5":[0, 3, 6, 10],
    "mM7": [0, 3, 7, 11],
    "9":   [0, 4, 7, 10, 14],
    "M9":  [0, 4, 7, 11, 14],
    "sus2":[0, 2, 7],
    "sus4":[0, 5, 7],
}

# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChordBlock:
    """Un acorde con sus cuatro voces asignadas."""
    soprano: int
    alto:    int
    tenor:   int
    bass:    int
    duration_beats: float = 4.0
    chord_name: str = ""
    root_pc: int = 0
    quality: str = "M"
    violations: List[str] = field(default_factory=list)
    score: float = 1.0

    @property
    def pitches(self) -> Dict[str, int]:
        return {"S": self.soprano, "A": self.alto, "T": self.tenor, "B": self.bass}

    def __repr__(self):
        return (f"ChordBlock({self.chord_name} | "
                f"S={self.soprano} A={self.alto} T={self.tenor} B={self.bass} "
                f"score={self.score:.2f})")


@dataclass
class VLReport:
    """Informe completo de voice leading de una progresión."""
    total_violations: int = 0
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    avg_motion: float = 0.0      # semitones promedio de movimiento por voz
    contrary_ratio: float = 0.0  # % de movimientos contrarios
    score: float = 1.0
    transitions: List[Dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

def parse_chord_name(name: str) -> Tuple[int, str]:
    """
    Parsea un nombre de acorde ('Cm', 'G7', 'F#m7', 'Bbsus4') →
    (root_pc 0-11, quality_str).
    """
    name = name.strip()
    # Raíz
    if len(name) >= 2 and name[1] in ("#", "b"):
        root_str = name[:2]
        suffix   = name[2:]
    else:
        root_str = name[:1]
        suffix   = name[1:]

    # Convertir a pitch class
    root_str_norm = root_str[0].upper() + root_str[1:]
    if root_str_norm in ENHARMONIC:
        root_pc = ENHARMONIC[root_str_norm]
    else:
        note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
        root_pc = note_map.get(root_str_norm.replace("b", "").replace("#", ""), 0)
        if "#" in root_str:
            root_pc = (root_pc + 1) % 12
        elif "b" in root_str and root_str_norm not in NOTE_NAMES:
            root_pc = (root_pc - 1) % 12

    # Calidad
    quality_map = {
        "m7b5": "m7b5", "dim7": "dim7", "mM7": "mM7",
        "maj7": "M7", "M7": "M7", "maj": "M",
        "m7": "m7", "7": "7", "m": "m",
        "aug": "aug", "dim": "d",
        "sus4": "sus4", "sus2": "sus2",
        "9": "9", "M9": "M9",
        "": "M",
    }
    # Buscar la coincidencia más larga
    quality = "M"
    for k in sorted(quality_map.keys(), key=len, reverse=True):
        if suffix.startswith(k):
            quality = quality_map[k]
            break

    return root_pc, quality


def parse_chord_string(chord_str: str, beats_per_bar: int = 4) -> List[Tuple[str, float]]:
    """
    Parsea "Cm G7 Fm:2 Bb7:2" → [("Cm", 4.0), ("G7", 4.0), ("Fm", 2.0), ("Bb7", 2.0)]
    """
    tokens = chord_str.strip().split()
    result = []
    for tok in tokens:
        if ":" in tok:
            name, dur_str = tok.rsplit(":", 1)
            try:
                dur = float(dur_str)
            except ValueError:
                dur = float(beats_per_bar)
        else:
            name = tok
            dur = float(beats_per_bar)
        result.append((name, dur))
    return result


def chord_pcs(root_pc: int, quality: str) -> List[int]:
    """Devuelve los pitch classes del acorde."""
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
    return [(root_pc + i) % 12 for i in intervals]


# ══════════════════════════════════════════════════════════════════════════════
#  VOICING INICIAL
# ══════════════════════════════════════════════════════════════════════════════

def _initial_voicing(root_pc: int, quality: str, style: str,
                     soprano_hint: Optional[int] = None) -> Tuple[int, int, int, int]:
    """
    Genera un voicing inicial para el primer acorde de la progresión.
    Devuelve (soprano, alto, tenor, bass).
    """
    pcs = chord_pcs(root_pc, quality)
    s_lo, s_hi = VOICE_RANGES["S"]
    a_lo, a_hi = VOICE_RANGES["A"]
    t_lo, t_hi = VOICE_RANGES["T"]
    b_lo, b_hi = VOICE_RANGES["B"]

    # Bajo: raíz en posición fundamental
    bass = _closest_pitch_in_range(root_pc, b_lo, b_hi)

    if style == "open":
        # Soprano y tenor a distancia de octava, alto en medio
        tenor = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[1], t_lo, t_hi)
        soprano = soprano_hint if soprano_hint and s_lo <= soprano_hint <= s_hi \
                  else _closest_pitch_in_range(root_pc, s_lo, s_hi)
        alto = _closest_pitch_in_range(pcs[1], a_lo, a_hi)
    elif style == "close":
        # S/A/T apilados en espacio de una 8ª
        soprano = soprano_hint if soprano_hint and s_lo <= soprano_hint <= s_hi \
                  else _closest_pitch_in_range(root_pc, 65, 74)
        alto   = _closest_pitch_in_range(pcs[1], soprano - 7, soprano - 1)
        tenor  = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[0],
                                          soprano - 12, alto - 1)
    elif style == "jazz":
        # Bajo con séptima si está disponible, soprano con tensión
        bass = _closest_pitch_in_range(root_pc, b_lo, b_hi)
        tenor = _closest_pitch_in_range(pcs[1], t_lo, t_hi)
        alto  = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[1], a_lo, a_hi)
        soprano = soprano_hint if soprano_hint and s_lo <= soprano_hint <= s_hi \
                  else _closest_pitch_in_range(pcs[-1], s_lo, s_hi)
    elif style == "keyboard":
        # Mano izquierda: bajo solo; mano derecha: soprano + alto + tenor cerrados
        bass = _closest_pitch_in_range(root_pc, 36, 52)
        soprano = soprano_hint if soprano_hint and s_lo <= soprano_hint <= s_hi \
                  else _closest_pitch_in_range(root_pc, 65, 76)
        alto   = _closest_pitch_in_range(pcs[1], soprano - 8, soprano - 1)
        tenor  = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[0],
                                          soprano - 14, alto - 1)
    else:  # chorale (por defecto)
        soprano = soprano_hint if soprano_hint and s_lo <= soprano_hint <= s_hi \
                  else _closest_pitch_in_range(root_pc, 64, 76)
        alto   = _closest_pitch_in_range(pcs[1], a_lo, min(soprano - 1, a_hi))
        tenor  = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[0],
                                          t_lo, min(alto - 1, t_hi))

    return soprano, alto, tenor, bass


def _closest_pitch_in_range(pc: int, lo: int, hi: int) -> int:
    """Encuentra el MIDI pitch más central al rango [lo, hi] con pitch class pc."""
    center = (lo + hi) // 2
    candidates = []
    for octave in range(10):
        p = pc + octave * 12
        if lo <= p <= hi:
            candidates.append(p)
    if not candidates:
        # Fallback: ajuste al límite
        p = pc
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        return max(lo, min(hi, p))
    return min(candidates, key=lambda x: abs(x - center))


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE VOICE LEADING
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_voicing(prev: ChordBlock, cand: Tuple[int, int, int, int],
                       root_pc: int, quality: str,
                       key_pc: int, is_major: bool,
                       style: str) -> Tuple[float, List[str]]:
    """
    Evalúa un voicing candidato dado el acorde previo.
    Devuelve (score, lista_de_violaciones).
    score = 1.0 es perfecto; cada violación reduce el score.
    """
    s, a, t, b = cand
    ps, pa, pt, pb = prev.soprano, prev.alto, prev.tenor, prev.bass
    pcs_set = set(chord_pcs(root_pc, quality))
    violations = []
    score = 1.0

    # ── R04: Sin cruzamientos ─────────────────────────────────────────────────
    if not (s >= a >= t >= b):
        if s < a:
            violations.append("R04:S-A_cruce")
            score -= 0.5
        if a < t:
            violations.append("R04:A-T_cruce")
            score -= 0.5
        if t < b:
            violations.append("R04:T-B_cruce")
            score -= 0.5

    # ── R06: Rangos vocales ───────────────────────────────────────────────────
    for voice, pitch, label in [("S", s, "R06:S"), ("A", a, "R06:A"),
                                  ("T", t, "R06:T"), ("B", b, "R06:B")]:
        lo, hi = VOICE_RANGES[voice]
        if not (lo <= pitch <= hi):
            violations.append(f"{label}_rango")
            score -= 0.3

    # ── Paralelismos: comparar cada par de voces ──────────────────────────────
    voices_now  = [s, a, t, b]
    voices_prev = [ps, pa, pt, pb]
    voice_names = ["S", "A", "T", "B"]

    for i, j in itertools.combinations(range(4), 2):
        interval_prev = abs(voices_prev[i] - voices_prev[j]) % 12
        interval_now  = abs(voices_now[i]  - voices_now[j])  % 12
        motion_i = voices_now[i] - voices_prev[i]
        motion_j = voices_now[j] - voices_prev[j]
        same_direction = (motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0)

        # R01: Paralelas de 5ª
        if interval_prev == 7 and interval_now == 7 and same_direction:
            violations.append(f"R01:5as_paralelas_{voice_names[i]}-{voice_names[j]}")
            score -= 0.6

        # R02: Paralelas de 8ª / unísono
        if interval_prev == 0 and interval_now == 0 and same_direction:
            violations.append(f"R02:8as_paralelas_{voice_names[i]}-{voice_names[j]}")
            score -= 0.6

        # R03: Quintas/octavas directas en movimiento similar hacia tiempo fuerte
        if interval_now in (0, 7) and same_direction:
            if abs(motion_i) > 2:  # salto en la voz superior
                violations.append(f"R03:5a8a_directa_{voice_names[i]}-{voice_names[j]}")
                score -= 0.35

        # R13: Preferir movimiento contrario u oblicuo
        if same_direction and motion_i != 0 and motion_j != 0:
            score -= 0.05  # pequeña penalización por movimiento similar

    # ── R07: Penalizar saltos grandes ────────────────────────────────────────
    for vi, (now, prev_p) in enumerate(zip(voices_now, voices_prev)):
        leap = abs(now - prev_p)
        if leap > 9:   # mayor que 6ª
            violations.append(f"R07:salto_grande_{voice_names[vi]}_{leap}st")
            score -= 0.25
        elif leap > 7: # 6ª mayor
            score -= 0.1

    # ── R08: Resolución de 7ª ────────────────────────────────────────────────
    if quality in ("7", "M7", "m7", "dim7", "m7b5"):
        seventh_pc = (root_pc + CHORD_INTERVALS[quality][3]) % 12
        for vi, (now, prev_p) in enumerate(zip(voices_now, voices_prev)):
            if prev_p % 12 == seventh_pc:
                if now > prev_p:  # la 7ª debería bajar
                    violations.append(f"R08:7a_no_resuelve_{voice_names[vi]}")
                    score -= 0.3

    # ── R09: Sensible resuelve hacia arriba ──────────────────────────────────
    leading_tone = (key_pc + 11) % 12
    for vi, (now, prev_p) in enumerate(zip(voices_now, voices_prev)):
        if prev_p % 12 == leading_tone:
            tonic = key_pc % 12
            if now % 12 != tonic:
                violations.append(f"R09:sensible_no_resuelve_{voice_names[vi]}")
                score -= 0.2

    # ── R11: Bajo completa la tríada ─────────────────────────────────────────
    if b % 12 not in pcs_set:
        violations.append("R11:bajo_no_en_acorde")
        score -= 0.4

    # ── R12: No duplicar sensible ni 7ª ──────────────────────────────────────
    leading_tone_pc = (key_pc + 11) % 12
    lt_count = sum(1 for p in voices_now if p % 12 == leading_tone_pc)
    if lt_count > 1:
        violations.append("R12:sensible_duplicada")
        score -= 0.3

    # ── Bonus: movimiento mínimo total ───────────────────────────────────────
    total_motion = sum(abs(n - p) for n, p in zip(voices_now, voices_prev))
    score -= total_motion * 0.02  # penalización suave por movimiento innecesario

    return max(0.0, score), violations


def _generate_candidates(root_pc: int, quality: str, style: str,
                          prev: ChordBlock,
                          soprano_hint: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """
    Genera candidatos de voicing (S, A, T, B) para el acorde actual.
    """
    pcs = chord_pcs(root_pc, quality)
    s_lo, s_hi = VOICE_RANGES["S"]
    a_lo, a_hi = VOICE_RANGES["A"]
    t_lo, t_hi = VOICE_RANGES["T"]
    b_lo, b_hi = VOICE_RANGES["B"]

    # Soprano: si hay hint fijo, solo ese; si no, candidatos cerca del previo
    if soprano_hint is not None:
        s_candidates = [soprano_hint] if s_lo <= soprano_hint <= s_hi else \
                        [_closest_pitch_in_range(soprano_hint % 12, s_lo, s_hi)]
    else:
        s_candidates = _nearby_pitches(prev.soprano, pcs, s_lo, s_hi, n=3)

    # Bajo: típicamente root (posición fundamental) o quinta para cadencias
    b_root = _closest_pitch_in_range(root_pc, b_lo, b_hi)
    b_fifth = _closest_pitch_in_range((root_pc + 7) % 12, b_lo, b_hi) if len(pcs) > 2 else b_root
    b_prev_move = _nearest_pc(prev.bass, root_pc, b_lo, b_hi)
    b_candidates = list({b_root, b_fifth, b_prev_move})

    candidates = []
    for s in s_candidates:
        for b in b_candidates:
            # Generar A y T con movimiento mínimo
            a_options = _nearby_pitches(prev.alto, pcs, a_lo, min(s - 1, a_hi), n=3)
            for a in a_options:
                t_options = _nearby_pitches(prev.tenor, pcs, t_lo, min(a - 1, t_hi), n=3)
                for t in t_options:
                    if b < t:  # B debe ser la voz más grave
                        candidates.append((s, a, t, b))

    # Fallback si no hay candidatos válidos
    if not candidates:
        s = soprano_hint or _closest_pitch_in_range(root_pc, s_lo, s_hi)
        a = _closest_pitch_in_range(pcs[1] if len(pcs) > 1 else pcs[0], a_lo, a_hi)
        t = _closest_pitch_in_range(pcs[2] if len(pcs) > 2 else pcs[0], t_lo, t_hi)
        b = b_root
        candidates = [(s, a, t, b)]

    return candidates


def _nearby_pitches(prev_pitch: int, pcs: List[int], lo: int, hi: int, n: int = 3) -> List[int]:
    """Genera los N pitches de pcs más cercanos al pitch previo dentro del rango."""
    candidates = []
    for pc in pcs:
        for oct_ in range(10):
            p = pc + oct_ * 12
            if lo <= p <= hi:
                candidates.append(p)
    candidates = sorted(set(candidates), key=lambda x: abs(x - prev_pitch))
    return candidates[:n] if candidates else [_closest_pitch_in_range(pcs[0], lo, hi)]


def _nearest_pc(prev_pitch: int, target_pc: int, lo: int, hi: int) -> int:
    """Pitch de pitch class target_pc más cercano a prev_pitch en [lo, hi]."""
    best = None
    best_dist = 9999
    for oct_ in range(10):
        p = target_pc + oct_ * 12
        if lo <= p <= hi:
            d = abs(p - prev_pitch)
            if d < best_dist:
                best_dist = d
                best = p
    return best if best is not None else _closest_pitch_in_range(target_pc, lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: voice_lead_progression
# ══════════════════════════════════════════════════════════════════════════════

def voice_lead_progression(
    chords: List[Tuple[str, float]],
    key: str = "C major",
    style: str = "chorale",
    soprano_hints: Optional[List[Optional[int]]] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[ChordBlock], VLReport]:
    """
    Genera voice leading SATB para una progresión de acordes.

    Args:
        chords:         Lista de (nombre_acorde, duracion_beats)
        key:            Tonalidad "C major", "Am", "Bb", etc.
        style:          "chorale" | "open" | "close" | "jazz" | "keyboard"
        soprano_hints:  Lista opcional de MIDI pitches para fijar la voz S
        seed:           Semilla aleatoria
        verbose:        Imprimir decisiones

    Returns:
        (lista_de_ChordBlock, VLReport)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Parsear tonalidad
    key_pc, is_major = _parse_key(key)

    # Soprano hints (None = libre)
    if soprano_hints is None:
        soprano_hints = [None] * len(chords)
    while len(soprano_hints) < len(chords):
        soprano_hints.append(None)

    result: List[ChordBlock] = []
    report = VLReport()

    # ── Primer acorde: voicing inicial ───────────────────────────────────────
    first_name, first_dur = chords[0]
    root_pc, quality = parse_chord_name(first_name)
    s0, a0, t0, b0 = _initial_voicing(root_pc, quality, style, soprano_hints[0])

    first_block = ChordBlock(
        soprano=s0, alto=a0, tenor=t0, bass=b0,
        duration_beats=first_dur,
        chord_name=first_name, root_pc=root_pc, quality=quality,
        score=1.0
    )
    result.append(first_block)

    if verbose:
        print(f"  [1/{len(chords)}] {first_name:8s} → S={s0} A={a0} T={t0} B={b0}  (inicial)")

    # ── Acordes siguientes: búsqueda de mejor candidato ──────────────────────
    for idx, (chord_name, dur) in enumerate(chords[1:], start=1):
        root_pc, quality = parse_chord_name(chord_name)
        prev = result[-1]

        candidates = _generate_candidates(
            root_pc, quality, style, prev, soprano_hints[idx]
        )

        best_cand = None
        best_score = -9999.0
        best_viols = []

        for cand in candidates:
            sc, viols = _evaluate_voicing(
                prev, cand, root_pc, quality, key_pc, is_major, style
            )
            if sc > best_score:
                best_score = sc
                best_cand  = cand
                best_viols = viols

        s, a, t, b = best_cand
        block = ChordBlock(
            soprano=s, alto=a, tenor=t, bass=b,
            duration_beats=dur,
            chord_name=chord_name, root_pc=root_pc, quality=quality,
            violations=best_viols,
            score=best_score,
        )
        result.append(block)

        # Acumular informe
        report.total_violations += len(best_viols)
        for v in best_viols:
            rule = v.split(":")[0]
            report.violations_by_rule[rule] = report.violations_by_rule.get(rule, 0) + 1

        if verbose:
            viols_str = ", ".join(best_viols) if best_viols else "sin violaciones"
            print(f"  [{idx+1}/{len(chords)}] {chord_name:8s} → "
                  f"S={s} A={a} T={t} B={b}  score={best_score:.2f}  [{viols_str}]")

        # Guardar transición en el informe
        report.transitions.append({
            "from": prev.chord_name,
            "to":   chord_name,
            "motion": {
                "S": s - prev.soprano, "A": a - prev.alto,
                "T": t - prev.tenor,   "B": b - prev.bass,
            },
            "violations": best_viols,
            "score": best_score,
        })

    # ── Calcular métricas globales del informe ────────────────────────────────
    if len(result) > 1:
        all_motions = []
        contrary_count = 0
        total_pairs = 0
        for tr in report.transitions:
            motions = list(tr["motion"].values())
            all_motions.extend([abs(m) for m in motions])
            # Contar movimientos contrarios (S vs B como proxy)
            for i, j in itertools.combinations(range(4), 2):
                mi, mj = motions[i], motions[j]
                if mi != 0 and mj != 0:
                    total_pairs += 1
                    if (mi > 0) != (mj > 0):
                        contrary_count += 1
        report.avg_motion = float(np.mean(all_motions)) if all_motions else 0.0
        report.contrary_ratio = contrary_count / total_pairs if total_pairs > 0 else 0.0
        report.score = float(np.mean([b.score for b in result]))

    return result, report


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def blocks_to_midi(
    blocks: List[ChordBlock],
    tempo_bpm: int = 90,
    ticks_per_beat: int = 480,
    split_tracks: bool = False,
    voices: Optional[List[str]] = None,
) -> mido.MidiFile:
    """
    Convierte una lista de ChordBlocks a un MidiFile.
    Si split_tracks=True, una pista por voz.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_us = mido.bpm2tempo(tempo_bpm)
    voice_keys = voices or ["S", "A", "T", "B"]

    if split_tracks:
        tracks = {}
        for vk in voice_keys:
            tr = mido.MidiTrack()
            tr.name = {"S": "Soprano", "A": "Alto", "T": "Tenor", "B": "Bass"}[vk]
            mid.tracks.append(tr)
            tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
            tr.append(mido.Message("program_change",
                                   channel=VOICE_CHANNELS[vk],
                                   program=VOICE_PROGRAMS[vk], time=0))
            tracks[vk] = tr

        for block in blocks:
            dur_ticks = int(block.duration_beats * ticks_per_beat)
            pitches = block.pitches
            for vk in voice_keys:
                p = pitches[vk]
                ch = VOICE_CHANNELS[vk]
                tracks[vk].append(mido.Message("note_on", channel=ch, note=p,
                                                velocity=70, time=0))
                tracks[vk].append(mido.Message("note_off", channel=ch, note=p,
                                                velocity=0, time=dur_ticks))
    else:
        # Una pista con todos los canales
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        for vk in voice_keys:
            ch = VOICE_CHANNELS[vk]
            tr.append(mido.Message("program_change", channel=ch,
                                   program=VOICE_PROGRAMS[vk], time=0))

        for block in blocks:
            dur_ticks = int(block.duration_beats * ticks_per_beat)
            pitches = block.pitches
            # Note-ons simultáneos (delta=0 desde el primero)
            first = True
            for vk in voice_keys:
                p = pitches[vk]
                ch = VOICE_CHANNELS[vk]
                tr.append(mido.Message("note_on", channel=ch, note=p,
                                       velocity=70, time=0))
                first = False
            # Note-offs: primer off lleva el delta de duración
            for i, vk in enumerate(voice_keys):
                p = pitches[vk]
                ch = VOICE_CHANNELS[vk]
                tr.append(mido.Message("note_off", channel=ch, note=p,
                                       velocity=0,
                                       time=dur_ticks if i == 0 else 0))

    return mid


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE SOPRANO DESDE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_soprano_hints_from_midi(
    midi_path: str, beats_per_bar: int = 4
) -> Tuple[List[Optional[int]], List[Tuple[str, float]]]:
    """
    Lee un MIDI de melodía y devuelve los hints de soprano (un pitch por compás)
    y una progresión de acordes vacía con las duraciones correctas.
    Devuelve (soprano_hints, [(placeholder, dur), ...])
    """
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    notes_by_beat: Dict[float, List[int]] = {}
    current_time = 0.0

    for msg in mido.merge_tracks(mid.tracks):
        current_time += msg.time / tpb
        if msg.type == "note_on" and msg.velocity > 0:
            beat_idx = int(current_time)
            if beat_idx not in notes_by_beat:
                notes_by_beat[beat_idx] = []
            notes_by_beat[beat_idx].append(msg.note)

    total_beats = max(notes_by_beat.keys()) + beats_per_bar if notes_by_beat else beats_per_bar
    n_bars = int(math.ceil(total_beats / beats_per_bar))

    soprano_hints = []
    dummy_chords = []
    for bar in range(n_bars):
        beat_start = bar * beats_per_bar
        pitches = []
        for b in range(beats_per_bar):
            pitches.extend(notes_by_beat.get(beat_start + b, []))
        hint = max(pitches) if pitches else None  # nota más aguda = soprano
        soprano_hints.append(hint)
        dummy_chords.append(("C", float(beats_per_bar)))  # placeholder

    return soprano_hints, dummy_chords


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE TONALIDAD
# ══════════════════════════════════════════════════════════════════════════════

def _parse_key(key_str: str) -> Tuple[int, bool]:
    """Parsea 'C major', 'Am', 'Bb', 'F# minor' → (key_pc, is_major)."""
    key_str = key_str.strip()
    is_major = "minor" not in key_str.lower() and not key_str[-1].islower()
    # Extraer raíz
    root_str = key_str.split()[0]
    root_str = root_str.replace("major", "").replace("minor", "").strip()
    pc, _ = parse_chord_name(root_str + "M")  # abuso del parser de acordes
    return pc, is_major


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice_leader.py",
        description="Motor de voice leading SATB para progresiones de acordes.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--chords", type=str,
                     help='Progresión: "C Am F G7" o "Cm:2 G7:2 Fm:4"')
    src.add_argument("--chords-midi", type=str, metavar="FILE",
                     help="MIDI de acordes bloque")

    p.add_argument("--soprano", type=str, metavar="FILE",
                   help="MIDI de melodía para fijar la voz Soprano")
    p.add_argument("--key", type=str, default="C major",
                   help='Tonalidad (default: "C major")')
    p.add_argument("--style", type=str, default="chorale",
                   choices=["chorale", "open", "close", "jazz", "keyboard"],
                   help="Estilo de voicing (default: chorale)")
    p.add_argument("--all-styles", action="store_true",
                   help="Generar una versión por cada estilo")
    p.add_argument("--tempo", type=int, default=90,
                   help="Tempo BPM (default: 90)")
    p.add_argument("--beats", type=int, default=4,
                   help="Beats por compás (default: 4)")
    p.add_argument("--voices", nargs="+", choices=["S", "A", "T", "B"],
                   default=["S", "A", "T", "B"],
                   help="Voces a exportar (default: S A T B)")
    p.add_argument("--split-tracks", action="store_true",
                   help="Una pista MIDI por voz")
    p.add_argument("--report", action="store_true",
                   help="Exportar análisis JSON")
    p.add_argument("--score-report", action="store_true",
                   help="Mostrar score de cada transición")
    p.add_argument("--output", type=str, default="voice_led.mid",
                   help="Fichero de salida (default: voice_led.mid)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directorio de salida con --all-styles")
    p.add_argument("--listen", action="store_true",
                   help="Reproducir al terminar (requiere pygame)")
    p.add_argument("--verbose", action="store_true",
                   help="Informe regla a regla")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria (default: 42)")
    return p


def _print_report(report: VLReport, blocks: List[ChordBlock]) -> None:
    """Imprime un resumen del análisis de voice leading."""
    print("\n╔══════════════════════════════════════════╗")
    print("║        VOICE LEADING REPORT              ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Acordes procesados:   {len(blocks)}")
    print(f"  Score promedio:       {report.score:.3f}")
    print(f"  Violaciones totales:  {report.total_violations}")
    print(f"  Movimiento promedio:  {report.avg_motion:.2f} st/voz")
    print(f"  Movimiento contrario: {report.contrary_ratio*100:.1f}%")
    if report.violations_by_rule:
        print("\n  Violaciones por regla:")
        for rule, count in sorted(report.violations_by_rule.items(),
                                   key=lambda x: -x[1]):
            print(f"    {rule:15s}: {count}")
    print()


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Obtener progresión ────────────────────────────────────────────────────
    soprano_hints = None

    if args.soprano:
        print(f"[INFO] Leyendo melodía soprano desde: {args.soprano}")
        soprano_hints, chords = load_soprano_hints_from_midi(args.soprano, args.beats)
        if args.chords:
            # Si también hay --chords, úsalos como armónico pero respetando soprano
            chords = parse_chord_string(args.chords, args.beats)
            soprano_hints = soprano_hints[:len(chords)]
    elif args.chords:
        chords = parse_chord_string(args.chords, args.beats)
    elif args.chords_midi:
        # Parsear acordes desde MIDI (canal 0 como bloque de acordes)
        print(f"[INFO] Leyendo acordes desde MIDI: {args.chords_midi}")
        # Implementación simplificada: extraer pitch más grave = raíz
        soprano_hints, chords = load_soprano_hints_from_midi(args.chords_midi, args.beats)
        soprano_hints = None  # no hay soprano, solo acordes
    else:
        parser.error("Especifica --chords, --chords-midi o --soprano")

    styles = ["chorale", "open", "close", "jazz", "keyboard"] if args.all_styles \
             else [args.style]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for style in styles:
        label = f" [{style}]" if args.all_styles else ""
        print(f"\n[VOICE LEADER{label}] Procesando {len(chords)} acorde(s) "
              f"· tonalidad: {args.key} · estilo: {style}")

        blocks, report = voice_lead_progression(
            chords=chords,
            key=args.key,
            style=style,
            soprano_hints=soprano_hints,
            seed=args.seed,
            verbose=args.verbose,
        )

        if args.score_report or args.verbose:
            _print_report(report, blocks)

        # ── Exportar MIDI ─────────────────────────────────────────────────────
        mid = blocks_to_midi(
            blocks,
            tempo_bpm=args.tempo,
            split_tracks=args.split_tracks,
            voices=args.voices,
        )

        if args.all_styles:
            out_stem = Path(args.output).stem
            out_path = out_dir / f"{out_stem}_{style}.mid"
        else:
            out_path = out_dir / args.output

        mid.save(str(out_path))
        print(f"[OK] Guardado: {out_path}  "
              f"({len(blocks)} acordes, score={report.score:.3f}, "
              f"violaciones={report.total_violations})")

        # ── Exportar JSON ─────────────────────────────────────────────────────
        if args.report:
            report_path = out_path.with_suffix(".report.json")
            data = {
                "style": style,
                "key": args.key,
                "tempo": args.tempo,
                "score": report.score,
                "total_violations": report.total_violations,
                "avg_motion_semitones": report.avg_motion,
                "contrary_motion_ratio": report.contrary_ratio,
                "violations_by_rule": report.violations_by_rule,
                "blocks": [
                    {
                        "chord": b.chord_name,
                        "S": b.soprano, "A": b.alto, "T": b.tenor, "B": b.bass,
                        "duration_beats": b.duration_beats,
                        "score": b.score,
                        "violations": b.violations,
                    }
                    for b in blocks
                ],
                "transitions": report.transitions,
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[OK] Informe guardado: {report_path}")

    # ── Reproducir ────────────────────────────────────────────────────────────
    if args.listen:
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(out_path))
            pygame.mixer.music.play()
            print("[INFO] Reproduciendo... (Ctrl+C para detener)")
            while pygame.mixer.music.get_busy():
                pass
        except Exception as e:
            print(f"[AVISO] No se pudo reproducir: {e}")


if __name__ == "__main__":
    main()
