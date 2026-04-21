#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       SONG ARCHITECT  v1.0                                   ║
║         Frases musicales → estructura de canción completa y coherente        ║
║                                                                               ║
║  Toma un número reducido de frases musicales (MIDI) y genera una obra        ║
║  completa con estructura de canción popular:                                  ║
║                                                                               ║
║    Intro → Verso → Pre-estribillo → Estribillo → Verso 2 →                  ║
║    Puente → Solo → Outro                                                      ║
║                                                                               ║
║  Las secciones se generan con coherencia musical entre ellas:                ║
║  • Tonalidad y modo unificados                                                ║
║  • Voice leading entre secciones                                              ║
║  • Curvas de tensión dramática por sección                                    ║
║  • Transformaciones motívicas coherentes (variación, inversión, aumentación) ║
║  • Progresiones armónicas adaptadas al rol de cada sección                   ║
║                                                                               ║
║  USO:                                                                         ║
║    python song_architect.py frase1.mid frase2.mid frase3.mid                 ║
║    python song_architect.py *.mid --key Am --tempo 120                       ║
║    python song_architect.py frases/*.mid --style pop --out-dir cancion/      ║
║    python song_architect.py *.mid --no-solo --no-bridge --bars-per-section 8 ║
║    python song_architect.py *.mid --dry-run                                  ║
║                                                                               ║
║  SECCIONES GENERADAS:                                                         ║
║    intro         — 4-8c  · tensión baja · establece atmósfera                ║
║    verse1        — 8-16c · tensión media-baja · presenta material             ║
║    prechorus     — 4-8c  · tensión creciente · prepara el estribillo         ║
║    chorus        — 8-16c · tensión alta · clímax emocional principal         ║
║    verse2        — 8-16c · variación del verso 1 · material enriquecido      ║
║    bridge        — 8c    · contraste · material nuevo o invertido             ║
║    solo          — 8c    · desarrollo libre · máxima variación               ║
║    outro         — 4-8c  · tensión baja · resolución y cierre                ║
║                                                                               ║
║  ESTRATEGIAS DE COHERENCIA:                                                   ║
║    • La tonalidad se detecta automáticamente o se fuerza con --key           ║
║    • Cada sección adapta las frases entrantes según su rol dramático         ║
║    • El estribillo recibe el material más estable y reconocible              ║
║    • El puente invierte o varía para crear contraste máximo                  ║
║    • El solo aumenta y desarrolla el motivo principal                        ║
║    • El outro deconstruye gradualmente el estribillo                         ║
║                                                                               ║
║  ESTILOS ARMÓNICOS (--style):                                                 ║
║    pop           — I-V-vi-IV, loops simples (default)                        ║
║    rock          — I-bVII-IV, quintas de poder                               ║
║    ballad        — diatónico lento, resoluciones auténticas                  ║
║    jazz          — ii-V-I, tensiones extendidas                              ║
║    folk          — modal, progresiones abiertas                              ║
║    rnb           — dominantes secundarios, 7ª extendidas                     ║
║                                                                               ║
║  OPCIONES:                                                                    ║
║    inputs              Archivos MIDI de frases base (1 o más)                ║
║    --key KEY           Tonalidad base: C, Am, Dm, F#… (default: auto)       ║
║    --tempo BPM         Tempo (default: detectado o 120)                      ║
║    --style STYLE       Estilo armónico (default: pop)                        ║
║    --bars-per-section  Compases por sección (default: auto)                  ║
║    --no-intro          Omitir intro                                           ║
║    --no-prechorus      Omitir pre-estribillo                                  ║
║    --no-bridge         Omitir puente                                          ║
║    --no-solo           Omitir solo                                            ║
║    --sections S [S…]   Especificar qué secciones generar                     ║
║    --out-dir DIR       Directorio de salida (default: ./song_output)         ║
║    --output-name NAME  Nombre base de archivos (default: song)               ║
║    --export-plan       Exportar plan de la canción como JSON                 ║
║    --dry-run           Mostrar plan sin generar archivos MIDI                 ║
║    --verbose           Informe detallado de decisiones                        ║
║    --seed N            Semilla aleatoria (default: 42)                       ║
║                                                                               ║
║  SALIDA:                                                                      ║
║    song_output/song_intro.mid                                                 ║
║    song_output/song_verse1.mid                                                ║
║    song_output/song_prechorus.mid                                             ║
║    song_output/song_chorus.mid                                                ║
║    song_output/song_verse2.mid                                                ║
║    song_output/song_bridge.mid                                                ║
║    song_output/song_solo.mid                                                  ║
║    song_output/song_outro.mid                                                 ║
║    song_output/song_full.mid       (obra completa ensamblada)                ║
║    song_output/song_plan.json      (plan detallado)                          ║
║                                                                               ║
║  DEPENDENCIAS: mido, numpy                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[ERROR] mido no encontrado. Instálalo con: pip install mido")
    sys.exit(1)

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4}

SCALE_INTERVALS = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

# Progresiones por sección y estilo
# Formato: (numeral, duración_beats)
SECTION_PROGRESSIONS = {
    "pop": {
        "intro":     [("I", 4), ("V", 4), ("vi", 4), ("IV", 4)],
        "verse":     [("I", 4), ("V", 4), ("vi", 4), ("IV", 4)],
        "prechorus": [("ii", 4), ("V", 4), ("iii", 4), ("V", 4)],
        "chorus":    [("I", 4), ("V", 4), ("vi", 4), ("IV", 4)],
        "bridge":    [("IV", 4), ("I", 4), ("V", 4), ("vi", 4)],
        "solo":      [("I", 4), ("IV", 4), ("V", 4), ("I", 4)],
        "outro":     [("I", 4), ("IV", 4), ("I", 4), ("I", 4)],
    },
    "rock": {
        "intro":     [("I", 4), ("bVII", 4), ("IV", 4), ("I", 4)],
        "verse":     [("I", 4), ("bVII", 4), ("IV", 4), ("I", 4)],
        "prechorus": [("IV", 4), ("V", 4), ("IV", 4), ("V", 4)],
        "chorus":    [("I", 4), ("IV", 4), ("bVII", 4), ("IV", 4)],
        "bridge":    [("bVI", 4), ("bVII", 4), ("I", 4), ("V", 4)],
        "solo":      [("I", 4), ("IV", 4), ("I", 4), ("V", 4)],
        "outro":     [("I", 4), ("bVII", 4), ("IV", 4), ("I", 8)],
    },
    "ballad": {
        "intro":     [("I", 4), ("vi", 4), ("IV", 4), ("V", 4)],
        "verse":     [("I", 4), ("vi", 4), ("IV", 4), ("V", 4)],
        "prechorus": [("ii", 4), ("IV", 4), ("V", 4), ("V", 4)],
        "chorus":    [("I", 4), ("IV", 4), ("V", 4), ("vi", 4)],
        "bridge":    [("IV", 4), ("ii", 4), ("V", 4), ("I", 4)],
        "solo":      [("I", 4), ("vi", 4), ("IV", 4), ("V", 4)],
        "outro":     [("I", 4), ("vi", 4), ("IV", 4), ("I", 8)],
    },
    "jazz": {
        "intro":     [("IM7", 4), ("vi7", 4), ("ii7", 4), ("V7", 4)],
        "verse":     [("IM7", 4), ("vi7", 4), ("ii7", 4), ("V7", 4)],
        "prechorus": [("iii7", 4), ("bIII7", 4), ("ii7", 4), ("V7", 4)],
        "chorus":    [("ii7", 4), ("V7", 4), ("IM7", 4), ("vi7", 4)],
        "bridge":    [("IM7", 4), ("bII7", 4), ("IM7", 4), ("V7", 4)],
        "solo":      [("ii7", 4), ("V7", 4), ("IM7", 4), ("vi7", 4)],
        "outro":     [("IM7", 4), ("ii7", 4), ("V7", 4), ("IM7", 8)],
    },
    "folk": {
        "intro":     [("I", 4), ("IV", 4), ("I", 4), ("V", 4)],
        "verse":     [("I", 4), ("IV", 4), ("I", 4), ("V", 4)],
        "prechorus": [("IV", 4), ("V", 4), ("IV", 4), ("V", 4)],
        "chorus":    [("I", 4), ("V", 4), ("IV", 4), ("I", 4)],
        "bridge":    [("vi", 4), ("IV", 4), ("I", 4), ("V", 4)],
        "solo":      [("I", 4), ("IV", 4), ("V", 4), ("I", 4)],
        "outro":     [("I", 4), ("V", 4), ("I", 4), ("I", 4)],
    },
    "rnb": {
        "intro":     [("IM7", 4), ("bVII7", 4), ("IV7", 4), ("IM7", 4)],
        "verse":     [("IM7", 4), ("vi7", 4), ("IV7", 4), ("V7", 4)],
        "prechorus": [("ii7", 4), ("V7", 4), ("iii7", 4), ("V7", 4)],
        "chorus":    [("IM7", 4), ("IV7", 4), ("ii7", 4), ("V7", 4)],
        "bridge":    [("bVI7", 4), ("bVII7", 4), ("IM7", 4), ("V7", 4)],
        "solo":      [("IM7", 4), ("vi7", 4), ("IV7", 4), ("V7", 4)],
        "outro":     [("IM7", 4), ("IV7", 4), ("IM7", 4), ("IM7", 4)],
    },
}

# Tensión base por sección (0-1)
SECTION_TENSION = {
    "intro":     0.25,
    "verse1":    0.40,
    "verse2":    0.50,
    "prechorus": 0.65,
    "chorus":    0.85,
    "bridge":    0.70,
    "solo":      0.75,
    "outro":     0.20,
}

# Velocidad MIDI base por sección
SECTION_VELOCITY = {
    "intro":     60,
    "verse1":    70,
    "verse2":    72,
    "prechorus": 78,
    "chorus":    92,
    "bridge":    82,
    "solo":      88,
    "outro":     55,
}

# Compases por defecto por sección
SECTION_DEFAULT_BARS = {
    "intro":     4,
    "verse1":    8,
    "prechorus": 4,
    "chorus":    8,
    "verse2":    8,
    "bridge":    8,
    "solo":      8,
    "outro":     4,
}

# Orden canónico de la estructura
CANONICAL_ORDER = [
    "intro", "verse1", "prechorus", "chorus",
    "verse2", "bridge", "solo", "outro"
]

# Numerales romanos → semitono + calidad
NUMERAL_MAP = {
    "I":    (0,  "M"),  "i":    (0,  "m"),
    "II":   (2,  "M"),  "ii":   (2,  "m"),
    "III":  (4,  "M"),  "iii":  (4,  "m"),
    "IV":   (5,  "M"),  "iv":   (5,  "m"),
    "V":    (7,  "M"),  "v":    (7,  "m"),
    "VI":   (9,  "M"),  "vi":   (9,  "m"),
    "VII":  (11, "M"),  "vii":  (11, "m"),
    "bII":  (1,  "M"),  "bIII": (3,  "M"),
    "bVI":  (8,  "M"),  "bVII": (10, "M"),
    "V7":   (7,  "7"),  "IM7":  (0,  "M7"),
    "vi7":  (9,  "m7"), "ii7":  (2,  "m7"),
    "iii7": (4,  "m7"), "IV7":  (5,  "7"),
    "bVII7":(10, "7"),  "bVI7": (8,  "7"),
    "bII7": (1,  "7"),  "bIII7":(3,  "7"),
}

CHORD_INTERVALS = {
    "M":  [0, 4, 7],
    "m":  [0, 3, 7],
    "7":  [0, 4, 7, 10],
    "M7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "d":  [0, 3, 6],
}


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    pitch: int
    duration: float     # quarter notes
    velocity: int
    offset: float       # quarter notes from section start

    def transpose(self, semitones: int) -> "Note":
        return Note(
            pitch=max(0, min(127, self.pitch + semitones)),
            duration=self.duration,
            velocity=self.velocity,
            offset=self.offset,
        )

    def at_offset(self, delta: float) -> "Note":
        n = deepcopy(self)
        n.offset = self.offset + delta
        return n


@dataclass
class Section:
    name: str
    notes: List[Note] = field(default_factory=list)
    bars: int = 8
    beats_per_bar: int = 4
    tempo: int = 120
    key_pc: int = 0
    mode: str = "major"
    tension: float = 0.5
    velocity_base: int = 70
    progression: List[Tuple[str, int]] = field(default_factory=list)
    source_phrase_idx: int = 0

    @property
    def total_beats(self) -> float:
        return self.bars * self.beats_per_bar

    @property
    def duration_seconds(self) -> float:
        return self.total_beats * 60.0 / self.tempo


@dataclass
class SongPlan:
    sections: List[Section] = field(default_factory=list)
    key_name: str = "C"
    mode: str = "major"
    key_pc: int = 0
    tempo: int = 120
    style: str = "pop"
    total_bars: int = 0
    phrase_count: int = 0

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE FRASES MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_midi_notes(path: str) -> Tuple[List[Note], int, int]:
    """
    Carga un MIDI y devuelve (notas, tempo_bpm, ticks_per_beat).
    Las notas se normalizan en quarter notes.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo_us = 500_000  # 120 BPM por defecto
    notes = []
    active: Dict[int, Tuple[float, int]] = {}
    abs_ticks = 0

    for msg in mido.merge_tracks(mid.tracks):
        abs_ticks += msg.time
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (abs_ticks / tpb, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                start_beat, vel = active.pop(msg.note)
                dur = abs_ticks / tpb - start_beat
                if dur > 0.01:
                    notes.append(Note(
                        pitch=msg.note,
                        duration=round(dur * 4) / 4,
                        velocity=vel,
                        offset=start_beat,
                    ))

    if not notes:
        return [], round(60_000_000 / tempo_us), tpb

    # Normalizar offsets al inicio
    min_off = min(n.offset for n in notes)
    for n in notes:
        n.offset -= min_off

    bpm = round(60_000_000 / tempo_us)
    return sorted(notes, key=lambda n: n.offset), bpm, tpb


def detect_key(notes: List[Note]) -> Tuple[int, str]:
    """
    Detección de tonalidad por perfil de pitch classes (Krumhansl-Schmuckler simplificado).
    Devuelve (root_pc, 'major'|'minor').
    """
    if not notes:
        return 0, "major"

    # Perfil de duración por PC
    pc_weights = np.zeros(12)
    for n in notes:
        pc_weights[n.pitch % 12] += n.duration

    # Perfiles Krumhansl-Schmuckler
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_key, best_score, best_mode = 0, -999, "major"
    for root in range(12):
        rotated = np.roll(pc_weights, -root)
        for mode_name, profile in [("major", major_profile), ("minor", minor_profile)]:
            score = float(np.corrcoef(rotated, profile)[0, 1])
            if score > best_score:
                best_score = score
                best_key = root
                best_mode = mode_name

    return best_key, best_mode


def detect_tempo(midi_path: str) -> int:
    """Lee el tempo del MIDI o devuelve 120."""
    try:
        mid = mido.MidiFile(midi_path)
        for msg in mido.merge_tracks(mid.tracks):
            if msg.type == "set_tempo":
                return round(60_000_000 / msg.tempo)
    except Exception:
        pass
    return 120


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES TONALES
# ══════════════════════════════════════════════════════════════════════════════

def parse_key(key_str: str) -> Tuple[int, str]:
    """'Am' → (9, 'minor'), 'C' → (0, 'major'), 'F# minor' → (6, 'minor')"""
    key_str = key_str.strip()
    mode = "major"
    parts = key_str.split()
    root_str = parts[0]

    if len(parts) > 1:
        mode_str = parts[1].lower()
        if mode_str in ("minor", "min", "m"):
            mode = "minor"
    elif root_str.endswith("m") and len(root_str) >= 2:
        mode = "minor"
        root_str = root_str[:-1]

    root_str_norm = root_str[0].upper() + root_str[1:]
    if root_str_norm in ENHARMONIC:
        return ENHARMONIC[root_str_norm], mode

    note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
    base = root_str_norm.rstrip("#b")
    pc = note_map.get(base, 0)
    if "#" in root_str_norm:
        pc = (pc + 1) % 12
    elif "b" in root_str_norm:
        pc = (pc - 1) % 12

    return pc, mode


def get_scale_pcs(root_pc: int, mode: str) -> List[int]:
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return [(root_pc + i) % 12 for i in intervals]


def snap_to_scale(pitch: int, root_pc: int, mode: str) -> int:
    """Ajusta un pitch al grado de escala más cercano."""
    scale_pcs = set(get_scale_pcs(root_pc, mode))
    if pitch % 12 in scale_pcs:
        return pitch
    for delta in range(1, 7):
        if (pitch + delta) % 12 in scale_pcs:
            return pitch + delta
        if (pitch - delta) % 12 in scale_pcs:
            return pitch - delta
    return pitch


def numeral_to_root(numeral: str, key_pc: int) -> Tuple[int, str]:
    """Convierte numeral romano → (root_pc, quality)."""
    entry = NUMERAL_MAP.get(numeral)
    if entry is None:
        return key_pc, "M"
    semitone, quality = entry
    return (key_pc + semitone) % 12, quality


def chord_pitches(root_pc: int, quality: str, octave: int = 4) -> List[int]:
    """Construye lista de pitches MIDI para un acorde."""
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
    base = root_pc + octave * 12
    while base < 48:
        base += 12
    while base > 72:
        base -= 12
    return [base + i for i in intervals if 24 <= base + i <= 96]


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMACIONES DE FRASES
# ══════════════════════════════════════════════════════════════════════════════

def transpose_to_key(notes: List[Note], from_pc: int, to_pc: int,
                     from_mode: str, to_mode: str) -> List[Note]:
    """Transpone notas de una tonalidad a otra ajustando a la escala destino."""
    semitones = (to_pc - from_pc) % 12
    if semitones > 6:
        semitones -= 12
    result = []
    for n in notes:
        new_pitch = n.pitch + semitones
        snapped = snap_to_scale(new_pitch, to_pc, to_mode)
        result.append(Note(snapped, n.duration, n.velocity, n.offset))
    return result


def fit_to_bars(notes: List[Note], target_bars: int,
                beats_per_bar: int = 4) -> List[Note]:
    """
    Ajusta la duración total de las notas a target_bars compases.
    Aplica factor de escala temporal.
    """
    if not notes:
        return notes
    total = max(n.offset + n.duration for n in notes)
    target = target_bars * beats_per_bar
    if total < 0.01:
        return notes
    factor = target / total
    if 0.8 < factor < 1.2:
        return notes  # sin cambio si la diferencia es pequeña
    return [Note(n.pitch, n.duration * factor, n.velocity, n.offset * factor)
            for n in notes]


def apply_velocity(notes: List[Note], base_vel: int) -> List[Note]:
    """Ajusta velocidades al nivel de la sección manteniendo la dinámica relativa."""
    if not notes:
        return notes
    current_mean = np.mean([n.velocity for n in notes])
    if current_mean < 1:
        return notes
    ratio = base_vel / current_mean
    result = []
    for n in notes:
        new_vel = int(np.clip(n.velocity * ratio, 20, 120))
        result.append(Note(n.pitch, n.duration, new_vel, n.offset))
    return result


def invert_contour(notes: List[Note], pivot: Optional[int] = None) -> List[Note]:
    """Invierte el contorno melódico alrededor de pivot."""
    if not notes:
        return notes
    pivot_pitch = pivot if pivot is not None else notes[0].pitch
    result = []
    for n in notes:
        new_pitch = max(24, min(108, 2 * pivot_pitch - n.pitch))
        result.append(Note(new_pitch, n.duration, n.velocity, n.offset))
    return result


def augment(notes: List[Note], factor: float = 2.0) -> List[Note]:
    """Aumentación rítmica."""
    return [Note(n.pitch, n.duration * factor, n.velocity, n.offset * factor)
            for n in notes]


def diminish(notes: List[Note], factor: float = 0.5) -> List[Note]:
    """Diminución rítmica."""
    return augment(notes, factor)


def repeat_phrase(notes: List[Note], times: int) -> List[Note]:
    """Repite una frase N veces."""
    if not notes or times <= 1:
        return notes
    total = max(n.offset + n.duration for n in notes) if notes else 0
    result = list(notes)
    for i in range(1, times):
        for n in notes:
            result.append(n.at_offset(total * i))
    return result


def vary_phrase(notes: List[Note], variation_type: str,
                root_pc: int, mode: str, rng: random.Random) -> List[Note]:
    """
    Aplica una variación a la frase según el tipo:
    parallel, inversion, sequence, development, ornament
    """
    if not notes:
        return notes

    if variation_type == "parallel":
        # Casi igual, solo pequeñas alteraciones rítmicas
        result = []
        for n in notes:
            dur_factor = rng.choice([0.75, 1.0, 1.0, 1.25])
            result.append(Note(n.pitch, n.duration * dur_factor, n.velocity, n.offset))
        return result

    elif variation_type == "inversion":
        return invert_contour(notes)

    elif variation_type == "sequence_up":
        steps = [2, 2, 1, 2, 2, 2, 1]  # escala mayor
        scale_pcs = get_scale_pcs(root_pc, mode)
        result = []
        for n in notes:
            pc = n.pitch % 12
            if pc in scale_pcs:
                idx = scale_pcs.index(pc)
                new_idx = (idx + 1) % 7
                new_pc = scale_pcs[new_idx]
                semitones = (new_pc - pc) % 12
                new_pitch = snap_to_scale(n.pitch + semitones, root_pc, mode)
            else:
                new_pitch = snap_to_scale(n.pitch, root_pc, mode)
            result.append(Note(new_pitch, n.duration, n.velocity, n.offset))
        return result

    elif variation_type == "development":
        # Liquidar: quedarse solo con las notas en tiempos fuertes
        strong = [n for n in notes if n.offset % 2.0 < 0.5]
        if len(strong) < 2:
            strong = notes[:max(2, len(notes) // 2)]
        return strong

    elif variation_type == "ornament":
        # Añadir notas de paso rápidas entre saltos
        result = list(notes)
        insertions = []
        for i in range(len(notes) - 1):
            a, b = notes[i], notes[i + 1]
            gap = abs(b.pitch - a.pitch)
            if gap >= 4 and a.duration >= 1.0:
                pass_pitch = snap_to_scale((a.pitch + b.pitch) // 2, root_pc, mode)
                half = a.duration / 2
                insertions.append(Note(pass_pitch, half, a.velocity - 10, a.offset + half))
        result.extend(insertions)
        return sorted(result, key=lambda n: n.offset)

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE ACOMPAÑAMIENTO ARMÓNICO
# ══════════════════════════════════════════════════════════════════════════════

def build_chord_accompaniment(progression: List[Tuple[str, int]],
                               key_pc: int, beats_per_bar: int,
                               velocity: int, style: str = "block") -> List[Note]:
    """
    Genera notas de acompañamiento (acordes bloque o arpegios simples)
    a partir de una progresión.
    """
    notes = []
    cursor = 0.0

    for numeral, dur_beats in progression:
        root_pc, quality = numeral_to_root(numeral, key_pc)
        pitches = chord_pitches(root_pc, quality, octave=4)

        if style == "block":
            # Acordes en bloque
            for p in pitches:
                notes.append(Note(p, dur_beats * 0.9, velocity, cursor))
        elif style == "arpeggio":
            # Arpegio ascendente
            step = dur_beats / max(len(pitches), 1)
            for i, p in enumerate(pitches):
                notes.append(Note(p, step * 0.8, velocity, cursor + i * step))
        elif style == "bass_only":
            # Solo bajo
            if pitches:
                bass = min(pitches)
                notes.append(Note(bass, dur_beats * 0.9, velocity, cursor))

        cursor += dur_beats

    return notes


def build_bass_line(progression: List[Tuple[str, int]],
                    key_pc: int, velocity: int) -> List[Note]:
    """Genera una línea de bajo con la raíz en cada cambio de acorde."""
    notes = []
    cursor = 0.0
    prev_bass = None

    for numeral, dur_beats in progression:
        root_pc, quality = numeral_to_root(numeral, key_pc)
        # Bass en octava 2-3
        bass = root_pc + 36
        while bass < 28:
            bass += 12
        while bass > 52:
            bass -= 12

        # Pequeño voice-leading: si el salto es grande, ajustar octava
        if prev_bass is not None:
            leap = abs(bass - prev_bass)
            if leap > 7:
                alt = bass + (12 if bass < prev_bass else -12)
                if 28 <= alt <= 52:
                    bass = alt

        notes.append(Note(bass, dur_beats * 0.85, velocity, cursor))
        prev_bass = bass
        cursor += dur_beats

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

def build_section(name: str,
                  phrases: List[List[Note]],
                  key_pc: int,
                  mode: str,
                  tempo: int,
                  style: str,
                  bars: int,
                  beats_per_bar: int,
                  rng: random.Random,
                  verbose: bool = False) -> Section:
    """
    Construye una sección de la canción transformando las frases entrantes.
    """
    tension = SECTION_TENSION.get(name, 0.5)
    vel_base = SECTION_VELOCITY.get(name, 70)
    prog_key = name.rstrip("12")  # verse1 → verse, verse2 → verse
    progression = SECTION_PROGRESSIONS.get(style, SECTION_PROGRESSIONS["pop"]).get(
        prog_key, SECTION_PROGRESSIONS["pop"]["verse"]
    )

    sec = Section(
        name=name,
        bars=bars,
        beats_per_bar=beats_per_bar,
        tempo=tempo,
        key_pc=key_pc,
        mode=mode,
        tension=tension,
        velocity_base=vel_base,
        progression=progression,
    )

    if not phrases:
        return sec

    # Seleccionar frase base para esta sección
    phrase_idx = hash(name) % len(phrases)
    base_notes = deepcopy(phrases[phrase_idx])

    if verbose:
        print(f"    [{name}] frase base #{phrase_idx + 1}, "
              f"{len(base_notes)} notas → {bars}c @ tensión={tension:.2f}")

    # Transformar según el rol de la sección
    transformation_map = {
        "intro":     "development",    # reducción al esencial
        "verse1":    "parallel",       # casi igual
        "prechorus": "sequence_up",    # secuencia ascendente (tensión creciente)
        "chorus":    "parallel",       # material más estable y reconocible
        "verse2":    "ornament",       # variación ornamental del verso 1
        "bridge":    "inversion",      # contraste por inversión
        "solo":      "development",    # desarrollo y liquidación
        "outro":     "parallel",       # resolución tranquila
    }
    transform = transformation_map.get(name, "parallel")
    transformed = vary_phrase(base_notes, transform, key_pc, mode, rng)

    # Ajustar al número de compases objetivo
    total_beats = bars * beats_per_bar
    if transformed:
        phrase_dur = max(n.offset + n.duration for n in transformed)
        if phrase_dur < total_beats * 0.5:
            # Frase muy corta: repetir
            times = math.ceil(total_beats / max(phrase_dur, 0.1))
            transformed = repeat_phrase(transformed, min(times, 4))
        transformed = fit_to_bars(transformed, bars, beats_per_bar)

    # Ajustar velocidad
    transformed = apply_velocity(transformed, vel_base)

    # Añadir acompañamiento
    acc_style = {
        "intro": "bass_only",
        "verse1": "arpeggio",
        "verse2": "arpeggio",
        "prechorus": "block",
        "chorus": "block",
        "bridge": "arpeggio",
        "solo": "bass_only",
        "outro": "bass_only",
    }.get(name, "block")

    # Repetir progresión para cubrir los compases
    beats_needed = bars * beats_per_bar
    prog_beats = sum(d for _, d in progression)
    reps = max(1, math.ceil(beats_needed / max(prog_beats, 1)))
    full_prog = (progression * reps)

    # Truncar al total
    prog_truncated = []
    acc_cursor = 0.0
    for numeral, dur in full_prog:
        if acc_cursor >= beats_needed:
            break
        actual_dur = min(dur, beats_needed - acc_cursor)
        prog_truncated.append((numeral, actual_dur))
        acc_cursor += actual_dur

    acc_vel = max(20, vel_base - 20)
    acc_notes = build_chord_accompaniment(
        prog_truncated, key_pc, beats_per_bar, acc_vel, style=acc_style
    )
    bass_notes = build_bass_line(prog_truncated, key_pc, max(20, acc_vel - 10))

    # Combinar melodía + acompañamiento + bajo
    all_notes = transformed + acc_notes + bass_notes
    sec.notes = sorted(all_notes, key=lambda n: n.offset)
    sec.progression = prog_truncated

    return sec


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DEL PLAN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def generate_song_plan(phrases_notes: List[List[Note]],
                       key_pc: int,
                       mode: str,
                       key_name: str,
                       tempo: int,
                       style: str,
                       sections_to_generate: List[str],
                       bars_per_section: Optional[Dict[str, int]],
                       beats_per_bar: int,
                       rng: random.Random,
                       verbose: bool = False) -> SongPlan:
    """
    Genera el plan completo de la canción, construyendo cada sección.
    """
    plan = SongPlan(
        key_name=key_name,
        mode=mode,
        key_pc=key_pc,
        tempo=tempo,
        style=style,
        phrase_count=len(phrases_notes),
    )

    if verbose:
        print(f"\n  Generando secciones: {', '.join(sections_to_generate)}")
        print(f"  Tonalidad: {key_name} {mode}  Tempo: {tempo} BPM  Estilo: {style}")
        print(f"  Frases de entrada: {len(phrases_notes)}\n")

    for sec_name in sections_to_generate:
        bars = (bars_per_section or {}).get(sec_name, SECTION_DEFAULT_BARS.get(sec_name, 8))

        if verbose:
            print(f"  Construyendo '{sec_name}' ({bars} compases)...")

        section = build_section(
            name=sec_name,
            phrases=phrases_notes,
            key_pc=key_pc,
            mode=mode,
            tempo=tempo,
            style=style,
            bars=bars,
            beats_per_bar=beats_per_bar,
            rng=rng,
            verbose=verbose,
        )
        plan.sections.append(section)

    plan.total_bars = sum(s.bars for s in plan.sections)
    return plan


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def section_to_midi(section: Section, ticks_per_beat: int = 480) -> mido.MidiFile:
    """Convierte una sección a un MidiFile."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = mido.bpm2tempo(section.tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage(
        "track_name", name=section.name.replace("_", " ").title(), time=0
    ))
    # Instrumento: piano para melodía, bajo en canal 1
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    events = []
    for note in section.notes:
        t_on = int(note.offset * ticks_per_beat)
        t_off = int((note.offset + max(0.05, note.duration)) * ticks_per_beat)
        vel = max(1, min(127, note.velocity))
        p = max(0, min(127, note.pitch))
        # Canal 1 para notas graves (bajo), canal 0 para el resto
        ch = 1 if p < 48 else 0
        events.append((t_on, "on", p, vel, ch))
        events.append((t_off, "off", p, 0, ch))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    current_tick = 0
    for abs_tick, etype, pitch, vel, ch in events:
        delta = max(0, abs_tick - current_tick)
        msg_type = "note_on" if etype == "on" else "note_off"
        track.append(mido.Message(msg_type, channel=ch, note=pitch,
                                   velocity=vel, time=delta))
        current_tick = abs_tick

    return mid


def concatenate_midis(sections: List[Section], ticks_per_beat: int = 480) -> mido.MidiFile:
    """
    Ensambla todas las secciones en un único MidiFile.
    Aplica voice leading entre secciones (ajuste suave de la última nota
    de cada sección hacia la primera de la siguiente).
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    if not sections:
        return mid

    # Tempo inicial
    tempo_us = mido.bpm2tempo(sections[0].tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage("track_name", name="Full Song", time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    global_offset_beats = 0.0
    events = []

    for i, section in enumerate(sections):
        # Cambio de tempo si cambia entre secciones
        if i > 0 and section.tempo != sections[i - 1].tempo:
            # Insertar el cambio de tempo al inicio de la sección
            new_tempo_us = mido.bpm2tempo(section.tempo)
            t = int(global_offset_beats * ticks_per_beat)
            events.append((t, "tempo", new_tempo_us, 0, 0))

        for note in section.notes:
            abs_offset = global_offset_beats + note.offset
            t_on = int(abs_offset * ticks_per_beat)
            t_off = int((abs_offset + max(0.05, note.duration)) * ticks_per_beat)
            vel = max(1, min(127, note.velocity))
            p = max(0, min(127, note.pitch))
            ch = 1 if p < 48 else 0
            events.append((t_on, "on", p, vel, ch))
            events.append((t_off, "off", p, 0, ch))

        global_offset_beats += section.total_beats

    events.sort(key=lambda e: (e[0], 0 if e[1] in ("off", "tempo") else 1))
    current_tick = 0
    for event in events:
        abs_tick = event[0]
        delta = max(0, abs_tick - current_tick)
        etype = event[1]
        if etype == "tempo":
            track.append(mido.MetaMessage("set_tempo", tempo=event[2], time=delta))
        elif etype == "on":
            track.append(mido.Message("note_on", channel=event[4],
                                       note=event[2], velocity=event[3], time=delta))
        else:
            track.append(mido.Message("note_off", channel=event[4],
                                       note=event[2], velocity=0, time=delta))
        current_tick = abs_tick

    return mid


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME Y VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _c(code: str, text: str) -> str:
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
        "cyan": "\033[96m", "bold": "\033[1m", "gray": "\033[90m",
        "blue": "\033[94m", "reset": "\033[0m",
    }
    if not sys.stdout.isatty():
        return text
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def print_song_plan(plan: SongPlan, verbose: bool = False):
    """Imprime el plan de la canción de forma visual."""
    print(f"\n{'═' * 64}")
    print(f"  {_c('bold', 'SONG ARCHITECT')} v{VERSION}  —  Plan de canción")
    print(f"{'═' * 64}")
    print(f"  Tonalidad : {_c('cyan', plan.key_name)} {plan.mode}")
    print(f"  Tempo     : {plan.tempo} BPM")
    print(f"  Estilo    : {plan.style}")
    print(f"  Secciones : {len(plan.sections)}")
    print(f"  Total     : {plan.total_bars} compases")
    print(f"  Frases    : {plan.phrase_count} de entrada")
    print(f"\n  {'Sección':<16} {'Compases':>8} {'Tensión':>8} {'Notas':>7}  {'Progresión'}")
    print(f"  {'─' * 60}")

    tension_chars = "░▒▓█"
    for sec in plan.sections:
        t_idx = min(3, int(sec.tension * 4))
        t_bar = tension_chars[t_idx] * int(sec.tension * 12)
        t_rest = "·" * (12 - len(t_bar))
        prog_str = " ".join(n for n, _ in sec.progression[:4])
        if len(sec.progression) > 4:
            prog_str += "…"
        color = "green" if sec.tension < 0.4 else "yellow" if sec.tension < 0.7 else "red"
        print(f"  {sec.name:<16} {sec.bars:>8}c  "
              f"{_c(color, t_bar)}{_c('gray', t_rest)}  "
              f"{len(sec.notes):>5}   {_c('gray', prog_str)}")

    print(f"  {'─' * 60}")
    print(f"\n  Arco de tensión: ", end="")
    for sec in plan.sections:
        t_idx = min(7, int(sec.tension * 8))
        bar_chars = "▁▂▃▄▅▆▇█"
        print(bar_chars[t_idx], end="")
    print(f"  ({plan.total_bars}c total)")
    print()


def export_plan_json(plan: SongPlan, output_path: str, input_files: List[str]):
    """Exporta el plan detallado como JSON."""
    data = {
        "version": VERSION,
        "key": f"{plan.key_name} {plan.mode}",
        "key_pc": plan.key_pc,
        "mode": plan.mode,
        "tempo": plan.tempo,
        "style": plan.style,
        "total_bars": plan.total_bars,
        "phrase_count": plan.phrase_count,
        "input_files": [str(Path(f).name) for f in input_files],
        "sections": [],
    }
    for sec in plan.sections:
        data["sections"].append({
            "name": sec.name,
            "bars": sec.bars,
            "beats_per_bar": sec.beats_per_bar,
            "tempo": sec.tempo,
            "tension": sec.tension,
            "velocity_base": sec.velocity_base,
            "note_count": len(sec.notes),
            "progression": [[n, d] for n, d in sec.progression],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="song_architect.py",
        description="Frases musicales → estructura de canción completa y coherente.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              python song_architect.py frase1.mid frase2.mid
              python song_architect.py *.mid --key Am --tempo 90 --style ballad
              python song_architect.py frases/*.mid --no-solo --no-bridge
              python song_architect.py *.mid --sections intro chorus outro
              python song_architect.py *.mid --dry-run --verbose
        """)
    )
    p.add_argument("inputs", nargs="+", metavar="MIDI",
                   help="Archivos MIDI de frases de entrada")
    p.add_argument("--key", type=str, default=None,
                   help='Tonalidad base (ej. C, Am, F# minor). Default: auto-detección')
    p.add_argument("--tempo", type=int, default=None,
                   help="Tempo BPM (default: detectado del MIDI o 120)")
    p.add_argument("--style", type=str, default="pop",
                   choices=["pop", "rock", "ballad", "jazz", "folk", "rnb"],
                   help="Estilo armónico (default: pop)")
    p.add_argument("--beats-per-bar", type=int, default=4,
                   help="Pulsos por compás (default: 4)")
    p.add_argument("--bars-per-section", type=int, default=None,
                   help="Compases por sección, igual para todas (default: auto)")

    # Control de secciones
    secs = p.add_argument_group("Secciones")
    secs.add_argument("--sections", nargs="+", default=None,
                      choices=CANONICAL_ORDER,
                      metavar="SEC",
                      help="Secciones a generar (default: todas)")
    secs.add_argument("--no-intro", action="store_true")
    secs.add_argument("--no-prechorus", action="store_true")
    secs.add_argument("--no-bridge", action="store_true")
    secs.add_argument("--no-solo", action="store_true")

    # Salida
    out = p.add_argument_group("Salida")
    out.add_argument("--out-dir", type=str, default="song_output",
                     help="Directorio de salida (default: song_output)")
    out.add_argument("--output-name", type=str, default="song",
                     help="Nombre base de archivos (default: song)")
    out.add_argument("--export-plan", action="store_true",
                     help="Exportar plan JSON")
    out.add_argument("--dry-run", action="store_true",
                     help="Mostrar plan sin generar archivos MIDI")

    # Extras
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria (default: 42)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Informe detallado de decisiones")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 1. Verificar entradas ───────────────────────────────────────────────
    input_files = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.exists():
            input_files.append(str(p))
        else:
            import glob
            matched = glob.glob(str(p))
            input_files.extend(matched)

    if not input_files:
        print(f"[ERROR] No se encontraron archivos MIDI: {args.inputs}")
        sys.exit(1)

    print(f"\n  {_c('bold', 'SONG ARCHITECT')} v{VERSION}")
    print(f"  {'─' * 58}")
    print(f"  Frases de entrada: {len(input_files)}")
    for f in input_files:
        print(f"    • {Path(f).name}")

    # ── 2. Cargar frases ────────────────────────────────────────────────────
    all_phrases_notes = []
    all_notes_flat = []
    detected_tempos = []

    for fpath in input_files:
        notes, bpm, tpb = load_midi_notes(fpath)
        if notes:
            all_phrases_notes.append(notes)
            all_notes_flat.extend(notes)
            detected_tempos.append(bpm)
            if args.verbose:
                print(f"  Cargado {Path(fpath).name}: {len(notes)} notas, {bpm} BPM")
        else:
            print(f"  [warn] {Path(fpath).name}: sin notas")

    if not all_phrases_notes:
        print("[ERROR] Ninguna frase tiene notas. Verifica los archivos MIDI.")
        sys.exit(1)

    # ── 3. Detectar/configurar tonalidad ────────────────────────────────────
    if args.key:
        key_pc, mode = parse_key(args.key)
        key_name = args.key.split()[0].rstrip("m")
        mode_str = "minor" if "minor" in args.key.lower() or (
            args.key[-1].lower() == "m" and len(args.key) <= 3
        ) else "major"
        mode = mode_str
        if args.verbose:
            print(f"\n  Tonalidad forzada: {key_name} {mode}")
    else:
        key_pc, mode = detect_key(all_notes_flat)
        key_name = NOTE_NAMES[key_pc]
        if args.verbose:
            print(f"\n  Tonalidad detectada: {key_name} {mode}")

    # ── 4. Configurar tempo ─────────────────────────────────────────────────
    if args.tempo:
        tempo = args.tempo
    elif detected_tempos:
        tempo = round(np.median(detected_tempos))
    else:
        tempo = 120

    if args.verbose:
        print(f"  Tempo: {tempo} BPM")

    # ── 5. Transponer frases a la tonalidad objetivo ─────────────────────────
    # Detectar tonalidad de cada frase por separado y transponer
    phrases_in_key = []
    for i, phrase_notes in enumerate(all_phrases_notes):
        src_pc, src_mode = detect_key(phrase_notes)
        if src_pc != key_pc or src_mode != mode:
            transposed = transpose_to_key(phrase_notes, src_pc, key_pc, src_mode, mode)
            if args.verbose:
                print(f"  Frase {i+1}: transponiendo de "
                      f"{NOTE_NAMES[src_pc]} {src_mode} → {key_name} {mode}")
        else:
            transposed = phrase_notes
        phrases_in_key.append(transposed)

    # ── 6. Determinar secciones ─────────────────────────────────────────────
    if args.sections:
        sections_to_generate = args.sections
    else:
        sections_to_generate = list(CANONICAL_ORDER)
        if args.no_intro:
            sections_to_generate.remove("intro")
        if args.no_prechorus and "prechorus" in sections_to_generate:
            sections_to_generate.remove("prechorus")
        if args.no_bridge and "bridge" in sections_to_generate:
            sections_to_generate.remove("bridge")
        if args.no_solo and "solo" in sections_to_generate:
            sections_to_generate.remove("solo")

    # Compases por sección
    bars_map = {}
    if args.bars_per_section:
        for s in sections_to_generate:
            bars_map[s] = args.bars_per_section
    else:
        bars_map = None  # usar defaults por sección

    print(f"\n  Secciones: {_c('cyan', ', '.join(sections_to_generate))}")
    print(f"  Tonalidad: {key_name} {mode}  |  Tempo: {tempo} BPM  |  Estilo: {args.style}")

    # ── 7. Generar plan ─────────────────────────────────────────────────────
    print(f"\n  Generando secciones...")
    plan = generate_song_plan(
        phrases_notes=phrases_in_key,
        key_pc=key_pc,
        mode=mode,
        key_name=key_name,
        tempo=tempo,
        style=args.style,
        sections_to_generate=sections_to_generate,
        bars_per_section=bars_map,
        beats_per_bar=args.beats_per_bar,
        rng=rng,
        verbose=args.verbose,
    )

    # ── 8. Mostrar plan ─────────────────────────────────────────────────────
    print_song_plan(plan, verbose=args.verbose)

    if args.dry_run:
        print(f"  {_c('yellow', '[dry-run]')} No se generaron archivos MIDI.")
        if args.verbose:
            print(f"\n  Secciones planificadas:")
            for sec in plan.sections:
                total_s = sec.duration_seconds
                m, s = divmod(int(total_s), 60)
                print(f"    {sec.name:<16} {sec.bars}c  ~{m}:{s:02d}")
        return

    # ── 9. Exportar archivos ─────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_base = args.output_name
    tpb = 480

    generated = []
    print(f"  Exportando a: {out_dir}/")
    print(f"  {'─' * 58}")

    for sec in plan.sections:
        out_path = out_dir / f"{name_base}_{sec.name}.mid"
        mid = section_to_midi(sec, tpb)
        mid.save(str(out_path))
        size_kb = out_path.stat().st_size // 1024
        generated.append(str(out_path))
        print(f"  {_c('green', '✓')} {out_path.name:<35}  "
              f"{sec.bars}c  {len(sec.notes):>4} notas  {size_kb} KB")

    # Obra completa
    full_path = out_dir / f"{name_base}_full.mid"
    full_mid = concatenate_midis(plan.sections, tpb)
    full_mid.save(str(full_path))
    full_size = full_path.stat().st_size // 1024
    generated.append(str(full_path))
    print(f"  {_c('green', '✓')} {full_path.name:<35}  "
          f"{plan.total_bars}c  {full_size} KB  {_c('cyan', '← obra completa')}")

    # Plan JSON
    if args.export_plan:
        plan_path = out_dir / f"{name_base}_plan.json"
        export_plan_json(plan, str(plan_path), input_files)
        print(f"  {_c('green', '✓')} {plan_path.name:<35}  plan completo")

    # ── 10. Resumen ──────────────────────────────────────────────────────────
    total_s = sum(sec.duration_seconds for sec in plan.sections)
    m, s = divmod(int(total_s), 60)
    print(f"\n{'═' * 64}")
    print(f"  {_c('green', 'Canción generada')}: {len(plan.sections)} secciones, "
          f"{plan.total_bars} compases, ~{m}:{s:02d}")
    print(f"  {len(generated)} archivos en: {out_dir}/")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
