#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       SONG ARCHITECT  v2.0                                   ║
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
║  • Voice leading entre secciones y entre acordes                             ║
║  • Curvas de tensión dramática por sección                                    ║
║  • Transformaciones motívicas coherentes (variación, inversión, aumentación) ║
║  • Progresiones armónicas adaptadas al rol de cada sección                   ║
║  • Selección inteligente de fragmentos por interés melódico                  ║
║  • Manejo correcto de MIDIs multi-track                                      ║
║  • Deduplicación de notas superpuestas                                       ║
║  • Detección automática de compás                                            ║
║  • Repeticiones con micro-variaciones                                        ║
║  • Exportación MusicXML (MuseScore/Sibelius)                                 ║
║                                                                               ║
║  USO:                                                                         ║
║    python song_architect.py frase1.mid frase2.mid frase3.mid                 ║
║    python song_architect.py *.mid --key Am --tempo 120                       ║
║    python song_architect.py frases/*.mid --style pop --out-dir cancion/      ║
║    python song_architect.py *.mid --no-solo --no-bridge --bars-per-section 8 ║
║    python song_architect.py *.mid --dry-run                                  ║
║    python song_architect.py *.mid --split-tracks --melody-track 1           ║
║    python song_architect.py *.mid --fragment-start 8 --fragment-end 24      ║
║    python song_architect.py *.mid --export-musicxml                          ║
║                                                                               ║
║  OPCIONES NUEVAS (v2.0):                                                      ║
║    --split-tracks       Tratar cada track MIDI como frase independiente      ║
║    --melody-track N     Índice del track melódico (default: auto)            ║
║    --fragment-start C   Compás de inicio del fragmento (1-based)             ║
║    --fragment-end C     Compás de fin del fragmento (1-based, inclusivo)     ║
║    --export-musicxml    Exportar también en MusicXML (.musicxml)             ║
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

VERSION = "2.0"

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

# Numerador de compás → beats por compás
TIME_SIG_BEATS = {2: 2, 3: 3, 4: 4, 6: 6, 9: 9, 12: 12}

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

CANONICAL_ORDER = [
    "intro", "verse1", "prechorus", "chorus",
    "verse2", "bridge", "solo", "outro"
]

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

# Duración mínima absoluta de nota en beats (evita artefactos de audio)
MIN_NOTE_DURATION = 0.125


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    pitch: int
    duration: float
    velocity: int
    offset: float

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
#  CARGA MIDI  (multi-track + detección de compás)
# ══════════════════════════════════════════════════════════════════════════════

def load_midi_tracks(path: str) -> Tuple[List[List[Note]], int, int, int]:
    """
    Carga un MIDI y devuelve (tracks_notes, tempo_bpm, ticks_per_beat, beats_per_bar).
    Cada elemento de tracks_notes contiene las notas de un track independiente.
    Detecta el compás automáticamente desde mensajes time_signature.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo_us = 500_000
    beats_per_bar = 4

    # Leer meta-mensajes globales del primer track
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
        elif msg.type == "time_signature":
            beats_per_bar = TIME_SIG_BEATS.get(msg.numerator, msg.numerator)

    tracks_notes: List[List[Note]] = []
    for track in mid.tracks:
        notes = _parse_track_notes(track, tpb)
        if notes:
            tracks_notes.append(notes)

    bpm = round(60_000_000 / tempo_us)
    return tracks_notes, bpm, tpb, beats_per_bar


def _parse_track_notes(track, tpb: int) -> List[Note]:
    """Extrae y normaliza notas de un único MidiTrack."""
    active: Dict[int, Tuple[float, int]] = {}
    notes = []
    abs_ticks = 0

    for msg in track:
        abs_ticks += msg.time
        beat = abs_ticks / tpb

        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (beat, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                start_beat, vel = active.pop(msg.note)
                dur = beat - start_beat
                if dur >= MIN_NOTE_DURATION:
                    notes.append(Note(
                        pitch=msg.note,
                        duration=max(MIN_NOTE_DURATION, round(dur * 4) / 4),
                        velocity=vel,
                        offset=start_beat,
                    ))

    if not notes:
        return []

    min_off = min(n.offset for n in notes)
    for n in notes:
        n.offset -= min_off

    return sorted(notes, key=lambda n: n.offset)


def load_midi_notes(path: str) -> Tuple[List[Note], int, int]:
    """Compatibilidad v1: devuelve todas las notas mezcladas."""
    tracks, bpm, tpb, _ = load_midi_tracks(path)
    all_notes = [n for track in tracks for n in track]
    if not all_notes:
        return [], bpm, tpb
    min_off = min(n.offset for n in all_notes)
    for n in all_notes:
        n.offset -= min_off
    return sorted(all_notes, key=lambda n: n.offset), bpm, tpb


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTUACIÓN DE INTERÉS MELÓDICO
# ══════════════════════════════════════════════════════════════════════════════

def score_melodic_interest(notes: List[Note]) -> float:
    """
    Puntúa el interés melódico de una lista de notas (0-1).
    Criterios: variedad de pitch classes, rango, densidad,
    variedad dinámica, variedad rítmica y registro.
    """
    if len(notes) < 2:
        return 0.0

    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    velocities = [n.velocity for n in notes]

    pc_variety = len(set(p % 12 for p in pitches)) / 12.0
    pitch_range = min(1.0, (max(pitches) - min(pitches)) / 24.0)

    total_dur = max(n.offset + n.duration for n in notes)
    density = len(notes) / max(total_dur, 1.0)
    density_score = max(0.0, min(1.0, 1.0 - abs(density - 2.0) / 4.0))

    vel_variety = min(1.0, float(np.std(velocities)) / 40.0)
    rhythm_variety = min(1.0, float(np.std(durations)) / 1.0)

    median_pitch = float(np.median(pitches))
    register_score = max(0.0, min(1.0, 1.0 - abs(median_pitch - 66.0) / 30.0))

    return float(
        pc_variety    * 0.25 +
        pitch_range   * 0.20 +
        density_score * 0.20 +
        vel_variety   * 0.10 +
        rhythm_variety* 0.10 +
        register_score* 0.15
    )


def select_melody_track(tracks: List[List[Note]],
                        melody_track_idx: Optional[int] = None,
                        verbose: bool = False) -> List[Note]:
    """Selecciona el track más melódico de un MIDI multi-track."""
    if not tracks:
        return []
    if len(tracks) == 1:
        return tracks[0]
    if melody_track_idx is not None:
        idx = max(0, min(melody_track_idx, len(tracks) - 1))
        if verbose:
            print(f"    Track melódico forzado: #{idx}")
        return tracks[idx]

    scores = [score_melodic_interest(t) for t in tracks]
    best_idx = int(np.argmax(scores))
    if verbose:
        for i, (t, s) in enumerate(zip(tracks, scores)):
            marker = " ← seleccionado" if i == best_idx else ""
            print(f"    Track {i}: {len(t):4d} notas  interés={s:.3f}{marker}")
    return tracks[best_idx]


def select_best_fragment(notes: List[Note], target_beats: float,
                         beats_per_bar: int = 4,
                         verbose: bool = False) -> List[Note]:
    """
    Desliza una ventana de target_beats sobre la frase y elige
    el fragmento con mayor interés melódico.
    """
    if not notes:
        return notes

    total = max(n.offset + n.duration for n in notes)
    if total <= target_beats * 1.2:
        return notes

    step = float(beats_per_bar)
    max_start = total - target_beats
    best_score = -1.0
    best_start = 0.0

    start = 0.0
    while start <= max_start + 0.01:
        fragment = [n for n in notes if n.offset >= start and n.offset < start + target_beats]
        if len(fragment) >= 2:
            score = score_melodic_interest(fragment)
            if score > best_score:
                best_score = score
                best_start = start
        start += step

    end = best_start + target_beats
    fragment = [n for n in notes if n.offset >= best_start and n.offset < end]
    if not fragment:
        fragment = notes[:max(2, len(notes) // 4)]

    min_off = min(n.offset for n in fragment)
    fragment = [Note(n.pitch, n.duration, n.velocity, n.offset - min_off)
                for n in fragment]

    if verbose:
        print(f"    Fragmento: inicio={best_start:.1f}b  fin={end:.1f}b  "
              f"notas={len(fragment)}  interés={best_score:.3f}")

    return fragment


def score_phrase_for_section(notes: List[Note], section_name: str) -> float:
    """Puntúa lo adecuada que es una frase para una sección concreta."""
    base = score_melodic_interest(notes)
    tension = SECTION_TENSION.get(section_name, 0.5)
    return float(base * tension + (1.0 - base) * (1.0 - tension))


def select_phrase_for_section(phrases: List[List[Note]],
                               section_name: str) -> Tuple[int, List[Note]]:
    """Elige la frase más adecuada para una sección según su rol dramático."""
    if not phrases:
        return 0, []
    if len(phrases) == 1:
        return 0, phrases[0]
    scores = [score_phrase_for_section(p, section_name) for p in phrases]
    best_idx = int(np.argmax(scores))
    return best_idx, phrases[best_idx]


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS TONAL
# ══════════════════════════════════════════════════════════════════════════════

def detect_key(notes: List[Note]) -> Tuple[int, str]:
    """Krumhansl-Schmuckler simplificado. Devuelve (root_pc, mode)."""
    if not notes:
        return 0, "major"

    pc_weights = np.zeros(12)
    for n in notes:
        pc_weights[n.pitch % 12] += n.duration

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_key, best_score, best_mode = 0, -999.0, "major"
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
    key_str = key_str.strip()
    mode = "major"
    parts = key_str.split()
    root_str = parts[0]

    if len(parts) > 1:
        if parts[1].lower() in ("minor", "min", "m"):
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
    entry = NUMERAL_MAP.get(numeral)
    if entry is None:
        return key_pc, "M"
    semitone, quality = entry
    return (key_pc + semitone) % 12, quality


def chord_pitches_with_voicing(root_pc: int, quality: str, octave: int = 4,
                                prev_chord: Optional[List[int]] = None) -> List[int]:
    """
    Construye pitches MIDI para un acorde con voice leading.
    Elige la inversión que minimiza el movimiento total de voces
    respecto al acorde anterior.
    """
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
    base = root_pc + octave * 12
    while base < 48:
        base += 12
    while base > 72:
        base -= 12
    raw = [base + i for i in intervals if 24 <= base + i <= 96]

    if not raw or prev_chord is None:
        return raw

    def total_motion(a: List[int], b: List[int]) -> float:
        n = min(len(a), len(b))
        return sum(abs(b[i] - a[i]) for i in range(n))

    best, best_motion = raw, total_motion(prev_chord, raw)
    rotated = list(raw)
    for _ in range(len(raw) - 1):
        rotated = rotated[1:] + [rotated[0] + 12]
        motion = total_motion(prev_chord, rotated)
        if motion < best_motion:
            best_motion = motion
            best = list(rotated)

    return best


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMACIONES DE FRASES
# ══════════════════════════════════════════════════════════════════════════════

def transpose_to_key(notes: List[Note], from_pc: int, to_pc: int,
                     from_mode: str, to_mode: str) -> List[Note]:
    semitones = (to_pc - from_pc) % 12
    if semitones > 6:
        semitones -= 12
    result = []
    for n in notes:
        snapped = snap_to_scale(n.pitch + semitones, to_pc, to_mode)
        result.append(Note(snapped, n.duration, n.velocity, n.offset))
    return result


def fit_to_bars(notes: List[Note], target_bars: int,
                beats_per_bar: int = 4) -> List[Note]:
    """
    Ajusta la frase al número de compases objetivo.

    factor > 1.2  → escala hacia arriba
    0.8-1.2       → sin cambio
    0.5-0.8       → escala suave
    < 0.5         → selecciona el fragmento más interesante (no comprime)
    """
    if not notes:
        return notes
    total = max(n.offset + n.duration for n in notes)
    target = target_bars * beats_per_bar
    if total < 0.01:
        return notes
    factor = target / total

    if 0.8 < factor < 1.2:
        return notes

    if factor < 0.5:
        sliced = select_best_fragment(notes, target, beats_per_bar)
        new_total = max(n.offset + n.duration for n in sliced) if sliced else 0
        if new_total > 0.01:
            fine = target / new_total
            if not (0.8 < fine < 1.2):
                sliced = [Note(n.pitch,
                               max(MIN_NOTE_DURATION, n.duration * fine),
                               n.velocity, n.offset * fine)
                          for n in sliced]
        return sliced

    return [Note(n.pitch, max(MIN_NOTE_DURATION, n.duration * factor),
                 n.velocity, n.offset * factor)
            for n in notes]


def apply_velocity(notes: List[Note], base_vel: int) -> List[Note]:
    if not notes:
        return notes
    current_mean = float(np.mean([n.velocity for n in notes]))
    if current_mean < 1:
        return notes
    ratio = base_vel / current_mean
    return [Note(n.pitch, n.duration, int(np.clip(n.velocity * ratio, 20, 120)), n.offset)
            for n in notes]


def invert_contour(notes: List[Note], pivot: Optional[int] = None) -> List[Note]:
    if not notes:
        return notes
    pivot_pitch = pivot if pivot is not None else notes[0].pitch
    return [Note(max(24, min(108, 2 * pivot_pitch - n.pitch)),
                 n.duration, n.velocity, n.offset)
            for n in notes]


def augment(notes: List[Note], factor: float = 2.0) -> List[Note]:
    return [Note(n.pitch, n.duration * factor, n.velocity, n.offset * factor)
            for n in notes]


def diminish(notes: List[Note], factor: float = 0.5) -> List[Note]:
    return augment(notes, factor)


def repeat_phrase(notes: List[Note], times: int,
                  rng: Optional[random.Random] = None) -> List[Note]:
    """
    Repite la frase N veces con micro-variaciones en cada repetición
    (±5% velocidad, ±10% duración) para evitar el efecto mecánico.
    """
    if not notes or times <= 1:
        return notes
    total = max(n.offset + n.duration for n in notes)
    result = list(deepcopy(notes))
    for i in range(1, times):
        for n in notes:
            new_vel = n.velocity
            new_dur = n.duration
            if rng is not None:
                new_vel = int(np.clip(n.velocity * rng.uniform(0.95, 1.05), 20, 120))
                new_dur = max(MIN_NOTE_DURATION, n.duration * rng.uniform(0.90, 1.10))
            result.append(Note(n.pitch, new_dur, new_vel, n.offset + total * i))
    return result


def vary_phrase(notes: List[Note], variation_type: str,
                root_pc: int, mode: str, rng: random.Random) -> List[Note]:
    if not notes:
        return notes

    if variation_type == "parallel":
        result = []
        for n in notes:
            dur_f = rng.choice([0.75, 1.0, 1.0, 1.25])
            result.append(Note(n.pitch, max(MIN_NOTE_DURATION, n.duration * dur_f),
                               n.velocity, n.offset))
        return result

    elif variation_type == "inversion":
        return invert_contour(notes)

    elif variation_type == "sequence_up":
        scale_pcs = get_scale_pcs(root_pc, mode)
        result = []
        for n in notes:
            pc = n.pitch % 12
            if pc in scale_pcs:
                idx = scale_pcs.index(pc)
                new_pc = scale_pcs[(idx + 1) % 7]
                semitones = (new_pc - pc) % 12
                new_pitch = snap_to_scale(n.pitch + semitones, root_pc, mode)
            else:
                new_pitch = snap_to_scale(n.pitch, root_pc, mode)
            result.append(Note(new_pitch, n.duration, n.velocity, n.offset))
        return result

    elif variation_type == "development":
        strong = [n for n in notes if n.offset % 2.0 < 0.5]
        if len(strong) < 2:
            strong = notes[:max(2, len(notes) // 2)]
        return strong

    elif variation_type == "ornament":
        result = list(notes)
        insertions = []
        for i in range(len(notes) - 1):
            a, b = notes[i], notes[i + 1]
            if abs(b.pitch - a.pitch) >= 4 and a.duration >= 1.0:
                pass_pitch = snap_to_scale((a.pitch + b.pitch) // 2, root_pc, mode)
                half = max(MIN_NOTE_DURATION, a.duration / 2)
                insertions.append(Note(pass_pitch, half, a.velocity - 10, a.offset + half))
        result.extend(insertions)
        return sorted(result, key=lambda n: n.offset)

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICACIÓN DE NOTAS SUPERPUESTAS
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_notes(notes: List[Note]) -> List[Note]:
    """
    Elimina solapamientos del mismo pitch: recorta la nota anterior
    para que termine justo antes de que comience la siguiente del mismo pitch.
    """
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: (n.offset, n.pitch))
    active: Dict[int, int] = {}  # pitch → índice en result
    result: List[Note] = []

    for n in sorted_notes:
        if n.pitch in active:
            prev_idx = active[n.pitch]
            prev = result[prev_idx]
            if prev.offset + prev.duration > n.offset:
                new_dur = max(MIN_NOTE_DURATION, n.offset - prev.offset - 0.01)
                result[prev_idx] = Note(prev.pitch, new_dur, prev.velocity, prev.offset)
        result.append(n)
        active[n.pitch] = len(result) - 1

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE LEADING ENTRE SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

def apply_section_voice_leading(prev_section: Section,
                                 next_section: Section) -> Section:
    """
    Suaviza la transición entre secciones consecutivas ajustando
    la primera nota melódica de la sección siguiente para que esté
    cerca (en semitones) de la última nota de la sección anterior,
    siempre dentro de la escala de destino.
    """
    if not prev_section.notes or not next_section.notes:
        return next_section

    prev_mel = [n for n in prev_section.notes if n.pitch >= 48]
    if not prev_mel:
        return next_section
    last_note = max(prev_mel, key=lambda n: n.offset + n.duration)

    next_mel = [n for n in next_section.notes if n.pitch >= 48]
    if not next_mel:
        return next_section
    first_note = min(next_mel, key=lambda n: n.offset)

    if abs(first_note.pitch - last_note.pitch) <= 2:
        return next_section

    scale_pcs = set(get_scale_pcs(next_section.key_pc, next_section.mode))
    best_pitch, best_dist = first_note.pitch, abs(first_note.pitch - last_note.pitch)
    for candidate in range(max(24, last_note.pitch - 12),
                            min(108, last_note.pitch + 13)):
        if candidate % 12 in scale_pcs:
            dist = abs(candidate - last_note.pitch)
            if dist < best_dist:
                best_dist = dist
                best_pitch = candidate

    if best_pitch == first_note.pitch:
        return next_section

    new_notes = []
    adjusted = False
    for n in next_section.notes:
        if not adjusted and n is first_note:
            new_notes.append(Note(best_pitch, n.duration, n.velocity, n.offset))
            adjusted = True
        else:
            new_notes.append(n)

    modified = deepcopy(next_section)
    modified.notes = new_notes
    return modified


# ══════════════════════════════════════════════════════════════════════════════
#  ACOMPAÑAMIENTO ARMÓNICO  (con voice leading de acordes)
# ══════════════════════════════════════════════════════════════════════════════

def build_chord_accompaniment(progression: List[Tuple[str, int]],
                               key_pc: int, beats_per_bar: int,
                               velocity: int, style: str = "block") -> List[Note]:
    """
    Genera acompañamiento con voice leading entre acordes consecutivos.
    En arpegios, garantiza duración mínima de 0.25 beats por nota.
    """
    notes = []
    cursor = 0.0
    prev_pitches: Optional[List[int]] = None

    for numeral, dur_beats in progression:
        root_pc, quality = numeral_to_root(numeral, key_pc)
        pitches = chord_pitches_with_voicing(root_pc, quality, octave=4,
                                             prev_chord=prev_pitches)
        prev_pitches = pitches

        if style == "block":
            for p in pitches:
                notes.append(Note(p, max(MIN_NOTE_DURATION, dur_beats * 0.9),
                                  velocity, cursor))

        elif style == "arpeggio":
            min_step = 0.25
            step = max(min_step, dur_beats / max(len(pitches), 1))
            if step * len(pitches) > dur_beats * 1.5:
                # Demasiado corto para arpegio → bloque
                for p in pitches:
                    notes.append(Note(p, max(MIN_NOTE_DURATION, dur_beats * 0.9),
                                      velocity, cursor))
            else:
                for i, p in enumerate(pitches):
                    notes.append(Note(p, max(MIN_NOTE_DURATION, step * 0.8),
                                      velocity, cursor + i * step))

        elif style == "bass_only":
            if pitches:
                notes.append(Note(min(pitches),
                                  max(MIN_NOTE_DURATION, dur_beats * 0.9),
                                  velocity, cursor))

        cursor += dur_beats

    return notes


def build_bass_line(progression: List[Tuple[str, int]],
                    key_pc: int, velocity: int) -> List[Note]:
    """Línea de bajo con voice leading: evita saltos de más de una séptima."""
    notes = []
    cursor = 0.0
    prev_bass = None

    for numeral, dur_beats in progression:
        root_pc, quality = numeral_to_root(numeral, key_pc)
        bass = root_pc + 36
        while bass < 28:
            bass += 12
        while bass > 52:
            bass -= 12

        if prev_bass is not None and abs(bass - prev_bass) > 7:
            alt = bass + (12 if bass < prev_bass else -12)
            if 28 <= alt <= 52:
                bass = alt

        notes.append(Note(bass, max(MIN_NOTE_DURATION, dur_beats * 0.85),
                          velocity, cursor))
        prev_bass = bass
        cursor += dur_beats

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

def build_section(name: str,
                  phrases: List[List[Note]],
                  key_pc: int, mode: str, tempo: int, style: str,
                  bars: int, beats_per_bar: int,
                  rng: random.Random, verbose: bool = False) -> Section:

    tension = SECTION_TENSION.get(name, 0.5)
    vel_base = SECTION_VELOCITY.get(name, 70)
    prog_key = name.rstrip("12")
    progression = SECTION_PROGRESSIONS.get(style, SECTION_PROGRESSIONS["pop"]).get(
        prog_key, SECTION_PROGRESSIONS["pop"]["verse"]
    )

    sec = Section(name=name, bars=bars, beats_per_bar=beats_per_bar,
                  tempo=tempo, key_pc=key_pc, mode=mode, tension=tension,
                  velocity_base=vel_base, progression=progression)

    if not phrases:
        return sec

    # Selección inteligente de frase por rol dramático
    phrase_idx, base_notes = select_phrase_for_section(phrases, name)
    base_notes = deepcopy(base_notes)
    sec.source_phrase_idx = phrase_idx

    if verbose:
        interest = score_melodic_interest(base_notes)
        print(f"    [{name}] frase #{phrase_idx + 1} (interés={interest:.3f}), "
              f"{len(base_notes)} notas → {bars}c @ tensión={tension:.2f}")

    transformation_map = {
        "intro":     "development",
        "verse1":    "parallel",
        "prechorus": "sequence_up",
        "chorus":    "parallel",
        "verse2":    "ornament",
        "bridge":    "inversion",
        "solo":      "development",
        "outro":     "parallel",
    }
    transformed = vary_phrase(base_notes, transformation_map.get(name, "parallel"),
                              key_pc, mode, rng)

    total_beats = bars * beats_per_bar
    if transformed:
        phrase_dur = max(n.offset + n.duration for n in transformed)
        if phrase_dur < total_beats * 0.5:
            times = math.ceil(total_beats / max(phrase_dur, 0.1))
            # Repetición con micro-variaciones
            transformed = repeat_phrase(transformed, min(times, 4), rng)
        transformed = fit_to_bars(transformed, bars, beats_per_bar)

    transformed = apply_velocity(transformed, vel_base)
    transformed = deduplicate_notes(transformed)

    acc_style = {
        "intro": "bass_only", "verse1": "arpeggio", "verse2": "arpeggio",
        "prechorus": "block",  "chorus": "block",   "bridge": "arpeggio",
        "solo": "bass_only",   "outro": "bass_only",
    }.get(name, "block")

    beats_needed = bars * beats_per_bar
    prog_beats = sum(d for _, d in progression)
    reps = max(1, math.ceil(beats_needed / max(prog_beats, 1)))
    full_prog = progression * reps

    prog_truncated = []
    acc_cursor = 0.0
    for numeral, dur in full_prog:
        if acc_cursor >= beats_needed:
            break
        actual_dur = min(dur, beats_needed - acc_cursor)
        prog_truncated.append((numeral, actual_dur))
        acc_cursor += actual_dur

    acc_vel = max(20, vel_base - 20)
    acc_notes = build_chord_accompaniment(prog_truncated, key_pc, beats_per_bar,
                                          acc_vel, style=acc_style)
    bass_notes = build_bass_line(prog_truncated, key_pc, max(20, acc_vel - 10))

    all_notes = deduplicate_notes(transformed + acc_notes + bass_notes)
    sec.notes = sorted(all_notes, key=lambda n: n.offset)
    sec.progression = prog_truncated
    return sec


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def generate_song_plan(phrases_notes: List[List[Note]],
                       key_pc: int, mode: str, key_name: str,
                       tempo: int, style: str,
                       sections_to_generate: List[str],
                       bars_per_section: Optional[Dict[str, int]],
                       beats_per_bar: int,
                       rng: random.Random,
                       verbose: bool = False) -> SongPlan:

    plan = SongPlan(key_name=key_name, mode=mode, key_pc=key_pc,
                    tempo=tempo, style=style,
                    phrase_count=len(phrases_notes))

    if verbose:
        print(f"\n  Generando secciones: {', '.join(sections_to_generate)}")
        print(f"  Tonalidad: {key_name} {mode}  Tempo: {tempo} BPM  Estilo: {style}")
        print(f"  Frases de entrada: {len(phrases_notes)}\n")

    for sec_name in sections_to_generate:
        bars = (bars_per_section or {}).get(sec_name,
               SECTION_DEFAULT_BARS.get(sec_name, 8))

        if verbose:
            print(f"  Construyendo '{sec_name}' ({bars} compases)...")

        section = build_section(
            name=sec_name, phrases=phrases_notes,
            key_pc=key_pc, mode=mode, tempo=tempo, style=style,
            bars=bars, beats_per_bar=beats_per_bar,
            rng=rng, verbose=verbose,
        )

        # Voice leading con la sección anterior
        if plan.sections:
            section = apply_section_voice_leading(plan.sections[-1], section)

        plan.sections.append(section)

    plan.total_bars = sum(s.bars for s in plan.sections)
    return plan


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def section_to_midi(section: Section, ticks_per_beat: int = 480) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(section.tempo), time=0))
    track.append(mido.MetaMessage("track_name",
                                   name=section.name.replace("_", " ").title(), time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    events = []
    for note in section.notes:
        t_on = int(note.offset * ticks_per_beat)
        t_off = int((note.offset + max(MIN_NOTE_DURATION, note.duration)) * ticks_per_beat)
        vel = max(1, min(127, note.velocity))
        p = max(0, min(127, note.pitch))
        ch = 1 if p < 48 else 0
        events.append((t_on, "on", p, vel, ch))
        events.append((t_off, "off", p, 0, ch))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    current_tick = 0
    for abs_tick, etype, pitch, vel, ch in events:
        delta = max(0, abs_tick - current_tick)
        track.append(mido.Message(
            "note_on" if etype == "on" else "note_off",
            channel=ch, note=pitch, velocity=vel, time=delta
        ))
        current_tick = abs_tick

    return mid


def concatenate_midis(sections: List[Section], ticks_per_beat: int = 480) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    if not sections:
        return mid

    track.append(mido.MetaMessage("set_tempo",
                                   tempo=mido.bpm2tempo(sections[0].tempo), time=0))
    track.append(mido.MetaMessage("track_name", name="Full Song", time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    global_offset = 0.0
    events = []

    for i, section in enumerate(sections):
        if i > 0 and section.tempo != sections[i - 1].tempo:
            t = int(global_offset * ticks_per_beat)
            events.append((t, "tempo", mido.bpm2tempo(section.tempo), 0, 0))

        for note in section.notes:
            abs_off = global_offset + note.offset
            t_on = int(abs_off * ticks_per_beat)
            t_off = int((abs_off + max(MIN_NOTE_DURATION, note.duration)) * ticks_per_beat)
            vel = max(1, min(127, note.velocity))
            p = max(0, min(127, note.pitch))
            ch = 1 if p < 48 else 0
            events.append((t_on, "on", p, vel, ch))
            events.append((t_off, "off", p, 0, ch))

        global_offset += section.total_beats

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
#  EXPORTACIÓN MUSICXML
# ══════════════════════════════════════════════════════════════════════════════

def note_to_musicxml_pitch(midi_pitch: int) -> Tuple[str, int, int]:
    """Convierte pitch MIDI → (step, alter, octave)."""
    pc = midi_pitch % 12
    octave = midi_pitch // 12 - 1
    pc_map = {
        0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0),
        5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0),
        10: ("A", 1), 11: ("B", 0),
    }
    step, alter = pc_map[pc]
    return step, alter, octave


def duration_to_musicxml_type(duration_beats: float) -> Tuple[int, str, int]:
    """duration_beats → (divisions_val, type_str, dots). divisions=4 (por negra)."""
    divisions = 4
    total_divs = max(1, round(duration_beats * divisions))
    type_table = [
        (32, "whole", 1), (16, "whole", 0), (12, "half", 1),
        (8, "half", 0), (6, "quarter", 1), (4, "quarter", 0),
        (3, "eighth", 1), (2, "eighth", 0), (1, "16th", 0),
    ]
    best = (total_divs, "quarter", 0)
    best_dist = 999
    for divs, name, dots in type_table:
        dist = abs(divs - total_divs)
        if dist < best_dist:
            best_dist = dist
            best = (divs, name, dots)
    return total_divs, best[1], best[2]


def section_to_musicxml_part(section: Section, divisions: int = 4) -> str:
    lines = [f'  <part id="{section.name}">']
    beats_per_bar = section.beats_per_bar

    bar_notes: Dict[int, List[Note]] = {}
    for n in section.notes:
        bar_idx = int(n.offset // beats_per_bar)
        bar_notes.setdefault(bar_idx, []).append(n)

    for bar_idx in range(section.bars):
        lines.append(f'    <measure number="{bar_idx + 1}">')
        if bar_idx == 0:
            lines.append(
                f'      <attributes>'
                f'<divisions>{divisions}</divisions>'
                f'<key><fifths>0</fifths></key>'
                f'<time><beats>{beats_per_bar}</beats><beat-type>4</beat-type></time>'
                f'<clef><sign>G</sign><line>2</line></clef>'
                f'</attributes>'
            )
            lines.append(
                f'      <direction placement="above"><direction-type>'
                f'<metronome><beat-unit>quarter</beat-unit>'
                f'<per-minute>{section.tempo}</per-minute></metronome>'
                f'</direction-type></direction>'
            )

        mel_notes = sorted(
            [n for n in bar_notes.get(bar_idx, []) if n.pitch >= 48],
            key=lambda n: n.offset
        )

        if not mel_notes:
            lines.append(
                f'      <note><rest/>'
                f'<duration>{divisions * beats_per_bar}</duration>'
                f'<type>whole</type></note>'
            )
        else:
            cursor = float(bar_idx * beats_per_bar)
            for n in mel_notes:
                gap = n.offset - cursor
                if gap >= 0.25:
                    gap_divs = max(1, round(gap * divisions))
                    lines.append(
                        f'      <note><rest/><duration>{gap_divs}</duration>'
                        f'<type>eighth</type></note>'
                    )
                step, alter, octave = note_to_musicxml_pitch(n.pitch)
                total_divs, type_str, dots = duration_to_musicxml_type(n.duration)
                vel_dyn = ("pp" if n.velocity < 40 else "mp" if n.velocity < 60
                           else "mf" if n.velocity < 80 else "f")
                note_lines = [f'      <note>']
                note_lines.append(f'        <pitch><step>{step}</step>')
                if alter:
                    note_lines.append(f'        <alter>{alter}</alter>')
                note_lines.append(f'        <octave>{octave}</octave></pitch>')
                note_lines.append(f'        <duration>{total_divs}</duration>')
                note_lines.append(f'        <type>{type_str}</type>')
                if dots:
                    note_lines.append(f'        <dot/>')
                note_lines.append(f'        <dynamics><{vel_dyn}/></dynamics>')
                note_lines.append(f'      </note>')
                lines.extend(note_lines)
                cursor = n.offset + n.duration

        lines.append(f'    </measure>')

    lines.append(f'  </part>')
    return "\n".join(lines)


def export_musicxml(plan: SongPlan, output_path: str):
    """Exporta el plan completo como archivo MusicXML 3.1."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC',
        '  "-//Recordare//DTD MusicXML 3.1 Partwise//EN"',
        '  "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        '  <work><work-title>Song Architect Export</work-title></work>',
        f'  <identification><encoding><software>Song Architect v{VERSION}</software>'
        f'</encoding></identification>',
        '  <part-list>',
    ]
    for sec in plan.sections:
        lines.append(f'    <score-part id="{sec.name}">')
        lines.append(f'      <part-name>{sec.name.replace("_"," ").title()}</part-name>')
        lines.append(f'    </score-part>')
    lines.append('  </part-list>')

    for sec in plan.sections:
        lines.append(section_to_musicxml_part(sec))

    lines.append('</score-partwise>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
        print("▁▂▃▄▅▆▇█"[t_idx], end="")
    print(f"  ({plan.total_bars}c total)")
    print()


def export_plan_json(plan: SongPlan, output_path: str, input_files: List[str]):
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
        "sections": [{
            "name": sec.name,
            "bars": sec.bars,
            "beats_per_bar": sec.beats_per_bar,
            "tempo": sec.tempo,
            "tension": sec.tension,
            "velocity_base": sec.velocity_base,
            "note_count": len(sec.notes),
            "source_phrase_idx": sec.source_phrase_idx,
            "progression": [[n, d] for n, d in sec.progression],
        } for sec in plan.sections],
    }
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
              python song_architect.py *.mid --split-tracks --melody-track 1
              python song_architect.py *.mid --fragment-start 8 --fragment-end 24
              python song_architect.py *.mid --export-musicxml
              python song_architect.py *.mid --dry-run --verbose
        """)
    )
    p.add_argument("inputs", nargs="+", metavar="MIDI",
                   help="Archivos MIDI de frases de entrada")
    p.add_argument("--key", type=str, default=None,
                   help="Tonalidad base (ej. C, Am, F# minor). Default: auto")
    p.add_argument("--tempo", type=int, default=None,
                   help="Tempo BPM (default: detectado del MIDI o 120)")
    p.add_argument("--style", type=str, default="pop",
                   choices=["pop", "rock", "ballad", "jazz", "folk", "rnb"],
                   help="Estilo armónico (default: pop)")
    p.add_argument("--beats-per-bar", type=int, default=None,
                   help="Pulsos por compás (default: auto-detección desde el MIDI)")
    p.add_argument("--bars-per-section", type=int, default=None,
                   help="Compases por sección, igual para todas (default: auto)")

    trk = p.add_argument_group("Tracks (v2.0)")
    trk.add_argument("--split-tracks", action="store_true",
                     help="Tratar cada track MIDI como frase independiente")
    trk.add_argument("--melody-track", type=int, default=None,
                     help="Índice del track melódico a usar (default: auto)")

    frag = p.add_argument_group("Fragmento (v2.0)")
    frag.add_argument("--fragment-start", type=int, default=None, metavar="BAR",
                      help="Compás de inicio del fragmento (1-based)")
    frag.add_argument("--fragment-end", type=int, default=None, metavar="BAR",
                      help="Compás de fin del fragmento (1-based, inclusivo)")

    secs = p.add_argument_group("Secciones")
    secs.add_argument("--sections", nargs="+", default=None,
                      choices=CANONICAL_ORDER, metavar="SEC",
                      help="Secciones a generar (default: todas)")
    secs.add_argument("--no-intro",     action="store_true")
    secs.add_argument("--no-prechorus", action="store_true")
    secs.add_argument("--no-bridge",    action="store_true")
    secs.add_argument("--no-solo",      action="store_true")

    out = p.add_argument_group("Salida")
    out.add_argument("--out-dir", type=str, default="song_output",
                     help="Directorio de salida (default: song_output)")
    out.add_argument("--output-name", type=str, default="song",
                     help="Nombre base de archivos (default: song)")
    out.add_argument("--export-plan", action="store_true",
                     help="Exportar plan JSON")
    out.add_argument("--export-musicxml", action="store_true",
                     help="Exportar también en formato MusicXML (v2.0)")
    out.add_argument("--dry-run", action="store_true",
                     help="Mostrar plan sin generar archivos MIDI")

    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
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

    # 1. Verificar entradas
    input_files = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.exists():
            input_files.append(str(p))
        else:
            import glob
            input_files.extend(glob.glob(str(p)))

    if not input_files:
        print(f"[ERROR] No se encontraron archivos MIDI: {args.inputs}")
        sys.exit(1)

    print(f"\n  {_c('bold', 'SONG ARCHITECT')} v{VERSION}")
    print(f"  {'─' * 58}")
    print(f"  Frases de entrada: {len(input_files)}")
    for f in input_files:
        print(f"    • {Path(f).name}")

    # 2. Cargar frases
    all_phrases_notes: List[List[Note]] = []
    all_notes_flat: List[Note] = []
    detected_tempos = []
    detected_bpb = 4

    for fpath in input_files:
        tracks, bpm, tpb, bpb = load_midi_tracks(fpath)
        detected_tempos.append(bpm)
        detected_bpb = bpb

        if not tracks:
            print(f"  [warn] {Path(fpath).name}: sin notas")
            continue

        if args.split_tracks:
            for ti, track_notes in enumerate(tracks):
                if track_notes:
                    all_phrases_notes.append(track_notes)
                    all_notes_flat.extend(track_notes)
                    if args.verbose:
                        interest = score_melodic_interest(track_notes)
                        print(f"  {Path(fpath).name} track {ti}: "
                              f"{len(track_notes)} notas  interés={interest:.3f}")
        else:
            if args.verbose:
                print(f"  Analizando tracks de {Path(fpath).name}:")
            melody = select_melody_track(tracks, args.melody_track, verbose=args.verbose)
            if melody:
                all_phrases_notes.append(melody)
                all_notes_flat.extend(melody)
                if args.verbose:
                    print(f"  Cargado {Path(fpath).name}: {len(melody)} notas, {bpm} BPM")
            else:
                print(f"  [warn] {Path(fpath).name}: sin notas utilizables")

    if not all_phrases_notes:
        print("[ERROR] Ninguna frase tiene notas. Verifica los archivos MIDI.")
        sys.exit(1)

    # 3. Compás
    beats_per_bar = args.beats_per_bar or detected_bpb

    # 4. Recorte manual de fragmento
    if args.fragment_start is not None or args.fragment_end is not None:
        frag_start = ((args.fragment_start or 1) - 1) * beats_per_bar
        frag_end = (args.fragment_end * beats_per_bar
                    if args.fragment_end is not None else None)
        new_phrases = []
        for phrase in all_phrases_notes:
            sliced = [n for n in phrase
                      if n.offset >= frag_start and
                      (frag_end is None or n.offset < frag_end)]
            if sliced:
                min_off = min(n.offset for n in sliced)
                sliced = [Note(n.pitch, n.duration, n.velocity, n.offset - min_off)
                          for n in sliced]
                new_phrases.append(sliced)
        if new_phrases:
            all_phrases_notes = new_phrases
            all_notes_flat = [n for p in new_phrases for n in p]
            if args.verbose:
                print(f"\n  Fragmento manual: compases "
                      f"{args.fragment_start or 1}–{args.fragment_end or '∞'} "
                      f"({len(all_notes_flat)} notas)")
        else:
            print("[warn] El rango de fragmento no contiene notas; usando frase completa.")

    # 5. Tonalidad
    if args.key:
        key_pc, mode = parse_key(args.key)
        key_name = args.key.split()[0].rstrip("m")
        mode = ("minor" if "minor" in args.key.lower() or
                (args.key[-1].lower() == "m" and len(args.key) <= 3) else "major")
        if args.verbose:
            print(f"\n  Tonalidad forzada: {key_name} {mode}")
    else:
        key_pc, mode = detect_key(all_notes_flat)
        key_name = NOTE_NAMES[key_pc]
        if args.verbose:
            print(f"\n  Tonalidad detectada: {key_name} {mode}")

    # 6. Tempo
    tempo = (args.tempo or
             round(float(np.median(detected_tempos))) if detected_tempos else 120)
    if args.verbose:
        print(f"  Tempo: {tempo} BPM  |  Compás: {beats_per_bar}/4")

    # 7. Transponer frases
    phrases_in_key = []
    for i, phrase_notes in enumerate(all_phrases_notes):
        src_pc, src_mode = detect_key(phrase_notes)
        if src_pc != key_pc or src_mode != mode:
            transposed = transpose_to_key(phrase_notes, src_pc, key_pc, src_mode, mode)
            if args.verbose:
                print(f"  Frase {i+1}: {NOTE_NAMES[src_pc]} {src_mode} → {key_name} {mode}")
        else:
            transposed = phrase_notes
        phrases_in_key.append(transposed)

    # 8. Secciones
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

    bars_map: Optional[Dict[str, int]] = None
    if args.bars_per_section:
        bars_map = {s: args.bars_per_section for s in sections_to_generate}

    print(f"\n  Secciones: {_c('cyan', ', '.join(sections_to_generate))}")
    print(f"  Tonalidad: {key_name} {mode}  |  Tempo: {tempo} BPM  |  "
          f"Compás: {beats_per_bar}/4  |  Estilo: {args.style}")

    # 9. Generar plan
    print(f"\n  Generando secciones...")
    plan = generate_song_plan(
        phrases_notes=phrases_in_key,
        key_pc=key_pc, mode=mode, key_name=key_name,
        tempo=tempo, style=args.style,
        sections_to_generate=sections_to_generate,
        bars_per_section=bars_map,
        beats_per_bar=beats_per_bar,
        rng=rng, verbose=args.verbose,
    )

    print_song_plan(plan, verbose=args.verbose)

    if args.dry_run:
        print(f"  {_c('yellow', '[dry-run]')} No se generaron archivos MIDI.")
        if args.verbose:
            for sec in plan.sections:
                m, s = divmod(int(sec.duration_seconds), 60)
                print(f"    {sec.name:<16} {sec.bars}c  ~{m}:{s:02d}")
        return

    # 10. Exportar
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_base = args.output_name
    tpb = 480

    generated = []
    print(f"  Exportando a: {out_dir}/")
    print(f"  {'─' * 58}")

    for sec in plan.sections:
        out_path = out_dir / f"{name_base}_{sec.name}.mid"
        section_to_midi(sec, tpb).save(str(out_path))
        size_kb = out_path.stat().st_size // 1024
        generated.append(str(out_path))
        print(f"  {_c('green', '✓')} {out_path.name:<35}  "
              f"{sec.bars}c  {len(sec.notes):>4} notas  {size_kb} KB")

    full_path = out_dir / f"{name_base}_full.mid"
    concatenate_midis(plan.sections, tpb).save(str(full_path))
    full_size = full_path.stat().st_size // 1024
    generated.append(str(full_path))
    print(f"  {_c('green', '✓')} {full_path.name:<35}  "
          f"{plan.total_bars}c  {full_size} KB  {_c('cyan', '← obra completa')}")

    if args.export_plan:
        plan_path = out_dir / f"{name_base}_plan.json"
        export_plan_json(plan, str(plan_path), input_files)
        print(f"  {_c('green', '✓')} {plan_path.name:<35}  plan completo")
        generated.append(str(plan_path))

    if args.export_musicxml:
        xml_path = out_dir / f"{name_base}_full.musicxml"
        export_musicxml(plan, str(xml_path))
        xml_size = xml_path.stat().st_size // 1024
        print(f"  {_c('green', '✓')} {xml_path.name:<35}  {xml_size} KB  "
              f"{_c('cyan', '← MusicXML (MuseScore/Sibelius)')}")
        generated.append(str(xml_path))

    # 11. Resumen
    total_s = sum(sec.duration_seconds for sec in plan.sections)
    m, s = divmod(int(total_s), 60)
    print(f"\n{'═' * 64}")
    print(f"  {_c('green', 'Canción generada')}: {len(plan.sections)} secciones, "
          f"{plan.total_bars} compases, ~{m}:{s:02d}")
    print(f"  {len(generated)} archivos en: {out_dir}/")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
