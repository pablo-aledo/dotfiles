#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     BASS LINE COMPOSER  v1.0                                 ║
║         Línea de bajo como voz compositiva autónoma                          ║
║                                                                              ║
║  A diferencia del walking bass auxiliar de midi_dna_unified (que acompaña   ║
║  sin mayor protagonismo) y de la voz B del voice_leader (que solo asegura   ║
║  las reglas de coral SATB), este módulo trata el bajo como una VOZ           ║
║  INDEPENDIENTE con su propio contrapunto contra la melodía, su propio        ║
║  arco expresivo y su propia lógica estilística.                              ║
║                                                                              ║
║  ESTILOS DISPONIBLES (--style):                                              ║
║    walking      — Jazz walking bass: negra en cada tiempo, notas de paso    ║
║                   cromáticas, aproximaciones diatónicas y cromáticas        ║
║    classical    — Bajo clásico/romántico: raíz en tiempo fuerte, 5ª/8ª      ║
║                   en tiempos débiles; soporte armónico Alberti opcional     ║
║    baroque      — Bajo continuo barroco: realización del bajo cifrado,      ║
║                   movimiento por grados, notas de paso, sin saltos bruscos  ║
║    latin        — Tumbao (son cubano): figura sincopada característica,     ║
║                   anticipaciones, dos y tres clave                           ║
║    blues        — Blues shuffle: root-5th-6th-7th, swung, boogie pattern   ║
║    pop          — Root en 1 y 5, octavas, fills en el 4º tiempo            ║
║    pedal        — Pedal de tónica o dominante bajo armonías cambiantes      ║
║    ostinato     — Figura rítmica repetida (ciaccona / passacaglia)          ║
║    auto         — Detecta el estilo más apropiado según la armonía          ║
║                                                                              ║
║  TÉCNICAS COMPOSITIVAS IMPLEMENTADAS:                                        ║
║  [T1]  Approach notes cromáticas: semitono ↑↓ al siguiente acorde          ║
║  [T2]  Approach notes diatónicas: grado de escala al siguiente acorde       ║
║  [T3]  Notas de enclosure: aproximación desde arriba Y abajo                ║
║  [T4]  Arpegio descendente: raíz→7ª→5ª→3ª                                 ║
║  [T5]  Notas de paso escalares entre dos raíces                             ║
║  [T6]  Anticipación: ataque de la siguiente raíz en el último corchea       ║
║  [T7]  Repetición rítmica: figuras de 2 o 4 notas que se repiten           ║
║  [T8]  Salto de octava con resolución                                       ║
║  [T9]  Pedal: nota sostenida bajo armonías cambiantes                       ║
║  [T10] Contrapunto melódico contra soprano: movimiento contrario            ║
║                                                                              ║
║  MODOS DE ENTRADA:                                                           ║
║    --chords "Cm G7 Fm Bb7"          — un acorde por compás                  ║
║    --chords "Cm:2 G7:2 Fm:1 Bb7:1" — duración en beats tras ':'            ║
║    --chords-midi  acordes.mid       — pista de acordes en MIDI              ║
║    --chords-json  prog.chords.json  — desde chord_progression_generator     ║
║    --melody-midi  melodia.mid       — melodía para contrapunto automático   ║
║    --from-theorist obra.theorist.json                                        ║
║    --from-narrator obra_plan.json                                            ║
║    --curves       curvas.curves.json                                         ║
║                                                                              ║
║  EXPORTACIÓN:                                                                ║
║    .mid                — MIDI de la línea de bajo (canal 2, program 32)     ║
║    .combined.mid       — Bajo + melodía combinados (si --melody-midi)       ║
║    .bass.json          — Partitura en JSON (notas, técnicas, análisis)      ║
║    .fingerprint.json   — Huella compatible con stitcher.py                  ║
║                                                                              ║
║  USO:                                                                        ║
║    # Desde descripción libre                                                 ║
║    python bass_line_composer.py "bajo de jazz oscuro en Re menor"           ║
║                                                                              ║
║    # Parámetros directos                                                     ║
║    python bass_line_composer.py --key Dm --style walking --bars 16          ║
║                                                                              ║
║    # Con progresión de acordes                                               ║
║    python bass_line_composer.py --key Am --chords "Am F C G" --bars 8      ║
║                                                                              ║
║    # Con melodía para contrapunto                                            ║
║    python bass_line_composer.py --key C --chords "C Am F G"                 ║
║        --melody-midi melody_out.mid --style classical                        ║
║                                                                              ║
║    # Desde chord_progression_generator                                       ║
║    python bass_line_composer.py --chords-json obra.chords.json              ║
║                                                                              ║
║    # Desde theorist                                                          ║
║    python bass_line_composer.py --from-theorist obra.theorist.json          ║
║                                                                              ║
║    # Con ostinato (passacaglia)                                              ║
║    python bass_line_composer.py --key Dm --style ostinato                   ║
║        --ostinato-pattern "D3:2 C3:1 Bb2:1" --bars 32                      ║
║                                                                              ║
║    # Múltiples candidatos                                                    ║
║    python bass_line_composer.py --key G --style auto --candidates 4         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    description          Descripción libre (entre comillas)                  ║
║    --key KEY             Tonalidad: C, Am, F#, Bb… (default: C)            ║
║    --mode MODE           Escala (default: auto desde key)                   ║
║    --bars N              Compases (default: 16)                             ║
║    --tempo BPM           Tempo (default: 120)                               ║
║    --time-sig S          Compás: 4/4, 3/4, 6/8 (default: 4/4)             ║
║    --style S             Estilo de bajo (default: auto)                     ║
║    --chords PROG         Progresión en texto                                ║
║    --chords-midi FILE    Pista de acordes en MIDI                           ║
║    --chords-json FILE    Acordes desde .chords.json                         ║
║    --melody-midi FILE    Melodía para contrapunto                           ║
║    --range LOW HIGH      Rango MIDI (default: 28 55 = E1-G3)               ║
║    --tension F           Tensión global 0-1 (default: auto)                 ║
║    --swing F             Swing 0-1 (0=straight, 1=full swing, default: auto)║
║    --approach-prob F     Probabilidad de approach notes 0-1 (default: 0.6) ║
║    --ostinato-pattern S  Patrón para style=ostinato ("D3:2 C3:1 Bb2:1")   ║
║    --pedal-note NOTE     Nota de pedal para style=pedal (ej: "D2")         ║
║    --candidates N        Generar N candidatos (default: 1)                  ║
║    --curves FILE         Curvas .curves.json de tension_designer            ║
║    --from-theorist FILE  Leer .theorist.json                                ║
║    --from-narrator FILE  Leer obra_plan.json                                ║
║    --export-fingerprint  Exportar .fingerprint.json                         ║
║    --export-combined     Exportar MIDI con bajo + melodía juntos            ║
║    --output FILE         Salida (default: bass_out.mid)                     ║
║    --seed N              Semilla (default: 42)                              ║
║    --verbose             Informe detallado                                   ║
║    --dry-run             Mostrar parámetros sin generar                     ║
║    --listen              Reproducir al final (requiere pygame)              ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:   mido, numpy                                                    ║
║    Opcionales: music21, scipy, pygame                                        ║
║    Pipeline:  midi_dna_unified.py en el mismo directorio                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
import re
from copy import deepcopy
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# ── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[WARN] mido no disponible. Instala con: pip install mido")

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

# ── Integración con el ecosistema ─────────────────────────────────────────────
_DNA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DNA_DIR)
try:
    from midi_dna_unified import (
        _snap_to_scale, _get_scale_pcs, _get_scale_midi,
        _quarter_to_ticks, _clamp_pitch,
        MAJOR_SCALE_DEGREES, MINOR_SCALE_DEGREES, INSTRUMENT_RANGES,
    )
    DNA_OK = True
except ImportError:
    DNA_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC  = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}

SCALE_INTERVALS: Dict[str, List[int]] = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "minor":             [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":    [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":     [0, 2, 3, 5, 7, 9, 11],
    "dorian":            [0, 2, 3, 5, 7, 9, 10],
    "phrygian":          [0, 1, 3, 5, 7, 8, 10],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "lydian":            [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":        [0, 2, 4, 5, 7, 9, 10],
    "locrian":           [0, 1, 3, 5, 6, 8, 10],
    "blues":             [0, 3, 5, 6, 7, 10],
    "pentatonic_minor":  [0, 3, 5, 7, 10],
    "pentatonic_major":  [0, 2, 4, 7, 9],
    "whole_tone":        [0, 2, 4, 6, 8, 10],
}

CHORD_INTERVALS: Dict[str, List[int]] = {
    "":      [0, 4, 7],
    "m":     [0, 3, 7],
    "7":     [0, 4, 7, 10],
    "M7":    [0, 4, 7, 11],
    "m7":    [0, 3, 7, 10],
    "dim":   [0, 3, 6],
    "dim7":  [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "aug":   [0, 4, 8],
    "sus4":  [0, 5, 7],
    "sus2":  [0, 2, 7],
    "9":     [0, 4, 7, 10, 14],
    "M9":    [0, 4, 7, 11, 14],
    "m9":    [0, 3, 7, 10, 14],
    "6":     [0, 4, 7, 9],
    "m6":    [0, 3, 7, 9],
    "add9":  [0, 4, 7, 14],
    "5":     [0, 7],         # power chord
}

# Rango típico del bajo (E1=28 a G3=55 en GM)
BASS_LOW_DEFAULT  = 28
BASS_HIGH_DEFAULT = 55

# Velocidades por zona de dinámica
VEL_MAP = {
    "pp": (25, 40),
    "p":  (40, 55),
    "mp": (55, 70),
    "mf": (70, 85),
    "f":  (85, 100),
    "ff": (100, 115),
}

# Programas GM para el bajo
GM_BASS_PROGRAMS = {
    "acoustic":   32,  # Acoustic Bass
    "fingered":   33,  # Electric Bass (finger)
    "picked":     34,  # Electric Bass (pick)
    "fretless":   35,  # Fretless Bass
    "slap":       36,  # Slap Bass 1
    "synth":      38,  # Synth Bass 1
    "contrabass": 43,  # Contrabass (orquestal)
    "tuba":       58,  # Tuba
}

STYLE_TO_PROGRAM = {
    "walking":   "fingered",
    "classical": "contrabass",
    "baroque":   "contrabass",
    "latin":     "fingered",
    "blues":     "fingered",
    "pop":       "picked",
    "pedal":     "contrabass",
    "ostinato":  "contrabass",
    "auto":      "fingered",
}

# Palabras clave para detectar estilo desde descripción
STYLE_KEYWORDS = {
    "walking":   ["jazz", "walking", "swing", "bebop", "bop"],
    "classical": ["clásic", "clasic", "orquest", "sinfonía", "sinfon", "romántic", "romant"],
    "baroque":   ["barroc", "baroque", "continuo", "bach", "handel", "vivaldi"],
    "latin":     ["latin", "salsa", "tumbao", "cubano", "bossa", "samba", "clave"],
    "blues":     ["blues", "boogie", "shuffle", "12 bar", "doce compás"],
    "pop":       ["pop", "rock", "indie", "moderno"],
    "pedal":     ["pedal", "drone", "estático", "estatico", "bordón"],
    "ostinato":  ["ostinato", "ciaccona", "passacaglia", "chacona", "basso continuo"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BassNote:
    """Una nota en la línea de bajo."""
    pitch:    int    # MIDI pitch
    duration: float  # quarter notes
    velocity: int
    offset:   float  # posición en quarter notes
    technique: str = ""  # T1-T10, "root", "fifth", "third", "seventh", etc.

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12

    @property
    def name(self) -> str:
        octave = self.pitch // 12 - 1
        return f"{NOTE_NAMES[self.pitch_class]}{octave}"


@dataclass
class ChordEvent:
    """Un acorde con su posición temporal."""
    root_pc:  int
    quality:  str
    offset:   float  # beats desde el inicio
    duration: float  # beats
    name:     str = ""

    @property
    def tones(self) -> List[int]:
        ivs = CHORD_INTERVALS.get(self.quality, CHORD_INTERVALS[""])
        return [(self.root_pc + iv) % 12 for iv in ivs]

    @property
    def root_name(self) -> str:
        return NOTE_NAMES[self.root_pc]


@dataclass
class BassResult:
    """Resultado completo de una línea de bajo generada."""
    notes:     List[BassNote]
    chords:    List[ChordEvent]
    key:       str
    mode:      str
    bars:      int
    tempo:     int
    style:     str
    score:     float = 0.0
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key":    self.key,
            "mode":   self.mode,
            "bars":   self.bars,
            "tempo":  self.tempo,
            "style":  self.style,
            "score":  round(self.score, 4),
            "total_notes": len(self.notes),
            "notes": [
                {"pitch": n.pitch, "name": n.name,
                 "duration": n.duration, "velocity": n.velocity,
                 "offset": round(n.offset, 4), "technique": n.technique}
                for n in self.notes
            ],
            "technique_breakdown": dict(Counter(n.technique for n in self.notes)),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES TONALES
# ══════════════════════════════════════════════════════════════════════════════

def parse_key(key_str: str) -> Tuple[int, str]:
    """'Am', 'Dm', 'C major', 'F# minor' → (root_pc, mode)."""
    key_str = key_str.strip()
    parts   = key_str.split()

    mode = "major"
    for m in ["harmonic_minor", "melodic_minor", "phrygian_dominant",
              "dorian", "phrygian", "lydian", "mixolydian", "locrian",
              "blues", "whole_tone", "pentatonic_minor", "pentatonic_major"]:
        if m.replace("_", " ") in key_str.lower() or m in key_str.lower():
            mode = m
            break
    else:
        if "minor" in key_str.lower():
            mode = "minor"
        elif "major" in key_str.lower():
            mode = "major"
        elif len(parts) == 1:
            s = parts[0]
            # Detecta 'Am', 'Dm', 'F#m', 'Bbm', etc. — el 'm' final no forma parte de 'maj'
            if re.search(r'^[A-G][#b]?m$', s):
                mode = "minor"

    root_str = re.sub(r"m(ajor|inor)?$", "", parts[0],
                      flags=re.IGNORECASE).strip()

    if root_str in ENHARMONIC:
        root_pc = ENHARMONIC[root_str]
    else:
        clean   = root_str.replace("b", "").replace("#", "")
        note_up = clean.upper()
        base    = NOTE_NAMES.index(note_up) if note_up in NOTE_NAMES else 0
        root_pc = (base + root_str.count("#") - root_str.count("b")) % 12

    return root_pc, mode


def parse_chord_name(name: str) -> Tuple[int, str]:
    """'Cm', 'G7', 'F#m7', 'Bbsus4' → (root_pc, quality)."""
    name = name.strip()
    if len(name) >= 2 and name[1] in ("#", "b"):
        root_str, suffix = name[:2], name[2:]
    else:
        root_str, suffix = name[:1], name[1:]

    root_norm = root_str[0].upper() + root_str[1:]
    if root_norm in ENHARMONIC:
        root_pc = ENHARMONIC[root_norm]
    else:
        base = NOTE_NAMES.index(root_norm.replace("b","").replace("#","").upper()) \
               if root_norm.replace("b","").replace("#","").upper() in NOTE_NAMES else 0
        root_pc = (base + root_str.count("#") - root_str.count("b")) % 12

    quality_map = [
        ("hdim7","hdim7"),("dim7","dim7"),("m7b5","hdim7"),
        ("mM7","mM7"),("maj7","M7"),("M7","M7"),("m9","m9"),
        ("M9","M9"),("add9","add9"),("m7","m7"),("7","7"),
        ("sus4","sus4"),("sus2","sus2"),("dim","dim"),("aug","aug"),
        ("m6","m6"),("6","6"),("m","m"),("5","5"),("9","9"),("",""),
    ]
    quality = ""
    s = suffix.strip()
    for suffix_key, q in quality_map:
        if s.startswith(suffix_key):
            quality = q
            break

    return root_pc % 12, quality


def parse_chords_text(text: str) -> List[Tuple[str, float]]:
    """'Cm G7 Fm Bb7' o 'Cm:2 G7:2' → [(name, beats), ...]."""
    tokens = text.strip().split()
    result = []
    for tok in tokens:
        if ":" in tok:
            name, beats = tok.rsplit(":", 1)
            result.append((name, float(beats)))
        else:
            result.append((tok, 4.0))
    return result


def get_scale_pitches(root_pc: int, mode: str,
                      low: int = 24, high: int = 60) -> List[int]:
    """Todos los MIDI pitches de la escala en el rango dado."""
    ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    result = []
    for oct_ in range(11):
        for iv in ivs:
            p = oct_ * 12 + root_pc + iv
            if low <= p <= high:
                result.append(p)
    return sorted(result)


def chord_pitches_in_range(chord: ChordEvent, low: int, high: int,
                            prefer_low: bool = True) -> List[int]:
    """Genera los MIDI pitches del acorde en el rango del bajo."""
    ivs   = CHORD_INTERVALS.get(chord.quality, CHORD_INTERVALS[""])
    result = []
    for oct_ in range(11):
        base = oct_ * 12 + chord.root_pc
        for iv in ivs:
            p = base + iv
            if low <= p <= high:
                result.append(p)
    result = sorted(set(result))
    if prefer_low:
        return result
    return list(reversed(result))


def best_root(chord: ChordEvent, prev_pitch: Optional[int],
              low: int, high: int) -> int:
    """Elige la mejor octava de la raíz aplicando voice-leading mínimo."""
    candidates = []
    for oct_ in range(11):
        p = oct_ * 12 + chord.root_pc
        if low <= p <= high:
            candidates.append(p)
    if not candidates:
        return low + chord.root_pc % 12

    if prev_pitch is None:
        # Preferir el centro del rango
        mid = (low + high) // 2
        return min(candidates, key=lambda p: abs(p - mid))

    # Movimiento mínimo desde el pitch anterior
    return min(candidates, key=lambda p: abs(p - prev_pitch))


def approach_chromatic(target: int, from_above: bool) -> int:
    """Semitono de aproximación cromática al target."""
    return target + (1 if from_above else -1)


def approach_diatonic(target: int, scale_pitches: List[int],
                       from_above: bool) -> Optional[int]:
    """Grado de escala adyacente al target."""
    if target not in scale_pitches:
        return None
    idx = scale_pitches.index(target)
    if from_above and idx + 1 < len(scale_pitches):
        return scale_pitches[idx + 1]
    elif not from_above and idx - 1 >= 0:
        return scale_pitches[idx - 1]
    return None


def snap_to_range(pitch: int, low: int, high: int) -> int:
    """Proyecta al rango por octavas."""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def infer_style_from_description(desc: str) -> str:
    """Detecta el estilo de bajo desde texto libre."""
    dl = desc.lower()
    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in dl for kw in keywords):
            return style
    return "auto"


def infer_style_from_chords(chords: List[ChordEvent]) -> str:
    """
    Infiere el estilo más apropiado según complejidad armónica.
    Acordes de 7ª extendida → walking; tríadas simples → classical o pop.
    """
    if not chords:
        return "classical"
    extended = sum(1 for c in chords if c.quality in
                   ("7", "M7", "m7", "9", "M9", "m9", "hdim7", "dim7"))
    ratio = extended / len(chords)
    if ratio > 0.5:
        return "walking"
    avg_dur = sum(c.duration for c in chords) / len(chords)
    if avg_dur <= 2.0:
        return "pop"
    return "classical"


def swing_offset(beat_pos: float, swing_ratio: float = 0.67) -> float:
    """
    Aplica swing a una posición de corchea.
    swing_ratio=0.67 → triplet swing estándar.
    """
    # Solo afecta a corcheas en posición débil (0.5, 1.5, 2.5, 3.5)
    frac = beat_pos % 1.0
    if abs(frac - 0.5) < 0.1:
        return beat_pos - 0.5 + swing_ratio
    return beat_pos


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def read_midi_notes(midi_path: str) -> Tuple[List[Tuple], int, int]:
    """
    Lee un MIDI y devuelve (notes, tempo_bpm, ticks_per_beat).
    notes = [(offset_beats, pitch, duration_beats, velocity), ...]
    """
    if not MIDO_OK:
        raise RuntimeError("mido requerido para leer MIDI")
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    notes = []
    pending: Dict[Tuple, Tuple] = {}

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
            elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                key = (msg.channel, msg.note)
                if key in pending:
                    on_t, vel = pending.pop(key)
                    dur_ticks = abs_t - on_t
                    notes.append((
                        on_t  / tpb,
                        msg.note,
                        max(0.1, dur_ticks / tpb),
                        vel,
                    ))

    notes.sort(key=lambda n: n[0])
    tempo_bpm = round(60_000_000 / max(tempo_us, 1))
    return notes, tempo_bpm, tpb


def extract_chords_from_midi(midi_path: str,
                              beats_per_bar: float = 4.0) -> List[ChordEvent]:
    """
    Extrae acordes de un MIDI usando detección de pitch classes por compás.
    """
    notes, _, _ = read_midi_notes(midi_path)
    if not notes:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    total_beats = max(o + d for o, _, d, _ in notes)
    n_bars = max(1, math.ceil(total_beats / beats_per_bar))
    chords = []

    for bar in range(n_bars):
        s = bar * beats_per_bar
        e = s + beats_per_bar
        bar_notes = [(o, p, d, v) for o, p, d, v in notes
                     if o < e and o + d > s]
        if not bar_notes:
            if chords:
                prev = chords[-1]
                chords.append(ChordEvent(prev.root_pc, prev.quality,
                                         s, beats_per_bar, prev.name))
            continue

        pcs = {p % 12 for _, p, _, _ in bar_notes}
        lowest = min(p for _, p, _, _ in bar_notes)
        root_pc = lowest % 12
        quality = _infer_quality(root_pc, pcs)
        name = NOTE_NAMES[root_pc] + quality
        chords.append(ChordEvent(root_pc, quality, s, beats_per_bar, name))

    return chords


def _infer_quality(root_pc: int, pcs: set) -> str:
    """Infiere la calidad de un acorde desde un conjunto de pitch classes."""
    rel = {(pc - root_pc) % 12 for pc in pcs}
    if 4 in rel and 7 in rel and 10 in rel:
        return "7"
    if 3 in rel and 7 in rel and 10 in rel:
        return "m7"
    if 4 in rel and 7 in rel and 11 in rel:
        return "M7"
    if 3 in rel and 6 in rel:
        return "dim"
    if 4 in rel and 8 in rel:
        return "aug"
    if 5 in rel and 7 in rel:
        return "sus4"
    if 3 in rel and 7 in rel:
        return "m"
    if 4 in rel and 7 in rel:
        return ""
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE PATRONES POR ESTILO
# ══════════════════════════════════════════════════════════════════════════════

class BassStyleGenerator:
    """Base para todos los generadores de estilo."""

    def __init__(self, root_pc: int, mode: str, low: int, high: int,
                 beats_per_bar: float, rng: random.Random,
                 approach_prob: float = 0.6, swing: float = 0.0,
                 tension_curve: Optional[List[float]] = None):
        self.root_pc       = root_pc
        self.mode          = mode
        self.low           = low
        self.high          = high
        self.bpb           = beats_per_bar
        self.rng           = rng
        self.approach_prob = approach_prob
        self.swing         = swing
        self.tension_curve = tension_curve or []
        self.scale_pitches = get_scale_pitches(root_pc, mode, low - 12, high + 12)
        self.scale_pitches = [p for p in self.scale_pitches if low <= p <= high]

    def tension_at(self, bar: int) -> float:
        if not self.tension_curve:
            return 0.5
        return float(self.tension_curve[min(bar, len(self.tension_curve) - 1)])

    def dyn_vel(self, bar: int, base: int = 80) -> int:
        t = self.tension_at(bar)
        return int(np.clip(base - 20 + int(t * 40), 30, 110))

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: WALKING
# ─────────────────────────────────────────────────────────────────────────────

class WalkingBassGenerator(BassStyleGenerator):
    """
    Walking bass jazzístico.
    Cada tiempo = una negra. Lógica:
      tiempo 1 → raíz del acorde
      tiempo 2 → 3ª, 5ª o nota de paso hacia la 5ª
      tiempo 3 → 5ª o 7ª (según tensión)
      tiempo 4 → approach note (cromática o diatónica) a la siguiente raíz [T1/T2]
    Con enclosure [T3] y arpegio descendente [T4] opcionales.
    """

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None

        for i, chord in enumerate(chords):
            bar_idx = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx)
            t_val = self.tension_at(bar_idx)

            next_chord = chords[i + 1] if i + 1 < len(chords) else chord

            # Raíz (tiempo 1)
            root = best_root(chord, prev_pitch, self.low, self.high)

            # Tones del acorde en rango
            chord_ps = chord_pitches_in_range(chord, self.low, self.high)
            if not chord_ps:
                chord_ps = [root]

            def nearest_chord_tone(target: int) -> int:
                return min(chord_ps, key=lambda p: abs(p - target))

            # 3ª, 5ª, 7ª relativas al root actual
            third  = snap_to_range(root + (3 if "m" in chord.quality else 4),
                                    self.low, self.high)
            fifth  = snap_to_range(root + 7, self.low, self.high)
            seventh = snap_to_range(
                root + (10 if "7" in chord.quality or "m7" in chord.quality else 9),
                self.low, self.high)

            # Siguiente raíz (para approach)
            next_root = best_root(next_chord, root, self.low, self.high)

            # Tiempo 4: approach note
            approach = self._choose_approach(root, next_root, chord, bar_idx)

            # Decisión de técnica para tiempos 2 y 3 según tensión
            if t_val > 0.7:
                # Alta tensión: arpegio descendente [T4]
                walk = [root, fifth, third, approach]
                techs = ["root", "fifth", "third", "approach"]
            elif t_val > 0.4:
                # Tensión media: mezcla funcional
                walk = [root, third if self.rng.random() > 0.5 else fifth,
                        seventh, approach]
                techs = ["root",
                         "third" if walk[1] == third else "fifth",
                         "seventh", "approach"]
            else:
                # Baja tensión: simple 1-5-1-approach
                walk = [root, fifth, root, approach]
                techs = ["root", "fifth", "root", "approach"]

            # Duración de cada paso
            n_steps = min(int(self.bpb), len(walk))
            step_dur = chord.duration / n_steps

            for step in range(n_steps):
                beat_pos = chord.offset + step * step_dur
                # Swing en el 4º tiempo
                swung_pos = beat_pos
                if self.swing > 0 and step == 3:
                    swung_pos = chord.offset + swing_offset(
                        step * step_dur, 0.5 + self.swing * 0.17)

                vel = vel_base if step == 0 else int(vel_base * 0.85)
                p = max(self.low, min(self.high, walk[step]))

                notes.append(BassNote(
                    pitch=p,
                    duration=step_dur * 0.9,
                    velocity=vel,
                    offset=round(swung_pos, 4),
                    technique=techs[step],
                ))

            prev_pitch = root

        return notes

    def _choose_approach(self, current_root: int, next_root: int,
                          chord: ChordEvent, bar_idx: int) -> int:
        """Elige la mejor approach note hacia next_root [T1/T2/T3]."""
        if self.rng.random() > self.approach_prob:
            # Sin approach: vuelve a la raíz o usa 7ª
            return snap_to_range(current_root + 10, self.low, self.high)

        dist = next_root - current_root
        if abs(dist) > 2:
            # Approach cromática desde abajo o arriba
            from_above = dist < 0
            ap = approach_chromatic(next_root, from_above=not from_above)
        else:
            # Enclosure [T3]: desde el semitono opuesto
            ap = next_root + (1 if next_root < current_root else -1)

        return max(self.low, min(self.high, ap))


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: CLASSICAL
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalBassGenerator(BassStyleGenerator):
    """
    Bajo clásico/romántico.
    Tiempo 1: raíz (con octava baja si 4/4).
    Tiempos débiles: 5ª, y si tensión alta, 7ª o 3ª.
    Contempla notas de paso escalares [T5] entre cambios de acorde.
    También puede generar contrapunto melódico contra la soprano [T10].
    """

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None
        melody_by_beat: Dict[float, int] = {}
        if melody:
            for o, p, d, v in melody:
                melody_by_beat[round(o, 2)] = p

        for i, chord in enumerate(chords):
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx)
            t_val    = self.tension_at(bar_idx)

            root   = best_root(chord, prev_pitch, self.low, self.high)
            fifth  = snap_to_range(root + 7, self.low, self.high)
            third  = snap_to_range(root + (3 if "m" in chord.quality else 4),
                                    self.low, self.high)
            seventh = snap_to_range(
                root + (10 if "7" in chord.quality else 9),
                self.low, self.high)

            # Calcular movimiento contrapuntístico [T10]
            cp_direction = 0  # 0=neutral, 1=up, -1=down
            if melody:
                mel_at = melody_by_beat.get(round(chord.offset, 2))
                if mel_at:
                    cp_direction = -1 if mel_at > 60 else 1

            if self.bpb == 4:
                # 4/4: root (2 beats), 5ª (1 beat), weak (1 beat)
                weak = seventh if t_val > 0.6 else (third if t_val > 0.3 else fifth)
                bar_notes = [
                    (chord.offset,           root,  2.0, vel_base,   "root"),
                    (chord.offset + 2.0,     fifth, 1.0, vel_base-10, "fifth"),
                    (chord.offset + 3.0,     weak,  1.0, vel_base-15, "seventh" if weak == seventh else "third"),
                ]
            elif self.bpb == 3:
                # 3/4: root (2 beats), 5ª (1 beat)
                bar_notes = [
                    (chord.offset,       root,  2.0, vel_base,   "root"),
                    (chord.offset + 2.0, fifth, 1.0, vel_base-10, "fifth"),
                ]
            else:
                # 2/4 o 6/8: root + 5ª
                half = chord.duration / 2
                bar_notes = [
                    (chord.offset,        root,  half, vel_base,   "root"),
                    (chord.offset + half, fifth, half, vel_base-10, "fifth"),
                ]

            # Notas de paso [T5] si el salto a la siguiente raíz es grande
            next_chord = chords[i + 1] if i + 1 < len(chords) else None
            passing_notes = []
            if next_chord and t_val > 0.4:
                next_root = best_root(next_chord, root, self.low, self.high)
                if abs(next_root - root) > 3:
                    passing_notes = self._passing_notes(root, next_root)

            for offset, pitch, dur, vel, tech in bar_notes:
                p = max(self.low, min(self.high, pitch))
                notes.append(BassNote(
                    pitch=p, duration=dur * 0.88, velocity=vel,
                    offset=round(offset, 4), technique=tech,
                ))

            # Insertar notas de paso en el último beat
            if passing_notes and bar_notes:
                last_offset = bar_notes[-1][0]
                last_dur    = bar_notes[-1][2]
                if len(passing_notes) <= 2:
                    sub = last_dur / (len(passing_notes) + 1)
                    for j, pp in enumerate(passing_notes):
                        notes.append(BassNote(
                            pitch=max(self.low, min(self.high, pp)),
                            duration=sub * 0.85,
                            velocity=vel_base - 20,
                            offset=round(last_offset + (j + 1) * sub, 4),
                            technique="passing",
                        ))

            prev_pitch = root

        return notes

    def _passing_notes(self, from_p: int, to_p: int) -> List[int]:
        """Notas de paso escalares entre dos pitches [T5]."""
        direction = 1 if to_p > from_p else -1
        result = []
        for sp in self.scale_pitches:
            if direction == 1 and from_p < sp < to_p:
                result.append(sp)
            elif direction == -1 and to_p < sp < from_p:
                result.append(sp)
        return sorted(result, key=lambda x: direction * x)[:2]


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: BAROQUE (Bajo continuo)
# ─────────────────────────────────────────────────────────────────────────────

class BaroqueBassGenerator(BassStyleGenerator):
    """
    Bajo continuo barroco.
    Se mueve principalmente por grados conjuntos de la escala.
    Evita saltos de > 6ª. Resuelve tensiones hacia la tónica.
    Las disonancias (7ª) se preparan y resuelven por grado descendente.
    """

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None

        for i, chord in enumerate(chords):
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx, base=70)
            t_val    = self.tension_at(bar_idx)

            root = best_root(chord, prev_pitch, self.low, self.high)

            # Baroque: preferir movimiento conjunto sobre saltos
            if prev_pitch is not None and abs(root - prev_pitch) > 7:
                alt = snap_to_range(root + 12
                                    if root < prev_pitch else root - 12,
                                    self.low, self.high)
                if abs(alt - prev_pitch) < abs(root - prev_pitch):
                    root = alt

            # Ornamentación en tiempos: figuras de 2 notas (tiempo fuerte + weak)
            fifth = snap_to_range(root + 7, self.low, self.high)
            if self.bpb == 4:
                # Bajo con figura de corcheas ocasionalmente
                if t_val > 0.5 and chord.duration >= 2:
                    # Figura ornamental: root-passing-fifth-root
                    passing = self._scale_step(root, fifth)
                    half_dur = chord.duration / 4
                    bar_ns = [
                        (chord.offset,                root,    half_dur*2, vel_base,    "root"),
                        (chord.offset + half_dur*2,   passing, half_dur,   vel_base-15, "passing"),
                        (chord.offset + half_dur*3,   fifth,   half_dur,   vel_base-10, "fifth"),
                    ]
                else:
                    bar_ns = [
                        (chord.offset,           root,  chord.duration*0.6, vel_base,    "root"),
                        (chord.offset + chord.duration*0.6, fifth, chord.duration*0.4, vel_base-10, "fifth"),
                    ]
            else:
                bar_ns = [(chord.offset, root, chord.duration, vel_base, "root")]

            for offset, pitch, dur, vel, tech in bar_ns:
                p = max(self.low, min(self.high, pitch))
                notes.append(BassNote(
                    pitch=p, duration=dur * 0.9, velocity=vel,
                    offset=round(offset, 4), technique=tech,
                ))

            prev_pitch = root

        return notes

    def _scale_step(self, from_p: int, to_p: int) -> int:
        """Un grado escalar entre dos pitches."""
        direction = 1 if to_p > from_p else -1
        for sp in self.scale_pitches:
            if direction == 1 and from_p < sp < to_p:
                return sp
            elif direction == -1 and to_p < sp < from_p:
                return sp
        return from_p + direction  # semitono si no hay grado


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: LATIN (Tumbao)
# ─────────────────────────────────────────────────────────────────────────────

class LatinBassGenerator(BassStyleGenerator):
    """
    Tumbao cubano (son/salsa).
    Figura característica: anticipación de la raíz en el 'and' del 4
    (4+), luego raíz en el 1, y cinco en el 'and' del 2 (2+).
    Notación en 4/4: [silencio en 1, root en 1+, root en 2, 5th en 3+, root en 4+]
    """

    # Tumbao: offsets dentro del compás (en quarter notes) y qué tono
    # (0=root, 1=fifth, 2=octave_root, 3=approach)
    TUMBAO_PATTERN_44 = [
        # (beat_offset, tone_type, dur, vel_factor)
        (0.5,  0, 0.5, 0.95),   # root en 1+
        (1.0,  0, 0.5, 0.90),   # root en 2
        (2.5,  1, 0.5, 0.85),   # fifth en 3+
        (3.0,  0, 0.5, 0.90),   # root en 4
        (3.5,  3, 0.5, 0.80),   # approach en 4+
    ]

    TUMBAO_PATTERN_34 = [
        (0.0,  0, 1.0, 1.00),   # root en 1
        (1.5,  1, 0.5, 0.85),   # fifth en 2+
        (2.0,  0, 1.0, 0.90),   # root en 3
    ]

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None
        pattern = self.TUMBAO_PATTERN_44 if self.bpb == 4 else self.TUMBAO_PATTERN_34

        for i, chord in enumerate(chords):
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx)
            t_val    = self.tension_at(bar_idx)

            root   = best_root(chord, prev_pitch, self.low, self.high)
            fifth  = snap_to_range(root + 7, self.low, self.high)
            root_oct = snap_to_range(root + 12, self.low, self.high)  # octava alta

            # Approach a la siguiente raíz
            next_chord  = chords[i + 1] if i + 1 < len(chords) else chord
            next_root   = best_root(next_chord, root, self.low, self.high)
            approach_p  = approach_chromatic(next_root, from_above=(next_root < root))

            tones = {0: root, 1: fifth, 2: root_oct, 3: approach_p}

            for beat_off, tone_type, dur, vel_f in pattern:
                # Escalar duración si el acorde dura < 4 beats
                if chord.duration < self.bpb:
                    ratio = chord.duration / self.bpb
                    beat_off_scaled = beat_off * ratio
                    dur_scaled = dur * ratio
                else:
                    beat_off_scaled = beat_off
                    dur_scaled = dur

                abs_offset = chord.offset + beat_off_scaled
                if abs_offset >= chord.offset + chord.duration:
                    continue

                p = max(self.low, min(self.high, tones.get(tone_type, root)))
                vel = int(vel_base * vel_f)

                tech_map = {0: "root", 1: "fifth", 2: "root_octave", 3: "approach"}
                notes.append(BassNote(
                    pitch=p, duration=dur_scaled * 0.85, velocity=vel,
                    offset=round(abs_offset, 4),
                    technique=tech_map.get(tone_type, "root"),
                ))

            prev_pitch = root

        return notes


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: BLUES (Shuffle / Boogie)
# ─────────────────────────────────────────────────────────────────────────────

class BluesBassGenerator(BassStyleGenerator):
    """
    Blues shuffle / boogie-woogie.
    Figura característica: root-5th-6th-flat7th en corcheas con swing.
    Variaciones: boogie ascendente, riff de 12 compases.
    """

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None
        sw = 0.5 + self.swing * 0.17  # ratio de swing

        for chord in chords:
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx, base=85)
            t_val    = self.tension_at(bar_idx)

            root = best_root(chord, prev_pitch, self.low, self.high)
            # Tonos del blues shuffle
            fifth     = snap_to_range(root + 7,  self.low, self.high)
            sixth     = snap_to_range(root + 9,  self.low, self.high)
            flat_sev  = snap_to_range(root + 10, self.low, self.high)
            octave    = snap_to_range(root + 12, self.low, self.high)

            # Patrón boogie: root-5-6-b7 / root-5-6-b7
            if t_val > 0.5:
                # Boogie ascendente
                pattern_pitches = [root, fifth, sixth, flat_sev,
                                    octave, flat_sev, sixth, fifth]
            else:
                # Shuffle simple: root-5 alternado
                pattern_pitches = [root, fifth, root, fifth,
                                    root, fifth, root, fifth]

            n_per_bar = min(len(pattern_pitches), int(self.bpb * 2))
            dur_each  = chord.duration / n_per_bar

            for step in range(n_per_bar):
                frac    = step % 2  # 0=downbeat, 1=upbeat
                # Swing en los upbeats
                if frac == 1 and self.swing > 0:
                    offset = chord.offset + (step - 1) * dur_each + dur_each * sw
                else:
                    offset = chord.offset + step * dur_each

                p   = max(self.low, min(self.high, pattern_pitches[step]))
                vel = vel_base if frac == 0 else int(vel_base * 0.82)
                notes.append(BassNote(
                    pitch=p,
                    duration=dur_each * (sw if frac == 0 else (1 - sw + 0.1)),
                    velocity=vel,
                    offset=round(offset, 4),
                    technique="root" if p == root else "blues_tone",
                ))

            prev_pitch = root

        return notes


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: POP
# ─────────────────────────────────────────────────────────────────────────────

class PopBassGenerator(BassStyleGenerator):
    """
    Bajo de pop/rock.
    Root en 1, octava en 3, fills en el 4º tiempo [T8].
    Anticipaciones ocasionales [T6].
    """

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        prev_pitch = None

        for i, chord in enumerate(chords):
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx, base=88)
            t_val    = self.tension_at(bar_idx)

            root     = best_root(chord, prev_pitch, self.low, self.high)
            root_oct = snap_to_range(root + 12, self.low, self.high)
            fifth    = snap_to_range(root + 7, self.low, self.high)

            next_chord = chords[i + 1] if i + 1 < len(chords) else chord
            next_root  = best_root(next_chord, root, self.low, self.high)

            if self.bpb == 4:
                # tiempo 1: root fuerte
                notes.append(BassNote(root, 0.9, vel_base,
                                       round(chord.offset, 4), "root"))
                # tiempo 2: quinta o silencio
                if t_val > 0.3:
                    notes.append(BassNote(fifth, 0.9, int(vel_base * 0.75),
                                           round(chord.offset + 1.0, 4), "fifth"))
                # tiempo 3: octava [T8]
                notes.append(BassNote(root_oct, 0.9, int(vel_base * 0.85),
                                       round(chord.offset + 2.0, 4), "octave"))
                # tiempo 4: fill hacia siguiente raíz [T6] o quinta
                if self.rng.random() < self.approach_prob and t_val > 0.4:
                    ap = approach_chromatic(next_root,
                                            from_above=(next_root < root))
                    ap = max(self.low, min(self.high, ap))
                    notes.append(BassNote(ap, 0.9, int(vel_base * 0.80),
                                           round(chord.offset + 3.0, 4), "approach"))
                else:
                    notes.append(BassNote(fifth, 0.9, int(vel_base * 0.78),
                                           round(chord.offset + 3.0, 4), "fifth"))
            else:
                # 3/4: root-root-fifth
                notes.append(BassNote(root,  0.9, vel_base,
                                       round(chord.offset, 4), "root"))
                notes.append(BassNote(root,  0.9, int(vel_base * 0.80),
                                       round(chord.offset + 1.0, 4), "root"))
                notes.append(BassNote(fifth, 0.9, int(vel_base * 0.75),
                                       round(chord.offset + 2.0, 4), "fifth"))

            prev_pitch = root

        return notes


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: PEDAL
# ─────────────────────────────────────────────────────────────────────────────

class PedalBassGenerator(BassStyleGenerator):
    """
    Pedal: nota sostenida (tónica o dominante) bajo armonías cambiantes [T9].
    Opcionalmente alterna entre tónica y dominante en cada compás.
    """

    def __init__(self, *args, pedal_pitch: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Nota de pedal: si no se especifica, usar la tónica en octava baja
        if pedal_pitch is not None:
            self.pedal_pitch = snap_to_range(pedal_pitch, self.low, self.high)
        else:
            # Tónica en la octava más baja del rango
            candidates = [p for p in range(self.low, self.high + 1)
                          if p % 12 == self.root_pc]
            self.pedal_pitch = candidates[0] if candidates else self.low + self.root_pc % 12

        dom_pc = (self.root_pc + 7) % 12
        dom_candidates = [p for p in range(self.low, self.high + 1)
                          if p % 12 == dom_pc]
        self.dominant_pitch = dom_candidates[0] if dom_candidates else self.pedal_pitch + 7

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes = []
        # Dividir la obra en dos mitades: primera = tónica, segunda = dominante
        total_beats = sum(c.duration for c in chords)
        half_beats  = total_beats / 2

        accumulated = 0.0
        for chord in chords:
            bar_idx  = int(chord.offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx)
            t_val    = self.tension_at(bar_idx)

            # Cambio de pedal según posición
            p = self.pedal_pitch if accumulated < half_beats else self.dominant_pitch
            accumulated += chord.duration

            # En alta tensión: articular el pedal en negras
            if t_val > 0.6 and chord.duration >= 2:
                n_pulses = int(chord.duration)
                dur_each = chord.duration / n_pulses
                for pulse in range(n_pulses):
                    vel = vel_base if pulse == 0 else int(vel_base * 0.78)
                    notes.append(BassNote(
                        pitch=p, duration=dur_each * 0.88,
                        velocity=vel,
                        offset=round(chord.offset + pulse * dur_each, 4),
                        technique="pedal",
                    ))
            else:
                # Nota larga
                notes.append(BassNote(
                    pitch=p, duration=chord.duration * 0.95,
                    velocity=vel_base,
                    offset=round(chord.offset, 4),
                    technique="pedal",
                ))

        return notes


# ─────────────────────────────────────────────────────────────────────────────
#  ESTILO: OSTINATO (Passacaglia / Ciaccona)
# ─────────────────────────────────────────────────────────────────────────────

class OstinatoBassGenerator(BassStyleGenerator):
    """
    Ostinato: figura rítmica/melódica que se repite de forma cíclica.
    Ideal para pasacalles, chaconne, basso ostinato barroco.
    El patrón puede especificarse externamente o generarse automáticamente.
    """

    def __init__(self, *args, pattern: Optional[List[Tuple[int, float]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # pattern: [(midi_pitch, duration_beats), ...]
        self.pattern = pattern or self._generate_default_pattern()

    def _generate_default_pattern(self) -> List[Tuple[int, float]]:
        """
        Genera un patrón de descenso tetracordal (La menor → Mi),
        la figura más común de los basso ostinati barrocos.
        """
        root = snap_to_range(self.root_pc + 5 * 12, self.low, self.high)
        # Descenso por grados de escala desde la tónica a la dominante
        ivs  = SCALE_INTERVALS.get(self.mode, SCALE_INTERVALS["minor"])
        start_idx = 0
        for i, iv in enumerate(ivs):
            if (root + iv) % 12 == self.root_pc:
                start_idx = i
                break

        tones = []
        for step in range(5):  # tónica → dominante (4 pasos = 5 notas)
            idx = (start_idx - step) % len(ivs)
            p   = root + ivs[idx] - (12 if step > 0 and ivs[idx] > ivs[start_idx] else 0)
            p   = snap_to_range(p, self.low, self.high)
            tones.append(p)

        return [(p, 1.0) for p in tones]  # negras

    def generate(self, chords: List[ChordEvent],
                 melody: Optional[List[Tuple]] = None) -> List[BassNote]:
        notes   = []
        total_b = sum(c.duration for c in chords)
        pat_dur = sum(d for _, d in self.pattern)

        offset  = 0.0
        pat_idx = 0

        while offset < total_b - 0.01:
            bar_idx  = int(offset / self.bpb)
            vel_base = self.dyn_vel(bar_idx)
            t_val    = self.tension_at(bar_idx)

            pitch, dur = self.pattern[pat_idx % len(self.pattern)]
            dur = min(dur, total_b - offset)

            # Leve variación de velocidad según tensión y posición en patrón
            pos_in_pat = (pat_idx % len(self.pattern)) / len(self.pattern)
            vel = int(vel_base * (0.85 + 0.15 * (1 - pos_in_pat)))

            p = max(self.low, min(self.high, pitch))
            notes.append(BassNote(
                pitch=p, duration=dur * 0.92, velocity=vel,
                offset=round(offset, 4), technique="ostinato",
            ))

            offset  += dur
            pat_idx += 1

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING DE LA LÍNEA DE BAJO
# ══════════════════════════════════════════════════════════════════════════════

def score_bass_line(notes: List[BassNote], chords: List[ChordEvent],
                    scale_pitches: List[int],
                    melody: Optional[List[Tuple]] = None,
                    beats_per_bar: float = 4.0) -> float:
    """
    Puntuación multicriterio de la línea de bajo [0, 1].

    Criterios:
      1. Consonancia armónica: % de notas que son tones del acorde en curso
      2. Solidez rítmica: raíz en tiempo fuerte
      3. Suavidad de movimiento: penalizar saltos excesivos
      4. Rango apropiado: ni demasiado agudo ni demasiado grave
      5. Variedad: diversidad de pitches y duraciones
      6. Contrapunto: movimiento contrario a la melodía (si existe)
    """
    if not notes:
        return 0.0

    pitches = [n.pitch for n in notes]
    durs    = [n.duration for n in notes]

    # Mapa rápido: beat → chord
    def chord_at(beat: float) -> Optional[ChordEvent]:
        for c in chords:
            if c.offset <= beat < c.offset + c.duration:
                return c
        return chords[-1] if chords else None

    # 1. Consonancia armónica
    harmonic_hits = 0
    for n in notes:
        c = chord_at(n.offset)
        if c and n.pitch_class in c.tones:
            harmonic_hits += 1
    consonance = harmonic_hits / len(notes)

    # 2. Solidez rítmica: notas en tiempo 1 de cada compás
    downbeat_roots = 0
    n_bars = max(1, int(max(n.offset for n in notes) / beats_per_bar) + 1)
    for bar in range(n_bars):
        bar_start = bar * beats_per_bar
        bar_notes = [n for n in notes
                     if abs(n.offset - bar_start) < 0.2]
        c = chord_at(bar_start)
        if bar_notes and c and bar_notes[0].pitch_class == c.root_pc:
            downbeat_roots += 1
    downbeat_score = downbeat_roots / n_bars

    # 3. Suavidad: penalizar saltos > octava entre notas consecutivas
    intervals  = [abs(pitches[i+1] - pitches[i])
                  for i in range(len(pitches) - 1)]
    big_leaps  = sum(1 for iv in intervals if iv > 12)
    smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 4. Rango apropiado
    center = (min(pitches) + max(pitches)) / 2
    rng_score = 1.0 - abs(center - 38) / 20  # centro ideal ~38 (D2)
    rng_score = max(0.0, min(1.0, rng_score))

    # 5. Variedad
    variety = min(1.0, len(set(pitches)) / max(len(pitches) * 0.5, 1))

    # 6. Contrapunto con melodía [T10]
    cp_score = 0.5  # neutral si no hay melodía
    if melody:
        contrary_motions = 0
        parallel_motions = 0
        mel_sorted = sorted(melody, key=lambda x: x[0])
        for i in range(len(mel_sorted) - 1):
            m_off1, m_p1 = mel_sorted[i][0],   mel_sorted[i][1]
            m_off2, m_p2 = mel_sorted[i+1][0], mel_sorted[i+1][1]
            b_ns = [n for n in notes if m_off1 <= n.offset < m_off2]
            if len(b_ns) >= 2:
                b_dir = b_ns[-1].pitch - b_ns[0].pitch
                m_dir = m_p2 - m_p1
                if b_dir * m_dir < 0:
                    contrary_motions += 1
                elif b_dir * m_dir > 0:
                    parallel_motions += 1
        total_m = contrary_motions + parallel_motions
        if total_m > 0:
            cp_score = contrary_motions / total_m

    return (consonance    * 0.30 +
            downbeat_score * 0.25 +
            smoothness     * 0.20 +
            rng_score      * 0.10 +
            variety        * 0.05 +
            cp_score       * 0.10)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE LA LÍNEA GENERADA
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bass_line(result: BassResult) -> Dict[str, Any]:
    """Genera un informe analítico de la línea de bajo."""
    notes = result.notes
    if not notes:
        return {}

    pitches   = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    vels      = [n.velocity for n in notes]
    techs     = Counter(n.technique for n in notes)

    intervals = [abs(pitches[i+1] - pitches[i])
                 for i in range(len(pitches) - 1)]

    return {
        "total_notes":       len(notes),
        "pitch_range":       {"min": min(pitches), "max": max(pitches),
                               "min_name": f"{NOTE_NAMES[min(pitches)%12]}{min(pitches)//12-1}",
                               "max_name": f"{NOTE_NAMES[max(pitches)%12]}{max(pitches)//12-1}",
                               "span_semitones": max(pitches) - min(pitches)},
        "pitch_mean":        round(float(np.mean(pitches)), 2),
        "duration_variety":  len(set(round(d, 2) for d in durations)),
        "mean_interval":     round(float(np.mean(intervals)), 2) if intervals else 0,
        "max_leap":          max(intervals) if intervals else 0,
        "velocity_range":    {"min": min(vels), "max": max(vels)},
        "technique_breakdown": dict(techs),
        "most_used_technique": techs.most_common(1)[0][0] if techs else "root",
        "score":             round(result.score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT: MIDI
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_midi(bass_notes: List[BassNote], melody_notes: Optional[List[Tuple]],
                   tempo_bpm: int, output_path: str,
                   ticks_per_beat: int = 480,
                   bass_program: int = 33,
                   include_melody: bool = False):
    """
    Exporta la línea de bajo a MIDI.
    Canal 0 = bajo; canal 1 = melodía (si include_melody=True).
    """
    if not MIDO_OK:
        print("[ERROR] mido no disponible.")
        return

    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def q2t(q: float) -> int:
        return max(0, int(round(q * ticks_per_beat)))

    # ── Pista de bajo ────────────────────────────────────────────────────────
    bass_track = mido.MidiTrack()
    mid.tracks.append(bass_track)
    bass_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    bass_track.append(mido.MetaMessage("track_name", name="Bass", time=0))
    bass_track.append(mido.Message("program_change", channel=0,
                                    program=bass_program, time=0))

    events = []
    for n in bass_notes:
        on_t  = q2t(n.offset)
        off_t = q2t(n.offset + n.duration)
        events.append((on_t,  "note_on",  0, n.pitch, n.velocity))
        events.append((off_t, "note_off", 0, n.pitch, 0))

    if include_melody and melody_notes:
        bass_track.append(mido.Message("program_change", channel=1,
                                        program=73, time=0))  # flauta
        for o, p, d, v in melody_notes:
            on_t  = q2t(o)
            off_t = q2t(o + d)
            events.append((on_t,  "note_on",  1, p, v))
            events.append((off_t, "note_off", 1, p, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
    cur_tick = 0
    for abs_tick, msg_type, ch, note, vel in events:
        delta = max(0, abs_tick - cur_tick)
        bass_track.append(mido.Message(msg_type, channel=ch,
                                        note=note, velocity=vel, time=delta))
        cur_tick = abs_tick

    mid.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT: FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════

def build_fingerprint(result: BassResult, output_path: str) -> dict:
    """Fingerprint compatible con stitcher.py."""
    notes = result.notes
    if not notes:
        return {}

    pitches = [n.pitch for n in notes]
    vels    = [n.velocity for n in notes]
    durs    = [n.duration for n in notes]

    root_pc, _ = parse_key(result.key)
    ivs   = SCALE_INTERVALS.get(result.mode, SCALE_INTERVALS["major"])
    scale_pcs = [(root_pc + iv) % 12 for iv in ivs]

    first_n = notes[:max(1, len(notes)//8)]
    last_n  = notes[-max(1, len(notes)//8):]

    return {
        "meta": {
            "source":      "bass_line_composer.py",
            "style":       result.style,
            "key":         result.key,
            "mode":        result.mode,
            "bars":        result.bars,
            "tempo_bpm":   result.tempo,
            "score":       result.score,
            "total_notes": len(notes),
        },
        "entry": {
            "key":             result.key,
            "mode":            result.mode,
            "dominant_pc":     Counter(n.pitch % 12 for n in first_n).most_common(1)[0][0],
            "mean_pitch":      float(np.mean([n.pitch for n in first_n])),
            "mean_velocity":   float(np.mean([n.velocity for n in first_n])),
            "density_per_beat": len(first_n) / max(4.0, 1),
            "tempo_bpm":       result.tempo,
        },
        "exit": {
            "key":             result.key,
            "mode":            result.mode,
            "dominant_pc":     Counter(n.pitch % 12 for n in last_n).most_common(1)[0][0],
            "mean_pitch":      float(np.mean([n.pitch for n in last_n])),
            "mean_velocity":   float(np.mean([n.velocity for n in last_n])),
            "density_per_beat": len(last_n) / max(4.0, 1),
            "last_pitch":      notes[-1].pitch,
            "last_pitch_pc":   notes[-1].pitch_class,
            "last_duration":   notes[-1].duration,
        },
        "tension": {
            "mean": float(np.mean(vels)) / 127,
            "max":  float(np.max(vels))  / 127,
            "min":  float(np.min(vels))  / 127,
            "std":  float(np.std(vels))  / 127,
        },
        "style": {
            "mean_pitch":    float(np.mean(pitches)),
            "pitch_range":   int(np.max(pitches) - np.min(pitches)),
            "mean_duration": float(np.mean(durs)),
            "scale_pcs":     scale_pcs,
            "bass_style":    result.style,
        },
    }


def export_fingerprint(fp: dict, output_path: str) -> str:
    json_path = output_path.replace(".mid", ".fingerprint.json")
    if not json_path.endswith(".fingerprint.json"):
        json_path += ".fingerprint.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)
    print(f"[fingerprint] {json_path}")
    return json_path


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIONES (loaders)
# ══════════════════════════════════════════════════════════════════════════════

def load_from_theorist(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    pp          = data.get("pipeline_parameters", {})
    narrator_pp = pp.get("narrator", {})
    midi_pp     = pp.get("midi_dna_unified", {})
    cpg_pp      = pp.get("chord_progression_generator", {})
    tension     = data.get("tension_curves", {})

    key    = midi_pp.get("--key",   narrator_pp.get("--key",   "C"))
    mode   = midi_pp.get("--mode",  narrator_pp.get("--mode",  "major"))
    bars   = int(midi_pp.get("--bars",  narrator_pp.get("--bars",  16)))
    tempo  = int(midi_pp.get("--tempo", narrator_pp.get("--tempo", 120)))
    style  = cpg_pp.get("--style", "auto")

    return {
        "key": key, "mode": mode, "bars": bars, "tempo": tempo,
        "style": style,
        "tension_curve": tension.get("tension", []),
    }


def load_from_narrator(path: str) -> dict:
    with open(path) as f:
        plan = json.load(f)
    key      = plan.get("key", "C")
    tempo    = int(plan.get("tempo", 120))
    sections = plan.get("sections", [])
    bars     = sum(s.get("bars", 8) for s in sections)
    tc       = plan.get("tension_curves", {})
    return {
        "key": key, "bars": bars, "tempo": tempo,
        "tension_curve": tc.get("tension", []),
    }


def load_from_curves(path: str) -> dict:
    with open(path) as f:
        curves = json.load(f)
    return {"tension_curve": curves.get("tension", [])}


def load_chords_json(path: str) -> List[Tuple[str, float]]:
    """Lee un .chords.json de chord_progression_generator."""
    with open(path) as f:
        data = json.load(f)
    # Formato esperado: {"chords": [{"name": "Cm", "duration": 4}, ...]}
    raw = data.get("chords", data.get("progression", []))
    result = []
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name", item.get("chord", "C"))
            dur  = float(item.get("duration", item.get("beats", 4)))
            result.append((name, dur))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append((str(item[0]), float(item[1])))
    return result


def parse_ostinato_pattern(text: str, low: int, high: int) -> List[Tuple[int, float]]:
    """
    Parsea un patrón de ostinato: "D3:2 C3:1 Bb2:1" →
    [(MIDI_pitch, duration_beats), ...]
    """
    result = []
    for token in text.strip().split():
        if ":" in token:
            note_str, dur_str = token.rsplit(":", 1)
            dur = float(dur_str)
        else:
            note_str, dur = token, 1.0

        # Parsear nota: "D3", "Bb2", "F#2"
        m = re.match(r"([A-G][#b]?)(-?\d+)", note_str)
        if m:
            note_name_raw = m.group(1)
            octave        = int(m.group(2))
            note_up = note_name_raw[0].upper() + note_name_raw[1:]
            if note_up in ENHARMONIC:
                pc = ENHARMONIC[note_up]
            else:
                pc = NOTE_NAMES.index(note_up.replace("b","").replace("#","").upper()) \
                     if note_up.replace("b","").replace("#","").upper() in NOTE_NAMES else 0
                pc = (pc + note_name_raw.count("#") - note_name_raw.count("b")) % 12
            midi_pitch = (octave + 1) * 12 + pc
            midi_pitch = snap_to_range(midi_pitch, low, high)
            result.append((midi_pitch, dur))

    return result if result else [(low + 2, 1.0), (low, 1.0)]  # fallback


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class BassLineComposer:
    """
    Coordina la generación de la línea de bajo completa.
    """

    VALID_PARAMS = {
        "key", "mode", "bars", "tempo", "time_sig", "style",
        "bass_low", "bass_high", "tension_curve", "approach_prob",
        "swing", "seed", "verbose", "pedal_pitch", "ostinato_pattern",
    }

    def __init__(self, key: str = "C", mode: str = "auto",
                 bars: int = 16, tempo: int = 120,
                 time_sig: str = "4/4", style: str = "auto",
                 bass_low: int = BASS_LOW_DEFAULT,
                 bass_high: int = BASS_HIGH_DEFAULT,
                 tension_curve: Optional[List[float]] = None,
                 approach_prob: float = 0.6, swing: float = 0.0,
                 pedal_pitch: Optional[int] = None,
                 ostinato_pattern: Optional[List[Tuple[int, float]]] = None,
                 seed: int = 42, verbose: bool = False):

        self.key           = key
        self.bars          = bars
        self.tempo         = tempo
        self.style         = style
        self.bass_low      = bass_low
        self.bass_high     = bass_high
        self.approach_prob = approach_prob
        self.swing         = swing
        self.verbose       = verbose
        self.pedal_pitch   = pedal_pitch
        self.ostinato_pat  = ostinato_pattern

        # Parsear compás
        num, den        = time_sig.split("/")
        self.bpb        = float(num)
        self.beat_unit  = int(den)

        # RNG
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Parsear tonalidad
        self.root_pc, inferred_mode = parse_key(key)
        self.mode = inferred_mode if mode == "auto" else mode

        # Curva de tensión
        self.tension_curve = tension_curve or []
        if len(self.tension_curve) < bars:
            # Extender con interpolación o rellenar con 0.5
            if self.tension_curve:
                self.tension_curve = list(np.interp(
                    np.linspace(0, 1, bars),
                    np.linspace(0, 1, len(self.tension_curve)),
                    self.tension_curve,
                ))
            else:
                self.tension_curve = [0.5] * bars

        if verbose:
            self._print_config()

    def _print_config(self):
        print(f"┌─ BASS LINE COMPOSER ────────────────────────────")
        print(f"│  Tonalidad : {self.key} {self.mode}")
        print(f"│  Estilo    : {self.style}")
        print(f"│  Compases  : {self.bars} ({int(self.bpb)}/4)")
        print(f"│  Tempo     : {self.tempo} BPM")
        print(f"│  Rango     : MIDI {self.bass_low}-{self.bass_high}")
        print(f"│  Swing     : {self.swing:.2f}  Approach: {self.approach_prob:.2f}")
        print(f"└─────────────────────────────────────────────────")

    def _build_chords_from_text(self, text: str) -> List[ChordEvent]:
        tokens = parse_chords_text(text)
        chords = []
        offset = 0.0
        for name, beats in tokens:
            rpc, qual = parse_chord_name(name)
            chords.append(ChordEvent(rpc, qual, offset, beats, name))
            offset += beats
        # Repetir si no cubre los compases solicitados
        total_needed = self.bars * self.bpb
        while offset < total_needed - 0.01:
            for c in list(chords):
                if offset >= total_needed:
                    break
                dur = min(c.duration, total_needed - offset)
                chords.append(ChordEvent(c.root_pc, c.quality, offset, dur, c.name))
                offset += dur
        return [c for c in chords if c.offset < total_needed]

    def _default_chords(self) -> List[ChordEvent]:
        """Progresión diatónica básica como fallback."""
        ivs = SCALE_INTERVALS.get(self.mode, SCALE_INTERVALS["major"])
        degrees = [0, 5, 3, 4]  # I-VI-IV-V (en índices de la escala)
        qualities = ["", "m", "m", ""] if self.mode == "major" else ["m", "", "", ""]

        chords  = []
        offset  = 0.0
        total   = self.bars * self.bpb
        bar_dur = self.bpb

        for i in range(self.bars):
            deg_idx = degrees[i % len(degrees)]
            pc      = (self.root_pc + ivs[min(deg_idx, len(ivs)-1)]) % 12
            qual    = qualities[i % len(qualities)]
            name    = NOTE_NAMES[pc] + qual
            chords.append(ChordEvent(pc, qual, offset, bar_dur, name))
            offset += bar_dur

        return chords

    def _get_generator(self, style: str,
                        chords: List[ChordEvent]) -> BassStyleGenerator:
        """Instancia el generador para el estilo dado."""
        common = dict(
            root_pc=self.root_pc, mode=self.mode,
            low=self.bass_low, high=self.bass_high,
            beats_per_bar=self.bpb, rng=self.rng,
            approach_prob=self.approach_prob, swing=self.swing,
            tension_curve=self.tension_curve,
        )
        if style == "walking":
            return WalkingBassGenerator(**common)
        elif style == "classical":
            return ClassicalBassGenerator(**common)
        elif style == "baroque":
            return BaroqueBassGenerator(**common)
        elif style == "latin":
            return LatinBassGenerator(**common)
        elif style == "blues":
            return BluesBassGenerator(**common)
        elif style == "pop":
            return PopBassGenerator(**common)
        elif style == "pedal":
            return PedalBassGenerator(**common, pedal_pitch=self.pedal_pitch)
        elif style == "ostinato":
            return OstinatoBassGenerator(**common, pattern=self.ostinato_pat)
        else:
            return WalkingBassGenerator(**common)

    def compose(self, chords: Optional[List[ChordEvent]] = None,
                chords_text: Optional[str] = None,
                melody: Optional[List[Tuple]] = None) -> BassResult:
        """
        Genera la línea de bajo completa.
        chords: lista de ChordEvent ya construida
        chords_text: progresión en texto (alternativa a chords)
        melody: notas de melodía para contrapunto
        """
        # Resolver acordes
        if chords is None:
            if chords_text:
                chords = self._build_chords_from_text(chords_text)
            else:
                chords = self._default_chords()

        # Resolver estilo
        effective_style = self.style
        if effective_style == "auto":
            effective_style = infer_style_from_chords(chords)
            if self.verbose:
                print(f"  [auto-style] → {effective_style}")

        # Generar
        gen   = self._get_generator(effective_style, chords)
        notes = gen.generate(chords, melody)

        # Score
        scale_ps = get_scale_pitches(self.root_pc, self.mode,
                                      self.bass_low, self.bass_high)
        score = score_bass_line(notes, chords, scale_ps, melody, self.bpb)

        return BassResult(
            notes=notes, chords=chords,
            key=self.key, mode=self.mode,
            bars=self.bars, tempo=self.tempo,
            style=effective_style, score=score,
        )

    def compose_candidates(self, n: int,
                            chords: Optional[List[ChordEvent]] = None,
                            chords_text: Optional[str] = None,
                            melody: Optional[List[Tuple]] = None
                            ) -> List[BassResult]:
        """Genera n candidatos variando la semilla y los ordena por score."""
        results = []
        base_rng_state = self.rng.getstate()
        for i in range(n):
            self.rng = random.Random(self.rng.randint(0, 2**31))
            res = self.compose(chords=deepcopy(chords) if chords else None,
                                chords_text=chords_text, melody=melody)
            results.append(res)
            if self.verbose:
                print(f"  [candidato {i+1}/{n}] score={res.score:.3f} "
                      f"notas={len(res.notes)}")
        self.rng.setstate(base_rng_state)
        return sorted(results, key=lambda r: -r.score)


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def play_midi(path: str, seconds: float = 15.0):
    if not PYGAME_OK:
        print("[WARN] pygame no disponible.")
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        import time; time.sleep(seconds)
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"[WARN] Error reproduciendo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BASS LINE COMPOSER — Línea de bajo como voz compositiva autónoma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("description", nargs="?", default=None,
                   help="Descripción libre en lenguaje natural")

    # Parámetros musicales
    p.add_argument("--key",      default="C",
                   help="Tonalidad (C, Am, F#, Bb…)")
    p.add_argument("--mode",     default="auto",
                   choices=list(SCALE_INTERVALS.keys()) + ["auto"])
    p.add_argument("--bars",     type=int,   default=16)
    p.add_argument("--tempo",    type=int,   default=120)
    p.add_argument("--time-sig", default="4/4")
    p.add_argument("--style",    default="auto",
                   choices=["walking", "classical", "baroque", "latin",
                            "blues", "pop", "pedal", "ostinato", "auto"])

    # Entrada armónica
    p.add_argument("--chords",       type=str, default=None,
                   help="Progresión en texto: 'C Am F G' o 'Cm:2 G7:2'")
    p.add_argument("--chords-midi",  type=str, default=None,
                   help="Pista de acordes en MIDI")
    p.add_argument("--chords-json",  type=str, default=None,
                   help="Acordes desde .chords.json")
    p.add_argument("--melody-midi",  type=str, default=None,
                   help="Melodía MIDI para contrapunto automático")

    # Rango
    p.add_argument("--range", nargs=2, type=int,
                   default=[BASS_LOW_DEFAULT, BASS_HIGH_DEFAULT],
                   metavar=("LOW", "HIGH"))

    # Parámetros de estilo
    p.add_argument("--approach-prob", type=float, default=0.6)
    p.add_argument("--swing",         type=float, default=0.0)
    p.add_argument("--ostinato-pattern", type=str, default=None,
                   help="Patrón para ostinato: 'D3:2 C3:1 Bb2:1'")
    p.add_argument("--pedal-note",    type=str,   default=None,
                   help="Nota de pedal para style=pedal (ej: 'D2')")

    # Candidatos
    p.add_argument("--candidates",    type=int,   default=1)

    # Integraciones
    p.add_argument("--curves",        type=str, default=None)
    p.add_argument("--from-theorist", type=str, default=None)
    p.add_argument("--from-narrator", type=str, default=None)

    # Exportación
    p.add_argument("--export-fingerprint", action="store_true")
    p.add_argument("--export-combined",    action="store_true",
                   help="Exportar MIDI con bajo + melodía juntos")
    p.add_argument("--output", type=str, default="bass_out.mid")

    # Control
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--listen",      action="store_true")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Parámetros base ──────────────────────────────────────────────────────
    params = {
        "key":           args.key,
        "mode":          args.mode,
        "bars":          args.bars,
        "tempo":         args.tempo,
        "time_sig":      args.time_sig,
        "style":         args.style,
        "bass_low":      args.range[0],
        "bass_high":     args.range[1],
        "approach_prob": args.approach_prob,
        "swing":         args.swing,
        "seed":          args.seed,
        "verbose":       args.verbose,
        "tension_curve": None,
    }

    chords_text  = args.chords
    melody_notes = None

    # ── Descripción libre ────────────────────────────────────────────────────
    if args.description:
        desc = args.description
        detected_style = infer_style_from_description(desc)
        if detected_style != "auto" and params["style"] == "auto":
            params["style"] = detected_style

        # Tonalidad desde descripción
        m = re.search(r"\b([A-G][b#]?)\s*(m(?:enor|inor|)\b|mayor\b|major\b)?",
                       desc, re.IGNORECASE)
        if m and args.key == "C":
            root_str = m.group(1)
            qual_str = (m.group(2) or "").lower()
            if qual_str in ("m", "minor", "menor"):
                params["key"] = root_str + "m"
            else:
                params["key"] = root_str

        # Tempo desde descripción
        dl = desc.lower()
        if any(w in dl for w in ["lento", "slow", "pausado"]):
            params["tempo"] = 60
        elif any(w in dl for w in ["rápido", "fast", "veloz", "allegro"]):
            params["tempo"] = 160

        if args.verbose:
            print(f"[descripción] '{desc}'")
            print(f"  → estilo={params['style']}  key={params['key']}")

    # ── Desde theorist ───────────────────────────────────────────────────────
    if args.from_theorist:
        if not os.path.exists(args.from_theorist):
            print(f"[ERROR] No se encuentra: {args.from_theorist}")
            sys.exit(1)
        ext = load_from_theorist(args.from_theorist)
        for k, v in ext.items():
            if k in BassLineComposer.VALID_PARAMS and v is not None:
                params[k] = v
        if args.verbose:
            print(f"[theorist] Cargado: {args.from_theorist}")

    # ── Desde narrator ───────────────────────────────────────────────────────
    if args.from_narrator:
        if not os.path.exists(args.from_narrator):
            print(f"[ERROR] No se encuentra: {args.from_narrator}")
            sys.exit(1)
        ext = load_from_narrator(args.from_narrator)
        for k, v in ext.items():
            if k in BassLineComposer.VALID_PARAMS and v is not None:
                params[k] = v
        if args.verbose:
            print(f"[narrator] Cargado: {args.from_narrator}")

    # ── Curvas ───────────────────────────────────────────────────────────────
    if args.curves:
        if not os.path.exists(args.curves):
            print(f"[ERROR] No se encuentra: {args.curves}")
            sys.exit(1)
        ext = load_from_curves(args.curves)
        if ext.get("tension_curve"):
            params["tension_curve"] = ext["tension_curve"]

    # ── Modo auto de escala ──────────────────────────────────────────────────
    if params["mode"] == "auto":
        _, inferred = parse_key(params["key"])
        params["mode"] = inferred

    # ── Ostinato pattern ─────────────────────────────────────────────────────
    ostinato_pat = None
    if args.ostinato_pattern:
        ostinato_pat = parse_ostinato_pattern(
            args.ostinato_pattern, params["bass_low"], params["bass_high"])
        params["style"] = "ostinato"

    # ── Pedal note ───────────────────────────────────────────────────────────
    pedal_pitch = None
    if args.pedal_note:
        m = re.match(r"([A-G][#b]?)(-?\d+)", args.pedal_note.strip())
        if m:
            note_raw = m.group(1)
            octave   = int(m.group(2))
            note_up  = note_raw[0].upper() + note_raw[1:]
            pc = ENHARMONIC.get(note_up,
                 NOTE_NAMES.index(note_up.replace("b","").replace("#","").upper())
                 if note_up.replace("b","").replace("#","").upper() in NOTE_NAMES else 0)
            pedal_pitch = (octave + 1) * 12 + pc
        params["style"] = "pedal"

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── PARÁMETROS RESUELTOS (dry-run) ─────────────────")
        for k, v in params.items():
            if k != "tension_curve":
                print(f"  {k:<22} = {v}")
        print("───────────────────────────────────────────────────")
        sys.exit(0)

    # ── Resolver acordes ─────────────────────────────────────────────────────
    chords: Optional[List[ChordEvent]] = None

    if args.chords_json:
        if not os.path.exists(args.chords_json):
            print(f"[ERROR] No se encuentra: {args.chords_json}")
            sys.exit(1)
        tokens = load_chords_json(args.chords_json)
        chords_text = " ".join(
            f"{name}:{dur}" for name, dur in tokens)
        if args.verbose:
            print(f"[chords-json] {len(tokens)} acordes cargados")

    if args.chords_midi:
        if not os.path.exists(args.chords_midi):
            print(f"[ERROR] No se encuentra: {args.chords_midi}")
            sys.exit(1)
        try:
            bpb = float(args.time_sig.split("/")[0])
            chords = extract_chords_from_midi(args.chords_midi, bpb)
            params["bars"] = max(params["bars"],
                                  math.ceil(
                                      sum(c.duration for c in chords) / bpb))
            if args.verbose:
                print(f"[chords-midi] {len(chords)} acordes extraídos")
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # ── Melodía ───────────────────────────────────────────────────────────────
    if args.melody_midi:
        if not os.path.exists(args.melody_midi):
            print(f"[ERROR] No se encuentra: {args.melody_midi}")
            sys.exit(1)
        try:
            melody_notes, mel_tempo, _ = read_midi_notes(args.melody_midi)
            if melody_notes and args.tempo == 120:
                params["tempo"] = mel_tempo
            if args.verbose:
                print(f"[melody-midi] {len(melody_notes)} notas cargadas")
        except Exception as e:
            print(f"[WARN] No se pudo leer melodía: {e}")
            melody_notes = None

    # ── Instanciar compositor ─────────────────────────────────────────────────
    valid_init = {k: v for k, v in params.items()
                  if k in BassLineComposer.VALID_PARAMS}
    if ostinato_pat:
        valid_init["ostinato_pattern"] = ostinato_pat
    if pedal_pitch is not None:
        valid_init["pedal_pitch"] = pedal_pitch

    try:
        composer = BassLineComposer(**valid_init)
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar: {e}")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Generar ───────────────────────────────────────────────────────────────
    output_base = args.output.replace(".mid", "")
    n_cands = max(1, args.candidates)

    print(f"\nGenerando {n_cands} candidato(s) — estilo: {params['style']}…")

    results = composer.compose_candidates(
        n=n_cands,
        chords=chords,
        chords_text=chords_text,
        melody=melody_notes,
    )

    # ── Exportar ──────────────────────────────────────────────────────────────
    output_paths = []
    for i, res in enumerate(results):
        suffix  = "" if i == 0 else f"_v{i+1}"
        out_p   = f"{output_base}{suffix}.mid"

        bass_prog = GM_BASS_PROGRAMS.get(
            STYLE_TO_PROGRAM.get(res.style, "fingered"), 33)

        notes_to_midi(res.notes, melody_notes, res.tempo, out_p,
                       bass_program=bass_prog,
                       include_melody=False)
        output_paths.append(out_p)

        tag = "★ MEJOR" if i == 0 else f"  v{i+1}"
        print(f"  {tag}  score={res.score:.3f}  "
              f"notas={len(res.notes)}  estilo={res.style}  → {out_p}")

    best     = results[0]
    best_p   = output_paths[0]

    # JSON de la línea
    json_path = best_p.replace(".mid", ".bass.json")
    analysis  = analyze_bass_line(best)
    out_data  = best.to_dict()
    out_data["analysis"] = analysis
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"[json]    {json_path}")

    # Combined MIDI (bajo + melodía)
    if args.export_combined and melody_notes:
        comb_p = best_p.replace(".mid", ".combined.mid")
        bass_prog = GM_BASS_PROGRAMS.get(
            STYLE_TO_PROGRAM.get(best.style, "fingered"), 33)
        notes_to_midi(best.notes, melody_notes, best.tempo, comb_p,
                       bass_program=bass_prog, include_melody=True)
        print(f"[combined] {comb_p}")

    # Fingerprint
    if args.export_fingerprint:
        fp = build_fingerprint(best, best_p)
        export_fingerprint(fp, best_p)

    # Reproducir
    if args.listen:
        print(f"\nReproduciendo {best_p}…")
        play_midi(best_p, seconds=min(20, best.bars * 2))

    # ── Resumen ────────────────────────────────────────────────────────────────
    analysis = analyze_bass_line(best)
    print(f"\n╔═ RESUMEN ════════════════════════════════════════╗")
    print(f"║  Tonalidad  : {best.key} {best.mode}")
    print(f"║  Estilo     : {best.style}")
    print(f"║  Compases   : {best.bars}  Tempo: {best.tempo} BPM")
    print(f"║  Notas      : {len(best.notes)}")
    print(f"║  Rango      : {analysis.get('pitch_range',{}).get('min_name','?')} → {analysis.get('pitch_range',{}).get('max_name','?')}")
    print(f"║  Score      : {best.score:.3f}")
    techs = analysis.get("technique_breakdown", {})
    print(f"║  Técnicas   : {dict(list(techs.items())[:4])}")
    print(f"╚═════════════════════════════════════════════════╝")
    print()
    print("Siguientes pasos sugeridos:")
    print(f"  voice_leader.py --chords '{args.chords or 'C Am F G'}' --soprano melody_out.mid")
    print(f"  stitcher.py {best_p.replace('.mid','.fingerprint.json')} *.fingerprint.json")
    print(f"  orchestrator.py {best_p} --template chamber")


if __name__ == "__main__":
    main()
