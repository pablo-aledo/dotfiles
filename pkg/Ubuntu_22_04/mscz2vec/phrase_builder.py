#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      PHRASE BUILDER  v1.0                                    ║
║         Motor de sintaxis melódica: antecedente / consecuente                ║
║                                                                              ║
║  Toma un motivo corto (2-4 compases, idealmente extraído por harvester.py)   ║
║  y lo expande en frases de 4/8/16 compases aplicando la lógica sintáctica   ║
║  clásica: antecedente → consecuente, pregunta → respuesta, semi-cadencia    ║
║  → cadencia auténtica. El resultado suena inevitablemente "correcto" porque  ║
║  sigue la gramática con la que el oído occidental está entrenado.            ║
║                                                                              ║
║  CONCEPTO CENTRAL:                                                           ║
║  Una frase musical es un par:                                                ║
║    Antecedente (A): presenta la idea, termina con tensión (semicadencia)    ║
║    Consecuente (C): responde la idea, termina con reposo (cadencia auténtica)║
║                                                                              ║
║  Tipos de par A→C:                                                           ║
║    parallel    — C es casi igual a A pero con cadencia diferente (Bach)     ║
║    contrasting — C tiene material nuevo pero mismo final (Haydn/Mozart)     ║
║    sequential  — C transpone el material de A (secuencia)                   ║
║    developmental — C desarrolla un motivo de A (Beethoven)                  ║
║    call_response — A y C alternan frases cortas (call & response)           ║
║                                                                              ║
║  FORMAS COMPLETAS GENERABLES:                                                ║
║    period       — [A: ant + cons]                          (8 compases)     ║
║    double_period — [A: ant + cons] [B: ant + cons]         (16 compases)   ║
║    sentence     — [Básica: 2c] [Repetición: 2c] [Liquidación: 4c]          ║
║    bar_form     — [Stollen: 2c] [Stollen: 2c] [Abgesang: 4c] (AAB)        ║
║    binary       — [A: 8c cadencia dom] [B: 8c cadencia tónica]             ║
║    ternary      — [A: 8c] [B: 8c] [A': 8c]                                 ║
║    rondo_kernel — [A: 8c] [B: 4c] [A: 4c]                                  ║
║                                                                              ║
║  TIPOS DE CADENCIA:                                                          ║
║    AC   — auténtica perfecta   V→I  (reposo total)                          ║
║    IAC  — auténtica imperfecta V→I  (reposo parcial, soprano no en tónica) ║
║    HC   — semicadencia         I→V  o ii→V (tensión abierta)               ║
║    DC   — engañosa             V→vi (sorpresa)                              ║
║    PC   — plagal               IV→I (postscript, "amén")                   ║
║    modal — final modal (en modo dórico/frigio: acorde característico)       ║
║                                                                              ║
║  FUENTES DE MOTIVO:                                                          ║
║    --motif FILE      — MIDI de motivo base (extraído por harvester.py)       ║
║    --motif-text S    — secuencia en texto: "C4:1 E4:0.5 G4:0.5 F4:1"       ║
║    --generate        — genera motivo interno aleatorio (para explorar)       ║
║                                                                              ║
║  USO STANDALONE:                                                             ║
║    python phrase_builder.py --motif motivo.mid                               ║
║    python phrase_builder.py --motif motivo.mid --form period --type parallel ║
║    python phrase_builder.py --motif motivo.mid --form sentence --key Am     ║
║    python phrase_builder.py --motif motivo.mid --form binary --tempo 120    ║
║    python phrase_builder.py --motif motivo.mid --all-forms                  ║
║    python phrase_builder.py --generate --key Dm --form period               ║
║    python phrase_builder.py --motif motivo.mid --cadences HC AC --report    ║
║                                                                              ║
║  USO COMO MÓDULO:                                                            ║
║    from phrase_builder import build_phrase, PhraseResult, MotifNote         ║
║                                                                              ║
║    motif = [MotifNote(pitch=60, duration=1.0), MotifNote(pitch=64, dur=0.5)]║
║    result = build_phrase(                                                    ║
║        motif=motif,                                                          ║
║        key="C major",                                                        ║
║        form="period",                                                        ║
║        phrase_type="parallel",                                               ║
║        cadence_ant="HC",        # cadencia del antecedente                  ║
║        cadence_cons="AC",       # cadencia del consecuente                  ║
║        bars=8,                                                               ║
║        tempo=90,                                                             ║
║        seed=42,                                                              ║
║    )                                                                         ║
║    # result.notes → lista de (beat_offset, pitch, duration, velocity)       ║
║    # result.sections → {"antecedent": [...], "consequent": [...]}           ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --motif FILE        MIDI de motivo de entrada                             ║
║    --motif-text S      Motivo en formato texto                               ║
║    --generate          Generar motivo interno                                ║
║    --key KEY           Tonalidad (default: C major)                         ║
║    --form S            period|double_period|sentence|bar_form|binary|       ║
║                        ternary|rondo_kernel (default: period)               ║
║    --type S            parallel|contrasting|sequential|developmental|       ║
║                        call_response (default: parallel)                    ║
║    --cadences S S      Cadencias: antecedente consecuente (default: HC AC)  ║
║    --bars N            Compases totales (default: 8)                        ║
║    --tempo BPM         Tempo (default: 90)                                  ║
║    --all-forms         Generar una versión por cada forma                   ║
║    --all-types         Generar una versión por cada tipo de frase           ║
║    --report            Exportar análisis de la frase en JSON                ║
║    --output FILE       Fichero de salida (default: phrase_out.mid)          ║
║    --output-dir DIR    Directorio de salida con --all-forms                 ║
║    --annotate          Exportar también MusicXML con anotaciones (requiere  ║
║                        music21)                                              ║
║    --listen            Reproducir al terminar (requiere pygame)             ║
║    --verbose           Imprimir decisiones de construcción                  ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    phrase_out.mid              — MIDI con la frase completa                  ║
║    phrase_out.report.json      — con --report                               ║
║    phrase_out.musicxml         — con --annotate (secciones marcadas)        ║
║                                                                              ║
║  INTEGRACIÓN CON EL PIPELINE:                                                ║
║    harvester.py → phrase_builder.py → stitcher.py → orchestrator.py        ║
║    narrator.py define la forma; phrase_builder la instancia en notas.       ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:  mido, numpy                                                     ║
║    --annotate / análisis tonal: music21 (opcional)                          ║
║    --listen: pygame                                                          ║
║    Integración: midi_dna_unified en el mismo directorio                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import mido

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

try:
    from music21 import stream, note as m21note, key as m21key, metadata
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC  = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}

# Escalas (semitonos desde la tónica)
SCALE_INTERVALS = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],   # natural
    "harmonic":   [0, 2, 3, 5, 7, 8, 11],   # armónica
    "melodic":    [0, 2, 3, 5, 7, 9, 11],   # melódica ascendente
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian":    [0, 1, 3, 5, 6, 8, 10],
}

# Grados de escala que forman los acordes de cadencia
CADENCE_CHORDS = {
    # (grado_root_en_escala, calidad)  — índices 0-6 en la escala
    "AC":    [("V", "dom"), ("I",  "tonic")],
    "IAC":   [("V", "dom"), ("I",  "tonic")],     # soprano no en tónica
    "HC":    [("I", "tonic"), ("V", "dom")],
    "DC":    [("V", "dom"), ("vi", "submed")],
    "PC":    [("IV","sub"),  ("I",  "tonic")],
    "modal": [("bVII","sub"), ("i", "tonic")],
}

# Duraciones válidas en quarters
VALID_DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MotifNote:
    """Una nota dentro de un motivo."""
    pitch:    int     # MIDI pitch
    duration: float   # en quarter notes
    velocity: int = 70
    offset:   float = 0.0  # posición en quarter notes desde inicio del motivo

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12

    def transpose(self, semitones: int) -> "MotifNote":
        return MotifNote(
            pitch=self.pitch + semitones,
            duration=self.duration,
            velocity=self.velocity,
            offset=self.offset,
        )

    def at_offset(self, offset: float) -> "MotifNote":
        n = deepcopy(self)
        n.offset = offset
        return n


@dataclass
class PhraseSection:
    """Una sección de la frase (antecedente, consecuente, etc.)."""
    name:          str
    notes:         List[MotifNote] = field(default_factory=list)
    start_beat:    float = 0.0
    duration_bars: int = 4
    cadence_type:  str = "HC"
    cadence_notes: List[MotifNote] = field(default_factory=list)

    @property
    def all_notes(self) -> List[MotifNote]:
        return self.notes + self.cadence_notes


@dataclass
class PhraseResult:
    """Resultado completo de una frase construida."""
    form:       str
    phrase_type: str
    key:        str
    bars:       int
    tempo:      int
    sections:   List[PhraseSection] = field(default_factory=list)
    motif_used: List[MotifNote] = field(default_factory=list)

    @property
    def all_notes(self) -> List[MotifNote]:
        """Todas las notas en orden cronológico."""
        all_n = []
        for sec in self.sections:
            for note in sec.all_notes:
                all_n.append(note)
        return sorted(all_n, key=lambda n: n.offset)

    def to_report(self) -> Dict[str, Any]:
        return {
            "form":        self.form,
            "phrase_type": self.phrase_type,
            "key":         self.key,
            "bars":        self.bars,
            "tempo":       self.tempo,
            "sections": [
                {
                    "name":          s.name,
                    "start_beat":    s.start_beat,
                    "duration_bars": s.duration_bars,
                    "cadence_type":  s.cadence_type,
                    "note_count":    len(s.all_notes),
                }
                for s in self.sections
            ],
            "total_notes": len(self.all_notes),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES TONALES
# ══════════════════════════════════════════════════════════════════════════════

def parse_key(key_str: str) -> Tuple[int, str]:
    """
    Parsea 'C major', 'Am', 'Dm', 'F# minor', 'G dorian' →
    (root_pc, mode_name).
    """
    key_str = key_str.strip()
    parts = key_str.split()

    # Detectar modo
    mode = "major"
    # Primero buscar modos explícitos (dorian, phrygian, etc.)
    for m in ["dorian", "phrygian", "lydian", "mixolydian", "locrian",
              "harmonic", "melodic"]:
        if m in key_str.lower():
            mode = m
            break
    else:
        # Mayor/menor
        if "minor" in key_str.lower():
            mode = "minor"
        elif "major" in key_str.lower():
            mode = "major"
        elif len(parts) == 1:
            # Convención: Am, Dm → menor si la letra va seguida de 'm'
            root_part = parts[0]
            if len(root_part) >= 2 and root_part[-1] == "m" and root_part[-2] != "#":
                mode = "minor"

    # Extraer root
    root_str = parts[0].rstrip("mM")
    if root_str in ENHARMONIC:
        root_pc = ENHARMONIC[root_str]
    else:
        base = root_str[0].upper()
        note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
        root_pc = note_map.get(base, 0)
        if len(root_str) > 1:
            if root_str[1] == "#":
                root_pc = (root_pc + 1) % 12
            elif root_str[1] == "b":
                root_pc = (root_pc - 1) % 12

    return root_pc, mode


def get_scale_pitches(root_pc: int, mode: str, octave_start: int = 4) -> List[int]:
    """Devuelve lista de MIDI pitches de la escala en octave_start y octave_start+1."""
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    base = 12 * octave_start + root_pc
    pitches = []
    for i in intervals:
        pitches.append(base + i)
    for i in intervals:
        pitches.append(base + i + 12)
    return pitches


def snap_to_scale(pitch: int, root_pc: int, mode: str) -> int:
    """Ajusta un pitch al grado de escala más cercano."""
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    scale_pcs = set((root_pc + i) % 12 for i in intervals)
    if pitch % 12 in scale_pcs:
        return pitch
    # Buscar el más cercano
    for delta in range(1, 7):
        if (pitch + delta) % 12 in scale_pcs:
            return pitch + delta
        if (pitch - delta) % 12 in scale_pcs:
            return pitch - delta
    return pitch


def scale_degree_pitch(degree_idx: int, root_pc: int, mode: str,
                        octave: int = 4) -> int:
    """
    Devuelve el MIDI pitch del grado degree_idx (0=tónica, 4=dominante, etc.)
    en la octava especificada.
    """
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    degree_idx = degree_idx % len(intervals)
    extra_oct = degree_idx // len(intervals)
    interval = intervals[degree_idx % len(intervals)]
    return 12 * (octave + extra_oct) + root_pc + interval


def leading_tone_pc(root_pc: int, mode: str) -> int:
    """Devuelve el pitch class de la sensible de la tonalidad."""
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return (root_pc + intervals[-1]) % 12  # último grado de la escala


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MOTIVOS
# ══════════════════════════════════════════════════════════════════════════════

def load_motif_from_midi(midi_path: str) -> List[MotifNote]:
    """Lee un MIDI y extrae las notas como lista de MotifNote."""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    notes = []
    active: Dict[int, Tuple[float, int]] = {}
    current_ticks = 0

    for msg in mido.merge_tracks(mid.tracks):
        current_ticks += msg.time
        beat = current_ticks / tpb
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (beat, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                start_beat, vel = active.pop(msg.note)
                dur = beat - start_beat
                if dur > 0:
                    notes.append(MotifNote(
                        pitch=msg.note, duration=round(dur * 2) / 2,
                        velocity=vel, offset=start_beat,
                    ))

    if not notes:
        return _generate_random_motif("C", "major")

    # Normalizar offset al inicio
    min_off = min(n.offset for n in notes)
    for n in notes:
        n.offset -= min_off

    return sorted(notes, key=lambda n: n.offset)


def load_motif_from_text(text: str) -> List[MotifNote]:
    """
    Parsea "C4:1 E4:0.5 G4:0.5 F4:1" → lista de MotifNote.
    Formato: nota:duracion (C4=60, D4=62, etc.)
    """
    NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    notes = []
    offset = 0.0

    for token in text.strip().split():
        if ":" in token:
            note_str, dur_str = token.rsplit(":", 1)
            dur = float(dur_str)
        else:
            note_str = token
            dur = 1.0

        # Parsear nota: "C4", "F#4", "Bb3"
        if len(note_str) >= 2:
            letter = note_str[0].upper()
            rest = note_str[1:]
            if rest[0] in ("#", "b"):
                accidental = rest[0]
                octave_str = rest[1:]
            else:
                accidental = ""
                octave_str = rest
            try:
                octave = int(octave_str)
            except ValueError:
                octave = 4
            pc = NOTE_PC.get(letter, 0)
            if accidental == "#":
                pc += 1
            elif accidental == "b":
                pc -= 1
            pitch = 12 * (octave + 1) + pc
        else:
            pitch = 60

        notes.append(MotifNote(pitch=pitch, duration=dur, offset=offset))
        offset += dur

    return notes


def _generate_random_motif(key_str: str, mode: str, bars: int = 2,
                            seed: int = 42) -> List[MotifNote]:
    """Genera un motivo aleatorio en la tonalidad dada."""
    random.seed(seed)
    root_pc, mode = parse_key(key_str + " " + mode)
    scale = get_scale_pitches(root_pc, mode, octave_start=4)

    notes = []
    offset = 0.0
    total_beats = bars * 4

    while offset < total_beats - 0.5:
        remaining = total_beats - offset
        dur = random.choice([d for d in VALID_DURS if d <= remaining])
        pitch = random.choice(scale[:9])  # octava principal
        notes.append(MotifNote(pitch=pitch, duration=dur, offset=offset))
        offset += dur

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMACIONES DE MOTIVO
# ══════════════════════════════════════════════════════════════════════════════

def transpose_motif(motif: List[MotifNote], semitones: int) -> List[MotifNote]:
    """Transpone el motivo N semitonos."""
    return [n.transpose(semitones) for n in motif]


def invert_motif(motif: List[MotifNote], pivot_pitch: Optional[int] = None) -> List[MotifNote]:
    """Invierte el contorno melódico del motivo alrededor del pivot_pitch."""
    if not motif:
        return motif
    pivot = pivot_pitch or motif[0].pitch
    result = []
    for n in motif:
        interval = n.pitch - pivot
        new_pitch = pivot - interval
        result.append(MotifNote(
            pitch=max(36, min(96, new_pitch)),
            duration=n.duration,
            velocity=n.velocity,
            offset=n.offset,
        ))
    return result


def augment_motif(motif: List[MotifNote], factor: float = 2.0) -> List[MotifNote]:
    """Aumentación rítmica: multiplica todas las duraciones y offsets."""
    return [
        MotifNote(
            pitch=n.pitch,
            duration=n.duration * factor,
            velocity=n.velocity,
            offset=n.offset * factor,
        )
        for n in motif
    ]


def diminish_motif(motif: List[MotifNote], factor: float = 0.5) -> List[MotifNote]:
    """Diminución rítmica."""
    return augment_motif(motif, factor)


def liquidate_motif(motif: List[MotifNote], keep_intervals: int = 2) -> List[MotifNote]:
    """
    Liquidación: reduce el motivo a sus keep_intervals primeros intervalos
    esenciales, eliminando notas de relleno. Clave en forma de sentencia.
    """
    if len(motif) <= keep_intervals:
        return motif
    # Mantener las notas en los tiempos fuertes (offsets enteros o medios)
    strong = [n for n in motif if n.offset % 1.0 == 0.0]
    if len(strong) >= keep_intervals:
        result = strong[:keep_intervals]
        # Recalcular duraciones
        for i in range(len(result) - 1):
            result[i].duration = result[i + 1].offset - result[i].offset
        return result
    return motif[:keep_intervals]


def sequence_motif(motif: List[MotifNote], root_pc: int, mode: str,
                    steps: int = 1) -> List[MotifNote]:
    """
    Secuencia: transpone el motivo N grados de la escala hacia arriba/abajo.
    Mantiene los intervalos diatónicos.
    """
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    scale_pcs_sorted = [(root_pc + i) % 12 for i in intervals]

    def diatonic_transpose(pitch: int, n_steps: int) -> int:
        pc = pitch % 12
        # Encontrar posición en la escala
        if pc in scale_pcs_sorted:
            idx = scale_pcs_sorted.index(pc)
        else:
            # Snap al más cercano
            idx = min(range(len(scale_pcs_sorted)),
                      key=lambda i: abs(scale_pcs_sorted[i] - pc))
        new_idx = (idx + n_steps) % len(intervals)
        extra_oct = (idx + n_steps) // len(intervals)
        new_pc = scale_pcs_sorted[new_idx]
        octave = pitch // 12
        return 12 * (octave + extra_oct) + new_pc

    return [
        MotifNote(
            pitch=diatonic_transpose(n.pitch, steps),
            duration=n.duration,
            velocity=n.velocity,
            offset=n.offset,
        )
        for n in motif
    ]


def add_passing_notes(motif: List[MotifNote], root_pc: int, mode: str) -> List[MotifNote]:
    """Añade notas de paso entre saltos de más de una 3ª."""
    result = []
    for i, note in enumerate(motif):
        result.append(deepcopy(note))
        if i + 1 < len(motif):
            next_note = motif[i + 1]
            interval = abs(next_note.pitch - note.pitch)
            if interval >= 4 and note.duration >= 1.0:
                # Hay espacio para una nota de paso
                passing_pitch = snap_to_scale(
                    (note.pitch + next_note.pitch) // 2, root_pc, mode
                )
                pass_dur = note.duration / 2
                # Acortar la nota actual
                result[-1].duration = pass_dur
                # Añadir nota de paso
                result.append(MotifNote(
                    pitch=passing_pitch,
                    duration=pass_dur,
                    velocity=note.velocity - 10,
                    offset=note.offset + pass_dur,
                ))
    return result


def resolve_to_cadence(motif: List[MotifNote], cadence_type: str,
                        root_pc: int, mode: str,
                        beats_per_bar: int = 4) -> List[MotifNote]:
    """
    Ajusta las últimas notas del motivo para que terminen en la cadencia indicada.
    Devuelve las notas de cadencia (2 acordes = 2 × beats_per_bar).
    """
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    def pitch_for_degree(degree_semitones: int, target_range=(60, 80)):
        pc = (root_pc + degree_semitones) % 12
        for p in range(target_range[0], target_range[1] + 1):
            if p % 12 == pc:
                return p
        return 60 + degree_semitones

    # Construir las notas de cadencia
    cadence_notes = []
    if not motif:
        return cadence_notes

    last_offset = max(n.offset + n.duration for n in motif)

    if cadence_type == "AC":
        # V → I: dominante → tónica
        dom_pitch   = pitch_for_degree(intervals[4])  # 5° grado
        tonic_pitch = pitch_for_degree(0)
        cadence_notes = [
            MotifNote(dom_pitch,   beats_per_bar,     70, last_offset),
            MotifNote(tonic_pitch, beats_per_bar,     75, last_offset + beats_per_bar),
        ]
    elif cadence_type == "IAC":
        # V → I pero soprano en 3ª o 5ª
        dom_pitch   = pitch_for_degree(intervals[4])
        third_pitch = pitch_for_degree(intervals[2])  # 3° de la tónica
        cadence_notes = [
            MotifNote(dom_pitch,   beats_per_bar, 70, last_offset),
            MotifNote(third_pitch, beats_per_bar, 72, last_offset + beats_per_bar),
        ]
    elif cadence_type == "HC":
        # I → V: tónica → dominante (tensión abierta)
        tonic_pitch = pitch_for_degree(0)
        dom_pitch   = pitch_for_degree(intervals[4])
        cadence_notes = [
            MotifNote(tonic_pitch, beats_per_bar, 70, last_offset),
            MotifNote(dom_pitch,   beats_per_bar, 65, last_offset + beats_per_bar),
        ]
    elif cadence_type == "DC":
        # V → vi: engañosa
        dom_pitch    = pitch_for_degree(intervals[4])
        submed_pitch = pitch_for_degree(intervals[5])
        cadence_notes = [
            MotifNote(dom_pitch,    beats_per_bar, 70, last_offset),
            MotifNote(submed_pitch, beats_per_bar, 68, last_offset + beats_per_bar),
        ]
    elif cadence_type == "PC":
        # IV → I: plagal
        sub_pitch   = pitch_for_degree(intervals[3])
        tonic_pitch = pitch_for_degree(0)
        cadence_notes = [
            MotifNote(sub_pitch,   beats_per_bar, 70, last_offset),
            MotifNote(tonic_pitch, beats_per_bar, 75, last_offset + beats_per_bar),
        ]
    elif cadence_type == "modal":
        # bVII → i: cadencia modal
        bvii_pitch  = pitch_for_degree((root_pc + 10) % 12 - root_pc)
        tonic_pitch = pitch_for_degree(0)
        cadence_notes = [
            MotifNote(bvii_pitch,  beats_per_bar, 70, last_offset),
            MotifNote(tonic_pitch, beats_per_bar, 73, last_offset + beats_per_bar),
        ]

    return cadence_notes


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTORES DE FORMA
# ══════════════════════════════════════════════════════════════════════════════

def _offset_notes(notes: List[MotifNote], delta: float) -> List[MotifNote]:
    """Desplaza todos los offsets en delta beats."""
    return [MotifNote(n.pitch, n.duration, n.velocity, n.offset + delta)
            for n in notes]


def _fit_motif_to_bars(motif: List[MotifNote], target_bars: int,
                        beats_per_bar: int = 4) -> List[MotifNote]:
    """
    Ajusta el motivo para que quepa en target_bars compases.
    Si el motivo es más corto, lo repite / aumenta.
    Si es más largo, lo trunca.
    """
    target_beats = target_bars * beats_per_bar
    motif_dur = max(n.offset + n.duration for n in motif) if motif else beats_per_bar

    if motif_dur < 0.01:
        return motif

    factor = target_beats / motif_dur

    if abs(factor - 1.0) < 0.1:
        return deepcopy(motif)
    elif factor >= 1.5:
        # Aumentación
        return augment_motif(deepcopy(motif), factor)
    elif factor <= 0.6:
        # Diminución
        return diminish_motif(deepcopy(motif), factor)
    else:
        # Ajuste fino: pequeño stretch temporal
        result = []
        for n in motif:
            result.append(MotifNote(
                pitch=n.pitch,
                duration=n.duration * factor,
                velocity=n.velocity,
                offset=n.offset * factor,
            ))
        return result


def _vary_motif(motif: List[MotifNote], phrase_type: str,
                root_pc: int, mode: str, section_idx: int) -> List[MotifNote]:
    """
    Aplica la variación apropiada al motivo según el tipo de frase
    y qué sección se está construyendo.
    """
    if phrase_type == "parallel":
        # Conservar casi todo; solo ajustar final en la cadencia
        return deepcopy(motif)
    elif phrase_type == "contrasting":
        if section_idx == 0:
            return deepcopy(motif)
        else:
            # Invertir o transponer para crear contraste
            inverted = invert_motif(deepcopy(motif))
            return [snap_note_to_scale(n, root_pc, mode) for n in inverted]
    elif phrase_type == "sequential":
        steps = [0, 1, -1, 2, -2]
        step = steps[section_idx % len(steps)]
        if step == 0:
            return deepcopy(motif)
        return sequence_motif(deepcopy(motif), root_pc, mode, steps=step)
    elif phrase_type == "developmental":
        if section_idx == 0:
            return deepcopy(motif)
        elif section_idx == 1:
            return add_passing_notes(deepcopy(motif), root_pc, mode)
        else:
            liquid = liquidate_motif(deepcopy(motif), keep_intervals=2)
            return _fit_motif_to_bars(liquid, 2)
    elif phrase_type == "call_response":
        if section_idx % 2 == 0:  # "call"
            return deepcopy(motif)
        else:  # "response": transponer +5 (a la dominante)
            return transpose_motif(deepcopy(motif), 5)
    return deepcopy(motif)


def snap_note_to_scale(note: MotifNote, root_pc: int, mode: str) -> MotifNote:
    new_pitch = snap_to_scale(note.pitch, root_pc, mode)
    return MotifNote(new_pitch, note.duration, note.velocity, note.offset)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTORES POR FORMA
# ══════════════════════════════════════════════════════════════════════════════

def build_period(motif: List[MotifNote], root_pc: int, mode: str,
                 phrase_type: str, cadence_ant: str, cadence_cons: str,
                 bars: int, beats_per_bar: int = 4,
                 verbose: bool = False) -> List[PhraseSection]:
    """
    Construye un período: [Antecedente (bars/2)] + [Consecuente (bars/2)]
    """
    half = bars // 2
    beats_half = half * beats_per_bar

    ant_motif = _vary_motif(motif, phrase_type, root_pc, mode, section_idx=0)
    ant_motif = _fit_motif_to_bars(ant_motif, half - 1, beats_per_bar)  # -1 para cadencia
    ant_cad   = resolve_to_cadence(ant_motif, cadence_ant, root_pc, mode, beats_per_bar)

    cons_motif = _vary_motif(motif, phrase_type, root_pc, mode, section_idx=1)
    cons_motif = _fit_motif_to_bars(cons_motif, half - 1, beats_per_bar)
    cons_motif = _offset_notes(cons_motif, beats_half)
    cons_cad   = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - beats_half)
             for n in cons_motif],
            cadence_cons, root_pc, mode, beats_per_bar,
        ),
        beats_half,
    )

    if verbose:
        print(f"  [period] antecedente: {len(ant_motif)} notas, cadencia {cadence_ant}")
        print(f"  [period] consecuente: {len(cons_motif)} notas, cadencia {cadence_cons}")

    return [
        PhraseSection("antecedent", ant_motif, 0.0, half, cadence_ant, ant_cad),
        PhraseSection("consequent", cons_motif, beats_half, half, cadence_cons, cons_cad),
    ]


def build_double_period(motif: List[MotifNote], root_pc: int, mode: str,
                         phrase_type: str, cadence_ant: str, cadence_cons: str,
                         bars: int, beats_per_bar: int = 4,
                         verbose: bool = False) -> List[PhraseSection]:
    """
    Período doble: [A: ant + cons] [B: ant' + cons']
    """
    quarter = bars // 4
    beats_q  = quarter * beats_per_bar

    sections = []
    labels = ["ant_A", "cons_A", "ant_B", "cons_B"]
    cadences = [cadence_ant, "IAC", cadence_ant, cadence_cons]
    phrase_types = [0, 1, 2, 3]

    for i, (label, cad, pt_idx) in enumerate(zip(labels, cadences, phrase_types)):
        m = _vary_motif(motif, phrase_type, root_pc, mode, section_idx=pt_idx)
        m = _fit_motif_to_bars(m, quarter - 1, beats_per_bar)
        start = i * beats_q
        m = _offset_notes(m, start)
        cad_notes = _offset_notes(
            resolve_to_cadence(
                [MotifNote(n.pitch, n.duration, n.velocity, n.offset - start) for n in m],
                cad, root_pc, mode, beats_per_bar,
            ),
            start,
        )
        sections.append(PhraseSection(label, m, start, quarter, cad, cad_notes))

    if verbose:
        print(f"  [double_period] 4 secciones de {quarter} compases cada una")

    return sections


def build_sentence(motif: List[MotifNote], root_pc: int, mode: str,
                   cadence_cons: str, bars: int, beats_per_bar: int = 4,
                   verbose: bool = False) -> List[PhraseSection]:
    """
    Sentencia: [Básica: bars/4] [Repetición: bars/4] [Liquidación: bars/2]
    Estructura clásica de Beethoven.
    """
    quarter = bars // 4
    half    = bars // 2
    beats_q = quarter * beats_per_bar
    beats_h = half * beats_per_bar

    # Idea básica (motivo original)
    basic = _fit_motif_to_bars(deepcopy(motif), quarter, beats_per_bar)

    # Repetición (transpuesta o variada)
    repetition = _vary_motif(deepcopy(motif), "sequential", root_pc, mode, section_idx=1)
    repetition = _fit_motif_to_bars(repetition, quarter, beats_per_bar)
    repetition = _offset_notes(repetition, beats_q)

    # Liquidación (reducción al núcleo + cadencia)
    liquid = liquidate_motif(deepcopy(motif), keep_intervals=2)
    liquid = _fit_motif_to_bars(liquid, half - 1, beats_per_bar)
    liquid = _offset_notes(liquid, 2 * beats_q)
    liquid_cad = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - 2 * beats_q) for n in liquid],
            cadence_cons, root_pc, mode, beats_per_bar,
        ),
        2 * beats_q,
    )

    if verbose:
        print(f"  [sentence] básica: {quarter}c, repetición: {quarter}c, liquidación: {half}c")

    return [
        PhraseSection("basic_idea", basic, 0.0, quarter, "", []),
        PhraseSection("repetition", repetition, beats_q, quarter, "", []),
        PhraseSection("liquidation", liquid, 2 * beats_q, half, cadence_cons, liquid_cad),
    ]


def build_bar_form(motif: List[MotifNote], root_pc: int, mode: str,
                   cadence_ant: str, cadence_cons: str, bars: int,
                   beats_per_bar: int = 4, verbose: bool = False) -> List[PhraseSection]:
    """
    Forma de Bar (AAB): [Stollen A] [Stollen A'] [Abgesang B]
    Clásica del Meistersinger y coral luterano.
    """
    third = bars // 3
    two_third = bars - third
    beats_t = third * beats_per_bar
    beats_2t = two_third * beats_per_bar

    stollen_a = _fit_motif_to_bars(deepcopy(motif), third - 1, beats_per_bar)
    stollen_a_cad = resolve_to_cadence(stollen_a, cadence_ant, root_pc, mode, beats_per_bar)

    stollen_b = _vary_motif(deepcopy(motif), "parallel", root_pc, mode, 1)
    stollen_b = _fit_motif_to_bars(stollen_b, third - 1, beats_per_bar)
    stollen_b = _offset_notes(stollen_b, beats_t)
    stollen_b_cad = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - beats_t) for n in stollen_b],
            cadence_ant, root_pc, mode, beats_per_bar,
        ),
        beats_t,
    )

    abgesang = _vary_motif(deepcopy(motif), "contrasting", root_pc, mode, 2)
    abgesang = _fit_motif_to_bars(abgesang, two_third - 1, beats_per_bar)
    abgesang = _offset_notes(abgesang, 2 * beats_t)
    abgesang_cad = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - 2 * beats_t) for n in abgesang],
            cadence_cons, root_pc, mode, beats_per_bar,
        ),
        2 * beats_t,
    )

    if verbose:
        print(f"  [bar_form] AAB: Stollen ({third}c) Stollen ({third}c) Abgesang ({two_third}c)")

    return [
        PhraseSection("stollen_A",  stollen_a, 0.0, third, cadence_ant, stollen_a_cad),
        PhraseSection("stollen_A2", stollen_b, beats_t, third, cadence_ant, stollen_b_cad),
        PhraseSection("abgesang_B", abgesang, 2 * beats_t, two_third, cadence_cons, abgesang_cad),
    ]


def build_binary(motif: List[MotifNote], root_pc: int, mode: str,
                 bars: int, beats_per_bar: int = 4,
                 verbose: bool = False) -> List[PhraseSection]:
    """
    Forma binaria: [A: bars/2 → cadencia dominante] [B: bars/2 → cadencia tónica]
    """
    half = bars // 2
    beats_h = half * beats_per_bar

    section_a = _fit_motif_to_bars(deepcopy(motif), half - 1, beats_per_bar)
    cad_a = resolve_to_cadence(section_a, "HC", root_pc, mode, beats_per_bar)

    section_b = _vary_motif(deepcopy(motif), "contrasting", root_pc, mode, 1)
    section_b = _fit_motif_to_bars(section_b, half - 1, beats_per_bar)
    section_b = _offset_notes(section_b, beats_h)
    cad_b = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - beats_h) for n in section_b],
            "AC", root_pc, mode, beats_per_bar,
        ),
        beats_h,
    )

    if verbose:
        print(f"  [binary] A({half}c → dom) B({half}c → tónica)")

    return [
        PhraseSection("A", section_a, 0.0, half, "HC", cad_a),
        PhraseSection("B", section_b, beats_h, half, "AC", cad_b),
    ]


def build_ternary(motif: List[MotifNote], root_pc: int, mode: str,
                  bars: int, beats_per_bar: int = 4,
                  verbose: bool = False) -> List[PhraseSection]:
    """
    Forma ternaria: [A: bars/3] [B: bars/3] [A': bars/3]
    """
    third = bars // 3
    beats_t = third * beats_per_bar

    section_a = _fit_motif_to_bars(deepcopy(motif), third - 1, beats_per_bar)
    cad_a = resolve_to_cadence(section_a, "AC", root_pc, mode, beats_per_bar)

    section_b = _vary_motif(deepcopy(motif), "contrasting", root_pc, mode, 1)
    section_b = _fit_motif_to_bars(section_b, third - 1, beats_per_bar)
    section_b = _offset_notes(section_b, beats_t)
    cad_b = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - beats_t) for n in section_b],
            "DC", root_pc, mode, beats_per_bar,
        ),
        beats_t,
    )

    section_a2 = _vary_motif(deepcopy(motif), "parallel", root_pc, mode, 2)
    section_a2 = _fit_motif_to_bars(section_a2, third - 1, beats_per_bar)
    section_a2 = _offset_notes(section_a2, 2 * beats_t)
    cad_a2 = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - 2 * beats_t) for n in section_a2],
            "AC", root_pc, mode, beats_per_bar,
        ),
        2 * beats_t,
    )

    if verbose:
        print(f"  [ternary] A({third}c) B({third}c) A'({third}c)")

    return [
        PhraseSection("A",  section_a,  0.0, third, "AC", cad_a),
        PhraseSection("B",  section_b,  beats_t, third, "DC", cad_b),
        PhraseSection("A2", section_a2, 2 * beats_t, third, "AC", cad_a2),
    ]


def build_rondo_kernel(motif: List[MotifNote], root_pc: int, mode: str,
                       bars: int, beats_per_bar: int = 4,
                       verbose: bool = False) -> List[PhraseSection]:
    """
    Núcleo de rondó: [A: 8c] [B: 4c] [A: 4c]
    """
    a_bars = bars // 2
    b_bars = bars // 4
    a2_bars = bars - a_bars - b_bars
    beats_a = a_bars * beats_per_bar
    beats_b = b_bars * beats_per_bar

    section_a = _fit_motif_to_bars(deepcopy(motif), a_bars - 1, beats_per_bar)
    cad_a = resolve_to_cadence(section_a, "AC", root_pc, mode, beats_per_bar)

    section_b = _vary_motif(deepcopy(motif), "contrasting", root_pc, mode, 1)
    section_b = _fit_motif_to_bars(section_b, b_bars - 1, beats_per_bar)
    section_b = _offset_notes(section_b, beats_a)
    cad_b = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - beats_a) for n in section_b],
            "HC", root_pc, mode, beats_per_bar,
        ),
        beats_a,
    )

    section_a2 = _fit_motif_to_bars(deepcopy(motif), a2_bars - 1, beats_per_bar)
    section_a2 = _offset_notes(section_a2, beats_a + beats_b)
    cad_a2 = _offset_notes(
        resolve_to_cadence(
            [MotifNote(n.pitch, n.duration, n.velocity, n.offset - (beats_a + beats_b))
             for n in section_a2],
            "AC", root_pc, mode, beats_per_bar,
        ),
        beats_a + beats_b,
    )

    if verbose:
        print(f"  [rondo_kernel] A({a_bars}c) B({b_bars}c) A'({a2_bars}c)")

    return [
        PhraseSection("A",  section_a,  0.0, a_bars, "AC", cad_a),
        PhraseSection("B",  section_b,  beats_a, b_bars, "HC", cad_b),
        PhraseSection("A2", section_a2, beats_a + beats_b, a2_bars, "AC", cad_a2),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: build_phrase
# ══════════════════════════════════════════════════════════════════════════════

FORM_BUILDERS = {
    "period":        build_period,
    "double_period": build_double_period,
    "sentence":      build_sentence,
    "bar_form":      build_bar_form,
    "binary":        build_binary,
    "ternary":       build_ternary,
    "rondo_kernel":  build_rondo_kernel,
}

ALL_FORMS  = list(FORM_BUILDERS.keys())
ALL_TYPES  = ["parallel", "contrasting", "sequential", "developmental", "call_response"]
ALL_CADENCES = ["AC", "IAC", "HC", "DC", "PC", "modal"]


def build_phrase(
    motif: List[MotifNote],
    key: str = "C major",
    form: str = "period",
    phrase_type: str = "parallel",
    cadence_ant: str = "HC",
    cadence_cons: str = "AC",
    bars: int = 8,
    beats_per_bar: int = 4,
    tempo: int = 90,
    seed: int = 42,
    verbose: bool = False,
) -> PhraseResult:
    """
    Construye una frase musical completa desde un motivo base.

    Args:
        motif:         Lista de MotifNote (motivo de entrada)
        key:           Tonalidad ("C major", "Am", "Bb dorian"…)
        form:          Forma musical (period, sentence, binary…)
        phrase_type:   Relación A→C (parallel, contrasting, sequential…)
        cadence_ant:   Cadencia del antecedente (HC, IAC…)
        cadence_cons:  Cadencia del consecuente (AC, PC…)
        bars:          Compases totales
        beats_per_bar: Beats por compás
        tempo:         BPM
        seed:          Semilla aleatoria
        verbose:       Imprimir decisiones

    Returns:
        PhraseResult con todas las secciones y notas.
    """
    random.seed(seed)
    np.random.seed(seed)

    root_pc, mode = parse_key(key)

    if verbose:
        print(f"\n[PHRASE BUILDER] forma={form} tipo={phrase_type} "
              f"tonalidad={key} ({bars} compases, {tempo} BPM)")

    builder = FORM_BUILDERS.get(form, build_period)

    # Llamar al constructor correcto según la forma
    if form == "period":
        sections = builder(motif, root_pc, mode, phrase_type,
                           cadence_ant, cadence_cons, bars, beats_per_bar, verbose)
    elif form == "double_period":
        sections = builder(motif, root_pc, mode, phrase_type,
                           cadence_ant, cadence_cons, bars, beats_per_bar, verbose)
    elif form == "sentence":
        sections = builder(motif, root_pc, mode, cadence_cons, bars, beats_per_bar, verbose)
    elif form == "bar_form":
        sections = builder(motif, root_pc, mode, cadence_ant, cadence_cons,
                           bars, beats_per_bar, verbose)
    elif form in ("binary", "ternary", "rondo_kernel"):
        sections = builder(motif, root_pc, mode, bars, beats_per_bar, verbose)
    else:
        sections = build_period(motif, root_pc, mode, phrase_type,
                                cadence_ant, cadence_cons, bars, beats_per_bar, verbose)

    return PhraseResult(
        form=form, phrase_type=phrase_type,
        key=key, bars=bars, tempo=tempo,
        sections=sections, motif_used=deepcopy(motif),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def phrase_to_midi(
    result: PhraseResult,
    ticks_per_beat: int = 480,
    channel: int = 0,
    program: int = 0,
) -> mido.MidiFile:
    """Convierte un PhraseResult a un MidiFile."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = mido.bpm2tempo(result.tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.Message("program_change", channel=channel,
                               program=program, time=0))

    # Construir eventos
    events = []
    for note in result.all_notes:
        t_on  = int(note.offset * ticks_per_beat)
        t_off = int((note.offset + note.duration) * ticks_per_beat)
        vel   = max(1, min(127, note.velocity))
        p     = max(0, min(127, note.pitch))
        events.append((t_on,  "on",  p, vel))
        events.append((t_off, "off", p, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    current_tick = 0
    for abs_tick, etype, pitch, vel in events:
        delta = abs_tick - current_tick
        if etype == "on":
            track.append(mido.Message("note_on", channel=channel,
                                       note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", channel=channel,
                                       note=pitch, velocity=0, time=delta))
        current_tick = abs_tick

    return mid


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phrase_builder.py",
        description="Motor de sintaxis melódica: antecedente/consecuente desde motivos.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--motif", type=str, metavar="FILE",
                     help="MIDI de motivo de entrada (extraído por harvester.py)")
    src.add_argument("--motif-text", type=str, metavar="S",
                     help='Motivo en texto: "C4:1 E4:0.5 G4:0.5"')
    src.add_argument("--generate", action="store_true",
                     help="Generar motivo interno aleatorio")

    p.add_argument("--key", type=str, default="C major",
                   help='Tonalidad (default: "C major")')
    p.add_argument("--form", type=str, default="period",
                   choices=ALL_FORMS,
                   help="Forma musical (default: period)")
    p.add_argument("--type", dest="phrase_type", type=str, default="parallel",
                   choices=ALL_TYPES,
                   help="Tipo de relación A→C (default: parallel)")
    p.add_argument("--cadences", nargs=2, metavar=("ANT", "CONS"),
                   default=["HC", "AC"],
                   help="Cadencias antecedente y consecuente (default: HC AC)")
    p.add_argument("--bars", type=int, default=8,
                   help="Compases totales (default: 8)")
    p.add_argument("--beats", type=int, default=4,
                   help="Beats por compás (default: 4)")
    p.add_argument("--tempo", type=int, default=90,
                   help="Tempo BPM (default: 90)")
    p.add_argument("--all-forms", action="store_true",
                   help="Generar una versión por cada forma")
    p.add_argument("--all-types", action="store_true",
                   help="Generar una versión por cada tipo")
    p.add_argument("--report", action="store_true",
                   help="Exportar análisis JSON")
    p.add_argument("--output", type=str, default="phrase_out.mid",
                   help="Fichero de salida (default: phrase_out.mid)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directorio de salida con --all-forms/--all-types")
    p.add_argument("--annotate", action="store_true",
                   help="Exportar MusicXML con anotaciones (requiere music21)")
    p.add_argument("--listen", action="store_true",
                   help="Reproducir al terminar (requiere pygame)")
    p.add_argument("--verbose", action="store_true",
                   help="Imprimir decisiones de construcción")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria (default: 42)")
    return p


def _print_phrase_summary(result: PhraseResult) -> None:
    total_notes = len(result.all_notes)
    total_beats = result.bars * 4
    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║        PHRASE SUMMARY                    ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  Forma:          {result.form}")
    print(f"  Tipo:           {result.phrase_type}")
    print(f"  Tonalidad:      {result.key}")
    print(f"  Compases:       {result.bars}")
    print(f"  Beats totales:  {total_beats}")
    print(f"  Notas totales:  {total_notes}")
    print(f"  Secciones:")
    for s in result.sections:
        cad_str = s.cadence_type if s.cadence_type else "(ninguna)"
        print(f"    [{s.name:15s}] {s.duration_bars:2d} compases  "
              f"@ beat {s.start_beat:.1f}  cadencia: {cad_str}")
    print()


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Cargar motivo ─────────────────────────────────────────────────────────
    if args.motif:
        print(f"[INFO] Cargando motivo desde: {args.motif}")
        motif = load_motif_from_midi(args.motif)
    elif args.motif_text:
        print(f"[INFO] Parseando motivo desde texto: {args.motif_text}")
        motif = load_motif_from_text(args.motif_text)
    elif args.generate:
        print(f"[INFO] Generando motivo aleatorio en {args.key}")
        root_pc, mode = parse_key(args.key)
        motif = _generate_random_motif(args.key, mode, bars=2, seed=args.seed)
    else:
        parser.error("Especifica --motif, --motif-text o --generate")

    print(f"[INFO] Motivo cargado: {len(motif)} notas, "
          f"duración={sum(n.duration for n in motif):.1f} beats")

    # ── Determinar qué combinaciones generar ─────────────────────────────────
    forms  = ALL_FORMS  if args.all_forms  else [args.form]
    types  = ALL_TYPES  if args.all_types  else [args.phrase_type]
    cadence_ant, cadence_cons = args.cadences

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for form in forms:
        for ptype in types:
            label = ""
            if args.all_forms or args.all_types:
                label = f"_{form}_{ptype}"

            print(f"\n[PHRASE BUILDER] forma={form} tipo={ptype} "
                  f"tonalidad={args.key} · {args.bars} compases · {args.tempo} BPM")

            result = build_phrase(
                motif=motif,
                key=args.key,
                form=form,
                phrase_type=ptype,
                cadence_ant=cadence_ant,
                cadence_cons=cadence_cons,
                bars=args.bars,
                beats_per_bar=args.beats,
                tempo=args.tempo,
                seed=args.seed,
                verbose=args.verbose,
            )

            if args.verbose or (not args.all_forms and not args.all_types):
                _print_phrase_summary(result)

            # ── Exportar MIDI ─────────────────────────────────────────────────
            mid = phrase_to_midi(result)
            stem = Path(args.output).stem
            out_path = out_dir / f"{stem}{label}.mid"
            mid.save(str(out_path))
            print(f"[OK] Guardado: {out_path}  ({len(result.all_notes)} notas)")

            # ── Exportar JSON ─────────────────────────────────────────────────
            if args.report:
                report_path = out_path.with_suffix(".report.json")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_report(), f, indent=2, ensure_ascii=False)
                print(f"[OK] Informe guardado: {report_path}")

            results.append((result, out_path))

    # ── Reproducir ────────────────────────────────────────────────────────────
    if args.listen and results:
        last_path = results[-1][1]
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(last_path))
            pygame.mixer.music.play()
            print("[INFO] Reproduciendo... (Ctrl+C para detener)")
            while pygame.mixer.music.get_busy():
                pass
        except Exception as e:
            print(f"[AVISO] No se pudo reproducir: {e}")


if __name__ == "__main__":
    main()
