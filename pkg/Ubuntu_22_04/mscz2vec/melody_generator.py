#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MELODY GENERATOR  v1.0                                  ║
║         Generación de melodías originales desde cero                         ║
║                                                                              ║
║  Construye una melodía completa ex nihilo a partir de parámetros            ║
║  musicales o de lenguaje natural. Es el «Bloque 0» melódico del             ║
║  pipeline: el equivalente a chord_progression_generator pero para la        ║
║  voz principal. No necesita ningún MIDI de entrada.                          ║
║                                                                              ║
║  MOTORES DE GENERACIÓN (--engine):                                           ║
║    markov     — Cadena de Markov de 2º orden (intervalo × duración).        ║
║                 Aprende estadísticas del corpus MIDI si se proporciona      ║
║                 (--corpus), o usa tablas estilísticas internas.             ║
║    grammar    — Gramática generativa basada en reglas de Schenkerian:       ║
║                 elabora progresivamente una estructura de capa profunda      ║
║                 (arpegio tónica) añadiendo notas de paso y ornamentos.      ║
║    search     — Búsqueda heurística A*: explora el espacio de alturas       ║
║                 guiada por una función de coste (consonancia, contorno,     ║
║                 tensión objetivo). Garantiza propiedades locales y globales. ║
║    genetic    — Algoritmo genético: población de frases evaluadas por       ║
║                 fitness multicriterio y evolucionadas por cruce/mutación.   ║
║    hybrid     — Combina grammar (estructura) + markov (ornamentación).      ║
║                 Produce los resultados más «musicales» por defecto.         ║
║                                                                              ║
║  PERFILES EMOCIONALES (--profile):                                           ║
║    heroic      — ascenso agresivo, saltos de 5ª/8ª, forte, major           ║
║    melancholic — descenso lento, semítonos, piano, minor/dorian             ║
║    playful     — ritmo irregular, saltos cortos, staccato implícito         ║
║    tense       — cromatismo, ritmo denso, disonancia controlada             ║
║    serene      — movimiento por grados, dinámicas suaves, arcos largos      ║
║    mysterious  — modo frigio/locrio, silencios, saltos inesperados          ║
║    triumphant  — ritmo marcado, forte, major, clímax en el último tercio    ║
║    custom      — usa directamente --tension-curve / --contour               ║
║                                                                              ║
║  PERFILES RÍTMICOS (--rhythm):                                               ║
║    flowing     — mezcla de negras y corcheas, legato                        ║
║    march       — pulso regular, negras dominantes, acentos en 1 y 3        ║
║    syncopated  — desplazamientos de acento, corcheas con puntillo           ║
║    baroque     — valores rítmicos variados, ornamentos, sin silencios       ║
║    jazz        — swing implícito, tresillos, síncopas                       ║
║    sparse      — notas largas (blancas/redondas), mucho espacio             ║
║    dense       — semicorcheas y fusas, texturas rápidas                     ║
║                                                                              ║
║  CONTROL DEL CONTORNO (--contour):                                           ║
║    arch        — sube al centro, baja al final (el más «natural»)           ║
║    ascending   — tendencia ascendente continua                               ║
║    descending  — tendencia descendente continua                              ║
║    wave        — ondulación suave, sube y baja periódicamente               ║
║    plateau     — plano en el centro, puntas bajas                           ║
║    inverted    — baja al centro, sube al final                              ║
║    erratic     — sin tendencia clara (útil para música experimental)        ║
║    custom      — lista de valores 0-1 separados por comas                   ║
║                                                                              ║
║  MODOS DE ESCALA (--mode):                                                   ║
║    major, minor, harmonic_minor, melodic_minor,                             ║
║    dorian, phrygian, lydian, mixolydian, locrian,                           ║
║    phrygian_dominant, whole_tone, pentatonic_major, pentatonic_minor,       ║
║    blues, chromatic                                                          ║
║                                                                              ║
║  INTEGRACIONES (--from-*):                                                   ║
║    theorist.py    → --from-theorist lee .theorist.json                      ║
║    narrator.py    → --from-narrator lee obra_plan.json                      ║
║    tension_designer → --curves lee .curves.json                             ║
║    chord_progression_generator → --chords lee .chords.json / texto         ║
║    harvester.py   → --corpus carpeta con MIDIs para entrenamiento Markov    ║
║                                                                              ║
║  EXPORTACIÓN:                                                                ║
║    .mid                   — MIDI de la melodía generada                     ║
║    .melody.json           — Partitura en JSON (notas, duraciones, offsets)  ║
║    .fingerprint.json      — Huella compatible con stitcher.py               ║
║    .motif.mid             — Solo el motivo semilla (primeros 2-4 compases)  ║
║                                                                              ║
║  USO:                                                                        ║
║    # Desde lenguaje natural                                                  ║
║    python melody_generator.py "una melodía melancólica en La menor"         ║
║                                                                              ║
║    # Con parámetros directos                                                 ║
║    python melody_generator.py --key Am --mode minor --bars 16               ║
║        --engine hybrid --profile melancholic --contour arch                 ║
║                                                                              ║
║    # Desde plan de theorist                                                  ║
║    python melody_generator.py --from-theorist obra.theorist.json            ║
║                                                                              ║
║    # Desde plan de narrator (genera melodía para cada sección)               ║
║    python melody_generator.py --from-narrator obra_plan.json --per-section  ║
║                                                                              ║
║    # Con curvas de tensión explícitas                                        ║
║    python melody_generator.py --key C --curves mis_curvas.json --bars 32    ║
║                                                                              ║
║    # Aprendiendo de corpus propio                                            ║
║    python melody_generator.py --key Dm --corpus ./mis_midis/ --engine markov║
║                                                                              ║
║    # Generar varios candidatos y exportar el mejor                          ║
║    python melody_generator.py --key G --candidates 5 --bars 16              ║
║                                                                              ║
║    # Con acordes como guía armónica                                          ║
║    python melody_generator.py --key C --chords "C Am F G" --bars 8         ║
║                                                                              ║
║    # Generar motivo semilla solo (para usar en phrase_builder / harvester)  ║
║    python melody_generator.py --key Dm --bars 4 --motif-only               ║
║                                                                              ║
║    # Modo interactivo: genera, escucha, acepta o regenera                   ║
║    python melody_generator.py --key C --interactive                         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    description        Descripción libre (entre comillas)                    ║
║    --key KEY           Tonalidad base: C, Am, F#, Bb… (default: C)         ║
║    --mode MODE         Modo de escala (default: major o minor según clave)  ║
║    --bars N            Compases a generar (default: 16)                     ║
║    --tempo BPM         Tempo en BPM (default: 120)                          ║
║    --time-sig S        Compás: 4/4, 3/4, 6/8… (default: 4/4)              ║
║    --engine E          Motor generativo (default: hybrid)                   ║
║    --profile P         Perfil emocional (default: serene)                   ║
║    --rhythm R          Perfil rítmico (default: flowing)                    ║
║    --contour C         Forma del contorno (default: arch)                   ║
║    --range LOW HIGH    Rango MIDI de la melodía (default: 60 84, C4-C6)    ║
║    --candidates N      Generar N candidatos y exportar el mejor (default: 3)║
║    --chords PROG       Acordes guía en texto: "C Am F G" o "Cm:2 G7:2"    ║
║    --chords-json FILE  Acordes guía desde .chords.json                      ║
║    --curves FILE       Curvas emocionales .curves.json                      ║
║    --from-theorist F   Leer parámetros de .theorist.json                   ║
║    --from-narrator F   Leer plan de obra_plan.json                          ║
║    --per-section       Con --from-narrator, genera una melodía por sección  ║
║    --corpus DIR        Carpeta con MIDIs para entrenar Markov               ║
║    --motif-only        Exportar solo el motivo semilla (2-4 compases)       ║
║    --export-fingerprint Exportar .fingerprint.json para stitcher.py        ║
║    --output FILE       Fichero de salida (default: melody_out.mid)          ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║    --interactive       Modo interactivo: escucha, acepta o regenera         ║
║    --listen            Reproducir al final (requiere pygame)                ║
║    --verbose           Informe detallado de decisiones                      ║
║    --dry-run           Mostrar parámetros sin generar MIDI                  ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:   mido, numpy                                                    ║
║    Opcionales: music21, pygame, scipy, sklearn                               ║
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
import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator
from collections import defaultdict, Counter

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[WARN] mido no disponible. Instala con: pip install mido")

# ── Integración con el ecosistema ─────────────────────────────────────────────
_DNA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DNA_DIR)
try:
    from midi_dna_unified import (
        _snap_to_scale, _get_scale_pcs, _get_scale_midi,
        _quarter_to_ticks, _clamp_pitch, score_candidate,
        MAJOR_SCALE_DEGREES, MINOR_SCALE_DEGREES, INSTRUMENT_RANGES,
    )
    DNA_OK = True
except ImportError:
    DNA_OK = False

try:
    from music21 import pitch, key as m21key, note as m21note, stream
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}

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
    "whole_tone":        [0, 2, 4, 6, 8, 10],
    "pentatonic_major":  [0, 2, 4, 7, 9],
    "pentatonic_minor":  [0, 3, 5, 7, 10],
    "blues":             [0, 3, 5, 6, 7, 10],
    "chromatic":         list(range(12)),
}

# Duraciones válidas en quarter notes (negra = 1.0)
DURATION_NAMES = {
    0.25: "semicorchea", 0.5: "corchea", 0.75: "corchea con puntillo",
    1.0: "negra", 1.5: "negra con puntillo", 2.0: "blanca",
    3.0: "blanca con puntillo", 4.0: "redonda",
}

# Tablas de intervalos probables por perfil emocional
# (intervalo_semitones → peso relativo)
INTERVAL_WEIGHTS: Dict[str, Dict[int, float]] = {
    "heroic":      {0: 0.5, 1: 0.5, 2: 2.0, 3: 1.5, 4: 2.0, 5: 2.5, 7: 3.0, 12: 2.5, -2: 2.0, -3: 1.0, -5: 1.5, -7: 1.5},
    "melancholic": {0: 0.3, 1: 2.5, 2: 3.0, 3: 2.0, -1: 3.0, -2: 3.5, -3: 2.0, -5: 1.5, 5: 1.0, 7: 0.5},
    "playful":     {0: 0.5, 1: 1.5, 2: 2.5, 3: 2.0, 4: 2.0, 5: 1.5, -1: 1.5, -2: 2.5, -3: 2.0, -4: 1.5, 7: 1.0, -7: 1.0},
    "tense":       {0: 0.5, 1: 3.0, 2: 2.0, 6: 2.0, -1: 3.0, -2: 2.0, -6: 2.0, 11: 1.5, -11: 1.5, 3: 1.0, -3: 1.0},
    "serene":      {0: 1.0, 2: 3.0, 3: 2.0, 4: 2.5, 5: 2.0, -2: 3.0, -3: 2.0, -4: 2.0, -5: 1.5, 7: 1.0, -7: 1.0},
    "mysterious":  {0: 1.0, 1: 2.0, 6: 2.5, -1: 2.0, -6: 2.5, 3: 1.5, -3: 1.5, 8: 1.5, -8: 1.5, 2: 1.0},
    "triumphant":  {0: 0.3, 2: 2.0, 4: 2.5, 5: 2.0, 7: 3.5, 12: 2.0, -2: 1.5, -3: 1.5, -5: 1.5, 3: 2.0},
}

# Distribuciones rítmicas por perfil
RHYTHM_DISTS: Dict[str, Dict[float, float]] = {
    "flowing":    {0.5: 3.0, 1.0: 4.0, 1.5: 1.5, 2.0: 1.5},
    "march":      {1.0: 5.0, 0.5: 2.0, 2.0: 1.5, 0.25: 1.0},
    "syncopated": {0.5: 3.5, 0.75: 3.0, 1.0: 2.5, 1.5: 2.0, 0.25: 2.0},
    "baroque":    {0.25: 2.0, 0.5: 3.0, 1.0: 3.0, 1.5: 1.5, 2.0: 1.0},
    "jazz":       {0.5: 3.0, 0.75: 3.5, 1.0: 2.5, 1.5: 2.0, 0.25: 1.5},
    "sparse":     {2.0: 3.5, 1.5: 2.0, 3.0: 2.0, 4.0: 1.5, 1.0: 2.0},
    "dense":      {0.25: 4.0, 0.5: 3.5, 0.75: 2.0, 1.0: 1.5},
}

# Perfil emocional → modo sugerido
PROFILE_TO_MODE: Dict[str, str] = {
    "heroic":      "major",
    "melancholic": "minor",
    "playful":     "major",
    "tense":       "phrygian",
    "serene":      "major",
    "mysterious":  "phrygian_dominant",
    "triumphant":  "major",
    "custom":      "major",
}

# Perfil emocional → rango de velocidades MIDI (min, max, typical_climax)
PROFILE_VELOCITY: Dict[str, Tuple[int, int, int]] = {
    "heroic":      (70, 110, 105),
    "melancholic": (40, 75,  65),
    "playful":     (60, 95,  85),
    "tense":       (55, 100, 98),
    "serene":      (40, 75,  68),
    "mysterious":  (35, 70,  60),
    "triumphant":  (75, 115, 112),
    "custom":      (50, 90,  80),
}

# Descripción libre → perfil emocional
DESCRIPTION_KEYWORDS: Dict[str, List[str]] = {
    "heroic":      ["héroe", "heroic", "épico", "epic", "valiente", "batalla", "triunfo"],
    "melancholic": ["melancol", "triste", "sad", "nostalg", "melanch", "dolor", "llanto", "lamento"],
    "playful":     ["alegre", "juguetón", "playful", "divertido", "vivaz", "jovial"],
    "tense":       ["tenso", "tense", "angustia", "ansie", "miedo", "terror", "oscuro", "dark"],
    "serene":      ["sereno", "serene", "tranquilo", "calm", "paz", "suave", "gentle"],
    "mysterious":  ["misterio", "mystery", "enigm", "sombrío", "ambiguo", "extraño"],
    "triumphant":  ["triunfal", "triumphant", "victoria", "glorioso", "poderoso"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MelodyNote:
    """Una nota en la melodía generada."""
    pitch:    int    # MIDI pitch (0-127)
    duration: float  # en quarter notes
    velocity: int    # 0-127
    offset:   float  # posición en quarter notes desde inicio

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12

    @property
    def octave(self) -> int:
        return self.pitch // 12 - 1

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.pitch_class]}{self.octave}"

    def transpose(self, semitones: int) -> "MelodyNote":
        return MelodyNote(
            pitch=max(0, min(127, self.pitch + semitones)),
            duration=self.duration,
            velocity=self.velocity,
            offset=self.offset,
        )


@dataclass
class MelodyResult:
    """Resultado completo de una melodía generada."""
    notes:   List[MelodyNote]
    key:     str
    mode:    str
    bars:    int
    tempo:   int
    engine:  str
    profile: str
    contour: str
    score:   float = 0.0
    motif:   List[MelodyNote] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key":     self.key,
            "mode":    self.mode,
            "bars":    self.bars,
            "tempo":   self.tempo,
            "engine":  self.engine,
            "profile": self.profile,
            "contour": self.contour,
            "score":   round(self.score, 4),
            "total_notes": len(self.notes),
            "notes": [
                {"pitch": n.pitch, "name": n.name,
                 "duration": n.duration, "velocity": n.velocity,
                 "offset": round(n.offset, 4)}
                for n in self.notes
            ],
            "motif": [
                {"pitch": n.pitch, "name": n.name,
                 "duration": n.duration, "velocity": n.velocity,
                 "offset": round(n.offset, 4)}
                for n in self.motif
            ],
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

    mode = "major"
    for m in ["harmonic_minor", "melodic_minor", "phrygian_dominant",
              "dorian", "phrygian", "lydian", "mixolydian", "locrian",
              "harmonic", "melodic", "whole_tone",
              "pentatonic_major", "pentatonic_minor", "blues", "chromatic"]:
        if m.replace("_", " ") in key_str.lower() or m in key_str.lower():
            mode = m
            break
    else:
        if "minor" in key_str.lower():
            mode = "minor"
        elif "major" in key_str.lower():
            mode = "major"
        elif len(parts) == 1:
            root_part = parts[0]
            if (len(root_part) >= 2 and root_part[-1] == "m"
                    and root_part[-2] not in ("#", "b")):
                mode = "minor"

    # Extraer root
    root_str = parts[0]
    root_str = re.sub(r'm(ajor|inor)?$', '', root_str, flags=re.IGNORECASE)
    root_str = root_str.strip()

    if root_str in ENHARMONIC:
        root_pc = ENHARMONIC[root_str]
    else:
        clean = root_str.replace("b", "").replace("#", "")
        base_pc = NOTE_NAMES.index(clean.upper()) if clean.upper() in NOTE_NAMES else 0
        sharps = root_str.count("#")
        flats  = root_str.count("b")
        root_pc = (base_pc + sharps - flats) % 12

    return root_pc, mode


def get_scale_pitches(root_pc: int, mode: str,
                      low: int = 48, high: int = 96) -> List[int]:
    """Devuelve todos los MIDI pitches de la escala en el rango dado."""
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    result = []
    for octave in range(11):
        for iv in intervals:
            p = octave * 12 + root_pc + iv
            if low <= p <= high:
                result.append(p)
    return sorted(result)


def snap_to_scale(pitch: int, scale_pitches: List[int]) -> int:
    """Proyecta un pitch al más cercano en la escala."""
    if not scale_pitches:
        return pitch
    return min(scale_pitches, key=lambda p: abs(p - pitch))


def build_contour_curve(shape: str, n_bars: int,
                        custom_vals: Optional[List[float]] = None) -> List[float]:
    """
    Genera una curva de contorno normalizada [0, 1] de longitud n_bars.
    0 = registro bajo, 1 = registro alto.
    """
    t = np.linspace(0, 1, n_bars)
    if shape == "arch":
        return list(np.sin(t * math.pi))
    elif shape == "ascending":
        return list(t)
    elif shape == "descending":
        return list(1.0 - t)
    elif shape == "wave":
        return list(0.5 + 0.5 * np.sin(t * 2 * math.pi))
    elif shape == "plateau":
        curve = np.zeros(n_bars)
        s = max(1, n_bars // 5)
        e = n_bars - s
        curve[s:e] = 0.8
        curve[:s] = np.linspace(0, 0.8, s)
        curve[e:] = np.linspace(0.8, 0, n_bars - e)
        return list(curve)
    elif shape == "inverted":
        arch = np.sin(t * math.pi)
        return list(1.0 - arch)
    elif shape == "erratic":
        rng = np.random.default_rng(42)
        return list(rng.random(n_bars))
    elif shape == "custom" and custom_vals:
        if len(custom_vals) == n_bars:
            return custom_vals
        return list(np.interp(np.linspace(0, 1, n_bars),
                               np.linspace(0, 1, len(custom_vals)),
                               custom_vals))
    return [0.5] * n_bars  # fallback neutral


def build_tension_curve(profile: str, n_bars: int) -> List[float]:
    """Curva de tensión sugerida para el perfil emocional."""
    t = np.linspace(0, 1, n_bars)
    if profile == "heroic":
        return list(0.3 + 0.6 * np.sin(t * math.pi))
    elif profile == "melancholic":
        return list(0.5 - 0.3 * t + 0.1 * np.sin(t * 3 * math.pi))
    elif profile == "tense":
        return list(np.clip(0.4 + 0.5 * t + 0.1 * np.sin(t * 6 * math.pi), 0, 1))
    elif profile == "playful":
        return list(0.4 + 0.2 * np.sin(t * 4 * math.pi))
    elif profile == "serene":
        return list(0.2 + 0.15 * np.sin(t * 2 * math.pi))
    elif profile == "mysterious":
        return list(0.3 + 0.4 * np.sin(t * math.pi) + 0.1 * np.sin(t * 7 * math.pi))
    elif profile == "triumphant":
        return list(np.clip(0.5 * t + 0.5 * np.sin(t * math.pi), 0, 1))
    return [0.5] * n_bars


def weighted_choice(options: List[Any], weights: List[float],
                    rng: random.Random) -> Any:
    """Selección aleatoria ponderada."""
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += w
        if r <= acc:
            return opt
    return options[-1]


def sample_duration(rhythm_profile: str, beats_remaining: float,
                    rng: random.Random) -> float:
    """Elige una duración apropiada dadas las restricciones del compás."""
    dist = RHYTHM_DISTS.get(rhythm_profile, RHYTHM_DISTS["flowing"])
    valid = {d: w for d, w in dist.items() if d <= beats_remaining and d > 0}
    if not valid:
        return min(beats_remaining, 1.0)
    return weighted_choice(list(valid.keys()), list(valid.values()), rng)


def detect_profile_from_description(desc: str) -> str:
    """Detecta el perfil emocional desde texto libre."""
    desc_lower = desc.lower()
    scores = {p: 0 for p in DESCRIPTION_KEYWORDS}
    for profile, keywords in DESCRIPTION_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                scores[profile] += 1
    best = max(scores, key=lambda p: scores[p])
    return best if scores[best] > 0 else "serene"


def detect_mode_from_description(desc: str) -> str:
    """Detecta el modo musical desde texto libre."""
    desc_lower = desc.lower()
    mode_keywords = {
        "minor":          ["menor", "minor", "triste", "sad", "oscuro"],
        "major":          ["mayor", "major", "alegre", "happy", "brillante"],
        "dorian":         ["dorian", "dórico", "dorico", "folk"],
        "phrygian":       ["frigio", "phrygian", "flamenco", "español"],
        "lydian":         ["lidio", "lydian", "onírico", "mágico"],
        "mixolydian":     ["mixolidio", "mixolydian", "rock", "modal"],
        "pentatonic_minor": ["pentatónic", "pentatonic", "oriental", "chino"],
        "blues":          ["blues", "blue"],
        "whole_tone":     ["impresion", "débussy", "debussy", "entero"],
    }
    for mode, kws in mode_keywords.items():
        for kw in kws:
            if kw in desc_lower:
                return mode
    return None


def detect_key_from_description(desc: str) -> Optional[str]:
    """Detecta la tonalidad desde texto libre."""
    pattern = r'\b([A-G][b#]?)\s*(m(?:ajor|inor)?|mayor|menor|m\b)?\b'
    match = re.search(pattern, desc)
    if match:
        root = match.group(1)
        qual = match.group(2) or ""
        if qual.lower() in ("m", "minor", "menor"):
            return f"{root}m"
        return root
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 1: MARKOV
# ══════════════════════════════════════════════════════════════════════════════

class MarkovEngine:
    """
    Cadena de Markov de 2º orden sobre (intervalo, duración_cuantizada).
    Puede aprenderse de un corpus externo o usar tablas internas por estilo.
    """

    # Tablas internas de transición por perfil: (interval, dur_bucket) → lista de (next_interval, next_dur_bucket)
    # dur_bucket: 0=corta(≤0.5), 1=media(0.5-1.5), 2=larga(≥2.0)
    _INTERNAL_TABLES = None  # Se construyen bajo demanda

    def __init__(self, profile: str = "serene", rhythm: str = "flowing",
                 rng: random.Random = None):
        self.profile = profile
        self.rhythm  = rhythm
        self.rng     = rng or random.Random(42)
        self._transitions: Dict[Tuple, List[Tuple]] = defaultdict(list)
        self._trained = False

    def _dur_bucket(self, dur: float) -> int:
        if dur <= 0.5:
            return 0
        elif dur <= 1.5:
            return 1
        return 2

    def train_from_midi(self, midi_path: str) -> int:
        """
        Aprende transiciones desde un MIDI. Devuelve el número de
        transiciones extraídas.
        """
        if not MIDO_OK:
            return 0
        try:
            mid = mido.MidiFile(midi_path)
        except Exception:
            return 0

        notes = []
        ticks_per_beat = mid.ticks_per_beat or 480

        for track in mid.tracks:
            current_time = 0
            pending: Dict[int, Tuple[int, int]] = {}  # pitch → (on_time, vel)
            for msg in track:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    pending[msg.note] = (current_time, msg.velocity)
                elif msg.type in ("note_off",) or (
                        msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in pending:
                        on_t, vel = pending.pop(msg.note)
                        dur_ticks = current_time - on_t
                        dur_q = dur_ticks / ticks_per_beat
                        notes.append((on_t / ticks_per_beat, msg.note,
                                      dur_q, vel))

        notes.sort(key=lambda n: n[0])

        count = 0
        for i in range(len(notes) - 2):
            iv1 = notes[i+1][1] - notes[i][1]
            iv2 = notes[i+2][1] - notes[i+1][1]
            db1 = self._dur_bucket(notes[i][2])
            db2 = self._dur_bucket(notes[i+1][2])
            db3 = self._dur_bucket(notes[i+2][2])
            self._transitions[(iv1, db1, db2)].append((iv2, db3))
            count += 1

        self._trained = (count > 0)
        return count

    def train_from_corpus(self, corpus_dir: str, verbose: bool = False):
        """Entrena desde todos los MIDIs de un directorio."""
        total = 0
        path = Path(corpus_dir)
        files = list(path.rglob("*.mid")) + list(path.rglob("*.midi"))
        for f in files:
            n = self.train_from_midi(str(f))
            if verbose and n > 0:
                print(f"  [corpus] {f.name}: {n} transiciones")
            total += n
        if verbose:
            print(f"  [corpus] Total: {total} transiciones de {len(files)} archivos")
        self._trained = (total > 0)

    def _build_internal_tables(self):
        """Construye tablas de Markov sintéticas para cada perfil."""
        # Para cada perfil usamos los INTERVAL_WEIGHTS como distribución
        # de siguiente intervalo y RHYTHM_DISTS para la duración
        iw = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        ivs = list(iw.keys())
        iws = [iw[k] for k in ivs]

        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])

        # Crear tabla sintética de 200 transiciones
        prev_iv = 0
        prev_db = 1
        for _ in range(200):
            iv = weighted_choice(ivs, iws, self.rng)
            dur = weighted_choice(list(rd.keys()), list(rd.values()), self.rng)
            db = self._dur_bucket(dur)
            next_iv = weighted_choice(ivs, iws, self.rng)
            next_db = self._dur_bucket(
                weighted_choice(list(rd.keys()), list(rd.values()), self.rng))
            self._transitions[(iv, prev_db, db)].append((next_iv, next_db))
            prev_iv, prev_db = iv, db

        self._trained = True

    def next_step(self, prev_iv: int, prev_db: int,
                  cur_db: int) -> Tuple[int, int]:
        """
        Dado el contexto actual, sugiere (siguiente_intervalo, siguiente_dur_bucket).
        """
        if not self._trained:
            self._build_internal_tables()

        key = (prev_iv, prev_db, cur_db)
        candidates = self._transitions.get(key)
        if not candidates:
            # Backoff a orden 1
            candidates = []
            for k, v in self._transitions.items():
                if k[2] == cur_db:
                    candidates.extend(v)
        if not candidates:
            # Fallback completo: usar pesos del perfil
            iw = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
            ivs = list(iw.keys())
            iws = [iw[k] for k in ivs]
            iv = weighted_choice(ivs, iws, self.rng)
            return iv, self.rng.choice([0, 1, 2])

        return self.rng.choice(candidates)

    def generate(self, scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float], vel_range: Tuple[int, int, int],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        """Genera una melodía completa por Markov."""
        if not self._trained:
            self._build_internal_tables()

        notes = []
        low, high = pitch_range
        vel_min, vel_max, vel_climax = vel_range

        # Punto de partida: tónica en octava media
        tonic = scale_pitches[len(scale_pitches) // 4]
        current_pitch = snap_to_scale(tonic, scale_pitches)

        prev_iv  = 0
        prev_db  = 1
        offset   = 0.0
        total_beats = n_bars * beats_per_bar

        bar_idx = 0

        while offset < total_beats - 0.01:
            beats_in_bar = beats_per_bar
            bar_offset   = 0.0

            while bar_offset < beats_in_bar - 0.01:
                # Tensión y contorno del compás actual
                t_val = tension[min(bar_idx, len(tension) - 1)]
                c_val = contour[min(bar_idx, len(contour) - 1)]

                # Duración
                beats_left = min(beats_in_bar - bar_offset,
                                 total_beats - offset - bar_offset)
                dur = sample_duration(self.rhythm, beats_left, self.rng)
                dur_db = self._dur_bucket(dur)

                # Siguiente intervalo desde Markov
                iv, _ = self.next_step(prev_iv, prev_db, dur_db)

                # Ajuste de dirección según contorno
                # c_val alto → preferir movimiento ascendente
                current_rel = (current_pitch - low) / max(high - low, 1)
                if current_rel > c_val + 0.15 and iv > 0:
                    iv = -abs(iv)
                elif current_rel < c_val - 0.15 and iv < 0:
                    iv = abs(iv)

                # Limitar salto máximo por tensión (más tensión = saltos más grandes permitidos)
                max_leap = int(3 + t_val * 9)
                if abs(iv) > max_leap:
                    iv = int(math.copysign(max_leap, iv))

                # Aplicar intervalo y proyectar a escala
                candidate = current_pitch + iv
                candidate = max(low, min(high, candidate))
                new_pitch = snap_to_scale(candidate, scale_pitches)

                # Velocidad dinámica según posición y tensión
                climax_factor = math.sin(
                    (offset / total_beats) * math.pi) * t_val
                vel = int(vel_min + (vel_max - vel_min) * climax_factor)
                vel = max(vel_min, min(vel_max, vel))

                notes.append(MelodyNote(
                    pitch=new_pitch,
                    duration=dur,
                    velocity=vel,
                    offset=round(offset + bar_offset, 4),
                ))

                prev_iv = new_pitch - current_pitch
                prev_db = dur_db
                current_pitch = new_pitch
                bar_offset += dur

            offset += beats_per_bar
            bar_idx += 1

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 2: GRAMMAR (Schenkerian)
# ══════════════════════════════════════════════════════════════════════════════

class GrammarEngine:
    """
    Generación por gramática generativa inspirada en Schenker.
    Parte de una estructura armónica de capa profunda y la elabora
    progresivamente: primero el arpegio de la tónica, luego notas
    de paso, luego ornamentos.

    Niveles de elaboración:
      [1] Capa profunda: notas de estructura (tónica, dominante, tónica)
      [2] Capa media: notas de arpegio y notas de paso principales
      [3] Capa superficial: ornamentos (bordaduras, appogiaturas)
    """

    def __init__(self, root_pc: int, mode: str, rng: random.Random = None):
        self.root_pc = root_pc
        self.mode    = mode
        self.rng     = rng or random.Random(42)
        self.scale   = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])

    def _scale_degree(self, degree: int, octave: int = 5) -> int:
        """Grado de escala (1-7) → MIDI pitch."""
        idx = (degree - 1) % len(self.scale)
        oct_adjust = (degree - 1) // len(self.scale)
        return (octave - 1) * 12 + self.root_pc + self.scale[idx] + oct_adjust * 12

    def _passing_notes(self, p1: int, p2: int,
                       scale_pitches: List[int]) -> List[int]:
        """Notas de paso escalonadas entre dos pitches."""
        direction = 1 if p2 > p1 else -1
        result = []
        for sp in scale_pitches:
            if direction == 1 and p1 < sp < p2:
                result.append(sp)
            elif direction == -1 and p2 < sp < p1:
                result.append(sp)
        return sorted(result, key=lambda x: direction * x)

    def _neighbor_note(self, pitch: int, scale_pitches: List[int],
                       upper: bool = True) -> int:
        """Bordadura superior o inferior."""
        idx = scale_pitches.index(pitch) if pitch in scale_pitches else -1
        if idx < 0:
            return pitch
        if upper and idx + 1 < len(scale_pitches):
            return scale_pitches[idx + 1]
        elif not upper and idx - 1 >= 0:
            return scale_pitches[idx - 1]
        return pitch

    def generate(self, scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float], vel_range: Tuple[int, int, int],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        """Genera una melodía por elaboración gramatical."""
        low, high = pitch_range
        vel_min, vel_max, _ = vel_range
        total_beats = n_bars * beats_per_bar

        # ── Capa 1: estructura de frase (cada 4 compases) ──────────────────
        # Divide la obra en frases de 4 compases con estructura T-D-T
        structural_pitches = []
        phrase_len = 4  # compases
        n_phrases = max(1, n_bars // phrase_len)
        remainder = n_bars % phrase_len

        for ph in range(n_phrases):
            c_mid = contour[min(ph * phrase_len + phrase_len // 2,
                                len(contour) - 1)]
            target_abs = low + int(c_mid * (high - low))
            target = snap_to_scale(target_abs, scale_pitches)
            structural_pitches.extend([target] * phrase_len)

        if remainder:
            structural_pitches.extend([structural_pitches[-1]] * remainder)

        # ── Capa 2: melodía barra a barra ──────────────────────────────────
        notes = []
        offset = 0.0

        for bar_idx in range(n_bars):
            struct_pitch = structural_pitches[bar_idx]
            t_val = tension[min(bar_idx, len(tension) - 1)]
            beats_remaining = beats_per_bar

            # Decide complejidad del compás por tensión
            if t_val < 0.3:
                # Simple: nota larga en el primer tiempo
                vel = int(vel_min + (vel_max - vel_min) * 0.3)
                notes.append(MelodyNote(struct_pitch, beats_per_bar, vel,
                                         round(offset, 4)))
                beats_remaining = 0.0

            elif t_val < 0.6:
                # Medio: nota de inicio + nota de paso + nota de destino
                dest = structural_pitches[min(bar_idx + 1, n_bars - 1)]
                passing = self._passing_notes(struct_pitch, dest, scale_pitches)

                if passing:
                    dur1 = beats_per_bar * 0.5
                    dur2 = beats_per_bar * 0.25
                    dur3 = beats_per_bar * 0.25
                    vel1 = int(vel_min + (vel_max - vel_min) * 0.5)
                    vel2 = vel1 - 10
                    vel3 = vel1 + 5
                    notes.append(MelodyNote(struct_pitch, dur1, vel1,
                                             round(offset, 4)))
                    notes.append(MelodyNote(passing[0], dur2, vel2,
                                             round(offset + dur1, 4)))
                    notes.append(MelodyNote(dest, dur3, vel3,
                                             round(offset + dur1 + dur2, 4)))
                else:
                    # Sin notas de paso: dos notas
                    dur1 = beats_per_bar * 0.6
                    dur2 = beats_per_bar * 0.4
                    vel = int(vel_min + (vel_max - vel_min) * 0.5)
                    notes.append(MelodyNote(struct_pitch, dur1, vel,
                                             round(offset, 4)))
                    notes.append(MelodyNote(dest, dur2, vel,
                                             round(offset + dur1, 4)))
                beats_remaining = 0.0

            else:
                # Alta tensión: figura ornamentada
                # bordadura superior + nota estructural + notas de paso
                upper_nb = self._neighbor_note(struct_pitch, scale_pitches,
                                                upper=True)
                dest = structural_pitches[min(bar_idx + 1, n_bars - 1)]
                passing = self._passing_notes(struct_pitch, dest, scale_pitches)

                beat = 0.0
                vel_base = int(vel_min + (vel_max - vel_min) * 0.8)

                # Appoggiatura (bordadura)
                notes.append(MelodyNote(upper_nb, 0.25, vel_base - 10,
                                         round(offset + beat, 4)))
                beat += 0.25

                # Nota estructural
                notes.append(MelodyNote(struct_pitch, 0.5, vel_base,
                                         round(offset + beat, 4)))
                beat += 0.5

                # Notas de paso hacia siguiente estructura
                for ps_p in passing[:2]:
                    if beat < beats_per_bar - 0.01:
                        notes.append(MelodyNote(ps_p, 0.25, vel_base - 15,
                                                 round(offset + beat, 4)))
                        beat += 0.25

                # Relleno si queda tiempo
                while beat < beats_per_bar - 0.01:
                    notes.append(MelodyNote(struct_pitch, 0.25, vel_base - 5,
                                             round(offset + beat, 4)))
                    beat += 0.25
                beats_remaining = 0.0

            offset += beats_per_bar

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 3: A* SEARCH
# ══════════════════════════════════════════════════════════════════════════════

class SearchEngine:
    """
    Búsqueda heurística A* sobre el espacio de notas.
    El estado es (pitch_actual, tiempo_actual).
    El coste acumula penalizaciones locales.
    La heurística estima el coste hasta completar el objetivo.
    """

    def __init__(self, profile: str = "serene", rng: random.Random = None):
        self.profile = profile
        self.rng     = rng or random.Random(42)

    def _local_cost(self, from_pitch: int, to_pitch: int,
                    duration: float, offset: float, total_beats: float,
                    scale_pitches: List[int], contour_val: float,
                    tension_val: float, pitch_range: Tuple[int, int]) -> float:
        """Coste de transición entre dos notas."""
        low, high = pitch_range
        cost = 0.0
        iv = abs(to_pitch - from_pitch)

        # Penalizar notas fuera de escala
        if to_pitch not in scale_pitches:
            cost += 3.0

        # Penalizar notas fuera de rango
        if to_pitch < low or to_pitch > high:
            cost += 10.0

        # Penalizar saltos muy grandes (> octava)
        if iv > 12:
            cost += (iv - 12) * 0.5

        # Penalizar estancamiento (misma nota repetida)
        if iv == 0:
            cost += 1.5

        # Premio por consonancias (3ª, 5ª, 6ª)
        iv_mod = iv % 12
        if iv_mod in (3, 4, 7, 8, 9):
            cost -= 0.3

        # Penalizar contorno incorrecto
        rel_pos = (to_pitch - low) / max(high - low, 1)
        contour_error = abs(rel_pos - contour_val)
        cost += contour_error * 0.8

        return cost

    def _heuristic(self, pitch: int, offset: float, total_beats: float,
                   scale_pitches: List[int]) -> float:
        """Estimación optimista del coste restante."""
        # Fracción restante de la obra
        fraction_done = offset / max(total_beats, 1)
        remaining = 1.0 - fraction_done
        # Penalización base proporcional al tiempo restante
        return remaining * 0.1

    def generate(self, scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float], vel_range: Tuple[int, int, int],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        """
        Genera la melodía mediante búsqueda A*.
        Para mantener la eficiencia, trabaja nota a nota con beam search (k=5).
        """
        low, high = pitch_range
        vel_min, vel_max, _ = vel_range
        total_beats = n_bars * beats_per_bar
        beam_width = 5

        # Estado: (coste_acumulado, offset, pitch, notas_generadas)
        start_pitch = snap_to_scale(
            low + (high - low) // 3, scale_pitches)

        # Beam: lista de (coste, offset, pitch, notas)
        beam: List[Tuple[float, float, int, List[MelodyNote]]] = [
            (0.0, 0.0, start_pitch, [])
        ]

        while beam:
            # Seleccionar el mejor estado del beam
            beam.sort(key=lambda x: x[0])
            best_cost, offset, cur_pitch, notes_so_far = beam[0]

            if offset >= total_beats - 0.01:
                return notes_so_far

            bar_idx = int(offset // beats_per_bar)
            c_val = contour[min(bar_idx, len(contour) - 1)]
            t_val = tension[min(bar_idx, len(tension) - 1)]

            # Duraciones candidatas
            beats_left_in_bar = beats_per_bar - (offset % beats_per_bar)
            dur_options = [d for d in [0.25, 0.5, 1.0, 1.5, 2.0]
                           if d <= min(beats_left_in_bar,
                                       total_beats - offset) + 0.01]
            if not dur_options:
                return notes_so_far

            # Pitches candidatos: vecinos en la escala
            neighbors = []
            for sp in scale_pitches:
                if low <= sp <= high:
                    neighbors.append(sp)

            new_beam = []
            seen = set()

            for dur in dur_options[:3]:
                for nxt_pitch in neighbors:
                    key = (round(offset, 2), nxt_pitch, dur)
                    if key in seen:
                        continue
                    seen.add(key)

                    lc = self._local_cost(cur_pitch, nxt_pitch, dur, offset,
                                          total_beats, scale_pitches,
                                          c_val, t_val, pitch_range)
                    h  = self._heuristic(nxt_pitch, offset + dur,
                                         total_beats, scale_pitches)
                    total_cost = best_cost + lc + h

                    climax_factor = math.sin(
                        (offset / total_beats) * math.pi) * t_val
                    vel = int(vel_min + (vel_max - vel_min) * climax_factor)
                    vel = max(vel_min, min(vel_max, vel))

                    new_note = MelodyNote(nxt_pitch, dur, vel,
                                          round(offset, 4))
                    new_beam.append((
                        total_cost,
                        offset + dur,
                        nxt_pitch,
                        notes_so_far + [new_note],
                    ))

            # Reducir beam
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]

        # Si el beam se vacía, devolver lo que hay
        if beam:
            return beam[0][3]
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 4: GENÉTICO
# ══════════════════════════════════════════════════════════════════════════════

class GeneticEngine:
    """
    Algoritmo genético para evolucionar una población de melodías.
    Cada individuo es una lista de (pitch_idx, dur_idx) sobre la escala.
    """

    def __init__(self, profile: str, rhythm: str,
                 population_size: int = 20, generations: int = 30,
                 rng: random.Random = None):
        self.profile    = profile
        self.rhythm     = rhythm
        self.pop_size   = population_size
        self.generations = generations
        self.rng        = rng or random.Random(42)

    def _random_individual(self, n_notes: int, n_pitches: int,
                            n_durs: int) -> List[Tuple[int, int]]:
        return [(self.rng.randint(0, n_pitches - 1),
                 self.rng.randint(0, n_durs - 1))
                for _ in range(n_notes)]

    def _fitness(self, individual: List[Tuple[int, int]],
                 scale_pitches: List[int], dur_options: List[float],
                 contour: List[float], tension: List[float],
                 beats_per_bar: float) -> float:
        """Función de fitness multicriterio."""
        pitches = [scale_pitches[min(idx, len(scale_pitches)-1)]
                   for idx, _ in individual]
        durs    = [dur_options[min(idx, len(dur_options)-1)]
                   for _, idx in individual]

        if len(pitches) < 2:
            return 0.0

        # 1. Suavidad de intervalos
        intervals = [abs(pitches[i+1] - pitches[i])
                     for i in range(len(pitches) - 1)]
        big_leaps = sum(1 for iv in intervals if iv > 7)
        smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

        # 2. Conformidad con el contorno
        contour_error = 0.0
        low, high = min(pitches), max(pitches)
        rng_val = high - low or 1
        for i, p in enumerate(pitches):
            bar_idx = min(int(i * beats_per_bar / max(len(pitches), 1)),
                          len(contour) - 1)
            rel = (p - low) / rng_val
            contour_error += abs(rel - contour[bar_idx])
        contour_score = max(0.0, 1.0 - contour_error / len(pitches))

        # 3. Variedad rítmica
        variety = min(1.0, len(set(round(d, 2) for d in durs)) / 4)

        # 4. Rango razonable (entre 5ª y 2 octavas)
        rng_semitones = max(pitches) - min(pitches)
        range_score = min(1.0, max(0.0, (rng_semitones - 5) / 19))

        # 5. Clímax en posición correcta (según perfil)
        max_pitch_pos = pitches.index(max(pitches)) / max(len(pitches) - 1, 1)
        if self.profile in ("heroic", "triumphant"):
            climax_score = 1.0 - abs(max_pitch_pos - 0.85)
        else:
            climax_score = 1.0 - abs(max_pitch_pos - 0.618)  # proporción áurea

        return (smoothness  * 0.25 +
                contour_score * 0.30 +
                variety       * 0.15 +
                range_score   * 0.15 +
                climax_score  * 0.15)

    def _crossover(self, p1: List[Tuple], p2: List[Tuple]) -> List[Tuple]:
        """Cruce de punto único."""
        if len(p1) < 2:
            return deepcopy(p1)
        point = self.rng.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:]

    def _mutate(self, individual: List[Tuple[int, int]],
                n_pitches: int, n_durs: int,
                rate: float = 0.1) -> List[Tuple[int, int]]:
        """Mutación puntual."""
        result = []
        for pitch_idx, dur_idx in individual:
            if self.rng.random() < rate:
                # Mutar pitch ±1-3 grados de escala
                delta = self.rng.randint(-3, 3)
                pitch_idx = max(0, min(n_pitches - 1, pitch_idx + delta))
            if self.rng.random() < rate * 0.5:
                dur_idx = self.rng.randint(0, n_durs - 1)
            result.append((pitch_idx, dur_idx))
        return result

    def generate(self, scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float], vel_range: Tuple[int, int, int],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        """Evoluciona la población y devuelve la mejor melodía."""
        low, high = pitch_range
        vel_min, vel_max, _ = vel_range
        total_beats = n_bars * beats_per_bar

        # Filtrar escala al rango
        scale_in_range = [p for p in scale_pitches if low <= p <= high]
        if not scale_in_range:
            scale_in_range = scale_pitches

        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])
        dur_options = sorted(rd.keys())

        # Estimar número de notas
        avg_dur = sum(d * w for d, w in rd.items()) / max(sum(rd.values()), 1)
        n_notes = max(4, int(total_beats / avg_dur))

        # Inicializar población
        population = [
            self._random_individual(n_notes, len(scale_in_range),
                                    len(dur_options))
            for _ in range(self.pop_size)
        ]

        # Evolución
        for gen in range(self.generations):
            # Evaluar
            scored = [
                (self._fitness(ind, scale_in_range, dur_options,
                               contour, tension, beats_per_bar), ind)
                for ind in population
            ]
            scored.sort(key=lambda x: -x[0])

            # Elitismo: conservar top 2
            elite = [deepcopy(ind) for _, ind in scored[:2]]

            # Nueva generación
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                # Selección por torneo
                t1 = self.rng.choice(scored[:self.pop_size // 2])[1]
                t2 = self.rng.choice(scored[:self.pop_size // 2])[1]
                child = self._crossover(t1, t2)
                child = self._mutate(child, len(scale_in_range),
                                     len(dur_options))
                new_pop.append(child)

            population = new_pop

        # Mejor individuo
        best = max(
            population,
            key=lambda ind: self._fitness(ind, scale_in_range, dur_options,
                                          contour, tension, beats_per_bar)
        )

        # Convertir a MelodyNote
        notes = []
        offset = 0.0
        for i, (pitch_idx, dur_idx) in enumerate(best):
            if offset >= total_beats - 0.01:
                break
            p = scale_in_range[min(pitch_idx, len(scale_in_range) - 1)]
            d = dur_options[min(dur_idx, len(dur_options) - 1)]
            d = min(d, total_beats - offset)

            bar_idx = int(offset // beats_per_bar)
            t_val = tension[min(bar_idx, len(tension) - 1)]
            climax = math.sin((offset / total_beats) * math.pi) * t_val
            vel = int(vel_min + (vel_max - vel_min) * climax)
            vel = max(vel_min, min(vel_max, vel))

            notes.append(MelodyNote(p, d, vel, round(offset, 4)))
            offset += d

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 5: HYBRID (grammar + markov)
# ══════════════════════════════════════════════════════════════════════════════

class HybridEngine:
    """
    Combina GrammarEngine (estructura) + MarkovEngine (ornamentación).
    GrammarEngine define los puntos estructurales y las cadencias.
    MarkovEngine rellena los espacios intermedios con fluidez estilística.
    """

    def __init__(self, root_pc: int, mode: str, profile: str,
                 rhythm: str, rng: random.Random = None):
        self.grammar = GrammarEngine(root_pc, mode, rng)
        self.markov  = MarkovEngine(profile, rhythm, rng)
        self.profile = profile
        self.rng     = rng or random.Random(42)

    def generate(self, scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float], vel_range: Tuple[int, int, int],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        """
        Genera estructura gramatical, luego enriquece con Markov
        donde la tensión es alta.
        """
        grammar_notes = self.grammar.generate(
            scale_pitches, n_bars, beats_per_bar,
            contour, tension, vel_range, pitch_range)
        markov_notes  = self.markov.generate(
            scale_pitches, n_bars, beats_per_bar,
            contour, tension, vel_range, pitch_range)

        if not grammar_notes:
            return markov_notes
        if not markov_notes:
            return grammar_notes

        # Mezclar: por compás, elegir grammar (tensión baja) o markov (alta)
        result = []
        total_beats = n_bars * beats_per_bar

        def notes_in_bar(notes: List[MelodyNote], bar: int) -> List[MelodyNote]:
            s = bar * beats_per_bar
            e = s + beats_per_bar
            return [n for n in notes if s <= n.offset < e]

        for bar_idx in range(n_bars):
            t_val = tension[min(bar_idx, len(tension) - 1)]
            g_bar = notes_in_bar(grammar_notes, bar_idx)
            m_bar = notes_in_bar(markov_notes,  bar_idx)

            # Alta tensión → Markov (más movimiento); baja → Grammar (estructura)
            if t_val > 0.55 and m_bar:
                result.extend(m_bar)
            elif g_bar:
                result.extend(g_bar)
            elif m_bar:
                result.extend(m_bar)

        return sorted(result, key=lambda n: n.offset)


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING DE MELODÍAS
# ══════════════════════════════════════════════════════════════════════════════

def score_melody(notes: List[MelodyNote], scale_pitches: List[int],
                 contour: List[float], beats_per_bar: float,
                 profile: str) -> float:
    """
    Puntuación multicriterio de la melodía generada.
    Retorna un valor en [0, 1].
    """
    if not notes:
        return 0.0

    pitches = [n.pitch for n in notes]
    durs    = [n.duration for n in notes]
    vels    = [n.velocity for n in notes]

    # 1. Consonancia: porcentaje de notas en la escala
    in_scale = sum(1 for p in pitches if p in scale_pitches)
    consonance = in_scale / len(pitches)

    # 2. Variedad rítmica
    variety = min(1.0, len(set(round(d, 2) for d in durs)) / 5)

    # 3. Rango melódico (óptimo: entre 7 y 17 semitonos)
    rng = max(pitches) - min(pitches) if pitches else 0
    range_score = min(1.0, max(0.0, 1 - abs(rng - 12) / 12))

    # 4. Suavidad de movimiento
    intervals = [abs(pitches[i+1] - pitches[i])
                 for i in range(len(pitches) - 1)]
    big_leaps  = sum(1 for iv in intervals if iv > 7)
    smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 5. Arco dinámico
    vel_std = float(np.std(vels)) if len(vels) >= 4 else 5.0
    arc_score = min(1.0, vel_std / 15.0)

    # 6. Conformidad con el contorno
    if pitches:
        low, high = min(pitches), max(pitches)
        rng_v = high - low or 1
        contour_errors = []
        for i, n in enumerate(notes):
            bar_idx = int(n.offset // beats_per_bar)
            c_val = contour[min(bar_idx, len(contour) - 1)]
            rel = (n.pitch - low) / rng_v
            contour_errors.append(abs(rel - c_val))
        contour_score = max(0.0, 1.0 - np.mean(contour_errors))
    else:
        contour_score = 0.5

    # 7. Proporcionalidad del clímax (oro: 0.618)
    if pitches:
        climax_pos = pitches.index(max(pitches)) / max(len(pitches) - 1, 1)
        if profile in ("heroic", "triumphant"):
            climax_score = 1.0 - abs(climax_pos - 0.85)
        else:
            climax_score = 1.0 - abs(climax_pos - 0.618)
    else:
        climax_score = 0.5

    return (consonance   * 0.20 +
            variety      * 0.15 +
            range_score  * 0.10 +
            smoothness   * 0.20 +
            arc_score    * 0.10 +
            contour_score * 0.15 +
            climax_score * 0.10)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MOTIVO SEMILLA
# ══════════════════════════════════════════════════════════════════════════════

def extract_motif(notes: List[MelodyNote], beats_per_bar: float,
                  motif_bars: int = 2) -> List[MelodyNote]:
    """
    Extrae el motivo semilla: las primeras motif_bars compases.
    El motivo es la «célula generativa» que puede importarse en
    phrase_builder, leitmotif_tracker o variation_engine.
    """
    threshold = motif_bars * beats_per_bar
    motif = [n for n in notes if n.offset < threshold]
    return motif


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT: MIDI
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_midi(notes: List[MelodyNote], tempo_bpm: int,
                  output_path: str, ticks_per_beat: int = 480,
                  track_name: str = "Melody"):
    """Convierte una lista de MelodyNote a un archivo MIDI."""
    if not MIDO_OK:
        print("[ERROR] mido no disponible. No se puede exportar MIDI.")
        return

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = int(60_000_000 / max(tempo_bpm, 1))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.Message("program_change", channel=0, program=73, time=0))
    # program 73 = flauta — voz melódica por defecto

    def q_to_ticks(q: float) -> int:
        return int(q * ticks_per_beat)

    # Convertir a eventos absolutos
    events = []
    for n in notes:
        on_tick  = q_to_ticks(n.offset)
        off_tick = q_to_ticks(n.offset + n.duration)
        events.append((on_tick,  "note_on",  n.pitch, n.velocity))
        events.append((off_tick, "note_off", n.pitch, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    current_tick = 0
    for abs_tick, msg_type, note, vel in events:
        delta = abs_tick - current_tick
        delta = max(0, delta)
        track.append(mido.Message(msg_type, channel=0, note=note,
                                   velocity=vel, time=delta))
        current_tick = abs_tick

    mid.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT: FINGERPRINT (compatible con stitcher.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_fingerprint(result: MelodyResult, output_midi_path: str) -> dict:
    """
    Construye un fingerprint compatible con stitcher.py y midi_dna_unified.
    """
    notes = result.notes
    if not notes:
        return {}

    pitches = [n.pitch for n in notes]
    durs    = [n.duration for n in notes]
    vels    = [n.velocity for n in notes]

    def dominant_pc(pitches_list: List[int]) -> int:
        counts = Counter(p % 12 for p in pitches_list)
        return counts.most_common(1)[0][0] if counts else 0

    root_pc, mode = parse_key(result.key)
    scale_pcs = [(root_pc + iv) % 12
                 for iv in SCALE_INTERVALS.get(result.mode, [0, 2, 4, 5, 7, 9, 11])]

    # Primeras y últimas notas (entrada y salida)
    first_notes = [n for n in notes if n.offset < result.bars * (4 / result.bars)]
    last_notes  = [n for n in notes[-max(1, len(notes)//8):]]

    fingerprint = {
        "meta": {
            "source":        "melody_generator.py",
            "engine":        result.engine,
            "profile":       result.profile,
            "key":           result.key,
            "mode":          result.mode,
            "bars":          result.bars,
            "tempo_bpm":     result.tempo,
            "score":         result.score,
            "total_notes":   len(notes),
        },
        "entry": {
            "key":           result.key,
            "mode":          result.mode,
            "dominant_pc":   dominant_pc(pitches[:max(1, len(pitches)//8)]),
            "mean_pitch":    float(np.mean(pitches[:max(1, len(pitches)//8)])),
            "mean_velocity": float(np.mean(vels[:max(1, len(vels)//8)])),
            "density_per_beat": len(first_notes) / max(4.0, 1),
            "tempo_bpm":     result.tempo,
        },
        "exit": {
            "key":           result.key,
            "mode":          result.mode,
            "dominant_pc":   dominant_pc([n.pitch for n in last_notes]),
            "mean_pitch":    float(np.mean([n.pitch for n in last_notes])),
            "mean_velocity": float(np.mean([n.velocity for n in last_notes])),
            "density_per_beat": len(last_notes) / max(4.0, 1),
            "last_pitch":    notes[-1].pitch if notes else 60,
            "last_pitch_pc": notes[-1].pitch_class if notes else 0,
            "last_duration": notes[-1].duration if notes else 1.0,
        },
        "tension": {
            "mean":  float(np.mean(vels)) / 127,
            "max":   float(np.max(vels))  / 127,
            "min":   float(np.min(vels))  / 127,
            "std":   float(np.std(vels))  / 127,
        },
        "style": {
            "mean_pitch":    float(np.mean(pitches)),
            "pitch_range":   int(np.max(pitches) - np.min(pitches)),
            "mean_duration": float(np.mean(durs)),
            "scale_pcs":     scale_pcs,
            "contour":       result.contour,
            "profile":       result.profile,
        },
        "motif": [
            {"pitch": n.pitch, "duration": n.duration,
             "velocity": n.velocity, "offset": n.offset}
            for n in result.motif
        ],
    }
    return fingerprint


def export_fingerprint(fingerprint: dict, output_midi_path: str):
    """Guarda el fingerprint como JSON junto al MIDI."""
    json_path = output_midi_path.replace(".mid", ".fingerprint.json")
    if not json_path.endswith(".fingerprint.json"):
        json_path += ".fingerprint.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)
    print(f"[fingerprint] Exportado: {json_path}")
    return json_path


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE INTEGRACIONES
# ══════════════════════════════════════════════════════════════════════════════

def load_from_theorist(path: str) -> dict:
    """Lee un .theorist.json y extrae parámetros para melody_generator."""
    with open(path) as f:
        data = json.load(f)
    pp = data.get("pipeline_parameters", {})
    narrator_pp = pp.get("narrator", {})
    midi_pp     = pp.get("midi_dna_unified", {})
    cpg_pp      = pp.get("chord_progression_generator", {})
    tension     = data.get("tension_curves", {})

    key_str  = midi_pp.get("--key",    narrator_pp.get("--key",    "C"))
    mode_str = midi_pp.get("--mode",   narrator_pp.get("--mode",   "major"))
    bars     = int(midi_pp.get("--bars",  narrator_pp.get("--bars",  32)))
    tempo    = int(midi_pp.get("--tempo", narrator_pp.get("--tempo", 120)))

    result = {
        "key": key_str, "mode": mode_str, "bars": bars, "tempo": tempo,
        "tension_curve": tension.get("tension", []),
        "activity_curve": tension.get("activity", []),
    }
    if "--style" in cpg_pp:
        result["style"] = cpg_pp["--style"]
    return result


def load_from_narrator(path: str) -> dict:
    """Lee un obra_plan.json de narrator.py."""
    with open(path) as f:
        plan = json.load(f)
    key   = plan.get("key", "C")
    tempo = int(plan.get("tempo", 120))
    sections = plan.get("sections", [])
    total_bars = sum(s.get("bars", 8) for s in sections)
    tc = plan.get("tension_curves", {})
    return {
        "key": key, "bars": total_bars, "tempo": tempo,
        "sections": sections,
        "tension_curve": tc.get("tension", []),
        "activity_curve": tc.get("activity", []),
    }


def load_from_curves(path: str) -> dict:
    """Lee un .curves.json de tension_designer.py."""
    with open(path) as f:
        curves = json.load(f)
    return {
        "tension_curve":  curves.get("tension",  []),
        "activity_curve": curves.get("activity", []),
        "register_curve": curves.get("register", []),
        "harmony_curve":  curves.get("harmony",  []),
    }


def parse_chords_text(chords_str: str) -> List[Tuple[str, float]]:
    """
    Parsea 'C Am F G' o 'Cm:2 G7:2 Fm:1 Bb7:1' →
    lista de (chord_name, beats).
    """
    tokens = chords_str.strip().split()
    result = []
    for token in tokens:
        if ":" in token:
            name, beats = token.split(":", 1)
            result.append((name, float(beats)))
        else:
            result.append((token, 4.0))  # default: 1 compás de 4/4
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class MelodyGenerator:
    """
    Clase principal que coordina los motores de generación.
    """

    def __init__(self, key: str = "C", mode: str = "major",
                 bars: int = 16, tempo: int = 120,
                 time_sig: str = "4/4", engine: str = "hybrid",
                 profile: str = "serene", rhythm: str = "flowing",
                 contour: str = "arch", custom_contour: Optional[List[float]] = None,
                 pitch_low: int = 60, pitch_high: int = 84,
                 tension_curve: Optional[List[float]] = None,
                 chords: Optional[List[Tuple[str, float]]] = None,
                 corpus_dir: Optional[str] = None,
                 seed: int = 42, verbose: bool = False):

        self.key     = key
        self.mode    = mode
        self.bars    = bars
        self.tempo   = tempo
        self.engine  = engine
        self.profile = profile
        self.rhythm  = rhythm
        self.contour_shape  = contour
        self.custom_contour = custom_contour
        self.pitch_low  = pitch_low
        self.pitch_high = pitch_high
        self.tension_curve_input = tension_curve
        self.chords  = chords
        self.corpus_dir = corpus_dir
        self.verbose = verbose

        # Parsear compás
        num, den = time_sig.split("/")
        self.beats_per_bar = float(num)
        self.beat_unit     = int(den)

        # RNG reproducible
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Calcular escala
        self.root_pc, detected_mode = parse_key(key)
        if mode == "auto":
            self.mode = detected_mode
        self.scale_pitches = get_scale_pitches(
            self.root_pc, self.mode, pitch_low - 12, pitch_high + 12)

        # Filtrar al rango real
        self.scale_pitches = [p for p in self.scale_pitches
                               if pitch_low <= p <= pitch_high]
        if not self.scale_pitches:
            # Ampliar rango si es necesario
            self.scale_pitches = get_scale_pitches(self.root_pc, self.mode,
                                                    pitch_low - 24, pitch_high + 24)

        # Curvas
        self.contour_curve = build_contour_curve(
            contour, bars, custom_contour)
        self.tension_curve = (
            tension_curve if tension_curve
            else build_tension_curve(profile, bars))

        # Ajustar longitud de curvas
        self.contour_curve = self._resize_curve(self.contour_curve, bars)
        self.tension_curve = self._resize_curve(self.tension_curve, bars)

        # Velocidades
        self.vel_range = PROFILE_VELOCITY.get(profile, (50, 90, 80))

        if verbose:
            self._print_config()

    def _resize_curve(self, curve: List[float], n: int) -> List[float]:
        if len(curve) == n:
            return curve
        if not curve:
            return [0.5] * n
        return list(np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(curve)),
            curve,
        ))

    def _print_config(self):
        scale_names = [NOTE_NAMES[(self.root_pc + iv) % 12]
                       for iv in SCALE_INTERVALS.get(self.mode, [])]
        print(f"┌─ MELODY GENERATOR ──────────────────────────────")
        print(f"│  Tonalidad : {self.key} {self.mode}")
        print(f"│  Escala    : {' '.join(scale_names)}")
        print(f"│  Compases  : {self.bars} ({self.beats_per_bar}/4)")
        print(f"│  Tempo     : {self.tempo} BPM")
        print(f"│  Motor     : {self.engine}")
        print(f"│  Perfil    : {self.profile}")
        print(f"│  Ritmo     : {self.rhythm}")
        print(f"│  Contorno  : {self.contour_shape}")
        print(f"│  Rango     : MIDI {self.pitch_low}-{self.pitch_high}")
        print(f"└─────────────────────────────────────────────────")

    def _get_engine(self) -> Any:
        """Instancia el motor de generación seleccionado."""
        e = self.engine.lower()
        rng = self.rng
        if e == "markov":
            eng = MarkovEngine(self.profile, self.rhythm, rng)
            if self.corpus_dir:
                eng.train_from_corpus(self.corpus_dir, self.verbose)
            return eng
        elif e == "grammar":
            return GrammarEngine(self.root_pc, self.mode, rng)
        elif e == "search":
            return SearchEngine(self.profile, rng)
        elif e == "genetic":
            return GeneticEngine(self.profile, self.rhythm,
                                 population_size=20, generations=30, rng=rng)
        else:  # hybrid (default)
            eng = HybridEngine(self.root_pc, self.mode, self.profile,
                               self.rhythm, rng)
            if self.corpus_dir:
                eng.markov.train_from_corpus(self.corpus_dir, self.verbose)
            return eng

    def generate_once(self, seed_offset: int = 0) -> MelodyResult:
        """Genera una melodía con el motor configurado."""
        rng_local = random.Random(self.rng.randint(0, 2**31) + seed_offset)

        eng = self._get_engine()
        notes = eng.generate(
            scale_pitches=self.scale_pitches,
            n_bars=self.bars,
            beats_per_bar=self.beats_per_bar,
            contour=self.contour_curve,
            tension=self.tension_curve,
            vel_range=self.vel_range,
            pitch_range=(self.pitch_low, self.pitch_high),
        )

        if not notes:
            notes = []

        motif = extract_motif(notes, self.beats_per_bar, motif_bars=2)

        score = score_melody(notes, set(self.scale_pitches),
                             self.contour_curve, self.beats_per_bar,
                             self.profile)

        return MelodyResult(
            notes=notes,
            key=self.key,
            mode=self.mode,
            bars=self.bars,
            tempo=self.tempo,
            engine=self.engine,
            profile=self.profile,
            contour=self.contour_shape,
            score=score,
            motif=motif,
        )

    def generate_candidates(self, n: int = 3) -> List[MelodyResult]:
        """Genera n candidatos y los ordena por score."""
        candidates = []
        for i in range(n):
            result = self.generate_once(seed_offset=i * 1337)
            candidates.append(result)
            if self.verbose:
                print(f"  [candidato {i+1}/{n}] score={result.score:.3f} "
                      f"notas={len(result.notes)}")
        return sorted(candidates, key=lambda r: -r.score)

    def generate_per_section(self, sections: List[dict]) -> List[MelodyResult]:
        """
        Genera una melodía por cada sección del narrator.
        Cada sección puede tener su propia tonalidad, tempo y curvas.
        """
        results = []
        for i, sec in enumerate(sections):
            sec_bars = sec.get("bars", 8)
            sec_key  = sec.get("key", self.key)
            sec_tempo = sec.get("tempo", self.tempo)
            sec_tension = sec.get("tension", [self.tension_curve[
                min(i, len(self.tension_curve)-1)]])

            gen = MelodyGenerator(
                key=sec_key, mode=self.mode,
                bars=sec_bars, tempo=sec_tempo,
                engine=self.engine, profile=self.profile,
                rhythm=self.rhythm, contour=self.contour_shape,
                pitch_low=self.pitch_low, pitch_high=self.pitch_high,
                tension_curve=sec_tension,
                seed=self.rng.randint(0, 2**31),
                verbose=self.verbose,
            )
            result = gen.generate_once()
            result.metadata["section_name"] = sec.get("name", f"S{i+1}")
            results.append(result)
            if self.verbose:
                print(f"  [sección {i+1}/{len(sections)}] "
                      f"{sec.get('name','?')} bars={sec_bars} "
                      f"score={result.score:.3f}")
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def play_midi(path: str, seconds: float = 15.0):
    """Reproduce un MIDI usando pygame."""
    if not PYGAME_OK:
        print("[WARN] pygame no disponible. No se puede reproducir.")
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        import time
        time.sleep(seconds)
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"[WARN] Error reproduciendo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(gen: MelodyGenerator, output_base: str):
    """
    Modo interactivo: genera, muestra info, el usuario acepta o regenera.
    """
    print("\n╔═ MODO INTERACTIVO ═══════════════════════════════════╗")
    print("║  g = generar nueva  |  a = aceptar  |  q = salir    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    iteration = 0
    while True:
        iteration += 1
        result = gen.generate_once(seed_offset=iteration * 999)
        out_path = f"{output_base}_iter{iteration}.mid"
        notes_to_midi(result.notes, result.tempo, out_path)

        print(f"\n[iter {iteration}] score={result.score:.3f}  "
              f"notas={len(result.notes)}  key={result.key} {result.mode}")
        print(f"  archivo: {out_path}")

        if PYGAME_OK:
            choice = input("  [reproducir? s/n] ").strip().lower()
            if choice == "s":
                play_midi(out_path, seconds=min(15, result.bars * 2))

        action = input("  [a]ceptar / [g]enerar otra / [q]uit: ").strip().lower()
        if action == "a":
            print(f"\n✓ Melodía aceptada: {out_path}")
            return result, out_path
        elif action == "q":
            print("Saliendo sin guardar.")
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  DESCRIPCIÓN LIBRE → PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

def params_from_description(desc: str) -> dict:
    """Extrae parámetros de una descripción en lenguaje natural."""
    params = {}

    # Perfil emocional
    params["profile"] = detect_profile_from_description(desc)

    # Modo
    detected_mode = detect_mode_from_description(desc)
    if detected_mode:
        params["mode"] = detected_mode
    else:
        params["mode"] = PROFILE_TO_MODE.get(params["profile"], "major")

    # Tonalidad
    key = detect_key_from_description(desc)
    if key:
        params["key"] = key

    desc_lower = desc.lower()

    # Tempo desde adjetivos
    if any(w in desc_lower for w in ["lento", "slow", "pausado", "tranquilo"]):
        params["tempo"] = 60
    elif any(w in desc_lower for w in ["rápido", "fast", "veloz", "allegro"]):
        params["tempo"] = 160
    elif any(w in desc_lower for w in ["moderado", "moderato"]):
        params["tempo"] = 100

    # Compases
    m = re.search(r'(\d+)\s*compas', desc_lower)
    if m:
        params["bars"] = int(m.group(1))

    # Contorno
    if any(w in desc_lower for w in ["ascend", "sube", "crece"]):
        params["contour"] = "ascending"
    elif any(w in desc_lower for w in ["descend", "baja", "decrece"]):
        params["contour"] = "descending"
    elif any(w in desc_lower for w in ["arco", "arch", "clímax"]):
        params["contour"] = "arch"

    return params


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MELODY GENERATOR — Generación de melodías originales desde cero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("description", nargs="?", default=None,
                   help="Descripción libre en lenguaje natural")

    # Parámetros musicales
    p.add_argument("--key",      default="C",    help="Tonalidad (ej: C, Am, F#, Bb)")
    p.add_argument("--mode",     default="auto",
                   choices=list(SCALE_INTERVALS.keys()) + ["auto"],
                   help="Modo de escala (default: auto)")
    p.add_argument("--bars",     type=int,   default=16, help="Compases a generar")
    p.add_argument("--tempo",    type=int,   default=120, help="Tempo BPM")
    p.add_argument("--time-sig", default="4/4", help="Compás (4/4, 3/4, 6/8…)")

    # Motor y estilo
    p.add_argument("--engine",  default="hybrid",
                   choices=["markov", "grammar", "search", "genetic", "hybrid"],
                   help="Motor generativo")
    p.add_argument("--profile", default="serene",
                   choices=list(PROFILE_VELOCITY.keys()),
                   help="Perfil emocional")
    p.add_argument("--rhythm",  default="flowing",
                   choices=list(RHYTHM_DISTS.keys()),
                   help="Perfil rítmico")
    p.add_argument("--contour", default="arch",
                   choices=["arch", "ascending", "descending", "wave",
                            "plateau", "inverted", "erratic", "custom"],
                   help="Forma del contorno melódico")
    p.add_argument("--contour-values", type=str, default=None,
                   help="Valores custom para --contour custom: '0.2,0.5,0.8,0.5'")

    # Rango
    p.add_argument("--range", nargs=2, type=int, default=[60, 84],
                   metavar=("LOW", "HIGH"),
                   help="Rango MIDI (default: 60 84 = C4-C6)")

    # Candidatos
    p.add_argument("--candidates", type=int, default=3,
                   help="Generar N candidatos y exportar el mejor")

    # Acordes guía
    p.add_argument("--chords", type=str, default=None,
                   help="Progresión guía en texto: 'C Am F G'")
    p.add_argument("--chords-json", type=str, default=None,
                   help="Acordes desde .chords.json")

    # Curvas e integraciones
    p.add_argument("--curves",        type=str, default=None,
                   help="Cargar curvas de tension_designer (.curves.json)")
    p.add_argument("--from-theorist", type=str, default=None,
                   help="Leer parámetros de .theorist.json")
    p.add_argument("--from-narrator", type=str, default=None,
                   help="Leer plan de obra_plan.json (narrator.py)")
    p.add_argument("--per-section",   action="store_true",
                   help="Con --from-narrator: genera una melodía por sección")

    # Corpus Markov
    p.add_argument("--corpus", type=str, default=None,
                   help="Carpeta con MIDIs para entrenar Markov")

    # Exportación
    p.add_argument("--motif-only",          action="store_true",
                   help="Exportar solo el motivo semilla (2-4 compases)")
    p.add_argument("--export-fingerprint",  action="store_true",
                   help="Exportar .fingerprint.json para stitcher.py")
    p.add_argument("--output", type=str, default="melody_out.mid",
                   help="Archivo MIDI de salida")

    # Control
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--interactive", action="store_true",
                   help="Modo interactivo: genera, escucha, acepta o regenera")
    p.add_argument("--listen",      action="store_true",
                   help="Reproducir al final (requiere pygame)")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--dry-run",     action="store_true",
                   help="Mostrar parámetros sin generar MIDI")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Construir parámetros ────────────────────────────────────────────────
    params = {
        "key":          args.key,
        "mode":         args.mode,
        "bars":         args.bars,
        "tempo":        args.tempo,
        "time_sig":     args.time_sig,
        "engine":       args.engine,
        "profile":      args.profile,
        "rhythm":       args.rhythm,
        "contour":      args.contour,
        "pitch_low":    args.range[0],
        "pitch_high":   args.range[1],
        "seed":         args.seed,
        "verbose":      args.verbose,
        "tension_curve": None,
        "corpus_dir":   args.corpus,
        "custom_contour": None,
    }

    # ── 1. Descripción libre ────────────────────────────────────────────────
    if args.description:
        detected = params_from_description(args.description)
        for k, v in detected.items():
            if k in params and params[k] == getattr(args, k.replace("-", "_"),
                                                     None):
                params[k] = v  # solo sobreescribir si no fue especificado manualmente
            elif k not in ["key", "mode", "bars", "tempo"] or \
                 getattr(args, k, None) is None:
                params[k] = v
        if args.verbose:
            print(f"[descripción] '{args.description}'")
            print(f"  → perfil={params['profile']}  modo={params['mode']}")

    # ── 2. Desde theorist ───────────────────────────────────────────────────
    if args.from_theorist:
        if not os.path.exists(args.from_theorist):
            print(f"[ERROR] No se encuentra: {args.from_theorist}")
            sys.exit(1)
        ext = load_from_theorist(args.from_theorist)
        params.update({k: v for k, v in ext.items() if v is not None})
        if args.verbose:
            print(f"[theorist] Cargado: {args.from_theorist}")

    # ── 3. Desde narrator ───────────────────────────────────────────────────
    narrator_sections = None
    if args.from_narrator:
        if not os.path.exists(args.from_narrator):
            print(f"[ERROR] No se encuentra: {args.from_narrator}")
            sys.exit(1)
        ext = load_from_narrator(args.from_narrator)
        narrator_sections = ext.pop("sections", None)
        params.update({k: v for k, v in ext.items() if v is not None})
        if args.verbose:
            print(f"[narrator] Cargado: {args.from_narrator}  "
                  f"secciones={len(narrator_sections or [])}")

    # ── 4. Curvas de tensión ────────────────────────────────────────────────
    if args.curves:
        if not os.path.exists(args.curves):
            print(f"[ERROR] No se encuentra: {args.curves}")
            sys.exit(1)
        ext = load_from_curves(args.curves)
        if ext.get("tension_curve"):
            params["tension_curve"] = ext["tension_curve"]
        if args.verbose:
            print(f"[curves] Cargado: {args.curves}")

    # ── 5. Contorno personalizado ───────────────────────────────────────────
    if args.contour == "custom" and args.contour_values:
        try:
            vals = [float(v) for v in args.contour_values.split(",")]
            params["custom_contour"] = vals
        except ValueError:
            print("[WARN] --contour-values inválido. Usando arch.")
            params["contour"] = "arch"

    # ── 6. Modo auto ────────────────────────────────────────────────────────
    if params["mode"] == "auto":
        # Inferir desde tonalidad
        _, inferred = parse_key(params["key"])
        params["mode"] = inferred
        if args.verbose:
            print(f"[modo auto] → {params['mode']}")

    # ── Dry run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── PARÁMETROS RESUELTOS (dry-run) ─────────────────")
        for k, v in params.items():
            if k != "tension_curve" and k != "custom_contour":
                print(f"  {k:<20} = {v}")
        print("───────────────────────────────────────────────────")
        sys.exit(0)

    # ── Instanciar generador ─────────────────────────────────────────────────
    # Filtrar solo los parámetros que acepta MelodyGenerator.__init__
    VALID_PARAMS = {
        "key", "mode", "bars", "tempo", "time_sig", "engine", "profile",
        "rhythm", "contour", "custom_contour", "pitch_low", "pitch_high",
        "tension_curve", "chords", "corpus_dir", "seed", "verbose",
    }
    params = {k: v for k, v in params.items() if k in VALID_PARAMS}

    try:
        gen = MelodyGenerator(**params)
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar el generador: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    output_base = args.output.replace(".mid", "")

    # ── Modo interactivo ────────────────────────────────────────────────────
    if args.interactive:
        result, out_path = interactive_loop(gen, output_base)
        if result is None:
            sys.exit(0)
        results = [result]
        output_paths = [out_path]

    # ── Por secciones ────────────────────────────────────────────────────────
    elif args.per_section and narrator_sections:
        section_results = gen.generate_per_section(narrator_sections)
        results = section_results
        output_paths = []
        for i, res in enumerate(section_results):
            sec_name = res.metadata.get("section_name", f"sec{i+1}")
            out_p = f"{output_base}_{sec_name}.mid"
            notes_to_midi(res.notes, res.tempo, out_p)
            output_paths.append(out_p)
            print(f"[sección] {sec_name}: {out_p}  score={res.score:.3f}")

    # ── Modo estándar: generar candidatos ────────────────────────────────────
    else:
        print(f"\nGenerando {args.candidates} candidato(s)…")
        candidates = gen.generate_candidates(args.candidates)
        results = candidates
        output_paths = []

        for i, res in enumerate(candidates):
            suffix = "" if i == 0 else f"_v{i+1}"
            out_p  = f"{output_base}{suffix}.mid"

            # Si --motif-only, recortar a 2-4 compases
            if args.motif_only:
                res.notes = res.motif
                out_p = out_p.replace(".mid", ".motif.mid")

            notes_to_midi(res.notes, res.tempo, out_p)
            output_paths.append(out_p)
            tag = "★ MEJOR" if i == 0 else f"  v{i+1}"
            print(f"  {tag}  score={res.score:.3f}  "
                  f"notas={len(res.notes)}  → {out_p}")

    # ── Exportaciones adicionales ─────────────────────────────────────────
    if results and output_paths:
        best = results[0]
        best_path = output_paths[0]

        # JSON de la melodía
        json_path = best_path.replace(".mid", ".melody.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[json]    {json_path}")

        # Motivo como MIDI separado (si no es motif-only ya)
        if not args.motif_only and best.motif:
            motif_path = best_path.replace(".mid", ".motif.mid")
            notes_to_midi(best.motif, best.tempo, motif_path,
                          track_name="Motif")
            print(f"[motivo]  {motif_path}")

        # Fingerprint
        if args.export_fingerprint:
            fp = build_fingerprint(best, best_path)
            fp_path = export_fingerprint(fp, best_path)

        # Reproducir
        if args.listen:
            print(f"\nReproduciendo {best_path}…")
            play_midi(best_path, seconds=min(20, best.bars * 2))

    # ── Resumen final ──────────────────────────────────────────────────────
    if results:
        best = results[0]
        print(f"\n╔═ RESUMEN ════════════════════════════════════════╗")
        print(f"║  Tonalidad : {best.key} {best.mode}")
        print(f"║  Compases  : {best.bars}  Tempo: {best.tempo} BPM")
        print(f"║  Motor     : {best.engine}  Perfil: {best.profile}")
        print(f"║  Notas     : {len(best.notes)}")
        print(f"║  Score     : {best.score:.3f}")
        print(f"╚═════════════════════════════════════════════════╝")
        print()
        print("Siguientes pasos sugeridos:")
        print(f"  phrase_builder.py --motif {output_paths[0].replace('.mid','.motif.mid')} --form period")
        print(f"  variation_engine.py {output_paths[0]} --all")
        print(f"  reharmonizer.py {output_paths[0]} --strategy all")
        if args.export_fingerprint:
            print(f"  stitcher.py {output_paths[0].replace('.mid','.fingerprint.json')} *.fingerprint.json")


if __name__ == "__main__":
    main()
