#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MELODY CONDITIONED  v1.1                                  ║
║       Generación de melodías condicionadas a progresión de acordes          ║
║                                                                              ║
║  Este módulo condiciona la generación a una progresión de acordes dada,    ║
║  mediante reglas armónicas determinísticas o mediante un LLM.               ║
║                                                                              ║
║  MOTORES DE GENERACIÓN (--engine):                                           ║
║    chord_guided — Condicionamiento determinístico: filtra y pondera las     ║
║                   notas según chord tones, tensiones y notas de paso del    ║
║                   acorde activo. Combina Markov condicionado por tensión,   ║
║                   contorno melódico emocional y tabla EMOTIONAL_CHORD_      ║
║                   WEIGHTS. No requiere entrenamiento previo.                ║
║    llm          — Usa Claude o ChatGPT como motor generativo mediante       ║
║                   prompting estructurado. Modos: direct (descripción        ║
║                   simbólica) y fewshot (ejemplos reales del corpus).        ║
║                                                                              ║
║  FUENTES DE ACORDES:                                                         ║
║    --chords "Am:2 G:2 F:2 E7:2"  texto directo (beats por acorde)          ║
║    --chords-json  prog.chords.json  JSON del ecosistema                     ║
║    --chords-midi  fichero.mid       extrae acordes del track de armonía     ║
║                                                                              ║
║  CONDICIONAMIENTO EMOCIONAL (sin LLM):                                       ║
║    1. Perfil de intervalos por estado, sesgado por grado del acorde         ║
║    2. Curvas tensión/valencia/arousal → densidad, saltos, disonancia        ║
║    3. Tabla EMOTIONAL_CHORD_WEIGHTS: estado × acorde → pesos de notas       ║
║    4. Contorno melódico inferido desde el estado emocional                  ║
║    5. Markov condicionado por tensión en tiempo real                        ║
║                                                                              ║
║  FORMATO DEL CORPUS:                                                         ║
║    Track 0 — metadatos (tempo, time signature)                              ║
║    Track 1 — melodía  (canal 0)                                             ║
║    Track 2 — acordes  (canal 1)                                             ║
║                                                                              ║
║  USO:                                                                        ║
║    # Modo chord_guided (sin entrenamiento):                                  ║
║    python melody_conditioned.py --engine chord_guided                        ║
║        --chords "Am:2 G:2 F:2 E7:2" --key Am --bars 8                      ║
║        --profile melancholic --contour descending                           ║
║                                                                              ║
║    # Con acordes desde MIDI (incluir acordes en la salida):                 ║
║    python melody_conditioned.py --engine chord_guided                        ║
║        --chords-midi harmony.mid --key Am --bars 8 --include-chords         ║
║                                                                              ║
║    # Generar con LLM (Claude):                                              ║
║    python melody_conditioned.py --engine llm                                 ║
║        --llm-provider claude --llm-mode fewshot                             ║
║        --llm-fewshot-bank ./midis/ --llm-fewshot-n 3                        ║
║        --chords "Dm:2 C:2 Bb:2 A7:2" --key Dm --bars 8                    ║
║                                                                              ║
║    # Generar con LLM (OpenAI):                                              ║
║    python melody_conditioned.py --engine llm                                 ║
║        --llm-provider openai --llm-mode direct                              ║
║        --chords "C:4 Am:4 F:4 G:4" --key C --bars 8 --profile serene       ║
║                                                                              ║
║  OPCIONES PRINCIPALES:                                                       ║
║    --engine E          Motor: chord_guided | llm                            ║
║    --key KEY           Tonalidad: C, Am, F#, Bb… (default: C)              ║
║    --mode MODE         Modo de escala (default: auto)                       ║
║    --bars N            Compases (default: 16)                               ║
║    --tempo BPM         Tempo (default: 120)                                 ║
║    --time-sig S        Compás: 4/4, 3/4, 6/8… (default: 4/4)             ║
║    --profile P         Estado emocional (default: serene)                   ║
║    --contour C         Contorno: arch|ascending|descending|wave|…          ║
║    --tension-curve V   Curva de tensión: "0.2,0.5,0.8,0.5,…"             ║
║    --range LOW HIGH    Rango MIDI (default: 60 84)                          ║
║    --candidates N      Generar N candidatos, exportar el mejor (default: 3) ║
║    --chords TEXT       Progresión en texto: "Am:2 G:2 F:2 E7:2"           ║
║    --chords-json FILE  Progresión desde .chords.json                        ║
║    --chords-midi FILE  Progresión desde MIDI (extrae track de armonía)      ║
║    --llm-provider P    Proveedor LLM: claude | openai                       ║
║    --llm-mode M        Modo LLM: direct | fewshot                           ║
║    --llm-fewshot-bank DIR  Corpus para ejemplos few-shot                    ║
║    --llm-fewshot-n N   Número de ejemplos en el prompt (default: 3)         ║
║    --llm-model M       Modelo concreto (default: claude-sonnet-4-20250514 / ║
║                        gpt-4o)                                              ║
║    --output FILE       Fichero de salida (default: melody_out.mid)          ║
║    --include-chords    Incluir track de acordes en el MIDI de salida        ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║    --verbose           Informe detallado                                    ║
║    --dry-run           Mostrar parámetros sin generar                       ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:    mido, numpy                                                   ║
║    LLM Claude: anthropic                                                     ║
║    LLM OpenAI: openai                                                        ║
║    Opcional:   music21 (detección de acordes mejorada desde MIDI)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
import re
import time
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[WARN] mido no disponible: pip install mido")


try:
    import anthropic as anthropic_sdk
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    import openai as openai_sdk
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from music21 import harmony, stream, midi as m21midi
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
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
    "whole_tone":        [0, 2, 4, 6, 8, 10],
    "pentatonic_major":  [0, 2, 4, 7, 9],
    "pentatonic_minor":  [0, 3, 5, 7, 10],
    "blues":             [0, 3, 5, 6, 7, 10],
    "chromatic":         list(range(12)),
}

# Intervalos de acorde por calidad
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
    "11":    [0, 4, 7, 10, 14, 17],
    "13":    [0, 4, 7, 10, 14, 21],
}

# Tensiones disponibles por calidad de acorde
CHORD_TENSIONS: Dict[str, List[int]] = {
    "":      [14, 9, 11],          # 9ª, 6ª, 7ªM
    "m":     [14, 10, 17],         # 9ª, 7ªm, 11ª
    "7":     [14, 17, 21, 1],      # 9ª, 11ª, 13ª, b9ª
    "M7":    [14, 9, 18],          # 9ª, 6ª, #11ª
    "m7":    [14, 17, 9],
    "dim":   [9, 15],
    "dim7":  [14, 2],
    "hdim7": [14, 17],
    "aug":   [14, 10],
    "sus4":  [14, 10],
    "sus2":  [17, 10],
}

# Pesos de notas del acorde por función armónica
# "CT" = chord tone, "T" = tension, "PT" = passing/scale tone, "NT" = non-chord
HARMONIC_FUNCTION_WEIGHTS = {
    "T":  {"CT": 5.0, "T": 2.0, "PT": 2.5, "NT": 0.3},   # Tónica: estable
    "S":  {"CT": 4.0, "T": 3.0, "PT": 3.0, "NT": 0.5},   # Subdominante: movimiento
    "D":  {"CT": 3.5, "T": 4.0, "PT": 2.5, "NT": 0.8},   # Dominante: tensión
    "?":  {"CT": 4.0, "T": 2.5, "PT": 2.5, "NT": 0.5},   # Desconocida
}

# Estado emocional → pesos de tipo de nota por acorde
# Modifica los pesos base de HARMONIC_FUNCTION_WEIGHTS
EMOTIONAL_CHORD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "heroic":      {"CT": 1.3, "T": 0.8, "PT": 0.9, "NT": 0.4},
    "melancholic": {"CT": 1.0, "T": 1.5, "PT": 1.3, "NT": 0.6},
    "playful":     {"CT": 1.2, "T": 1.2, "PT": 1.4, "NT": 0.5},
    "tense":       {"CT": 0.8, "T": 2.0, "PT": 1.2, "NT": 1.0},
    "serene":      {"CT": 1.5, "T": 0.8, "PT": 1.0, "NT": 0.2},
    "mysterious":  {"CT": 0.7, "T": 1.8, "PT": 1.0, "NT": 1.2},
    "triumphant":  {"CT": 1.4, "T": 0.9, "PT": 0.8, "NT": 0.3},
    "custom":      {"CT": 1.0, "T": 1.0, "PT": 1.0, "NT": 0.5},
}

# Pesos de intervalos por perfil emocional
INTERVAL_WEIGHTS: Dict[str, Dict[int, float]] = {
    "heroic":      {0: 0.5, 2: 2.0, 3: 1.5, 4: 2.0, 5: 2.5, 7: 3.0, 12: 2.5,
                    -2: 2.0, -3: 1.0, -5: 1.5, -7: 1.5},
    "melancholic": {0: 0.3, 1: 2.5, 2: 3.0, 3: 2.0, -1: 3.0, -2: 3.5,
                    -3: 2.0, -5: 1.5, 5: 1.0, 7: 0.5},
    "playful":     {0: 0.5, 1: 1.5, 2: 2.5, 3: 2.0, 4: 2.0, 5: 1.5,
                    -1: 1.5, -2: 2.5, -3: 2.0, -4: 1.5, 7: 1.0, -7: 1.0},
    "tense":       {0: 0.5, 1: 3.0, 2: 2.0, 6: 2.0, -1: 3.0, -2: 2.0,
                    -6: 2.0, 11: 1.5, -11: 1.5, 3: 1.0, -3: 1.0},
    "serene":      {0: 1.0, 2: 3.0, 3: 2.0, 4: 2.5, 5: 2.0, -2: 3.0,
                    -3: 2.0, -4: 2.0, -5: 1.5, 7: 1.0, -7: 1.0},
    "mysterious":  {0: 1.0, 1: 2.0, 6: 2.5, -1: 2.0, -6: 2.5, 3: 1.5,
                    -3: 1.5, 8: 1.5, -8: 1.5, 2: 1.0},
    "triumphant":  {0: 0.3, 2: 2.0, 4: 2.5, 5: 2.0, 7: 3.5, 12: 2.0,
                    -2: 1.5, -3: 1.5, -5: 1.5, 3: 2.0},
    "custom":      {0: 1.0, 2: 2.5, -2: 2.5, 3: 1.5, -3: 1.5, 5: 1.5,
                    -5: 1.5, 7: 1.5, -7: 1.0},
}

# Contorno melódico inferido desde el estado emocional
PROFILE_CONTOUR: Dict[str, str] = {
    "heroic":      "arch",
    "melancholic": "descending",
    "playful":     "wave",
    "tense":       "ascending",
    "serene":      "arch",
    "mysterious":  "erratic",
    "triumphant":  "arch",
    "custom":      "arch",
}

# Velocidades MIDI por perfil
PROFILE_VELOCITY: Dict[str, Tuple[int, int]] = {
    "heroic":      (70, 110),
    "melancholic": (40, 75),
    "playful":     (60, 95),
    "tense":       (55, 100),
    "serene":      (40, 75),
    "mysterious":  (35, 70),
    "triumphant":  (75, 115),
    "custom":      (50, 90),
}

# Modo sugerido por perfil
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

# Distribuciones rítmicas
RHYTHM_DISTS: Dict[str, Dict[float, float]] = {
    "flowing":    {0.5: 3.0, 1.0: 4.0, 1.5: 1.5, 2.0: 1.5},
    "march":      {1.0: 5.0, 0.5: 2.0, 2.0: 1.5, 0.25: 1.0},
    "syncopated": {0.5: 3.5, 0.75: 3.0, 1.0: 2.5, 1.5: 2.0, 0.25: 2.0},
    "baroque":    {0.25: 2.0, 0.5: 3.0, 1.0: 3.0, 1.5: 1.5, 2.0: 1.0},
    "jazz":       {0.5: 3.0, 0.75: 3.5, 1.0: 2.5, 1.5: 2.0, 0.25: 1.5},
    "sparse":     {2.0: 3.5, 1.5: 2.0, 3.0: 2.0, 4.0: 1.5, 1.0: 2.0},
    "dense":      {0.25: 4.0, 0.5: 3.5, 0.75: 2.0, 1.0: 1.5},
}



# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MelodyNote:
    pitch:    int    # MIDI pitch 0-127; -1 = silencio
    duration: float  # en quarter notes
    velocity: int
    offset:   float  # en quarter notes desde inicio

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12 if self.pitch >= 0 else -1

    @property
    def name(self) -> str:
        if self.pitch < 0:
            return "REST"
        oct_ = self.pitch // 12 - 1
        return f"{NOTE_NAMES[self.pitch % 12]}{oct_}"


@dataclass
class ChordEvent:
    """Un acorde activo durante un intervalo de tiempo."""
    name:       str          # e.g. "Am7"
    root_pc:    int          # pitch class de la raíz
    quality:    str          # "", "m", "7", "M7", …
    chord_pcs:  List[int]    # pitch classes de las notas del acorde
    tension_pcs: List[int]   # pitch classes de las tensiones disponibles
    duration:   float        # en quarter notes
    offset:     float        # en quarter notes desde inicio
    harm_func:  str = "?"    # T, S, D, ?


@dataclass
class MelodyResult:
    notes:    List[MelodyNote]
    key:      str
    mode:     str
    bars:     int
    tempo:    int
    engine:   str
    profile:  str
    score:    float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "mode": self.mode, "bars": self.bars,
            "tempo": self.tempo, "engine": self.engine,
            "profile": self.profile, "score": round(self.score, 4),
            "total_notes": len(self.notes),
            "notes": [
                {"pitch": n.pitch, "name": n.name,
                 "duration": n.duration, "velocity": n.velocity,
                 "offset": round(n.offset, 4)}
                for n in self.notes
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES TONALES
# ══════════════════════════════════════════════════════════════════════════════

def parse_note_name(s: str) -> int:
    """'C', 'F#', 'Bb' → pitch class 0-11."""
    s = s.strip()
    if s in ENHARMONIC:
        return ENHARMONIC[s]
    base = s[0].upper()
    if base not in NOTE_NAMES:
        return 0
    pc = NOTE_NAMES.index(base)
    for ch in s[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
    return pc % 12


def parse_key(key_str: str) -> Tuple[int, str]:
    """'Am', 'C major', 'F# minor' → (root_pc, mode)."""
    key_str = key_str.strip()
    mode = "major"
    for m in ["harmonic_minor", "melodic_minor", "phrygian_dominant",
              "dorian", "phrygian", "lydian", "mixolydian", "locrian",
              "whole_tone", "pentatonic_major", "pentatonic_minor",
              "blues", "chromatic"]:
        if m.replace("_", " ") in key_str.lower() or m in key_str.lower():
            mode = m
            break
    else:
        if "minor" in key_str.lower() or "menor" in key_str.lower():
            mode = "minor"
        elif "major" in key_str.lower() or "mayor" in key_str.lower():
            mode = "major"
        else:
            # 'Am', 'Dm' → minor por sufijo m
            root_part = key_str.split()[0]
            clean = re.sub(r'[#b]', '', root_part)
            if (len(root_part) >= 2
                    and root_part[-1].lower() == "m"
                    and root_part[-2] not in ("#", "b")):
                mode = "minor"

    root_str = key_str.split()[0]
    root_str = re.sub(r'm(ajor|inor)?$', '', root_str, flags=re.IGNORECASE).strip()
    root_pc  = parse_note_name(root_str)
    return root_pc, mode


def get_scale_pitches(root_pc: int, mode: str,
                      low: int = 48, high: int = 96) -> List[int]:
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    result = []
    for oct_ in range(11):
        for iv in intervals:
            p = oct_ * 12 + root_pc + iv
            if low <= p <= high:
                result.append(p)
    return sorted(result)


def snap_to_scale(pitch: int, scale: List[int]) -> int:
    if not scale:
        return pitch
    return min(scale, key=lambda p: abs(p - pitch))


def weighted_choice(options: list, weights: list, rng: random.Random):
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    r = rng.random() * total
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += w
        if r <= acc:
            return opt
    return options[-1]


def build_contour_curve(shape: str, n: int,
                        custom: Optional[List[float]] = None) -> List[float]:
    t = np.linspace(0, 1, n)
    if shape == "arch":
        return list(np.sin(t * math.pi))
    elif shape == "ascending":
        return list(t)
    elif shape == "descending":
        return list(1.0 - t)
    elif shape == "wave":
        return list(0.5 + 0.5 * np.sin(t * 2 * math.pi))
    elif shape == "plateau":
        c = np.zeros(n)
        s, e = max(1, n // 5), n - max(1, n // 5)
        c[s:e] = 0.8
        c[:s]  = np.linspace(0, 0.8, s)
        c[e:]  = np.linspace(0.8, 0, n - e)
        return list(c)
    elif shape == "inverted":
        return list(1.0 - np.sin(t * math.pi))
    elif shape == "erratic":
        return list(np.random.default_rng(42).random(n))
    elif shape == "custom" and custom:
        return list(np.interp(np.linspace(0, 1, n),
                               np.linspace(0, 1, len(custom)), custom))
    return [0.5] * n


def resize_curve(curve: List[float], n: int) -> List[float]:
    if not curve:
        return [0.5] * n
    if len(curve) == n:
        return curve
    return list(np.interp(np.linspace(0, 1, n),
                           np.linspace(0, 1, len(curve)), curve))


def sample_duration(rhythm: str, beats_left: float,
                    rng: random.Random) -> float:
    dist = RHYTHM_DISTS.get(rhythm, RHYTHM_DISTS["flowing"])
    valid = {d: w for d, w in dist.items() if 0 < d <= beats_left}
    if not valid:
        return min(beats_left, 1.0)
    return weighted_choice(list(valid.keys()), list(valid.values()), rng)


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

def parse_chord_name(chord_str: str) -> Tuple[int, str]:
    """
    'Am7' → (9, 'm7')   'C' → (0, '')   'F#M7' → (6, 'M7')
    Devuelve (root_pc, quality).
    """
    chord_str = chord_str.strip()
    # Extraer raíz (nota + accidental)
    m = re.match(r'^([A-G][#b]?)(.*)', chord_str)
    if not m:
        return 0, ""
    root_str, quality = m.group(1), m.group(2)
    root_pc = parse_note_name(root_str)

    # Normalizar calidad
    quality = quality.strip("/").strip()
    # Eliminar inversiones ('/E', '/G')
    quality = re.sub(r'/[A-G][#b]?$', '', quality)

    # Mapear aliases comunes
    aliases = {
        "maj7": "M7", "Maj7": "M7", "maj": "", "MAJ": "",
        "min7": "m7", "min": "m", "MIN": "m",
        "°": "dim", "ø": "hdim7", "ø7": "hdim7",
        "+": "aug", "sus": "sus4",
    }
    for alias, canonical in aliases.items():
        if quality == alias:
            quality = canonical
            break

    if quality not in CHORD_INTERVALS:
        # Intentar prefijo más largo conocido
        for q in sorted(CHORD_INTERVALS.keys(), key=len, reverse=True):
            if quality.startswith(q):
                quality = q
                break
        else:
            quality = ""  # fallback a mayor

    return root_pc, quality


def build_chord_event(chord_name: str, duration: float,
                      offset: float, root_pc_key: int,
                      mode: str) -> ChordEvent:
    """Construye un ChordEvent completo desde nombre + contexto tonal."""
    root_pc, quality = parse_chord_name(chord_name)
    ivs = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])
    chord_pcs = [(root_pc + iv) % 12 for iv in ivs]

    tens_ivs = CHORD_TENSIONS.get(quality, [])
    tension_pcs = [(root_pc + iv) % 12 for iv in tens_ivs]

    # Función armónica aproximada
    scale_ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    degree = (root_pc - root_pc_key) % 12
    harm_func = "?"
    dom_degrees  = {scale_ivs[4] if len(scale_ivs) > 4 else 7}
    sub_degrees  = {scale_ivs[3] if len(scale_ivs) > 3 else 5,
                    scale_ivs[1] if len(scale_ivs) > 1 else 2}
    ton_degrees  = {0, scale_ivs[2] if len(scale_ivs) > 2 else 4,
                    scale_ivs[5] if len(scale_ivs) > 5 else 9}
    if degree in ton_degrees:
        harm_func = "T"
    elif degree in dom_degrees:
        harm_func = "D"
    elif degree in sub_degrees:
        harm_func = "S"

    return ChordEvent(
        name=chord_name, root_pc=root_pc, quality=quality,
        chord_pcs=chord_pcs, tension_pcs=tension_pcs,
        duration=duration, offset=offset, harm_func=harm_func,
    )


def parse_chords_text(text: str, root_pc_key: int,
                      mode: str) -> List[ChordEvent]:
    """'Am:2 G:2 F:2 E7:2' → lista de ChordEvent."""
    tokens = text.strip().split()
    events = []
    offset = 0.0
    for tok in tokens:
        if ":" in tok:
            name, beats = tok.split(":", 1)
            dur = float(beats)
        else:
            name, dur = tok, 4.0
        ev = build_chord_event(name, dur, offset, root_pc_key, mode)
        events.append(ev)
        offset += dur
    return events


def parse_chords_json(path: str, root_pc_key: int,
                      mode: str) -> List[ChordEvent]:
    """Carga un .chords.json del ecosistema."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chords_raw = []
    # Soportar varios formatos posibles del ecosistema
    if isinstance(data, list):
        chords_raw = data
    elif "chords" in data:
        chords_raw = data["chords"]
    elif "progression" in data:
        chords_raw = data["progression"]

    events = []
    offset = 0.0
    for item in chords_raw:
        if isinstance(item, str):
            name, dur = (item.split(":") + ["4.0"])[:2]
            dur = float(dur)
        elif isinstance(item, dict):
            name = item.get("name", item.get("chord", "C"))
            dur  = float(item.get("duration", item.get("beats", 4.0)))
        else:
            continue
        ev = build_chord_event(name.strip(), dur, offset, root_pc_key, mode)
        events.append(ev)
        offset += dur
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ACORDES DESDE MIDI (--chords-midi)
# ══════════════════════════════════════════════════════════════════════════════

def _chord_pcs_to_name(pcs: List[int]) -> Tuple[str, int, str]:
    """
    Template matching: dado un conjunto de pitch classes, devuelve
    (chord_name, root_pc, quality) del acorde más probable.
    Prueba todas las raíces posibles y todas las calidades conocidas.
    """
    if not pcs:
        return "C", 0, ""

    pcs_set = set(pcs)
    best_name, best_root, best_quality = "C", 0, ""
    best_score = -1

    for root_pc in range(12):
        for quality, ivs in CHORD_INTERVALS.items():
            template = {(root_pc + iv) % 12 for iv in ivs}
            # Score: notas coincidentes - notas extra
            matches = len(pcs_set & template)
            extras  = len(pcs_set - template)
            missing = len(template - pcs_set)
            score   = matches * 2 - extras * 0.5 - missing * 1.0
            if score > best_score:
                best_score   = score
                best_root    = root_pc
                best_quality = quality
                root_name    = NOTE_NAMES[root_pc]
                best_name    = root_name + quality

    return best_name, best_root, best_quality


def extract_chords_from_midi(midi_path: str, root_pc_key: int,
                              mode: str,
                              harmony_track_idx: int = 2,
                              harmony_channel: int = 1,
                              min_window_beats: float = 0.5
                              ) -> List[ChordEvent]:
    """
    Extrae acordes del track de armonía de un MIDI.

    Estrategia:
      1. Lee todas las notas del track de armonía (track_idx o canal).
      2. Agrupa las notas simultáneas en ventanas temporales.
      3. Para cada ventana, aplica template matching para identificar el acorde.
      4. Fusiona ventanas consecutivas con el mismo acorde.
      5. Si music21 está disponible, lo usa como fallback para acordes ambiguos.

    Parámetros:
      harmony_track_idx — índice del track de armonía (default: 2)
      harmony_channel   — canal MIDI del track de armonía (default: 1)
      min_window_beats  — ventana mínima de agrupación en quarter notes
    """
    if not MIDO_OK:
        raise RuntimeError("mido requerido para leer MIDI: pip install mido")

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480

    # Seleccionar track a analizar
    # Primero intentar por índice, luego por canal
    target_track = None
    if harmony_track_idx < len(mid.tracks):
        target_track = mid.tracks[harmony_track_idx]
    else:
        # Fallback: buscar el track que usa el canal indicado
        for tr in mid.tracks:
            for msg in tr:
                if hasattr(msg, "channel") and msg.channel == harmony_channel:
                    target_track = tr
                    break
            if target_track:
                break

    if target_track is None:
        raise ValueError(
            f"No se encontró track de armonía (idx={harmony_track_idx}, "
            f"canal={harmony_channel}) en {midi_path}"
        )

    # Extraer notas con tiempo absoluto en quarter notes
    notes: List[Tuple[float, float, int]] = []  # (onset_q, offset_q, pitch)
    current_tick = 0
    active: Dict[int, float] = {}  # pitch → onset_tick

    for msg in target_track:
        current_tick += msg.time
        onset_q = current_tick / tpb

        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_tick
        elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                on_tick = active.pop(msg.note)
                off_q   = current_tick / tpb
                on_q    = on_tick / tpb
                notes.append((on_q, off_q, msg.note))

    if not notes:
        # Si no hay notas, devolver tónica como placeholder
        dur_total = sum(
            msg.time for msg in target_track
            if not msg.is_meta
        ) / tpb
        dur_total = max(4.0, dur_total)
        root_name = NOTE_NAMES[root_pc_key]
        return [build_chord_event(root_name, dur_total, 0.0,
                                  root_pc_key, mode)]

    # Cuantizar a ventanas de min_window_beats
    total_duration = max(off for _, off, _ in notes)
    n_windows = max(1, int(math.ceil(total_duration / min_window_beats)))
    window_pcs: List[List[int]] = [[] for _ in range(n_windows)]

    for on_q, off_q, pitch in notes:
        pc = pitch % 12
        # Asignar la nota a todas las ventanas que abarca
        win_start = int(on_q / min_window_beats)
        win_end   = max(win_start + 1,
                        int(math.ceil(off_q / min_window_beats)))
        for w in range(win_start, min(win_end, n_windows)):
            if pc not in window_pcs[w]:
                window_pcs[w].append(pc)

    # Identificar acorde de cada ventana
    window_chords: List[Tuple[str, int, str]] = []
    for pcs in window_pcs:
        if pcs:
            window_chords.append(_chord_pcs_to_name(pcs))
        else:
            window_chords.append(window_chords[-1] if window_chords
                                  else ("C", 0, ""))

    # Fusionar ventanas consecutivas con el mismo acorde
    events: List[ChordEvent] = []
    if not window_chords:
        return events

    current_name, current_root, current_quality = window_chords[0]
    current_start  = 0.0
    current_length = min_window_beats

    for i in range(1, len(window_chords)):
        name, root, quality = window_chords[i]
        if name == current_name:
            current_length += min_window_beats
        else:
            ev = build_chord_event(
                current_name, current_length, current_start,
                root_pc_key, mode
            )
            events.append(ev)
            current_name    = name
            current_root    = root
            current_quality = quality
            current_start  += current_length
            current_length  = min_window_beats

    # Último segmento
    ev = build_chord_event(
        current_name, current_length, current_start,
        root_pc_key, mode
    )
    events.append(ev)

    # Opcional: refinar con music21 los acordes con score bajo
    if MUSIC21_OK:
        events = _refine_chords_music21(midi_path, events,
                                         harmony_channel, root_pc_key, mode)

    return events


def _refine_chords_music21(midi_path: str, events: List[ChordEvent],
                            channel: int, root_pc_key: int,
                            mode: str) -> List[ChordEvent]:
    """
    Usa music21.harmony.chordify para refinar acordes ambiguos.
    Solo sobreescribe eventos cuyo nombre no encaja bien.
    """
    try:
        score = m21midi.MidiFile()
        score.open(midi_path)
        score.read()
        score.close()
        s = m21midi.translate.midiFileToStream(score)
        chordified = s.chordify()

        m21_chords = []
        for c in chordified.flatten().getElementsByClass("Chord"):
            offset_q = float(c.offset)
            duration_q = float(c.duration.quarterLength)
            root = c.root()
            quality = c.commonName
            if root:
                m21_chords.append((offset_q, duration_q,
                                    root.name, quality))

        # Mapear acordes de music21 a los eventos existentes
        for ev in events:
            for off, dur, root_name, qual in m21_chords:
                if abs(off - ev.offset) < 0.25:
                    root_pc = parse_note_name(root_name)
                    # Normalizar quality de music21
                    q_map = {"minor triad": "m", "major triad": "",
                              "dominant seventh chord": "7",
                              "minor seventh chord": "m7",
                              "major seventh chord": "M7",
                              "diminished triad": "dim",
                              "diminished seventh chord": "dim7",
                              "half-diminished seventh chord": "hdim7",
                              "augmented triad": "aug"}
                    q = q_map.get(qual, "")
                    new_name = NOTE_NAMES[root_pc] + q
                    new_ev = build_chord_event(new_name, ev.duration,
                                               ev.offset, root_pc_key, mode)
                    ev.name       = new_ev.name
                    ev.root_pc    = new_ev.root_pc
                    ev.quality    = new_ev.quality
                    ev.chord_pcs  = new_ev.chord_pcs
                    ev.tension_pcs = new_ev.tension_pcs
                    ev.harm_func  = new_ev.harm_func
                    break
    except Exception:
        pass  # Si music21 falla, devolvemos los eventos sin refinar

    return events


def chord_events_to_text(events: List[ChordEvent]) -> str:
    """Serializa ChordEvents a formato texto 'Am:2 G:2 ...' ."""
    return " ".join(f"{e.name}:{e.duration}" for e in events)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLUCIÓN DE ACORDES EN EL TIEMPO
# ══════════════════════════════════════════════════════════════════════════════

def get_chord_at(events: List[ChordEvent], offset: float) -> ChordEvent:
    """Devuelve el ChordEvent activo en un offset dado."""
    for ev in reversed(events):
        if ev.offset <= offset:
            return ev
    return events[0] if events else ChordEvent(
        "C", 0, "", [0, 4, 7], [10, 14], 4.0, 0.0, "T")


def chord_note_type(pitch: int, chord: ChordEvent,
                    scale_pitches: List[int]) -> str:
    """
    Clasifica un pitch respecto al acorde activo:
    'CT' (chord tone), 'T' (tension), 'PT' (passing/scale), 'NT' (non-chord).
    """
    pc = pitch % 12
    if pc in chord.chord_pcs:
        return "CT"
    if pc in chord.tension_pcs:
        return "T"
    if pitch in scale_pitches:
        return "PT"
    return "NT"


def get_note_weights_for_chord(chord: ChordEvent, profile: str,
                                scale_pitches: List[int],
                                pitch_range: Tuple[int, int],
                                tension_val: float) -> Dict[int, float]:
    """
    Calcula el peso de cada pitch en el rango dado el acorde activo,
    el perfil emocional y la tensión actual.

    La tensión modulada permite más notas no-chord a medida que sube.
    """
    low, high = pitch_range
    func_weights = HARMONIC_FUNCTION_WEIGHTS.get(
        chord.harm_func, HARMONIC_FUNCTION_WEIGHTS["?"]
    )
    emo_mult = EMOTIONAL_CHORD_WEIGHTS.get(
        profile, EMOTIONAL_CHORD_WEIGHTS["custom"]
    )

    # Modulación por tensión: más tensión → más peso a T y NT
    tension_bonus = {
        "CT": 1.0 - tension_val * 0.3,
        "T":  1.0 + tension_val * 1.5,
        "PT": 1.0 + tension_val * 0.5,
        "NT": tension_val * 1.2,
    }

    weights: Dict[int, float] = {}
    for p in range(low, high + 1):
        ntype = chord_note_type(p, chord, scale_pitches)
        w = (func_weights[ntype]
             * emo_mult[ntype]
             * max(0.01, tension_bonus[ntype]))
        weights[p] = w

    return weights


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR: CHORD_GUIDED
# ══════════════════════════════════════════════════════════════════════════════

class ChordGuidedEngine:
    """
    Generación determinística condicionada a acordes.

    Combina cinco mecanismos de condicionamiento emocional:
      1. Perfil de intervalos por estado, sesgado por grado del acorde
      2. Curvas tensión/valencia/arousal por compás
      3. Tabla EMOTIONAL_CHORD_WEIGHTS: estado × acorde → pesos de notas
      4. Contorno melódico inferido/forzado desde el estado
      5. Markov de 2º orden condicionado por tensión en tiempo real
    """

    def __init__(self, profile: str, rhythm: str,
                 rng: random.Random):
        self.profile = profile
        self.rhythm  = rhythm
        self.rng     = rng
        # Tabla de Markov: (iv_prev, db_prev, db_cur) → [(iv_next, db_next)]
        self._markov: Dict[Tuple, List[Tuple]] = defaultdict(list)
        self._markov_trained = False

    # ── Markov interno ────────────────────────────────────────────────────────

    def _dur_bucket(self, d: float) -> int:
        return 0 if d <= 0.5 else (1 if d <= 1.5 else 2)

    def _build_markov(self):
        """Construye tabla de Markov sintética desde el perfil."""
        iw = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        ivs, iws = list(iw.keys()), list(iw.values())
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])

        prev_db = 1
        for _ in range(300):
            iv = weighted_choice(ivs, iws, self.rng)
            dur = weighted_choice(list(rd.keys()), list(rd.values()), self.rng)
            db  = self._dur_bucket(dur)
            niv = weighted_choice(ivs, iws, self.rng)
            ndb = self._dur_bucket(
                weighted_choice(list(rd.keys()), list(rd.values()), self.rng))
            self._markov[(iv, prev_db, db)].append((niv, ndb))
            prev_db = db
        self._markov_trained = True

    def _markov_next(self, prev_iv: int, prev_db: int,
                     cur_db: int, tension_val: float) -> int:
        """
        Siguiente intervalo desde Markov condicionado por tensión.
        A mayor tensión: sesgo hacia intervalos del perfil más extremos.
        """
        if not self._markov_trained:
            self._build_markov()

        # Candidatos del Markov
        key = (prev_iv, prev_db, cur_db)
        candidates = self._markov.get(key, [])
        if not candidates:
            for k, v in self._markov.items():
                if k[2] == cur_db:
                    candidates.extend(v)

        iw = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        ivs, iws = list(iw.keys()), list(iw.values())

        if candidates:
            base_iv, _ = self.rng.choice(candidates)
        else:
            base_iv = weighted_choice(ivs, iws, self.rng)

        # Sesgo por tensión: alta tensión → favorecer intervalos grandes
        if tension_val > 0.6:
            tension_ivs = [iv for iv in ivs if abs(iv) >= 3]
            tension_iws = [iw[iv] for iv in tension_ivs]
            if tension_ivs and self.rng.random() < tension_val * 0.6:
                base_iv = weighted_choice(tension_ivs, tension_iws, self.rng)

        return base_iv

    # ── Generación principal ──────────────────────────────────────────────────

    def generate(self,
                 chord_events: List[ChordEvent],
                 scale_pitches: List[int],
                 n_bars: int,
                 beats_per_bar: float,
                 contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar

        if not self._markov_trained:
            self._build_markov()

        # Inicio: tónica o 3ª del primer acorde, en zona media
        first_chord = chord_events[0] if chord_events else None
        if first_chord and first_chord.chord_pcs:
            mid_range = (low + high) // 2
            # Buscar chord tone más cercano al centro del rango
            start_candidates = [
                p for p in range(low, high + 1)
                if p % 12 in first_chord.chord_pcs
            ]
            if start_candidates:
                current_pitch = min(start_candidates,
                                    key=lambda p: abs(p - mid_range))
            else:
                current_pitch = snap_to_scale(mid_range, scale_pitches)
        else:
            current_pitch = snap_to_scale((low + high) // 2, scale_pitches)

        notes: List[MelodyNote] = []
        offset   = 0.0
        prev_iv  = 0
        prev_db  = 1

        bar_idx = 0
        while offset < total_beats - 0.01:
            bar_start  = bar_idx * beats_per_bar
            bar_offset = 0.0

            t_val = tension[min(bar_idx, len(tension) - 1)]
            c_val = contour[min(bar_idx, len(contour) - 1)]

            while bar_offset < beats_per_bar - 0.01:
                beats_left = min(beats_per_bar - bar_offset,
                                 total_beats - offset)
                dur = sample_duration(self.rhythm, beats_left, self.rng)
                db  = self._dur_bucket(dur)

                cur_offset = offset + bar_offset
                chord = get_chord_at(chord_events, cur_offset)

                # Pesos de notas: chord × emoción × tensión
                note_weights = get_note_weights_for_chord(
                    chord, self.profile, scale_pitches, pitch_range, t_val
                )

                # Intervalo Markov condicionado por tensión
                iv = self._markov_next(prev_iv, prev_db, db, t_val)

                # Ajuste de dirección por contorno
                cur_rel = (current_pitch - low) / max(high - low, 1)
                if cur_rel > c_val + 0.15 and iv > 0:
                    iv = -abs(iv)
                elif cur_rel < c_val - 0.15 and iv < 0:
                    iv = abs(iv)

                # Limitar salto según tensión
                max_leap = int(3 + t_val * 9)
                if abs(iv) > max_leap:
                    iv = int(math.copysign(max_leap, iv))

                # Candidato por intervalo
                candidate = max(low, min(high, current_pitch + iv))

                # Refinar usando pesos del acorde: buscar el pitch con mayor
                # peso en la vecindad del candidato (±3 semitonos)
                neighborhood = range(max(low, candidate - 3),
                                     min(high, candidate + 3) + 1)
                nbr_weights = [note_weights.get(p, 0.01) for p in neighborhood]
                if any(w > 0 for w in nbr_weights):
                    new_pitch = weighted_choice(list(neighborhood),
                                                nbr_weights, self.rng)
                else:
                    new_pitch = snap_to_scale(candidate, scale_pitches)

                # Velocidad dinámica
                climax = math.sin((cur_offset / total_beats) * math.pi) * t_val
                vel = int(vel_min + (vel_max - vel_min) * max(0.0, climax))
                vel = max(vel_min, min(vel_max, vel))

                notes.append(MelodyNote(
                    pitch=new_pitch, duration=dur,
                    velocity=vel, offset=round(cur_offset, 4)
                ))

                prev_iv = new_pitch - current_pitch
                prev_db = db
                current_pitch = new_pitch
                bar_offset += dur

            offset += beats_per_bar
            bar_idx += 1

        return notes



# ══════════════════════════════════════════════════════════════════════════════
#  BANCO DE EJEMPLOS FEW-SHOT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FewShotExample:
    harmony_text: str    # "Am:2 G:2 F:2 E7:2"
    melody_json:  str    # JSON con lista de notas


def build_fewshot_bank(corpus_dir: str, n_examples: int = 50,
                       seed: int = 42,
                       verbose: bool = False) -> List[FewShotExample]:
    """
    Carga n_examples MIDIs del corpus y los serializa como pares
    (armonía_texto, melodía_json) para uso como ejemplos few-shot.

    Se seleccionan aleatoriamente del corpus para variedad estilística.
    """
    if not MIDO_OK:
        return []

    rng = random.Random(seed)
    path  = Path(corpus_dir)
    files = list(path.rglob("*.mid")) + list(path.rglob("*.midi"))
    if not files:
        return []

    rng.shuffle(files)
    files = files[:n_examples * 3]  # pool amplio para filtrar

    examples: List[FewShotExample] = []
    for f in files:
        if len(examples) >= n_examples:
            break
        try:
            ex = _midi_to_fewshot(str(f))
            if ex:
                examples.append(ex)
        except Exception:
            continue

    if verbose:
        print(f"  [fewshot] banco={len(examples)} ejemplos de {corpus_dir}")

    return examples


def _midi_to_fewshot(midi_path: str) -> Optional[FewShotExample]:
    """Convierte un MIDI del corpus a FewShotExample."""
    mid = mido.MidiFile(midi_path)
    if len(mid.tracks) < 3:
        return None

    tpb = mid.ticks_per_beat or 480
    root_pc_key = 0  # sin contexto tonal, usamos C como placeholder

    # Extraer acordes desde track 2
    try:
        chord_events = extract_chords_from_midi(
            midi_path, root_pc_key, "major",
            harmony_track_idx=2, harmony_channel=1,
            min_window_beats=1.0
        )
    except Exception:
        return None

    if not chord_events:
        return None

    # Limitar a 8 compases máximo para no inflar el prompt
    max_beats = 32.0
    chord_events = [ev for ev in chord_events if ev.offset < max_beats]
    if not chord_events:
        return None

    harmony_text = chord_events_to_text(chord_events)

    # Extraer melodía desde track 1
    notes: List[Tuple[float, float, int]] = []
    current_tick = 0
    active: Dict[int, int] = {}
    for msg in mid.tracks[1]:
        current_tick += msg.time
        if hasattr(msg, "channel") and msg.channel != 0:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_tick
        elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                on_tick = active.pop(msg.note)
                on_q  = on_tick / tpb
                dur_q = (current_tick - on_tick) / tpb
                if on_q < max_beats:
                    notes.append((on_q, dur_q, msg.note))

    notes.sort(key=lambda n: n[0])
    if not notes:
        return None

    melody_notes = [
        {"pitch": pitch, "duration": round(dur_q, 3), "offset": round(on_q, 3)}
        for on_q, dur_q, pitch in notes[:64]  # máx 64 notas
    ]
    melody_json = json.dumps(melody_notes, separators=(",", ":"))

    return FewShotExample(harmony_text=harmony_text, melody_json=melody_json)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR: LLM
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Eres un compositor experto en teoría musical.
Tu tarea es generar melodías que armonicen con una progresión de acordes dada.
Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto adicional,
sin comillas de código, sin explicaciones.
El formato exacto es:
{"notes":[{"pitch":<int 0-127>,"duration":<float quarter notes>,"offset":<float>},...]}
Las notas deben:
- Usar chord tones (1ª, 3ª, 5ª) del acorde activo en los tiempos fuertes
- Usar notas de paso y tensiones en los tiempos débiles
- Seguir el perfil emocional y el contorno indicados
- Estar dentro del rango MIDI indicado
- Tener una duración total aproximada de bars × beats_per_bar quarter notes"""


def _build_direct_prompt(chord_events: List[ChordEvent],
                          key: str, mode: str, bars: int,
                          beats_per_bar: float, tempo: int,
                          profile: str, contour: str,
                          pitch_range: Tuple[int, int]) -> str:
    """Prompt para modo 'direct': descripción simbólica completa."""
    low, high = pitch_range
    total_beats = bars * beats_per_bar

    chord_lines = []
    for ev in chord_events:
        notes_in_chord = [NOTE_NAMES[(ev.root_pc + iv) % 12]
                          for iv in CHORD_INTERVALS.get(ev.quality, [0, 4, 7])]
        chord_lines.append(
            f"  offset={ev.offset:.1f}  duración={ev.duration:.1f}beats  "
            f"acorde={ev.name}  función={ev.harm_func}  "
            f"notas=[{', '.join(notes_in_chord)}]"
        )

    return (
        f"Genera una melodía con estos parámetros:\n"
        f"  Tonalidad: {key} {mode}\n"
        f"  Compases: {bars}  Tempo: {tempo}BPM  "
        f"Compás: {beats_per_bar}/4\n"
        f"  Rango MIDI: {low}-{high}\n"
        f"  Perfil emocional: {profile}\n"
        f"  Contorno melódico: {contour}\n"
        f"  Duración total: {total_beats} quarter notes\n\n"
        f"Progresión de acordes:\n"
        + "\n".join(chord_lines)
        + "\n\nResponde SOLO con el JSON de notas."
    )


def _build_fewshot_prompt(chord_events: List[ChordEvent],
                           examples: List[FewShotExample],
                           key: str, mode: str, bars: int,
                           beats_per_bar: float, tempo: int,
                           profile: str, contour: str,
                           pitch_range: Tuple[int, int],
                           n_shots: int = 3) -> str:
    """Prompt para modo 'fewshot': incluye ejemplos del corpus."""
    low, high = pitch_range
    total_beats = bars * beats_per_bar

    # Seleccionar n_shots ejemplos del banco
    selected = examples[:n_shots] if len(examples) >= n_shots else examples

    prompt_parts = [
        f"A continuación hay {len(selected)} ejemplos de pares "
        f"armonía→melodía del corpus de referencia:\n"
    ]

    for i, ex in enumerate(selected, 1):
        prompt_parts.append(
            f"=== EJEMPLO {i} ===\n"
            f"ARMONÍA: {ex.harmony_text}\n"
            f"MELODÍA: {ex.melody_json}\n"
        )

    harmony_text = chord_events_to_text(chord_events)
    prompt_parts.append(
        f"\nAhora genera una melodía en el mismo estilo para:\n"
        f"  Tonalidad: {key} {mode}  |  Compases: {bars}  "
        f"|  Tempo: {tempo}BPM  |  Compás: {beats_per_bar}/4\n"
        f"  Rango MIDI: {low}-{high}  |  Perfil: {profile}  "
        f"|  Contorno: {contour}\n"
        f"  Duración total: {total_beats} quarter notes\n\n"
        f"ARMONÍA: {harmony_text}\n"
        f"MELODÍA: (responde SOLO con el JSON de notas)"
    )

    return "\n".join(prompt_parts)


def _parse_llm_response(response_text: str) -> Optional[List[MelodyNote]]:
    """
    Parsea la respuesta JSON del LLM → lista de MelodyNote.
    Tolerante a variaciones de formato menores.
    """
    text = response_text.strip()
    # Eliminar posibles bloques de código
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")

    # Buscar el primer objeto JSON válido
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    raw_notes = data.get("notes", [])
    if not isinstance(raw_notes, list):
        return None

    notes = []
    for item in raw_notes:
        try:
            pitch    = int(item["pitch"])
            duration = float(item["duration"])
            offset   = float(item["offset"])
            if not (0 <= pitch <= 127 and duration > 0 and offset >= 0):
                continue
            notes.append(MelodyNote(
                pitch=pitch, duration=duration,
                velocity=80, offset=round(offset, 4)
            ))
        except (KeyError, ValueError, TypeError):
            continue

    return notes if notes else None


class LLMEngine:
    """
    Generación condicionada mediante LLM (Claude o ChatGPT).

    Modos:
      direct  — prompt con descripción simbólica completa de la progresión
      fewshot — añade ejemplos reales del corpus como contexto few-shot

    La respuesta JSON del LLM se parsea y convierte a MelodyNote.
    Si el parsing falla, reintenta hasta max_retries veces con temperatura
    ligeramente aumentada.
    """

    def __init__(self, provider: str, mode: str,
                 llm_model: Optional[str],
                 fewshot_bank: Optional[List[FewShotExample]],
                 n_shots: int, profile: str,
                 rng: random.Random, max_retries: int = 3):
        self.provider      = provider.lower()
        self.mode          = mode.lower()
        self.llm_model     = llm_model
        self.fewshot_bank  = fewshot_bank or []
        self.n_shots       = n_shots
        self.profile       = profile
        self.rng           = rng
        self.max_retries   = max_retries

        # Validar disponibilidad
        if self.provider == "claude" and not ANTHROPIC_OK:
            raise RuntimeError(
                "anthropic SDK requerido: pip install anthropic")
        if self.provider == "openai" and not OPENAI_OK:
            raise RuntimeError(
                "openai SDK requerido: pip install openai")

        # Modelos por defecto
        self._default_models = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
        }

    def _get_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return self._default_models.get(self.provider, "gpt-4o")

    def _call_claude(self, prompt: str,
                     temperature: float) -> Optional[str]:
        """Llama a la API de Anthropic."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Variable de entorno ANTHROPIC_API_KEY no definida")

        client = anthropic_sdk.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self._get_model(),
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return msg.content[0].text if msg.content else None

    def _call_openai(self, prompt: str,
                     temperature: float) -> Optional[str]:
        """Llama a la API de OpenAI."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Variable de entorno OPENAI_API_KEY no definida")

        client = openai_sdk.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self._get_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2048,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content if resp.choices else None

    def _call_llm(self, prompt: str,
                  temperature: float = 0.8) -> Optional[str]:
        if self.provider == "claude":
            return self._call_claude(prompt, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, temperature)
        else:
            raise ValueError(f"Proveedor desconocido: {self.provider}")

    def generate(self,
                 chord_events: List[ChordEvent],
                 scale_pitches: List[int],
                 n_bars: int,
                 beats_per_bar: float,
                 contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int],
                 key: str = "C", mode: str = "major",
                 tempo: int = 120,
                 contour_shape: str = "arch") -> List[MelodyNote]:

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))

        # Construir prompt
        if self.mode == "fewshot" and self.fewshot_bank:
            prompt = _build_fewshot_prompt(
                chord_events, self.fewshot_bank,
                key, mode, n_bars, beats_per_bar, tempo,
                self.profile, contour_shape, pitch_range, self.n_shots
            )
        else:
            prompt = _build_direct_prompt(
                chord_events, key, mode, n_bars, beats_per_bar, tempo,
                self.profile, contour_shape, pitch_range
            )

        # Intentar generación con reintentos
        notes = None
        temperature = 0.8

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_llm(prompt, temperature)
                if response:
                    notes = _parse_llm_response(response)
                if notes:
                    break
                temperature = min(temperature + 0.1, 1.2)
            except Exception as e:
                print(f"  [llm] intento {attempt}/{self.max_retries} falló: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # backoff exponencial

        if not notes:
            print("[llm] No se pudo obtener una respuesta válida del LLM")
            return []

        # Post-procesado: aplicar perfil emocional + snap a escala
        total_beats = n_bars * beats_per_bar
        processed: List[MelodyNote] = []

        for n in sorted(notes, key=lambda x: x.offset):
            if n.offset >= total_beats:
                break

            # Snap a escala si la nota está muy fuera
            pc = n.pitch % 12
            scale_pcs = {p % 12 for p in scale_pitches}
            if pc not in scale_pcs:
                n = MelodyNote(
                    pitch=snap_to_scale(n.pitch, scale_pitches),
                    duration=n.duration,
                    velocity=n.velocity,
                    offset=n.offset,
                )

            # Ajustar al rango
            pitch = n.pitch
            while pitch < low and pitch + 12 <= high:
                pitch += 12
            while pitch > high and pitch - 12 >= low:
                pitch -= 12
            pitch = max(low, min(high, pitch))

            # Velocidad desde perfil + tensión
            bar_idx = int(n.offset / beats_per_bar)
            t_val   = tension[min(bar_idx, len(tension) - 1)]
            climax  = math.sin((n.offset / total_beats) * math.pi) * t_val
            vel     = int(vel_min + (vel_max - vel_min) * max(0.0, climax))
            vel     = max(vel_min, min(vel_max, vel))

            processed.append(MelodyNote(
                pitch=pitch, duration=n.duration,
                velocity=vel, offset=n.offset,
            ))

        return processed


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING DE MELODÍAS
# ══════════════════════════════════════════════════════════════════════════════

def score_melody(notes: List[MelodyNote],
                 chord_events: List[ChordEvent],
                 scale_pitches: List[int],
                 contour: List[float],
                 beats_per_bar: float,
                 profile: str) -> float:
    """
    Puntuación multicriterio [0, 1].

    Criterios:
      1. Consonancia con la escala (20%)
      2. Armonización con los acordes activos (25%)
      3. Suavidad de movimiento (20%)
      4. Conformidad con el contorno (15%)
      5. Rango melódico razonable (10%)
      6. Variedad rítmica (10%)
    """
    if not notes:
        return 0.0

    pitches = [n.pitch for n in notes]
    durs    = [n.duration for n in notes]
    scale_set = {p % 12 for p in scale_pitches}

    # 1. Consonancia con la escala
    in_scale = sum(1 for p in pitches if p % 12 in scale_set)
    consonance = in_scale / len(pitches)

    # 2. Armonización con acordes activos
    chord_scores = []
    for n in notes:
        chord = get_chord_at(chord_events, n.offset)
        ntype = chord_note_type(n.pitch, chord, scale_pitches)
        chord_scores.append({"CT": 1.0, "T": 0.7, "PT": 0.5, "NT": 0.1}[ntype])
    harmony_score = float(np.mean(chord_scores)) if chord_scores else 0.5

    # 3. Suavidad de movimiento
    intervals  = [abs(pitches[i+1] - pitches[i])
                   for i in range(len(pitches) - 1)]
    big_leaps  = sum(1 for iv in intervals if iv > 7)
    smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 4. Conformidad con el contorno
    low, high = min(pitches), max(pitches)
    rng_v = high - low or 1
    c_errors = []
    for n in notes:
        bar_idx = int(n.offset // beats_per_bar)
        c_val   = contour[min(bar_idx, len(contour) - 1)]
        rel     = (n.pitch - low) / rng_v
        c_errors.append(abs(rel - c_val))
    contour_score = max(0.0, 1.0 - float(np.mean(c_errors)))

    # 5. Rango melódico (óptimo: 7-17 semitonos)
    rng_st    = max(pitches) - min(pitches)
    range_sc  = min(1.0, max(0.0, 1 - abs(rng_st - 12) / 12))

    # 6. Variedad rítmica
    variety   = min(1.0, len(set(round(d, 2) for d in durs)) / 5)

    return (consonance    * 0.20 +
            harmony_score * 0.25 +
            smoothness    * 0.20 +
            contour_score * 0.15 +
            range_sc      * 0.10 +
            variety       * 0.10)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT MIDI
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_midi(notes: List[MelodyNote], chord_events: List[ChordEvent],
                  tempo_bpm: int, output_path: str,
                  ticks_per_beat: int = 480,
                  include_chords: bool = False,
                  source_midi_path: Optional[str] = None,
                  source_chord_track_idx: int = 2):
    """
    Exporta la melodía a MIDI.

    Si include_chords=True, añade un track de acordes de referencia:
      - Si source_midi_path apunta a un MIDI válido, copia el track de
        armonía original (índice source_chord_track_idx) tal cual,
        preservando todas las notas y duraciones exactas.
      - En caso contrario, genera el track desde los ChordEvent resueltos.
    """
    if not MIDO_OK:
        print("[ERROR] mido no disponible. No se puede exportar MIDI.")
        return

    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = int(60_000_000 / max(tempo_bpm, 1))

    def q_ticks(q: float) -> int:
        return int(q * ticks_per_beat)

    # ── Track 0: metadatos ────────────────────────────────────────────────
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # ── Track 1: melodía ──────────────────────────────────────────────────
    mel_track = mido.MidiTrack()
    mid.tracks.append(mel_track)
    mel_track.append(mido.MetaMessage("track_name", name="Melody", time=0))
    mel_track.append(mido.Message("program_change", channel=0,
                                   program=73, time=0))  # flauta

    events = []
    for n in notes:
        if n.pitch < 0:
            continue
        on  = q_ticks(n.offset)
        off = q_ticks(n.offset + n.duration)
        events.append((on,  "note_on",  n.pitch, n.velocity))
        events.append((off, "note_off", n.pitch, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
    cur = 0
    for abs_t, mtype, note, vel in events:
        delta = max(0, abs_t - cur)
        mel_track.append(mido.Message(mtype, channel=0, note=note,
                                       velocity=vel, time=delta))
        cur = abs_t

    # ── Track 2: acordes (opcional) ───────────────────────────────────────
    if include_chords:
        # Intentar copiar el track original verbatim
        copied = False
        if source_midi_path and os.path.exists(source_midi_path):
            try:
                src = mido.MidiFile(source_midi_path)
                if source_chord_track_idx < len(src.tracks):
                    src_track = src.tracks[source_chord_track_idx]
                    # Remap ticks si el tpb de origen difiere
                    src_tpb = src.ticks_per_beat or 480
                    ratio   = ticks_per_beat / src_tpb

                    acc_track = mido.MidiTrack()
                    mid.tracks.append(acc_track)
                    acc_track.append(mido.MetaMessage(
                        "track_name", name="Chords", time=0))

                    for msg in src_track:
                        if msg.is_meta:
                            # Copiar solo track_name y program_change meta;
                            # saltar set_tempo y time_signature (ya en track 0)
                            if msg.type == "track_name":
                                # ya añadido arriba
                                pass
                            # ignorar el resto de meta del track original
                        else:
                            new_time = int(round(msg.time * ratio))
                            acc_track.append(msg.copy(time=new_time))
                    copied = True
            except Exception as e:
                print(f"[WARN] No se pudo copiar track de acordes original: {e}")

        if not copied and chord_events:
            # Fallback: generar desde ChordEvent
            acc_track = mido.MidiTrack()
            mid.tracks.append(acc_track)
            acc_track.append(mido.MetaMessage("track_name",
                                               name="Chords", time=0))
            acc_track.append(mido.Message("program_change", channel=1,
                                           program=0, time=0))  # piano

            chord_evts = []
            for ev in chord_events:
                pitches_oct3 = [(ev.root_pc + 48)]
                for iv in CHORD_INTERVALS.get(ev.quality, [0, 4, 7]):
                    pitches_oct3.append(ev.root_pc + 48 + iv)

                on_t  = q_ticks(ev.offset)
                off_t = q_ticks(ev.offset + ev.duration)
                for p in pitches_oct3:
                    if 0 <= p <= 127:
                        chord_evts.append((on_t,  "note_on",  p, 60))
                        chord_evts.append((off_t, "note_off", p, 0))

            chord_evts.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
            cur = 0
            for abs_t, mtype, note, vel in chord_evts:
                delta = max(0, abs_t - cur)
                acc_track.append(mido.Message(mtype, channel=1, note=note,
                                               velocity=vel, time=delta))
                cur = abs_t

    mid.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class ConditionedMelodyGenerator:
    """
    Clase principal que coordina los tres motores de generación,
    el condicionamiento emocional y la resolución de acordes.
    """

    def __init__(self,
                 # Parámetros musicales
                 key: str = "C", mode: str = "auto",
                 bars: int = 16, tempo: int = 120,
                 time_sig: str = "4/4",
                 # Motor
                 engine: str = "chord_guided",
                 # Emocional
                 profile: str = "serene",
                 rhythm: str = "flowing",
                 contour: str = "auto",
                 custom_contour: Optional[List[float]] = None,
                 tension_curve: Optional[List[float]] = None,
                 # Rango
                 pitch_low: int = 60, pitch_high: int = 84,
                 # Acordes
                 chord_events: Optional[List[ChordEvent]] = None,
                 # LLM
                 llm_provider: str = "claude",
                 llm_mode: str = "direct",
                 llm_model_name: Optional[str] = None,
                 fewshot_bank: Optional[List[FewShotExample]] = None,
                 n_shots: int = 3,
                 # Control
                 seed: int = 42, verbose: bool = False):

        self.key      = key
        self.bars     = bars
        self.tempo    = tempo
        self.engine   = engine
        self.profile  = profile
        self.rhythm   = rhythm
        self.verbose  = verbose
        self.chord_events = chord_events or []

        # LLM params (guardados para pasarlos al motor)
        self._llm_provider   = llm_provider
        self._llm_mode       = llm_mode
        self._llm_model_name = llm_model_name
        self._fewshot_bank   = fewshot_bank
        self._n_shots        = n_shots

        # Compás
        num, den = time_sig.split("/")
        self.beats_per_bar = float(num)

        # RNG
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Tonalidad y modo
        self.root_pc, detected_mode = parse_key(key)
        self.mode = detected_mode if mode == "auto" else mode

        # Modo por defecto desde perfil si sigue siendo auto
        if self.mode == "auto":
            self.mode = PROFILE_TO_MODE.get(profile, "major")

        # Escala
        self.scale_pitches = get_scale_pitches(
            self.root_pc, self.mode, pitch_low - 12, pitch_high + 12)
        self.scale_pitches = [p for p in self.scale_pitches
                               if pitch_low <= p <= pitch_high]
        if not self.scale_pitches:
            self.scale_pitches = get_scale_pitches(
                self.root_pc, self.mode, pitch_low - 24, pitch_high + 24)

        self.pitch_range = (pitch_low, pitch_high)

        # Contorno: si es 'auto', inferir desde el perfil
        contour_shape = contour
        if contour_shape == "auto":
            contour_shape = PROFILE_CONTOUR.get(profile, "arch")
        self.contour_shape = contour_shape
        self.contour_curve = build_contour_curve(
            contour_shape, bars, custom_contour)
        self.contour_curve = resize_curve(self.contour_curve, bars)

        # Curva de tensión
        if tension_curve:
            self.tension_curve = resize_curve(tension_curve, bars)
        else:
            self.tension_curve = self._default_tension(profile, bars)

        if verbose:
            self._print_config()

    def _default_tension(self, profile: str, n: int) -> List[float]:
        t = np.linspace(0, 1, n)
        curves = {
            "heroic":      0.3 + 0.6 * np.sin(t * math.pi),
            "melancholic": 0.5 - 0.3 * t,
            "tense":       np.clip(0.4 + 0.5 * t, 0, 1),
            "playful":     0.4 + 0.2 * np.sin(t * 4 * math.pi),
            "serene":      0.2 + 0.15 * np.sin(t * 2 * math.pi),
            "mysterious":  0.3 + 0.4 * np.sin(t * math.pi),
            "triumphant":  np.clip(0.5 * t + 0.5 * np.sin(t * math.pi), 0, 1),
        }
        return list(curves.get(profile, np.full(n, 0.4)))

    def _print_config(self):
        scale_names = [NOTE_NAMES[(self.root_pc + iv) % 12]
                       for iv in SCALE_INTERVALS.get(self.mode, [])]
        print(f"┌─ MELODY CONDITIONED ────────────────────────────────")
        print(f"│  Motor     : {self.engine}")
        print(f"│  Tonalidad : {self.key} {self.mode}")
        print(f"│  Escala    : {' '.join(scale_names)}")
        print(f"│  Compases  : {self.bars}  Tempo: {self.tempo} BPM")
        print(f"│  Perfil    : {self.profile}  Contorno: {self.contour_shape}")
        print(f"│  Acordes   : {len(self.chord_events)} eventos")
        if self.chord_events:
            preview = chord_events_to_text(self.chord_events[:8])
            print(f"│  Progresión: {preview}")
        print(f"└─────────────────────────────────────────────────────")

    def generate_once(self, seed_offset: int = 0) -> MelodyResult:
        rng = random.Random(self.rng.randint(0, 2**31) + seed_offset)

        if self.engine == "chord_guided":
            eng = ChordGuidedEngine(self.profile, self.rhythm, rng)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve,
                self.pitch_range,
            )

        elif self.engine == "llm":
            eng = LLMEngine(
                provider=self._llm_provider,
                mode=self._llm_mode,
                llm_model=self._llm_model_name,
                fewshot_bank=self._fewshot_bank,
                n_shots=self._n_shots,
                profile=self.profile,
                rng=rng,
            )
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve,
                self.pitch_range,
                key=self.key, mode=self.mode, tempo=self.tempo,
                contour_shape=self.contour_shape,
            )

        else:
            raise ValueError(f"Motor desconocido: {self.engine}. "
                             f"Usa: chord_guided | llm")

        sc = score_melody(notes, self.chord_events, self.scale_pitches,
                          self.contour_curve, self.beats_per_bar, self.profile)

        return MelodyResult(
            notes=notes, key=self.key, mode=self.mode,
            bars=self.bars, tempo=self.tempo,
            engine=self.engine, profile=self.profile,
            score=sc,
            metadata={"contour": self.contour_shape,
                      "rhythm": self.rhythm,
                      "chord_progression": chord_events_to_text(
                          self.chord_events)},
        )

    def generate_candidates(self, n: int = 3) -> List[MelodyResult]:
        candidates = []
        for i in range(n):
            r = self.generate_once(seed_offset=i * 1337)
            candidates.append(r)
            if self.verbose:
                print(f"  [candidato {i+1}/{n}] "
                      f"score={r.score:.3f}  notas={len(r.notes)}")
        return sorted(candidates, key=lambda r: -r.score)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MELODY CONDITIONED — Generación condicionada a acordes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Motor ─────────────────────────────────────────────────────────────
    p.add_argument("--engine", default="chord_guided",
                   choices=["chord_guided", "llm"],
                   help="Motor de generación (default: chord_guided)")

    # ── Parámetros musicales ──────────────────────────────────────────────
    p.add_argument("--key",      default="C")
    p.add_argument("--mode",     default="auto",
                   choices=list(SCALE_INTERVALS.keys()) + ["auto"])
    p.add_argument("--bars",     type=int,   default=16)
    p.add_argument("--tempo",    type=int,   default=120)
    p.add_argument("--time-sig", default="4/4")

    # ── Perfil emocional ──────────────────────────────────────────────────
    p.add_argument("--profile", default="serene",
                   choices=list(PROFILE_VELOCITY.keys()))
    p.add_argument("--rhythm",  default="flowing",
                   choices=list(RHYTHM_DISTS.keys()))
    p.add_argument("--contour", default="auto",
                   choices=["auto", "arch", "ascending", "descending",
                            "wave", "plateau", "inverted", "erratic", "custom"])
    p.add_argument("--contour-values", type=str, default=None)
    p.add_argument("--tension-curve",  type=str, default=None,
                   help="Curva de tensión: '0.2,0.5,0.8,0.5'")

    # ── Rango ─────────────────────────────────────────────────────────────
    p.add_argument("--range", nargs=2, type=int, default=[60, 84],
                   metavar=("LOW", "HIGH"))

    # ── Fuentes de acordes ────────────────────────────────────────────────
    p.add_argument("--chords",      type=str, default=None,
                   help="Progresión en texto: 'Am:2 G:2 F:2 E7:2'")
    p.add_argument("--chords-json", type=str, default=None,
                   help="Progresión desde .chords.json")
    p.add_argument("--chords-midi", type=str, default=None,
                   help="Extraer acordes del track de armonía de un MIDI")
    p.add_argument("--chords-midi-track",   type=int, default=2,
                   help="Índice del track de armonía (default: 2)")
    p.add_argument("--chords-midi-channel", type=int, default=1,
                   help="Canal MIDI de armonía (default: 1)")
    p.add_argument("--chords-midi-window",  type=float, default=0.5,
                   help="Ventana mínima de agrupación en beats (default: 0.5)")

    # ── LLM ───────────────────────────────────────────────────────────────
    p.add_argument("--llm-provider", default="claude",
                   choices=["claude", "openai"])
    p.add_argument("--llm-mode",     default="direct",
                   choices=["direct", "fewshot"])
    p.add_argument("--llm-model",    type=str, default=None,
                   help="Modelo LLM concreto (default: claude-sonnet-4-20250514 / gpt-4o)")
    p.add_argument("--llm-fewshot-bank", type=str, default=None,
                   help="Corpus para ejemplos few-shot (carpeta de MIDIs)")
    p.add_argument("--llm-fewshot-n",    type=int, default=3,
                   help="Número de ejemplos few-shot en el prompt (default: 3)")

    # ── Exportación ───────────────────────────────────────────────────────
    p.add_argument("--output",         type=str, default="melody_conditioned.mid")
    p.add_argument("--include-chords", action="store_true",
                   help="Incluir track de acordes en el MIDI de salida")
    p.add_argument("--candidates",     type=int, default=3)

    # ── Control ───────────────────────────────────────────────────────────
    p.add_argument("--seed",    type=int,  default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolver tonalidad y modo ─────────────────────────────────────────
    root_pc, detected_mode = parse_key(args.key)
    mode = detected_mode if args.mode == "auto" else args.mode
    if mode == "auto":
        mode = PROFILE_TO_MODE.get(args.profile, "major")

    # ── Resolver acordes ──────────────────────────────────────────────────
    chord_events: List[ChordEvent] = []

    if args.chords_midi:
        if not os.path.exists(args.chords_midi):
            print(f"[ERROR] No se encuentra: {args.chords_midi}")
            sys.exit(1)
        print(f"[acordes] Extrayendo desde MIDI: {args.chords_midi}")
        chord_events = extract_chords_from_midi(
            midi_path         = args.chords_midi,
            root_pc_key       = root_pc,
            mode              = mode,
            harmony_track_idx = args.chords_midi_track,
            harmony_channel   = args.chords_midi_channel,
            min_window_beats  = args.chords_midi_window,
        )
        if args.verbose:
            print(f"  → {len(chord_events)} acordes extraídos")
            for ev in chord_events[:8]:
                print(f"    {ev.offset:.1f}  {ev.name}  ({ev.harm_func})")

    elif args.chords_json:
        if not os.path.exists(args.chords_json):
            print(f"[ERROR] No se encuentra: {args.chords_json}")
            sys.exit(1)
        chord_events = parse_chords_json(args.chords_json, root_pc, mode)

    elif args.chords:
        chord_events = parse_chords_text(args.chords, root_pc, mode)

    if not chord_events:
        # Fallback: progresión I-V-VI-IV en la tonalidad dada
        scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
        fallback_degrees = [0, 4, 5, 3] if len(scale) > 5 else [0, 4, 0, 4]
        print("[WARN] No se especificaron acordes. "
              "Usando progresión I-V-VI-IV por defecto.")
        text_parts = []
        for deg in fallback_degrees:
            r = NOTE_NAMES[(root_pc + scale[min(deg, len(scale)-1)]) % 12]
            text_parts.append(f"{r}:4")
        chord_events = parse_chords_text(
            " ".join(text_parts), root_pc, mode)

    # ── Curvas ────────────────────────────────────────────────────────────
    tension_curve = None
    if args.tension_curve:
        try:
            tension_curve = [float(v) for v in args.tension_curve.split(",")]
        except ValueError:
            print("[WARN] --tension-curve inválida. Usando curva por defecto.")

    custom_contour = None
    if args.contour == "custom" and args.contour_values:
        try:
            custom_contour = [float(v)
                               for v in args.contour_values.split(",")]
        except ValueError:
            print("[WARN] --contour-values inválido. Usando arch.")

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── PARÁMETROS RESUELTOS (dry-run) ─────────────────────")
        print(f"  engine   = {args.engine}")
        print(f"  key      = {args.key}  mode = {mode}")
        print(f"  bars     = {args.bars}  tempo = {args.tempo}")
        print(f"  profile  = {args.profile}  rhythm = {args.rhythm}")
        print(f"  contour  = {args.contour}")
        print(f"  range    = {args.range[0]}-{args.range[1]}")
        print(f"  chords   = {chord_events_to_text(chord_events)}")
        if args.engine == "llm":
            print(f"  llm_provider = {args.llm_provider}")
            print(f"  llm_mode     = {args.llm_mode}")
            print(f"  llm_model    = {args.llm_model or '(default)'}")
        print("────────────────────────────────────────────────────────")
        sys.exit(0)

    # ── Banco few-shot ────────────────────────────────────────────────────
    fewshot_bank = None
    if args.engine == "llm" and args.llm_mode == "fewshot":
        bank_dir = args.llm_fewshot_bank or args.corpus
        if bank_dir and os.path.isdir(bank_dir):
            print(f"[llm] Construyendo banco few-shot desde: {bank_dir}")
            fewshot_bank = build_fewshot_bank(
                bank_dir, n_examples=max(50, args.llm_fewshot_n * 10),
                seed=args.seed, verbose=args.verbose)
        else:
            print("[WARN] --llm-fewshot-bank no especificado o no existe. "
                  "Cambiando a modo 'direct'.")

    # ── Instanciar generador ──────────────────────────────────────────────
    gen = ConditionedMelodyGenerator(
        key           = args.key,
        mode          = args.mode,
        bars          = args.bars,
        tempo         = args.tempo,
        time_sig      = args.time_sig,
        engine        = args.engine,
        profile       = args.profile,
        rhythm        = args.rhythm,
        contour       = args.contour,
        custom_contour= custom_contour,
        tension_curve = tension_curve,
        pitch_low     = args.range[0],
        pitch_high    = args.range[1],
        chord_events  = chord_events,
        llm_provider  = args.llm_provider,
        llm_mode      = args.llm_mode,
        llm_model_name= args.llm_model,
        fewshot_bank  = fewshot_bank,
        n_shots       = args.llm_fewshot_n,
        seed          = args.seed,
        verbose       = args.verbose,
    )

    # ── Generar candidatos ────────────────────────────────────────────────
    n_cand = 1 if args.engine == "llm" else args.candidates
    print(f"\nGenerando {n_cand} candidato(s) con motor '{args.engine}'…")
    candidates = gen.generate_candidates(n_cand)

    # ── Exportar ──────────────────────────────────────────────────────────
    base = args.output.replace(".mid", "")
    output_paths = []

    for i, res in enumerate(candidates):
        suffix = "" if i == 0 else f"_v{i+1}"
        out_p  = f"{base}{suffix}.mid"
        notes_to_midi(
            res.notes, chord_events, res.tempo, out_p,
            include_chords=args.include_chords,
            source_midi_path=args.chords_midi,
            source_chord_track_idx=args.chords_midi_track,
        )
        output_paths.append(out_p)
        tag = "★ MEJOR" if i == 0 else f"  v{i+1}"
        print(f"  {tag}  score={res.score:.3f}  "
              f"notas={len(res.notes)}  → {out_p}")

    # ── JSON del mejor resultado ──────────────────────────────────────────
    if candidates:
        best      = candidates[0]
        json_path = f"{base}.melody.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[json]  {json_path}")

    # ── Resumen ───────────────────────────────────────────────────────────
    if candidates:
        best = candidates[0]
        print(f"\n╔═ RESUMEN ═════════════════════════════════════════╗")
        print(f"║  Motor     : {best.engine}")
        print(f"║  Tonalidad : {best.key} {best.mode}")
        print(f"║  Compases  : {best.bars}  Tempo: {best.tempo} BPM")
        print(f"║  Perfil    : {best.profile}")
        print(f"║  Notas     : {len(best.notes)}")
        print(f"║  Score     : {best.score:.3f}")
        prog_preview = chord_events_to_text(chord_events[:6])
        if len(chord_events) > 6:
            prog_preview += " …"
        print(f"║  Progresión: {prog_preview}")
        print(f"╚═══════════════════════════════════════════════════╝\n")

        if output_paths:
            print(f"  Salida principal: {output_paths[0]}")


if __name__ == "__main__":
    main()
