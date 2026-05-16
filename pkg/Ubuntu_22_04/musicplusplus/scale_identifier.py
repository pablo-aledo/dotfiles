#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      SCALE IDENTIFIER  v1.0                                  ║
║         Identificación de escalas desde notas MIDI                           ║
║                                                                              ║
║  Portado desde music++ / scaleDictionary.h (Francesco Balena – Scale Omnibus)║
║                                                                              ║
║  Dado un conjunto de notas MIDI (o pitch-classes), identifica todas las      ║
║  escalas que contienen exactamente ese conjunto de clases de altura.         ║
║  También calcula una puntuación de cobertura parcial para escalas            ║
║  que contienen el conjunto como subconjunto.                                 ║
║                                                                              ║
║  USO:                                                                        ║
║    python scale_identifier.py 60 62 64 65 67 69 71                          ║
║    python scale_identifier.py notas.mid                                      ║
║    python scale_identifier.py notas.mid --top 5                              ║
║    python scale_identifier.py 60 62 63 67 70 --partial --top 10             ║
║    python scale_identifier.py notas.mid --root C --output resultado.json    ║
║    python scale_identifier.py notas.mid --category "Jazz Scales"            ║
║                                                                              ║
║  MODOS:                                                                      ║
║    exact    — coincidencia exacta de pitch-classes (default)                 ║
║    partial  — el input es subconjunto de la escala (--partial)               ║
║    superset — la escala es subconjunto del input (--superset)                ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    notas...          MIDI pitch values (0-127) o archivo .mid               ║
║    --root NOTE       Raíz de la escala detectada (default: auto)            ║
║    --top N           Mostrar los N mejores resultados (default: todos)       ║
║    --partial         Modo parcial: input ⊆ escala                           ║
║    --superset        Modo superset: escala ⊆ input                          ║
║    --category CAT    Filtrar por categoría de escala                         ║
║    --list-categories Listar categorías disponibles                           ║
║    --output FILE     Guardar resultado en JSON                               ║
║    --verbose         Mostrar pitch-classes del input y de cada escala        ║
║    --no-color        Sin colores ANSI                                        ║
║                                                                              ║
║  CATEGORÍAS DISPONIBLES:                                                     ║
║    Major and minor scales · Symmetrical scales · European Scales             ║
║    Modal Scales · Pentatonic Scales · Jazz Scales · Asian Scales             ║
║    Indian Scales · Miscellaneous scales                                      ║
║                                                                              ║
║  DEPENDENCIAS: mido (solo para lectura de .mid), sin otras dependencias      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  BASE DE DATOS DE ESCALAS
#  Portada directamente desde scaleDictionary.h (music++)
#  Fuente: Francesco Balena – The Scale Omnibus
# ─────────────────────────────────────────────────────────────────────────────

SCALE_DATABASE: list[dict] = [
    # ── Major and minor scales ────────────────────────────────────────────────
    {"category": "Major and minor scales", "name": "Ionian (Major)",       "pcs": [0,2,4,5,7,9,11]},
    {"category": "Major and minor scales", "name": "Dorian",               "pcs": [0,2,3,5,7,9,10]},
    {"category": "Major and minor scales", "name": "Phrygian",             "pcs": [0,1,3,5,7,8,10]},
    {"category": "Major and minor scales", "name": "Lydian",               "pcs": [0,2,4,6,7,9,11]},
    {"category": "Major and minor scales", "name": "Mixolydian",           "pcs": [0,2,4,5,7,9,10]},
    {"category": "Major and minor scales", "name": "Aeolian (Natural Minor)","pcs":[0,2,3,5,7,8,10]},
    {"category": "Major and minor scales", "name": "Locrian",              "pcs": [0,1,3,5,6,8,10]},
    {"category": "Major and minor scales", "name": "Melodic Minor",        "pcs": [0,2,3,5,7,9,11]},
    {"category": "Major and minor scales", "name": "Dorian b2",            "pcs": [0,1,3,5,7,9,10]},
    {"category": "Major and minor scales", "name": "Lydian Augmented",     "pcs": [0,2,4,6,8,9,11]},
    {"category": "Major and minor scales", "name": "Lydian Dominant",      "pcs": [0,2,4,6,7,9,10]},
    {"category": "Major and minor scales", "name": "Melodic Major",        "pcs": [0,2,4,5,7,8,10]},
    {"category": "Major and minor scales", "name": "Half Diminished",      "pcs": [0,2,3,5,6,8,10]},
    {"category": "Major and minor scales", "name": "Altered Dominant",     "pcs": [0,1,3,4,6,8,10]},
    {"category": "Major and minor scales", "name": "Harmonic Minor",       "pcs": [0,2,3,5,7,8,11]},
    {"category": "Major and minor scales", "name": "Locrian #6",           "pcs": [0,1,3,5,6,9,10]},
    {"category": "Major and minor scales", "name": "Ionian Augmented",     "pcs": [0,2,4,5,8,9,11]},
    {"category": "Major and minor scales", "name": "Romanian Minor",       "pcs": [0,2,3,6,7,9,10]},
    {"category": "Major and minor scales", "name": "Phrygian Dominant",    "pcs": [0,1,4,5,7,8,10]},
    {"category": "Major and minor scales", "name": "Lydian #2",            "pcs": [0,3,4,6,7,9,11]},
    {"category": "Major and minor scales", "name": "Ultralocrian",         "pcs": [0,1,3,4,6,8,9]},
    {"category": "Major and minor scales", "name": "Harmonic Major",       "pcs": [0,2,4,5,7,8,11]},
    # ── Symmetrical scales ────────────────────────────────────────────────────
    {"category": "Symmetrical scales", "name": "Whole-Tone",               "pcs": [0,2,4,6,8,10]},
    {"category": "Symmetrical scales", "name": "Augmented",                "pcs": [0,3,4,7,8,11]},
    {"category": "Symmetrical scales", "name": "Inverted Augmented",       "pcs": [0,1,4,5,8,9]},
    {"category": "Symmetrical scales", "name": "Diminished",               "pcs": [0,2,3,5,6,8,9,11]},
    {"category": "Symmetrical scales", "name": "Diminished Half-tone",     "pcs": [0,1,3,4,6,7,9,10]},
    {"category": "Symmetrical scales", "name": "Chromatic",                "pcs": [0,1,2,3,4,5,6,7,8,9,10,11]},
    {"category": "Symmetrical scales", "name": "Tritone",                  "pcs": [0,1,4,6,7,10]},
    {"category": "Symmetrical scales", "name": "Raga Neelangi",            "pcs": [0,2,3,6,8,9]},
    {"category": "Symmetrical scales", "name": "Messiaen 2nd Mode Truncated","pcs":[0,1,3,6,7,9]},
    {"category": "Symmetrical scales", "name": "Messiaen 3rd Mode",        "pcs": [0,2,3,4,6,7,8,10,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 4th Mode",        "pcs": [0,1,2,5,6,7,8,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 4th Mode Inverse","pcs": [0,3,4,5,6,9,10,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 5th Mode",        "pcs": [0,1,5,6,7,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 5th Mode Inverse","pcs": [0,4,5,6,10,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 6th Mode",        "pcs": [0,2,4,5,6,8,10,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 6th Mode Inverse","pcs": [0,1,2,4,6,7,8,10]},
    {"category": "Symmetrical scales", "name": "Messiaen 7th Mode",        "pcs": [0,1,2,3,5,6,7,8,9,11]},
    {"category": "Symmetrical scales", "name": "Messiaen 7th Mode Inverse","pcs": [0,2,3,4,5,6,8,9,10,11]},
    {"category": "Symmetrical scales", "name": "Genus Chromaticum",        "pcs": [0,1,3,4,5,7,8,9,11]},
    {"category": "Symmetrical scales", "name": "Two-semitone Tritone",     "pcs": [0,1,2,6,7,8]},
    {"category": "Symmetrical scales", "name": "Symmetrical Decatonic",    "pcs": [0,1,2,4,5,6,7,8,10,11]},
    {"category": "Symmetrical scales", "name": "Van Der Host",             "pcs": [0,1,3,5,6,7,9,11]},
    # ── European Scales ───────────────────────────────────────────────────────
    {"category": "European Scales", "name": "Adonai Malakh",               "pcs": [0,1,2,3,5,7,9,10]},
    {"category": "European Scales", "name": "Enigmatic (asc)",             "pcs": [0,1,4,6,8,10,11]},
    {"category": "European Scales", "name": "Enigmatic (desc)",            "pcs": [0,1,4,5,8,10,11]},
    {"category": "European Scales", "name": "Enigmatic Minor",             "pcs": [0,1,3,6,8,10,11]},
    {"category": "European Scales", "name": "Enigmatic Mixed",             "pcs": [0,1,4,5,6,8,10,11]},
    {"category": "European Scales", "name": "Flamenco",                    "pcs": [0,1,3,4,5,7,8,10]},
    {"category": "European Scales", "name": "Gypsy",                       "pcs": [0,2,3,6,7,8,10]},
    {"category": "European Scales", "name": "Gypsy Hexatonic",             "pcs": [0,1,4,5,7,8,9]},
    {"category": "European Scales", "name": "Gypsy Inverse",               "pcs": [0,1,4,5,7,9,11]},
    {"category": "European Scales", "name": "Gypsy Minor",                 "pcs": [0,2,3,6,7,8,11]},
    {"category": "European Scales", "name": "Hijaz Major",                 "pcs": [0,1,5,6,8,9,10]},
    {"category": "European Scales", "name": "Hungarian Major",             "pcs": [0,3,4,6,7,9,10]},
    {"category": "European Scales", "name": "Hungarian Minor b2",          "pcs": [0,1,2,3,6,7,8,11]},
    {"category": "European Scales", "name": "Istrian",                     "pcs": [0,1,3,4,6,7]},
    {"category": "European Scales", "name": "Neapolitan Major",            "pcs": [0,1,3,5,7,9,11]},
    {"category": "European Scales", "name": "Neapolitan Minor",            "pcs": [0,1,3,5,7,8,11]},
    {"category": "European Scales", "name": "Prometheus",                  "pcs": [0,2,4,6,9,10]},
    {"category": "European Scales", "name": "Prometheus Neapolitan",       "pcs": [0,1,4,6,9,10]},
    {"category": "European Scales", "name": "Romanian Major",              "pcs": [0,1,4,6,7,9,10]},
    {"category": "European Scales", "name": "Scottish Hexatonic",          "pcs": [0,2,4,5,7,9]},
    {"category": "European Scales", "name": "Shostakovich",                "pcs": [0,1,3,4,6,7,9,11]},
    {"category": "European Scales", "name": "Spanish Heptatonic",          "pcs": [0,3,4,5,6,8,10]},
    {"category": "European Scales", "name": "Spanish Octatonic",           "pcs": [0,1,3,4,5,6,8,10]},
    {"category": "European Scales", "name": "Double Harmonic",             "pcs": [0,1,4,5,7,8,11]},
    # ── Modal Scales ──────────────────────────────────────────────────────────
    {"category": "Modal Scales", "name": "Harmonic Major 2",               "pcs": [0,2,4,5,8,9,11]},
    {"category": "Modal Scales", "name": "Lydian #6",                      "pcs": [0,2,4,6,7,10,11]},
    {"category": "Modal Scales", "name": "Lydian Dominant b6",             "pcs": [0,2,4,6,7,8,10]},
    {"category": "Modal Scales", "name": "Lydian Augmented Dominant",      "pcs": [0,2,4,6,8,9,10]},
    {"category": "Modal Scales", "name": "Lydian Diminished",              "pcs": [0,2,3,6,7,9,11]},
    {"category": "Modal Scales", "name": "Mixolydian Augmented",           "pcs": [0,2,4,5,8,9,10]},
    {"category": "Modal Scales", "name": "Mixolydian b5",                  "pcs": [0,2,4,5,6,9,10]},
    {"category": "Modal Scales", "name": "Phrygian Dominant",              "pcs": [0,1,4,5,7,8,10]},
    {"category": "Modal Scales", "name": "Phrygian b4",                    "pcs": [0,1,3,4,7,8,10]},
    {"category": "Modal Scales", "name": "Dorian b2 b4",                   "pcs": [0,1,3,4,7,9,10]},
    {"category": "Modal Scales", "name": "Leading Whole-Tone",             "pcs": [0,2,4,6,8,10,11]},
    {"category": "Modal Scales", "name": "Major Locrian",                  "pcs": [0,2,4,5,6,8,10]},
    {"category": "Modal Scales", "name": "Minor Hexatonic",                "pcs": [0,2,3,5,7,10]},
    {"category": "Modal Scales", "name": "Lydian Hexatonic",               "pcs": [0,2,4,7,9,11]},
    {"category": "Modal Scales", "name": "Mixolydian Hexatonic",           "pcs": [0,2,5,7,9,10]},
    {"category": "Modal Scales", "name": "Phrygian Hexatonic",             "pcs": [0,3,5,7,8,10]},
    # ── Pentatonic Scales ─────────────────────────────────────────────────────
    {"category": "Pentatonic Scales", "name": "Major Pentatonic",          "pcs": [0,2,4,7,9]},
    {"category": "Pentatonic Scales", "name": "Minor Pentatonic",          "pcs": [0,3,5,7,10]},
    {"category": "Pentatonic Scales", "name": "Suspended Pentatonic",      "pcs": [0,2,5,7,10]},
    {"category": "Pentatonic Scales", "name": "Man Gong",                  "pcs": [0,3,5,8,10]},
    {"category": "Pentatonic Scales", "name": "Ritusen",                   "pcs": [0,2,5,7,9]},
    {"category": "Pentatonic Scales", "name": "Dorian Pentatonic",         "pcs": [0,2,3,7,9]},
    {"category": "Pentatonic Scales", "name": "Kokin-Choshi",              "pcs": [0,1,5,7,10]},
    {"category": "Pentatonic Scales", "name": "Raga Hindol",               "pcs": [0,4,6,9,11]},
    {"category": "Pentatonic Scales", "name": "Han-Kumoi",                 "pcs": [0,2,5,7,8]},
    {"category": "Pentatonic Scales", "name": "Hirajoshi",                 "pcs": [0,4,6,7,11]},
    {"category": "Pentatonic Scales", "name": "Ake-Bono",                  "pcs": [0,2,3,7,8]},
    {"category": "Pentatonic Scales", "name": "Iwato",                     "pcs": [0,1,5,6,10]},
    {"category": "Pentatonic Scales", "name": "In",                        "pcs": [0,1,5,7,8]},
    {"category": "Pentatonic Scales", "name": "Dominant Pentatonic",       "pcs": [0,2,4,7,10]},
    {"category": "Pentatonic Scales", "name": "Pentatonic Whole-Tone",     "pcs": [0,4,6,8,10]},
    {"category": "Pentatonic Scales", "name": "Locrian Pentatonic",        "pcs": [0,3,4,6,10]},
    {"category": "Pentatonic Scales", "name": "Major Pentatonic b2",       "pcs": [0,1,4,7,9]},
    {"category": "Pentatonic Scales", "name": "Mixolydian Pentatonic",     "pcs": [0,4,5,7,10]},
    {"category": "Pentatonic Scales", "name": "Altered Pentatonic",        "pcs": [0,1,5,7,9]},
    {"category": "Pentatonic Scales", "name": "Pygmy",                     "pcs": [0,2,3,7,10]},
    {"category": "Pentatonic Scales", "name": "Kyemyonjo",                 "pcs": [0,3,5,7,9]},
    {"category": "Pentatonic Scales", "name": "Ionian Pentatonic",         "pcs": [0,4,5,7,11]},
    {"category": "Pentatonic Scales", "name": "Pelog Pentatonic",          "pcs": [0,1,3,7,8]},
    {"category": "Pentatonic Scales", "name": "Raga Hamsadhvani 2",        "pcs": [0,2,4,7,11]},
    # ── Jazz Scales ───────────────────────────────────────────────────────────
    {"category": "Jazz Scales", "name": "Blues",                           "pcs": [0,3,5,6,7,10]},
    {"category": "Jazz Scales", "name": "Blues Heptatonic",                "pcs": [0,2,3,5,6,9,10]},
    {"category": "Jazz Scales", "name": "Blues Octatonic",                 "pcs": [0,2,3,5,6,7,9,10]},
    {"category": "Jazz Scales", "name": "Blues Phrygian",                  "pcs": [0,1,3,5,6,7,10]},
    {"category": "Jazz Scales", "name": "Bebop",                           "pcs": [0,2,4,5,7,9,10,11]},
    {"category": "Jazz Scales", "name": "Bebop Major",                     "pcs": [0,2,4,5,7,8,9,11]},
    {"category": "Jazz Scales", "name": "Bebop Minor",                     "pcs": [0,2,3,4,7,9,10]},
    {"category": "Jazz Scales", "name": "Bebop Dorian",                    "pcs": [0,2,3,4,5,7,9,10]},
    {"category": "Jazz Scales", "name": "Bebop Melodic Minor",             "pcs": [0,2,3,5,7,8,9,11]},
    {"category": "Jazz Scales", "name": "Bebop Harmonic Minor",            "pcs": [0,2,3,5,7,8,10,11]},
    {"category": "Jazz Scales", "name": "Bebop Half-diminished",           "pcs": [0,1,3,5,6,7,8,11]},
    {"category": "Jazz Scales", "name": "Bebop Locrian",                   "pcs": [0,1,3,5,6,7,8,10]},
    {"category": "Jazz Scales", "name": "Rock 'n Roll",                    "pcs": [0,3,4,5,7,9,10]},
    # ── Asian Scales ──────────────────────────────────────────────────────────
    {"category": "Asian Scales", "name": "Oriental",                       "pcs": [0,1,4,5,6,9,10]},
    {"category": "Asian Scales", "name": "Persian",                        "pcs": [0,1,4,5,6,8,11]},
    {"category": "Asian Scales", "name": "Pelog",                          "pcs": [0,2,4,6,7,8,11]},
    {"category": "Asian Scales", "name": "Noh",                            "pcs": [0,2,5,7,8,9,11]},
    {"category": "Asian Scales", "name": "Maqam Hijaz",                    "pcs": [0,1,4,5,7,8,10,11]},
    {"category": "Asian Scales", "name": "Ichilkotsucho",                  "pcs": [0,2,4,5,6,7,9,11]},
    {"category": "Asian Scales", "name": "Insen",                          "pcs": [0,1,5,7,8,10]},
    {"category": "Asian Scales", "name": "Honkoshi",                       "pcs": [0,1,3,5,6,10]},
    {"category": "Asian Scales", "name": "Takemitzu Tree 1",               "pcs": [0,2,3,6,8,11]},
    {"category": "Asian Scales", "name": "Takemitzu Tree 2",               "pcs": [0,2,3,6,8,10]},
    # ── Indian Scales (selección) ─────────────────────────────────────────────
    {"category": "Indian Scales", "name": "Raga Bhairavi",                 "pcs": [0,1,3,5,7,8,10]},
    {"category": "Indian Scales", "name": "Raga Malkauns",                 "pcs": [0,3,5,8,10,11]},
    {"category": "Indian Scales", "name": "Raga Hamsadhvani",              "pcs": [0,2,3,7,11]},
    {"category": "Indian Scales", "name": "Raga Bhupali",                  "pcs": [0,2,4,7,9]},
    {"category": "Indian Scales", "name": "Raga Yaman",                    "pcs": [0,2,4,6,7,9,11]},
    {"category": "Indian Scales", "name": "Raga Kafi",                     "pcs": [0,2,3,5,7,9,10]},
    {"category": "Indian Scales", "name": "Raga Bhairav",                  "pcs": [0,1,4,5,7,8,11]},
    {"category": "Indian Scales", "name": "Raga Todi",                     "pcs": [0,1,3,6,7,8,11]},
    {"category": "Indian Scales", "name": "Raga Purvi",                    "pcs": [0,1,4,6,7,8,11]},
    {"category": "Indian Scales", "name": "Raga Marwa",                    "pcs": [0,1,4,6,7,9,11]},
    {"category": "Indian Scales", "name": "Raga Ahir Bhairav",             "pcs": [0,2,3,5,7,8,11]},
    {"category": "Indian Scales", "name": "Raga Chandrakauns",             "pcs": [0,3,5,8,11]},
    {"category": "Indian Scales", "name": "Raga Darbari Kanada",           "pcs": [0,2,3,5,7,8,10]},
    {"category": "Indian Scales", "name": "Mela Kalyani",                  "pcs": [0,2,4,6,7,9,11]},
    {"category": "Indian Scales", "name": "Mela Shankarabharana",          "pcs": [0,2,4,5,7,9,11]},
    {"category": "Indian Scales", "name": "Mela Natabhairavi",             "pcs": [0,2,3,5,7,8,10]},
    # ── Miscellaneous scales ─────────────────────────────────────────────────
    {"category": "Miscellaneous scales", "name": "Algerian",               "pcs": [0,2,3,6,7,8,11]},
    {"category": "Miscellaneous scales", "name": "Algerian Octatonic",     "pcs": [0,2,3,5,6,7,8,11]},
    {"category": "Miscellaneous scales", "name": "Hawaiian",               "pcs": [0,2,3,7,9,11]},
    {"category": "Miscellaneous scales", "name": "Eskimo Hexatonic",       "pcs": [0,2,4,6,8,9]},
    {"category": "Miscellaneous scales", "name": "Pyramid Hexatonic",      "pcs": [0,2,3,5,6,9]},
    {"category": "Miscellaneous scales", "name": "Symmetrical Nonatonic",  "pcs": [0,1,2,4,6,7,8,10,11]},
    {"category": "Miscellaneous scales", "name": "LG Octatonic",           "pcs": [0,1,3,4,5,7,9,10]},
    {"category": "Miscellaneous scales", "name": "Hamel",                  "pcs": [0,1,3,5,7,8,10,11]},
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONICS = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"
}

ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "cyan":   "\033[36m",
    "gray":   "\033[90m",
    "red":    "\033[31m",
    "magenta":"\033[35m",
}

USE_COLOR = True


def c(key: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"{ANSI[key]}{text}{ANSI['reset']}"


# ─────────────────────────────────────────────────────────────────────────────
#  LECTURA DE MIDI
# ─────────────────────────────────────────────────────────────────────────────

def read_midi_pitches(path: str) -> list[int]:
    """Extrae todos los pitches únicos de un archivo MIDI."""
    try:
        import mido
    except ImportError:
        print(c("red", "Error: mido no está instalado. Instala con: pip install mido"), file=sys.stderr)
        sys.exit(1)

    mid = mido.MidiFile(path)
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitches.add(msg.note)
    return sorted(pitches)


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALIZACIÓN (pitch → pitch-class, relativo a raíz)
# ─────────────────────────────────────────────────────────────────────────────

def detect_root(pitches: list[int]) -> int:
    """Heurística simple: el pitch-class más grave de los más frecuentes."""
    return min(p % 12 for p in pitches)


def normalize(pitches: list[int], root: int) -> frozenset[int]:
    """Convierte pitches a pitch-classes relativos a la raíz dada."""
    return frozenset((p % 12 - root) % 12 for p in pitches)


def all_transpositions(pcs: list[int]) -> list[tuple[int, frozenset[int]]]:
    """Genera las 12 transposiciones de un conjunto de pitch-classes."""
    base = frozenset(pcs)
    result = []
    for t in range(12):
        transposed = frozenset((p + t) % 12 for p in base)
        result.append((t, transposed))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  BÚSQUEDA EN BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def find_exact(input_pcs: frozenset[int], category_filter: str | None) -> list[dict]:
    """
    Busca escalas cuyo conjunto de pitch-classes (en alguna transposición)
    coincide exactamente con el input.
    Devuelve lista de resultados con root y nombre.
    """
    results = []
    for scale in SCALE_DATABASE:
        if category_filter and scale["category"] != category_filter:
            continue
        for transposition, transposed in all_transpositions(scale["pcs"]):
            if transposed == input_pcs:
                root_pc = (12 - transposition) % 12
                results.append({
                    "category": scale["category"],
                    "name": scale["name"],
                    "root_pc": root_pc,
                    "root_name": NOTE_NAMES[root_pc],
                    "pcs": sorted(transposed),
                    "match_type": "exact",
                    "coverage": 1.0,
                    "missing": [],
                    "extra": [],
                })
                break  # solo uno por escala (la primera transposición que encaja)
    return results


def find_partial(input_pcs: frozenset[int], category_filter: str | None) -> list[dict]:
    """
    Busca escalas que contienen el input como subconjunto.
    Ordena por menor número de notas extra (cobertura más alta).
    """
    results = []
    for scale in SCALE_DATABASE:
        if category_filter and scale["category"] != category_filter:
            continue
        for transposition, transposed in all_transpositions(scale["pcs"]):
            if input_pcs.issubset(transposed):
                root_pc = (12 - transposition) % 12
                extra = sorted(transposed - input_pcs)
                coverage = len(input_pcs) / len(transposed)
                results.append({
                    "category": scale["category"],
                    "name": scale["name"],
                    "root_pc": root_pc,
                    "root_name": NOTE_NAMES[root_pc],
                    "pcs": sorted(transposed),
                    "match_type": "partial",
                    "coverage": round(coverage, 3),
                    "missing": [],
                    "extra": extra,
                })
                break
    results.sort(key=lambda r: (-r["coverage"], len(r["extra"])))
    return results


def find_superset(input_pcs: frozenset[int], category_filter: str | None) -> list[dict]:
    """
    Busca escalas que son subconjunto del input.
    La escala queda completamente contenida en las notas dadas.
    """
    results = []
    for scale in SCALE_DATABASE:
        if category_filter and scale["category"] != category_filter:
            continue
        for transposition, transposed in all_transpositions(scale["pcs"]):
            if transposed.issubset(input_pcs):
                root_pc = (12 - transposition) % 12
                missing = sorted(input_pcs - transposed)
                coverage = len(transposed) / len(input_pcs)
                results.append({
                    "category": scale["category"],
                    "name": scale["name"],
                    "root_pc": root_pc,
                    "root_name": NOTE_NAMES[root_pc],
                    "pcs": sorted(transposed),
                    "match_type": "superset",
                    "coverage": round(coverage, 3),
                    "missing": missing,
                    "extra": [],
                })
                break
    results.sort(key=lambda r: (-r["coverage"], -len(r["pcs"])))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PRESENTACIÓN DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def pcs_to_names(pcs: list[int], root_pc: int = 0) -> str:
    """Convierte lista de pitch-classes a nombres de notas."""
    names = []
    for pc in sorted(pcs):
        absolute = (pc + root_pc) % 12
        names.append(NOTE_NAMES[absolute])
    return " ".join(names)


def print_results(results: list[dict], input_pcs: frozenset[int], verbose: bool, top: int | None):
    shown = results[:top] if top else results

    if not shown:
        print(c("yellow", "\nNo se encontraron escalas coincidentes."))
        return

    match_type = shown[0]["match_type"]
    type_labels = {
        "exact":    c("green", "COINCIDENCIA EXACTA"),
        "partial":  c("cyan", "COBERTURA PARCIAL (input ⊆ escala)"),
        "superset": c("magenta", "SUPERSET (escala ⊆ input)"),
    }
    print(f"\n{type_labels.get(match_type, '')}  —  {len(shown)} resultado(s)")
    print(c("gray", "─" * 68))

    if verbose:
        print(c("gray", f"Input pitch-classes: {sorted(input_pcs)}"))
        print(c("gray", "─" * 68))

    prev_cat = None
    for r in shown:
        if r["category"] != prev_cat:
            print(f"\n  {c('yellow', r['category'])}")
            prev_cat = r["category"]

        cov_str = ""
        if r["match_type"] != "exact":
            cov_str = c("gray", f"  [{r['coverage']*100:.0f}%]")

        full_name = f"{r['root_name']} {r['name']}"
        print(f"    {c('bold', full_name)}{cov_str}")

        if verbose:
            note_str = pcs_to_names(r["pcs"], 0)
            print(c("gray", f"      pitch-classes: {r['pcs']}"))
            if r["extra"]:
                extra_names = [NOTE_NAMES[(pc + r["root_pc"]) % 12] for pc in r["extra"]]
                print(c("gray", f"      notas extra:   {r['extra']} ({' '.join(extra_names)})"))
            if r["missing"]:
                miss_names = [NOTE_NAMES[(pc + r["root_pc"]) % 12] for pc in r["missing"]]
                print(c("gray", f"      notas ausentes:{r['missing']} ({' '.join(miss_names)})"))

    print()


def list_categories():
    cats = sorted(set(s["category"] for s in SCALE_DATABASE))
    print(c("bold", "\nCategorías disponibles:"))
    for cat in cats:
        n = sum(1 for s in SCALE_DATABASE if s["category"] == cat)
        print(f"  {c('cyan', cat)}  ({n} escalas)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="scale_identifier — Identifica escalas desde notas MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("notes", nargs="*",
                        help="Pitches MIDI (0-127) o ruta a un archivo .mid")
    parser.add_argument("--root", metavar="NOTE",
                        help="Raíz de la escala (C, D#, Bb, …). Auto-detecta si no se especifica.")
    parser.add_argument("--top", type=int, metavar="N",
                        help="Mostrar solo los N mejores resultados")
    parser.add_argument("--partial", action="store_true",
                        help="Modo parcial: busca escalas que contienen el input")
    parser.add_argument("--superset", action="store_true",
                        help="Modo superset: busca escalas contenidas en el input")
    parser.add_argument("--category", metavar="CAT",
                        help="Filtrar por categoría de escala")
    parser.add_argument("--list-categories", action="store_true",
                        help="Listar categorías disponibles y salir")
    parser.add_argument("--output", metavar="FILE",
                        help="Guardar resultado en JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar detalle de pitch-classes")
    parser.add_argument("--no-color", action="store_true",
                        help="Sin colores ANSI")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    if args.list_categories:
        list_categories()
        return

    if not args.notes:
        parser.print_help()
        sys.exit(1)

    # ── Obtener pitches ────────────────────────────────────────────────────────
    if len(args.notes) == 1 and args.notes[0].endswith(".mid"):
        pitches = read_midi_pitches(args.notes[0])
        source = args.notes[0]
    else:
        try:
            pitches = [int(n) for n in args.notes]
        except ValueError:
            print(c("red", "Error: los valores deben ser enteros (MIDI 0-127) o una ruta .mid"), file=sys.stderr)
            sys.exit(1)
        source = "input directo"

    if not pitches:
        print(c("red", "Error: no se encontraron notas."), file=sys.stderr)
        sys.exit(1)

    # ── Determinar raíz ────────────────────────────────────────────────────────
    if args.root:
        root_name = args.root.capitalize()
        if root_name not in NOTE_NAMES:
            # intentar enarmónico
            root_name = next(
                (k for k, v in ENHARMONICS.items() if v == root_name), None
            )
            if root_name is None:
                print(c("red", f"Error: raíz '{args.root}' no reconocida."), file=sys.stderr)
                sys.exit(1)
        root_pc = NOTE_NAMES.index(root_name)
    else:
        root_pc = detect_root(pitches)
        root_name = NOTE_NAMES[root_pc]

    input_pcs = normalize(pitches, root_pc)

    print(f"\n{c('bold', 'SCALE IDENTIFIER')}  |  {source}")
    print(c("gray", f"Pitches: {sorted(set(pitches))}"))
    print(c("gray", f"Raíz detectada: {root_name} (pc {root_pc})"))
    print(c("gray", f"Pitch-classes: {sorted(input_pcs)}"))

    # ── Búsqueda ───────────────────────────────────────────────────────────────
    if args.partial:
        results = find_partial(input_pcs, args.category)
    elif args.superset:
        results = find_superset(input_pcs, args.category)
    else:
        results = find_exact(input_pcs, args.category)
        if not results:
            print(c("gray", "\nNo hay coincidencia exacta. Intentando modo parcial…"))
            results = find_partial(input_pcs, args.category)

    print_results(results, input_pcs, args.verbose, args.top)

    # ── Exportar JSON ──────────────────────────────────────────────────────────
    if args.output:
        payload = {
            "source": source,
            "root_pc": root_pc,
            "root_name": root_name,
            "input_pcs": sorted(input_pcs),
            "match_type": args.partial and "partial" or args.superset and "superset" or "exact",
            "results": results[:args.top] if args.top else results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(c("green", f"Resultado guardado en: {args.output}"))


if __name__ == "__main__":
    main()
