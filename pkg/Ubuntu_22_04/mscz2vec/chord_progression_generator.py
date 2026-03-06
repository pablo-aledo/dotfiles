#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  CHORD PROGRESSION GENERATOR  v1.0                          ║
║         Intención musical → progresión de acordes fundamentada              ║
║                                                                              ║
║  Genera progresiones de acordes desde cero a partir de:                     ║
║    · Descripción libre en lenguaje natural ("melancólico y tenso")          ║
║    · Parámetros directos (tonalidad, modo, estilo, complejidad)             ║
║    · Progresión de texto libre (--from-chords "Dm Dm/A C Am Dm")           ║
║    · Plan de theorist.py (.theorist.json)                                   ║
║    · Curvas de tensión de tension_designer.py (.curves.json)                ║
║                                                                              ║
║  A diferencia de reharmonizer.py (que transforma una melodía existente),    ║
║  este módulo construye la progresión desde reglas teóricas puras, sin       ║
║  necesitar ningún MIDI previo. Es el eslabón inicial de la cadena cuando    ║
║  no tienes material de referencia.                                           ║
║                                                                              ║
║  FLUJO INTERNO:                                                              ║
║  [1] FUNCIÓN ARMÓNICA — asigna T/S/D a cada compás según curva de tensión  ║
║  [2] SELECCIÓN DE PATRÓN — elige patrones del catálogo según estilo/modo    ║
║  [3] RESOLUCIÓN — convierte numerales romanos a acordes concretos           ║
║  [4] SCORING — puntúa candidatos por coherencia y variedad                  ║
║  [5] EXPORTACIÓN — texto, JSON y MIDI de acordes en bloque                  ║
║                                                                              ║
║  ESTILOS DISPONIBLES (--style):                                              ║
║    diatonic        — progresiones diatónicas clásicas                       ║
║    baroque         — funcionalidad barroca (I-V-vi-iii-IV-I-V)             ║
║    jazz            — ii-V-I, tensiones extendidas, tritono                  ║
║    modal           — movimiento modal, évita cadencias V-I                  ║
║    romantic        — mediantes cromáticas, préstamos modales                ║
║    impressionist   — paralelismo, tonos enteros, cuartas                    ║
║    pop             — loops de 4 acordes, vi-IV-I-V, I-V-vi-IV              ║
║    flamenco        — frigio dominante, cadencia andaluza                    ║
║    auto            — elige el más adecuado según intención (default)        ║
║                                                                              ║
║  MODOS DISPONIBLES (--mode):                                                 ║
║    major, minor, dorian, phrygian, lydian, mixolydian, locrian,             ║
║    harmonic_minor, melodic_minor, phrygian_dominant                         ║
║                                                                              ║
║  USO:                                                                        ║
║    # Desde progresión de texto libre                                         ║
║    python chord_progression_generator.py --from-chords "Dm Dm/A C Am Dm"  ║
║    python chord_progression_generator.py \                                  ║
║        --from-chords "Dm:2 Dm:2 Dm/A:2 C:4 Am:4 Dm:4" --tempo 90         ║
║                                                                              ║
║    # Desde intención libre                                                   ║
║    python chord_progression_generator.py "melancolía urbana bajo la lluvia" ║
║                                                                              ║
║    # Con parámetros directos                                                 ║
║    python chord_progression_generator.py --key Am --mode minor --style jazz ║
║        --bars 16 --complexity 0.7                                            ║
║                                                                              ║
║    # Desde plan de theorist                                                  ║
║    python chord_progression_generator.py --from-theorist obra.theorist.json ║
║                                                                              ║
║    # Desde curvas de tensión                                                 ║
║    python chord_progression_generator.py --key Dm --curves curvas.json      ║
║                                                                              ║
║    # Generar múltiples candidatos y elegir                                   ║
║    python chord_progression_generator.py "épico y ascendente" --candidates 5║
║                                                                              ║
║    # Exportar directamente como fuente para otros módulos                   ║
║    python chord_progression_generator.py "jazz nocturno" --key Cm           ║
║        --export-midi acordes.mid --export-json progresion.json              ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                              ║
║    theorist.py      → --from-theorist lee .theorist.json                    ║
║    tension_designer → --curves lee .curves.json                             ║
║    narrator.py      → --from-narrator lee obra_plan.json                    ║
║    voice_leader.py  → --export-midi genera acordes.mid compatible           ║
║    melody_adapter.py→ --export-text genera string compatible con --chords   ║
║    reharmonizer.py  → --export-midi genera fuente para reharmonización      ║
║    midi_dna_unified → --export-midi genera fuente de armonía                ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <base>.chords.txt   — progresión en texto ("Am:2 G:2 F:2 E7:2")         ║
║    <base>.chords.json  — progresión completa con metadatos                  ║
║    <base>.chords.mid   — MIDI de acordes en bloque (compatible --chords-midi)║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    description         Intención en lenguaje natural (entre comillas)       ║
║    --from-chords STR   Progresión en texto: "Dm Dm/A C Am" (exporta        ║
║                        directamente a TXT, JSON y MIDI)                     ║
║    --chord-duration N  Beats por acorde en --from-chords sin ":" (def: 4)  ║
║    --key KEY           Tónica: C, Dm, F#, Bb… (default: auto)              ║
║    --mode MODE         Modo (default: auto desde intención)                 ║
║    --style STYLE       Estilo armónico (default: auto)                      ║
║    --bars N            Compases totales (default: 16)                       ║
║    --beats N           Pulsos por compás (default: 4)                       ║
║    --tempo N           BPM (default: 120)                                   ║
║    --complexity F      Complejidad 0-1: 0=tríadas simples, 1=9ª/13ª (0.5) ║
║    --tension-profile S Perfil de tensión: arch|crescendo|decrescendo|wave  ║
║                        |late_climax|neutral (default: arch)                 ║
║    --candidates N      Candidatos a evaluar (default: 3)                    ║
║    --from-theorist F   Leer plan de theorist.py (.theorist.json)            ║
║    --from-narrator F   Leer plan de narrator.py (obra_plan.json)            ║
║    --curves F          Leer curvas de tension_designer (.curves.json)       ║
║    --export-midi F     Exportar MIDI de acordes                             ║
║    --export-json F     Exportar JSON de progresión                          ║
║    --export-text F     Exportar texto compatible con --chords               ║
║    --output BASE       Nombre base para todas las salidas (default: obra)   ║
║    --per-section       Generar progresión distinta por sección narrativa    ║
║    --verbose           Informe detallado con justificaciones teóricas       ║
║    --report            Mostrar análisis de calidad de los candidatos        ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy (music21 opcional para detección avanzada)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import random
import re
from pathlib import Path
from copy import deepcopy

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[AVISO] mido no encontrado. La exportación MIDI no estará disponible.")
    print("        pip install mido")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MUSICALES
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Tónica → pitch class
NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})
NOTE_PC.update({'Cb': 11, 'Fb': 4, 'B#': 0, 'E#': 5})

# Intervalos de acorde por calidad
CHORD_INTERVALS = {
    '':     [0, 4, 7],          # mayor
    'm':    [0, 3, 7],          # menor
    '7':    [0, 4, 7, 10],      # dominante 7ª
    'M7':   [0, 4, 7, 11],      # mayor 7ª
    'm7':   [0, 3, 7, 10],      # menor 7ª
    'dim':  [0, 3, 6],          # disminuido
    'dim7': [0, 3, 6, 9],       # disminuido 7ª
    'hdim7':[0, 3, 6, 10],      # semidisminuido
    'aug':  [0, 4, 8],          # aumentado
    'sus4': [0, 5, 7],          # sus4
    'sus2': [0, 2, 7],          # sus2
    '9':    [0, 4, 7, 10, 14],  # dominante 9ª
    'M9':   [0, 4, 7, 11, 14],  # mayor 9ª
    'm9':   [0, 3, 7, 10, 14],  # menor 9ª
    '6':    [0, 4, 7, 9],       # mayor 6ª
    'm6':   [0, 3, 7, 9],       # menor 6ª
    '13':   [0, 4, 7, 10, 14, 21], # dominante 13ª
}

# Modos: grados desde la tónica
MODES = {
    'major':            [0, 2, 4, 5, 7, 9, 11],
    'minor':            [0, 2, 3, 5, 7, 8, 10],
    'dorian':           [0, 2, 3, 5, 7, 9, 10],
    'phrygian':         [0, 1, 3, 5, 7, 8, 10],
    'lydian':           [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':       [0, 2, 4, 5, 7, 9, 10],
    'locrian':          [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor':   [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor':    [0, 2, 3, 5, 7, 9, 11],
    'phrygian_dominant':[0, 1, 4, 5, 7, 8, 10],
}

# Función armónica por grado: T=tónica, S=subdominante, D=dominante, P=preparación
HARMONIC_FUNCTION = {
    'major': {
        'I': 'T', 'ii': 'S', 'iii': 'T', 'IV': 'S',
        'V': 'D', 'V7': 'D', 'vi': 'T', 'vii°': 'D',
        'bII': 'S', 'bVI': 'T', 'bVII': 'S', 'bIII': 'T',
        'IV/I': 'S', 'V/I': 'D', 'ii/I': 'S',
    },
    'minor': {
        'i': 'T', 'ii°': 'S', 'III': 'T', 'iv': 'S',
        'V': 'D', 'V7': 'D', 'VI': 'T', 'VII': 'S',
        'bII': 'S', 'bVI': 'T', 'bIII': 'T',
    },
}

# Tensión asociada a cada función
FUNCTION_TENSION = {'T': 0.2, 'S': 0.5, 'D': 0.8, 'P': 0.6}

# ═══════════════════════════════════════════════════════════════════════════════
# CATÁLOGO DE PATRONES POR ESTILO Y MODO
# (numerales romanos + duración en beats)
# ═══════════════════════════════════════════════════════════════════════════════

PATTERNS = {

    # ── DIATÓNICO ─────────────────────────────────────────────────────────────
    'diatonic': {
        'major': [
            [('I',2),('IV',2),('V',2),('I',2)],
            [('I',2),('vi',2),('ii',2),('V',2)],
            [('I',1),('V',1),('vi',2),('IV',2)],
            [('I',2),('iii',2),('IV',2),('V',2)],
            [('IV',2),('I',2),('V',2),('I',2)],
            [('I',2),('ii',2),('V',2),('I',2)],
            [('vi',2),('IV',2),('I',2),('V',2)],
        ],
        'minor': [
            [('i',2),('iv',2),('V',2),('i',2)],
            [('i',2),('VII',2),('VI',2),('VII',2)],
            [('i',2),('VI',2),('III',2),('VII',2)],
            [('i',2),('iv',2),('VII',2),('III',2)],
            [('i',1),('VII',1),('VI',2),('V',2)],
            [('i',2),('ii°',2),('V',2),('i',2)],
        ],
    },

    # ── BARROCO ───────────────────────────────────────────────────────────────
    'baroque': {
        'major': [
            [('I',1),('V',1),('vi',1),('iii',1),('IV',1),('I',1),('V',2)],
            [('I',2),('IV',2),('ii',2),('V',2)],
            [('I',1),('ii',1),('V',1),('vi',1),('IV',2),('V',2)],
            [('I',2),('V',1),('IV',1),('ii',2),('V',2)],
        ],
        'minor': [
            [('i',1),('V',1),('i',1),('iv',1),('i',1),('V',1),('i',2)],
            [('i',2),('iv',2),('ii°',2),('V',2)],
            [('i',1),('VII',1),('VI',1),('V',1),('i',2)],
            [('i',2),('iv',1),('V',1),('i',2),('V',2)],
        ],
    },

    # ── JAZZ ──────────────────────────────────────────────────────────────────
    'jazz': {
        'major': [
            [('IM7',2),('vi7',2),('ii7',2),('V7',2)],
            [('IM7',2),('IV7',2),('iii7',1),('bIII7',1),('ii7',2)],
            [('ii7',2),('V7',2),('IM7',2),('vi7',2)],
            [('IM7',1),('bII7',1),('IM7',2),('IV7',2)],
            [('iii7',2),('bIII7',2),('ii7',2),('V7',2)],
            [('I6',2),('ii7',2),('V7',2),('IM7',2)],
        ],
        'minor': [
            [('im7',2),('iv7',2),('VII7',2),('III7',2)],
            [('im7',2),('bII7',2),('im7',2),('V7',2)],
            [('ii°7',2),('V7',2),('im7',2),('VI7',2)],
            [('im9',2),('iv9',2),('V7',2),('im7',2)],
        ],
    },

    # ── MODAL ─────────────────────────────────────────────────────────────────
    'modal': {
        'dorian': [
            [('i',4),('IV',4)],
            [('i',2),('IV',2),('i',2),('VII',2)],
            [('i',2),('II',2),('VII',2),('i',2)],
            [('i',3),('VII',1),('IV',2),('i',2)],
        ],
        'phrygian': [
            [('i',2),('bII',2),('i',2),('bII',2)],
            [('i',2),('bII',2),('VII',2),('i',2)],
            [('i',3),('bII',1),('i',4)],
        ],
        'lydian': [
            [('I',2),('II',2),('I',2),('VII',2)],
            [('I',2),('II',2),('vii',2),('I',2)],
            [('I',4),('II',4)],
        ],
        'mixolydian': [
            [('I',2),('bVII',2),('IV',2),('I',2)],
            [('I',2),('bVII',2),('I',2),('bVII',2)],
            [('I',3),('bVII',1),('IV',2),('I',2)],
        ],
        'minor': [
            [('i',2),('VII',2),('VI',2),('VII',2)],
            [('i',2),('iv',2),('V',2),('i',2)],
        ],
        'major': [
            [('I',2),('IV',2),('V',2),('I',2)],
        ],
        'phrygian_dominant': [
            [('I',2),('bII',2),('i',2),('bII',2)],
            [('I',3),('bII',1),('VII',2),('I',2)],
        ],
    },

    # ── ROMÁNTICO ─────────────────────────────────────────────────────────────
    'romantic': {
        'major': [
            [('I',2),('bVI',2),('bVII',2),('I',2)],
            [('I',2),('III',2),('IV',2),('V',2)],
            [('I',2),('iv',2),('I',2),('V',2)],
            [('I',2),('bIII',2),('bVI',2),('bVII',2)],
            [('I',1),('bVI',1),('IV',2),('V',2)],
        ],
        'minor': [
            [('i',2),('bVI',2),('bVII',2),('i',2)],
            [('i',2),('III',2),('bVI',2),('V',2)],
            [('i',2),('bII',2),('V',2),('i',2)],
            [('i',1),('bVI',1),('bIII',2),('V',2)],
        ],
    },

    # ── IMPRESIONISTA ─────────────────────────────────────────────────────────
    'impressionist': {
        'major': [
            [('I',2),('II',2),('III',2),('II',2)],
            [('I',2),('bIII',2),('bV',2),('bVII',2)],
            [('I',2),('bVI',2),('bIII',2),('bVII',2)],
            [('I',1),('bII',1),('bIII',1),('bII',1),('I',2)],
        ],
        'minor': [
            [('i',2),('bII',2),('bIII',2),('bII',2)],
            [('i',2),('VI',2),('bIII',2),('bVII',2)],
        ],
    },

    # ── POP ───────────────────────────────────────────────────────────────────
    'pop': {
        'major': [
            [('I',2),('V',2),('vi',2),('IV',2)],
            [('vi',2),('IV',2),('I',2),('V',2)],
            [('I',2),('IV',2),('vi',2),('V',2)],
            [('I',2),('iii',2),('vi',2),('IV',2)],
            [('IV',2),('I',2),('V',2),('vi',2)],
        ],
        'minor': [
            [('i',2),('VI',2),('III',2),('VII',2)],
            [('i',2),('iv',2),('VI',2),('VII',2)],
            [('i',2),('VII',2),('VI',2),('VII',2)],
        ],
    },

    # ── FLAMENCO ──────────────────────────────────────────────────────────────
    'flamenco': {
        'phrygian': [
            [('I',2),('bII',2),('i',2),('bII',2)],
            [('i',1),('VII',1),('VI',1),('bII',1),('I',2)],
            [('I',2),('bII',2),('VII',2),('i',2)],
        ],
        'phrygian_dominant': [
            [('I',2),('bII',2),('I',2),('bII',2)],
            [('I',1),('bVII',1),('VI',1),('bII',1),('I',2)],
            [('I',2),('VII',1),('VI',1),('bII',2)],
        ],
        'minor': [
            [('i',2),('VII',1),('VI',1),('bII',2)],
            [('i',2),('VI',2),('bII',2),('i',2)],
        ],
    },
}

# Mapa de modo canónico a familia de patrones
MODE_TO_PATTERN_FAMILY = {
    'major':             'major',
    'minor':             'minor',
    'harmonic_minor':    'minor',
    'melodic_minor':     'minor',
    'dorian':            'dorian',
    'phrygian':          'phrygian',
    'phrygian_dominant': 'phrygian_dominant',
    'lydian':            'lydian',
    'mixolydian':        'mixolydian',
    'locrian':           'minor',
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAPA SEMÁNTICO: palabras clave → parámetros
# ═══════════════════════════════════════════════════════════════════════════════

SEMANTIC_MAP = {
    # Modo
    'oscuro':       {'mode': 'minor', 'complexity': +0.1},
    'oscuridad':    {'mode': 'minor', 'complexity': +0.1},
    'sombrio':      {'mode': 'minor', 'complexity': +0.1},
    'sombrío':      {'mode': 'minor', 'complexity': +0.1},
    'triste':       {'mode': 'minor'},
    'tristeza':     {'mode': 'minor'},
    'melancolia':   {'mode': 'minor', 'style': 'romantic'},
    'melancolico':  {'mode': 'minor', 'style': 'romantic'},
    'melancólico':  {'mode': 'minor', 'style': 'romantic'},
    'lugubre':      {'mode': 'minor', 'complexity': +0.2},
    'lúgubre':      {'mode': 'minor', 'complexity': +0.2},
    'alegre':       {'mode': 'major'},
    'alegria':      {'mode': 'major'},
    'luminoso':     {'mode': 'major', 'style': 'diatonic'},
    'esperanza':    {'mode': 'major', 'style': 'romantic'},
    'esperanzador': {'mode': 'major', 'style': 'romantic'},
    'misterioso':   {'mode': 'dorian', 'style': 'modal', 'complexity': +0.2},
    'misterio':     {'mode': 'dorian', 'style': 'modal', 'complexity': +0.2},
    'ambiguo':      {'mode': 'dorian', 'style': 'modal'},
    'epico':        {'mode': 'minor', 'style': 'romantic', 'complexity': +0.15},
    'épico':        {'mode': 'minor', 'style': 'romantic', 'complexity': +0.15},
    'heroico':      {'mode': 'major', 'style': 'romantic'},
    'intimo':       {'complexity': -0.2, 'style': 'diatonic'},
    'íntimo':       {'complexity': -0.2, 'style': 'diatonic'},
    'agresivo':     {'mode': 'minor', 'complexity': +0.2, 'style': 'baroque'},
    'tranquilo':    {'complexity': -0.15, 'tension_profile': 'neutral'},
    'tenso':        {'complexity': +0.2, 'tension_profile': 'crescendo'},
    'tension':      {'complexity': +0.2, 'tension_profile': 'crescendo'},
    'urgente':      {'complexity': +0.15, 'tension_profile': 'late_climax'},
    'urgencia':     {'complexity': +0.15, 'tension_profile': 'late_climax'},
    'nostalgico':   {'mode': 'minor', 'style': 'romantic'},
    'nostálgico':   {'mode': 'minor', 'style': 'romantic'},
    'nostalgia':    {'mode': 'minor', 'style': 'romantic'},
    'oriental':     {'mode': 'phrygian_dominant', 'style': 'flamenco'},
    'espanol':      {'mode': 'phrygian', 'style': 'flamenco'},
    'español':      {'mode': 'phrygian', 'style': 'flamenco'},
    'flamenco':     {'mode': 'phrygian_dominant', 'style': 'flamenco'},
    # Estilos
    'jazz':         {'style': 'jazz', 'complexity': +0.3},
    'barroco':      {'style': 'baroque'},
    'clásico':      {'style': 'baroque'},
    'impresionista':{'style': 'impressionist'},
    'modal':        {'style': 'modal'},
    'pop':          {'style': 'pop', 'complexity': -0.2},
    # Complejidad
    'simple':       {'complexity': -0.3},
    'complejo':     {'complexity': +0.3},
    'elaborado':    {'complexity': +0.2},
    'sofisticado':  {'complexity': +0.25},
    # Tensión
    'ascendente':   {'tension_profile': 'crescendo'},
    'descendente':  {'tension_profile': 'decrescendo'},
    'clímax':       {'tension_profile': 'late_climax'},
    'circular':     {'tension_profile': 'wave'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE NUMERALES ROMANOS → ACORDES CONCRETOS
# ═══════════════════════════════════════════════════════════════════════════════

# Numeral → (semitono desde tónica, calidad base)
NUMERAL_TO_INTERVAL = {
    'I':    (0,  ''),    'i':    (0,  'm'),
    'II':   (2,  ''),    'ii':   (2,  'm'),   'ii°':  (2,  'dim'),
    'III':  (4,  ''),    'iii':  (4,  'm'),
    'IV':   (5,  ''),    'iv':   (5,  'm'),
    'V':    (7,  ''),    'v':    (7,  'm'),
    'VI':   (9,  ''),    'vi':   (9,  'm'),
    'VII':  (11, ''),    'vii':  (11, 'm'),   'vii°': (11, 'dim'),
    # Alterados
    'bII':  (1,  ''),    'bii':  (1,  'm'),
    'bIII': (3,  ''),    'biii': (3,  'm'),
    'bV':   (6,  ''),
    'bVI':  (8,  ''),    'bvi':  (8,  'm'),
    'bVII': (10, ''),    'bvii': (10, 'm'),
    # Con séptima
    'V7':   (7,  '7'),   'i7':   (0,  'm7'),
    'IM7':  (0,  'M7'),  'im7':  (0,  'm7'),
    'IVM7': (5,  'M7'),  'iv7':  (5,  'm7'),
    'vi7':  (9,  'm7'),  'ii7':  (2,  'm7'),
    'ii°7': (2,  'dim7'),'iii7': (4,  'm7'),
    'VII7': (11, '7'),   'IV7':  (5,  '7'),
    'bII7': (1,  '7'),   'bIII7':(3,  '7'),
    'bVII7':(10, '7'),
    # Con 9ª
    'im9':  (0,  'm9'),  'iv9':  (5,  'm9'),  'IM9':  (0, 'M9'),
    # Con 6ª
    'I6':   (0,  '6'),   'i6':   (0,  'm6'),
    # Semidisminuido
    'ii°7': (2, 'hdim7'), 'vii°7':(11,'dim7'),
    # Sus
    'Isus4':(0, 'sus4'), 'Vsus4':(7, 'sus4'),
    # Pedal / sobre bajo (simplificado: ignoramos el bajo)
    'IV/I': (5, ''),     'V/I':  (7, ''),    'ii/I': (2, 'm'),
    'bVII/I':(10,''),
    'I/V':  (0, ''),     'IV/V': (5, ''),    'V/i':  (7, ''),
    'iv/i': (5, 'm'),    'VII/i':(11,''),     'bII/i':(1, ''),
    # Dominantes secundarios (simplificados al dominante sin resolver)
    'V/ii': (9,  '7'),   'V/IV': (0,  '7'),  'V/V':  (2,  '7'),
    'V/vi': (4,  '7'),   'V/III':(11, '7'),  'V/iv': (0,  '7'),
    'V/VII':(6,  '7'),   'V/bVI':(3,  '7'),  'V/bIII':(8, '7'),
}


def resolve_numeral(numeral: str, tonic_pc: int, complexity: float = 0.5) -> tuple[str, list[int]]:
    """
    Convierte un numeral romano en (nombre_acorde, [pitch_classes]).
    complexity 0-1 decide si añadir extensiones (7ª, 9ª).
    """
    entry = NUMERAL_TO_INTERVAL.get(numeral)
    if entry is None:
        # Fallback: intentar sin alteración
        entry = NUMERAL_TO_INTERVAL.get(numeral.lstrip('b'), (0, ''))

    interval, quality = entry
    root_pc = (tonic_pc + interval) % 12

    # Añadir extensiones según complejidad
    if complexity > 0.6 and quality == '' and numeral.isupper():
        quality = 'M7'
    elif complexity > 0.6 and quality == 'm':
        quality = 'm7'
    elif complexity > 0.75 and quality == '7':
        quality = '9'

    root_name = PITCH_NAMES[root_pc]
    chord_name = f"{root_name}{quality}"
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[''])
    pitches = [(root_pc + iv) % 12 for iv in intervals]

    return chord_name, pitches


# ═══════════════════════════════════════════════════════════════════════════════
# PERFIL DE TENSIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def build_tension_profile(profile_name: str, n_bars: int) -> list[float]:
    """Genera una curva de tensión compás a compás."""
    t = np.linspace(0, 1, n_bars)
    profiles = {
        'arch':        np.sin(t * np.pi),
        'crescendo':   t,
        'decrescendo': 1 - t,
        'wave':        0.5 + 0.5 * np.sin(t * np.pi * 3),
        'late_climax': np.where(t < 0.7, t * 0.5, (t - 0.7) * 3.3),
        'neutral':     np.ones(n_bars) * 0.4,
        'plateau':     np.where((t > 0.2) & (t < 0.8), 0.8, 0.3),
    }
    curve = profiles.get(profile_name, profiles['arch'])
    return np.clip(curve, 0, 1).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS SEMÁNTICO
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    """Elimina acentos y pasa a minúsculas para comparación robusta."""
    replacements = {'á':'a','é':'e','í':'i','ó':'o','ú':'u','ü':'u','ñ':'n'}
    s = s.lower()
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def parse_intent(description: str, defaults: dict) -> dict:
    """
    Extrae parámetros musicales de una descripción en lenguaje natural.
    Aplica el SEMANTIC_MAP de forma aditiva. Ignora acentos.
    """
    params = deepcopy(defaults)
    desc_norm = _normalize(description)

    for keyword, deltas in SEMANTIC_MAP.items():
        if _normalize(keyword) in desc_norm:
            for k, v in deltas.items():
                if isinstance(v, str):
                    params[k] = v          # override directo
                elif isinstance(v, (int, float)):
                    if k not in params or not isinstance(params[k], (int, float)):
                        params[k] = 0.5
                    params[k] = float(np.clip(params[k] + v, 0.0, 1.0))

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# SELECCIÓN Y PUNTUACIÓN DE PATRONES
# ═══════════════════════════════════════════════════════════════════════════════

def get_pattern_pool(style: str, mode: str) -> list:
    """Devuelve el pool de patrones para un estilo y modo dados."""
    style_patterns = PATTERNS.get(style, PATTERNS['diatonic'])
    family = MODE_TO_PATTERN_FAMILY.get(mode, 'major')

    # Para modal, intentar la familia específica del modo
    if style == 'modal':
        pool = style_patterns.get(mode, style_patterns.get(family, []))
    else:
        pool = style_patterns.get(family, style_patterns.get('major', []))

    # Fallback a diatónico si no hay patrones
    if not pool:
        pool = PATTERNS['diatonic'].get(family, PATTERNS['diatonic']['major'])

    return pool


def score_pattern(pattern: list, tension_curve: list, n_bars: int,
                  complexity: float) -> float:
    """
    Puntúa un patrón según su adecuación a la curva de tensión.
    Devuelve float [0, 1].
    """
    if not pattern:
        return 0.0

    # Expandir patrón al número de compases
    beats_total = sum(dur for _, dur in pattern)
    bars_per_pattern = beats_total / 4  # asumimos 4/4

    scores = []
    bar = 0
    for numeral, dur in (pattern * (int(n_bars / bars_per_pattern) + 2)):
        if bar >= n_bars:
            break
        t_target = tension_curve[min(bar, len(tension_curve) - 1)]
        fn = HARMONIC_FUNCTION.get('major', {}).get(numeral,
             HARMONIC_FUNCTION.get('minor', {}).get(numeral, None))
        t_chord = FUNCTION_TENSION.get(fn, 0.5)
        scores.append(1.0 - abs(t_chord - t_target))
        bar += int(dur / 4) or 1

    base = float(np.mean(scores)) if scores else 0.5

    # Bonus por variedad de acordes
    unique_numerals = len(set(n for n, _ in pattern))
    variety_bonus = min(unique_numerals / 5, 1.0) * 0.15

    # Penalización si es demasiado simple para la complejidad pedida
    has_extensions = any('7' in n or '9' in n or 'M7' in n for n, _ in pattern)
    complexity_fit = 0.0 if (complexity > 0.6 and not has_extensions) else 0.1

    return float(np.clip(base + variety_bonus - complexity_fit, 0.0, 1.0))


def select_best_patterns(pool: list, tension_curve: list, n_bars: int,
                          complexity: float, n_candidates: int = 3) -> list:
    """
    Evalúa todos los patrones del pool y devuelve los N mejores.
    Devuelve lista de (score, pattern).
    """
    scored = []
    for pat in pool:
        s = score_pattern(pat, tension_curve, n_bars, complexity)
        scored.append((s, pat))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n_candidates]


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE PROGRESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def build_progression(pattern: list, tonic_pc: int, n_bars: int,
                      beats_per_bar: int, complexity: float) -> list:
    """
    Expande un patrón de numerales a una lista de acordes concretos.
    Devuelve lista de dicts:
        {'numeral': str, 'chord': str, 'pitches': list, 'duration_beats': float,
         'bar': int, 'beat': float}
    """
    result = []
    beats_total = sum(dur for _, dur in pattern)
    total_beats = n_bars * beats_per_bar

    # Repetir el patrón hasta cubrir todos los compases
    repeated = []
    accumulated = 0
    while accumulated < total_beats:
        for numeral, dur in pattern:
            repeated.append((numeral, dur))
            accumulated += dur
            if accumulated >= total_beats:
                break

    # Resolver cada numeral
    current_beat = 0
    for numeral, dur in repeated:
        if current_beat >= total_beats:
            break
        dur = min(dur, total_beats - current_beat)
        chord_name, pitches = resolve_numeral(numeral, tonic_pc, complexity)
        bar = int(current_beat // beats_per_bar)
        beat_in_bar = current_beat % beats_per_bar
        result.append({
            'numeral':        numeral,
            'chord':          chord_name,
            'pitches':        pitches,
            'duration_beats': float(dur),
            'bar':            bar,
            'beat':           float(beat_in_bar),
        })
        current_beat += dur

    return result


def progression_to_text(progression: list) -> str:
    """
    Convierte la progresión al formato de texto compatible con --chords.
    Ejemplo: "Am:4 G:4 F:4 E7:4"
    """
    parts = []
    for entry in progression:
        chord = entry['chord']
        dur   = entry['duration_beats']
        if dur == int(dur):
            parts.append(f"{chord}:{int(dur)}")
        else:
            parts.append(f"{chord}:{dur:.1f}")
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def progression_to_midi(progression: list, tonic_pc: int, ticks_per_beat: int,
                         tempo_bpm: float, output_path: str,
                         octave: int = 4, velocity: int = 64) -> None:
    """
    Exporta la progresión como MIDI de acordes en bloque.
    Compatible con --chords-midi de voice_leader, melody_adapter y reharmonizer.
    """
    if not MIDO_OK:
        print("[ERROR] mido no disponible. No se puede exportar MIDI.")
        return

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo
    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.MetaMessage('track_name', name='Chord Progression', time=0))

    # Nota de programa: piano
    track.append(mido.Message('program_change', channel=0, program=0, time=0))

    for entry in progression:
        pitches_pc = entry['pitches']
        dur_beats  = entry['duration_beats']
        dur_ticks  = int(dur_beats * ticks_per_beat)

        # Construir notas en la octava adecuada
        midi_notes = []
        for pc in pitches_pc:
            note = octave * 12 + pc
            # Ajustar para que quede en rango 48-84
            while note < 48: note += 12
            while note > 84: note -= 12
            midi_notes.append(note)

        # note_on
        for i, note in enumerate(midi_notes):
            track.append(mido.Message('note_on', channel=0, note=note,
                                       velocity=velocity, time=(0 if i > 0 else 0)))
        # note_off después de la duración
        for i, note in enumerate(midi_notes):
            track.append(mido.Message('note_off', channel=0, note=note,
                                       velocity=0, time=(dur_ticks if i == 0 else 0)))

    mid.save(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# LECTURA DE PLANES EXTERNOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_from_theorist(path: str) -> dict:
    """Lee un .theorist.json y extrae parámetros relevantes."""
    with open(path) as f:
        plan = json.load(f)

    pp = plan.get('pipeline_params', {})
    narrator_pp = pp.get('narrator', {})
    midi_pp = pp.get('midi_dna_unified', {})
    rehar_pp = pp.get('reharmonizer', {})

    # Extraer tonalidad
    key_str = midi_pp.get('--key', narrator_pp.get('--key', 'C'))
    # Limpiar formato "C minor" → tónica="C", modo="minor"
    parts = key_str.strip().split()
    tonic = parts[0] if parts else 'C'
    mode_str = parts[1] if len(parts) > 1 else 'major'

    # Modo desde key_mode del plan
    decisions = plan.get('decisions', [])
    for d in decisions:
        if d.get('parameter') == 'Modo / Escala':
            val = d.get('value', '')
            if val:
                mode_str = val.split()[0].lower() if val else mode_str

    # Estrategia de reharmonización → estilo
    strategies = rehar_pp.get('--strategy', ['diatonic'])
    if isinstance(strategies, str):
        strategies = strategies.split()
    style = strategies[0] if strategies else 'diatonic'
    if 'jazz' in ' '.join(strategies).lower():
        style = 'jazz'
    elif 'baroque' in ' '.join(strategies).lower():
        style = 'baroque'
    elif 'impressionist' in ' '.join(strategies).lower():
        style = 'impressionist'

    # Complejidad
    complexity_str = midi_pp.get('--mt-harmony-complexity', '0.5')
    try:
        if ':' in str(complexity_str):
            complexity = float(str(complexity_str).split(':')[0])
        else:
            complexity = float(complexity_str)
    except (ValueError, TypeError):
        complexity = 0.5

    # Curvas de tensión
    tc = plan.get('tension_curves', {})
    tension_curve = tc.get('tension', [])

    n_bars = int(midi_pp.get('--bars', 32))
    tempo  = float(midi_pp.get('--tempo', 120))

    return {
        'tonic':          tonic,
        'mode':           mode_str,
        'style':          style,
        'complexity':     float(np.clip(complexity, 0.0, 1.0)),
        'n_bars':         n_bars,
        'tempo':          tempo,
        'tension_curve':  tension_curve,
        'source':         path,
    }


def load_from_narrator(path: str) -> dict:
    """Lee un obra_plan.json de narrator y extrae parámetros."""
    with open(path) as f:
        plan = json.load(f)

    key_raw = plan.get('key', 'C')
    parts = key_raw.strip().split()
    tonic = parts[0]
    mode  = parts[1] if len(parts) > 1 else 'major'

    n_bars = plan.get('total_bars', 32)
    tempo  = plan.get('tempo', 120)
    arc    = plan.get('arc', 'hero')

    tension_profile_map = {
        'hero': 'arch', 'tragedy': 'late_climax', 'romance': 'arch',
        'mystery': 'wave', 'meditation': 'neutral', 'sonata': 'arch',
        'rondo': 'wave',
    }
    tension_profile = tension_profile_map.get(arc, 'arch')

    return {
        'tonic':           tonic,
        'mode':            mode,
        'style':           'auto',
        'complexity':      0.5,
        'n_bars':          n_bars,
        'tempo':           float(tempo),
        'tension_profile': tension_profile,
        'tension_curve':   [],
        'source':          path,
    }


def load_from_curves(path: str) -> dict:
    """Lee un .curves.json de tension_designer."""
    with open(path) as f:
        curves = json.load(f)
    tension = curves.get('tension', [])
    harmony = curves.get('harmony', [])

    # Complejidad media desde curva de armonía
    complexity = float(np.mean(harmony)) if harmony else 0.5

    return {
        'tension_curve': tension,
        'complexity':    complexity,
        'n_bars':        len(tension),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PARSE DE PROGRESIÓN EN TEXTO LIBRE
# ═══════════════════════════════════════════════════════════════════════════════

# Calidades reconocidas (orden importante: más largas primero para evitar
# coincidencias parciales como "m7" antes de "m")
_QUALITY_RE = re.compile(
    r'^(M9|M7|m9|m7|dim7|hdim7|dim|aug|sus4|sus2|13|m6|6|9|7|m|)'
    r'(?:/([A-G][b#]?))?$'
)

def parse_chord_string(chord_string: str, beats_per_chord: float = 4.0) -> list:
    """
    Parsea una cadena de texto libre de acordes al formato interno de progresión.

    Formatos de entrada aceptados:
        "Dm Dm/A C Am"             — acordes separados por espacios
        "Dm:2 G:4 C:2 Am:4"       — con duración explícita en beats
        "Dm Dm/A C Am Dm"          — con bajo alternativo (slash chords)

    Devuelve lista de dicts con la misma estructura que build_progression():
        {'numeral': str, 'chord': str, 'pitches': list,
         'duration_beats': float, 'bar': int, 'beat': float}
    """
    # Normalizar alias de calidades antes de parsear
    #   maj7 → M7,  maj → '',  min → m,  Maj7 → M7, etc.
    QUALITY_ALIASES = [
        (re.compile(r'(?i)maj7'),  'M7'),
        (re.compile(r'(?i)maj9'),  'M9'),
        (re.compile(r'(?i)maj'),   ''),
        (re.compile(r'(?i)min7'),  'm7'),
        (re.compile(r'(?i)min9'),  'm9'),
        (re.compile(r'(?i)min'),   'm'),
        (re.compile(r'(?i)△7'),   'M7'),
        (re.compile(r'(?i)△'),    'M7'),
        (re.compile(r'ø7'),        'hdim7'),
        (re.compile(r'ø'),         'hdim7'),
        (re.compile(r'°7'),        'dim7'),
        (re.compile(r'°'),         'dim'),
        (re.compile(r'\+'),        'aug'),
    ]

    tokens = chord_string.strip().split()
    if not tokens:
        raise ValueError("La cadena de acordes está vacía.")

    result = []
    current_beat = 0.0
    beats_per_bar = 4  # asumimos 4/4 para el cálculo de compás/beat

    for token in tokens:
        # Separar duración opcional: "Dm:2" → ("Dm", 2.0)
        if ':' in token:
            chord_part, dur_part = token.rsplit(':', 1)
            try:
                duration = float(dur_part)
            except ValueError:
                duration = beats_per_chord
        else:
            chord_part = token
            duration = beats_per_chord

        # Separar bajo en slash chord: "Dm/A" → raíz="Dm", bajo="A"
        bass_note = None
        if '/' in chord_part:
            chord_part, bass_note = chord_part.split('/', 1)

        # Aplicar alias de calidad sobre la parte de calidad (después de la raíz)
        root_match = re.match(r'^([A-G][b#]{0,2})', chord_part)
        if not root_match:
            raise ValueError(f"No se reconoce el acorde: '{token}'")

        root_str = root_match.group(1)
        qual_str = chord_part[len(root_str):]

        for alias_re, alias_repl in QUALITY_ALIASES:
            qual_str = alias_re.sub(alias_repl, qual_str)

        # Validar/normalizar calidad
        qm2 = re.match(r'^(M9|M7|m9|m7|dim7|hdim7|dim|aug|sus4|sus2|13|m6|6|9|7|m|)$', qual_str)
        if not qm2:
            # Calidad desconocida → tratar como mayor
            qual_norm = ''
        else:
            qual_norm = qm2.group(1)

        root_pc = NOTE_PC.get(root_str)
        if root_pc is None:
            raise ValueError(f"Nota raíz desconocida: '{root_str}' en '{token}'")

        intervals = CHORD_INTERVALS.get(qual_norm, CHORD_INTERVALS[''])
        pitches   = [(root_pc + iv) % 12 for iv in intervals]

        # Nombre del acorde (incluye el bajo si es slash chord)
        chord_name = f"{root_str}{qual_norm}"
        if bass_note:
            chord_name = f"{chord_name}/{bass_note}"

        bar        = int(current_beat // beats_per_bar)
        beat_in_bar = current_beat % beats_per_bar

        result.append({
            'numeral':        chord_name,   # sin numeral real; usamos el nombre
            'chord':          chord_name,
            'pitches':        pitches,
            'duration_beats': float(duration),
            'bar':            bar,
            'beat':           float(beat_in_bar),
        })
        current_beat += duration

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# INFORME
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(candidates: list, tonic_pc: int, params: dict) -> None:
    """Muestra un informe detallado de los candidatos."""
    c = lambda color, s: f"\033[{color}m{s}\033[0m"
    print(f"\n{c('36', '═' * 70)}")
    print(f"{c('36', '  CHORD PROGRESSION GENERATOR — Informe de candidatos')}")
    print(f"{c('36', '═' * 70)}")
    print(f"  Tónica:      {PITCH_NAMES[tonic_pc]}  |  Modo: {params['mode']}")
    print(f"  Estilo:      {params['style']}  |  Complejidad: {params['complexity']:.2f}")
    print(f"  Compases:    {params['n_bars']}  |  Tempo: {params['tempo']} BPM")
    print()

    for rank, (score, pattern, progression) in enumerate(candidates, 1):
        print(f"  {c('33', f'[{rank}]')} Score: {c('32', f'{score:.3f}')}  —  "
              f"{len(set(n for n,_ in pattern))} acordes únicos")
        # Mostrar numerales
        nums = ' → '.join(f"{n}({d}b)" for n, d in pattern)
        print(f"       Patrón: {nums}")
        # Mostrar acordes concretos (primeros 8)
        chords_preview = [e['chord'] for e in progression[:8]]
        print(f"       Acordes: {' '.join(chords_preview)}")
        print(f"       Texto:   {progression_to_text(progression[:8])}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Genera progresiones de acordes desde una intención musical.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Entrada principal
    parser.add_argument('description', nargs='?', default='',
                        help='Intención musical en lenguaje natural')

    # Parámetros directos
    parser.add_argument('--key', type=str, default=None,
                        help='Tónica: C, Am, F#, Bb… (default: C)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=list(MODES.keys()),
                        help='Modo musical (default: auto desde intención)')
    parser.add_argument('--style', type=str, default=None,
                        choices=list(PATTERNS.keys()) + ['auto'],
                        help='Estilo armónico (default: auto)')
    parser.add_argument('--bars', type=int, default=16,
                        help='Número de compases (default: 16)')
    parser.add_argument('--beats', type=int, default=4,
                        help='Pulsos por compás (default: 4)')
    parser.add_argument('--tempo', type=float, default=120.0,
                        help='Tempo en BPM (default: 120)')
    parser.add_argument('--complexity', type=float, default=0.5,
                        help='Complejidad armónica 0-1 (default: 0.5)')
    parser.add_argument('--tension-profile', type=str, default='arch',
                        choices=['arch','crescendo','decrescendo','wave',
                                 'late_climax','neutral','plateau'],
                        help='Perfil de tensión predefinido (default: arch)')

    # Progresión en texto libre
    parser.add_argument('--from-chords', type=str, metavar='CHORDS',
                        help='Progresión de acordes en texto (ej: "Dm Dm/A C Am Dm"). '
                             'Acepta duración con ":" (ej: "Dm:2 G:4"). '
                             'Exporta directamente a TXT, JSON y MIDI sin generar.')
    parser.add_argument('--chord-duration', type=float, default=4.0,
                        help='Duración por defecto de cada acorde en beats cuando '
                             'se usa --from-chords sin duración explícita (default: 4)')

    # Fuentes externas
    parser.add_argument('--from-theorist', type=str, metavar='FILE',
                        help='Leer parámetros de un .theorist.json')
    parser.add_argument('--from-narrator', type=str, metavar='FILE',
                        help='Leer parámetros de un obra_plan.json')
    parser.add_argument('--curves', type=str, metavar='FILE',
                        help='Leer curvas de un .curves.json de tension_designer')

    # Generación
    parser.add_argument('--candidates', type=int, default=3,
                        help='Número de candidatos a evaluar (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')

    # Exportación
    parser.add_argument('--output', type=str, default='obra',
                        help='Nombre base para archivos de salida (default: obra)')
    parser.add_argument('--export-midi', type=str, metavar='FILE',
                        help='Exportar MIDI de acordes a este archivo')
    parser.add_argument('--export-json', type=str, metavar='FILE',
                        help='Exportar JSON de progresión a este archivo')
    parser.add_argument('--export-text', type=str, metavar='FILE',
                        help='Exportar texto compatible con --chords')

    # Control
    parser.add_argument('--verbose', action='store_true',
                        help='Informe detallado con justificaciones')
    parser.add_argument('--report', action='store_true',
                        help='Mostrar análisis de calidad de los candidatos')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 0. MODO --from-chords: parsear progresión de texto y exportar ─────────

    if args.from_chords:
        print(f"\n[from-chords] Parseando: \"{args.from_chords}\"")
        try:
            progression = parse_chord_string(
                args.from_chords,
                beats_per_chord=args.chord_duration
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        chord_text = progression_to_text(progression)
        total_beats = sum(e['duration_beats'] for e in progression)
        n_bars      = int(total_beats / args.beats) or 1
        tempo       = args.tempo

        print(f"✓ Progresión parseada  [{len(progression)} acordes | "
              f"{total_beats:.0f} beats | ~{n_bars} compases]")
        print(f"  {chord_text}\n")

        base = args.output

        # — JSON completo
        json_path = args.export_json or f"{base}.chords.json"
        out_data = {
            'meta': {
                'source':        '--from-chords',
                'input':         args.from_chords,
                'n_bars':        n_bars,
                'beats_per_bar': args.beats,
                'tempo':         tempo,
                'generator':     'chord_progression_generator.py v1.0',
            },
            'progression':  progression,
            'chord_string': chord_text,
        }
        with open(json_path, 'w') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"  → JSON:  {json_path}")

        # — Texto para --chords
        txt_path = args.export_text or f"{base}.chords.txt"
        with open(txt_path, 'w') as f:
            f.write(chord_text + '\n')
        print(f"  → Texto: {txt_path}  (compatible con --chords)")

        # — MIDI de acordes
        if MIDO_OK:
            mid_path = args.export_midi or f"{base}.chords.mid"
            # Para MIDI necesitamos pitch classes sin el bajo en el nombre
            # progression_to_midi trabaja con 'pitches' que ya están resueltos
            progression_to_midi(
                progression, tonic_pc=0,   # tonic_pc no se usa; pitches ya calculados
                ticks_per_beat=480, tempo_bpm=tempo,
                output_path=mid_path
            )
            print(f"  → MIDI:  {mid_path}  (compatible con --chords-midi)")
        else:
            print("  [AVISO] MIDI omitido (mido no disponible).")

        sys.exit(0)

    # ── 1. RECOPILAR PARÁMETROS BASE ─────────────────────────────────────────

    params = {
        'tonic':           args.key or 'C',
        'mode':            args.mode or 'major',
        'style':           args.style or 'auto',
        'complexity':      args.complexity,
        'n_bars':          args.bars,
        'beats_per_bar':   args.beats,
        'tempo':           args.tempo,
        'tension_profile': args.tension_profile,
        'tension_curve':   [],
    }

    # ── 2. CARGAR FUENTES EXTERNAS ────────────────────────────────────────────

    if args.from_theorist:
        if not os.path.exists(args.from_theorist):
            print(f"[ERROR] No se encuentra: {args.from_theorist}")
            sys.exit(1)
        ext = load_from_theorist(args.from_theorist)
        if args.verbose:
            print(f"[theorist] Cargado: {args.from_theorist}")
        params.update({k: v for k, v in ext.items() if v})
        # CLI overrides tienen prioridad
        if args.key:   params['tonic'] = args.key
        if args.mode:  params['mode']  = args.mode
        if args.style: params['style'] = args.style

    if args.from_narrator:
        if not os.path.exists(args.from_narrator):
            print(f"[ERROR] No se encuentra: {args.from_narrator}")
            sys.exit(1)
        ext = load_from_narrator(args.from_narrator)
        if args.verbose:
            print(f"[narrator] Cargado: {args.from_narrator}")
        params.update({k: v for k, v in ext.items() if v})
        if args.key:   params['tonic'] = args.key
        if args.mode:  params['mode']  = args.mode
        if args.style: params['style'] = args.style

    if args.curves:
        if not os.path.exists(args.curves):
            print(f"[ERROR] No se encuentra: {args.curves}")
            sys.exit(1)
        ext = load_from_curves(args.curves)
        if args.verbose:
            print(f"[curves] Cargado: {args.curves}")
        params['tension_curve'] = ext.get('tension_curve', [])
        if not args.complexity and ext.get('complexity'):
            params['complexity'] = ext['complexity']
        if ext.get('n_bars') and not args.bars:
            params['n_bars'] = ext['n_bars']

    # ── 2b. PRE-PARSEAR --key ANTES DE SEMÁNTICA ─────────────────────────────
    # Así "Dm" → tónica=D, modo=minor, y la semántica puede refinar el estilo
    # pero no sobreescribir tónica ni modo si ya vienen de CLI

    _cli_key   = args.key
    _cli_mode  = args.mode
    _cli_style = args.style

    if _cli_key:
        k = _cli_key.strip()
        if k.endswith('m') and len(k) <= 3 and not k.endswith('dim'):
            params['tonic'] = k[:-1]
            params['mode']  = _cli_mode or 'minor'
        else:
            params['tonic'] = k
            if _cli_mode:
                params['mode'] = _cli_mode

    if args.description:
        if args.verbose:
            print(f"\n[semántica] Analizando: \"{args.description}\"")
        params = parse_intent(args.description, params)
        if args.verbose:
            print(f"  → modo: {params['mode']}, estilo: {params['style']}, "
                  f"complejidad: {params['complexity']:.2f}, "
                  f"perfil tensión: {params.get('tension_profile', 'arch')}")

    # CLI tiene prioridad absoluta sobre semántica
    if _cli_key:
        k = _cli_key.strip()
        if k.endswith('m') and len(k) <= 3 and not k.endswith('dim'):
            params['tonic'] = k[:-1]
            if not _cli_mode: params['mode'] = 'minor'
        else:
            params['tonic'] = k
    if _cli_mode:  params['mode']  = _cli_mode
    if _cli_style and _cli_style != 'auto': params['style'] = _cli_style

    # ── 4. RESOLVER TÓNICA ────────────────────────────────────────────────────

    # Parsear "Am" → tónica="A", modo="minor"
    tonic_str = params['tonic'].strip()
    if tonic_str.endswith('m') and not tonic_str.endswith('dim') \
            and len(tonic_str) <= 3 and params['mode'] == 'major':
        tonic_str = tonic_str[:-1]
        params['mode'] = 'minor'

    tonic_pc = NOTE_PC.get(tonic_str, 0)
    if args.verbose:
        print(f"\n[resolución] Tónica: {PITCH_NAMES[tonic_pc]} ({tonic_pc})  "
              f"Modo: {params['mode']}")

    # ── 5. CONSTRUIR CURVA DE TENSIÓN ─────────────────────────────────────────

    if not params.get('tension_curve'):
        params['tension_curve'] = build_tension_profile(
            params.get('tension_profile', 'arch'), params['n_bars']
        )

    # ── 6. ELEGIR ESTILO ──────────────────────────────────────────────────────

    if params['style'] == 'auto':
        mode_style_map = {
            'major': 'diatonic', 'minor': 'romantic',
            'dorian': 'modal', 'phrygian': 'modal',
            'phrygian_dominant': 'flamenco', 'mixolydian': 'modal',
            'lydian': 'modal', 'harmonic_minor': 'baroque',
            'melodic_minor': 'jazz',
        }
        params['style'] = mode_style_map.get(params['mode'], 'diatonic')
        if args.verbose:
            print(f"[estilo] Auto-seleccionado: {params['style']}")

    # ── 7. OBTENER POOL DE PATRONES ───────────────────────────────────────────

    pool = get_pattern_pool(params['style'], params['mode'])

    if args.verbose:
        print(f"[patrones] Pool '{params['style']}/{params['mode']}': "
              f"{len(pool)} patrones disponibles")

    # ── 8. EVALUAR CANDIDATOS ─────────────────────────────────────────────────

    best = select_best_patterns(
        pool, params['tension_curve'], params['n_bars'],
        params['complexity'], args.candidates
    )

    if not best:
        print("[ERROR] No se encontraron patrones para los parámetros dados.")
        sys.exit(1)

    # Construir progresiones completas para cada candidato
    candidates_full = []
    for score, pattern in best:
        progression = build_progression(
            pattern, tonic_pc, params['n_bars'],
            params.get('beats_per_bar', 4), params['complexity']
        )
        candidates_full.append((score, pattern, progression))

    # ── 9. INFORME / VERBOSE ──────────────────────────────────────────────────

    if args.report or args.verbose:
        print_report(candidates_full, tonic_pc, params)

    # Seleccionar el mejor
    best_score, best_pattern, best_progression = candidates_full[0]

    # Mostrar resultado principal
    # Limitar a exactamente n_bars compases para la salida principal
    beats_target = params['n_bars'] * params.get('beats_per_bar', 4)
    prog_trimmed, acc = [], 0.0
    for entry in best_progression:
        if acc >= beats_target - 0.01:
            break
        remaining = beats_target - acc
        if entry['duration_beats'] > remaining:
            entry = dict(entry)
            entry['duration_beats'] = remaining
        prog_trimmed.append(entry)
        acc += entry['duration_beats']
    best_progression = prog_trimmed

    chord_text = progression_to_text(best_progression)
    print(f"\n✓ Progresión generada  [{params['style']} | "
          f"{PITCH_NAMES[tonic_pc]} {params['mode']} | "
          f"score: {best_score:.3f}]")
    print(f"  {chord_text}\n")

    # ── 10. EXPORTACIONES ─────────────────────────────────────────────────────

    base = args.output

    # — JSON completo
    json_path = args.export_json or f"{base}.chords.json"
    out_data = {
        'meta': {
            'tonic':       PITCH_NAMES[tonic_pc],
            'mode':        params['mode'],
            'style':       params['style'],
            'complexity':  params['complexity'],
            'n_bars':      params['n_bars'],
            'beats_per_bar': params.get('beats_per_bar', 4),
            'tempo':       params['tempo'],
            'score':       best_score,
            'generator':   'chord_progression_generator.py v1.0',
        },
        'pattern':     [[n, d] for n, d in best_pattern],
        'progression': best_progression,
        'chord_string': chord_text,
        'tension_curve': params['tension_curve'],
        'candidates': [
            {
                'score': s,
                'pattern': [[n, d] for n, d in p],
                'chord_string': progression_to_text(prog),
            }
            for s, p, prog in candidates_full
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  → JSON:  {json_path}")

    # — Texto para --chords
    txt_path = args.export_text or f"{base}.chords.txt"
    with open(txt_path, 'w') as f:
        f.write(chord_text + '\n')
    print(f"  → Texto: {txt_path}  (compatible con --chords)")

    # — MIDI de acordes
    if MIDO_OK:
        mid_path = args.export_midi or f"{base}.chords.mid"
        progression_to_midi(
            best_progression, tonic_pc,
            ticks_per_beat=480, tempo_bpm=params['tempo'],
            output_path=mid_path
        )
        print(f"  → MIDI:  {mid_path}  (compatible con --chords-midi)")

    # ── 11. MOSTRAR INTEGRACIÓN CON EL ECOSISTEMA ────────────────────────────

    if args.verbose:
        print("\n┌─ Cómo usar esta progresión con el resto del ecosistema ─────┐")
        print(f"│  voice_leader.py  --chords-midi {base}.chords.mid           │")
        print(f"│  melody_adapter.py melodia.mid --chords-midi {base}.chords.mid│")
        print(f"│  reharmonizer.py melodia.mid --chords-midi {base}.chords.mid │")
        print(f"│  midi_dna_unified.py {base}.chords.mid [otros.mid]           │")
        print("└──────────────────────────────────────────────────────────────┘\n")


if __name__ == '__main__':
    main()
