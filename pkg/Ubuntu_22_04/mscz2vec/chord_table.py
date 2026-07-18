#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CHORD TABLE  v1.0                                     ║
║         Catálogo curado de progresiones con tensión y emoción                ║
║                                                                              ║
║  Complemento de chord_progression_generator.py: en lugar de construir       ║
║  progresiones algorítmicamente, ofrece un catálogo curado de 70 patrones    ║
║  con etiquetas de tensión (1-10) y emoción, buscables y exportables.        ║
║                                                                              ║
║  Las progresiones cubren los mismos estilos y modos que el generador,       ║
║  añadiendo etiquetas emocionales explícitas ausentes en aquel:               ║
║    · Modos: mayor, menor, dórico, frigio, frigio dominante, lidio,          ║
║             mixolidio, armónico menor                                        ║
║    · Estilos: diatónico, barroco, jazz, modal, romántico,                   ║
║               impresionista, pop, flamenco, blues                            ║
║    · Emociones: reposo, alegría, tristeza, melancolía, nostalgia,            ║
║                 misterio, ambigüedad, drama, angustia, esperanza,            ║
║                 euforia, flotación, solemnidad, exotismo                     ║
║                                                                              ║

║  USO:                                                                        ║
║    # Listar todas                                                            ║
║    python chord_table.py --list                                              ║
║                                                                              ║
║    # Filtrar por emoción                                                     ║
║    python chord_table.py --emocion angustia                                  ║
║                                                                              ║
║    # Filtrar por estilo y modo                                               ║
║    python chord_table.py --style jazz --mode minor                           ║
║                                                                              ║
║    # Filtrar por rango de tensión                                            ║
║    python chord_table.py --tension-min 7 --tension-max 10                   ║
║                                                                              ║
║    # Buscar por texto libre                                                  ║
║    python chord_table.py --buscar "blues"                                    ║
║                                                                              ║
║    # Resolver progresión a acordes concretos en una tónica dada             ║
║    python chord_table.py --id 16 --key Am                                   ║
║                                                                              ║
║    # Progresión personalizada por grados (modo custom)                      ║
║    python chord_table.py --custom "I vi IV V" --key C                      ║
║                                                                              ║
║    # Custom con duración por grado (numeral:beats) y export                 ║
║    python chord_table.py --custom "I:4 V:2 vi:2 IV:4" --key G              ║
║        --export-midi progresion.mid                                         ║
║                                                                              ║
║    # Exportar tabla filtrada a JSON                                          ║
║    python chord_table.py --style flamenco --export-json flamenco.json       ║
║                                                                              ║
║    # Exportar progresión concreta a texto compatible con --chords            ║
║    python chord_table.py --id 5 --key C --export-text ii_V_I.txt           ║
║                                                                              ║
║    # Exportar progresión concreta a MIDI                                     ║
║    python chord_table.py --id 5 --key C --export-midi ii_V_I.mid           ║
║                                                                              ║
║    # Mostrar estadísticas del catálogo                                       ║
║    python chord_table.py --stats                                             ║
║                                                                              ║
║    # Buscar la progresión más adecuada para una intención                    ║
║    python chord_table.py --intencion "melancólico y tenso"                  ║
║                                                                              ║
║    # Analizar un MIDI y detectar progresiones del catálogo                  ║
║    python chord_table.py --analyze-midi cancion.mid                         ║
║                                                                              ║
║    # Análisis con tónica forzada, umbral alto y exportación JSON            ║
║    python chord_table.py --analyze-midi song.mid --key Am                   ║
║        --min-score 0.75 --export-json matches.json --verbose                ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                              ║
║    chord_progression_generator.py → complementario (algorítmico vs curado)  ║
║    voice_leader.py      → --export-midi genera acordes.mid compatible       ║
║    melody_adapter.py    → --export-text genera string compatible --chords   ║
║    reharmonizer.py      → --export-midi genera fuente para reharmonización  ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <base>.chords.txt   — progresión en texto ("Am:2 G:2 F:2 E7:2")         ║
║    <base>.chords.json  — progresión completa con metadatos                  ║
║    <base>.chords.mid   — MIDI de acordes en bloque                          ║
║                                                                              ║
║  DEPENDENCIAS: mido (opcional, solo para exportación MIDI)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import re
from copy import deepcopy

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MUSICALES  (idénticas a chord_progression_generator.py)
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})
NOTE_PC.update({'Cb': 11, 'Fb': 4, 'B#': 0, 'E#': 5})

CHORD_INTERVALS = {
    '':      [0, 4, 7],
    'm':     [0, 3, 7],
    '7':     [0, 4, 7, 10],
    'M7':    [0, 4, 7, 11],
    'm7':    [0, 3, 7, 10],
    'dim':   [0, 3, 6],
    'dim7':  [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],
    'aug':   [0, 4, 8],
    'sus4':  [0, 5, 7],
    'sus2':  [0, 2, 7],
    '9':     [0, 4, 7, 10, 14],
    'M9':    [0, 4, 7, 11, 14],
    'm9':    [0, 3, 7, 10, 14],
    '6':     [0, 4, 7, 9],
    'm6':    [0, 3, 7, 9],
    '13':    [0, 4, 7, 10, 14, 21],
    'b9':    [0, 4, 7, 10, 13],
}

NUMERAL_TO_INTERVAL = {
    'I':    (0,  ''),    'i':    (0,  'm'),
    'II':   (2,  ''),    'ii':   (2,  'm'),    'ii°':  (2,  'dim'),
    'III':  (4,  ''),    'iii':  (4,  'm'),
    'IV':   (5,  ''),    'iv':   (5,  'm'),
    'V':    (7,  ''),    'v':    (7,  'm'),
    'VI':   (9,  ''),    'vi':   (9,  'm'),
    'VII':  (11, ''),    'vii':  (11, 'm'),    'vii°': (11, 'dim'),
    'bII':  (1,  ''),    'bii':  (1,  'm'),
    'bIII': (3,  ''),    'bIII': (3,  ''),
    'bV':   (6,  ''),
    'bVI':  (8,  ''),    'bvi':  (8,  'm'),
    'bVII': (10, ''),    'bvii': (10, 'm'),
    'V7':   (7,  '7'),   'i7':   (0,  'm7'),
    'IM7':  (0,  'M7'),  'im7':  (0,  'm7'),
    'IVM7': (5,  'M7'),  'iv7':  (5,  'm7'),
    'vi7':  (9,  'm7'),  'ii7':  (2,  'm7'),
    'ii°7': (2,  'hdim7'), 'iii7': (4, 'm7'),
    'VII7': (11, '7'),   'IV7':  (5,  '7'),
    'bII7': (1,  '7'),   'bIII7':(3,  '7'),
    'bVII7':(10, '7'),   'VI7':  (9,  '7'),
    'im9':  (0,  'm9'),  'iv9':  (5,  'm9'),  'IM9':  (0,  'M9'),
    'I6':   (0,  '6'),   'i6':   (0,  'm6'),
    'IIø7': (2,  'hdim7'), 'vii°7':(11,'dim7'),
    'Isus4':(0,  'sus4'), 'Vsus4':(7,  'sus4'),
    'IV/I': (5,  ''),    'V/I':  (7,  ''),    'ii/I': (2,  'm'),
    'I/V':  (0,  ''),    'IV/V': (5,  ''),
    'V/ii': (9,  '7'),   'V/IV': (0,  '7'),   'V/V':  (2,  '7'),
    'V/vi': (4,  '7'),   'V7b9': (7,  'b9'),
}

# ═══════════════════════════════════════════════════════════════════════════════
# CATÁLOGO CURADO DE PROGRESIONES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cada entrada:
#   id       — identificador único
#   prog     — progresión en numerales romanos (texto)
#   pattern  — lista de (numeral, duracion_beats) para resolución MIDI
#   nombre   — nombre descriptivo
#   style    — estilo armónico (diatonic | baroque | jazz | modal |
#               romantic | impressionist | pop | flamenco | blues)
#   mode     — modo (major | minor | dorian | phrygian | phrygian_dominant |
#               lydian | mixolydian)
#   tension  — nivel de tensión 1-10 (1=máximo reposo, 10=máxima tensión)
#   emocion  — emoción principal asociada
#   desc     — descripción breve del efecto y uso
#
TABLE = [
    {
        'id': 1, 'prog': 'I – IV – V – I',
        'pattern': [('I',2),('IV',2),('V',2),('I',2)],
        'nombre': 'Cadencia perfecta',
        'style': 'diatonic', 'mode': 'major', 'tension': 2,
        'emocion': 'reposo',
        'desc': 'La más estable del repertorio tonal. Cierra con autoridad. '
                'Himnos, folclore, rock clásico.',
    },
    {
        'id': 2, 'prog': 'I – V – vi – IV',
        'pattern': [('I',2),('V',2),('vi',2),('IV',2)],
        'nombre': 'Cuatro acordes pop',
        'style': 'pop', 'mode': 'major', 'tension': 3,
        'emocion': 'alegría',
        'desc': 'Omnipresente en pop desde los 80. Circular, sin resolver del todo.',
    },
    {
        'id': 3, 'prog': 'vi – IV – I – V',
        'pattern': [('vi',2),('IV',2),('I',2),('V',2)],
        'nombre': 'Cuatro acordes (rot.)',
        'style': 'pop', 'mode': 'major', 'tension': 3,
        'emocion': 'melancolía',
        'desc': 'Rotación comenzando en el relativo menor. Mismo loop, color más oscuro.',
    },
    {
        'id': 4, 'prog': 'I – vi – IV – V',
        'pattern': [('I',2),('vi',2),('IV',2),('V',2)],
        'nombre': 'Doo-wop / 50s',
        'style': 'diatonic', 'mode': 'major', 'tension': 3,
        'emocion': 'nostalgia',
        'desc': 'Definió el pop de los 50. Melancólica y circular.',
    },
    {
        'id': 5, 'prog': 'ii – V – I',
        'pattern': [('ii',2),('V',2),('I',4)],
        'nombre': 'Cadencia jazz',
        'style': 'jazz', 'mode': 'major', 'tension': 5,
        'emocion': 'esperanza',
        'desc': 'Motor armónico del jazz. El ii prepara el V con más riqueza que el IV.',
    },
    {
        'id': 6, 'prog': 'i – VII – VI – VII',
        'pattern': [('i',2),('VII',2),('VI',2),('VII',2)],
        'nombre': 'Dórica circular',
        'style': 'modal', 'mode': 'dorian', 'tension': 4,
        'emocion': 'misterio',
        'desc': 'Stairway to Heaven, Oye Como Va. Estática e hipnótica.',
    },
    {
        'id': 7, 'prog': 'i – VI – III – VII',
        'pattern': [('i',2),('VI',2),('III',2),('VII',2)],
        'nombre': 'Andaluza / rock menor',
        'style': 'diatonic', 'mode': 'minor', 'tension': 4,
        'emocion': 'tristeza',
        'desc': 'Loop aeólico del rock. Nothing Else Matters, Stairway.',
    },
    {
        'id': 8, 'prog': 'i – iv – V – i',
        'pattern': [('i',2),('iv',2),('V',2),('i',2)],
        'nombre': 'Cadencia menor',
        'style': 'diatonic', 'mode': 'minor', 'tension': 5,
        'emocion': 'tristeza',
        'desc': 'Equivalente menor de la cadencia perfecta.',
    },
    {
        'id': 9, 'prog': 'I – III – IV – iv',
        'pattern': [('I',2),('III',2),('IV',2),('iv',2)],
        'nombre': 'Cromatismo de plagal',
        'style': 'romantic', 'mode': 'major', 'tension': 6,
        'emocion': 'melancolía',
        'desc': 'El iv prestado del modo paralelo da un giro oscuro. Beatles, Radiohead.',
    },
    {
        'id': 10, 'prog': 'I – bVII – IV – I',
        'pattern': [('I',2),('bVII',2),('IV',2),('I',2)],
        'nombre': 'Mixolidia',
        'style': 'modal', 'mode': 'mixolydian', 'tension': 4,
        'emocion': 'euforia',
        'desc': 'Rock and roll, folk celta. Abierta y sin resolver del todo.',
    },
    {
        'id': 11, 'prog': 'I – V – I',
        'pattern': [('I',2),('V',2),('I',4)],
        'nombre': 'Cadencia auténtica simple',
        'style': 'diatonic', 'mode': 'major', 'tension': 2,
        'emocion': 'reposo',
        'desc': 'La más básica. Define la tonalidad con dos acordes.',
    },
    {
        'id': 12, 'prog': 'IV – I',
        'pattern': [('IV',4),('I',4)],
        'nombre': 'Cadencia plagal (amén)',
        'style': 'diatonic', 'mode': 'major', 'tension': 1,
        'emocion': 'reposo',
        'desc': 'Cierre suave, sin tensión previa. Música sacra, finales tranquilos.',
    },
    {
        'id': 13, 'prog': 'I7 – IV7 – I7 – V7 – IV7 – I7',
        'pattern': [('I',4),('IV',4),('I',4),('V7',2),('IV',2),('I',4)],
        'nombre': '12 compases (blues)',
        'style': 'blues', 'mode': 'major', 'tension': 6,
        'emocion': 'melancolía',
        'desc': 'El blueprint del blues. Subdominante en c.5, dominante en c.9.',
    },
    {
        'id': 14, 'prog': 'ii – V – I – VI',
        'pattern': [('ii',2),('V',2),('I',2),('VI',2)],
        'nombre': 'Turnaround jazz',
        'style': 'jazz', 'mode': 'major', 'tension': 6,
        'emocion': 'ambigüedad',
        'desc': 'Cierre de sección que vuelve a empezar. Ciclo infinito.',
    },
    {
        'id': 15, 'prog': 'I – VI – ii – V',
        'pattern': [('I',2),('VI',2),('ii',2),('V',2)],
        'nombre': 'Ciclo de quintas',
        'style': 'jazz', 'mode': 'major', 'tension': 5,
        'emocion': 'esperanza',
        'desc': 'Recorre el círculo de quintas. Cada acorde lleva al siguiente con naturalidad.',
    },
    {
        'id': 16, 'prog': 'IIø7 – V7 – i',
        'pattern': [('IIø7',2),('V7',2),('i',4)],
        'nombre': 'Cadencia frigia menor',
        'style': 'baroque', 'mode': 'minor', 'tension': 8,
        'emocion': 'angustia',
        'desc': 'El semidisminuido aporta máxima tensión antes de la resolución. Tango, clásica.',
    },
    {
        'id': 17, 'prog': 'i – bVI – bIII – bVII',
        'pattern': [('i',2),('bVI',2),('bIII',2),('bVII',2)],
        'nombre': 'Aeolia rock menor',
        'style': 'diatonic', 'mode': 'minor', 'tension': 4,
        'emocion': 'tristeza',
        'desc': 'Stairway (parte final), Nothing Else Matters.',
    },
    {
        'id': 18, 'prog': 'I – bII – I',
        'pattern': [('I',2),('bII',2),('I',4)],
        'nombre': 'Napolitana',
        'style': 'romantic', 'mode': 'major', 'tension': 7,
        'emocion': 'drama',
        'desc': 'El bII introduce una disonancia cromática intensa. Muy teatral.',
    },
    {
        'id': 19, 'prog': 'I – V – vi – iii – IV',
        'pattern': [('I',1),('V',1),('vi',2),('iii',2),('IV',2)],
        'nombre': 'Canon de Pachelbel',
        'style': 'baroque', 'mode': 'major', 'tension': 3,
        'emocion': 'nostalgia',
        'desc': 'Uno de los patrones más reconocibles. El iii añade suavidad antes del IV.',
    },
    {
        'id': 20, 'prog': 'IV – V – iii – vi',
        'pattern': [('IV',2),('V',2),('iii',2),('vi',2)],
        'nombre': 'Deceptiva hacia vi',
        'style': 'diatonic', 'mode': 'major', 'tension': 5,
        'emocion': 'ambigüedad',
        'desc': 'La cadencia no resuelve en I sino en vi. Sorpresa emotiva.',
    },
    {
        'id': 21, 'prog': 'i – bVI – bVII – i',
        'pattern': [('i',2),('bVI',2),('bVII',2),('i',2)],
        'nombre': 'Menor circular',
        'style': 'pop', 'mode': 'minor', 'tension': 4,
        'emocion': 'misterio',
        'desc': 'Loop menor sin dominante real. Música de cine, videojuegos.',
    },
    {
        'id': 22, 'prog': 'I – bVII – bVI – V',
        'pattern': [('I',2),('bVII',2),('bVI',2),('V',2)],
        'nombre': 'Descenso andaluz mayor',
        'style': 'flamenco', 'mode': 'major', 'tension': 6,
        'emocion': 'drama',
        'desc': 'Versión mayor del descenso andaluz. Directo hacia el V.',
    },
    {
        'id': 23, 'prog': 'i – bVII – bVI – V',
        'pattern': [('i',2),('bVII',2),('bVI',2),('V',2)],
        'nombre': 'Descenso andaluz menor',
        'style': 'flamenco', 'mode': 'minor', 'tension': 7,
        'emocion': 'drama',
        'desc': 'Hit the Road Jack, Sultans of Swing. Inevitable y tenso.',
    },
    {
        'id': 24, 'prog': 'I – IV – bVII – IV',
        'pattern': [('I',2),('IV',2),('bVII',2),('IV',2)],
        'nombre': 'Mixolidia rock',
        'style': 'modal', 'mode': 'mixolydian', 'tension': 4,
        'emocion': 'euforia',
        'desc': 'Sweet Home Alabama, Hey Jude. Loop abierto sin resolución real.',
    },
    {
        'id': 25, 'prog': 'ii7 – V7 – Imaj7',
        'pattern': [('ii7',2),('V7',2),('IM7',4)],
        'nombre': 'Jazz major con 7as',
        'style': 'jazz', 'mode': 'major', 'tension': 5,
        'emocion': 'esperanza',
        'desc': 'Versión extendida del ii-V-I con séptimas. Color más rico.',
    },
    {
        'id': 26, 'prog': 'bIII – bVII – I',
        'pattern': [('bIII',2),('bVII',2),('I',4)],
        'nombre': 'Rock modal plagal',
        'style': 'modal', 'mode': 'major', 'tension': 3,
        'emocion': 'euforia',
        'desc': 'Final épico del rock. Jump (Van Halen), Born to Run.',
    },
    {
        'id': 27, 'prog': 'I – IV – V – IV',
        'pattern': [('I',2),('IV',2),('V',2),('IV',2)],
        'nombre': 'Blues rock',
        'style': 'blues', 'mode': 'major', 'tension': 4,
        'emocion': 'euforia',
        'desc': 'No resuelve en I sino vuelve al IV. Abierta y cíclica.',
    },
    {
        'id': 28, 'prog': 'V – IV – I',
        'pattern': [('V',2),('IV',2),('I',4)],
        'nombre': 'Cadencia plagal de blues',
        'style': 'blues', 'mode': 'major', 'tension': 4,
        'emocion': 'melancolía',
        'desc': 'El V va al IV antes que al I. Característica del blues y rock sureño.',
    },
    {
        'id': 29, 'prog': 'I – ii – iii – IV',
        'pattern': [('I',2),('ii',2),('iii',2),('IV',2)],
        'nombre': 'Ascendente cromática',
        'style': 'diatonic', 'mode': 'major', 'tension': 3,
        'emocion': 'alegría',
        'desc': 'Sube un grado cada vez. Soul, pop, Motown. Optimista y en aumento.',
    },
    {
        'id': 30, 'prog': 'IVmaj7 – V7 – iii7 – vi',
        'pattern': [('IVM7',2),('V7',2),('iii7',2),('vi',2)],
        'nombre': 'Jazz balada',
        'style': 'jazz', 'mode': 'major', 'tension': 6,
        'emocion': 'melancolía',
        'desc': 'Lenta y rica en tensión no resuelta. Estándar de jazz balada.',
    },
    {
        'id': 31, 'prog': 'I – bVI – bVII – I',
        'pattern': [('I',2),('bVI',2),('bVII',2),('I',2)],
        'nombre': 'Préstamo modal mayor',
        'style': 'modal', 'mode': 'major', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'bVI y bVII prestados del modo paralelo. Rock progresivo.',
    },
    {
        'id': 32, 'prog': 'I – bVII – I – bVII',
        'pattern': [('I',2),('bVII',2),('I',2),('bVII',2)],
        'nombre': 'Mixolidia estática',
        'style': 'modal', 'mode': 'mixolydian', 'tension': 3,
        'emocion': 'flotación',
        'desc': 'Oscila entre I y bVII sin resolver. Meditativa. Drone-rock, ambient.',
    },
    {
        'id': 33, 'prog': 'i – II – VII – i',
        'pattern': [('i',2),('II',2),('VII',2),('i',2)],
        'nombre': 'Dórica con II mayor',
        'style': 'modal', 'mode': 'dorian', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'El II mayor es la nota característica del dórico (sexta mayor).',
    },
    {
        'id': 34, 'prog': 'i – IV – i – VII',
        'pattern': [('i',2),('IV',2),('i',2),('VII',2)],
        'nombre': 'Dórica alterna',
        'style': 'modal', 'mode': 'dorian', 'tension': 4,
        'emocion': 'misterio',
        'desc': 'Alterna entre tónica y el IV mayor dórico. Groove oscuro con luz modal.',
    },
    {
        'id': 35, 'prog': 'i – bII – i – bII',
        'pattern': [('i',2),('bII',2),('i',2),('bII',2)],
        'nombre': 'Frigia hipnótica',
        'style': 'modal', 'mode': 'phrygian', 'tension': 6,
        'emocion': 'exotismo',
        'desc': 'Oscilación i–bII característica del flamenco puro y música andaluza.',
    },
    {
        'id': 36, 'prog': 'i – bII – VII – i',
        'pattern': [('i',2),('bII',2),('VII',2),('i',2)],
        'nombre': 'Frigia con VII',
        'style': 'modal', 'mode': 'phrygian', 'tension': 6,
        'emocion': 'exotismo',
        'desc': 'El VII natural añade movimiento antes de la resolución frigia.',
    },
    {
        'id': 37, 'prog': 'I – II – I – VII',
        'pattern': [('I',2),('II',2),('I',2),('VII',2)],
        'nombre': 'Lidia oscilante',
        'style': 'modal', 'mode': 'lydian', 'tension': 3,
        'emocion': 'flotación',
        'desc': 'El II aumentado del lidio da una luminosidad irreal. Ciencia ficción, ambient.',
    },
    {
        'id': 38, 'prog': 'I – II – vii – I',
        'pattern': [('I',2),('II',2),('vii',2),('I',2)],
        'nombre': 'Lidia con sensible',
        'style': 'modal', 'mode': 'lydian', 'tension': 4,
        'emocion': 'flotación',
        'desc': 'El vii menor añade tensión sin destruir el carácter etéreo del lidio.',
    },
    {
        'id': 39, 'prog': 'I – bII – bIII – bII – I',
        'pattern': [('I',1),('bII',1),('bIII',1),('bII',1),('I',4)],
        'nombre': 'Impresionista cromática',
        'style': 'impressionist', 'mode': 'major', 'tension': 6,
        'emocion': 'ambigüedad',
        'desc': 'Movimiento paralelo cromático. Debussy, Satie. Sin funcionalidad tonal.',
    },
    {
        'id': 40, 'prog': 'I – bIII – bV – bVII',
        'pattern': [('I',2),('bIII',2),('bV',2),('bVII',2)],
        'nombre': 'Tonos enteros',
        'style': 'impressionist', 'mode': 'major', 'tension': 7,
        'emocion': 'ambigüedad',
        'desc': 'Progresión por tonos enteros. Máxima ambigüedad tonal. Debussy.',
    },
    {
        'id': 41, 'prog': 'I – bVI – bIII – bVII',
        'pattern': [('I',2),('bVI',2),('bIII',2),('bVII',2)],
        'nombre': 'Mediantes cromáticas',
        'style': 'impressionist', 'mode': 'major', 'tension': 6,
        'emocion': 'misterio',
        'desc': 'Mediantes de tercera. Romántico tardío, cine de Hollywood clásico.',
    },
    {
        'id': 42, 'prog': 'i – bII – bIII – bII',
        'pattern': [('i',2),('bII',2),('bIII',2),('bII',2)],
        'nombre': 'Impresionista menor',
        'style': 'impressionist', 'mode': 'minor', 'tension': 6,
        'emocion': 'ambigüedad',
        'desc': 'Versión menor del paralelismo cromático. Más oscura e inquietante.',
    },
    {
        'id': 43, 'prog': 'I – V – vi – iii',
        'pattern': [('I',2),('V',2),('vi',2),('iii',2)],
        'nombre': 'Canon simplificado',
        'style': 'baroque', 'mode': 'major', 'tension': 3,
        'emocion': 'nostalgia',
        'desc': 'Versión abreviada del canon. La más usada en pop contemporáneo.',
    },
    {
        'id': 44, 'prog': 'I – ii – V – vi – IV – V',
        'pattern': [('I',1),('ii',1),('V',1),('vi',1),('IV',2),('V',2)],
        'nombre': 'Barroco ampliado',
        'style': 'baroque', 'mode': 'major', 'tension': 4,
        'emocion': 'solemnidad',
        'desc': 'Ciclo funcional completo con paso por vi. Barroco tardío, contrapunto.',
    },
    {
        'id': 45, 'prog': 'i – VII – VI – V – i',
        'pattern': [('i',1),('VII',1),('VI',1),('V',1),('i',4)],
        'nombre': 'Descenso frigio',
        'style': 'baroque', 'mode': 'minor', 'tension': 7,
        'emocion': 'drama',
        'desc': 'Descenso diatónico completo hacia el V. Lamento barroco.',
    },
    {
        'id': 46, 'prog': 'i – iv – ii° – V',
        'pattern': [('i',2),('iv',2),('ii°',2),('V',2)],
        'nombre': 'Menor con ii°',
        'style': 'baroque', 'mode': 'minor', 'tension': 7,
        'emocion': 'angustia',
        'desc': 'El ii° antes del V aumenta la carga armónica. Muy oscuro.',
    },
    {
        'id': 47, 'prog': 'IM7 – vi7 – ii7 – V7',
        'pattern': [('IM7',2),('vi7',2),('ii7',2),('V7',2)],
        'nombre': 'Jazz mayor completo',
        'style': 'jazz', 'mode': 'major', 'tension': 5,
        'emocion': 'esperanza',
        'desc': 'Ciclo I-vi-ii-V con séptimas. Estándar de jazz. Smooth y sofisticado.',
    },
    {
        'id': 48, 'prog': 'IM7 – IV7 – iii7 – bIII7 – ii7',
        'pattern': [('IM7',2),('IV7',2),('iii7',1),('bIII7',1),('ii7',2)],
        'nombre': 'Jazz cromático',
        'style': 'jazz', 'mode': 'major', 'tension': 7,
        'emocion': 'ambigüedad',
        'desc': 'El bIII7 es sustitución de tritono. Hard bop, post-bop.',
    },
    {
        'id': 49, 'prog': 'ii°7 – V7 – im7 – VI7',
        'pattern': [('ii°7',2),('V7',2),('im7',2),('VI7',2)],
        'nombre': 'Jazz menor con turnaround',
        'style': 'jazz', 'mode': 'minor', 'tension': 7,
        'emocion': 'angustia',
        'desc': 'El VI7 crea un turnaround que vuelve al ii°7. Muy tenso.',
    },
    {
        'id': 50, 'prog': 'im9 – iv9 – V7 – im7',
        'pattern': [('im9',2),('iv9',2),('V7',2),('im7',2)],
        'nombre': 'Jazz menor con novenas',
        'style': 'jazz', 'mode': 'minor', 'tension': 6,
        'emocion': 'melancolía',
        'desc': 'Las novenas añaden color sin aumentar la tensión. Jazz contemporáneo.',
    },
    {
        'id': 51, 'prog': 'im7 – bII7 – im7 – V7',
        'pattern': [('im7',2),('bII7',2),('im7',2),('V7',2)],
        'nombre': 'Jazz frigio',
        'style': 'jazz', 'mode': 'minor', 'tension': 7,
        'emocion': 'exotismo',
        'desc': 'El bII7 (tritono sustituto) da color frigio al jazz. Muy característico.',
    },
    {
        'id': 52, 'prog': 'I – III – IV – V',
        'pattern': [('I',2),('III',2),('IV',2),('V',2)],
        'nombre': 'Ascendente con III',
        'style': 'romantic', 'mode': 'major', 'tension': 4,
        'emocion': 'euforia',
        'desc': 'El III mayor prestado da potencia antes del IV. Épico y ascendente.',
    },
    {
        'id': 53, 'prog': 'I – iv – I – V',
        'pattern': [('I',2),('iv',2),('I',2),('V',2)],
        'nombre': 'Romántica con iv',
        'style': 'romantic', 'mode': 'major', 'tension': 5,
        'emocion': 'melancolía',
        'desc': 'El iv menor prestado añade sombra cromática al modo mayor. Schubert.',
    },
    {
        'id': 54, 'prog': 'i – III – bVI – V',
        'pattern': [('i',2),('III',2),('bVI',2),('V',2)],
        'nombre': 'Romántica menor',
        'style': 'romantic', 'mode': 'minor', 'tension': 6,
        'emocion': 'drama',
        'desc': 'El III mayor da luminosidad momentánea antes del V. Romanticismo tardío.',
    },
    {
        'id': 55, 'prog': 'i – bII – V – i',
        'pattern': [('i',2),('bII',2),('V',2),('i',2)],
        'nombre': 'Napolitana menor',
        'style': 'romantic', 'mode': 'minor', 'tension': 8,
        'emocion': 'angustia',
        'desc': 'El bII en modo menor es aún más oscuro. Muy dramático.',
    },
    {
        'id': 56, 'prog': 'I – bIII – bVI – bVII',
        'pattern': [('I',2),('bIII',2),('bVI',2),('bVII',2)],
        'nombre': 'Mediante doble',
        'style': 'romantic', 'mode': 'major', 'tension': 6,
        'emocion': 'misterio',
        'desc': 'Doble mediante cromática. Romanticismo tardío, Wagner.',
    },
    {
        'id': 57, 'prog': 'I – bVI – IV – V',
        'pattern': [('I',2),('bVI',2),('IV',2),('V',2)],
        'nombre': 'Romántica con bVI',
        'style': 'romantic', 'mode': 'major', 'tension': 5,
        'emocion': 'nostalgia',
        'desc': 'El bVI da color antes de la cadencia. Baladas de cine, pop romántico.',
    },
    {
        'id': 58, 'prog': 'I – bII – bIII – bII',
        'pattern': [('I',2),('bII',2),('bIII',2),('bII',2)],
        'nombre': 'Frigio dominante hip',
        'style': 'flamenco', 'mode': 'phrygian_dominant', 'tension': 7,
        'emocion': 'exotismo',
        'desc': 'Versión del frigio dominante. Flamenco moderno, hip-hop oriental.',
    },
    {
        'id': 59, 'prog': 'I – bVII – VI – bII – I',
        'pattern': [('I',1),('bVII',1),('VI',1),('bII',1),('I',4)],
        'nombre': 'Cadencia flamenca completa',
        'style': 'flamenco', 'mode': 'phrygian_dominant', 'tension': 8,
        'emocion': 'exotismo',
        'desc': 'Cadencia andaluza con bII final. Flamenco clásico, copla.',
    },
    {
        'id': 60, 'prog': 'i – VII – VI – bII',
        'pattern': [('i',2),('VII',2),('VI',2),('bII',2)],
        'nombre': 'Andaluza con napolitano',
        'style': 'flamenco', 'mode': 'minor', 'tension': 8,
        'emocion': 'drama',
        'desc': 'El bII en lugar del V da un final más oscuro y frigio al descenso.',
    },
    {
        'id': 61, 'prog': 'IV – I – V – vi',
        'pattern': [('IV',2),('I',2),('V',2),('vi',2)],
        'nombre': 'Pop con final en vi',
        'style': 'pop', 'mode': 'major', 'tension': 4,
        'emocion': 'nostalgia',
        'desc': 'Termina en vi en lugar de I. Evita el cierre, sensación de continuidad.',
    },
    {
        'id': 62, 'prog': 'i – iv – VI – VII',
        'pattern': [('i',2),('iv',2),('VI',2),('VII',2)],
        'nombre': 'Pop menor ascendente',
        'style': 'pop', 'mode': 'minor', 'tension': 5,
        'emocion': 'tristeza',
        'desc': 'Sube hacia el VII sin resolver. Pop alternativo y rock oscuro.',
    },
    {
        'id': 63, 'prog': 'i – bVI – bVII – bVII',
        'pattern': [('i',2),('bVI',2),('bVII',2),('bVII',2)],
        'nombre': 'Pedal menor',
        'style': 'modal', 'mode': 'minor', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'El bVII repetido crea pedal y estasis. Suspense, Radiohead.',
    },
    {
        'id': 64, 'prog': 'IIø7 – V7b9 – i',
        'pattern': [('IIø7',2),('V7b9',2),('i',4)],
        'nombre': 'Cadencia frigia alterada',
        'style': 'jazz', 'mode': 'minor', 'tension': 9,
        'emocion': 'angustia',
        'desc': 'V con novena bemol. Tensión máxima antes de resolver. Jazz, tango, flamenco.',
    },
    {
        'id': 65, 'prog': 'iv – I',
        'pattern': [('iv',4),('I',4)],
        'nombre': 'Plagal menor',
        'style': 'diatonic', 'mode': 'minor', 'tension': 3,
        'emocion': 'melancolía',
        'desc': 'El iv prestado en mayor da cierre oscuro. Final de Yesterday (Beatles).',
    },
    {
        'id': 66, 'prog': 'i – IV – i – VII',
        'pattern': [('i',2),('IV',2),('i',2),('VII',2)],
        'nombre': 'Dórica groove',
        'style': 'modal', 'mode': 'dorian', 'tension': 4,
        'emocion': 'misterio',
        'desc': 'IV mayor natural del dórico. Groove oscuro con destellos de luz.',
    },
    {
        'id': 67, 'prog': 'I – II – VII – i',
        'pattern': [('I',2),('II',2),('VII',2),('i',2)],
        'nombre': 'Lidia resolutiva',
        'style': 'modal', 'mode': 'lydian', 'tension': 5,
        'emocion': 'flotación',
        'desc': 'La #4 del lidio (II mayor) resuelve inesperadamente al i menor. Onírico.',
    },
    {
        'id': 68, 'prog': 'i – bVI – bIII – V',
        'pattern': [('i',2),('bVI',2),('bIII',2),('V',2)],
        'nombre': 'Épica menor',
        'style': 'romantic', 'mode': 'minor', 'tension': 6,
        'emocion': 'drama',
        'desc': 'El III mayor da pausa luminosa antes del V. Cine épico, trailer music.',
    },
    {
        'id': 69, 'prog': 'I – iii – vi – IV',
        'pattern': [('I',2),('iii',2),('vi',2),('IV',2)],
        'nombre': 'Descendente luminosa',
        'style': 'diatonic', 'mode': 'major', 'tension': 3,
        'emocion': 'nostalgia',
        'desc': 'Cae por terceras desde I. Introspectiva. Baladas pop.',
    },
    {
        'id': 70, 'prog': 'i – bVI – V – i',
        'pattern': [('i',2),('bVI',2),('V',2),('i',2)],
        'nombre': 'Menor con bVI',
        'style': 'romantic', 'mode': 'minor', 'tension': 6,
        'emocion': 'drama',
        'desc': 'El bVI añade color modal antes del dominante. Ópera, cine épico.',
    },
    # ── NUEVAS (71–100) ────────────────────────────────────────────────────────
    {
        'id': 71, 'prog': 'I – V – IV – V',
        'pattern': [('I',2),('V',2),('IV',2),('V',2)],
        'nombre': 'Rock alternante',
        'style': 'diatonic', 'mode': 'major', 'tension': 3,
        'emocion': 'alegría',
        'desc': 'El V se repite como pivote. Energía sostenida sin resolver. '
                'Rock de estadio, anthems de los 70.',
    },
    {
        'id': 72, 'prog': 'I – iii – IV – V',
        'pattern': [('I',2),('iii',2),('IV',2),('V',2)],
        'nombre': 'Ascendente con iii',
        'style': 'diatonic', 'mode': 'major', 'tension': 3,
        'emocion': 'esperanza',
        'desc': 'El iii suaviza el paso al IV. Sensación de crecimiento gradual. '
                'Pop de los 80, baladas de soul.',
    },
    {
        'id': 73, 'prog': 'I – IV – ii – V',
        'pattern': [('I',2),('IV',2),('ii',2),('V',2)],
        'nombre': 'Subdominante doble',
        'style': 'diatonic', 'mode': 'major', 'tension': 4,
        'emocion': 'esperanza',
        'desc': 'El ii sustituye al IV en la segunda mitad. Más grave y sofisticado '
                'que el I-IV-V-I estándar. Jazz ligero, bossa nova.',
    },
    {
        'id': 74, 'prog': 'i – III – VII – VI',
        'pattern': [('i',2),('III',2),('VII',2),('VI',2)],
        'nombre': 'Menor por terceras',
        'style': 'diatonic', 'mode': 'minor', 'tension': 4,
        'emocion': 'melancolía',
        'desc': 'Descenso por terceras desde el III. Muy usada en música de cine '
                'y bandas sonoras de videojuegos. Circular e hipnótica.',
    },
    {
        'id': 75, 'prog': 'IV – V – vi – I',
        'pattern': [('IV',2),('V',2),('vi',2),('I',2)],
        'nombre': 'Deceptiva hacia tónica',
        'style': 'diatonic', 'mode': 'major', 'tension': 4,
        'emocion': 'nostalgia',
        'desc': 'Llega al vi antes que al I y luego cierra. Dos resoluciones '
                'encadenadas. Frecuente en corales y canciones de autor.',
    },
    {
        'id': 76, 'prog': 'I – bVII – IV – bVI',
        'pattern': [('I',2),('bVII',2),('IV',2),('bVI',2)],
        'nombre': 'Modal con bVI',
        'style': 'modal', 'mode': 'mixolydian', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'El bVI añade un giro inesperado al loop mixolidio. '
                'Rock alternativo, post-rock, Muse.',
    },
    {
        'id': 77, 'prog': 'i – iv – i – V7',
        'pattern': [('i',2),('iv',2),('i',2),('V7',2)],
        'nombre': 'Menor con dominante al final',
        'style': 'diatonic', 'mode': 'minor', 'tension': 5,
        'emocion': 'tristeza',
        'desc': 'La tónica se repite para crear peso antes del V7. '
                'Clásico de tangos y pasodobles. El V7 final pide continuar.',
    },
    {
        'id': 78, 'prog': 'I – IV – I – IV',
        'pattern': [('I',2),('IV',2),('I',2),('IV',2)],
        'nombre': 'Tónica-subdominante loop',
        'style': 'blues', 'mode': 'major', 'tension': 2,
        'emocion': 'reposo',
        'desc': 'Oscilación pura entre tónica y subdominante. '
                'Base del gospel, soul y blues lento. Estática y contenida.',
    },
    {
        'id': 79, 'prog': 'ii – IV – V – I',
        'pattern': [('ii',2),('IV',2),('V',2),('I',2)],
        'nombre': 'Cadencia con ii inicial',
        'style': 'diatonic', 'mode': 'major', 'tension': 4,
        'emocion': 'esperanza',
        'desc': 'El ii en primera posición da más profundidad que empezar en I. '
                'Usado en soul, R&B y jazz vocal.',
    },
    {
        'id': 80, 'prog': 'I – bVI – bVII – bVI',
        'pattern': [('I',2),('bVI',2),('bVII',2),('bVI',2)],
        'nombre': 'Préstamo modal oscilante',
        'style': 'modal', 'mode': 'major', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'bVI y bVII se alternan sin resolver en I. '
                'Rock progresivo, Pink Floyd, Radiohead.',
    },
    {
        'id': 81, 'prog': 'I – vi – ii – V – I',
        'pattern': [('I',2),('vi',2),('ii',2),('V',2),('I',2)],
        'nombre': 'Ciclo completo de quintas',
        'style': 'jazz', 'mode': 'major', 'tension': 4,
        'emocion': 'solemnidad',
        'desc': 'Cinco acordes recorriendo el círculo de quintas completo. '
                'Muy cadencial y satisfactorio. Barroco, jazz académico.',
    },
    {
        'id': 82, 'prog': 'IM7 – iiim7 – vim7 – IVM7',
        'pattern': [('IM7',2),('iii7',2),('vi7',2),('IVM7',2)],
        'nombre': 'Jazz diatónico suave',
        'style': 'jazz', 'mode': 'major', 'tension': 4,
        'emocion': 'reposo',
        'desc': 'Cuatro acordes diatónicos con séptimas. '
                'Dreamy, sin tensión real. Bossa nova, lofi, jazz suave.',
    },
    {
        'id': 83, 'prog': 'i – VI – III – bVII',
        'pattern': [('i',2),('VI',2),('III',2),('bVII',2)],
        'nombre': 'Menor modal ampliado',
        'style': 'modal', 'mode': 'minor', 'tension': 4,
        'emocion': 'misterio',
        'desc': 'El bVII al final abre el loop en lugar de cerrarlo. '
                'Metal melódico, bandas sonoras de fantasía.',
    },
    {
        'id': 84, 'prog': 'V – vi – IV – I',
        'pattern': [('V',2),('vi',2),('IV',2),('I',2)],
        'nombre': 'Resolución invertida',
        'style': 'diatonic', 'mode': 'major', 'tension': 4,
        'emocion': 'nostalgia',
        'desc': 'Empieza en V (sin preparación) y resuelve en I al final. '
                'Crea sensación de que la música ya estaba en marcha.',
    },
    {
        'id': 85, 'prog': 'I – V/vi – vi – V',
        'pattern': [('I',2),('V/vi',2),('vi',2),('V',2)],
        'nombre': 'Dominante secundaria al vi',
        'style': 'diatonic', 'mode': 'major', 'tension': 6,
        'emocion': 'ambigüedad',
        'desc': 'V/vi es dominante secundaria del relativo menor. '
                'Giro inesperado hacia la oscuridad. Muy expresivo.',
    },
    {
        'id': 86, 'prog': 'i – bVII – IV – bVI',
        'pattern': [('i',2),('bVII',2),('IV',2),('bVI',2)],
        'nombre': 'Menor con IV mayor',
        'style': 'modal', 'mode': 'minor', 'tension': 5,
        'emocion': 'drama',
        'desc': 'El IV mayor prestado del modo dórico añade luz momentánea. '
                'Metal progresivo, power metal. Dramático y épico.',
    },
    {
        'id': 87, 'prog': 'I – II – IV – I',
        'pattern': [('I',2),('II',2),('IV',2),('I',2)],
        'nombre': 'Lidia con IV',
        'style': 'modal', 'mode': 'lydian', 'tension': 4,
        'emocion': 'flotación',
        'desc': 'El II mayor del lidio combinado con el IV diatónico. '
                'Muy luminoso y abierto. Cine de aventuras, John Williams.',
    },
    {
        'id': 88, 'prog': 'i – bII – bVII – i',
        'pattern': [('i',2),('bII',2),('bVII',2),('i',2)],
        'nombre': 'Frigio con bVII',
        'style': 'modal', 'mode': 'phrygian', 'tension': 6,
        'emocion': 'exotismo',
        'desc': 'El bVII añade un paso intermedio al loop frigio. '
                'Flamenco moderno, metal extremo, música turca.',
    },
    {
        'id': 89, 'prog': 'I – IV – bVII – bIII',
        'pattern': [('I',2),('IV',2),('bVII',2),('bIII',2)],
        'nombre': 'Modal por cuartas',
        'style': 'modal', 'mode': 'mixolydian', 'tension': 5,
        'emocion': 'euforia',
        'desc': 'Progresión que asciende por cuartas desde el IV. '
                'Característica del rock psicodélico. Abierta y expansiva.',
    },
    {
        'id': 90, 'prog': 'i – III – iv – V',
        'pattern': [('i',2),('III',2),('iv',2),('V',2)],
        'nombre': 'Menor armónica clásica',
        'style': 'baroque', 'mode': 'minor', 'tension': 6,
        'emocion': 'solemnidad',
        'desc': 'El III mayor seguido del iv y el V crea una cadencia muy '
                'característica del barroco y el clasicismo. Bach, Händel.',
    },
    {
        'id': 91, 'prog': 'IV – bVII – I – V',
        'pattern': [('IV',2),('bVII',2),('I',2),('V',2)],
        'nombre': 'Rock épico',
        'style': 'modal', 'mode': 'major', 'tension': 5,
        'emocion': 'euforia',
        'desc': 'El bVII antes del I le da potencia épica al loop. '
                'Muy usada en rock de estadio y música de superhéroes.',
    },
    {
        'id': 92, 'prog': 'ii – ii – V – I',
        'pattern': [('ii',4),('ii',2),('V',2),('I',4)],
        'nombre': 'Doble subdominante',
        'style': 'jazz', 'mode': 'major', 'tension': 5,
        'emocion': 'esperanza',
        'desc': 'El ii repetido intensifica la preparación del V. '
                'Muy usado en jazz vocal y bossa nova lenta.',
    },
    {
        'id': 93, 'prog': 'I – bIII – IV – bVII',
        'pattern': [('I',2),('bIII',2),('IV',2),('bVII',2)],
        'nombre': 'Rock modal mixto',
        'style': 'modal', 'mode': 'major', 'tension': 5,
        'emocion': 'misterio',
        'desc': 'bIII y bVII prestados del modo paralelo. Sin resolución real. '
                'Rock alternativo de los 90, grunge.',
    },
    {
        'id': 94, 'prog': 'i – iv – bVII – bIII – bVI – bII – V – i',
        'pattern': [('i',1),('iv',1),('bVII',1),('bIII',1),
                    ('bVI',1),('bII',1),('V',1),('i',1)],
        'nombre': 'Ciclo de quintas menor',
        'style': 'baroque', 'mode': 'minor', 'tension': 7,
        'emocion': 'drama',
        'desc': 'Recorre todas las quintas en modo menor. Muy dramático y '
                'estructurado. Passacaglia, chacona, música barroca tardía.',
    },
    {
        'id': 95, 'prog': 'I – V7/IV – IV – I',
        'pattern': [('I',2),('V/IV',2),('IV',2),('I',2)],
        'nombre': 'Con dominante secundaria del IV',
        'style': 'diatonic', 'mode': 'major', 'tension': 5,
        'emocion': 'ambigüedad',
        'desc': 'V7/IV (dominante secundaria) resuelve en el IV. '
                'Giro cromático inesperado. Blues avanzado, gospel.',
    },
    {
        'id': 96, 'prog': 'im7 – iv7 – VII7 – IIIM7',
        'pattern': [('im7',2),('iv7',2),('VII7',2),('III7',2)],
        'nombre': 'Jazz menor por quintas',
        'style': 'jazz', 'mode': 'minor', 'tension': 6,
        'emocion': 'melancolía',
        'desc': 'Ciclo de quintas en modo menor con séptimas. '
                'Característico del jazz modal de Miles Davis.',
    },
    {
        'id': 97, 'prog': 'I – IV – V – vi – IV',
        'pattern': [('I',2),('IV',2),('V',2),('vi',2),('IV',2)],
        'nombre': 'Pop extendido con vi',
        'style': 'pop', 'mode': 'major', 'tension': 3,
        'emocion': 'alegría',
        'desc': 'Variante de 5 acordes del loop pop estándar. '
                'El vi en cuarta posición añade un giro melancólico breve.',
    },
    {
        'id': 98, 'prog': 'i – bVII – bVI – bVII – i',
        'pattern': [('i',2),('bVII',2),('bVI',2),('bVII',2),('i',4)],
        'nombre': 'Aeolia con pivote bVII',
        'style': 'diatonic', 'mode': 'minor', 'tension': 4,
        'emocion': 'tristeza',
        'desc': 'El bVII actúa como pivote entre bVI y la resolución en i. '
                'Muy usada en metal, rock gótico y darkwave.',
    },
    {
        'id': 99, 'prog': 'IM7 – bVIIM7 – bVIM7 – bVIIM7',
        'pattern': [('IM7',2),('bVII',2),('bVI',2),('bVII',2)],
        'nombre': 'Romántico con maj7',
        'style': 'romantic', 'mode': 'major', 'tension': 5,
        'emocion': 'nostalgia',
        'desc': 'Los maj7 suavizan el préstamo modal. '
                'Característico del pop sofisticado y R&B moderno.',
    },
    {
        'id': 100, 'prog': 'i – bVI – bIII – bVII – IV – i',
        'pattern': [('i',1),('bVI',1),('bIII',1),('bVII',1),('IV',2),('i',2)],
        'nombre': 'Menor épica de 6 acordes',
        'style': 'romantic', 'mode': 'minor', 'tension': 6,
        'emocion': 'drama',
        'desc': 'Loop de 6 acordes que recorre el modo menor con IV mayor dórico. '
                'Bandas sonoras de cine épico, videojuegos de rol.',
    },
    {
        'id': 101, 'prog': 'I – V – ii – IV',
        'pattern': [('I',2),('V',2),('ii',2),('IV',2)],
        'nombre': 'Loop pop con ii',
        'style': 'pop', 'mode': 'major', 'tension': 3,
        'emocion': 'alegría',
        'desc': 'Variante del four-chord loop donde el ii sustituye al vi. '
                'Más diatónica y resuelta que I–V–vi–IV: el ii funciona como '
                'pre-dominante hacia el IV, creando un movimiento circular estable '
                'sin la sombra melancólica del relativo menor. '
                'Let Her Go (Passenger), Torn (Imbruglia), No Woman No Cry (Marley). '
                'En inversión (I–V/vii–ii–IV) muy frecuente en folk y música litúrgica.',
    },
]

# Índice por id para acceso rápido
TABLE_BY_ID = {e['id']: e for e in TABLE}

# ═══════════════════════════════════════════════════════════════════════════════
# MAPA SEMÁNTICO: palabras → emociones → filtros
# (complementa el de chord_progression_generator.py)
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_TO_EMOCION = {
    'oscuro':        ['tristeza', 'melancolía', 'angustia'],
    'oscuridad':     ['tristeza', 'angustia'],
    'triste':        ['tristeza', 'melancolía'],
    'tristeza':      ['tristeza', 'melancolía'],
    'alegre':        ['alegría', 'euforia'],
    'alegria':       ['alegría', 'euforia'],
    'alegría':       ['alegría', 'euforia'],
    'melancolia':    ['melancolía', 'nostalgia'],
    'melancolía':    ['melancolía', 'nostalgia'],
    'melancolico':   ['melancolía', 'nostalgia'],
    'nostalgico':    ['nostalgia', 'melancolía'],
    'nostalgia':     ['nostalgia', 'melancolía'],
    'misterioso':    ['misterio', 'ambigüedad'],
    'misterio':      ['misterio', 'ambigüedad'],
    'tenso':         ['angustia', 'tensión', 'drama'],
    'tension':       ['angustia', 'drama'],
    'tensión':       ['angustia', 'drama'],
    'angustia':      ['angustia'],
    'drama':         ['drama', 'angustia'],
    'dramatico':     ['drama'],
    'dramático':     ['drama'],
    'epico':         ['drama', 'euforia'],
    'épico':         ['drama', 'euforia'],
    'esperanza':     ['esperanza'],
    'esperanzador':  ['esperanza', 'alegría'],
    'euforia':       ['euforia'],
    'euforico':      ['euforia'],
    'eufórico':      ['euforia'],
    'tranquilo':     ['reposo', 'flotación'],
    'reposo':        ['reposo'],
    'exotico':       ['exotismo'],
    'exótico':       ['exotismo'],
    'exotismo':      ['exotismo'],
    'flamenco':      ['exotismo', 'drama'],
    'oriental':      ['exotismo'],
    'solemne':       ['solemnidad'],
    'solemnidad':    ['solemnidad'],
    'ambiguo':       ['ambigüedad', 'misterio'],
    'ambigüedad':    ['ambigüedad'],
    'flotacion':     ['flotación'],
    'flotación':     ['flotación'],
    'luminoso':      ['alegría', 'esperanza', 'flotación'],
    'oscilante':     ['flotación', 'misterio'],
}

INTENT_TO_TENSION = {
    'mucha tension':   (7, 10),
    'mucha tensión':   (7, 10),
    'alta tension':    (7, 10),
    'alta tensión':    (7, 10),
    'poca tension':    (1, 3),
    'poca tensión':    (1, 3),
    'baja tension':    (1, 3),
    'baja tensión':    (1, 3),
    'media tension':   (4, 6),
    'media tensión':   (4, 6),
}

# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE NUMERALES → ACORDES CONCRETOS
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_numeral(numeral: str, tonic_pc: int) -> tuple:
    """Convierte un numeral romano en (nombre_acorde, [pitch_classes])."""
    entry = NUMERAL_TO_INTERVAL.get(numeral)
    if entry is None:
        stripped = re.sub(r'^b', '', numeral)
        entry = NUMERAL_TO_INTERVAL.get(stripped, (0, ''))

    interval, quality = entry
    root_pc   = (tonic_pc + interval) % 12
    root_name = PITCH_NAMES[root_pc]
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[''])
    pitches   = [(root_pc + iv) % 12 for iv in intervals]
    return f"{root_name}{quality}", pitches


# ═══════════════════════════════════════════════════════════════════════════════
# MODO CUSTOM — parser genérico de numerales para progresiones introducidas
# directamente por el usuario (--custom)
# ═══════════════════════════════════════════════════════════════════════════════

# Grados diatónicos de referencia (semitonos sobre la tónica, escala mayor)
DEGREE_SEMITONES = {
    'VII': 11, 'VI': 9, 'IV': 5, 'V': 7, 'III': 4, 'II': 2, 'I': 0,
}
# Orden de intento: numerales más largos primero para no confundir p.ej. 'VI' con 'V'
_DEGREE_ORDER = sorted(DEGREE_SEMITONES.keys(), key=len, reverse=True)

# Alias de sufijo de calidad → clave válida en CHORD_INTERVALS
_QUALITY_ALIASES = {
    '°': 'dim', 'o': 'dim', 'ø7': 'hdim7', 'ø': 'hdim7',
    '+': 'aug', 'dim': 'dim', 'dim7': 'dim7', 'hdim7': 'hdim7',
    'aug': 'aug', 'sus4': 'sus4', 'sus2': 'sus2',
    '7': '7', 'm7': 'm7', 'M7': 'M7', '9': '9', 'M9': 'M9', 'm9': 'm9',
    '6': '6', 'm6': 'm6', '13': '13', 'b9': 'b9', 'm': 'm', '': '',
}
# Sufijos ordenados por longitud (desc) para el matching greedy
_SUFFIX_ORDER = sorted(_QUALITY_ALIASES.keys(), key=len, reverse=True)


def parse_generic_numeral(token: str) -> tuple:
    """
    Parser de numerales romanos flexible, más allá de las entradas fijas de
    NUMERAL_TO_INTERVAL. Soporta:
      - Alteración opcional delante: b / #  (bVI, #iv...)
      - Grado diatónico I-VII (mayúscula=tríada mayor, minúscula=tríada menor
        por defecto)
      - Sufijo de calidad opcional: 7, M7, dim, dim7, hdim7/ø7, aug/+,
        sus4, sus2, 9, M9, m9, 6, m6, 13, b9, °

    Devuelve (interval_semitonos, quality) o None si no se reconoce.
    """
    s = token.strip()
    if not s:
        return None

    accidental = 0
    if s[0] == 'b':
        accidental, s = -1, s[1:]
    elif s[0] == '#':
        accidental, s = 1, s[1:]

    degree = None
    for d in _DEGREE_ORDER:
        if s[:len(d)] == d or s[:len(d)] == d.lower():
            degree = d
            break
    if degree is None:
        return None

    is_lower = s[:len(degree)] == degree.lower()
    suffix   = s[len(degree):]

    interval = (DEGREE_SEMITONES[degree] + accidental) % 12
    default_quality = 'm' if is_lower else ''

    if suffix == '':
        return (interval, default_quality)

    resolved_suffix = None
    for suf in _SUFFIX_ORDER:
        if suf and suf == suffix:
            resolved_suffix = suf
            break
    if resolved_suffix is None:
        return None

    quality = _QUALITY_ALIASES[resolved_suffix]
    # 'm7'/'7' explícitos en el sufijo ya determinan la calidad final;
    # una minúscula + '7' desnudo se interpreta como m7 (ej. 'ii7' -> m7),
    # una mayúscula + '7' desnudo es dominante (ej. 'V7' -> 7)
    if quality == '7' and is_lower:
        quality = 'm7'

    if quality not in CHORD_INTERVALS:
        return None

    return (interval, quality)


def resolve_custom_numeral(numeral: str, tonic_pc: int) -> tuple:
    """
    Resuelve un numeral de una progresión --custom. Primero intenta el
    diccionario fijo NUMERAL_TO_INTERVAL (para casos especiales como
    dominantes secundarias V/vi, V/V...); si no está, usa el parser
    genérico. Lanza ValueError con mensaje claro si no se reconoce.
    """
    entry = NUMERAL_TO_INTERVAL.get(numeral)
    if entry is None:
        entry = parse_generic_numeral(numeral)
    if entry is None:
        raise ValueError(f"grado no reconocido: '{numeral}'")

    interval, quality = entry
    root_pc   = (tonic_pc + interval) % 12
    root_name = PITCH_NAMES[root_pc]
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[''])
    pitches   = [(root_pc + iv) % 12 for iv in intervals]
    return f"{root_name}{quality}", pitches


def parse_custom_progression(text: str, default_beats: float) -> list:
    """
    Parsea el texto de --custom en una lista de (numeral, duracion_beats),
    compatible con el formato 'pattern' de las entradas del catálogo.
    Cada token es un numeral, opcionalmente seguido de ':duracion'
    (ej. 'I:4 V:2 vi:2 IV:4'). Sin duración explícita se usa default_beats.
    """
    pattern = []
    for raw_tok in text.strip().split():
        if ':' in raw_tok:
            numeral, dur_str = raw_tok.rsplit(':', 1)
            try:
                dur = float(dur_str)
            except ValueError:
                raise ValueError(
                    f"duración inválida en '{raw_tok}' (debe ser numérica)")
        else:
            numeral, dur = raw_tok, default_beats
        if not numeral:
            raise ValueError(f"token vacío en la progresión: '{raw_tok}'")
        pattern.append((numeral, dur))
    if not pattern:
        raise ValueError("la progresión --custom está vacía")
    return pattern


def make_custom_entry(text: str, default_beats: float) -> dict:
    """Construye una 'entry' sintética compatible con resolve_entry() a
    partir del texto de --custom, validando todos los grados."""
    pattern = parse_custom_progression(text, default_beats)
    # Validar cada numeral cuanto antes (tonic_pc=0 es solo para chequear
    # que resuelve sin error; el resultado real se recalcula en resolve_entry)
    for numeral, _ in pattern:
        resolve_custom_numeral(numeral, 0)
    return {
        'id':      None,
        'prog':    text.strip(),
        'pattern': pattern,
        'nombre':  'Progresión personalizada',
        'style':   'custom',
        'mode':    'custom',
        'tension': None,
        'emocion': None,
        'desc':    'Definida por el usuario mediante --custom',
    }


def resolve_entry(entry: dict, tonic_pc: int, bars: int = 8,
                  beats_per_bar: int = 4, reps: int = None) -> list:
    """
    Expande el patrón de una entrada del catálogo a una lista de acordes
    concretos.

    Si `reps` se especifica, la progresión se repite exactamente `reps`
    veces en su totalidad (sin truncar el último ciclo), ignorando
    `bars`. Si no, se rellena hasta cubrir `bars` compases (comportamiento
    original, puede truncar el ciclo final).
    Devuelve lista de dicts compatibles con chord_progression_generator.
    """
    pattern   = entry['pattern']
    beats_pat = sum(d for _, d in pattern)
    total     = beats_pat * reps if reps is not None else bars * beats_per_bar

    repeated = []
    acc = 0
    while acc < total:
        for numeral, dur in pattern:
            repeated.append((numeral, dur))
            acc += dur
            if acc >= total:
                break

    result, current = [], 0.0
    for numeral, dur in repeated:
        if current >= total:
            break
        dur = min(dur, total - current)
        chord_name, pitches = resolve_custom_numeral(numeral, tonic_pc)
        result.append({
            'numeral':        numeral,
            'chord':          chord_name,
            'pitches':        pitches,
            'duration_beats': float(dur),
            'bar':            int(current // beats_per_bar),
            'beat':           current % beats_per_bar,
        })
        current += dur
    return result


def progression_to_text(progression: list) -> str:
    """Formato compatible con --chords de los otros módulos."""
    parts = []
    for e in progression:
        dur = e['duration_beats']
        parts.append(f"{e['chord']}:{int(dur) if dur == int(dur) else f'{dur:.1f}'}")
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def progression_to_midi(progression: list, tonic_pc: int, ticks_per_beat: int,
                        tempo_bpm: float, output_path: str,
                        octave: int = 4, velocity: int = 64) -> None:
    """Exporta la progresión como MIDI de acordes en bloque."""
    if not MIDO_OK:
        print("[ERROR] mido no disponible. Instala con: pip install mido")
        return

    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',
                                  tempo=int(60_000_000 / tempo_bpm), time=0))
    track.append(mido.MetaMessage('track_name', name='Chord Table', time=0))
    track.append(mido.Message('program_change', channel=0, program=0, time=0))

    for entry in progression:
        dur_ticks  = int(entry['duration_beats'] * ticks_per_beat)
        midi_notes = []
        for pc in entry['pitches']:
            note = octave * 12 + pc
            while note < 48: note += 12
            while note > 84: note -= 12
            midi_notes.append(note)
        for i, note in enumerate(midi_notes):
            track.append(mido.Message('note_on', channel=0, note=note,
                                      velocity=velocity, time=0))
        for i, note in enumerate(midi_notes):
            track.append(mido.Message('note_off', channel=0, note=note,
                                      velocity=0,
                                      time=(dur_ticks if i == 0 else 0)))
    mid.save(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS MIDI → PROGRESIONES DEL CATÁLOGO
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_midi_raw(midi_path: str):
    """
    Parser MIDI nativo (sin dependencias externas).
    Devuelve (events, tpb) donde events es lista de (abs_tick, 'on'|'off', note_midi).
    Soporta running status y meta-mensajes.
    """
    import struct as _struct
    with open(midi_path, 'rb') as f:
        data = f.read()
    pos = 0

    def read_bytes(n):
        nonlocal pos; b = data[pos:pos+n]; pos += n; return b

    def read_uint32(): return _struct.unpack('>I', read_bytes(4))[0]
    def read_uint16(): return _struct.unpack('>H', read_bytes(2))[0]

    def read_varlen():
        nonlocal pos
        result = 0
        while True:
            b = data[pos]; pos += 1
            result = (result << 7) | (b & 0x7F)
            if not (b & 0x80): break
        return result

    assert read_bytes(4) == b'MThd', "Not a MIDI file"
    read_uint32()  # header length
    read_uint16()  # format
    ntracks = read_uint16()
    tpb     = read_uint16()

    all_events = []
    for _ in range(ntracks):
        assert read_bytes(4) == b'MTrk', "Bad track header"
        tlen    = read_uint32()
        end     = pos + tlen
        abs_tick = 0
        running = 0
        while pos < end:
            dt = read_varlen(); abs_tick += dt
            b = data[pos]
            if b & 0x80:
                running = b; pos += 1
            msg_type = running & 0xF0
            if msg_type == 0x90:
                note = data[pos]; vel = data[pos+1]; pos += 2
                etype = 'on' if vel > 0 else 'off'
                all_events.append((abs_tick, etype, note))
            elif msg_type == 0x80:
                note = data[pos]; pos += 2
                all_events.append((abs_tick, 'off', note))
            elif msg_type in (0xB0, 0xA0, 0xE0): pos += 2
            elif msg_type in (0xC0, 0xD0):        pos += 1
            elif running == 0xFF:
                pos += 1; mlen = read_varlen(); pos += mlen
            elif running in (0xF0, 0xF7):
                slen = read_varlen(); pos += slen
            else:
                break
        pos = end

    all_events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))
    return all_events, tpb


def midi_to_chord_sequence(midi_path: str, bass_only: bool = False) -> tuple:
    """
    Lee un archivo MIDI y devuelve una secuencia limpia de acordes con su duracion.

    Usa un parser MIDI nativo (sin mido) para mayor robustez.

    Si bass_only=True, solo se consideran las notas hasta el umbral
    BASS_THRESHOLD (MIDI 61 = C#4, inclusive de C4) para la detección de acordes, permitiendo
    analizar la línea de bajo de forma independiente.

    Estrategia:
    - Construye intervalos nota-encendida/nota-apagada para cada nota.
    - Muestrea en ventanas de 1 beat: PCs activos + nota MIDI más grave activa.
    - Comprime ventanas contiguas con el mismo conjunto de PCs.
    - Elimina segmentos con < 2 pitch-classes.
    - Fusiona segmentos cortos (<=1 beat) con el vecino anterior.
    - Devuelve (pcs, bass_midi, start_tick, dur_ticks) por segmento.
    """
    BASS_THRESHOLD = 61   # C4 inclusive: notas <= C4 (MIDI <=60) se consideran "bajo"
    # Antes era 60 (excluía C4), ahora C4 queda dentro del rango de bajo,
    # lo que permite identificar correctamente acordes en primera inversión
    # cuya tercera en el bajo coincide con C4 (p.ej. C/E con C4 en el acorde).

    all_events, tpb = _parse_midi_raw(midi_path)

    if not all_events:
        return [], tpb

    # ── Construir intervalos (start_tick, end_tick, note_midi) ──────────────
    active = {}
    intervals = []
    for tick, etype, note in all_events:
        if etype == 'on':
            active[(note,)] = tick
        else:
            key = (note,)
            if key in active:
                intervals.append((active.pop(key), tick, note))
    # Cerrar notas que quedaron abiertas
    max_tick = max((t for t, _, _ in all_events), default=0) + tpb
    for (note,), start in active.items():
        intervals.append((start, max_tick, note))

    total_ticks = max(e for _, e, _ in intervals) if intervals else tpb

    # ── Muestrear por ventanas de 1 beat ────────────────────────────────────
    WINDOW    = tpb
    n_windows = int(total_ticks / WINDOW) + 1

    if bass_only:
        # Recolectamos TODOS los PCs activos en el registro grave (< BASS_THRESHOLD)
        # en cada ventana, incluyendo díadas y notas dobles del bajo.
        # La nota más grave de la ventana se guarda como bass_midi para inversiones.
        window_bass_pcs  = [frozenset() for _ in range(n_windows)]
        window_bass_midi = [None] * n_windows

        for start, end, note in intervals:
            if note >= BASS_THRESHOLD:
                continue
            pc      = note % 12
            w_start = int(start / WINDOW)
            w_end   = int(end   / WINDOW)
            for w in range(max(0, w_start), min(n_windows, w_end + 1)):
                window_bass_pcs[w] = window_bass_pcs[w] | {pc}
                if window_bass_midi[w] is None or note < window_bass_midi[w]:
                    window_bass_midi[w] = note

        # Comprimir ventanas contiguas con el mismo conjunto de PCs de bajo
        compressed_bass = []
        cur_pcs   = window_bass_pcs[0]
        cur_bass  = window_bass_midi[0]
        cur_start = 0
        cur_count = 1
        for w in range(1, n_windows):
            if window_bass_pcs[w] == cur_pcs:
                cur_count += 1
                if window_bass_midi[w] is not None:
                    if cur_bass is None or window_bass_midi[w] < cur_bass:
                        cur_bass = window_bass_midi[w]
            else:
                if cur_pcs:
                    compressed_bass.append(
                        (cur_pcs, cur_bass, cur_start * WINDOW, cur_count * WINDOW)
                    )
                cur_pcs   = window_bass_pcs[w]
                cur_bass  = window_bass_midi[w]
                cur_start = w
                cur_count = 1
        if cur_pcs:
            compressed_bass.append(
                (cur_pcs, cur_bass, cur_start * WINDOW, cur_count * WINDOW)
            )

        # Filtrar segmentos vacíos y fusionar cortos (≤1 beat) con el anterior
        filtered = [(p, b, s, c) for p, b, s, c in compressed_bass if len(p) >= 1]
        merged = []
        for pcs, bass, start, count in filtered:
            if merged and count <= 1:
                pp, pb, ps, pc_ = merged[-1]
                nb = pb if (bass is None or (pb is not None and pb <= bass)) else bass
                merged[-1] = (pp, nb, ps, pc_ + count)
            elif merged and pcs == merged[-1][0]:
                pp, pb, ps, pc_ = merged[-1]
                nb = pb if (bass is None or (pb is not None and pb <= bass)) else bass
                merged[-1] = (pp, nb, ps, pc_ + count)
            else:
                merged.append((pcs, bass, start, count))

        return [(pcs, bass, s, c) for pcs, bass, s, c in merged], tpb

    # ── Modo normal: PCS de todas las notas ──────────────────────────────────
    # Para cada ventana: set de PCs activos y nota MIDI mínima activa
    window_pcs  = [frozenset() for _ in range(n_windows)]
    window_bass = [None]        * n_windows   # nota MIDI más grave en la ventana

    for start, end, note in intervals:
        pc = note % 12
        # Si bass_only, ignorar notas que no son bajo
        if bass_only and note >= BASS_THRESHOLD:
            continue
        w_start = int(start / WINDOW)
        w_end   = int(end   / WINDOW)
        for w in range(max(0, w_start), min(n_windows, w_end + 1)):
            window_pcs[w]  = window_pcs[w] | {pc}
            if window_bass[w] is None or note < window_bass[w]:
                window_bass[w] = note

    # ── Comprimir ventanas contiguas con el mismo PCS ────────────────────────
    compressed = []
    if window_pcs:
        cur_pcs   = window_pcs[0]
        cur_bass  = window_bass[0]
        cur_start = 0
        cur_count = 1
        for w in range(1, n_windows):
            if window_pcs[w] == cur_pcs:
                cur_count += 1
                # Actualizar bajo más grave del segmento
                if window_bass[w] is not None:
                    if cur_bass is None or window_bass[w] < cur_bass:
                        cur_bass = window_bass[w]
            else:
                compressed.append((cur_pcs, cur_bass, cur_start, cur_count))
                cur_pcs   = window_pcs[w]
                cur_bass  = window_bass[w]
                cur_start = w
                cur_count = 1
        compressed.append((cur_pcs, cur_bass, cur_start, cur_count))

    # ── Filtrar segmentos con < 2 pitch-classes ──────────────────────────────
    filtered = [(pcs, bass, s, c) for pcs, bass, s, c in compressed if len(pcs) >= 2]

    # ── Fusionar segmentos cortos (1 beat) con el vecino anterior ────────────
    MERGE_THR = 1
    merged = []
    for pcs, bass, start, count in filtered:
        if merged and count <= MERGE_THR:
            prev_pcs, prev_bass, prev_s, prev_c = merged[-1]
            new_bass = prev_bass
            if bass is not None and (new_bass is None or bass < new_bass):
                new_bass = bass
            merged[-1] = (prev_pcs, new_bass, prev_s, prev_c + count)
        elif merged and pcs == merged[-1][0]:
            prev_pcs, prev_bass, prev_s, prev_c = merged[-1]
            new_bass = prev_bass
            if bass is not None and (new_bass is None or bass < new_bass):
                new_bass = bass
            merged[-1] = (prev_pcs, new_bass, prev_s, prev_c + count)
        else:
            merged.append((pcs, bass, start, count))

    # Convertir ventanas a ticks — resultado: (pcs, bass_midi, start_tick, dur_ticks)
    result = [(pcs, bass, s * WINDOW, c * WINDOW) for pcs, bass, s, c in merged]
    return result, tpb



def pcs_to_chord_name(pcs: frozenset, bass_midi: int = None) -> tuple:
    """
    Identifica el nombre del acorde, la tónica y la inversión a partir de un
    conjunto de pitch-classes y, opcionalmente, la nota MIDI más grave (bajo).

    Devuelve (tónica_pc, calidad, nombre_completo, bass_pc) donde:
      - nombre_completo incluye notación de inversión "Root/Bass" si corresponde
      - bass_pc es el PC de la nota más grave (None si no se dispone)
    """
    pcs  = frozenset(pcs)

    # Caso especial: un solo PC (nota pedal en modo bass_only)
    if len(pcs) == 1:
        pc = next(iter(pcs))
        name = PITCH_NAMES_FLAT[pc]
        return pc, '', name, (bass_midi % 12 if bass_midi is not None else pc)

    # Caso díada (2 PCs): elegir el template más probable usando una puntuación
    # que combina tamaño del template (triada < séptima), si la raíz coincide con
    # el bajo, y un bonus de "calidad diatónica" (mayor/menor > sus/aug/dim).
    if len(pcs) == 2:
        dyad         = frozenset(pcs)
        bass_pc_hint = bass_midi % 12 if bass_midi is not None else None
        # Peso de calidad: triadas diatónicas > sus > aumentadas > séptimas > disminuidas
        QUALITY_SCORE = {
            '': 10, 'm': 10, '7': 7, 'M7': 7, 'm7': 7,
            'sus4': 5, 'sus2': 5, 'aug': 4,
            'dim': 3, 'dim7': 2, 'hdim7': 2,
        }
        best_dyad = (None, None, '?', -1)  # (root_pc, quality, name, score)
        for quality, ivs in CHORD_INTERVALS.items():
            for root in range(12):
                candidate = frozenset((root + iv) % 12 for iv in ivs)
                if not (dyad <= candidate):
                    continue
                q_score = QUALITY_SCORE.get(quality, 3)
                size_bonus = 10 - len(ivs)           # triada(3)=7, séptima(4)=6
                bass_bonus = 5 if root == bass_pc_hint else 0
                total = q_score + size_bonus + bass_bonus
                if total > best_dyad[3]:
                    name = f"{PITCH_NAMES_FLAT[root]}{quality}"
                    best_dyad = (root, quality, name, total)
        if best_dyad[0] is not None:
            root_pc, quality = best_dyad[0], best_dyad[1]
            name = best_dyad[2]
            bass_pc = bass_pc_hint
            if bass_pc is not None and bass_pc != root_pc:
                chord_pcs = frozenset((root_pc + iv) % 12
                                       for iv in CHORD_INTERVALS.get(quality, []))
                if bass_pc in chord_pcs:
                    name = f"{name}/{PITCH_NAMES_FLAT[bass_pc]}"
            return root_pc, quality, name, bass_pc

    best = (None, None, '?', -1)   # (root_pc, quality, name, score)

    for quality, intervals in CHORD_INTERVALS.items():
        for root in range(12):
            candidate = frozenset((root + iv) % 12 for iv in intervals)
            overlap = len(pcs & candidate)
            total   = max(len(pcs), len(candidate))
            score   = overlap / total if total else 0
            if overlap == len(candidate) and score > best[3]:
                name = f"{PITCH_NAMES[root]}{quality}"
                best = (root, quality, name, score)

    root_pc, quality, name = best[0], best[1], best[2]

    # ── Detección de inversión ───────────────────────────────────────────────
    bass_pc = bass_midi % 12 if bass_midi is not None else None
    if (bass_pc is not None and root_pc is not None and bass_pc != root_pc):
        # Solo añadir /Bajo si el bajo es una nota del acorde
        chord_pcs = frozenset((root_pc + iv) % 12
                               for iv in CHORD_INTERVALS.get(quality, []))
        if bass_pc in chord_pcs:
            bass_name = PITCH_NAMES_FLAT[bass_pc]
            name = f"{name}/{bass_name}"

    return root_pc, quality, name, bass_pc


def chord_sequence_to_numerals(chord_events: list, tpb: int) -> list:
    """
    Convierte la secuencia de acordes en una lista de dicts con tónica, numeral,
    nombre (con inversión si corresponde) y duración en beats.

    Acepta el formato extendido (pcs, bass_midi, start_tick, dur_ticks) devuelto
    por midi_to_chord_sequence, así como el formato legado (pcs, start, dur).
    """
    recognized = []
    for item in chord_events:
        if len(item) == 4:
            pcs, bass_midi, start, dur = item
        else:                          # compatibilidad legado
            pcs, start, dur = item; bass_midi = None

        root_pc, quality, chord_name, bass_pc = pcs_to_chord_name(pcs, bass_midi)
        dur_beats = dur / tpb
        recognized.append({
            'root_pc':    root_pc,
            'quality':    quality,
            'chord_name': chord_name,
            'bass_pc':    bass_pc,
            'dur_beats':  dur_beats,
            'start':      start / tpb,
        })
    return recognized


# Perfiles de Krumhansl-Schmuckler (correlacion notas de escala con tónica)
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def infer_tonic(recognized: list) -> int:
    """
    Estima la tónica mas probable usando dos metodos combinados:

    1. Krumhansl-Schmuckler: correlacion del perfil de duraciones de pitch-classes
       con los perfiles de escala mayor y menor (12 tónicas x 2 modos).
    2. Funcionalidad armónica: bonificacion a tónicas cuyo I, IV y V
       coincidan con acordes de larga duracion.

    El resultado es la tónica con mayor puntuacion combinada.
    """
    if not recognized:
        return 0

    # Acumular duracion de cada pitch-class, clippeando outliers (>4x mediana)
    # para que notas pedal muy largas no dominen la inferencia de tónica.
    durs = [c['dur_beats'] for c in recognized if c['root_pc'] is not None]
    if not durs:
        return 0
    sorted_durs = sorted(durs)
    median_dur  = sorted_durs[len(sorted_durs) // 2]
    dur_cap     = max(median_dur * 4, 1.0)   # cap: 4x la mediana

    pc_dur = [0.0] * 12
    for c in recognized:
        if c['root_pc'] is None:
            continue
        quality  = c['quality'] or ''
        ivs      = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[''])
        clipped  = min(c['dur_beats'], dur_cap)
        for iv in ivs:
            pc = (c['root_pc'] + iv) % 12
            pc_dur[pc] += clipped

    total_dur = sum(pc_dur) or 1.0
    pc_norm = [d / total_dur for d in pc_dur]

    # Media y desv. de los perfiles KS para correlacion de Pearson
    def pearson(profile, observed):
        n = len(profile)
        mp = sum(profile) / n
        mo = sum(observed) / n
        num = sum((profile[i]-mp)*(observed[i]-mo) for i in range(n))
        dp  = (sum((profile[i]-mp)**2 for i in range(n)) ** 0.5) or 1e-9
        do  = (sum((observed[i]-mo)**2 for i in range(n)) ** 0.5) or 1e-9
        return num / (dp * do)

    # Score KS para cada tónica y modo (rotacion del perfil)
    ks_scores = {}
    for t in range(12):
        obs   = pc_norm[t:] + pc_norm[:t]   # rotar para que t sea el 0
        r_maj = pearson(_KS_MAJOR, obs)
        r_min = pearson(_KS_MINOR, obs)
        ks_scores[t] = max(r_maj, r_min)

    # Normalizar KS a [0,1]
    ks_min = min(ks_scores.values())
    ks_max = max(ks_scores.values()) - ks_min or 1e-9
    ks_norm = {t: (v - ks_min) / ks_max for t, v in ks_scores.items()}

    # Bonificacion funcional: I(0), IV(5), V(7) con mayor duracion → tónica candidata
    root_dur = {}
    for c in recognized:
        if c['root_pc'] is not None:
            clipped = min(c['dur_beats'], dur_cap)
            root_dur[c['root_pc']] = root_dur.get(c['root_pc'], 0) + clipped
    max_root_dur = max(root_dur.values()) if root_dur else 1.0

    func_scores = {}
    for t in range(12):
        i_dur  = root_dur.get(t,            0) / max_root_dur
        iv_dur = root_dur.get((t + 5) % 12, 0) / max_root_dur
        v_dur  = root_dur.get((t + 7) % 12, 0) / max_root_dur
        # I cuenta doble (es la tónica), IV y V suman
        func_scores[t] = 2 * i_dur + iv_dur + v_dur

    func_max = max(func_scores.values()) or 1e-9
    func_norm = {t: v / func_max for t, v in func_scores.items()}

    # Score de "calidad de numerales": cuantos acordes del MIDI serian grados
    # comunes del catalogo si T fuera la tónica.
    # Grados muy comunes en el catalogo: I(0), ii(2,m), IV(5), V(7), vi(9,m)
    # Grados raros/prestados: bII, bIII, bV, bVI, bVII
    # La idea: F-C-Gm-Bb en tónica F → I, V, ii, IV (todos comunes = alta puntuacion)
    #           en tónica C → IV, I, v(raro), bVII(prestado = baja puntuacion)
    COMMON_DEGREES = {
        # (intervalo, es_menor): peso
        (0,  False): 3.0,   # I
        (2,  True):  2.5,   # ii
        (4,  True):  1.5,   # iii
        (5,  False): 2.5,   # IV
        (7,  False): 3.0,   # V
        (7,  True):  0.5,   # v (poco comun)
        (9,  True):  2.0,   # vi
        (11, True):  1.0,   # vii°
        # menores como tónica
        (0,  True):  3.0,   # i
        (2,  False): 1.5,   # II (dorico)
        (3,  False): 1.5,   # bIII
        (5,  True):  2.0,   # iv
        (10, False): 1.5,   # bVII (mixolidio/aeolico)
        (8,  False): 1.0,   # bVI
    }

    catalog_scores = {}
    for t in range(12):
        score_c = 0.0
        for c in recognized:
            if c['root_pc'] is None:
                continue
            iv   = (c['root_pc'] - t) % 12
            is_m = (c['quality'] or '') in ('m', 'm7', 'm9', 'hdim7', 'dim', 'dim7')
            w    = COMMON_DEGREES.get((iv, is_m), 0.0)
            score_c += w * min(c['dur_beats'], dur_cap)
        catalog_scores[t] = score_c

    cat_max = max(catalog_scores.values()) or 1e-9
    cat_norm = {t: v / cat_max for t, v in catalog_scores.items()}

    # Puntuacion combinada: 25% KS + 15% funcional + 60% calidad de numerales
    combined = {t: 0.25 * ks_norm[t] + 0.15 * func_norm[t] + 0.60 * cat_norm[t]
                for t in range(12)}
    return max(combined, key=lambda k: combined[k])


def chord_to_numeral(root_pc: int, quality: str, tonic_pc: int) -> str:
    """
    Convierte (root_pc, quality) en numeral romano relativo a tonic_pc.
    """
    interval = (root_pc - tonic_pc) % 12
    # Tabla inversa: (interval, quality) → numeral
    for numeral, (iv, q) in NUMERAL_TO_INTERVAL.items():
        if iv == interval and q == quality:
            return numeral
    # Fallback: solo por intervalo
    INTERVAL_TO_NUMERAL = {
        0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III', 5: 'IV',
        6: 'bV', 7: 'V', 8: 'bVI', 9: 'VI', 10: 'bVII', 11: 'VII',
    }
    base = INTERVAL_TO_NUMERAL.get(interval, '?')
    if quality in ('m', 'm7', 'm9', 'hdim7'):
        base = base.lower()
    return base


# Tabla de equivalencias entre numerales (distintas grafías del mismo grado)
_NUMERAL_ALIASES = {
    'I':   {'I','IM7','I6','Isus4','IM9','I7'},
    'i':   {'i','im7','im9','i7','i6'},
    'II':  {'II','II7'},
    'ii':  {'ii','ii7','IIø7','ii°7','ii°'},
    'III': {'III','III7','bIII','bIII7'},
    'iii': {'iii','iii7'},
    'IV':  {'IV','IVM7','IV7','IV/I'},
    'iv':  {'iv','iv7','iv9'},
    'V':   {'V','V7','Vsus4','V7b9','V/I'},
    'v':   {'v'},
    'VI':  {'VI','VI7'},
    'vi':  {'vi','vi7','vi°'},
    'VII': {'VII','VII7'},
    'vii': {'vii','vii°','vii°7'},
    'bII': {'bII','bII7','bii'},
    'bIII':{'bIII','bIII7'},
    'bVI': {'bVI','bvi'},
    'bVII':{'bVII','bvii','bVII7'},
}
# Índice inverso: numeral_exacto → canonical
_CANON = {}
for canon, aliases in _NUMERAL_ALIASES.items():
    for a in aliases:
        _CANON[a] = canon

def _canonicalize(num: str) -> str:
    """Reduce un numeral a su forma canónica (sin extensiones)."""
    return _CANON.get(num, num)


def match_pattern_in_sequence(midi_numerals: list, pattern: list,
                               tonic_pc: int) -> list:
    """
    Busca todas las ocurrencias del patron de una entrada del catalogo
    dentro de la secuencia de numerales del MIDI.

    Score final = accuracy * coverage_penalty * length_bonus
    - accuracy:        fraccion de acordes reconocidos que coinciden
    - coverage_penalty: penaliza si hay muchos ? en la ventana
    - length_bonus:    premia patrones largos (evita que IV-I domine)

    Solo se devuelve el mejor hit por patron (el de mayor score).
    """
    pat_numerals = [_canonicalize(n) for n, _ in pattern]
    midi_canon   = [_canonicalize(n) if n != '?' else '?' for n in midi_numerals]
    n, m = len(pat_numerals), len(midi_canon)
    if n > m:
        return []

    # length_bonus: patrones de 2 acordes tienen bonus 0.7, de 4+ tienen 1.0
    # formula: 0.7 + 0.3 * min(1, (n-2)/3)  → llega a 1.0 en n>=5
    length_bonus = 0.75 + 0.25 * min(1.0, max(0.0, (n - 2) / 3.0))

    all_hits = []

    for i in range(m - n + 1):
        window     = midi_canon[i:i + n]
        comparable = [(a, b) for a, b in zip(window, pat_numerals) if a != '?']
        # Necesitamos al menos ceil(n/2) acordes comparables
        if len(comparable) < max(2, (n + 1) // 2):
            continue
        match    = sum(1 for a, b in comparable if a == b)
        accuracy = match / len(comparable)
        coverage = len(comparable) / n

        # Bonus si el primer acorde del patron coincide exactamente
        first_match = 1.0 if (window[0] != '?' and window[0] == pat_numerals[0]) else 0.85

        # Penalizar coverage baja + bonus por primer acorde + bonus por longitud
        adj = accuracy * (0.5 + 0.5 * coverage) * length_bonus * first_match
        if adj >= 0.62:
            all_hits.append((i, round(adj, 3)))

    # Devolver todos los hits ordenados por score desc
    all_hits.sort(key=lambda x: -x[1])
    return all_hits


def analyze_midi(midi_path: str, tonic_override: int = None,
                 min_score: float = 0.6, verbose: bool = False,
                 bass_only: bool = False, min_dur: float = 0.0) -> list:
    """
    Función principal de análisis.
    Devuelve lista de matches: cada uno es un dict con entrada del catálogo
    y la información de coincidencia.

    Si bass_only=True, solo las notas por debajo de C#4 (MIDI 61) se usan
    para detectar los acordes, lo que permite analizar la línea de bajo
    de forma independiente de la melodía. C4 queda incluido en el bajo.
    """
    chord_events, tpb = midi_to_chord_sequence(midi_path, bass_only=bass_only)
    recognized = chord_sequence_to_numerals(chord_events, tpb)

    if not recognized:
        return []

    # Filtrar acordes con duración inferior al umbral --min-dur
    if min_dur > 0:
        recognized = [c for c in recognized if c['dur_beats'] >= min_dur]
    if not recognized:
        return []

    tonic_pc = tonic_override if tonic_override is not None else infer_tonic(recognized)
    tonic_name = PITCH_NAMES[tonic_pc]

    # Convertir cada acorde reconocido a numeral relativo
    midi_numerals = []
    for c in recognized:
        if c['root_pc'] is not None:
            num = chord_to_numeral(c['root_pc'], c['quality'] or '', tonic_pc)
        else:
            num = '?'
        midi_numerals.append(num)

    # Secuencia solo con acordes reconocidos (sin '?') para display limpio
    clean_numerals = [n for n in midi_numerals if n != '?']

    # Inferir compas predominante a partir de la mediana de duraciones de acordes.
    dur_list = sorted(c['dur_beats'] for c in recognized if c['dur_beats'] > 0)
    median_dur = dur_list[len(dur_list) // 2] if dur_list else 4.0
    beats_per_measure = min([2, 4, 8], key=lambda x: abs(x - median_dur))

    def beat_to_measure(beat: float) -> int:
        return int(beat / beats_per_measure) + 1

    # Buscar coincidencias agrupando todas las ocurrencias por entrada del catalogo
    by_entry = {}
    pat_set_cache = {}

    for entry in TABLE:
        hits = match_pattern_in_sequence(midi_numerals, entry['pattern'], tonic_pc)
        if not hits:
            continue
        eid = entry['id']
        pat_set = pat_set_cache.setdefault(
            eid, set(_canonicalize(n) for n, _ in entry['pattern'])
        )
        midi_unique = set(_canonicalize(n) for n in clean_numerals if n != '?')
        midi_cov  = len(midi_unique & pat_set) / max(len(midi_unique), 1)
        pat_prec  = len(midi_unique & pat_set) / max(len(pat_set), 1)
        f1_cov    = (2 * midi_cov * pat_prec / (midi_cov + pat_prec)
                     if (midi_cov + pat_prec) > 0 else 0.0)

        best_idx, best_score = hits[0]
        window = midi_numerals[best_idx:best_idx + len(entry['pattern'])]

        # Calcular compas de inicio de cada ocurrencia (deduplicado)
        measures = []
        seen_m = set()
        for idx, _ in hits:
            start_beat = recognized[idx]['start'] if idx < len(recognized) else 0.0
            m_num = beat_to_measure(start_beat)
            if m_num not in seen_m:
                measures.append(m_num)
                seen_m.add(m_num)
        measures.sort()

        by_entry[eid] = {
            'entry':             entry,
            'tonic_pc':          tonic_pc,
            'tonic_name':        tonic_name,
            'score':             best_score,
            'midi_coverage':     round(f1_cov, 3),
            'start_index':       best_idx,
            'matched_numerals':  [n for n in window if n != '?'],
            'beats_per_measure': beats_per_measure,
            'measures':          measures,
        }

    all_matches = list(by_entry.values())

    # Ordenar: score desc, luego midi_coverage desc, luego longitud desc, luego id
    all_matches.sort(key=lambda x: (
        -round(x['score'] + x.get('midi_coverage', 0) * 0.15, 3),
        -len(x['entry']['pattern']),
        x['entry']['id']
    ))

    # Filtrar por min_score
    all_matches = [m for m in all_matches if m['score'] >= min_score]

    return all_matches, recognized, clean_numerals, tonic_name


def print_midi_analysis(midi_path: str, matches: list, recognized: list,
                         midi_numerals: list, tonic_name: str,
                         verbose: bool = False, min_score: float = 0.6,
                         bass_only: bool = False,
                         sort_compas: bool = False) -> None:
    """Imprime el resultado del análisis MIDI de forma legible."""
    print(f"\n{COL['cyan']}{'═'*72}{COL['reset']}")
    print(f"{COL['bold']}  ANÁLISIS MIDI: {os.path.basename(midi_path)}{COL['reset']}")
    print(f"{COL['cyan']}{'═'*72}{COL['reset']}\n")
    if bass_only:
        print(f"  {COL['yellow']}[modo --bass-only: solo notas graves ≤ C4]{COL['reset']}\n")

    print(f"  {COL['bold']}Tónica inferida:{COL['reset']} {COL['yellow']}{tonic_name}{COL['reset']}")
    print(f"  {COL['bold']}Acordes detectados:{COL['reset']} {len(recognized)}")

    # Mostrar secuencia detectada
    chord_str = '  →  '.join(
        f"{COL['cyan']}{c['chord_name']}{COL['reset']} "
        f"{COL['gray']}({c['dur_beats']:.1f}b){COL['reset']}"
        for c in recognized
    )
    print(f"\n  {COL['bold']}Secuencia:{COL['reset']}")
    # Partir en líneas de ≤ 5 acordes
    for i in range(0, len(recognized), 5):
        chunk = recognized[i:i+5]
        line  = '  →  '.join(
            f"{COL['cyan']}{c['chord_name']}{COL['reset']}"
            f"{COL['gray']}({c['dur_beats']:.0f}b){COL['reset']}"
            for c in chunk
        )
        print(f"    {line}")

    print(f"\n  {COL['bold']}Numerales (tónica {tonic_name}):{COL['reset']}")
    nums_str = ' – '.join(midi_numerals)
    print(f"    {COL['yellow']}{nums_str}{COL['reset']}")

    filtered = [m for m in matches if m['score'] >= min_score]
    print(f"\n  {COL['bold']}Progresiones del catálogo encontradas:{COL['reset']} {len(filtered)}\n")

    if not filtered:
        print(f"  {COL['gray']}Sin coincidencias con score ≥ {min_score:.0%}{COL['reset']}\n")
        return

    # En modo sort_compas: expandir cada match en una fila por compás,
    # ordenar el conjunto por compás de aparición.
    if sort_compas:
        rows = []
        for m in filtered:
            for meas in (m.get('measures') or [None]):
                rows.append((meas, m))
        rows.sort(key=lambda x: (x[0] if x[0] is not None else 0))
    else:
        rows = [(None, m) for m in filtered]

    col_compas = 'compás' if sort_compas else 'compases'
    print(f"  {COL['gray']}{'score':>6}  {'cob':>4}  {'#':>3}  {'progresión':<36}  "
          f"{'t':>2}  {'emoción':<12}  {'estilo':<12} {'modo':<10} {col_compas}{COL['reset']}")
    print(f"  {'─'*100}")

    for meas_single, m in rows:
        e   = m['entry']
        sc  = m['score']
        cov = m.get('midi_coverage', 0)
        sc_col = COL['green'] if sc >= 0.75 and cov >= 0.6 else \
                 COL['yellow'] if sc >= 0.65 else COL['gray']
        tc = tension_col(e['tension'])
        if sort_compas:
            meas_str = f'c.{meas_single}' if meas_single is not None else ''
        else:
            measures = m.get('measures', [])
            meas_str = ('c.' + ', '.join(str(x) for x in measures)) if measures else ''
        print(
            f"  {sc_col}{sc:>5.0%}{COL['reset']}  "
            f"{COL['gray']}{cov:>3.0%}{COL['reset']}  "
            f"{COL['gray']}#{e['id']:>3}{COL['reset']}  "
            f"{COL['bold']}{e['prog']:<36}{COL['reset']}  "
            f"{tc}{e['tension']}{COL['reset']}  "
            f"{COL['cyan']}{e['emocion']:<12}{COL['reset']}  "
            f"{COL['gray']}{e['style']:<12} {e['mode']:<10} "
            f"{COL['yellow']}{meas_str}{COL['reset']}"
        )
        if verbose:
            print(f"         {COL['gray']}{e['nombre']}{COL['reset']}")
            print(f"         {e['desc']}")
            print(f"         Numerales coincidentes: {' – '.join(m['matched_numerals'])}")
            print()

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRADO Y BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    for src, dst in {'á':'a','é':'e','í':'i','ó':'o','ú':'u','ü':'u','ñ':'n'}.items():
        s = s.lower().replace(src, dst)
    return s


def filter_table(style=None, mode=None, emocion=None,
                 tension_min=None, tension_max=None,
                 buscar=None) -> list:
    """Filtra el catálogo según los criterios dados."""
    result = list(TABLE)
    if style:
        result = [e for e in result if e['style'] == style]
    if mode:
        result = [e for e in result if e['mode'] == mode]
    if emocion:
        result = [e for e in result if e['emocion'] == emocion]
    if tension_min is not None:
        result = [e for e in result if e['tension'] >= tension_min]
    if tension_max is not None:
        result = [e for e in result if e['tension'] <= tension_max]
    if buscar:
        q = _normalize(buscar)
        result = [
            e for e in result
            if q in _normalize(e['prog'])
            or q in _normalize(e['nombre'])
            or q in _normalize(e['desc'])
            or q in _normalize(e['emocion'])
            or q in _normalize(e['style'])
            or q in _normalize(e['mode'])
        ]
    return result


def parse_intent(description: str) -> dict:
    """
    Extrae filtros desde una descripción en lenguaje natural.
    Devuelve dict con claves opcionales: emocion, tension_min, tension_max.
    """
    desc = _normalize(description)
    filters = {}

    # Detectar rango de tensión
    for phrase, (tmin, tmax) in INTENT_TO_TENSION.items():
        if _normalize(phrase) in desc:
            filters['tension_min'] = tmin
            filters['tension_max'] = tmax
            break

    # Detectar emoción
    for word, emociones in INTENT_TO_EMOCION.items():
        if _normalize(word) in desc:
            filters['emocion'] = emociones[0]
            break

    return filters


# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

COL = {
    'reset': '\033[0m', 'bold': '\033[1m',
    'cyan':  '\033[36m', 'yellow': '\033[33m',
    'green': '\033[32m', 'red': '\033[31m',
    'gray':  '\033[90m', 'blue': '\033[34m',
    'magenta': '\033[35m',
}

TENSION_COLOR = {
    range(1, 4):  COL['green'],
    range(4, 7):  COL['yellow'],
    range(7, 11): COL['red'],
}

def tension_col(t: int) -> str:
    for r, c in TENSION_COLOR.items():
        if t in r:
            return c
    return COL['reset']

def print_entry(e: dict, verbose: bool = False) -> None:
    tc = tension_col(e['tension'])
    print(
        f"  {COL['gray']}#{e['id']:2d}{COL['reset']}  "
        f"{COL['bold']}{e['prog']:<38}{COL['reset']} "
        f"{tc}t={e['tension']}{COL['reset']}  "
        f"{COL['cyan']}{e['emocion']:<12}{COL['reset']}  "
        f"{COL['gray']}{e['style']:<12} {e['mode']:<18}{COL['reset']}  "

    )
    if verbose:
        print(f"        {COL['gray']}{e['nombre']}{COL['reset']}")
        print(f"        {e['desc']}")


def print_table(entries: list, verbose: bool = False) -> None:
    print(
        f"\n  {COL['gray']}{'#':>3}  {'progresión':<38} {'t':>2}  "
        f"{'emoción':<12}  {'estilo':<12} {'modo':<18}{COL['reset']}"
    )
    print(f"  {'─'*88}")
    for e in entries:
        print_entry(e, verbose)
    print(f"\n  {len(entries)} progresión(es)\n")


def print_stats() -> None:
    print(f"\n{COL['cyan']}{'═'*60}{COL['reset']}")
    print(f"{COL['bold']}  CHORD TABLE — Estadísticas del catálogo{COL['reset']}")
    print(f"{COL['cyan']}{'═'*60}{COL['reset']}\n")

    from collections import Counter
    styles   = Counter(e['style']   for e in TABLE)
    modes    = Counter(e['mode']    for e in TABLE)
    emocions = Counter(e['emocion'] for e in TABLE)
    tensions = [e['tension'] for e in TABLE]

    print(f"  Total progresiones: {len(TABLE)}")
    print(f"  Tensión media: {sum(tensions)/len(tensions):.1f}  "
          f"min={min(tensions)}  max={max(tensions)}\n")

    print(f"  {COL['bold']}Por estilo:{COL['reset']}")
    for k, v in sorted(styles.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por modo:{COL['reset']}")
    for k, v in sorted(modes.items(), key=lambda x: -x[1]):
        print(f"    {k:<22} {v:>3}")

    print(f"\n  {COL['bold']}Por emoción:{COL['reset']}")
    for k, v in sorted(emocions.items(), key=lambda x: -x[1]):
        print(f"    {k:<16} {v:>3}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Catálogo curado de progresiones de acordes con tensión y emoción.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Análisis MIDI
    parser.add_argument('--analyze-midi', type=str, metavar='MIDI_FILE',
                        help='Analizar un MIDI y mostrar las progresiones del '
                             'catálogo que contiene')
    parser.add_argument('--min-score',   type=float, default=0.65, metavar='F',
                        help='Score mínimo de coincidencia 0-1 (default: 0.6)')

    # Entrada / búsqueda
    parser.add_argument('--list',      action='store_true',
                        help='Listar todas las progresiones')
    parser.add_argument('--stats',     action='store_true',
                        help='Estadísticas del catálogo')
    parser.add_argument('--custom',    type=str, metavar='GRADOS',
                        help="Progresión personalizada en numerales romanos, "
                             "en vez de seleccionar del catálogo. "
                             "Ej: --custom \"I vi IV V\". "
                             "Duración opcional por grado con ':beats' "
                             "(ej: \"I:4 V:2 vi:2 IV:4\"); sin ella se usa "
                             "--beats por grado. Se resuelve en la tónica "
                             "dada por --key igual que --id.")
    parser.add_argument('--id',        type=int, metavar='N',
                        help='Seleccionar progresión por id')
    parser.add_argument('--buscar',    type=str, metavar='TEXTO',
                        help='Búsqueda por texto libre')
    parser.add_argument('--intencion', type=str, metavar='DESC',
                        help='Buscar por intención en lenguaje natural')

    # Filtros
    parser.add_argument('--style',       type=str,
                        choices=['diatonic','baroque','jazz','modal','romantic',
                                 'impressionist','pop','flamenco','blues'],
                        help='Filtrar por estilo')
    parser.add_argument('--mode',        type=str,
                        choices=['major','minor','dorian','phrygian',
                                 'phrygian_dominant','lydian','mixolydian'],
                        help='Filtrar por modo')
    parser.add_argument('--emocion',     type=str,
                        help='Filtrar por emoción (reposo, alegría, tristeza…)')
    parser.add_argument('--tension-min', type=int, metavar='N', dest='tension_min',
                        help='Tensión mínima (1-10)')
    parser.add_argument('--tension-max', type=int, metavar='N', dest='tension_max',
                        help='Tensión máxima (1-10)')

    # Resolución y exportación
    parser.add_argument('--key',         type=str, default='C',
                        help='Tónica para resolución concreta (default: C)')
    parser.add_argument('--reps',        type=int, default=None, metavar='N',
                        help='Número de repeticiones exactas de la progresión/ciclo '
                             'a exportar (sustituye a --bars; no trunca el último ciclo)')
    parser.add_argument('--bars',        type=int, default=8,
                        help='Compases a generar (default: 8)')
    parser.add_argument('--beats',       type=int, default=4,
                        help='Pulsos por compás (default: 4)')
    parser.add_argument('--tempo',       type=float, default=120.0,
                        help='Tempo BPM para exportación MIDI (default: 120)')
    parser.add_argument('--export-json', type=str, metavar='FILE',
                        help='Exportar tabla filtrada o progresión a JSON')
    parser.add_argument('--export-text', type=str, metavar='FILE',
                        help='Exportar progresión concreta a texto --chords')
    parser.add_argument('--export-midi', type=str, metavar='FILE',
                        help='Exportar progresión concreta a MIDI')
    parser.add_argument('--output',      type=str, default='obra',
                        help='Nombre base para salidas (default: obra)')
    parser.add_argument('--bass-only',   action='store_true', dest='bass_only',
                        help='Analizar solo notas graves (≤ C4) para la detección '
                             'de acordes — útil para aislar la línea de bajo')
    parser.add_argument('--min-dur',      type=float, default=0.0, metavar='BEATS',
                        dest='min_dur',
                        help='Duración mínima en beats para considerar un acorde '
                             '(default: 0, sin filtro). Útil para ignorar notas '
                             'de paso y acordes fugaces. Ej: --min-dur 2')
    parser.add_argument('--sort-compas', action='store_true', dest='sort_compas',
                        help='Ordenar resultados por compás de aparición; '
                             'cuando una progresión aparece varias veces se muestra '
                             'una fila por aparición en lugar de agruparlas')
    parser.add_argument('--verbose',     action='store_true',
                        help='Mostrar nombre y descripción de cada progresión')

    args = parser.parse_args()

    # ── Análisis MIDI ────────────────────────────────────────────────────────
    if args.analyze_midi:
        if not os.path.isfile(args.analyze_midi):
            print(f"[ERROR] No se encuentra el archivo: {args.analyze_midi}")
            sys.exit(1)

        # Tónica manual (opcional)
        tonic_override = None
        if args.key and args.key != 'C':
            k = args.key.strip()
            tonic_override = NOTE_PC.get(k[:-1] if k.endswith('m') else k, None)

        result = analyze_midi(
            args.analyze_midi,
            tonic_override = tonic_override,
            min_score      = args.min_score,
            verbose        = args.verbose,
            bass_only      = args.bass_only,
            min_dur        = args.min_dur,
        )
        matches, recognized, midi_numerals, tonic_name = result

        print_midi_analysis(
            args.analyze_midi,
            matches,
            recognized,
            midi_numerals,
            tonic_name,
            verbose      = args.verbose,
            min_score    = args.min_score,
            bass_only    = args.bass_only,
            sort_compas  = args.sort_compas,
        )

        if args.export_json:
            exportable = []
            for m in matches:
                if m['score'] >= args.min_score:
                    exportable.append({
                        'score':    m['score'],
                        'tonic':    m['tonic_name'],
                        'matched_numerals': m['matched_numerals'],
                        'catalog':  {k: v for k, v in m['entry'].items()
                                     if k != 'pattern'},
                    })
            with open(args.export_json, 'w') as f:
                json.dump(exportable, f, indent=2, ensure_ascii=False)
            print(f"  → JSON: {args.export_json}\n")
        return

    # ── Stats ────────────────────────────────────────────────────────────────
    if args.stats:
        print_stats()
        return

    # ── Resolver tónica ──────────────────────────────────────────────────────
    key_str = args.key.strip()
    if key_str.endswith('m') and len(key_str) <= 3:
        tonic_pc = NOTE_PC.get(key_str[:-1], 0)
    else:
        tonic_pc = NOTE_PC.get(key_str, 0)

    # ── Validación: --id y --custom son mutuamente excluyentes ──────────────
    if args.id is not None and args.custom is not None:
        print("[ERROR] --id y --custom no se pueden usar a la vez")
        sys.exit(1)

    # ── Selección por id o por progresión personalizada ──────────────────────
    if args.id is not None or args.custom is not None:
        if args.custom is not None:
            try:
                entry = make_custom_entry(args.custom, default_beats=args.beats)
            except ValueError as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
            print(f"\n  {COL['bold']}{entry['prog']}{COL['reset']}  "
                  f"{COL['gray']}(custom){COL['reset']}")
        else:
            entry = TABLE_BY_ID.get(args.id)
            if not entry:
                print(f"[ERROR] No existe progresión con id={args.id}")
                sys.exit(1)
            print(f"\n  #{entry['id']}  {COL['bold']}{entry['prog']}{COL['reset']}")
            print(f"  {entry['nombre']}  ·  {entry['style']} / {entry['mode']}")
            print(f"  Tensión: {entry['tension']}/10  ·  Emoción: {entry['emocion']}")
            print(f"  {entry['desc']}\n")

        progression = resolve_entry(entry, tonic_pc, args.bars, args.beats,
                                    reps=args.reps)
        chord_text  = progression_to_text(progression)
        print(f"  → {COL['cyan']}{chord_text}{COL['reset']}\n")

        # Exportaciones
        json_path = args.export_json or f"{args.output}.chords.json"
        txt_path  = args.export_text or f"{args.output}.chords.txt"
        mid_path  = args.export_midi or f"{args.output}.chords.mid"

        out = {
            'meta':        {k: v for k, v in entry.items() if k != 'pattern'},
            'tonic':       PITCH_NAMES[tonic_pc],
            'bars':        args.bars,
            'beats':       args.beats,
            'reps':        args.reps,
            'tempo':       args.tempo,
            'chord_string': chord_text,
            'progression': progression,
            'generator':   'chord_table.py v1.0',
        }
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  → JSON:  {json_path}")

        with open(txt_path, 'w') as f:
            f.write(chord_text + '\n')
        print(f"  → Texto: {txt_path}")

        if args.export_midi or True:
            if MIDO_OK:
                progression_to_midi(progression, tonic_pc, 480, args.tempo, mid_path)
                print(f"  → MIDI:  {mid_path}")
            else:
                print("  [AVISO] mido no instalado — MIDI omitido. pip install mido")
        return

    # ── Búsqueda por intención ───────────────────────────────────────────────
    if args.intencion:
        intent_filters = parse_intent(args.intencion)
        entries = filter_table(
            style       = args.style,
            mode        = args.mode,
            emocion     = intent_filters.get('emocion', args.emocion),
            tension_min = intent_filters.get('tension_min', args.tension_min),
            tension_max = intent_filters.get('tension_max', args.tension_max),
            buscar      = args.buscar,
        )
        if not entries:
            print(f"\n  Sin resultados para: \"{args.intencion}\"\n")
            sys.exit(0)
        print(f"\n  Resultados para: \"{args.intencion}\"")
        print_table(entries, args.verbose)
        if args.export_json:
            with open(args.export_json, 'w') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False, default=str)
            print(f"  → JSON: {args.export_json}")
        return

    # ── Filtrado general + listado ───────────────────────────────────────────
    entries = filter_table(
        style       = args.style,
        mode        = args.mode,
        emocion     = args.emocion,
        tension_min = args.tension_min,
        tension_max = args.tension_max,
        buscar      = args.buscar,
    )

    if not entries and not args.list:
        parser.print_help()
        return

    print_table(entries if entries else TABLE, args.verbose)

    if args.export_json:
        data = entries if entries else TABLE
        exportable = [{k: v for k, v in e.items() if k != 'pattern'}
                      for e in data]
        with open(args.export_json, 'w') as f:
            json.dump(exportable, f, indent=2, ensure_ascii=False)
        print(f"  → JSON: {args.export_json}\n")


if __name__ == '__main__':
    main()
