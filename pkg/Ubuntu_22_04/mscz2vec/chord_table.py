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


def resolve_entry(entry: dict, tonic_pc: int, bars: int = 8,
                  beats_per_bar: int = 4) -> list:
    """
    Expande el patrón de una entrada del catálogo a una lista de acordes
    concretos repetidos para cubrir `bars` compases.
    Devuelve lista de dicts compatibles con chord_progression_generator.
    """
    pattern   = entry['pattern']
    beats_pat = sum(d for _, d in pattern)
    total     = bars * beats_per_bar

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
        chord_name, pitches = resolve_numeral(numeral, tonic_pc)
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

    # Entrada / búsqueda
    parser.add_argument('--list',      action='store_true',
                        help='Listar todas las progresiones')
    parser.add_argument('--stats',     action='store_true',
                        help='Estadísticas del catálogo')
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
    parser.add_argument('--verbose',     action='store_true',
                        help='Mostrar nombre y descripción de cada progresión')

    args = parser.parse_args()

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

    # ── Selección por id ─────────────────────────────────────────────────────
    if args.id is not None:
        entry = TABLE_BY_ID.get(args.id)
        if not entry:
            print(f"[ERROR] No existe progresión con id={args.id}")
            sys.exit(1)

        print(f"\n  #{entry['id']}  {COL['bold']}{entry['prog']}{COL['reset']}")
        print(f"  {entry['nombre']}  ·  {entry['style']} / {entry['mode']}")
        print(f"  Tensión: {entry['tension']}/10  ·  Emoción: {entry['emocion']}")

        print(f"  {entry['desc']}\n")

        progression = resolve_entry(entry, tonic_pc, args.bars, args.beats)
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
