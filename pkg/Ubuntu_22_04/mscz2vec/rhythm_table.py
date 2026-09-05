#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RHYTHM TABLE  v1.0                                    ║
║         Catálogo curado de patrones rítmicos con tensión y emoción           ║
║                                                                              ║
║  Complemento de chord_table.py: en lugar de acordes, ofrece un catálogo     ║
║  curado de patrones rítmicos —células de una sola línea y grooves          ║
║  multi-voz— con etiquetas de tensión (1-10) y emoción, buscables y          ║
║  exportables. Un patrón puede usarse como base percusiva independiente o    ║
║  como "ritmo armónico" (puntos de cambio de acorde) aplicado sobre una      ║
║  progresión de chord_table.py.                                              ║
║                                                                              ║
║    · Tipos:    célula (una línea de onsets), groove (varias voces)         ║
║    · Estilos:  afrocubano, tango, habanera, balcánico, flamenco, jazz,      ║
║                blues, bossa, baião, rock, hip_hop, reggaeton, drum_and_bass ║
║    · Usos:     percusión, ritmo_armonico                                    ║
║    · Emociones: reposo, alegría, tristeza, melancolía, nostalgia,           ║
║                 misterio, ambigüedad, drama, angustia, esperanza,           ║
║                 euforia, flotación, solemnidad, exotismo                    ║
║                                                                              ║
║  USO:                                                                        ║
║    # Listar todos                                                           ║
║    python rhythm_table.py --list                                            ║
║                                                                              ║
║    # Filtrar por emoción / estilo / tipo / uso                              ║
║    python rhythm_table.py --emocion euforia                                 ║
║    python rhythm_table.py --style flamenco --tipo celula                    ║
║    python rhythm_table.py --uso ritmo_armonico                              ║
║                                                                              ║
║    # Filtrar por rango de tensión                                           ║
║    python rhythm_table.py --tension-min 7 --tension-max 10                  ║
║                                                                              ║
║    # Buscar por texto libre / por intención en lenguaje natural             ║
║    python rhythm_table.py --buscar "clave"                                  ║
║    python rhythm_table.py --intencion "urgente y denso"                     ║
║                                                                              ║
║    # Resolver un patrón del catálogo a tiempo real y exportar               ║
║    python rhythm_table.py --id 1 --tempo 100 --bars 4 --export-midi c.mid   ║
║                                                                              ║
║    # Patrón personalizado — célula (onsets 'x'/'.')                        ║
║    python rhythm_table.py --custom "x..x..x." --meter 4/4 --bars 2          ║
║                                                                              ║
║    # Patrón personalizado — groove multi-voz                                ║
║    python rhythm_table.py --custom "kick:x...x... snare:....x..." \\        ║
║        --tipo groove --meter 4/4 --export-midi groove.mid                   ║
║                                                                              ║
║    # Ritmo armónico: aplicar los puntos de cambio de un patrón a acordes   ║
║    python rhythm_table.py --id 5 --apply-to-chords "Am G F E7" --bars 2     ║
║        --export-text cambios.chords.txt                                     ║
║                                                                              ║
║    # Exportar tabla filtrada a JSON                                         ║
║    python rhythm_table.py --style bossa --export-json bossa.json            ║
║                                                                              ║
║    # Mostrar estadísticas del catálogo                                      ║
║    python rhythm_table.py --stats                                           ║
║                                                                              ║
║    # Analizar un MIDI y detectar patrones del catálogo                      ║
║    python rhythm_table.py --analyze-midi cancion.mid                        ║
║                                                                              ║
║    # Análisis con resolución fina, compás forzado y exportación JSON        ║
║    python rhythm_table.py --analyze-midi song.mid --meter 4/4 --bars 4 \\   ║
║        --resolution 4 --min-score 0.6 --export-json matches.json --verbose  ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                              ║
║    chord_table.py       → --apply-to-chords toma una progresión suya y le  ║
║                            impone la temporización del patrón rítmico       ║
║    melody_adapter.py    → el texto de --apply-to-chords es compatible      ║
║                            con su entrada --chords                          ║
║    layer_graph_composer.py / adaptive_playback_runtime.py → --export-midi  ║
║                            genera groove de percusión fuente                ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <base>.rhythm.txt    — patrón en notación de onsets (grid legible)      ║
║    <base>.rhythm.json   — patrón resuelto con metadatos                     ║
║    <base>.rhythm.mid    — MIDI de percusión (canal 10 GM)                   ║
║    <base>.chords.txt/.json — con --apply-to-chords, progresión resultante  ║
║                                                                              ║
║  DEPENDENCIAS: mido (opcional, solo para exportación MIDI)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import re
import math
from collections import Counter

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# CATÁLOGO CURADO DE PATRONES RÍTMICOS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cada entrada:
#   id          — identificador único
#   nombre      — nombre descriptivo
#   tipo        — 'celula' (pattern: string de onsets) o 'groove'
#                 (pattern: dict {voz: string de onsets}, todas las voces con
#                 la misma longitud)
#   meter       — compás, ej. '4/4', '7/8', '12/8'
#   bars        — número de compases que abarca UNA repetición del patrón
#   subdivision — informativo: subdivisión nominal (corcheas=8, semicorcheas=16,
#                 tresillos=12, o 1 para métricas aditivas contadas en corcheas)
#   feel        — 'straight', 'swing' o 'shuffle'
#   estilo      — estilo/tradición del patrón
#   tension     — 1-10
#   emocion     — etiqueta emocional
#   uso         — lista con 'percusion' y/o 'ritmo_armonico'
#   desc        — descripción breve
#
# La duración real de cada carácter se calcula en tiempo de resolución a
# partir de meter/bars/longitud del patrón (ver entry_beats_per_char), no de
# 'subdivision' — ese campo es solo documentación para el usuario.

TABLE = [
    {
        'id': 1, 'nombre': 'Clave son 3-2', 'tipo': 'celula',
        'pattern': 'x..x..x...x..x..',
        'meter': '4/4', 'bars': 2, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Clave de son en su forma 3-2: lado de tresillo seguido de lado de dos golpes.',
    },
    {
        'id': 2, 'nombre': 'Clave son 2-3', 'tipo': 'celula',
        'pattern': '..x..x..x..x..x.',
        'meter': '4/4', 'bars': 2, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Clave de son en su forma 2-3: lado de dos golpes seguido de lado de tresillo.',
    },
    {
        'id': 3, 'nombre': 'Tresillo cubano', 'tipo': 'celula',
        'pattern': 'x..x..x.',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 4, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Célula de 3+3+2 corcheas, base de habaneras, tangos y bajos tumbao.',
    },
    {
        'id': 4, 'nombre': 'Cinquillo cubano', 'tipo': 'celula',
        'pattern': 'x.xx.xx.',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 5, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Cinco golpes por compás, típico de comparsa y danzón.',
    },
    {
        'id': 5, 'nombre': 'Habanera', 'tipo': 'celula',
        'pattern': 'x..xx.x.',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'habanera', 'tension': 3, 'emocion': 'nostalgia',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Puntillo + dos corcheas, célula madre de tango y habanera.',
    },
    {
        'id': 6, 'nombre': 'Milonga porteña', 'tipo': 'celula',
        'pattern': 'x.x.xx..',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'tango', 'tension': 5, 'emocion': 'drama',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Variante acelerada de la habanera usada en la milonga rioplatense.',
    },
    {
        'id': 7, 'nombre': 'Aksak 7/8 (3+2+2)', 'tipo': 'celula',
        'pattern': 'x..x.x.',
        'meter': '7/8', 'bars': 1, 'subdivision': 1, 'feel': 'straight',
        'estilo': 'balcanico', 'tension': 7, 'emocion': 'exotismo',
        'uso': ['percusion'],
        'desc': 'Métrica aditiva balcánica, agrupación asimétrica 3+2+2.',
    },
    {
        'id': 8, 'nombre': 'Bulgarian 9/8 (2+2+2+3)', 'tipo': 'celula',
        'pattern': 'x.x.x.x..',
        'meter': '9/8', 'bars': 1, 'subdivision': 1, 'feel': 'straight',
        'estilo': 'balcanico', 'tension': 7, 'emocion': 'misterio',
        'uso': ['percusion'],
        'desc': 'Agrupación aditiva 2+2+2+3, común en danzas búlgaras.',
    },
    {
        'id': 9, 'nombre': 'Compás de soleá', 'tipo': 'celula',
        'pattern': '..x.x..x.x.x',
        'meter': '12/8', 'bars': 1, 'subdivision': 1, 'feel': 'straight',
        'estilo': 'flamenco', 'tension': 8, 'emocion': 'angustia',
        'uso': ['percusion'],
        'desc': 'Acentuación en 3, 6, 8, 10 y 12 sobre ciclo de doce tiempos.',
    },
    {
        'id': 10, 'nombre': 'Compás de bulería', 'tipo': 'celula',
        'pattern': 'x.x.x..x.x.x',
        'meter': '12/8', 'bars': 1, 'subdivision': 1, 'feel': 'straight',
        'estilo': 'flamenco', 'tension': 9, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Ciclo de doce tiempos con acentos desplazados, más urgente que la soleá.',
    },
    {
        'id': 11, 'nombre': 'Charleston', 'tipo': 'celula',
        'pattern': 'x..x....',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'swing',
        'estilo': 'jazz', 'tension': 3, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Anticipación clásica del jazz temprano: tiempo 1 y el "y" del 2.',
    },
    {
        'id': 12, 'nombre': 'Shuffle de blues', 'tipo': 'celula',
        'pattern': 'x.xx.xx.xx.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 12, 'feel': 'shuffle',
        'estilo': 'blues', 'tension': 4, 'emocion': 'nostalgia',
        'uso': ['percusion'],
        'desc': 'Tresillos con acento en el segundo golpe, sensación de "cojeo".',
    },
    {
        'id': 13, 'nombre': 'Ride swing walking', 'tipo': 'celula',
        'pattern': 'x.xx.xx.xx.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 12, 'feel': 'swing',
        'estilo': 'jazz', 'tension': 3, 'emocion': 'flotacion',
        'uso': ['percusion'],
        'desc': 'Patrón de platillo ride "ding-ding-a-ding" en swing moderado.',
    },
    {
        'id': 14, 'nombre': 'Clave de bossa nova', 'tipo': 'celula',
        'pattern': 'x..x..x...x.x...',
        'meter': '4/4', 'bars': 2, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'bossa', 'tension': 3, 'emocion': 'reposo',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Derivada del tresillo, marca los acentos característicos de la bossa.',
    },
    {
        'id': 15, 'nombre': 'Baião', 'tipo': 'celula',
        'pattern': 'x..x.x.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'baiao', 'tension': 5, 'emocion': 'esperanza',
        'uso': ['percusion'],
        'desc': 'Célula del nordeste brasileño, base de zabumba y baixo.',
    },
    {
        'id': 16, 'nombre': 'Rock básico 4/4', 'tipo': 'groove',
        'pattern': {
            'kick':  'x...x...',
            'snare': '....x...',
            'hihat': 'x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'rock', 'tension': 4, 'emocion': 'alegria',
        'uso': ['percusion'],
        'desc': 'Patrón fundacional: bombo en 1 y 3, caja en 2 y 4, hihat en corcheas.',
    },
    {
        'id': 17, 'nombre': 'Groove bossa nova', 'tipo': 'groove',
        'pattern': {
            'kick':  'x..x..x...x.x...',
            'rim':   '..x..x..x..x..x.',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 2, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'bossa', 'tension': 3, 'emocion': 'reposo',
        'uso': ['percusion'],
        'desc': 'Bombo siguiendo el tresillo, aro (rim click) marcando la clave.',
    },
    {
        'id': 18, 'nombre': 'Boom-bap hip hop', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.......x...x...',
            'snare': '....x.......x...',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'swing',
        'estilo': 'hip_hop', 'tension': 5, 'emocion': 'melancolia',
        'uso': ['percusion'],
        'desc': 'Bombo sincopado, caja en 2 y 4, hihat en semicorcheas con swing ligero.',
    },
    {
        'id': 19, 'nombre': 'Dembow reggaetón', 'tipo': 'groove',
        'pattern': {
            'kick':  'x..x..x.x..x..x.',
            'snare': '..x...x...x...x.',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'reggaeton', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Patrón dembow estándar, célula tresillo+dos repetida en bombo y caja.',
    },
    {
        'id': 20, 'nombre': 'Breakbeat drum and bass', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.......x.x.....',
            'snare': '....x.......x.x.',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'drum_and_bass', 'tension': 8, 'emocion': 'angustia',
        'uso': ['percusion'],
        'desc': 'Break sincopado con doble bombo tardío y caja anticipada, alta densidad.',
    },
    {
        'id': 21, 'nombre': 'Son montuno', 'tipo': 'celula',
        'pattern': 'x.x..x.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 5, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Anticipación característica del tumbao de piano en el son montuno.',
    },
    {
        'id': 22, 'nombre': 'Guaguancó', 'tipo': 'groove',
        'pattern': {
            'claves': 'x..x..x...x.x...',
            'kick':   'x...x...x...x...',
            'perc':   '..x.x..x..x.x...',
        },
        'meter': '4/4', 'bars': 2, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'rumba', 'tension': 7, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Rumba de salón cubana: clave, tumbadora grave y quinto improvisando encima.',
    },
    {
        'id': 23, 'nombre': 'Songo', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.x.x..x........',
            'snare': '....x.......x...',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Fusión de tumbao, funk y jazz creada por Los Van Van, base de la salsa moderna.',
    },
    {
        'id': 24, 'nombre': 'Cha cha chá', 'tipo': 'celula',
        'pattern': 'x.x.x.xx',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'chachacha', 'tension': 4, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Los tres golpes finales marcan el característico "cha-cha-chá".',
    },
    {
        'id': 25, 'nombre': 'Mambo', 'tipo': 'groove',
        'pattern': {
            'kick':    'x..x..x.',
            'cowbell': '..x..x.x',
            'hihat':   'x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'mambo', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Sección de metales y campana propulsando el mambo de big band cubana.',
    },
    {
        'id': 26, 'nombre': 'Merengue', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.x.',
            'snare': '.x.x',
            'hihat': 'xx.x',
        },
        'meter': '2/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'merengue', 'tension': 7, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Tambora y güira dominicanas a tempo vivo, patrón binario muy bailable.',
    },
    {
        'id': 27, 'nombre': 'Bachata', 'tipo': 'celula',
        'pattern': 'x.xx.x.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'bachata', 'tension': 4, 'emocion': 'nostalgia',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Güira y bongó marcando el requinto de guitarra típico de la bachata dominicana.',
    },
    {
        'id': 28, 'nombre': 'Cumbia', 'tipo': 'groove',
        'pattern': {
            'kick':  'x..x..x.',
            'perc':  '.x.x.x.x',
            'hihat': 'x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'cumbia', 'tension': 5, 'emocion': 'esperanza',
        'uso': ['percusion'],
        'desc': 'Guacharaca y tambor alegre marcando el vaivén costeño colombiano.',
    },
    {
        'id': 29, 'nombre': 'Samba batucada', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.x.',
            'rim':   '.x.x',
            'hihat': 'xxxx',
        },
        'meter': '2/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'samba', 'tension': 7, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Surdo y tamborim de batería de escuela de samba, pulso de carnaval carioca.',
    },
    {
        'id': 30, 'nombre': 'Partido alto', 'tipo': 'celula',
        'pattern': 'x.xx.x.x',
        'meter': '2/4', 'bars': 1, 'subdivision': 16, 'feel': 'swing',
        'estilo': 'samba', 'tension': 5, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Célula sincopada de samba de raíz, base de innumerables partidos-altos.',
    },
    {
        'id': 31, 'nombre': 'Forró', 'tipo': 'groove',
        'pattern': {
            'kick': 'x.x.',
            'rim':  '.xx.',
        },
        'meter': '2/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'forro', 'tension': 5, 'emocion': 'esperanza',
        'uso': ['percusion'],
        'desc': 'Zabumba nordestina alternando golpe grave y seco, tren rítmico del forró.',
    },
    {
        'id': 32, 'nombre': 'Bolero', 'tipo': 'celula',
        'pattern': 'x..x.x..',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'bolero', 'tension': 2, 'emocion': 'tristeza',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Célula pausada emparentada con la habanera, propia del bolero romántico.',
    },
    {
        'id': 33, 'nombre': 'Guajira', 'tipo': 'celula',
        'pattern': 'x..x..',
        'meter': '6/8', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 3, 'emocion': 'nostalgia',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Aire campesino cubano en compás binario compuesto, evoca el punto guajiro.',
    },
    {
        'id': 34, 'nombre': 'Danzón', 'tipo': 'celula',
        'pattern': 'x.xx.x.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'afrocubano', 'tension': 3, 'emocion': 'melancolia',
        'uso': ['percusion'],
        'desc': 'Variante elegante y pausada del cinquillo, salón cubano del siglo XIX.',
    },
    {
        'id': 35, 'nombre': 'Calypso', 'tipo': 'groove',
        'pattern': {
            'kick':  'x..x..x.',
            'rim':   '.x.x.x.x',
            'hihat': 'x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'swing',
        'estilo': 'calypso', 'tension': 5, 'emocion': 'alegria',
        'uso': ['percusion'],
        'desc': 'Steel drum y tambora caribeñas balanceándose con acentos en contratiempo.',
    },
    {
        'id': 36, 'nombre': 'Soca', 'tipo': 'celula',
        'pattern': 'x.xx.xx.x.xx.xx.',
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'calypso', 'tension': 8, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Evolución acelerada y densa del calypso trinitense para carnaval.',
    },
    {
        'id': 37, 'nombre': 'Ska', 'tipo': 'groove',
        'pattern': {
            'kick':  'x...x...',
            'rim':   '.x.x.x.x',
            'hihat': 'x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'ska', 'tension': 5, 'emocion': 'alegria',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Guitarra en contratiempo ("skank") marcando el "upstroke" jamaicano.',
    },
    {
        'id': 38, 'nombre': 'Reggae one drop', 'tipo': 'groove',
        'pattern': {
            'kick':  '....x...',
            'rim':   '....x...',
            'hihat': '..x...x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'reggae', 'tension': 3, 'emocion': 'reposo',
        'uso': ['percusion'],
        'desc': 'El bombo "cae" únicamente en el tercer tiempo, dejando el uno en silencio.',
    },
    {
        'id': 39, 'nombre': 'Rocksteady', 'tipo': 'celula',
        'pattern': '.x.x.x.x',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'reggae', 'tension': 3, 'emocion': 'nostalgia',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Guitarra y piano en contratiempo, tempo más lento que el ska precedente.',
    },
    {
        'id': 40, 'nombre': 'Funk (The Funky Drummer)', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.....x..x......',
            'snare': '....x.......x...',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'funk', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Bombo muy sincopado al estilo de la batería funk clásica de los años 70.',
    },
    {
        'id': 41, 'nombre': 'Disco four-on-the-floor', 'tipo': 'groove',
        'pattern': {
            'kick':       'x...x...x...x...',
            'hihat_open': '..x...x...x...x.',
            'snare':      '....x.......x...',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'disco', 'tension': 5, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Bombo en cada negra, hihat abierto en los contratiempos, pista setentera.',
    },
    {
        'id': 42, 'nombre': 'House four-on-the-floor', 'tipo': 'groove',
        'pattern': {
            'kick':  'x...x...x...x...',
            'clap':  '....x.......x...',
            'hihat': 'x.x.x.x.x.x.x.x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'house', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion'],
        'desc': 'Heredero del disco: bombo constante en negras con palmada (clap) programada.',
    },
    {
        'id': 43, 'nombre': 'Techno driving', 'tipo': 'groove',
        'pattern': {
            'kick':       'x...x...x...x...',
            'hihat_open': '..x...x...x...x.',
            'cowbell':    'x.......x.......',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'techno', 'tension': 7, 'emocion': 'misterio',
        'uso': ['percusion'],
        'desc': 'Bombo hipnótico y regular con acentos metálicos esporádicos, pulso de club.',
    },
    {
        'id': 44, 'nombre': 'Trap hi-hat rolls', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.......x..x....',
            'snare': '........x.......',
            'hihat': 'x.xxx.xxx.xxx.xx',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'trap', 'tension': 6, 'emocion': 'drama',
        'uso': ['percusion'],
        'desc': 'Caja a medio tiempo con redobles rápidos de hihat característicos del trap.',
    },
    {
        'id': 45, 'nombre': 'Dubstep half-time', 'tipo': 'groove',
        'pattern': {
            'kick':  'x.......x.......',
            'snare': '........x.......',
            'hihat': 'x..x..x.x..x..x.',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 16, 'feel': 'straight',
        'estilo': 'dubstep', 'tension': 8, 'emocion': 'angustia',
        'uso': ['percusion'],
        'desc': 'Sensación de medio tiempo: la caja cae en el "3", dejando espacio al bajo.',
    },
    {
        'id': 46, 'nombre': 'Vals vienés', 'tipo': 'groove',
        'pattern': {
            'kick':  'x..',
            'snare': '.x.',
            'hihat': '..x',
        },
        'meter': '3/4', 'bars': 1, 'subdivision': 4, 'feel': 'straight',
        'estilo': 'vals', 'tension': 2, 'emocion': 'reposo',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Clásico "oom-pah-pah": bajo en el tiempo fuerte, acordes en los dos débiles.',
    },
    {
        'id': 47, 'nombre': 'Marcha militar', 'tipo': 'groove',
        'pattern': {
            'kick':   'x...x...',
            'snare':  'x.x.x.x.',
            'cymbal': 'x.......',
        },
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'marcha', 'tension': 6, 'emocion': 'solemnidad',
        'uso': ['percusion'],
        'desc': 'Redoble constante de caja sobre bombo en tiempos fuertes, paso de desfile.',
    },
    {
        'id': 48, 'nombre': 'Klezmer freylekh', 'tipo': 'celula',
        'pattern': 'xx.x.xx.',
        'meter': '4/4', 'bars': 1, 'subdivision': 8, 'feel': 'straight',
        'estilo': 'klezmer', 'tension': 6, 'emocion': 'euforia',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Danza judía askenazí de celebración, viva y sincopada, propia de la boda.',
    },
    {
        'id': 49, 'nombre': 'Gnawa', 'tipo': 'celula',
        'pattern': 'x..x.x..x.x.',
        'meter': '12/8', 'bars': 1, 'subdivision': 1, 'feel': 'straight',
        'estilo': 'gnawa', 'tension': 7, 'emocion': 'exotismo',
        'uso': ['percusion'],
        'desc': 'Krakebs y guembri marroquíes tejiendo un ciclo hipnótico de trance ritual.',
    },
    {
        'id': 50, 'nombre': 'Jazz waltz quíntuplo (5/4)', 'tipo': 'celula',
        'pattern': 'x.x.x',
        'meter': '5/4', 'bars': 1, 'subdivision': 4, 'feel': 'swing',
        'estilo': 'jazz', 'tension': 5, 'emocion': 'ambiguedad',
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Métrica asimétrica de cinco tiempos agrupados 3+2, célebre por "Take Five".',
    },
]

TABLE_BY_ID = {e['id']: e for e in TABLE}

# ═══════════════════════════════════════════════════════════════════════════════
# MAPEO DE VOCES A PERCUSIÓN GM (canal 10)
# ═══════════════════════════════════════════════════════════════════════════════

GM_DRUM_MAP = {
    'kick': 36, 'snare': 38, 'rim': 37, 'clap': 39,
    'hihat': 42, 'hihat_open': 46,
    'tom_low': 41, 'tom_mid': 45, 'tom_high': 48,
    'cymbal': 49, 'ride': 51, 'cowbell': 56,
    'claves': 75, 'click': 76, 'perc': 63,
}

# Mapeo inverso (nota GM → voz canónica), usado en el análisis de MIDI para
# reconstruir qué voz tocó cada golpe de percusión.
VOICE_FROM_GM_NOTE = {
    35: 'kick', 36: 'kick',
    38: 'snare', 40: 'snare', 37: 'rim', 39: 'clap',
    42: 'hihat', 44: 'hihat', 46: 'hihat_open',
    41: 'tom_low', 43: 'tom_low', 45: 'tom_mid', 47: 'tom_mid',
    48: 'tom_high', 50: 'tom_high',
    49: 'cymbal', 55: 'cymbal', 57: 'cymbal',
    51: 'ride', 59: 'ride',
    56: 'cowbell', 75: 'claves', 76: 'click', 63: 'perc',
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE MÉTRICA / TIEMPO
# ═══════════════════════════════════════════════════════════════════════════════

def meter_beats_per_bar(meter: str) -> float:
    """Convierte '7/8' -> 3.5 pulsos (negra) por compás, '4/4' -> 4.0, etc."""
    num, den = meter.strip().split('/')
    return int(num) * 4.0 / int(den)


def pattern_length(entry: dict) -> int:
    """Longitud en caracteres de una entrada (celula: string; groove: primera voz)."""
    if entry['tipo'] == 'celula':
        return len(entry['pattern'])
    return len(next(iter(entry['pattern'].values())))


def entry_beats_per_char(entry: dict) -> float:
    """
    Duración de cada carácter del patrón en pulsos (negra), derivada de
    meter/bars/longitud — no de 'subdivision', que es solo informativo.
    """
    bpb = meter_beats_per_bar(entry['meter'])
    length = pattern_length(entry)
    total_beats = bpb * entry['bars']
    return total_beats / length


# ═══════════════════════════════════════════════════════════════════════════════
# MODO CUSTOM — patrones introducidos directamente por el usuario (--custom)
# ═══════════════════════════════════════════════════════════════════════════════

_ONSET_CHARS = set('x.')


def parse_custom_celula(text: str) -> str:
    s = text.strip()
    if not s or set(s) - _ONSET_CHARS:
        raise ValueError(
            f"patrón de célula inválido: '{text}' (solo se admiten 'x' y '.')")
    return s


def parse_custom_groove(text: str) -> dict:
    """
    Parsea 'kick:x...x... snare:....x...' en {'kick': 'x...x...', ...}.
    Todas las voces deben tener la misma longitud.
    """
    voices = {}
    for tok in text.strip().split():
        if ':' not in tok:
            raise ValueError(
                f"token de groove inválido: '{tok}' (formato voz:onsets)")
        voice, onsets = tok.split(':', 1)
        voice = voice.strip()
        if not voice or set(onsets) - _ONSET_CHARS:
            raise ValueError(f"patrón inválido para la voz '{voice}': '{onsets}'")
        voices[voice] = onsets
    if not voices:
        raise ValueError("el groove --custom está vacío")
    lengths = {len(v) for v in voices.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"todas las voces deben tener la misma longitud (encontradas: {lengths})")
    return voices


def make_custom_entry(text: str, tipo: str, meter: str, bars: int) -> dict:
    """Construye una 'entry' sintética compatible con resolve_entry()."""
    if tipo == 'groove' or ':' in text:
        pattern = parse_custom_groove(text)
        tipo = 'groove'
    else:
        pattern = parse_custom_celula(text)
        tipo = 'celula'
    return {
        'id': None, 'nombre': 'Patrón personalizado', 'tipo': tipo,
        'pattern': pattern, 'meter': meter, 'bars': bars, 'subdivision': None,
        'feel': 'straight', 'estilo': 'custom', 'tension': None, 'emocion': None,
        'uso': ['percusion', 'ritmo_armonico'],
        'desc': 'Definido por el usuario mediante --custom',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN A TIEMPO REAL
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_entry(entry: dict, bars: int = None, reps: int = None) -> tuple:
    """
    Expande el patrón de una entrada a una lista de eventos concretos.

    Si `reps` se especifica, el patrón se repite exactamente `reps` veces en
    su totalidad (sin truncar). Si no, se rellena hasta cubrir `bars`
    compases del propio compás del patrón (puede truncar el último ciclo).

    Devuelve (events, total_chars, beats_per_char) donde cada evento es
    {'voice': str|None, 'beat': float, 'bar': int}.
    """
    beats_per_char = entry_beats_per_char(entry)
    bpb = meter_beats_per_bar(entry['meter'])
    length = pattern_length(entry)

    if reps is not None:
        total_chars = length * reps
    else:
        target_bars = bars if bars is not None else entry['bars']
        total_beats = bpb * target_bars
        total_chars = max(1, round(total_beats / beats_per_char))

    def tiled(s: str) -> str:
        reps_needed = total_chars // len(s) + 2
        return (s * reps_needed)[:total_chars]

    events = []
    if entry['tipo'] == 'celula':
        s = tiled(entry['pattern'])
        for i, ch in enumerate(s):
            if ch == 'x':
                beat = i * beats_per_char
                events.append({'voice': None, 'beat': beat,
                              'bar': int(beat // bpb) + 1})
    else:
        for voice, base in entry['pattern'].items():
            s = tiled(base)
            for i, ch in enumerate(s):
                if ch == 'x':
                    beat = i * beats_per_char
                    events.append({'voice': voice, 'beat': beat,
                                  'bar': int(beat // bpb) + 1})

    events.sort(key=lambda e: (e['beat'], e['voice'] or ''))
    return events, total_chars, beats_per_char


def tiled_pattern(entry: dict, total_chars: int) -> dict:
    """Devuelve {voz_o_None: string_tileado_a_total_chars}."""
    def tile(s):
        return (s * (total_chars // len(s) + 2))[:total_chars]
    if entry['tipo'] == 'celula':
        return {None: tile(entry['pattern'])}
    return {v: tile(s) for v, s in entry['pattern'].items()}


def pattern_to_text(entry: dict, total_chars: int) -> str:
    """Notación de onsets legible; una línea por voz si es groove."""
    tiled = tiled_pattern(entry, total_chars)
    if entry['tipo'] == 'celula':
        return tiled[None]
    width = max(len(v) for v in tiled) + 1
    return '\n'.join(f"{voice:<{width}}{s}" for voice, s in tiled.items())


# ═══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN A PROGRESIONES DE ACORDES ("ritmo armónico")
# ═══════════════════════════════════════════════════════════════════════════════

def apply_to_chords(entry: dict, chords: list, bars: int = None,
                    reps: int = None, change_voice: str = None) -> list:
    """
    Usa los onsets del patrón (o de una única voz de un groove, vía
    `change_voice`) como puntos de cambio de acorde. Devuelve una lista de
    (chord, duration_beats) que cubre exactamente el rango resuelto.
    """
    events, total_chars, beats_per_char = resolve_entry(entry, bars=bars, reps=reps)

    if entry['tipo'] == 'groove':
        if change_voice is None:
            raise ValueError(
                "--apply-to-chords sobre un groove requiere --change-voice "
                f"(voces disponibles: {', '.join(entry['pattern'].keys())})")
        if change_voice not in entry['pattern']:
            raise ValueError(
                f"voz '{change_voice}' no existe en este groove "
                f"(disponibles: {', '.join(entry['pattern'].keys())})")
        onset_beats = sorted({e['beat'] for e in events if e['voice'] == change_voice})
    else:
        onset_beats = sorted({e['beat'] for e in events})

    if not onset_beats:
        raise ValueError("el patrón no tiene onsets con los que marcar cambios")

    if onset_beats[0] != 0.0:
        onset_beats = [0.0] + onset_beats

    total_beats = total_chars * beats_per_char
    boundaries = onset_beats + [total_beats]

    result = []
    for i in range(len(boundaries) - 1):
        dur = boundaries[i + 1] - boundaries[i]
        if dur <= 1e-9:
            continue
        chord = chords[len(result) % len(chords)]
        result.append((chord, dur))
    return result


def chords_progression_to_text(pairs: list) -> str:
    """Formato compatible con --chords de los otros módulos del ecosistema."""
    parts = []
    for chord, dur in pairs:
        parts.append(f"{chord}:{int(dur) if dur == int(dur) else f'{dur:.2f}'}")
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_to_midi(entry: dict, events: list, beats_per_char: float,
                    ticks_per_beat: int, tempo_bpm: float, output_path: str,
                    default_note: int = 75, velocity: int = 100) -> None:
    """Exporta el patrón resuelto como MIDI de percusión (canal 10 GM)."""
    if not MIDO_OK:
        print("[ERROR] mido no disponible. Instala con: pip install mido")
        return

    note_dur_beats = min(beats_per_char * 0.9, 0.4)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',
                                  tempo=int(60_000_000 / tempo_bpm), time=0))
    track.append(mido.MetaMessage('track_name', name='Rhythm Table', time=0))

    abs_events = []
    for e in events:
        note = GM_DRUM_MAP.get(e['voice'], default_note) if e['voice'] else default_note
        on_tick = round(e['beat'] * ticks_per_beat)
        off_tick = on_tick + max(1, round(note_dur_beats * ticks_per_beat))
        abs_events.append((on_tick, 1, note))   # 1 = 'on' ordena tras 'off' en el mismo tick
        abs_events.append((off_tick, 0, note))  # 0 = 'off'
    abs_events.sort(key=lambda x: (x[0], x[1]))

    last = 0
    for tick, kind, note in abs_events:
        delta = tick - last
        last = tick
        if kind == 1:
            track.append(mido.Message('note_on', channel=9, note=note,
                                      velocity=velocity, time=delta))
        else:
            track.append(mido.Message('note_off', channel=9, note=note,
                                      velocity=0, time=delta))
    mid.save(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER MIDI NATIVO (sin dependencias externas — idéntico en espíritu al de
# chord_table.py, adaptado aquí para extracción de onsets)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_midi_raw(midi_path: str):
    """Devuelve (events, tpb) con events = lista de (abs_tick, 'on'|'off', note)."""
    import struct as _struct
    with open(midi_path, 'rb') as f:
        data = f.read()
    pos = 0

    def read_bytes(n):
        nonlocal pos
        b = data[pos:pos + n]
        pos += n
        return b

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
    read_uint32()
    read_uint16()
    ntracks = read_uint16()
    tpb = read_uint16()

    all_events = []
    for _ in range(ntracks):
        assert read_bytes(4) == b'MTrk', "Bad track header"
        tlen = read_uint32()
        end = pos + tlen
        abs_tick = 0
        running = 0
        while pos < end:
            dt = read_varlen(); abs_tick += dt
            b = data[pos]
            if b & 0x80:
                running = b; pos += 1
            msg_type = running & 0xF0
            if msg_type == 0x90:
                note = data[pos]; vel = data[pos + 1]; pos += 2
                etype = 'on' if vel > 0 else 'off'
                all_events.append((abs_tick, etype, note))
            elif msg_type == 0x80:
                note = data[pos]; pos += 2
                all_events.append((abs_tick, 'off', note))
            elif msg_type in (0xB0, 0xA0, 0xE0): pos += 2
            elif msg_type in (0xC0, 0xD0): pos += 1
            elif running == 0xFF:
                pos += 1; mlen = read_varlen(); pos += mlen
            elif running in (0xF0, 0xF7):
                slen = read_varlen(); pos += slen
            else:
                break
        pos = end

    all_events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))
    return all_events, tpb


def midi_to_onset_grid(midi_path: str, meter: str = '4/4', bars: int = None,
                       resolution: int = 4) -> dict:
    """
    Lee un MIDI y devuelve una rejilla de onsets cuantizada.

    `resolution` = subdivisiones por pulso (negra); 4 = semicorcheas.
    Devuelve dict con 'combined' (string de onsets con todas las notas),
    'voices' ({voz: string}, para notas reconocidas como percusión GM),
    'n_slots', 'slot_beats', 'total_bars', 'bpb'.
    """
    events, tpb = _parse_midi_raw(midi_path)
    note_ons = [(tick / tpb, note) for tick, etype, note in events if etype == 'on']
    if not note_ons:
        return {'combined': '', 'voices': {}, 'n_slots': 0,
               'slot_beats': 0.0, 'total_bars': 0, 'bpb': meter_beats_per_bar(meter)}

    bpb = meter_beats_per_bar(meter)
    max_beat = max(b for b, _ in note_ons)
    total_bars = bars if bars else max(1, math.ceil(max_beat / bpb))
    total_beats = total_bars * bpb
    slot_beats = 1.0 / resolution
    n_slots = max(1, round(total_beats / slot_beats))

    combined = ['.'] * n_slots
    voice_grids = {}
    for beat, note in note_ons:
        idx = int(round(beat / slot_beats))
        if idx >= n_slots:
            continue
        combined[idx] = 'x'
        voice = VOICE_FROM_GM_NOTE.get(note)
        if voice:
            grid = voice_grids.setdefault(voice, ['.'] * n_slots)
            grid[idx] = 'x'

    return {
        'combined': ''.join(combined),
        'voices': {v: ''.join(g) for v, g in voice_grids.items()},
        'n_slots': n_slots, 'slot_beats': slot_beats,
        'total_bars': total_bars, 'bpb': bpb,
    }


def entry_grids_at_resolution(entry: dict, total_bars: int, slot_beats: float) -> tuple:
    """
    Re-muestrea el patrón del catálogo (que puede estar en corcheas,
    semicorcheas, tresillos...) a la MISMA rejilla temporal (slot_beats) que
    se usó para extraer los onsets del MIDI, cubriendo `total_bars` compases.
    Sin este re-muestreo, comparar dos strings de distinta resolución
    temporal carácter-a-carácter no tiene sentido.

    Devuelve (grids, period_slots) donde grids es {'combined': str} para
    células o {voz: str} para grooves, y period_slots es la duración de UNA
    repetición del patrón en slots (para la búsqueda de fase).
    """
    bpb = meter_beats_per_bar(entry['meter'])
    total_beats = total_bars * bpb
    n_slots = max(1, round(total_beats / slot_beats))
    beats_per_char = entry_beats_per_char(entry)
    period_slots = max(1, round((bpb * entry['bars']) / slot_beats))
    total_chars_needed = max(1, math.ceil(total_beats / beats_per_char))

    def make_grid(pattern_str: str) -> str:
        length = len(pattern_str)
        grid = ['.'] * n_slots
        for i in range(total_chars_needed):
            if pattern_str[i % length] == 'x':
                idx = round((i * beats_per_char) / slot_beats)
                if idx < n_slots:
                    grid[idx] = 'x'
        return ''.join(grid)

    if entry['tipo'] == 'celula':
        return {'combined': make_grid(entry['pattern'])}, period_slots
    return {v: make_grid(s) for v, s in entry['pattern'].items()}, period_slots


def score_grid_vs_grid(entry_grid: str, midi_grid: str, period_slots: int) -> tuple:
    """
    Compara dos rejillas binarias YA en la misma resolución temporal,
    probando desplazamientos circulares de `entry_grid` de hasta un período
    del patrón (para tolerar que el MIDI no empiece exactamente en el primer
    onset del patrón). Score = F1 sobre las posiciones de onset, análogo en
    espíritu a match_pattern_in_sequence de chord_table.py pero para
    conjuntos binarios de golpes en vez de numerales de acorde.
    """
    n = len(midi_grid)
    entry_x = entry_grid.count('x')
    midi_x = midi_grid.count('x')
    if entry_x == 0 or midi_x == 0 or n == 0:
        return 0, 0.0

    best_shift, best_score = 0, 0.0
    for shift in range(min(period_slots, n)):
        shifted = entry_grid[-shift:] + entry_grid[:-shift] if shift else entry_grid
        match_x = sum(1 for a, b in zip(shifted, midi_grid) if a == 'x' and b == 'x')
        precision = match_x / midi_x
        recall = match_x / entry_x
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_score:
            best_shift, best_score = shift, f1
    return best_shift, best_score


def analyze_midi(midi_path: str, meter: str = '4/4', bars: int = None,
                 resolution: int = 4, min_score: float = 0.5) -> tuple:
    """Compara la rejilla de onsets extraída del MIDI contra todo el catálogo."""
    grid = midi_to_onset_grid(midi_path, meter=meter, bars=bars, resolution=resolution)
    matches = []
    if not grid['combined']:
        return matches, grid

    for entry in TABLE:
        entry_grids, period_slots = entry_grids_at_resolution(
            entry, grid['total_bars'], grid['slot_beats'])

        if entry['tipo'] == 'celula':
            shift, score = score_grid_vs_grid(
                entry_grids['combined'], grid['combined'], period_slots)
            if score >= min_score:
                matches.append({
                    'entry': entry, 'score': round(score, 3),
                    'offset': shift, 'detail': None,
                })
        else:
            per_voice, scores = {}, []
            for voice, eg in entry_grids.items():
                if voice not in grid['voices']:
                    continue
                shift, sc = score_grid_vs_grid(eg, grid['voices'][voice], period_slots)
                per_voice[voice] = {'offset': shift, 'score': round(sc, 3)}
                scores.append(sc)
            if not scores:
                continue
            coverage = len(scores) / len(entry_grids)
            score = (sum(scores) / len(scores)) * coverage
            if score >= min_score:
                matches.append({
                    'entry': entry, 'score': round(score, 3),
                    'offset': None, 'detail': per_voice,
                    'coverage': round(coverage, 3),
                })

    matches.sort(key=lambda m: -m['score'])
    return matches, grid


def print_midi_analysis(midi_path: str, matches: list, grid: dict,
                        verbose: bool = False) -> None:
    print(f"\n{COL['cyan']}{'═'*72}{COL['reset']}")
    print(f"{COL['bold']}  Análisis rítmico: {midi_path}{COL['reset']}")
    print(f"{COL['cyan']}{'═'*72}{COL['reset']}\n")
    print(f"  Compases analizados: {grid['total_bars']}  ·  "
          f"resolución: {round(1/grid['slot_beats'])} subdiv/pulso  ·  "
          f"voces detectadas: {', '.join(sorted(grid['voices'])) or '(ninguna reconocida como percusión GM)'}\n")

    if not matches:
        print("  Sin coincidencias por encima del umbral.\n")
        return

    for m in matches:
        e = m['entry']
        tc = tension_col(e['tension'])
        print(f"  {COL['gray']}#{e['id']:2d}{COL['reset']}  "
              f"{COL['bold']}{e['nombre']:<26}{COL['reset']} "
              f"score={COL['cyan']}{m['score']:.2f}{COL['reset']}  "
              f"{tc}t={e['tension']}{COL['reset']}  "
              f"{COL['gray']}{e['tipo']:<7} {e['estilo']:<14}{COL['reset']}")
        if verbose:
            print(f"        {e['desc']}")
            if m['detail']:
                for voice, d in m['detail'].items():
                    print(f"          {voice:<10} score={d['score']:.2f} offset={d['offset']}")
    print(f"\n  {len(matches)} coincidencia(s)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRADO Y BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    for src, dst in {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                     'ü': 'u', 'ñ': 'n'}.items():
        s = s.lower().replace(src, dst)
    return s


def filter_table(estilo=None, tipo=None, uso=None, meter=None, feel=None,
                 emocion=None, tension_min=None, tension_max=None,
                 buscar=None) -> list:
    result = list(TABLE)
    if estilo:
        result = [e for e in result if e['estilo'] == estilo]
    if tipo:
        result = [e for e in result if e['tipo'] == tipo]
    if uso:
        result = [e for e in result if uso in e['uso']]
    if meter:
        result = [e for e in result if e['meter'] == meter]
    if feel:
        result = [e for e in result if e['feel'] == feel]
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
            if q in _normalize(e['nombre'])
            or q in _normalize(e['desc'])
            or q in _normalize(e['emocion'])
            or q in _normalize(e['estilo'])
            or q in _normalize(e['tipo'])
        ]
    return result


INTENT_TO_EMOCION = {
    'oscuro': 'angustia', 'oscuridad': 'angustia',
    'triste': 'tristeza', 'tristeza': 'tristeza',
    'alegre': 'alegria', 'alegria': 'alegria', 'alegría': 'alegria',
    'eufórico': 'euforia', 'euforico': 'euforia', 'euforia': 'euforia',
    'melancolico': 'melancolia', 'melancólico': 'melancolia', 'melancolia': 'melancolia',
    'nostalgico': 'nostalgia', 'nostálgico': 'nostalgia', 'nostalgia': 'nostalgia',
    'misterioso': 'misterio', 'misterio': 'misterio',
    'tenso': 'drama', 'tension': 'drama', 'tensión': 'drama', 'drama': 'drama',
    'angustia': 'angustia', 'angustioso': 'angustia',
    'esperanza': 'esperanza', 'esperanzador': 'esperanza',
    'relajado': 'reposo', 'reposo': 'reposo', 'calmado': 'reposo',
    'flotante': 'flotacion', 'flotacion': 'flotacion',
    'exotico': 'exotismo', 'exótico': 'exotismo', 'exotismo': 'exotismo',
    'solemne': 'solemnidad', 'solemnidad': 'solemnidad',
}

INTENT_TO_TENSION = {
    'muy denso': (7, 10), 'mucha tension': (7, 10), 'mucha tensión': (7, 10),
    'urgente': (7, 10), 'denso': (6, 10), 'alta tension': (7, 10),
    'poco denso': (1, 3), 'poca tension': (1, 3), 'poca tensión': (1, 3),
    'suave': (1, 3), 'relajado': (1, 4),
    'media tension': (4, 6), 'media tensión': (4, 6), 'moderado': (4, 6),
}


def parse_intent(description: str) -> dict:
    desc = _normalize(description)
    filters = {}
    for phrase, rng in INTENT_TO_TENSION.items():
        if _normalize(phrase) in desc:
            filters['tension_min'], filters['tension_max'] = rng
            break
    for word, emocion in INTENT_TO_EMOCION.items():
        if _normalize(word) in desc:
            filters['emocion'] = emocion
            break
    return filters


# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

COL = {
    'reset': '\033[0m', 'bold': '\033[1m',
    'cyan': '\033[36m', 'yellow': '\033[33m',
    'green': '\033[32m', 'red': '\033[31m',
    'gray': '\033[90m', 'blue': '\033[34m',
    'magenta': '\033[35m',
}

TENSION_COLOR = {
    range(1, 4): COL['green'],
    range(4, 7): COL['yellow'],
    range(7, 11): COL['red'],
}


def tension_col(t) -> str:
    if t is None:
        return COL['reset']
    for r, c in TENSION_COLOR.items():
        if t in r:
            return c
    return COL['reset']


def print_entry(e: dict, verbose: bool = False) -> None:
    tc = tension_col(e['tension'])
    tension_str = str(e['tension']) if e['tension'] is not None else '-'
    print(
        f"  {COL['gray']}#{'' if e['id'] is None else e['id']:>2}{COL['reset']}  "
        f"{COL['bold']}{e['nombre']:<26}{COL['reset']} "
        f"{COL['gray']}{e['tipo']:<7}{COL['reset']} "
        f"{tc}t={tension_str:<2}{COL['reset']}  "
        f"{COL['cyan']}{(e['emocion'] or '-'):<12}{COL['reset']}  "
        f"{COL['gray']}{e['estilo']:<14} {e['meter']:<6}{COL['reset']}"
    )
    if verbose:
        print(f"        {COL['gray']}uso: {', '.join(e['uso'])} · feel: {e['feel']}{COL['reset']}")
        print(f"        {e['desc']}")


def print_table(entries: list, verbose: bool = False) -> None:
    print(
        f"\n  {COL['gray']}{'#':>3}  {'nombre':<26} {'tipo':<7} {'t':>3}  "
        f"{'emoción':<12}  {'estilo':<14} {'compás':<6}{COL['reset']}"
    )
    print(f"  {'─'*88}")
    for e in entries:
        print_entry(e, verbose)
    print(f"\n  {len(entries)} patrón(es)\n")


def print_stats() -> None:
    print(f"\n{COL['cyan']}{'═'*60}{COL['reset']}")
    print(f"{COL['bold']}  RHYTHM TABLE — Estadísticas del catálogo{COL['reset']}")
    print(f"{COL['cyan']}{'═'*60}{COL['reset']}\n")

    tipos = Counter(e['tipo'] for e in TABLE)
    estilos = Counter(e['estilo'] for e in TABLE)
    usos = Counter(u for e in TABLE for u in e['uso'])
    emociones = Counter(e['emocion'] for e in TABLE)
    tensions = [e['tension'] for e in TABLE]

    print(f"  Total patrones: {len(TABLE)}")
    print(f"  Tensión media: {sum(tensions)/len(tensions):.1f}  "
          f"min={min(tensions)}  max={max(tensions)}\n")

    print(f"  {COL['bold']}Por tipo:{COL['reset']}")
    for k, v in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por uso:{COL['reset']}")
    for k, v in sorted(usos.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por estilo:{COL['reset']}")
    for k, v in sorted(estilos.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por emoción:{COL['reset']}")
    for k, v in sorted(emociones.items(), key=lambda x: -x[1]):
        print(f"    {k:<16} {v:>3}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Catálogo curado de patrones rítmicos con tensión y emoción.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Análisis MIDI
    parser.add_argument('--analyze-midi', type=str, metavar='MIDI_FILE',
                        help='Analizar un MIDI y mostrar los patrones del catálogo '
                             'que contiene')
    parser.add_argument('--min-score', type=float, default=0.5, metavar='F',
                        help='Score mínimo de coincidencia 0-1 (default: 0.5)')
    parser.add_argument('--resolution', type=int, default=4, metavar='N',
                        help='Subdivisiones por pulso para la rejilla de análisis '
                             '(default: 4 = semicorcheas)')

    # Entrada / búsqueda
    parser.add_argument('--list', action='store_true',
                        help='Listar todos los patrones')
    parser.add_argument('--stats', action='store_true',
                        help='Estadísticas del catálogo')
    parser.add_argument('--custom', type=str, metavar='ONSETS',
                        help="Patrón personalizado en vez de seleccionar del "
                             "catálogo. Célula: --custom \"x..x..x.\". "
                             "Groove: --custom \"kick:x...x... snare:....x...\" "
                             "(auto-detecta tipo groove si contiene ':').")
    parser.add_argument('--tipo', type=str, choices=['celula', 'groove'],
                        help='Tipo de patrón (filtro, o fuerza el tipo de --custom)')
    parser.add_argument('--id', type=int, metavar='N',
                        help='Seleccionar patrón por id')
    parser.add_argument('--buscar', type=str, metavar='TEXTO',
                        help='Búsqueda por texto libre')
    parser.add_argument('--intencion', type=str, metavar='DESC',
                        help='Buscar por intención en lenguaje natural')

    # Filtros
    parser.add_argument('--style', type=str, dest='estilo',
                        choices=sorted({e['estilo'] for e in TABLE}),
                        help='Filtrar por estilo')
    parser.add_argument('--uso', type=str, choices=['percusion', 'ritmo_armonico'],
                        help='Filtrar por uso')
    parser.add_argument('--meter', type=str,
                        help='Filtrar por compás (ej. 4/4), o compás para --custom/análisis')
    parser.add_argument('--feel', type=str, choices=['straight', 'swing', 'shuffle'],
                        help='Filtrar por feel')
    parser.add_argument('--emocion', type=str,
                        help='Filtrar por emoción (euforia, angustia, reposo…)')
    parser.add_argument('--tension-min', type=int, metavar='N', dest='tension_min',
                        help='Tensión mínima (1-10)')
    parser.add_argument('--tension-max', type=int, metavar='N', dest='tension_max',
                        help='Tensión máxima (1-10)')

    # Resolución y exportación
    parser.add_argument('--bars', type=int, default=None,
                        help='Compases a generar (default: los del propio patrón)')
    parser.add_argument('--reps', type=int, default=None, metavar='N',
                        help='Repeticiones exactas del patrón (sustituye a --bars)')
    parser.add_argument('--tempo', type=float, default=120.0,
                        help='Tempo BPM para exportación MIDI (default: 120)')
    parser.add_argument('--note', type=int, default=75, metavar='N',
                        help='Nota MIDI GM para células sin voz (default: 75, claves)')

    # Ritmo armónico
    parser.add_argument('--apply-to-chords', type=str, metavar='ACORDES',
                        help='Aplica los onsets del patrón como puntos de cambio '
                             'a una lista de acordes, ej. "Am G F E7". Compatible '
                             'con el texto --chords de chord_table.py / melody_adapter.py')
    parser.add_argument('--change-voice', type=str, metavar='VOZ',
                        help='Voz del groove cuyos onsets marcan los cambios de '
                             'acorde (requerido si --apply-to-chords se usa sobre un groove)')

    parser.add_argument('--export-json', type=str, metavar='FILE',
                        help='Exportar tabla filtrada o patrón resuelto a JSON')
    parser.add_argument('--export-text', type=str, metavar='FILE',
                        help='Exportar patrón resuelto (o progresión de --apply-to-chords) a texto')
    parser.add_argument('--export-midi', type=str, metavar='FILE',
                        help='Exportar patrón resuelto a MIDI de percusión')
    parser.add_argument('--output', type=str, default='ritmo',
                        help='Nombre base para salidas (default: ritmo)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar nombre y descripción de cada patrón')

    args = parser.parse_args()

    # ── Análisis MIDI ────────────────────────────────────────────────────────
    if args.analyze_midi:
        if not os.path.isfile(args.analyze_midi):
            print(f"[ERROR] No se encuentra el archivo: {args.analyze_midi}")
            sys.exit(1)
        matches, grid = analyze_midi(
            args.analyze_midi, meter=args.meter or '4/4', bars=args.bars,
            resolution=args.resolution, min_score=args.min_score,
        )
        print_midi_analysis(args.analyze_midi, matches, grid, verbose=args.verbose)
        if args.export_json:
            exportable = [{
                'id': m['entry']['id'], 'nombre': m['entry']['nombre'],
                'score': m['score'], 'detail': m['detail'],
            } for m in matches]
            with open(args.export_json, 'w') as f:
                json.dump(exportable, f, indent=2, ensure_ascii=False)
            print(f"  → JSON: {args.export_json}\n")
        return

    # ── Stats ────────────────────────────────────────────────────────────────
    if args.stats:
        print_stats()
        return

    # ── Validación --id / --custom mutuamente excluyentes ────────────────────
    if args.id is not None and args.custom is not None:
        print("[ERROR] --id y --custom no se pueden usar a la vez")
        sys.exit(1)

    # ── Selección por id o patrón personalizado ───────────────────────────────
    if args.id is not None or args.custom is not None:
        if args.custom is not None:
            try:
                entry = make_custom_entry(
                    args.custom, args.tipo or 'celula',
                    args.meter or '4/4', args.bars or 1,
                )
            except ValueError as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
            print(f"\n  {COL['bold']}{entry['nombre']}{COL['reset']}  "
                  f"{COL['gray']}({entry['tipo']}, custom){COL['reset']}")
        else:
            entry = TABLE_BY_ID.get(args.id)
            if not entry:
                print(f"[ERROR] No existe patrón con id={args.id}")
                sys.exit(1)
            print(f"\n  #{entry['id']}  {COL['bold']}{entry['nombre']}{COL['reset']}")
            print(f"  {entry['tipo']} · {entry['estilo']} · {entry['meter']} · {entry['feel']}")
            print(f"  Tensión: {entry['tension']}/10  ·  Emoción: {entry['emocion']}")
            print(f"  {entry['desc']}\n")

        try:
            events, total_chars, beats_per_char = resolve_entry(
                entry, bars=args.bars, reps=args.reps)
        except ZeroDivisionError:
            print("[ERROR] patrón vacío, no se puede resolver")
            sys.exit(1)

        text = pattern_to_text(entry, total_chars)
        print(f"{COL['cyan']}{text}{COL['reset']}\n")

        # ── Ritmo armónico ────────────────────────────────────────────────────
        if args.apply_to_chords:
            chords = args.apply_to_chords.replace(',', ' ').split()
            if not chords:
                print("[ERROR] --apply-to-chords está vacío")
                sys.exit(1)
            try:
                pairs = apply_to_chords(entry, chords, bars=args.bars,
                                        reps=args.reps, change_voice=args.change_voice)
            except ValueError as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
            chord_text = chords_progression_to_text(pairs)
            print(f"  → {COL['cyan']}{chord_text}{COL['reset']}\n")

            json_path = args.export_json or f"{args.output}.chords.json"
            txt_path = args.export_text or f"{args.output}.chords.txt"
            with open(json_path, 'w') as f:
                json.dump({
                    'meta': {k: v for k, v in entry.items() if k != 'pattern'},
                    'change_voice': args.change_voice,
                    'chord_string': chord_text,
                    'progression': [{'chord': c, 'duration_beats': d} for c, d in pairs],
                    'generator': 'rhythm_table.py v1.0',
                }, f, indent=2, ensure_ascii=False)
            print(f"  → JSON:  {json_path}")
            with open(txt_path, 'w') as f:
                f.write(chord_text + '\n')
            print(f"  → Texto: {txt_path}")
            return

        # ── Exportación estándar del patrón ────────────────────────────────────
        json_path = args.export_json or f"{args.output}.rhythm.json"
        txt_path = args.export_text or f"{args.output}.rhythm.txt"
        mid_path = args.export_midi or f"{args.output}.rhythm.mid"

        with open(json_path, 'w') as f:
            json.dump({
                'meta': {k: v for k, v in entry.items() if k != 'pattern'},
                'bars': args.bars, 'reps': args.reps, 'tempo': args.tempo,
                'total_chars': total_chars, 'beats_per_char': beats_per_char,
                'pattern_text': text,
                'events': events,
                'generator': 'rhythm_table.py v1.0',
            }, f, indent=2, ensure_ascii=False)
        print(f"  → JSON:  {json_path}")

        with open(txt_path, 'w') as f:
            f.write(text + '\n')
        print(f"  → Texto: {txt_path}")

        if MIDO_OK:
            pattern_to_midi(entry, events, beats_per_char, 480, args.tempo,
                            mid_path, default_note=args.note)
            print(f"  → MIDI:  {mid_path}")
        else:
            print("  [AVISO] mido no instalado — MIDI omitido. pip install mido")
        return

    # ── Búsqueda por intención ────────────────────────────────────────────────
    if args.intencion:
        intent_filters = parse_intent(args.intencion)
        entries = filter_table(
            estilo=args.estilo, tipo=args.tipo, uso=args.uso,
            meter=args.meter, feel=args.feel,
            emocion=intent_filters.get('emocion', args.emocion),
            tension_min=intent_filters.get('tension_min', args.tension_min),
            tension_max=intent_filters.get('tension_max', args.tension_max),
            buscar=args.buscar,
        )
        if not entries:
            print(f"\n  Sin resultados para: \"{args.intencion}\"\n")
            sys.exit(0)
        print(f"\n  Resultados para: \"{args.intencion}\"")
        print_table(entries, args.verbose)
        if args.export_json:
            exportable = [{k: v for k, v in e.items()} for e in entries]
            with open(args.export_json, 'w') as f:
                json.dump(exportable, f, indent=2, ensure_ascii=False)
            print(f"  → JSON: {args.export_json}")
        return

    # ── Filtrado general + listado ────────────────────────────────────────────
    entries = filter_table(
        estilo=args.estilo, tipo=args.tipo, uso=args.uso,
        meter=args.meter, feel=args.feel, emocion=args.emocion,
        tension_min=args.tension_min, tension_max=args.tension_max,
        buscar=args.buscar,
    )

    if not entries and not args.list:
        parser.print_help()
        return

    print_table(entries if entries else TABLE, args.verbose)

    if args.export_json:
        data = entries if entries else TABLE
        with open(args.export_json, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  → JSON: {args.export_json}\n")


if __name__ == '__main__':
    main()
