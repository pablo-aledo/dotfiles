#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MOTIF TABLE  v2.0                                     ║
║      Catálogo curado de motivos y frases melódicas con tensión y emoción    ║
║                                                                              ║
║  Complemento de chord_table.py y rhythm_table.py: en lugar de acordes o     ║
║  ritmos, ofrece un catálogo curado de 100 gestos melódicos —motivos cortos  ║
║  (2-5 notas) y frases más largas (6-9 notas, con arco antecedente-          ║
║  consecuente)— descritos por grados de escala, con etiquetas de tensión     ║
║  (1-10), contorno y emoción, buscables y exportables.                       ║
║                                                                              ║
║    · Tipos:    motivo (célula corta), frase (gesto largo con arco,         ║
║                lista para insertar), escala (recorrido modal completo,     ║
║                material de referencia/estudio, no una frase con forma)     ║
║    · Modos:    mayor, menor, dórico, frigio, frigio dominante, lidio,       ║
║                mixolidio, armónico menor                                    ║
║    · Estilos:  diatónico, barroco, jazz, modal, romántico, impresionista,   ║
║                pop, flamenco, blues, folk                                   ║
║    · Contorno: ascendente, descendente, arco, ondulante, zigzag, salto,     ║
║                estático                                                     ║
║    · Emociones: reposo, alegría, tristeza, melancolía, nostalgia,           ║
║                  misterio, ambigüedad, drama, angustia, esperanza,          ║
║                  euforia, flotación, solemnidad, exotismo                   ║
║                                                                              ║
║  NOTACIÓN DE GRADOS (relativa a la tónica, no a acordes):                  ║
║    1-8      grado de escala según el modo de la entrada (8 = octava de 1)  ║
║    b / #    altera el grado un semitono (nota cromática fuera del modo)    ║
║    '        sube una octava el grado (puede repetirse)                     ║
║    ,        baja una octava el grado (puede repetirse)                     ║
║    r        silencio (ocupa tiempo, no suena)                              ║
║    Ej: "1 3 5 8" = tríada ascendente. "7," = sensible una octava abajo.   ║
║                                                                              ║
║  USO:                                                                        ║
║    # Listar todos                                                           ║
║    python motif_table.py --list                                            ║
║                                                                              ║
║    # Filtrar por emoción / estilo / tipo / contorno                        ║
║    python motif_table.py --emocion angustia                                ║
║    python motif_table.py --style flamenco --tipo frase                    ║
║    python motif_table.py --contorno ascendente --mode dorian              ║
║                                                                              ║
║    # Filtrar por rango de tensión                                           ║
║    python motif_table.py --tension-min 7 --tension-max 10                  ║
║                                                                              ║
║    # Buscar por texto libre / por intención en lenguaje natural             ║
║    python motif_table.py --buscar "cromático"                              ║
║    python motif_table.py --intencion "misterioso y flotante"               ║
║                                                                              ║
║    # Resolver un motivo del catálogo en una tónica dada y exportar         ║
║    python motif_table.py --id 4 --key Cm --octave 5 --export-midi m.mid   ║
║                                                                              ║
║    # Motivo personalizado por grados (modo custom)                         ║
║    python motif_table.py --custom "1 3 5 8" --mode major --key D          ║
║                                                                              ║
║    # Custom con duración por grado (grado:beats)                           ║
║    python motif_table.py --custom "1:0.5 3:0.5 5:1 8:2" --key G           ║
║                                                                              ║
║    # Exportar tabla filtrada a JSON                                        ║
║    python motif_table.py --style baroque --export-json barroco.json       ║
║                                                                              ║
║    # Mostrar estadísticas del catálogo                                     ║
║    python motif_table.py --stats                                           ║
║                                                                              ║
║    # Analizar un MIDI y detectar motivos del catálogo en la melodía       ║
║    python motif_table.py --analyze-midi cancion.mid                        ║
║                                                                              ║
║    # Análisis con canal forzado, umbral alto y exportación JSON            ║
║    python motif_table.py --analyze-midi song.mid --channel 0               ║
║        --min-score 0.75 --export-json matches.json --verbose               ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                            ║
║    chord_table.py       → complementario: acordes vs. línea melódica       ║
║    rhythm_table.py      → --apply-to-chords puede dar la temporización     ║
║                            que aquí se resuelve como línea de grados        ║
║    melody_adapter.py    → el texto exportado (nota:beats) es una línea     ║
║                            melódica lista para adaptar sobre acordes        ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <base>.motif.txt    — línea melódica en texto ("C5:1 E5:1 G5:1 C6:1")  ║
║    <base>.motif.json   — motivo resuelto con metadatos                     ║
║    <base>.motif.mid    — MIDI de línea melódica monofónica                 ║
║                                                                              ║
║  DEPENDENCIAS: mido (opcional, solo para exportación / análisis MIDI)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import re
from collections import Counter

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MUSICALES
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})
NOTE_PC.update({'Cb': 11, 'Fb': 4, 'B#': 0, 'E#': 5})

# Intervalos en semitonos de cada grado (1-7) respecto a la tónica, por modo.
MODE_INTERVALS = {
    'major':              [0, 2, 4, 5, 7, 9, 11],
    'minor':              [0, 2, 3, 5, 7, 8, 10],   # eólico / menor natural
    'dorian':             [0, 2, 3, 5, 7, 9, 10],
    'phrygian':           [0, 1, 3, 5, 7, 8, 10],
    'lydian':             [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':         [0, 2, 4, 5, 7, 9, 10],
    'harmonic_minor':     [0, 2, 3, 5, 7, 8, 11],
    'phrygian_dominant':  [0, 1, 4, 5, 7, 8, 10],
}

MODES = list(MODE_INTERVALS.keys())

DEGREE_RE = re.compile(r"^(b|#)?([1-8])([',]*)$")
REST_TOKEN = 'r'


def is_rest(token: str) -> bool:
    return token.strip().lower() == REST_TOKEN


def resolve_degree(token: str, mode: str) -> int:
    """
    Convierte un token de grado (ej. '1', 'b3', '#4', "7,", "1'") en un
    desplazamiento en semitonos absoluto respecto a la tónica (puede ser
    negativo o mayor que 12; no se envuelve a una sola octava, porque el
    contorno melódico real importa).

    No debe llamarse con un token de silencio ('r'); compruébalo antes con
    is_rest().
    """
    m = DEGREE_RE.match(token.strip())
    if not m:
        raise ValueError(f"Grado inválido: '{token}'")
    accidental, digit, octmarks = m.groups()
    d = int(digit)
    scale = MODE_INTERVALS[mode]
    base = scale[0] + 12 if d == 8 else scale[d - 1]
    if accidental == 'b':
        base -= 1
    elif accidental == '#':
        base += 1
    octshift = octmarks.count("'") - octmarks.count(",")
    return base + 12 * octshift


def degrees_display(pattern: list) -> str:
    return ' – '.join(('𝄽' if is_rest(tok) else tok) for tok, _ in pattern)



# ═══════════════════════════════════════════════════════════════════════════════
# CATÁLOGO CURADO DE MOTIVOS Y FRASES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cada entrada:
#   id       — identificador único
#   pattern  — lista de (grado, duracion_beats)
#   nombre   — nombre descriptivo
#   tipo     — 'motivo' (célula corta, 2-5 notas, con forma/gesto real) |
#               'frase' (gesto largo, 6-9 notas, con arco antecedente-
#               consecuente y ritmo variado, lista para insertarse con
#               pocos cambios) | 'escala' (recorrido completo de un modo,
#               ritmo uniforme: es material de referencia/estudio para oír
#               el color del modo, no una frase con forma propia)
#   contorno — ascendente | descendente | arco | ondulante | zigzag |
#               salto | estatico
#   style    — estilo (diatonic | baroque | jazz | modal | romantic |
#               impressionist | pop | flamenco | blues | folk)
#   mode     — modo (ver MODE_INTERVALS)
#   tension  — nivel de tensión 1-10
#   emocion  — emoción principal asociada
#   desc     — descripción breve del efecto y uso
#
TABLE = [
    {
        'id': 1, 'nombre': 'Arpegio ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('8', 1)],
        'contorno': 'ascendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'alegría',
        'desc': 'Tríada mayor recorrida hacia arriba. Abierto, luminoso, muy común como anacrusis.',
    },
    {
        'id': 2, 'nombre': 'Arpegio descendente', 'tipo': 'motivo',
        'pattern': [('8', 1), ('5', 1), ('3', 1), ('1', 1)],
        'contorno': 'descendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Tríada mayor recorrida hacia abajo. Cierre natural, gesto de aterrizaje.',
    },
    {
        'id': 3, 'nombre': 'Tríada menor ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('8', 1)],
        'contorno': 'ascendente', 'style': 'diatonic', 'mode': 'minor',
        'tension': 4, 'emocion': 'tristeza',
        'desc': 'Mismo gesto que el arpegio mayor pero en modo menor: mismo contorno, color sombrío.',
    },
    {
        'id': 4, 'nombre': 'Motivo del destino', 'tipo': 'motivo',
        'pattern': [('3', 0.5), ('3', 0.5), ('3', 0.5), ('1', 2)],
        'contorno': 'estatico', 'style': 'romantic', 'mode': 'minor',
        'tension': 8, 'emocion': 'angustia',
        'desc': 'Tres repeticiones cortas seguidas de un salto de tercera hacia abajo largo. Beethoven 5ª.',
    },
    {
        'id': 5, 'nombre': 'Escala ascendente', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5),
                    ('5', .5), ('6', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 3, 'emocion': 'esperanza',
        'desc': 'Recorrido diatónico completo hacia arriba. Sensación de apertura y avance.',
    },
    {
        'id': 6, 'nombre': 'Escala descendente', 'tipo': 'motivo',
        'pattern': [('8', .5), ('7', .5), ('6', .5), ('5', .5),
                    ('4', .5), ('3', .5), ('2', .5), ('1', .5)],
        'contorno': 'descendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Recorrido diatónico completo hacia abajo. Distensión progresiva hasta la tónica.',
    },
    {
        'id': 7, 'nombre': 'Escala menor descendente', 'tipo': 'motivo',
        'pattern': [('8', .5), ('7', .5), ('6', .5), ('5', .5),
                    ('4', .5), ('3', .5), ('2', .5), ('1', .5)],
        'contorno': 'descendente', 'style': 'diatonic', 'mode': 'minor',
        'tension': 5, 'emocion': 'melancolía',
        'desc': 'Descenso completo en modo menor. Más pesado y resignado que su equivalente mayor.',
    },
    {
        'id': 8, 'nombre': 'Vuelta (grupetto)', 'tipo': 'motivo',
        'pattern': [('3', .5), ('4', .5), ('3', .5), ('2', .5), ('3', 1)],
        'contorno': 'ondulante', 'style': 'baroque', 'mode': 'major',
        'tension': 3, 'emocion': 'alegría',
        'desc': 'Ornamento clásico que rodea una nota central por arriba y por abajo antes de asentarse.',
    },
    {
        'id': 9, 'nombre': 'Apoyatura simple', 'tipo': 'motivo',
        'pattern': [('2', 1.5), ('1', .5)],
        'contorno': 'descendente', 'style': 'romantic', 'mode': 'major',
        'tension': 5, 'emocion': 'nostalgia',
        'desc': 'Nota acentuada un grado por encima que resuelve por debajo. Suspiro melódico clásico.',
    },
    {
        'id': 10, 'nombre': 'Apoyatura cromática ascendente', 'tipo': 'motivo',
        'pattern': [('#4', 1.5), ('5', .5)],
        'contorno': 'ascendente', 'style': 'romantic', 'mode': 'major',
        'tension': 6, 'emocion': 'drama',
        'desc': 'Sensible cromática que tira con fuerza hacia el grado siguiente. Muy expresiva.',
    },
    {
        'id': 11, 'nombre': 'Paso cromático ascendente', 'tipo': 'motivo',
        'pattern': [('1', .5), ('#1', .5), ('2', .5), ('#2', .5), ('3', .5)],
        'contorno': 'ascendente', 'style': 'jazz', 'mode': 'major',
        'tension': 4, 'emocion': 'misterio',
        'desc': 'Notas de paso cromáticas entre grados diatónicos. Color jazzístico, ambiguo.',
    },
    {
        'id': 12, 'nombre': 'Cambiata', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('2', 1), ('4', 1)],
        'contorno': 'zigzag', 'style': 'baroque', 'mode': 'major',
        'tension': 4, 'emocion': 'ambigüedad',
        'desc': 'Salto que se aleja de la nota esperada antes de continuar por grados. Contrapunto renacentista.',
    },
    {
        'id': 13, 'nombre': 'Salto de sexta ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('6', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'esperanza',
        'desc': 'Salto amplio y consonante. My Bonnie, NBC chimes. Sensación de anhelo optimista.',
    },
    {
        'id': 14, 'nombre': 'Salto de séptima tenso', 'tipo': 'motivo',
        'pattern': [('1', 1), ('7', 2)],
        'contorno': 'salto', 'style': 'romantic', 'mode': 'minor',
        'tension': 7, 'emocion': 'angustia',
        'desc': 'Salto disonante hacia la sensible. Muy tenso, pide resolución inmediata.',
    },
    {
        'id': 15, 'nombre': 'Motivo pentatónico ascendente', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('5', .5), ('6', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'folk', 'mode': 'major',
        'tension': 2, 'emocion': 'alegría',
        'desc': 'Salta el 4º y el 7º grado, imitando la escala pentatónica. Abierto y folclórico.',
    },
    {
        'id': 16, 'nombre': 'Motivo blues ascendente', 'tipo': 'motivo',
        'pattern': [('1', .5), ('3', .5), ('4', .5), ('b5', .5), ('5', 1)],
        'contorno': 'ascendente', 'style': 'blues', 'mode': 'minor',
        'tension': 5, 'emocion': 'melancolía',
        'desc': 'Incluye la "blue note" (5ª disminuida) como nota de paso hacia la 5ª justa.',
    },
    {
        'id': 17, 'nombre': 'Tetracordo descendente (lamento)', 'tipo': 'motivo',
        'pattern': [('8', 1), ('7', 1), ('6', 1), ('5', 1)],
        'contorno': 'descendente', 'style': 'baroque', 'mode': 'minor',
        'tension': 7, 'emocion': 'angustia',
        'desc': 'Bajo de lamento barroco trasladado a la melodía: descenso implacable de cuatro grados.',
    },
    {
        'id': 18, 'nombre': 'Nota vecina superior', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('1', 1)],
        'contorno': 'ondulante', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Adorno mínimo: se toca el grado superior y se vuelve de inmediato a la nota de partida.',
    },
    {
        'id': 19, 'nombre': 'Nota vecina inferior', 'tipo': 'motivo',
        'pattern': [('1', .5), ('7,', .5), ('1', 1)],
        'contorno': 'ondulante', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Como la vecina superior pero tocando la sensible una octava abajo antes de volver.',
    },
    {
        'id': 20, 'nombre': 'Giro mixolidio', 'tipo': 'motivo',
        'pattern': [('5', 1), ('7', 1), ('8', 2)],
        'contorno': 'ascendente', 'style': 'modal', 'mode': 'mixolydian',
        'tension': 4, 'emocion': 'euforia',
        'desc': 'El 7º grado rebajado del modo mixolidio da un color abierto, sin sensible clásica.',
    },
    {
        'id': 21, 'nombre': 'Giro dórico', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('4', 1), ('6', 2)],
        'contorno': 'ascendente', 'style': 'modal', 'mode': 'dorian',
        'tension': 4, 'emocion': 'misterio',
        'desc': 'La 6ª natural del modo dórico (más brillante que la menor natural) define este giro.',
    },
    {
        'id': 22, 'nombre': 'Giro lidio', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('4', 1), ('5', 2)],
        'contorno': 'ascendente', 'style': 'modal', 'mode': 'lydian',
        'tension': 3, 'emocion': 'flotación',
        'desc': 'El 4º grado elevado del lidio da una sensación suspendida y sin gravedad.',
    },
    {
        'id': 23, 'nombre': 'Giro frigio', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('1', 1)],
        'contorno': 'ondulante', 'style': 'modal', 'mode': 'phrygian',
        'tension': 6, 'emocion': 'exotismo',
        'desc': 'La segunda menor del modo frigio hace que este vecino superior suene tenso y oscuro.',
    },
    {
        'id': 24, 'nombre': 'Giro frigio dominante (2ª aumentada)', 'tipo': 'motivo',
        'pattern': [('1', .75), ('2', .25), ('3', 1)],
        'contorno': 'salto', 'style': 'flamenco', 'mode': 'phrygian_dominant',
        'tension': 7, 'emocion': 'exotismo',
        'desc': 'El salto de segunda aumentada entre el 2º y 3º grado es la firma del cante flamenco.',
    },
    {
        'id': 25, 'nombre': 'Giro armónico menor (2ª aumentada)', 'tipo': 'motivo',
        'pattern': [('5', .5), ('6', .5), ('7', .5), ('8', 1)],
        'contorno': 'ascendente', 'style': 'romantic', 'mode': 'harmonic_minor',
        'tension': 7, 'emocion': 'drama',
        'desc': 'La sensible elevada del menor armónico crea un salto de 2ª aumentada muy característico.',
    },
    {
        'id': 26, 'nombre': 'Arco simple', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('2', .5), ('1', 1)],
        'contorno': 'arco', 'style': 'pop', 'mode': 'major',
        'tension': 2, 'emocion': 'alegría',
        'desc': 'Sube por grados y baja por el mismo camino. Gesto redondo, fácil de cantar.',
    },
    {
        'id': 27, 'nombre': 'Frase clásica (periodo)', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .5), ('5', .5), ('8', 1),
                    ('7', .5), ('6', .5), ('5', .5), ('r', .5), ('1', 1.5)],
        'contorno': 'arco', 'style': 'diatonic', 'mode': 'major',
        'tension': 3, 'emocion': 'esperanza',
        'desc': 'Sube en arpegio hasta la octava y desciende por grados, con una pequeña respiración antes '
                'de resolver en la tónica. Frase-tipo mozartiana, lista para insertarse con pocos cambios.',
    },
    {
        'id': 28, 'nombre': 'Frase de terceras descendentes (suspiro romántico)', 'tipo': 'frase',
        'pattern': [('8', .75), ('6', .25), ('4', .75), ('2', .25), ('r', .5), ('1', 2.5)],
        'contorno': 'descendente', 'style': 'romantic', 'mode': 'minor',
        'tension': 6, 'emocion': 'melancolía',
        'desc': 'Descenso por terceras con ritmo punteado (corchea con puntillo) y una respiración antes de '
                'la resolución final. Efecto de resignación progresiva, muy usado en baladas.',
    },
    {
        'id': 29, 'nombre': 'Secuencia ascendente', 'tipo': 'frase',
        'pattern': [('1', .5), ('2', .5), ('3', 1), ('2', .5), ('3', .5),
                    ('4', 1), ('3', .5), ('4', .5), ('5', 1.5)],
        'contorno': 'ascendente', 'style': 'baroque', 'mode': 'major',
        'tension': 5, 'emocion': 'esperanza',
        'desc': 'La misma célula corta-corta-larga se repite transportada un grado más arriba cada vez. '
                'El empuje mecánico es intencional: así se usan las secuencias en el estilo barroco.',
    },
    {
        'id': 30, 'nombre': 'Secuencia descendente', 'tipo': 'frase',
        'pattern': [('8', .5), ('7', .5), ('6', 1), ('7', .5), ('6', .5),
                    ('5', 1), ('6', .5), ('5', .5), ('4', 1.5)],
        'contorno': 'descendente', 'style': 'baroque', 'mode': 'major',
        'tension': 4, 'emocion': 'nostalgia',
        'desc': 'Célula secuencial descendente con el mismo perfil corta-corta-larga, imagen especular '
                'de la secuencia ascendente.',
    },
    {
        'id': 31, 'nombre': 'Semifrase — pregunta', 'tipo': 'frase',
        'pattern': [('1', .5), ('2', .25), ('3', .25), ('5', 1.5), ('r', .5)],
        'contorno': 'ascendente', 'style': 'pop', 'mode': 'major',
        'tension': 4, 'emocion': 'ambigüedad',
        'desc': 'Termina sostenida en el 5º grado y se apaga en un silencio, sin resolver. '
                'Pide una respuesta (ver #32): encadénalas con la respuesta empezando justo tras el silencio.',
    },
    {
        'id': 32, 'nombre': 'Semifrase — respuesta', 'tipo': 'frase',
        'pattern': [('r', .25), ('6', .25), ('5', .5), ('3', .5), ('1', 2.5)],
        'contorno': 'descendente', 'style': 'pop', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Empieza con una breve anacrusa tras un silencio y cierra firme sobre la tónica. '
                'Diseñada para encadenarse justo después de la semifrase de pregunta (#31).',
    },
    {
        'id': 33, 'nombre': 'Frase con clímax', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .25), ('5', .25), ('6', .5), ('8', 1.5),
                    ('r', .25), ('6', .25), ('5', .5), ('3', .5), ('1', 1.5)],
        'contorno': 'arco', 'style': 'romantic', 'mode': 'major',
        'tension': 6, 'emocion': 'drama',
        'desc': 'Arco amplio con un pico claro en la octava y una breve pausa dramática justo después, '
                'antes del descenso simétrico.',
    },
    {
        'id': 34, 'nombre': 'Frase modal circular', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .5), ('5', .5), ('3', .5),
                    ('1', .5), ('7,', .5), ('r', .25), ('1', .75)],
        'contorno': 'estatico', 'style': 'modal', 'mode': 'dorian',
        'tension': 3, 'emocion': 'misterio',
        'desc': 'Gira alrededor de la tríada dórica sin desarrollo direccional, con una respiración antes '
                'de cerrar cada vuelta. Hipnótica, tipo banda sonora.',
    },
    {
        'id': 35, 'nombre': 'Descenso frigio (cadencia andaluza melódica)', 'tipo': 'frase',
        'pattern': [('4', .75), ('3', .25), ('2', 1), ('1', 2)],
        'contorno': 'descendente', 'style': 'flamenco', 'mode': 'phrygian',
        'tension': 7, 'emocion': 'drama',
        'desc': 'El equivalente melódico del descenso andaluz armónico, con el arrastre punteado típico '
                'del cante. Ineludible y teatral.',
    },
    {
        'id': 36, 'nombre': 'Frase blues (call)', 'tipo': 'frase',
        'pattern': [('1', .67), ('3', .33), ('4', .67), ('b5', .33),
                    ('5', 1), ('3', .5), ('1', 1.5)],
        'contorno': 'arco', 'style': 'blues', 'mode': 'minor',
        'tension': 5, 'emocion': 'melancolía',
        'desc': 'Frase de "llamada" típica del blues, con swing (corcheas desiguales): sube con la blue '
                'note y baja de vuelta a la tónica.',
    },
    {
        'id': 37, 'nombre': 'Frase de planeo (impresionista)', 'tipo': 'frase',
        'pattern': [('1', 1.5), ('2', .5), ('3', 1), ('5', .75), ('6', .75), ('8', 2)],
        'contorno': 'ascendente', 'style': 'impressionist', 'mode': 'lydian',
        'tension': 2, 'emocion': 'flotación',
        'desc': 'Ascenso de duraciones irregulares y sin acento métrico claro, saltando el 4º grado sobre '
                'el modo lidio. Sensación de rubato y de flotar sin gravedad, no de escala.',
    },
    {
        'id': 38, 'nombre': 'Fanfarria (saltos de quinta y octava)', 'tipo': 'motivo',
        'pattern': [('1', 1), ('5', 1), ('8', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'Saltos amplios y consonantes. Efecto de llamada, apertura o fanfarria.',
    },
    {
        'id': 39, 'nombre': 'Salto de tritono (blue note jazz)', 'tipo': 'motivo',
        'pattern': [('1', .5), ('#4', .5), ('5', 1)],
        'contorno': 'salto', 'style': 'jazz', 'mode': 'major',
        'tension': 6, 'emocion': 'misterio',
        'desc': 'Salto de tritono que resuelve por semitono a la 5ª. Color disonante muy jazzístico.',
    },
    {
        'id': 40, 'nombre': 'Ostinato de quinta', 'tipo': 'frase',
        'pattern': [('1', .5), ('5', .5), ('1', .5), ('5', .5)],
        'contorno': 'estatico', 'style': 'modal', 'mode': 'minor',
        'tension': 3, 'emocion': 'flotación',
        'desc': 'Vaivén repetitivo entre tónica y quinta. La uniformidad rítmica es intencional: es la '
                'base hipnótica sobre la que se superponen otras capas, no una frase con arco propio.',
    },
    # ── Ampliación v2 (41-100) ──────────────────────────────────────────────
    {
        'id': 41, 'nombre': 'Arpegio de séptima mayor ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('7', 1)],
        'contorno': 'ascendente', 'style': 'jazz', 'mode': 'major',
        'tension': 3, 'emocion': 'esperanza',
        'desc': 'Tétrada de séptima mayor recorrida hacia arriba. Color jazz suave y abierto.',
    },
    {
        'id': 42, 'nombre': 'Arpegio de séptima mayor descendente', 'tipo': 'motivo',
        'pattern': [('7', 1), ('5', 1), ('3', 1), ('1', 1)],
        'contorno': 'descendente', 'style': 'jazz', 'mode': 'major',
        'tension': 3, 'emocion': 'reposo',
        'desc': 'La misma tétrada de séptima mayor recorrida hacia abajo. Cierre suave.',
    },
    {
        'id': 43, 'nombre': 'Arpegio de séptima menor ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('7', 1)],
        'contorno': 'ascendente', 'style': 'jazz', 'mode': 'minor',
        'tension': 5, 'emocion': 'melancolía',
        'desc': 'Tétrada m7 (1-b3-5-b7) ascendente: los grados del modo menor ya dan la séptima menor.',
    },
    {
        'id': 44, 'nombre': 'Salto de cuarta ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('4', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 2, 'emocion': 'alegría',
        'desc': 'Anacrusa clásica: salto de cuarta justa hacia arriba antes de asentarse.',
    },
    {
        'id': 45, 'nombre': 'Salto de cuarta descendente (V-I melódico)', 'tipo': 'motivo',
        'pattern': [('5', 1), ('1', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Del 5º grado a la tónica: el gesto melódico más común de cierre conclusivo.',
    },
    {
        'id': 46, 'nombre': 'Salto de quinta ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('5', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'Salto amplio y estable hacia la dominante. Abre espacio, sensación expansiva.',
    },
    {
        'id': 47, 'nombre': 'Salto de quinta descendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('4,', 2)],
        'contorno': 'salto', 'style': 'romantic', 'mode': 'major',
        'tension': 4, 'emocion': 'melancolía',
        'desc': 'Caída de una quinta desde la tónica. Gesto de abatimiento o resignación.',
    },
    {
        'id': 48, 'nombre': 'Salto de sexta descendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3,', 2)],
        'contorno': 'salto', 'style': 'romantic', 'mode': 'major',
        'tension': 4, 'emocion': 'nostalgia',
        'desc': 'Caída de una sexta desde la tónica hasta la mediante de la octava inferior.',
    },
    {
        'id': 49, 'nombre': 'Salto de séptima descendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('2,', 2)],
        'contorno': 'salto', 'style': 'jazz', 'mode': 'major',
        'tension': 5, 'emocion': 'drama',
        'desc': 'Caída amplia y poco común: un salto de séptima genera tensión inmediata al oído.',
    },
    {
        'id': 50, 'nombre': 'Salto de octava ascendente', 'tipo': 'motivo',
        'pattern': [('1', 1), ('8', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'El salto melódico más reconocible ("Over the Rainbow"): apertura total, sin ambigüedad.',
    },
    {
        'id': 51, 'nombre': 'Salto de octava descendente', 'tipo': 'motivo',
        'pattern': [('8', 1), ('1', 2)],
        'contorno': 'salto', 'style': 'romantic', 'mode': 'major',
        'tension': 3, 'emocion': 'drama',
        'desc': 'Caída de una octava completa. Gesto de derrumbe o de gran gesto teatral.',
    },
    {
        'id': 52, 'nombre': 'Trino', 'tipo': 'motivo',
        'pattern': [('1', .25), ('2', .25), ('1', .25), ('2', .25), ('1', 1)],
        'contorno': 'ondulante', 'style': 'baroque', 'mode': 'major',
        'tension': 3, 'emocion': 'alegría',
        'desc': 'Alternancia rápida entre la nota principal y el grado superior antes de asentarse.',
    },
    {
        'id': 53, 'nombre': 'Mordente inferior', 'tipo': 'motivo',
        'pattern': [('1', .25), ('7,', .25), ('1', 1.5)],
        'contorno': 'ondulante', 'style': 'baroque', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Adorno rápido hacia el vecino inferior antes de sostener la nota principal.',
    },
    {
        'id': 54, 'nombre': 'Giro barroco (Fonte)', 'tipo': 'motivo',
        'pattern': [('4', .5), ('3', .5), ('3', .5), ('2', .5)],
        'contorno': 'descendente', 'style': 'baroque', 'mode': 'major',
        'tension': 4, 'emocion': 'melancolía',
        'desc': 'Célula de dos notas que cae y se repite transportada un grado más abajo. Figura barroca clásica.',
    },
    {
        'id': 55, 'nombre': 'Giro barroco (Monte)', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('2', .5), ('3', .5)],
        'contorno': 'ascendente', 'style': 'baroque', 'mode': 'major',
        'tension': 4, 'emocion': 'esperanza',
        'desc': 'Imagen especular del Fonte: la célula sube transportada un grado cada vez.',
    },
    {
        'id': 56, 'nombre': 'Rosalía (secuencia por cuartas)', 'tipo': 'frase',
        'pattern': [('1', .5), ('4', 1), ('2', .5), ('5', 1), ('3', .5), ('6', 1.5)],
        'contorno': 'ascendente', 'style': 'baroque', 'mode': 'major',
        'tension': 5, 'emocion': 'esperanza',
        'desc': 'Célula corta-larga de cuarta ascendente repetida un grado más arriba cada vez. '
                'Muy usada en música española antigua.',
    },
    {
        'id': 57, 'nombre': 'Frase blues (call-response completa)', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .5), ('4', .5), ('b5', .5),
                    ('5', 1), ('r', .5), ('4', .5), ('3', .5), ('1', 1.5)],
        'contorno': 'arco', 'style': 'blues', 'mode': 'minor',
        'tension': 6, 'emocion': 'melancolía',
        'desc': 'Sube con la blue note hasta la 5ª, respira, y baja de vuelta: llamada y respuesta '
                'en una sola frase, con el silencio marcando el relevo entre ambas.',
    },
    {
        'id': 58, 'nombre': 'Riff pentatónico menor (rock/blues)', 'tipo': 'motivo',
        'pattern': [('1', .5), ('3', .5), ('4', .5), ('5', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'blues', 'mode': 'minor',
        'tension': 5, 'emocion': 'euforia',
        'desc': 'Pentatónica menor (1-b3-4-5-b7) recorrida hacia arriba. El riff más usado del rock y el blues.',
    },
    {
        'id': 59, 'nombre': 'Riff dórico (rock progresivo)', 'tipo': 'frase',
        'pattern': [('1', .75), ('3', .25), ('4', .5), ('5', .5), ('6', 1),
                    ('5', .5), ('4', .25), ('3', .25), ('1', 1)],
        'contorno': 'arco', 'style': 'modal', 'mode': 'dorian',
        'tension': 4, 'emocion': 'misterio',
        'desc': 'Arco sincopado sobre la tríada dórica, con la 6ª natural como color distintivo '
                'y como pico rítmico del riff.',
    },
    {
        'id': 60, 'nombre': 'Hook pop (nota repetida + vecina)', 'tipo': 'motivo',
        'pattern': [('5', .5), ('5', .5), ('6', .5), ('5', 1.5)],
        'contorno': 'estatico', 'style': 'pop', 'mode': 'major',
        'tension': 2, 'emocion': 'alegría',
        'desc': 'Repetición de una nota con un pequeño desvío al vecino superior. Gancho pegadizo típico.',
    },
    {
        'id': 61, 'nombre': 'Vals melódico', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('3', 1), ('1', 3)],
        'contorno': 'arco', 'style': 'romantic', 'mode': 'major',
        'tension': 3, 'emocion': 'nostalgia',
        'desc': 'Arco de tríada que sube y baja con aire ternario, típico del vals decimonónico.',
    },
    {
        'id': 62, 'nombre': 'Berceuse (vaivén de tercera)', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('1', 1), ('3', 1)],
        'contorno': 'estatico', 'style': 'romantic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Vaivén suave entre la tónica y la tercera. Efecto de arrullo, canción de cuna.',
    },
    {
        'id': 63, 'nombre': 'Himno (grados conjuntos, valores largos)', 'tipo': 'motivo',
        'pattern': [('1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 2)],
        'contorno': 'ascendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'solemnidad',
        'desc': 'Movimiento por grados conjuntos con valores largos. Carácter solemne, tipo himno o coral.',
    },
    {
        'id': 64, 'nombre': 'Coral (movimiento conjunto)', 'tipo': 'motivo',
        'pattern': [('5', 1), ('4', 1), ('3', 1), ('4', 1), ('5', 2)],
        'contorno': 'ondulante', 'style': 'baroque', 'mode': 'major',
        'tension': 3, 'emocion': 'solemnidad',
        'desc': 'Vaivén conjunto alrededor del 4º grado, en el estilo de un coral de Bach.',
    },
    {
        'id': 65, 'nombre': 'Fanfarria de cuarta y octava', 'tipo': 'motivo',
        'pattern': [('1', 1), ('4', 1), ('8', 2)],
        'contorno': 'salto', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'Saltos ascendentes consecutivos de cuarta y quinta. Efecto de llamada o apertura triunfal.',
    },
    {
        'id': 66, 'nombre': 'Llamada de corno (quinta-octava)', 'tipo': 'motivo',
        'pattern': [('5', 1), ('8', 1), ('5', 2)],
        'contorno': 'salto', 'style': 'romantic', 'mode': 'major',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'Salto de quinta a octava y vuelta, imitando las llamadas de caza de trompa.',
    },
    {
        'id': 67, 'nombre': 'Escala bebop mayor', 'tipo': 'frase',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5), ('5', .5),
                    ('#5', .5), ('6', .5), ('7', .5), ('8', 1)],
        'contorno': 'ascendente', 'style': 'jazz', 'mode': 'major',
        'tension': 5, 'emocion': 'esperanza',
        'desc': 'Escala mayor con una nota cromática añadida entre el 5º y el 6º grado, para que los '
                'tonos del acorde caigan en tiempo fuerte al tocarla en corcheas. Se toca deliberadamente '
                'pareja (así funciona el recurso bebop); solo la nota de llegada se alarga.',
    },
    {
        'id': 68, 'nombre': 'ii–V–I melódico', 'tipo': 'motivo',
        'pattern': [('2', 1), ('5', 1), ('1', 2)],
        'contorno': 'zigzag', 'style': 'jazz', 'mode': 'major',
        'tension': 4, 'emocion': 'esperanza',
        'desc': 'Traza melódicamente el movimiento armónico ii-V-I: sube una cuarta, baja una quinta.',
    },
    {
        'id': 69, 'nombre': 'Turnaround melódico', 'tipo': 'motivo',
        'pattern': [('1', 1), ('6', 1), ('2', 1), ('5', 1)],
        'contorno': 'zigzag', 'style': 'jazz', 'mode': 'major',
        'tension': 5, 'emocion': 'ambigüedad',
        'desc': 'Sigue el movimiento armónico I-vi-ii-V. Gesto que empuja hacia una nueva vuelta.',
    },
    {
        'id': 70, 'nombre': 'Lick de blues (bend de tercera)', 'tipo': 'motivo',
        'pattern': [('1', .5), ('b3', .25), ('3', .25), ('5', 1.5)],
        'contorno': 'ondulante', 'style': 'blues', 'mode': 'major',
        'tension': 5, 'emocion': 'melancolía',
        'desc': 'Simula el "bend" del blues: la tercera menor se desliza hacia la mayor antes de saltar a la 5ª.',
    },
    {
        'id': 71, 'nombre': 'Arpegio de novena', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('5', 1), ('7', 1), ("2'", 1)],
        'contorno': 'ascendente', 'style': 'jazz', 'mode': 'major',
        'tension': 4, 'emocion': 'esperanza',
        'desc': 'Arpegio extendido hasta la novena (2ª una octava arriba). Color jazz abierto y moderno.',
    },
    {
        'id': 72, 'nombre': 'Patrón de campanas (bell pattern)', 'tipo': 'motivo',
        'pattern': [('1', .5), ('5', .5), ('2', .5), ('6', .5), ('3', .5)],
        'contorno': 'zigzag', 'style': 'modal', 'mode': 'major',
        'tension': 3, 'emocion': 'flotación',
        'desc': 'Saltos alternados entre grados no contiguos. Textura minimalista tipo "bell pattern".',
    },
    {
        'id': 73, 'nombre': 'Ostinato de cuarta', 'tipo': 'motivo',
        'pattern': [('1', .5), ('4', .5), ('1', .5), ('4', .5)],
        'contorno': 'estatico', 'style': 'modal', 'mode': 'minor',
        'tension': 3, 'emocion': 'misterio',
        'desc': 'Vaivén repetitivo entre la tónica y la cuarta. Base hipnótica en modo menor.',
    },
    {
        'id': 74, 'nombre': 'Pedal de tónica con vecinas', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('1', .5), ('7,', .5), ('1', 1)],
        'contorno': 'ondulante', 'style': 'modal', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'La tónica se adorna con ambos vecinos, superior e inferior, sin abandonar el centro tonal.',
    },
    {
        'id': 75, 'nombre': 'Pedal de dominante', 'tipo': 'motivo',
        'pattern': [('5', .5), ('5', .5), ('5', 1), ('1', 2)],
        'contorno': 'estatico', 'style': 'romantic', 'mode': 'major',
        'tension': 5, 'emocion': 'esperanza',
        'desc': 'Insistencia sobre el 5º grado antes de resolver: acumula expectativa antes del cierre.',
    },
    {
        'id': 76, 'nombre': 'Giro napolitano melódico', 'tipo': 'motivo',
        'pattern': [('b2', 1.5), ('1', .5)],
        'contorno': 'descendente', 'style': 'romantic', 'mode': 'major',
        'tension': 7, 'emocion': 'drama',
        'desc': 'El 2º grado rebajado (acorde napolitano) resuelve a la tónica. Muy expresivo y oscuro.',
    },
    {
        'id': 77, 'nombre': 'Lamento cromático completo (octava a quinta)', 'tipo': 'frase',
        'pattern': [('8', .5), ('7', .5), ('b7', .5), ('6', .5), ('b6', .5), ('5', 1)],
        'contorno': 'descendente', 'style': 'baroque', 'mode': 'major',
        'tension': 8, 'emocion': 'angustia',
        'desc': 'Descenso cromático nota a nota desde la octava hasta la 5ª. El paso uniforme es intencional: '
                'es el "ground" del lamento barroco, un tranco inexorable que representa el destino o el duelo.',
    },
    {
        'id': 78, 'nombre': 'Escala armónica menor ascendente', 'tipo': 'escala',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5),
                    ('5', .5), ('6', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'romantic', 'mode': 'harmonic_minor',
        'tension': 5, 'emocion': 'drama',
        'desc': 'Recorrido completo de la escala menor armónica, con su característico salto de 2ª aumentada. '
                'Referencia de estudio: no tiene arco propio, es solo el material en bruto del modo.',
    },
    {
        'id': 79, 'nombre': 'Escala armónica menor descendente', 'tipo': 'escala',
        'pattern': [('8', .5), ('7', .5), ('6', .5), ('5', .5),
                    ('4', .5), ('3', .5), ('2', .5), ('1', .5)],
        'contorno': 'descendente', 'style': 'romantic', 'mode': 'harmonic_minor',
        'tension': 5, 'emocion': 'drama',
        'desc': 'Descenso completo de la escala menor armónica. Referencia de estudio, no una frase con forma propia.',
    },
    {
        'id': 80, 'nombre': 'Escala lidia ascendente', 'tipo': 'escala',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5),
                    ('5', .5), ('6', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'impressionist', 'mode': 'lydian',
        'tension': 2, 'emocion': 'flotación',
        'desc': 'Recorrido completo del modo lidio: el 4º grado elevado da una sensación suspendida. '
                'Referencia de estudio para oír el color del modo, no una frase con forma propia.',
    },
    {
        'id': 81, 'nombre': 'Escala mixolidia descendente', 'tipo': 'escala',
        'pattern': [('8', .5), ('7', .5), ('6', .5), ('5', .5),
                    ('4', .5), ('3', .5), ('2', .5), ('1', .5)],
        'contorno': 'descendente', 'style': 'modal', 'mode': 'mixolydian',
        'tension': 3, 'emocion': 'euforia',
        'desc': 'Descenso completo del modo mixolidio, sin sensible clásica: color abierto y rockero. '
                'Referencia de estudio, no una frase con forma propia.',
    },
    {
        'id': 82, 'nombre': 'Escala frigia descendente', 'tipo': 'escala',
        'pattern': [('8', .5), ('7', .5), ('6', .5), ('5', .5),
                    ('4', .5), ('3', .5), ('2', .5), ('1', .5)],
        'contorno': 'descendente', 'style': 'modal', 'mode': 'phrygian',
        'tension': 6, 'emocion': 'exotismo',
        'desc': 'Descenso completo del modo frigio: la 2ª menor le da un color oscuro y exótico. '
                'Referencia de estudio, no una frase con forma propia.',
    },
    {
        'id': 83, 'nombre': 'Escala dórica ascendente', 'tipo': 'escala',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5),
                    ('5', .5), ('6', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'modal', 'mode': 'dorian',
        'tension': 3, 'emocion': 'misterio',
        'desc': 'Recorrido completo del modo dórico: la 6ª natural lo distingue del menor natural. '
                'Referencia de estudio, no una frase con forma propia.',
    },
    {
        'id': 84, 'nombre': 'Escala frigia dominante ascendente', 'tipo': 'escala',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5),
                    ('5', .5), ('6', .5), ('7', .5), ('8', .5)],
        'contorno': 'ascendente', 'style': 'flamenco', 'mode': 'phrygian_dominant',
        'tension': 6, 'emocion': 'exotismo',
        'desc': 'Recorrido completo de la escala "española": el salto de 2ª aumentada entre el 2º y 3º grado. '
                'Referencia de estudio, no una frase con forma propia.',
    },
    {
        'id': 85, 'nombre': 'Arpegio disminuido (vii°)', 'tipo': 'motivo',
        'pattern': [('7', 1), ("2'", 1), ("4'", 1)],
        'contorno': 'ascendente', 'style': 'baroque', 'mode': 'major',
        'tension': 7, 'emocion': 'angustia',
        'desc': 'Tríada disminuida construida sobre la sensible, toda diatónica. Muy inestable, pide resolución.',
    },
    {
        'id': 86, 'nombre': 'Arpegio aumentado', 'tipo': 'motivo',
        'pattern': [('1', 1), ('3', 1), ('#5', 1)],
        'contorno': 'ascendente', 'style': 'impressionist', 'mode': 'major',
        'tension': 6, 'emocion': 'misterio',
        'desc': 'Tríada aumentada (5ª elevada). Simétrica y ambigua, muy usada en el impresionismo.',
    },
    {
        'id': 87, 'nombre': 'Motivo de tresillo ondulante', 'tipo': 'motivo',
        'pattern': [('1', .33), ('3', .33), ('2', .34), ('1', .33), ('3', .33), ('2', .34)],
        'contorno': 'ondulante', 'style': 'jazz', 'mode': 'major',
        'tension': 3, 'emocion': 'alegría',
        'desc': 'Célula de tresillo que sube y baja, repetida dos veces. Sensación fluida y rebotante.',
    },
    {
        'id': 88, 'nombre': 'Secuencia por terceras ascendente', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', 1), ('2', .5), ('4', 1), ('3', .5), ('5', 1.5)],
        'contorno': 'ascendente', 'style': 'baroque', 'mode': 'major',
        'tension': 5, 'emocion': 'esperanza',
        'desc': 'La célula corta-larga de tercera se repite transportada un grado más arriba en cada compás.',
    },
    {
        'id': 89, 'nombre': 'Secuencia por terceras descendente', 'tipo': 'frase',
        'pattern': [('8', .5), ('6', 1), ('7', .5), ('5', 1), ('6', .5), ('4', 1.5)],
        'contorno': 'descendente', 'style': 'baroque', 'mode': 'major',
        'tension': 4, 'emocion': 'nostalgia',
        'desc': 'Imagen especular de la secuencia por terceras ascendente.',
    },
    {
        'id': 90, 'nombre': 'Arco romántico extendido', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .5), ('5', .5), ('8', 1),
                    ('r', .25), ('6', .5), ('4', .5), ('2', .5), ('1', 1.75)],
        'contorno': 'arco', 'style': 'romantic', 'mode': 'minor',
        'tension': 6, 'emocion': 'drama',
        'desc': 'Arco amplio que sube en arpegio hasta la octava, respira en el pico, y baja por terceras '
                'hasta la tónica.',
    },
    {
        'id': 91, 'nombre': 'Pregunta-respuesta modal (dórico)', 'tipo': 'frase',
        'pattern': [('1', .5), ('3', .5), ('5', .5), ('6', 1),
                    ('r', .25), ('5', .5), ('3', .5), ('1', 1.25)],
        'contorno': 'arco', 'style': 'modal', 'mode': 'dorian',
        'tension': 4, 'emocion': 'misterio',
        'desc': 'Sube hasta la 6ª dórica característica, respira, y regresa a la tónica: pregunta y '
                'respuesta en un solo gesto.',
    },
    {
        'id': 92, 'nombre': 'Frase con anacrusa', 'tipo': 'motivo',
        'pattern': [('5', .25), ('1', .75), ('3', 1), ('5', 1), ('8', 1.5)],
        'contorno': 'ascendente', 'style': 'pop', 'mode': 'major',
        'tension': 3, 'emocion': 'alegría',
        'desc': 'Nota de anacrusa corta antes del tiempo fuerte, seguida de un ascenso en arpegio.',
    },
    {
        'id': 93, 'nombre': 'Cadencia melódica auténtica (V-I)', 'tipo': 'motivo',
        'pattern': [('5', 1), ('7', 1), ('8', 2)],
        'contorno': 'ascendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 4, 'emocion': 'esperanza',
        'desc': 'La sensible empuja hacia la tónica octavada. Versión melódica de la cadencia auténtica.',
    },
    {
        'id': 94, 'nombre': 'Cadencia melódica plagal (IV-I)', 'tipo': 'motivo',
        'pattern': [('4', 1), ('3', 1), ('2', 1), ('1', 1)],
        'contorno': 'descendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 2, 'emocion': 'reposo',
        'desc': 'Descenso suave por grados desde la subdominante. Versión melódica del "amén" plagal.',
    },
    {
        'id': 95, 'nombre': 'Media cadencia frigia', 'tipo': 'motivo',
        'pattern': [('4', 1), ('3', 1), ('2', 2)],
        'contorno': 'descendente', 'style': 'flamenco', 'mode': 'phrygian',
        'tension': 6, 'emocion': 'drama',
        'desc': 'Descenso que se detiene en el 2º grado frigio, medio tono por encima de la tónica. Final abierto y tenso.',
    },
    {
        'id': 96, 'nombre': 'Clímax tardío (arco asimétrico)', 'tipo': 'frase',
        'pattern': [('1', .5), ('2', .5), ('3', .5), ('4', .5), ('5', .5),
                    ('6', .5), ('7', .5), ('8', 1.5), ('r', .25), ('6', 1.75)],
        'contorno': 'arco', 'style': 'romantic', 'mode': 'major',
        'tension': 6, 'emocion': 'drama',
        'desc': 'Ascenso escalonado completo con el pico desplazado hacia el final, una pausa dramática '
                'justo tras el clímax, y una caída breve que lo resuelve.',
    },
    {
        'id': 97, 'nombre': 'Ostinato mixolidio', 'tipo': 'motivo',
        'pattern': [('1', .5), ('5', .5), ('7', .5), ('5', .5)],
        'contorno': 'estatico', 'style': 'modal', 'mode': 'mixolydian',
        'tension': 4, 'emocion': 'euforia',
        'desc': 'Vaivén entre tónica, quinta y séptima rebajada. Color abierto, muy usado en rock modal.',
    },
    {
        'id': 98, 'nombre': 'Drone con giro melódico', 'tipo': 'motivo',
        'pattern': [('1', 1), ('1', 1), ('3', 1), ('1', 1)],
        'contorno': 'estatico', 'style': 'modal', 'mode': 'minor',
        'tension': 3, 'emocion': 'flotación',
        'desc': 'La tónica se repite como un pedal, interrumpida por un breve giro a la tercera.',
    },
    {
        'id': 99, 'nombre': 'Giro de blues mayor', 'tipo': 'motivo',
        'pattern': [('1', .5), ('2', .5), ('b3', .5), ('3', .5), ('5', 1)],
        'contorno': 'ascendente', 'style': 'blues', 'mode': 'major',
        'tension': 5, 'emocion': 'nostalgia',
        'desc': 'Sube por grados y pasa por la tercera menor como nota de blues antes de asentarse en la mayor.',
    },
    {
        'id': 100, 'nombre': 'Frase final (coda melódica)', 'tipo': 'motivo',
        'pattern': [('3', 1), ('2', 1), ('1', 3)],
        'contorno': 'descendente', 'style': 'diatonic', 'mode': 'major',
        'tension': 1, 'emocion': 'reposo',
        'desc': 'Descenso simple y sostenido hasta la tónica. El cierre más neutro y conclusivo del catálogo.',
    },
]

TABLE_BY_ID = {e['id']: e for e in TABLE}


def _compute_signature(entry: dict) -> tuple:
    """Intervalos (deltas en semitonos) y razones de duración entre notas
    consecutivas del patrón — invariantes por transposición, usados para
    emparejar contra una melodía real en --analyze-midi. Los silencios se
    excluyen: la firma se calcula solo sobre las notas que realmente suenan."""
    sounding = [(tok, d) for tok, d in entry['pattern'] if not is_rest(tok)]
    semis = [resolve_degree(tok, entry['mode']) for tok, _ in sounding]
    intervals = [semis[i + 1] - semis[i] for i in range(len(semis) - 1)]
    durs = [d for _, d in sounding]
    dur_ratios = [
        (durs[i + 1] / durs[i]) if durs[i] else 1.0
        for i in range(len(durs) - 1)
    ]
    return intervals, dur_ratios


for _e in TABLE:
    _e['intervals'], _e['dur_ratios'] = _compute_signature(_e)
    _e['degrees'] = degrees_display(_e['pattern'])
del _e

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
    'tenso':         ['angustia', 'drama'],
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
    'flotante':      ['flotación'],
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
# ENTRADA PERSONALIZADA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_custom_motif(text: str, default_beats: float) -> list:
    """Parsea 'grado[:beats] grado[:beats] ...' -> lista de (grado, beats).
    'r' (o 'r:beats') representa un silencio."""
    result = []
    for token in text.strip().split():
        if ':' in token:
            deg, beats = token.split(':', 1)
            try:
                beats = float(beats)
            except ValueError:
                raise ValueError(f"Duración inválida en '{token}'")
        else:
            deg, beats = token, default_beats
        if not is_rest(deg) and not DEGREE_RE.match(deg):
            raise ValueError(f"Grado inválido: '{deg}'")
        result.append((deg.lower() if is_rest(deg) else deg, beats))
    if not result:
        raise ValueError("Motivo personalizado vacío")
    return result


def make_custom_entry(text: str, mode: str, default_beats: float) -> dict:
    pattern = parse_custom_motif(text, default_beats)
    entry = {
        'id': None, 'nombre': 'Personalizado', 'tipo': 'custom',
        'pattern': pattern, 'contorno': '—', 'style': 'custom', 'mode': mode,
        'tension': None, 'emocion': '—', 'desc': 'Motivo definido por el usuario.',
    }
    entry['intervals'], entry['dur_ratios'] = _compute_signature(entry)
    entry['degrees'] = degrees_display(pattern)
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE GRADOS → NOTAS CONCRETAS
# ═══════════════════════════════════════════════════════════════════════════════

def midi_to_name(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def resolve_entry(entry: dict, tonic_pc: int, octave: int = 5,
                  bars: int = 4, beats_per_bar: int = 4,
                  reps: int = None) -> list:
    """
    Expande el patrón de grados de una entrada a una lista de notas
    concretas (pitch MIDI absoluto, no solo clase de altura, porque el
    registro/contorno de una melodía importa).

    Igual que en chord_table.py: si `reps` se especifica, el motivo se
    repite exactamente esa cantidad de veces en su totalidad; si no, se
    rellena hasta cubrir `bars` compases (puede truncar el último ciclo).
    """
    pattern   = entry['pattern']
    mode      = entry['mode']
    beats_pat = sum(d for _, d in pattern)
    total     = beats_pat * reps if reps is not None else bars * beats_per_bar

    repeated = []
    acc = 0
    while acc < total:
        for tok, dur in pattern:
            repeated.append((tok, dur))
            acc += dur
            if acc >= total:
                break

    result, current = [], 0.0
    for tok, dur in repeated:
        if current >= total:
            break
        dur = min(dur, total - current)
        if is_rest(tok):
            result.append({
                'degree':         'r',
                'note':           'R',
                'midi':           None,
                'duration_beats': float(dur),
                'bar':            int(current // beats_per_bar),
                'beat':           current % beats_per_bar,
            })
            current += dur
            continue
        semis = resolve_degree(tok, mode)
        midi = tonic_pc + octave * 12 + semis
        midi = max(0, min(127, midi))
        result.append({
            'degree':         tok,
            'note':           midi_to_name(midi),
            'midi':           midi,
            'duration_beats': float(dur),
            'bar':            int(current // beats_per_bar),
            'beat':           current % beats_per_bar,
        })
        current += dur
    return result


def motif_to_text(notes: list) -> str:
    """Formato 'nota:beats' — línea melódica lista para melody_adapter.py."""
    parts = []
    for n in notes:
        dur = n['duration_beats']
        parts.append(f"{n['note']}:{int(dur) if dur == int(dur) else f'{dur:.2f}'}")
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def motif_to_midi(notes: list, ticks_per_beat: int, tempo_bpm: float,
                  output_path: str, velocity: int = 80) -> None:
    """Exporta la línea melódica como MIDI monofónico."""
    if not MIDO_OK:
        print("[ERROR] mido no disponible. Instala con: pip install mido")
        return

    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',
                                  tempo=int(60_000_000 / tempo_bpm), time=0))
    track.append(mido.MetaMessage('track_name', name='Motif Table', time=0))
    track.append(mido.Message('program_change', channel=0, program=0, time=0))

    pending_delay = 0
    for n in notes:
        dur_ticks = max(1, int(n['duration_beats'] * ticks_per_beat))
        if n['midi'] is None:  # silencio: no emite nota, solo acumula tiempo
            pending_delay += dur_ticks
            continue
        track.append(mido.Message('note_on', channel=0, note=n['midi'],
                                  velocity=velocity, time=pending_delay))
        pending_delay = 0
        track.append(mido.Message('note_off', channel=0, note=n['midi'],
                                  velocity=0, time=dur_ticks))
    mid.save(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS MIDI → MOTIVOS DEL CATÁLOGO
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_midi_raw(midi_path: str):
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    events = []
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            events.append((abs_t, msg))
    events.sort(key=lambda x: x[0])
    return events, tpb


def extract_melody(midi_path: str, channel: int = None,
                   min_dur: float = 0.0) -> tuple:
    """
    Extrae una línea melódica monofónica aproximada de un MIDI: si hay
    varias notas simultáneas, se queda con la más aguda (asunción
    razonable para una melodía principal por encima de un acompañamiento).
    Devuelve (lista de eventos {start,end,note,duration_beats}, ticks_per_beat).
    """
    events, tpb = _parse_midi_raw(midi_path)
    active = {}
    raw_notes = []
    for t, msg in events:
        if msg.type not in ('note_on', 'note_off'):
            continue
        if channel is not None and msg.channel != channel:
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            active.setdefault(msg.note, []).append(t)
        else:
            starts = active.get(msg.note)
            if starts:
                start = starts.pop(0)
                raw_notes.append((start, t, msg.note))

    raw_notes.sort(key=lambda x: x[0])

    # Reducir polifonía: agrupar por tiempo de inicio, quedarse con la más aguda
    by_start = {}
    for start, end, note in raw_notes:
        cur = by_start.get(start)
        if cur is None or note > cur[1]:
            by_start[start] = (end, note)
    starts_sorted = sorted(by_start.keys())

    melody = []
    for start in starts_sorted:
        end, note = by_start[start]
        if melody and start < melody[-1]['end'] and note <= melody[-1]['note']:
            continue  # nota inferior solapada con la anterior: ignorar
        if melody and start < melody[-1]['end']:
            melody[-1]['end'] = start  # recortar nota anterior por solape
        dur_beats = (end - start) / tpb
        melody.append({'start': start, 'end': end, 'note': note,
                       'duration_beats': dur_beats})

    melody = [n for n in melody if n['duration_beats'] >= min_dur]
    return melody, tpb


def match_motifs_in_melody(melody: list, min_score: float = 0.6) -> list:
    """
    Compara, con una ventana deslizante, los intervalos (deltas en
    semitonos) de la melodía extraída contra la firma de cada entrada
    del catálogo. Score = 75% coincidencia de intervalos exactos +
    25% similitud de proporciones de duración.
    """
    if len(melody) < 2:
        return []

    mel_intervals = [melody[i + 1]['note'] - melody[i]['note']
                     for i in range(len(melody) - 1)]
    mel_durs = [n['duration_beats'] for n in melody]
    mel_dur_ratios = [
        (mel_durs[i + 1] / mel_durs[i]) if mel_durs[i] else 1.0
        for i in range(len(mel_durs) - 1)
    ]

    matches = []
    for entry in TABLE:
        eint = entry['intervals']
        n = len(eint)
        if n == 0 or n > len(mel_intervals):
            continue
        edur = entry['dur_ratios']
        for start in range(len(mel_intervals) - n + 1):
            window = mel_intervals[start:start + n]
            pitch_matches = sum(1 for a, b in zip(window, eint) if a == b)
            pitch_score = pitch_matches / n

            wdur = mel_dur_ratios[start:start + n]
            rdiffs = [abs(a - b) for a, b in zip(wdur, edur)]
            rhythm_score = 1 - min(1.0, (sum(rdiffs) / len(rdiffs)) / 2) if rdiffs else 1.0

            score = 0.75 * pitch_score + 0.25 * rhythm_score
            if score >= min_score:
                matches.append({
                    'entry':       entry,
                    'note_index':  start,
                    'start_beat':  sum(mel_durs[:start]),
                    'score':       round(score, 3),
                })
    matches.sort(key=lambda m: (-m['score'], m['start_beat']))
    return matches


def print_midi_analysis(midi_path: str, matches: list, melody: list,
                        min_score: float, verbose: bool = False) -> None:
    print(f"\n{COL['cyan']}{'═'*70}{COL['reset']}")
    print(f"{COL['bold']}  Análisis melódico: {midi_path}{COL['reset']}")
    print(f"{COL['cyan']}{'═'*70}{COL['reset']}\n")
    print(f"  Notas extraídas: {len(melody)}  ·  Umbral de score: {min_score}\n")

    if not matches:
        print("  Sin coincidencias con el catálogo por encima del umbral.\n")
        return

    print(f"  {COL['gray']}{'motivo':<42} {'t':>2}  {'nota#':>6}  {'compás~':>8}  {'score':>6}{COL['reset']}")
    print(f"  {'─'*72}")
    seen = set()
    for m in matches:
        e = m['entry']
        key = (e['id'], m['note_index']) if not verbose else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        tc = tension_col(e['tension'])
        print(
            f"  {COL['bold']}{e['nombre']:<42}{COL['reset']} "
            f"{tc}{e['tension']:>2}{COL['reset']}  "
            f"{m['note_index']:>6}  {m['start_beat']:>8.1f}  "
            f"{COL['cyan']}{m['score']:>6.2f}{COL['reset']}"
        )
        if verbose:
            print(f"        {COL['gray']}{e['degrees']}  ({e['mode']}, {e['style']}, {e['emocion']}){COL['reset']}")
    print(f"\n  {len(matches)} coincidencia(s)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRADO Y BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    for src, dst in {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                     'ü': 'u', 'ñ': 'n'}.items():
        s = s.lower().replace(src, dst)
    return s


def filter_table(style=None, mode=None, tipo=None, contorno=None,
                 emocion=None, tension_min=None, tension_max=None,
                 buscar=None) -> list:
    result = list(TABLE)
    if style:
        result = [e for e in result if e['style'] == style]
    if mode:
        result = [e for e in result if e['mode'] == mode]
    if tipo:
        result = [e for e in result if e['tipo'] == tipo]
    if contorno:
        result = [e for e in result if e['contorno'] == contorno]
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
            or q in _normalize(e['style'])
            or q in _normalize(e['mode'])
            or q in _normalize(e['contorno'])
        ]
    return result


def parse_intent(description: str) -> dict:
    desc = _normalize(description)
    filters = {}
    for phrase, (tmin, tmax) in INTENT_TO_TENSION.items():
        if _normalize(phrase) in desc:
            filters['tension_min'] = tmin
            filters['tension_max'] = tmax
            break
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
        f"{COL['bold']}{e['degrees']:<32}{COL['reset']} "
        f"{tc}t={e['tension']}{COL['reset']}  "
        f"{COL['cyan']}{e['emocion']:<12}{COL['reset']}  "
        f"{COL['gray']}{e['tipo']:<7} {e['contorno']:<11} "
        f"{e['style']:<13} {e['mode']:<18}{COL['reset']}"
    )
    if verbose:
        print(f"        {COL['gray']}{e['nombre']}{COL['reset']}")
        print(f"        {e['desc']}")


def print_table(entries: list, verbose: bool = False) -> None:
    print(
        f"\n  {COL['gray']}{'#':>3}  {'grados':<32} {'t':>2}  "
        f"{'emoción':<12}  {'tipo':<7} {'contorno':<11} "
        f"{'estilo':<13} {'modo':<18}{COL['reset']}"
    )
    print(f"  {'─'*112}")
    for e in entries:
        print_entry(e, verbose)
    print(f"\n  {len(entries)} motivo(s)/frase(s)\n")


def print_stats() -> None:
    print(f"\n{COL['cyan']}{'═'*60}{COL['reset']}")
    print(f"{COL['bold']}  MOTIF TABLE — Estadísticas del catálogo{COL['reset']}")
    print(f"{COL['cyan']}{'═'*60}{COL['reset']}\n")

    tipos    = Counter(e['tipo']     for e in TABLE)
    contornos = Counter(e['contorno'] for e in TABLE)
    styles   = Counter(e['style']    for e in TABLE)
    modes    = Counter(e['mode']     for e in TABLE)
    emocions = Counter(e['emocion']  for e in TABLE)
    tensions = [e['tension'] for e in TABLE]

    print(f"  Total motivos/frases: {len(TABLE)}")
    print(f"  Tensión media: {sum(tensions)/len(tensions):.1f}  "
          f"min={min(tensions)}  max={max(tensions)}\n")

    print(f"  {COL['bold']}Por tipo:{COL['reset']}")
    for k, v in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por contorno:{COL['reset']}")
    for k, v in sorted(contornos.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>3}")

    print(f"\n  {COL['bold']}Por estilo:{COL['reset']}")
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
        description='Catálogo curado de motivos y frases melódicas con tensión y emoción.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Análisis MIDI
    parser.add_argument('--analyze-midi', type=str, metavar='MIDI_FILE',
                        help='Analizar un MIDI y mostrar los motivos del '
                             'catálogo que contiene su melodía')
    parser.add_argument('--min-score', type=float, default=0.7, metavar='F',
                        help='Score mínimo de coincidencia 0-1 (default: 0.7)')
    parser.add_argument('--channel', type=int, default=None, metavar='N',
                        help='Restringir el análisis a un canal MIDI concreto')
    parser.add_argument('--min-dur', type=float, default=0.0, metavar='BEATS',
                        dest='min_dur',
                        help='Duración mínima en beats para considerar una nota '
                             '(default: 0, sin filtro). Útil para ignorar notas '
                             'de adorno u ornamentos')

    # Entrada / búsqueda
    parser.add_argument('--list',      action='store_true',
                        help='Listar todos los motivos y frases')
    parser.add_argument('--stats',     action='store_true',
                        help='Estadísticas del catálogo')
    parser.add_argument('--custom',    type=str, metavar='GRADOS',
                        help='Motivo personalizado por grados de escala, en vez '
                             'de seleccionar del catálogo. Ej: --custom "1 3 5 8". '
                             'Duración opcional por grado con \':beats\' '
                             '(ej: "1:0.5 3:0.5 5:1 8:2"); sin ella se usa '
                             '--beats por grado. Se interpreta con el modo dado '
                             'por --mode y se resuelve en la tónica de --key '
                             'igual que --id.')
    parser.add_argument('--id',        type=int, metavar='N',
                        help='Seleccionar motivo/frase por id')
    parser.add_argument('--buscar',    type=str, metavar='TEXTO',
                        help='Búsqueda por texto libre')
    parser.add_argument('--intencion', type=str, metavar='DESC',
                        help='Buscar por intención en lenguaje natural')

    # Filtros
    parser.add_argument('--style',    type=str,
                        choices=['diatonic', 'baroque', 'jazz', 'modal', 'romantic',
                                 'impressionist', 'pop', 'flamenco', 'blues', 'folk'],
                        help='Filtrar por estilo')
    parser.add_argument('--mode',     type=str, choices=MODES,
                        help='Filtrar por modo (también usado como modo de '
                             'interpretación para --custom, default: major)')
    parser.add_argument('--tipo',     type=str, choices=['motivo', 'frase', 'escala'],
                        help='Filtrar por tipo')
    parser.add_argument('--contorno', type=str,
                        choices=['ascendente', 'descendente', 'arco', 'ondulante',
                                 'zigzag', 'salto', 'estatico'],
                        help='Filtrar por contorno melódico')
    parser.add_argument('--emocion',     type=str,
                        help='Filtrar por emoción (reposo, alegría, tristeza…)')
    parser.add_argument('--tension-min', type=int, metavar='N', dest='tension_min',
                        help='Tensión mínima (1-10)')
    parser.add_argument('--tension-max', type=int, metavar='N', dest='tension_max',
                        help='Tensión máxima (1-10)')

    # Resolución y exportación
    parser.add_argument('--key',    type=str, default='C',
                        help='Tónica para resolución concreta (default: C)')
    parser.add_argument('--octave', type=int, default=5,
                        help='Octava base de la tónica (default: 5, ~C5=60)')
    parser.add_argument('--reps',   type=int, default=None, metavar='N',
                        help='Número de repeticiones exactas del motivo a exportar '
                             '(sustituye a --bars; no trunca el último ciclo)')
    parser.add_argument('--bars',   type=int, default=4,
                        help='Compases a generar (default: 4)')
    parser.add_argument('--beats',  type=int, default=4,
                        help='Pulsos por compás, y duración por defecto en '
                             '--custom sin duración explícita (default: 4)')
    parser.add_argument('--tempo',  type=float, default=100.0,
                        help='Tempo BPM para exportación MIDI (default: 100)')
    parser.add_argument('--export-json', type=str, metavar='FILE',
                        help='Exportar tabla filtrada o motivo resuelto a JSON')
    parser.add_argument('--export-text', type=str, metavar='FILE',
                        help='Exportar motivo resuelto a texto (nota:beats)')
    parser.add_argument('--export-midi', type=str, metavar='FILE',
                        help='Exportar motivo resuelto a MIDI')
    parser.add_argument('--output', type=str, default='obra',
                        help='Nombre base para salidas (default: obra)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar nombre y descripción de cada entrada')

    args = parser.parse_args()

    # ── Análisis MIDI ────────────────────────────────────────────────────────
    if args.analyze_midi:
        if not os.path.isfile(args.analyze_midi):
            print(f"[ERROR] No se encuentra el archivo: {args.analyze_midi}")
            sys.exit(1)
        if not MIDO_OK:
            print("[ERROR] mido no disponible. Instala con: pip install mido")
            sys.exit(1)

        melody, tpb = extract_melody(args.analyze_midi, channel=args.channel,
                                     min_dur=args.min_dur)
        matches = match_motifs_in_melody(melody, min_score=args.min_score)
        print_midi_analysis(args.analyze_midi, matches, melody,
                            args.min_score, verbose=args.verbose)

        if args.export_json:
            exportable = [{
                'id':          m['entry']['id'],
                'nombre':      m['entry']['nombre'],
                'degrees':     m['entry']['degrees'],
                'score':       m['score'],
                'note_index':  m['note_index'],
                'start_beat':  m['start_beat'],
            } for m in matches]
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

    custom_mode = args.mode or 'major'

    # ── Validación: --id y --custom son mutuamente excluyentes ──────────────
    if args.id is not None and args.custom is not None:
        print("[ERROR] --id y --custom no se pueden usar a la vez")
        sys.exit(1)

    # ── Selección por id o por motivo personalizado ──────────────────────────
    if args.id is not None or args.custom is not None:
        if args.custom is not None:
            try:
                entry = make_custom_entry(args.custom, custom_mode, args.beats)
            except ValueError as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
            print(f"\n  {COL['bold']}{entry['degrees']}{COL['reset']}  "
                  f"{COL['gray']}(custom, {custom_mode}){COL['reset']}")
        else:
            entry = TABLE_BY_ID.get(args.id)
            if not entry:
                print(f"[ERROR] No existe motivo/frase con id={args.id}")
                sys.exit(1)
            print(f"\n  #{entry['id']}  {COL['bold']}{entry['degrees']}{COL['reset']}")
            print(f"  {entry['nombre']}  ·  {entry['tipo']} / {entry['style']} / {entry['mode']}")
            print(f"  Tensión: {entry['tension']}/10  ·  Emoción: {entry['emocion']}  ·  Contorno: {entry['contorno']}")
            print(f"  {entry['desc']}\n")

        notes = resolve_entry(entry, tonic_pc, octave=args.octave,
                              bars=args.bars, beats_per_bar=args.beats,
                              reps=args.reps)
        motif_text = motif_to_text(notes)
        print(f"  → {COL['cyan']}{motif_text}{COL['reset']}\n")

        # Exportaciones
        json_path = args.export_json or f"{args.output}.motif.json"
        txt_path  = args.export_text or f"{args.output}.motif.txt"
        mid_path  = args.export_midi or f"{args.output}.motif.mid"

        out = {
            'meta':        {k: v for k, v in entry.items()
                            if k not in ('pattern', 'intervals', 'dur_ratios')},
            'tonic':       PITCH_NAMES[tonic_pc],
            'octave':      args.octave,
            'bars':        args.bars,
            'beats':       args.beats,
            'reps':        args.reps,
            'tempo':       args.tempo,
            'motif_string': motif_text,
            'notes':       notes,
            'generator':   'motif_table.py v2.0',
        }
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  → JSON:  {json_path}")

        with open(txt_path, 'w') as f:
            f.write(motif_text + '\n')
        print(f"  → Texto: {txt_path}")

        if MIDO_OK:
            motif_to_midi(notes, 480, args.tempo, mid_path)
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
            tipo        = args.tipo,
            contorno    = args.contorno,
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
            exportable = [{k: v for k, v in e.items()
                          if k not in ('pattern', 'intervals', 'dur_ratios')}
                         for e in entries]
            with open(args.export_json, 'w') as f:
                json.dump(exportable, f, indent=2, ensure_ascii=False)
            print(f"  → JSON: {args.export_json}")
        return

    # ── Filtrado general + listado ───────────────────────────────────────────
    entries = filter_table(
        style       = args.style,
        mode        = args.mode,
        tipo        = args.tipo,
        contorno    = args.contorno,
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
        exportable = [{k: v for k, v in e.items()
                      if k not in ('pattern', 'intervals', 'dur_ratios')}
                     for e in data]
        with open(args.export_json, 'w') as f:
            json.dump(exportable, f, indent=2, ensure_ascii=False)
        print(f"  → JSON: {args.export_json}\n")


if __name__ == '__main__':
    main()
