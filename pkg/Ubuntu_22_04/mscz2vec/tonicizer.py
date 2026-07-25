#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          TONICIZER  v1.0                                     ║
║      Motor de modulación tonal — puentes armónicos entre dos tonalidades    ║
║                                                                                ║
║  Construye (y, sobre melodías reales, inserta) el "puente" armónico que      ║
║  lleva de una tonalidad A a una tonalidad B usando una técnica de            ║
║  modulación clásica concreta. Cada técnica se modela con su lógica propia    ║
║  (no son variaciones cosméticas de la misma plantilla).                      ║
║                                                                                ║
║  TÉCNICAS (--technique):                                                     ║
║    pivot        — acorde pivote diatónico común a ambas tonalidades         ║
║    common_tone   — modulación por nota común (mediante cromática)            ║
║    diatonic      — modulación diatónica a tonalidad(es) cercana(s)          ║
║                    (encadena saltos de 5ª/relativa si B no es cercana)      ║
║    chromatic     — acorde de aproximación cromática (fuera de la tonalidad)  ║
║    enharmonic    — 7ª disminuida reinterpretada enarmónicamente              ║
║    direct        — modulación directa/abrupta, sin transición                ║
║    sequential    — secuencia armónica transportada en pasos hasta B          ║
║    dominant      — dominante secundario (V7 de B) como bisagra funcional     ║
║    all           — las ocho técnicas                                         ║
║                                                                                ║
║  MODOS:                                                                       ║
║    demo          — genera un MIDI sintético A → puente → B por técnica       ║
║    modulate      — inserta la modulación dentro de una melodía MIDI real     ║
║    analyze       — detecta puntos de cambio de tonalidad en un MIDI          ║
║    catalog       — imprime el catálogo teórico de las 8 técnicas             ║
║    selftest      — valida las 8 técnicas con ejemplos sintéticos y asserts   ║
║                                                                                ║
║  USO:                                                                         ║
║    python tonicizer.py demo --from-key "C major" --to-key "G major" \\       ║
║                        --technique pivot                                     ║
║    python tonicizer.py demo --from-key "A minor" --to-key "F major" \\       ║
║                        --technique all --out-dir salidas/                    ║
║    python tonicizer.py demo --from-key "C major" --to-key "F# major" \\      ║
║                        --technique enharmonic --verbose                      ║
║    python tonicizer.py modulate melodia.mid --to-key "E major" \\            ║
║                        --technique dominant --at 8                          ║
║    python tonicizer.py analyze obra_modulante.mid --window 2                ║
║    python tonicizer.py catalog                                               ║
║    python tonicizer.py selftest                                              ║
║                                                                                ║
║  OPCIONES COMUNES:                                                            ║
║    --from-key "T modo"   Tonalidad de origen, ej. "D minor", "Ab major"      ║
║    --to-key "T modo"     Tonalidad de destino                                ║
║    --technique T [T…]    Técnica(s) (default: pivot)                        ║
║    --bpb N               Pulsos por compás (default: 4)                      ║
║    --tempo BPM           Tempo (default: 100)                                ║
║    --bars-pre N          Compases estableciendo A (default: 4)               ║
║    --bars-post N         Compases estableciendo B (default: 4)               ║
║    --at BAR              (modulate) compás donde ocurre el cambio            ║
║    --window N            (analyze) tamaño de ventana en compases             ║
║    --out-dir DIR         Carpeta de salida (default: ./tonicizer_out)        ║
║    --seed N              Semilla para desambiguar candidatos (default: 42)   ║
║    --verbose             Muestra el análisis armónico del puente             ║
║                                                                                ║
║  COMO MÓDULO:                                                                 ║
║    from tonicizer import Key, build_modulation, modulation_to_midi           ║
║    events, meta = build_modulation(Key(0,'major'), Key(7,'major'), 'pivot')  ║
║                                                                                ║
║  DEPENDENCIAS: mido, numpy   (sin dependencias externas de teoría musical)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import random
import argparse
import traceback
from pathlib import Path
from collections import namedtuple, defaultdict, deque

import numpy as np
import mido

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE TEORÍA MUSICAL
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Orden del círculo de quintas empezando en C, subiendo de 5ª en 5ª.
# Se usa tanto para decidir la grafía (sostenidos/bemoles) como para medir
# "distancia tonal" entre dos tonalidades.
FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
FIFTHS_POS = {pc: i for i, pc in enumerate(FIFTHS_ORDER)}
# Tonalidades mayores desde Cb(6 flats-side) .. hasta C#(7 sharps): usamos la
# posición en FIFTHS_ORDER para decidir si el nombre "natural" de esa tónica
# se escribe con sostenidos o bemoles.
SHARP_SIDE_POS = set(range(0, 7))   # C, G, D, A, E, B, F#

# Intervalos (semitonos desde la tónica) de las escalas usadas para validar
# que un grupo de notas "pertenece" a una tonalidad. Para modo menor se
# incluye tanto la 7ª natural (b7) como la sensible elevada (menor armónica),
# porque ambas se usan libremente en la práctica compositiva.
SCALE_ALLOWED_PCS = {
    'major': {0, 2, 4, 5, 7, 9, 11},
    'minor': {0, 2, 3, 5, 7, 8, 9, 10, 11},   # incluye vi natural/armónico y VII/vii°
}

# Tríadas diatónicas: grado -> (semitonos desde la tónica, calidad)
MAJOR_TRIADS = {
    'I':    (0, 'M'),
    'ii':   (2, 'm'),
    'iii':  (4, 'm'),
    'IV':   (5, 'M'),
    'V':    (7, 'M'),
    'vi':   (9, 'm'),
    'vii°': (11, 'd'),
}
MINOR_TRIADS = {
    'i':    (0, 'm'),
    'ii°':  (2, 'd'),
    'III':  (3, 'M'),
    'iv':   (5, 'm'),
    'v':    (7, 'm'),    # dominante natural (modal)
    'V':    (7, 'M'),    # dominante armónica (funcional, sensible elevada)
    'VI':   (8, 'M'),
    'VII':  (10, 'M'),   # subtónica (menor natural)
    'vii°': (11, 'd'),   # sensible disminuida (menor armónica)
}

# Intervalos de cada calidad de acorde (desde la raíz)
CHORD_INTERVALS = {
    'M':   [0, 4, 7],
    'm':   [0, 3, 7],
    'd':   [0, 3, 6],
    'A':   [0, 4, 8],
    'Mm7': [0, 4, 7, 10],   # 7ª de dominante
    'M7':  [0, 4, 7, 11],
    'm7':  [0, 3, 7, 10],
    'd7':  [0, 3, 6, 9],    # 7ª disminuida (simétrica)
    'hd7': [0, 3, 6, 10],   # semidisminuida
}

# Perfiles de Krumhansl-Schmuckler (para el modo `analyze`)
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

ALL_TECHNIQUES = [
    'pivot', 'common_tone', 'diatonic', 'chromatic',
    'enharmonic', 'direct', 'sequential', 'dominant',
]

TECHNIQUE_DESCRIPTIONS = {
    'pivot':       'Acorde diatónico común a ambas tonalidades usado como bisagra',
    'common_tone': 'Una nota sostenida enlaza dos acordes sin relación diatónica (mediante cromática)',
    'diatonic':    'Modulación a tonalidad(es) cercana(s) en el círculo de quintas, sin alterar notas',
    'chromatic':   'Acorde de aproximación alterado cromáticamente, ajeno a la tonalidad de origen',
    'enharmonic':  '7ª disminuida (simétrica) reinterpretada enarmónicamente para resolver en B',
    'direct':      'Salto abrupto sin transición armónica alguna',
    'sequential':  'Célula armónica repetida y transportada en pasos iguales hasta B',
    'dominant':    'V7 de la tonalidad de destino insertado como dominante secundario',
}

Chord = namedtuple('Chord', ['start_beat', 'dur_beats', 'root_pc', 'quality', 'label', 'section'])


# ══════════════════════════════════════════════════════════════════════════════
#  TONALIDAD
# ══════════════════════════════════════════════════════════════════════════════

class Key:
    """Tonalidad mínima: clase de altura de la tónica + modo ('major'|'minor')."""

    __slots__ = ('tonic_pc', 'mode')

    def __init__(self, tonic_pc, mode='major'):
        self.tonic_pc = tonic_pc % 12
        self.mode = mode if mode in ('major', 'minor') else 'major'

    def spelling(self):
        """Elige sostenidos o bemoles según el lado del círculo de quintas."""
        pos = FIFTHS_POS[self.tonic_pc]
        # El modo menor se referencia a su relativo mayor para decidir grafía
        ref_pos = FIFTHS_POS[(self.tonic_pc + 3) % 12] if self.mode == 'minor' else pos
        names = NOTE_NAMES_SHARP if ref_pos in SHARP_SIDE_POS else NOTE_NAMES_FLAT
        return names[self.tonic_pc]

    def scale_pcs(self):
        return {(self.tonic_pc + iv) % 12 for iv in SCALE_ALLOWED_PCS[self.mode]}

    def __eq__(self, other):
        return isinstance(other, Key) and self.tonic_pc == other.tonic_pc and self.mode == other.mode

    def __hash__(self):
        return hash((self.tonic_pc, self.mode))

    def __repr__(self):
        return f"{self.spelling()} {self.mode}"

    __str__ = __repr__


def parse_key(key_str):
    """Parsea 'D minor', 'F# major', 'Bb major'... -> Key."""
    parts = key_str.strip().replace('♭', 'b').replace('♯', '#').split()
    if not parts:
        raise ValueError(f"Tonalidad vacía: {key_str!r}")
    tonic_raw = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    mode = 'minor' if mode.startswith('min') else 'major'

    letter = tonic_raw[0].upper()
    accidental = tonic_raw[1:]
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}.get(letter)
    if base is None:
        raise ValueError(f"Tónica no reconocida: {tonic_raw!r}")
    for ch in accidental:
        if ch in ('#', '♯'):
            base += 1
        elif ch in ('b', '♭'):
            base -= 1
    return Key(base % 12, mode)


def fifths_distance(key_a, key_b):
    """Distancia mínima (en pasos de 5ª) entre las tónicas en el círculo de quintas."""
    pa, pb = FIFTHS_POS[key_a.tonic_pc], FIFTHS_POS[key_b.tonic_pc]
    d = abs(pa - pb)
    return min(d, 12 - d)


def is_relative(key_a, key_b):
    if key_a.mode == key_b.mode:
        return False
    maj, mnr = (key_a, key_b) if key_a.mode == 'major' else (key_b, key_a)
    return mnr.tonic_pc == (maj.tonic_pc - 3) % 12


def closely_related(key_a, key_b):
    """Tonalidades cercanas en el sentido clásico: relativa, o a un paso de 5ª."""
    if key_a == key_b:
        return True
    return is_relative(key_a, key_b) or fifths_distance(key_a, key_b) <= 1


def key_neighbors(key):
    """Vecinos 'diatónicamente alcanzables' de una tonalidad: dominante,
    subdominante y relativa. Es el grafo que usa la técnica `diatonic` para
    encadenar saltos cuando el destino no es cercano."""
    dom = Key((key.tonic_pc + 7) % 12, key.mode)
    sub = Key((key.tonic_pc + 5) % 12, key.mode)
    if key.mode == 'major':
        rel = Key((key.tonic_pc - 3) % 12, 'minor')
    else:
        rel = Key((key.tonic_pc + 3) % 12, 'major')
    return [dom, sub, rel]


def shortest_diatonic_path(key_a, key_b, max_depth=4):
    """BFS sobre el grafo de tonalidades cercanas. Devuelve lista de Key
    (incluye extremos) o None si no se encuentra dentro de max_depth."""
    if key_a == key_b:
        return [key_a]
    visited = {key_a}
    parent = {}
    q = deque([key_a])
    depth = {key_a: 0}
    while q:
        cur = q.popleft()
        if depth[cur] >= max_depth:
            continue
        for nb in key_neighbors(cur):
            if nb in visited:
                continue
            visited.add(nb)
            parent[nb] = cur
            depth[nb] = depth[cur] + 1
            if nb == key_b:
                path = [nb]
                while path[-1] != key_a:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            q.append(nb)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ACORDES DIATÓNICOS Y PIVOTES
# ══════════════════════════════════════════════════════════════════════════════

def diatonic_triads(key):
    """dict label -> (root_pc_absoluto, calidad) para la tonalidad dada."""
    table = MAJOR_TRIADS if key.mode == 'major' else MINOR_TRIADS
    return {label: ((key.tonic_pc + iv) % 12, qual) for label, (iv, qual) in table.items()}


def chord_pcs(root_pc, quality):
    return {(root_pc + iv) % 12 for iv in CHORD_INTERVALS[quality]}


def find_pivot(key_a, key_b, rng=None):
    """Busca una tríada (mayor o menor, no disminuida) diatónica a ambas
    tonalidades. Devuelve dict con label_a, label_b, root_pc, quality, exact.
    Si no hay coincidencia exacta, devuelve el "pivote parcial" con más
    notas en común (exact=False)."""
    triads_a = diatonic_triads(key_a)
    triads_b = diatonic_triads(key_b)

    exact = []
    for la, (ra, qa) in triads_a.items():
        if qa not in ('M', 'm'):
            continue
        for lb, (rb, qb) in triads_b.items():
            if qb not in ('M', 'm'):
                continue
            if ra == rb and qa == qb:
                interest = 0
                if la in ('I', 'i') or lb in ('I', 'i'):
                    interest -= 1
                exact.append({'label_a': la, 'label_b': lb, 'root_pc': ra,
                              'quality': qa, 'exact': True, 'interest': interest})
    if exact:
        exact.sort(key=lambda d: -d['interest'])
        best_score = exact[0]['interest']
        top = [d for d in exact if d['interest'] == best_score]
        rng = rng or random
        return rng.choice(top)

    # Sin pivote exacto: buscar el par de tríadas con más pitch-classes compartidas
    best = None
    for la, (ra, qa) in triads_a.items():
        pcs_a = chord_pcs(ra, qa)
        for lb, (rb, qb) in triads_b.items():
            pcs_b = chord_pcs(rb, qb)
            shared = len(pcs_a & pcs_b)
            if best is None or shared > best['shared']:
                best = {'label_a': la, 'label_b': lb, 'root_pc': ra,
                        'quality': qa, 'exact': False, 'shared': shared}
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE ACORDES / VOICE-LEADING
# ══════════════════════════════════════════════════════════════════════════════

def build_chord_pitches(root_pc, quality, prev_pitches=None, lo=48, hi=72):
    """Devuelve pitches MIDI de un acorde, con voice-leading mínimo si se
    pasa el acorde anterior."""
    ints = CHORD_INTERVALS.get(quality, [0, 4, 7])
    if prev_pitches:
        candidates = []
        for octv in range(2, 8):
            for iv in ints:
                m = root_pc + iv + octv * 12
                if lo - 6 <= m <= hi + 6:
                    candidates.append(m)
        result, used = [], set()
        for pp in sorted(prev_pitches):
            best = min((c for c in candidates if c not in used),
                       key=lambda c: abs(c - pp), default=None)
            if best is None:
                continue
            result.append(best)
            used.add(best)
        # Asegura que estén representadas todas las notas del acorde (fundamental incluida)
        needed_pcs = {(root_pc + iv) % 12 for iv in ints}
        present_pcs = {p % 12 for p in result}
        for pc in needed_pcs - present_pcs:
            cand = min((c for c in candidates if c % 12 == pc and c not in used),
                       key=lambda c: abs(c - (sum(prev_pitches) / len(prev_pitches))),
                       default=None)
            if cand is not None:
                result.append(cand)
                used.add(cand)
        return sorted(set(result))

    base = root_pc + 48
    while base < lo:
        base += 12
    while base > lo + 12:
        base -= 12
    return sorted(base + iv for iv in ints)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTABLECIMIENTO DE TONALIDAD (secciones "pre" y "post")
# ══════════════════════════════════════════════════════════════════════════════

def establishing_progression(key, function='pre'):
    """Progresión cadencial corta que fija auditivamente una tonalidad."""
    if key.mode == 'major':
        labels = ['I', 'vi', 'IV', 'V'] if function == 'pre' else ['IV', 'V', 'I', 'I']
    else:
        labels = ['i', 'VI', 'iv', 'V'] if function == 'pre' else ['iv', 'V', 'i', 'i']
    triads = diatonic_triads(key)
    return [(triads[l][0], triads[l][1], f"{l}_{key}") for l in labels]


def tonic_triad(key):
    triads = diatonic_triads(key)
    label = 'I' if key.mode == 'major' else 'i'
    root, qual = triads[label]
    return root, qual, label


# ══════════════════════════════════════════════════════════════════════════════
#  LAS OCHO TÉCNICAS DE MODULACIÓN
#  Cada función devuelve (bridge_chords, meta) donde bridge_chords es una
#  lista de (root_pc, quality, label) de UN compás cada uno.
# ══════════════════════════════════════════════════════════════════════════════

def _technique_pivot(key_a, key_b, rng):
    piv = find_pivot(key_a, key_b, rng)
    tag = "pivote" if piv['exact'] else "pivote aproximado"
    bridge = [
        (piv['root_pc'], piv['quality'], f"{piv['label_a']}_A={piv['label_b']}_B ({tag})"),
        ((key_b.tonic_pc + 7) % 12, 'Mm7', f"V7_{key_b}"),
    ]
    meta = {'pivot': piv}
    return bridge, meta


def _technique_common_tone(key_a, key_b, rng):
    ra, qa, la = tonic_triad(key_a)
    rb, qb, lb = tonic_triad(key_b)
    shared = chord_pcs(ra, qa) & chord_pcs(rb, qb)
    tag = f"nota común: pc {sorted(shared)}" if shared else "sin nota común exacta (aproximado)"
    bridge = [(rb, qb, f"{lb}_B (vía {tag})")]
    meta = {'shared_pcs': sorted(shared)}
    return bridge, meta


def _technique_diatonic(key_a, key_b, rng):
    if closely_related(key_a, key_b):
        bridge, meta = _technique_pivot(key_a, key_b, rng)
        meta['chain'] = [key_a, key_b]
        return bridge, meta

    path = shortest_diatonic_path(key_a, key_b, max_depth=4)
    if not path or len(path) < 2:
        bridge, meta = _technique_pivot(key_a, key_b, rng)
        meta['chain'] = [key_a, key_b]
        meta['warning'] = 'no se encontró cadena diatónica corta; se usó pivote directo'
        return bridge, meta

    bridge = []
    for i in range(len(path) - 1):
        piv = find_pivot(path[i], path[i + 1], rng)
        tag = "pivote" if piv['exact'] else "pivote aprox."
        bridge.append((piv['root_pc'], piv['quality'],
                       f"{piv['label_a']}_{path[i]}={piv['label_b']}_{path[i+1]} ({tag})"))
        if i < len(path) - 2:
            r2, q2, l2 = tonic_triad(path[i + 1])
            bridge.append((r2, q2, f"{l2}_{path[i+1]} (tonalidad de paso)"))
    bridge.append(((key_b.tonic_pc + 7) % 12, 'Mm7', f"V7_{key_b}"))
    return bridge, {'chain': path}


def _technique_chromatic(key_a, key_b, rng):
    # Acorde de aproximación cromática que planea por semitono hasta V7_B.
    # Se prueban candidatos alrededor del V7 de B (semitono arriba, abajo,
    # 3ª menor arriba...) hasta encontrar uno que sea genuinamente ajeno
    # (no diatónico) a la tonalidad de origen A.
    v7_root = (key_b.tonic_pc + 7) % 12
    candidate_offsets = [1, -1, 3, -3, 6]
    approach_root = None
    for off in candidate_offsets:
        cand = (v7_root + off) % 12
        if cand not in key_a.scale_pcs():
            approach_root = cand
            break
    if approach_root is None:
        # Tonalidad de origen inusualmente "cromática" (no debería pasar con
        # escalas diatónicas de 7 notas): usar semitono superior igualmente.
        approach_root = (v7_root + 1) % 12

    bridge = [
        (approach_root, 'Mm7', f"acorde cromático de aproximación (ajeno a {key_a})"),
        (v7_root, 'Mm7', f"V7_{key_b}"),
    ]
    in_key_a = approach_root in key_a.scale_pcs()
    meta = {'approach_root_pc': approach_root, 'diatonic_to_A': in_key_a}
    return bridge, meta


def _technique_enharmonic(key_a, key_b, rng):
    # 7ma disminuida cuyo root está un semitono por debajo de la tónica de B:
    # resuelve "hacia arriba" a I_B, pero por ser simétrica también podría
    # reinterpretarse como vii°7 de otras tres tonalidades (a distancia de 3ª m).
    dim_root = (key_b.tonic_pc - 1) % 12
    other_roots = sorted({(dim_root + s) % 12 for s in (0, 3, 6, 9)})
    rb, qb, lb = tonic_triad(key_b)
    bridge = [
        (dim_root, 'd7', f"vii°7 (reinterpretado enarmónicamente hacia {key_b})"),
        (rb, qb, f"{lb}_B"),
    ]
    meta = {'dim7_root_pc': dim_root, 'dim7_alt_roots_pc': other_roots}
    return bridge, meta


def _technique_direct(key_a, key_b, rng):
    # Sin acorde puente: el "corte" ocurre entre el final de A y el inicio de B.
    return [], {'abrupt': True}


def _technique_sequential(key_a, key_b, rng):
    pa, pb = key_a.tonic_pc, key_b.tonic_pc
    d = ((pb - pa + 6) % 12) - 6           # distancia con signo en [-6, 6]
    n_steps = 3 if abs(d) >= 2 else (1 if d != 0 else 1)
    base_step = d // n_steps
    remainder = d - base_step * n_steps
    steps = [base_step] * n_steps
    if n_steps:
        steps[-1] += remainder
    bridge = []
    cum = 0
    mode_for_step = key_a.mode
    for i, s in enumerate(steps):
        cum += s
        local_pc = (pa + cum) % 12
        mode_for_step = key_b.mode if i == len(steps) - 1 else key_a.mode
        qual = 'M' if mode_for_step == 'major' else 'm'
        bridge.append(((local_pc + 7) % 12, 'Mm7', f"V7 (paso seq. {i+1})"))
        bridge.append((local_pc, qual, f"I paso {i+1} (célula transportada {cum:+d}st)"))
    meta = {'steps_semitones': steps, 'n_steps': n_steps, 'signed_distance': d}
    return bridge, meta


def _technique_dominant(key_a, key_b, rng):
    bridge = [((key_b.tonic_pc + 7) % 12, 'Mm7', f"V7_{key_b} (dominante secundario)")]
    return bridge, {}


TECHNIQUE_FUNCS = {
    'pivot': _technique_pivot,
    'common_tone': _technique_common_tone,
    'diatonic': _technique_diatonic,
    'chromatic': _technique_chromatic,
    'enharmonic': _technique_enharmonic,
    'direct': _technique_direct,
    'sequential': _technique_sequential,
    'dominant': _technique_dominant,
}


# ══════════════════════════════════════════════════════════════════════════════
#  ENSAMBLAJE: pre + puente + post -> lista de Chord con tiempos absolutos
# ══════════════════════════════════════════════════════════════════════════════

def build_modulation(key_a, key_b, technique, bpb=4, bars_pre=4, bars_post=4, seed=42):
    """API pública. Devuelve (events: list[Chord], meta: dict)."""
    if technique not in TECHNIQUE_FUNCS:
        raise ValueError(f"Técnica desconocida: {technique!r}. Opciones: {ALL_TECHNIQUES}")
    rng = random.Random(seed)

    pre_prog = establishing_progression(key_a, 'pre')
    post_prog = establishing_progression(key_b, 'post')

    # Repetir/recortar pre y post para ajustarse a bars_pre / bars_post
    def resize(prog, n_bars):
        out = []
        i = 0
        while len(out) < n_bars:
            out.append(prog[i % len(prog)])
            i += 1
        return out[:n_bars]

    pre_prog = resize(pre_prog, max(1, bars_pre))
    post_prog = resize(post_prog, max(1, bars_post))

    bridge_raw, meta = TECHNIQUE_FUNCS[technique](key_a, key_b, rng)

    events = []
    beat = 0.0
    for root, qual, label in pre_prog:
        events.append(Chord(beat, bpb, root, qual, label, 'pre'))
        beat += bpb
    for root, qual, label in bridge_raw:
        events.append(Chord(beat, bpb, root, qual, label, 'bridge'))
        beat += bpb
    for root, qual, label in post_prog:
        events.append(Chord(beat, bpb, root, qual, label, 'post'))
        beat += bpb

    meta.update({
        'technique': technique,
        'key_a': key_a, 'key_b': key_b,
        'total_beats': beat, 'bpb': bpb,
        'n_bridge_chords': len(bridge_raw),
    })
    return events, meta


# ══════════════════════════════════════════════════════════════════════════════
#  REALIZACIÓN A NOTAS / ESCRITURA MIDI (mido puro, sin dependencias extra)
# ══════════════════════════════════════════════════════════════════════════════

def realize_events(events, lo=48, hi=72):
    """events -> (chord_notes, bass_notes, melody_notes) cada uno lista de
    (offset_beats, pitch_midi, dur_beats, velocidad)."""
    chord_notes, bass_notes, melody_notes = [], [], []
    prev_pitches = None
    for ev in events:
        pitches = build_chord_pitches(ev.root_pc, ev.quality, prev_pitches, lo, hi)
        prev_pitches = pitches
        dur = ev.dur_beats * 0.92
        vel_chord = 62 if ev.section != 'bridge' else 70
        for p in pitches:
            chord_notes.append((ev.start_beat, p, dur, vel_chord))
        bass_pitch = ev.root_pc + 36
        while bass_pitch > 52:
            bass_pitch -= 12
        while bass_pitch < 28:
            bass_pitch += 12
        bass_notes.append((ev.start_beat, bass_pitch, dur * 0.95, 78))
        top = max(pitches) + 12
        if top > 96:
            top -= 12
        melody_notes.append((ev.start_beat, top, dur, 85 if ev.section == 'bridge' else 72))
    return chord_notes, bass_notes, melody_notes


def write_midi(chord_notes, bass_notes, melody_notes, tempo_bpm, bpb, output_path):
    tpb = 480
    tempo_us = int(60_000_000 / max(1.0, tempo_bpm))
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    header = mido.MidiTrack()
    header.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    header.append(mido.MetaMessage('time_signature', numerator=bpb, denominator=4,
                                   clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    header.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(header)

    def to_ticks(beats):
        return max(1, int(round(beats * tpb)))

    def track_from_notes(notes, ch, program, name):
        events = []
        for offset, pitch, dur, vel in notes:
            p = max(0, min(127, int(round(pitch))))
            v = max(1, min(127, int(vel)))
            t_on = to_ticks(offset)
            t_off = to_ticks(offset + max(0.05, dur))
            events.append((t_on, 1, p, v))
            events.append((t_off, 0, p, 0))
        events.sort(key=lambda x: (x[0], x[1]))  # note_off antes que note_on en el mismo tick
        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=name, time=0))
        trk.append(mido.Message('program_change', channel=ch, program=program, time=0))
        prev_t = 0
        for abs_t, kind, p, v in events:
            dt = max(0, abs_t - prev_t)
            msg_type = 'note_on' if kind == 1 else 'note_off'
            trk.append(mido.Message(msg_type, channel=ch, note=p, velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    mid.tracks.append(track_from_notes(melody_notes, ch=0, program=0, name='Melody (guide)'))
    mid.tracks.append(track_from_notes(chord_notes, ch=1, program=0, name='Chords'))
    mid.tracks.append(track_from_notes(bass_notes, ch=2, program=32, name='Bass'))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mid.save(output_path)


def modulation_to_midi(key_a, key_b, technique, output_path, bpb=4, bars_pre=4,
                       bars_post=4, tempo_bpm=100, seed=42):
    events, meta = build_modulation(key_a, key_b, technique, bpb, bars_pre, bars_post, seed)
    chord_notes, bass_notes, melody_notes = realize_events(events)
    write_midi(chord_notes, bass_notes, melody_notes, tempo_bpm, bpb, output_path)
    return events, meta


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS: detección de tonalidad por ventana (Krumhansl-Schmuckler)
# ══════════════════════════════════════════════════════════════════════════════

def load_notes(midi_path):
    """Carga todas las notas de un MIDI (todas las pistas fusionadas).
    Devuelve list of (offset_beats, pitch, dur_beats, vel), tpb, tempo_bpm."""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    notes = []
    for track in mid.tracks:
        abs_t = 0
        pending = {}
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = abs_t
            elif msg.type in ('note_off', 'note_on'):
                k = (msg.channel, msg.note)
                if k in pending:
                    on_t = pending.pop(k)
                    notes.append((on_t / tpb, msg.note, max(0.05, (abs_t - on_t) / tpb), msg.velocity or 64))
    notes.sort(key=lambda n: n[0])
    tempo_bpm = round(60_000_000 / tempo_us, 2)
    return notes, tpb, tempo_bpm


def estimate_key_window(pcs_weighted):
    """pcs_weighted: array(12,) de peso por pitch-class. Devuelve Key estimada
    vía correlación con los perfiles de Krumhansl-Schmuckler."""
    if pcs_weighted.sum() == 0:
        return Key(0, 'major'), 0.0
    best = None
    for tonic in range(12):
        maj = np.roll(_KK_MAJOR, tonic)
        mnr = np.roll(_KK_MINOR, tonic)
        for mode, profile in (('major', maj), ('minor', mnr)):
            corr = np.corrcoef(pcs_weighted, profile)[0, 1]
            if np.isnan(corr):
                corr = -1.0
            if best is None or corr > best[0]:
                best = (corr, tonic, mode)
    corr, tonic, mode = best
    return Key(tonic, mode), float(corr)


def analyze_modulations(midi_path, window_bars=2, bpb=4, verbose=False):
    notes, tpb, tempo_bpm = load_notes(midi_path)
    if not notes:
        raise RuntimeError("El MIDI no contiene notas.")
    total_beats = max(o + d for o, _, d, _ in notes)
    win_beats = window_bars * bpb
    n_windows = max(1, math.ceil(total_beats / win_beats))

    segments = []
    for w in range(n_windows):
        w_start, w_end = w * win_beats, (w + 1) * win_beats
        weights = np.zeros(12)
        for offset, pitch, dur, vel in notes:
            if offset < w_end and offset + dur > w_start:
                weights[pitch % 12] += vel * min(dur, w_end - w_start)
        key_est, corr = estimate_key_window(weights)
        segments.append({'bar_start': w * window_bars, 'key': key_est, 'confidence': round(corr, 3)})
        if verbose:
            print(f"    compás {w*window_bars:>3d}: {key_est}  (r={corr:.3f})")

    # Detectar puntos de cambio (colapsar ventanas consecutivas con misma tonalidad)
    changes = []
    for i in range(1, len(segments)):
        prev_k, cur_k = segments[i - 1]['key'], segments[i]['key']
        if prev_k != cur_k:
            changes.append({
                'at_bar': segments[i]['bar_start'],
                'from_key': prev_k, 'to_key': cur_k,
                'guessed_technique': _guess_technique(prev_k, cur_k),
            })
    return segments, changes, tempo_bpm


def _guess_technique(key_a, key_b):
    """Heurística aproximada: sólo orientativa, basada en la relación entre
    las dos tonalidades detectadas."""
    if key_a == key_b:
        return None
    if is_relative(key_a, key_b):
        return 'common_tone / diatonic (relativas)'
    d = fifths_distance(key_a, key_b)
    if d == 1:
        return 'diatonic / pivot / dominant (5ª cercana)'
    if d == 6:
        return 'chromatic / enharmonic (polo opuesto del círculo)'
    if (key_b.tonic_pc - key_a.tonic_pc) % 12 in (3, 4, 8, 9):
        return 'common_tone (relación de 3ª / mediante cromática)'
    return 'chromatic / direct (relación lejana)'


# ══════════════════════════════════════════════════════════════════════════════
#  MODO `modulate`: insertar la modulación dentro de una melodía real
# ══════════════════════════════════════════════════════════════════════════════

def modulate_midi_file(input_path, to_key, technique, at_bar, output_path,
                       from_key=None, bpb=4, seed=42, verbose=False):
    notes, tpb, tempo_bpm = load_notes(input_path)
    if not notes:
        raise RuntimeError("El MIDI de entrada no contiene notas.")

    if from_key is None:
        win_beats = at_bar * bpb if at_bar > 0 else 8 * bpb
        weights = np.zeros(12)
        for offset, pitch, dur, vel in notes:
            if offset < win_beats:
                weights[pitch % 12] += vel * dur
        from_key, corr = estimate_key_window(weights)
        if verbose:
            print(f"  Tonalidad de origen autodetectada: {from_key} (r={corr:.3f})")

    seam_beat = at_bar * bpb

    before = [(o, p, d, v) for o, p, d, v in notes if o < seam_beat]
    after = [(o, p, d, v) for o, p, d, v in notes if o >= seam_beat]

    # Transposición simple de la continuación al nuevo centro tonal (semitonos,
    # camino más corto en el círculo cromático).
    shift = ((to_key.tonic_pc - from_key.tonic_pc + 6) % 12) - 6
    after_shifted = [(o, p + shift, d, v) for o, p, d, v in after]

    events, meta = build_modulation(from_key, to_key, technique, bpb=bpb,
                                    bars_pre=1, bars_post=1, seed=seed)
    bridge_events = [e for e in events if e.section == 'bridge']
    bridge_offset = seam_beat
    chord_notes, bass_notes, melody_notes = [], [], []
    prev_pitches = None
    beat_cursor = bridge_offset
    for ev in bridge_events:
        pitches = build_chord_pitches(ev.root_pc, ev.quality, prev_pitches)
        prev_pitches = pitches
        dur = ev.dur_beats * 0.92
        for p in pitches:
            chord_notes.append((beat_cursor, p, dur, 66))
        bass_pitch = ev.root_pc + 36
        while bass_pitch > 52:
            bass_pitch -= 12
        while bass_pitch < 28:
            bass_pitch += 12
        bass_notes.append((beat_cursor, bass_pitch, dur * 0.95, 76))
        beat_cursor += ev.dur_beats

    bridge_beats = beat_cursor - bridge_offset
    after_shifted = [(o + bridge_beats, p, d, v) for o, p, d, v in after_shifted]

    melody_notes = before + after_shifted

    write_midi(chord_notes, bass_notes, melody_notes, tempo_bpm, bpb, output_path)

    if verbose:
        print(f"  {from_key} --[{technique}]--> {to_key}, puente en compás {at_bar} "
              f"({len(bridge_events)} acorde(s), {bridge_beats:.1f} tiempos)")
        for ev in bridge_events:
            print(f"    · {ev.label}")

    return {'from_key': from_key, 'to_key': to_key, 'technique': technique,
            'at_bar': at_bar, 'bridge_beats': bridge_beats, 'meta': meta}


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO
# ══════════════════════════════════════════════════════════════════════════════

def print_catalog():
    print("\n╔══ CATÁLOGO DE TÉCNICAS DE MODULACIÓN ══════════════════════════════════╗")
    for t in ALL_TECHNIQUES:
        print(f"  {t:13s} — {TECHNIQUE_DESCRIPTIONS[t]}")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST — ejemplos sintéticos + asserts estructurales por técnica
# ══════════════════════════════════════════════════════════════════════════════

def _assert_subset_scale(events, section, key, label):
    bad = []
    for ev in events:
        if ev.section != section:
            continue
        pcs = chord_pcs(ev.root_pc, ev.quality)
        if not pcs.issubset(key.scale_pcs()):
            bad.append((ev.label, sorted(pcs - key.scale_pcs())))
    assert not bad, f"[{label}] notas fuera de {key} en sección '{section}': {bad}"


def _check_technique(technique, key_a, key_b, out_dir, seed=42, bpb=4):
    label = f"{technique}: {key_a} -> {key_b}"
    events, meta = build_modulation(key_a, key_b, technique, bpb=bpb, bars_pre=2, bars_post=2, seed=seed)
    bridge = [e for e in events if e.section == 'bridge']

    # 1) Estructura general
    assert any(e.section == 'pre' for e in events), f"[{label}] falta sección 'pre'"
    assert any(e.section == 'post' for e in events), f"[{label}] falta sección 'post'"

    # 2) pre/post deben caer dentro de la escala de su tonalidad
    _assert_subset_scale(events, 'pre', key_a, label)
    _assert_subset_scale(events, 'post', key_b, label)

    # 3) Aserciones específicas de cada técnica
    if technique == 'pivot':
        piv = meta['pivot']
        pcs = chord_pcs(piv['root_pc'], piv['quality'])
        if piv['exact']:
            assert pcs.issubset(key_a.scale_pcs()) and pcs.issubset(key_b.scale_pcs()), \
                f"[{label}] el pivote 'exacto' no es diatónico a ambas tonalidades"
    elif technique == 'common_tone':
        ra, qa, _ = tonic_triad(key_a)
        pcs_a = chord_pcs(ra, qa)
        pcs_bridge = chord_pcs(*bridge[0][2:4]) if False else chord_pcs(bridge[0].root_pc, bridge[0].quality)
        if meta['shared_pcs']:
            assert pcs_a & pcs_bridge, f"[{label}] no hay nota común real entre I_A y el puente"
    elif technique == 'diatonic':
        chain = meta['chain']
        assert chain[0] == key_a and chain[-1] == key_b, f"[{label}] la cadena no conecta A con B"
    elif technique == 'chromatic':
        assert not meta['diatonic_to_A'], f"[{label}] el acorde de aproximación no es realmente cromático"
    elif technique == 'enharmonic':
        dim_pcs = chord_pcs(meta['dim7_root_pc'], 'd7')
        intervals = sorted((p - meta['dim7_root_pc']) % 12 for p in dim_pcs)
        assert intervals == [0, 3, 6, 9], f"[{label}] la 7ª disminuida no es simétrica: {intervals}"
        resolves_to = (meta['dim7_root_pc'] + 1) % 12
        assert resolves_to == key_b.tonic_pc, f"[{label}] la resolución no llega a la tónica de B"
    elif technique == 'direct':
        assert len(bridge) == 0, f"[{label}] 'direct' no debería tener acordes puente"
    elif technique == 'sequential':
        steps = meta['steps_semitones']
        assert sum(steps) == meta['signed_distance'], f"[{label}] los pasos no suman la distancia esperada"
        last_local_pc = (key_a.tonic_pc + sum(steps)) % 12
        assert last_local_pc == key_b.tonic_pc, f"[{label}] la secuencia no llega a la tónica de B"
    elif technique == 'dominant':
        assert bridge[0].quality == 'Mm7', f"[{label}] el puente no es un acorde de dominante (7ª)"
        assert bridge[0].root_pc == (key_b.tonic_pc + 7) % 12, f"[{label}] la raíz no es V de B"

    # 4) Renderizado a MIDI real (verifica que no revienta al escribir)
    out_path = os.path.join(out_dir, f"selftest_{technique}_{key_a}_{key_b}.mid".replace(' ', '_').replace('#', 's').replace('°', ''))
    chord_notes, bass_notes, melody_notes = realize_events(events)
    write_midi(chord_notes, bass_notes, melody_notes, 100, bpb, out_path)
    assert os.path.isfile(out_path), f"[{label}] no se generó el archivo MIDI"

    return out_path


def run_selftest(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    key_pairs = [
        (Key(0, 'major'), Key(7, 'major')),     # C -> G   (cercanas, dominante)
        (Key(0, 'major'), Key(9, 'minor')),     # C -> Am  (relativas)
        (Key(0, 'major'), Key(6, 'major')),     # C -> F#  (tritono, distante)
        (Key(0, 'minor'), Key(3, 'major')),     # Cm -> Eb (relativa mayor)
        (Key(2, 'major'), Key(11, 'minor')),    # D -> Bm
        (Key(0, 'major'), Key(4, 'major')),     # C -> E   (3ª mayor, mediante)
    ]

    total, passed, failed = 0, 0, []
    print(f"\n{'═'*72}\n  TONICIZER — selftest ({len(ALL_TECHNIQUES)} técnicas × {len(key_pairs)} pares de tonalidades)\n{'═'*72}")
    for technique in ALL_TECHNIQUES:
        for key_a, key_b in key_pairs:
            total += 1
            label = f"{technique:12s} {str(key_a):12s} -> {key_b}"
            try:
                path = _check_technique(technique, key_a, key_b, out_dir)
                print(f"  [OK]   {label}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {label}\n         {e}")
                failed.append((technique, key_a, key_b, str(e)))
            except Exception as e:
                print(f"  [ERROR]{label}\n         {type(e).__name__}: {e}")
                failed.append((technique, key_a, key_b, f"{type(e).__name__}: {e}"))
                if os.environ.get('TONICIZER_DEBUG'):
                    traceback.print_exc()

    print(f"{'─'*72}")
    print(f"  Total: {total}   OK: {passed}   FALLOS: {len(failed)}")
    print(f"  MIDI de prueba en: {out_dir}")
    print(f"{'═'*72}\n")
    return passed, failed


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='TONICIZER — motor de modulación tonal entre tonalidades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Técnicas disponibles:
  pivot, common_tone, diatonic, chromatic, enharmonic, direct, sequential, dominant, all

Ejemplos:
  python tonicizer.py demo --from-key "C major" --to-key "G major" --technique pivot
  python tonicizer.py demo --from-key "A minor" --to-key "F major" --technique all
  python tonicizer.py modulate melodia.mid --to-key "E major" --technique dominant --at 8
  python tonicizer.py analyze obra.mid --window 2
  python tonicizer.py catalog
  python tonicizer.py selftest
        """
    )
    sub = p.add_subparsers(dest='mode', required=True)

    d = sub.add_parser('demo', help='Genera un MIDI sintético A -> puente -> B')
    d.add_argument('--from-key', required=True, metavar='KEY')
    d.add_argument('--to-key', required=True, metavar='KEY')
    d.add_argument('--technique', nargs='+', default=['pivot'], metavar='T')
    d.add_argument('--bpb', type=int, default=4)
    d.add_argument('--tempo', type=float, default=100)
    d.add_argument('--bars-pre', type=int, default=4)
    d.add_argument('--bars-post', type=int, default=4)
    d.add_argument('--out-dir', default='tonicizer_out')
    d.add_argument('--seed', type=int, default=42)
    d.add_argument('--verbose', action='store_true')

    m = sub.add_parser('modulate', help='Inserta una modulación dentro de un MIDI real')
    m.add_argument('input', help='Archivo MIDI de entrada')
    m.add_argument('--to-key', required=True, metavar='KEY')
    m.add_argument('--from-key', default=None, metavar='KEY', help='Si se omite, se autodetecta')
    m.add_argument('--technique', default='pivot')
    m.add_argument('--at', type=int, required=True, metavar='BAR', help='Compás donde ocurre el cambio')
    m.add_argument('--bpb', type=int, default=4)
    m.add_argument('--out-dir', default=None)
    m.add_argument('--seed', type=int, default=42)
    m.add_argument('--verbose', action='store_true')

    a = sub.add_parser('analyze', help='Detecta puntos de cambio de tonalidad en un MIDI')
    a.add_argument('input')
    a.add_argument('--window', type=int, default=2, metavar='BARS')
    a.add_argument('--bpb', type=int, default=4)
    a.add_argument('--verbose', action='store_true')

    sub.add_parser('catalog', help='Imprime el catálogo de técnicas')

    st = sub.add_parser('selftest', help='Valida las 8 técnicas con ejemplos sintéticos')
    st.add_argument('--out-dir', default='tonicizer_selftest')

    return p


def cmd_demo(args):
    key_a = parse_key(args.from_key)
    key_b = parse_key(args.to_key)
    techniques = ALL_TECHNIQUES if 'all' in args.technique else args.technique
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'═'*64}\n  TONICIZER — {key_a} → {key_b}\n{'═'*64}")
    for tech in techniques:
        if tech not in TECHNIQUE_FUNCS:
            print(f"  [AVISO] técnica desconocida: {tech}")
            continue
        stem = f"modulation_{tech}_{key_a}_to_{key_b}".replace(' ', '_').replace('#', 's').replace('°', '')
        out_path = os.path.join(args.out_dir, f"{stem}.mid")
        events, meta = modulation_to_midi(key_a, key_b, tech, out_path,
                                          bpb=args.bpb, bars_pre=args.bars_pre,
                                          bars_post=args.bars_post,
                                          tempo_bpm=args.tempo, seed=args.seed)
        bridge = [e for e in events if e.section == 'bridge']
        print(f"\n  ▸ {tech}  ({TECHNIQUE_DESCRIPTIONS[tech]})")
        print(f"    → {out_path}")
        if args.verbose:
            for e in events:
                tag = {'pre': f'  {key_a}', 'bridge': '  PUENTE', 'post': f'  {key_b}'}[e.section]
                print(f"      [{tag}] {e.label}")
        else:
            for e in bridge:
                print(f"      · {e.label}")
    print(f"\n{'═'*64}\n")


def cmd_modulate(args):
    to_key = parse_key(args.to_key)
    from_key = parse_key(args.from_key) if args.from_key else None
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    stem = Path(args.input).stem
    out_path = os.path.join(out_dir, f"{stem}.modulated_{args.technique}.mid")
    info = modulate_midi_file(args.input, to_key, args.technique, args.at, out_path,
                              from_key=from_key, bpb=args.bpb, seed=args.seed, verbose=args.verbose)
    print(f"\n  {info['from_key']} --[{args.technique}]--> {info['to_key']}  (compás {args.at})")
    print(f"  → {out_path}\n")


def cmd_analyze(args):
    segments, changes, tempo_bpm = analyze_modulations(args.input, window_bars=args.window,
                                                        bpb=args.bpb, verbose=args.verbose)
    print(f"\n{'═'*64}\n  TONICIZER — análisis de {args.input}  ({tempo_bpm} BPM)\n{'═'*64}")
    if not changes:
        print("  No se detectaron cambios de tonalidad claros entre ventanas.")
    for c in changes:
        print(f"  compás {c['at_bar']:>3d}: {c['from_key']} → {c['to_key']}"
              f"   (técnica probable: {c['guessed_technique']})")
    print(f"{'═'*64}\n")


def cmd_catalog(args):
    print_catalog()


def cmd_selftest(args):
    passed, failed = run_selftest(args.out_dir)
    sys.exit(0 if not failed else 1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.mode == 'demo':
            cmd_demo(args)
        elif args.mode == 'modulate':
            cmd_modulate(args)
        elif args.mode == 'analyze':
            cmd_analyze(args)
        elif args.mode == 'catalog':
            cmd_catalog(args)
        elif args.mode == 'selftest':
            cmd_selftest(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        if os.environ.get('TONICIZER_DEBUG'):
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
