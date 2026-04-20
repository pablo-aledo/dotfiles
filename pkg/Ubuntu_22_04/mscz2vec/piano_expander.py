#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      PIANO EXPANDER  v1.0                                    ║
║     Expansión orquestal de bocetos pianísticos / reducciones MIDI           ║
║                                                                              ║
║  Lee un MIDI de piano (una o dos pistas) y genera un MIDI multitracks       ║
║  con orquestación completa, idiomática y lista para evaluar en FL Studio.   ║
║                                                                              ║
║  PIPELINE INTERNO:                                                           ║
║  [1] Análisis       — extrae melodía, bajo, armonía, forma, dinámica        ║
║  [2] Textura        — clasifica cada sección en 5 arquetipos orquestales    ║
║  [3] Expansión arm. — voicing orquestal de 4-8 voces desde acordes piano    ║
║  [4] Contramelodías — 1-2 voces secundarias conservadoras (sin disonancias) ║
║  [5] Asignación     — material → instrumentos según arquetipo + plantilla   ║
║  [6] Articulación   — keyswitches, CC1/CC11, ornamentos idiomáticos         ║
║  [7] Salida         — MIDI multitracks + informe + fingerprint opcional      ║
║                                                                              ║
║  ARQUETIPOS DE TEXTURA:                                                      ║
║    A  Tutti homofónico  — tutti ataca junto, dinámica alta                  ║
║    B  Melodía+colchón   — melodía clara + voces largas de acompañamiento    ║
║    C  Melodía+arpegios  — figuración en acomp., melodía por encima          ║
║    D  Contrapuntístico  — líneas independientes, polifonía real             ║
║    E  Sparse/cámara     — pocas notas, mucho espacio, dinámica suave        ║
║                                                                              ║
║  PLANTILLAS (--template):                                                    ║
║    chamber    Cuerdas + flauta + oboe  (6 instr.)                           ║
║    strings    Solo sección de cuerdas  (5 instr.)                           ║
║    full       Orquesta completa        (16 instr.)                          ║
║    custom     JSON con lista de instrumentos (mismo fmt que orchestrator)   ║
║                                                                              ║
║  ESTILOS (--style):                                                          ║
║    auto         Detecta desde el propio piano (default)                     ║
║    romantic      Cuerdas líricas, metales en clímax, maderas melódicas      ║
║    baroque       Contrapunto denso, bajo continuo, ornamentos               ║
║    impressionist Texturas sparse, maderas como color, dinámica suave        ║
║    cinematic     Tutti frecuentes, strings tresillos, metales heroicos      ║
║    chamber       Ensemble pequeño, polifonía real                           ║
║                                                                              ║
║  USO:                                                                        ║
║    python piano_expander.py boceto.mid                                       ║
║    python piano_expander.py boceto.mid --template full --style romantic     ║
║    python piano_expander.py boceto.mid --split G4                           ║
║    python piano_expander.py boceto.mid --no-counter                         ║
║    python piano_expander.py boceto.mid --counter-density 0.3                ║
║    python piano_expander.py boceto.mid --template mi_plantilla.json         ║
║    python piano_expander.py boceto.mid --export-fingerprint                 ║
║    python piano_expander.py boceto.mid --report-only                        ║
║    python piano_expander.py boceto.mid --verbose                            ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --template      chamber | strings | full | <archivo.json>                ║
║    --style         auto | romantic | baroque | impressionist | cinematic |  ║
║                    chamber                                                   ║
║    --split         Nota de split si el piano es una sola pista (def: G4)   ║
║    --no-counter    No generar contramelodías                                 ║
║    --counter-density  Densidad de contramelodías 0.0-1.0 (default: 0.35)   ║
║    --texture-sensitivity  Sensibilidad clasificación textura 0.0-1.0        ║
║    --no-ornaments  No añadir ornamentos ni articulación                     ║
║    --no-cc         No insertar CC1/CC11                                     ║
║    --export-fingerprint  Exportar fingerprint.json para orchestrator.py     ║
║    --export-yaml   Exportar expansión como partitura.yaml                   ║
║    --output        Nombre base de salida (default: <stem>_orquestado)       ║
║    --report-only   Solo análisis, sin generar MIDI                          ║
║    --verbose       Decisiones detalladas por sección                        ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install mido numpy                                                    ║
║    (opcional) pip install PyYAML   ← para --export-yaml                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import argparse
import random
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy")
    sys.exit(1)

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
TICKS       = 480   # se sobreescribe al leer el MIDI
PERC_CH     = 9

def midi_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

def name_to_midi(s: str) -> int:
    flat = {'Db':1,'Eb':3,'Fb':4,'Gb':6,'Ab':8,'Bb':10,'Cb':11}
    s = s.strip()
    u = s[0].upper() + s[1:]
    if len(u) >= 2 and u[1]=='b' and u[:2] in flat:
        return (int(u[2:])+1)*12 + flat[u[:2]]
    if len(u) >= 2 and u[1]=='#':
        nm = {n:i for i,n in enumerate(NOTE_NAMES)}
        return (int(u[2:])+1)*12 + nm.get(u[:2],0)
    nm = {n:i for i,n in enumerate(NOTE_NAMES)}
    return (int(u[1:])+1)*12 + nm.get(u[0],0)


# ══════════════════════════════════════════════════════════════════════════════
#  RANGOS IDIOMÁTICOS  (igual que orchestrator.py para compatibilidad)
# ══════════════════════════════════════════════════════════════════════════════

RANGES = {
    'violin1':    (55, 96),
    'violin2':    (55, 91),
    'viola':      (48, 84),
    'cello':      (36, 76),
    'contrabass': (28, 60),
    'flute':      (60, 96),
    'oboe':       (58, 91),
    'clarinet':   (50, 94),
    'bassoon':    (34, 75),
    'horn':       (34, 77),
    'trumpet':    (52, 82),
    'trombone':   (34, 72),
    'tuba':       (28, 58),
}

SWEET = {
    'violin1':    (64, 88),
    'violin2':    (60, 84),
    'viola':      (55, 79),
    'cello':      (43, 72),
    'contrabass': (33, 52),
    'flute':      (65, 90),
    'oboe':       (62, 86),
    'clarinet':   (55, 86),
    'bassoon':    (38, 67),
    'horn':       (40, 70),
    'trumpet':    (56, 77),
    'trombone':   (40, 67),
}

INSTR_PROGRAM = {
    'violin1': 40, 'violin2': 40, 'viola': 41, 'cello': 42, 'contrabass': 43,
    'flute': 73, 'oboe': 68, 'clarinet': 71, 'bassoon': 70,
    'horn': 60, 'trumpet': 56, 'trombone': 57, 'tuba': 58,
}

INSTR_DISPLAY = {
    'violin1': 'Violin I', 'violin2': 'Violin II', 'viola': 'Viola',
    'cello': 'Cello', 'contrabass': 'Contrabass',
    'flute': 'Flute', 'oboe': 'Oboe', 'clarinet': 'Clarinet', 'bassoon': 'Bassoon',
    'horn': 'Horn', 'trumpet': 'Trumpet', 'trombone': 'Trombone', 'tuba': 'Tuba',
}

FAMILY = {
    'violin1': 'strings', 'violin2': 'strings', 'viola': 'strings',
    'cello': 'strings', 'contrabass': 'strings',
    'flute': 'woodwind', 'oboe': 'woodwind', 'clarinet': 'woodwind', 'bassoon': 'woodwind',
    'horn': 'brass', 'trumpet': 'brass', 'trombone': 'brass', 'tuba': 'brass',
}


# ══════════════════════════════════════════════════════════════════════════════
#  PLANTILLAS ORQUESTALES
# ══════════════════════════════════════════════════════════════════════════════
#  Cada instrumento tiene:
#    role   : melody | countermelody | harmony | bass | pad
#    source : melody | bass | harmony | inner
#    octave : transposición en octavas respecto a la fuente (0 = igual)

TEMPLATES = {
    'chamber': {
        'name': 'Chamber',
        'instruments': [
            {'name': 'violin1',   'role': 'melody',        'source': 'melody',  'octave':  0},
            {'name': 'violin2',   'role': 'countermelody', 'source': 'inner',   'octave':  0},
            {'name': 'viola',     'role': 'harmony',       'source': 'harmony', 'octave':  0},
            {'name': 'cello',     'role': 'bass',          'source': 'bass',    'octave':  0},
            {'name': 'flute',     'role': 'melody',        'source': 'melody',  'octave':  1},
            {'name': 'oboe',      'role': 'countermelody', 'source': 'inner',   'octave':  0},
        ]
    },
    'strings': {
        'name': 'Strings',
        'instruments': [
            {'name': 'violin1',   'role': 'melody',   'source': 'melody',  'octave':  0},
            {'name': 'violin2',   'role': 'harmony',  'source': 'harmony', 'octave':  0},
            {'name': 'viola',     'role': 'inner',    'source': 'inner',   'octave':  0},
            {'name': 'cello',     'role': 'bass',     'source': 'bass',    'octave':  0},
            {'name': 'contrabass','role': 'bass',     'source': 'bass',    'octave': -1},
        ]
    },
    'full': {
        'name': 'Full Orchestra',
        'instruments': [
            {'name': 'violin1',   'role': 'melody',        'source': 'melody',  'octave':  0},
            {'name': 'violin2',   'role': 'harmony',       'source': 'harmony', 'octave':  0},
            {'name': 'viola',     'role': 'inner',         'source': 'inner',   'octave':  0},
            {'name': 'cello',     'role': 'bass',          'source': 'bass',    'octave':  0},
            {'name': 'contrabass','role': 'bass',          'source': 'bass',    'octave': -1},
            {'name': 'flute',     'role': 'melody',        'source': 'melody',  'octave':  1},
            {'name': 'oboe',      'role': 'countermelody', 'source': 'counter', 'octave':  0},
            {'name': 'clarinet',  'role': 'harmony',       'source': 'harmony', 'octave':  0},
            {'name': 'bassoon',   'role': 'bass',          'source': 'bass',    'octave':  0},
            {'name': 'horn',      'role': 'pad',           'source': 'harmony', 'octave': -1},
            {'name': 'trumpet',   'role': 'melody',        'source': 'melody',  'octave':  0},
            {'name': 'trombone',  'role': 'bass',          'source': 'harmony', 'octave': -1},
            {'name': 'tuba',      'role': 'bass',          'source': 'bass',    'octave': -1},
        ]
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUETIPOS Y ESTILOS
# ══════════════════════════════════════════════════════════════════════════════

TEXTURE_ARCHETYPES = {
    'A': 'Tutti homofónico',
    'B': 'Melodía + colchón',
    'C': 'Melodía + arpegios',
    'D': 'Contrapuntístico',
    'E': 'Sparse / cámara',
}

# Para cada estilo: pesos relativos de cada arquetipo (suman 1.0)
# Ajustan la clasificación cuando el análisis está en zona limítrofe
STYLE_ARCHETYPE_BIAS = {
    'romantic':     {'A': 0.20, 'B': 0.35, 'C': 0.25, 'D': 0.10, 'E': 0.10},
    'baroque':      {'A': 0.10, 'B': 0.15, 'C': 0.20, 'D': 0.45, 'E': 0.10},
    'impressionist':{'A': 0.05, 'B': 0.25, 'C': 0.20, 'D': 0.15, 'E': 0.35},
    'cinematic':    {'A': 0.40, 'B': 0.25, 'C': 0.20, 'D': 0.05, 'E': 0.10},
    'chamber':      {'A': 0.10, 'B': 0.20, 'C': 0.20, 'D': 0.40, 'E': 0.10},
    'auto':         {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.20, 'E': 0.20},
}

# Instrumentos activos por arquetipo dentro de cada plantilla
# (cuáles tocan en ese arquetipo, el resto silencia o reduce velocidad)
ARCHETYPE_ACTIVITY = {
    'A': {'strings': 1.0, 'woodwind': 0.9, 'brass': 1.0},
    'B': {'strings': 1.0, 'woodwind': 0.8, 'brass': 0.3},
    'C': {'strings': 1.0, 'woodwind': 0.7, 'brass': 0.2},
    'D': {'strings': 0.8, 'woodwind': 1.0, 'brass': 0.2},
    'E': {'strings': 0.6, 'woodwind': 0.7, 'brass': 0.0},
}

# Escalas de dinámica por arquetipo
ARCHETYPE_VEL_SCALE = {
    'A': 1.10,
    'B': 0.85,
    'C': 0.80,
    'D': 0.75,
    'E': 0.60,
}


# ══════════════════════════════════════════════════════════════════════════════
#  TEORÍA MUSICAL — utilidades armónicas
# ══════════════════════════════════════════════════════════════════════════════

# Intervalos consonantes (sin disonancia) para contramelodías
CONSONANT_INTERVALS = {3, 4, 7, 8, 9, 12, 15, 16}   # 3ª m, 3ª M, 5ª, 6ª m, 6ª M, 8ª, 10ª m, 10ª M

def pc(n: int) -> int:
    """Clase de altura (pitch class) 0-11."""
    return n % 12

def interval_class(a: int, b: int) -> int:
    return min((b-a) % 12, (a-b) % 12)

def is_consonant(a: int, b: int) -> bool:
    diff = abs(a - b) % 12
    return diff in {0, 3, 4, 7, 8, 9}

def chord_tones_from_notes(notes: List[dict]) -> List[int]:
    """Extrae las clases de altura únicas de un grupo de notas."""
    return list(set(pc(n['pitch']) for n in notes))

def nearest_chord_tone(pitch: int, chord_pcs: List[int], direction: int = 0) -> int:
    """
    Encuentra la nota más cercana a pitch que pertenezca a chord_pcs.
    direction: 0=cualquiera, 1=arriba, -1=abajo
    """
    candidates = []
    for octave in range(-2, 3):
        for cp in chord_pcs:
            p = cp + (pitch // 12 + octave) * 12
            if 21 <= p <= 108:
                if direction == 0 or (direction == 1 and p >= pitch) or (direction == -1 and p <= pitch):
                    candidates.append(p)
    if not candidates:
        return pitch
    return min(candidates, key=lambda p: abs(p - pitch))

def detect_key(notes: List[dict]) -> Tuple[int, str]:
    """
    Estima tonalidad por perfil de Krumhansl-Schmuckler simplificado.
    Devuelve (tonic_pc, 'major'|'minor').
    """
    profile_major = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    profile_minor = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

    counts = [0.0] * 12
    for n in notes:
        counts[pc(n['pitch'])] += n['duration']
    total = sum(counts) or 1
    counts = [c/total for c in counts]

    best_key, best_mode, best_score = 0, 'major', -999
    for tonic in range(12):
        for mode, profile in [('major', profile_major), ('minor', profile_minor)]:
            rotated = [profile[(i - tonic) % 12] for i in range(12)]
            score = float(np.corrcoef(counts, rotated)[0, 1])
            if score > best_score:
                best_score, best_key, best_mode = score, tonic, mode

    return best_key, best_mode

def scale_pcs(tonic: int, mode: str) -> List[int]:
    """Clases de altura de la escala."""
    intervals = {
        'major': [0,2,4,5,7,9,11],
        'minor': [0,2,3,5,7,8,10],
    }
    return [(tonic + i) % 12 for i in intervals.get(mode, intervals['major'])]

def snap_to_scale(pitch: int, scale: List[int]) -> int:
    """Mueve la nota al grado de escala más cercano."""
    p = pc(pitch)
    if p in scale:
        return pitch
    # buscar el más cercano hacia arriba o abajo
    for delta in range(1, 7):
        if (p + delta) % 12 in scale:
            return pitch + delta
        if (p - delta) % 12 in scale:
            return pitch - delta
    return pitch

def infer_chord(notes: List[dict]) -> List[int]:
    """
    Infiere los tonos del acorde a partir de notas simultáneas.
    Devuelve lista de pitch classes.
    """
    if not notes:
        return [0, 4, 7]
    pcs = list(set(pc(n['pitch']) for n in notes))
    return pcs if pcs else [0, 4, 7]


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 1 — LECTURA Y ANÁLISIS DEL PIANO
# ══════════════════════════════════════════════════════════════════════════════

def load_midi(path: str) -> Tuple[MidiFile, int, int]:
    mid = MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    return mid, tpb, tempo

def extract_notes(mid: MidiFile) -> Dict[str, List[dict]]:
    """Extrae notas por pista, tick absoluto."""
    result = {}
    for i, track in enumerate(mid.tracks):
        name = track.name.strip().rstrip('\x00') or f"Track_{i}"
        on = {}
        notes = []
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != PERC_CH:
                on[(msg.channel, msg.note)] = (t, msg.velocity)
            elif (msg.type == 'note_off' or
                  (msg.type == 'note_on' and msg.velocity == 0)) and msg.channel != PERC_CH:
                key = (msg.channel, msg.note)
                if key in on:
                    t0, vel = on.pop(key)
                    dur = t - t0
                    if dur > 0:
                        notes.append({'tick': t0, 'pitch': msg.note,
                                      'velocity': vel, 'duration': dur,
                                      'channel': msg.channel})
        if notes:
            result[name] = notes
    return result

def split_piano_hands(notes: List[dict], split: int, tpb: int) -> Tuple[List[dict], List[dict]]:
    """Separa pista única en MD/MI por registro y polifonía."""
    SLACK = tpb // 8
    sn = sorted(notes, key=lambda n: (n['tick'], -n['pitch']))
    clusters, i = [], 0
    while i < len(sn):
        g = [sn[i]]
        j = i+1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= SLACK:
            g.append(sn[j])
            j += 1
        clusters.append(g)
        i = j

    rh, lh = [], []
    for c in clusters:
        cs = sorted(c, key=lambda n: -n['pitch'])
        above = [n for n in cs if n['pitch'] >= split]
        below = [n for n in cs if n['pitch'] <  split]
        if above and below:
            rh.extend(above); lh.extend(below)
        elif len(cs) == 1:
            (rh if cs[0]['pitch'] >= split else lh).extend(cs)
        else:
            rh.append(cs[0]); lh.extend(cs[1:])
    return rh, lh

def segment_sections(notes: List[dict], tpb: int,
                     sensitivity: float = 0.5) -> List[dict]:
    """
    Segmenta la obra en secciones por cambios de densidad y dinámica.
    Devuelve lista de {start, end, notes, density, mean_vel, mean_dur}.
    """
    if not notes:
        return []

    bar = tpb * 4
    max_tick = max(n['tick'] + n['duration'] for n in notes)
    n_bars = max(int(math.ceil(max_tick / bar)), 1)

    # Perfil por compás: densidad y dinámica
    bar_density = np.zeros(n_bars)
    bar_vel     = np.zeros(n_bars)
    for n in notes:
        b = min(n['tick'] // bar, n_bars - 1)
        bar_density[b] += 1
        bar_vel[b]     += n['velocity']

    # Normalizar
    bar_density = bar_density / max(bar_density.max(), 1)
    bar_vel_n   = bar_vel / max(bar_vel.max(), 1)

    # Combinar en señal de cambio
    signal = (bar_density + bar_vel_n) / 2.0

    # Detectar fronteras: cambios bruscos
    threshold = 0.20 + (1.0 - sensitivity) * 0.25
    boundaries = [0]
    window = max(2, int(4 * (1 - sensitivity)))
    for i in range(window, n_bars - window):
        left  = np.mean(signal[max(0, i-window):i])
        right = np.mean(signal[i:min(n_bars, i+window)])
        if abs(right - left) > threshold:
            # Evitar secciones demasiado cortas (< 4 compases)
            if i - boundaries[-1] >= 4:
                boundaries.append(i)
    boundaries.append(n_bars)

    # Construir secciones
    sections = []
    for i in range(len(boundaries) - 1):
        b_start = boundaries[i]
        b_end   = boundaries[i+1]
        t_start = b_start * bar
        t_end   = b_end   * bar
        sec_notes = [n for n in notes if t_start <= n['tick'] < t_end]
        if not sec_notes:
            continue
        sections.append({
            'index':    i,
            'bar_start': b_start + 1,
            'bar_end':   b_end,
            'tick_start': t_start,
            'tick_end':   t_end,
            'notes':     sec_notes,
            'density':   np.mean(bar_density[b_start:b_end]),
            'mean_vel':  np.mean([n['velocity'] for n in sec_notes]),
            'mean_dur':  np.mean([n['duration'] for n in sec_notes]),
            'n_notes':   len(sec_notes),
        })
    return sections

def extract_melody(notes: List[dict], tpb: int) -> List[dict]:
    """Extrae la voz superior (nota más aguda por ventana temporal)."""
    if not notes:
        return []
    SLACK = tpb // 8
    sn = sorted(notes, key=lambda n: n['tick'])
    melody, i = [], 0
    while i < len(sn):
        g = [sn[i]]
        j = i+1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= SLACK:
            g.append(sn[j])
            j += 1
        top = max(g, key=lambda n: n['pitch'])
        melody.append(top)
        i = j
    return melody

def extract_bass(notes: List[dict], tpb: int) -> List[dict]:
    """Extrae la voz más grave por ventana temporal."""
    if not notes:
        return []
    SLACK = tpb // 8
    sn = sorted(notes, key=lambda n: n['tick'])
    bass, i = [], 0
    while i < len(sn):
        g = [sn[i]]
        j = i+1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= SLACK:
            g.append(sn[j])
            j += 1
        bot = min(g, key=lambda n: n['pitch'])
        bass.append(bot)
        i = j
    return bass

def extract_harmony_chords(all_notes: List[dict], tpb: int) -> List[dict]:
    """
    Agrupa notas en acordes por ventana temporal.
    Devuelve lista de {tick, duration, pitches, chord_pcs}.
    """
    SLACK = tpb // 4
    sn = sorted(all_notes, key=lambda n: n['tick'])
    chords, i = [], 0
    while i < len(sn):
        g = [sn[i]]
        j = i+1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= SLACK:
            g.append(sn[j])
            j += 1
        dur = max(n['duration'] for n in g)
        pitches = sorted(set(n['pitch'] for n in g))
        chords.append({
            'tick':      sn[i]['tick'],
            'duration':  dur,
            'pitches':   pitches,
            'chord_pcs': infer_chord(g),
            'mean_vel':  np.mean([n['velocity'] for n in g]),
        })
        i = j
    return chords


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 2 — CLASIFICACIÓN DE TEXTURA
# ══════════════════════════════════════════════════════════════════════════════

def classify_texture(section: dict, tpb: int, style: str,
                     sensitivity: float) -> str:
    """
    Clasifica una sección en uno de los 5 arquetipos.
    Retorna 'A'|'B'|'C'|'D'|'E'.
    """
    density   = section['density']
    mean_vel  = section['mean_vel']
    mean_dur  = section['mean_dur']
    n         = section['n_notes']
    bar_span  = max(section['bar_end'] - section['bar_start'], 1)
    notes_per_bar = n / bar_span

    # Movimiento melódico: % de pasos por grado vs. saltos
    notes_sorted = sorted(section['notes'], key=lambda x: x['tick'])
    steps = 0
    for i in range(1, len(notes_sorted)):
        if abs(notes_sorted[i]['pitch'] - notes_sorted[i-1]['pitch']) <= 2:
            steps += 1
    step_ratio = steps / max(len(notes_sorted)-1, 1)

    # Puntuaciones brutas para cada arquetipo
    scores = {
        'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0
    }

    # A: Tutti — densa, fuerte
    scores['A'] += density * 0.4
    scores['A'] += (mean_vel / 127) * 0.4
    scores['A'] += (1 - step_ratio) * 0.2

    # B: Melodía + colchón — densidad media, notas largas, movimiento por grados
    scores['B'] += (1 - density) * 0.3
    scores['B'] += step_ratio * 0.4
    scores['B'] += (mean_dur / (tpb * 2)) * 0.3

    # C: Melodía + arpegios — densidad media-alta, notas cortas en acomp.
    scores['C'] += density * 0.3
    scores['C'] += (1 - mean_dur / (tpb * 2)) * 0.4
    scores['C'] += step_ratio * 0.3

    # D: Contrapuntístico — densidad alta, muchos saltos, velocidad moderada
    scores['D'] += density * 0.3
    scores['D'] += (1 - step_ratio) * 0.4
    scores['D'] += (1 - mean_vel / 127) * 0.3

    # E: Sparse — muy baja densidad, notas muy largas, piano
    scores['E'] += (1 - density) * 0.5
    scores['E'] += (mean_dur / (tpb * 4)) * 0.3
    scores['E'] += (1 - mean_vel / 127) * 0.2

    # Aplicar bias de estilo
    bias = STYLE_ARCHETYPE_BIAS.get(style, STYLE_ARCHETYPE_BIAS['auto'])
    for k in scores:
        scores[k] = scores[k] * (1.0 - sensitivity * 0.3) + bias[k] * sensitivity * 0.3

    return max(scores, key=lambda k: scores[k])


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 3 — EXPANSIÓN ARMÓNICA
# ══════════════════════════════════════════════════════════════════════════════

def expand_chord_to_voices(chord: dict, instr: dict,
                           tonic: int, scale: List[int],
                           archetype: str) -> Optional[dict]:
    """
    Genera una nota para un instrumento dado un acorde del piano.
    Respeta el rango idiomático del instrumento y la consonancia.
    Devuelve nota dict o None si no hay nota idiomática.
    """
    name     = instr['name']
    role     = instr['role']
    source   = instr['source']
    octave   = instr['octave']
    lo, hi   = RANGES.get(name, (36, 96))
    sw_lo, sw_hi = SWEET.get(name, (lo, hi))

    chord_pcs = chord['chord_pcs']
    pitches   = chord['pitches']
    if not pitches:
        return None

    # Selección de pitch base según rol
    if role == 'melody' or source == 'melody':
        base = max(pitches)
    elif role == 'bass' or source == 'bass':
        base = min(pitches)
    elif source in ('harmony', 'inner', 'counter'):
        # Voz interior: elegir nota del acorde en zona media
        mid_idx = len(pitches) // 2
        base = pitches[mid_idx] if len(pitches) > 1 else pitches[0]
    else:
        base = max(pitches)

    # Transponer por octava indicada en la plantilla
    target = base + octave * 12

    # Ajustar al rango idiomático
    while target < lo and target + 12 <= hi:
        target += 12
    while target > hi and target - 12 >= lo:
        target -= 12

    if not (lo <= target <= hi):
        return None

    # Para voces interiores/contramelodía: snap a tono del acorde más cercano
    if source in ('harmony', 'inner', 'counter'):
        target = nearest_chord_tone(target, chord_pcs)
        # Verificar consonancia con melodía (nota más aguda del acorde)
        mel_pitch = max(pitches)
        if not is_consonant(target, mel_pitch):
            # Intentar tono alternativo del acorde
            for alt_pc in chord_pcs:
                for oct_offset in [-1, 0, 1]:
                    alt = alt_pc + (target // 12 + oct_offset) * 12
                    if lo <= alt <= hi and is_consonant(alt, mel_pitch):
                        target = alt
                        break
                else:
                    continue
                break

    # Asegurar que es nota de escala (no cromatismo accidental)
    target = snap_to_scale(target, scale)
    # Snap final a tono del acorde más cercano (mantiene consonancia)
    target = nearest_chord_tone(target, chord_pcs)

    if not (lo <= target <= hi):
        return None

    # Dinámica según actividad del arquetipo
    family      = FAMILY.get(name, 'strings')
    vel_scale   = ARCHETYPE_ACTIVITY.get(archetype, {}).get(family, 0.8)
    vel_scale  *= ARCHETYPE_VEL_SCALE.get(archetype, 0.85)
    velocity    = max(30, min(127, int(chord['mean_vel'] * vel_scale)))

    # Pad y brass en arquetipo E casi silenciosos
    if archetype == 'E' and family == 'brass':
        return None

    return {
        'tick':     chord['tick'],
        'pitch':    target,
        'velocity': velocity,
        'duration': chord['duration'],
        'channel':  0,  # se asignará al construir el MIDI
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 4 — CONTRAMELODÍAS CONSERVADORAS
# ══════════════════════════════════════════════════════════════════════════════

def generate_countermelody(melody: List[dict], chords: List[dict],
                           tonic: int, scale: List[int],
                           density: float, tpb: int,
                           target_range: Tuple[int,int] = (55, 79)) -> List[dict]:
    """
    Genera una contramelodía conservadora:
    - Solo notas del acorde activo en cada momento
    - Preferencia por 3ªs y 6ªs bajo la melodía (intervalos dulces)
    - Movimiento contrario al de la melodía cuando es posible
    - No disonancias: nunca 2ª menor, 7ª mayor o tritono sin preparar

    density: 0.0-1.0, controla qué % de notas de melodía tienen contra.
    """
    if not melody or not chords:
        return []

    lo, hi = target_range
    counter = []

    # Índice de acordes por tick para búsqueda rápida
    chord_list = sorted(chords, key=lambda c: c['tick'])

    def chord_at(tick: int) -> dict:
        """Acorde activo en un tick dado."""
        active = chord_list[0]
        for c in chord_list:
            if c['tick'] <= tick:
                active = c
            else:
                break
        return active

    prev_pitch = None
    prev_mel   = None

    for i, mel_note in enumerate(melody):
        # Decisión de densidad: no generar nota en todos los pulsos
        if random.random() > density:
            prev_mel = mel_note['pitch']
            continue

        chord = chord_at(mel_note['tick'])
        chord_pcs = chord['chord_pcs']
        mel_p = mel_note['pitch']

        # Intervalos preferidos bajo la melodía: 3ª M(4), 3ª m(3), 6ª M(9), 6ª m(8), 5ª(7)
        preferred_intervals = [4, 3, 9, 8, 7]

        candidate = None
        for interval in preferred_intervals:
            p = mel_p - interval
            # Ajustar octava al rango del instrumento
            while p < lo:
                p += 12
            while p > hi:
                p -= 12
            if not (lo <= p <= hi):
                continue
            # Debe ser tono del acorde
            if pc(p) not in chord_pcs:
                p2 = nearest_chord_tone(p, chord_pcs)
                if abs(p2 - p) > 2:
                    continue
                p = p2
            # No cruzar con la melodía (siempre por debajo)
            if p >= mel_p:
                continue
            # Consonante con la melodía
            if not is_consonant(p, mel_p):
                continue
            # Movimiento suave desde la nota anterior (no saltar más de una 6ª)
            if prev_pitch is not None and abs(p - prev_pitch) > 9:
                continue
            # Movimiento contrario a la melodía si es posible
            if prev_pitch is not None and prev_mel is not None:
                mel_dir = mel_p - prev_mel
                cnt_dir = p - prev_pitch
                # Aceptar si es contrario o oblicuo, penalizar pero no prohibir paralelo
                if mel_dir * cnt_dir < 0 or cnt_dir == 0:
                    candidate = p
                    break
                else:
                    candidate = p  # paralelo: aceptar si no hay mejor opción
            else:
                candidate = p
                break

        if candidate is not None:
            velocity = max(30, min(100, int(mel_note['velocity'] * 0.75)))
            counter.append({
                'tick':     mel_note['tick'],
                'pitch':    candidate,
                'velocity': velocity,
                'duration': mel_note['duration'],
                'channel':  0,
            })
            prev_pitch = candidate

        prev_mel = mel_p

    return counter


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 5+6 — ASIGNACIÓN, ARTICULACIÓN Y CC
# ══════════════════════════════════════════════════════════════════════════════

def add_cc_envelope(events: List[Tuple], tick_start: int, tick_end: int,
                    vel_mean: float, archetype: str, tpb: int) -> List[Tuple]:
    """
    Añade CC1 (mod/dynamics xfade) y CC11 (expression) interpolados.
    Devuelve lista de eventos CC adicionales.
    """
    cc_events = []
    n_points = max(4, (tick_end - tick_start) // tpb)
    base_cc1 = int(vel_mean * ARCHETYPE_VEL_SCALE.get(archetype, 0.85))
    base_cc1 = max(30, min(110, base_cc1))

    for i in range(n_points + 1):
        t = tick_start + i * (tick_end - tick_start) // n_points
        # Leve arco dinámico dentro de la sección
        progress = i / n_points
        arc = math.sin(math.pi * progress) * 15  # sube y baja
        cc1_val = max(1, min(127, int(base_cc1 + arc)))
        cc11_val = max(1, min(127, int(cc1_val * 0.9)))
        cc_events.append((t, 'cc', 1,  cc1_val))
        cc_events.append((t, 'cc', 11, cc11_val))

    return cc_events

def apply_articulation(notes: List[dict], archetype: str,
                       tpb: int, instr_name: str) -> List[dict]:
    """
    Ajusta duración de notas según arquetipo e instrumento.
    No añade notas — solo modifica duración y velocidad.
    """
    family = FAMILY.get(instr_name, 'strings')
    result = []
    for n in notes:
        dur = n['duration']
        vel = n['velocity']

        if archetype == 'A':
            # Tutti: notas ligeramente separadas para claridad
            dur = int(dur * 0.92)
        elif archetype == 'C':
            # Arpegios: cuerdas más cortas, maderas más largas
            dur = int(dur * (0.75 if family == 'strings' else 0.95))
        elif archetype == 'D':
            # Contrapunto: legato estricto
            dur = int(dur * 0.98)
        elif archetype == 'E':
            # Sparse: molto legato, diminuendo
            dur = int(dur * 1.02)
            vel = max(20, int(vel * 0.85))

        result.append({**n, 'duration': max(dur, tpb//8), 'velocity': vel})
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MIDI DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def build_track(notes: List[dict], cc_events: List[Tuple],
                channel: int, name: str, program: int, tpb: int) -> MidiTrack:
    """Construye MidiTrack desde lista de notas + CC events."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name=name, time=0))
    track.append(Message('program_change', channel=channel,
                         program=program, time=0))

    events = []
    for n in notes:
        t   = n['tick']
        dur = max(n['duration'], 1)
        events.append((t,       'on',  n['pitch'], n['velocity']))
        events.append((t + dur, 'off', n['pitch'], 0))
    for ev in cc_events:
        t, _, cc_num, cc_val = ev
        events.append((t, 'cc', cc_num, cc_val))

    events.sort(key=lambda e: (e[0], {'off':0,'cc':1,'on':2}.get(e[1],1)))

    prev = 0
    for abs_t, etype, a, b in events:
        delta = abs_t - prev
        prev  = abs_t
        if etype == 'on':
            track.append(Message('note_on',  channel=channel,
                                 note=a, velocity=b, time=delta))
        elif etype == 'off':
            track.append(Message('note_off', channel=channel,
                                 note=a, velocity=0, time=delta))
        elif etype == 'cc':
            track.append(Message('control_change', channel=channel,
                                 control=a, value=b, time=delta))
    return track

def build_output_midi(instr_notes: Dict[str, List[dict]],
                      instr_cc:    Dict[str, List[Tuple]],
                      template: dict, tempo: int, tpb: int,
                      output_path: str) -> List[str]:
    """Genera el MIDI multitracks final."""
    mid = MidiFile(type=1, ticks_per_beat=tpb)

    # Pista de tempo
    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    t0.append(MetaMessage('time_signature', numerator=4, denominator=4,
                           clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    mid.tracks.append(t0)

    created = []
    for ch, instr in enumerate(template['instruments']):
        name = instr['name']
        notes = instr_notes.get(name, [])
        cc    = instr_cc.get(name, [])
        if not notes:
            continue
        prog = INSTR_PROGRAM.get(name, 40)
        disp = INSTR_DISPLAY.get(name, name)
        track = build_track(notes, cc, ch % 16, disp, prog, tpb)
        mid.tracks.append(track)
        created.append(disp)

    mid.save(output_path)
    return created


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def expand(rh: List[dict], lh: List[dict], tpb: int, tempo: int,
           template: dict, style: str, args) -> Tuple[Dict, Dict, dict]:
    """
    Pipeline completo de expansión.
    Devuelve (instr_notes, instr_cc, report_data).
    """
    all_piano = sorted(rh + lh, key=lambda n: n['tick'])
    report = {'sections': [], 'style': style,
              'template': template.get('name','?'), 'steps': []}

    # ── Detectar tonalidad ──────────────────────────────────────────────────
    tonic, mode = detect_key(all_piano)
    scale = scale_pcs(tonic, mode)
    key_name = f"{NOTE_NAMES[tonic]} {mode}"
    print(f"  Tonalidad detectada : {key_name}")
    report['key'] = key_name

    # ── Detectar estilo automático ──────────────────────────────────────────
    if style == 'auto':
        mean_vel = np.mean([n['velocity'] for n in all_piano]) if all_piano else 64
        mean_dur = np.mean([n['duration'] for n in all_piano]) if all_piano else tpb
        density  = len(all_piano) / max(
            (max(n['tick']+n['duration'] for n in all_piano) / (tpb*4)), 1)
        if mean_vel > 80 and density > 8:
            style = 'cinematic'
        elif mean_dur > tpb * 2 and mean_vel < 65:
            style = 'impressionist'
        elif density > 10:
            style = 'romantic'
        else:
            style = 'romantic'
        print(f"  Estilo auto-detectado: {style}")
        report['style'] = style

    # ── Extraer voces ───────────────────────────────────────────────────────
    melody = extract_melody(rh if rh else all_piano, tpb)
    bass   = extract_bass(lh if lh else all_piano, tpb)
    chords = extract_harmony_chords(all_piano, tpb)
    report['steps'].append(f"Melodía: {len(melody)} notas  "
                           f"Bajo: {len(bass)} notas  Acordes: {len(chords)}")

    # ── Segmentar secciones ─────────────────────────────────────────────────
    sections = segment_sections(all_piano, tpb, args.texture_sensitivity)
    print(f"  Secciones detectadas: {len(sections)}")

    # ── Clasificar texturas ─────────────────────────────────────────────────
    for sec in sections:
        arch = classify_texture(sec, tpb, style, args.texture_sensitivity)
        sec['archetype'] = arch
        if args.verbose:
            print(f"    Sección {sec['index']+1:>2} "
                  f"(cc {sec['bar_start']}-{sec['bar_end']}) "
                  f"→ {arch}: {TEXTURE_ARCHETYPES[arch]}  "
                  f"vel={sec['mean_vel']:.0f}  "
                  f"dens={sec['density']:.2f}")
        report['sections'].append({
            'index':     sec['index'],
            'bars':      f"{sec['bar_start']}-{sec['bar_end']}",
            'archetype': arch,
            'label':     TEXTURE_ARCHETYPES[arch],
            'density':   round(sec['density'], 3),
            'mean_vel':  round(sec['mean_vel'], 1),
        })

    # ── Construir mapa de arquetipo por tick ────────────────────────────────
    def archetype_at(tick: int) -> str:
        arch = 'B'
        for sec in sections:
            if sec['tick_start'] <= tick < sec['tick_end']:
                arch = sec['archetype']
                break
        return arch

    def section_at(tick: int) -> Optional[dict]:
        for sec in sections:
            if sec['tick_start'] <= tick < sec['tick_end']:
                return sec
        return sections[-1] if sections else None

    # ── Generar contramelodía ───────────────────────────────────────────────
    counter_notes = []
    if not args.no_counter and melody:
        counter_range = SWEET.get('oboe', (62, 86))
        counter_notes = generate_countermelody(
            melody, chords, tonic, scale,
            density=args.counter_density,
            tpb=tpb,
            target_range=counter_range,
        )
        print(f"  Contramelodía       : {len(counter_notes)} notas")
        report['steps'].append(f"Contramelodía: {len(counter_notes)} notas "
                               f"(densidad={args.counter_density:.2f})")

    # ── Asignar material a instrumentos ────────────────────────────────────
    instr_notes: Dict[str, List[dict]] = defaultdict(list)
    instr_cc:    Dict[str, List[Tuple]] = defaultdict(list)

    # Construir mapa chord por tick para acceso O(1)
    chord_sorted = sorted(chords, key=lambda c: c['tick'])
    def chord_at_tick(tick: int) -> dict:
        active = chord_sorted[0] if chord_sorted else {'tick':0,'duration':tpb,'pitches':[60],'chord_pcs':[0,4,7],'mean_vel':64}
        for c in chord_sorted:
            if c['tick'] <= tick:
                active = c
            else:
                break
        return active

    for chord in chords:
        arch = archetype_at(chord['tick'])
        for instr in template['instruments']:
            note = expand_chord_to_voices(chord, instr, tonic, scale, arch)
            if note:
                if not args.no_ornaments:
                    note = apply_articulation([note], arch, tpb, instr['name'])[0]
                instr_notes[instr['name']].append(note)

    # Asignar contramelodía al instrumento más idiomático disponible
    if counter_notes:
        counter_instr = None
        # Prioridad: oboe > clarinet > violin2 > viola (según plantilla)
        priority = ['oboe', 'clarinet', 'violin2', 'viola', 'flute']
        instr_names_in_template = {i['name'] for i in template['instruments']}
        for candidate in priority:
            if candidate in instr_names_in_template:
                counter_instr = candidate
                break
        if counter_instr:
            lo_c, hi_c = RANGES.get(counter_instr, (36, 96))
            for n in counter_notes:
                p = n['pitch']
                while p < lo_c: p += 12
                while p > hi_c: p -= 12
                if lo_c <= p <= hi_c:
                    instr_notes[counter_instr].append({**n, 'pitch': p})
            report['steps'].append(
                f"Contramelodía → {INSTR_DISPLAY.get(counter_instr, counter_instr)}")

    # ── CC envelopes por sección ────────────────────────────────────────────
    if not args.no_cc:
        for sec in sections:
            arch     = sec['archetype']
            mean_vel = sec['mean_vel']
            for instr in template['instruments']:
                name = instr['name']
                if instr_notes.get(name):
                    cc_ev = add_cc_envelope(
                        [], sec['tick_start'], sec['tick_end'],
                        mean_vel, arch, tpb)
                    instr_cc[name].extend(cc_ev)

    # ── Ordenar notas ───────────────────────────────────────────────────────
    for name in instr_notes:
        instr_notes[name].sort(key=lambda n: (n['tick'], n['pitch']))

    # ── Stats ───────────────────────────────────────────────────────────────
    total_notes = sum(len(v) for v in instr_notes.values())
    report['total_notes_out'] = total_notes
    report['total_notes_in']  = len(all_piano)
    report['expansion_ratio'] = round(total_notes / max(len(all_piano), 1), 2)

    return dict(instr_notes), dict(instr_cc), report


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(report: dict, midi_in: str, midi_out: str,
                    created: List[str], template: dict) -> str:
    W = 70
    lines = ['═'*W, '  PIANO EXPANDER — INFORME DE EXPANSIÓN', '═'*W]
    lines += [
        f"  Fuente        : {midi_in}",
        f"  Salida        : {midi_out}",
        f"  Tonalidad     : {report.get('key','?')}",
        f"  Estilo        : {report.get('style','?')}",
        f"  Plantilla     : {report.get('template','?')}",
        f"  Notas entrada : {report.get('total_notes_in',0)}",
        f"  Notas salida  : {report.get('total_notes_out',0)}",
        f"  Ratio expansión: {report.get('expansion_ratio',0):.1f}x",
        '',
    ]
    lines += ['  SECCIONES DETECTADAS', '  ' + '─'*(W-2)]
    for s in report.get('sections', []):
        lines.append(f"    Sec {s['index']+1:>2}  cc {s['bars']:<12} "
                     f"[{s['archetype']}] {s['label']:<22} "
                     f"vel={s['mean_vel']:.0f}  dens={s['density']:.2f}")
    lines += ['', '  PASOS DE PROCESAMIENTO', '  ' + '─'*(W-2)]
    for step in report.get('steps', []):
        lines.append(f"    • {step}")
    lines += ['', '  PISTAS GENERADAS', '  ' + '─'*(W-2)]
    for i, name in enumerate(created):
        lines.append(f"    Canal {i+1:>2} → {name}")
    lines += [
        '',
        '  ARQUETIPOS DE TEXTURA',
        '  ' + '─'*(W-2),
        '    A Tutti homofónico  — tutti attacks, dinámica alta',
        '    B Melodía+colchón   — melodía + voces largas',
        '    C Melodía+arpegios  — figuración + melodía encima',
        '    D Contrapuntístico  — líneas independientes',
        '    E Sparse/cámara     — pocas notas, mucho espacio',
        '',
        '  NOTA PARA FL STUDIO',
        '  ' + '─'*(W-2),
        '    Cada pista lleva CC1 (dynamics xfade) y CC11 (expression)',
        '    interpolados según la dinámica original del piano.',
        '    Asigna los mismos sample libraries que usas con orchestrator.py.',
        '═'*W,
    ]
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN FINGERPRINT (compatible con orchestrator.py)
# ══════════════════════════════════════════════════════════════════════════════

def export_fingerprint(report: dict, sections_data: List[dict],
                       output_path: str) -> None:
    """Genera fingerprint.json compatible con orchestrator.py."""
    fp = {
        'source':         'piano_expander',
        'key':            report.get('key', '?'),
        'style':          report.get('style', '?'),
        'template':       report.get('template', '?'),
        'expansion_ratio': report.get('expansion_ratio', 1.0),
        'tension_curve':  [s.get('mean_vel', 64) / 127.0
                           for s in report.get('sections', [])],
        'sections':       [
            {
                'bars':     s['bars'],
                'archetype': s['archetype'],
                'label':    s['label'],
                'density':  s['density'],
            }
            for s in report.get('sections', [])
        ],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)
    print(f"  Fingerprint: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE PLANTILLA PERSONALIZADA
# ══════════════════════════════════════════════════════════════════════════════

def load_template(arg_template: str) -> Tuple[dict, str]:
    """
    Carga plantilla por nombre ('chamber','strings','full')
    o desde archivo JSON personalizado.
    """
    if arg_template in TEMPLATES:
        return TEMPLATES[arg_template], arg_template

    path = Path(arg_template)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                custom = json.load(f)
            # Normalizar al formato interno si viene de orchestrator.py
            if 'instruments' not in custom and 'palette' in custom:
                custom = {'name': path.stem, 'instruments': custom['palette']}
            if 'name' not in custom:
                custom['name'] = path.stem
            return custom, path.stem
        except Exception as e:
            print(f"  ⚠ No se pudo cargar plantilla '{arg_template}': {e}")
            print("  → Usando plantilla 'chamber' por defecto")

    print(f"  ⚠ Plantilla '{arg_template}' no reconocida. Usando 'chamber'.")
    return TEMPLATES['chamber'], 'chamber'


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Piano Expander — Expansión orquestal de bocetos pianísticos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('midi', help='MIDI de piano de entrada')
    p.add_argument('--template', default='chamber',
                   help='chamber | strings | full | <archivo.json> (default: chamber)')
    p.add_argument('--style',
                   choices=['auto','romantic','baroque','impressionist','cinematic','chamber'],
                   default='auto', help='Estilo orquestal (default: auto)')
    p.add_argument('--split', default='G4',
                   help='Nota de split si el piano es una sola pista (default: G4)')
    p.add_argument('--no-counter',  action='store_true',
                   help='No generar contramelodías')
    p.add_argument('--counter-density', type=float, default=0.35,
                   help='Densidad de contramelodías 0.0-1.0 (default: 0.35)')
    p.add_argument('--texture-sensitivity', type=float, default=0.5,
                   help='Sensibilidad de clasificación de textura 0.0-1.0 (default: 0.5)')
    p.add_argument('--no-ornaments', action='store_true',
                   help='No aplicar articulación idiomática')
    p.add_argument('--no-cc',        action='store_true',
                   help='No insertar CC1/CC11')
    p.add_argument('--export-fingerprint', action='store_true',
                   help='Exportar fingerprint.json para orchestrator.py')
    p.add_argument('--export-yaml',  action='store_true',
                   help='Exportar expansión como partitura.yaml')
    p.add_argument('--output',       default=None,
                   help='Nombre base de salida (default: <stem>_orquestado)')
    p.add_argument('--report-only',  action='store_true',
                   help='Solo análisis, sin generar MIDI')
    p.add_argument('--verbose',      action='store_true',
                   help='Decisiones detalladas por sección')
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Validar densidad
    args.counter_density    = max(0.0, min(1.0, args.counter_density))
    args.texture_sensitivity = max(0.0, min(1.0, args.texture_sensitivity))

    # Nota de split
    try:
        split_note = name_to_midi(args.split)
    except Exception:
        split_note = 67  # G4
        print(f"  ⚠ No se pudo parsear --split '{args.split}'. Usando G4.")

    print('═' * 65)
    print('  PIANO EXPANDER  v1.0')
    print('═' * 65)
    print(f"  Entrada   : {args.midi}")
    print(f"  Plantilla : {args.template}")
    print(f"  Estilo    : {args.style}")

    if not Path(args.midi).exists():
        print(f"  ERROR: Fichero no encontrado: {args.midi}")
        sys.exit(1)

    # ── Cargar MIDI ─────────────────────────────────────────────────────────
    mid, tpb, tempo = load_midi(args.midi)
    global TICKS
    TICKS = tpb
    bpm   = round(60_000_000 / tempo)
    print(f"  TPB={tpb}  Tempo={bpm} BPM")

    tracks = extract_notes(mid)
    if not tracks:
        print("  ERROR: No se encontraron notas.")
        sys.exit(1)

    # ── Separar manos ────────────────────────────────────────────────────────
    non_empty = [k for k, v in tracks.items() if v]
    print(f"  Pistas detectadas: {len(non_empty)}")

    if len(non_empty) >= 2:
        # Asumir primera pista = MD, segunda = MI (orden standard)
        names = sorted(non_empty)
        all_notes_flat = [n for k in non_empty for n in tracks[k]]
        all_notes_flat.sort(key=lambda n: (n['tick'], -n['pitch']))
        # Split por registro para clasificar MD/MI
        rh_all = [n for n in all_notes_flat if n['pitch'] >= split_note]
        lh_all = [n for n in all_notes_flat if n['pitch'] <  split_note]
        rh = rh_all if rh_all else all_notes_flat
        lh = lh_all
        print(f"  Manos separadas por registro ({midi_to_name(split_note)}): "
              f"MD={len(rh)}  MI={len(lh)}")
    else:
        all_notes = list(tracks.values())[0] if tracks else []
        rh, lh = split_piano_hands(all_notes, split_note, tpb)
        print(f"  Pista única → split interno ({midi_to_name(split_note)}): "
              f"MD={len(rh)}  MI={len(lh)}")

    if not rh and not lh:
        print("  ERROR: No se encontraron notas en el piano.")
        sys.exit(1)

    # ── Cargar plantilla ─────────────────────────────────────────────────────
    template, template_name = load_template(args.template)
    n_instr = len(template.get('instruments', []))
    print(f"  Instrumentos  : {n_instr}  ({template.get('name','?')})")

    # ── Solo informe ─────────────────────────────────────────────────────────
    if args.report_only:
        all_piano = sorted(rh + lh, key=lambda n: n['tick'])
        tonic, mode = detect_key(all_piano)
        sections    = segment_sections(all_piano, tpb, args.texture_sensitivity)
        print(f"\n  Tonalidad detectada: {NOTE_NAMES[tonic]} {mode}")
        print(f"  Secciones          : {len(sections)}")
        for sec in sections:
            arch = classify_texture(sec, tpb, args.style, args.texture_sensitivity)
            print(f"    cc {sec['bar_start']:>3}-{sec['bar_end']:<4} "
                  f"[{arch}] {TEXTURE_ARCHETYPES[arch]:<22} "
                  f"vel={sec['mean_vel']:.0f} dens={sec['density']:.2f}")
        sys.exit(0)

    # ── Nombres de salida ────────────────────────────────────────────────────
    stem = Path(args.midi).stem
    base = args.output or (stem + '_orquestado')
    base = base.replace('.mid', '')
    out_midi    = base + '.mid'
    out_report  = base + '_report.txt'
    out_fp      = base + '.fingerprint.json'
    out_yaml    = base + '.yaml'

    # ── Expansión ────────────────────────────────────────────────────────────
    print('\n  Expandiendo...')
    instr_notes, instr_cc, report_data = expand(
        rh, lh, tpb, tempo, template, args.style, args)

    # ── Escribir MIDI ────────────────────────────────────────────────────────
    print(f'\n  Escribiendo: {out_midi}')
    created = build_output_midi(instr_notes, instr_cc,
                                template, tempo, tpb, out_midi)
    print(f"  Pistas creadas: {len(created)}")

    # ── Fingerprint ──────────────────────────────────────────────────────────
    if args.export_fingerprint:
        export_fingerprint(report_data, report_data.get('sections', []), out_fp)

    # ── YAML ─────────────────────────────────────────────────────────────────
    if args.export_yaml and YAML_OK:
        # Estructura mínima compatible con partitura.yaml
        ya = {
            'obra': {'titulo': stem, 'generado_por': 'piano_expander'},
            'tonalidad': report_data.get('key', '?'),
            'estilo':    report_data.get('style', '?'),
            'instrumentos': [INSTR_DISPLAY.get(i['name'], i['name'])
                             for i in template['instruments']],
        }
        with open(out_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(ya, f, allow_unicode=True, default_flow_style=False)
        print(f"  YAML: {out_yaml}")
    elif args.export_yaml and not YAML_OK:
        print("  ⚠ PyYAML no disponible. Instala con: pip install PyYAML")

    # ── Informe ───────────────────────────────────────────────────────────────
    report_text = generate_report(report_data, args.midi, out_midi,
                                  created, template)
    with open(out_report, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Informe: {out_report}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print('\n' + '═'*65)
    print('  RESUMEN')
    print('═'*65)
    print(f"  MIDI orquestado : {out_midi}")
    print(f"  Informe         : {out_report}")
    print(f"  Pistas          : {len(created)}")
    print(f"  Tonalidad       : {report_data.get('key','?')}")
    print(f"  Estilo          : {report_data.get('style','?')}")
    print(f"  Expansión       : {report_data.get('expansion_ratio',0):.1f}x notas")
    secs = report_data.get('sections', [])
    if secs:
        archcount = defaultdict(int)
        for s in secs:
            archcount[s['archetype']] += 1
        for k, v in sorted(archcount.items()):
            print(f"    [{k}] {TEXTURE_ARCHETYPES[k]:<22} {v} sección(es)")
    if args.export_fingerprint:
        print(f"  Fingerprint     : {out_fp}")
    print('═'*65)
    print(f'\n  Importa {out_midi} en FL Studio.')
    for i, name in enumerate(created):
        prog = INSTR_PROGRAM.get(
            next((ins['name'] for ins in template['instruments']
                  if INSTR_DISPLAY.get(ins['name'], ins['name']) == name), ''), 40)
        print(f"  Canal {i+1:>2} → {name}")


if __name__ == '__main__':
    main()
