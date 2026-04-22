#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ML EXPANDER  v1.0                                     ║
║     Expansión orquestal de bocetos pianísticos mediante machine learning     ║
║                                                                              ║
║  Aprende la transformación piano→orquesta desde partituras reales usando    ║
║  reducción sintética: colapsa obras orquestales a piano para generar pares  ║
║  de entrenamiento (piano_reducido, orquesta_original).                       ║
║                                                                              ║
║  TRES MODELOS PyTorch (MLP, entrenables en CPU):                            ║
║  [M1] TextureNet     — features de sección → arquetipo + tensión            ║
║  [M2] InstrumentNet  — features + rol → secuencia de eventos MIDI           ║
║  [M3] ExpressionNet  — tensión + instrumento → curvas CC1/CC11              ║
║                                                                              ║
║  PIPELINE DE FEATURES (31 dimensiones con semántica musical real):          ║
║  · analyze_bars (orchestrator.py) — 16 métricas por compás con gradientes  ║
║  · score_bass_line (bass_line_composer.py) — 6 criterios de calidad         ║
║  · detect_key (piano_expander.py) — tónica + modo                          ║
║  · posición relativa, gap_ratio, histograma de intervalos comprimido        ║
║                                                                              ║
║  CORRECCIÓN IDIOMÁTICA POST-ML (reglas, sin ML):                            ║
║  · chord_tone_category / admissible_pcs (melody_harmonizer.py)             ║
║  · _best_candidate / _motion_type (counterpoint.py)                        ║
║  · classify_articulation (orchestrator.py)                                  ║
║  · RANGES / SWEET idiomáticos (piano_expander.py)                          ║
║                                                                              ║
║  MODOS:                                                                      ║
║    train   <dir_midi>    Entrena M1/M2/M3 desde MIDIs orquestales reales   ║
║    expand  <boceto.mid>  Expande un boceto de piano usando checkpoints       ║
║    reduce  <midi.mid>    Solo ejecuta SyntheticReducer (inspección)         ║
║                                                                              ║
║  USO:                                                                        ║
║    python ml_expander.py train  /datos/maestro/ --epochs 50                ║
║    python ml_expander.py expand boceto.mid --template full                  ║
║    python ml_expander.py reduce orquesta.mid --output reduccion.mid        ║
║                                                                              ║
║  DATOS: Descargar MIDIs orquestales manualmente desde:                      ║
║    MAESTRO  → https://magenta.tensorflow.org/datasets/maestro               ║
║    MuseScore → https://musescore.com (filtrar por orquesta, CC0/public)    ║
║    Apuntar al directorio con --train <dir>                                  ║
║                                                                              ║
║  DEPENDENCIAS: pip install mido numpy torch                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── MIDI ──────────────────────────────────────────────────────────────────────
try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)

# ── NumPy ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    # Se comprueba en los modos que lo necesitan


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 1 — CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
PERC_CH    = 9

# Rangos idiomáticos [lo, hi] en MIDI notes
RANGES = {
    'violin1':    (55, 96), 'violin2':   (55, 91), 'viola':      (48, 84),
    'cello':      (36, 76), 'contrabass':(28, 60),
    'flute':      (60, 96), 'oboe':      (58, 91), 'clarinet':   (50, 94),
    'bassoon':    (34, 75),
    'horn':       (34, 77), 'trumpet':   (52, 82), 'trombone':   (34, 72),
    'tuba':       (28, 58),
}

# Registro cómodo / expresivo
SWEET = {
    'violin1':    (64, 88), 'violin2':   (60, 84), 'viola':      (55, 79),
    'cello':      (43, 72), 'contrabass':(33, 52),
    'flute':      (65, 90), 'oboe':      (62, 86), 'clarinet':   (55, 86),
    'bassoon':    (38, 67),
    'horn':       (40, 70), 'trumpet':   (56, 77), 'trombone':   (40, 67),
}

INSTR_PROGRAM = {
    'violin1':40, 'violin2':40, 'viola':41, 'cello':42, 'contrabass':43,
    'flute':73, 'oboe':68, 'clarinet':71, 'bassoon':70,
    'horn':60, 'trumpet':56, 'trombone':57, 'tuba':58,
}

INSTR_DISPLAY = {
    'violin1':'Violin I', 'violin2':'Violin II', 'viola':'Viola',
    'cello':'Cello', 'contrabass':'Contrabass',
    'flute':'Flute', 'oboe':'Oboe', 'clarinet':'Clarinet', 'bassoon':'Bassoon',
    'horn':'Horn', 'trumpet':'Trumpet', 'trombone':'Trombone', 'tuba':'Tuba',
}

FAMILY = {
    'violin1':'strings', 'violin2':'strings', 'viola':'strings',
    'cello':'strings',   'contrabass':'strings',
    'flute':'woodwind',  'oboe':'woodwind', 'clarinet':'woodwind',
    'bassoon':'woodwind',
    'horn':'brass',      'trumpet':'brass', 'trombone':'brass', 'tuba':'brass',
}

# Rol orquestal de cada instrumento
INSTR_ROLE = {
    'violin1':'melody',      'violin2':'harmony',    'viola':'inner',
    'cello':'bass_melodic',  'contrabass':'bass_root',
    'flute':'melody_high',   'oboe':'counter',       'clarinet':'harmony',
    'bassoon':'bass_melodic','horn':'pad',            'trumpet':'melody',
    'trombone':'pad_low',    'tuba':'bass_root',
}

# Embedding índice por instrumento (para M2/M3)
INSTR_IDX = {name: i for i, name in enumerate(sorted(INSTR_ROLE.keys()))}
N_INSTRUMENTS = len(INSTR_IDX)

# Embedding índice por rol (para M2)
ROLE_IDX = {r: i for i, r in enumerate(sorted(set(INSTR_ROLE.values())))}
N_ROLES = len(ROLE_IDX)

# Arquetipos de textura
ARCHETYPES = ['A', 'B', 'C', 'D', 'E']  # índices 0-4
N_ARCHETYPES = len(ARCHETYPES)

# Plantillas de instrumentos
TEMPLATES = {
    'chamber': ['violin1','violin2','viola','cello','flute','oboe'],
    'strings': ['violin1','violin2','viola','cello','contrabass'],
    'full':    ['violin1','violin2','viola','cello','contrabass',
                'flute','oboe','clarinet','bassoon',
                'horn','trumpet','trombone','tuba'],
}

# Umbrales de tensión para entrada de familias
FAMILY_ENTRY_THRESHOLD = {'strings': 0.0, 'woodwind': 0.15, 'brass': 0.45}

# Intervalos consonantes (semitones mod 12)
PERFECT_CONSONANCES    = {0, 7, 12}
IMPERFECT_CONSONANCES  = {3, 4, 8, 9}
ALL_CONSONANCES        = PERFECT_CONSONANCES | IMPERFECT_CONSONANCES

# Categorías de grado sobre un acorde
NOTE_CATEGORY = {
    0: 'chord_tone',  # raíz
    4: 'chord_tone',  # 3ª mayor
    3: 'chord_tone',  # 3ª menor
    7: 'chord_tone',  # 5ª
    2: 'tension',     # 9ª
    5: 'avoid',       # 4ª (sobre acorde mayor)
    6: 'tension',     # #4 / b5
    9: 'tension',     # 6ª
   10: 'tension',     # 7ª menor
   11: 'tension',     # 7ª mayor
    1: 'avoid',       # b9
    8: 'avoid',       # b6 / #5
}

# Intervalos de acorde por calidad
CHORD_INTERVALS = {
    'M':   [0, 4, 7],
    'm':   [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'M7':  [0, 4, 7, 11],
    'm7':  [0, 3, 7, 10],
    'Mm7': [0, 4, 7, 10],
    'dom7':[0, 4, 7, 10],
    'd7':  [0, 3, 6, 9],
    'hd7': [0, 3, 6, 10],
}

# Articulaciones (umbrales en beats)
ART_THRESHOLDS = {'legato': 2.0, 'sustain': 0.75, 'portato': 0.4, 'spiccato': 0.0}

# ── Bins de cuantización para M2 (defaults, sobreescribibles por CLI) ─────────
DEFAULT_DELTA_BINS   = 32   # delta_tick:  0–2 compases
DEFAULT_PITCH_BINS   = 88   # pitch MIDI:  21–108
DEFAULT_VEL_BINS     = 16   # velocity:    0–127 en 8 unidades
DEFAULT_DUR_BINS     = 24   # duration:    logarítmica

# Dimensión del vector de features de sección (SectionFeaturizer)
FEAT_DIM = 31

# ── Hiperparámetros por defecto ────────────────────────────────────────────────
DEFAULT_HIDDEN_DIM  = 128
DEFAULT_EPOCHS      = 30
DEFAULT_LR          = 1e-3
DEFAULT_BATCH_SIZE  = 32
DEFAULT_CHECKPOINT  = './checkpoints'


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

def pc(n: int) -> int:
    return n % 12

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def is_consonant(a: int, b: int) -> bool:
    return abs(a - b) % 12 in ALL_CONSONANCES

def is_perfect(a: int, b: int) -> bool:
    return abs(a - b) % 12 in PERFECT_CONSONANCES

def ticks_to_beats(ticks: int, tpb: int) -> float:
    return ticks / tpb

def beats_to_ticks(beats: float, tpb: int) -> int:
    return int(beats * tpb)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — LECTURA Y ANÁLISIS MIDI
#  (reutilizado de piano_expander.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_midi(path: str):
    """Carga un MIDI y devuelve (MidiFile, tpb, tempo)."""
    mid  = MidiFile(path)
    tpb  = mid.ticks_per_beat
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    return mid, tpb, tempo


def extract_notes(mid: MidiFile) -> Dict[str, List[dict]]:
    """Extrae notas de todas las pistas. Devuelve {nombre_pista: [nota, ...]}.
    Usa índice numérico como clave para evitar colisiones entre pistas con
    el mismo nombre (e.g. dos pistas 'Piano' en el mismo MIDI)."""
    result = {}
    for i, track in enumerate(mid.tracks):
        name = track.name.strip().rstrip('\x00') or f'Track_{i}'
        key  = f'{i}_{name}'   # índice garantiza unicidad
        on   = {}
        notes= []
        t    = 0
        for msg in track:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != PERC_CH:
                on[(msg.channel, msg.note)] = (t, msg.velocity)
            elif (msg.type == 'note_off' or
                  (msg.type == 'note_on' and msg.velocity == 0)) and msg.channel != PERC_CH:
                mk = (msg.channel, msg.note)
                if mk in on:
                    t0, vel = on.pop(mk)
                    dur = t - t0
                    if dur > 0:
                        notes.append({'tick': t0, 'pitch': msg.note,
                                      'velocity': vel, 'duration': dur,
                                      'channel': msg.channel})
        if notes:
            result[key] = sorted(notes, key=lambda n: n['tick'])
    return result


def extract_notes_flat(mid: MidiFile) -> List[dict]:
    """Todas las notas de todas las pistas en una lista plana ordenada."""
    all_notes = []
    for notes in extract_notes(mid).values():
        all_notes.extend(notes)
    return sorted(all_notes, key=lambda n: n['tick'])


def split_hands(notes: List[dict], split: int, tpb: int):
    """Divide notas en mano derecha e izquierda por pitch split."""
    slack = tpb // 8
    sn    = sorted(notes, key=lambda n: (n['tick'], -n['pitch']))
    clusters = []
    i = 0
    while i < len(sn):
        g = [sn[i]]
        j = i + 1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= slack:
            g.append(sn[j]); j += 1
        clusters.append(g); i = j
    rh, lh = [], []
    for c in clusters:
        cs     = sorted(c, key=lambda n: -n['pitch'])
        above  = [n for n in cs if n['pitch'] >= split]
        below  = [n for n in cs if n['pitch'] <  split]
        if above and below:
            rh.extend(above); lh.extend(below)
        elif len(cs) == 1:
            (rh if cs[0]['pitch'] >= split else lh).extend(cs)
        else:
            rh.append(cs[0]); lh.extend(cs[1:])
    return rh, lh


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — TEORÍA MUSICAL
#  (reutilizado de piano_expander.py + melody_harmonizer.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_key(notes: List[dict]) -> Tuple[int, str]:
    """Detecta tónica y modo (Krumhansl-Schmuckler). Devuelve (tonic_pc, mode)."""
    pm = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    pn = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    counts = [0.0] * 12
    for n in notes:
        counts[pc(n['pitch'])] += n['duration']
    tot = sum(counts) or 1
    counts = [c / tot for c in counts]
    best = -999; bk = 0; bm = 'major'
    for t in range(12):
        for mode, prof in [('major', pm), ('minor', pn)]:
            rot = [prof[(i - t) % 12] for i in range(12)]
            s   = float(np.corrcoef(counts, rot)[0, 1])
            if s > best:
                best = s; bk = t; bm = mode
    return bk, bm


def scale_pcs(tonic: int, mode: str) -> List[int]:
    itvs = {'major': [0,2,4,5,7,9,11], 'minor': [0,2,3,5,7,8,10]}
    return [(tonic + i) % 12 for i in itvs.get(mode, [0,2,4,5,7,9,11])]


def snap_to_scale(pitch: int, scale: List[int]) -> int:
    p = pc(pitch)
    if p in scale:
        return pitch
    for d in range(1, 7):
        if (p + d) % 12 in scale: return pitch + d
        if (p - d) % 12 in scale: return pitch - d
    return pitch


def fit_range(pitch: int, lo: int, hi: int) -> Optional[int]:
    p = pitch
    while p < lo:
        if p + 12 > hi: return None
        p += 12
    while p > hi:
        if p - 12 < lo: return None
        p -= 12
    return p if lo <= p <= hi else None


def nearest_chord_tone(pitch: int, chord_pcs: List[int]) -> int:
    best = pitch; best_d = 999
    for oct_ in range(-2, 3):
        for cp in chord_pcs:
            p = cp + (pitch // 12 + oct_) * 12
            if 21 <= p <= 108:
                d = abs(p - pitch)
                if d < best_d:
                    best_d = d; best = p
    return best


def infer_chord(notes: List[dict]) -> List[int]:
    if not notes:
        return [0, 4, 7]
    return list(set(pc(n['pitch']) for n in notes)) or [0, 4, 7]


def chord_tone_category(mel_pc: int, root_pc: int, quality: str) -> str:
    """Clasifica una nota respecto a un acorde: chord_tone | tension | avoid."""
    interval   = (mel_pc - root_pc) % 12
    base_cat   = NOTE_CATEGORY.get(interval, 'tension')
    if quality in ('Mm7', 'M7', 'd7', 'hd7', 'dom7'):
        if interval == 11: return 'avoid'
        if interval == 6:  return 'tension'
    if quality in ('M', 'M7'):
        if interval == 5:  return 'avoid'
    return base_cat


def chord_tones_for_quality(root_pc: int, quality: str) -> set:
    return {(root_pc + i) % 12 for i in CHORD_INTERVALS.get(quality, [0, 4, 7])}


def admissible_pcs(root_pc: int, quality: str,
                   scale: Optional[List[int]] = None) -> set:
    """Pitch classes admisibles (chord_tone + tension no evitada)."""
    ok = set()
    for p in range(12):
        cat = chord_tone_category(p, root_pc, quality)
        if cat in ('chord_tone', 'tension'):
            ok.add(p)
    if scale:
        ok &= set(scale)
        ok |= chord_tones_for_quality(root_pc, quality)
    return ok


def nearest_admissible(pitch: int, adm: set, direction: str = 'nearest') -> int:
    """Mueve pitch al PC admisible más cercano en la dirección dada."""
    if not adm:
        return pitch
    p = pc(pitch)
    octave = pitch // 12

    def candidates():
        for oct_ in range(-1, 2):
            for a in adm:
                yield a + (octave + oct_) * 12

    cands = [c for c in candidates() if 21 <= c <= 108]
    if not cands:
        return pitch
    if direction == 'nearest':
        return min(cands, key=lambda c: abs(c - pitch))
    elif direction == 'above':
        above = [c for c in cands if c >= pitch]
        return min(above, key=lambda c: c - pitch) if above else min(cands, key=lambda c: abs(c - pitch))
    else:  # below
        below = [c for c in cands if c <= pitch]
        return max(below, key=lambda c: pitch - c) if below else min(cands, key=lambda c: abs(c - pitch))


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — SYNTHETIC REDUCER
#  Colapsa una partitura orquestal real a piano sintético.
#  Genera los pares (piano_reducido, orquesta_original) para entrenamiento.
# ══════════════════════════════════════════════════════════════════════════════

def reduce_orchestra_to_piano(notes: List[dict], tpb: int,
                               split_pitch: int = 60) -> Tuple[List[dict], List[dict]]:
    """
    Colapsa un conjunto de notas orquestales en una reducción de piano:
      · Derecha (rh): notas >= split_pitch, máx 4 simultáneas (por vel)
      · Izquierda (lh): notas < split_pitch, máx 2 simultáneas + siempre el bajo
      · Dinámica normalizada al rango típico de piano (40–100)
      · Articulation unificada: todos los gaps > 0.1 beat se respetan
    Devuelve (rh, lh).
    """
    if not notes:
        return [], []

    slack    = tpb // 8
    sorted_n = sorted(notes, key=lambda n: n['tick'])
    rh: List[dict] = []
    lh: List[dict] = []

    i = 0
    while i < len(sorted_n):
        # Agrupar notas simultáneas (dentro del slack)
        group = [sorted_n[i]]
        j = i + 1
        while j < len(sorted_n) and sorted_n[j]['tick'] - sorted_n[i]['tick'] <= slack:
            group.append(sorted_n[j]); j += 1

        high = [n for n in group if n['pitch'] >= split_pitch]
        low  = [n for n in group if n['pitch'] <  split_pitch]

        # Normalizar velocidades: media del grupo → rango piano
        def norm_vel(v: int) -> int:
            return clamp(int(40 + (v / 127) * 60), 20, 100)

        # Mano derecha: top-4 por velocidad
        high_sorted = sorted(high, key=lambda n: -n['velocity'])
        for n in high_sorted[:4]:
            rh.append({**n, 'velocity': norm_vel(n['velocity'])})

        # Mano izquierda: bajo siempre + hasta 1 más
        if low:
            bass_note = min(low, key=lambda n: n['pitch'])
            lh.append({**bass_note, 'velocity': norm_vel(bass_note['velocity'])})
            others = [n for n in low if n is not bass_note]
            others.sort(key=lambda n: -n['velocity'])
            for n in others[:1]:
                lh.append({**n, 'velocity': norm_vel(n['velocity'])})

        i = j

    rh.sort(key=lambda n: n['tick'])
    lh.sort(key=lambda n: n['tick'])
    return rh, lh


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 6 — SECTION FEATURIZER
#  Extrae un vector de 31 dimensiones por sección con semántica musical real.
#  Combina analyze_bars (orchestrator.py) + score_bass_line (bass_line_composer.py)
#  + detect_key (piano_expander.py).
# ══════════════════════════════════════════════════════════════════════════════

def segment_sections(notes: List[dict], tpb: int, sensitivity: float = 0.5) -> List[dict]:
    """
    Segmenta las notas en secciones por cambios de densidad/dinámica.
    Devuelve lista de secciones con sus notas y métricas básicas.
    """
    if not notes:
        return []
    bar         = tpb * 4
    max_tick    = max(n['tick'] + n['duration'] for n in notes)
    n_bars      = max(int(math.ceil(max_tick / bar)), 1)
    bd          = np.zeros(n_bars)
    bv          = np.zeros(n_bars)
    for n in notes:
        b = min(n['tick'] // bar, n_bars - 1)
        bd[b] += 1
        bv[b] += n['velocity']
    bd /= max(bd.max(), 1)
    bv /= max(bv.max(), 1)
    signal    = (bd + bv) / 2.0
    threshold = 0.20 + (1.0 - sensitivity) * 0.25
    window    = max(2, int(4 * (1 - sensitivity)))
    boundaries = [0]
    for i in range(window, n_bars - window):
        left  = np.mean(signal[max(0, i - window):i])
        right = np.mean(signal[i:min(n_bars, i + window)])
        if abs(right - left) > threshold and i - boundaries[-1] >= 4:
            boundaries.append(i)
    boundaries.append(n_bars)
    sections = []
    for k in range(len(boundaries) - 1):
        bs, be  = boundaries[k], boundaries[k + 1]
        ts, te  = bs * bar, be * bar
        sn      = [n for n in notes if ts <= n['tick'] < te]
        if not sn:
            continue
        sections.append({
            'index':      k,
            'bar_start':  bs + 1,
            'bar_end':    be,
            'tick_start': ts,
            'tick_end':   te,
            'notes':      sn,
            'density':    float(np.mean(bd[bs:be])),
            'mean_vel':   float(np.mean([n['velocity'] for n in sn])),
            'mean_dur':   float(np.mean([n['duration'] for n in sn])),
            'n_notes':    len(sn),
        })
    return sections


def _bar_metrics(notes: List[dict], tpb: int, beats_per_bar: int = 4) -> dict:
    """
    Calcula métricas musicales para un conjunto de notas (un compás o sección).
    Adaptado de analyze_bars (orchestrator.py).
    """
    m: dict = {}
    if not notes:
        return {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
            'pitch_direction': 0, 'velocity_mean': 0, 'velocity_std': 0,
            'note_density': 0.0, 'dur_mean_beats': 0,
            'gap_beats': float(beats_per_bar),
            'strong_beat_ratio': 0.0, 'max_leap': 0,
            'interval_mean': 0.0, 'is_empty': True,
        }

    pitches = [n['pitch']    for n in notes]
    vels    = [n['velocity'] for n in notes]
    durs    = [n['duration'] / tpb for n in notes]

    m['pitch_mean']      = float(np.mean(pitches))
    m['pitch_std']       = float(np.std(pitches))
    m['pitch_range']     = int(max(pitches) - min(pitches))
    m['velocity_mean']   = float(np.mean(vels))
    m['velocity_std']    = float(np.std(vels))
    m['note_density']    = len(notes) / beats_per_bar
    m['dur_mean_beats']  = float(np.mean(durs))
    m['is_empty']        = False

    # Dirección melódica
    sorted_n = sorted(notes, key=lambda n: n['tick'])
    if len(sorted_n) >= 2:
        diff = sorted_n[-1]['pitch'] - sorted_n[0]['pitch']
        m['pitch_direction'] = 1 if diff > 1 else (-1 if diff < -1 else 0)
    else:
        m['pitch_direction'] = 0

    # Silencio
    occupied    = sum(min(d, beats_per_bar) for d in durs)
    m['gap_beats'] = max(0.0, beats_per_bar - occupied)

    # Strong-beat ratio (beats 1 y 3 en 4/4)
    bar_start      = notes[0]['tick'] // (tpb * beats_per_bar) * (tpb * beats_per_bar)
    strong_ticks   = {bar_start, bar_start + tpb * 2}
    strong_count   = sum(1 for n in notes
                         if min(abs(n['tick'] - s) for s in strong_ticks) < tpb // 4)
    m['strong_beat_ratio'] = strong_count / max(len(notes), 1)

    # Intervalos
    pitches_sorted = [n['pitch'] for n in sorted_n]
    if len(pitches_sorted) >= 2:
        intervals       = [abs(pitches_sorted[k+1] - pitches_sorted[k])
                           for k in range(len(pitches_sorted) - 1)]
        m['max_leap']       = max(intervals)
        m['interval_mean']  = float(np.mean(intervals))
    else:
        m['max_leap']      = 0
        m['interval_mean'] = 0.0

    return m


def _score_bass_quality(notes: List[dict], chord_pcs: List[int],
                         tpb: int, beats_per_bar: int = 4) -> Tuple[float, ...]:
    """
    Calcula 6 scores de calidad del bajo (adaptado de score_bass_line en
    bass_line_composer.py). Devuelve tupla de 6 floats en [0,1].
    """
    if not notes:
        return (0.0,) * 6

    pitches = [n['pitch'] for n in notes]

    # 1. Consonancia armónica
    hits       = sum(1 for n in notes if pc(n['pitch']) in chord_pcs)
    consonance = hits / len(notes)

    # 2. Solidez rítmica: raíz en tiempo 1 de cada compás
    bar       = tpb * beats_per_bar
    n_bars_l  = max(1, int(max(n['tick'] for n in notes) / bar) + 1)
    downbeats = 0
    for b in range(n_bars_l):
        bar_start = b * bar
        bar_n     = [n for n in notes if abs(n['tick'] - bar_start) < tpb // 4]
        if bar_n and chord_pcs and pc(bar_n[0]['pitch']) == chord_pcs[0]:
            downbeats += 1
    downbeat_score = downbeats / n_bars_l

    # 3. Suavidad: penalizar saltos > octava
    intervals   = [abs(pitches[k+1] - pitches[k]) for k in range(len(pitches) - 1)]
    big_leaps   = sum(1 for iv in intervals if iv > 12)
    smoothness  = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 4. Rango apropiado (centro ideal ~38 = D2)
    center      = (min(pitches) + max(pitches)) / 2
    rng_score   = max(0.0, min(1.0, 1.0 - abs(center - 38) / 20))

    # 5. Variedad de pitches
    variety     = min(1.0, len(set(pitches)) / max(len(pitches) * 0.5, 1))

    # 6. Dirección: simplicidad de placeholder (0.5 neutral)
    direction_score = 0.5

    return (consonance, downbeat_score, smoothness, rng_score, variety, direction_score)


def _interval_histogram_4(notes: List[dict]) -> List[float]:
    """Histograma de intervalos en 4 bins: [semitono, 2a-3a, 4a-5a, 6a+]."""
    if len(notes) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    sorted_n  = sorted(notes, key=lambda n: n['tick'])
    intervals = [abs(sorted_n[k+1]['pitch'] - sorted_n[k]['pitch'])
                 for k in range(len(sorted_n) - 1)]
    bins = [0, 0, 0, 0]
    for iv in intervals:
        if iv <= 1:   bins[0] += 1
        elif iv <= 4: bins[1] += 1
        elif iv <= 7: bins[2] += 1
        else:         bins[3] += 1
    total = sum(bins) or 1
    return [b / total for b in bins]


def section_to_feature_vector(section: dict, tpb: int,
                               total_sections: int = 1) -> np.ndarray:
    """
    Extrae vector de 31 dimensiones para una sección.
    Combina métricas de orchestrator.py, bass_line_composer.py y piano_expander.py.

    Dimensiones:
      [0-15]  métricas de compás (analyze_bars)
      [16-21] scores de calidad del bajo (score_bass_line)
      [22-23] tónica one-hot simplificada (pc/12, mode 0=major/1=minor)
      [24]    posición relativa en la obra [0,1]
      [25]    gap_ratio (silencio sobre duración total)
      [26-29] histograma de intervalos 4 bins
      [30]    tensión estimada desde densidad+velocidad
    """
    notes      = section.get('notes', [])
    n_bars_sec = max(section['bar_end'] - section['bar_start'] + 1, 1)
    bpm        = 4  # beats per bar

    m = _bar_metrics(notes, tpb, bpm * n_bars_sec)

    # Features 0-15: métricas de compás
    feat = [
        m['pitch_mean'] / 127.0,          # 0
        m['pitch_std']  / 40.0,           # 1
        m['pitch_range']/ 60.0,           # 2
        float(m['pitch_direction'] + 1) / 2.0,  # 3  (-1→0, 0→0.5, 1→1)
        m['velocity_mean'] / 127.0,       # 4
        m['velocity_std']  / 40.0,        # 5
        min(m['note_density'] / 8.0, 1.0),# 6
        min(m['dur_mean_beats'] / 4.0, 1.0), # 7
        m['gap_beats'] / max(bpm * n_bars_sec, 1), # 8
        m['strong_beat_ratio'],           # 9
        min(m['max_leap'] / 12.0, 1.0),  # 10
        min(m['interval_mean'] / 6.0, 1.0), # 11
        section['density'],               # 12  (normalizado en segment_sections)
        section['mean_vel'] / 127.0,      # 13
        min(section['mean_dur'] / (tpb * 4), 1.0), # 14
        min(section['n_notes'] / 200.0, 1.0),       # 15
    ]

    # Features 16-21: scores de calidad del bajo
    chord_pcs = infer_chord(notes)
    bass_scores = _score_bass_quality(notes, chord_pcs, tpb, bpm)
    feat.extend(bass_scores)  # 6 valores → índices 16-21

    # Features 22-23: tonalidad
    tonic, mode = detect_key(notes) if notes else (0, 'major')
    feat.append(tonic / 12.0)                         # 22
    feat.append(0.0 if mode == 'major' else 1.0)      # 23

    # Feature 24: posición relativa en la obra
    pos_norm = section['index'] / max(total_sections - 1, 1)
    feat.append(pos_norm)                              # 24

    # Feature 25: gap_ratio
    tick_span = section['tick_end'] - section['tick_start']
    if tick_span > 0 and notes:
        total_dur = sum(n['duration'] for n in notes)
        feat.append(max(0.0, 1.0 - total_dur / tick_span))  # 25
    else:
        feat.append(0.0)                               # 25

    # Features 26-29: histograma de intervalos 4 bins
    feat.extend(_interval_histogram_4(notes))          # 26-29

    # Feature 30: tensión estimada
    tension = clamp(section['density'] * 0.5 + (section['mean_vel'] / 127) * 0.5, 0, 1)
    feat.append(tension)                               # 30

    assert len(feat) == FEAT_DIM, f"Feature dim mismatch: {len(feat)} != {FEAT_DIM}"
    return np.array(feat, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 7 — CUANTIZACIÓN DE EVENTOS MIDI PARA M2
# ══════════════════════════════════════════════════════════════════════════════

class EventQuantizer:
    """
    Cuantiza eventos MIDI (delta_tick, pitch, velocity, duration) en bins discretos
    para que M2 pueda generar distribuciones de probabilidad sobre ellos.
    La duración se cuantiza en escala logarítmica.
    """
    def __init__(self, delta_bins: int = DEFAULT_DELTA_BINS,
                 pitch_bins:  int = DEFAULT_PITCH_BINS,
                 vel_bins:    int = DEFAULT_VEL_BINS,
                 dur_bins:    int = DEFAULT_DUR_BINS,
                 tpb:         int = 480):
        self.delta_bins = delta_bins
        self.pitch_bins = pitch_bins
        self.vel_bins   = vel_bins
        self.dur_bins   = dur_bins
        self.tpb        = tpb

        # Delta tick: 0 a 2 beats — rango más musical y denso.
        # Con 8 beats (valor anterior) el modelo sin entrenamiento real
        # genera medias de 4 beats entre notas, dejando silencios enormes.
        self.delta_max  = tpb * 2
        # Duración: 1 tick a 4 compases logarítmica
        self.dur_min    = 1
        self.dur_max    = tpb * 16
        self.pitch_min  = 21  # A0
        self.pitch_max  = 108 # C8

    def encode(self, delta_tick: int, pitch: int,
               velocity: int, duration: int) -> Tuple[int, int, int, int]:
        d_bin  = clamp(int(delta_tick / self.delta_max * self.delta_bins),
                       0, self.delta_bins - 1)
        p_bin  = clamp(pitch - self.pitch_min, 0, self.pitch_bins - 1)
        v_bin  = clamp(int(velocity / 128 * self.vel_bins), 0, self.vel_bins - 1)

        # Duración logarítmica
        log_dur = math.log(max(duration, self.dur_min))
        log_min = math.log(self.dur_min)
        log_max = math.log(self.dur_max)
        dur_bin = clamp(int((log_dur - log_min) / (log_max - log_min) * self.dur_bins),
                        0, self.dur_bins - 1)

        return d_bin, p_bin, v_bin, dur_bin

    def decode(self, d_bin: int, p_bin: int,
               v_bin: int, dur_bin: int) -> Tuple[int, int, int, int]:
        delta_tick = int(d_bin / self.delta_bins * self.delta_max)
        pitch      = clamp(p_bin + self.pitch_min, self.pitch_min, self.pitch_max)
        velocity   = clamp(int((v_bin + 0.5) / self.vel_bins * 128), 1, 127)

        log_min  = math.log(self.dur_min)
        log_max  = math.log(self.dur_max)
        duration = int(math.exp(log_min + (dur_bin + 0.5) / self.dur_bins * (log_max - log_min)))

        return delta_tick, pitch, velocity, max(duration, 30)

    @property
    def output_dim(self) -> int:
        return self.delta_bins + self.pitch_bins + self.vel_bins + self.dur_bins

    def total_bins(self) -> Tuple[int, int, int, int]:
        return self.delta_bins, self.pitch_bins, self.vel_bins, self.dur_bins


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 8 — MODELOS PYTORCH
# ══════════════════════════════════════════════════════════════════════════════

def _mlp(in_dim: int, hidden: int, out_dim: int,
         layers: int = 2, dropout: float = 0.2) -> nn.Sequential:
    """Construye un MLP genérico con BatchNorm y Dropout."""
    mods = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    if dropout > 0:
        mods.append(nn.Dropout(dropout))
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        hidden = hidden // 2
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class TextureNet(nn.Module):
    """
    M1 — Clasifica textura y tensión desde features de sección.
    Entrada:  vector [FEAT_DIM]
    Salida:   logits [N_ARCHETYPES] + tensión escalar [1]
    """
    def __init__(self, feat_dim: int = FEAT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 n_archetypes: int = N_ARCHETYPES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.archetype_head = nn.Linear(hidden_dim // 2, n_archetypes)
        self.tension_head   = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        h     = self.backbone(x)
        arch  = self.archetype_head(h)   # logits, sin softmax
        tens  = self.tension_head(h).squeeze(-1)
        return arch, tens


class InstrumentNet(nn.Module):
    """
    M2 — Genera una secuencia de eventos MIDI para un instrumento dado el
    contexto de sección y su rol.

    Entrada:  features [FEAT_DIM] + rol embedding [rol_emb_dim]
    Salida:   4 cabezas independientes (una distribución por dimensión):
              delta_bins | pitch_bins | vel_bins | dur_bins
    El modelo es "autoregresivo de un paso": genera un evento por llamada.
    La generación secuencial se orquesta externamente en expand_with_ml().
    """
    def __init__(self, feat_dim: int = FEAT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 n_roles: int = N_ROLES,
                 rol_emb_dim: int = 8,
                 delta_bins: int = DEFAULT_DELTA_BINS,
                 pitch_bins: int = DEFAULT_PITCH_BINS,
                 vel_bins:   int = DEFAULT_VEL_BINS,
                 dur_bins:   int = DEFAULT_DUR_BINS):
        super().__init__()
        self.rol_emb   = nn.Embedding(n_roles, rol_emb_dim)
        in_dim         = feat_dim + rol_emb_dim
        self.backbone  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        h2 = hidden_dim // 2
        self.delta_head = nn.Linear(h2, delta_bins)
        self.pitch_head = nn.Linear(h2, pitch_bins)
        self.vel_head   = nn.Linear(h2, vel_bins)
        self.dur_head   = nn.Linear(h2, dur_bins)

        # Guardar config para serialización
        self.cfg = dict(feat_dim=feat_dim, hidden_dim=hidden_dim,
                        n_roles=n_roles, rol_emb_dim=rol_emb_dim,
                        delta_bins=delta_bins, pitch_bins=pitch_bins,
                        vel_bins=vel_bins, dur_bins=dur_bins)

    def forward(self, feat: torch.Tensor, rol_idx: torch.Tensor):
        emb  = self.rol_emb(rol_idx)
        x    = torch.cat([feat, emb], dim=-1)
        h    = self.backbone(x)
        return (self.delta_head(h),
                self.pitch_head(h),
                self.vel_head(h),
                self.dur_head(h))


class ExpressionNet(nn.Module):
    """
    M3 — Predice curvas CC1/CC11 desde la tensión y el instrumento.
    Entrada:  tensión escalar + instr embedding
    Salida:   [cc1, cc11] en [0,1] (se escala a 0-127 al usar)
    """
    def __init__(self, hidden_dim: int = 32,
                 n_instruments: int = N_INSTRUMENTS,
                 instr_emb_dim: int = 4):
        super().__init__()
        self.instr_emb = nn.Embedding(n_instruments, instr_emb_dim)
        in_dim         = 1 + instr_emb_dim
        self.net       = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.cfg = dict(hidden_dim=hidden_dim,
                        n_instruments=n_instruments,
                        instr_emb_dim=instr_emb_dim)

    def forward(self, tension: torch.Tensor, instr_idx: torch.Tensor):
        emb = self.instr_emb(instr_idx)
        x   = torch.cat([tension.unsqueeze(-1), emb], dim=-1)
        return self.net(x)  # [batch, 2] → cc1, cc11 en [0,1]


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 9 — DATASET Y ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class OrchestraDataset(Dataset):
    """
    Cada muestra: (feat_vector, archetype_idx, tension, rol_idx, instr_idx,
                   delta_bin, pitch_bin, vel_bin, dur_bin)
    Generada por build_dataset() desde MIDIs orquestales reales.
    """
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['feat'],       dtype=torch.float32),
            torch.tensor(s['arch_idx'],   dtype=torch.long),
            torch.tensor(s['tension'],    dtype=torch.float32),
            torch.tensor(s['rol_idx'],    dtype=torch.long),
            torch.tensor(s['instr_idx'],  dtype=torch.long),
            torch.tensor(s['delta_bin'],  dtype=torch.long),
            torch.tensor(s['pitch_bin'],  dtype=torch.long),
            torch.tensor(s['vel_bin'],    dtype=torch.long),
            torch.tensor(s['dur_bin'],    dtype=torch.long),
        )


def _classify_texture_heuristic(section: dict, tpb: int) -> int:
    """
    Heurístico de arquetipo para generar labels de entrenamiento desde
    secciones de orquesta real (donde no hay un label "ground truth" explícito).
    Devuelve índice 0-4 (A-E).
    """
    density  = section['density']
    mean_vel = section['mean_vel']
    mean_dur = section['mean_dur']
    notes    = section['notes']
    ns       = sorted(notes, key=lambda x: x['tick'])
    steps    = sum(1 for i in range(1, len(ns))
                   if abs(ns[i]['pitch'] - ns[i-1]['pitch']) <= 2)
    step_r   = steps / max(len(ns) - 1, 1)

    scores = [0.0] * 5
    scores[0] += density * 0.4 + (mean_vel / 127) * 0.4 + (1 - step_r) * 0.2
    scores[1] += (1 - density) * 0.3 + step_r * 0.4 + (mean_dur / (tpb * 2)) * 0.3
    scores[2] += density * 0.3 + (1 - mean_dur / (tpb * 2)) * 0.4 + step_r * 0.3
    scores[3] += density * 0.3 + (1 - step_r) * 0.4 + (1 - mean_vel / 127) * 0.3
    scores[4] += (1 - density) * 0.5 + (mean_dur / (tpb * 4)) * 0.3 + (1 - mean_vel / 127) * 0.2
    return int(np.argmax(scores))


def build_dataset(midi_dir: str, quantizer: EventQuantizer,
                  verbose: bool = False) -> OrchestraDataset:
    """
    Construye el dataset de entrenamiento desde un directorio de MIDIs orquestales.

    Para cada MIDI:
      1. Extrae todas las notas como orquesta original
      2. Aplica SyntheticReducer → piano reducido
      3. Segmenta el piano reducido en secciones
      4. Extrae features de cada sección
      5. Para cada nota de la orquesta original, la asigna a la sección
         correspondiente y genera una muestra de entrenamiento.
    """
    samples   = []
    midi_dir  = Path(midi_dir)
    midi_files= list(midi_dir.rglob('*.mid')) + list(midi_dir.rglob('*.midi'))

    if not midi_files:
        print(f"  ⚠ No se encontraron MIDIs en {midi_dir}")
        return OrchestraDataset([])

    print(f"  Procesando {len(midi_files)} MIDIs...")

    for midi_path in midi_files:
        try:
            mid, tpb, tempo = load_midi(str(midi_path))
            all_notes       = extract_notes_flat(mid)
            if len(all_notes) < 20:
                continue

            # Reducción sintética: orquesta → piano
            rh, lh   = reduce_orchestra_to_piano(all_notes, tpb)
            piano    = sorted(rh + lh, key=lambda n: n['tick'])
            if not piano:
                continue

            # Segmentar el piano reducido
            sections = segment_sections(piano, tpb)
            if not sections:
                continue

            total_sec = len(sections)

            for sec_idx, section in enumerate(sections):
                feat      = section_to_feature_vector(section, tpb, total_sec)
                arch_idx  = _classify_texture_heuristic(section, tpb)
                tension   = float(feat[30])  # último elemento del vector

                # Notas de orquesta que caen en esta sección
                ts, te    = section['tick_start'], section['tick_end']
                orch_notes= [n for n in all_notes if ts <= n['tick'] < te]
                if not orch_notes:
                    continue

                # Agrupar por instrumento aproximado (por pitch register)
                # En orquestales reales sin tracks etiquetados, usamos heurístico
                # de registro para asignar rol
                orch_notes.sort(key=lambda n: n['tick'])

                # Crear muestras: una por nota de orquesta en la sección
                prev_tick = ts
                for note in orch_notes[:64]:  # máx 64 eventos por sección
                    delta_tick = note['tick'] - prev_tick
                    prev_tick  = note['tick']

                    # Asignar instrumento/rol por registro (heurístico)
                    p = note['pitch']
                    if p >= 72:   instr_name = 'violin1'
                    elif p >= 60: instr_name = 'violin2'
                    elif p >= 48: instr_name = 'viola'
                    elif p >= 36: instr_name = 'cello'
                    else:         instr_name = 'contrabass'

                    rol_name  = INSTR_ROLE.get(instr_name, 'harmony')
                    rol_idx   = ROLE_IDX.get(rol_name, 0)
                    instr_idx = INSTR_IDX.get(instr_name, 0)

                    d_bin, p_bin, v_bin, dur_bin = quantizer.encode(
                        delta_tick, note['pitch'],
                        note['velocity'], note['duration']
                    )

                    samples.append({
                        'feat':      feat,
                        'arch_idx':  arch_idx,
                        'tension':   tension,
                        'rol_idx':   rol_idx,
                        'instr_idx': instr_idx,
                        'delta_bin': d_bin,
                        'pitch_bin': p_bin,
                        'vel_bin':   v_bin,
                        'dur_bin':   dur_bin,
                    })

            if verbose:
                print(f"    {midi_path.name}: {len(sections)} secciones → {len(samples)} muestras acumuladas")

        except Exception as e:
            if verbose:
                print(f"    ⚠ {midi_path.name}: {e}")
            continue

    print(f"  Dataset: {len(samples)} muestras desde {len(midi_files)} MIDIs")
    return OrchestraDataset(samples)


def train(midi_dir: str, checkpoint_dir: str, quantizer: EventQuantizer,
          hidden_dim: int = DEFAULT_HIDDEN_DIM,
          epochs: int = DEFAULT_EPOCHS,
          lr: float = DEFAULT_LR,
          batch_size: int = DEFAULT_BATCH_SIZE,
          verbose: bool = False):
    """
    Loop de entrenamiento unificado para M1, M2 y M3.
    Guarda m1.pt, m2.pt, m3.pt en checkpoint_dir.
    """
    if not TORCH_OK:
        print("ERROR: pip install torch"); sys.exit(1)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = build_dataset(midi_dir, quantizer, verbose=verbose)
    if len(dataset) == 0:
        print("ERROR: dataset vacío. Verifica que el directorio contiene MIDIs.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

    # ── Instanciar modelos ───────────────────────────────────────────────────
    d_bins, p_bins, v_bins, dur_bins = quantizer.total_bins()

    m1 = TextureNet(FEAT_DIM, hidden_dim, N_ARCHETYPES)
    m2 = InstrumentNet(FEAT_DIM, hidden_dim, N_ROLES, 8,
                       d_bins, p_bins, v_bins, dur_bins)
    m3 = ExpressionNet(32, N_INSTRUMENTS, 4)

    opt1 = optim.Adam(m1.parameters(), lr=lr)
    opt2 = optim.Adam(m2.parameters(), lr=lr)
    opt3 = optim.Adam(m3.parameters(), lr=lr)

    ce   = nn.CrossEntropyLoss()
    mse  = nn.MSELoss()

    print(f"\n  Entrenando M1/M2/M3  |  epochs={epochs}  lr={lr}  batch={batch_size}")
    print(f"  Muestras: {len(dataset)}  |  Pasos por epoch: {len(loader)}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print()

    for epoch in range(1, epochs + 1):
        m1.train(); m2.train(); m3.train()
        total_l1 = total_l2 = total_l3 = 0.0

        for batch in loader:
            (feat, arch_idx, tension, rol_idx, instr_idx,
             d_bin, p_bin, v_bin, dur_bin) = batch

            # ── M1: TextureNet ───────────────────────────────────────────────
            opt1.zero_grad()
            arch_logits, pred_tension = m1(feat)
            loss1 = ce(arch_logits, arch_idx) + mse(pred_tension, tension)
            loss1.backward()
            opt1.step()
            total_l1 += loss1.item()

            # ── M2: InstrumentNet ────────────────────────────────────────────
            opt2.zero_grad()
            d_logits, p_logits, v_logits, dur_logits = m2(feat, rol_idx)
            loss2 = (ce(d_logits,   d_bin)   +
                     ce(p_logits,   p_bin)   +
                     ce(v_logits,   v_bin)   +
                     ce(dur_logits, dur_bin))
            loss2.backward()
            opt2.step()
            total_l2 += loss2.item()

            # ── M3: ExpressionNet ────────────────────────────────────────────
            opt3.zero_grad()
            pred_cc = m3(tension, instr_idx)   # [batch, 2]
            # Target: cc1/cc11 desde tensión (supervisión débil)
            cc1_target  = (tension * 0.8 + 0.1).unsqueeze(-1)
            cc11_target = (tension * 0.6 + 0.2).unsqueeze(-1)
            target_cc   = torch.cat([cc1_target, cc11_target], dim=-1)
            loss3       = mse(pred_cc, target_cc)
            loss3.backward()
            opt3.step()
            total_l3 += loss3.item()

        n = len(loader)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"L1(texture)={total_l1/n:.4f}  "
                  f"L2(instr)={total_l2/n:.4f}  "
                  f"L3(expr)={total_l3/n:.4f}")

    # ── Guardar checkpoints ──────────────────────────────────────────────────
    cfg_quantizer = {'delta_bins': d_bins, 'pitch_bins': p_bins,
                     'vel_bins': v_bins, 'dur_bins': dur_bins,
                     'tpb': quantizer.tpb}

    torch.save({'model_state': m1.state_dict(),
                'config': {'feat_dim': FEAT_DIM, 'hidden_dim': hidden_dim,
                           'n_archetypes': N_ARCHETYPES}},
               Path(checkpoint_dir) / 'm1.pt')

    torch.save({'model_state': m2.state_dict(),
                'config': {**m2.cfg, **cfg_quantizer}},
               Path(checkpoint_dir) / 'm2.pt')

    torch.save({'model_state': m3.state_dict(),
                'config': m3.cfg},
               Path(checkpoint_dir) / 'm3.pt')

    print(f"\n  Checkpoints guardados en {checkpoint_dir}/")
    print(f"    m1.pt  (TextureNet)     — arquetipo + tensión")
    print(f"    m2.pt  (InstrumentNet)  — generación de eventos")
    print(f"    m3.pt  (ExpressionNet)  — CC1/CC11")

    return m1, m2, m3


def load_checkpoints(checkpoint_dir: str):
    """
    Carga m1.pt, m2.pt, m3.pt desde checkpoint_dir.
    Reconstruye los modelos con la arquitectura guardada en cada checkpoint.
    """
    if not TORCH_OK:
        print("ERROR: pip install torch"); sys.exit(1)

    cdir = Path(checkpoint_dir)
    for fname in ('m1.pt', 'm2.pt', 'm3.pt'):
        if not (cdir / fname).exists():
            print(f"\nERROR: No se encontró {cdir / fname}")
            print("\nPara entrenar el modelo primero ejecuta:")
            print(f"  python ml_expander.py train <directorio_midis> "
                  f"--checkpoint-dir {checkpoint_dir}")
            print("\nDescarga MIDIs orquestales desde:")
            print("  https://magenta.tensorflow.org/datasets/maestro")
            sys.exit(1)

    ck1 = torch.load(cdir / 'm1.pt', map_location='cpu')
    ck2 = torch.load(cdir / 'm2.pt', map_location='cpu')
    ck3 = torch.load(cdir / 'm3.pt', map_location='cpu')

    m1  = TextureNet(**{k: v for k, v in ck1['config'].items()
                        if k in ('feat_dim', 'hidden_dim', 'n_archetypes')})
    m1.load_state_dict(ck1['model_state'])
    m1.eval()

    m2cfg  = ck2['config']
    m2     = InstrumentNet(
        feat_dim  = m2cfg.get('feat_dim',   FEAT_DIM),
        hidden_dim= m2cfg.get('hidden_dim', DEFAULT_HIDDEN_DIM),
        n_roles   = m2cfg.get('n_roles',    N_ROLES),
        rol_emb_dim=m2cfg.get('rol_emb_dim',8),
        delta_bins= m2cfg.get('delta_bins', DEFAULT_DELTA_BINS),
        pitch_bins= m2cfg.get('pitch_bins', DEFAULT_PITCH_BINS),
        vel_bins  = m2cfg.get('vel_bins',   DEFAULT_VEL_BINS),
        dur_bins  = m2cfg.get('dur_bins',   DEFAULT_DUR_BINS),
    )
    m2.load_state_dict(ck2['model_state'])
    m2.eval()

    # Reconstruir quantizer desde config guardada en m2
    quantizer = EventQuantizer(
        delta_bins= m2cfg.get('delta_bins', DEFAULT_DELTA_BINS),
        pitch_bins= m2cfg.get('pitch_bins', DEFAULT_PITCH_BINS),
        vel_bins  = m2cfg.get('vel_bins',   DEFAULT_VEL_BINS),
        dur_bins  = m2cfg.get('dur_bins',   DEFAULT_DUR_BINS),
        tpb       = m2cfg.get('tpb',        480),
    )

    m3 = ExpressionNet(**ck3['config'])
    m3.load_state_dict(ck3['model_state'])
    m3.eval()

    return m1, m2, m3, quantizer


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10 — IDIOMATIC CORRECTOR
#  Corrección post-ML de las notas generadas por M2.
#  Combina: melody_harmonizer.py + counterpoint.py + orchestrator.py
# ══════════════════════════════════════════════════════════════════════════════

def _motion_type(prev_mel: int, prev_cp: int, curr_mel: int, curr_cp: int) -> str:
    """Tipo de movimiento entre dos pares de voces. De counterpoint.py."""
    mel_dir = curr_mel - prev_mel
    cp_dir  = curr_cp  - prev_cp
    if mel_dir == 0 and cp_dir == 0:   return 'static'
    if mel_dir == 0 or cp_dir == 0:    return 'oblique'
    if (mel_dir > 0) == (cp_dir > 0):
        if (curr_mel - curr_cp) == (prev_mel - prev_cp): return 'parallel'
        return 'similar'
    return 'contrary'


def _best_candidate(candidates: List[int], mel_p: int, prev_cp: Optional[int],
                    prev_mel: Optional[int],
                    chord_pcs: List[int],
                    voice_position: str = 'below') -> int:
    """
    Puntúa candidatos de nota y elige el mejor aplicando reglas de contrapunto.
    Adaptado de _best_candidate en counterpoint.py.
    """
    best_cp, best_score = None, -9999
    for cp in candidates:
        iv    = abs(mel_p - cp) % 12
        score = 0

        # Consonancia
        if iv in IMPERFECT_CONSONANCES:  score += 4
        elif iv in PERFECT_CONSONANCES:  score += 2
        else:                            score -= 3

        # Tono de acorde
        if pc(cp) in chord_pcs:          score += 2

        # Movimiento respecto a voz anterior
        if prev_cp is not None and prev_mel is not None:
            motion = _motion_type(prev_mel, prev_cp, mel_p, cp)
            if motion == 'contrary':     score += 3
            elif motion == 'oblique':    score += 1
            elif motion == 'parallel':
                if iv in PERFECT_CONSONANCES: score -= 5
                else:                          score -= 1

            jump = abs(cp - prev_cp)
            if jump > 12:   score -= 4
            elif jump > 7:  score -= 2
            elif jump <= 2: score += 1

            # Evitar unísono consecutivo
            if iv == 0 and abs(mel_p - prev_cp) % 12 == 0: score -= 3

        # Posición relativa
        if voice_position == 'below' and cp < mel_p:  score += 1
        elif voice_position == 'above' and cp > mel_p: score += 1
        if voice_position == 'below' and cp >= mel_p:  score -= 6
        elif voice_position == 'above' and cp <= mel_p: score -= 6

        if score > best_score:
            best_score = score; best_cp = cp

    return best_cp if best_cp is not None else (candidates[0] if candidates else mel_p)


def classify_articulation(pitch: int, dur_ticks: int, velocity: int,
                           prev_end_tick: Optional[int], curr_tick: int,
                           tension: float, family: str, role: str,
                           tpb: int) -> str:
    """
    Clasifica articulación apropiada para una nota generada.
    Adaptado de classify_articulation en orchestrator.py.
    """
    dur_beats = ticks_to_beats(dur_ticks, tpb)
    gap_beats = ticks_to_beats(curr_tick - prev_end_tick, tpb) if prev_end_tick else 0.0
    high_ten  = tension > 0.65
    low_ten   = tension < 0.30

    if family == 'strings':
        if role in ('melody', 'melody_high') and dur_beats >= ART_THRESHOLDS['legato'] and gap_beats < 0.25:
            return 'legato'
        if role in ('melody', 'melody_high') and dur_beats >= ART_THRESHOLDS['sustain']:
            return 'sustain'
        if high_ten and dur_beats < ART_THRESHOLDS['portato']:
            return 'spiccato'
        if role in ('harmony', 'inner') and low_ten and dur_beats >= 0.4:
            return 'pizzicato'
        if role in ('harmony', 'inner') and high_ten and dur_beats >= 1.0:
            return 'tremolo'
        if dur_beats >= ART_THRESHOLDS['portato']:
            return 'portato'
        return 'spiccato' if dur_beats < ART_THRESHOLDS['spiccato'] else 'sustain'

    elif family == 'woodwind':
        if dur_beats >= ART_THRESHOLDS['legato'] and gap_beats < 0.2: return 'legato'
        if dur_beats >= ART_THRESHOLDS['sustain']:                     return 'sustain'
        if dur_beats >= ART_THRESHOLDS['portato']:                     return 'portato'
        return 'staccato'

    elif family == 'brass':
        if dur_beats >= ART_THRESHOLDS['legato']:
            return 'marcato_l' if high_ten and dur_beats >= 2.0 else (
                   'marcato_s' if high_ten else 'legato')
        if dur_beats >= ART_THRESHOLDS['sustain']: return 'sustain'
        if dur_beats >= ART_THRESHOLDS['portato']: return 'portato'
        return 'staccato'

    return 'sustain'


def correct_note(pitch: int, instr_name: str, chord_pcs: List[int],
                 scale: List[int], prev_pitch: Optional[int],
                 prev_mel_pitch: Optional[int],
                 tension: float) -> Optional[int]:
    """
    Aplica corrección idiomática completa a una nota generada por M2:
    1. Ajusta al rango idiomático del instrumento (RANGES/SWEET)
    2. Verifica que el PC sea admisible (chord_tone o tension, no avoid)
    3. Aplica reglas de contrapunto si hay contexto de voz anterior
    Devuelve el pitch corregido, o None si no hay solución válida.
    """
    lo, hi     = RANGES.get(instr_name, (21, 108))
    sw_lo, sw_hi = SWEET.get(instr_name, (lo + 5, hi - 5))

    # 1. Ajustar al rango por octavas, priorizando el sweet spot
    p = fit_range(pitch, lo, hi)
    if p is None:
        # Forzar al sweet spot más cercano
        p = clamp(pitch, lo, hi)

    # 2. PC admisible: inferir calidad del acorde desde los PCs disponibles
    quality = 'm' if len(chord_pcs) >= 2 and (chord_pcs[1] - chord_pcs[0]) % 12 == 3 else 'M'
    root_pc = chord_pcs[0] if chord_pcs else 0
    adm     = admissible_pcs(root_pc, quality, scale)

    if pc(p) not in adm:
        # Buscar el PC admisible más cercano manteniendo la octava
        candidates = [cp + (p // 12) * 12 for cp in adm]
        candidates += [cp + (p // 12 + 1) * 12 for cp in adm]
        candidates += [cp + (p // 12 - 1) * 12 for cp in adm]
        candidates = [c for c in candidates if lo <= c <= hi]
        if candidates:
            p = min(candidates, key=lambda c: abs(c - p))

    # 3. Reglas de contrapunto si hay contexto
    if prev_pitch is not None and prev_mel_pitch is not None:
        role = INSTR_ROLE.get(instr_name, 'harmony')
        vpos = 'below' if role in ('bass_melodic', 'bass_root', 'pad_low') else 'auto'
        # Generar candidatos locales y puntuar
        local_candidates = [p]
        for delta in [-2, -1, 1, 2, -12, 12]:
            c = p + delta
            if lo <= c <= hi and pc(c) in adm:
                local_candidates.append(c)
        if local_candidates:
            p = _best_candidate(local_candidates, prev_mel_pitch,
                                prev_pitch, prev_mel_pitch,
                                chord_pcs, vpos)

    return p if lo <= p <= hi else None


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 11 — PIPELINE DE INFERENCIA
#
#  Arquitectura de tres capas:
#  [1] melody / melody_high  →  melodía directa del piano (reconocible)
#  [2] acompañamiento        →  FigurationEngine genera patrón rítmico
#                                por rol + arquetipo; M2 elige pitches
#  [3] contraste             →  instrumentación varía por tensión y arquetipo;
#                                arquetipo por heurístico si M1 poco entrenado
# ══════════════════════════════════════════════════════════════════════════════

def _extract_melody_line(notes: List[dict], tpb: int) -> List[dict]:
    """Nota más aguda en cada grupo simultáneo → línea melódica del piano."""
    if not notes:
        return []
    slack = tpb // 8
    sn    = sorted(notes, key=lambda n: n['tick'])
    mel   = []
    i     = 0
    while i < len(sn):
        g = [sn[i]]; j = i + 1
        while j < len(sn) and sn[j]['tick'] - sn[i]['tick'] <= slack:
            g.append(sn[j]); j += 1
        mel.append(max(g, key=lambda n: n['pitch']))
        i = j
    return mel


def _assign_melody_to_instrument(melody: List[dict], instr_name: str,
                                  octave_shift: int = 0) -> List[dict]:
    """Transpone la melodía al rango idiomático del instrumento."""
    lo, hi = RANGES.get(instr_name, (21, 108))
    result = []
    for n in melody:
        p = fit_range(n['pitch'] + octave_shift * 12, lo, hi)
        if p is not None:
            result.append({**n, 'pitch': p, 'channel': 0})
    return result


def _classify_texture_heuristic_section(section: dict, tpb: int) -> int:
    """
    Fallback heurístico de arquetipo cuando M1 no está bien entrenado.
    Reproduces la lógica de classify_texture del piano_expander original.
    Devuelve índice 0-4 (A-E).
    """
    density  = section['density']
    mean_vel = section['mean_vel']
    mean_dur = section['mean_dur']
    notes    = section['notes']
    ns       = sorted(notes, key=lambda x: x['tick'])
    steps    = sum(1 for i in range(1, len(ns))
                   if abs(ns[i]['pitch'] - ns[i-1]['pitch']) <= 2)
    step_r   = steps / max(len(ns) - 1, 1)
    scores   = [0.0] * 5
    scores[0] += density * 0.4 + (mean_vel / 127) * 0.4 + (1 - step_r) * 0.2
    scores[1] += (1 - density) * 0.3 + step_r * 0.4 + (mean_dur / (tpb * 2)) * 0.3
    scores[2] += density * 0.3 + (1 - mean_dur / (tpb * 2)) * 0.4 + step_r * 0.3
    scores[3] += density * 0.3 + (1 - step_r) * 0.4 + (1 - mean_vel / 127) * 0.3
    scores[4] += (1 - density) * 0.5 + (mean_dur / (tpb * 4)) * 0.3 + (1 - mean_vel / 127) * 0.2
    return int(np.argmax(scores))


def _active_instruments_for_section(instruments: List[str], tension: float,
                                     archetype: str, sec_idx: int,
                                     total_sec: int) -> List[str]:
    """
    Contraste entre secciones: decide qué instrumentos suenan según tensión
    y arquetipo. Garantiza que la instrumentación cambia visiblemente entre
    secciones consecutivas.

    Reglas:
    · Cuerdas: siempre activas (threshold 0.0)
    · Maderas:  entran cuando tensión > 0.20 O posición > 1/3 de la obra
    · Metales:  entran cuando tensión > 0.45 O posición > 2/3 de la obra
    · Arquetipo E (sparse): solo los 3 instrumentos más graves de la plantilla
    """
    pos = sec_idx / max(total_sec - 1, 1)
    active = []
    for instr in instruments:
        family   = FAMILY.get(instr, 'strings')
        rol_name = INSTR_ROLE.get(instr, 'harmony')

        # Roles melódicos siempre entran (la melodía no puede desaparecer)
        if rol_name in ('melody', 'melody_high'):
            active.append(instr); continue

        # Metales
        if instr in ('tuba', 'trombone') and tension < 0.55 and pos < 0.60:
            continue
        if instr == 'trumpet' and tension < 0.40 and pos < 0.55:
            continue
        if instr == 'horn' and tension < 0.30 and pos < 0.45:
            continue

        # Maderas: entran progresivamente
        if family == 'woodwind':
            if tension < 0.20 and pos < 0.33:
                continue

        # Arquetipo E (sparse): solo cuerdas graves + 1 madera máximo
        if archetype == 'E':
            lo, _ = RANGES.get(instr, (60, 96))
            if lo > 55 and family != 'woodwind':
                continue
            if family == 'woodwind' and instr not in ('oboe', 'clarinet'):
                continue

        active.append(instr)
    return active


# ── FigurationEngine ──────────────────────────────────────────────────────────
# Genera el esqueleto rítmico de cada rol según el arquetipo de textura.
# M2 luego elige los pitches dentro de este esqueleto.

def _fig_bass_root(ts: int, te: int, tpb: int,
                   chord_pcs: List[int], scale: List[int],
                   instr_name: str, tension: float,
                   mean_vel: float) -> List[dict]:
    """
    Bajo raíz: ataca en tiempos 1 y 3. Más activo en tensión alta.
    Adaptado de gen_bass_root (piano_expander.py).
    """
    bar  = tpb * 4
    lo, hi = RANGES.get(instr_name, (28, 62))
    notes  = []
    root_pc = chord_pcs[0] if chord_pcs else 0
    # Nota base: raíz del acorde en registro grave
    base_p = root_pc + 36  # C2 + root_pc
    p = fit_range(base_p, lo, hi)
    if p is None:
        p = lo

    t = ts
    while t < te:
        beat_in_bar = (t - ts) % bar
        on_1 = beat_in_bar < tpb // 4
        on_3 = abs(beat_in_bar - tpb * 2) < tpb // 4
        if on_1 or on_3:
            dur = tpb * 2 if tension < 0.5 else tpb
            vel = int(clamp(mean_vel * 0.85 + tension * 10, 30, 100))
            notes.append({'tick': t, 'pitch': p, 'velocity': vel,
                          'duration': min(dur, te - t - 10), 'channel': 0})
            t += tpb  # avanzar 1 beat tras cada ataque
        else:
            t += tpb // 2  # avanzar media negra buscando siguiente tiempo
    return notes


def _fig_bass_melodic(ts: int, te: int, tpb: int,
                      chord_pcs: List[int], scale: List[int],
                      instr_name: str, tension: float,
                      mean_vel: float, melody_ref: List[dict]) -> List[dict]:
    """
    Bajo melódico: sigue el contorno del bajo del piano con notas de paso.
    Adaptado de gen_bass_melodic (piano_expander.py).
    """
    lo, hi = RANGES.get(instr_name, (36, 76))
    notes  = []
    # Usar la nota más grave de cada compás como referencia de bajo
    bar    = tpb * 4
    cur    = ts
    prev_p = None
    while cur < te:
        bar_end  = min(cur + bar, te)
        bar_mel  = [n for n in melody_ref if cur <= n['tick'] < bar_end]
        if bar_mel:
            ref_p = min(bar_mel, key=lambda n: n['pitch'])['pitch']
        else:
            ref_p = chord_pcs[0] + 48 if chord_pcs else 48
        p = fit_range(ref_p, lo, hi)
        if p is None:
            p = lo
        vel = int(clamp(mean_vel * 0.80 + tension * 12, 25, 100))
        dur = tpb * 2
        notes.append({'tick': cur, 'pitch': p, 'velocity': vel,
                      'duration': min(dur, bar_end - cur - 10), 'channel': 0})
        # Nota de paso si hay salto
        if prev_p is not None and abs(p - prev_p) > 4 and (bar_end - cur) > tpb * 3:
            mid_p = (p + prev_p) // 2
            mid_p = snap_to_scale(mid_p, scale)
            mid_p_fit = fit_range(mid_p, lo, hi)
            if mid_p_fit:
                notes.append({'tick': cur + tpb, 'pitch': mid_p_fit,
                              'velocity': max(vel - 15, 20),
                              'duration': tpb - 20, 'channel': 0})
        prev_p = p
        cur    = bar_end
    return notes


def _fig_harmony(ts: int, te: int, tpb: int,
                 chord_pcs: List[int], scale: List[int],
                 instr_name: str, archetype: str,
                 tension: float, mean_vel: float) -> List[dict]:
    """
    Voces interiores con figuración dependiente del arquetipo:
    A → negras sincronizadas (tutti homofónico)
    B → blancas (colchón armónico)
    C → arpegios de tresillo (movimiento activo)
    D → negras con pequeño desplazamiento de fase (contrapunto)
    E → redondas (sparse)
    Adaptado de gen_harmony (piano_expander.py).
    """
    lo, hi = SWEET.get(instr_name, RANGES.get(instr_name, (48, 84)))
    fam    = FAMILY.get(instr_name, 'strings')
    notes  = []

    # Pitch de referencia: tercera del acorde en registro medio
    root_pc = chord_pcs[0] if chord_pcs else 0
    third_pc= chord_pcs[1] if len(chord_pcs) > 1 else (root_pc + 4) % 12
    base_p  = third_pc + 60  # C4 + intervalo
    p       = fit_range(base_p, lo, hi)
    if p is None:
        p = nearest_chord_tone(lo + (hi - lo) // 2, chord_pcs)
        p = fit_range(p, lo, hi) or (lo + hi) // 2

    vel_base = int(clamp(mean_vel * 0.88 + tension * 8, 25, 110))
    # Pequeño desplazamiento de fase para arquetipo D
    phase = tpb // 4 if archetype == 'D' else 0

    t = ts + phase
    while t < te:
        if archetype in ('A', 'D'):
            # Negras
            dur = int(tpb * 0.92)
            notes.append({'tick': t, 'pitch': p, 'velocity': vel_base,
                          'duration': min(dur, te - t - 10), 'channel': 0})
            t += tpb
        elif archetype == 'B':
            # Blancas
            dur = tpb * 2
            notes.append({'tick': t, 'pitch': p, 'velocity': max(vel_base - 10, 20),
                          'duration': min(dur, te - t - 10), 'channel': 0})
            t += tpb * 2
        elif archetype == 'C':
            # Arpegios de tresillo: raíz → 5ª → raíz
            tresillo = tpb // 3
            fifth_pc = chord_pcs[2] if len(chord_pcs) > 2 else (root_pc + 7) % 12
            p5       = nearest_chord_tone(p + 7, chord_pcs)
            p5       = fit_range(p5, lo, hi) or p
            for pp in [p, p5, p]:
                if t >= te: break
                notes.append({'tick': t, 'pitch': pp,
                              'velocity': vel_base,
                              'duration': min(int(tresillo * 0.85), te - t - 5),
                              'channel': 0})
                t += tresillo
        else:  # E
            # Redondas
            dur = tpb * 4
            notes.append({'tick': t, 'pitch': p,
                          'velocity': max(vel_base - 20, 18),
                          'duration': min(dur, te - t - 10), 'channel': 0})
            t += tpb * 4
    return notes


def _fig_pad(ts: int, te: int, tpb: int,
             chord_pcs: List[int], scale: List[int],
             instr_name: str, tension: float,
             mean_vel: float) -> List[dict]:
    """
    Pad (metales en sustain): una nota larga por sección o por cada 4 compases.
    Adaptado de gen_pad (piano_expander.py).
    """
    lo, hi = RANGES.get(instr_name, (34, 77))
    notes  = []
    root_pc = chord_pcs[0] if chord_pcs else 0
    mid_pc  = chord_pcs[1] if len(chord_pcs) > 1 else (root_pc + 4) % 12
    base_p  = mid_pc + 48
    p       = fit_range(base_p, lo, hi)
    if p is None:
        p = nearest_chord_tone(lo + (hi - lo) // 2, chord_pcs)
        p = fit_range(p, lo, hi) or lo

    vel = int(clamp(mean_vel * 0.90 * tension, 20, 95))
    bar = tpb * 4
    t   = ts
    while t < te:
        block_end = min(t + bar * 4, te)
        dur       = block_end - t - 10
        notes.append({'tick': t, 'pitch': p, 'velocity': vel,
                      'duration': max(dur, 30), 'channel': 0})
        t = block_end
    return notes


def _fig_counter(ts: int, te: int, tpb: int,
                 chord_pcs: List[int], scale: List[int],
                 instr_name: str, tension: float,
                 mean_vel: float, melody_ref: List[dict]) -> List[dict]:
    """
    Contrapunto real: movimiento contrario a la melodía, consonante con el acorde.
    Versión simplificada de gen_counterpoint (piano_expander.py).
    """
    lo, hi  = SWEET.get(instr_name, RANGES.get(instr_name, (48, 84)))
    notes   = []
    prev_p  = None
    prev_mel= None
    sec_mel = [n for n in melody_ref if ts <= n['tick'] < te]
    if not sec_mel:
        return _fig_harmony(ts, te, tpb, chord_pcs, scale, instr_name, 'B',
                            tension, mean_vel)

    for mn in sec_mel:
        mel_p = mn['pitch']
        # Intervalo preferido: 3ª o 6ª por debajo
        for interval in [9, 8, 4, 3, 7, 12]:
            p = mel_p - interval
            p = fit_range(p, lo, hi)
            if p is None:
                continue
            if pc(p) not in chord_pcs and pc(p) not in scale:
                p = nearest_chord_tone(p, chord_pcs)
                p = fit_range(p, lo, hi)
            if p is None or not is_consonant(p, mel_p):
                continue
            if p >= mel_p:
                continue
            # Evitar paralelas de 5ª/8ª
            if prev_p is not None and prev_mel is not None:
                prev_iv = abs(prev_mel - prev_p) % 12
                curr_iv = abs(mel_p - p) % 12
                if prev_iv in (7, 0) and curr_iv == prev_iv:
                    continue
            if prev_p is not None and abs(p - prev_p) > 9:
                continue
            vel = int(clamp(mn['velocity'] * 0.72, 22, 90))
            notes.append({'tick': mn['tick'], 'pitch': p, 'velocity': vel,
                          'duration': mn['duration'], 'channel': 0})
            prev_p   = p
            prev_mel = mel_p
            break

    return notes


def _apply_m2_pitch(raw_notes: List[dict], feat_t: 'torch.Tensor',
                    rol_t: 'torch.Tensor', m2: 'InstrumentNet',
                    quantizer: 'EventQuantizer',
                    instr_name: str, chord_pcs: List[int],
                    scale: List[int], tension: float) -> List[dict]:
    """
    Sustituye el pitch de cada nota del patrón de figuración por el pitch
    que propone M2, corregido idiomáticamente.
    Si M2 propone un pitch inválido, se conserva el pitch del patrón.
    """
    result = []
    prev_pitch = None
    for n in raw_notes:
        with torch.no_grad():
            _, p_l, v_l, _ = m2(feat_t, rol_t)

        # Temperatura adaptativa: alta tensión → más determinista
        temp = 1.5 - tension * 0.7
        probs_p = torch.softmax(p_l / max(temp, 0.2), dim=-1)
        p_bin   = int(torch.multinomial(probs_p, 1).item())
        probs_v = torch.softmax(v_l / 1.0, dim=-1)
        v_bin   = int(torch.multinomial(probs_v, 1).item())

        _, raw_pitch, velocity, _ = quantizer.decode(p_bin, p_bin, v_bin, 0)
        # decode usa p_bin en posición de pitch
        raw_pitch = clamp(p_bin + quantizer.pitch_min,
                          quantizer.pitch_min, quantizer.pitch_max)

        corrected = correct_note(raw_pitch, instr_name, chord_pcs, scale,
                                 prev_pitch, n['pitch'], tension)
        if corrected is None:
            corrected = n['pitch']  # fallback: pitch del patrón

        result.append({**n, 'pitch': corrected,
                        'velocity': clamp(velocity, 20, 120)})
        prev_pitch = corrected
    return result


def expand_with_ml(piano_notes: List[dict], tpb: int, tempo: int,
                   instruments: List[str],
                   m1: 'TextureNet', m2: 'InstrumentNet', m3: 'ExpressionNet',
                   quantizer: 'EventQuantizer',
                   verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Pipeline de inferencia con FigurationEngine y contraste entre secciones.

    Tres capas:
    [1] melody / melody_high  →  melodía directa del piano (reconocible)
    [2] acompañamiento        →  FigurationEngine por rol+arquetipo, M2 elige pitch
    [3] contraste             →  instrumentación varía por tensión, arquetipo y
                                  posición en la obra; arquetipo por heurístico
                                  si M1 tiene baja confianza
    """
    sections = segment_sections(piano_notes, tpb)
    if not sections:
        print("  ⚠ No se detectaron secciones en el boceto.")
        return {}, {}

    total_sec  = len(sections)
    tonic, mode= detect_key(piano_notes)
    scale      = scale_pcs(tonic, mode)
    melody_global = _extract_melody_line(piano_notes, tpb)

    print(f"  Tonalidad  : {NOTE_NAMES[tonic]} {mode}")
    print(f"  Secciones  : {total_sec}")

    instr_notes: Dict[str, List[dict]] = {i: [] for i in instruments}
    instr_cc:    Dict[str, List[tuple]]= {i: [] for i in instruments}

    with torch.no_grad():
        for sec_idx, section in enumerate(sections):
            feat    = section_to_feature_vector(section, tpb, total_sec)
            feat_t  = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

            # M1: arquetipo y tensión
            arch_logits, tension_t = m1(feat_t)
            arch_probs = torch.softmax(arch_logits, dim=-1)
            arch_conf  = float(arch_probs.max().item())
            arch_idx   = int(torch.argmax(arch_logits, dim=-1).item())
            tension    = float(tension_t.item())

            # Fallback heurístico si M1 tiene baja confianza (< 0.4)
            # — ocurre cuando el modelo no está bien entrenado con datos reales
            if arch_conf < 0.40:
                arch_idx = _classify_texture_heuristic_section(section, tpb)

            archetype  = ARCHETYPES[arch_idx]
            chord_pcs  = infer_chord(section['notes'])
            ts         = section['tick_start']
            te         = section['tick_end']
            mean_vel   = section['mean_vel']
            sec_melody = [n for n in melody_global if ts <= n['tick'] < te]

            if verbose:
                conf_str = f"M1({arch_conf:.2f})" if arch_conf >= 0.40 else "heurístico"
                print(f"  Sec {sec_idx+1:>2}  [{archetype}] {conf_str}  "
                      f"tensión={tension:.2f}  "
                      f"cc {section['bar_start']}-{section['bar_end']}")

            # Contraste: decidir qué instrumentos suenan en esta sección
            active = _active_instruments_for_section(
                instruments, tension, archetype, sec_idx, total_sec)

            for instr_name in active:
                family   = FAMILY.get(instr_name, 'strings')
                rol_name = INSTR_ROLE.get(instr_name, 'harmony')
                rol_idx  = ROLE_IDX.get(rol_name, 0)
                rol_t    = torch.tensor([rol_idx], dtype=torch.long)

                # ── Capa 1: melodía directa del piano ─────────────────────────
                if rol_name in ('melody', 'melody_high'):
                    shift    = 1 if rol_name == 'melody_high' else 0
                    assigned = _assign_melody_to_instrument(
                        sec_melody, instr_name, shift)
                    instr_notes[instr_name].extend(assigned)
                    continue

                # ── Capa 2: FigurationEngine → patrón rítmico por rol ─────────
                if rol_name in ('bass_root',):
                    raw = _fig_bass_root(ts, te, tpb, chord_pcs, scale,
                                         instr_name, tension, mean_vel)
                elif rol_name == 'bass_melodic':
                    raw = _fig_bass_melodic(ts, te, tpb, chord_pcs, scale,
                                            instr_name, tension, mean_vel,
                                            piano_notes)
                elif rol_name in ('harmony', 'inner'):
                    raw = _fig_harmony(ts, te, tpb, chord_pcs, scale,
                                       instr_name, archetype, tension, mean_vel)
                elif rol_name == 'counter':
                    raw = _fig_counter(ts, te, tpb, chord_pcs, scale,
                                       instr_name, tension, mean_vel,
                                       melody_global)
                elif rol_name in ('pad', 'pad_low'):
                    raw = _fig_pad(ts, te, tpb, chord_pcs, scale,
                                   instr_name, tension, mean_vel)
                else:
                    raw = _fig_harmony(ts, te, tpb, chord_pcs, scale,
                                       instr_name, 'B', tension, mean_vel)

                if not raw:
                    continue

                # ── Capa 3: M2 sustituye pitches del patrón ───────────────────
                voiced = _apply_m2_pitch(raw, feat_t, rol_t, m2, quantizer,
                                         instr_name, chord_pcs, scale, tension)
                instr_notes[instr_name].extend(voiced)

                # M3: CC envelope
                instr_idx_t = torch.tensor([INSTR_IDX.get(instr_name, 0)],
                                            dtype=torch.long)
                tens_t      = torch.tensor([tension], dtype=torch.float32)
                cc_pred     = m3(tens_t, instr_idx_t).squeeze(0)
                cc1_val     = int(clamp(float(cc_pred[0]) * 127, 1, 127))
                cc11_val    = int(clamp(float(cc_pred[1]) * 127, 1, 127))
                sec_len     = te - ts
                n_points    = max(2, sec_len // (tpb * 8))
                for pt in range(n_points):
                    t_pt  = ts + pt * sec_len // n_points
                    alpha = pt / max(n_points - 1, 1)
                    swell = math.sin(math.pi * alpha) * 10
                    instr_cc[instr_name].append(
                        (t_pt, 'cc', 1,  clamp(cc1_val  + int(swell), 1, 127)))
                    instr_cc[instr_name].append(
                        (t_pt, 'cc', 11, clamp(cc11_val + int(swell * 0.8), 1, 127)))

    # Ordenar y reportar
    for instr_name in instruments:
        instr_notes[instr_name].sort(key=lambda n: (n['tick'], n['pitch']))
        n    = len(instr_notes[instr_name])
        role = INSTR_ROLE.get(instr_name, '?')
        disp = INSTR_DISPLAY.get(instr_name, instr_name)
        if verbose or n > 0:
            print(f"  [{disp:<16}] {n:>5} notas  rol={role}")

    return instr_notes, instr_cc


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 12 — CONSTRUCCIÓN MIDI DE SALIDA
#  (reutilizado de piano_expander.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_track(notes: List[dict], cc_events: List[tuple],
                channel: int, name: str, program: int) -> MidiTrack:
    track = MidiTrack()
    track.append(MetaMessage('track_name', name=name, time=0))
    track.append(Message('program_change', channel=channel, program=program, time=0))

    events = []
    for n in notes:
        t   = n['tick']
        dur = max(n['duration'], 1)
        events.append((t,       'on',  n['pitch'], n['velocity']))
        events.append((t + dur, 'off', n['pitch'], 0))
    for ev in cc_events:
        t, _, cc_num, cc_val = ev
        events.append((t, 'cc', cc_num, cc_val))

    events.sort(key=lambda e: (e[0], {'off': 0, 'cc': 1, 'on': 2}.get(e[1], 1)))
    prev = 0
    for abs_t, etype, a, b in events:
        delta = abs_t - prev; prev = abs_t
        if etype == 'on':
            track.append(Message('note_on',  channel=channel, note=a, velocity=b, time=delta))
        elif etype == 'off':
            track.append(Message('note_off', channel=channel, note=a, velocity=0, time=delta))
        elif etype == 'cc':
            track.append(Message('control_change', channel=channel, control=a, value=b, time=delta))
    return track


def build_midi(instr_notes: Dict, instr_cc: Dict, instruments: List[str],
               tempo: int, tpb: int, output_path: str) -> List[str]:
    mid = MidiFile(type=1, ticks_per_beat=tpb)
    t0  = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    t0.append(MetaMessage('time_signature', numerator=4, denominator=4,
                           clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    mid.tracks.append(t0)

    created = []
    for ch, name in enumerate(instruments):
        notes = instr_notes.get(name, [])
        cc    = instr_cc.get(name, [])
        if not notes:
            continue
        prog  = INSTR_PROGRAM.get(name, 40)
        disp  = INSTR_DISPLAY.get(name, name)
        track = build_track(notes, cc, ch % 16, disp, prog)
        mid.tracks.append(track)
        created.append(disp)

    mid.save(output_path)
    return created


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 13 — INFORME
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(midi_in: str, midi_out: str, instruments: List[str],
                    instr_notes: Dict, tonic: int, mode: str,
                    n_sections: int, created: List[str]) -> str:
    W  = 70
    L  = ['═' * W, '  ML EXPANDER v1.0 — INFORME', '═' * W]
    L += [f"  Fuente    : {midi_in}",
          f"  Salida    : {midi_out}",
          f"  Tonalidad : {NOTE_NAMES[tonic]} {mode}",
          f"  Secciones : {n_sections}",
          f"  Pistas    : {len(created)}", '']

    total_in  = 0
    total_out = sum(len(v) for v in instr_notes.values())

    L += ['  LÍNEAS GENERADAS', '  ' + '─' * (W - 2)]
    for name in instruments:
        n    = len(instr_notes.get(name, []))
        role = INSTR_ROLE.get(name, '?')
        disp = INSTR_DISPLAY.get(name, name)
        L.append(f"    {disp:<18} {n:>5} notas  rol={role}")

    L += ['', f"  Total notas generadas: {total_out}", '═' * W]
    return '\n'.join(L)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 14 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='ML Expander v1.0 — Expansión orquestal por machine learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modos:
  train   <dir_midi>   Entrena M1/M2/M3 desde MIDIs orquestales reales
  expand  <boceto.mid> Expande un boceto de piano usando checkpoints
  reduce  <midi.mid>   Ejecuta solo el SyntheticReducer (inspección)

ejemplos:
  python ml_expander.py train  /datos/maestro/ --epochs 50 --lr 0.001
  python ml_expander.py expand boceto.mid --template full --verbose
  python ml_expander.py reduce orquesta.mid --output piano_reducido.mid
        """)

    sub = parser.add_subparsers(dest='mode', required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Entrenar M1/M2/M3')
    p_train.add_argument('midi_dir', help='Directorio con MIDIs orquestales')
    p_train.add_argument('--epochs',       type=int,   default=DEFAULT_EPOCHS)
    p_train.add_argument('--lr',           type=float, default=DEFAULT_LR)
    p_train.add_argument('--batch-size',   type=int,   default=DEFAULT_BATCH_SIZE)
    p_train.add_argument('--hidden-dim',   type=int,   default=DEFAULT_HIDDEN_DIM)
    p_train.add_argument('--delta-bins',   type=int,   default=DEFAULT_DELTA_BINS)
    p_train.add_argument('--pitch-bins',   type=int,   default=DEFAULT_PITCH_BINS)
    p_train.add_argument('--vel-bins',     type=int,   default=DEFAULT_VEL_BINS)
    p_train.add_argument('--dur-bins',     type=int,   default=DEFAULT_DUR_BINS)
    p_train.add_argument('--checkpoint-dir', default=DEFAULT_CHECKPOINT)
    p_train.add_argument('--verbose',      action='store_true')

    # ── expand ───────────────────────────────────────────────────────────────
    p_exp = sub.add_parser('expand', help='Expandir boceto de piano')
    p_exp.add_argument('midi',      help='Boceto MIDI de piano')
    p_exp.add_argument('--template', choices=['chamber', 'strings', 'full'],
                       default='chamber')
    p_exp.add_argument('--split',   default='G4',
                       help='Nota de split para separar manos (default: G4)')
    p_exp.add_argument('--output',  default=None)
    p_exp.add_argument('--checkpoint-dir', default=DEFAULT_CHECKPOINT)
    p_exp.add_argument('--verbose', action='store_true')

    # ── reduce ───────────────────────────────────────────────────────────────
    p_red = sub.add_parser('reduce', help='Reducir orquesta a piano (inspección)')
    p_red.add_argument('midi',     help='MIDI orquestal de entrada')
    p_red.add_argument('--output', default=None)
    p_red.add_argument('--split',  type=int, default=60,
                       help='Pitch de split MD/MI (default: 60 = C4)')
    p_red.add_argument('--verbose',action='store_true')

    return parser


def _parse_split_note(s: str) -> int:
    """Convierte 'G4' → MIDI note number."""
    note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
    s = s.strip()
    if len(s) >= 2 and s[1] == '#':
        pc_  = note_map.get(s[:2], 7)
        oct_ = int(s[2:])
    elif len(s) >= 2 and s[1] == 'b':
        pc_  = (note_map.get(s[0], 7) - 1) % 12
        oct_ = int(s[2:])
    else:
        pc_  = note_map.get(s[0], 7)
        oct_ = int(s[1:])
    return (oct_ + 1) * 12 + pc_


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print('═' * 65)
    print('  ML EXPANDER  v1.0')
    print('═' * 65)

    # ── MODO: train ──────────────────────────────────────────────────────────
    if args.mode == 'train':
        if not TORCH_OK:
            print("ERROR: pip install torch"); sys.exit(1)

        print(f"  Modo      : entrenamiento")
        print(f"  Datos     : {args.midi_dir}")
        print(f"  Epochs    : {args.epochs}")
        print(f"  LR        : {args.lr}")
        print(f"  Hidden    : {args.hidden_dim}")
        print(f"  Bins      : Δtick={args.delta_bins} pitch={args.pitch_bins} "
              f"vel={args.vel_bins} dur={args.dur_bins}")
        print()

        quantizer = EventQuantizer(
            delta_bins=args.delta_bins, pitch_bins=args.pitch_bins,
            vel_bins=args.vel_bins,     dur_bins=args.dur_bins,
        )

        m1, m2, m3 = train(
            midi_dir       = args.midi_dir,
            checkpoint_dir = args.checkpoint_dir,
            quantizer      = quantizer,
            hidden_dim     = args.hidden_dim,
            epochs         = args.epochs,
            lr             = args.lr,
            batch_size     = args.batch_size,
            verbose        = args.verbose,
        )

        print('\n' + '═' * 65)
        print('  ENTRENAMIENTO COMPLETO')
        print('═' * 65)
        print(f"  Checkpoints: {args.checkpoint_dir}/")
        print(f"  Siguiente paso:")
        print(f"    python ml_expander.py expand <boceto.mid> "
              f"--checkpoint-dir {args.checkpoint_dir}")

    # ── MODO: expand ─────────────────────────────────────────────────────────
    elif args.mode == 'expand':
        if not TORCH_OK:
            print("ERROR: pip install torch"); sys.exit(1)

        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no encontrado"); sys.exit(1)

        print(f"  Modo      : expansión")
        print(f"  Entrada   : {args.midi}")
        print(f"  Plantilla : {args.template}")

        # Cargar checkpoints
        print(f"\n  Cargando checkpoints desde {args.checkpoint_dir}...")
        m1, m2, m3, quantizer = load_checkpoints(args.checkpoint_dir)

        # Cargar y preparar piano
        mid, tpb, tempo = load_midi(args.midi)
        split_note      = _parse_split_note(args.split)
        all_notes       = extract_notes_flat(mid)

        if not all_notes:
            print("ERROR: No se encontraron notas en el MIDI."); sys.exit(1)

        # Separar manos: si hay notas en ambos lados del split, usar split_hands;
        # si todo está en un solo lado (piano de pista única o boceto condensado),
        # usar directamente todas las notas como piano combinado.
        above = [n for n in all_notes if n['pitch'] >= split_note]
        below = [n for n in all_notes if n['pitch'] <  split_note]
        if above and below:
            rh, lh = split_hands(all_notes, split_note, tpb)
        else:
            # Una sola región de registro: todo es el piano
            rh, lh = all_notes, []
        piano      = sorted(rh + lh, key=lambda n: n['tick'])
        instruments= TEMPLATES[args.template]

        print(f"  TPB={tpb}  Tempo={round(60_000_000/tempo)} BPM")
        print(f"  Notas piano : {len(piano)}")
        print(f"  Instrumentos: {len(instruments)} ({args.template})")
        print(f"\n  Expandiendo...")

        instr_notes, instr_cc = expand_with_ml(
            piano, tpb, tempo, instruments, m1, m2, m3, quantizer,
            verbose=args.verbose)

        # Salida
        stem     = Path(args.midi).stem
        base     = args.output or (stem + '_ml_orquestado')
        base     = base.replace('.mid', '')
        out_midi = base + '.mid'
        out_rep  = base + '_report.txt'

        print(f"\n  Escribiendo: {out_midi}")
        tonic, mode = detect_key(piano)
        created     = build_midi(instr_notes, instr_cc, instruments,
                                  tempo, tpb, out_midi)

        n_sec = len(segment_sections(piano, tpb))
        report = generate_report(args.midi, out_midi, instruments,
                                  instr_notes, tonic, mode, n_sec, created)
        with open(out_rep, 'w', encoding='utf-8') as f:
            f.write(report)

        print('\n' + '═' * 65)
        print('  RESUMEN')
        print('═' * 65)
        print(f"  MIDI       : {out_midi}")
        print(f"  Informe    : {out_rep}")
        print(f"  Pistas     : {len(created)}")
        print(f"  Tonalidad  : {NOTE_NAMES[tonic]} {mode}")
        total = sum(len(v) for v in instr_notes.values())
        print(f"  Notas gen. : {total}")
        print(f"  Expansión  : {total / max(len(piano), 1):.1f}x")
        print('═' * 65)
        print(f"\n  Importa {out_midi} en tu DAW.")
        for i, name in enumerate(created):
            print(f"  Canal {i+1:>2} → {name}")

    # ── MODO: reduce ─────────────────────────────────────────────────────────
    elif args.mode == 'reduce':
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no encontrado"); sys.exit(1)

        print(f"  Modo      : reducción sintética")
        print(f"  Entrada   : {args.midi}")

        mid, tpb, tempo = load_midi(args.midi)
        all_notes       = extract_notes_flat(mid)

        if not all_notes:
            print("ERROR: No se encontraron notas."); sys.exit(1)

        print(f"  Notas orig.: {len(all_notes)}")
        rh, lh = reduce_orchestra_to_piano(all_notes, tpb, args.split)
        print(f"  MD (piano) : {len(rh)}")
        print(f"  MI (piano) : {len(lh)}")

        # Guardar el piano reducido como MIDI de dos pistas
        stem     = Path(args.midi).stem
        out_path = args.output or (stem + '_reduccion.mid')
        out_midi = MidiFile(type=1, ticks_per_beat=tpb)
        t0_track = MidiTrack()
        t0_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        out_midi.tracks.append(t0_track)

        for hand_notes, hand_name, ch, prog in [
            (rh, 'Piano RH', 0, 0),
            (lh, 'Piano LH', 1, 0),
        ]:
            if hand_notes:
                trk = build_track(hand_notes, [], ch, hand_name, prog)
                out_midi.tracks.append(trk)

        out_midi.save(out_path)
        print(f"\n  Reducción guardada: {out_path}")
        print(f"  Total notas piano: {len(rh) + len(lh)} "
              f"(de {len(all_notes)} originales, ratio "
              f"{(len(rh)+len(lh))/max(len(all_notes),1):.2f})")

    print()


if __name__ == '__main__':
    main()
