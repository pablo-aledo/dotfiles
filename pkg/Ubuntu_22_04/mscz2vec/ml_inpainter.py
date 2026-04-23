#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ML INPAINTER  v1.0                                    ║
║     Relleno inteligente de compases vacíos en obras MIDI multi-pista         ║
║                                                                              ║
║  Detecta compases vacíos o de baja densidad y los rellena usando una         ║
║  combinación de cadenas de Markov bidireccionales, interpolación de           ║
║  contorno y (opcionalmente) un modelo PyTorch entrenado sobre corpus.         ║
║  Soporta múltiples pistas con coordinación armónica entre ellas.             ║
║                                                                              ║
║  MODOS:                                                                      ║
║    scan     — detecta huecos y genera reporte detallado                     ║
║    fill     — rellena huecos con la mejor estrategia (una salida)           ║
║    variants — genera N versiones completas de la pieza para comparar        ║
║    train    — entrena modelo ML sobre corpus de MIDIs                       ║
║    inspect  — inspecciona un modelo entrenado                               ║
║                                                                              ║
║  ── SCAN ─────────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py scan obra.mid                                     ║
║    python ml_inpainter.py scan obra.mid --min-density 0.1                  ║
║    python ml_inpainter.py scan obra.mid --report gaps.txt                  ║
║                                                                              ║
║  ── FILL ─────────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py fill obra.mid                                     ║
║    python ml_inpainter.py fill obra.mid --gaps 1 3                         ║
║    python ml_inpainter.py fill obra.mid --strategy contour_blend            ║
║    python ml_inpainter.py fill obra.mid --length keep                       ║
║    python ml_inpainter.py fill obra.mid --length 4                          ║
║    python ml_inpainter.py fill obra.mid --length auto                       ║
║    python ml_inpainter.py fill obra.mid --checkpoint-dir ./ckpt/ -o out.mid║
║                                                                              ║
║  ── VARIANTS ─────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py variants obra.mid                                 ║
║    python ml_inpainter.py variants obra.mid --n 8                           ║
║    python ml_inpainter.py variants obra.mid --n 6 --out-dir ./variantes/   ║
║    python ml_inpainter.py variants obra.mid --strategies markov interpolate ║
║    python ml_inpainter.py variants obra.mid --gaps 2 --seeds 0 1 2 3       ║
║                                                                              ║
║  ── TRAIN ────────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py train corpus/                                     ║
║    python ml_inpainter.py train corpus/ --epochs 80 --lr 5e-4              ║
║    python ml_inpainter.py train corpus/ --checkpoint-dir ./mis_modelos/    ║
║    python ml_inpainter.py train corpus/ --mask-ratio 0.25                  ║
║                                                                              ║
║  ── INSPECT ──────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py inspect ./ckpt/inpainter.pkl                     ║
║                                                                              ║
║  ESTRATEGIAS (--strategy):                                                  ║
║    auto          — elige automáticamente según longitud y contexto          ║
║    markov_bi     — cadenas de Markov bidireccionales fusionadas             ║
║    interpolate   — interpolación de contorno entre bordes del hueco         ║
║    contour_blend — combinación ponderada de markov_bi e interpolate         ║
║    ml_guided     — guiado por modelo PyTorch (requiere --checkpoint-dir)    ║
║                                                                              ║
║  SCORING AUTOMÁTICO (variants):                                             ║
║    tonal_coherence   35% — notas dentro de la escala del contexto          ║
║    boundary_smooth   30% — suavidad de las uniones en los bordes           ║
║    tension_continuity 20% — continuidad de la curva de tensión             ║
║    rhythm_diversity   15% — diversidad rítmica no mecánica                 ║
║                                                                              ║
║  DEPENDENCIAS: pip install mido numpy (+ torch para --strategy ml_guided)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import math
import json
import pickle
import random
import argparse
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ── MIDO ──────────────────────────────────────────────────────────────────────
try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_OK = True
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)

# ── NUMPY ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    NP_OK = True
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

# ── PYTORCH (opcional) ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

SCALE_INTERVALS: Dict[str, List[int]] = {
    'major':    [0, 2, 4, 5, 7, 9, 11],
    'minor':    [0, 2, 3, 5, 7, 8, 10],
    'dorian':   [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
}

# Consonancia por intervalo (0=disonancia, 1=consonancia)
CONSONANCE_MAP: Dict[int, float] = {
    0: 1.0, 1: 0.1, 2: 0.3, 3: 0.7, 4: 0.8,
    5: 0.9, 6: 0.2, 7: 1.0, 8: 0.7, 9: 0.7,
    10: 0.5, 11: 0.3,
}

# Perfiles Krumhansl-Schmuckler
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# IOI bins en beats
IOI_VALS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.75]

MIN_NOTE_DUR  = 0.125   # duración mínima en beats
MELODY_MIN_DUR = 0.25

# Rol de pista por registro (para coordinación multi-pista)
ROLE_ORDER = ['melody', 'inner', 'harmony', 'bass_melodic', 'bass_root', 'unknown']

# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawNote:
    pitch:    int
    duration: float   # en beats
    velocity: int
    offset:   float   # en beats desde inicio de la pieza
    track_idx: int = 0

@dataclass
class GapRegion:
    """Región de compases vacíos/escasos detectada en la obra."""
    gap_id:        int
    bar_start:     int              # primer compás vacío (1-indexed)
    bar_end:       int              # último compás vacío (inclusivo)
    n_bars:        int
    affected_tracks: List[int]      # índices de pistas afectadas
    all_tracks_empty: bool          # True si ninguna pista tiene notas
    context_key_pc:   int = 0
    context_key_mode: str = 'major'
    context_tension_left:  float = 0.5
    context_tension_right: float = 0.5
    density_left:  float = 0.5     # densidad del contexto izquierdo
    density_right: float = 0.5
    recommended_strategy: str = 'auto'
    anchor_chord_pcs: List[int] = field(default_factory=list)  # PCs de pistas ancla

@dataclass
class TrackInfo:
    """Metadatos de una pista MIDI."""
    idx:       int
    name:      str
    program:   int
    channel:   int
    notes:     List[RawNote]
    role:      str = 'unknown'
    register_lo: int = 36
    register_hi: int = 84

@dataclass
class PieceContext:
    """Contexto global extraído de la pieza."""
    tpb:         int = 480
    tempo_us:    int = 500_000
    tempo_bpm:   float = 120.0
    bpb:         int = 4           # beats per bar
    n_bars:      int = 0
    key_pc:      int = 0
    key_mode:    str = 'major'
    tracks:      List[TrackInfo] = field(default_factory=list)
    raw_mid:     Any = None        # mido.MidiFile original

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 1 — CARGA Y ANÁLISIS MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_piece(midi_path: str) -> PieceContext:
    """Carga un MIDI y extrae el contexto completo de la pieza."""
    mid = MidiFile(midi_path)
    ctx = PieceContext(tpb=mid.ticks_per_beat, raw_mid=mid)

    # Leer tempo y compás del track 0
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            ctx.tempo_us = msg.tempo
            ctx.tempo_bpm = round(60_000_000 / msg.tempo, 2)
        elif msg.type == 'time_signature':
            ctx.bpb = msg.numerator

    # Parsear todas las pistas
    all_notes_global: List[RawNote] = []
    for i, track in enumerate(mid.tracks):
        notes = _parse_track(track, ctx.tpb)
        if not notes:
            continue
        name = track.name or f'Track {i}'
        program, channel = _get_track_program_channel(track)
        lo = min(n.pitch for n in notes)
        hi = max(n.pitch for n in notes)
        ti = TrackInfo(idx=i, name=name, program=program, channel=channel,
                       notes=notes, register_lo=lo, register_hi=hi)
        ti.role = _infer_role(ti)
        ctx.tracks.append(ti)
        for n in notes:
            all_notes_global.append(n)

    # Detectar tonalidad global
    if all_notes_global:
        ctx.key_pc, ctx.key_mode = _detect_key(all_notes_global)

    # Número de compases
    if all_notes_global:
        max_offset = max(n.offset + n.duration for n in all_notes_global)
        ctx.n_bars = max(1, math.ceil(max_offset / ctx.bpb))

    return ctx


def _parse_track(track, tpb: int) -> List[RawNote]:
    """Parsea una pista MIDI a lista de RawNote con offsets en beats."""
    active: Dict[int, Tuple[float, int]] = {}
    notes: List[RawNote] = []
    abs_ticks = 0
    for msg in track:
        abs_ticks += msg.time
        beat = abs_ticks / tpb
        if msg.type == 'note_on' and msg.velocity > 0:
            active[msg.note] = (beat, msg.velocity)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active:
                start, vel = active.pop(msg.note)
                dur = beat - start
                if dur >= MIN_NOTE_DUR:
                    notes.append(RawNote(msg.note, round(dur * 4) / 4,
                                         vel, start))
    return sorted(notes, key=lambda n: n.offset)


def _get_track_program_channel(track) -> Tuple[int, int]:
    program = 0
    channel = 0
    for msg in track:
        if msg.type == 'program_change':
            program = msg.program
            channel = msg.channel
            break
        if hasattr(msg, 'channel'):
            channel = msg.channel
    return program, channel


def _infer_role(ti: TrackInfo) -> str:
    """Infiere el rol de la pista por registro y programa."""
    mid_pitch = (ti.register_lo + ti.register_hi) / 2
    if mid_pitch >= 70:
        return 'melody'
    elif mid_pitch >= 55:
        return 'inner'
    elif mid_pitch >= 45:
        return 'bass_melodic'
    else:
        return 'bass_root'

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — DETECCIÓN DE TONALIDAD Y UTILIDADES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

def _detect_key(notes: List[RawNote]) -> Tuple[int, str]:
    """Detección de tonalidad por correlación de Krumhansl-Schmuckler."""
    if not notes:
        return 0, 'major'
    w = np.zeros(12)
    for n in notes:
        w[n.pitch % 12] += n.duration
    M = np.array(KS_MAJOR)
    m = np.array(KS_MINOR)
    best_k, best_s, best_mode = 0, -999.0, 'major'
    for r in range(12):
        rot = np.roll(w, -r)
        for name, p in [('major', M), ('minor', m)]:
            s = float(np.corrcoef(rot, p)[0, 1])
            if s > best_s:
                best_s = s; best_k = r; best_mode = name
    return best_k, best_mode


def _get_scale_pcs(root_pc: int, mode: str) -> List[int]:
    return [(root_pc + i) % 12 for i in SCALE_INTERVALS.get(mode, SCALE_INTERVALS['major'])]


def _snap_to_scale(pitch: int, root_pc: int, mode: str) -> int:
    sc = set(_get_scale_pcs(root_pc, mode))
    if pitch % 12 in sc:
        return pitch
    for d in range(1, 7):
        if (pitch + d) % 12 in sc:
            return pitch + d
        if (pitch - d) % 12 in sc:
            return pitch - d
    return pitch


def _harmonic_tension(notes: List[RawNote]) -> float:
    """Tensión armónica media por disonancia de intervalos."""
    if len(notes) < 2:
        return 0.0
    pcs = list(set(n.pitch % 12 for n in notes))
    if len(pcs) < 2:
        return 0.0
    td = 0.0
    cnt = 0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            iv = abs(pcs[i] - pcs[j]) % 12
            td += 1.0 - CONSONANCE_MAP.get(iv, 0.5)
            cnt += 1
    return td / cnt if cnt > 0 else 0.0


def _bar_density(notes: List[RawNote], bar: int, bpb: int) -> float:
    """Densidad de notas en un compás (notas por beat)."""
    lo = (bar - 1) * bpb
    hi = bar * bpb
    bn = [n for n in notes if lo <= n.offset < hi]
    return len(bn) / bpb


def _notes_in_bar(notes: List[RawNote], bar: int, bpb: int) -> List[RawNote]:
    lo = (bar - 1) * bpb
    hi = bar * bpb
    return [n for n in notes if lo <= n.offset < hi]


def _notes_in_range(notes: List[RawNote], beat_lo: float, beat_hi: float) -> List[RawNote]:
    return [n for n in notes if beat_lo <= n.offset < beat_hi]


def _build_ioi_histogram(notes: List[RawNote]) -> np.ndarray:
    """Histograma de Inter-Onset Intervals cuantizados en 8 bins."""
    mel = sorted([n for n in notes if n.duration >= MELODY_MIN_DUR],
                  key=lambda n: n.offset)
    if len(mel) < 2:
        return np.ones(8) / 8
    iois = [mel[i+1].offset - mel[i].offset for i in range(len(mel)-1)]
    bins = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    hist = np.zeros(8)
    for ioi in iois:
        idx = min(6, max(0, int(np.searchsorted(bins, ioi))))
        hist[idx] += 1
    s = hist.sum()
    return hist / s if s > 0 else np.ones(8) / 8

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — DETECCIÓN DE HUECOS (SCAN)
# ══════════════════════════════════════════════════════════════════════════════

def detect_gaps(ctx: PieceContext, min_density: float = 0.0) -> List[GapRegion]:
    """
    Detecta regiones de compases vacíos o de baja densidad.
    min_density: umbral de densidad (notas/beat) por debajo del cual
                 se considera un compás "vacío". 0.0 = solo vacíos completos.
    Agrupa compases consecutivos en regiones.
    """
    if not ctx.tracks or ctx.n_bars == 0:
        return []

    # Para cada compás, calcular qué pistas tienen contenido
    bar_track_density: Dict[int, Dict[int, float]] = {}
    for bar in range(1, ctx.n_bars + 1):
        bar_track_density[bar] = {}
        for ti in ctx.tracks:
            d = _bar_density(ti.notes, bar, ctx.bpb)
            bar_track_density[bar][ti.idx] = d

    # Identificar compases vacíos por pista
    empty_per_bar: Dict[int, List[int]] = {}  # bar -> lista de track_idx vacíos
    for bar in range(1, ctx.n_bars + 1):
        empty_tracks = [ti.idx for ti in ctx.tracks
                        if bar_track_density[bar].get(ti.idx, 0.0) <= min_density]
        if empty_tracks:
            empty_per_bar[bar] = empty_tracks

    if not empty_per_bar:
        return []

    # Agrupar compases consecutivos con pistas vacías en común
    gaps: List[GapRegion] = []
    gap_id = 1
    sorted_bars = sorted(empty_per_bar.keys())
    i = 0
    while i < len(sorted_bars):
        bar_start = sorted_bars[i]
        empty_tracks_set = set(empty_per_bar[bar_start])
        bar_end = bar_start
        # Extender mientras el siguiente compás tenga al menos una pista en común
        j = i + 1
        while j < len(sorted_bars):
            next_bar = sorted_bars[j]
            if next_bar != bar_end + 1:
                break
            next_empty = set(empty_per_bar[next_bar])
            if not (empty_tracks_set & next_empty):
                break
            empty_tracks_set &= next_empty  # mantener solo las pistas comunes
            bar_end = next_bar
            j += 1

        n_bars_gap = bar_end - bar_start + 1
        all_tracks = [ti.idx for ti in ctx.tracks]
        all_empty = (empty_tracks_set == set(all_tracks))

        # Analizar contexto
        ctx_window = max(4, n_bars_gap * 2)
        left_notes = _get_context_notes(ctx, bar_start - 1, ctx_window, 'left')
        right_notes = _get_context_notes(ctx, bar_end + 1, ctx_window, 'right')

        # Tonalidad local (basada en contexto)
        ctx_notes = left_notes + right_notes
        if ctx_notes:
            key_pc, key_mode = _detect_key(ctx_notes)
        else:
            key_pc, key_mode = ctx.key_pc, ctx.key_mode

        # Tensión en los bordes
        t_left  = _harmonic_tension(left_notes[-16:] if left_notes else [])
        t_right = _harmonic_tension(right_notes[:16] if right_notes else [])

        # Densidad del contexto
        d_left  = _mean_density_window(ctx, bar_start - ctx_window, bar_start - 1)
        d_right = _mean_density_window(ctx, bar_end + 1, bar_end + ctx_window)

        # Acordes de pistas ancla (las que NO están vacías en este hueco)
        anchor_tracks = [ti for ti in ctx.tracks if ti.idx not in empty_tracks_set]
        anchor_pcs = []
        if anchor_tracks:
            gap_beat_lo = (bar_start - 1) * ctx.bpb
            gap_beat_hi = bar_end * ctx.bpb
            anchor_notes = []
            for ti in anchor_tracks:
                anchor_notes.extend(_notes_in_range(ti.notes, gap_beat_lo, gap_beat_hi))
            anchor_pcs = list(set(n.pitch % 12 for n in anchor_notes))

        # Estrategia recomendada
        if n_bars_gap <= 2:
            rec = 'interpolate'
        elif n_bars_gap <= 4:
            rec = 'contour_blend'
        else:
            rec = 'markov_bi'
        # Unión delicada: tensión muy diferente entre lados
        if abs(t_left - t_right) > 0.4:
            rec = 'contour_blend'

        gaps.append(GapRegion(
            gap_id=gap_id,
            bar_start=bar_start,
            bar_end=bar_end,
            n_bars=n_bars_gap,
            affected_tracks=sorted(empty_tracks_set),
            all_tracks_empty=all_empty,
            context_key_pc=key_pc,
            context_key_mode=key_mode,
            context_tension_left=round(t_left, 3),
            context_tension_right=round(t_right, 3),
            density_left=round(d_left, 3),
            density_right=round(d_right, 3),
            recommended_strategy=rec,
            anchor_chord_pcs=anchor_pcs,
        ))
        gap_id += 1
        i = j

    return gaps


def _get_context_notes(ctx: PieceContext, bar_ref: int, window: int,
                        side: str) -> List[RawNote]:
    """Extrae notas del contexto izquierdo o derecho de un compás."""
    all_notes = []
    for ti in ctx.tracks:
        all_notes.extend(ti.notes)
    if side == 'left':
        bar_lo = max(1, bar_ref - window + 1)
        bar_hi = bar_ref
    else:
        bar_lo = bar_ref
        bar_hi = min(ctx.n_bars, bar_ref + window - 1)
    beat_lo = (bar_lo - 1) * ctx.bpb
    beat_hi = bar_hi * ctx.bpb
    return [n for n in all_notes if beat_lo <= n.offset < beat_hi]


def _mean_density_window(ctx: PieceContext, bar_lo: int, bar_hi: int) -> float:
    """Densidad media de compases en un rango."""
    all_notes = []
    for ti in ctx.tracks:
        all_notes.extend(ti.notes)
    if not all_notes:
        return 0.0
    densities = []
    for bar in range(max(1, bar_lo), min(ctx.n_bars, bar_hi) + 1):
        densities.append(_bar_density(all_notes, bar, ctx.bpb))
    return float(np.mean(densities)) if densities else 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — ESTRATEGIAS DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.0 ANÁLISIS DE ESTILO POR PISTA ─────────────────────────────────────────

@dataclass
class TrackStyle:
    """Perfil rítmico y de textura aprendido de una pista."""
    # Ritmo
    ioi_mode: float          # IOI más frecuente (beats)
    ioi_regularity: float    # 0=irregular, 1=exactamente regular (ostinato)
    ioi_counts: Dict         # {ioi_quantized: count}
    notes_per_bar: float     # densidad media
    # Polifonía
    polyphony: float         # grado de simultaneidad (0=monofónico, 1=acordes densos)
    simultaneous_offsets: List[float]  # offsets donde hay acordes
    # Pitch
    pitch_pool: List[int]    # pitches usados (sin octava)
    pitch_pcs: List[int]     # pitch classes usados
    vel_mean: float
    vel_std: float
    # Patrón ostinato (si existe)
    is_ostinato: bool        # True si el patrón rítmico se repite cada compás
    bar_template: List[dict] # plantilla de un compás: [{rel_offset, pitch_pc, dur, vel}]


def _analyze_track_style(notes: List[RawNote], bpb: int,
                          n_context_bars: int = 8) -> TrackStyle:
    """
    Extrae el perfil de estilo de una pista a partir de los últimos n_context_bars compases.
    Detecta ostinatos, densidad rítmica, polifonía y pool de pitches.
    """
    if not notes:
        return TrackStyle(ioi_mode=1.0, ioi_regularity=0.0, ioi_counts={},
                          notes_per_bar=0.0, polyphony=0.0,
                          simultaneous_offsets=[], pitch_pool=[], pitch_pcs=[],
                          vel_mean=72.0, vel_std=10.0,
                          is_ostinato=False, bar_template=[])

    # Usar solo el contexto más cercano (últimos n_context_bars compases con notas)
    sorted_notes = sorted(notes, key=lambda n: n.offset)
    max_offset = sorted_notes[-1].offset
    window_lo  = max(0.0, max_offset - n_context_bars * bpb)
    ctx = [n for n in sorted_notes if n.offset >= window_lo]
    if not ctx:
        ctx = sorted_notes[-min(32, len(sorted_notes)):]

    # ── IOI ──────────────────────────────────────────────────────────────────
    # Cuantizar a 1/8 de beat para capturar patrones repetitivos con precisión
    Q = 8  # subdivisiones por beat
    ioi_raw = []
    for i in range(len(ctx) - 1):
        raw = ctx[i+1].offset - ctx[i].offset
        if raw > 0.02:  # ignorar simultaneidades
            ioi_raw.append(raw)

    if ioi_raw:
        ioi_q = [round(x * Q) / Q for x in ioi_raw]
        ioi_counter = Counter(ioi_q)
        total_ioi = len(ioi_q)
        ioi_mode_val = ioi_counter.most_common(1)[0][0]
        ioi_mode_count = ioi_counter.most_common(1)[0][1]
        ioi_regularity = ioi_mode_count / total_ioi
    else:
        ioi_counter = Counter()
        ioi_mode_val = 1.0
        ioi_regularity = 0.0

    notes_per_bar = len(ctx) / max(n_context_bars, 1)

    # ── POLIFONÍA ─────────────────────────────────────────────────────────────
    # Agrupar notas por onset (dentro de tolerancia)
    TOL = 0.05  # beats de tolerancia para considerar simultáneas
    onset_groups: Dict[float, List[RawNote]] = {}
    for n in ctx:
        key = round(n.offset / TOL) * TOL
        onset_groups.setdefault(key, []).append(n)

    poly_counts = [len(g) for g in onset_groups.values()]
    polyphony = (sum(1 for c in poly_counts if c > 1) /
                 max(len(poly_counts), 1))
    simult_offsets = [k for k, g in onset_groups.items() if len(g) > 1]

    # ── PITCH POOL ────────────────────────────────────────────────────────────
    pitch_pool = sorted(set(n.pitch for n in ctx))
    pitch_pcs  = sorted(set(n.pitch % 12 for n in ctx))
    vels = [n.velocity for n in ctx]
    vel_mean = float(np.mean(vels))
    vel_std  = float(np.std(vels)) if len(vels) > 1 else 10.0

    # ── DETECCIÓN DE OSTINATO ─────────────────────────────────────────────────
    # Comparar el patrón rítmico de compás a compás (offsets relativos al inicio del compás)
    # Si la varianza de los patrones es baja → ostinato
    bar_patterns = []
    n_ctx_bars_actual = max(1, int((max_offset - window_lo) / bpb))
    for b in range(n_ctx_bars_actual):
        bar_lo = window_lo + b * bpb
        bar_hi = bar_lo + bpb
        bar_notes = [n for n in ctx if bar_lo <= n.offset < bar_hi]
        if bar_notes:
            rel = tuple(round((n.offset - bar_lo) * Q) / Q
                        for n in sorted(bar_notes, key=lambda x: x.offset))
            bar_patterns.append(rel)

    is_ostinato = False
    bar_template: List[dict] = []

    if len(bar_patterns) >= 2:
        # Comparar longitudes y offsets relativos
        lengths = [len(p) for p in bar_patterns]
        len_mode = Counter(lengths).most_common(1)[0][0]
        consistent_len = sum(1 for l in lengths if l == len_mode) / len(lengths)

        if consistent_len >= 0.75 and len_mode >= 1:
            # Calcular varianza de posición por slot
            same_len = [p for p in bar_patterns if len(p) == len_mode]
            if len(same_len) >= 2:
                slots = list(zip(*same_len))
                slot_vars = [float(np.var(s)) for s in slots]
                mean_var = float(np.mean(slot_vars)) if slot_vars else 1.0
                is_ostinato = mean_var < 0.05  # varianza muy baja → ostinato

    # Construir plantilla de un compás si hay ostinato o patrón claro
    if is_ostinato or ioi_regularity >= 0.5:
        # Tomar el último compás completo como plantilla
        last_bar_lo = max_offset - (max_offset % bpb) if max_offset % bpb > 0 \
                      else max_offset - bpb
        last_bar_lo = max(window_lo, last_bar_lo)
        template_notes = [n for n in ctx
                          if last_bar_lo <= n.offset < last_bar_lo + bpb]
        if not template_notes:
            # Tomar penúltimo compás
            last_bar_lo -= bpb
            template_notes = [n for n in ctx
                               if last_bar_lo <= n.offset < last_bar_lo + bpb]
        if template_notes:
            bar_template = [
                {'rel_offset': round((n.offset - last_bar_lo) * Q) / Q,
                 'pitch_pc': n.pitch % 12,
                 'pitch': n.pitch,
                 'dur': round(n.duration * Q) / Q,
                 'vel': n.velocity}
                for n in sorted(template_notes, key=lambda x: x.offset)
            ]

    return TrackStyle(
        ioi_mode=ioi_mode_val,
        ioi_regularity=ioi_regularity,
        ioi_counts=dict(ioi_counter),
        notes_per_bar=notes_per_bar,
        polyphony=polyphony,
        simultaneous_offsets=simult_offsets,
        pitch_pool=pitch_pool,
        pitch_pcs=pitch_pcs,
        vel_mean=vel_mean,
        vel_std=vel_std,
        is_ostinato=is_ostinato,
        bar_template=bar_template,
    )


def _build_markov_chain(notes: List[RawNote], order: int = 1) -> Dict[int, Dict[int, float]]:
    """Cadena de Markov de intervalos melódicos."""
    pitches = [n.pitch for n in sorted(notes, key=lambda n: n.offset)
               if n.duration >= MELODY_MIN_DUR]
    if len(pitches) < order + 1:
        return {}
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    chain: Dict[int, Dict[int, int]] = {}
    for i in range(len(intervals) - order):
        key = intervals[i]
        nxt = intervals[i + order]
        chain.setdefault(key, {})
        chain[key][nxt] = chain[key].get(nxt, 0) + 1
    prob: Dict[int, Dict[int, float]] = {}
    for k, counts in chain.items():
        total = sum(counts.values())
        prob[k] = {iv: cnt / total for iv, cnt in counts.items()}
    return prob


def _merge_markov_chains(chains: List[Dict]) -> Dict[int, Dict[int, float]]:
    """Combina varias cadenas de Markov sumando pesos."""
    merged: Dict[int, Dict[int, float]] = {}
    for chain in chains:
        for k, nexts in chain.items():
            merged.setdefault(k, {})
            for iv, p in nexts.items():
                merged[k][iv] = merged[k].get(iv, 0) + p
    for k in merged:
        total = sum(merged[k].values())
        if total > 0:
            merged[k] = {iv: v / total for iv, v in merged[k].items()}
    return merged


def _sample_from_chain(chain: Dict[int, Dict[int, float]], prev_iv: int,
                        rng: random.Random, contour_bias: float = 0.0) -> int:
    """Muestrea el siguiente intervalo de una cadena de Markov."""
    nexts = chain.get(prev_iv, {})
    if not nexts:
        candidates = list(range(-4, 5))
        weights = [1.0] * len(candidates)
    else:
        candidates = list(nexts.keys())
        weights = list(nexts.values())

    # Aplicar sesgo de contorno
    if abs(contour_bias) > 0.1:
        weights = [w * (1.5 if (contour_bias > 0 and iv > 0) or
                               (contour_bias < 0 and iv < 0) else 0.7)
                   for w, iv in zip(weights, candidates)]

    total_w = sum(weights)
    r = rng.random() * total_w
    acc = 0.0
    for iv, w in zip(candidates, weights):
        acc += w
        if r <= acc:
            return iv
    return candidates[-1]


# ── 4.1 GENERADOR BASADO EN ESTILO (NÚCLEO PRINCIPAL) ───────────────────────

def _generate_from_style(
        style: TrackStyle,
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random,
        anchor_pcs: Optional[List[int]] = None,
        contour_bias: float = 0.0) -> List[RawNote]:
    """
    Generador principal consciente del estilo de la pista.
    Prioridades:
      1. Si hay ostinato claro (ioi_regularity >= 0.8 o is_ostinato):
         replica la plantilla compás a compás con variación melódica.
      2. Si hay patrón rítmico fuerte (ioi_regularity >= 0.5):
         usa el IOI modal como grid fijo y aplica Markov solo al pitch.
      3. En caso contrario: Markov bidireccional estándar.
    En todos los casos respeta densidad, polifonía y rango de la pista.
    """
    if style.is_ostinato or style.ioi_regularity >= 0.75:
        return _generate_ostinato(style, left_notes, right_notes,
                                   n_bars, bpb, key_pc, key_mode, rng, anchor_pcs)
    elif style.ioi_regularity >= 0.4 or style.notes_per_bar >= 3:
        return _generate_grid_markov(style, left_notes, right_notes,
                                      n_bars, bpb, key_pc, key_mode, rng,
                                      anchor_pcs, contour_bias)
    else:
        return _generate_notes_markov_bi(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode, rng,
            ioi_hist=None, anchor_pcs=anchor_pcs, contour_bias=contour_bias,
            style=style)


def _generate_ostinato(
        style: TrackStyle,
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random,
        anchor_pcs: Optional[List[int]] = None) -> List[RawNote]:
    """
    Replica la plantilla de compás del ostinato con variación armónica suave.
    Los pitches se adaptan según el acorde subyacente (anchor_pcs) pero
    mantienen los intervalos internos de la plantilla.
    """
    if not style.bar_template:
        # Sin plantilla: fallback a grid regular
        return _generate_grid_markov(
            style, left_notes, right_notes, n_bars, bpb,
            key_pc, key_mode, rng, anchor_pcs)

    result: List[RawNote] = []
    scale_pcs = set(_get_scale_pcs(key_pc, key_mode))
    if anchor_pcs:
        scale_pcs |= set(anchor_pcs)

    # Pitch de referencia: última nota del contexto izquierdo en el registro correcto
    left_sorted = sorted(left_notes, key=lambda n: n.offset)
    if left_sorted:
        ref_pitch = left_sorted[-1].pitch
    elif style.pitch_pool:
        ref_pitch = style.pitch_pool[len(style.pitch_pool) // 2]
    else:
        ref_pitch = key_pc + 60

    # Pitch objetivo (para guiar la variación a lo largo del hueco)
    right_sorted = sorted(right_notes, key=lambda n: n.offset)
    target_pitch = right_sorted[0].pitch if right_sorted else ref_pitch

    for bar_idx in range(n_bars):
        bar_offset = bar_idx * bpb
        t = bar_idx / max(n_bars - 1, 1)  # 0..1

        # Pitch central interpolado hacia el objetivo
        center_pitch = int(ref_pitch + t * (target_pitch - ref_pitch))
        # Pequeña variación aleatoria por compás (una vez por compás, no por nota)
        bar_jitter = rng.randint(-2, 2)
        center_pitch = max(style.pitch_pool[0] if style.pitch_pool else 36,
                           min(style.pitch_pool[-1] if style.pitch_pool else 96,
                               center_pitch + bar_jitter))

        # Transponer la plantilla al pitch central manteniendo intervalos internos
        if style.bar_template:
            tpl_pitches = [t_note['pitch'] for t_note in style.bar_template]
            tpl_center  = int(np.median(tpl_pitches))
            transpose   = center_pitch - tpl_center

            for t_note in style.bar_template:
                raw_pitch = t_note['pitch'] + transpose
                # Snap al pool de pitches de la pista (respeta el rango)
                if style.pitch_pool:
                    raw_pitch = min(style.pitch_pool,
                                    key=lambda p: (abs(p - raw_pitch),
                                                   0 if p % 12 in scale_pcs else 1))
                else:
                    raw_pitch = _snap_to_scale(
                        max(36, min(96, raw_pitch)), key_pc, key_mode)

                vel = int(np.clip(t_note['vel'] + rng.randint(-5, 5), 30, 120))
                result.append(RawNote(
                    pitch=raw_pitch,
                    duration=t_note['dur'],
                    velocity=vel,
                    offset=bar_offset + t_note['rel_offset'],
                ))

    return result


def _generate_grid_markov(
        style: TrackStyle,
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random,
        anchor_pcs: Optional[List[int]] = None,
        contour_bias: float = 0.0) -> List[RawNote]:
    """
    Usa el IOI modal como grid rítmico fijo y aplica Markov bidireccional
    solo al pitch. Respeta la densidad característica de la pista.
    """
    chain_fwd = _build_markov_chain(left_notes)
    right_rev  = list(reversed(sorted(right_notes, key=lambda n: n.offset)))
    chain_rev  = _build_markov_chain(right_rev)
    chain      = _merge_markov_chains([chain_fwd, chain_rev]) \
                 if chain_fwd or chain_rev else {}

    target_beats = n_bars * bpb

    # Nota semilla y objetivo
    left_sorted  = sorted(left_notes,  key=lambda n: n.offset)
    right_sorted = sorted(right_notes, key=lambda n: n.offset)
    seed_pitch   = left_sorted[-1].pitch  if left_sorted  else (style.pitch_pool[len(style.pitch_pool)//2] if style.pitch_pool else key_pc + 60)
    goal_pitch   = right_sorted[0].pitch  if right_sorted else seed_pitch

    if abs(goal_pitch - seed_pitch) > 0:
        contour_bias = 0.4 * float(np.sign(goal_pitch - seed_pitch))

    scale_pcs = set(_get_scale_pcs(key_pc, key_mode))
    if anchor_pcs:
        scale_pcs |= set(anchor_pcs)

    # Construir grid rítmico: usar el IOI modal más frecuente
    # Si hay varios IOIs, samplear según su frecuencia relativa
    ioi_total = sum(style.ioi_counts.values()) if style.ioi_counts else 1
    ioi_items = sorted(style.ioi_counts.items(), key=lambda x: -x[1]) \
                if style.ioi_counts else [(1.0, 1)]

    # Duración modal
    modal_ioi = ioi_items[0][0] if ioi_items else 1.0
    # Duración media ponderada
    dur_options = [ioi for ioi, _ in ioi_items[:6]]
    dur_weights = [cnt / ioi_total for _, cnt in ioi_items[:6]]

    # Velocidad de referencia
    base_vel = int(style.vel_mean)

    result: List[RawNote] = []
    cursor = 0.0
    current_pitch = seed_pitch
    prev_iv = 0

    while cursor < target_beats:
        # Samplear duración del pool de la pista
        if dur_weights and dur_options:
            r = rng.random()
            acc = 0.0
            chosen_ioi = dur_options[0]
            for ioi_val, w in zip(dur_options, dur_weights):
                acc += w
                if r <= acc:
                    chosen_ioi = ioi_val
                    break
        else:
            chosen_ioi = modal_ioi

        dur = min(chosen_ioi, target_beats - cursor)
        if dur < 0.05:
            break

        # Pitch por Markov o pool directo
        if chain:
            iv = _sample_from_chain(chain, prev_iv, rng, contour_bias)
            new_pitch = current_pitch + iv
        else:
            # Sin cadena: samplear del pool de pitches
            if style.pitch_pool:
                new_pitch = rng.choice(style.pitch_pool)
            else:
                new_pitch = current_pitch + rng.randint(-3, 3)

        # Forzar al pool de pitches si existe (respeta el vocabulario de la pista)
        if style.pitch_pool:
            new_pitch = min(style.pitch_pool,
                            key=lambda p: (abs(p - new_pitch),
                                           0 if p % 12 in scale_pcs else 1))
        else:
            new_pitch = _snap_to_scale(max(36, min(96, new_pitch)), key_pc, key_mode)

        vel = int(np.clip(base_vel + rng.randint(
            -int(style.vel_std), int(style.vel_std)), 30, 120))

        result.append(RawNote(new_pitch, dur, vel, cursor))
        cursor += chosen_ioi
        current_pitch = new_pitch
        prev_iv = new_pitch - current_pitch

        # Polifonía: si la pista tiene acordes, añadir nota simultánea
        if style.polyphony > 0.3 and rng.random() < style.polyphony:
            # Nota adicional simultánea a intervalos de 3ª o 4ª
            chord_iv = rng.choice([-7, -5, -4, -3, 3, 4, 5, 7])
            chord_pitch = new_pitch + chord_iv
            if style.pitch_pool:
                chord_pitch = min(style.pitch_pool,
                                  key=lambda p: abs(p - chord_pitch))
            else:
                chord_pitch = _snap_to_scale(
                    max(36, min(96, chord_pitch)), key_pc, key_mode)
            chord_vel = int(np.clip(vel - rng.randint(5, 15), 25, 110))
            result.append(RawNote(chord_pitch,
                                  min(dur, target_beats - cursor + chosen_ioi),
                                  chord_vel, cursor))

    return sorted(result, key=lambda n: n.offset)


def _generate_notes_markov_bi(
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random, ioi_hist: Optional[np.ndarray] = None,
        anchor_pcs: Optional[List[int]] = None,
        contour_bias: float = 0.0,
        style: Optional['TrackStyle'] = None) -> List[RawNote]:
    """
    Cadenas de Markov bidireccionales fusionadas.
    Construye una cadena desde la izquierda y otra desde la derecha (invertida)
    y las combina para generar el relleno.
    """
    chain_fwd = _build_markov_chain(left_notes)
    # Cadena inversa: construida desde las notas derechas en orden invertido
    right_rev = list(reversed(sorted(right_notes, key=lambda n: n.offset)))
    chain_rev = _build_markov_chain(right_rev)
    chain = _merge_markov_chains([chain_fwd, chain_rev]) if chain_fwd or chain_rev \
            else {}

    target_beats = n_bars * bpb

    # Nota semilla: última del contexto izquierdo
    if left_notes:
        seed = sorted(left_notes, key=lambda n: n.offset)[-1].pitch
    elif right_notes:
        seed = sorted(right_notes, key=lambda n: n.offset)[0].pitch
    else:
        seed = key_pc + 60

    # Nota objetivo: primera del contexto derecho
    if right_notes:
        target_pitch = sorted(right_notes, key=lambda n: n.offset)[0].pitch
    else:
        target_pitch = seed

    # Calcular sesgo de contorno hacia la nota objetivo
    if abs(target_pitch - seed) > 0:
        contour_bias = 0.4 * np.sign(target_pitch - seed)

    # IOI para ritmo
    if ioi_hist is None:
        all_ctx = left_notes + right_notes
        ioi_hist = _build_ioi_histogram(all_ctx) if all_ctx else None

    notes: List[RawNote] = []
    cursor = 0.0
    prev_iv = 0
    current_pitch = seed
    ioi_options = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.25, 0.5]

    scale_pcs = set(_get_scale_pcs(key_pc, key_mode))
    # Añadir PCs de acordes ancla si existen
    if anchor_pcs:
        scale_pcs |= set(anchor_pcs)

    while cursor < target_beats:
        # Samplear IOI
        if ioi_hist is not None and ioi_hist.sum() > 0:
            probs = ioi_hist / ioi_hist.sum()
            r = rng.random()
            acc = 0.0
            chosen_ioi = 1.0
            for j, p in enumerate(probs):
                acc += p
                if r <= acc:
                    chosen_ioi = ioi_options[j % len(ioi_options)]
                    break
        else:
            chosen_ioi = rng.choice([0.5, 1.0, 1.0, 2.0, 0.25])

        dur = min(chosen_ioi, target_beats - cursor)
        if dur < 0.125:
            break

        # Samplear intervalo
        iv = _sample_from_chain(chain, prev_iv, rng, contour_bias)
        new_pitch = current_pitch + iv

        # Clamp al pitch pool de la pista o a la escala
        if style and style.pitch_pool:
            new_pitch = min(style.pitch_pool,
                            key=lambda p: (abs(p - new_pitch),
                                           0 if p % 12 in scale_pcs else 1))
        else:
            new_pitch = max(36, min(96, new_pitch))
            new_pitch = _snap_to_scale(new_pitch, key_pc, key_mode)

        # Velocidad con variación suave
        if notes:
            base_vel = notes[-1].velocity + rng.randint(-6, 6)
        elif left_notes:
            base_vel = int(np.mean([n.velocity for n in left_notes[-4:]]))
        else:
            base_vel = 72
        vel = int(np.clip(base_vel, 35, 110))

        notes.append(RawNote(new_pitch, dur, vel, cursor))
        cursor += chosen_ioi
        prev_iv = iv
        current_pitch = new_pitch

    return notes


def _generate_notes_interpolate(
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random, ioi_hist: Optional[np.ndarray] = None,
        anchor_pcs: Optional[List[int]] = None,
        style: Optional['TrackStyle'] = None) -> List[RawNote]:
    """
    Interpolación de contorno: navega suavemente entre el borde izquierdo
    y el borde derecho del hueco.
    Si existe un estilo con patrón rítmico claro, usa ese grid rítmico
    en lugar de un histograma genérico de IOI.
    """
    target_beats = n_bars * bpb

    # Extraer contornos de borde
    left_sorted = sorted(left_notes, key=lambda n: n.offset)
    right_sorted = sorted(right_notes, key=lambda n: n.offset)

    left_edge  = left_sorted[-4:]  if len(left_sorted)  >= 4 else left_sorted
    right_edge = right_sorted[:4]  if len(right_sorted) >= 4 else right_sorted

    start_pitch = int(np.mean([n.pitch for n in left_edge])) if left_edge \
                  else key_pc + 60
    end_pitch   = int(np.mean([n.pitch for n in right_edge])) if right_edge \
                  else start_pitch

    start_vel = int(np.mean([n.velocity for n in left_edge])) if left_edge else 72
    end_vel   = int(np.mean([n.velocity for n in right_edge])) if right_edge else 72

    # Grid rítmico: preferir el estilo de la pista si está disponible
    if style and style.ioi_counts:
        ioi_items   = sorted(style.ioi_counts.items(), key=lambda x: -x[1])
        dur_options = [ioi for ioi, _ in ioi_items[:6]]
        ioi_total   = sum(style.ioi_counts.values())
        dur_weights = [cnt / ioi_total for _, cnt in ioi_items[:6]]
    else:
        if ioi_hist is None:
            ioi_hist = _build_ioi_histogram(left_notes + right_notes)
        ioi_options = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.25, 0.5]
        dur_options = ioi_options
        dur_weights = (ioi_hist / ioi_hist.sum()).tolist() \
                      if ioi_hist is not None and ioi_hist.sum() > 0 \
                      else [1.0 / len(ioi_options)] * len(ioi_options)

    pitch_pool = style.pitch_pool if style else []
    scale_pcs  = set(_get_scale_pcs(key_pc, key_mode))

    notes: List[RawNote] = []
    cursor = 0.0

    while cursor < target_beats:
        t = cursor / target_beats
        # Pitch interpolado con variación suave
        target_pitch_now = start_pitch + t * (end_pitch - start_pitch)
        jitter    = rng.randint(-2, 2)
        raw_pitch = int(round(target_pitch_now)) + jitter

        if pitch_pool:
            new_pitch = min(pitch_pool,
                            key=lambda p: (abs(p - raw_pitch),
                                           0 if p % 12 in scale_pcs else 1))
        else:
            new_pitch = _snap_to_scale(max(36, min(96, raw_pitch)), key_pc, key_mode)

        vel = int(start_vel + t * (end_vel - start_vel) + rng.randint(-5, 5))
        vel = int(np.clip(vel, 35, 110))

        # Samplear IOI del pool de la pista
        r = rng.random()
        acc = 0.0
        chosen_ioi = dur_options[0] if dur_options else 1.0
        for ioi_val, w in zip(dur_options, dur_weights):
            acc += w
            if r <= acc:
                chosen_ioi = ioi_val
                break

        dur = min(chosen_ioi, target_beats - cursor)
        if dur < 0.05:
            break

        notes.append(RawNote(new_pitch, dur, vel, cursor))
        cursor += chosen_ioi

    return notes


def _generate_notes_contour_blend(
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random, ioi_hist: Optional[np.ndarray] = None,
        anchor_pcs: Optional[List[int]] = None) -> List[RawNote]:
    """
    Combinación ponderada: usa markov_bi para la estructura rítmica/melódica
    e interpolate para la tendencia de contorno, mezclando según la longitud.
    """
    # Para huecos cortos, interpolate tiene más peso; para largos, markov
    blend_t = min(1.0, n_bars / 6.0)  # 0=todo interpolate, 1=todo markov

    notes_mk = _generate_notes_markov_bi(
        left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
        rng, ioi_hist, anchor_pcs)
    notes_ip = _generate_notes_interpolate(
        left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
        rng, ioi_hist, anchor_pcs)

    # Tomar el set más largo y ajustar pitches mezclando contornos
    if not notes_mk:
        return notes_ip
    if not notes_ip:
        return notes_mk

    # Alinear por tiempo y mezclar pitch
    target_beats = n_bars * bpb
    result: List[RawNote] = []

    # Usar estructura rítmica de markov_bi, corregir pitch con blend
    for nm in notes_mk:
        t = nm.offset / target_beats
        # Pitch de interpolate en el mismo t relativo
        ip_candidates = [n for n in notes_ip if abs(n.offset - nm.offset) < bpb]
        if ip_candidates:
            ip_pitch = min(ip_candidates, key=lambda n: abs(n.offset - nm.offset)).pitch
            mixed_pitch = int(round(nm.pitch * blend_t + ip_pitch * (1 - blend_t)))
            mixed_pitch = max(36, min(96, mixed_pitch))
            mixed_pitch = _snap_to_scale(mixed_pitch, key_pc, key_mode)
        else:
            mixed_pitch = nm.pitch
        result.append(RawNote(mixed_pitch, nm.duration, nm.velocity, nm.offset))

    return result


def _generate_notes_ml(
        left_notes: List[RawNote], right_notes: List[RawNote],
        n_bars: int, bpb: int, key_pc: int, key_mode: str,
        rng: random.Random, model_data: Dict,
        ioi_hist: Optional[np.ndarray] = None,
        anchor_pcs: Optional[List[int]] = None) -> List[RawNote]:
    """
    Generación guiada por el modelo BiLSTM entrenado.
    Si el modelo falla, cae back a contour_blend.
    """
    if not TORCH_OK or model_data is None:
        return _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

    try:
        model = InpainterNet(model_data['config'])
        model.load_state_dict(model_data['state_dict'])
        model.eval()

        target_beats = n_bars * bpb
        feat = _build_inpainter_features(left_notes, right_notes,
                                          key_pc, key_mode, n_bars, bpb)
        feat_t = torch.FloatTensor(feat).unsqueeze(0)

        scale_pcs = _get_scale_pcs(key_pc, key_mode)
        seed_pitch = sorted(left_notes, key=lambda n: n.offset)[-1].pitch \
                     if left_notes else key_pc + 60

        notes: List[RawNote] = []
        cursor = 0.0
        prev_pitch = seed_pitch

        with torch.no_grad():
            while cursor < target_beats:
                pos_t = torch.FloatTensor([[cursor / max(target_beats, 1)]])
                pitch_logits, dur_logits, vel_out = model(feat_t, pos_t)

                # Samplear pitch
                pitch_probs = torch.softmax(pitch_logits[0] / 0.8, dim=-1).numpy()
                pitch_bins = model_data['config']['pitch_bins']
                pitch_step = 61 / pitch_bins
                pitch_idx = rng.choices(range(pitch_bins),
                                         weights=pitch_probs.tolist())[0]
                raw_pitch = int(36 + pitch_idx * pitch_step)
                new_pitch = _snap_to_scale(
                    max(36, min(96, raw_pitch)), key_pc, key_mode)

                # Samplear duración
                dur_probs = torch.softmax(dur_logits[0], dim=-1).numpy()
                dur_vals = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.125, 0.333]
                dur_idx = rng.choices(range(len(dur_vals)),
                                       weights=dur_probs[:len(dur_vals)].tolist())[0]
                chosen_dur = dur_vals[dur_idx]

                # Velocidad
                vel = int(np.clip(float(vel_out[0, 0]) * 127, 35, 110))
                vel += rng.randint(-5, 5)
                vel = int(np.clip(vel, 35, 110))

                dur = min(chosen_dur, target_beats - cursor)
                if dur < 0.125:
                    break

                notes.append(RawNote(new_pitch, dur, vel, cursor))
                cursor += chosen_dur
                prev_pitch = new_pitch

        return notes if notes else _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

    except Exception as e:
        return _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — COORDINACIÓN MULTI-PISTA
# ══════════════════════════════════════════════════════════════════════════════

def fill_gap_multitrack(ctx: PieceContext, gap: GapRegion,
                         strategy: str = 'auto', seed: int = 42,
                         length_mode: str = 'keep', length_bars: int = 0,
                         model_data: Optional[Dict] = None) -> Dict[int, List[RawNote]]:
    """
    Rellena un hueco en todas las pistas afectadas con coordinación armónica.
    Devuelve dict {track_idx: [RawNote]} con las notas generadas.
    """
    rng = random.Random(seed)

    # Determinar longitud real del relleno
    if length_mode == 'keep':
        n_bars_fill = gap.n_bars
    elif length_mode == 'fixed':
        n_bars_fill = max(1, length_bars)
    elif length_mode == 'auto':
        # Heurística: entre la longitud actual y la del contexto más corto
        ctx_window = max(2, gap.n_bars)
        d_left  = gap.density_left
        d_right = gap.density_right
        if d_left < 0.2 and d_right < 0.2:
            n_bars_fill = max(1, gap.n_bars - 1)
        elif d_left > 0.8 and d_right > 0.8:
            n_bars_fill = gap.n_bars + 1
        else:
            n_bars_fill = gap.n_bars
    else:
        n_bars_fill = gap.n_bars

    # Resolver estrategia
    if strategy == 'auto':
        strategy = gap.recommended_strategy

    bpb = ctx.bpb
    key_pc = gap.context_key_pc
    key_mode = gap.context_key_mode

    # Ordenar pistas por rol (melody primero, bass al final)
    tracks_to_fill = [ti for ti in ctx.tracks if ti.idx in gap.affected_tracks]
    tracks_to_fill.sort(key=lambda ti: ROLE_ORDER.index(ti.role)
                        if ti.role in ROLE_ORDER else len(ROLE_ORDER))

    # Pistas ancla: las que tienen notas durante el hueco
    anchor_tracks = [ti for ti in ctx.tracks if ti.idx not in gap.affected_tracks]

    # IOI global de la pieza para referencia rítmica
    all_notes = [n for ti in ctx.tracks for n in ti.notes]
    global_ioi = _build_ioi_histogram(all_notes)

    result: Dict[int, List[RawNote]] = {}
    generated_so_far: List[RawNote] = []  # notas ya generadas (para constraint de las siguientes)

    for ti in tracks_to_fill:
        # Contexto izquierdo de esta pista
        left_notes = [n for n in ti.notes
                      if n.offset < (gap.bar_start - 1) * bpb]
        right_notes = [n for n in ti.notes
                       if n.offset >= gap.bar_end * bpb]

        # Últimas/primeras notas del contexto (ventana de 8 compases)
        left_ctx = sorted(left_notes, key=lambda n: n.offset)
        left_ctx = left_ctx[-max(8 * bpb, 1):]  # aproximación por notas
        right_ctx = sorted(right_notes, key=lambda n: n.offset)[:max(8 * bpb, 1)]

        # Construir anchor_pcs desde pistas ancla + ya generadas
        anchor_pcs = list(gap.anchor_chord_pcs)
        gap_beat_lo = (gap.bar_start - 1) * bpb
        gap_beat_hi = gap.bar_end * bpb
        for anc_ti in anchor_tracks:
            anchor_pcs.extend(n.pitch % 12 for n in anc_ti.notes
                               if gap_beat_lo <= n.offset < gap_beat_hi)
        for gn in generated_so_far:
            if gap_beat_lo <= gn.offset < gap_beat_hi:
                anchor_pcs.append(gn.pitch % 12)
        anchor_pcs = list(set(anchor_pcs))

        # IOI de la pista
        ioi_hist = _build_ioi_histogram(ti.notes) if ti.notes else global_ioi

        # ── ANÁLISIS DE ESTILO DE LA PISTA ───────────────────────────────────
        # Usar el contexto izquierdo para aprender el estilo
        style = _analyze_track_style(left_ctx if left_ctx else ti.notes,
                                      bpb, n_context_bars=8)

        # Ajustar rango de pitch según registro de la pista
        def clamp_pitch(n: RawNote) -> RawNote:
            p = n.pitch
            while p < ti.register_lo:
                p += 12
            while p > ti.register_hi:
                p -= 12
            p = max(ti.register_lo, min(ti.register_hi, p))
            return RawNote(p, n.duration, n.velocity, n.offset, ti.idx)

        # Sesgo de contorno
        left_sorted_ctx  = sorted(left_ctx,  key=lambda n: n.offset)
        right_sorted_ctx = sorted(right_ctx, key=lambda n: n.offset)
        seed_p   = left_sorted_ctx[-1].pitch  if left_sorted_ctx  else key_pc + 60
        target_p = right_sorted_ctx[0].pitch  if right_sorted_ctx else seed_p
        contour_bias = 0.4 * float(np.sign(target_p - seed_p)) \
                       if abs(target_p - seed_p) > 1 else 0.0

        # ── GENERAR SEGÚN ESTRATEGIA ──────────────────────────────────────────
        if strategy == 'interpolate':
            # interpolate usa estilo para el ritmo pero interpola el pitch
            raw_notes = _generate_notes_interpolate(
                left_ctx, right_ctx, n_bars_fill, bpb, key_pc, key_mode,
                rng, ioi_hist, anchor_pcs, style=style)
        elif strategy == 'markov_bi':
            raw_notes = _generate_from_style(
                style, left_ctx, right_ctx, n_bars_fill, bpb,
                key_pc, key_mode, rng, anchor_pcs, contour_bias)
        elif strategy == 'ml_guided':
            raw_notes = _generate_notes_ml(
                left_ctx, right_ctx, n_bars_fill, bpb, key_pc, key_mode,
                rng, model_data, ioi_hist, anchor_pcs)
        else:  # contour_blend y auto ya resuelto
            raw_notes = _generate_from_style(
                style, left_ctx, right_ctx, n_bars_fill, bpb,
                key_pc, key_mode, rng, anchor_pcs, contour_bias)

        # Ajustar offset al inicio real del hueco
        gap_offset = (gap.bar_start - 1) * bpb
        for n in raw_notes:
            n.offset += gap_offset
            n.track_idx = ti.idx

        # Clampar al registro
        raw_notes = [clamp_pitch(n) for n in raw_notes]

        result[ti.idx] = raw_notes
        generated_so_far.extend(raw_notes)

    return result

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 6 — ESCRITURA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def write_filled_midi(ctx: PieceContext, fills: Dict[int, Dict[int, List[RawNote]]],
                       output_path: str,
                       gap_length_adjustments: Optional[Dict[int, int]] = None):
    """
    Escribe el MIDI de salida con los huecos rellenos.
    fills: {gap_id: {track_idx: [RawNote]}}
    gap_length_adjustments: {gap_id: nuevo_n_bars} si --length != keep
    """
    mid_out = MidiFile(ticks_per_beat=ctx.tpb)

    # Construir mapa de notas por pista: originales + relleno
    for ti in ctx.tracks:
        # Determinar notas a excluir (las que están dentro de algún hueco rellenado)
        # y añadir las generadas
        filled_notes_for_track: List[RawNote] = []
        excluded_ranges: List[Tuple[float, float]] = []

        for gap_id, track_fills in fills.items():
            if ti.idx in track_fills:
                # Encontrar el gap
                gap_beat_lo: float = 0
                gap_beat_hi: float = 0
                # Los huecos vienen de las notas generadas — inferir rango del primer/último
                fn = track_fills[ti.idx]
                if fn:
                    gap_beat_lo = min(n.offset for n in fn)
                    gap_beat_hi = max(n.offset + n.duration for n in fn)
                else:
                    continue
                excluded_ranges.append((gap_beat_lo, gap_beat_hi))
                filled_notes_for_track.extend(fn)

        # Filtrar notas originales fuera de huecos
        kept_notes = [n for n in ti.notes
                      if not any(lo <= n.offset < hi
                                 for lo, hi in excluded_ranges)]
        all_notes = sorted(kept_notes + filled_notes_for_track,
                            key=lambda n: n.offset)

        track_out = MidiTrack()
        mid_out.tracks.append(track_out)
        track_out.name = ti.name

        track_out.append(MetaMessage('set_tempo',
                                      tempo=ctx.tempo_us, time=0))
        if ti == ctx.tracks[0]:
            track_out.append(MetaMessage('time_signature',
                                          numerator=ctx.bpb,
                                          denominator=4, time=0))
        track_out.append(Message('program_change',
                                   program=ti.program,
                                   channel=ti.channel, time=0))

        # Construir eventos
        events = []
        for n in all_notes:
            start_t = int(n.offset * ctx.tpb)
            end_t   = int((n.offset + n.duration) * ctx.tpb)
            events.append((start_t, 'on',  n.pitch, n.velocity))
            events.append((end_t,   'off', n.pitch, 0))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        current_tick = 0
        for tick, typ, pitch, vel in events:
            delta = max(0, tick - current_tick)
            ch = ti.channel
            if typ == 'on':
                track_out.append(Message('note_on', note=pitch,
                                          velocity=vel, channel=ch, time=delta))
            else:
                track_out.append(Message('note_off', note=pitch,
                                          velocity=0, channel=ch, time=delta))
            current_tick = tick

    mid_out.save(output_path)


def write_original_with_gap_info(ctx: PieceContext, output_path: str):
    """Reconstruye el MIDI original (para validación)."""
    fills: Dict[int, Dict[int, List[RawNote]]] = {}
    write_filled_midi(ctx, fills, output_path)

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 7 — SCORING DE VARIANTES
# ══════════════════════════════════════════════════════════════════════════════

def score_fill(ctx: PieceContext, gap: GapRegion,
               filled_notes: Dict[int, List[RawNote]]) -> Dict[str, float]:
    """
    Puntúa el relleno de un hueco según cuatro criterios.
    Devuelve dict con métricas y puntuación total.
    """
    bpb = ctx.bpb
    key_pc = gap.context_key_pc
    key_mode = gap.context_key_mode
    scale_pcs = set(_get_scale_pcs(key_pc, key_mode))
    gap_beat_lo = (gap.bar_start - 1) * bpb
    gap_beat_hi = gap.bar_end * bpb

    all_filled: List[RawNote] = [n for notes in filled_notes.values() for n in notes]

    if not all_filled:
        return {'tonal_coherence': 0.0, 'boundary_smooth': 0.0,
                'tension_continuity': 0.0, 'rhythm_diversity': 0.0,
                'total': 0.0}

    # 1. Coherencia tonal (35%)
    in_scale = sum(1 for n in all_filled if n.pitch % 12 in scale_pcs)
    tonal_coherence = in_scale / len(all_filled)

    # 2. Suavidad de uniones (30%)
    # Distancia entre última nota del contexto izq y primera del relleno,
    # y entre última del relleno y primera del contexto dcho
    smooth_scores = []
    for ti in ctx.tracks:
        fn = filled_notes.get(ti.idx, [])
        if not fn:
            continue
        left_notes  = [n for n in ti.notes if n.offset < gap_beat_lo]
        right_notes = [n for n in ti.notes if n.offset >= gap_beat_hi]
        fn_sorted   = sorted(fn, key=lambda n: n.offset)

        if left_notes and fn_sorted:
            last_left = sorted(left_notes, key=lambda n: n.offset)[-1].pitch
            first_fill = fn_sorted[0].pitch
            dist = abs(last_left - first_fill)
            smooth_scores.append(max(0.0, 1.0 - dist / 24.0))

        if right_notes and fn_sorted:
            last_fill   = fn_sorted[-1].pitch
            first_right = sorted(right_notes, key=lambda n: n.offset)[0].pitch
            dist = abs(last_fill - first_right)
            smooth_scores.append(max(0.0, 1.0 - dist / 24.0))

    boundary_smooth = float(np.mean(smooth_scores)) if smooth_scores else 0.5

    # 3. Continuidad de tensión (20%)
    t_fill = _harmonic_tension(all_filled)
    t_left  = gap.context_tension_left
    t_right = gap.context_tension_right
    t_expected = (t_left + t_right) / 2.0
    tension_continuity = max(0.0, 1.0 - abs(t_fill - t_expected) * 2.0)

    # 4. Diversidad rítmica (15%)
    if len(all_filled) > 1:
        durs = [n.duration for n in all_filled]
        dur_std = float(np.std(durs))
        rhythm_diversity = min(1.0, dur_std / 0.5)
    else:
        rhythm_diversity = 0.5

    total = (tonal_coherence   * 0.35 +
             boundary_smooth   * 0.30 +
             tension_continuity * 0.20 +
             rhythm_diversity   * 0.15)

    return {
        'tonal_coherence':    round(tonal_coherence, 3),
        'boundary_smooth':    round(boundary_smooth, 3),
        'tension_continuity': round(tension_continuity, 3),
        'rhythm_diversity':   round(rhythm_diversity, 3),
        'total':              round(total, 3),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 8 — MODELO PYTORCH (InpainterNet)
# ══════════════════════════════════════════════════════════════════════════════

FEAT_DIM   = 40   # dimensión del vector de features
PITCH_BINS = 61   # 36..96 en semitonos
DUR_BINS   = 8
HIDDEN_DIM = 128

def _build_inpainter_features(left_notes: List[RawNote], right_notes: List[RawNote],
                                key_pc: int, key_mode: str,
                                n_bars: int, bpb: int) -> np.ndarray:
    """
    Vector de 40 features que describe el contexto de un hueco:
    [0-11]  histograma PC del contexto izquierdo
    [12-23] histograma PC del contexto derecho
    [24-31] IOI histogram del contexto izquierdo
    [32]    tónica normalizada
    [33]    modo (0=mayor, 1=menor)
    [34]    longitud del hueco (normalizada)
    [35]    tensión izquierda
    [36]    tensión derecha
    [37]    densidad izquierda
    [38]    densidad derecha
    [39]    ratio longitud/contexto
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)

    # PC histograms
    if left_notes:
        for n in left_notes[-32:]:
            feat[n.pitch % 12] += n.duration
        s = feat[:12].sum()
        if s > 0: feat[:12] /= s

    if right_notes:
        for n in right_notes[:32]:
            feat[12 + n.pitch % 12] += n.duration
        s = feat[12:24].sum()
        if s > 0: feat[12:24] /= s

    # IOI histogram
    ioi = _build_ioi_histogram(left_notes) if left_notes else np.ones(8)/8
    feat[24:32] = ioi

    feat[32] = key_pc / 12.0
    feat[33] = 0.0 if key_mode == 'major' else 1.0
    feat[34] = min(1.0, n_bars / 16.0)
    feat[35] = _harmonic_tension(left_notes[-16:]) if left_notes else 0.5
    feat[36] = _harmonic_tension(right_notes[:16]) if right_notes else 0.5
    feat[37] = min(1.0, len(left_notes) / max(n_bars * bpb * 2, 1))
    feat[38] = min(1.0, len(right_notes) / max(n_bars * bpb * 2, 1))
    feat[39] = n_bars / max(len(left_notes) + len(right_notes), 1)

    return feat


if TORCH_OK:
    class InpainterNet(nn.Module):
        """
        MLP que predice pitch, duración y velocidad para cada evento del relleno,
        condicionado por features de contexto y posición relativa en el hueco.
        """
        def __init__(self, config: Dict):
            super().__init__()
            feat_dim   = config.get('feat_dim', FEAT_DIM)
            hidden_dim = config.get('hidden_dim', HIDDEN_DIM)
            pitch_bins = config.get('pitch_bins', PITCH_BINS)
            dur_bins   = config.get('dur_bins', DUR_BINS)

            self.config = config

            # Encoder de contexto
            self.context_enc = nn.Sequential(
                nn.Linear(feat_dim + 1, hidden_dim),  # +1 para posición relativa
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )

            h = hidden_dim // 2
            self.pitch_head = nn.Linear(h, pitch_bins)
            self.dur_head   = nn.Linear(h, dur_bins)
            self.vel_head   = nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())

        def forward(self, feat: 'torch.Tensor', pos: 'torch.Tensor'):
            x = torch.cat([feat, pos], dim=-1)
            h = self.context_enc(x)
            return self.pitch_head(h), self.dur_head(h), self.vel_head(h)


    class InpainterDataset(Dataset):
        """Dataset de pares (features, target_note) extraídos de un corpus."""
        def __init__(self, samples: List[Dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            return (torch.FloatTensor(s['feat']),
                    torch.FloatTensor([s['pos']]),
                    torch.LongTensor([s['pitch_bin']]),
                    torch.LongTensor([s['dur_bin']]),
                    torch.FloatTensor([s['vel']]))


def extract_training_samples(midi_path: str, mask_ratio: float = 0.2,
                               bpb_override: int = 0) -> List[Dict]:
    """
    Extrae muestras de entrenamiento de un MIDI:
    enmascara segmentos aleatorios y genera pares (contexto_izq+dcho, nota_enmascarada).
    """
    try:
        mid = MidiFile(midi_path)
        tpb = mid.ticks_per_beat
        tempo_us = 500_000
        bpb = bpb_override or 4
        for msg in mid.tracks[0]:
            if msg.type == 'set_tempo':   tempo_us = msg.tempo
            elif msg.type == 'time_signature': bpb = msg.numerator

        all_notes: List[RawNote] = []
        for i, track in enumerate(mid.tracks):
            notes = _parse_track(track, tpb)
            all_notes.extend(notes)

        if not all_notes:
            return []

        key_pc, key_mode = _detect_key(all_notes)
        max_offset = max(n.offset + n.duration for n in all_notes)
        n_bars = max(1, math.ceil(max_offset / bpb))

        if n_bars < 8:
            return []

        samples = []
        rng = random.Random(hash(midi_path) % (2**31))

        # Generar huecos sintéticos
        n_gaps = max(1, int(n_bars * mask_ratio / 2))
        for _ in range(n_gaps):
            gap_len = rng.randint(1, min(4, n_bars // 4))
            gap_start = rng.randint(4, max(5, n_bars - gap_len - 4))
            gap_beat_lo = (gap_start - 1) * bpb
            gap_beat_hi = (gap_start + gap_len - 1) * bpb

            masked = [n for n in all_notes
                      if gap_beat_lo <= n.offset < gap_beat_hi]
            if not masked:
                continue

            left_notes = [n for n in all_notes if n.offset < gap_beat_lo]
            right_notes = [n for n in all_notes if n.offset >= gap_beat_hi]

            feat = _build_inpainter_features(
                left_notes[-32:], right_notes[:32], key_pc, key_mode,
                gap_len, bpb)

            for n in masked:
                pos = (n.offset - gap_beat_lo) / max(gap_len * bpb, 1)
                pitch_bin = max(0, min(PITCH_BINS - 1, n.pitch - 36))
                dur_vals = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.125, 0.333]
                dur_bin = min(range(len(dur_vals)),
                               key=lambda i: abs(dur_vals[i] - n.duration))
                samples.append({
                    'feat':      feat,
                    'pos':       float(pos),
                    'pitch_bin': pitch_bin,
                    'dur_bin':   dur_bin,
                    'vel':       n.velocity / 127.0,
                })

        return samples

    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 9 — REPORTE DE SCAN
# ══════════════════════════════════════════════════════════════════════════════

def _tension_label(t: float) -> str:
    if t < 0.2: return 'baja'
    if t < 0.5: return 'media'
    if t < 0.7: return 'alta'
    return 'muy alta'


def _density_label(d: float) -> str:
    if d < 0.15: return 'sparse'
    if d < 0.4:  return 'moderada'
    if d < 0.7:  return 'densa'
    return 'muy densa'


def build_scan_report(midi_path: str, ctx: PieceContext,
                       gaps: List[GapRegion]) -> str:
    """Genera el texto del reporte de scan."""
    key_name = NOTE_NAMES[ctx.key_pc] + ' ' + ctx.key_mode
    total_gap_bars = sum(g.n_bars for g in gaps)
    track_names = {ti.idx: ti.name for ti in ctx.tracks}

    lines = []
    W = 62
    lines.append('═' * W)
    lines.append('  ML INPAINTER  v{} — SCAN REPORT'.format(VERSION))
    lines.append('═' * W)
    lines.append(f'  Archivo  : {Path(midi_path).name}')
    lines.append(f'  Pistas   : {len(ctx.tracks)}  |  '
                 f'Compases: {ctx.n_bars}  |  '
                 f'Tonalidad: {key_name}')
    lines.append(f'  Tempo    : {ctx.tempo_bpm:.0f} BPM  |  '
                 f'Compás: {ctx.bpb}/4')
    lines.append('─' * W)

    if not gaps:
        lines.append('  ✓ No se detectaron huecos.')
        lines.append('═' * W)
        return '\n'.join(lines)

    lines.append(f'  HUECOS DETECTADOS: {len(gaps)} región(es)  '
                 f'({total_gap_bars} compás/es en total)')
    lines.append('')

    for g in gaps:
        bar_range = (f'Compás {g.bar_start}'
                     if g.n_bars == 1
                     else f'Compases {g.bar_start}–{g.bar_end}')
        gap_type = 'TOTAL' if g.all_tracks_empty else 'PARCIAL'
        affected = ', '.join(track_names.get(idx, f'Track {idx}')
                             for idx in g.affected_tracks)

        lines.append(f'  [GAP-{g.gap_id}]  {bar_range}  ({g.n_bars}c)  '
                     f'{gap_type} — {affected}')

        ctx_key = NOTE_NAMES[g.context_key_pc] + ' ' + g.context_key_mode
        lines.append(f'           Tonalidad local : {ctx_key}')
        lines.append(f'           Tensión         : '
                     f'izq {g.context_tension_left:.2f} ({_tension_label(g.context_tension_left)})  '
                     f'→  dcho {g.context_tension_right:.2f} ({_tension_label(g.context_tension_right)})')
        lines.append(f'           Densidad vecinos: '
                     f'izq {g.density_left:.2f} ({_density_label(g.density_left)})  '
                     f'dcho {g.density_right:.2f} ({_density_label(g.density_right)})')
        if g.anchor_chord_pcs and not g.all_tracks_empty:
            chord_str = ' '.join(NOTE_NAMES[pc] for pc in sorted(g.anchor_chord_pcs)[:6])
            lines.append(f'           Acordes ancla   : {chord_str}')

        warn = ''
        if abs(g.context_tension_left - g.context_tension_right) > 0.4:
            warn = '  ⚠ transición de tensión brusca'
        elif g.n_bars >= 8:
            warn = '  ⚠ hueco largo — considerar revisión manual'

        lines.append(f'           Estrategia rec. : {g.recommended_strategy}{warn}')
        lines.append('')

    # Mapa ASCII
    lines.append('─' * W)
    lines.append('  MAPA DE LA OBRA:')

    map_width = min(ctx.n_bars, 56)
    scale_factor = ctx.n_bars / map_width if ctx.n_bars > map_width else 1.0

    # Construir mapa de densidad
    all_notes = [n for ti in ctx.tracks for n in ti.notes]
    bar_has_gap = set()
    for g in gaps:
        for b in range(g.bar_start, g.bar_end + 1):
            bar_has_gap.add(b)

    row = '  '
    for i in range(map_width):
        bar_idx = int(i * scale_factor) + 1
        if bar_idx in bar_has_gap:
            row += '░'
        else:
            d = _bar_density(all_notes, bar_idx, ctx.bpb)
            if d < 0.1:   row += ' '
            elif d < 0.3: row += '▒'
            elif d < 0.6: row += '▓'
            else:          row += '█'

    lines.append(f'  1{" " * (map_width // 2 - 1)}{ctx.n_bars // 2}'
                 f'{" " * (map_width // 2 - len(str(ctx.n_bars // 2)))}{ctx.n_bars}')
    lines.append(row)

    # Indicadores de huecos
    gap_row = '  '
    for i in range(map_width):
        bar_idx = int(i * scale_factor) + 1
        found_gap = next((g for g in gaps
                           if g.bar_start <= bar_idx <= g.bar_end), None)
        if found_gap:
            gap_row += str(found_gap.gap_id)
        else:
            gap_row += ' '
    lines.append(gap_row + '  ← GAP-N')

    lines.append('')
    lines.append('  Leyenda: █ densa  ▓ moderada  ▒ sparse  ' +
                 '░ vacío/hueco  (número = GAP-ID)')
    lines.append('═' * W)
    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 10 — MODOS CLI
# ══════════════════════════════════════════════════════════════════════════════

def mode_scan(args):
    """Modo scan: detecta y reporta huecos."""
    print(f'\n  Cargando {args.input}...')
    ctx = load_piece(args.input)
    gaps = detect_gaps(ctx, min_density=args.min_density)
    report = build_scan_report(args.input, ctx, gaps)
    print('\n' + report)

    if args.report:
        Path(args.report).write_text(report, encoding='utf-8')
        print(f'\n  Reporte guardado en: {args.report}')

    if not gaps:
        print('  No hay nada que rellenar.')
    else:
        print(f'\n  Ejecuta fill o variants para rellenar los huecos.')

    return gaps


def mode_fill(args):
    """Modo fill: rellena huecos y genera un único MIDI de salida."""
    print(f'\n  Cargando {args.input}...')
    ctx = load_piece(args.input)
    gaps = detect_gaps(ctx, min_density=args.min_density)

    if not gaps:
        print('  No se detectaron huecos. Salida = entrada.')
        out = args.output or _default_output(args.input, 'filled')
        import shutil; shutil.copy(args.input, out)
        print(f'  Guardado: {out}')
        return

    # Filtrar huecos si se especificaron
    if args.gaps:
        gaps = [g for g in gaps if g.gap_id in args.gaps]
        if not gaps:
            print(f'  ERROR: los GAP-IDs indicados no existen. '
                  f'Ejecuta scan primero.')
            sys.exit(1)

    # Parsear modo de longitud
    length_mode, length_bars = _parse_length(args.length)

    # Cargar modelo si existe
    model_data = _load_model(args.checkpoint_dir)

    print(f'  Huecos a rellenar: {len(gaps)}')
    fills: Dict[int, Dict[int, List[RawNote]]] = {}

    for g in gaps:
        strat = args.strategy
        if strat == 'ml_guided' and model_data is None:
            print(f'  [GAP-{g.gap_id}] ml_guided requiere modelo entrenado. '
                  f'Usando contour_blend.')
            strat = 'contour_blend'
        print(f'  [GAP-{g.gap_id}] compases {g.bar_start}-{g.bar_end}  '
              f'estrategia={strat if strat != "auto" else g.recommended_strategy}  '
              f'pistas={len(g.affected_tracks)}')
        fills[g.gap_id] = fill_gap_multitrack(
            ctx, g, strategy=strat, seed=args.seed,
            length_mode=length_mode, length_bars=length_bars,
            model_data=model_data)

    out = args.output or _default_output(args.input, 'filled')
    write_filled_midi(ctx, fills, out)
    print(f'\n  ✓ MIDI guardado: {out}')


def mode_variants(args):
    """Modo variants: genera N versiones completas de la pieza."""
    print(f'\n  Cargando {args.input}...')
    ctx = load_piece(args.input)
    gaps = detect_gaps(ctx, min_density=args.min_density)

    if not gaps:
        print('  No se detectaron huecos.')
        return

    # Filtrar huecos si se especificaron
    if args.gaps:
        gaps = [g for g in gaps if g.gap_id in args.gaps]

    model_data = _load_model(args.checkpoint_dir)

    # Determinar estrategias a usar
    all_strats = ['markov_bi', 'interpolate', 'contour_blend']
    if model_data is not None:
        all_strats.append('ml_guided')
    if args.strategies and args.strategies != ['all']:
        strategies = [s for s in args.strategies if s in all_strats] or all_strats
    else:
        strategies = all_strats

    # Seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(args.n))

    # Generar combinaciones de (estrategia, seed) hasta tener args.n variantes
    combos = []
    for seed in seeds:
        for strat in strategies:
            combos.append((strat, seed))
    combos = combos[:args.n]
    while len(combos) < args.n:
        combos.append((strategies[len(combos) % len(strategies)],
                        len(combos) + 100))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.input).stem
    length_mode, length_bars = _parse_length(args.length)
    results = []

    print(f'  Generando {len(combos)} variantes para {len(gaps)} hueco(s)...')

    for v_idx, (strat, seed) in enumerate(combos):
        fills: Dict[int, Dict[int, List[RawNote]]] = {}
        gap_scores = []

        for g in gaps:
            f = fill_gap_multitrack(ctx, g, strategy=strat, seed=seed,
                                     length_mode=length_mode,
                                     length_bars=length_bars,
                                     model_data=model_data)
            fills[g.gap_id] = f
            sc = score_fill(ctx, g, f)
            gap_scores.append(sc['total'])

        mean_score = round(float(np.mean(gap_scores)), 3) if gap_scores else 0.0
        fname = f'{stem}_v{v_idx+1:02d}_{strat}_s{seed}.mid'
        fpath = out_dir / fname
        write_filled_midi(ctx, fills, str(fpath))

        results.append({'variant': v_idx + 1, 'file': fname,
                         'strategy': strat, 'seed': seed,
                         'score': mean_score})
        print(f'  [{v_idx+1:02d}/{len(combos)}] {fname}  score={mean_score:.3f}')

    # Ranking
    results.sort(key=lambda r: -r['score'])
    print('\n  ── RANKING (mayor score = mejor coherencia musical) ──')
    for i, r in enumerate(results, 1):
        bar = '█' * int(r['score'] * 20) + '░' * (20 - int(r['score'] * 20))
        print(f'  {i:2d}. [{bar}] {r["score"]:.3f}  {r["file"]}')

    print(f'\n  ✓ {len(results)} variantes guardadas en: {out_dir}')


def mode_train(args):
    """Modo train: entrena el modelo InpainterNet sobre un corpus de MIDIs."""
    if not TORCH_OK:
        print('ERROR: PyTorch no disponible. pip install torch'); sys.exit(1)

    corpus_dir = Path(args.corpus)
    midi_files = list(corpus_dir.rglob('*.mid')) + list(corpus_dir.rglob('*.midi'))
    if not midi_files:
        print(f'ERROR: no se encontraron MIDIs en {corpus_dir}'); sys.exit(1)

    print(f'\n  Extrayendo muestras de {len(midi_files)} MIDIs...')
    all_samples = []
    for mf in midi_files:
        s = extract_training_samples(str(mf), mask_ratio=args.mask_ratio)
        all_samples.extend(s)
        if len(s) > 0:
            print(f'  {mf.name}: {len(s)} muestras')

    if not all_samples:
        print('ERROR: no se pudieron extraer muestras del corpus.'); sys.exit(1)

    print(f'\n  Total muestras: {len(all_samples)}')
    print(f'  Entrenando InpainterNet (hidden={args.hidden_dim}, '
          f'epochs={args.epochs}, lr={args.lr})...')

    config = {'feat_dim': FEAT_DIM, 'hidden_dim': args.hidden_dim,
              'pitch_bins': PITCH_BINS, 'dur_bins': DUR_BINS}
    model = InpainterNet(config)
    dataset = InpainterDataset(all_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for feat, pos, pitch_tgt, dur_tgt, vel_tgt in loader:
            optimizer.zero_grad()
            pitch_logits, dur_logits, vel_out = model(feat, pos)
            loss = (ce(pitch_logits, pitch_tgt.squeeze(1)) +
                    ce(dur_logits,   dur_tgt.squeeze(1)) +
                    mse(vel_out.squeeze(-1), vel_tgt.squeeze(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / max(batches, 1)
            print(f'  Época {epoch:4d}/{args.epochs}  loss={avg:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

    # Guardar
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'inpainter.pkl'
    data = {'state_dict': model.state_dict(), 'config': config,
            'n_samples': len(all_samples), 'epochs': args.epochs,
            'version': VERSION}
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\n  ✓ Modelo guardado: {ckpt_path}')


def mode_inspect(args):
    """Modo inspect: muestra información de un modelo guardado."""
    path = Path(args.model)
    if not path.exists():
        print(f'ERROR: {path} no encontrado'); sys.exit(1)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    cfg = data.get('config', {})
    print('\n  ── MODELO INPAINTER ──────────────────────────────')
    print(f'  Versión      : {data.get("version", "?")}')
    print(f'  Épocas       : {data.get("epochs", "?")}')
    print(f'  Muestras     : {data.get("n_samples", "?")}')
    print(f'  feat_dim     : {cfg.get("feat_dim", "?")}')
    print(f'  hidden_dim   : {cfg.get("hidden_dim", "?")}')
    print(f'  pitch_bins   : {cfg.get("pitch_bins", "?")}')
    print(f'  dur_bins     : {cfg.get("dur_bins", "?")}')
    if TORCH_OK:
        model = InpainterNet(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Parámetros   : {n_params:,}')
    print('  ─────────────────────────────────────────────────\n')

# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES CLI
# ══════════════════════════════════════════════════════════════════════════════

def _default_output(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    return str(p.parent / f'{p.stem}_{suffix}.mid')


def _parse_length(length_arg: str) -> Tuple[str, int]:
    if length_arg == 'keep':
        return 'keep', 0
    elif length_arg == 'auto':
        return 'auto', 0
    else:
        try:
            n = int(length_arg)
            return 'fixed', n
        except ValueError:
            return 'keep', 0


def _load_model(checkpoint_dir: Optional[str]) -> Optional[Dict]:
    if not checkpoint_dir:
        return None
    ckpt_path = Path(checkpoint_dir) / 'inpainter.pkl'
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        print(f'  Modelo cargado: {ckpt_path}')
        return data
    except Exception as e:
        print(f'  Aviso: no se pudo cargar el modelo ({e})')
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='ml_inpainter',
        description='Relleno inteligente de compases vacíos en MIDIs multi-pista',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='mode', required=True)

    # ── scan ──
    ps = sub.add_parser('scan', help='Detecta huecos y genera reporte')
    ps.add_argument('input', help='MIDI de entrada')
    ps.add_argument('--min-density', type=float, default=0.0,
                    help='Umbral de densidad mínima (def: 0.0 = solo vacíos)')
    ps.add_argument('--report', default=None,
                    help='Guardar reporte en fichero .txt')

    # ── fill ──
    pf = sub.add_parser('fill', help='Rellena huecos (una salida)')
    pf.add_argument('input', help='MIDI de entrada')
    pf.add_argument('--gaps', type=int, nargs='+', default=None,
                    help='IDs de huecos a rellenar (default: todos)')
    pf.add_argument('--strategy', default='auto',
                    choices=['auto', 'markov_bi', 'interpolate',
                             'contour_blend', 'ml_guided'],
                    help='Estrategia de generación (def: auto)')
    pf.add_argument('--length', default='keep',
                    help='Longitud del relleno: keep | auto | N (nº compases)')
    pf.add_argument('--min-density', type=float, default=0.0)
    pf.add_argument('--checkpoint-dir', default=None,
                    help='Directorio con modelo entrenado')
    pf.add_argument('--seed', type=int, default=42)
    pf.add_argument('-o', '--output', default=None,
                    help='MIDI de salida (def: <entrada>_filled.mid)')

    # ── variants ──
    pv = sub.add_parser('variants',
                         help='Genera N versiones completas para comparar')
    pv.add_argument('input', help='MIDI de entrada')
    pv.add_argument('--n', type=int, default=6,
                    help='Número de variantes (def: 6)')
    pv.add_argument('--gaps', type=int, nargs='+', default=None)
    pv.add_argument('--strategies', nargs='+', default=['all'],
                    choices=['all', 'markov_bi', 'interpolate',
                             'contour_blend', 'ml_guided'],
                    help='Estrategias a usar (def: all)')
    pv.add_argument('--seeds', type=int, nargs='+', default=None)
    pv.add_argument('--length', default='keep')
    pv.add_argument('--min-density', type=float, default=0.0)
    pv.add_argument('--checkpoint-dir', default=None)
    pv.add_argument('--out-dir', default='./inpainter_variants',
                    help='Directorio de salida (def: ./inpainter_variants)')

    # ── train ──
    pt = sub.add_parser('train', help='Entrena modelo sobre corpus de MIDIs')
    pt.add_argument('corpus', help='Directorio con MIDIs de entrenamiento')
    pt.add_argument('--epochs', type=int, default=60)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--batch-size', type=int, default=64)
    pt.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM)
    pt.add_argument('--mask-ratio', type=float, default=0.2,
                    help='Fracción de compases a enmascarar por MIDI (def: 0.2)')
    pt.add_argument('--checkpoint-dir', default='./inpainter_ckpt')

    # ── inspect ──
    pi = sub.add_parser('inspect', help='Inspecciona un modelo entrenado')
    pi.add_argument('model', help='Ruta al fichero .pkl del modelo')

    return p

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        'scan':     mode_scan,
        'fill':     mode_fill,
        'variants': mode_variants,
        'train':    mode_train,
        'inspect':  mode_inspect,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
