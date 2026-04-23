#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ML INPAINTER  v2.0                                    ║
║     Relleno inteligente de compases vacíos en obras MIDI multi-pista         ║
║                                                                              ║
║  Detecta compases vacíos o de baja densidad y los rellena usando una         ║
║  combinación de análisis de estilo por pista, cadenas de Markov              ║
║  bidireccionales y (opcionalmente) un modelo PyTorch autoregresivo           ║
║  entrenado sobre corpus. Coordina múltiples pistas con constraint            ║
║  armónico entre ellas.                                                       ║
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
║    python ml_inpainter.py fill obra.mid --length keep|auto|N               ║
║    python ml_inpainter.py fill obra.mid --checkpoint-dir ./ckpt/            ║
║    python ml_inpainter.py fill obra.mid --no-chord-conditioning             ║
║    python ml_inpainter.py fill obra.mid --context-bars 12                  ║
║    python ml_inpainter.py fill obra.mid -o salida.mid                      ║
║                                                                              ║
║  ── VARIANTS ─────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py variants obra.mid --n 8                           ║
║    python ml_inpainter.py variants obra.mid --n 6 --out-dir ./variantes/   ║
║    python ml_inpainter.py variants obra.mid --strategies markov interpolate ║
║    python ml_inpainter.py variants obra.mid --gaps 2 --seeds 0 1 2 3       ║
║    python ml_inpainter.py variants obra.mid --score-weights                 ║
║        tonal=0.4 boundary=0.3 tension=0.2 rhythm=0.1                       ║
║                                                                              ║
║  ── TRAIN ────────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py train corpus/                                     ║
║    python ml_inpainter.py train corpus/ --epochs 80 --lr 5e-4              ║
║    python ml_inpainter.py train corpus/ --hidden-dim 256                   ║
║    python ml_inpainter.py train corpus/ --mask-ratio 0.25                  ║
║    python ml_inpainter.py train corpus/ --mask-phrase                      ║
║    python ml_inpainter.py train corpus/ --augment-transpose 5              ║
║    python ml_inpainter.py train corpus/ --augment-reverse                  ║
║    python ml_inpainter.py train corpus/ --role melody                      ║
║    python ml_inpainter.py train corpus/ --role bass                        ║
║    python ml_inpainter.py train corpus/ --checkpoint-dir ./mis_modelos/    ║
║                                                                              ║
║  ── INSPECT ──────────────────────────────────────────────────────────────  ║
║    python ml_inpainter.py inspect ./ckpt/inpainter.pkl                     ║
║                                                                              ║
║  ESTRATEGIAS (--strategy):                                                  ║
║    auto          — elige según estilo detectado de la pista                 ║
║    markov_bi     — Markov bidireccional con análisis de estilo              ║
║    interpolate   — interpolación de contorno entre bordes del hueco         ║
║    contour_blend — combinación ponderada de markov_bi e interpolate         ║
║    ml_guided     — guiado por modelo PyTorch (requiere --checkpoint-dir)    ║
║                                                                              ║
║  SCORING AUTOMÁTICO (variants):                                             ║
║    Pesos por defecto (configurables con --score-weights):                   ║
║    tonal_coherence    35% — notas dentro de la escala del contexto         ║
║    boundary_smooth    30% — suavidad de las uniones en los bordes          ║
║    tension_continuity 20% — continuidad de la curva de tensión             ║
║    rhythm_consistency 15% — consistencia rítmica respecto al estilo        ║
║    El criterio rhythm_consistency es sensible al rol de la pista:           ║
║    para melody favorece la diversidad; para bass/inner, la regularidad.     ║
║                                                                              ║
║  MEJORAS v2.0 (respecto a v1.0):                                            ║
║  [M1] Features de estilo rítmico en el vector ML: IOI modal, regularidad,  ║
║       notas/compás, polifonía y rol de pista (8 dims extra → FEAT_DIM=56). ║
║  [M2] Enmascaramiento basado en límites de frase (--mask-phrase): los       ║
║       huecos sintéticos de entrenamiento se colocan en transiciones de       ║
║       densidad real, no de forma aleatoria.                                  ║
║  [M3] Scoring de ritmo sensible al rol: rhythm_consistency premia la        ║
║       regularidad en bajo/acompañamiento y la variedad en melodía.          ║
║  [M4] Modelo autoregresivo: el forward recibe el pitch y duración de la     ║
║       nota anterior, eliminando saltos incoherentes entre eventos.          ║
║  [M5] Contexto derecho enriquecido: el vector de features incluye IOI       ║
║       del lado derecho y los pitches absolutos de borde izq./dcho.         ║
║  [M6] Encoders duales con atención: contextos izq./dcho. codificados por   ║
║       separado y combinados mediante una capa de atención aditiva simple.   ║
║  [M7] Modelos por rol (--role melody|inner|bass|all): entrena modelos       ║
║       especializados por función orquestal. En inferencia se selecciona     ║
║       automáticamente el modelo más adecuado para cada pista.               ║
║  [M8] Condicionamiento armónico explícito: los chord_pcs del compás se      ║
║       añaden como one-hot de 12 dims al vector de features (desactivable    ║
║       con --no-chord-conditioning).                                          ║
║  [M9] Augmentation de corpus: transposición cromática (--augment-transpose  ║
║       N genera ±N copias en semitonos aleatorios) e inversión temporal      ║
║       dentro de compás (--augment-reverse).                                  ║
║                                                                              ║
║  DEPENDENCIAS: pip install mido numpy (+ torch para ml_guided/train)        ║
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

VERSION = "2.0"

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
        anchor_pcs: Optional[List[int]] = None,
        style: Optional['TrackStyle'] = None,
        use_chord_conditioning: bool = True) -> List[RawNote]:
    """
    Generación autoregresiva guiada por el modelo InpainterNet v2.
    M4: pasa pitch y duración previos en cada paso.
    M7: selecciona el modelo adecuado según el rol de la pista (si existen
        modelos especializados en model_data['role_models']).
    M8: incluye chord_pcs en el vector de features si use_chord_conditioning.
    """
    if not TORCH_OK or model_data is None:
        return _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

    try:
        # M7: seleccionar modelo por rol si existen modelos especializados
        track_role = style.role if style and hasattr(style, 'role') else 'unknown'
        role_models = model_data.get('role_models', {})
        if track_role in role_models:
            active_data = role_models[track_role]
        elif 'bass' in role_models and track_role in ('bass_melodic', 'bass_root'):
            active_data = role_models['bass']
        else:
            active_data = model_data

        model = InpainterNet(active_data['config'])
        model.load_state_dict(active_data['state_dict'])
        model.eval()

        target_beats = n_bars * bpb
        feat = _build_inpainter_features(
            left_notes, right_notes, key_pc, key_mode, n_bars, bpb,
            style=style,
            chord_pcs=anchor_pcs,
            use_chord_conditioning=use_chord_conditioning)
        feat_t = torch.FloatTensor(feat).unsqueeze(0)

        # Estado inicial autoregresivo (M4)
        if left_notes:
            prev_p = sorted(left_notes, key=lambda n: n.offset)[-1].pitch / 127.0
            prev_d = sorted(left_notes, key=lambda n: n.offset)[-1].duration / 4.0
        else:
            prev_p = (key_pc + 60) / 127.0
            prev_d = 0.25

        pitch_pool = style.pitch_pool if style else []
        scale_pcs  = set(_get_scale_pcs(key_pc, key_mode))
        dur_vals   = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.125, 0.333]

        notes: List[RawNote] = []
        cursor = 0.0

        with torch.no_grad():
            while cursor < target_beats:
                pos_t      = torch.FloatTensor([[cursor / max(target_beats, 1)]])
                prev_p_t   = torch.FloatTensor([[float(np.clip(prev_p, 0, 1))]])
                prev_d_t   = torch.FloatTensor([[float(np.clip(prev_d, 0, 1))]])

                pitch_logits, dur_logits, vel_out = model(
                    feat_t, pos_t, prev_p_t, prev_d_t)

                # Samplear pitch con temperatura 0.8
                pitch_probs = torch.softmax(pitch_logits[0] / 0.8, dim=-1).numpy()
                pitch_idx   = rng.choices(range(PITCH_BINS),
                                           weights=pitch_probs.tolist())[0]
                raw_pitch   = 36 + pitch_idx
                if pitch_pool:
                    new_pitch = min(pitch_pool,
                                    key=lambda p: (abs(p - raw_pitch),
                                                   0 if p % 12 in scale_pcs else 1))
                else:
                    new_pitch = _snap_to_scale(
                        max(36, min(96, raw_pitch)), key_pc, key_mode)

                # Samplear duración
                dur_probs  = torch.softmax(dur_logits[0], dim=-1).numpy()
                dur_idx    = rng.choices(range(len(dur_vals)),
                                          weights=dur_probs[:len(dur_vals)].tolist())[0]
                chosen_dur = dur_vals[dur_idx]

                vel = int(np.clip(float(vel_out[0, 0]) * 127 + rng.randint(-5, 5),
                                   30, 120))
                dur = min(chosen_dur, target_beats - cursor)
                if dur < 0.05:
                    break

                notes.append(RawNote(new_pitch, dur, vel, cursor))
                cursor += chosen_dur
                # Actualizar estado previo (M4)
                prev_p = new_pitch / 127.0
                prev_d = min(chosen_dur / 4.0, 1.0)

        return notes if notes else _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

    except Exception:
        return _generate_notes_contour_blend(
            left_notes, right_notes, n_bars, bpb, key_pc, key_mode,
            rng, ioi_hist, anchor_pcs)

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — COORDINACIÓN MULTI-PISTA
# ══════════════════════════════════════════════════════════════════════════════

def fill_gap_multitrack(ctx: PieceContext, gap: GapRegion,
                         strategy: str = 'auto', seed: int = 42,
                         length_mode: str = 'keep', length_bars: int = 0,
                         model_data: Optional[Dict] = None,
                         use_chord_conditioning: bool = True,
                         context_bars: int = 8) -> Dict[int, List[RawNote]]:
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
                                      bpb, n_context_bars=context_bars)
        try:
            style.role = ti.role  # type: ignore
        except Exception:
            pass

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
                rng, model_data, ioi_hist, anchor_pcs,
                style=style,
                use_chord_conditioning=use_chord_conditioning)
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
               filled_notes: Dict[int, List[RawNote]],
               score_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Puntúa el relleno de un hueco según cuatro criterios.

    M3: rhythm_consistency es sensible al rol de la pista.
        Para melody: premia la diversidad rítmica.
        Para bass/inner: premia la regularidad (consistencia con el ostinato).

    score_weights: dict opcional con pesos para cada criterio.
        Claves: 'tonal', 'boundary', 'tension', 'rhythm'. Deben sumar 1.0.
    """
    W = score_weights or {}
    w_tonal    = W.get('tonal',    0.35)
    w_boundary = W.get('boundary', 0.30)
    w_tension  = W.get('tension',  0.20)
    w_rhythm   = W.get('rhythm',   0.15)
    # Renormalizar por si no suman 1
    total_w = w_tonal + w_boundary + w_tension + w_rhythm
    if total_w > 0:
        w_tonal /= total_w; w_boundary /= total_w
        w_tension /= total_w; w_rhythm /= total_w

    bpb = ctx.bpb
    key_pc = gap.context_key_pc
    key_mode = gap.context_key_mode
    scale_pcs = set(_get_scale_pcs(key_pc, key_mode))
    gap_beat_lo = (gap.bar_start - 1) * bpb
    gap_beat_hi = gap.bar_end * bpb

    all_filled: List[RawNote] = [n for notes in filled_notes.values() for n in notes]

    if not all_filled:
        return {'tonal_coherence': 0.0, 'boundary_smooth': 0.0,
                'tension_continuity': 0.0, 'rhythm_consistency': 0.0,
                'total': 0.0}

    # 1. Coherencia tonal
    in_scale = sum(1 for n in all_filled if n.pitch % 12 in scale_pcs)
    tonal_coherence = in_scale / len(all_filled)

    # 2. Suavidad de uniones
    smooth_scores = []
    for ti in ctx.tracks:
        fn = filled_notes.get(ti.idx, [])
        if not fn:
            continue
        left_notes  = [n for n in ti.notes if n.offset < gap_beat_lo]
        right_notes = [n for n in ti.notes if n.offset >= gap_beat_hi]
        fn_sorted   = sorted(fn, key=lambda n: n.offset)

        if left_notes and fn_sorted:
            dist = abs(sorted(left_notes, key=lambda n: n.offset)[-1].pitch
                       - fn_sorted[0].pitch)
            smooth_scores.append(max(0.0, 1.0 - dist / 24.0))
        if right_notes and fn_sorted:
            dist = abs(fn_sorted[-1].pitch
                       - sorted(right_notes, key=lambda n: n.offset)[0].pitch)
            smooth_scores.append(max(0.0, 1.0 - dist / 24.0))

    boundary_smooth = float(np.mean(smooth_scores)) if smooth_scores else 0.5

    # 3. Continuidad de tensión
    t_fill     = _harmonic_tension(all_filled)
    t_expected = (gap.context_tension_left + gap.context_tension_right) / 2.0
    tension_continuity = max(0.0, 1.0 - abs(t_fill - t_expected) * 2.0)

    # 4. Consistencia rítmica — sensible al rol (M3)
    rhythm_scores = []
    for ti in ctx.tracks:
        fn = filled_notes.get(ti.idx, [])
        if not fn:
            continue
        role = ti.role
        durs = [n.duration for n in fn]
        if len(durs) < 2:
            rhythm_scores.append(0.5)
            continue
        dur_std = float(np.std(durs))

        if role in ('melody',):
            # Melody: diversidad es buena
            rhythm_scores.append(min(1.0, dur_std / 0.5))
        else:
            # Bass, inner, harmony: regularidad es buena
            # Comparar IOI modal del relleno con IOI modal del contexto
            left_ctx = [n for n in ti.notes if n.offset < gap_beat_lo]
            if left_ctx:
                style = _analyze_track_style(left_ctx, bpb)
                ctx_modal = style.ioi_mode
                fill_iois = [fn[i+1].offset - fn[i].offset
                             for i in range(len(fn)-1)
                             if fn[i+1].offset - fn[i].offset > 0.02]
                if fill_iois:
                    fill_modal = Counter(round(x * 8) / 8
                                         for x in fill_iois).most_common(1)[0][0]
                    match = 1.0 if abs(fill_modal - ctx_modal) < 0.125 else \
                            max(0.0, 1.0 - abs(fill_modal - ctx_modal))
                    rhythm_scores.append(match)
                else:
                    rhythm_scores.append(0.5)
            else:
                rhythm_scores.append(0.5)

    rhythm_consistency = float(np.mean(rhythm_scores)) if rhythm_scores else 0.5

    total = (tonal_coherence    * w_tonal    +
             boundary_smooth    * w_boundary +
             tension_continuity * w_tension  +
             rhythm_consistency * w_rhythm)

    return {
        'tonal_coherence':    round(tonal_coherence, 3),
        'boundary_smooth':    round(boundary_smooth, 3),
        'tension_continuity': round(tension_continuity, 3),
        'rhythm_consistency': round(rhythm_consistency, 3),
        'total':              round(total, 3),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 8 — MODELO PYTORCH (InpainterNet)
# ══════════════════════════════════════════════════════════════════════════════

FEAT_DIM   = 56   # v2: 40 base + 8 estilo rítmico + 8 IOI dcho + 12 chord one-hot → 68 total
                  # con --no-chord-conditioning: 56 dims
CHORD_FEAT = 12   # dims adicionales para condicionamiento armónico (M8)
PITCH_BINS = 61   # 36..96 en semitonos
DUR_BINS   = 8
HIDDEN_DIM = 128

# Mapeo de rol a índice para la feature de rol de pista (M1)
ROLE_IDX = {'melody': 0, 'inner': 1, 'harmony': 2,
             'bass_melodic': 3, 'bass_root': 4, 'unknown': 5}

def _build_inpainter_features(
        left_notes: List[RawNote], right_notes: List[RawNote],
        key_pc: int, key_mode: str,
        n_bars: int, bpb: int,
        style: Optional['TrackStyle'] = None,
        chord_pcs: Optional[List[int]] = None,
        use_chord_conditioning: bool = True) -> np.ndarray:
    """
    Vector de features que describe el contexto de un hueco.

    Dims base (56):
      [0-11]   histograma PC ponderado por duración — contexto izquierdo
      [12-23]  histograma PC ponderado por duración — contexto derecho
      [24-31]  IOI histogram del contexto izquierdo            (M5)
      [32-39]  IOI histogram del contexto derecho              (M5)
      [40]     tónica normalizada (0..1)
      [41]     modo (0=mayor, 1=menor)
      [42]     longitud del hueco (normalizada 0..1)
      [43]     tensión izquierda
      [44]     tensión derecha
      [45]     densidad izquierda
      [46]     densidad derecha
      [47]     pitch absoluto borde izquierdo (normalizado)    (M5)
      [48]     pitch absoluto borde derecho  (normalizado)     (M5)
      [49]     IOI modal de la pista (normalizado)             (M1)
      [50]     regularidad rítmica de la pista (0..1)         (M1)
      [51]     notas/compás normalizadas                       (M1)
      [52]     grado de polifonía (0..1)                      (M1)
      [53-55]  rol de pista one-hot reducido (melody/inner/bass) (M1)

    Dims opcionales (12, añadidas si use_chord_conditioning=True):
      [56-67]  chord PCs one-hot del compás actual             (M8)
    """
    dim = FEAT_DIM + (CHORD_FEAT if use_chord_conditioning else 0)
    feat = np.zeros(dim, dtype=np.float32)

    # ── PC histograms ─────────────────────────────────────────────────────────
    if left_notes:
        for n in left_notes[-48:]:
            feat[n.pitch % 12] += n.duration
        s = feat[:12].sum()
        if s > 0: feat[:12] /= s

    if right_notes:
        for n in right_notes[:48]:
            feat[12 + n.pitch % 12] += n.duration
        s = feat[12:24].sum()
        if s > 0: feat[12:24] /= s

    # ── IOI histograms izq. y dcho. (M5) ─────────────────────────────────────
    feat[24:32] = _build_ioi_histogram(left_notes)  if left_notes  else np.ones(8)/8
    feat[32:40] = _build_ioi_histogram(right_notes) if right_notes else np.ones(8)/8

    # ── Escalares básicos ─────────────────────────────────────────────────────
    feat[40] = key_pc / 12.0
    feat[41] = 0.0 if key_mode == 'major' else 1.0
    feat[42] = min(1.0, n_bars / 16.0)
    feat[43] = _harmonic_tension(left_notes[-16:])  if left_notes  else 0.5
    feat[44] = _harmonic_tension(right_notes[:16])  if right_notes else 0.5
    feat[45] = min(1.0, len(left_notes)  / max(n_bars * bpb * 2, 1))
    feat[46] = min(1.0, len(right_notes) / max(n_bars * bpb * 2, 1))

    # ── Pitches absolutos de borde (M5) ──────────────────────────────────────
    if left_notes:
        feat[47] = sorted(left_notes,  key=lambda n: n.offset)[-1].pitch / 127.0
    if right_notes:
        feat[48] = sorted(right_notes, key=lambda n: n.offset)[0].pitch  / 127.0

    # ── Features de estilo rítmico (M1) ──────────────────────────────────────
    if style is not None:
        feat[49] = min(1.0, style.ioi_mode / 4.0)
        feat[50] = style.ioi_regularity
        feat[51] = min(1.0, style.notes_per_bar / 20.0)
        feat[52] = style.polyphony
        # Rol: melody=0, inner=1, bass=2 (reducido a 3 clases)
        role_idx = ROLE_IDX.get(style.role if hasattr(style, 'role') else 'unknown', 5)
        reduced  = min(2, role_idx // 2)   # 0→melody, 1→inner, 2→bass
        feat[53 + reduced] = 1.0

    # ── Condicionamiento armónico one-hot (M8) ────────────────────────────────
    if use_chord_conditioning and chord_pcs:
        for pc in chord_pcs:
            feat[FEAT_DIM + (pc % 12)] = 1.0

    return feat


if TORCH_OK:
    class InpainterNet(nn.Module):
        """
        Modelo autoregresivo con encoders duales y atención (v2).

        Arquitectura (M4, M6):
          - LeftEncoder:  MLP sobre features del contexto izquierdo
          - RightEncoder: MLP sobre features del contexto derecho
          - Atención aditiva: peso adaptativo entre ambos encoders según
            la posición relativa en el hueco (izq pesa más al inicio,
            dcho pesa más al final)
          - Autoregresivo: recibe (pitch_prev, dur_prev) como input
            adicional para evitar saltos incoherentes entre eventos (M4)
          - Heads: pitch (clasificación), duración (clasificación),
            velocidad (regresión sigmoide)
        """
        def __init__(self, config: Dict):
            super().__init__()
            feat_dim   = config.get('feat_dim', FEAT_DIM)
            hidden_dim = config.get('hidden_dim', HIDDEN_DIM)
            pitch_bins = config.get('pitch_bins', PITCH_BINS)
            dur_bins   = config.get('dur_bins', DUR_BINS)
            self.config = config

            half = feat_dim // 2   # mitad izq / mitad dcha del vector

            # Encoders separados para contexto izq. y dcho. (M6)
            enc_out = hidden_dim // 2
            self.left_enc = nn.Sequential(
                nn.Linear(half, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, enc_out),
                nn.LayerNorm(enc_out),
                nn.ReLU(),
            )
            self.right_enc = nn.Sequential(
                nn.Linear(feat_dim - half, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, enc_out),
                nn.LayerNorm(enc_out),
                nn.ReLU(),
            )

            # Capa de atención aditiva (M6): combina left/right según posición
            self.attn_w = nn.Linear(enc_out * 2 + 1, 2)  # +1 para posición

            # Input de la capa de fusión:
            #   enc_out (contexto fusionado) + 1 (pos) + 2 (prev_pitch, prev_dur)
            fusion_in = enc_out + 1 + 2   # (M4: +2 para autoregresivo)
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )

            h = hidden_dim // 2
            self.pitch_head = nn.Linear(h, pitch_bins)
            self.dur_head   = nn.Linear(h, dur_bins)
            self.vel_head   = nn.Sequential(nn.Linear(h, 1), nn.Sigmoid())

        def forward(self, feat: 'torch.Tensor', pos: 'torch.Tensor',
                    prev_pitch: 'torch.Tensor', prev_dur: 'torch.Tensor'):
            """
            feat:       [B, feat_dim]
            pos:        [B, 1]   posición relativa en el hueco (0..1)
            prev_pitch: [B, 1]   pitch anterior normalizado (0..1)  — M4
            prev_dur:   [B, 1]   duración anterior normalizada      — M4
            """
            half = feat.shape[-1] // 2
            left_h  = self.left_enc(feat[:, :half])
            right_h = self.right_enc(feat[:, half:])

            # Atención aditiva: peso según posición (M6)
            attn_in  = torch.cat([left_h, right_h, pos], dim=-1)
            attn_w   = torch.softmax(self.attn_w(attn_in), dim=-1)  # [B, 2]
            ctx      = attn_w[:, 0:1] * left_h + attn_w[:, 1:2] * right_h

            # Fusión con posición y estado previo (M4)
            fused = self.fusion(torch.cat([ctx, pos, prev_pitch, prev_dur], dim=-1))
            return self.pitch_head(fused), self.dur_head(fused), self.vel_head(fused)


    class InpainterDataset(Dataset):
        """Dataset de pares (features, prev_state, target_note) para entrenamiento."""
        def __init__(self, samples: List[Dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            return (torch.FloatTensor(s['feat']),
                    torch.FloatTensor([s['pos']]),
                    torch.FloatTensor([s['prev_pitch']]),   # M4
                    torch.FloatTensor([s['prev_dur']]),     # M4
                    torch.LongTensor([s['pitch_bin']]),
                    torch.LongTensor([s['dur_bin']]),
                    torch.FloatTensor([s['vel']]))


def _detect_phrase_boundaries(all_notes: List[RawNote], bpb: int,
                                n_bars: int) -> List[int]:
    """
    M2: detecta límites de frase por caídas de densidad.
    Devuelve lista de números de compás donde empieza una nueva frase.
    """
    if n_bars < 4:
        return []
    densities = []
    for bar in range(1, n_bars + 1):
        lo = (bar - 1) * bpb
        hi = bar * bpb
        densities.append(len([n for n in all_notes if lo <= n.offset < hi]) / bpb)

    boundaries = []
    window = max(2, n_bars // 16)
    for i in range(window, n_bars - window):
        left_mean  = sum(densities[max(0, i-window):i]) / window
        right_mean = sum(densities[i:min(n_bars, i+window)]) / window
        # Cambio de densidad significativo → límite de frase
        if abs(right_mean - left_mean) > 0.3 * max(left_mean, right_mean, 0.1):
            if not boundaries or i - boundaries[-1] >= 2:
                boundaries.append(i + 1)   # +1: 1-indexed
    return boundaries


def _augment_notes_transpose(notes: List[RawNote], semitones: int,
                               key_pc: int) -> Tuple[List[RawNote], int]:
    """M9: transpone todas las notas por N semitonos, ajustando key_pc."""
    transposed = [RawNote(
        pitch    = max(21, min(108, n.pitch + semitones)),
        duration = n.duration,
        velocity = n.velocity,
        offset   = n.offset,
        track_idx= n.track_idx,
    ) for n in notes]
    return transposed, (key_pc + semitones) % 12


def _augment_notes_reverse_bars(notes: List[RawNote], bpb: int,
                                  gap_beat_lo: float,
                                  gap_beat_hi: float) -> List[RawNote]:
    """
    M9: invierte temporalmente las notas dentro de cada compás del hueco.
    El contorno global se mantiene pero el orden interno se invierte,
    generando variación sin perder el vocabulario de pitches.
    """
    result = []
    bar_lo = gap_beat_lo
    while bar_lo < gap_beat_hi:
        bar_hi = min(bar_lo + bpb, gap_beat_hi)
        bar_notes = [n for n in notes if bar_lo <= n.offset < bar_hi]
        if bar_notes:
            # Invertir posición relativa dentro del compás
            rev = []
            for n in bar_notes:
                new_offset = bar_lo + (bar_hi - bar_lo) - (n.offset - bar_lo) - n.duration
                rev.append(RawNote(n.pitch, n.duration, n.velocity,
                                   max(bar_lo, new_offset), n.track_idx))
            result.extend(sorted(rev, key=lambda x: x.offset))
        bar_lo = bar_hi
    return result


def _role_from_notes(notes: List[RawNote]) -> str:
    """Infiere el rol de una pista por registro medio."""
    if not notes:
        return 'unknown'
    mid = sum(n.pitch for n in notes) / len(notes)
    if mid >= 70:   return 'melody'
    if mid >= 55:   return 'inner'
    if mid >= 45:   return 'bass_melodic'
    return 'bass_root'


def extract_training_samples(
        midi_path: str,
        mask_ratio: float = 0.2,
        bpb_override: int = 0,
        mask_phrase: bool = False,
        augment_transpose: int = 0,
        augment_reverse: bool = False,
        role_filter: Optional[str] = None,
        use_chord_conditioning: bool = True) -> List[Dict]:
    """
    Extrae muestras de entrenamiento de un MIDI.

    mask_phrase (M2): si True, coloca los huecos en límites de frase
                      detectados automáticamente.
    augment_transpose (M9): genera ±N copias transpuestas.
    augment_reverse (M9):   añade versión con inversión temporal intra-compás.
    role_filter (M7):       filtra pistas por rol ('melody'|'inner'|'bass'|None).
    use_chord_conditioning (M8): incluye chord one-hot en el vector de features.
    """
    try:
        mid = MidiFile(midi_path)
        tpb = mid.ticks_per_beat
        bpb = bpb_override or 4
        for msg in mid.tracks[0]:
            if msg.type == 'set_tempo':        pass
            elif msg.type == 'time_signature': bpb = msg.numerator

        # Parsear por pista para mantener info de rol
        track_notes: Dict[int, List[RawNote]] = {}
        for i, track in enumerate(mid.tracks):
            ns = _parse_track(track, tpb)
            if ns:
                track_notes[i] = ns

        if not track_notes:
            return []

        all_notes = [n for ns in track_notes.values() for n in ns]
        key_pc, key_mode = _detect_key(all_notes)
        max_offset = max(n.offset + n.duration for n in all_notes)
        n_bars = max(1, math.ceil(max_offset / bpb))

        if n_bars < 8:
            return []

        # M2: límites de frase para enmascaramiento dirigido
        phrase_bounds = _detect_phrase_boundaries(all_notes, bpb, n_bars) \
                        if mask_phrase else []

        rng = random.Random(hash(midi_path) % (2**31))
        samples = []

        def _process_single(notes_by_track: Dict[int, List[RawNote]],
                             k_pc: int, k_mode: str) -> List[Dict]:
            """Genera muestras para una versión (original o augmentada)."""
            flat = [n for ns in notes_by_track.values() for n in ns]
            s_list = []

            # Determinar posiciones de huecos
            if mask_phrase and phrase_bounds:
                # M2: colocar huecos en/cerca de límites de frase
                gap_starts = []
                for pb in phrase_bounds:
                    if 4 <= pb <= n_bars - 5:
                        gap_starts.append(pb)
                # Completar con aleatorios si hay pocas frases
                n_extra = max(0, max(1, int(n_bars * mask_ratio / 2)) - len(gap_starts))
                for _ in range(n_extra):
                    gap_starts.append(rng.randint(4, max(5, n_bars - 5)))
            else:
                n_gaps = max(1, int(n_bars * mask_ratio / 2))
                gap_starts = [rng.randint(4, max(5, n_bars - 5))
                              for _ in range(n_gaps)]

            for gap_start in gap_starts:
                gap_len = rng.randint(1, min(4, n_bars // 4))
                gap_beat_lo = (gap_start - 1) * bpb
                gap_beat_hi = (gap_start + gap_len - 1) * bpb

                for track_idx, t_notes in notes_by_track.items():
                    # M7: filtrar por rol
                    t_role = _role_from_notes(t_notes)
                    if role_filter and role_filter != 'all':
                        if role_filter == 'bass' and t_role not in ('bass_melodic', 'bass_root'):
                            continue
                        elif role_filter not in ('bass',) and t_role != role_filter:
                            continue

                    masked = [n for n in t_notes
                              if gap_beat_lo <= n.offset < gap_beat_hi]
                    if not masked:
                        continue

                    left_n  = [n for n in t_notes if n.offset < gap_beat_lo]
                    right_n = [n for n in t_notes if n.offset >= gap_beat_hi]

                    # Construir estilo para M1
                    style_obj = _analyze_track_style(left_n, bpb, n_context_bars=8)
                    # Añadir rol al style_obj dinámicamente
                    object.__setattr__(style_obj, 'role', t_role) \
                        if hasattr(style_obj, '__dataclass_fields__') else None
                    try:
                        style_obj.role = t_role  # type: ignore
                    except Exception:
                        pass

                    # M8: chord PCs de pistas ancla en el hueco
                    chord_pcs_gap: List[int] = []
                    for other_idx, other_notes in notes_by_track.items():
                        if other_idx != track_idx:
                            chord_pcs_gap.extend(
                                n.pitch % 12 for n in other_notes
                                if gap_beat_lo <= n.offset < gap_beat_hi)
                    chord_pcs_gap = list(set(chord_pcs_gap))

                    feat = _build_inpainter_features(
                        left_n[-48:], right_n[:48], k_pc, k_mode,
                        gap_len, bpb,
                        style=style_obj,
                        chord_pcs=chord_pcs_gap,
                        use_chord_conditioning=use_chord_conditioning)

                    # Estado anterior para autoregresivo (M4)
                    sorted_masked = sorted(masked, key=lambda n: n.offset)
                    prev_pitch_abs = left_n[-1].pitch if left_n else 60
                    prev_dur_val   = left_n[-1].duration if left_n else 1.0

                    dur_vals = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.125, 0.333]

                    for n in sorted_masked:
                        pos = (n.offset - gap_beat_lo) / max(gap_len * bpb, 1)
                        pitch_bin = max(0, min(PITCH_BINS - 1, n.pitch - 36))
                        dur_bin = min(range(len(dur_vals)),
                                      key=lambda i: abs(dur_vals[i] - n.duration))
                        s_list.append({
                            'feat':       feat,
                            'pos':        float(np.clip(pos, 0.0, 1.0)),
                            'prev_pitch': float(prev_pitch_abs / 127.0),  # M4
                            'prev_dur':   float(min(prev_dur_val / 4.0, 1.0)),  # M4
                            'pitch_bin':  pitch_bin,
                            'dur_bin':    dur_bin,
                            'vel':        n.velocity / 127.0,
                        })
                        prev_pitch_abs = n.pitch
                        prev_dur_val   = n.duration

            return s_list

        # ── Original ──────────────────────────────────────────────────────────
        samples.extend(_process_single(track_notes, key_pc, key_mode))

        # ── M9: augmentation por transposición ────────────────────────────────
        if augment_transpose > 0:
            semitones_list = list(range(-augment_transpose, augment_transpose + 1))
            semitones_list = [s for s in semitones_list if s != 0]
            # Limitar a máximo 8 transposiciones para no inflar demasiado
            if len(semitones_list) > 8:
                semitones_list = rng.sample(semitones_list, 8)
            for semitones in semitones_list:
                transposed_tracks = {}
                new_kpc = key_pc
                for tid, tnotes in track_notes.items():
                    t_trans, new_kpc = _augment_notes_transpose(tnotes, semitones, key_pc)
                    transposed_tracks[tid] = t_trans
                samples.extend(_process_single(transposed_tracks, new_kpc, key_mode))

        # ── M9: augmentation por inversión temporal ───────────────────────────
        if augment_reverse:
            # Invertir dentro de cada compás para todos los huecos de la pasada original
            # (se procesa como una nueva versión independiente del MIDI)
            reversed_tracks = {}
            for tid, tnotes in track_notes.items():
                rev_notes = []
                for bar in range(1, n_bars + 1):
                    bar_lo = (bar - 1) * bpb
                    bar_hi = bar * bpb
                    bar_notes = [n for n in tnotes if bar_lo <= n.offset < bar_hi]
                    if bar_notes:
                        rev_notes.extend(
                            _augment_notes_reverse_bars(
                                bar_notes, bpb, bar_lo, bar_hi))
                    else:
                        pass  # compases vacíos se mantienen vacíos
                reversed_tracks[tid] = sorted(rev_notes, key=lambda n: n.offset)
            samples.extend(_process_single(reversed_tracks, key_pc, key_mode))

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

    use_chord = not getattr(args, 'no_chord_conditioning', False)
    ctx_bars  = getattr(args, 'context_bars', 8)

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
            model_data=model_data,
            use_chord_conditioning=use_chord,
            context_bars=ctx_bars)

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

    use_chord = not getattr(args, 'no_chord_conditioning', False)
    ctx_bars  = getattr(args, 'context_bars', 8)

    # Parsear score_weights si se proporcionaron
    score_weights: Optional[Dict[str, float]] = None
    if getattr(args, 'score_weights', None):
        score_weights = {}
        for token in args.score_weights:
            if '=' in token:
                k, v = token.split('=', 1)
                score_weights[k.strip()] = float(v.strip())

    print(f'  Generando {len(combos)} variantes para {len(gaps)} hueco(s)...')

    for v_idx, (strat, seed) in enumerate(combos):
        fills: Dict[int, Dict[int, List[RawNote]]] = {}
        gap_scores = []

        for g in gaps:
            f = fill_gap_multitrack(ctx, g, strategy=strat, seed=seed,
                                     length_mode=length_mode,
                                     length_bars=length_bars,
                                     model_data=model_data,
                                     use_chord_conditioning=use_chord,
                                     context_bars=ctx_bars)
            fills[g.gap_id] = f
            sc = score_fill(ctx, g, f, score_weights=score_weights)
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
    """Modo train: entrena InpainterNet v2 sobre corpus de MIDIs."""
    if not TORCH_OK:
        print('ERROR: PyTorch no disponible. pip install torch'); sys.exit(1)

    corpus_dir = Path(args.corpus)
    midi_files = list(corpus_dir.rglob('*.mid')) + list(corpus_dir.rglob('*.midi'))
    if not midi_files:
        print(f'ERROR: no se encontraron MIDIs en {corpus_dir}'); sys.exit(1)

    use_chord = not args.no_chord_conditioning
    feat_dim  = FEAT_DIM + (CHORD_FEAT if use_chord else 0)
    role      = getattr(args, 'role', 'all')

    print(f'\n  Extrayendo muestras de {len(midi_files)} MIDIs...')
    print(f'  mask_phrase={args.mask_phrase}  '
          f'augment_transpose={args.augment_transpose}  '
          f'augment_reverse={args.augment_reverse}  '
          f'role={role}  chord_conditioning={use_chord}')

    all_samples = []
    for mf in midi_files:
        s = extract_training_samples(
            str(mf),
            mask_ratio=args.mask_ratio,
            mask_phrase=args.mask_phrase,
            augment_transpose=args.augment_transpose,
            augment_reverse=args.augment_reverse,
            role_filter=role if role != 'all' else None,
            use_chord_conditioning=use_chord,
        )
        all_samples.extend(s)
        if len(s) > 0:
            print(f'  {mf.name}: {len(s)} muestras')

    if not all_samples:
        print('ERROR: no se pudieron extraer muestras del corpus.'); sys.exit(1)

    print(f'\n  Total muestras : {len(all_samples)}')
    print(f'  feat_dim       : {feat_dim}')
    print(f'  Entrenando InpainterNet v2 '
          f'(hidden={args.hidden_dim}, epochs={args.epochs}, lr={args.lr})...')

    config = {'feat_dim': feat_dim, 'hidden_dim': args.hidden_dim,
              'pitch_bins': PITCH_BINS, 'dur_bins': DUR_BINS}
    model   = InpainterNet(config)
    dataset = InpainterDataset(all_samples)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    ce  = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0; batches = 0
        for feat, pos, prev_p, prev_d, pitch_tgt, dur_tgt, vel_tgt in loader:
            optimizer.zero_grad()
            pitch_logits, dur_logits, vel_out = model(feat, pos, prev_p, prev_d)
            loss = (ce(pitch_logits, pitch_tgt.squeeze(1)) +
                    ce(dur_logits,   dur_tgt.squeeze(1))   +
                    mse(vel_out.squeeze(-1), vel_tgt.squeeze(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / max(batches, 1)
            print(f'  Época {epoch:4d}/{args.epochs}  loss={avg:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

    # Guardar — incluyendo rol si es especializado (M7)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_key = f'inpainter_{role}.pkl' if role != 'all' else 'inpainter.pkl'
    ckpt_path = ckpt_dir / model_key

    data = {'state_dict': model.state_dict(), 'config': config,
            'n_samples': len(all_samples), 'epochs': args.epochs,
            'version': VERSION, 'role': role,
            'use_chord_conditioning': use_chord}

    # Se los roles especializados se añaden dentro del fichero principal
    # para que _load_model pueda seleccionarlos automáticamente (M7)
    main_ckpt = ckpt_dir / 'inpainter.pkl'
    if main_ckpt.exists() and role != 'all':
        with open(main_ckpt, 'rb') as f:
            main_data = pickle.load(f)
        if 'role_models' not in main_data:
            main_data['role_models'] = {}
        main_data['role_models'][role] = {
            'state_dict': model.state_dict(),
            'config': config,
        }
        with open(main_ckpt, 'wb') as f:
            pickle.dump(main_data, f)
        print(f'\n  ✓ Modelo rol={role} añadido a: {main_ckpt}')
    else:
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
    print(f'  Versión             : {data.get("version", "?")}')
    print(f'  Épocas              : {data.get("epochs", "?")}')
    print(f'  Muestras            : {data.get("n_samples", "?")}')
    print(f'  Rol                 : {data.get("role", "all")}')
    print(f'  feat_dim            : {cfg.get("feat_dim", "?")}')
    print(f'  hidden_dim          : {cfg.get("hidden_dim", "?")}')
    print(f'  chord_conditioning  : {data.get("use_chord_conditioning", "?")}')
    role_models = data.get('role_models', {})
    if role_models:
        print(f'  Modelos por rol     : {", ".join(role_models.keys())}')
    if TORCH_OK and 'state_dict' in data:
        model = InpainterNet(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Parámetros          : {n_params:,}')
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
        description='Relleno inteligente de compases vacíos en MIDIs multi-pista (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='mode', required=True)

    # ── scan ──────────────────────────────────────────────────────────────────
    ps = sub.add_parser('scan', help='Detecta huecos y genera reporte')
    ps.add_argument('input', help='MIDI de entrada')
    ps.add_argument('--min-density', type=float, default=0.0,
                    help='Umbral de densidad mínima (def: 0.0 = solo vacíos)')
    ps.add_argument('--report', default=None,
                    help='Guardar reporte en fichero .txt')

    # ── fill ──────────────────────────────────────────────────────────────────
    pf = sub.add_parser('fill', help='Rellena huecos (una salida)')
    pf.add_argument('input', help='MIDI de entrada')
    pf.add_argument('--gaps', type=int, nargs='+', default=None,
                    help='IDs de huecos a rellenar (default: todos)')
    pf.add_argument('--strategy', default='auto',
                    choices=['auto', 'markov_bi', 'interpolate',
                             'contour_blend', 'ml_guided'],
                    help='Estrategia de generación (def: auto)')
    pf.add_argument('--length', default='keep',
                    help='Longitud: keep | auto | N (nº compases)')
    pf.add_argument('--min-density', type=float, default=0.0)
    pf.add_argument('--checkpoint-dir', default=None,
                    help='Directorio con modelo entrenado')
    pf.add_argument('--seed', type=int, default=42)
    pf.add_argument('--no-chord-conditioning', action='store_true',
                    help='M8: desactiva el condicionamiento armónico explícito')
    pf.add_argument('--context-bars', type=int, default=8,
                    help='M1: compases de contexto para análisis de estilo (def: 8)')
    pf.add_argument('-o', '--output', default=None,
                    help='MIDI de salida (def: <entrada>_filled.mid)')

    # ── variants ──────────────────────────────────────────────────────────────
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
    pv.add_argument('--no-chord-conditioning', action='store_true',
                    help='M8: desactiva el condicionamiento armónico explícito')
    pv.add_argument('--context-bars', type=int, default=8,
                    help='M1: compases de contexto para análisis de estilo (def: 8)')
    pv.add_argument('--score-weights', nargs='+', default=None,
                    metavar='KEY=VAL',
                    help='M3: pesos del scoring, ej: tonal=0.4 boundary=0.3 '
                         'tension=0.2 rhythm=0.1')
    pv.add_argument('--out-dir', default='./inpainter_variants',
                    help='Directorio de salida (def: ./inpainter_variants)')

    # ── train ──────────────────────────────────────────────────────────────────
    pt = sub.add_parser('train', help='Entrena modelo sobre corpus de MIDIs')
    pt.add_argument('corpus', help='Directorio con MIDIs de entrenamiento')
    pt.add_argument('--epochs', type=int, default=60)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--batch-size', type=int, default=64)
    pt.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM)
    pt.add_argument('--mask-ratio', type=float, default=0.2,
                    help='Fracción de compases a enmascarar por MIDI (def: 0.2)')
    pt.add_argument('--mask-phrase', action='store_true',
                    help='M2: coloca huecos en límites de frase detectados '
                         'automáticamente (más musical que aleatorio)')
    pt.add_argument('--augment-transpose', type=int, default=0,
                    metavar='N',
                    help='M9: genera ±N copias transpuestas por semitono '
                         '(def: 0 = sin augmentación)')
    pt.add_argument('--augment-reverse', action='store_true',
                    help='M9: añade versión con inversión temporal intra-compás')
    pt.add_argument('--role', default='all',
                    choices=['all', 'melody', 'inner', 'bass'],
                    help='M7: entrena solo para el rol indicado. Con "all" entrena '
                         'un modelo genérico. Ejecutar varias veces con roles distintos '
                         'añade modelos especializados al checkpoint principal.')
    pt.add_argument('--no-chord-conditioning', action='store_true',
                    help='M8: desactiva el one-hot de acorde en el vector de features')
    pt.add_argument('--checkpoint-dir', default='./inpainter_ckpt')

    # ── inspect ───────────────────────────────────────────────────────────────
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
