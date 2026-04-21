#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ML ARCHITECT  v1.0                                    ║
║     Composición estructural dirigida por gramáticas aprendidas de corpus     ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║                                                                              ║
║  ── SEGMENT ──────────────────────────────────────────────────────────────  ║
║  Analiza un MIDI y extrae sus secciones formales mediante gramática          ║
║  Sequitur + clustering espectral. Útil para inspeccionar el corpus.          ║
║                                                                              ║
║    python ml_architect.py segment  cancion.mid                              ║
║    python ml_architect.py segment  cancion.mid --num-sections 4             ║
║    python ml_architect.py segment  cancion.mid --track 1 --all-tracks       ║
║    python ml_architect.py segment  cancion.mid --fuzzy 0.85 --verbose       ║
║                                                                              ║
║  ── TRAIN ────────────────────────────────────────────────────────────────  ║
║  Entrena el modelo de transformaciones sobre un directorio de MIDIs.         ║
║  Aprende cómo cada pieza transforma su material fuente en secciones.         ║
║                                                                              ║
║    python ml_architect.py train  corpus/  --output modelo.pkl               ║
║    python ml_architect.py train  corpus/  --n-roles 5 --assignment hybrid   ║
║    python ml_architect.py train  corpus/  --num-sections 4 --verbose        ║
║                                                                              ║
║  ── GENERATE ─────────────────────────────────────────────────────────────  ║
║  Genera una obra completa a partir de frases MIDI y un modelo entrenado.     ║
║                                                                              ║
║    python ml_architect.py generate frase1.mid frase2.mid --model m.pkl      ║
║    python ml_architect.py generate *.mid --model m.pkl --n-sections 6       ║
║    python ml_architect.py generate *.mid --model m.pkl --assignment pos     ║
║    python ml_architect.py generate *.mid --model m.pkl --role-order 0 2 1 3 ║
║    python ml_architect.py generate *.mid --model m.pkl --out-dir salida/    ║
║    python ml_architect.py generate *.mid --model m.pkl --dry-run            ║
║                                                                              ║
║  MODOS DE ASIGNACIÓN DE ROLES (--assignment):                               ║
║    similarity  — roles por similitud de características al material fuente  ║
║    position    — roles por posición relativa aprendida en el corpus          ║
║    hybrid      — posición determina el orden, similitud el rol concreto      ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, scikit-learn, scipy                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import math
import copy
import json
import random
import pickle
import argparse
import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import mido
except ImportError:
    print("[ERROR] mido no encontrado. Instálalo con: pip install mido")
    sys.exit(1)

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize as sk_normalize
except ImportError:
    print("[ERROR] scikit-learn no encontrado. Instálalo con: pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
except ImportError:
    print("[ERROR] scipy no encontrado. Instálalo con: pip install scipy")
    sys.exit(1)

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MIN_NOTE_DUR = 0.125   # beats

SCALE_INTERVALS = {
    "major":    [0,2,4,5,7,9,11],
    "minor":    [0,2,3,5,7,8,10],
    "dorian":   [0,2,3,5,7,9,10],
    "phrygian": [0,1,3,5,7,8,10],
}

# Mapas de progresiones por función de sección
SECTION_PROGRESSIONS = {
    "pop": {
        0: [("I",4),("V",4),("vi",4),("IV",4)],
        1: [("I",4),("vi",4),("IV",4),("V",4)],
        2: [("IV",4),("V",4),("I",4),("vi",4)],
        3: [("I",4),("IV",4),("I",4),("I",4)],
    }
}

CHORD_INTERVALS = {
    "M": [0,4,7], "m": [0,3,7],
    "7": [0,4,7,10], "M7": [0,4,7,11], "m7": [0,3,7,10],
}

NUMERAL_MAP = {
    "I":(0,"M"),"i":(0,"m"),"II":(2,"M"),"ii":(2,"m"),
    "III":(4,"M"),"iii":(4,"m"),"IV":(5,"M"),"iv":(5,"m"),
    "V":(7,"M"),"v":(7,"m"),"VI":(9,"M"),"vi":(9,"m"),
    "VII":(11,"M"),"vii":(11,"m"),"V7":(7,"7"),"IM7":(0,"M7"),
    "vi7":(9,"m7"),"ii7":(2,"m7"),"bVII":(10,"M"),
}

# Pesos de las dimensiones del TransformVector para estimación
PITCH_OPS   = ["identity","transpose","invert","retrograde","retro_invert",
               "modal_shift","diatonic_sequence","liquidate"]
RHYTHM_OPS  = ["identity","augment","diminish","rhythmic_repattern","syncopate"]
DYNAMIC_OPS = ["identity","velocity_scale","tension_curve","ornament","add_passing_notes"]
HARMONY_OPS = ["identity","progression_select","acc_style","reharmonize"]

ACC_STYLES  = ["block","arpeggio","alberti","bass_only","waltz"]
TENSION_SHAPES = ["flat","rise","fall","arch","inverse_arch"]

# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawNote:
    """Nota en beats, formato interno común."""
    pitch:    int
    duration: float   # beats
    velocity: int
    offset:   float   # beats desde inicio del segmento

    def transpose(self, semitones: int) -> "RawNote":
        return RawNote(max(0,min(127,self.pitch+semitones)),
                       self.duration, self.velocity, self.offset)

    def at_offset(self, delta: float) -> "RawNote":
        n = deepcopy(self)
        n.offset = self.offset + delta
        return n


@dataclass
class SectionSegment:
    label:     str              # "A", "B", "C"…
    bar_start: int              # compás inicio (base 1)
    bar_end:   int              # compás fin (inclusivo)
    notes:     List[RawNote]
    symbol:    str              # símbolo Sequitur de origen


@dataclass
class SectionSignature:
    """Vector de características de un fragmento musical."""
    position_ratio: float         # [0,1] posición relativa en la pieza
    tension:        float         # [0,1]
    velocity_mean:  float         # [0,1]
    interval_hist:  np.ndarray    # [24] invariante a transposición
    rhythm_hist:    np.ndarray    # [8]
    contour_dir:    float         # [-1,1]
    density:        float         # notas/beat normalizado

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            [self.position_ratio, self.tension, self.velocity_mean,
             self.contour_dir, min(self.density/4.0, 1.0)],
            self.interval_hist,
            self.rhythm_hist,
        ])


@dataclass
class TransformVector:
    op_pitch:       str = "identity"
    param_pitch:    float = 0.0
    op_rhythm:      str = "identity"
    param_rhythm:   float = 1.0
    op_dynamics:    str = "identity"
    param_dynamics: float = 1.0
    op_harmony:     str = "identity"
    param_harmony:  float = 0.0

    def to_array(self) -> np.ndarray:
        """Representación numérica para estadísticas."""
        return np.array([
            PITCH_OPS.index(self.op_pitch)   / max(len(PITCH_OPS)-1,1),
            self.param_pitch / 12.0,
            RHYTHM_OPS.index(self.op_rhythm) / max(len(RHYTHM_OPS)-1,1),
            self.param_rhythm / 2.0,
            DYNAMIC_OPS.index(self.op_dynamics) / max(len(DYNAMIC_OPS)-1,1),
            self.param_dynamics / 1.5,
            HARMONY_OPS.index(self.op_harmony) / max(len(HARMONY_OPS)-1,1),
            self.param_harmony / max(len(ACC_STYLES)-1,1),
        ])

    @staticmethod
    def from_array(arr: np.ndarray) -> "TransformVector":
        def nearest_idx(val, n): return int(round(np.clip(val,0,1) * (n-1)))
        return TransformVector(
            op_pitch    = PITCH_OPS[nearest_idx(arr[0], len(PITCH_OPS))],
            param_pitch = float(arr[1]) * 12.0,
            op_rhythm   = RHYTHM_OPS[nearest_idx(arr[2], len(RHYTHM_OPS))],
            param_rhythm= float(arr[3]) * 2.0,
            op_dynamics = DYNAMIC_OPS[nearest_idx(arr[4], len(DYNAMIC_OPS))],
            param_dynamics = float(arr[5]) * 1.5,
            op_harmony  = HARMONY_OPS[nearest_idx(arr[6], len(HARMONY_OPS))],
            param_harmony  = float(arr[7]) * max(len(ACC_STYLES)-1,1),
        )


@dataclass
class RoleProfile:
    centroid:      SectionSignature
    theta_mean:    TransformVector
    theta_cov:     np.ndarray         # [8×8]
    position_dist: Tuple[float,float] # (mean, std)
    tension_dist:  Tuple[float,float]
    label:         str = "?"          # etiqueta descriptiva inferida


@dataclass
class TransformationModel:
    n_roles:         int
    roles:           List[RoleProfile]
    assignment_mode: str   # "similarity"|"position"|"hybrid"
    corpus_size:     int
    version:         str = VERSION


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

def _detect_key(notes: List[RawNote]) -> Tuple[int, str]:
    """Krumhansl-Schmuckler simplificado."""
    if not notes:
        return 0, "major"
    pc_weights = np.zeros(12)
    for n in notes:
        pc_weights[n.pitch % 12] += n.duration
    major_p = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_p = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    best_key, best_score, best_mode = 0, -999.0, "major"
    for root in range(12):
        rotated = np.roll(pc_weights, -root)
        for mode_name, profile in [("major",major_p),("minor",minor_p)]:
            score = float(np.corrcoef(rotated, profile)[0,1])
            if score > best_score:
                best_score = score; best_key = root; best_mode = mode_name
    return best_key, best_mode


def _get_scale_pcs(root_pc: int, mode: str) -> List[int]:
    ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return [(root_pc + i) % 12 for i in ivs]


def _snap_to_scale(pitch: int, root_pc: int, mode: str) -> int:
    scale = set(_get_scale_pcs(root_pc, mode))
    if pitch % 12 in scale:
        return pitch
    for d in range(1, 7):
        if (pitch+d) % 12 in scale: return pitch+d
        if (pitch-d) % 12 in scale: return pitch-d
    return pitch


def _numeral_to_root(numeral: str, key_pc: int) -> Tuple[int, str]:
    entry = NUMERAL_MAP.get(numeral, (0,"M"))
    return (key_pc + entry[0]) % 12, entry[1]


def _chord_pitches(root_pc: int, quality: str, octave: int = 4,
                   prev: Optional[List[int]] = None) -> List[int]:
    intervals = CHORD_INTERVALS.get(quality, [0,4,7])
    base = root_pc + octave * 12
    while base < 48: base += 12
    while base > 72: base -= 12
    raw = [base + i for i in intervals if 24 <= base+i <= 96]
    if not raw or prev is None:
        return raw
    def motion(a,b): return sum(abs(b[i]-a[i]) for i in range(min(len(a),len(b))))
    best, bm = raw, motion(prev, raw)
    rot = list(raw)
    for _ in range(len(raw)-1):
        rot = rot[1:] + [rot[0]+12]
        m = motion(prev, rot)
        if m < bm: bm = m; best = list(rot)
    return best


def _score_melodic_interest(notes: List[RawNote]) -> float:
    if len(notes) < 2: return 0.0
    pitches = [n.pitch for n in notes]
    durs    = [n.duration for n in notes]
    vels    = [n.velocity for n in notes]
    pc_var  = len(set(p%12 for p in pitches)) / 12.0
    p_range = min(1.0, (max(pitches)-min(pitches)) / 24.0)
    total   = max(n.offset+n.duration for n in notes)
    density = min(1.0, len(notes)/max(total,1)/2.0)
    vel_var = min(1.0, float(np.std(vels))/40.0)
    rhy_var = min(1.0, float(np.std(durs))/1.0)
    return pc_var*0.3 + p_range*0.2 + density*0.2 + vel_var*0.1 + rhy_var*0.2


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi_raw(path: str) -> Tuple[mido.MidiFile, int, int, int]:
    """Devuelve (mid, tpb, tempo_bpm, beats_per_bar)."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo_us = 500_000
    bpb = 4
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":   tempo_us = msg.tempo
        elif msg.type == "time_signature": bpb = msg.numerator
    bpm = round(60_000_000 / tempo_us)
    return mid, tpb, bpm, bpb


def _parse_track_to_notes(track, tpb: int) -> List[RawNote]:
    active: Dict[int, Tuple[float,int]] = {}
    notes: List[RawNote] = []
    abs_ticks = 0
    for msg in track:
        abs_ticks += msg.time
        beat = abs_ticks / tpb
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (beat, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                start, vel = active.pop(msg.note)
                dur = beat - start
                if dur >= MIN_NOTE_DUR:
                    notes.append(RawNote(msg.note,
                                         max(MIN_NOTE_DUR, round(dur*4)/4),
                                         vel, start))
    if not notes: return []
    mn = min(n.offset for n in notes)
    for n in notes: n.offset -= mn
    return sorted(notes, key=lambda n: n.offset)


def _choose_melody_track_idx(mid: mido.MidiFile) -> int:
    """Elige el track con mayor interés melódico."""
    tpb = mid.ticks_per_beat
    best_idx, best_score = 0, -1.0
    for i, track in enumerate(mid.tracks):
        notes = _parse_track_to_notes(track, tpb)
        if notes:
            s = _score_melodic_interest(notes)
            if s > best_score:
                best_score = s; best_idx = i
    return best_idx


def _load_all_tracks(mid: mido.MidiFile) -> Dict[int, List[RawNote]]:
    tpb = mid.ticks_per_beat
    result = {}
    for i, track in enumerate(mid.tracks):
        notes = _parse_track_to_notes(track, tpb)
        if notes:
            result[i] = notes
    return result


def _tick_to_bar(tick: int, tpb: int, bpb: int) -> int:
    """Compás (base 1) para un tick absoluto."""
    bar_ticks = tpb * bpb
    return tick // bar_ticks + 1


def _bar_start_tick(bar: int, tpb: int, bpb: int) -> int:
    return (bar - 1) * tpb * bpb


def _notes_with_bar(track_notes: List[RawNote],
                    tpb: int, bpb: int) -> List[Tuple[RawNote, int]]:
    """Lista de (nota, número_de_compás)."""
    return [(n, _tick_to_bar(int(n.offset * tpb), tpb, bpb))
            for n in track_notes]


# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DE FIRMA (SectionSignature)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signature(notes: List[RawNote],
                       position_ratio: float = 0.5) -> SectionSignature:
    if not notes:
        return SectionSignature(
            position_ratio=position_ratio, tension=0.0,
            velocity_mean=0.5, interval_hist=np.zeros(24),
            rhythm_hist=np.zeros(8), contour_dir=0.0, density=0.0)

    pitches  = [n.pitch for n in notes]
    durs     = [n.duration for n in notes]
    vels     = [n.velocity for n in notes]
    total_dur = max(n.offset+n.duration for n in notes)

    # Histograma de intervalos [-12, +12] → 24 bins
    intervals = [pitches[i+1]-pitches[i] for i in range(len(pitches)-1)]
    ihist, _ = np.histogram(intervals, bins=24, range=(-12,12), density=False)
    s = ihist.sum()
    ihist = ihist / s if s > 0 else ihist

    # Histograma rítmico (duraciones cuantizadas)
    q_durs = [min(7, max(0, int(round(math.log2(max(d, 0.125)) + 3)))) for d in durs]
    rhist = np.zeros(8)
    for q in q_durs: rhist[q] += 1
    rs = rhist.sum()
    rhist = rhist / rs if rs > 0 else rhist

    # Tensión: proporción de notas fuera de escala detectada
    key_pc, mode = _detect_key(notes)
    scale_pcs = set(_get_scale_pcs(key_pc, mode))
    out_of_scale = sum(1 for p in pitches if p % 12 not in scale_pcs)
    tension = out_of_scale / len(pitches)

    # Dirección del contorno
    contour_dir = float(np.mean([1 if i > 0 else (-1 if i < 0 else 0)
                                  for i in intervals])) if intervals else 0.0

    density = len(notes) / max(total_dur, 1.0)
    vel_mean = float(np.mean(vels)) / 127.0

    return SectionSignature(
        position_ratio = position_ratio,
        tension        = float(tension),
        velocity_mean  = vel_mean,
        interval_hist  = ihist,
        rhythm_hist    = rhist,
        contour_dir    = contour_dir,
        density        = density,
    )


def _signature_distance(a: SectionSignature, b: SectionSignature) -> float:
    va = a.to_vector()
    vb = b.to_vector()
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9: return 1.0
    return float(1.0 - np.dot(va/na, vb/nb))


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITMO RE-PAIR (Sequitur simplificado)
# ══════════════════════════════════════════════════════════════════════════════

class _RePair:
    """Re-Pair sobre secuencia de enteros (pitches MIDI)."""

    def __init__(self):
        self.rules: Dict[str, List] = {}
        self._next_id = 1
        self.root: List = []

    def _new_rule(self) -> str:
        name = f"R{self._next_id}"; self._next_id += 1
        return name

    def build(self, symbols: List[int]) -> None:
        from collections import Counter
        seq = list(symbols)
        while True:
            pairs = Counter()
            for i in range(len(seq)-1):
                pairs[(seq[i], seq[i+1])] += 1
            if not pairs: break
            best, count = pairs.most_common(1)[0]
            if count < 2: break
            name = self._new_rule()
            self.rules[name] = list(best)
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq)-1 and seq[i] == best[0] and seq[i+1] == best[1]:
                    new_seq.append(name); i += 2
                else:
                    new_seq.append(seq[i]); i += 1
            seq = new_seq
        self.root = seq

    def expand(self, sym) -> List[int]:
        if sym not in self.rules: return [sym]
        result = []
        for s in self.rules[sym]: result.extend(self.expand(s))
        return result

    def expand_with_indices(self, sym, start_idx: int = 0) -> Tuple[List[int], List[int]]:
        if sym not in self.rules:
            return [sym], [start_idx]
        notes, indices = [], []
        pos = start_idx
        for s in self.rules[sym]:
            sub_n, sub_i = self.expand_with_indices(s, pos)
            notes.extend(sub_n); indices.extend(sub_i)
            pos += len(sub_n)
        return notes, indices


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN  (midi_segmenter)
# ══════════════════════════════════════════════════════════════════════════════

def _ssm(descriptors: List[np.ndarray]) -> np.ndarray:
    X = np.array(descriptors, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    return np.dot(Xn, Xn.T)


def _checkerboard_kernel(size: int) -> np.ndarray:
    m = size // 2
    k = np.array([[1,-1],[-1,1]])
    k = np.kron(k, np.ones((m,m)))
    return gaussian_filter(k.astype(float), sigma=size/6)


def _infer_num_sections(notes: List[RawNote], tpb: int, bpb: int,
                         kernel_size: int = 8,
                         threshold: float = 0.15) -> int:
    """Detecta número de secciones por curva de novedad sobre SSM."""
    bar_ticks = tpb * bpb
    total = max(int((n.offset + n.duration) * tpb) for n in notes)
    n_bars = max(1, total // bar_ticks)

    if n_bars < kernel_size:
        return max(2, n_bars // 2)

    def bar_descriptor(bar_n: int) -> np.ndarray:
        lo = (bar_n - 1) * bar_ticks / tpb
        hi = bar_n * bar_ticks / tpb
        bar_notes = [n for n in notes if lo <= n.offset < hi]
        if not bar_notes:
            return np.zeros(8)
        pitches   = [n.pitch for n in bar_notes]
        durs      = [n.duration for n in bar_notes]
        vels      = [n.velocity for n in bar_notes]
        intervals = np.diff(pitches) if len(pitches) > 1 else [0]
        density   = len(bar_notes) / max(bar_ticks / tpb, 1)
        return np.array([
            min(density / 4, 1.0),
            np.mean(pitches) / 127,
            np.std(pitches) / 64 if len(pitches) > 1 else 0,
            np.mean(vels) / 127,
            np.std(vels) / 64 if len(vels) > 1 else 0,
            np.mean(durs),
            float(np.mean(np.array(intervals) > 0)) if len(intervals) > 0 else 0.5,
            float(np.mean(np.abs(np.array(intervals)) > 5)) if len(intervals) > 0 else 0,
        ])

    descs = [bar_descriptor(b) for b in range(1, n_bars+1)]
    ssm   = _ssm(descs)
    ks    = min(kernel_size, len(descs) // 2 * 2)
    if ks < 2: return 2
    kernel = _checkerboard_kernel(ks)
    N = ssm.shape[0]
    novelty = np.zeros(N)
    m = ks // 2
    for i in range(m, N-m):
        sub = ssm[i-m:i+m, i-m:i+m]
        if sub.shape == kernel.shape:
            novelty[i] = np.sum(sub * kernel)

    mx = np.max(np.abs(novelty))
    if mx > 0: novelty /= mx

    peaks, _ = find_peaks(novelty, height=threshold, distance=2)
    n_detected = len(peaks) + 1
    max_sections = max(2, n_bars // 4)
    return int(np.clip(n_detected, 2, max_sections))


def _segment_to_notes(notes: List[RawNote], bar_from: int, bar_to: int,
                       tpb: int, bpb: int) -> List[RawNote]:
    lo = (bar_from - 1) * bpb
    hi = bar_to * bpb
    result = [n for n in notes if lo <= n.offset < hi]
    if result:
        mn = min(n.offset for n in result)
        result = [RawNote(n.pitch, n.duration, n.velocity, n.offset - mn)
                  for n in result]
    return result


def _compute_segment_vector(notes: List[int], n: int = 3, bins: int = 12) -> np.ndarray:
    """Vector de n-gramas de intervalos para comparar segmentos."""
    if len(notes) < 2:
        return np.zeros(bins * 2)
    intervals = np.diff(notes)
    hist, _ = np.histogram(intervals, bins=bins, range=(-12,12), density=True)
    quantized = [max(-12,min(12,int(i))) for i in intervals]
    ngrams = [tuple(quantized[i:i+n]) for i in range(len(quantized)-n+1)]
    ngvec = np.zeros(bins)
    for ng in ngrams:
        h = hash(str(ng)) % bins
        ngvec[abs(h)] += 1
    norm = np.linalg.norm(ngvec)
    if norm > 0: ngvec /= norm
    return np.concatenate([hist, ngvec])


def _merge_similar_clusters(labels: np.ndarray, vectors: np.ndarray,
                              sim_threshold: float) -> np.ndarray:
    unique = sorted(set(labels))
    centroids = {}
    for l in unique:
        idxs = np.where(labels == l)[0]
        c = np.mean(vectors[idxs], axis=0)
        n = np.linalg.norm(c)
        centroids[l] = c / n if n > 0 else c

    mapping = {l: l for l in unique}
    for l1 in unique:
        for l2 in unique:
            if l1 >= l2: continue
            if mapping[l1] == mapping[l2]: continue
            sim = float(np.dot(centroids[l1], centroids[l2]))
            if sim > sim_threshold:
                old = mapping[l2]; new = mapping[l1]
                for k in mapping:
                    if mapping[k] == old: mapping[k] = new

    return np.array([mapping[l] for l in labels])


def extract_sections(midi_path: str,
                     num_sections: Optional[int] = None,
                     track_idx: Optional[int] = None,
                     all_tracks: bool = False,
                     fuzzy_threshold: float = 1.0,
                     similarity_threshold: float = 0.85,
                     verbose: bool = False) -> List[SectionSegment]:
    """
    Extrae las secciones formales de un MIDI mediante gramática Sequitur
    + clustering espectral.
    """
    mid, tpb, bpm, bpb = _load_midi_raw(midi_path)

    # Selección del track melódico
    if track_idx is None:
        track_idx = _choose_melody_track_idx(mid)

    melody_notes = _parse_track_to_notes(mid.tracks[track_idx], tpb)
    if not melody_notes:
        if verbose: print(f"  [warn] Track {track_idx} sin notas")
        return []

    all_track_notes = _load_all_tracks(mid) if all_tracks else None

    if verbose:
        print(f"  Track melódico: #{track_idx}  |  {len(melody_notes)} notas  "
              f"|  {bpm} BPM  |  {bpb}/4")

    # Inferir número de secciones si no se especifica
    if num_sections is None:
        num_sections = _infer_num_sections(melody_notes, tpb, bpb)
        if verbose:
            print(f"  Secciones inferidas: {num_sections}")
    else:
        if verbose:
            print(f"  Secciones forzadas: {num_sections}")

    # Secuencia de pitches para Re-Pair
    symbols = [n.pitch for n in melody_notes]
    if len(symbols) < 4:
        if verbose: print("  [warn] Secuencia demasiado corta para Re-Pair")
        return []

    grammar = _RePair()
    grammar.build(symbols)
    root_seq = grammar.root

    if verbose:
        print(f"  Reglas Sequitur: {len(grammar.rules)}  "
              f"|  Secuencia raíz: {len(root_seq)} símbolos")

    # Expandir raíz con tracking de índices de notas
    if len(root_seq) == 1 and isinstance(root_seq[0], str):
        root_seq = grammar.rules[root_seq[0]]

    segments_raw = []
    current_idx = 0
    for sym in root_seq:
        if isinstance(sym, str) and sym in grammar.rules:
            notes_exp, indices = grammar.expand_with_indices(sym, current_idx)
        else:
            notes_exp = [sym] if isinstance(sym, int) else grammar.expand(sym)
            indices = list(range(current_idx, current_idx + len(notes_exp)))
        segments_raw.append({
            "symbol": str(sym),
            "notes_vals": notes_exp,
            "note_indices": indices,
            "len": len(notes_exp),
        })
        current_idx += len(notes_exp)

    # Simplificar al número de secciones objetivo
    def simplify(segs, target):
        current = deepcopy(segs)
        while len(current) > target:
            min_len = min(s["len"] for s in current)
            min_idx = next(i for i, s in enumerate(current) if s["len"] == min_len)
            if min_idx == 0:
                merge = 1
            elif min_idx == len(current) - 1:
                merge = min_idx - 1; min_idx = len(current) - 1
            else:
                prev_l = current[min_idx-1]["len"]
                next_l = current[min_idx+1]["len"]
                if prev_l <= next_l:
                    merge = min_idx; min_idx = min_idx - 1
                else:
                    merge = min_idx + 1
            if merge > min_idx: min_idx, merge = merge, min_idx
            a, b = current[min_idx], current[merge]
            current[min_idx] = {
                "symbol": f"({a['symbol']}+{b['symbol']})",
                "notes_vals": a["notes_vals"] + b["notes_vals"],
                "note_indices": a["note_indices"] + b["note_indices"],
                "len": a["len"] + b["len"],
            }
            del current[merge]
        return current

    segments_raw = simplify(segments_raw, num_sections)

    if verbose:
        print(f"  Segmentos tras simplificación: {len(segments_raw)}")

    # Vectorizar segmentos y clustering
    vectors = []
    for seg in segments_raw:
        v = _compute_segment_vector(seg["notes_vals"])
        vectors.append(v if not np.all(v == 0) else v + 1e-9)

    X = np.array(vectors)
    k = min(num_sections, len(segments_raw))

    if k < 2:
        raw_labels = np.zeros(len(segments_raw), dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(X)
        raw_labels = _merge_similar_clusters(raw_labels, X, similarity_threshold)

    # Asignar letras A/B/C...
    label_map: Dict[int, str] = {}
    next_char = ord("A")
    form_labels = []
    for l in raw_labels:
        if l not in label_map:
            label_map[l] = chr(next_char); next_char += 1
        form_labels.append(label_map[l])

    # Construir SectionSegments con rango de compases
    result: List[SectionSegment] = []
    bar_map = {i: _tick_to_bar(int(n.offset * tpb), tpb, bpb)
               for i, n in enumerate(melody_notes)}

    for i, (seg, lbl) in enumerate(zip(segments_raw, form_labels)):
        indices = [idx for idx in seg["note_indices"]
                   if 0 <= idx < len(melody_notes)]
        if indices:
            measures = [bar_map.get(idx, 1) for idx in indices]
            bar_start = min(measures)
            bar_end   = max(measures)
        else:
            bar_start = bar_end = i + 1

        # Notas del segmento
        if all_tracks and all_track_notes:
            seg_notes = []
            for tk_notes in all_track_notes.values():
                seg_notes.extend(
                    _segment_to_notes(tk_notes, bar_start, bar_end, tpb, bpb))
        else:
            seg_notes = _segment_to_notes(melody_notes, bar_start, bar_end, tpb, bpb)

        result.append(SectionSegment(
            label=lbl, bar_start=bar_start, bar_end=bar_end,
            notes=seg_notes, symbol=seg["symbol"],
        ))

    if verbose:
        form_str = "".join(s.label for s in result)
        print(f"  Forma detectada: {form_str}")
        for s in result:
            print(f"    {s.label}  compases {s.bar_start}–{s.bar_end}"
                  f"  ({len(s.notes)} notas)  [{s.symbol[:30]}]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ESTIMACIÓN DE TransformVector (de source → section)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_transform(source_sig: SectionSignature,
                         target_sig: SectionSignature) -> TransformVector:
    """
    Estima el TransformVector θ que lleva source_sig hacia target_sig
    analizando las diferencias entre sus características.
    """
    tv = TransformVector()

    # ── PITCH ────────────────────────────────────────────────────────────────
    # Dirección de contorno: si difieren mucho → inversión
    contour_diff = target_sig.contour_dir - source_sig.contour_dir
    # Histograma de intervalos: dominancia de intervalos pequeños vs grandes
    src_small  = float(np.sum(source_sig.interval_hist[10:14]))  # ±2 semitonos
    tgt_small  = float(np.sum(target_sig.interval_hist[10:14]))
    src_large  = float(np.sum(source_sig.interval_hist[:4] + source_sig.interval_hist[20:]))
    tgt_large  = float(np.sum(target_sig.interval_hist[:4] + target_sig.interval_hist[20:]))

    if abs(contour_diff) > 0.5:
        tv.op_pitch = "invert"
        tv.param_pitch = contour_diff
    elif tgt_small > src_small + 0.15:
        tv.op_pitch = "diatonic_sequence"
        tv.param_pitch = 1.0 if contour_diff >= 0 else -1.0
    elif tgt_large > src_large + 0.15:
        tv.op_pitch = "transpose"
        # Transposición estimada como el desplazamiento del centroide de intervalos
        src_centroid = float(np.sum(
            np.arange(-12,12) * source_sig.interval_hist))
        tgt_centroid = float(np.sum(
            np.arange(-12,12) * target_sig.interval_hist))
        tv.param_pitch = float(np.clip(tgt_centroid - src_centroid, -12, 12))
    elif target_sig.tension > source_sig.tension + 0.2:
        tv.op_pitch = "modal_shift"
        tv.param_pitch = 1.0
    else:
        tv.op_pitch = "identity"
        tv.param_pitch = 0.0

    # ── RHYTHM ───────────────────────────────────────────────────────────────
    src_long = float(np.sum(source_sig.rhythm_hist[5:]))  # negras y más largas
    tgt_long = float(np.sum(target_sig.rhythm_hist[5:]))
    density_ratio = (target_sig.density / max(source_sig.density, 0.01))

    if density_ratio > 1.4:
        tv.op_rhythm = "diminish"
        tv.param_rhythm = 1.0 / min(density_ratio, 2.0)
    elif density_ratio < 0.7:
        tv.op_rhythm = "augment"
        tv.param_rhythm = 1.0 / max(density_ratio, 0.5)
    elif tgt_long < src_long - 0.2:
        tv.op_rhythm = "rhythmic_repattern"
        tv.param_rhythm = 1.0
    else:
        tv.op_rhythm = "identity"
        tv.param_rhythm = 1.0

    # ── DYNAMICS ─────────────────────────────────────────────────────────────
    vel_ratio = target_sig.velocity_mean / max(source_sig.velocity_mean, 0.01)
    tension_delta = target_sig.tension - source_sig.tension

    if abs(vel_ratio - 1.0) > 0.15:
        tv.op_dynamics = "velocity_scale"
        tv.param_dynamics = float(np.clip(vel_ratio, 0.5, 1.5))
    elif abs(tension_delta) > 0.2:
        tv.op_dynamics = "tension_curve"
        # 0=flat,1=rise,2=fall,3=arch,4=inverse_arch
        if tension_delta > 0.3:   tv.param_dynamics = 1.0  # rise
        elif tension_delta < -0.3: tv.param_dynamics = 2.0  # fall
        elif target_sig.tension > 0.6: tv.param_dynamics = 3.0  # arch
        else:                          tv.param_dynamics = 4.0  # inverse
    else:
        tv.op_dynamics = "identity"
        tv.param_dynamics = 1.0

    # ── HARMONY ──────────────────────────────────────────────────────────────
    if target_sig.tension > 0.6:
        tv.op_harmony = "progression_select"
        tv.param_harmony = 2.0   # índice alto = más tenso
    elif target_sig.tension < 0.2:
        tv.op_harmony = "progression_select"
        tv.param_harmony = 3.0   # cierre
    else:
        tv.op_harmony = "acc_style"
        tv.param_harmony = float(ACC_STYLES.index("arpeggio"))

    return tv


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO  (transformation_learner)
# ══════════════════════════════════════════════════════════════════════════════

def train(midi_dir: str,
          output_model: str,
          n_roles: int = 4,
          assignment_mode: str = "hybrid",
          num_sections: Optional[int] = None,
          verbose: bool = False) -> TransformationModel:
    """
    Entrena el modelo de transformaciones sobre un directorio de MIDIs.
    """
    midi_files = sorted(Path(midi_dir).glob("**/*.mid")) + \
                 sorted(Path(midi_dir).glob("**/*.midi"))

    if not midi_files:
        print(f"[ERROR] No se encontraron MIDIs en {midi_dir}")
        sys.exit(1)

    print(f"\n  Corpus: {len(midi_files)} MIDIs en {midi_dir}")
    print(f"  Roles: {n_roles}  |  Asignación: {assignment_mode}\n")

    all_signatures:  List[SectionSignature]  = []
    all_transforms:  List[TransformVector]   = []
    corpus_size = 0

    for i, midi_path in enumerate(midi_files):
        try:
            sections = extract_sections(
                str(midi_path),
                num_sections=num_sections,
                verbose=False,
            )
            if len(sections) < 2:
                continue

            # Firma del material fuente = todas las notas juntas
            source_notes = [n for s in sections for n in s.notes]
            if not source_notes:
                continue

            source_sig = _compute_signature(source_notes, position_ratio=0.0)
            total_sections = len(sections)

            for j, section in enumerate(sections):
                if not section.notes:
                    continue
                pos = j / max(total_sections - 1, 1)
                sig = _compute_signature(section.notes, position_ratio=pos)
                theta = _estimate_transform(source_sig, sig)
                all_signatures.append(sig)
                all_transforms.append(theta)

            corpus_size += 1
            if verbose:
                form = "".join(s.label for s in sections)
                print(f"  [{i+1:3d}/{len(midi_files)}] {Path(midi_path).name:<40}"
                      f"  forma={form}  secciones={len(sections)}")
            elif (i+1) % 10 == 0 or i == 0:
                print(f"  Procesados: {i+1}/{len(midi_files)}  "
                      f"(secciones acumuladas: {len(all_signatures)})")

        except Exception as e:
            if verbose:
                print(f"  [warn] {Path(midi_path).name}: {e}")
            continue

    if len(all_signatures) < n_roles:
        print(f"[ERROR] Solo {len(all_signatures)} secciones en corpus, "
              f"insuficiente para {n_roles} roles.")
        sys.exit(1)

    print(f"\n  Secciones totales: {len(all_signatures)}  |  "
          f"MIDIs procesados: {corpus_size}")
    print(f"  Clustering en {n_roles} roles...")

    # Clustering de SectionSignatures
    sig_vectors = np.array([s.to_vector() for s in all_signatures])
    sig_vectors_norm = sk_normalize(sig_vectors)

    clustering = AgglomerativeClustering(
        n_clusters=n_roles,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(sig_vectors_norm)

    # Construir RoleProfiles
    roles: List[RoleProfile] = []
    for role_idx in range(n_roles):
        idxs = [i for i, l in enumerate(labels) if l == role_idx]
        if not idxs:
            continue

        # Centroide de firmas
        role_sigs   = [all_signatures[i] for i in idxs]
        role_thetas = [all_transforms[i]  for i in idxs]

        mean_vec = np.mean([s.to_vector() for s in role_sigs], axis=0)
        # Reconstruir SectionSignature del centroide
        centroid = SectionSignature(
            position_ratio = float(mean_vec[0]),
            tension        = float(mean_vec[1]),
            velocity_mean  = float(mean_vec[2]),
            contour_dir    = float(mean_vec[3]),
            density        = float(mean_vec[4]) * 4.0,
            interval_hist  = mean_vec[5:29],
            rhythm_hist    = mean_vec[29:37],
        )

        # Estadísticas de TransformVector
        theta_arrays = np.array([t.to_array() for t in role_thetas])
        theta_mean_arr = np.mean(theta_arrays, axis=0)
        theta_mean = TransformVector.from_array(theta_mean_arr)
        theta_cov  = np.cov(theta_arrays.T) if len(idxs) > 1 else np.eye(8) * 0.01

        positions = [all_signatures[i].position_ratio for i in idxs]
        tensions  = [all_signatures[i].tension for i in idxs]
        pos_dist  = (float(np.mean(positions)), float(np.std(positions)) + 1e-6)
        ten_dist  = (float(np.mean(tensions)),  float(np.std(tensions)) + 1e-6)

        # Etiqueta descriptiva
        p_mean = pos_dist[0]
        t_mean = ten_dist[0]
        if p_mean < 0.2:
            lbl = "intro"
        elif p_mean > 0.8:
            lbl = "outro"
        elif t_mean > 0.65:
            lbl = "climax"
        elif t_mean > 0.45:
            lbl = "development"
        else:
            lbl = "verse"

        roles.append(RoleProfile(
            centroid=centroid, theta_mean=theta_mean,
            theta_cov=theta_cov, position_dist=pos_dist,
            tension_dist=ten_dist, label=lbl,
        ))

    # Ordenar roles por posición media
    roles.sort(key=lambda r: r.position_dist[0])

    model = TransformationModel(
        n_roles=len(roles), roles=roles,
        assignment_mode=assignment_mode,
        corpus_size=corpus_size,
    )

    # Guardar modelo
    out_path = Path(output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n  Modelo guardado: {out_path}")
    print(f"  Roles aprendidos:")
    for i, r in enumerate(model.roles):
        print(f"    [{i}] {r.label:<15}  pos={r.position_dist[0]:.2f}±{r.position_dist[1]:.2f}"
              f"  tensión={r.tension_dist[0]:.2f}  "
              f"θ_pitch={r.theta_mean.op_pitch}  θ_rhythm={r.theta_mean.op_rhythm}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN DE TRANSFORMACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _apply_transform(notes: List[RawNote],
                     theta: TransformVector,
                     key_pc: int, mode: str,
                     target_bars: int,
                     beats_per_bar: int,
                     rng: random.Random) -> List[RawNote]:
    """Aplica un TransformVector sobre una lista de notas."""
    if not notes: return []
    result = deepcopy(notes)

    # ── PITCH ────────────────────────────────────────────────────────────────
    op = theta.op_pitch
    param = theta.param_pitch

    if op == "transpose":
        semitones = int(round(np.clip(param, -12, 12)))
        result = [RawNote(max(0,min(127,n.pitch+semitones)),
                           n.duration, n.velocity, n.offset)
                  for n in result]

    elif op == "invert":
        pitches = [n.pitch for n in result]
        pivot   = float(np.mean(pitches))
        result = [RawNote(max(0,min(127,int(round(2*pivot - n.pitch)))),
                           n.duration, n.velocity, n.offset)
                  for n in result]
        # Snap to scale
        result = [RawNote(_snap_to_scale(n.pitch, key_pc, mode),
                           n.duration, n.velocity, n.offset)
                  for n in result]

    elif op == "retrograde":
        pitches = [n.pitch for n in reversed(result)]
        result = [RawNote(p, n.duration, n.velocity, n.offset)
                  for p, n in zip(pitches, result)]

    elif op == "retro_invert":
        pitches = [n.pitch for n in result]
        pivot   = float(np.mean(pitches))
        inv     = [int(round(2*pivot - p)) for p in pitches]
        inv_ret = list(reversed(inv))
        result = [RawNote(max(0,min(127,_snap_to_scale(p, key_pc, mode))),
                           n.duration, n.velocity, n.offset)
                  for p, n in zip(inv_ret, result)]

    elif op == "modal_shift":
        # Altera terceras mayor↔menor
        pitches = [n.pitch for n in result]
        new_pitches = [pitches[0]]
        for i in range(len(pitches)-1):
            d = pitches[i+1] - pitches[i]
            if d == 4: d = 3
            elif d == -4: d = -3
            elif d == 3: d = 4
            elif d == -3: d = -4
            new_pitches.append(max(0,min(127, new_pitches[-1] + d)))
        result = [RawNote(p, n.duration, n.velocity, n.offset)
                  for p, n in zip(new_pitches, result)]

    elif op == "diatonic_sequence":
        steps = int(round(np.clip(param, -4, 4)))
        scale_pcs = _get_scale_pcs(key_pc, mode)
        def diat_transpose(pitch, n_steps):
            pc = pitch % 12
            if pc in scale_pcs:
                idx = scale_pcs.index(pc)
            else:
                idx = min(range(len(scale_pcs)), key=lambda i: abs(scale_pcs[i]-pc))
            new_idx = (idx + n_steps) % 7
            extra_oct = (idx + n_steps) // 7
            new_pc = scale_pcs[new_idx]
            octave = pitch // 12
            return 12 * (octave + extra_oct) + new_pc
        result = [RawNote(max(0,min(127,diat_transpose(n.pitch, steps))),
                           n.duration, n.velocity, n.offset)
                  for n in result]

    elif op == "liquidate":
        keep = max(2, int(round(abs(param))) if param != 0 else 3)
        # Mantener solo notas en tiempos fuertes
        strong = [n for n in result if n.offset % 1.0 < 0.1]
        if len(strong) >= keep:
            result = strong[:keep]
        else:
            result = result[:keep]

    # ── RHYTHM ───────────────────────────────────────────────────────────────
    op = theta.op_rhythm
    param = theta.param_rhythm

    if op == "augment":
        factor = float(np.clip(param, 1.0, 2.0))
        result = [RawNote(n.pitch, n.duration*factor, n.velocity, n.offset*factor)
                  for n in result]

    elif op == "diminish":
        factor = float(np.clip(param, 0.5, 1.0))
        result = [RawNote(n.pitch, max(MIN_NOTE_DUR, n.duration*factor),
                           n.velocity, n.offset*factor)
                  for n in result]

    elif op == "rhythmic_repattern":
        pitches = [n.pitch for n in result]
        vels    = [n.velocity for n in result]
        patterns = [
            [1.0, 0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
            [0.25, 0.25, 0.5, 1.0],
            [2.0, 0.5, 0.5, 1.0],
        ]
        pat = patterns[int(round(param)) % len(patterns)]
        total_beats = target_bars * beats_per_bar
        cursor = 0.0; new_notes = []; pi = 0; pat_i = 0
        while cursor < total_beats - 0.05 and pi < len(pitches):
            d = min(pat[pat_i % len(pat)], total_beats - cursor)
            new_notes.append(RawNote(pitches[pi % len(pitches)], d,
                                      vels[pi % len(vels)], cursor))
            cursor += d; pi += 1; pat_i += 1
        result = new_notes

    elif op == "syncopate":
        shift = float(np.clip(param if param != 1.0 else 0.25, 0.125, 0.5))
        result = [RawNote(n.pitch, n.duration, n.velocity,
                           n.offset + (shift if n.offset % 1.0 < 0.1 else 0.0))
                  for n in result]

    # ── DYNAMICS ─────────────────────────────────────────────────────────────
    op = theta.op_dynamics
    param = theta.param_dynamics

    if op == "velocity_scale":
        ratio = float(np.clip(param, 0.5, 1.5))
        result = [RawNote(n.pitch, n.duration,
                           int(np.clip(n.velocity * ratio, 20, 120)), n.offset)
                  for n in result]

    elif op == "tension_curve":
        shape_idx = int(round(param)) % len(TENSION_SHAPES)
        shape = TENSION_SHAPES[shape_idx]
        total = max(n.offset + n.duration for n in result) if result else 1.0
        def get_vel_scale(offset):
            t = offset / max(total, 1.0)
            if shape == "flat":          return 1.0
            elif shape == "rise":        return 0.7 + 0.3 * t
            elif shape == "fall":        return 1.0 - 0.3 * t
            elif shape == "arch":        return 0.7 + 0.6 * math.sin(math.pi * t)
            elif shape == "inverse_arch":return 1.0 - 0.3 * math.sin(math.pi * t)
            return 1.0
        result = [RawNote(n.pitch, n.duration,
                           int(np.clip(n.velocity * get_vel_scale(n.offset), 20, 120)),
                           n.offset)
                  for n in result]

    elif op == "ornament":
        density = float(np.clip(param, 0.0, 1.0))
        ornamented = []
        for i, n in enumerate(result):
            ornamented.append(n)
            # Solo ornanentar notas largas (>=1.0 beat) para evitar micronotas
            if (i + 1 < len(result) and n.duration >= 1.0 and rng.random() < density):
                next_n = result[i+1]
                if abs(next_n.pitch - n.pitch) >= 4:
                    pass_pitch = _snap_to_scale(
                        (n.pitch + next_n.pitch) // 2, key_pc, mode)
                    half = max(0.5, n.duration / 2)  # minimo 0.5 beats
                    ornamented[-1] = RawNote(n.pitch, half, n.velocity, n.offset)
                    ornamented.append(RawNote(pass_pitch, half,
                                               max(20, n.velocity - 10),
                                               n.offset + half))
        result = ornamented

    elif op == "add_passing_notes":
        threshold = int(np.clip(param if param > 0 else 4, 3, 6))
        expanded = []
        for i, n in enumerate(result):
            expanded.append(n)
            if i + 1 < len(result) and n.duration >= 1.0:
                nxt = result[i+1]
                if abs(nxt.pitch - n.pitch) >= threshold:
                    pp = _snap_to_scale((n.pitch + nxt.pitch)//2, key_pc, mode)
                    half = max(0.5, n.duration / 2)  # mínimo 0.5 beats
                    expanded[-1] = RawNote(n.pitch, half, n.velocity, n.offset)
                    expanded.append(RawNote(pp, half, max(20, n.velocity-8),
                                            n.offset + half))
        result = expanded

    # ── Fit to target_bars ───────────────────────────────────────────────────
    # Se trabaja siempre en beats de 4/4 para independizarse del compás fuente.
    if result:
        target_beats = target_bars * 4  # siempre en negras (4/4 equivalente)
        current_total = max(n.offset + n.duration for n in result)
        if current_total > 0.01:
            factor = target_beats / current_total
            if factor < 0.5:
                # Compresión extrema: seleccionar el fragmento más interesante
                # (las notas con mayor variedad de pitch en la ventana objetivo)
                result_sorted = sorted(result, key=lambda n: n.offset)
                # Tomar las primeras target_beats / dur_media notas
                dur_media = float(np.mean([n.duration for n in result_sorted])) or 1.0
                n_keep = max(4, int(target_beats / dur_media))
                # Preferir el fragmento central (más desarrollo)
                start_idx = max(0, (len(result_sorted) - n_keep) // 2)
                fragment = result_sorted[start_idx:start_idx + n_keep]
                if fragment:
                    mn = min(n.offset for n in fragment)
                    fragment = [RawNote(n.pitch, n.duration, n.velocity, n.offset - mn)
                                for n in fragment]
                    frag_total = max(n.offset + n.duration for n in fragment)
                    fine = target_beats / max(frag_total, 0.01)
                    if not (0.8 < fine < 1.2):
                        fragment = [RawNote(n.pitch,
                                            max(0.25, n.duration * fine),
                                            n.velocity, n.offset * fine)
                                    for n in fragment]
                    result = fragment
            elif not (0.8 < factor < 1.2):
                # Escalado normal: respetar un mínimo de 0.25 beats por nota
                min_factor = 0.25 / max(min(n.duration for n in result), 0.01)
                safe_factor = max(factor, min_factor)
                result = [RawNote(n.pitch,
                                   max(0.25, n.duration * safe_factor),
                                   n.velocity, n.offset * safe_factor)
                          for n in result]

    # ── Micro-variaciones para evitar efecto mecánico ────────────────────────
    # Solo se aplican a notas suficientemente largas para no crear ruido
    result = [RawNote(n.pitch,
                       max(0.25, n.duration * rng.uniform(0.95, 1.05))
                           if n.duration >= 0.25 else n.duration,
                       int(np.clip(n.velocity * rng.uniform(0.97, 1.03), 20, 120)),
                       n.offset)
              for n in result]

    return sorted(result, key=lambda n: n.offset)


# ══════════════════════════════════════════════════════════════════════════════
#  ACOMPAÑAMIENTO Y EXPORTACIÓN  (infraestructura de song_architect)
# ══════════════════════════════════════════════════════════════════════════════

def _build_accompaniment(prog: List[Tuple[str,int]], key_pc: int,
                          beats_per_bar: int, velocity: int,
                          style: str = "arpeggio") -> List[RawNote]:
    notes: List[RawNote] = []
    cursor = 0.0
    prev_pitches = None
    for numeral, dur in prog:
        root_pc, quality = _numeral_to_root(numeral, key_pc)
        pitches = _chord_pitches(root_pc, quality, octave=4, prev=prev_pitches)
        prev_pitches = pitches
        if style == "block":
            for p in pitches:
                notes.append(RawNote(p, max(MIN_NOTE_DUR, dur*0.9),
                                      velocity, cursor))
        elif style in ("arpeggio", "alberti"):
            # Minimo 0.5 beats por nota de arpeggio para evitar micronotas
            step = max(0.5, dur / max(len(pitches), 1))
            # Si el paso excede la duración del acorde, usar bloque
            if step * len(pitches) > dur * 1.5:
                for p in pitches:
                    notes.append(RawNote(p, max(0.25, dur * 0.9),
                                          velocity, cursor))
            else:
                for i, p in enumerate(pitches):
                    notes.append(RawNote(p, max(0.25, step * 0.85),
                                          velocity, cursor + i * step))
        elif style == "bass_only":
            if pitches:
                notes.append(RawNote(min(pitches), max(MIN_NOTE_DUR, dur*0.9),
                                      velocity, cursor))
        cursor += dur
    return notes


def _build_bass_line(prog: List[Tuple[str,int]], key_pc: int,
                      velocity: int) -> List[RawNote]:
    notes: List[RawNote] = []
    cursor = 0.0; prev_bass = None
    for numeral, dur in prog:
        root_pc, _ = _numeral_to_root(numeral, key_pc)
        bass = root_pc + 36
        while bass < 28: bass += 12
        while bass > 52: bass -= 12
        if prev_bass is not None and abs(bass - prev_bass) > 7:
            alt = bass + (12 if bass < prev_bass else -12)
            if 28 <= alt <= 52: bass = alt
        notes.append(RawNote(bass, max(MIN_NOTE_DUR, dur*0.85), velocity, cursor))
        prev_bass = bass; cursor += dur
    return notes


def _select_progression(theta: TransformVector, key_pc: int,
                          bars: int, beats_per_bar: int) -> List[Tuple[str,int]]:
    """Selecciona una progresión armónica según el rol."""
    progs = SECTION_PROGRESSIONS["pop"]
    idx = int(round(theta.param_harmony)) % len(progs)
    prog = progs[idx]
    # Siempre trabajar en beats de 4/4 para independizarse del compás fuente
    beats_needed = bars * 4
    beats_in_prog = sum(d for _, d in prog)
    reps = max(1, math.ceil(beats_needed / max(beats_in_prog, 1)))
    full = prog * reps
    # Truncar al tamaño exacto
    truncated = []; cursor = 0.0
    for numeral, dur in full:
        if cursor >= beats_needed: break
        actual = min(dur, beats_needed - cursor)
        truncated.append((numeral, actual)); cursor += actual
    return truncated


def _deduplicate(notes: List[RawNote]) -> List[RawNote]:
    sorted_notes = sorted(notes, key=lambda n: (n.offset, n.pitch))
    active: Dict[int,int] = {}; result: List[RawNote] = []
    for n in sorted_notes:
        if n.pitch in active:
            pi = active[n.pitch]; prev = result[pi]
            if prev.offset + prev.duration > n.offset:
                nd = max(MIN_NOTE_DUR, n.offset - prev.offset - 0.01)
                result[pi] = RawNote(prev.pitch, nd, prev.velocity, prev.offset)
        result.append(n); active[n.pitch] = len(result)-1
    return result


def _apply_voice_leading(prev_notes: List[RawNote], next_notes: List[RawNote],
                          key_pc: int, mode: str) -> List[RawNote]:
    if not prev_notes or not next_notes: return next_notes
    mel_prev = [n for n in prev_notes if n.pitch >= 48]
    mel_next = [n for n in next_notes if n.pitch >= 48]
    if not mel_prev or not mel_next: return next_notes
    last = max(mel_prev, key=lambda n: n.offset + n.duration)
    first = min(mel_next, key=lambda n: n.offset)
    if abs(first.pitch - last.pitch) <= 2: return next_notes
    scale_pcs = set(_get_scale_pcs(key_pc, mode))
    best_p, best_d = first.pitch, abs(first.pitch - last.pitch)
    for candidate in range(max(24, last.pitch-12), min(108, last.pitch+13)):
        if candidate % 12 in scale_pcs:
            d = abs(candidate - last.pitch)
            if d < best_d: best_d = d; best_p = candidate
    if best_p == first.pitch: return next_notes
    adjusted = False; new_notes = []
    for n in next_notes:
        if not adjusted and n is first:
            new_notes.append(RawNote(best_p, n.duration, n.velocity, n.offset))
            adjusted = True
        else:
            new_notes.append(n)
    return new_notes


def _notes_to_midi_track(notes: List[RawNote], tpb: int,
                          tempo_bpm: int, track_name: str) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo",
                                   tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    events = []
    for n in notes:
        t_on  = int(n.offset * tpb)
        t_off = int((n.offset + max(MIN_NOTE_DUR, n.duration)) * tpb)
        vel   = max(1, min(127, n.velocity))
        p     = max(0, min(127, n.pitch))
        ch    = 1 if p < 48 else 0
        events.append((t_on,  "on",  p, vel, ch))
        events.append((t_off, "off", p, 0,   ch))

    events.sort(key=lambda e: (e[0], 0 if e[1]=="off" else 1))
    current_tick = 0
    for abs_tick, etype, pitch, vel, ch in events:
        delta = max(0, abs_tick - current_tick)
        track.append(mido.Message(
            "note_on" if etype == "on" else "note_off",
            channel=ch, note=pitch, velocity=vel, time=delta))
        current_tick = abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN  (song_architect_ml)
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(model_path: str) -> TransformationModel:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _assign_roles_similarity(source_sig: SectionSignature,
                               model: TransformationModel,
                               n_sections: int) -> List[int]:
    """Asigna roles por similitud al material fuente, ordenados por posición."""
    dists = [_signature_distance(source_sig, r.centroid) for r in model.roles]
    sorted_by_dist = sorted(range(len(model.roles)), key=lambda i: dists[i])
    # Repetir roles si hay más secciones que roles
    role_order = []
    for i in range(n_sections):
        role_order.append(sorted_by_dist[i % len(sorted_by_dist)])
    return role_order


def _assign_roles_position(model: TransformationModel,
                             n_sections: int) -> List[int]:
    """Asigna roles por posición relativa aprendida."""
    positions = [i / max(n_sections-1, 1) for i in range(n_sections)]
    role_order = []
    for pos in positions:
        # Rol con position_dist.mean más cercana
        best = min(range(len(model.roles)),
                   key=lambda i: abs(model.roles[i].position_dist[0] - pos))
        role_order.append(best)
    return role_order


def _assign_roles_hybrid(source_sig: SectionSignature,
                          model: TransformationModel,
                          n_sections: int) -> List[int]:
    """Híbrido: posición determina el orden, similitud refina el rol."""
    position_order = _assign_roles_position(model, n_sections)
    dists = [_signature_distance(source_sig, r.centroid) for r in model.roles]
    role_order = []
    for pos_idx, base_role in enumerate(position_order):
        # Buscar el rol más similar al base_role que esté cerca en posición
        pos = pos_idx / max(n_sections-1, 1)
        candidates = sorted(range(len(model.roles)),
                             key=lambda i: dists[i] + abs(model.roles[i].position_dist[0]-pos))
        role_order.append(candidates[0])
    return role_order


def generate(input_midis: List[str],
             model_path: str,
             assignment_mode: Optional[str] = None,
             n_sections: Optional[int] = None,
             role_order_override: Optional[List[int]] = None,
             bars_per_section: int = 8,
             out_dir: str = "output",
             output_name: str = "song",
             tempo: Optional[int] = None,
             dry_run: bool = False,
             seed: int = 42,
             verbose: bool = False) -> None:
    """
    Genera una obra completa a partir de frases MIDI y un modelo entrenado.
    """
    rng = random.Random(seed)

    # Cargar modelo
    model = _load_model(model_path)
    mode  = assignment_mode or model.assignment_mode

    print(f"\n  Modelo: {model_path}  "
          f"({model.n_roles} roles, corpus={model.corpus_size})")
    print(f"  Asignación: {mode}")

    # Cargar frases de entrada.
    # Solo se usan notas melódicas (dur >= 0.25 beats) para evitar que
    # las micronotas del acompañamiento contaminen la fuente.
    MELODY_MIN_DUR = 0.25
    source_notes_all: List[RawNote] = []
    bpm_detected = 120
    bpb_detected = 4

    for midi_path in input_midis:
        try:
            mid, tpb_in, bpm, bpb = _load_midi_raw(midi_path)
            track_idx = _choose_melody_track_idx(mid)
            notes = _parse_track_to_notes(mid.tracks[track_idx], tpb_in)
            notes = [n for n in notes if n.duration >= MELODY_MIN_DUR]
            if notes:
                source_notes_all.extend(notes)
                bpm_detected = bpm; bpb_detected = bpb
                if verbose:
                    print(f"  Frase: {Path(midi_path).name}  "
                          f"{len(notes)} notas melódicas  {bpm} BPM")
            elif verbose:
                print(f"  [warn] {Path(midi_path).name}: sin notas melódicas válidas")
        except Exception as e:
            print(f"  [warn] {midi_path}: {e}")

    if not source_notes_all:
        print("[ERROR] No se pudieron cargar frases de entrada.")
        sys.exit(1)

    final_tempo = tempo or bpm_detected
    key_pc, mode_str = _detect_key(source_notes_all)
    source_sig = _compute_signature(source_notes_all, position_ratio=0.0)

    print(f"  Tonalidad: {NOTE_NAMES[key_pc]} {mode_str}  "
          f"Tempo: {final_tempo} BPM  Compás: {bpb_detected}/4")

    # Determinar número de secciones
    if n_sections is None:
        n_sections = model.n_roles
    print(f"  Secciones: {n_sections}")

    # Asignar roles
    if role_order_override is not None:
        role_assignment = [r % model.n_roles for r in role_order_override]
        while len(role_assignment) < n_sections:
            role_assignment.append(role_assignment[-1])
        role_assignment = role_assignment[:n_sections]
    elif mode == "similarity":
        role_assignment = _assign_roles_similarity(source_sig, model, n_sections)
    elif mode == "position":
        role_assignment = _assign_roles_position(model, n_sections)
    else:  # hybrid
        role_assignment = _assign_roles_hybrid(source_sig, model, n_sections)

    if verbose:
        print(f"  Roles asignados: "
              f"{[model.roles[i].label for i in role_assignment]}")

    if dry_run:
        print(f"\n  [dry-run] Secciones que se generarían:")
        for i, ri in enumerate(role_assignment):
            r = model.roles[ri]
            print(f"    Sección {i+1}: rol={r.label}  "
                  f"θ_pitch={r.theta_mean.op_pitch}  "
                  f"θ_rhythm={r.theta_mean.op_rhythm}  "
                  f"θ_dynamics={r.theta_mean.op_dynamics}")
        return

    # Generar secciones
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tpb = 480
    # generated_sections: lista de (sec_name, melody_notes, all_notes)
    generated_sections: List[Tuple[str, List[RawNote], List[RawNote]]] = []
    prev_melody: Optional[List[RawNote]] = None

    print(f"\n  Generando secciones...")
    print(f"  {'─'*60}")

    for i, role_idx in enumerate(role_assignment):
        role = model.roles[role_idx]
        sec_name = f"sec{i+1:02d}_{role.label}"

        # Samplear θ con algo de varianza controlada
        theta_arr = role.theta_mean.to_array()
        try:
            noise = np.random.multivariate_normal(
                np.zeros(8), role.theta_cov * 0.05)
            theta_arr_noisy = np.clip(theta_arr + noise, 0, 1)
            theta = TransformVector.from_array(theta_arr_noisy)
        except Exception:
            theta = role.theta_mean

        # Aplicar transformación sobre el material fuente
        # Seleccionar la frase más representativa (la más larga) como base
        if len(input_midis) > 1:
            base_notes = max(
                [_parse_track_to_notes(
                    mido.MidiFile(p).tracks[_choose_melody_track_idx(mido.MidiFile(p))],
                    mido.MidiFile(p).ticks_per_beat)
                 for p in input_midis if Path(p).exists()],
                key=lambda ns: len(ns),
                default=source_notes_all,
            )
        else:
            base_notes = source_notes_all

        melody_notes = _apply_transform(
            base_notes, theta, key_pc, mode_str,
            bars_per_section, bpb_detected, rng)

        # Voice leading con sección anterior
        if prev_melody:
            melody_notes = _apply_voice_leading(
                prev_melody, melody_notes, key_pc, mode_str)

        # Construir progresión y acompañamiento
        acc_style_map = {
            "intro": "bass_only", "verse": "arpeggio",
            "development": "block", "climax": "block",
            "outro": "bass_only",
        }
        acc_style = acc_style_map.get(role.label, "arpeggio")
        progression = _select_progression(theta, key_pc,
                                           bars_per_section, bpb_detected)
        acc_vel = max(20, int(np.mean([n.velocity for n in melody_notes])) - 20)
        acc_notes  = _build_accompaniment(progression, key_pc,
                                           4, acc_vel, acc_style)
        bass_notes = _build_bass_line(progression, key_pc, max(20, acc_vel - 10))

        # Combinar y deduplicar
        all_notes = _deduplicate(melody_notes + acc_notes + bass_notes)
        all_notes.sort(key=lambda n: n.offset)

        generated_sections.append((sec_name, melody_notes, all_notes))
        prev_melody = melody_notes

        # Guardar MIDI individual de la sección
        sec_mid = mido.MidiFile(ticks_per_beat=tpb)
        sec_mid.tracks.append(
            _notes_to_midi_track(all_notes, tpb, final_tempo, sec_name))
        sec_path = out_path / f"{output_name}_{sec_name}.mid"
        sec_mid.save(str(sec_path))

        dur_s = bars_per_section * bpb_detected * 60.0 / final_tempo
        m_s, s_s = divmod(int(dur_s), 60)
        print(f"  ✓ {sec_path.name:<45}  {bars_per_section}c  "
              f"{len(all_notes):>4} notas  ~{m_s}:{s_s:02d}")

    # ── Concatenar todo en un MIDI completo ──────────────────────────────────
    # Recopilar todos los eventos con tiempo absoluto en ticks
    all_events: List[Tuple[int, str, int, int, int]] = []  # (abs_tick, on/off, pitch, vel, ch)
    marker_events: List[Tuple[int, str]] = []              # (abs_tick, name)

    cursor_ticks = 0
    for sec_name, _, sec_notes in generated_sections:
        marker_events.append((cursor_ticks, sec_name))
        sec_dur_ticks = bars_per_section * bpb_detected * tpb

        for n in sec_notes:
            t_on  = cursor_ticks + int(n.offset * tpb)
            t_off = cursor_ticks + int((n.offset + max(MIN_NOTE_DUR, n.duration)) * tpb)
            vel   = max(1, min(127, n.velocity))
            p     = max(0, min(127, n.pitch))
            ch    = 1 if p < 48 else 0
            all_events.append((t_on,  "on",  p, vel, ch))
            all_events.append((t_off, "off", p, 0,   ch))

        cursor_ticks += sec_dur_ticks

    # Ordenar: primero offs, luego ons para el mismo tick
    all_events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    # Construir track único
    full_mid = mido.MidiFile(ticks_per_beat=tpb)
    full_track = mido.MidiTrack()
    full_mid.tracks.append(full_track)

    full_track.append(mido.MetaMessage("set_tempo",
                                        tempo=mido.bpm2tempo(final_tempo), time=0))
    full_track.append(mido.MetaMessage("track_name",
                                        name=f"{output_name}_full", time=0))
    full_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    full_track.append(mido.Message("program_change", channel=1, program=32, time=0))

    # Intercalar markers con eventos de nota
    combined: List[Tuple[int, int, Any]] = []  # (abs_tick, priority, msg)
    for abs_tick, name in marker_events:
        combined.append((abs_tick, 0, mido.MetaMessage("marker", text=name, time=0)))
    for abs_tick, etype, pitch, vel, ch in all_events:
        priority = 1 if etype == "off" else 2
        combined.append((abs_tick, priority,
                          mido.Message("note_on" if etype == "on" else "note_off",
                                        channel=ch, note=pitch, velocity=vel, time=0)))

    combined.sort(key=lambda x: (x[0], x[1]))

    current_tick = 0
    for abs_tick, _, msg in combined:
        msg.time = max(0, abs_tick - current_tick)
        full_track.append(msg)
        current_tick = abs_tick

    full_track.append(mido.MetaMessage("end_of_track", time=0))
    full_path = out_path / f"{output_name}_full.mid"
    full_mid.save(str(full_path))

    n_gen       = len(generated_sections)
    total_bars  = n_gen * bars_per_section
    total_s     = total_bars * bpb_detected * 60.0 / final_tempo
    m, s = divmod(int(total_s), 60)

    print(f"\n{'═'*64}")
    print(f"  Obra generada: {n_gen} secciones, "
          f"{total_bars} compases, ~{m}:{s:02d}")
    print(f"  Roles: {' → '.join(model.roles[i].label for i in role_assignment)}")
    print(f"  Salida: {out_path}/")
    print(f"{'═'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: segment
# ══════════════════════════════════════════════════════════════════════════════

def cmd_segment(args) -> None:
    print(f"\n{'═'*64}")
    print(f"  ML ARCHITECT v{VERSION} — Segmentación")
    print(f"{'═'*64}")
    print(f"  MIDI: {args.midi}")

    sections = extract_sections(
        midi_path        = args.midi,
        num_sections     = args.num_sections,
        track_idx        = args.track,
        all_tracks       = args.all_tracks,
        fuzzy_threshold  = args.fuzzy,
        similarity_threshold = args.similarity,
        verbose          = args.verbose,
    )

    if not sections:
        print("  [ERROR] No se detectaron secciones.")
        return

    print(f"\n  Forma detectada: {''.join(s.label for s in sections)}")
    print(f"  {'─'*60}")
    for s in sections:
        print(f"  {s.label}  compases {s.bar_start:3d}–{s.bar_end:3d}"
              f"  {len(s.notes):>4} notas"
              f"  [{s.symbol[:30]}{'…' if len(s.symbol)>30 else ''}]")

    # Calcular y mostrar firma de cada sección
    if args.verbose:
        total = len(sections)
        print(f"\n  Firmas por sección:")
        for i, s in enumerate(sections):
            sig = _compute_signature(s.notes, i / max(total-1, 1))
            print(f"  {s.label}  tensión={sig.tension:.2f}  "
                  f"vel={sig.velocity_mean:.2f}  "
                  f"densidad={sig.density:.2f}  "
                  f"contorno={'↑' if sig.contour_dir>0.1 else '↓' if sig.contour_dir<-0.1 else '→'}")

    # Exportar JSON si se solicita
    if args.output:
        data = {
            "source": args.midi,
            "form": "".join(s.label for s in sections),
            "sections": [
                {
                    "label": s.label,
                    "bar_start": s.bar_start,
                    "bar_end": s.bar_end,
                    "n_notes": len(s.notes),
                    "symbol": s.symbol,
                }
                for s in sections
            ]
        }
        Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\n  Informe guardado: {args.output}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args) -> None:
    print(f"\n{'═'*64}")
    print(f"  ML ARCHITECT v{VERSION} — Entrenamiento")
    print(f"{'═'*64}")

    train(
        midi_dir        = args.corpus,
        output_model    = args.output,
        n_roles         = args.n_roles,
        assignment_mode = args.assignment,
        num_sections    = args.num_sections,
        verbose         = args.verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args) -> None:
    print(f"\n{'═'*64}")
    print(f"  ML ARCHITECT v{VERSION} — Generación")
    print(f"{'═'*64}")

    generate(
        input_midis         = args.input,
        model_path          = args.model,
        assignment_mode     = args.assignment,
        n_sections          = args.n_sections,
        role_order_override = args.role_order,
        bars_per_section    = args.bars_per_section,
        out_dir             = args.out_dir,
        output_name         = args.output_name,
        tempo               = args.tempo,
        dry_run             = args.dry_run,
        seed                = args.seed,
        verbose             = args.verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml_architect",
        description="ML ARCHITECT — Composición estructural dirigida por gramáticas aprendidas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    # ── SEGMENT ──────────────────────────────────────────────────────────────
    seg = sub.add_parser("segment",
        help="Detecta secciones formales de un MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Analiza un MIDI y extrae sus secciones mediante "
                    "gramática Sequitur + clustering espectral.")
    seg.add_argument("midi", type=str,
        help="Fichero MIDI de entrada")
    seg.add_argument("--num-sections", "-n", type=int, default=None,
        metavar="N",
        help="Número de secciones a detectar (default: inferencia automática)")
    seg.add_argument("--track", type=int, default=None,
        metavar="N",
        help="Índice del track a analizar (default: auto — mayor interés melódico)")
    seg.add_argument("--all-tracks", action="store_true",
        help="Incluir notas de todos los tracks en la salida de cada sección "
             "(default: solo track melódico)")
    seg.add_argument("--fuzzy", type=float, default=1.0, metavar="T",
        help="Umbral de similitud fuzzy para comparación de compases [0.0–1.0] "
             "(default: 1.0 = exacto)")
    seg.add_argument("--similarity", type=float, default=0.85, metavar="T",
        help="Umbral de similitud para fusión de clusters (default: 0.85)")
    seg.add_argument("--output", "-o", type=str, default=None, metavar="FILE",
        help="Guardar informe de secciones en JSON")
    seg.add_argument("--verbose", action="store_true",
        help="Informe detallado")
    seg.set_defaults(func=cmd_segment)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    trn = sub.add_parser("train",
        help="Entrena el modelo de transformaciones sobre un corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Analiza un directorio de MIDIs y aprende cómo cada pieza "
                    "transforma su material fuente en secciones.")
    trn.add_argument("corpus", type=str,
        help="Directorio con los MIDIs de entrenamiento (búsqueda recursiva)")
    trn.add_argument("--output", "-o", type=str, default="model.pkl",
        metavar="FILE",
        help="Ruta del modelo entrenado a guardar (default: model.pkl)")
    trn.add_argument("--n-roles", type=int, default=4, metavar="N",
        help="Número de roles a aprender (default: 4)")
    trn.add_argument("--assignment", type=str,
        choices=["similarity","position","hybrid"], default="hybrid",
        help="Modo de asignación de roles (default: hybrid)")
    trn.add_argument("--num-sections", "-n", type=int, default=None,
        metavar="N",
        help="Forzar nº de secciones en todo el corpus (default: inferencia)")
    trn.add_argument("--verbose", action="store_true",
        help="Informe detallado por fichero")
    trn.set_defaults(func=cmd_train)

    # ── GENERATE ─────────────────────────────────────────────────────────────
    gen = sub.add_parser("generate",
        help="Genera una obra completa a partir de frases y un modelo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Toma una o varias frases MIDI y un modelo entrenado y "
                    "genera una obra completa aplicando las transformaciones aprendidas.")
    gen.add_argument("input", nargs="+", type=str,
        help="Ficheros MIDI de entrada (frases semilla)")
    gen.add_argument("--model", "-m", type=str, required=True,
        metavar="FILE",
        help="Modelo entrenado (.pkl)")
    gen.add_argument("--assignment", type=str,
        choices=["similarity","position","hybrid"], default=None,
        help="Modo de asignación de roles (default: heredado del modelo)")
    gen.add_argument("--n-sections", type=int, default=None, metavar="N",
        help="Número de secciones a generar (default: n_roles del modelo)")
    gen.add_argument("--role-order", type=int, nargs="+", default=None,
        metavar="R",
        help="Orden manual de roles por índice (ej: --role-order 0 1 0 2 3)")
    gen.add_argument("--bars-per-section", type=int, default=8, metavar="N",
        help="Compases por sección (default: 8)")
    gen.add_argument("--out-dir", type=str, default="output", metavar="DIR",
        help="Directorio de salida (default: output/)")
    gen.add_argument("--output-name", type=str, default="song", metavar="NAME",
        help="Nombre base de los ficheros generados (default: song)")
    gen.add_argument("--tempo", type=int, default=None, metavar="BPM",
        help="Tempo de salida en BPM (default: detectado de los MIDIs de entrada)")
    gen.add_argument("--dry-run", action="store_true",
        help="Solo mostrar qué se generaría sin escribir ficheros")
    gen.add_argument("--seed", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)")
    gen.add_argument("--verbose", action="store_true",
        help="Informe detallado")
    gen.set_defaults(func=cmd_generate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
