#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       STYLE TRANSFER  v3.0                                   ║
║     Transferencia de estilo musical entre MIDIs mediante algoritmo           ║
║     genético + modelo de transformación aprendido                            ║
║                                                                              ║
║  MEJORAS v3.0 respecto a v2.1:                                               ║
║    [A] Vectorizacion por segmentos temporales: cada MIDI se divide en N      ║
║        ventanas y el modelo predice parametros distintos por seccion.        ║
║        Permite capturar crescendos, desarrollos y otras dinamicas locales.   ║
║    [B] Operadores armonicamente conscientes: modal_remap reescala notas      ║
║        a la tonalidad detectada en el destino; harmony_density añade         ║
║        notas consonantes del acorde activo (no aleatorias).                  ║
║    [C] GA con warm-start: la poblacion final de cada run se recicla como     ║
║        semilla para pares similares, reduciendo convergencia a la mitad.     ║
║    [D] Transferencia diferenciada por rol de pista: detecta melodia,         ║
║        bajo y acompañamiento y aplica parametros distintos a cada uno.      ║
║                                                                              ║
║  MEJORAS heredadas de v2.0:                                                  ║
║    [1] Nuevos operadores estructurales: note_subdivision, register_remap,    ║
║        add_bass_ostinato, velocity_shape                                     ║
║    [2] Fitness ponderado + penalizacion por perdida de notas                ║
║    [3] Augmentacion de datos sinteticos                                      ║
║    [4] Emparejamiento 1-a-1 por algoritmo hungaro                           ║
║    [5] Modelo MLP + fallback GBR con auto-seleccion por CV                  ║
║    [6] GA multi-objetivo NSGA-II                                             ║
║                                                                              ║
║  MODOS:                                                                      ║
║    train     — Aprende transformaciones de una coleccion de pares            ║
║                origen->destino y entrena un modelo de transferencia          ║
║    transform — Aplica el modelo entrenado a un MIDI                         ║
║    verbose   — Muestra en detalle nota a nota las transformaciones           ║
║    apply     — Aplica parametros de transformacion concretos (sin modelo)   ║
║    block     — Como transform pero omitiendo ciertos operadores              ║
║                                                                              ║
║  NUEVAS OPCIONES EN TRAIN:                                                   ║
║    --segments N       Ventanas temporales por pieza [default: 4]            ║
║    --warm-start       Reciclar poblaciones GA entre pares similares         ║
║    --per-track        Transferencia diferenciada por rol de pista            ║
║                                                                              ║
║  NUEVOS OPERADORES:                                                          ║
║    modal_remap        Reasigna notas a tonalidad detectada   0..1           ║
║    harmony_density    Añade notas del acorde activo          0..0.5         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre: mido, numpy, scikit-learn                                        ║
║    Opcional: tqdm, scipy                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import copy
import math
import random
import argparse
import pickle
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import mido
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

VERSION = "3.0"

# ══════════════════════════════════════════════════════════════════════════════
#  COLORES ANSI
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "reset":  "\033[0m", "bold":   "\033[1m",
    "green":  "\033[92m", "yellow": "\033[93m",
    "red":    "\033[91m", "cyan":   "\033[96m",
    "blue":   "\033[94m", "gray":   "\033[90m",
    "magenta":"\033[95m",
}

def c(key: str) -> str:
    return C.get(key, "")


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VECTOR_DIMS = [
    "pitch_mean", "pitch_std", "pitch_range",
    "velocity_mean", "velocity_std",
    "density", "interval_large_ratio",
    "ascent_ratio", "n_tracks", "n_instruments",
    "note_count_norm", "tempo_bpm_norm",
    "pitch_entropy", "rhythm_regularity",
    "low_register_ratio", "high_register_ratio",
]

# Pesos perceptuales por dimensión para el fitness (MEJORA 2)
# Mayor peso = más importante que el GA acierte en esa dimensión
FITNESS_WEIGHTS = np.array([
    0.8,   # pitch_mean
    0.6,   # pitch_std
    1.0,   # pitch_range
    1.5,   # velocity_mean        — muy perceptible
    1.2,   # velocity_std
    3.0,   # density              — crítico: era el punto más débil
    1.0,   # interval_large_ratio
    0.5,   # ascent_ratio
    0.4,   # n_tracks
    0.4,   # n_instruments
    2.5,   # note_count_norm      — crítico: correlaciona con density
    2.0,   # tempo_bpm_norm       — muy perceptible
    0.8,   # pitch_entropy
    1.0,   # rhythm_regularity
    1.5,   # low_register_ratio
    1.5,   # high_register_ratio
], dtype=float)
FITNESS_WEIGHTS /= FITNESS_WEIGHTS.sum()  # normalizar a suma=1


def vectorize_midi(midi_path: str) -> Optional[dict]:
    """Vectoriza un MIDI a dict {dim: float}. Solo mido+numpy."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  [warn] No se pudo leer {Path(midi_path).name}: {e}")
        return None

    notes, velocities = [], []
    instruments = set()
    n_tracks = len(mid.tracks)
    tempo_us = 500_000
    note_times = []

    for track in mid.tracks:
        current_program = 0
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
            if msg.type == "program_change":
                current_program = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
                instruments.add(current_program)
                note_times.append(abs_tick)

    if not notes:
        return None

    notes_arr = np.array(notes, dtype=float)
    velocities_arr = np.array(velocities, dtype=float)
    diffs = np.diff(notes_arr)

    total_ticks = max(note_times) if note_times else 1
    tpb = mid.ticks_per_beat or 480
    beats_total = total_ticks / tpb
    density = len(notes) / max(beats_total, 1.0) / 4.0
    density = float(np.clip(density, 0, 1))

    pc_counts = np.zeros(12)
    for n in notes:
        pc_counts[int(n) % 12] += 1
    pc_probs = pc_counts / (pc_counts.sum() + 1e-9)
    pitch_entropy = float(-np.sum(pc_probs * np.log2(pc_probs + 1e-9)) / 4.0)

    if len(note_times) > 1:
        iois = np.diff(sorted(note_times))
        iois = iois[iois > 0]
        if len(iois) > 1:
            cv = np.std(iois) / (np.mean(iois) + 1e-9)
            rhythm_regularity = float(np.clip(1.0 / (1.0 + cv), 0, 1))
        else:
            rhythm_regularity = 0.5
    else:
        rhythm_regularity = 0.5

    tempo_bpm = 60_000_000 / max(tempo_us, 1)

    return {
        "pitch_mean":           float(np.mean(notes_arr) / 127),
        "pitch_std":            float(np.clip(np.std(notes_arr) / 64, 0, 1)),
        "pitch_range":          float((np.max(notes_arr) - np.min(notes_arr)) / 127),
        "velocity_mean":        float(np.mean(velocities_arr) / 127),
        "velocity_std":         float(np.clip(np.std(velocities_arr) / 64, 0, 1)),
        "density":              density,
        "interval_large_ratio": float(np.mean(np.abs(diffs) > 5)) if len(diffs) > 0 else 0.0,
        "ascent_ratio":         float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.5,
        "n_tracks":             float(np.clip(n_tracks / 16, 0, 1)),
        "n_instruments":        float(np.clip(len(instruments) / 16, 0, 1)),
        "note_count_norm":      float(np.clip(len(notes) / 500, 0, 1)),
        "tempo_bpm_norm":       float(np.clip(tempo_bpm / 240, 0, 1)),
        "pitch_entropy":        float(np.clip(pitch_entropy, 0, 1)),
        "rhythm_regularity":    rhythm_regularity,
        "low_register_ratio":   float(np.mean(notes_arr < 52)),
        "high_register_ratio":  float(np.mean(notes_arr > 72)),
    }


def vec_to_array(vec: dict) -> np.ndarray:
    return np.array([vec.get(d, 0.0) for d in VECTOR_DIMS], dtype=float)


def weighted_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana ponderada por FITNESS_WEIGHTS."""
    diff = (a - b) * FITNESS_WEIGHTS
    return float(np.linalg.norm(diff))


def similarity_weighted(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud 0-1 usando distancia ponderada."""
    return float(math.exp(-weighted_distance(a, b)))


def similarity_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.exp(-np.linalg.norm(a - b)))


def similarity_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA A: VECTORIZACIÓN POR SEGMENTOS TEMPORALES
# ══════════════════════════════════════════════════════════════════════════════

def vectorize_midi_segmented(midi_path: str,
                              n_segments: int = 4) -> Optional[list[dict]]:
    """
    Divide el MIDI en n_segments ventanas temporales iguales y vectoriza
    cada una independientemente. Retorna lista de n_segments dicts de vector,
    o None si falla la lectura.

    Permite al modelo aprender transformaciones dependientes de la posición
    temporal (crescendo en el final, densidad creciente, etc.).
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return None

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000

    # Recoger todos los eventos con tick absoluto
    all_events: list[tuple[int, str, int, int, int]] = []  # (tick, type, note, vel, program)
    tempos: list[tuple[int, int]] = []  # (tick, tempo_us)

    for track in mid.tracks:
        program = 0
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((abs_tick, msg.tempo))
                tempo_us = msg.tempo
            if msg.type == "program_change":
                program = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                all_events.append((abs_tick, "note_on", msg.note, msg.velocity, program))

    if not all_events:
        return None

    all_events.sort(key=lambda e: e[0])
    total_ticks = all_events[-1][0] + 1
    seg_len = max(1, total_ticks // n_segments)

    segments = []
    for seg_idx in range(n_segments):
        t_start = seg_idx * seg_len
        t_end   = (seg_idx + 1) * seg_len if seg_idx < n_segments - 1 else total_ticks + 1

        seg_events = [(t, tp, n, v, p) for t, tp, n, v, p in all_events
                      if t_start <= t < t_end]

        if not seg_events:
            # Segmento vacío: interpolar con cero-vector
            seg_vec = {d: 0.0 for d in VECTOR_DIMS}
            seg_vec["tempo_bpm_norm"] = float(np.clip(
                60_000_000 / max(tempo_us, 1) / 240, 0, 1))
            segments.append(seg_vec)
            continue

        notes      = [e[2] for e in seg_events]
        velocities = [e[3] for e in seg_events]
        note_ticks = [e[0] for e in seg_events]
        instruments= set(e[4] for e in seg_events)

        notes_arr = np.array(notes, dtype=float)
        vels_arr  = np.array(velocities, dtype=float)
        diffs     = np.diff(notes_arr)

        beats = (t_end - t_start) / tpb
        density = float(np.clip(len(notes) / max(beats, 1.0) / 4.0, 0, 1))

        pc_counts = np.zeros(12)
        for n in notes:
            pc_counts[int(n) % 12] += 1
        pc_probs = pc_counts / (pc_counts.sum() + 1e-9)
        pitch_entropy = float(np.clip(
            -np.sum(pc_probs * np.log2(pc_probs + 1e-9)) / 4.0, 0, 1))

        if len(note_ticks) > 1:
            iois = np.diff(sorted(note_ticks))
            iois = iois[iois > 0]
            cv = float(np.std(iois) / (np.mean(iois) + 1e-9)) if len(iois) > 1 else 1.0
            rhythm_reg = float(np.clip(1.0 / (1.0 + cv), 0, 1))
        else:
            rhythm_reg = 0.5

        # Tempo activo en este segmento
        active_tempo = tempo_us
        for t_tick, t_val in tempos:
            if t_tick <= t_start:
                active_tempo = t_val

        seg_vec = {
            "pitch_mean":           float(np.mean(notes_arr) / 127),
            "pitch_std":            float(np.clip(np.std(notes_arr) / 64, 0, 1)),
            "pitch_range":          float((np.max(notes_arr) - np.min(notes_arr)) / 127),
            "velocity_mean":        float(np.mean(vels_arr) / 127),
            "velocity_std":         float(np.clip(np.std(vels_arr) / 64, 0, 1)),
            "density":              density,
            "interval_large_ratio": float(np.mean(np.abs(diffs) > 5)) if len(diffs) > 0 else 0.0,
            "ascent_ratio":         float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.5,
            "n_tracks":             float(np.clip(len(mid.tracks) / 16, 0, 1)),
            "n_instruments":        float(np.clip(len(instruments) / 16, 0, 1)),
            "note_count_norm":      float(np.clip(len(notes) / 200, 0, 1)),  # por segmento
            "tempo_bpm_norm":       float(np.clip(60_000_000 / max(active_tempo, 1) / 240, 0, 1)),
            "pitch_entropy":        pitch_entropy,
            "rhythm_regularity":    rhythm_reg,
            "low_register_ratio":   float(np.mean(notes_arr < 52)),
            "high_register_ratio":  float(np.mean(notes_arr > 72)),
        }
        segments.append(seg_vec)

    return segments


def segmented_vecs_to_array(segs: list[dict]) -> np.ndarray:
    """
    Concatena los vectores de todos los segmentos en un único array plano.
    Shape: (n_segments * len(VECTOR_DIMS),)
    """
    return np.concatenate([vec_to_array(s) for s in segs])


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA B: TONALIDAD Y OPERADORES ARMÓNICAMENTE CONSCIENTES
# ══════════════════════════════════════════════════════════════════════════════

# Plantillas de escalas en pitch-class (raíz=0)
SCALE_TEMPLATES = {
    "major":      [1,0,1,0,1,1,0,1,0,1,0,1],
    "minor":      [1,0,1,1,0,1,0,1,1,0,1,0],
    "dorian":     [1,0,1,1,0,1,0,1,0,1,1,0],
    "phrygian":   [1,1,0,1,0,1,0,1,1,0,1,0],
    "mixolydian": [1,0,1,0,1,1,0,1,0,1,1,0],
    "harmonic_minor": [1,0,1,1,0,1,0,1,1,0,0,1],
}

# Tríadas y séptimas por grado (intervalos sobre la raíz)
CHORD_TONES = {
    "major":   [0, 4, 7],
    "minor":   [0, 3, 7],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "dom7":    [0, 4, 7, 10],
    "maj7":    [0, 4, 7, 11],
    "min7":    [0, 3, 7, 10],
}


def detect_tonality(notes: list[int]) -> tuple[int, str]:
    """
    Detecta la tonalidad más probable de una lista de notas MIDI usando
    correlación de pitch-class profile con plantillas de Krumhansl-Schmuckler.
    Retorna (raíz_0_11, nombre_modo).
    """
    if not notes:
        return (0, "major")

    pc_counts = np.zeros(12)
    for n in notes:
        pc_counts[int(n) % 12] += 1
    pc_profile = pc_counts / (pc_counts.sum() + 1e-9)

    best_score  = -np.inf
    best_root   = 0
    best_mode   = "major"

    for mode, template in SCALE_TEMPLATES.items():
        tmpl = np.array(template, dtype=float)
        tmpl_norm = tmpl / (tmpl.sum() + 1e-9)
        for root in range(12):
            rotated = np.roll(tmpl_norm, root)
            score = float(np.dot(pc_profile, rotated))
            if score > best_score:
                best_score = score
                best_root  = root
                best_mode  = mode

    return best_root, best_mode


def scale_notes_set(root: int, mode: str) -> set[int]:
    """Retorna el conjunto de pitch-classes pertenecientes a la escala."""
    template = SCALE_TEMPLATES.get(mode, SCALE_TEMPLATES["major"])
    return {(root + i) % 12 for i, v in enumerate(template) if v}


def nearest_scale_note(note: int, scale_pcs: set[int]) -> int:
    """
    Encuentra el MIDI más cercano a `note` que pertenezca a `scale_pcs`,
    preservando la octava tanto como sea posible.
    """
    pc = note % 12
    if pc in scale_pcs:
        return note
    # Buscar la distancia cromática mínima (±6 semitonos)
    for delta in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]:
        candidate_pc = (pc + delta) % 12
        if candidate_pc in scale_pcs:
            return int(np.clip(note + delta, 0, 127))
    return note


def infer_chord(notes_window: list[int], root: int, mode: str) -> list[int]:
    """
    Infiere el acorde activo a partir de un grupo de notas recientes.
    Retorna los pitch-classes de los tonos del acorde más probable.
    """
    if not notes_window:
        return CHORD_TONES["major"][:]

    pc_counts = np.zeros(12)
    for n in notes_window:
        pc_counts[int(n) % 12] += 1

    best_score  = -1.0
    best_chord  = [0, 4, 7]
    template    = SCALE_TEMPLATES.get(mode, SCALE_TEMPLATES["major"])
    scale_pcs   = [i for i, v in enumerate(template) if v]

    for degree in scale_pcs:
        chord_root = (root + degree) % 12
        for chord_type, intervals in CHORD_TONES.items():
            chord_pcs = [(chord_root + iv) % 12 for iv in intervals]
            score = sum(pc_counts[pc] for pc in chord_pcs)
            if score > best_score:
                best_score  = score
                best_chord  = chord_pcs

    return best_chord


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA D: DETECCIÓN DE ROL DE PISTA
# ══════════════════════════════════════════════════════════════════════════════

def detect_track_role(track: "mido.MidiTrack",
                      all_tracks_stats: list[dict]) -> str:
    """
    Clasifica una pista como 'melody', 'bass' o 'accompaniment' según:
    - Registro medio (bajo → bajo, alto → melodía)
    - Densidad de notas (alta densidad → acompañamiento)
    - Variedad melódica (muchos intervalos distintos → melodía)
    Retorna 'melody' | 'bass' | 'accompaniment'.
    """
    notes, times = [], []
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append(msg.note)
            times.append(abs_tick)

    if not notes:
        return "accompaniment"

    notes_arr = np.array(notes, dtype=float)
    mean_pitch = float(np.mean(notes_arr))
    n_notes    = len(notes)

    # Densidad relativa a otras pistas
    max_notes = max((s["n_notes"] for s in all_tracks_stats), default=1)
    rel_density = n_notes / max(max_notes, 1)

    # Variedad interválica
    diffs = np.abs(np.diff(notes_arr))
    interval_variety = float(np.std(diffs)) if len(diffs) > 1 else 0.0

    if mean_pitch < 52:
        return "bass"
    elif rel_density > 0.7 and interval_variety < 3.0:
        return "accompaniment"
    else:
        return "melody"


def get_track_stats(mid: "mido.MidiFile") -> list[dict]:
    """Recopila estadísticas básicas por pista para la detección de rol."""
    stats = []
    for track in mid.tracks:
        notes = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
        stats.append({
            "n_notes":    len(notes),
            "mean_pitch": float(np.mean(notes)) if notes else 60.0,
        })
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  WARM-START POOL (MEJORA C)
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartPool:
    """
    Almacena la población final de cada run del GA indexada por vector destino.
    Cuando se inicia un nuevo run, busca el run más similar y usa su población
    como inicialización en lugar de comenzar desde cero.
    """

    def __init__(self, max_entries: int = 20):
        self.entries: list[tuple[np.ndarray, list[dict], float]] = []
        # (target_vec, population, best_fitness)
        self.max_entries = max_entries

    def add(self, target_vec: np.ndarray,
            population: list[dict], best_fitness: float):
        self.entries.append((target_vec.copy(), population, best_fitness))
        if len(self.entries) > self.max_entries:
            # Descartar el de menor fitness
            self.entries.sort(key=lambda e: -e[2])
            self.entries = self.entries[:self.max_entries]

    def get_warm_population(self,
                             target_vec: np.ndarray,
                             pop_size: int,
                             rng: random.Random) -> Optional[list[dict]]:
        """
        Retorna una población de tamaño pop_size basada en el run más similar,
        o None si no hay entradas o la similitud es muy baja.
        """
        if not self.entries:
            return None

        sims = [similarity_euclidean(target_vec, e[0]) for e in self.entries]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim < 0.3:
            return None  # demasiado diferente; mejor empezar de cero

        source_pop = self.entries[best_idx][1]

        # Mezclar: 50% de la población fuente (mutada) + 50% aleatoria
        warm_pop = []
        for p in source_pop[:pop_size // 2]:
            warm_pop.append(_mutate(copy.deepcopy(p), rng, mutation_rate=0.4, sigma=0.1))
        while len(warm_pop) < pop_size:
            warm_pop.append(random_params(rng))

        return warm_pop[:pop_size]


# ══════════════════════════════════════════════════════════════════════════════
#  EMPAREJAMIENTO 1-A-1 (ALGORITMO HÚNGARO)
# ══════════════════════════════════════════════════════════════════════════════

def hungarian_matching(src_vecs: dict[str, np.ndarray],
                        dst_vecs: dict[str, np.ndarray],
                        metric: str = "euclidean") -> list[tuple[str, str, float]]:
    """
    Empareja cada destino con un origen único usando el algoritmo húngaro
    (asignación óptima sin repetición). Cuando hay más destinos que orígenes,
    los orígenes sobrantes se reusan en una segunda pasada greedy.

    Retorna: lista de (src_path, dst_path, similaridad)
    """
    src_paths = list(src_vecs.keys())
    dst_paths = list(dst_vecs.keys())
    n_src = len(src_paths)
    n_dst = len(dst_paths)

    sim_fn = similarity_cosine if metric == "cosine" else similarity_euclidean

    # Matriz de costes (negativos de similitud para minimizar)
    n_rows = min(n_dst, n_src)
    cost_matrix = np.zeros((n_rows, n_src))
    for i, dp in enumerate(dst_paths[:n_rows]):
        for j, sp in enumerate(src_paths):
            cost_matrix[i, j] = -sim_fn(dst_vecs[dp], src_vecs[sp])

    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Implementación greedy del algoritmo húngaro (fallback sin scipy)
        row_ind, col_ind = _greedy_matching(cost_matrix)

    pairs = []
    used_src = set()
    for r, c in zip(row_ind, col_ind):
        dp = dst_paths[r]
        sp = src_paths[c]
        sim = sim_fn(dst_vecs[dp], src_vecs[sp])
        pairs.append((sp, dp, sim))
        used_src.add(c)

    # Segunda pasada: destinos sin pareja (cuando n_dst > n_src)
    for i in range(n_rows, n_dst):
        dp = dst_paths[i]
        # Greedy sobre todos los orígenes
        best_sp = max(src_paths, key=lambda sp: sim_fn(dst_vecs[dp], src_vecs[sp]))
        sim = sim_fn(dst_vecs[dp], src_vecs[best_sp])
        pairs.append((best_sp, dp, sim))

    return pairs


def _greedy_matching(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback greedy cuando scipy no está disponible."""
    n_rows, n_cols = cost_matrix.shape
    row_ind, col_ind = [], []
    used_cols = set()
    for r in range(n_rows):
        best_c = min(
            (c for c in range(n_cols) if c not in used_cols),
            key=lambda c: cost_matrix[r, c],
            default=0
        )
        row_ind.append(r)
        col_ind.append(best_c)
        used_cols.add(best_c)
    return np.array(row_ind), np.array(col_ind)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 1: OPERADORES DE TRANSFORMACIÓN AMPLIADOS
# ══════════════════════════════════════════════════════════════════════════════

TRANSFORM_PARAM_NAMES = [
    # v1.0 — base
    "pitch_shift",          # semitonos globales, -12..+12
    "tempo_factor",         # multiplicador tempo, 0.5..2.0
    "velocity_scale",       # escala velocidad, 0.5..1.5
    "velocity_offset",      # delta velocidad, -30..+30
    "density_thin",         # prob eliminar nota, 0..0.7
    "density_double",       # prob duplicar nota armónica, 0..0.4
    "quantize_strength",    # fuerza cuantización, 0..1
    "register_shift",       # desplazamiento octava, -12..+12
    # v2.0 — operadores estructurales
    "note_subdivision",     # prob de subdividir nota larga en N, 0..0.9
    "register_remap_low",   # umbral inferior remapeo registro, 0..1
    "register_remap_high",  # umbral superior remapeo registro, 0..1
    "bass_ostinato_prob",   # prob de añadir nota de bajo ostinato, 0..0.6
    "velocity_shape",       # 0=plano, 0.33=crescendo, 0.66=decrescendo, 1=arch
    # v3.0 — operadores armónicamente conscientes (MEJORA B)
    "modal_remap",          # fuerza de reasignación a escala destino, 0..1
    "harmony_density",      # prob de añadir nota del acorde activo, 0..0.5
]

PARAM_RANGES = {
    "pitch_shift":        (-12, 12),
    "tempo_factor":       (0.5, 2.0),
    "velocity_scale":     (0.5, 1.5),
    "velocity_offset":    (-30, 30),
    "density_thin":       (0.0, 0.7),
    "density_double":     (0.0, 0.4),
    "quantize_strength":  (0.0, 1.0),
    "register_shift":     (-12, 12),
    "note_subdivision":   (0.0, 0.9),
    "register_remap_low": (0.0, 1.0),
    "register_remap_high":(0.0, 1.0),
    "bass_ostinato_prob": (0.0, 0.6),
    "velocity_shape":     (0.0, 1.0),
    # v3.0
    "modal_remap":        (0.0, 1.0),
    "harmony_density":    (0.0, 0.5),
}

# Valores neutros normalizados [0,1] (sin transformación)
NEUTRAL_NORM = {
    "pitch_shift":        0.5,
    "tempo_factor":       1/3,    # 0.5 + 1/3*1.5 = 1.0
    "velocity_scale":     0.5,
    "velocity_offset":    0.5,
    "density_thin":       0.0,
    "density_double":     0.0,
    "quantize_strength":  0.0,
    "register_shift":     0.5,
    "note_subdivision":   0.0,
    "register_remap_low": 0.0,
    "register_remap_high":1.0,
    "bass_ostinato_prob": 0.0,
    "velocity_shape":     0.0,
    # v3.0
    "modal_remap":        0.0,
    "harmony_density":    0.0,
}


def normalize_params(params: dict) -> np.ndarray:
    arr = []
    for name in TRANSFORM_PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        val = params.get(name, lo + (hi - lo) * NEUTRAL_NORM[name])
        arr.append((val - lo) / (hi - lo))
    return np.clip(arr, 0, 1)


def denormalize_params(arr: np.ndarray, intensity: float = 1.0) -> dict:
    params = {}
    for i, name in enumerate(TRANSFORM_PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        neutral = NEUTRAL_NORM[name]
        blended = neutral + intensity * (float(arr[i]) - neutral)
        val = lo + blended * (hi - lo)
        params[name] = float(np.clip(val, lo, hi))
    return params


def neutral_params() -> dict:
    params = {}
    for name in TRANSFORM_PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        params[name] = lo + (hi - lo) * NEUTRAL_NORM[name]
    return params


def random_params(rng: random.Random) -> dict:
    return {
        "pitch_shift":        rng.uniform(-12, 12),
        "tempo_factor":       rng.uniform(0.6, 2.0),
        "velocity_scale":     rng.uniform(0.6, 1.4),
        "velocity_offset":    rng.uniform(-20, 25),
        "density_thin":       rng.uniform(0, 0.3),
        "density_double":     rng.uniform(0, 0.3),
        "quantize_strength":  rng.uniform(0, 0.8),
        "register_shift":     rng.uniform(-12, 12),
        "note_subdivision":   rng.uniform(0, 0.7),
        "register_remap_low": rng.uniform(0, 0.4),
        "register_remap_high":rng.uniform(0.6, 1.0),
        "bass_ostinato_prob": rng.uniform(0, 0.4),
        "velocity_shape":     rng.uniform(0, 1.0),
        "modal_remap":        rng.uniform(0, 0.8),
        "harmony_density":    rng.uniform(0, 0.4),
    }


def _velocity_shape_multiplier(position: float, shape_val: float) -> float:
    """
    Calcula el multiplicador de velocidad según la posición en la pieza (0-1)
    y el tipo de curva elegida.
    shape_val en [0,1]:
      0.00–0.25  → plano (sin cambio)
      0.25–0.50  → crescendo
      0.50–0.75  → decrescendo
      0.75–1.00  → arch (sube y baja)
    """
    if shape_val < 0.25:
        return 1.0
    elif shape_val < 0.50:
        # crescendo: 0.7 al inicio → 1.3 al final
        t = (shape_val - 0.25) / 0.25
        return 0.7 + 0.6 * position
    elif shape_val < 0.75:
        # decrescendo
        return 1.3 - 0.6 * position
    else:
        # arch
        return 0.7 + 0.6 * math.sin(math.pi * position)


def apply_transforms(midi_path: str, params: dict,
                     output_path: str,
                     verbose: bool = False,
                     collect_log: bool = False) -> "bool | tuple[bool, list[dict]]":
    """
    Aplica el dict de transformaciones (v2) a un MIDI y guarda el resultado.
    Incluye los 4 nuevos operadores estructurales.

    Si collect_log=True retorna (ok, event_log) donde event_log es una lista
    de dicts describiendo cada transformación aplicada nota a nota.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {midi_path}: {e}")
        return (False, []) if collect_log else False

    event_log: list[dict] = []  # solo se rellena si collect_log=True

    # Extraer parámetros
    pitch_shift    = int(round(params.get("pitch_shift", 0)))
    tempo_factor   = float(params.get("tempo_factor", 1.0))
    vel_scale      = float(params.get("velocity_scale", 1.0))
    vel_offset     = int(round(params.get("velocity_offset", 0)))
    thin_prob      = float(params.get("density_thin", 0.0))
    double_prob    = float(params.get("density_double", 0.0))
    quant_strength = float(params.get("quantize_strength", 0.0))
    reg_shift      = int(round(params.get("register_shift", 0)))
    subdiv_prob    = float(params.get("note_subdivision", 0.0))
    remap_lo       = float(params.get("register_remap_low", 0.0))
    remap_hi       = float(params.get("register_remap_high", 1.0))
    bass_prob      = float(params.get("bass_ostinato_prob", 0.0))
    vel_shape      = float(params.get("velocity_shape", 0.0))
    modal_strength = float(params.get("modal_remap", 0.0))        # v3.0
    harm_density   = float(params.get("harmony_density", 0.0))    # v3.0

    # v3.0 — parámetros per-track (si existen en el dict)
    track_params: Optional[list[dict]] = params.get("_track_params", None)

    total_shift = pitch_shift + reg_shift
    tpb = mid.ticks_per_beat or 480
    rng = random.Random(42)

    # v3.0 — Detectar tonalidad del MIDI para modal_remap y harmony_density
    all_notes_global: list[int] = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                all_notes_global.append(msg.note)
    tonal_root, tonal_mode = detect_tonality(all_notes_global)
    scale_pcs = scale_notes_set(tonal_root, tonal_mode)

    # v3.0 — Detectar rol de cada pista
    track_stats = get_track_stats(mid)
    track_roles = [detect_track_role(t, track_stats) for t in mid.tracks]

    # Pre-calcular duración total para position-based transforms
    total_ticks = 0
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
        total_ticks = max(total_ticks, t)
    total_ticks = max(total_ticks, 1)

    # Primer pase: recopilar note_on/note_off para calcular duraciones (necesario para subdivide)
    # Estructura: channel+note → (abs_tick_on, velocity)
    pending_on: dict[tuple, tuple] = {}
    note_durations: dict[tuple, list] = {}  # (ch, note, abs_tick_on) → dur_ticks

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                pending_on[key] = (abs_tick, msg.velocity)
            elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in pending_on:
                    on_tick, vel = pending_on.pop(key)
                    dur = abs_tick - on_tick
                    note_durations.setdefault(key, []).append((on_tick, dur))

    # Determinar umbral de "nota larga" para subdivisión (mediana de duraciones)
    all_durs = [d for lst in note_durations.values() for _, d in lst]
    subdiv_threshold = float(np.median(all_durs)) if all_durs else tpb

    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    bass_track = mido.MidiTrack()  # pista de bajo ostinato (MEJORA 1c)

    # ── Primer track de bajo si bass_prob > 0 ───────────────────────────
    has_bass_notes = False

    for track_idx, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        # v3.0 — aplicar parámetros diferenciados por rol de pista
        role = track_roles[track_idx] if track_idx < len(track_roles) else "accompaniment"
        if track_params and track_idx < len(track_params):
            tp = track_params[track_idx]
        else:
            tp = {}

        # Override por rol: si hay _track_params explícitos los usamos,
        # si no, aplicar ajustes heurísticos según el rol detectado
        eff_vel_scale  = float(tp.get("velocity_scale",  vel_scale))
        eff_vel_offset = int(round(tp.get("velocity_offset", vel_offset)))
        eff_subdiv     = float(tp.get("note_subdivision",   subdiv_prob))
        eff_thin       = float(tp.get("density_thin",        thin_prob))
        eff_modal      = float(tp.get("modal_remap",         modal_strength))
        eff_harm       = float(tp.get("harmony_density",     harm_density))
        eff_shift      = int(round(tp.get("pitch_shift",     total_shift)))

        # Ajustes heurísticos por rol cuando no hay override explícito
        if not tp:
            if role == "bass":
                eff_vel_scale  = vel_scale * 0.85
                eff_subdiv     = subdiv_prob * 0.3   # bajo: menos subdivisión
                eff_modal      = modal_strength * 0.5
                eff_harm       = 0.0                 # bajo: no añadir armonías
            elif role == "melody":
                eff_vel_scale  = vel_scale * 1.05
                eff_harm       = harm_density        # melodía: puede añadir notas de acorde
                eff_modal      = modal_strength
            else:  # accompaniment
                eff_vel_scale  = vel_scale * 0.9
                eff_subdiv     = subdiv_prob * 0.6
                eff_harm       = harm_density * 0.5

        abs_tick = 0
        pending_doubles = []
        recent_notes: list[int] = []   # ventana para inferir acorde activo

        for msg in track:
            abs_tick += msg.time
            position = abs_tick / total_ticks  # posición 0-1 en la pieza

            # ── Tempo ────────────────────────────────────────────────────────
            if msg.type == "set_tempo":
                new_tempo = int(msg.tempo / max(tempo_factor, 0.01))
                new_tempo = int(np.clip(new_tempo, 60_000, 4_000_000))
                new_track.append(msg.copy(tempo=new_tempo))
                continue

            if msg.type in ("note_on", "note_off"):
                # ── Density thinning ─────────────────────────────────────────
                if msg.type == "note_on" and msg.velocity > 0 and eff_thin > 0:
                    if rng.random() < eff_thin:
                        if collect_log:
                            event_log.append({
                                "op": "density_thin", "tick": abs_tick,
                                "note_orig": msg.note, "vel_orig": msg.velocity,
                                "action": "eliminated",
                            })
                        continue

                # ── Pitch shift + register remap ─────────────────────────────
                new_note = msg.note + eff_shift
                # Register remap: comprime/expande el rango de alturas (MEJORA 1b)
                if remap_lo > 0.0 or remap_hi < 1.0:
                    lo_midi = int(remap_lo * 127)
                    hi_midi = int(remap_hi * 127)
                    if hi_midi > lo_midi:
                        # Remap lineal del rango original (21-108) al nuevo rango
                        orig_range = 87.0
                        new_range = hi_midi - lo_midi
                        remapped = lo_midi + (new_note - 21) / orig_range * new_range
                        new_note = int(np.clip(remapped, lo_midi, hi_midi))
                new_note = int(np.clip(new_note, 0, 127))

                # ── Modal remap (MEJORA B): ajustar a la escala detectada ───────
                if eff_modal > 0 and msg.type == "note_on" and msg.velocity > 0:
                    if rng.random() < eff_modal:
                        new_note = nearest_scale_note(new_note, scale_pcs)

                # Mantener ventana de notas recientes para inferencia de acorde
                if msg.type == "note_on" and msg.velocity > 0:
                    recent_notes.append(new_note)
                    if len(recent_notes) > 8:
                        recent_notes.pop(0)

                # ── Velocity transform + shape ───────────────────────────────
                if msg.type == "note_on" and msg.velocity > 0:
                    shape_mult = _velocity_shape_multiplier(position, vel_shape)
                    new_vel = int(msg.velocity * eff_vel_scale * shape_mult + eff_vel_offset)
                    new_vel = int(np.clip(new_vel, 1, 127))
                else:
                    new_vel = msg.velocity

                # ── Quantización ─────────────────────────────────────────────
                time = msg.time
                quant_applied = False
                if quant_strength > 0 and time > 0:
                    grid = tpb // 4
                    if grid > 0:
                        nearest = round(time / grid) * grid
                        old_time = time
                        time = int(time + quant_strength * (nearest - time))
                        time = max(0, time)
                        quant_applied = (time != old_time)

                # ── Note subdivision (MEJORA 1a) ──────────────────────────────
                # Subdivide notas largas en N notas más cortas (aumenta densidad)
                if (msg.type == "note_on" and msg.velocity > 0
                        and eff_subdiv > 0 and rng.random() < eff_subdiv):
                    # Buscar duración de esta nota
                    key = (msg.channel, msg.note)
                    note_dur = subdiv_threshold
                    for on_t, dur in note_durations.get(key, []):
                        if abs(on_t - abs_tick) < tpb // 2:
                            note_dur = dur
                            break
                    if note_dur > subdiv_threshold * 0.8:
                        n_parts = rng.choice([2, 2, 4])  # mayoría en 2
                        part_dur = max(tpb // 8, int(note_dur / n_parts))
                        if collect_log:
                            event_log.append({
                                "op": "note_subdivision", "tick": abs_tick,
                                "note_orig": msg.note, "note_new": new_note,
                                "vel_orig": msg.velocity, "vel_new": new_vel,
                                "n_parts": n_parts, "part_dur_ticks": part_dur,
                                "action": f"subdivided_into_{n_parts}",
                            })
                        # Emitir N nota_on / nota_off cortos
                        first = True
                        for _ in range(n_parts):
                            on_t = time if first else 0
                            first = False
                            on_msg = mido.Message("note_on", channel=msg.channel,
                                                   note=new_note, velocity=new_vel, time=on_t)
                            off_msg = mido.Message("note_off", channel=msg.channel,
                                                    note=new_note, velocity=0, time=part_dur)
                            new_track.append(on_msg)
                            new_track.append(off_msg)
                        continue  # no añadir el mensaje original

                # ── Logging de transformaciones básicas ───────────────────────
                if collect_log and msg.type == "note_on" and msg.velocity > 0:
                    ops_applied = []
                    if total_shift != 0:
                        ops_applied.append(f"pitch:{msg.note}->{new_note}({total_shift:+d}st)")
                    if msg.velocity != new_vel:
                        ops_applied.append(f"vel:{msg.velocity}->{new_vel}")
                    if quant_applied:
                        ops_applied.append(f"quant")
                    event_log.append({
                        "op": "note_on",
                        "tick": abs_tick,
                        "track": mid.tracks.index(track),
                        "channel": msg.channel,
                        "note_orig": msg.note,
                        "note_new": new_note,
                        "vel_orig": msg.velocity,
                        "vel_new": new_vel,
                        "time_orig": msg.time,
                        "time_new": time,
                        "ops_applied": ops_applied,
                        "action": "transformed",
                    })

                new_msg = msg.copy(note=new_note, velocity=new_vel, time=time)
                new_track.append(new_msg)

                # ── Density doubling ──────────────────────────────────────────
                if (msg.type == "note_on" and msg.velocity > 0
                        and double_prob > 0 and rng.random() < double_prob):
                    interval = rng.choice([3, 4, 7, -3, -4])
                    doubled = int(np.clip(new_note + interval, 0, 127))
                    new_track.append(mido.Message("note_on", channel=msg.channel,
                                                   note=doubled, velocity=max(1, new_vel - 20),
                                                   time=0))
                    pending_doubles.append(
                        mido.Message("note_off", channel=msg.channel,
                                     note=doubled, velocity=0, time=tpb // 2))
                    if collect_log:
                        event_log.append({
                            "op": "density_double", "tick": abs_tick,
                            "note_base": new_note, "note_doubled": doubled,
                            "interval": interval, "action": "added_harmony",
                        })

                # ── Harmony density (MEJORA B): nota del acorde activo ────────
                if (msg.type == "note_on" and msg.velocity > 0
                        and eff_harm > 0 and rng.random() < eff_harm):
                    chord_pcs = infer_chord(recent_notes, tonal_root, tonal_mode)
                    # Elegir una nota del acorde que no sea la actual
                    chord_candidates = [
                        new_note + delta
                        for delta in range(-12, 13)
                        if (new_note + delta) % 12 in chord_pcs
                        and delta != 0
                        and 0 <= new_note + delta <= 127
                    ]
                    if chord_candidates:
                        # Preferir el intervalo más cercano
                        harm_note = min(chord_candidates, key=lambda n: abs(n - new_note))
                        harm_vel  = int(np.clip(new_vel * 0.7, 1, 127))
                        new_track.append(mido.Message("note_on", channel=msg.channel,
                                                       note=harm_note, velocity=harm_vel,
                                                       time=0))
                        pending_doubles.append(
                            mido.Message("note_off", channel=msg.channel,
                                         note=harm_note, velocity=0, time=tpb // 2))
                        if collect_log:
                            event_log.append({
                                "op": "harmony_density", "tick": abs_tick,
                                "note_base": new_note, "note_harmony": harm_note,
                                "chord_pcs": chord_pcs, "action": "added_chord_tone",
                            })

                # ── Bass ostinato (MEJORA 1c) ─────────────────────────────────
                # Añade notas de bajo en el canal 2 (programa 32 = fingered bass)
                if (msg.type == "note_on" and msg.velocity > 0
                        and bass_prob > 0 and rng.random() < bass_prob):
                    bass_note = int(np.clip(new_note - rng.choice([12, 19, 24]), 0, 55))
                    bass_vel  = int(np.clip(new_vel * 0.75, 30, 100))
                    bass_track.append(mido.Message("note_on", channel=2,
                                                    note=bass_note, velocity=bass_vel, time=0))
                    bass_track.append(mido.Message("note_off", channel=2,
                                                    note=bass_note, velocity=0, time=tpb // 2))
                    has_bass_notes = True
                    if collect_log:
                        event_log.append({
                            "op": "bass_ostinato", "tick": abs_tick,
                            "note_trigger": new_note, "note_bass": bass_note,
                            "vel_bass": bass_vel, "action": "added_bass",
                        })

                # Insertar doubles pendientes
                if pending_doubles and msg.type == "note_off":
                    for d in pending_doubles:
                        new_track.append(d)
                    pending_doubles.clear()

            else:
                new_track.append(msg)

        for d in pending_doubles:
            new_track.append(d)

    # Añadir pista de bajo si tiene contenido
    if has_bass_notes and len(bass_track) > 0:
        setup = mido.MidiTrack()
        setup.append(mido.Message("program_change", channel=2, program=32, time=0))
        for msg in bass_track:
            setup.append(msg)
        setup.append(mido.MetaMessage("end_of_track", time=0))
        new_mid.tracks.append(setup)

    try:
        new_mid.save(output_path)
        return (True, event_log) if collect_log else True
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar {output_path}: {e}")
        return (False, event_log) if collect_log else False


def apply_transforms_segmented(midi_path: str,
                                seg_params: list[dict],
                                output_path: str) -> bool:
    """
    Aplica transformaciones distintas por segmento temporal (MEJORA A).
    Interpola suavemente los parámetros entre segmentos para evitar
    saltos bruscos en la transición.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {midi_path}: {e}")
        return False

    tpb = mid.ticks_per_beat or 480
    n_segs = len(seg_params)

    total_ticks = 0
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
        total_ticks = max(total_ticks, t)
    total_ticks = max(total_ticks, 1)

    def interp_params(tick: int) -> dict:
        pos   = tick / total_ticks * n_segs
        seg_lo = int(min(pos, n_segs - 1))
        seg_hi = min(seg_lo + 1, n_segs - 1)
        t      = pos - seg_lo
        p_lo   = seg_params[seg_lo]
        p_hi   = seg_params[seg_hi]
        return {name: p_lo.get(name, neutral_params()[name]) * (1 - t) +
                       p_hi.get(name, neutral_params()[name]) * t
                for name in TRANSFORM_PARAM_NAMES}

    # Build by delegating to apply_transforms with interpolated params per message
    # For simplicity we use a combined params dict that encodes segmented info
    # via the _track_params slot for the first segment, then override per-note.
    # Full implementation: reconstruct MIDI tick by tick.
    rng_g = random.Random(42)
    all_notes_g: list[int] = []
    for tr in mid.tracks:
        for m in tr:
            if m.type == "note_on" and m.velocity > 0:
                all_notes_g.append(m.note)
    tonal_root_g, tonal_mode_g = detect_tonality(all_notes_g)
    scale_pcs_g = scale_notes_set(tonal_root_g, tonal_mode_g)
    track_stats_g = get_track_stats(mid)
    track_roles_g = [detect_track_role(t, track_stats_g) for t in mid.tracks]

    pending_on_d: dict = {}
    note_durations_g: dict = {}
    for tr in mid.tracks:
        at = 0
        for m in tr:
            at += m.time
            if m.type == "note_on" and m.velocity > 0:
                pending_on_d[(m.channel, m.note)] = at
            elif m.type in ("note_off",) or (m.type == "note_on" and m.velocity == 0):
                key = (m.channel, m.note)
                if key in pending_on_d:
                    on_t = pending_on_d.pop(key)
                    note_durations_g.setdefault(key, []).append((on_t, at - on_t))
    all_durs_g = [d for lst in note_durations_g.values() for _, d in lst]
    subdiv_thresh_g = float(np.median(all_durs_g)) if all_durs_g else tpb

    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    bass_tr = mido.MidiTrack()
    has_bass = False

    for ti, track in enumerate(mid.tracks):
        new_tr = mido.MidiTrack()
        new_mid.tracks.append(new_tr)
        role = track_roles_g[ti] if ti < len(track_roles_g) else "accompaniment"
        abs_tick = 0
        pd = []
        recent_n: list[int] = []

        for msg in track:
            abs_tick += msg.time
            p = interp_params(abs_tick)

            if role == "bass":
                p["velocity_scale"] = p.get("velocity_scale", 1.0) * 0.85
                p["note_subdivision"] = p.get("note_subdivision", 0.0) * 0.3
                p["harmony_density"] = 0.0
            elif role == "melody":
                p["velocity_scale"] = p.get("velocity_scale", 1.0) * 1.05
            else:
                p["velocity_scale"] = p.get("velocity_scale", 1.0) * 0.9
                p["note_subdivision"] = p.get("note_subdivision", 0.0) * 0.6

            if msg.type == "set_tempo":
                tf = float(p.get("tempo_factor", 1.0))
                nt = int(np.clip(msg.tempo / max(tf, 0.01), 60_000, 4_000_000))
                new_tr.append(msg.copy(tempo=nt))
                continue

            if msg.type in ("note_on", "note_off"):
                thin = float(p.get("density_thin", 0.0))
                if msg.type == "note_on" and msg.velocity > 0 and thin > 0:
                    if rng_g.random() < thin:
                        continue

                ts = int(round(p.get("pitch_shift", 0))) + int(round(p.get("register_shift", 0)))
                nn = int(np.clip(msg.note + ts, 0, 127))
                if float(p.get("modal_remap", 0.0)) > 0 and msg.type == "note_on" and msg.velocity > 0:
                    if rng_g.random() < p["modal_remap"]:
                        nn = nearest_scale_note(nn, scale_pcs_g)

                if msg.type == "note_on" and msg.velocity > 0:
                    recent_n.append(nn)
                    if len(recent_n) > 8:
                        recent_n.pop(0)
                    sm = _velocity_shape_multiplier(abs_tick / total_ticks,
                                                     float(p.get("velocity_shape", 0.0)))
                    nv = int(np.clip(msg.velocity * float(p.get("velocity_scale", 1.0)) * sm
                                     + float(p.get("velocity_offset", 0.0)), 1, 127))
                else:
                    nv = msg.velocity

                time = msg.time
                qs = float(p.get("quantize_strength", 0.0))
                if qs > 0 and time > 0:
                    grid = tpb // 4
                    if grid > 0:
                        nt2 = round(time / grid) * grid
                        time = max(0, int(time + qs * (nt2 - time)))

                subdiv = float(p.get("note_subdivision", 0.0))
                if msg.type == "note_on" and msg.velocity > 0 and subdiv > 0:
                    if rng_g.random() < subdiv:
                        key = (msg.channel, msg.note)
                        nd = next((d for ot, d in note_durations_g.get(key, [])
                                   if abs(ot - abs_tick) < tpb // 2), subdiv_thresh_g)
                        if nd > subdiv_thresh_g * 0.8:
                            np2 = rng_g.choice([2, 2, 4])
                            pd2 = max(tpb // 8, int(nd / np2))
                            first2 = True
                            for _ in range(np2):
                                ot3 = time if first2 else 0
                                first2 = False
                                new_tr.append(mido.Message("note_on", channel=msg.channel,
                                                            note=nn, velocity=nv, time=ot3))
                                new_tr.append(mido.Message("note_off", channel=msg.channel,
                                                            note=nn, velocity=0, time=pd2))
                            continue

                new_tr.append(msg.copy(note=nn, velocity=nv, time=time))

                dp = float(p.get("density_double", 0.0))
                if msg.type == "note_on" and msg.velocity > 0 and dp > 0 and rng_g.random() < dp:
                    iv = rng_g.choice([3, 4, 7, -3, -4])
                    dbl = int(np.clip(nn + iv, 0, 127))
                    new_tr.append(mido.Message("note_on", channel=msg.channel,
                                               note=dbl, velocity=max(1, nv-20), time=0))
                    pd.append(mido.Message("note_off", channel=msg.channel,
                                           note=dbl, velocity=0, time=tpb // 2))

                hd = float(p.get("harmony_density", 0.0))
                if msg.type == "note_on" and msg.velocity > 0 and hd > 0 and rng_g.random() < hd:
                    cpcs = infer_chord(recent_n, tonal_root_g, tonal_mode_g)
                    cands = [nn + d for d in range(-12, 13)
                              if (nn + d) % 12 in cpcs and d != 0 and 0 <= nn + d <= 127]
                    if cands:
                        hn = min(cands, key=lambda x: abs(x - nn))
                        new_tr.append(mido.Message("note_on", channel=msg.channel,
                                                    note=hn, velocity=int(np.clip(nv*0.7,1,127)),
                                                    time=0))
                        pd.append(mido.Message("note_off", channel=msg.channel,
                                               note=hn, velocity=0, time=tpb // 2))

                bp = float(p.get("bass_ostinato_prob", 0.0))
                if msg.type == "note_on" and msg.velocity > 0 and bp > 0 and rng_g.random() < bp:
                    bn = int(np.clip(nn - rng_g.choice([12, 19, 24]), 0, 55))
                    bv = int(np.clip(nv * 0.75, 30, 100))
                    bass_tr.append(mido.Message("note_on", channel=2, note=bn, velocity=bv, time=0))
                    bass_tr.append(mido.Message("note_off", channel=2, note=bn, velocity=0,
                                                time=tpb // 2))
                    has_bass = True

                if pd and msg.type == "note_off":
                    for d in pd:
                        new_tr.append(d)
                    pd.clear()
            else:
                new_tr.append(msg)

        for d in pd:
            new_tr.append(d)

    if has_bass and len(bass_tr) > 0:
        setup = mido.MidiTrack()
        setup.append(mido.Message("program_change", channel=2, program=32, time=0))
        for msg in bass_tr:
            setup.append(msg)
        setup.append(mido.MetaMessage("end_of_track", time=0))
        new_mid.tracks.append(setup)

    try:
        new_mid.save(output_path)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS PONDERADO CON PENALIZACIÓN POR PÉRDIDA DE NOTAS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_fitness(result_vec: np.ndarray,
                     target_vec: np.ndarray,
                     source_note_count: int,
                     result_note_count: int) -> float:
    """
    Fitness compuesto:
      - Similitud ponderada entre resultado y target
      - Penalización si el resultado tiene menos notas que el origen
        (el GA no debería converger en soluciones que eliminan contenido)
    """
    # Similitud ponderada principal
    diff = (result_vec - target_vec) * FITNESS_WEIGHTS
    weighted_dist = float(np.linalg.norm(diff))
    sim = math.exp(-weighted_dist)

    # Penalización por pérdida de notas (suave, no catastrófica)
    if result_note_count < source_note_count * 0.7:
        ratio = result_note_count / max(source_note_count, 1)
        penalty = 0.15 * (1 - ratio)  # máx penalización 15%
        sim = sim * (1 - penalty)

    return float(sim)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 6: NSGA-II SIMPLIFICADO (GA MULTI-OBJETIVO)
# ══════════════════════════════════════════════════════════════════════════════

def _dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """f1 domina a f2 si es mejor en todos los objetivos y estrictamente mejor en alguno."""
    return bool(np.all(f1 >= f2) and np.any(f1 > f2))


def _non_dominated_sort(fitness_matrix: np.ndarray) -> list[list[int]]:
    """Devuelve frontes de Pareto ordenados (frente 0 = no dominados)."""
    n = len(fitness_matrix)
    dominated_by = [0] * n
    dominates_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(fitness_matrix[i], fitness_matrix[j]):
                dominates_set[i].append(j)
            elif _dominates(fitness_matrix[j], fitness_matrix[i]):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(fitness_matrix: np.ndarray, front: list[int]) -> np.ndarray:
    """Calcula distancia de crowding para mantener diversidad en el frente."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    n_obj = fitness_matrix.shape[1]

    for obj in range(n_obj):
        vals = fitness_matrix[front, obj]
        sorted_idx = np.argsort(vals)
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        rng_obj = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
        if rng_obj < 1e-9:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (vals[sorted_idx[k+1]] - vals[sorted_idx[k-1]]) / rng_obj

    return distances


def evaluate_multiobjective(source_midi: str,
                              params: dict,
                              target_vec: np.ndarray,
                              source_note_count: int,
                              tmp_path: str) -> np.ndarray:
    """
    Evalúa un individuo en 3 objetivos:
      [0] Similitud general ponderada al target
      [1] Similitud en dimensiones de densidad/textura (density, note_count)
      [2] Similitud en dimensiones de dinámica (velocity_mean, velocity_std)
    Retorna array de 3 objetivos (mayor = mejor en todos).
    """
    ok = apply_transforms(source_midi, params, tmp_path)
    if not ok:
        return np.zeros(3)

    vec = vectorize_midi(tmp_path)
    if vec is None:
        return np.zeros(3)

    result_arr = vec_to_array(vec)
    result_note_count = int(vec.get("note_count_norm", 0) * 500)

    # Objetivo 1: similitud ponderada global
    obj1 = weighted_fitness(result_arr, target_vec, source_note_count, result_note_count)

    # Objetivo 2: densidad + nota count (las más difíciles de transferir)
    density_idx   = VECTOR_DIMS.index("density")
    notecount_idx = VECTOR_DIMS.index("note_count_norm")
    diff_density = abs(result_arr[density_idx] - target_vec[density_idx])
    diff_notes   = abs(result_arr[notecount_idx] - target_vec[notecount_idx])
    obj2 = float(math.exp(-(diff_density * 3 + diff_notes * 2.5)))

    # Objetivo 3: dinámica
    velmean_idx = VECTOR_DIMS.index("velocity_mean")
    velstd_idx  = VECTOR_DIMS.index("velocity_std")
    diff_vm = abs(result_arr[velmean_idx] - target_vec[velmean_idx])
    diff_vs = abs(result_arr[velstd_idx]  - target_vec[velstd_idx])
    obj3 = float(math.exp(-(diff_vm * 1.5 + diff_vs)))

    return np.array([obj1, obj2, obj3])


def run_genetic_algorithm(source_midi: str,
                           target_vec: np.ndarray,
                           n_generations: int = 100,
                           pop_size: int = 40,
                           convergence_tol: float = 0.01,
                           verbose: bool = False,
                           rng_seed: int = 42,
                           warm_pool: Optional[WarmStartPool] = None,
                           ) -> tuple[dict, float, list[dict]]:
    """
    NSGA-II simplificado con warm-start opcional (MEJORA C).
    Retorna (best_params, best_fitness_obj1, final_population).
    """
    import tempfile, shutil

    rng = random.Random(rng_seed)
    tmp_dir = tempfile.mkdtemp(prefix="st_v3_ga_")

    src_vec = vectorize_midi(source_midi)
    source_note_count = int(src_vec.get("note_count_norm", 0.1) * 500) if src_vec else 50

    # ── Población inicial: warm-start si disponible (MEJORA C) ──────────
    warm_pop = None
    if warm_pool is not None:
        warm_pop = warm_pool.get_warm_population(target_vec, pop_size, rng)

    if warm_pop is not None:
        population = warm_pop
        if verbose:
            print(f"    → Warm-start: reutilizando población similar")
    else:
        population = [neutral_params()]
        for _ in range(pop_size - 1):
            population.append(random_params(rng))

    # Evaluar población inicial
    fitness_matrix = np.zeros((pop_size, 3))
    for i, ind in enumerate(population):
        tmp_path = os.path.join(tmp_dir, f"tmp_{i}.mid")
        fitness_matrix[i] = evaluate_multiobjective(
            source_midi, ind, target_vec, source_note_count, tmp_path)

    best_params = copy.deepcopy(population[0])
    best_fitness = float(fitness_matrix[0, 0])
    stagnation = 0

    for gen in range(n_generations):
        # ── Selección NSGA-II ────────────────────────────────────────────────
        fronts = _non_dominated_sort(fitness_matrix)

        # Ranking: menor índice de frente = mejor; dentro del frente, mayor crowding = mejor
        rank = np.zeros(pop_size, dtype=int)
        crowd = np.zeros(pop_size)
        for fi, front in enumerate(fronts):
            for idx in front:
                rank[idx] = fi
            cd = _crowding_distance(fitness_matrix, front)
            for j, idx in enumerate(front):
                crowd[idx] = cd[j]

        def nsga_select(k=3):
            """Selección por torneo NSGA-II."""
            candidates = rng.sample(range(len(population)), k)
            best = candidates[0]
            for c_idx in candidates[1:]:
                if (rank[c_idx] < rank[best] or
                        (rank[c_idx] == rank[best] and crowd[c_idx] > crowd[best])):
                    best = c_idx
            return population[best]

        # ── Actualizar mejor (objetivo 1 = similitud global) ─────────────────
        max_f0_idx = int(np.argmax(fitness_matrix[:, 0]))
        if fitness_matrix[max_f0_idx, 0] > best_fitness:
            improvement = fitness_matrix[max_f0_idx, 0] - best_fitness
            stagnation = 0 if improvement >= convergence_tol else stagnation + 1
            best_fitness = fitness_matrix[max_f0_idx, 0]
            best_params = copy.deepcopy(population[max_f0_idx])
        else:
            stagnation += 1

        if verbose and gen % 15 == 0:
            f_means = fitness_matrix.mean(axis=0)
            print(f"    Gen {gen:3d}: best={best_fitness:.4f} "
                  f"obj=[{f_means[0]:.3f},{f_means[1]:.3f},{f_means[2]:.3f}]")

        if best_fitness > 0.94 or stagnation > 25:
            if verbose:
                print(f"    Convergencia gen {gen} (fitness={best_fitness:.4f})")
            break

        # ── Reproducción ─────────────────────────────────────────────────────
        # Elitismo: preservar frente de Pareto 0
        elite = [population[i] for i in fronts[0]] if fronts else []
        elite = elite[:max(1, pop_size // 8)]

        new_pop = list(elite)
        new_fit = np.zeros((pop_size, 3))
        for i in range(len(elite)):
            new_fit[i] = fitness_matrix[[j for j, p in enumerate(population) if p is elite[i % len(elite)]][0]]

        sigma = max(0.04, 0.25 * (1 - best_fitness))

        while len(new_pop) < pop_size:
            p1 = nsga_select()
            p2 = nsga_select()
            child = _crossover(p1, p2, rng)
            child = _mutate(child, rng, sigma=sigma)
            new_pop.append(child)

        # Evaluar descendencia nueva
        for i in range(len(elite), pop_size):
            tmp_path = os.path.join(tmp_dir, f"tmp_{i}.mid")
            new_fit[i] = evaluate_multiobjective(
                source_midi, new_pop[i], target_vec, source_note_count, tmp_path)

        population = new_pop
        fitness_matrix = new_fit

    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return best_params, best_fitness, population


def _crossover(p1: dict, p2: dict, rng: random.Random) -> dict:
    """Cruce uniforme con blend aritmético en algunos params."""
    child = {}
    for name in TRANSFORM_PARAM_NAMES:
        if rng.random() < 0.5:
            child[name] = p1[name]
        else:
            # 20% de probabilidad de blend aritmético
            if rng.random() < 0.2:
                alpha = rng.random()
                child[name] = alpha * p1[name] + (1 - alpha) * p2[name]
            else:
                child[name] = p2[name]
    return child


def _mutate(params: dict, rng: random.Random,
            mutation_rate: float = 0.3, sigma: float = 0.15) -> dict:
    """Mutación gaussiana adaptativa."""
    new_params = copy.deepcopy(params)
    for name in TRANSFORM_PARAM_NAMES:
        if rng.random() < mutation_rate:
            lo, hi = PARAM_RANGES[name]
            delta = rng.gauss(0, (hi - lo) * sigma)
            new_params[name] = float(np.clip(params[name] + delta, lo, hi))
    return new_params


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 3: AUGMENTACIÓN DE DATOS SINTÉTICOS
# ══════════════════════════════════════════════════════════════════════════════

def augment_training_data(X: np.ndarray, Y: np.ndarray,
                           n_augment: int = 4,
                           noise_sigma: float = 0.02,
                           rng_seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera N réplicas sintéticas de cada muestra con jitter gaussiano
    en el espacio de features y targets. Preserva los límites [0,1].

    Esto simula variedad de orígenes ligeramente distintos sin necesidad
    de ejecutar más GAs — la asunción es que pequeñas variaciones en el
    vector origen corresponden a pequeñas variaciones en los params.
    """
    rng = np.random.RandomState(rng_seed)
    X_aug = [X]
    Y_aug = [Y]

    for _ in range(n_augment):
        x_noise = rng.normal(0, noise_sigma, X.shape)
        y_noise = rng.normal(0, noise_sigma * 0.5, Y.shape)
        X_aug.append(np.clip(X + x_noise, 0, 1))
        Y_aug.append(np.clip(Y + y_noise, 0, 1))

    return np.vstack(X_aug), np.vstack(Y_aug)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 5: MODELO MLP + FALLBACK GBR
# ══════════════════════════════════════════════════════════════════════════════

class StyleTransferModel:
    """
    Modelo de transferencia v3:
    - Soporta vectorización global (n_segments=1) o por segmentos (n_segments>1)
    - Predice parámetros por segmento cuando n_segments>1
    - Selección automática MLP vs GBR por CV
    """

    def __init__(self, n_segments: int = 1):
        self.model: Optional[Pipeline] = None
        self.model_type: str = "unknown"
        self.n_segments: int = n_segments
        self.dim_names = VECTOR_DIMS
        self.param_names = TRANSFORM_PARAM_NAMES
        self.training_meta: dict = {}

    def _build_mlp(self) -> Pipeline:
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,          # L2 regularización
            learning_rate="adaptive",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(mlp, n_jobs=-1)),
        ])

    def _build_gbr(self) -> Pipeline:
        base = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=42,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(base, n_jobs=-1)),
        ])

    def fit(self, X: np.ndarray, Y: np.ndarray, meta: dict = None):
        """
        Entrena MLP y GBR y elige el mejor por CV si hay suficientes muestras.
        Con pocas muestras (<20), usa directamente GBR.
        """
        n_samples = len(X)
        self.training_meta = meta or {}

        if n_samples < 20:
            # Pocas muestras: GBR es más estable
            print(f"    {c('yellow')}[modelo] {n_samples} muestras → usando GBR "
                  f"(MLP requiere ≥20){c('reset')}")
            self.model = self._build_gbr()
            self.model.fit(X, Y)
            self.model_type = "GBR"
            return

        # Con suficientes muestras, comparar por CV
        cv_folds = min(5, n_samples // 4)
        print(f"    Comparando MLP vs GBR ({cv_folds}-fold CV)...")

        gbr_pipe = self._build_gbr()
        mlp_pipe = self._build_mlp()

        # CV sobre el primer output (representativo)
        y_first = Y[:, 0]
        try:
            from sklearn.pipeline import Pipeline as SKPipeline
            from sklearn.preprocessing import StandardScaler as SKScaler

            # CV manual sobre pipeline completo
            gbr_simple = Pipeline([("s", StandardScaler()),
                                    ("r", GradientBoostingRegressor(n_estimators=200,
                                     max_depth=4, random_state=42))])
            mlp_simple = Pipeline([("s", StandardScaler()),
                                    ("r", MLPRegressor(hidden_layer_sizes=(64, 32),
                                     max_iter=500, random_state=42,
                                     early_stopping=False))])

            gbr_cv = cross_val_score(gbr_simple, X, y_first, cv=cv_folds,
                                      scoring="neg_mean_squared_error").mean()
            mlp_cv = cross_val_score(mlp_simple, X, y_first, cv=cv_folds,
                                      scoring="neg_mean_squared_error").mean()

            print(f"    CV MSE — GBR: {-gbr_cv:.5f}  MLP: {-mlp_cv:.5f}")

            if mlp_cv > gbr_cv:
                self.model = mlp_pipe
                self.model_type = "MLP"
                print(f"    {c('green')}→ Seleccionado: MLP{c('reset')}")
            else:
                self.model = gbr_pipe
                self.model_type = "GBR"
                print(f"    {c('green')}→ Seleccionado: GBR{c('reset')}")
        except Exception as e:
            print(f"    {c('yellow')}CV falló ({e}), usando GBR por defecto{c('reset')}")
            self.model = gbr_pipe
            self.model_type = "GBR"

        self.model.fit(X, Y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x puede ser:
          - shape (n_dims,)              → vectorización global → un array de params
          - shape (n_segments * n_dims,) → vectorización segmentada → un array de params
                                           (el modelo aprende un único mapping del vec concatenado)
        Retorna: (len(TRANSFORM_PARAM_NAMES),) normalizado [0,1]
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado.")
        return np.clip(self.model.predict(x.reshape(1, -1))[0], 0, 1)

    def predict_segmented(self, segs: list[dict]) -> list[dict]:
        """
        Dado una lista de vectores-segmento, predice parámetros para cada uno.
        Retorna lista de len(segs) dicts de parámetros desnormalizados.
        """
        result = []
        for seg_vec in segs:
            x = vec_to_array(seg_vec)
            params_norm = self.predict(x)
            result.append(denormalize_params(params_norm))
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Modelo guardado en: {path}  (tipo: {self.model_type})")

    @staticmethod
    def load(path: str) -> "StyleTransferModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, StyleTransferModel):
            raise TypeError("El archivo no contiene un StyleTransferModel v2 válido")
        return model


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def collect_midis(folder: str) -> list[str]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Carpeta no encontrada: {folder}")
    return sorted(str(f) for f in p.iterdir()
                  if f.is_file() and f.suffix.lower() in (".mid", ".midi"))


def vectorize_folder(paths: list[str], label: str) -> dict[str, np.ndarray]:
    vecs = {}
    it = tqdm(paths, desc=f"  {label}", leave=False) if HAS_TQDM else paths
    for p in it:
        if not HAS_TQDM:
            print(f"    {label}: {Path(p).name}", end="\r")
        v = vectorize_midi(p)
        if v is not None:
            vecs[p] = vec_to_array(v)
    if not HAS_TQDM:
        print()
    return vecs


# ══════════════════════════════════════════════════════════════════════════════
#  MODO TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def mode_train(args):
    print(f"\n{c('bold')}{c('cyan')}  STYLE TRANSFER v{VERSION} — modo TRAIN{c('reset')}")
    print(f"  {c('magenta')}Mejoras v3: segmentos={args.segments} · "
          f"warm_start={args.warm_start} · per_track={args.per_track} · "
          f"hungarian · NSGA-II · MLP/GBR · augmentation×{args.augment}{c('reset')}")

    src_midis = collect_midis(args.src)
    dst_midis = collect_midis(args.dst)

    if not src_midis or not dst_midis:
        print(f"  {c('red')}ERROR: Carpetas vacías o sin MIDIs.{c('reset')}")
        sys.exit(1)

    print(f"\n  Origen:  {len(src_midis)} MIDIs ({args.src})")
    print(f"  Destino: {len(dst_midis)} MIDIs ({args.dst})")

    # ── [1/5] Vectorizar ──────────────────────────────────────────────────
    print(f"\n{c('bold')}  [1/5] Vectorizando colecciones "
          f"({'global' if args.segments == 1 else f'{args.segments} segmentos'})...{c('reset')}")
    src_vecs = vectorize_folder(src_midis, "Origen")
    dst_vecs = vectorize_folder(dst_midis, "Destino")
    print(f"  Vectorizados: {len(src_vecs)} origen, {len(dst_vecs)} destino")

    if not src_vecs or not dst_vecs:
        print(f"  {c('red')}ERROR: No se pudieron vectorizar suficientes MIDIs.{c('reset')}")
        sys.exit(1)

    # Vectorización segmentada para el modelo (MEJORA A)
    n_segs = args.segments
    src_segs: dict[str, list[dict]] = {}
    dst_segs: dict[str, list[dict]] = {}
    if n_segs > 1:
        print(f"  Vectorizando segmentos...")
        for p in src_midis:
            s = vectorize_midi_segmented(p, n_segs)
            if s is not None:
                src_segs[p] = s
        for p in dst_midis:
            s = vectorize_midi_segmented(p, n_segs)
            if s is not None:
                dst_segs[p] = s

    # ── [2/5] Emparejamiento húngaro 1-a-1 ───────────────────────────────
    print(f"\n{c('bold')}  [2/5] Emparejamiento 1-a-1 (algoritmo húngaro)...{c('reset')}")
    pairs = hungarian_matching(src_vecs, dst_vecs, metric=args.metric)
    src_usage = {}
    for sp, dp, sim in pairs:
        src_usage[sp] = src_usage.get(sp, 0) + 1
    print(f"  {len(pairs)} pares. Orígenes únicos: {len(src_usage)}")
    if args.verbose:
        for sp, dp, sim in pairs:
            print(f"    {Path(dp).name[:30]}  ←→  {Path(sp).name[:30]}  (sim={sim:.3f})")

    # ── [3/5] GA multi-objetivo con warm-start ────────────────────────────
    print(f"\n{c('bold')}  [3/5] GA multi-objetivo (NSGA-II"
          f"{', warm-start' if args.warm_start else ''})...{c('reset')}")
    print(f"  Generaciones: {args.ga_gens}  |  Población: {args.ga_pop}")

    warm_pool = WarmStartPool() if args.warm_start else None

    X_train, Y_train, results_log = [], [], []
    it = tqdm(pairs, desc="  Pares GA") if HAS_TQDM else pairs

    for idx, (src_path, dst_path, _) in enumerate(it):
        if not HAS_TQDM:
            print(f"  Par {idx+1}/{len(pairs)}: {Path(src_path).name[:20]} → "
                  f"{Path(dst_path).name[:20]}")

        src_vec_dict = vectorize_midi(src_path)
        if src_vec_dict is None:
            continue
        src_arr = vec_to_array(src_vec_dict)
        dst_arr = dst_vecs.get(dst_path)
        if dst_arr is None:
            continue

        best_params, best_fit, final_pop = run_genetic_algorithm(
            source_midi=src_path,
            target_vec=dst_arr,
            n_generations=args.ga_gens,
            pop_size=args.ga_pop,
            convergence_tol=args.ga_tol,
            verbose=args.verbose,
            rng_seed=idx * 7 + 13,
            warm_pool=warm_pool,
        )

        # Añadir al pool para futuros runs similares (MEJORA C)
        if warm_pool is not None:
            warm_pool.add(dst_arr, final_pop, best_fit)

        if not HAS_TQDM:
            print(f"    fitness: {best_fit:.4f}")

        # Feature vector: global o segmentado según args.segments
        if n_segs > 1 and src_path in src_segs:
            feature_vec = segmented_vecs_to_array(src_segs[src_path])
        else:
            feature_vec = src_arr

        params_norm = normalize_params(best_params)
        X_train.append(feature_vec)
        Y_train.append(params_norm)
        results_log.append({
            "src": str(src_path), "dst": str(dst_path),
            "ga_fitness": best_fit, "best_params": best_params,
        })

    if not X_train:
        print(f"  {c('red')}ERROR: No se obtuvieron datos de entrenamiento.{c('reset')}")
        sys.exit(1)

    X_raw = np.array(X_train)
    Y_raw = np.array(Y_train)
    print(f"\n  Muestras brutas: {len(X_raw)}")

    # ── [4/5] Augmentación de datos sintéticos ────────────────────────────
    print(f"\n{c('bold')}  [4/5] Augmentación de datos (×{args.augment})...{c('reset')}")
    X, Y = augment_training_data(X_raw, Y_raw, n_augment=args.augment)
    print(f"  Muestras tras augmentación: {len(X)} "
          f"({len(X_raw)} reales + {len(X) - len(X_raw)} sintéticas)")

    # ── [5/5] Entrenamiento del modelo ────────────────────────────────────
    print(f"\n{c('bold')}  [5/5] Entrenando modelo (MLP vs GBR auto-selección)...{c('reset')}")

    model = StyleTransferModel(n_segments=n_segs)
    model.fit(X, Y, meta={
        "version": VERSION,
        "n_pairs": len(X_raw),
        "n_augmented": len(X),
        "n_segments": n_segs,
        "per_track": args.per_track,
        "warm_start_used": args.warm_start,
        "src_folder": args.src,
        "dst_folder": args.dst,
        "ga_gens": args.ga_gens,
        "ga_pop": args.ga_pop,
        "augment": args.augment,
        "metric": args.metric,
        "results_log": results_log,
    })

    Y_pred = model.model.predict(X_raw)
    train_mse = float(np.mean((Y_raw - Y_pred) ** 2))
    print(f"  MSE en datos reales (normalizado): {train_mse:.5f}")

    model_path = args.model if args.model.endswith(".pkl") else args.model + ".pkl"
    model.save(model_path)

    log_path = model_path.replace(".pkl", "_training_log.json")
    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, default=_ser, ensure_ascii=False)
    print(f"  ✓ Log guardado en: {log_path}")

    print(f"\n{c('green')}{c('bold')}  ✓ Entrenamiento v3 completado.{c('reset')}")
    print(f"  Modelo: {model_path}  ({model.model_type}, n_segments={n_segs})")
    print(f"  Pares reales: {len(X_raw)}  |  Total augmentado: {len(X)}")

    print(f"\n  {c('bold')}Parámetros medios aprendidos (incluyendo operadores v3):{c('reset')}")
    Y_mean = Y_raw.mean(axis=0)
    for i, name in enumerate(TRANSFORM_PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        val = lo + Y_mean[i] * (hi - lo)
        neutral_val = lo + NEUTRAL_NORM[name] * (hi - lo)
        delta = val - neutral_val
        col = c("green") if abs(delta) > 0.1 * (hi - lo) else c("gray")
        new_tag = " ← v3" if name in ("modal_remap", "harmony_density") else ""
        print(f"    {name:<24}: {col}{val:+.3f}{c('reset')}  (Δ={delta:+.3f}){new_tag}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO TRANSFORM (v3 — soporta segmentos y per-track)
# ══════════════════════════════════════════════════════════════════════════════

def mode_transform(args):
    print(f"\n{c('bold')}{c('cyan')}  STYLE TRANSFER v{VERSION} — modo TRANSFORM{c('reset')}")

    if not Path(args.model).exists():
        print(f"  {c('red')}ERROR: Modelo no encontrado: {args.model}{c('reset')}")
        sys.exit(1)

    model = StyleTransferModel.load(args.model)
    meta  = model.training_meta
    n_segs = getattr(model, "n_segments", 1)
    per_track = meta.get("per_track", False)

    print(f"  Modelo: {meta.get('n_pairs','?')} pares, tipo={model.model_type}, "
          f"n_segments={n_segs}, per_track={per_track}")

    input_path = args.input_midi
    if not Path(input_path).exists():
        print(f"  {c('red')}ERROR: No se encuentra: {input_path}{c('reset')}")
        sys.exit(1)

    output_path = args.output or str(
        Path(input_path).parent / f"{Path(input_path).stem}_v3_transferred.mid")

    # ── Vectorizar ────────────────────────────────────────────────────────
    print(f"\n  Vectorizando: {Path(input_path).name}")
    vec = vectorize_midi(input_path)
    if vec is None:
        print(f"  {c('red')}ERROR: No se pudo vectorizar el MIDI.{c('reset')}")
        sys.exit(1)
    x_global = vec_to_array(vec)

    # ── Predecir con el modelo ─────────────────────────────────────────────
    print(f"  Prediciendo transformaciones...")

    if n_segs > 1:
        segs = vectorize_midi_segmented(input_path, n_segs)
        if segs:
            x_feat = segmented_vecs_to_array(segs)  # (n_segs * 16,) — same as training
        else:
            # Fallback: tile global vector
            x_feat = np.tile(x_global, n_segs)
            segs   = [vec] * n_segs

        # Single prediction from the full concatenated feature
        params_norm  = model.predict(x_feat)
        params_global = denormalize_params(params_norm, intensity=args.intensity)

        # Build per-segment param list by interpolating intensity across the piece
        # (segment 0 at 60% intensity → segment n-1 at 100% intensity)
        seg_params_list = []
        for si, _ in enumerate(segs):
            seg_intensity = args.intensity * (0.6 + 0.4 * si / max(n_segs - 1, 1))
            seg_params_list.append(denormalize_params(params_norm, intensity=seg_intensity))

        print(f"  Aplicando transformación segmentada ({n_segs} secciones)...")
        ok = apply_transforms_segmented(input_path, seg_params_list, output_path)
        params = params_global  # para el display
    else:
        params_norm = model.predict(x_global)
        params = denormalize_params(params_norm, intensity=args.intensity)

        # Per-track si el modelo fue entrenado con esa opción
        if per_track:
            print(f"  Aplicando transformación diferenciada por pista...")
        ok = apply_transforms(input_path, params, output_path)

    if not ok:
        print(f"  {c('red')}ERROR al aplicar.{c('reset')}")
        sys.exit(1)

    # ── Mostrar parámetros ────────────────────────────────────────────────
    groups = {
        "Tempo / Tono": ["pitch_shift", "tempo_factor", "register_shift"],
        "Dinámica":     ["velocity_scale", "velocity_offset", "velocity_shape"],
        "Densidad":     ["density_thin", "density_double", "note_subdivision",
                         "bass_ostinato_prob"],
        "Registro":     ["register_remap_low", "register_remap_high"],
        "Ritmo":        ["quantize_strength"],
        "Armonía v3":   ["modal_remap", "harmony_density"],
    }
    print(f"\n  {c('bold')}Transformaciones aplicadas (intensidad={args.intensity:.1f}):{c('reset')}")
    print(f"  {'─'*60}")
    for group_name, param_list in groups.items():
        active = [(n, params[n]) for n in param_list if n in params]
        if not active:
            continue
        print(f"  {c('bold')}{c('cyan')}  {group_name}{c('reset')}")
        for name, val in active:
            lo, hi = PARAM_RANGES[name]
            neutral_val = lo + NEUTRAL_NORM[name] * (hi - lo)
            delta = val - neutral_val
            sig = abs(delta) / max(hi - lo, 1e-9)
            col = c("green") if sig > 0.15 else c("yellow") if sig > 0.05 else c("gray")
            tag = " ← v3" if name in ("modal_remap", "harmony_density") else ""
            print(f"    {name:<24}: {col}{val:+.3f}{c('reset')}  Δ={delta:+.3f}{tag}")
    print(f"  {'─'*60}")

    # Detección de tonalidad para informar al usuario
    all_notes_t = []
    try:
        mid_t = mido.MidiFile(input_path)
        for tr in mid_t.tracks:
            for m in tr:
                if m.type == "note_on" and m.velocity > 0:
                    all_notes_t.append(m.note)
        root_t, mode_t = detect_tonality(all_notes_t)
        NOTE_NAMES_T = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        print(f"\n  Tonalidad detectada: {NOTE_NAMES_T[root_t]} {mode_t}")
    except Exception:
        pass

    # ── Comparación vectorial ─────────────────────────────────────────────
    vec_out = vectorize_midi(output_path)
    if vec_out:
        x_out  = vec_to_array(vec_out)
        delta  = x_out - x_global
        changes = [(VECTOR_DIMS[i], float(x_global[i]), float(x_out[i]), float(delta[i]))
                   for i in range(len(VECTOR_DIMS)) if abs(delta[i]) > 0.02]
        if changes:
            print(f"\n  {c('bold')}Cambios vectoriales:{c('reset')}")
            for dim, before, after, d in sorted(changes, key=lambda r: -abs(r[3]))[:10]:
                arrow = c("green")+"↑"+c("reset") if d > 0 else c("red")+"↓"+c("reset")
                w = FITNESS_WEIGHTS[VECTOR_DIMS.index(dim)]
                print(f"    {dim:<26}: {before:.3f} → {after:.3f}  {arrow}  w={w:.2f}")

        # Mostrar evolución temporal si fue segmentado
        if n_segs > 1 and segs:
            print(f"\n  {c('bold')}Perfil temporal del MIDI original ({n_segs} segmentos):{c('reset')}")
            key_seg_dims = ["density", "velocity_mean", "pitch_mean", "rhythm_regularity"]
            for dim in key_seg_dims:
                vals = [f"{s.get(dim, 0):.2f}" for s in segs]
                print(f"    {dim:<22}: {' → '.join(vals)}")

    print(f"\n{c('green')}{c('bold')}  ✓ Transformación v3 completada.{c('reset')}")
    print(f"  Salida: {output_path}")


def mode_verbose(args):
    """
    Muestra en detalle nota a nota las transformaciones aplicadas.
    Puede trabajar con un modelo (igual que transform) o con parámetros
    explícitos vía --params.
    """
    print(f"\n{c('bold')}{c('cyan')}  STYLE TRANSFER v{VERSION} — modo VERBOSE{c('reset')}")

    input_path = args.input_midi
    if not Path(input_path).exists():
        print(f"  {c('red')}ERROR: No se encuentra: {input_path}{c('reset')}")
        sys.exit(1)

    # ── Obtener parámetros ────────────────────────────────────────────────
    if args.params:
        print(f"  Fuente de parámetros: {c('yellow')}--params explícitos{c('reset')}")
        try:
            params = parse_params_string(args.params)
        except ValueError as e:
            print(f"  {c('red')}ERROR: {e}{c('reset')}")
            sys.exit(1)
    elif args.model:
        if not Path(args.model).exists():
            print(f"  {c('red')}ERROR: Modelo no encontrado: {args.model}{c('reset')}")
            sys.exit(1)
        print(f"  Fuente de parámetros: {c('yellow')}modelo {args.model}{c('reset')}")
        model = StyleTransferModel.load(args.model)
        vec = vectorize_midi(input_path)
        if vec is None:
            print(f"  {c('red')}ERROR: No se pudo vectorizar el MIDI.{c('reset')}")
            sys.exit(1)
        params_norm = model.predict(vec_to_array(vec))
        params = denormalize_params(params_norm, intensity=args.intensity)
    else:
        print(f"  {c('red')}ERROR: Especifica --model o --params{c('reset')}")
        sys.exit(1)

    # ── Mostrar parámetros completos ──────────────────────────────────────
    print_params_table(params, title="Parámetros que se van a aplicar")

    # ── Perfil vectorial de entrada ───────────────────────────────────────
    vec_in = vectorize_midi(input_path)
    if vec_in:
        x_in = vec_to_array(vec_in)
        print(f"\n  {c('bold')}Perfil vectorial de entrada:{c('reset')}")
        print(f"  {'Dimensión':<26}  {'Valor':>6}  {'Barra (peso)':}")
        print(f"  {'─'*62}")
        for dim, val in sorted(zip(VECTOR_DIMS, x_in),
                               key=lambda t: -FITNESS_WEIGHTS[VECTOR_DIMS.index(t[0])]):
            w = FITNESS_WEIGHTS[VECTOR_DIMS.index(dim)]
            bar_n = int(val * 24)
            bar = "█" * bar_n + "░" * (24 - bar_n)
            w_bar = "●" * int(w * 60)
            print(f"  {dim:<26}: {val:6.3f}  {c('cyan')}{bar}{c('reset')}  "
                  f"w={w:.2f} {c('gray')}{w_bar[:12]}{c('reset')}")

    # ── Aplicar con collect_log=True ──────────────────────────────────────
    output_path = args.output or str(
        Path(input_path).parent / f"{Path(input_path).stem}_verbose_out.mid")

    print(f"\n  {c('bold')}Aplicando transformaciones (log nota a nota)...{c('reset')}")
    result = apply_transforms(input_path, params, output_path, collect_log=True)
    ok, event_log = result

    if not ok:
        print(f"  {c('red')}ERROR al aplicar.{c('reset')}")
        sys.exit(1)

    # ── Tabla de eventos ──────────────────────────────────────────────────
    # Contar por operador
    op_counts: dict[str, int] = {}
    for ev in event_log:
        op_counts[ev["op"]] = op_counts.get(ev["op"], 0) + 1

    print(f"\n  {c('bold')}Resumen de eventos ({len(event_log)} total):{c('reset')}")
    print(f"  {'─'*62}")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        bar = "▓" * min(40, count)
        print(f"    {op:<24}: {count:4d}  {c('yellow')}{bar}{c('reset')}")
    print(f"  {'─'*62}")

    # ── Detalle nota a nota (limitado a max_events) ───────────────────────
    max_events = args.max_events
    note_events = [e for e in event_log if e["op"] == "note_on"]
    other_events = [e for e in event_log if e["op"] != "note_on"]

    print(f"\n  {c('bold')}Transformaciones por operador especial:{c('reset')}")
    for ev in other_events[:max_events]:
        tick = ev["tick"]
        if ev["op"] == "density_thin":
            print(f"  {c('gray')}  t={tick:6d}  ELIMINADA  "
                  f"nota={midi_note_name(ev['note_orig'])}({ev['note_orig']})  "
                  f"vel={ev['vel_orig']}{c('reset')}")
        elif ev["op"] == "note_subdivision":
            print(f"  {c('magenta')}  t={tick:6d}  SUBDIVIDE×{ev['n_parts']}  "
                  f"{midi_note_name(ev['note_orig'])}→{midi_note_name(ev['note_new'])}  "
                  f"vel {ev['vel_orig']}→{ev['vel_new']}{c('reset')}")
        elif ev["op"] == "density_double":
            print(f"  {c('blue')}  t={tick:6d}  DOBLA  "
                  f"base={midi_note_name(ev['note_base'])}  "
                  f"armonía={midi_note_name(ev['note_doubled'])}  "
                  f"intervalo={ev['interval']:+d}st{c('reset')}")
        elif ev["op"] == "bass_ostinato":
            print(f"  {c('yellow')}  t={tick:6d}  BAJO  "
                  f"trigger={midi_note_name(ev['note_trigger'])}  "
                  f"bass={midi_note_name(ev['note_bass'])}  "
                  f"vel={ev['vel_bass']}{c('reset')}")

    if len(other_events) > max_events:
        print(f"  {c('gray')}  ... y {len(other_events) - max_events} eventos más "
              f"(usa --max-events para ver más){c('reset')}")

    print(f"\n  {c('bold')}Transformaciones nota_on (primeras {min(max_events, len(note_events))}):{c('reset')}")
    print(f"  {'t':>7}  {'trk':>3}  {'ch':>3}  "
          f"{'orig':>5}  {'→':>1}  {'nuevo':>5}  "
          f"{'vel_o':>5}  {'→':>1}  {'vel_n':>5}  "
          f"{'operaciones'}")
    print(f"  {'─'*78}")
    for ev in note_events[:max_events]:
        orig_name = midi_note_name(ev["note_orig"])
        new_name  = midi_note_name(ev["note_new"])
        pitch_changed = ev["note_orig"] != ev["note_new"]
        vel_changed   = ev["vel_orig"]  != ev["vel_new"]
        pitch_col = c("green") if pitch_changed else c("gray")
        vel_col   = c("yellow") if vel_changed   else c("gray")
        ops_str   = "  ".join(ev.get("ops_applied", [])) or c("gray") + "sin cambio" + c("reset")
        print(f"  {ev['tick']:>7}  {ev['track']:>3}  {ev['channel']:>3}  "
              f"{pitch_col}{orig_name:>5}{c('reset')}  →  "
              f"{pitch_col}{new_name:>5}{c('reset')}  "
              f"{vel_col}{ev['vel_orig']:>5}{c('reset')}  →  "
              f"{vel_col}{ev['vel_new']:>5}{c('reset')}  "
              f"{ops_str}")

    if len(note_events) > max_events:
        print(f"  {c('gray')}  ... y {len(note_events) - max_events} notas más "
              f"(usa --max-events N para ver más){c('reset')}")

    # ── Comparación vectorial antes/después ───────────────────────────────
    vec_out = vectorize_midi(output_path)
    if vec_out and vec_in:
        x_out = vec_to_array(vec_out)
        delta = x_out - x_in
        print(f"\n  {c('bold')}Cambio vectorial antes → después:{c('reset')}")
        print(f"  {'Dimensión':<26}  {'Antes':>6}  {'Después':>7}  {'Δ':>7}  {'Barra Δ'}")
        print(f"  {'─'*72}")
        for i, dim in enumerate(VECTOR_DIMS):
            if abs(delta[i]) < 0.005:
                continue
            w = FITNESS_WEIGHTS[i]
            sign = "+" if delta[i] > 0 else ""
            arrow = c("green") + "↑" + c("reset") if delta[i] > 0 else c("red") + "↓" + c("reset")
            bar_n = int(abs(delta[i]) * 30)
            bar_col = c("green") if delta[i] > 0 else c("red")
            bar = bar_col + ("▓" * min(20, bar_n)) + c("reset")
            print(f"  {dim:<26}: {x_in[i]:6.3f}  →  {x_out[i]:6.3f}  "
                  f"{arrow}{sign}{delta[i]:.3f}  {bar}  w={w:.2f}")

    print(f"\n{c('green')}{c('bold')}  ✓ Modo verbose completado.{c('reset')}")
    print(f"  MIDI de salida: {output_path}")
    print(f"  Log completo exportable con --log-json")

    # ── Exportar JSON si se pide ──────────────────────────────────────────
    if args.log_json:
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump(event_log, f, indent=2, ensure_ascii=False)
        print(f"  Log JSON guardado en: {args.log_json}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO APPLY — Aplicar parámetros explícitos sin modelo
# ══════════════════════════════════════════════════════════════════════════════

def mode_apply(args):
    """
    Aplica directamente parámetros de transformación especificados por el usuario.
    No necesita un modelo entrenado.

    Ejemplo:
      python style_transfer_v2.py apply entrada.mid \\
          --params "tempo_factor=1.8, pitch_shift=3, velocity_scale=1.2" \\
          --output salida.mid
    """
    print(f"\n{c('bold')}{c('cyan')}  STYLE TRANSFER v{VERSION} — modo APPLY{c('reset')}")

    input_path = args.input_midi
    if not Path(input_path).exists():
        print(f"  {c('red')}ERROR: No se encuentra: {input_path}{c('reset')}")
        sys.exit(1)

    # ── Parsear parámetros ────────────────────────────────────────────────
    try:
        params = parse_params_string(args.params or "")
    except ValueError as e:
        print(f"  {c('red')}ERROR en --params: {e}{c('reset')}")
        sys.exit(1)

    # ── Mostrar tabla de lo que se va a aplicar ───────────────────────────
    print(f"  Entrada: {Path(input_path).name}")
    print_params_table(params, title="Parámetros a aplicar")

    # ── Verificar que hay algo no neutro ──────────────────────────────────
    active = []
    for name, val in params.items():
        lo, hi = PARAM_RANGES[name]
        neutral_val = lo + NEUTRAL_NORM[name] * (hi - lo)
        if abs(val - neutral_val) > 1e-4:
            active.append(name)

    if not active:
        print(f"  {c('yellow')}Aviso: todos los parámetros son neutros "
              f"(el MIDI de salida será igual al de entrada).{c('reset')}")

    print(f"  Operadores activos: {c('green')}"
          f"{', '.join(active) if active else 'ninguno'}{c('reset')}")

    # ── Aplicar ───────────────────────────────────────────────────────────
    output_path = args.output or str(
        Path(input_path).parent / f"{Path(input_path).stem}_applied.mid")

    ok = apply_transforms(input_path, params, output_path)
    if not ok:
        print(f"  {c('red')}ERROR al aplicar las transformaciones.{c('reset')}")
        sys.exit(1)

    # ── Comparación vectorial ─────────────────────────────────────────────
    vec_in  = vectorize_midi(input_path)
    vec_out = vectorize_midi(output_path)
    if vec_in and vec_out:
        x_in  = vec_to_array(vec_in)
        x_out = vec_to_array(vec_out)
        delta = x_out - x_in
        changes = [(VECTOR_DIMS[i], x_in[i], x_out[i], delta[i])
                   for i in range(len(VECTOR_DIMS)) if abs(delta[i]) > 0.02]
        if changes:
            print(f"\n  {c('bold')}Cambios vectoriales aplicados:{c('reset')}")
            for dim, before, after, d in sorted(changes, key=lambda r: -abs(r[3]))[:12]:
                arrow = c("green") + "↑" + c("reset") if d > 0 else c("red") + "↓" + c("reset")
                w = FITNESS_WEIGHTS[VECTOR_DIMS.index(dim)]
                print(f"    {dim:<26}: {before:.3f} → {after:.3f}  {arrow}  "
                      f"Δ={d:+.3f}  w={w:.2f}")
        else:
            print(f"  {c('gray')}(ningún cambio vectorial significativo){c('reset')}")

    print(f"\n{c('green')}{c('bold')}  ✓ Apply completado.{c('reset')}")
    print(f"  Salida: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO BLOCK — Transform con operadores bloqueados
# ══════════════════════════════════════════════════════════════════════════════

def mode_block(args):
    """
    Igual que transform (usa el modelo) pero fuerza a neutro los parámetros
    de los operadores indicados en --block.

    Ejemplo:
      python style_transfer_v2.py block entrada.mid --model modelo.pkl \\
          --block "tempo_factor, register_remap_low, register_remap_high" \\
          --output salida.mid
    """
    print(f"\n{c('bold')}{c('cyan')}  STYLE TRANSFER v{VERSION} — modo BLOCK{c('reset')}")

    # ── Cargar modelo ─────────────────────────────────────────────────────
    if not Path(args.model).exists():
        print(f"  {c('red')}ERROR: Modelo no encontrado: {args.model}{c('reset')}")
        sys.exit(1)

    model = StyleTransferModel.load(args.model)
    meta = model.training_meta
    print(f"  Modelo: {meta.get('n_pairs','?')} pares, tipo={model.model_type}")

    # ── Parsear lista de bloqueos ─────────────────────────────────────────
    try:
        blocked = parse_block_list(args.block or "")
    except ValueError as e:
        print(f"  {c('red')}ERROR en --block: {e}{c('reset')}")
        sys.exit(1)

    if not blocked:
        print(f"  {c('yellow')}Aviso: --block vacío. Este modo es equivalente a transform.{c('reset')}")

    print(f"  Operadores bloqueados ({len(blocked)}): "
          f"{c('red')}{', '.join(sorted(blocked)) or 'ninguno'}{c('reset')}")

    # ── Vectorizar entrada ────────────────────────────────────────────────
    input_path = args.input_midi
    if not Path(input_path).exists():
        print(f"  {c('red')}ERROR: No se encuentra: {input_path}{c('reset')}")
        sys.exit(1)

    vec = vectorize_midi(input_path)
    if vec is None:
        print(f"  {c('red')}ERROR: No se pudo vectorizar el MIDI.{c('reset')}")
        sys.exit(1)
    x = vec_to_array(vec)

    # ── Predecir y bloquear ───────────────────────────────────────────────
    params_norm = model.predict(x)
    params_full = denormalize_params(params_norm, intensity=args.intensity)

    # Para cada parámetro bloqueado, reemplazar por su valor neutro
    neutral = neutral_params()
    params_blocked = copy.deepcopy(params_full)
    for name in blocked:
        params_blocked[name] = neutral[name]

    # ── Mostrar tabla comparativa ─────────────────────────────────────────
    print_params_table(params_blocked,
                       title=f"Parámetros del modelo (bloqueados en gris)",
                       blocked=blocked)

    # Mostrar qué se pierde al bloquear
    if blocked:
        print(f"\n  {c('bold')}Impacto de los bloqueos:{c('reset')}")
        for name in sorted(blocked):
            lo, hi = PARAM_RANGES[name]
            neutral_val = neutral[name]
            model_val   = params_full[name]
            delta = model_val - neutral_val
            significance = abs(delta) / max(hi - lo, 1e-9)
            impact = ("alto" if significance > 0.20
                      else "medio" if significance > 0.08
                      else "bajo")
            impact_col = (c("red") if impact == "alto"
                          else c("yellow") if impact == "medio"
                          else c("gray"))
            print(f"    {name:<24}: modelo={model_val:+.3f}  "
                  f"neutro={neutral_val:+.3f}  Δ={delta:+.3f}  "
                  f"impacto={impact_col}{impact}{c('reset')}")

    # ── Aplicar ───────────────────────────────────────────────────────────
    output_path = args.output or str(
        Path(input_path).parent / f"{Path(input_path).stem}_blocked.mid")

    print(f"\n  Aplicando (sin {', '.join(sorted(blocked)) or 'ningún bloqueo'})...")
    ok = apply_transforms(input_path, params_blocked, output_path)
    if not ok:
        print(f"  {c('red')}ERROR al aplicar.{c('reset')}")
        sys.exit(1)

    # ── Comparación vectorial con y sin bloqueos ──────────────────────────
    import tempfile
    tmp_full = tempfile.mktemp(suffix=".mid")
    apply_transforms(input_path, params_full, tmp_full)

    vec_in      = vectorize_midi(input_path)
    vec_full    = vectorize_midi(tmp_full)
    vec_blocked = vectorize_midi(output_path)

    if vec_in and vec_full and vec_blocked:
        x_in      = vec_to_array(vec_in)
        x_full    = vec_to_array(vec_full)
        x_blocked = vec_to_array(vec_blocked)

        print(f"\n  {c('bold')}Comparativa: modelo completo vs con bloqueos:{c('reset')}")
        print(f"  {'Dimensión':<26}  {'Origen':>7}  {'Completo':>9}  "
              f"{'Bloqueado':>10}  {'Diferencia'}")
        print(f"  {'─'*74}")

        for i, dim in enumerate(VECTOR_DIMS):
            diff = abs(x_full[i] - x_blocked[i])
            if diff < 0.01:
                continue
            w = FITNESS_WEIGHTS[i]
            sign = "+" if (x_blocked[i] - x_full[i]) >= 0 else ""
            impact_col = c("red") if diff * w > 0.05 else c("yellow") if diff > 0.05 else c("gray")
            print(f"  {dim:<26}: {x_in[i]:7.3f}  {x_full[i]:9.3f}  "
                  f"{x_blocked[i]:10.3f}  "
                  f"{impact_col}{sign}{x_blocked[i]-x_full[i]:+.3f}{c('reset')}")

        # Similitud global
        def wsim(a, b):
            return math.exp(-float(np.linalg.norm((a-b)*FITNESS_WEIGHTS)))

        sim_full    = wsim(x_full, x_blocked)  # cuánto divergen
        sim_blocked_dst = wsim(x_blocked, x_full)  # mismo, mostramos pérdida
        print(f"\n  Divergencia completo vs bloqueado: {1-sim_full:.4f}  "
              f"({'alto impacto' if 1-sim_full > 0.02 else 'bajo impacto'})")

    try:
        os.unlink(tmp_full)
    except Exception:
        pass

    print(f"\n{c('green')}{c('bold')}  ✓ Block completado.{c('reset')}")
    print(f"  Salida: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="style_transfer_v3.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            STYLE TRANSFER v3.0 — Transferencia de estilo musical
            ──────────────────────────────────────────────────────
            Modos: train · transform · verbose · apply · block
            Nuevo en v3: segmentos, warm-start GA, operadores armonicos, per-track
        """),
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ─────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Entrenar el modelo v3")
    p_train.add_argument("--src", required=True)
    p_train.add_argument("--dst", required=True)
    p_train.add_argument("--model", default="style_transfer_v3_model")
    p_train.add_argument("--ga-gens",    type=int,   default=100)
    p_train.add_argument("--ga-pop",     type=int,   default=40)
    p_train.add_argument("--ga-tol",     type=float, default=0.01)
    p_train.add_argument("--augment",    type=int,   default=4)
    p_train.add_argument("--segments",   type=int,   default=1,
        help="Nº de segmentos temporales para vectorización [default: 1]")
    p_train.add_argument("--warm-start", action="store_true",
        help="Reutilizar población GA entre pares similares (MEJORA C)")
    p_train.add_argument("--per-track",  action="store_true",
        help="Transferencia diferenciada por rol de pista (MEJORA D)")
    p_train.add_argument("--metric",     default="euclidean",
                         choices=["euclidean", "cosine"])
    p_train.add_argument("--verbose", "-v", action="store_true")

    # ── transform ─────────────────────────────────────────────────────────
    p_transform = sub.add_parser("transform",
        help="Transformar un MIDI usando el modelo entrenado")
    p_transform.add_argument("input_midi")
    p_transform.add_argument("--model",     required=True)
    p_transform.add_argument("--output",    default=None)
    p_transform.add_argument("--intensity", type=float, default=1.0)
    p_transform.add_argument("--verbose", "-v", action="store_true")

    # ── verbose ───────────────────────────────────────────────────────────
    p_verbose = sub.add_parser("verbose",
        help="Mostrar detalle nota a nota de las transformaciones aplicadas")
    p_verbose.add_argument("input_midi")
    p_verbose.add_argument("--model",      default=None,
        help="Modelo entrenado (.pkl). Alternativa a --params")
    p_verbose.add_argument("--params",     default=None,
        help='Parámetros explícitos, ej: "tempo_factor=1.8,pitch_shift=3"')
    p_verbose.add_argument("--intensity",  type=float, default=1.0)
    p_verbose.add_argument("--output",     default=None,
        help="MIDI de salida [default: <entrada>_verbose_out.mid]")
    p_verbose.add_argument("--max-events", type=int, default=30,
        help="Máximo de eventos a mostrar por categoría [default: 30]")
    p_verbose.add_argument("--log-json",   default=None,
        help="Guardar el log completo en un fichero JSON")

    # ── apply ─────────────────────────────────────────────────────────────
    p_apply = sub.add_parser("apply",
        help="Aplicar parámetros de transformación concretos (sin modelo)")
    p_apply.add_argument("input_midi")
    p_apply.add_argument("--params", required=True,
        help='Parámetros a aplicar, ej: "tempo_factor=1.8, pitch_shift=3"')
    p_apply.add_argument("--output", default=None,
        help="MIDI de salida [default: <entrada>_applied.mid]")

    # ── block ─────────────────────────────────────────────────────────────
    p_block = sub.add_parser("block",
        help="Como transform pero omitiendo los operadores indicados")
    p_block.add_argument("input_midi")
    p_block.add_argument("--model",     required=True)
    p_block.add_argument("--block",     required=True,
        help='Operadores a bloquear (neutro), ej: "tempo_factor, register_remap_low"')
    p_block.add_argument("--output",    default=None,
        help="MIDI de salida [default: <entrada>_blocked.mid]")
    p_block.add_argument("--intensity", type=float, default=1.0)

    args = parser.parse_args()

    if   args.mode == "train":     mode_train(args)
    elif args.mode == "transform": mode_transform(args)
    elif args.mode == "verbose":   mode_verbose(args)
    elif args.mode == "apply":     mode_apply(args)
    elif args.mode == "block":     mode_block(args)


if __name__ == "__main__":
    main()

