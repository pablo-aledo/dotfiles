#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ANALYZER  v1.1                                       ║
║         Análisis comparativo y de trayectoria entre MIDIs                   ║
║                                                                              ║
║  Responde preguntas del tipo:                                                ║
║    • ¿En qué dimensiones se parecen A y B?                                  ║
║    • ¿Qué tiene A que no tenga B?                                           ║
║    • ¿Cuáles son los 5 MIDIs de mi colección más cercanos a este?           ║
║    • ¿Cómo se agrupan estos MIDIs entre sí?                                 ║
║    • ¿Cómo evoluciona esta obra a lo largo de sus secciones?  [NUEVO]       ║
║    • ¿Cuál es el camino vectorial de A a B?                   [NUEVO]       ║
║                                                                              ║
║  MODOS:                                                                      ║
║    compare    — Compara dos MIDIs dimensión a dimensión                      ║
║    nearest    — Busca los N más cercanos a un MIDI en una colección         ║
║    diff       — Qué tiene A que no tenga B (perfil de diferencias)          ║
║    matrix     — Matriz de similitud entre todos los MIDIs dados             ║
║    profile    — Perfil vectorial de un MIDI individual                      ║
║    trajectory — Evolución interna de un MIDI en ventanas temporales [NUEVO] ║
║    path       — Camino vectorial entre dos MIDIs (o una secuencia)  [NUEVO] ║
║                                                                              ║
║  NIVELES DE VECTORIZACIÓN:                                                   ║
║    fast    — Sólo mido: notas, velocidades, densidad (sin dependencias)     ║
║    full    — mscz2vec completo: 17 dimensiones musicales (requiere music21) ║
║    auto    — full si music21 disponible, fast si no (default)               ║
║                                                                              ║
║  USO:                                                                        ║
║    python analyzer.py compare A.mid B.mid                                   ║
║    python analyzer.py compare A.mid B.mid --dimensions melodic harmonic     ║
║    python analyzer.py nearest query.mid coleccion/*.mid                     ║
║    python analyzer.py nearest query.mid coleccion/*.mid --top 5             ║
║    python analyzer.py diff A.mid B.mid                                      ║
║    python analyzer.py diff A.mid B.mid --threshold 0.2                      ║
║    python analyzer.py matrix A.mid B.mid C.mid D.mid                        ║
║    python analyzer.py matrix coleccion/*.mid --plot                         ║
║    python analyzer.py profile obra.mid                                       ║
║    python analyzer.py nearest query.mid *.mid --use-cache cache.json       ║
║    python analyzer.py trajectory obra.mid                                    ║
║    python analyzer.py trajectory obra.mid --windows 8 --plot                ║
║    python analyzer.py trajectory obra.mid --highlight pitch_mean density    ║
║    python analyzer.py path A.mid B.mid                                      ║
║    python analyzer.py path A.mid B.mid --steps 5 --plot                     ║
║    python analyzer.py path A.mid B.mid C.mid D.mid                          ║
║                                                                              ║
║  OPCIONES GLOBALES:                                                          ║
║    --level      fast | full | auto   (default: auto)                        ║
║    --dimensions Dimensiones a analizar (default: según nivel)               ║
║    --output     Guardar resultado en JSON                                    ║
║    --plot       Visualización matplotlib (heatmap / radar / scatter)        ║
║    --no-color   Sin colores ANSI en la salida                               ║
║    --verbose    Informe detallado                                            ║
║                                                                              ║
║  OPCIONES POR MODO:                                                          ║
║    compare:    --no-radar                                                    ║
║    nearest:    --top N (default: 5) · --metric euclidean|cosine|manhattan   ║
║    diff:       --threshold F (default: 0.15) · --show-similar               ║
║    matrix:     --sort · --plot                                               ║
║    trajectory: --windows N (default: 8) · --highlight dim1 dim2...         ║
║                --boundary silence|beat|equal (default: equal)               ║
║    path:       --steps N (default: 4) · --collection *.mid                  ║
║                (si se pasa colección, muestra los vecinos reales de la ruta)║
║                                                                              ║
║  CACHÉ:                                                                      ║
║    --save-cache FILE   Guardar vectores calculados en JSON                  ║
║    --use-cache FILE    Cargar vectores previos (evita recalcular)           ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:   mido, numpy                                                    ║
║    --level full: mscz2vec.py en el mismo directorio + music21               ║
║    --plot:   matplotlib                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import textwrap
import math
import copy
import time
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.1"

# Dimensiones disponibles en nivel "fast" (sólo mido)
FAST_DIMENSIONS = [
    "pitch_mean", "pitch_std", "pitch_range",
    "velocity_mean", "velocity_std",
    "density", "interval_large_ratio",
    "ascent_ratio", "n_tracks", "n_instruments",
]

# Dimensiones disponibles en nivel "full" (mscz2vec)
FULL_DIMENSIONS = [
    "melodic", "harmonic", "rhythmic", "instrumental",
    "emotion", "tension", "stability",
    "novelty", "entropy", "lerdahl", "tonnetz",
    "psychoacoustic", "modal", "brightness", "unified",
]

# Descripción legible de cada dimensión
DIM_LABELS = {
    # fast
    "pitch_mean":           "Altura media",
    "pitch_std":            "Variedad de alturas",
    "pitch_range":          "Rango de alturas",
    "velocity_mean":        "Dinámica media",
    "velocity_std":         "Expresividad dinámica",
    "density":              "Densidad rítmica",
    "interval_large_ratio": "Saltos melódicos",
    "ascent_ratio":         "Prop. melódica ascendente",
    "n_tracks":             "Número de pistas",
    "n_instruments":        "Número de instrumentos",
    # full (mscz2vec - vector unified 6D)
    "melodic":      "Perfil melódico",
    "harmonic":     "Progresión armónica",
    "rhythmic":     "Patrón rítmico",
    "instrumental": "Orquestación",
    "emotion":      "Vector emocional",
    "tension":      "Tensión armónica",
    "stability":    "Estabilidad tonal",
    "novelty":      "Novedad estructural",
    "entropy":      "Entropía / sorpresa",
    "lerdahl":      "Energía (Lerdahl)",
    "tonnetz":      "Distancias Tonnetz",
    "psychoacoustic": "Psicoaacústica",
    "modal":        "Intercambio modal",
    "brightness":   "Luminosidad tímbrica",
    "unified":      "Vector unificado 6D",
}

# Colores ANSI
C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "blue":   "\033[94m",
    "gray":   "\033[90m",
    "white":  "\033[97m",
}
USE_COLOR = True


def c(key: str) -> str:
    return C[key] if USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def vectorize_fast(midi_path: str) -> dict | None:
    """
    Vectorización rápida usando sólo mido.
    Retorna dict {dimension: float} o None si falla.
    """
    try:
        import mido
    except ImportError:
        print("ERROR: pip install mido")
        return None

    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  [warn] No se pudo leer {Path(midi_path).name}: {e}")
        return None

    notes, velocities = [], []
    instruments = set()
    n_tracks = len(mid.tracks)

    for track in mid.tracks:
        current_program = 0
        for msg in track:
            if msg.type == "program_change":
                current_program = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
                instruments.add(current_program)

    if not notes:
        return None

    notes = np.array(notes, dtype=float)
    velocities = np.array(velocities, dtype=float)
    diffs = np.diff(notes)

    density = len(notes) / max(1.0, sum(
        msg.time for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0 and msg.time > 0
    ) or 1.0)
    density = min(1.0, density * 10)  # normalizar aprox a 0-1

    return {
        "pitch_mean":           float(np.mean(notes) / 127),
        "pitch_std":            float(np.std(notes) / 64),
        "pitch_range":          float((np.max(notes) - np.min(notes)) / 127),
        "velocity_mean":        float(np.mean(velocities) / 127),
        "velocity_std":         float(np.std(velocities) / 64),
        "density":              float(np.clip(density, 0, 1)),
        "interval_large_ratio": float(np.mean(np.abs(diffs) > 5)) if len(diffs) > 0 else 0.0,
        "ascent_ratio":         float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.5,
        "n_tracks":             float(min(n_tracks / 16, 1.0)),
        "n_instruments":        float(min(len(instruments) / 16, 1.0)),
    }


def vectorize_full(midi_path: str, dimensions: list[str],
                   verbose: bool = False) -> dict | None:
    """
    Vectorización completa usando mscz2vec.py.
    Llama al script como subproceso y parsea el JSON de salida.
    """
    script = _find_script("mscz2vec.py")
    if not script:
        if verbose:
            print("  [warn] mscz2vec.py no encontrado, usando fast")
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    # Sólo pedir los vectores que son arrays numéricos directos
    # (tension y stability devuelven dicts con perfiles; los tratamos aparte)
    array_dims  = [d for d in dimensions if d not in ("tension", "stability")]
    profile_dims = [d for d in dimensions if d in ("tension", "stability")]

    results = {}

    if array_dims:
        cmd = [
            sys.executable, script,
            midi_path,
            "--vector", *array_dims,
            "--output", out_path,
            "--no-debug",
        ]
        try:
            subprocess.run(cmd, capture_output=not verbose, text=True, timeout=300)
            if Path(out_path).exists():
                with open(out_path) as f:
                    data = json.load(f)
                for dim, val in data.get("vectors", {}).items():
                    if val is not None:
                        results[dim] = val
        except Exception as e:
            if verbose:
                print(f"  [warn] mscz2vec error en {Path(midi_path).name}: {e}")
        finally:
            Path(out_path).unlink(missing_ok=True)

    if profile_dims:
        cmd2 = [
            sys.executable, script,
            midi_path,
            "--vector", *profile_dims,
            "--output", out_path,
            "--no-debug",
        ]
        try:
            subprocess.run(cmd2, capture_output=not verbose, text=True, timeout=300)
            if Path(out_path).exists():
                with open(out_path) as f:
                    data = json.load(f)
                for dim, val in data.get("vectors", {}).items():
                    if val is not None:
                        # tension/stability devuelven dicts; extraer valor escalar
                        if isinstance(val, dict):
                            # tension: {'tension_values': [...], ...}
                            # stability: {'values': [...], 'statistics': {...}}
                            tv = val.get("tension_values") or val.get("values")
                            if tv:
                                results[dim] = float(np.mean(tv))
                            else:
                                stats = val.get("statistics", {})
                                results[dim] = float(stats.get("mean", 0.5))
                        else:
                            results[dim] = val
        except Exception as e:
            if verbose:
                print(f"  [warn] mscz2vec (profiles) error: {e}")
        finally:
            Path(out_path).unlink(missing_ok=True)

    return results if results else None


def vectorize(midi_path: str, level: str, dimensions: list[str] | None,
              verbose: bool = False) -> dict | None:
    """
    Punto de entrada unificado de vectorización.
    Retorna {dim: valor_o_array} o None.
    """
    if level == "fast":
        vec = vectorize_fast(midi_path)
        if vec and dimensions:
            vec = {k: v for k, v in vec.items() if k in dimensions}
        return vec

    if level == "full":
        dims = dimensions or FULL_DIMENSIONS
        vec = vectorize_full(midi_path, dims, verbose=verbose)
        return vec

    # auto: intentar full, caer a fast
    dims = dimensions or FULL_DIMENSIONS
    vec = vectorize_full(midi_path, dims, verbose=verbose)
    if vec:
        return vec
    if verbose:
        print(f"  [auto] mscz2vec no disponible, usando vectorización fast")
    vec = vectorize_fast(midi_path)
    if vec and dimensions:
        vec = {k: v for k, v in vec.items() if k in dimensions}
    return vec


# ══════════════════════════════════════════════════════════════════════════════
#  CACHÉ DE VECTORES
# ══════════════════════════════════════════════════════════════════════════════

class VectorCache:
    """
    Caché de vectores en disco. Clave: ruta absoluta del MIDI.
    Invalida automáticamente entradas cuyo MIDI fue modificado.
    """
    def __init__(self, path: str | None = None):
        self.path = path
        self._data: dict = {}
        if path and Path(path).exists():
            try:
                with open(path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def get(self, midi_path: str) -> dict | None:
        key = str(Path(midi_path).resolve())
        entry = self._data.get(key)
        if not entry:
            return None
        # Validar que el MIDI no cambió desde que se cacheó
        try:
            mtime = Path(midi_path).stat().st_mtime
            if abs(mtime - entry.get("mtime", 0)) > 1.0:
                return None
        except Exception:
            return None
        return entry.get("vectors")

    def set(self, midi_path: str, vectors: dict):
        key = str(Path(midi_path).resolve())
        try:
            mtime = Path(midi_path).stat().st_mtime
        except Exception:
            mtime = 0.0
        self._data[key] = {"mtime": mtime, "vectors": vectors}

    def save(self):
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)

    def __len__(self):
        return len(self._data)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE VECTORES A ESCALARES COMPARABLES
# ══════════════════════════════════════════════════════════════════════════════

def flatten_vector(vec_value) -> float:
    """
    Convierte cualquier valor de vector (float, list, array) a un escalar
    representativo [0, 1] para comparación dimensional.
    """
    if isinstance(vec_value, (int, float)):
        return float(vec_value)
    arr = np.asarray(vec_value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    # Usar la norma L2 normalizada como escalar representativo
    norm = np.linalg.norm(arr)
    n = len(arr.flatten())
    return float(norm / math.sqrt(n)) if n > 0 else 0.0


def to_comparable_vector(vec_dict: dict) -> tuple[list[str], np.ndarray]:
    """
    Convierte un dict de vectores a (lista_de_dims, array_numpy).
    Expande vectores multi-dimensionales: 'unified' → 'unified_0'..'unified_5'.
    """
    dims, vals = [], []
    for k in sorted(vec_dict.keys()):
        v = vec_dict[k]
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v, dtype=float).flatten()
            if len(arr) <= 20:  # expandir si es razonablemente corto
                for i, x in enumerate(arr):
                    dims.append(f"{k}_{i}")
                    vals.append(float(x))
            else:
                # Para vectores largos, usar estadísticas resumidas
                dims.extend([f"{k}_mean", f"{k}_std", f"{k}_p25", f"{k}_p75"])
                vals.extend([
                    float(np.mean(arr)),
                    float(np.std(arr)),
                    float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 75)),
                ])
        else:
            dims.append(k)
            vals.append(float(v) if v is not None else 0.0)
    return dims, np.array(vals, dtype=float)


def per_dim_scalars(vec_dict: dict) -> dict[str, float]:
    """
    Convierte cada entrada del vector a un único escalar representativo.
    Usado para comparación dimensión a dimensión en compare y diff.
    """
    result = {}
    for k, v in vec_dict.items():
        result[k] = flatten_vector(v)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE SIMILITUD
# ══════════════════════════════════════════════════════════════════════════════

def _safe_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trunca o rellena con ceros para igualar longitudes."""
    if len(a) == len(b):
        return a, b
    n = min(len(a), len(b))
    return a[:n], b[:n]


def distance_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _safe_align(a, b)
    return float(np.linalg.norm(a - b))


def distance_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _safe_align(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def distance_manhattan(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _safe_align(a, b)
    return float(np.sum(np.abs(a - b)))


def similarity(a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> float:
    """Retorna similitud 0-1 (1 = idénticos)."""
    fns = {
        "euclidean": distance_euclidean,
        "cosine":    distance_cosine,
        "manhattan": distance_manhattan,
    }
    dist = fns.get(metric, distance_euclidean)(a, b)
    # Convertir distancia a similitud con decaimiento exponencial
    return float(math.exp(-dist))


def similarity_per_dim(va: dict, vb: dict) -> dict[str, float]:
    """
    Calcula similitud escalar por dimensión entre dos vectores.
    Retorna {dim: similitud_0_a_1}.
    """
    result = {}
    all_dims = sorted(set(va) | set(vb))
    for dim in all_dims:
        a_val = va.get(dim)
        b_val = vb.get(dim)
        if a_val is None or b_val is None:
            continue
        a_arr = np.asarray(flatten_vector(a_val)).flatten()
        b_arr = np.asarray(flatten_vector(b_val)).flatten()
        diff = abs(float(a_arr[0]) - float(b_arr[0]))
        result[dim] = float(max(0.0, 1.0 - diff))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA Y CACHE DE MÚLTIPLES MIDIS
# ══════════════════════════════════════════════════════════════════════════════

def load_vectors_batch(midi_paths: list[str], level: str,
                       dimensions: list[str] | None,
                       cache: VectorCache,
                       verbose: bool = False) -> dict[str, dict]:
    """
    Vectoriza una lista de MIDIs. Usa caché cuando está disponible.
    Retorna {ruta: vector_dict}.
    """
    results = {}
    total = len(midi_paths)

    for i, path in enumerate(midi_paths, 1):
        cached = cache.get(path)
        if cached is not None:
            results[path] = cached
            if verbose:
                print(f"  [{i}/{total}] {Path(path).name}  (caché)")
            continue

        if verbose:
            print(f"  [{i}/{total}] Vectorizando {Path(path).name}...")
        else:
            print(f"  Vectorizando {i}/{total}: {Path(path).name}", end="\r")

        vec = vectorize(path, level, dimensions, verbose=verbose)
        if vec:
            results[path] = vec
            cache.set(path, vec)
        else:
            print(f"\n  [warn] No se pudo vectorizar: {Path(path).name}")

    if not verbose:
        print()  # salto de línea después del \r

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def mode_profile(midi_path: str, args) -> dict:
    """Perfil vectorial completo de un MIDI individual."""
    cache = VectorCache(args.use_cache)
    print(f"\n{c('bold')}  Perfil: {Path(midi_path).name}{c('reset')}")
    print(f"  Nivel: {args.level}")

    vec = (cache.get(midi_path) or
           vectorize(midi_path, args.level, args.dimensions, verbose=args.verbose))
    if not vec:
        print("  ERROR: No se pudo vectorizar el MIDI.")
        return {}

    cache.set(midi_path, vec)
    if args.save_cache:
        cache.save()

    scalars = per_dim_scalars(vec)

    print(f"\n{'─'*58}")
    print(f"  {'Dimensión':<28} {'Valor':>8}  {'Barra'}")
    print(f"{'─'*58}")

    for dim, val in sorted(scalars.items()):
        label = DIM_LABELS.get(dim, dim)[:27]
        bar_len = int(val * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        color = c("green") if val > 0.65 else c("yellow") if val > 0.35 else c("blue")
        print(f"  {label:<28} {val:>7.3f}  {color}{bar}{c('reset')}")

    print(f"{'─'*58}")

    result = {"source": midi_path, "vectors": vec}
    _maybe_save(result, args)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def mode_compare(path_a: str, path_b: str, args) -> dict:
    """Comparación dimensión a dimensión entre dos MIDIs."""
    cache = VectorCache(args.use_cache)

    name_a = Path(path_a).name
    name_b = Path(path_b).name

    print(f"\n{c('bold')}  Comparando:{c('reset')}")
    print(f"    A: {name_a}")
    print(f"    B: {name_b}")
    print(f"  Nivel: {args.level}\n")

    vec_a = cache.get(path_a) or vectorize(path_a, args.level, args.dimensions, args.verbose)
    vec_b = cache.get(path_b) or vectorize(path_b, args.level, args.dimensions, args.verbose)

    if not vec_a or not vec_b:
        print("  ERROR: No se pudieron vectorizar ambos MIDIs.")
        return {}

    cache.set(path_a, vec_a)
    cache.set(path_b, vec_b)
    if args.save_cache:
        cache.save()

    sc_a = per_dim_scalars(vec_a)
    sc_b = per_dim_scalars(vec_b)

    common_dims = sorted(set(sc_a) & set(sc_b))

    # Similitud global
    _, arr_a = to_comparable_vector(vec_a)
    _, arr_b = to_comparable_vector(vec_b)
    global_sim = similarity(arr_a, arr_b, metric=args.metric)

    # Tabla comparativa
    col_w = min(20, max(len(name_a), len(name_b), 10))
    print(f"{'─'*68}")
    print(f"  {'Dimensión':<26}  {name_a[:col_w]:>{col_w}}  {name_b[:col_w]:>{col_w}}  {'Sim':>5}  {'Δ':>6}")
    print(f"{'─'*68}")

    dim_sims = {}
    for dim in common_dims:
        a_v = sc_a[dim]
        b_v = sc_b[dim]
        diff = b_v - a_v
        sim = max(0.0, 1.0 - abs(diff))
        dim_sims[dim] = sim
        label = DIM_LABELS.get(dim, dim)[:25]

        if sim >= 0.85:
            sim_color = c("green")
        elif sim >= 0.6:
            sim_color = c("yellow")
        else:
            sim_color = c("red")

        diff_str = f"{diff:+.3f}"
        print(f"  {label:<26}  {a_v:>{col_w}.3f}  {b_v:>{col_w}.3f}  "
              f"{sim_color}{sim:>5.2f}{c('reset')}  {diff_str:>6}")

    print(f"{'─'*68}")
    print(f"\n  {c('bold')}Similitud global ({args.metric}): "
          f"{c('green') if global_sim > 0.7 else c('yellow') if global_sim > 0.4 else c('red')}"
          f"{global_sim:.4f}{c('reset')}")

    # Puntos de mayor similitud y mayor diferencia
    if dim_sims:
        top_similar = sorted(dim_sims.items(), key=lambda x: -x[1])[:3]
        top_different = sorted(dim_sims.items(), key=lambda x: x[1])[:3]

        print(f"\n  {c('green')}Más parecidos:{c('reset')}")
        for dim, s in top_similar:
            print(f"    • {DIM_LABELS.get(dim, dim):<28}  sim={s:.3f}")

        print(f"\n  {c('red')}Más diferentes:{c('reset')}")
        for dim, s in top_different:
            label = DIM_LABELS.get(dim, dim)
            a_v = sc_a.get(dim, 0)
            b_v = sc_b.get(dim, 0)
            direction = f"A={a_v:.3f}, B={b_v:.3f}"
            print(f"    • {label:<28}  sim={s:.3f}  ({direction})")

    result = {
        "mode": "compare",
        "a": path_a, "b": path_b,
        "global_similarity": global_sim,
        "dimension_similarities": dim_sims,
        "vectors": {"a": vec_a, "b": vec_b},
    }

    if args.plot:
        _plot_compare_radar(sc_a, sc_b, name_a, name_b, common_dims)

    _maybe_save(result, args)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: DIFF
# ══════════════════════════════════════════════════════════════════════════════

def mode_diff(path_a: str, path_b: str, args) -> dict:
    """
    Qué tiene A que no tenga B: diferencias significativas
    con descripción semántica de la dirección del cambio.
    """
    cache = VectorCache(args.use_cache)
    name_a = Path(path_a).name
    name_b = Path(path_b).name
    threshold = args.threshold

    print(f"\n{c('bold')}  Diff: {name_a}  →  {name_b}{c('reset')}")
    print(f"  Umbral de diferencia significativa: {threshold:.2f}\n")

    vec_a = cache.get(path_a) or vectorize(path_a, args.level, args.dimensions, args.verbose)
    vec_b = cache.get(path_b) or vectorize(path_b, args.level, args.dimensions, args.verbose)

    if not vec_a or not vec_b:
        print("  ERROR: No se pudieron vectorizar los MIDIs.")
        return {}

    cache.set(path_a, vec_a)
    cache.set(path_b, vec_b)
    if args.save_cache:
        cache.save()

    sc_a = per_dim_scalars(vec_a)
    sc_b = per_dim_scalars(vec_b)
    common_dims = sorted(set(sc_a) & set(sc_b))

    diffs_sig = []    # diferencias significativas
    diffs_minor = []  # diferencias menores

    for dim in common_dims:
        a_v = sc_a[dim]
        b_v = sc_b[dim]
        delta = b_v - a_v
        if abs(delta) >= threshold:
            diffs_sig.append((dim, a_v, b_v, delta))
        else:
            diffs_minor.append((dim, a_v, b_v, delta))

    diffs_sig.sort(key=lambda x: -abs(x[3]))

    # A tiene más que B
    a_more = [(d, av, bv, delta) for d, av, bv, delta in diffs_sig if delta < 0]
    # B tiene más que A
    b_more = [(d, av, bv, delta) for d, av, bv, delta in diffs_sig if delta > 0]

    if not diffs_sig:
        print(f"  {c('green')}No hay diferencias significativas "
              f"(umbral={threshold}).{c('reset')}")
        print(f"  Los MIDIs son muy similares en todas las dimensiones analizadas.")
    else:
        if a_more:
            print(f"  {c('cyan')}{c('bold')}Lo que A tiene más que B:{c('reset')}")
            for dim, av, bv, delta in a_more:
                label = DIM_LABELS.get(dim, dim)
                bar = _delta_bar(abs(delta))
                print(f"    {c('cyan')}↑{c('reset')} {label:<28}  "
                      f"A={av:.3f}  B={bv:.3f}  Δ={delta:+.3f}  {bar}")

        if b_more:
            print(f"\n  {c('yellow')}{c('bold')}Lo que B tiene más que A:{c('reset')}")
            for dim, av, bv, delta in b_more:
                label = DIM_LABELS.get(dim, dim)
                bar = _delta_bar(abs(delta))
                print(f"    {c('yellow')}↑{c('reset')} {label:<28}  "
                      f"A={av:.3f}  B={bv:.3f}  Δ={delta:+.3f}  {bar}")

    if args.show_similar and diffs_minor:
        print(f"\n  {c('gray')}Dimensiones similares (|Δ| < {threshold}):{c('reset')}")
        for dim, av, bv, delta in sorted(diffs_minor, key=lambda x: abs(x[3])):
            label = DIM_LABELS.get(dim, dim)
            print(f"    {c('gray')}≈{c('reset')} {label:<28}  "
                  f"A={av:.3f}  B={bv:.3f}  Δ={delta:+.3f}")

    # Resumen semántico
    print(f"\n  {c('bold')}Resumen:{c('reset')}")
    n_sig = len(diffs_sig)
    n_total = len(common_dims)
    pct = 100 * n_sig / n_total if n_total > 0 else 0
    print(f"  {n_sig} de {n_total} dimensiones difieren significativamente ({pct:.0f}%)")

    result = {
        "mode": "diff",
        "a": path_a, "b": path_b,
        "threshold": threshold,
        "significant_differences": [
            {"dim": d, "a": av, "b": bv, "delta": delta}
            for d, av, bv, delta in diffs_sig
        ],
        "vectors": {"a": vec_a, "b": vec_b},
    }

    _maybe_save(result, args)
    return result


def _plot_trajectory(dim_series: dict, windows: list, show_dims: list, title: str):
    """
    Gráfico matplotlib de la trayectoria: una línea por dimensión a lo largo
    de las ventanas temporales.
    Cada subplot tiene escala propia + margen para anotaciones.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_dims = len(show_dims)
    if n_dims == 0:
        return

    pcts = [w["pct_start"] for w in windows]
    row_h = 2.2  # altura por subplot
    fig, axes = plt.subplots(
        n_dims, 1,
        figsize=(13, max(4, n_dims * row_h)),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if n_dims == 1:
        axes = [axes]

    colors = cm.tab10(np.linspace(0, 1, max(n_dims, 2)))

    for ax, dim, col in zip(axes, show_dims, colors):
        series = np.array(dim_series[dim], dtype=float)

        ax.plot(pcts, series, "o-", color=col, linewidth=2,
                markersize=5, solid_capstyle="round")
        ax.fill_between(pcts, series, alpha=0.12, color=col)

        # ── Escala adaptativa por subplot ──────────────────────────────────
        mn, mx = series.min(), series.max()
        rng = mx - mn if mx != mn else 0.1
        pad = rng * 0.35           # margen extra para las anotaciones de valor
        ax.set_ylim(mn - pad * 0.5, mx + pad)

        # Etiquetas de valor en cada punto
        for x, y in zip(pcts, series):
            ax.annotate(
                f"{y:.3f}", (x, y),
                textcoords="offset points", xytext=(0, 7),
                fontsize=6, ha="center", color=col, alpha=0.85,
            )

        # Línea de media punteada
        ax.axhline(series.mean(), linestyle=":", linewidth=0.8,
                   color=col, alpha=0.5)

        # Etiqueta del eje Y
        label = DIM_LABELS.get(dim, dim)
        ax.set_ylabel(label, fontsize=8, labelpad=4)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.tick_params(axis="y", length=3)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.grid(axis="y", linestyle=":", alpha=0.2)

        # Rango mostrado en margen derecho para referencia rápida
        ax.annotate(
            f"[{mn:.2f}–{mx:.2f}]",
            xy=(1.0, 0.92), xycoords="axes fraction",
            fontsize=6, ha="right", va="top", color="gray",
        )

    axes[-1].set_xlabel("Posición en la obra (%)", fontsize=9)
    axes[-1].xaxis.set_tick_params(labelsize=8)

    fig.suptitle(
        f"Trayectoria musical — {title}",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def _plot_path(vecs: dict, midi_paths: list, coll_vec_map: dict,
               names: list, metric: str):
    """
    Visualización 2D (PCA) del camino vectorial entre los MIDIs waypoint,
    con la colección de fondo si está disponible.
    Márgenes automáticos para que las etiquetas no se salgan del canvas.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    all_vecs   = [vecs[p] for p in midi_paths] + list(coll_vec_map.values())
    all_arrays = [to_comparable_vector(v)[1] for v in all_vecs]

    # Rellenar a la misma longitud
    max_len = max(len(a) for a in all_arrays)
    padded  = np.array([np.pad(a, (0, max_len - len(a))) for a in all_arrays],
                       dtype=float)

    # PCA 2D manual
    mat = padded - padded.mean(axis=0)
    cov = np.cov(mat.T)
    if cov.ndim < 2:
        cov = np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(-eigvals)
    emb = mat @ eigvecs[:, idx[:2]]

    n_wp   = len(midi_paths)
    wp_emb = emb[:n_wp]

    fig, ax = plt.subplots(figsize=(11, 8))

    # Colección de fondo
    if coll_vec_map:
        ax.scatter(emb[n_wp:, 0], emb[n_wp:, 1],
                   c="#cccccc", s=20, alpha=0.45, zorder=1, label="Colección")

    # Flechas del camino
    for i in range(n_wp - 1):
        ax.annotate(
            "", xy=wp_emb[i + 1], xytext=wp_emb[i],
            arrowprops=dict(arrowstyle="-|>", color="#E84040",
                            lw=2, mutation_scale=16),
            zorder=4,
        )

    # Puntos waypoint
    wp_colors = (["#E84040"] + ["#4C72B0"] * max(0, n_wp - 2) + ["#2ca02c"])[:n_wp]
    ax.scatter(wp_emb[:, 0], wp_emb[:, 1],
               c=wp_colors, s=200, zorder=5,
               edgecolors="white", linewidths=1.8)

    # Etiquetas con offset adaptativo por cuadrante
    cx, cy = wp_emb[:, 0].mean(), wp_emb[:, 1].mean()
    for i, (pt, name) in enumerate(zip(wp_emb, names[:n_wp])):
        role = "Inicio" if i == 0 else ("Fin" if i == n_wp - 1 else f"·{i}")
        text = f"{role}\n{name[:22]}"
        dx = 10 if pt[0] >= cx else -10
        dy = 10 if pt[1] >= cy else -10
        ha = "left" if dx > 0 else "right"
        va = "bottom" if dy > 0 else "top"
        ax.annotate(
            text, pt,
            textcoords="offset points", xytext=(dx, dy),
            fontsize=8, fontweight="bold", ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.75, lw=0.6),
        )

    # Márgenes con padding proporcional al rango de datos
    all_x, all_y = emb[:, 0], emb[:, 1]
    x_rng = all_x.ptp() or 1.0
    y_rng = all_y.ptp() or 1.0
    ax.set_xlim(all_x.min() - x_rng * 0.20, all_x.max() + x_rng * 0.20)
    ax.set_ylim(all_y.min() - y_rng * 0.20, all_y.max() + y_rng * 0.20)

    ax.set_title(f"Camino vectorial ({metric}) — PCA 2D", fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(linestyle=":", alpha=0.3)
    if coll_vec_map:
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def _delta_bar(delta: float, width: int = 20) -> str:
    n = int(min(delta, 1.0) * width)
    return "▓" * n + "░" * (width - n)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: NEAREST
# ══════════════════════════════════════════════════════════════════════════════

def mode_nearest(query_path: str, collection_paths: list[str], args) -> dict:
    """
    Busca los N MIDIs más cercanos al query en la colección.
    """
    cache = VectorCache(args.use_cache)
    top_n = args.top
    metric = args.metric

    query_name = Path(query_path).name
    print(f"\n{c('bold')}  Búsqueda de similitud{c('reset')}")
    print(f"  Consulta:   {query_name}")
    print(f"  Colección:  {len(collection_paths)} MIDIs")
    print(f"  Métrica:    {metric}  |  Top: {top_n}\n")

    # Vectorizar query
    vec_query = cache.get(query_path) or vectorize(
        query_path, args.level, args.dimensions, args.verbose)
    if not vec_query:
        print("  ERROR: No se pudo vectorizar el MIDI de consulta.")
        return {}
    cache.set(query_path, vec_query)
    _, arr_query = to_comparable_vector(vec_query)

    # Vectorizar colección (excluir el query si aparece)
    candidates = [p for p in collection_paths if Path(p).resolve() != Path(query_path).resolve()]
    print(f"  Vectorizando colección ({len(candidates)} MIDIs)...")
    vec_collection = load_vectors_batch(candidates, args.level, args.dimensions,
                                        cache, verbose=args.verbose)

    if args.save_cache:
        cache.save()

    # Calcular similitudes
    sims = []
    for path, vec in vec_collection.items():
        _, arr = to_comparable_vector(vec)
        sim = similarity(arr_query, arr, metric=metric)
        sims.append((path, sim, vec))

    sims.sort(key=lambda x: -x[1])
    top_results = sims[:top_n]

    # Tabla de resultados
    print(f"\n{'─'*64}")
    print(f"  {'#':>3}  {'Similitud':>10}  {'Barra':<24}  MIDI")
    print(f"{'─'*64}")

    for rank, (path, sim, _) in enumerate(top_results, 1):
        bar_len = int(sim * 24)
        bar = "█" * bar_len + "░" * (24 - bar_len)
        name = Path(path).name[:35]
        color = c("green") if sim > 0.7 else c("yellow") if sim > 0.4 else c("red")
        print(f"  {rank:>3}  {color}{sim:>10.4f}{c('reset')}  {color}{bar}{c('reset')}  {name}")

    print(f"{'─'*64}")

    # Análisis dimensional del top-1
    if top_results:
        best_path, best_sim, best_vec = top_results[0]
        best_name = Path(best_path).name
        print(f"\n  {c('bold')}Similitud dimensional con el más cercano "
              f"({best_name}):{c('reset')}")
        dim_sims = similarity_per_dim(per_dim_scalars(vec_query),
                                      per_dim_scalars(best_vec))
        top_d = sorted(dim_sims.items(), key=lambda x: -x[1])[:5]
        bot_d = sorted(dim_sims.items(), key=lambda x: x[1])[:3]
        print(f"  {c('green')}Más similares:{c('reset')}")
        for dim, s in top_d:
            print(f"    • {DIM_LABELS.get(dim, dim):<28}  {s:.3f}")
        print(f"  {c('red')}Más diferentes:{c('reset')}")
        for dim, s in bot_d:
            print(f"    • {DIM_LABELS.get(dim, dim):<28}  {s:.3f}")

    result = {
        "mode": "nearest",
        "query": query_path,
        "metric": metric,
        "top_n": top_n,
        "results": [
            {"rank": i + 1, "path": p, "similarity": sim}
            for i, (p, sim, _) in enumerate(top_results)
        ],
        "query_vector": vec_query,
    }

    if args.plot and top_results:
        _plot_nearest_scatter(arr_query, vec_collection, top_results,
                               query_name, metric)

    _maybe_save(result, args)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def mode_matrix(midi_paths: list[str], args) -> dict:
    """
    Matriz de similitud N×N entre todos los MIDIs dados.
    """
    cache = VectorCache(args.use_cache)
    metric = args.metric

    print(f"\n{c('bold')}  Matriz de similitud{c('reset')}")
    print(f"  {len(midi_paths)} MIDIs  |  Métrica: {metric}\n")

    print(f"  Vectorizando {len(midi_paths)} MIDIs...")
    vec_map = load_vectors_batch(midi_paths, args.level, args.dimensions,
                                 cache, verbose=args.verbose)

    if args.save_cache:
        cache.save()

    paths = list(vec_map.keys())
    n = len(paths)
    if n < 2:
        print("  ERROR: se necesitan al menos 2 MIDIs con vectores válidos.")
        return {}

    names = [Path(p).stem[:14] for p in paths]

    # Construir matriz
    arr_map = {p: to_comparable_vector(v)[1] for p, v in vec_map.items()}
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                sim = similarity(arr_map[paths[i]], arr_map[paths[j]], metric)
                mat[i, j] = mat[j, i] = sim

    # Ordenar por similitud media si se pide
    if args.sort:
        order = np.argsort(-mat.mean(axis=1))
        paths = [paths[i] for i in order]
        names = [names[i] for i in order]
        mat = mat[np.ix_(order, order)]

    # Imprimir matriz
    name_w = max(len(n) for n in names) + 1
    header = " " * (name_w + 2) + "  ".join(f"{n[:6]:>6}" for n in names)
    print(f"{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for i, row_name in enumerate(names):
        row_parts = []
        for j in range(n):
            v = mat[i, j]
            if i == j:
                cell = f"{c('gray')}  1.00{c('reset')}"
            elif v > 0.75:
                cell = f"{c('green')}{v:6.2f}{c('reset')}"
            elif v > 0.5:
                cell = f"{c('yellow')}{v:6.2f}{c('reset')}"
            else:
                cell = f"{c('red')}{v:6.2f}{c('reset')}"
            row_parts.append(cell)
        print(f"  {row_name:<{name_w}}  {'  '.join(row_parts)}")

    print(f"{'─' * len(header)}")

    # Par más similar y más diferente
    best_sim, best_i, best_j = 0, 0, 1
    worst_sim, worst_i, worst_j = 1, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] > best_sim:
                best_sim, best_i, best_j = mat[i, j], i, j
            if mat[i, j] < worst_sim:
                worst_sim, worst_i, worst_j = mat[i, j], i, j

    print(f"\n  {c('green')}Par más similar:{c('reset')}    "
          f"{names[best_i]} ↔ {names[best_j]}  ({best_sim:.3f})")
    print(f"  {c('red')}Par más diferente:{c('reset')}  "
          f"{names[worst_i]} ↔ {names[worst_j]}  ({worst_sim:.3f})")

    result = {
        "mode": "matrix",
        "paths": paths,
        "names": names,
        "metric": metric,
        "matrix": mat.tolist(),
    }

    if args.plot:
        _plot_matrix_heatmap(mat, names)

    _maybe_save(result, args)
    return result




# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN TEMPORAL DE UN MIDI EN VENTANAS
# ══════════════════════════════════════════════════════════════════════════════

def _midi_total_ticks(mid) -> int:
    """Devuelve el tick máximo de cualquier nota en el MIDI."""
    max_tick = 0
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            max_tick = max(max_tick, abs_time)
    return max_tick


def _detect_silence_boundaries(mid, n_windows: int) -> list[int]:
    """
    Detecta fronteras por silencios largos. Divide en up to n_windows segmentos.
    Devuelve lista de ticks de inicio de cada segmento.
    """
    # Recopilar todos los eventos note_on / note_off con tiempo absoluto
    events = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type in ("note_on", "note_off"):
                is_on = msg.type == "note_on" and msg.velocity > 0
                events.append((abs_time, is_on))
    if not events:
        return [0]

    events.sort()

    # Calcular silencios entre nota_off y siguiente nota_on
    last_on = 0
    last_off = 0
    silences = []  # (tick_inicio_silencio, duracion)
    note_count = 0

    for tick, is_on in events:
        if is_on:
            if note_count == 0 and tick > last_off:
                silences.append((last_off, tick - last_off))
            last_on = tick
            note_count += 1
        else:
            note_count = max(0, note_count - 1)
            last_off = tick

    if not silences:
        return None  # señal: usar segmentación igual

    # Tomar los n_windows-1 silencios más largos como fronteras
    silences.sort(key=lambda x: -x[1])
    boundary_ticks = sorted([s[0] for s in silences[:n_windows - 1]])
    return [0] + boundary_ticks


def _split_midi_by_ticks(midi_path: str, boundaries: list[int]) -> list[str]:
    """
    Divide un MIDI por tick boundaries y guarda fragmentos en tempfiles.
    Devuelve lista de rutas temporales.
    """
    import mido
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []

    total = _midi_total_ticks(mid)
    # Añadir tick final
    boundaries = sorted(set(boundaries + [total + 1]))
    segments = list(zip(boundaries[:-1], boundaries[1:]))

    temp_paths = []
    for seg_start, seg_end in segments:
        new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        has_notes = False
        for track in mid.tracks:
            new_track = mido.MidiTrack()
            abs_time = 0
            last_included_abs = seg_start
            for msg in track:
                abs_time += msg.time
                if abs_time < seg_start:
                    # Preservar mensajes de estado (program_change, tempo, etc.)
                    if msg.is_meta or msg.type == "program_change":
                        new_track.append(msg.copy(time=0))
                    continue
                if abs_time >= seg_end:
                    break
                dt = abs_time - last_included_abs
                new_track.append(msg.copy(time=dt))
                last_included_abs = abs_time
                if msg.type == "note_on" and msg.velocity > 0:
                    has_notes = True
            if new_track:
                new_mid.tracks.append(new_track)

        if has_notes:
            tf = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
            new_mid.save(tf.name)
            temp_paths.append(tf.name)
        else:
            temp_paths.append(None)  # segmento vacío

    return temp_paths


def _vectorize_windows(midi_path: str, n_windows: int, boundary_mode: str,
                        level: str, dimensions: list | None,
                        verbose: bool) -> list[dict]:
    """
    Divide el MIDI en n_windows ventanas y vectoriza cada una.
    Devuelve lista de dicts {tick_start, tick_end, pct_start, pct_end, vectors}.
    """
    import mido
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  ERROR leyendo {midi_path}: {e}")
        return []

    total_ticks = _midi_total_ticks(mid)
    if total_ticks == 0:
        return []

    # Calcular boundaries según modo
    if boundary_mode == "silence":
        boundaries = _detect_silence_boundaries(mid, n_windows)
        if boundaries is None:
            boundary_mode = "equal"  # fallback

    if boundary_mode in ("equal", "beat"):
        step = total_ticks // n_windows
        boundaries = [i * step for i in range(n_windows)]

    # Segmentar
    seg_paths = _split_midi_by_ticks(midi_path, boundaries)
    boundary_pairs = list(zip(
        sorted(set(boundaries + [total_ticks + 1]))[:-1],
        sorted(set(boundaries + [total_ticks + 1]))[1:],
    ))

    results = []
    temp_files = []

    for i, (seg_path, (t_start, t_end)) in enumerate(zip(seg_paths, boundary_pairs)):
        if seg_path is None:
            if verbose:
                print(f"  Ventana {i+1}: vacía, omitida")
            continue
        temp_files.append(seg_path)

        if verbose:
            pct = 100 * t_start / total_ticks
            print(f"  Ventana {i+1}/{len(seg_paths)}: ticks {t_start}–{t_end} "
                  f"({pct:.0f}%)")
        else:
            print(f"  Vectorizando ventana {i+1}/{len(seg_paths)}...", end="\r")

        vec = vectorize(seg_path, level, dimensions, verbose=False)
        if vec:
            results.append({
                "window": i + 1,
                "tick_start": t_start,
                "tick_end": min(t_end, total_ticks),
                "pct_start": round(100 * t_start / total_ticks, 1),
                "pct_end":   round(100 * min(t_end, total_ticks) / total_ticks, 1),
                "vectors": vec,
            })

    # Limpiar temporales
    for p in temp_files:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    if not verbose:
        print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

# Emojis / símbolos para tendencia de cada dimensión
_TREND_UP   = "↑"
_TREND_DOWN = "↓"
_TREND_FLAT = "─"

def _trend_symbol(delta: float, threshold: float = 0.05) -> str:
    if delta > threshold:
        return _TREND_UP
    if delta < -threshold:
        return _TREND_DOWN
    return _TREND_FLAT


def _arc_label(trajectory: list[float]) -> str:
    """Clasifica el arco dramático de una dimensión a lo largo de la obra."""
    if len(trajectory) < 3:
        return "sin datos"
    start = np.mean(trajectory[:max(1, len(trajectory)//4)])
    mid   = np.mean(trajectory[len(trajectory)//4: 3*len(trajectory)//4])
    end   = np.mean(trajectory[max(1, 3*len(trajectory)//4):])

    rises   = end > start + 0.08
    falls   = end < start - 0.08
    peak    = mid > start + 0.08 and mid > end + 0.08
    valley  = mid < start - 0.08 and mid < end - 0.08
    flat    = abs(end - start) < 0.08 and abs(mid - start) < 0.08

    if flat:
        return "plano"
    if peak:
        return "arco↑ (clímax central)"
    if valley:
        return "arco↓ (valle central)"
    if rises and not falls:
        return "ascendente"
    if falls and not rises:
        return "descendente"
    if rises and falls:
        return "ondulado"
    return "irregular"


def mode_trajectory(midi_path: str, args) -> dict:
    """
    Analiza cómo evoluciona un MIDI a lo largo de su duración dividiendo
    la obra en N ventanas temporales y vectorizando cada una.

    Muestra:
      · Tabla de evolución por dimensión (valor en cada ventana)
      · Arco dramático detectado por dimensión
      · Dimensiones con mayor y menor variación
      · Curva ASCII de las dimensiones más variables
      · Eventualmente: gráfico matplotlib si --plot
    """
    n_windows   = getattr(args, "windows", 8)
    highlight   = getattr(args, "highlight", None) or []
    boundary    = getattr(args, "boundary", "equal")

    name = Path(midi_path).name
    print(f"\n{c('bold')}  Trayectoria: {name}{c('reset')}")
    print(f"  Ventanas: {n_windows}  |  Modo de frontera: {boundary}  |  Nivel: {args.level}\n")

    windows = _vectorize_windows(
        midi_path, n_windows, boundary,
        args.level, args.dimensions, args.verbose
    )

    if len(windows) < 2:
        print("  ERROR: no se obtuvieron suficientes ventanas con notas.")
        return {}

    # Recopilar dimensiones presentes en todas las ventanas
    all_dims = set(windows[0]["vectors"].keys())
    for w in windows[1:]:
        all_dims &= set(w["vectors"].keys())
    all_dims = sorted(all_dims)

    # Convertir a escalares por ventana
    # shape: {dim: [val_w1, val_w2, ...]}
    dim_series: dict[str, list[float]] = {}
    for dim in all_dims:
        dim_series[dim] = [
            flatten_vector(w["vectors"][dim]) for w in windows
        ]

    # Calcular variación total por dimensión (rango normalizado)
    dim_variation = {
        dim: float(np.max(series) - np.min(series))
        for dim, series in dim_series.items()
    }

    # Dimensiones más variables
    top_variable = sorted(dim_variation.items(), key=lambda x: -x[1])
    top_variable_dims = [d for d, _ in top_variable if dim_variation[d] > 0.02]

    # Dimensiones a mostrar:
    # · las que el usuario pidió en --highlight
    # · + las top variables
    # · siempre al menos 5 si existen
    show_dims = list(dict.fromkeys(
        highlight
        + [d for d, _ in top_variable[:max(5, len(highlight))]]
    ))
    # Filtrar las que existan en dim_series
    show_dims = [d for d in show_dims if d in dim_series]

    # ── Cabecera tabla ─────────────────────────────────────────────────────
    window_labels = [f"W{w['window']:02d}" for w in windows]
    pct_labels    = [f"{w['pct_start']:.0f}%" for w in windows]

    col_w  = 7
    name_w = 28

    hdr = f"  {'Dimensión':<{name_w}}  " + "  ".join(f"{l:>{col_w}}" for l in pct_labels)
    hdr2 = f"  {'(porcentaje obra)':>{name_w}}  " + "  ".join(f"{l:>{col_w}}" for l in window_labels)
    sep  = "─" * len(hdr)

    print(sep)
    print(hdr)
    print(hdr2)
    print(sep)

    # ── Filas de la tabla ──────────────────────────────────────────────────
    for dim in show_dims:
        series = dim_series[dim]
        label  = DIM_LABELS.get(dim, dim)[:name_w - 1]

        row_cells = []
        for i, val in enumerate(series):
            # Color según valor absoluto
            col = (c("green") if val > 0.65
                   else c("yellow") if val > 0.35
                   else c("blue"))
            # Flecha de tendencia respecto a ventana anterior
            if i == 0:
                arrow = " "
            else:
                delta = val - series[i-1]
                arrow = _trend_symbol(delta)
                arrow_col = (c("green") if arrow == _TREND_UP
                             else c("red") if arrow == _TREND_DOWN
                             else c("gray"))
                arrow = f"{arrow_col}{arrow}{c('reset')}"
            row_cells.append(f"{col}{val:5.3f}{c('reset')}{arrow}")

        # Variación y arco
        var   = dim_variation[dim]
        arc   = _arc_label(series)
        var_col = c("green") if var > 0.15 else c("yellow") if var > 0.05 else c("gray")

        print(f"  {label:<{name_w}}  {'  '.join(row_cells)}  "
              f"{var_col}Δ={var:.3f}{c('reset')}  {c('cyan')}{arc}{c('reset')}")

    print(sep)

    # ── Curvas ASCII de las dimensiones más variables ──────────────────────
    CURVE_HEIGHT = 6
    CURVE_CHARS  = " ▁▂▃▄▅▆▇█"

    most_variable = [d for d, _ in top_variable[:3] if d in dim_series]

    if most_variable:
        print(f"\n{c('bold')}  Curvas de evolución (dimensiones más variables):{c('reset')}\n")
        for dim in most_variable:
            series = np.array(dim_series[dim])
            mn, mx = series.min(), series.max()
            if mx - mn < 1e-6:
                normalized = [0.5] * len(series)
            else:
                normalized = list((series - mn) / (mx - mn))

            # Renderizar en CURVE_HEIGHT líneas
            label = DIM_LABELS.get(dim, dim)[:25]
            sparkline = ""
            for val in normalized:
                idx = int(round(val * (len(CURVE_CHARS) - 1)))
                sparkline += CURVE_CHARS[idx]

            # Fila con etiquetas de valor
            print(f"  {c('bold')}{label}{c('reset')}")
            print(f"  max {mx:.3f} ┤ {c('cyan')}{sparkline}{c('reset')}")
            # Dividir en cuartiles para dar contexto
            q1, q2, q3 = np.percentile(series, [25, 50, 75])
            print(f"        Q3 {q3:.3f}  Q2 {q2:.3f}  Q1 {q1:.3f}")
            print(f"  min {mn:.3f} ┘")
            print()

    # ── Resumen ────────────────────────────────────────────────────────────
    print(f"{c('bold')}  Resumen de arcos dramáticos:{c('reset')}")
    arcs_by_type: dict[str, list[str]] = {}
    for dim in show_dims:
        arc = _arc_label(dim_series[dim])
        arcs_by_type.setdefault(arc, []).append(DIM_LABELS.get(dim, dim))

    for arc_type, dims_in_arc in sorted(arcs_by_type.items()):
        print(f"  {c('cyan')}{arc_type:<25}{c('reset')}  {', '.join(dims_in_arc)}")

    # Punto de mayor cambio global (ventana a ventana)
    total_delta_per_transition: list[float] = []
    for i in range(1, len(windows)):
        delta = sum(
            abs(dim_series[d][i] - dim_series[d][i-1])
            for d in all_dims
        )
        total_delta_per_transition.append(delta)

    if total_delta_per_transition:
        max_trans = int(np.argmax(total_delta_per_transition))
        wa = windows[max_trans]
        wb = windows[max_trans + 1]
        print(f"\n  {c('bold')}Mayor cambio global:{c('reset')} "
              f"entre {wa['pct_start']:.0f}% y {wb['pct_start']:.0f}% de la obra "
              f"(ventanas W{wa['window']:02d}→W{wb['window']:02d})")

    # ── Plot matplotlib ────────────────────────────────────────────────────
    if args.plot:
        _plot_trajectory(dim_series, windows, show_dims, name)

    result = {
        "mode":       "trajectory",
        "source":     midi_path,
        "n_windows":  len(windows),
        "windows":    [
            {k: v for k, v in w.items() if k != "vectors"} | {"scalars": per_dim_scalars(w["vectors"])}
            for w in windows
        ],
        "dim_variation":  dim_variation,
        "dim_series":     dim_series,
    }

    _maybe_save(result, args)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: PATH
# ══════════════════════════════════════════════════════════════════════════════

def mode_path(midi_paths: list[str], args) -> dict:
    """
    Calcula el camino vectorial entre dos o más MIDIs en el espacio musical.

    Para 2 MIDIs: muestra la distancia, dirección dominante del cambio y
      genera N puntos interpolados que describen el "viaje" entre A y B.
    Para N MIDIs: muestra la secuencia de cambios A→B→C→... como un mapa
      de ruta con distancias acumuladas y cambios dominantes en cada etapa.

    Si se pasa --collection, busca en la colección los MIDIs reales más
    cercanos a cada punto interpolado del camino.
    """
    cache       = VectorCache(args.use_cache)
    n_steps     = getattr(args, "steps", 4)
    collection  = getattr(args, "collection", None) or []
    metric      = args.metric

    names = [Path(p).name for p in midi_paths]

    print(f"\n{c('bold')}  Camino vectorial:{c('reset')}")
    for i, (p, n) in enumerate(zip(midi_paths, names)):
        marker = "◉" if i == 0 else ("◎" if i == len(midi_paths)-1 else "·")
        print(f"    {marker} {n}")
    print(f"  Pasos interpolados: {n_steps}  |  Métrica: {metric}\n")

    # Vectorizar todos los MIDIs del camino
    vecs = {}
    for p in midi_paths:
        v = cache.get(p) or vectorize(p, args.level, args.dimensions, args.verbose)
        if not v:
            print(f"  ERROR: no se pudo vectorizar {Path(p).name}")
            return {}
        cache.set(p, v)
        vecs[p] = v

    if args.save_cache:
        cache.save()

    # Vectorizar colección si se pasó
    coll_vec_map = {}
    if collection:
        coll_candidates = [p for p in collection
                           if Path(p).resolve() not in {Path(mp).resolve() for mp in midi_paths}]
        print(f"  Vectorizando colección ({len(coll_candidates)} MIDIs para mapear ruta)...")
        coll_vec_map = load_vectors_batch(coll_candidates, args.level,
                                          args.dimensions, cache, verbose=args.verbose)
        if args.save_cache:
            cache.save()

    # Para cada etapa A→B, analizar el cambio
    legs = list(zip(midi_paths[:-1], midi_paths[1:]))
    all_leg_results = []
    total_distance = 0.0

    for leg_idx, (path_a, path_b) in enumerate(legs):
        name_a = Path(path_a).name
        name_b = Path(path_b).name
        vec_a  = vecs[path_a]
        vec_b  = vecs[path_b]

        dims_a, arr_a = to_comparable_vector(vec_a)
        dims_b, arr_b = to_comparable_vector(vec_b)

        # Alinear dimensiones
        common_dim_labels = sorted(set(dims_a) & set(dims_b))
        idx_a = [dims_a.index(d) for d in common_dim_labels if d in dims_a]
        idx_b = [dims_b.index(d) for d in common_dim_labels if d in dims_b]
        arr_a_aligned = arr_a[idx_a]
        arr_b_aligned = arr_b[idx_b]

        leg_dist = distance_euclidean(arr_a_aligned, arr_b_aligned)
        total_distance += leg_dist

        # Dirección del cambio por dimensión
        deltas = arr_b_aligned - arr_a_aligned
        abs_deltas = np.abs(deltas)

        # Top dimensiones que más cambian
        top_k = min(5, len(common_dim_labels))
        top_idx = np.argsort(-abs_deltas)[:top_k]

        # Imprimir etapa
        print(f"  {c('bold')}Etapa {leg_idx+1}:{c('reset')} "
              f"{c('cyan')}{name_a}{c('reset')} → {c('yellow')}{name_b}{c('reset')}")
        print(f"  {'─'*56}")
        print(f"  Distancia euclidiana: {c('bold')}{leg_dist:.4f}{c('reset')}")
        print(f"\n  {'Dimensión':<28}  {'Δ':>7}  {'Dirección':<14}  Barras")

        for i in top_idx:
            dim = common_dim_labels[i]
            delta = float(deltas[i])
            label = DIM_LABELS.get(dim.split("_")[0], dim)[:27]
            direction = (f"{c('green')}aumenta{c('reset')}" if delta > 0
                         else f"{c('red')}disminuye{c('reset')}")
            bar_full  = int(abs(delta) * 20)
            bar_empty = 20 - bar_full
            col = c("green") if delta > 0 else c("red")
            bar = f"{col}{'█' * bar_full}{'░' * bar_empty}{c('reset')}"
            print(f"  {label:<28}  {delta:>+7.4f}  {direction:<23}  {bar}")

        # Arco semántico de esta etapa
        print(f"\n  {c('gray')}Resumen de la transformación:{c('reset')}")
        increasing = [common_dim_labels[i] for i in top_idx if deltas[i] > 0.03]
        decreasing = [common_dim_labels[i] for i in top_idx if deltas[i] < -0.03]
        stable     = [common_dim_labels[i] for i in top_idx if abs(deltas[i]) <= 0.03]

        if increasing:
            labels_inc = [DIM_LABELS.get(d.split("_")[0], d) for d in increasing]
            print(f"  {c('green')}↑ Sube:{c('reset')}  {', '.join(labels_inc)}")
        if decreasing:
            labels_dec = [DIM_LABELS.get(d.split("_")[0], d) for d in decreasing]
            print(f"  {c('red')}↓ Baja:{c('reset')}  {', '.join(labels_dec)}")
        if stable:
            labels_sta = [DIM_LABELS.get(d.split("_")[0], d) for d in stable]
            print(f"  {c('gray')}≈ Estable:{c('reset')}  {', '.join(labels_sta)}")

        # Puntos interpolados
        interp_points = []
        for step in range(1, n_steps + 1):
            t = step / (n_steps + 1)
            interp_vec = arr_a_aligned + t * (arr_b_aligned - arr_a_aligned)
            interp_points.append((t, interp_vec))

        # Si hay colección, buscar el vecino real más cercano a cada punto
        coll_matches = []
        if coll_vec_map and interp_points:
            print(f"\n  {c('bold')}Vecinos reales del camino interpolado:{c('reset')}")
            print(f"  {'t':>5}  {'Similitud':>9}  MIDI más cercano")
            print(f"  {'─'*52}")

            for t, interp_arr in interp_points:
                best_sim_val  = -1.0
                best_path_val = None
                for cpath, cvec in coll_vec_map.items():
                    _, carr = to_comparable_vector(cvec)
                    # Alinear a las mismas dims
                    cdims, carr_full = to_comparable_vector(cvec)
                    c_idx = [cdims.index(d) for d in common_dim_labels if d in cdims]
                    if len(c_idx) < 2:
                        continue
                    carr_aligned = carr_full[c_idx]
                    n_align = min(len(interp_arr), len(carr_aligned))
                    dist = float(np.linalg.norm(interp_arr[:n_align] - carr_aligned[:n_align]))
                    sim  = float(math.exp(-dist))
                    if sim > best_sim_val:
                        best_sim_val  = sim
                        best_path_val = cpath

                if best_path_val:
                    coll_matches.append({"t": round(t, 2), "path": best_path_val,
                                         "similarity": round(best_sim_val, 4)})
                    cname = Path(best_path_val).name[:35]
                    color = c("green") if best_sim_val > 0.7 else c("yellow")
                    print(f"  {t:>5.2f}  {color}{best_sim_val:>9.4f}{c('reset')}  {cname}")

        all_leg_results.append({
            "from": path_a, "to": path_b,
            "distance": round(leg_dist, 5),
            "top_deltas": [
                {"dim": common_dim_labels[i], "delta": round(float(deltas[i]), 5)}
                for i in top_idx
            ],
            "interp_matches": coll_matches,
        })

        if leg_idx < len(legs) - 1:
            print(f"\n  {'═'*56}\n")

    # ── Resumen global del camino ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {c('bold')}Distancia total del camino:{c('reset')} {total_distance:.4f}")
    if len(legs) > 1:
        print(f"  Etapa más larga:  {max(all_leg_results, key=lambda x: x['distance'])['distance']:.4f}")
        print(f"  Etapa más corta:  {min(all_leg_results, key=lambda x: x['distance'])['distance']:.4f}")

    # ── Plot matplotlib ────────────────────────────────────────────────────
    if args.plot:
        _plot_path(vecs, midi_paths, coll_vec_map, names, metric)

    result = {
        "mode":           "path",
        "waypoints":      midi_paths,
        "metric":         metric,
        "total_distance": round(total_distance, 5),
        "legs":           all_leg_results,
    }

    _maybe_save(result, args)
    return result




def _check_matplotlib() -> bool:
    try:
        import matplotlib
        return True
    except ImportError:
        print("  [warn] matplotlib no disponible (pip install matplotlib)")
        return False


def _plot_compare_radar(sc_a: dict, sc_b: dict,
                         name_a: str, name_b: str,
                         dims: list[str]):
    """Radar chart comparando dos MIDIs por dimensión."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    labels = [DIM_LABELS.get(d, d) for d in dims]
    vals_a = [sc_a.get(d, 0.0) for d in dims]
    vals_b = [sc_b.get(d, 0.0) for d in dims]
    n = len(dims)
    if n < 3:
        print("  [warn] Se necesitan ≥ 3 dimensiones para el radar")
        return

    angles = [2 * math.pi * i / n for i in range(n)] + [0]
    vals_a = vals_a + [vals_a[0]]
    vals_b = vals_b + [vals_b[0]]

    # Tamaño dinámico según número de dimensiones
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size),
                            subplot_kw=dict(polar=True))

    ax.plot(angles, vals_a, "o-", linewidth=2, color="#4C72B0", label=name_a[:35])
    ax.fill(angles, vals_a, alpha=0.15, color="#4C72B0")
    ax.plot(angles, vals_b, "o-", linewidth=2, color="#DD8452", label=name_b[:35])
    ax.fill(angles, vals_b, alpha=0.15, color="#DD8452")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6, color="gray")

    # Fuente adaptativa para etiquetas radiales
    lbl_fs = max(6, min(9, int(80 / n)))
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=lbl_fs)
    ax.tick_params(axis="x", pad=8)  # separar etiquetas del centro

    ax.set_title("Comparación dimensional", pad=20, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def _plot_nearest_scatter(arr_query: np.ndarray,
                           vec_collection: dict,
                           top_results: list,
                           query_name: str,
                           metric: str):
    """Scatter PCA-2D de la colección con el query resaltado."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    paths  = list(vec_collection.keys())
    arrays = [to_comparable_vector(v)[1] for v in vec_collection.values()]
    all_arrays_raw = [arr_query] + arrays
    pt_names = [query_name] + [Path(p).stem[:14] for p in paths]

    # Alinear longitudes
    max_len = max(len(a) for a in all_arrays_raw)
    all_arr = np.array([np.pad(a, (0, max_len - len(a)))
                        for a in all_arrays_raw], dtype=float)

    # Reducción 2D
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(5, len(all_arr) - 1),
                            min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(all_arr)
    except ImportError:
        mat = all_arr - all_arr.mean(axis=0)
        cov = np.cov(mat.T)
        if cov.ndim < 2:
            cov = np.eye(2)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(-eigvals)
        emb = mat @ eigvecs[:, idx[:2]]

    top_paths = {r[0] for r in top_results}

    fig, ax = plt.subplots(figsize=(11, 8))

    # Fondo: colección
    for i, (pt, name) in enumerate(zip(emb[1:], pt_names[1:]), 1):
        in_top = paths[i - 1] in top_paths
        ax.scatter(*pt, color="#DD8452" if in_top else "#4C72B0",
                   s=130 if in_top else 35, zorder=2,
                   alpha=0.9 if in_top else 0.55,
                   edgecolors="white" if in_top else "none", linewidths=0.8)
        if in_top:
            ax.annotate(name, pt, textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.9, color="#DD8452")

    # Query
    ax.scatter(*emb[0], color="#E84040", s=220, marker="*",
               zorder=3, label=f"Query: {query_name[:25]}")

    # Márgenes proporcionales
    x_rng = emb[:, 0].ptp() or 1.0
    y_rng = emb[:, 1].ptp() or 1.0
    ax.set_xlim(emb[:, 0].min() - x_rng * 0.12, emb[:, 0].max() + x_rng * 0.12)
    ax.set_ylim(emb[:, 1].min() - y_rng * 0.12, emb[:, 1].max() + y_rng * 0.12)

    ax.set_title(f"Espacio de similitud ({metric}) — naranja: top resultados",
                 fontsize=11)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_matrix_heatmap(mat: np.ndarray, names: list[str]):
    """Heatmap de la matriz de similitud con tamaño adaptativo."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    n = len(names)
    # Tamaño de celda mínimo 0.7 pulgadas, ajustado para etiquetas largas
    max_label = max(len(nm) for nm in names) if names else 4
    label_margin = max(0.8, max_label * 0.07)
    cell_size = max(0.7, min(1.5, 14 / n))

    fig_w = n * cell_size + label_margin * 2
    fig_h = n * cell_size + label_margin * 2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Rotación y alineación correcta para etiquetas largas
    ax.set_xticklabels(names, rotation=45, ha="right",
                       fontsize=max(6, min(9, int(90 / n))))
    ax.set_yticklabels(names, fontsize=max(6, min(9, int(90 / n))))

    # Texto en cada celda, color según contraste
    cell_fs = max(5, min(8, int(70 / n)))
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            txt_color = "black" if 0.25 < v < 0.85 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=cell_fs, color=txt_color)

    plt.colorbar(im, ax=ax, label="Similitud", fraction=0.03, pad=0.02)
    ax.set_title("Matriz de similitud entre MIDIs", fontsize=11, pad=12)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _find_script(name: str) -> str | None:
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    cwd = Path.cwd() / name
    if cwd.exists():
        return str(cwd)
    found = shutil.which(name)
    return found


def _maybe_save(result: dict, args):
    if not args.output:
        return
    # Serializar numpy arrays
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"No serializable: {type(obj)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_convert)
    print(f"\n  ✓ Resultado guardado en: {args.output}")


def _detect_level() -> str:
    """Detecta si mscz2vec y music21 están disponibles."""
    script = _find_script("mscz2vec.py")
    if not script:
        return "fast"
    try:
        import music21  # noqa: F401
        return "full"
    except ImportError:
        return "fast"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        prog="analyzer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ANALYZER v1.0 — Análisis comparativo entre MIDIs
            ─────────────────────────────────────────────────
            Modos disponibles:
              compare  A.mid B.mid                → comparación dimensional
              nearest  query.mid colección/*.mid  → búsqueda por similitud
              diff     A.mid B.mid                → qué tiene A que no tenga B
              matrix   A.mid B.mid C.mid...       → matriz de similitud N×N
              profile  obra.mid                   → perfil vectorial individual
        """),
    )

    # ── Subcomandos ───────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="mode", required=True)

    # compare
    p_compare = sub.add_parser("compare", help="Comparación dimensional A vs B")
    p_compare.add_argument("a", help="MIDI A")
    p_compare.add_argument("b", help="MIDI B")
    p_compare.add_argument("--no-radar", action="store_true",
                           help="No mostrar radar chart")

    # nearest
    p_nearest = sub.add_parser("nearest", help="Búsqueda de similitud en colección")
    p_nearest.add_argument("query", help="MIDI de consulta")
    p_nearest.add_argument("collection", nargs="+", help="MIDIs de la colección")
    p_nearest.add_argument("--top", type=int, default=5,
                           help="Número de resultados (default: 5)")

    # diff
    p_diff = sub.add_parser("diff", help="Diferencias significativas A vs B")
    p_diff.add_argument("a", help="MIDI A")
    p_diff.add_argument("b", help="MIDI B")
    p_diff.add_argument("--threshold", type=float, default=0.15,
                        help="Umbral mínimo de diferencia (default: 0.15)")
    p_diff.add_argument("--show-similar", action="store_true",
                        help="Mostrar también dimensiones similares")

    # matrix
    p_matrix = sub.add_parser("matrix", help="Matriz de similitud N×N")
    p_matrix.add_argument("midis", nargs="+", help="MIDIs a comparar")
    p_matrix.add_argument("--sort", action="store_true",
                          help="Ordenar filas/columnas por similitud media")

    # profile
    p_profile = sub.add_parser("profile", help="Perfil vectorial de un MIDI")
    p_profile.add_argument("midi", help="MIDI a analizar")

    # trajectory
    p_traj = sub.add_parser("trajectory", help="Evolución interna de un MIDI en ventanas temporales")
    p_traj.add_argument("midi",  help="MIDI a analizar")
    p_traj.add_argument("--windows",   type=int, default=8,
                        help="Número de ventanas temporales (default: 8)")
    p_traj.add_argument("--highlight", nargs="+", default=None,
                        help="Dimensiones a destacar en la tabla (ej. pitch_mean density)")
    p_traj.add_argument("--boundary",  default="equal",
                        choices=["equal", "silence", "beat"],
                        help="Método de segmentación (default: equal)")

    # path
    p_path = sub.add_parser("path", help="Camino vectorial entre dos o más MIDIs")
    p_path.add_argument("waypoints", nargs="+",
                        help="MIDIs que definen el camino (mínimo 2)")
    p_path.add_argument("--steps",      type=int, default=4,
                        help="Puntos interpolados entre cada par (default: 4)")
    p_path.add_argument("--collection", nargs="*", default=None,
                        help="Colección en la que buscar vecinos reales del camino")


    for p in (p_compare, p_nearest, p_diff, p_matrix, p_profile, p_traj, p_path):
        p.add_argument("--level",      default="auto",
                       choices=["fast", "full", "auto"],
                       help="Nivel de vectorización (default: auto)")
        p.add_argument("--dimensions", nargs="+", default=None,
                       help="Dimensiones específicas a analizar")
        p.add_argument("--metric",     default="euclidean",
                       choices=["euclidean", "cosine", "manhattan"],
                       help="Métrica de distancia (default: euclidean)")
        p.add_argument("--output",     default=None,
                       help="Guardar resultado en JSON")
        p.add_argument("--plot",       action="store_true",
                       help="Visualización matplotlib")
        p.add_argument("--no-color",   action="store_true",
                       help="Sin colores ANSI")
        p.add_argument("--save-cache", default=None, metavar="FILE",
                       help="Guardar vectores calculados en JSON")
        p.add_argument("--use-cache",  default=None, metavar="FILE",
                       help="Cargar vectores de caché previa")
        p.add_argument("--verbose",    "-v", action="store_true",
                       help="Informe detallado")

    args = parser.parse_args()
    USE_COLOR = not args.no_color

    # Auto-detectar nivel si no se especificó
    if args.level == "auto":
        detected = _detect_level()
        if args.verbose:
            print(f"  [auto] Nivel de vectorización detectado: {detected}")
        args.level = detected

    # Banner
    print(f"\n{c('bold')}{c('cyan')}  ANALYZER v{VERSION}{c('reset')}  "
          f"— nivel={args.level}  métrica={args.metric}")

    # Despachar modo
    if args.mode == "compare":
        args.plot = args.plot and not args.no_radar
        mode_compare(args.a, args.b, args)

    elif args.mode == "nearest":
        mode_nearest(args.query, args.collection, args)

    elif args.mode == "diff":
        mode_diff(args.a, args.b, args)

    elif args.mode == "matrix":
        mode_matrix(args.midis, args)

    elif args.mode == "profile":
        mode_profile(args.midi, args)

    elif args.mode == "trajectory":
        mode_trajectory(args.midi, args)

    elif args.mode == "path":
        if len(args.waypoints) < 2:
            print("  ERROR: path necesita al menos 2 MIDIs.")
            sys.exit(1)
        mode_path(args.waypoints, args)


if __name__ == "__main__":
    main()
