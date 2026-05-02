#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                   PREFERENCE TRAINER  v1.1                          ║
║         Aprende tus preferencias musicales. Evalúa MIDIs.          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Tres comandos — pipeline completo autónomo:                        ║
║                                                                      ║
║  INDEX — vectoriza un corpus de MIDIs y genera el índice .npz      ║
║                                                                      ║
║    python preference_trainer.py index ./midis/                      ║
║    python preference_trainer.py index ./midis/ --output corpus.npz ║
║                                                                      ║
║  Procesa todos los .mid/.midi del directorio (recursivo) y guarda  ║
║  un vector de 10 dimensiones por archivo:                           ║
║    pitch_center · pitch_range · interval_mean · contour            ║
║    density · rhythm_variance · polyphony                           ║
║    velocity_mean · velocity_variance · silence_ratio               ║
║                                                                      ║
║  100.000 MIDIs ≈ 1-3 horas. Solo hay que correrlo una vez.         ║
║                                                                      ║
║  Opciones:                                                           ║
║    --output FILE   Nombre del índice (default: corpus.npz)          ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  TRAIN — aprende tus preferencias desde un corpus indexado          ║
║                                                                      ║
║    # Modo rating: puntúa MIDIs de 1 a 5                             ║
║    python preference_trainer.py train corpus.npz                    ║
║    python preference_trainer.py train corpus.npz --mode rating      ║
║                                                                      ║
║    # Modo contraste: elige entre dos MIDIs                           ║
║    python preference_trainer.py train corpus.npz --mode contrast    ║
║                                                                      ║
║    # Guardar modelo en archivo específico                            ║
║    python preference_trainer.py train corpus.npz --model mis_prefs  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  EVAL — evalúa un MIDI con el modelo entrenado                       ║
║                                                                      ║
║    # Solo preferencia aprendida                                      ║
║    python preference_trainer.py eval pieza.mid --model mis_prefs    ║
║                                                                      ║
║    # Preferencia + afinidad a obras de referencia                   ║
║    python preference_trainer.py eval pieza.mid                      ║
║        --model mis_prefs --affinity ./obras_afines/ --weight 0.4    ║
║                                                                      ║
║    # Evaluar varios MIDIs a la vez                                  ║
║    python preference_trainer.py eval *.mid --model mis_prefs        ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  REPRODUCCIÓN INTEGRADA                                              ║
║                                                                      ║
║  Cada MIDI se reproduce automáticamente al presentarse.             ║
║  La música continúa mientras el usuario decide; se para al votar.  ║
║                                                                      ║
║  Controles durante la votación:                                      ║
║  · Rating:   p=volver a reproducir · x=parar                        ║
║  · Contraste: pa/pb=reproducir A o B · x=parar                      ║
║                                                                      ║
║  Backends de audio (se detectan automáticamente):                   ║
║    pygame   pip install pygame   (sin proceso externo)              ║
║    timidity sudo apt install timidity  (mejor con soundfont)        ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SELECCIÓN ACTIVA DE MUESTRAS                                        ║
║                                                                      ║
║  El programa no presenta muestras al azar. En cada paso elige       ║
║  la muestra (o el par) que maximiza la información ganada:          ║
║                                                                      ║
║  · Modo rating:    selección por máxima incertidumbre del modelo    ║
║    La muestra donde el modelo predice con menos confianza es        ║
║    la que más puede cambiar los pesos → se presenta primero.        ║
║                                                                      ║
║  · Modo contraste: selección por máximo desacuerdo esperado         ║
║    Se buscan pares donde los dos MIDIs son similares en el espacio  ║
║    vectorial pero el modelo los separa con poca confianza,          ║
║    forzando una decisión informativa.                                ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODELO DE PREFERENCIAS                                              ║
║                                                                      ║
║  Regresión lineal sobre los 10 vectores musicales del corpus.       ║
║  Entrenamiento incremental con SGD. Se guarda como JSON             ║
║  entre sesiones y acumula ejemplos de forma indefinida.             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  OPCIONES INDEX                                                      ║
║    --output FILE             Archivo .npz de salida (def: corpus.npz)║
║                                                                      ║
║  OPCIONES TRAIN                                                      ║
║    --mode rating|contrast   Modalidad de anotación (def: rating)    ║
║    --model FILE              Nombre base del modelo (def: prefs)     ║
║    --rounds N                Número de rondas (def: 20)             ║
║    --soundfont FILE          Soundfont .sf2 para timidity            ║
║    --no-autoplay             No reproducir automáticamente           ║
║    --candidate-pool N        Pool de candidatos para selección activa║
║                              (def: 50, mayor = mejor selección)      ║
║                                                                      ║
║  OPCIONES EVAL                                                       ║
║    --model FILE              Nombre base del modelo (def: prefs)     ║
║    --affinity DIR            Carpeta de obras afines (opcional)      ║
║    --weight W                Peso de la afinidad 0-1 (def: 0.5)      ║
║                              0 = solo preferencia, 1 = solo afinidad ║
║    --verbose                 Desglose completo de dimensiones        ║
║    --json                    Salida JSON machine-readable            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  FLUJO TÍPICO                                                         ║
║                                                                      ║
║    # 1. Indexar el corpus (una sola vez)                             ║
║    python preference_trainer.py index ./midis/                      ║
║                                                                      ║
║    # 2. Entrenar el modelo de preferencias                          ║
║    python preference_trainer.py train corpus.npz --rounds 30        ║
║    python preference_trainer.py train corpus.npz --mode contrast    ║
║                                                                      ║
║    # 3. Evaluar una pieza propia                                     ║
║    python preference_trainer.py eval mi_pieza.mid                   ║
║                                                                      ║
║    # 4. Evaluar con afinidad a obras de referencia                  ║
║    python preference_trainer.py eval mi_pieza.mid                   ║
║        --affinity ./referencias/ --weight 0.4                       ║
║                                                                      ║
║    # 5. Inspeccionar el modelo aprendido                             ║
║    python preference_trainer.py info                                 ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS                                                        ║
║    pip install mido numpy          # obligatorio                    ║
║    pip install pygame              # opcional: audio (recomendado)  ║
║    sudo apt install timidity       # opcional: audio alternativo    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import time
import random
import argparse
import subprocess
import threading
from pathlib import Path
from collections import defaultdict

# UTF-8 en stdin/stdout
if hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── Dependencias opcionales ─────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ════════════════════════════════════════════════════════════════════
#  COLORES
# ════════════════════════════════════════════════════════════════════

USE_COLOR = sys.stdout.isatty()

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")
def green(t):   return _c(t, "32")
def yellow(t):  return _c(t, "33")
def cyan(t):    return _c(t, "36")
def red(t):     return _c(t, "31")
def magenta(t): return _c(t, "35")
def blue(t):    return _c(t, "34")


# ════════════════════════════════════════════════════════════════════
#  VECTORIZADOR — 10 dimensiones (mido + math, sin numpy)
# ════════════════════════════════════════════════════════════════════

DIM_NAMES = [
    "pitch_center",       # 0  altura media normalizada (0-1)
    "pitch_range",        # 1  rango de alturas normalizado (0-1)
    "interval_mean",      # 2  intervalo medio entre notas (0-1)
    "contour",            # 3  dirección global: 0 baja, 0.5 plana, 1 sube
    "density",            # 4  notas por segundo (0-1, saturado a 10 nps)
    "rhythm_variance",    # 5  irregularidad rítmica (0-1)
    "polyphony",          # 6  media de voces simultáneas (0-1, sat. a 8)
    "velocity_mean",      # 7  dinámica media (0-1)
    "velocity_variance",  # 8  variación dinámica (0-1)
    "silence_ratio",      # 9  proporción de silencio (0-1)
]

N_DIMS = len(DIM_NAMES)

DIM_LABELS = {
    "pitch_center":     ("grave", "agudo"),
    "pitch_range":      ("estrecho", "amplio"),
    "interval_mean":    ("stepwise", "saltos grandes"),
    "contour":          ("descendente", "ascendente"),
    "density":          ("sparse", "denso"),
    "rhythm_variance":  ("regular", "irregular"),
    "polyphony":        ("monofónico", "polifónico"),
    "velocity_mean":    ("piano", "forte"),
    "velocity_variance":("dinámica estable", "dinámica variable"),
    "silence_ratio":    ("continuo", "silencioso"),
}


def vectorize_midi(path: str) -> dict:
    """Extrae vector de 10 dimensiones de un MIDI. Solo mido + math."""
    result = {
        "path": str(path),
        "vector": None,
        "duration_s": 0.0,
        "n_tracks": 0,
        "n_notes": 0,
        "tempo_bpm": 120,
        "error": None,
    }

    if not HAS_MIDO:
        result["error"] = "mido_not_installed"
        return result

    try:
        mid = mido.MidiFile(str(path), clip=True)
    except Exception as e:
        result["error"] = str(e)
        return result

    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat or 480
    notes = []
    active = {}
    current_time_s = 0.0

    for track in mid.tracks:
        current_time_s = 0.0
        active = {}
        for msg in track:
            delta_s = mido.tick2second(msg.time, ticks_per_beat, tempo)
            current_time_s += delta_s

            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (current_time_s, msg.velocity)
            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in active:
                    on_time, vel = active.pop(msg.note)
                    dur = current_time_s - on_time
                    notes.append((on_time, msg.note, vel, max(dur, 0.05)))

    result["n_tracks"] = len(mid.tracks)
    result["n_notes"] = len(notes)
    result["tempo_bpm"] = round(60_000_000 / tempo)

    if len(notes) < 4:
        result["error"] = "too_few_notes"
        return result

    end_times = [n[0] + n[3] for n in notes]
    total_dur = max(end_times) if end_times else 0.0
    result["duration_s"] = total_dur

    if total_dur < 2.0:
        result["error"] = "too_short"
        return result

    pitches     = [n[1] for n in notes]
    velocities  = [n[2] for n in notes]
    onset_times = sorted([n[0] for n in notes])

    pitch_mean   = sum(pitches) / len(pitches)
    pitch_center = (pitch_mean - 21) / (108 - 21)

    pitch_range = (max(pitches) - min(pitches)) / 87.0

    intervals    = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    interval_mean = min((sum(intervals) / len(intervals)) / 12.0, 1.0)

    half = len(pitches) // 2
    first_half  = pitches[:half]
    second_half = pitches[half:]
    mean_first  = sum(first_half) / len(first_half)
    mean_second = sum(second_half) / len(second_half)
    contour_raw = mean_second - mean_first
    contour = (max(-1.0, min(1.0, contour_raw / 12.0)) + 1) / 2

    density = min(len(notes) / max(total_dur, 1.0) / 10.0, 1.0)

    iois = [onset_times[i+1] - onset_times[i] for i in range(len(onset_times)-1)]
    if iois:
        mean_ioi = sum(iois) / len(iois)
        var_ioi  = sum((x - mean_ioi)**2 for x in iois) / len(iois)
        rhythm_variance = min(math.sqrt(var_ioi) / 2.0, 1.0)
    else:
        rhythm_variance = 0.0

    sample_times = [total_dur * i / 100 for i in range(100)]
    poly_counts = [
        sum(1 for (on, p, v, dur) in notes if on <= t <= on + dur)
        for t in sample_times
    ]
    polyphony = min(sum(poly_counts) / len(poly_counts) / 8.0, 1.0)

    velocity_mean = sum(velocities) / len(velocities) / 127.0

    vel_mean = sum(velocities) / len(velocities)
    vel_var  = sum((v - vel_mean)**2 for v in velocities) / len(velocities)
    velocity_variance = min(math.sqrt(vel_var) / 64.0, 1.0)

    total_note_time = sum(n[3] for n in notes)
    silence_ratio   = max(0.0, 1.0 - (total_note_time / max(total_dur, 1.0)))

    result["vector"] = [
        pitch_center, pitch_range, interval_mean, contour,
        density, rhythm_variance, polyphony,
        velocity_mean, velocity_variance, silence_ratio,
    ]
    return result


def describe_vector(v: list) -> str:
    """Descripción textual breve de un vector de 10 dims."""
    parts = []
    labels = [
        ("pitch_center", 0), ("density", 4), ("polyphony", 6),
        ("contour", 3), ("silence_ratio", 9)
    ]
    for name, idx in labels:
        lo, hi = DIM_LABELS[name]
        val = v[idx]
        if val < 0.35:
            parts.append(lo)
        elif val > 0.65:
            parts.append(hi)
    return ", ".join(parts) if parts else "neutro"


def format_duration(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}:{sec:02d}"


# ════════════════════════════════════════════════════════════════════
#  MODELO DE PREFERENCIAS — regresión lineal con SGD
# ════════════════════════════════════════════════════════════════════

class PreferenceModel:
    """
    Modelo lineal que mapea vector musical (10 dims) → score de preferencia [0,1].

    Internamente trabaja con pesos w[0..9] y sesgo b.
    score = sigmoid(dot(w, x) + b)

    Entrenamiento incremental con SGD:
      · Modo rating: target = (rating - 1) / 4  → [0, 1]
      · Modo contraste: Bradley-Terry pair loss

    Selección activa:
      · Rating:    incertidumbre = |score - 0.5| mínimo → muestra más dudosa
      · Contraste: pares donde |score_a - score_b| mínimo → decisión más informativa
    """

    VERSION = "1.0"

    def __init__(self):
        self.weights          = [0.0] * N_DIMS
        self.bias             = 0.0
        self.lr               = 0.05        # learning rate
        self.n_train          = 0           # nº de ejemplos vistos
        self.history          = []          # lista de {type, path(s), rating/winner}
        self.annotated_paths  = {}          # path → lista de ratings/outcomes vistos
        self.feature_stats    = {           # para normalización interna
            "mean": [0.5] * N_DIMS,
            "std":  [0.3] * N_DIMS,
        }

    # ── Predicción ──────────────────────────────────────────────────

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def score(self, vector: list) -> float:
        """Devuelve score de preferencia en [0, 1]."""
        logit = sum(self.weights[i] * vector[i] for i in range(N_DIMS)) + self.bias
        return self._sigmoid(logit)

    def uncertainty(self, vector: list) -> float:
        """Incertidumbre: 1 - 2*|score - 0.5|. Máxima en 0.5."""
        s = self.score(vector)
        return 1.0 - 2.0 * abs(s - 0.5)

    # ── Actualización con rating (1-5) ───────────────────────────────

    def update_rating(self, vector: list, rating: int, path: str = ""):
        """SGD paso con un ejemplo de rating."""
        target = (rating - 1) / 4.0  # [0, 1]
        pred   = self.score(vector)
        error  = target - pred
        # gradiente de MSE con sigmoid
        grad_logit = error * pred * (1 - pred)
        for i in range(N_DIMS):
            self.weights[i] += self.lr * grad_logit * vector[i]
        self.bias    += self.lr * grad_logit
        self.n_train += 1
        self._decay_lr()
        if path:
            self.annotated_paths.setdefault(path, []).append(rating)

    # ── Actualización con par (winner/loser) ────────────────────────

    def update_contrast(self, winner_vec: list, loser_vec: list,
                        winner_path: str = "", loser_path: str = ""):
        """Bradley-Terry: winner > loser."""
        s_w = self.score(winner_vec)
        s_l = self.score(loser_vec)
        # P(w > l) = sigmoid(logit_w - logit_l)
        # Gradient de log-loss
        prob_correct = self._sigmoid(
            sum((self.weights[i]) * (winner_vec[i] - loser_vec[i])
                for i in range(N_DIMS))
            # bias cancela al restar
        )
        # grad respecto a w_i: (1 - prob_correct) * (w_i - l_i)
        grad = 1.0 - prob_correct
        for i in range(N_DIMS):
            diff = winner_vec[i] - loser_vec[i]
            self.weights[i] += self.lr * grad * diff
        self.n_train += 1
        self._decay_lr()
        if winner_path:
            self.annotated_paths.setdefault(winner_path, []).append("win")
        if loser_path:
            self.annotated_paths.setdefault(loser_path, []).append("lose")

    def _decay_lr(self):
        """Learning rate decay suave."""
        self.lr = max(0.005, 0.05 / (1 + self.n_train * 0.02))

    # ── Selección activa ─────────────────────────────────────────────

    def annotation_count(self, path: str) -> int:
        """Número de veces que este path ha sido anotado."""
        return len(self.annotated_paths.get(path, []))

    def novelty_score(self, path: str) -> float:
        """1.0 = nunca visto; decrece con cada anotación previa."""
        n = self.annotation_count(path)
        return 1.0 / (1.0 + n)

    def select_rating_candidate(self, vectors: list, indices: list,
                                 paths: list = None) -> int:
        """
        Selecciona el índice del vector más informativo para modo rating.
        Estrategia: máxima incertidumbre si hay datos; aleatorio si el modelo es virgen.
        """
        if self.n_train < 3:
            # Sin modelo: preferir no vistos, luego aleatorio
            if paths:
                unseen = [i for i in indices if self.annotation_count(str(paths[i])) == 0]
                return random.choice(unseen) if unseen else random.choice(indices)
            return random.choice(indices)

        # Score combinado: incertidumbre × novedad
        # · incertidumbre alta → el modelo no sabe qué pensar de él
        # · novedad alta      → nunca o pocas veces anotado
        # La novedad evita re-presentar MIDIs ya bien calibrados
        def _info(i):
            unc = self.uncertainty(vectors[i])
            nov = self.novelty_score(str(paths[i])) if paths else 1.0
            return unc * nov

        best_idx = max(indices, key=_info)
        return best_idx

    def select_contrast_pair(self, vectors: list, indices: list,
                              n_candidates: int = 50,
                              paths: list = None) -> tuple:
        """
        Selecciona el par (i, j) más informativo para modo contraste.

        Criterio: par con score_diff mínimo (decisión difícil) entre
        los pares vectorialmente más similares (comparación justa).
        """
        if len(indices) < 2:
            raise ValueError("Se necesitan al menos 2 candidatos.")

        # Submuestreo para eficiencia
        pool = random.sample(indices, min(n_candidates, len(indices)))

        if self.n_train < 3:
            # Sin modelo todavía: par más similar (aporta más información
            # sobre qué dimensión importa)
            best_pair = None
            best_sim  = -1.0
            for ii in range(len(pool)):
                for jj in range(ii + 1, len(pool)):
                    a, b = pool[ii], pool[jj]
                    sim = _cosine_similarity(vectors[a], vectors[b])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (a, b)
            return best_pair

        # Con modelo: maximizar informatividad del par
        # Criterio: similitud vectorial alta + score_diff bajo + novedad alta
        # La novedad desalienta repetir pares donde ambos ya están bien calibrados
        best_pair  = None
        best_score = -1.0
        for ii in range(len(pool)):
            for jj in range(ii + 1, len(pool)):
                a, b = pool[ii], pool[jj]
                sa  = self.score(vectors[a])
                sb  = self.score(vectors[b])
                diff = abs(sa - sb)
                sim  = _cosine_similarity(vectors[a], vectors[b])
                # Novedad del par: media geométrica de las novedades individuales
                nov_a = self.novelty_score(str(paths[a])) if paths else 1.0
                nov_b = self.novelty_score(str(paths[b])) if paths else 1.0
                nov   = math.sqrt(nov_a * nov_b)
                # Informatividad total
                info = sim * (1.0 - diff) * nov
                if info > best_score:
                    best_score = info
                    best_pair = (a, b)
        return best_pair

    # ── Diagnóstico ─────────────────────────────────────────────────

    def top_dimensions(self, n: int = 5) -> list:
        """Devuelve las n dimensiones con mayor peso absoluto."""
        indexed = sorted(
            enumerate(self.weights),
            key=lambda x: abs(x[1]), reverse=True
        )
        return [(DIM_NAMES[i], w) for i, w in indexed[:n]]

    def summary(self) -> str:
        lines = []
        lines.append(f"  Ejemplos de entrenamiento: {bold(str(self.n_train))}")
        lines.append(f"  Learning rate actual:       {self.lr:.4f}")
        lines.append(f"  Bias:                       {self.bias:+.3f}")
        lines.append("")
        lines.append(f"  {bold('Dimensiones más influyentes:')}")
        for name, w in self.top_dimensions(5):
            bar_len = int(abs(w) * 15)
            bar = ("+" if w >= 0 else "-") * bar_len
            lo, hi = DIM_LABELS[name]
            direction = hi if w > 0 else lo
            lines.append(
                f"    {name:<20} {bar:<16} {w:+.3f}  → {dim(direction)}"
            )
        return "\n".join(lines)

    # ── Serialización ───────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "version":          self.VERSION,
            "weights":          self.weights,
            "bias":             self.bias,
            "lr":               self.lr,
            "n_train":          self.n_train,
            "history":          self.history[-200:],
            "annotated_paths":  self.annotated_paths,
            "feature_stats":    self.feature_stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PreferenceModel":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        m = cls()
        m.weights          = data.get("weights", [0.0] * N_DIMS)
        m.bias             = data.get("bias", 0.0)
        m.lr               = data.get("lr", 0.05)
        m.n_train          = data.get("n_train", 0)
        m.history          = data.get("history", [])
        m.annotated_paths  = data.get("annotated_paths", {})
        m.feature_stats    = data.get("feature_stats", m.feature_stats)
        return m


# ════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x*x for x in a))
    nb   = math.sqrt(sum(x*x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _euclidean_distance(a: list, b: list) -> float:
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))


def _load_corpus_npz(path: str) -> tuple:
    """
    Carga un corpus.npz generado por corpus_explorer.
    Devuelve (vectors, paths, meta) como listas Python puras.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy no instalado. pip install numpy")

    data    = np.load(path, allow_pickle=True)
    vectors = data["vectors"].tolist()
    paths   = data["paths"].tolist()
    _meta_key = "meta" if "meta" in data else ("metadata" if "metadata" in data else None)
    meta    = data[_meta_key].tolist() if _meta_key else ["{}"] * len(paths)
    return vectors, paths, meta


def _model_path(name: str) -> str:
    return f"{name}.prefs.json"


# Estado global del reproductor (una sola instancia activa a la vez)
_player_proc  = None   # subprocess de timidity, si está activo
_player_stop  = None   # threading.Event para el hilo pygame


def _print_continuation_banner(model, vectors: list, paths: list):
    """
    Muestra el estado del modelo previo y el progreso de cobertura del corpus
    al continuar una sesión de entrenamiento ya iniciada.
    """
    n_corpus    = len(vectors)
    n_annotated = len(model.annotated_paths)
    n_seen      = sum(1 for p in paths if str(p) in model.annotated_paths)
    coverage    = n_seen / n_corpus if n_corpus > 0 else 0.0

    # Distribucion de incertidumbre actual sobre el corpus
    uncertainties = [model.uncertainty(v) for v in vectors]
    high_unc  = sum(1 for u in uncertainties if u > 0.7)   # muy incierto
    med_unc   = sum(1 for u in uncertainties if 0.3 < u <= 0.7)
    low_unc   = sum(1 for u in uncertainties if u <= 0.3)  # bien calibrado

    # Barra de cobertura
    bar_w   = 30
    filled  = int(coverage * bar_w)
    cov_bar = green("█" * filled) + dim("░" * (bar_w - filled))

    print(f"\n  {bold('─ Modelo existente cargado ─')}")
    print(f"  Ejemplos acumulados:   {bold(str(model.n_train))}")
    print(f"  MIDIs distintos vistos:{bold(str(n_seen))} / {n_corpus}")
    print(f"  Cobertura del corpus:  {cov_bar}  {coverage*100:.1f}%")
    print()
    print(f"  {bold('Distribución de incertidumbre sobre el corpus:')}")
    # Mini histograma inline
    unc_total = n_corpus or 1
    hi_bar  = green("▓" * int(high_unc / unc_total * 25))
    med_bar = yellow("▓" * int(med_unc  / unc_total * 25))
    lo_bar  = dim("▓"   * int(low_unc  / unc_total * 25))
    print(f"  Alta  (>0.7)  {hi_bar:<26} {high_unc:>5}  ← prioritarios")
    print(f"  Media (0.3-0.7) {med_bar:<24} {med_unc:>5}")
    print(f"  Baja  (<0.3)  {lo_bar:<26} {low_unc:>5}  ← bien calibrados")
    print()

    if model.n_train > 0:
        print(f"  {bold('Lo que el modelo ha aprendido:')}")
        for name, w in model.top_dimensions(5):
            bar_len = int(abs(w) * 15)
            bar     = ("+" if w >= 0 else "-") * bar_len
            lo, hi  = DIM_LABELS[name]
            direction = hi if w > 0 else lo
            print(f"    {name:<20} {bar:<16} {w:+.3f}  → {dim(direction)}")
        print()

    if high_unc > 0:
        print(f"  {green(str(high_unc))} MIDIs con alta incertidumbre se presentarán primero.")
    else:
        print(f"  {yellow('El modelo cubre bien el corpus.')} "
              f"Las nuevas rondas refinarán casos límite.")
    print()


def _stop_playback():
    """Para cualquier reproducción activa (timidity o pygame)."""
    global _player_proc, _player_stop
    if _player_proc is not None:
        try:
            _player_proc.terminate()
            _player_proc.wait(timeout=2)
        except Exception:
            try:
                _player_proc.kill()
            except Exception:
                pass
        _player_proc = None
    if _player_stop is not None:
        _player_stop.set()
        _player_stop = None
    if HAS_PYGAME:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass


def _play_midi(midi_path: str, soundfont: str = None, blocking: bool = False):
    """
    Reproduce un MIDI en background.

    Backends (en orden de preferencia):
      1. pygame  — sin proceso externo, control preciso, sin soundfont propio
      2. timidity — mejor calidad con soundfont, proceso externo

    Si blocking=True espera a que termine o a que el usuario pulse Enter.
    Si blocking=False lanza la reproducción y retorna inmediatamente;
    llamar a _stop_playback() para parar.
    """
    global _player_proc, _player_stop

    _stop_playback()   # parar cualquier reproducción anterior

    name = Path(midi_path).name

    # ── Backend 1: pygame ────────────────────────────────────────────
    if HAS_PYGAME and not soundfont:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(midi_path)
            pygame.mixer.music.play()
            print(f"  {green(chr(9654))} {bold(name)}  "
                  f"{dim('(pygame — Enter para parar)')}", flush=True)

            if blocking:
                _pygame_wait_for_enter()
            else:
                stop_ev = threading.Event()
                _player_stop = stop_ev
                def _pygame_auto_stop():
                    while pygame.mixer.music.get_busy() and not stop_ev.is_set():
                        time.sleep(0.1)
                    if not stop_ev.is_set():
                        stop_ev.set()
                threading.Thread(target=_pygame_auto_stop, daemon=True).start()
            return
        except Exception as e:
            print(yellow(f"  (pygame error: {e} — intentando timidity)"))

    # ── Backend 2: timidity ──────────────────────────────────────────
    cmd = ["timidity"]
    if soundfont:
        cmd += ["-x", f"soundfont {soundfont}"]
    cmd.append(midi_path)

    try:
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        _player_proc = proc
        print(f"  {green(chr(9654))} {bold(name)}  "
              f"{dim('(timidity — Enter para parar)')}", flush=True)

        if blocking:
            _timidity_wait_for_enter(proc)
            _player_proc = None
        else:
            # Hilo que limpia _player_proc cuando el proceso termina solo
            def _watch():
                global _player_proc
                proc.wait()
                if _player_proc is proc:
                    _player_proc = None
            threading.Thread(target=_watch, daemon=True).start()

    except FileNotFoundError:
        _player_proc = None
        print(yellow("  (timidity no instalado — sin audio)"))
        print(yellow("    sudo apt install timidity   # Ubuntu/Debian"))
        print(yellow("    brew install timidity       # macOS"))
    except Exception as e:
        _player_proc = None
        print(yellow(f"  (error de reproducción: {e})"))


def _pygame_wait_for_enter():
    """Espera Enter o fin de canción con pygame (usa select en Unix)."""
    try:
        import select as _select
        while pygame.mixer.music.get_busy():
            ready, _, _ = _select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.readline()
                pygame.mixer.music.stop()
                print(f"  {yellow(chr(9646))} {dim('Parado.')}")
                return
        print(f"  {green(chr(9646))} {dim('Fin.')}")
    except (AttributeError, OSError):
        # Windows fallback
        try:
            import msvcrt
            while pygame.mixer.music.get_busy():
                if msvcrt.kbhit():
                    msvcrt.getwch()
                    pygame.mixer.music.stop()
                    print(f"  {yellow(chr(9646))} {dim('Parado.')}")
                    return
                time.sleep(0.1)
        except ImportError:
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        print(f"  {green(chr(9646))} {dim('Fin.')}")


def _timidity_wait_for_enter(proc):
    """Espera Enter o fin de proceso timidity con un hilo listener."""
    stop_ev = threading.Event()

    def _listener():
        try:
            input()
        except Exception:
            pass
        stop_ev.set()

    t = threading.Thread(target=_listener, daemon=True)
    t.start()

    while proc.poll() is None and not stop_ev.is_set():
        time.sleep(0.1)

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    if stop_ev.is_set():
        print(f"  {yellow(chr(9646))} {dim('Parado.')}")
    else:
        print(f"  {green(chr(9646))} {dim('Fin.')}")


def _midi_header(info: dict, idx: int = None, label: str = None) -> str:
    name   = Path(str(info.get("path", "?"))).name
    dur    = format_duration(info.get("duration_s", 0))
    bpm    = info.get("tempo_bpm", "?")
    notes  = info.get("n_notes", "?")
    prefix = f"[{label}] " if label else (f"#{idx+1}  " if idx is not None else "")
    return f"{prefix}{cyan(name)}  {dim(f'{dur} · {bpm}bpm · {notes} notas')}"


# ════════════════════════════════════════════════════════════════════
#  MODO TRAIN — rating
# ════════════════════════════════════════════════════════════════════

def cmd_train_rating(args, model: PreferenceModel, vectors, paths, meta):
    """Sesión de entrenamiento por rating (1-5)."""

    indices  = list(range(len(vectors)))
    rounds   = args.rounds
    played   = set()
    soundfont = getattr(args, "soundfont", None)

    print(f"\n{bold('MODO RATING')}  — puntúa cada MIDI de 1 a 5")
    print(dim("  1=no me gusta nada · 3=indiferente · 5=me encanta"))
    print(dim("  p=reproducir · x=parar · s=saltar · q=guardar y salir"))
    print()

    done = 0
    while done < rounds:
        available = [i for i in indices if i not in played]
        if not available:
            print(yellow("  Has puntuado todos los MIDIs disponibles."))
            break

        idx  = model.select_rating_candidate(vectors, available, paths=paths)
        played.add(idx)

        v    = vectors[idx]
        path = str(paths[idx])
        try:
            m = json.loads(str(meta[idx]))
        except Exception:
            m = {}
        m["path"]       = path
        m["duration_s"] = m.get("duration_s", 0)
        m["tempo_bpm"]  = m.get("tempo_bpm", "?")
        m["n_notes"]    = m.get("n_notes", "?")

        n_seen_this = model.annotation_count(path)
        seen_tag    = dim(f"  [visto {n_seen_this}x]") if n_seen_this > 0 else ""
        print(f"  {'─'*52}")
        print(f"  {_midi_header(m, done)}{seen_tag}")
        print(f"  {dim(describe_vector(v))}")
        if model.n_train > 0:
            pred = model.score(v)
            unc  = model.uncertainty(v)
            unc_label = green("alta") if unc > 0.7 else (yellow("media") if unc > 0.3 else dim("baja"))
            print(f"  {dim(f'Predicción: {pred:.2f}  ·  incertidumbre: {unc:.2f} (')}{unc_label}{dim(')')}")
        print()

        # Reproducción automática en background (salvo --no-autoplay)
        if not getattr(args, 'no_autoplay', False):
            _play_midi(path, soundfont=soundfont, blocking=False)

        # Bucle de input — la música sigue sonando mientras el usuario decide
        rating = None
        while rating is None:
            raw = input("  Puntuación [1-5 / p=repetir / x=parar / s=saltar / q=salir]: ").strip().lower()
            if raw in ("q", "quit", "exit"):
                _stop_playback()
                return "quit"
            elif raw in ("p", "play", "r"):
                _play_midi(path, soundfont=soundfont, blocking=False)
            elif raw in ("x", "stop", "0"):
                _stop_playback()
            elif raw in ("s", "skip"):
                rating = "skip"
            elif raw.isdigit() and 1 <= int(raw) <= 5:
                rating = int(raw)
                _stop_playback()
            else:
                print(red("  Introduce un número del 1 al 5."))

        if rating == "skip":
            print(dim("  Saltado.\n"))
            continue

        model.update_rating(v, rating, path=path)
        model.history.append({"type": "rating", "path": path, "rating": rating})

        stars = "★" * rating + "☆" * (5 - rating)
        print(green(f"  {stars}  Anotado.\n"))
        done += 1

        if done % 5 == 0 and done > 0:
            print(f"  {bold('─ Estado del modelo ─')}")
            print(model.summary())
            print()

    _stop_playback()
    return "done"


# ════════════════════════════════════════════════════════════════════
#  MODO TRAIN — contraste
# ════════════════════════════════════════════════════════════════════

def cmd_train_contrast(args, model: PreferenceModel, vectors, paths, meta):
    """Sesión de entrenamiento por contraste (A vs B)."""

    indices = list(range(len(vectors)))
    rounds  = args.rounds
    seen_pairs: set = set()

    print(f"\n{bold('MODO CONTRASTE')}  — elige el MIDI que prefieres")
    print(dim("  a=elige A · b=elige B · e=empate · s=saltar · q=guardar y salir"))
    print(dim("  pa=reproducir A · pb=reproducir B · x=parar"))
    print()

    done = 0
    while done < rounds:
        # Selección activa del par
        try:
            pair = model.select_contrast_pair(
                vectors, indices,
                n_candidates=getattr(args, "candidate_pool", 50),
                paths=paths
            )
        except ValueError:
            print(yellow("  Corpus demasiado pequeño."))
            break

        if pair is None:
            break

        a_idx, b_idx = pair
        pair_key = (min(a_idx, b_idx), max(a_idx, b_idx))
        if pair_key in seen_pairs:
            # Intentar un par aleatorio
            alt = random.sample(indices, min(50, len(indices)))
            try:
                pair = model.select_contrast_pair(vectors, alt,
                    n_candidates=getattr(args, "candidate_pool", 50),
                    paths=paths)
                a_idx, b_idx = pair
                pair_key = (min(a_idx, b_idx), max(a_idx, b_idx))
            except Exception:
                pass

        seen_pairs.add(pair_key)

        va   = vectors[a_idx]
        vb   = vectors[b_idx]
        path_a = str(paths[a_idx])
        path_b = str(paths[b_idx])

        def get_meta(idx, path):
            try:
                m = json.loads(str(meta[idx]))
            except Exception:
                m = {}
            m.setdefault("path", path)
            m.setdefault("duration_s", 0)
            m.setdefault("tempo_bpm", "?")
            m.setdefault("n_notes", "?")
            return m

        ma = get_meta(a_idx, path_a)
        mb = get_meta(b_idx, path_b)

        seen_a = model.annotation_count(path_a)
        seen_b = model.annotation_count(path_b)
        tag_a  = dim(f"  [{seen_a}x]") if seen_a > 0 else ""
        tag_b  = dim(f"  [{seen_b}x]") if seen_b > 0 else ""
        print(f"  {'─'*52}  Ronda {done+1}/{rounds}")
        print(f"  {bold('A')}  {_midi_header(ma)}{tag_a}")
        print(f"     {dim(describe_vector(va))}")
        print(f"  {bold('B')}  {_midi_header(mb)}{tag_b}")
        print(f"     {dim(describe_vector(vb))}")

        soundfont = getattr(args, "soundfont", None)

        if model.n_train > 0:
            sa  = model.score(va)
            sb  = model.score(vb)
            ua  = model.uncertainty(va)
            ub  = model.uncertainty(vb)
            print(f"  {dim(f'Predicción: A={sa:.2f} (unc={ua:.2f})  B={sb:.2f} (unc={ub:.2f})')}")
        print()

        # Reproducción automática de A en background (salvo --no-autoplay)
        if not getattr(args, 'no_autoplay', False):
            _play_midi(path_a, soundfont=soundfont, blocking=False)

        choice = None
        while choice is None:
            raw = input("  ¿Qué prefieres? [a/b / pa/pb=reproducir / x=parar / e=empate / s=saltar / q=salir]: ").strip().lower()
            if raw in ("q", "quit"):
                _stop_playback()
                return "quit"
            elif raw in ("pa", "ra"):
                _play_midi(path_a, soundfont=soundfont, blocking=False)
            elif raw in ("pb", "rb"):
                _play_midi(path_b, soundfont=soundfont, blocking=False)
            elif raw in ("x", "stop"):
                _stop_playback()
            elif raw in ("s", "skip"):
                choice = "skip"
            elif raw in ("a",):
                choice = "a"
                _stop_playback()
            elif raw in ("b",):
                choice = "b"
                _stop_playback()
            elif raw in ("e", "empate", "="):
                choice = "tie"
                _stop_playback()
            else:
                print(red("  Introduce a, b, e, pa, pb o s."))

        if choice == "skip":
            print(dim("  Saltado.\n"))
            continue

        if choice == "a":
            model.update_contrast(va, vb, winner_path=path_a, loser_path=path_b)
            model.history.append({"type": "contrast", "winner": path_a, "loser": path_b})
            print(green(f"  ✓ A elegida.\n"))
        elif choice == "b":
            model.update_contrast(vb, va, winner_path=path_b, loser_path=path_a)
            model.history.append({"type": "contrast", "winner": path_b, "loser": path_a})
            print(green(f"  ✓ B elegida.\n"))
        elif choice == "tie":
            # Empate: ambas actualizaciones con gradient pequeño
            model.lr *= 0.3
            model.update_contrast(va, vb)
            model.update_contrast(vb, va)
            model.lr /= 0.3
            model.history.append({"type": "contrast", "winner": None,
                                   "paths": [path_a, path_b]})
            print(dim(f"  Empate anotado.\n"))

        done += 1

        if done % 5 == 0 and done > 0:
            print(f"  {bold('─ Estado del modelo ─')}")
            print(model.summary())
            print()

    _stop_playback()
    return "done"


# ════════════════════════════════════════════════════════════════════
#  MODO EVAL
# ════════════════════════════════════════════════════════════════════

def cmd_eval(args):
    """Evalúa uno o varios MIDIs con el modelo entrenado."""

    model_file = _model_path(args.model)
    if not Path(model_file).exists():
        print(red(f"Modelo no encontrado: {model_file}"))
        print(yellow("  Entrena primero con:  python preference_trainer.py train <corpus.npz>"))
        sys.exit(1)

    model = PreferenceModel.load(model_file)
    print(f"\n{bold('PREFERENCE TRAINER — EVAL')}")
    print(f"  Modelo: {cyan(model_file)}  ({model.n_train} ejemplos de entrenamiento)")

    # Cargar vectores de afinidad si se especifica carpeta
    affinity_vectors = []
    affinity_label   = ""
    if args.affinity:
        affinity_dir = Path(args.affinity)
        if not affinity_dir.exists():
            print(red(f"  Carpeta de afinidad no encontrada: {affinity_dir}"))
        else:
            midi_files = (
                list(affinity_dir.rglob("*.mid")) +
                list(affinity_dir.rglob("*.midi"))
            )
            print(f"  Cargando {len(midi_files)} obras afines de {cyan(str(affinity_dir))}...")
            for mf in midi_files:
                res = vectorize_midi(str(mf))
                if res["vector"]:
                    affinity_vectors.append(res["vector"])
            affinity_label = str(affinity_dir)
            print(f"  {green(str(len(affinity_vectors)))} vectorizadas correctamente.")

    weight_pref = 1.0 - args.weight  if affinity_vectors else 1.0
    weight_aff  = args.weight         if affinity_vectors else 0.0

    # Recoger archivos a evaluar
    midi_files = []
    for pattern in args.midi:
        p = Path(pattern)
        if p.is_dir():
            midi_files += list(p.rglob("*.mid")) + list(p.rglob("*.midi"))
        elif p.exists():
            midi_files.append(p)
        else:
            # Glob manual (Windows no expande)
            import glob
            midi_files += [Path(f) for f in glob.glob(pattern)]

    if not midi_files:
        print(red("  No se encontraron archivos MIDI."))
        sys.exit(1)

    results = []

    print(f"\n{'─'*60}")
    for mf in midi_files:
        res = vectorize_midi(str(mf))
        if res["error"]:
            err_msg = res["error"]
            print(f"  " + yellow("!") + f"  {mf.name}  " + dim("error: " + err_msg))
            continue

        v = res["vector"]

        # Score de preferencia
        pref_score = model.score(v)

        # Score de afinidad (media de similitudes coseno con el corpus de referencia)
        aff_score = 0.0
        if affinity_vectors:
            sims = [_cosine_similarity(v, av) for av in affinity_vectors]
            # Normalizar similitudes coseno (rango [-1, 1]) → [0, 1]
            aff_score = (sum(sims) / len(sims) + 1) / 2

        # Score combinado
        if affinity_vectors:
            final_score = weight_pref * pref_score + weight_aff * aff_score
        else:
            final_score = pref_score

        results.append({
            "path":        str(mf),
            "name":        mf.name,
            "pref_score":  pref_score,
            "aff_score":   aff_score,
            "final_score": final_score,
            "vector":      v,
            "meta":        res,
        })

    if not results:
        print(red("  Ningún MIDI pudo ser procesado."))
        sys.exit(1)

    # Ordenar por score final descendente
    results.sort(key=lambda r: r["final_score"], reverse=True)

    # ── Salida ────────────────────────────────────────────────────────

    print(f"\n{'═'*60}")
    if affinity_vectors:
        print(f"  {bold('RESULTADOS')}  "
              f"[pref×{weight_pref:.2f} + afinidad×{weight_aff:.2f}]")
    else:
        print(f"  {bold('RESULTADOS')}  [solo preferencia]")
    print(f"{'═'*60}\n")

    for rank, r in enumerate(results, 1):
        bar_len = int(r["final_score"] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)

        score_color = green if r["final_score"] >= 0.65 else (
            yellow if r["final_score"] >= 0.40 else red
        )

        dur  = format_duration(r["meta"]["duration_s"])
        bpm  = r["meta"]["tempo_bpm"]
        desc = describe_vector(r["vector"])

        print(f"  {rank:>2}. {cyan(r['name'])}")
        fs = f"{r['final_score']:.3f}"
        print(f"      {score_color(bar)}  {bold(fs)}")

        if affinity_vectors:
            print(f"      Preferencia: {r['pref_score']:.3f}  "
                  f"·  Afinidad: {r['aff_score']:.3f}")

        print(f"      {dim(f'{dur} · {bpm} bpm · {desc}')}")

        if args.verbose:
            print(f"      {dim('─ Detalle de dimensiones:')}")
            for i, name in enumerate(DIM_NAMES):
                val  = r["vector"][i]
                w    = model.weights[i]
                lo, hi = DIM_LABELS[name]
                bar2 = "▮" * int(val * 10) + "▯" * (10 - int(val * 10))
                contrib = w * val
                sign    = "+" if contrib >= 0 else ""
                print(f"        {name:<20} {bar2}  {val:.2f}  "
                      f"{dim(f'w={w:+.3f} → {sign}{contrib:.3f}')}")
        print()

    # ── Resumen numérico (formato mínimo garantizado) ─────────────────
    print(f"{'─'*60}")
    print(f"  {bold('RESUMEN NUMÉRICO')}")
    print()
    for r in results:
        if affinity_vectors:
            print(f"  {r['name']:<40}  "
                  f"score={r['final_score']:.4f}  "
                  f"pref={r['pref_score']:.4f}  "
                  f"aff={r['aff_score']:.4f}")
        else:
            print(f"  {r['name']:<40}  score={r['final_score']:.4f}")
    print()

    # JSON machine-readable (opcional con --json)
    if getattr(args, "json_out", False):
        output = []
        for r in results:
            entry = {
                "path":        r["path"],
                "score":       round(r["final_score"], 4),
                "pref_score":  round(r["pref_score"], 4),
                "vector":      [round(x, 4) for x in r["vector"]],
            }
            if affinity_vectors:
                entry["aff_score"] = round(r["aff_score"], 4)
            output.append(entry)
        print(json.dumps(output, indent=2, ensure_ascii=False))


# ════════════════════════════════════════════════════════════════════
#  COMANDO TRAIN (orquestador)
# ════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Orquesta la sesión de entrenamiento."""

    if not HAS_MIDO:
        print(red("Error: mido no instalado.  pip install mido"))
        sys.exit(1)
    if not HAS_NUMPY:
        print(red("Error: numpy no instalado.  pip install numpy"))
        sys.exit(1)

    corpus_path = args.corpus
    if not Path(corpus_path).exists():
        print(red(f"Corpus no encontrado: {corpus_path}"))
        sys.exit(1)

    print(f"\n{bold('PREFERENCE TRAINER — TRAIN')}")
    print(f"  Corpus: {cyan(corpus_path)}")

    # Cargar corpus
    print(f"  Cargando índice...", end=" ", flush=True)
    try:
        vectors, paths, meta = _load_corpus_npz(corpus_path)
    except Exception as e:
        print(red(f"\n  Error cargando corpus: {e}"))
        sys.exit(1)

    valid = [(v, p, m) for v, p, m in zip(vectors, paths, meta) if v is not None]
    if not valid:
        print(red("  El corpus no contiene vectores válidos."))
        sys.exit(1)

    vectors, paths, meta = zip(*valid)
    vectors = list(vectors)
    paths   = list(paths)
    meta    = list(meta)
    print(green(f"{len(vectors)} MIDIs"))

    # Cargar o crear modelo
    model_file = _model_path(args.model)
    if Path(model_file).exists():
        model = PreferenceModel.load(model_file)
        _print_continuation_banner(model, vectors, paths)
    else:
        model = PreferenceModel()
        print(f"  {yellow('Modelo nuevo')} — se guardará en {cyan(model_file)}")
        print()

    # Sesión de entrenamiento
    try:
        if args.mode == "contrast":
            result = cmd_train_contrast(args, model, vectors, paths, meta)
        else:
            result = cmd_train_rating(args, model, vectors, paths, meta)
    except KeyboardInterrupt:
        result = "quit"

    # Guardar modelo
    model.save(model_file)
    print(f"\n{green('✓')} Modelo guardado: {cyan(model_file)}")
    print(f"  Total de ejemplos de entrenamiento: {bold(str(model.n_train))}")
    print()
    print(f"  {bold('─ Estado final del modelo ─')}")
    print(model.summary())
    print()
    print(f"  Para evaluar un MIDI:")
    print(f"    {dim(f'python preference_trainer.py eval tu_pieza.mid --model {args.model}')}")


# ════════════════════════════════════════════════════════════════════
#  COMANDO INFO — inspeccionar el modelo
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
#  COMANDO INDEX — vectorizar corpus
# ════════════════════════════════════════════════════════════════════

def cmd_index(args):
    """Vectoriza un directorio de MIDIs y guarda el índice .npz."""

    if not HAS_MIDO:
        print(red("Error: mido no instalado.  pip install mido"))
        sys.exit(1)
    if not HAS_NUMPY:
        print(red("Error: numpy no instalado.  pip install numpy"))
        sys.exit(1)

    midi_dir = Path(args.path)
    if not midi_dir.exists():
        print(red(f"Error: no existe '{midi_dir}'"))
        sys.exit(1)

    output = Path(args.output)

    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    total = len(midi_files)

    if total == 0:
        print(yellow("No se encontraron archivos MIDI en el directorio."))
        sys.exit(0)

    print(f"\n{bold('PREFERENCE TRAINER — INDEX')}")
    print(f"  Directorio:  {cyan(str(midi_dir))}")
    print(f"  Archivos:    {total:,}")
    print(f"  Salida:      {cyan(str(output))}\n")

    vectors  = []
    paths    = []
    metadata = []
    errors   = 0

    bar_width = 40
    t_start   = time.time()

    for i, path in enumerate(midi_files):
        pct     = (i + 1) / total
        filled  = int(bar_width * pct)
        bar     = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.time() - t_start
        eta     = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  [{bar}] {i+1}/{total}  ETA:{eta:.0f}s  "
              f"OK:{len(vectors)}  Err:{errors}",
              end="", flush=True)

        r = vectorize_midi(str(path))

        if r["error"]:
            errors += 1
            continue

        vectors.append(r["vector"])
        paths.append(str(path))
        metadata.append({
            "duration_s": r["duration_s"],
            "n_notes":    r["n_notes"],
            "n_tracks":   r["n_tracks"],
            "tempo_bpm":  r["tempo_bpm"],
        })

    print()

    if not vectors:
        print(red("\nNo se pudo vectorizar ningún archivo."))
        sys.exit(1)

    # Filtrar vectores con NaN o dimensión incorrecta
    clean_v, clean_p, clean_m = [], [], []
    bad = 0
    for v, p, m in zip(vectors, paths, metadata):
        if v is not None and len(v) == N_DIMS and all(x == x for x in v):
            clean_v.append(v)
            clean_p.append(p)
            clean_m.append(m)
        else:
            bad += 1

    if bad:
        print(yellow(f"  {bad} vectores descartados (NaN o dimensión incorrecta)"))

    arr_vectors = np.array(clean_v, dtype=np.float32)
    arr_paths   = np.array(clean_p)
    arr_meta    = np.array([json.dumps(m) for m in clean_m])

    np.savez_compressed(
        str(output),
        vectors   = arr_vectors,
        paths     = arr_paths,
        meta      = arr_meta,       # clave canónica
        dim_names = np.array(DIM_NAMES),
    )

    elapsed = time.time() - t_start
    print(f"\n{bold('ÍNDICE GENERADO')}")
    print(f"  Vectorizados: {green(str(len(clean_v)))}")
    print(f"  Errores:      {red(str(errors + bad))}")
    print(f"  Tiempo:       {elapsed:.1f}s")
    print(f"  Archivo:      {cyan(str(output))}")
    out_str = str(output)
    print(f"\n  Siguiente paso:")
    print(f"    {dim('python preference_trainer.py train ' + out_str)}\n")


def cmd_info(args):
    """Muestra información del modelo guardado."""
    model_file = _model_path(args.model)
    if not Path(model_file).exists():
        print(red(f"Modelo no encontrado: {model_file}"))
        sys.exit(1)

    model = PreferenceModel.load(model_file)
    print(f"\n{bold('PREFERENCE TRAINER — INFO')}")
    print(f"  Archivo: {cyan(model_file)}")
    print()
    print(model.summary())
    print()

    if model.history:
        print(f"  {bold('Últimas anotaciones:')}")
        for entry in model.history[-10:]:
            if entry["type"] == "rating":
                stars = "★" * entry["rating"] + "☆" * (5 - entry["rating"])
                name  = Path(entry["path"]).name
                print(f"    {stars}  {dim(name)}")
            elif entry["type"] == "contrast" and entry.get("winner"):
                winner = Path(entry["winner"]).name
                loser  = Path(entry["loser"]).name
                print(f"    {green('▶')} {winner}  {dim('sobre')}  {loser}")
        print()


# ════════════════════════════════════════════════════════════════════
#  ENTRADA PRINCIPAL
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Preference Trainer — aprende y evalúa preferencias musicales MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── index ─────────────────────────────────────────────────────
    p_index = sub.add_parser("index", help="Vectorizar corpus de MIDIs → .npz")
    p_index.add_argument("path",
                         help="Directorio raíz que contiene los MIDIs (recursivo)")
    p_index.add_argument("--output", default="corpus.npz",
                         help="Archivo .npz de salida (default: corpus.npz)")

    # ── train ──────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Sesión de entrenamiento interactiva")
    p_train.add_argument("corpus",
                         help="Archivo .npz generado con: python preference_trainer.py index <dir/>")
    p_train.add_argument("--mode", default="rating", choices=["rating", "contrast"],
                         help="Modalidad de anotación (default: rating)")
    p_train.add_argument("--model", default="prefs",
                         help="Nombre base del modelo (default: prefs → prefs.prefs.json)")
    p_train.add_argument("--rounds", type=int, default=20,
                         help="Número de rondas de anotación (default: 20)")
    p_train.add_argument("--soundfont", default=None,
                         help="Soundfont .sf2 para timidity o pygame")
    p_train.add_argument("--no-autoplay", action="store_true",
                         help="No reproducir automáticamente al presentar cada MIDI")
    p_train.add_argument("--candidate-pool", type=int, default=50,
                         help="Pool de candidatos para selección activa (default: 50)")

    # ── eval ───────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluar uno o varios MIDIs")
    p_eval.add_argument("midi", nargs="+",
                        help="Archivo(s) MIDI a evaluar")
    p_eval.add_argument("--model", default="prefs",
                        help="Nombre base del modelo (default: prefs)")
    p_eval.add_argument("--affinity", default=None,
                        help="Carpeta con obras afines para ponderar afinidad")
    p_eval.add_argument("--weight", type=float, default=0.5,
                        help="Peso de la afinidad [0-1] (default: 0.5). "
                             "0=solo preferencia, 1=solo afinidad")
    p_eval.add_argument("--verbose", action="store_true",
                        help="Desglose completo de dimensiones")
    p_eval.add_argument("--json", dest="json_out", action="store_true",
                        help="Salida JSON machine-readable al final")

    # ── info ───────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Inspeccionar modelo guardado")
    p_info.add_argument("--model", default="prefs",
                        help="Nombre base del modelo (default: prefs)")

    args = parser.parse_args()

    if args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
