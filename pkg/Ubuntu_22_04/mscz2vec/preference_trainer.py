#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                   PREFERENCE TRAINER  v1.2                          ║
║         Aprende tus preferencias musicales. Evalúa MIDIs.          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  7 comandos — pipeline completo autónomo:                           ║
║  index · train · eval · rank · set · validate · info               ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  INDEX — vectoriza un corpus de MIDIs y genera el índice .npz      ║
║                                                                      ║
║    python preference_trainer.py index ./midis/                      ║
║    python preference_trainer.py index ./midis/ --output corpus.npz ║
║    python preference_trainer.py index ./midis/ --extended-features  ║
║    python preference_trainer.py index ./melodias/ --melodic-features║
║                                                                      ║
║  Procesa todos los .mid/.midi del directorio (recursivo). Solo      ║
║  hay que ejecutarlo una vez (100 000 MIDIs ≈ 1-3 h).               ║
║                                                                      ║
║  Modos de features (elige uno; incompatibles entre sí):            ║
║                                                                      ║
║  · Base (10 dims, defecto)                                          ║
║      pitch_center · pitch_range · interval_mean · contour          ║
║      density · rhythm_variance · polyphony                         ║
║      velocity_mean · velocity_variance · silence_ratio             ║
║                                                                      ║
║  · --extended-features (20 dims)                                    ║
║      Las 10 base + tension_arc · density_arc · pitch_entropy       ║
║      interval_entropy · repetition · harmonic_tension              ║
║      rhythmic_complexity · note_duration_mean                      ║
║      climax_position · dynamic_arc                                  ║
║      Captura estructura temporal y armonía; mejor discriminación.  ║
║                                                                      ║
║  · --melodic-features (16 dims, solo melodía monofónica)           ║
║      pitch_center · pitch_range · interval_mean · contour          ║
║      density · rhythm_variance · velocity_mean · silence_ratio     ║
║      melodic_peak_ratio · step_ratio · leap_ratio · phrase_arch    ║
║      climax_position · note_duration_mean · pitch_entropy          ║
║      interval_entropy                                               ║
║                                                                      ║
║  El modelo detecta el modo automáticamente por el nº de pesos      ║
║  (10=base · 16=melódico · 20=extendido). Modelos de distintos      ║
║  modos son incompatibles — usa archivos .prefs.json separados.     ║
║                                                                      ║
║  Opciones:                                                           ║
║    --output FILE           Archivo .npz de salida (def: corpus.npz) ║
║    --extended-features     Vectores de 20 dimensiones               ║
║    --melodic-features      Vectores de 16 dims para melodía mono    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  TRAIN — sesión interactiva de anotación                            ║
║                                                                      ║
║    # Modo rating: puntúa MIDIs de 1 a 5 (defecto)                  ║
║    python preference_trainer.py train corpus.npz                    ║
║    python preference_trainer.py train corpus.npz --rounds 30       ║
║                                                                      ║
║    # Modo contraste: elige entre dos MIDIs                          ║
║    python preference_trainer.py train corpus.npz --mode contrast   ║
║                                                                      ║
║    # Con soundfont y modelo específico                              ║
║    python preference_trainer.py train corpus.npz                   ║
║        --model mis_prefs --soundfont ./gm.sf2 --rounds 20          ║
║                                                                      ║
║  Controles durante la sesión:                                        ║
║    Rating:    1-5=puntuar · p=reproducir · x=parar · q=guardar     ║
║    Contraste: A/B=elegir · pa/pb=reproducir A o B · x=parar        ║
║                                                                      ║
║  Selección activa de muestras (no aleatoria):                       ║
║    Rating:    muestra la pieza con mayor incertidumbre del modelo   ║
║    Contraste: muestra el par con mayor desacuerdo esperado          ║
║                                                                      ║
║  Opciones:                                                           ║
║    --mode rating|contrast  Modalidad de anotación (def: rating)     ║
║    --model FILE            Nombre base del modelo (def: prefs)      ║
║    --rounds N              Número de rondas (def: 20)               ║
║    --soundfont FILE        Soundfont .sf2 para timidity o pygame    ║
║    --no-autoplay           No reproducir automáticamente            ║
║    --candidate-pool N      Pool para selección activa (def: 50)     ║
║                            Mayor valor = mejor selección, más lento ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  EVAL — evalúa uno o varios MIDIs con el modelo entrenado           ║
║                                                                      ║
║    # Evaluación básica                                               ║
║    python preference_trainer.py eval pieza.mid --model mis_prefs   ║
║                                                                      ║
║    # Con percentil, histograma y vecinos del corpus                 ║
║    python preference_trainer.py eval pieza.mid                      ║
║        --model mis_prefs --corpus corpus.npz                        ║
║                                                                      ║
║    # Preferencia + afinidad a obras de referencia                   ║
║    python preference_trainer.py eval pieza.mid                      ║
║        --model mis_prefs --affinity ./obras_afines/ --weight 0.4   ║
║                                                                      ║
║    # Múltiples MIDIs + desglose de dimensiones + salida JSON        ║
║    python preference_trainer.py eval *.mid                          ║
║        --model mis_prefs --corpus corpus.npz --verbose --json       ║
║                                                                      ║
║  Salida con --corpus (opcional pero recomendado):                   ║
║    · Score final y score de preferencia/afinidad por separado       ║
║    · Percentil sobre el corpus completo                             ║
║    · Histograma ASCII con posición marcada                          ║
║    · Dimensiones que favorecen / perjudican el score (--verbose)    ║
║    · Vecinos más cercanos con rating conocido                       ║
║                                                                      ║
║  Opciones:                                                           ║
║    --model FILE    Nombre base del modelo (def: prefs)              ║
║    --corpus FILE   .npz para percentil, histograma y vecinos        ║
║    --affinity DIR  Carpeta de obras afines                          ║
║    --weight W      Peso afinidad 0-1 (def: 0.5)                     ║
║                    0 = solo preferencia · 1 = solo afinidad         ║
║    --verbose       Desglose completo por dimensión                  ║
║    --json          Salida JSON machine-readable al final            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  RANK — escanear y clasificar el corpus completo                    ║
║                                                                      ║
║    python preference_trainer.py rank corpus.npz --model prefs      ║
║                                                                      ║
║    # Con afinidad y exportación CSV                                 ║
║    python preference_trainer.py rank corpus.npz                     ║
║        --model prefs --affinity ./referencias/ --weight 0.4        ║
║        --top 30 --csv ranking.csv                                   ║
║                                                                      ║
║    # Solo MIDIs no anotados, con anotación interactiva              ║
║    python preference_trainer.py rank corpus.npz --model prefs      ║
║        --exclude-annotated --interactive --soundfont ./gm.sf2      ║
║                                                                      ║
║  Muestra:                                                            ║
║    · Histograma global de preferencia sobre el corpus               ║
║    · TOP N — las más preferidas según el modelo                     ║
║    · BOTTOM N — las menos preferidas                                ║
║    · INCIERTAS N — donde el modelo duda más (candidatas a train)    ║
║    · Exportación opcional a CSV con ranking completo                ║
║                                                                      ║
║  Opciones:                                                           ║
║    --model FILE          Nombre base del modelo (def: prefs)        ║
║    --top N               Entradas por lista (def: 20)               ║
║    --affinity DIR        Carpeta de obras afines                    ║
║    --weight W            Peso afinidad 0-1 (def: 0.5)               ║
║    --exclude-annotated   Excluir MIDIs ya anotados en train         ║
║    --interactive         Anotar MIDIs directamente desde el ranking ║
║    --soundfont FILE      Soundfont .sf2 para modo interactivo       ║
║    --csv FILE            Exportar ranking completo a CSV            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SET — anotar preferencia directamente sin sesión interactiva       ║
║                                                                      ║
║    python preference_trainer.py set cancion.mid --score 5           ║
║    python preference_trainer.py set *.mid --score 3 --model prefs  ║
║    python preference_trainer.py set ./carpeta/ --score 4            ║
║                                                                      ║
║    # Con fuerza de ajuste personalizada                             ║
║    python preference_trainer.py set pieza.mid --score 5            ║
║        --model mis_prefs --repetitions 20                           ║
║                                                                      ║
║  Asigna un rating directamente y actualiza el modelo               ║
║  inmediatamente. Muestra el cambio de score antes/después.          ║
║  --repetitions controla la fuerza del ajuste (se aplica N veces).  ║
║                                                                      ║
║  Opciones:                                                           ║
║    --score N         Preferencia a asignar, obligatorio (1-5)       ║
║    --model FILE      Nombre base del modelo (def: prefs)            ║
║    --repetitions N   Fuerza del ajuste, nº de aplicaciones (def:10) ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  VALIDATE — validación cruzada del modelo                           ║
║                                                                      ║
║    python preference_trainer.py validate --model prefs              ║
║    python preference_trainer.py validate --model prefs              ║
║        --corpus corpus.npz --folds 10                               ║
║                                                                      ║
║  Mide con k-fold cross-validation:                                  ║
║    · MAE — error absoluto medio en predicción de rating             ║
║    · Pair accuracy — % de pares ordenados correctamente             ║
║    · Correlación de Spearman entre predicciones y ratings reales    ║
║  Incluye interpretación automática y recomendaciones de mejora.     ║
║  Requiere al menos 10 ejemplos anotados para resultados fiables.    ║
║                                                                      ║
║  Opciones:                                                           ║
║    --model FILE    Nombre base del modelo (def: prefs)              ║
║    --corpus FILE   .npz para recuperar vectores del historial       ║
║    --folds N       Número de folds (def: 5)                         ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  INFO — inspeccionar el modelo guardado                             ║
║                                                                      ║
║    python preference_trainer.py info                                ║
║    python preference_trainer.py info --model mis_prefs             ║
║                                                                      ║
║  Muestra los pesos aprendidos por dimensión, número de ejemplos    ║
║  anotados, historial de sesiones y estadísticas del modelo.         ║
║                                                                      ║
║  Opciones:                                                           ║
║    --model FILE    Nombre base del modelo (def: prefs)              ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  REPRODUCCIÓN INTEGRADA                                              ║
║                                                                      ║
║  Cada MIDI se reproduce automáticamente al presentarse.             ║
║  La música continúa mientras el usuario decide; se para al votar.  ║
║                                                                      ║
║  Backends de audio (se detectan automáticamente):                   ║
║    pygame    pip install pygame          (recomendado, sin externo) ║
║    timidity  sudo apt install timidity   (mejor calidad con sf2)    ║
║                                                                      ║
║  Para usar soundfont con timidity:                                  ║
║    python preference_trainer.py train corpus.npz                   ║
║        --soundfont /usr/share/sounds/sf2/FluidR3_GM.sf2            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  FLUJO TÍPICO                                                        ║
║                                                                      ║
║    # 1. Indexar el corpus (una sola vez)                            ║
║    python preference_trainer.py index ./midis/                      ║
║    python preference_trainer.py index ./midis/ --extended-features  ║
║                                                                      ║
║    # 2. Entrenar el modelo (repetir hasta tener ≥30 ejemplos)       ║
║    python preference_trainer.py train corpus.npz --rounds 20       ║
║    python preference_trainer.py train corpus.npz --mode contrast   ║
║                                                                      ║
║    # 3. Validar la calidad del modelo                               ║
║    python preference_trainer.py validate --model prefs             ║
║                                                                      ║
║    # 4. Evaluar una pieza propia                                    ║
║    python preference_trainer.py eval mi_pieza.mid                  ║
║        --model prefs --corpus corpus.npz --verbose                 ║
║                                                                      ║
║    # 5. Clasificar el corpus entero y exportar                      ║
║    python preference_trainer.py rank corpus.npz                    ║
║        --model prefs --top 50 --csv ranking.csv                    ║
║                                                                      ║
║    # 6. Anotar manualmente sin sesión interactiva                   ║
║    python preference_trainer.py set favorita.mid --score 5         ║
║                                                                      ║
║    # 7. Inspeccionar el modelo                                      ║
║    python preference_trainer.py info --model prefs                 ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODELO DE PREFERENCIAS                                              ║
║                                                                      ║
║  Regresión lineal sobre los vectores musicales del corpus.          ║
║  Entrenamiento incremental con SGD. Se guarda como JSON             ║
║  entre sesiones y acumula ejemplos de forma indefinida.             ║
║  Nombre del archivo: <model>.prefs.json  (def: prefs.prefs.json)   ║
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

# Features base (siempre calculadas)
DIM_NAMES_BASE = [
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

# Features extendidas (--extended-features, más costosas)
DIM_NAMES_EXTENDED = [
    "tension_arc",        # 10 arco de tensión: sube(1) / plana(0.5) / baja(0)
    "density_arc",        # 11 arco de densidad: cómo cambia la densidad a lo largo
    "pitch_entropy",      # 12 entropía de la distribución de alturas (diversidad tonal)
    "interval_entropy",   # 13 entropía de intervalos (diversidad melódica)
    "repetition",         # 14 índice de repetición motívica (0=ninguna, 1=muy repetitivo)
    "harmonic_tension",   # 15 tensión armónica media (disonancias vs consonancias)
    "rhythmic_complexity",# 16 complejidad rítmica por densidad de cambios de IOI
    "note_duration_mean", # 17 duración media de notas normalizada
    "climax_position",    # 18 posición temporal del clímax (0=inicio, 1=final)
    "dynamic_arc",        # 19 arco dinámico: crescendo(1) / decrescendo(0) / plano(0.5)
]

DIM_NAMES = DIM_NAMES_BASE  # se amplía a DIM_NAMES_BASE + DIM_NAMES_EXTENDED con --extended-features

N_DIMS = len(DIM_NAMES)

DIM_LABELS_BASE = {
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

DIM_LABELS_EXTENDED = {
    "tension_arc":         ("tensión decae", "tensión crece"),
    "density_arc":         ("rarefacción", "densificación"),
    "pitch_entropy":       ("tonal fijo", "tonal diverso"),
    "interval_entropy":    ("melódica repetitiva", "melódica variada"),
    "repetition":          ("sin repetición", "muy repetitivo"),
    "harmonic_tension":    ("consonante", "disonante"),
    "rhythmic_complexity": ("ritmo simple", "ritmo complejo"),
    "note_duration_mean":  ("notas cortas", "notas largas"),
    "climax_position":     ("clímax al inicio", "clímax al final"),
    "dynamic_arc":         ("decrescendo", "crescendo"),
}

DIM_LABELS = {**DIM_LABELS_BASE, **DIM_LABELS_EXTENDED}

# Features melódicas — para MIDIs de una sola voz (--melodic-features)
# Reemplaza el set base: elimina polyphony/velocity_variance (irrelevantes
# en melodía monofónica) y añade análisis melódico fino.
DIM_NAMES_MELODIC = [
    "pitch_center",        # 0  altura media normalizada (0-1)
    "pitch_range",         # 1  rango de alturas normalizado (0-1)
    "interval_mean",       # 2  intervalo medio (0-1)
    "contour",             # 3  dirección global (0 baja, 0.5 plana, 1 sube)
    "density",             # 4  notas por segundo (0-1)
    "rhythm_variance",     # 5  irregularidad rítmica (0-1)
    "velocity_mean",       # 6  dinámica media (0-1)
    "silence_ratio",       # 7  proporción de silencio (0-1)
    "melodic_peak_ratio",  # 8  posición de la nota más alta dentro del rango (0-1)
    "step_ratio",          # 9  proporción de movimientos por grado (<=2 semitonos)
    "leap_ratio",          # 10 proporción de saltos grandes (>=7 semitonos)
    "phrase_arch",         # 11 curvatura de frase: sube-y-baja(1) vs monótona(0)
    "climax_position",     # 12 posición temporal del clímax (0=inicio, 1=final)
    "note_duration_mean",  # 13 duración media de notas normalizada
    "pitch_entropy",       # 14 entropía de distribución de alturas (diversidad)
    "interval_entropy",    # 15 entropía de intervalos (variedad melódica)
]

DIM_LABELS_MELODIC = {
    "pitch_center":       ("grave", "agudo"),
    "pitch_range":        ("estrecho", "amplio"),
    "interval_mean":      ("stepwise", "saltos grandes"),
    "contour":            ("descendente", "ascendente"),
    "density":            ("sparse", "denso"),
    "rhythm_variance":    ("regular", "irregular"),
    "velocity_mean":      ("piano", "forte"),
    "silence_ratio":      ("continuo", "silencioso"),
    "melodic_peak_ratio": ("cima baja", "cima alta"),
    "step_ratio":         ("pocas notas conjuntas", "movimiento conjunto"),
    "leap_ratio":         ("sin saltos", "muchos saltos"),
    "phrase_arch":        ("línea recta", "arco de frase"),
    "climax_position":    ("clímax al inicio", "clímax al final"),
    "note_duration_mean": ("notas cortas", "notas largas"),
    "pitch_entropy":      ("tonal fijo", "tonal diverso"),
    "interval_entropy":   ("melódica repetitiva", "melódica variada"),
}

DIM_LABELS = {**DIM_LABELS_BASE, **DIM_LABELS_EXTENDED, **DIM_LABELS_MELODIC}


def vectorize_midi(path: str, extended: bool = False,
                   melodic: bool = False) -> dict:
    """
    Extrae vector de características de un MIDI. Solo mido + math, sin numpy.

    Modos:
      base     (default) — 10 features generales
      extended            — 20 features (base + temporales/armónicas)
      melodic             — 16 features optimizadas para melodía monofónica
    """
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

    base_vector = [
        pitch_center, pitch_range, interval_mean, contour,
        density, rhythm_variance, polyphony,
        velocity_mean, velocity_variance, silence_ratio,
    ]

    if not extended and not melodic:
        result["vector"] = base_vector
        return result

    # ── Features melódicas (melodic=True) ────────────────────────────
    if melodic:
        # 8. melodic_peak_ratio — altura de la nota más alta dentro del rango
        p_min, p_max = min(pitches), max(pitches)
        melodic_peak_ratio = (p_max - p_min) / 87.0 if p_max > p_min else 0.5

        # 9. step_ratio — proporción de movimientos por grado (≤2 semit.)
        if intervals:
            step_ratio = sum(1 for iv in intervals if iv <= 2) / len(intervals)
        else:
            step_ratio = 0.5

        # 10. leap_ratio — proporción de saltos grandes (≥7 semit.)
        if intervals:
            leap_ratio = sum(1 for iv in intervals if iv >= 7) / len(intervals)
        else:
            leap_ratio = 0.0

        # 11. phrase_arch — curvatura de frase media
        # Divide la melodía en frases de ~8 notas y mide si cada una
        # tiene forma de arco (sube luego baja o viceversa)
        phrase_size = 8
        arches = []
        for start in range(0, len(pitches) - phrase_size, phrase_size // 2):
            phrase = pitches[start:start + phrase_size]
            if len(phrase) < 4:
                continue
            mid = len(phrase) // 2
            first_half  = phrase[:mid]
            second_half = phrase[mid:]
            m1 = sum(first_half)  / len(first_half)
            m2 = sum(second_half) / len(second_half)
            center = sum(phrase) / len(phrase)
            # Arco: si el centro de la frase está por encima/debajo de ambos extremos
            extremes_mean = (phrase[0] + phrase[-1]) / 2
            arch = abs(center - extremes_mean) / max(p_max - p_min, 1)
            arches.append(min(arch, 1.0))
        phrase_arch = sum(arches) / len(arches) if arches else 0.0

        # 12. climax_position — posición temporal de la nota más alta
        max_pitch_val = max(pitches)
        max_indices   = [i for i, p in enumerate(pitches) if p == max_pitch_val]
        climax_idx    = max_indices[len(max_indices) // 2]
        climax_onset  = (sorted(onset_times)[climax_idx]
                         if climax_idx < len(onset_times) else total_dur / 2)
        climax_position = min(climax_onset / max(total_dur, 1.0), 1.0)

        # 13. note_duration_mean — duración media normalizada (sat. a 4s)
        durations = [n[3] for n in notes]
        note_duration_mean = min(sum(durations) / len(durations) / 4.0, 1.0)

        # 14. pitch_entropy — entropía de clases de altura (12 clases)
        pitch_classes = [0] * 12
        for p in pitches:
            pitch_classes[p % 12] += 1
        n_p = sum(pitch_classes) or 1
        pitch_entropy = 0.0
        for c in pitch_classes:
            if c > 0:
                p_i = c / n_p
                pitch_entropy -= p_i * math.log2(p_i)
        pitch_entropy = min(pitch_entropy / math.log2(12), 1.0)

        # 15. interval_entropy — entropía de intervalos (13 clases)
        interval_classes = [0] * 13
        for iv in intervals:
            interval_classes[min(iv, 12)] += 1
        n_iv = sum(interval_classes) or 1
        interval_entropy = 0.0
        for c in interval_classes:
            if c > 0:
                p_i = c / n_iv
                interval_entropy -= p_i * math.log2(p_i)
        interval_entropy = min(interval_entropy / math.log2(13), 1.0)

        result["vector"] = [
            pitch_center, pitch_range, interval_mean, contour,
            density, rhythm_variance, velocity_mean, silence_ratio,
            melodic_peak_ratio, step_ratio, leap_ratio, phrase_arch,
            climax_position, note_duration_mean, pitch_entropy, interval_entropy,
        ]
        return result

    # ── Features extendidas ──────────────────────────────────────────

    # 10. tension_arc — diferencia de "tensión" entre segunda y primera mitad
    # Proxy: densidad × rango de pitch en cada mitad
    half_t  = total_dur / 2
    first_n = [n for n in notes if n[0] < half_t]
    second_n= [n for n in notes if n[0] >= half_t]

    def _section_tension(ns):
        if not ns:
            return 0.0
        ps = [n[1] for n in ns]
        rng = (max(ps) - min(ps)) / 87.0
        den = min(len(ns) / max(half_t, 0.1) / 10.0, 1.0)
        return rng * den

    t1 = _section_tension(first_n)
    t2 = _section_tension(second_n)
    tension_arc = (math.tanh(t2 - t1) + 1) / 2  # 0=baja, 0.5=plana, 1=sube

    # 11. density_arc — cambio de densidad entre tercios
    third = total_dur / 3
    def _third_density(start, end):
        ns = [n for n in notes if start <= n[0] < end]
        return min(len(ns) / max(end - start, 0.1) / 10.0, 1.0)
    d1 = _third_density(0, third)
    d3 = _third_density(2*third, total_dur)
    density_arc = (math.tanh((d3 - d1) * 3) + 1) / 2

    # 12. pitch_entropy — entropía de distribución de clases de altura (12 clases)
    pitch_classes = [0] * 12
    for p in pitches:
        pitch_classes[p % 12] += 1
    n_p = sum(pitch_classes) or 1
    pitch_entropy = 0.0
    for c in pitch_classes:
        if c > 0:
            p_i = c / n_p
            pitch_entropy -= p_i * math.log2(p_i)
    pitch_entropy = min(pitch_entropy / math.log2(12), 1.0)  # normalizar a [0,1]

    # 13. interval_entropy — entropía de intervalos (clases 0-11 semítonos)
    interval_classes = [0] * 13
    for iv in intervals:
        interval_classes[min(iv, 12)] += 1
    n_iv = sum(interval_classes) or 1
    interval_entropy = 0.0
    for c in interval_classes:
        if c > 0:
            p_i = c / n_iv
            interval_entropy -= p_i * math.log2(p_i)
    interval_entropy = min(interval_entropy / math.log2(13), 1.0)

    # 14. repetition — similitud entre ventanas de 4 notas consecutivas
    # Cuenta cuántos n-gramas de 4 notas se repiten
    n_gram_size = 4
    if len(pitches) >= n_gram_size * 2:
        ngrams = []
        for i in range(len(pitches) - n_gram_size + 1):
            ngrams.append(tuple(pitches[i:i+n_gram_size]))
        unique = len(set(ngrams))
        total_ng = len(ngrams)
        repetition = 1.0 - (unique / total_ng)
    else:
        repetition = 0.0

    # 15. harmonic_tension — proporción de intervalos disonantes
    # Consonancias perfectas: 0, 5, 7, 12 semítonos
    consonant = {0, 5, 7, 12, 4, 3}
    if intervals:
        dissonant_count = sum(1 for iv in intervals if (iv % 12) not in consonant)
        harmonic_tension = dissonant_count / len(intervals)
    else:
        harmonic_tension = 0.0

    # 16. rhythmic_complexity — proporción de cambios de IOI >50%
    if len(iois) > 1:
        changes = sum(
            1 for i in range(len(iois)-1)
            if iois[i] > 0 and abs(iois[i+1] - iois[i]) / iois[i] > 0.5
        )
        rhythmic_complexity = changes / (len(iois) - 1)
    else:
        rhythmic_complexity = 0.0

    # 17. note_duration_mean — duración media normalizada (saturado a 4 segundos)
    durations = [n[3] for n in notes]
    note_duration_mean = min(sum(durations) / len(durations) / 4.0, 1.0)

    # 18. climax_position — posición temporal (0-1) de la nota más alta
    max_pitch   = max(pitches)
    max_indices = [i for i, p in enumerate(pitches) if p == max_pitch]
    climax_note_idx = max_indices[len(max_indices)//2]
    climax_onset    = sorted(onset_times)[climax_note_idx] if climax_note_idx < len(onset_times) else total_dur / 2
    climax_position = min(climax_onset / max(total_dur, 1.0), 1.0)

    # 19. dynamic_arc — diferencia de velocidad media entre mitades
    vel_first  = [n[2] for n in notes if n[0] < half_t]
    vel_second = [n[2] for n in notes if n[0] >= half_t]
    vm1 = sum(vel_first)  / len(vel_first)  if vel_first  else 64.0
    vm2 = sum(vel_second) / len(vel_second) if vel_second else 64.0
    dynamic_arc = (math.tanh((vm2 - vm1) / 32.0) + 1) / 2

    result["vector"] = base_vector + [
        tension_arc, density_arc, pitch_entropy, interval_entropy,
        repetition, harmonic_tension, rhythmic_complexity,
        note_duration_mean, climax_position, dynamic_arc,
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
        n = min(len(self.weights), len(vector))
        logit = sum(self.weights[i] * vector[i] for i in range(n)) + self.bias
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
        if len(self.weights) != len(vector):
            self.weights = self.weights[:len(vector)] + [0.0] * max(0, len(vector) - len(self.weights))
        for i in range(len(vector)):
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
        if len(self.weights) != len(winner_vec):
            self.weights = self.weights[:len(winner_vec)] + [0.0] * max(0, len(winner_vec) - len(self.weights))
        grad = 1.0 - prob_correct
        for i in range(len(winner_vec)):
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

    def calibrate(self, vectors: list):
        """
        Escala los pesos post-entrenamiento para que el rango efectivo
        de scores sobre 'vectors' sea aproximadamente [0.1, 0.9].

        El modelo lineal con SGD tiende a comprimir los scores cerca de
        0.5 cuando los datos son similares entre sí. Este paso corrige
        eso sin cambiar el ranking aprendido.
        """
        if not vectors or self.n_train < 5:
            return
        scores = [self.score(v) for v in vectors]
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        if rng < 1e-6:
            return  # todos iguales, no hay nada que calibrar
        # Escalar pesos y bias para mapear [lo, hi] → [0.1, 0.9]
        target_lo, target_hi = 0.1, 0.9
        # score = sigmoid(w·x + b)
        # Queremos sigmoid(k*(w·x + b) + b2) tal que los extremos mapeen
        # Aproximación: escalar los logits por factor k
        import math
        lo_logit = math.log(lo / (1 - lo)) if 0 < lo < 1 else -4.0
        hi_logit = math.log(hi / (1 - hi)) if 0 < hi < 1 else  4.0
        tl_logit = math.log(target_lo / (1 - target_lo))
        th_logit = math.log(target_hi / (1 - target_hi))
        if abs(hi_logit - lo_logit) < 1e-6:
            return
        scale  = (th_logit - tl_logit) / (hi_logit - lo_logit)
        offset = tl_logit - scale * lo_logit
        self.weights = [w * scale for w in self.weights]
        self.bias    = self.bias * scale + offset

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

    def annotated_distribution(self) -> list:
        """
        Devuelve lista de (path, pred_score, rating_norm) para todos los
        MIDIs del historial con rating conocido. Útil para anclar percentiles.
        """
        dist = []
        for entry in self.history:
            if entry.get("type") == "rating" and "rating" in entry:
                path       = entry["path"]
                rating_norm = (entry["rating"] - 1) / 4.0
                dist.append((path, rating_norm))
        return dist

    def top_dimensions(self, n: int = 5) -> list:
        """Devuelve las n dimensiones con mayor peso absoluto."""
        n_w = len(self.weights)
        if n_w == 16:
            all_names = DIM_NAMES_MELODIC
        elif n_w == 20:
            all_names = DIM_NAMES_BASE + DIM_NAMES_EXTENDED
        else:
            all_names = DIM_NAMES_BASE
        indexed = sorted(
            enumerate(self.weights),
            key=lambda x: abs(x[1]), reverse=True
        )
        return [(all_names[i] if i < len(all_names) else f"dim_{i}", w)
                for i, w in indexed[:n]]

    def summary(self) -> str:
        n_features = len(self.weights)
        if n_features == 16:
            feat_label = f"melódico ({n_features} dims)"
        elif n_features == 20:
            feat_label = f"extendido ({n_features} dims)"
        else:
            feat_label = f"base ({n_features} dims)"
        lines = []
        lines.append(f"  Ejemplos de entrenamiento: {bold(str(self.n_train))}")
        lines.append(f"  Modo de features:          {feat_label}")
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
    _label_1321 = f"{dur} · {bpm}bpm · {notes} notas"
    return f"{prefix}{cyan(name)}  {dim(_label_1321)}"


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
            _label_1371 = f"Predicción: {pred:.2f}  ·  incertidumbre: {unc:.2f} ("
            print(f"  {dim(_label_1371)}{unc_label}{dim(')')}")
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
            _label_1501 = f"Predicción: A={sa:.2f} (unc={ua:.2f})  B={sb:.2f} (unc={ub:.2f})"
            print(f"  {dim(_label_1501)}")
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
#  CONTEXTO DE EVALUACIÓN
# ════════════════════════════════════════════════════════════════════

def _percentile(score: float, reference_scores: list) -> float:
    """Percentil de score dentro de reference_scores (0-100)."""
    if not reference_scores:
        return 50.0
    below = sum(1 for s in reference_scores if s < score)
    return below / len(reference_scores) * 100.0


def _dimension_narrative(vector: list, weights: list) -> tuple:
    """
    Devuelve (favor, contra) como strings con las dimensiones que más
    empujan el score hacia arriba y hacia abajo, en lenguaje natural.
    """
    contribs = [(DIM_NAMES[i], weights[i] * vector[i], weights[i], vector[i])
                for i in range(N_DIMS)]
    contribs.sort(key=lambda x: x[1], reverse=True)

    favor  = []
    contra = []
    for name, contrib, w, val in contribs:
        lo, hi = DIM_LABELS[name]
        label  = hi if val > 0.5 else lo
        if contrib > 0.005:
            favor.append(label)
        elif contrib < -0.005:
            contra.append(label)

    favor_str  = ", ".join(favor[:3])  if favor  else "—"
    contra_str = ", ".join(contra[:3]) if contra else "—"
    return favor_str, contra_str


def _nearest_annotated(vector: list, model, n: int = 3) -> list:
    """
    Devuelve los n MIDIs más cercanos en el historial con rating conocido,
    como lista de (nombre, rating, distancia).
    """
    candidates = []
    seen_paths = set()
    for entry in reversed(model.history):
        if entry.get("type") != "rating" or "rating" not in entry:
            continue
        path = entry["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        ann = model.annotated_paths.get(path, [])
        # media de ratings anotados (pueden ser múltiples)
        ratings = [x for x in ann if isinstance(x, int)]
        if not ratings:
            ratings = [entry["rating"]]
        avg_rating = sum(ratings) / len(ratings)
        # No tenemos el vector guardado, usamos el score del modelo como proxy
        # de similitud si no podemos calcular distancia real
        candidates.append((path, avg_rating))

    # Sin vectores de referencia almacenados, devolvemos los más recientes
    # con rating extremo primero (más informativos)
    candidates.sort(key=lambda x: abs(x[1] - 2.5), reverse=True)
    result = []
    for path, avg_rating in candidates[:n]:
        result.append((Path(path).name, avg_rating))
    return result


def _nearest_annotated_with_vectors(vector: list, model,
                                     corpus_vectors: list,
                                     corpus_paths: list,
                                     n: int = 3) -> list:
    """
    Versión que usa vectores reales del corpus para calcular distancia.
    Devuelve (nombre, rating_medio, distancia) de los n más cercanos anotados.
    """
    annotated_indices = []
    for i, path in enumerate(corpus_paths):
        p = str(path)
        if p in model.annotated_paths:
            ann = model.annotated_paths[p]
            ratings = [x for x in ann if isinstance(x, int)]
            if ratings:
                avg_rating_raw = sum(ratings) / len(ratings)  # escala 1-5
                avg_rating_norm = (avg_rating_raw - 1) / 4.0  # normalizar a 0-1
                annotated_indices.append((i, avg_rating_norm))

    if not annotated_indices:
        return []

    scored = []
    for i, avg_r in annotated_indices:
        if avg_r is None:
            continue
        dist = _euclidean_distance(vector, corpus_vectors[i])
        # Convertir rating normalizado (0-1) a escala 1-5 para las estrellas
        # avg_r ya está en escala 1-5 (se guarda el rating original)
        scored.append((Path(str(corpus_paths[i])).name, avg_r, dist))

    scored.sort(key=lambda x: x[2])
    return scored[:n]


def _corpus_histogram(score: float, all_scores: list, width: int = 40) -> str:
    """
    Histograma ASCII de la distribución de scores del corpus con la
    posición del MIDI evaluado marcada con ▲.
    """
    if not all_scores:
        return ""

    n_bins  = 10
    bin_w   = 1.0 / n_bins
    counts  = [0] * n_bins
    for s in all_scores:
        b = min(int(s / bin_w), n_bins - 1)
        counts[b] += 1

    max_count = max(counts) or 1
    bar_h     = 6   # altura en líneas

    # Construir columnas verticales
    cols = []
    for c in counts:
        filled = round(c / max_count * bar_h)
        col = ["█"] * filled + [" "] * (bar_h - filled)
        col.reverse()   # índice 0 = arriba
        cols.append(col)

    # Marcador de posición
    marker_bin = min(int(score / bin_w), n_bins - 1)

    lines = []
    for row in range(bar_h):
        line = "  "
        for b, col in enumerate(cols):
            ch = col[row]
            if b == marker_bin and row == bar_h - 1:
                line += cyan("▓▓▓▓")
            else:
                line += (dim("░░░░") if ch == " " else "████")
        lines.append(line)

    # Eje X con labels cada 2 bins (4 chars por bin)
    axis = "  "
    for b in range(n_bins):
        if b % 2 == 0:
            label = f"{b/n_bins:.1f}"
            axis += label + " " * (4 - len(label))
        else:
            axis += "    "

    # Marcador ▲ centrado en el bin
    marker_pos = 2 + marker_bin * 4 + 1
    marker_row = " " * marker_pos + cyan("▲")

    lines.append(axis)
    lines.append(marker_row)
    return "\n".join(lines)


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

    # ── Cargar corpus de referencia (para percentil, vecinos, histograma) ──
    corpus_vectors = []
    corpus_paths   = []
    if getattr(args, "corpus", None):
        corpus_path = args.corpus
        if Path(corpus_path).exists():
            try:
                corpus_vectors, corpus_paths, _ = _load_corpus_npz(corpus_path)
                print(f"  Corpus: {cyan(corpus_path)}  ({len(corpus_vectors):,} MIDIs)")
            except Exception as e:
                print(yellow(f"  No se pudo cargar corpus: {e}"))
        else:
            print(yellow(f"  Corpus no encontrado: {corpus_path}"))

    # Calibrar el modelo sobre los vectores disponibles para expandir el rango
    if corpus_vectors:
        model.calibrate(corpus_vectors)
    elif model.n_train >= 5:
        # Sin corpus, calibrar sobre los vectores del historial si hay suficientes
        hist_vecs = []
        seen = set()
        for entry in model.history:
            p = entry.get("path","")
            if p and p not in seen:
                seen.add(p)
                r = vectorize_midi(p)
                if r["vector"]:
                    hist_vecs.append(r["vector"])
        if hist_vecs:
            model.calibrate(hist_vecs)

    # Scores del corpus completo para histograma y percentil
    corpus_scores = [model.score(v) for v in corpus_vectors] if corpus_vectors else []

    # Scores del historial anotado para percentil alternativo
    annotated_dist = model.annotated_distribution()   # [(path, rating_norm)]
    annotated_scores_norm = [r for _, r in annotated_dist]

    # Detect if model was trained with extended features
    _n_weights = len(model.weights) if model.weights else 10
    _use_extended_eval = _n_weights == 20
    _use_melodic_eval  = _n_weights == 16

    # ── Cargar vectores de afinidad ──────────────────────────────────
    affinity_vectors = []
    if args.affinity:
        affinity_dir = Path(args.affinity)
        if not affinity_dir.exists():
            print(red(f"  Carpeta de afinidad no encontrada: {affinity_dir}"))
        else:
            aff_files = (list(affinity_dir.rglob("*.mid")) +
                         list(affinity_dir.rglob("*.midi")))
            print(f"  Cargando {len(aff_files)} obras afines de {cyan(str(affinity_dir))}...")
            for mf in aff_files:
                res = vectorize_midi(str(mf),
                                     extended=_use_extended_eval,
                                     melodic=_use_melodic_eval)
                if res["vector"]:
                    affinity_vectors.append(res["vector"])
            print(f"  {green(str(len(affinity_vectors)))} vectorizadas correctamente.")

    weight_pref = 1.0 - args.weight if affinity_vectors else 1.0
    weight_aff  = args.weight        if affinity_vectors else 0.0

    # ── Recoger archivos a evaluar ───────────────────────────────────
    midi_files = []
    for pattern in args.midi:
        p = Path(pattern)
        if p.is_dir():
            midi_files += list(p.rglob("*.mid")) + list(p.rglob("*.midi"))
        elif p.exists():
            midi_files.append(p)
        else:
            import glob
            midi_files += [Path(f) for f in glob.glob(pattern)]

    if not midi_files:
        print(red("  No se encontraron archivos MIDI."))
        sys.exit(1)

    results = []
    print(f"\n{'─'*60}")

    for mf in midi_files:
        res = vectorize_midi(str(mf),
                             extended=_use_extended_eval,
                             melodic=_use_melodic_eval)
        if res["error"]:
            err_msg = res["error"]
            print("  " + yellow("!") + f"  {mf.name}  " + dim("error: " + err_msg))
            continue

        v          = res["vector"]
        pref_score = model.score(v)

        aff_score = 0.0
        if affinity_vectors:
            sims      = [_cosine_similarity(v, av) for av in affinity_vectors]
            aff_score = (sum(sims) / len(sims) + 1) / 2

        final_score = (weight_pref * pref_score + weight_aff * aff_score
                       if affinity_vectors else pref_score)

        # ── Contexto ────────────────────────────────────────────────
        # Percentil: sobre corpus si disponible, si no sobre anotaciones
        if corpus_scores:
            pct = _percentile(final_score, corpus_scores)
            pct_label = f"percentil {pct:.0f}% del corpus ({len(corpus_scores):,} MIDIs)"
        elif annotated_scores_norm:
            # Convertir ratings normalizados a scores del modelo aproximados
            pct = _percentile(final_score, annotated_scores_norm)
            pct_label = f"percentil {pct:.0f}% de lo anotado ({len(annotated_scores_norm)} ejemplos)"
        else:
            pct        = None
            pct_label  = "sin referencia (entrena más rondas)"

        # Vecinos anotados
        if corpus_vectors:
            neighbors = _nearest_annotated_with_vectors(
                v, model, corpus_vectors, corpus_paths, n=3)
        else:
            neighbors = _nearest_annotated(v, model, n=3)

        # Desglose dimensional narrativo
        favor_str, contra_str = _dimension_narrative(v, model.weights)

        results.append({
            "path":        str(mf),
            "name":        mf.name,
            "pref_score":  pref_score,
            "aff_score":   aff_score,
            "final_score": final_score,
            "vector":      v,
            "meta":        res,
            "pct":         pct,
            "pct_label":   pct_label,
            "neighbors":   neighbors,
            "favor":       favor_str,
            "contra":      contra_str,
        })

    if not results:
        print(red("  Ningún MIDI pudo ser procesado."))
        sys.exit(1)

    results.sort(key=lambda r: r["final_score"], reverse=True)

    # ── Salida ───────────────────────────────────────────────────────

    print(f"\n{'═'*60}")
    if affinity_vectors:
        print(f"  {bold('RESULTADOS')}  "
              f"[pref×{weight_pref:.2f} + afinidad×{weight_aff:.2f}]")
    else:
        print(f"  {bold('RESULTADOS')}  [solo preferencia]")
    print(f"{'═'*60}\n")

    for rank, r in enumerate(results, 1):
        score_color = green if r["final_score"] >= 0.65 else (
            yellow if r["final_score"] >= 0.40 else red)
        dur  = format_duration(r["meta"]["duration_s"])
        bpm  = r["meta"]["tempo_bpm"]

        print(f"  {rank:>2}. {bold(cyan(r['name']))}")
        _label_1909 = f"{dur} · {bpm} bpm"
        print(f"      {dim(_label_1909)}")
        print()

        # ── Score principal ──────────────────────────────────────────
        bar_len = int(r["final_score"] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        fs      = f"{r['final_score']:.4f}"
        print(f"      {score_color(bar)}  {bold(fs)}")

        if affinity_vectors:
            _aff_label = f"preferencia={r['pref_score']:.4f}  ·  afinidad={r['aff_score']:.4f}"
            print(f"      {dim(_aff_label)}")
        print()

        # ── Percentil ────────────────────────────────────────────────
        if r["pct"] is not None:
            pct_bar_w = 30
            pct_filled = int(r["pct"] / 100 * pct_bar_w)
            pct_bar    = dim("─" * pct_filled) + cyan("┼") + dim("─" * (pct_bar_w - pct_filled))
            pct_color  = green if r["pct"] >= 70 else (yellow if r["pct"] >= 40 else red)
            _label_1929 = f'{r["pct"]:.0f}%'
            print(f"      {dim('Posición:')}  {pct_bar}  {pct_color(_label_1929)}")
            print(f"      {dim(r['pct_label'])}")
            print()

        # ── Histograma del corpus ─────────────────────────────────────
        if corpus_scores:
            print(f"      {dim('Distribución del corpus (▲ = esta pieza):')}")
            hist = _corpus_histogram(r["final_score"], corpus_scores)
            for line in hist.split("\n"):
                print(f"    {line}")
            print()

        # ── Dimensiones que favorecen / perjudican ────────────────────
        print(f"      {green('+ favorece:')}  {r['favor']}")
        print(f"      {red('- perjudica:')} {r['contra']}")
        print()

        # ── Vecinos anotados ─────────────────────────────────────────
        if r["neighbors"]:
            print(f"      {dim('Más parecido a lo que ya anotaste:')}")
            for nb in r["neighbors"]:
                if len(nb) == 3:
                    nb_name, nb_rating, nb_dist = nb
                    stars = "★" * round(nb_rating * 4 + 1) + "☆" * (5 - round(nb_rating * 4 + 1))
                    _label_1953 = f"(dist={nb_dist:.3f})"
                    print(f"        {stars}  {dim(nb_name)}  {dim(_label_1953)}")
                else:
                    nb_name, nb_rating = nb
                    n_stars = max(1, min(5, round(nb_rating * 4 + 1)))
                    stars = "★" * n_stars + "☆" * (5 - n_stars)
                    print(f"        {stars}  {dim(nb_name)}")
            print()

        # ── Desglose dimensional completo (--verbose) ─────────────────
        if args.verbose:
            print(f"      {dim('─ Detalle de dimensiones:')}")
            dims_sorted = sorted(range(N_DIMS),
                                 key=lambda i: abs(model.weights[i] * r["vector"][i]),
                                 reverse=True)
            for i in dims_sorted:
                val     = r["vector"][i]
                w       = model.weights[i]
                lo, hi  = DIM_LABELS[i] if isinstance(DIM_LABELS, list) else DIM_LABELS[DIM_NAMES[i]]
                bar2    = "▮" * int(val * 10) + "▯" * (10 - int(val * 10))
                contrib = w * val
                sign    = "+" if contrib >= 0 else ""
                c_color = green if contrib > 0.01 else (red if contrib < -0.01 else dim)
                _label_1976 = f"{sign}{contrib:.4f}"
                print(f"        {DIM_NAMES[i]:<20} {bar2}  {val:.2f}  "
                      f"{c_color(_label_1976)}")
            print()

        print(f"  {dim('─'*56)}")
        print()

    # ── Resumen numérico (formato mínimo garantizado) ─────────────────
    print(f"  {bold('RESUMEN NUMÉRICO')}")
    print()
    for r in results:
        pct_str = f"  pct={r['pct']:.0f}%" if r["pct"] is not None else ""
        if affinity_vectors:
            print(f"  {r['name']:<40}  score={r['final_score']:.4f}  "
                  f"pref={r['pref_score']:.4f}  aff={r['aff_score']:.4f}{pct_str}")
        else:
            print(f"  {r['name']:<40}  score={r['final_score']:.4f}{pct_str}")
    print()

    # JSON machine-readable
    if getattr(args, "json_out", False):
        output = []
        for r in results:
            entry = {
                "path":        r["path"],
                "score":       round(r["final_score"], 4),
                "pref_score":  round(r["pref_score"], 4),
                "percentile":  round(r["pct"], 1) if r["pct"] is not None else None,
                "favor":       r["favor"],
                "contra":      r["contra"],
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

    # Calibrar y guardar modelo
    model.calibrate(vectors)
    model.save(model_file)
    print(f"\n{green('✓')} Modelo guardado: {cyan(model_file)}")
    print(f"  Total de ejemplos de entrenamiento: {bold(str(model.n_train))}")
    print()
    print(f"  {bold('─ Estado final del modelo ─')}")
    print(model.summary())
    print()
    print(f"  Para evaluar un MIDI:")
    _label_2083 = f"python preference_trainer.py eval tu_pieza.mid --model {args.model}"
    print(f"    {dim(_label_2083)}")


# ════════════════════════════════════════════════════════════════════
#  COMANDO INFO — inspeccionar el modelo
# ════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════
#  COMANDO SET — anotar preferencia directamente
# ════════════════════════════════════════════════════════════════════

def cmd_set(args):
    """Asigna un score de preferencia directamente a uno o varios MIDIs."""

    model_file = _model_path(args.model)
    if not Path(model_file).exists():
        print(red(f"Modelo no encontrado: {model_file}"))
        print(yellow(f"  Crea uno primero con: python preference_trainer.py train <corpus.npz>"))
        sys.exit(1)

    model = PreferenceModel.load(model_file)

    # Recoger archivos
    midi_files = []
    for pattern in args.midi:
        p = Path(pattern)
        if p.is_dir():
            midi_files += sorted(p.rglob("*.mid")) + sorted(p.rglob("*.midi"))
        elif p.exists():
            midi_files.append(p)
        else:
            import glob as _glob
            midi_files += [Path(f) for f in sorted(_glob.glob(pattern))]

    if not midi_files:
        print(red("  No se encontraron archivos MIDI."))
        sys.exit(1)

    rating  = args.score
    reps    = getattr(args, 'repetitions', 10)
    _n_w_set = len(model.weights) if model.weights else 10
    use_ext  = _n_w_set == 20
    use_mel  = _n_w_set == 16

    print(f"\n{bold('PREFERENCE TRAINER — SET')}")
    print(f"  Modelo: {cyan(model_file)}  ({model.n_train} ejemplos previos)")
    print(f"  Score:  {'★' * rating + '☆' * (5 - rating)}  ({rating}/5)")
    print()

    updated = 0
    for mf in midi_files:
        res = vectorize_midi(str(mf), extended=use_ext, melodic=use_mel)
        if res["error"]:
            print(f"  {yellow('!')} {mf.name}  {dim('error: ' + res['error'])}")
            continue

        v    = res["vector"]
        path = str(mf)

        pred_before = model.score(v)
        n_prev      = model.annotation_count(path)

        for _ in range(reps):
            model.update_rating(v, rating, path=path)
        model.history.append({"type": "rating", "path": path, "rating": rating})

        pred_after = model.score(v)
        delta      = pred_after - pred_before
        delta_str  = (green(f"+{delta:.3f}") if delta > 0.001
                      else red(f"{delta:.3f}") if delta < -0.001
                      else dim(f"{delta:+.3f}"))

        seen_tag = dim(f"  (era {n_prev}x anotado)") if n_prev > 0 else ""
        print(f"  {green('✓')} {cyan(mf.name)}{seen_tag}")
        print(f"     score: {pred_before:.3f} → {pred_after:.3f}  Δ{delta_str}")
        updated += 1

    if not updated:
        print(red("  Ningún MIDI pudo ser procesado."))
        sys.exit(1)

    model.save(model_file)
    print(f"\n  {green('✓')} {updated} anotación(es) guardada(s) en {cyan(model_file)}")
    print(f"  Total ejemplos: {bold(str(model.n_train))}")

# ════════════════════════════════════════════════════════════════════
#  COMANDO VALIDATE — validación cruzada
# ════════════════════════════════════════════════════════════════════

def cmd_validate(args):
    """
    Validación cruzada k-fold sobre el historial de anotaciones.

    Mide qué tan bien generaliza el modelo aprendido:
    - MAE (error absoluto medio en predicción de rating)
    - Accuracy de ranking (pares ordenados correctamente)
    - Correlación de Spearman entre predicciones y ratings reales
    """
    model_file = _model_path(args.model)
    if not Path(model_file).exists():
        print(red(f"Modelo no encontrado: {model_file}"))
        sys.exit(1)

    model_ref = PreferenceModel.load(model_file)

    # Recoger ejemplos de rating con vectores disponibles
    corpus_vectors = {}
    if getattr(args, "corpus", None) and Path(args.corpus).exists():
        vecs, paths, _ = _load_corpus_npz(args.corpus)
        for v, p in zip(vecs, paths):
            corpus_vectors[str(p)] = v

    # Construir dataset desde historial
    dataset = []
    seen = set()
    for entry in model_ref.history:
        if entry.get("type") != "rating" or "rating" not in entry:
            continue
        path = entry["path"]
        if path in seen:
            continue
        seen.add(path)
        ratings = [x for x in model_ref.annotated_paths.get(path, [])
                   if isinstance(x, int)]
        if not ratings:
            ratings = [entry["rating"]]
        avg_r = sum(ratings) / len(ratings)

        # Obtener vector
        if path in corpus_vectors:
            v = corpus_vectors[path]
        else:
            res = vectorize_midi(path,
                extended=len(model_ref.weights) > 10)
            if res["error"] or not res["vector"]:
                continue
            v = res["vector"]

        dataset.append({"path": path, "vector": v, "rating": avg_r})

    n = len(dataset)
    if n < 4:
        print(red(f"  Historial insuficiente: {n} ejemplos con vector disponible."))
        print(yellow("  Se necesitan al menos 4. Entrena más rondas."))
        sys.exit(1)

    print(f"\n{bold('PREFERENCE TRAINER — VALIDATE')}")
    print(f"  Modelo:   {cyan(model_file)}")
    print(f"  Ejemplos: {n}  (con vector recuperable)")

    k = min(args.folds, n)
    print(f"  Folds:    {k}-fold\n")

    # K-fold estratificado (ordenar por rating y distribuir)
    dataset_sorted = sorted(dataset, key=lambda x: x["rating"])
    folds = [[] for _ in range(k)]
    for i, item in enumerate(dataset_sorted):
        folds[i % k].append(item)

    fold_results = []
    all_pred, all_true = [], []

    for fold_i in range(k):
        test  = folds[fold_i]
        train = [item for j, fold in enumerate(folds) if j != fold_i for item in fold]

        if not train:
            continue

        # Entrenar modelo limpio en este fold
        m = PreferenceModel()
        m.lr = 0.3
        for _ in range(30):
            random.shuffle(train)
            for item in train:
                m.update_rating(item["vector"], round(item["rating"]))
        m.calibrate([item["vector"] for item in train])

        # Evaluar en test
        maes, corrects, total_pairs = [], [], 0
        preds_fold, trues_fold = [], []

        for item in test:
            pred = m.score(item["vector"])
            true_norm = (item["rating"] - 1) / 4.0
            maes.append(abs(pred - true_norm))
            preds_fold.append(pred)
            trues_fold.append(true_norm)
            all_pred.append(pred)
            all_true.append(true_norm)

        # Pair accuracy en test
        correct_pairs = 0
        pair_total    = 0
        for ii in range(len(test)):
            for jj in range(ii+1, len(test)):
                ra = (test[ii]["rating"] - 1) / 4.0
                rb = (test[jj]["rating"] - 1) / 4.0
                if abs(ra - rb) < 0.01:
                    continue  # empate, no cuenta
                pa = m.score(test[ii]["vector"])
                pb = m.score(test[jj]["vector"])
                if (ra > rb and pa > pb) or (ra < rb and pa < pb):
                    correct_pairs += 1
                pair_total += 1

        mae   = sum(maes) / len(maes) if maes else 1.0
        p_acc = correct_pairs / pair_total if pair_total > 0 else 0.0
        fold_results.append({"fold": fold_i+1, "mae": mae, "pair_acc": p_acc,
                              "n_test": len(test)})

    # ── Métricas globales ────────────────────────────────────────────
    mean_mae  = sum(r["mae"]      for r in fold_results) / len(fold_results)
    mean_pacc = sum(r["pair_acc"] for r in fold_results) / len(fold_results)

    # Spearman global
    n_all  = len(all_pred)
    def _rank_list(lst):
        s = sorted(range(len(lst)), key=lambda i: lst[i])
        r = [0] * len(lst)
        for rank, idx in enumerate(s):
            r[idx] = rank + 1
        return r
    pred_ranks = _rank_list(all_pred)
    true_ranks = _rank_list(all_true)
    ds = [pred_ranks[i] - true_ranks[i] for i in range(n_all)]
    spearman = 1 - 6*sum(d**2 for d in ds) / (n_all*(n_all**2-1)) if n_all > 2 else 0.0

    # ── Salida ───────────────────────────────────────────────────────
    print(f"  {'Fold':<6} {'MAE':>6}  {'Pair Acc':>9}  {'n_test':>6}")
    print(f"  {'─'*36}")
    for r in fold_results:
        mae_color   = green if r["mae"]      < 0.15 else (yellow if r["mae"]      < 0.25 else red)
        pacc_color  = green if r["pair_acc"] > 0.75 else (yellow if r["pair_acc"] > 0.60 else red)
        _mae_2319 = f'{r["mae"]:.3f}'
        _pacc_2319 = f'{r["pair_acc"]*100:.1f}%'
        print(f"  {r['fold']:<6} {mae_color(_mae_2319):>6}  "
              f"{pacc_color(_pacc_2319):>9}  {r['n_test']:>6}")

    print(f"  {'─'*36}")
    mae_c  = green if mean_mae  < 0.15 else (yellow if mean_mae  < 0.25 else red)
    pacc_c = green if mean_pacc > 0.75 else (yellow if mean_pacc > 0.60 else red)
    sp_c   = green if spearman  > 0.70 else (yellow if spearman  > 0.50 else red)
    _mae_2326 = f"{mean_mae:.3f}"
    _pacc_2326 = f"{mean_pacc*100:.1f}%"
    print(f"  {'Media':<6} {mae_c(_mae_2326):>6}  {pacc_c(_pacc_2326):>9}")
    print()
    _sp_2328 = f"{spearman:.3f}"
    print(f"  Correlación de Spearman (global): {sp_c(_sp_2328)}")
    print()

    # ── Interpretación ───────────────────────────────────────────────
    print(f"  {bold('Interpretación:')}")

    if mean_mae < 0.10:
        print(f"  {green('●')} MAE excelente — el modelo predice el rating con alta precisión.")
    elif mean_mae < 0.20:
        print(f"  {yellow('●')} MAE aceptable — margen de ~1 estrella en promedio.")
    else:
        print(f"  {red('●')} MAE alto — el modelo generaliza mal. Entrena más rondas o")
        print(f"              revisa si los ratings son consistentes.")

    if mean_pacc > 0.75:
        print(f"  {green('●')} Pair accuracy alta — el modelo ordena bien tus preferencias.")
    elif mean_pacc > 0.60:
        print(f"  {yellow('●')} Pair accuracy moderada — el orden es mayormente correcto.")
    else:
        print(f"  {red('●')} Pair accuracy baja — el modelo no discrimina bien.")
        print(f"              Considera más rondas de contraste (--mode contrast).")

    if spearman > 0.70:
        print(f"  {green('●')} Ranking sólido — las predicciones reflejan fielmente tu gusto.")
    elif spearman > 0.45:
        print(f"  {yellow('●')} Ranking parcial — captura tendencias pero no casos límite.")
    else:
        print(f"  {red('●')} Ranking débil — necesitas más datos o más variedad en el corpus.")

    print()
    if n < 20:
        _label_2359 = f"Nota: {n} ejemplos es poco para validación fiable."
        print(f"  {dim(_label_2359)}")
        print(f"  {dim('Los resultados mejorarán con más sesiones de entrenamiento.')}")

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

        r = vectorize_midi(str(path),
                           extended=getattr(args, 'extended_features', False),
                           melodic=getattr(args, 'melodic_features', False))

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
        if getattr(args, 'melodic_features', False):
            expected_dims = 16
        elif getattr(args, 'extended_features', False):
            expected_dims = 20
        else:
            expected_dims = 10
        if v is not None and len(v) == expected_dims and all(x == x for x in v):
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

    if getattr(args, 'melodic_features', False):
        actual_dim_names = DIM_NAMES_MELODIC
    elif getattr(args, 'extended_features', False):
        actual_dim_names = DIM_NAMES_BASE + DIM_NAMES_EXTENDED
    else:
        actual_dim_names = DIM_NAMES_BASE
    np.savez_compressed(
        str(output),
        vectors   = arr_vectors,
        paths     = arr_paths,
        meta      = arr_meta,
        dim_names = np.array(actual_dim_names),
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




def _rank_interactive(args, model, entries, model_file):
    """
    Sesión interactiva post-rank: navegar el ranking, escuchar y anotar
    MIDIs sin salir del comando rank.
    """
    soundfont = getattr(args, "soundfont", None)

    # Construir lista navegable: top + inciertas (sin duplicados)
    top_n    = sorted(entries, key=lambda e: e["score"],  reverse=True)
    unc_n    = sorted(entries, key=lambda e: e["unc"],    reverse=True)
    # Intercalar: primero los más inciertos no anotados, luego el top
    queue = []
    seen_paths = set()
    for e in unc_n:
        if e["n_seen"] == 0 and e["path"] not in seen_paths:
            queue.append(e)
            seen_paths.add(e["path"])
    for e in top_n:
        if e["path"] not in seen_paths:
            queue.append(e)
            seen_paths.add(e["path"])

    _label_2510 = f"{len(queue)} MIDIs en cola"
    print(f"\n{bold('─ MODO INTERACTIVO ─')}  "
          f"{dim(_label_2510)}")
    print(dim("  Puntúa [1-5] · p=reproducir · x=parar · s=saltar · q=guardar y salir"))
    print()

    mode_str = getattr(args, 'mode', 'rating')
    done = 0

    for e in queue:
        v    = e["vector"]
        path = e["path"]
        name = e["name"]

        seen_tag = dim(f"  [{e['n_seen']}x anotado]") if e["n_seen"] > 0 else ""
        unc_col  = green if e["unc"] > 0.7 else (yellow if e["unc"] > 0.3 else dim)

        print(f"  {'─'*52}")
        print(f"  {cyan(name)}{seen_tag}")
        _sc_2528 = f'{e["score"]:.3f}'
        _uc_2528 = f'{e["unc"]:.2f}'
        print(f"  {dim(format_duration(e['dur']))} · {dim(str(e['bpm']))} bpm  "
              f"score={bold(_sc_2528)}  "
              f"unc={unc_col(_uc_2528)}")
        print()

        # Reproducción automática
        _play_midi(path, soundfont=soundfont, blocking=False)

        rating = None
        while rating is None:
            raw = input("  [1-5 / p=reproducir / x=parar / s=saltar / q=salir]: ").strip().lower()
            if raw in ("q", "quit"):
                _stop_playback()
                break
            elif raw in ("p", "play"):
                _play_midi(path, soundfont=soundfont, blocking=False)
            elif raw in ("x", "stop"):
                _stop_playback()
            elif raw in ("s", "skip", ""):
                rating = "skip"
            elif raw.isdigit() and 1 <= int(raw) <= 5:
                rating = int(raw)
                _stop_playback()
            else:
                print(red("  Introduce 1-5, p, x, s o q."))

        if raw in ("q", "quit"):
            break

        if rating == "skip":
            print(dim("  Saltado.\n"))
            continue

        model.update_rating(v, rating, path=path)
        model.history.append({"type": "rating", "path": path, "rating": rating})
        e["n_seen"] += 1

        stars = "★" * rating + "☆" * (5 - rating)
        print(green(f"  {stars}  Anotado.\n"))
        done += 1

        if done % 5 == 0:
            print(f"  {bold('─ Estado del modelo ─')}")
            print(model.summary())
            print()

    _stop_playback()

    if done > 0:
        model.calibrate([e["vector"] for e in entries])
        model.save(model_file)
        print(f"\n{green('✓')} {done} anotaciones guardadas en {cyan(model_file)}")
    else:
        print(dim("  Sin cambios."))

# ════════════════════════════════════════════════════════════════════
#  COMANDO RANK — escanear corpus completo
# ════════════════════════════════════════════════════════════════════

def cmd_rank(args):
    """Escanea el corpus completo y devuelve top/bottom/inciertas."""

    model_file = _model_path(args.model)
    if not Path(model_file).exists():
        print(red(f"Modelo no encontrado: {model_file}"))
        sys.exit(1)
    if not Path(args.corpus).exists():
        print(red(f"Corpus no encontrado: {args.corpus}"))
        sys.exit(1)

    model = PreferenceModel.load(model_file)

    print(f"\n{bold('PREFERENCE TRAINER — RANK')}")
    print(f"  Modelo: {cyan(model_file)}  ({model.n_train} ejemplos de entrenamiento)")
    print(f"  Corpus: {cyan(args.corpus)}")

    vectors, paths, meta = _load_corpus_npz(args.corpus)
    n_total = len(vectors)
    print(f"  Total:  {n_total:,} MIDIs")

    if n_total == 0:
        print(red("  Corpus vacío."))
        sys.exit(1)

    # ── Cargar vectores de afinidad ──────────────────────────────────
    affinity_vectors = []
    weight_pref = 1.0
    weight_aff  = 0.0
    if getattr(args, "affinity", None):
        affinity_dir = Path(args.affinity)
        if not affinity_dir.exists():
            print(red(f"  Carpeta de afinidad no encontrada: {affinity_dir}"))
        else:
            aff_files = (list(affinity_dir.rglob("*.mid")) +
                         list(affinity_dir.rglob("*.midi")))
            print(f"  Cargando {len(aff_files)} obras afines de {cyan(str(affinity_dir))}...",
                  end=" ", flush=True)
            _n_w_rank = len(model.weights) if model.weights else 10
            _use_ext_rank = _n_w_rank == 20
            _use_mel_rank = _n_w_rank == 16
            for mf in aff_files:
                res = vectorize_midi(str(mf),
                                     extended=_use_ext_rank,
                                     melodic=_use_mel_rank)
                if res["vector"]:
                    affinity_vectors.append(res["vector"])
            print(green(f"{len(affinity_vectors)} OK"))
            weight_pref = 1.0 - args.weight
            weight_aff  = args.weight
            print(f"  Ponderación: preferencia×{weight_pref:.2f} + afinidad×{weight_aff:.2f}")

    # Calibrar el modelo sobre el corpus completo
    model.calibrate(vectors)

    # Calcular score e incertidumbre para cada MIDI
    print(f"  Calculando scores...", end=" ", flush=True)
    t0 = time.time()
    entries = []
    for i, (v, p, m) in enumerate(zip(vectors, paths, meta)):
        pref_score = model.score(v)
        unc        = model.uncertainty(v)

        aff_score = 0.0
        if affinity_vectors:
            sims      = [_cosine_similarity(v, av) for av in affinity_vectors]
            aff_score = (sum(sims) / len(sims) + 1) / 2

        score = (weight_pref * pref_score + weight_aff * aff_score
                 if affinity_vectors else pref_score)

        try:
            md = json.loads(str(m))
        except Exception:
            md = {}
        n_seen = model.annotation_count(str(p))
        entries.append({
            "idx":        i,
            "path":       str(p),
            "name":       Path(str(p)).name,
            "score":      score,
            "pref_score": pref_score,
            "aff_score":  aff_score,
            "unc":        unc,
            "dur":        md.get("duration_s", 0),
            "bpm":        md.get("tempo_bpm", "?"),
            "n_seen":     n_seen,
            "vector":     v,
        })
    print(green(f"{time.time()-t0:.1f}s"))

    # ── Filtrar anotados si se pide ──────────────────────────────────
    if getattr(args, 'exclude_annotated', False):
        n_before = len(entries)
        entries  = [e for e in entries if e["n_seen"] == 0]
        n_after  = len(entries)
        excluded = n_before - n_after
        if excluded:
            _label_2684 = f"Excluidos {excluded} MIDIs ya anotados → quedan {n_after:,}"
            print(f"  {dim(_label_2684)}")
        if not entries:
            print(yellow("  Todos los MIDIs del corpus ya han sido anotados."))
            return

    n     = min(args.top, len(entries))
    top   = sorted(entries, key=lambda e: e["score"],  reverse=True)[:n]
    bot   = sorted(entries, key=lambda e: e["score"])[:n]
    unc_l = sorted(entries, key=lambda e: e["unc"],    reverse=True)[:n]

    # ── helpers de display ────────────────────────────────────────────
    def _score_bar(score, width=25):
        filled = int(score * width)
        bar    = "█" * filled + "░" * (width - filled)
        color  = green if score >= 0.65 else (yellow if score >= 0.40 else red)
        return color(bar)

    def _entry_line(rank, e, show_unc=False):
        seen_tag = dim(f" [{e['n_seen']}x]") if e["n_seen"] > 0 else ""
        dur      = format_duration(e["dur"])
        _unc_2704 = f'{e["unc"]:.2f}'
        extra    = (f"  {dim(f'unc={_unc_2704}')}" if show_unc else "")
        aff_tag  = (dim(f"  pref={e['pref_score']:.3f} aff={e['aff_score']:.3f}")
                    if affinity_vectors else "")
        _sc_2727 = f'{e["score"]:.3f}'
        print(f"  {rank:>4}.  {_score_bar(e['score'])}  "
              f"{bold(_sc_2727)}  "
              f"{cyan(e['name'])}{seen_tag}")
        _label_2710 = f'{dur} · {e["bpm"]} bpm'
        print(f"         {dim(_label_2710)}{extra}{aff_tag}")

    # ── Estadísticas globales ─────────────────────────────────────────
    all_scores = [e["score"] for e in entries]
    mean_score = sum(all_scores) / len(all_scores)
    sorted_s   = sorted(all_scores)
    median_s   = sorted_s[len(sorted_s)//2]
    p25        = sorted_s[int(len(sorted_s)*0.25)]
    p75        = sorted_s[int(len(sorted_s)*0.75)]
    n_high     = sum(1 for s in all_scores if s >= 0.65)
    n_low      = sum(1 for s in all_scores if s <= 0.35)
    n_mid      = n_total - n_high - n_low

    print(f"\n{bold('─ Distribución global ─')}")
    hist_full = _corpus_histogram(mean_score, all_scores)
    for line in hist_full.split("\n"):
        print(f"  {line}")
    print()
    print(f"  Media:   {mean_score:.3f}   Mediana: {median_s:.3f}")
    print(f"  P25:     {p25:.3f}          P75:     {p75:.3f}")
    print(f"  Alta preferencia (≥0.65): {green(str(n_high)):>6}  ({n_high/n_total*100:.1f}%)")
    print(f"  Indiferente    (0.35-0.65):{yellow(str(n_mid)):>6}  ({n_mid/n_total*100:.1f}%)")
    print(f"  Baja preferencia (≤0.35): {red(str(n_low)):>6}  ({n_low/n_total*100:.1f}%)")

    # ── TOP ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    _label_2736 = f"TOP {n} — más preferidas"
    print(f"  {bold(green(_label_2736))}")
    print(f"{'═'*60}")
    for rank, e in enumerate(top, 1):
        _entry_line(rank, e)
    print()

    # ── BOTTOM ────────────────────────────────────────────────────────
    print(f"{'═'*60}")
    _label_2744 = f"BOTTOM {n} — menos preferidas"
    print(f"  {bold(red(_label_2744))}")
    print(f"{'═'*60}")
    for rank, e in enumerate(bot, 1):
        _entry_line(rank, e)
    print()

    # ── INCIERTAS ─────────────────────────────────────────────────────
    print(f"{'═'*60}")
    _label_2752 = f"INCIERTAS {n} — mayor incertidumbre"
    print(f"  {bold(yellow(_label_2752))}")
    print(f"  {dim('(donde más ganarías entrenando)')}")
    print(f"{'═'*60}")
    for rank, e in enumerate(unc_l, 1):
        _entry_line(rank, e, show_unc=True)
    print()

    # ── Modo interactivo ─────────────────────────────────────────────
    if getattr(args, 'interactive', False):
        _rank_interactive(args, model, entries, model_file)

    # ── Exportar CSV si se pide ───────────────────────────────────────
    if args.csv:
        import csv
        out_path = args.csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["rank_pref", "score", "uncertainty", "n_seen",
                      "duration_s", "bpm", "name", "path"]
            if affinity_vectors:
                header += ["pref_score", "aff_score"]
            w.writerow(header)
            for rank, e in enumerate(
                sorted(entries, key=lambda x: x["score"], reverse=True), 1
            ):
                row = [rank, f"{e['score']:.4f}", f"{e['unc']:.4f}",
                       e["n_seen"], f"{e['dur']:.1f}", e["bpm"],
                       e["name"], e["path"]]
                if affinity_vectors:
                    row += [f"{e['pref_score']:.4f}", f"{e['aff_score']:.4f}"]
                w.writerow(row)
        print(f"  {green('✓')} CSV exportado: {cyan(out_path)}")
        print(f"  {n_total:,} filas, ordenadas por preferencia descendente.")

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
    p_index.add_argument("--extended-features", dest="extended_features",
                         action="store_true",
                         help="Calcular 20 features en lugar de 10 (más lento, mejor discriminación)")
    p_index.add_argument("--melodic-features", dest="melodic_features",
                         action="store_true",
                         help="Calcular 16 features melódicas (para MIDIs de una sola voz)")

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
    p_eval.add_argument("--corpus", default=None,
                        help="Corpus .npz para percentil y vecinos (opcional pero recomendado)")
    p_eval.add_argument("--verbose", action="store_true",
                        help="Desglose completo de dimensiones")
    p_eval.add_argument("--json", dest="json_out", action="store_true",
                        help="Salida JSON machine-readable al final")

    # ── rank ───────────────────────────────────────────────────────
    p_rank = sub.add_parser("rank", help="Escanear corpus → top/bottom/inciertas")
    p_rank.add_argument("corpus",
                        help="Archivo .npz del corpus a escanear")
    p_rank.add_argument("--model", default="prefs",
                        help="Nombre base del modelo (default: prefs)")
    p_rank.add_argument("--top", type=int, default=20,
                        help="Número de entradas por lista (default: 20)")
    p_rank.add_argument("--affinity", default=None,
                        help="Carpeta con obras afines para ponderar afinidad")
    p_rank.add_argument("--weight", type=float, default=0.5,
                        help="Peso de la afinidad [0-1] (default: 0.5). "
                             "0=solo preferencia, 1=solo afinidad")
    p_rank.add_argument("--exclude-annotated", dest="exclude_annotated",
                        action="store_true",
                        help="Excluir MIDIs ya anotados en sesiones previas")
    p_rank.add_argument("--interactive", action="store_true",
                        help="Anotar MIDIs directamente desde el ranking")
    p_rank.add_argument("--soundfont", default=None,
                        help="Soundfont .sf2 para reproducción en modo interactivo")
    p_rank.add_argument("--csv", default=None,
                        help="Exportar ranking completo a CSV")

    # ── set ────────────────────────────────────────────────────────
    p_set = sub.add_parser("set", help="Asignar preferencia directamente a un MIDI")
    p_set.add_argument("midi", nargs="+",
                       help="Archivo(s) MIDI o directorio")
    p_set.add_argument("--score", type=int, required=True, choices=[1,2,3,4,5],
                       help="Preferencia a asignar (1-5)")
    p_set.add_argument("--model", default="prefs",
                       help="Nombre base del modelo (default: prefs)")
    p_set.add_argument("--repetitions", type=int, default=10,
                       help="Fuerza del ajuste: nº de veces que se aplica (default: 10)")

    # ── validate ───────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Validación cruzada del modelo")
    p_val.add_argument("--model", default="prefs",
                       help="Nombre base del modelo (default: prefs)")
    p_val.add_argument("--corpus", default=None,
                       help="Corpus .npz para recuperar vectores del historial")
    p_val.add_argument("--folds", type=int, default=5,
                       help="Número de folds (default: 5)")

    # ── info ───────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Inspeccionar modelo guardado")
    p_info.add_argument("--model", default="prefs",
                        help="Nombre base del modelo (default: prefs)")

    args = parser.parse_args()

    if args.cmd == "set":
        cmd_set(args)
    elif args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "rank":
        cmd_rank(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
