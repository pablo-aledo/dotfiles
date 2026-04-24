#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     MIDI SPLIT HARMONY  v1.0                                 ║
║     Separación automática de melodía y acordes desde un MIDI de entrada     ║
║                                                                              ║
║  Dado un MIDI de entrada (monofónico o polifónico, con armonía limpia       ║
║  o mal transcrita), produce:                                                 ║
║    · Un MIDI de melodía  — voz principal, limpia y continua                 ║
║    · Un MIDI de acordes  — progresión armónica cuantizada y correcta        ║
║                                                                              ║
║  PIPELINE INTERNO:                                                           ║
║  [1] PRE-ANÁLISIS     — tempo, compás, tonalidad, canales, pistas           ║
║  [2] SEPARACIÓN       — skyline mejorado + continuidad + verosimilitud      ║
║                         interválica → extrae voz melódica principal          ║
║  [3] EXTRACCIÓN       — chroma por ventanas solapadas + template matching   ║
║                         + HMM de funciones armónicas → acordes coherentes   ║
║  [4] POST-LIMPIEZA    — quantización al beat, fusión de acordes cortos,     ║
║                         corrección por voz-leading, coherencia tonal        ║
║  [5] EXPORTACIÓN      — MIDI separado o fichero de dos pistas               ║
║                                                                              ║
║  MODOS DE ANÁLISIS:                                                          ║
║    deterministic  — algoritmos puramente basados en teoría musical (default)║
║    corpus         — aprende matrices de transición y emisión desde un       ║
║                     directorio de MIDIs de referencia; combina con reglas   ║
║                                                                              ║
║  USO BÁSICO:                                                                 ║
║    python midi_split_harmony.py entrada.mid                                  ║
║    python midi_split_harmony.py entrada.mid --out-dir salidas/              ║
║    python midi_split_harmony.py entrada.mid --split-files                   ║
║    python midi_split_harmony.py entrada.mid --key "A minor" --verbose       ║
║                                                                              ║
║  MODO CORPUS:                                                                ║
║    # Entrenar el modelo desde corpus (sin analizar ninguna canción)          ║
║    python midi_split_harmony.py --train-corpus ./mis_midis/                 ║
║        --save-model modelo.json                                              ║
║                                                                              ║
║    # Entrenar con tónica de referencia explícita                             ║
║    python midi_split_harmony.py --train-corpus ./mis_midis/                 ║
║        --key "C minor" --save-model modelo_menor.json --verbose             ║
║                                                                              ║
║    # Entrenar y analizar en un solo paso (guarda el modelo de paso)         ║
║    python midi_split_harmony.py entrada.mid --corpus ./mis_midis/           ║
║        --save-model modelo.json                                              ║
║                                                                              ║
║    # Usar un modelo ya entrenado                                             ║
║    python midi_split_harmony.py entrada.mid --load-model modelo.json        ║
║                                                                              ║
║  OPCIONES COMPLETAS:                                                         ║
║    input              Ruta al MIDI de entrada                               ║
║    --out-dir DIR       Carpeta de salida (default: junto al MIDI de entrada) ║
║    --output BASE       Nombre base de salida (default: <stem>)              ║
║    --split-files       Genera dos ficheros separados en vez de uno          ║
║                        con dos pistas (default: fichero único)              ║
║    --key "C major"     Tonalidad forzada (default: autodetectada)           ║
║    --beats-per-bar N   Pulsos por compás (default: detectado del MIDI)      ║
║    --tempo BPM         Tempo de salida (default: detectado)                 ║
║    --chord-res BEATS   Resolución mínima de acorde en beats (default: 1.0) ║
║    --melody-ch N       Canal MIDI a forzar como melodía (0-15)              ║
║    --harmony-ch N      Canal MIDI a forzar como armonía (0-15)             ║
║    --chord-octave N    Octava base de los acordes de salida (default: 4)   ║
║    --melody-octave N   Octava de transposición de melodía (default: 5)     ║
║    --velocity-melody N Velocity de las notas de melodía (default: 90)      ║
║    --velocity-chords N Velocity de los acordes (default: 70)               ║
║    --corpus DIR        Directorio de MIDIs de referencia para modo corpus   ║
║    --save-model FILE   Guardar el modelo HMM aprendido en JSON              ║
║    --load-model FILE   Cargar un modelo HMM previo                          ║
║    --min-corpus-score  Umbral de calidad para incluir un MIDI del corpus    ║
║                        en el entrenamiento (default: 0.55)                  ║
║    --report            Guardar reporte JSON con análisis detallado          ║
║    --verbose           Información detallada del proceso                    ║
║                                                                              ║
║  SALIDA (modo --split-files):                                               ║
║    <base>_melody.mid   — pista de melodía                                   ║
║    <base>_chords.mid   — pista de acordes                                   ║
║                                                                              ║
║  SALIDA (modo por defecto):                                                  ║
║    <base>_split.mid    — MIDI con dos pistas: ch0=melodía, ch1=acordes     ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                  ║
║  OPCIONALES:   music21 (mejora detección de tonalidad)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import copy
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import mido

# ── music21 opcional ──────────────────────────────────────────────────────────
try:
    from music21 import converter as m21conv, key as m21key
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MUSICALES
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES       = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE_PC: Dict[str, int] = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})
NOTE_PC.update({'Cb': 11, 'Fb': 4, 'B#': 0, 'E#': 5})

# Intervalos de acorde por calidad (semitonos desde la raíz)
CHORD_INTERVALS: Dict[str, List[int]] = {
    'maj':   [0, 4, 7],
    'min':   [0, 3, 7],
    'dom7':  [0, 4, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'min7':  [0, 3, 7, 10],
    'dim':   [0, 3, 6],
    'dim7':  [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],
    'aug':   [0, 4, 8],
    'sus4':  [0, 5, 7],
    'sus2':  [0, 2, 7],
    'dom9':  [0, 4, 7, 10, 14],
    'maj9':  [0, 4, 7, 11, 14],
    'min9':  [0, 3, 7, 10, 14],
    'maj6':  [0, 4, 7, 9],
    'min6':  [0, 3, 7, 9],
}

# Todos los acordes como (raíz_pc, calidad) para template matching
ALL_CHORD_TEMPLATES: List[Tuple[int, str]] = [
    (root, qual)
    for root in range(12)
    for qual in CHORD_INTERVALS
]

# Escalas diatónicas
SCALES: Dict[str, List[int]] = {
    'major':            [0, 2, 4, 5, 7, 9, 11],
    'minor':            [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor':   [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor':    [0, 2, 3, 5, 7, 9, 11],
    'dorian':           [0, 2, 3, 5, 7, 9, 10],
    'phrygian':         [0, 1, 3, 5, 7, 8, 10],
    'lydian':           [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':       [0, 2, 4, 5, 7, 9, 10],
    'phrygian_dominant':[0, 1, 4, 5, 7, 8, 10],
}

# Acordes diatónicos por modo: (grado, calidad)
DIATONIC_CHORDS: Dict[str, List[Tuple[int, str]]] = {
    'major':  [(0,'maj'),(2,'min'),(4,'min'),(5,'maj'),(7,'maj'),(9,'min'),(11,'dim')],
    'minor':  [(0,'min'),(2,'dim'),(3,'maj'),(5,'min'),(7,'min'),(8,'maj'),(10,'maj')],
    'harmonic_minor': [(0,'min'),(2,'dim'),(3,'aug'),(5,'min'),(7,'maj'),(8,'maj'),(11,'dim7')],
    'dorian': [(0,'min'),(2,'min'),(3,'maj'),(5,'maj'),(7,'min'),(9,'dim'),(10,'maj')],
    'phrygian':[(0,'min'),(1,'maj'),(3,'maj'),(5,'min'),(7,'dim'),(8,'maj'),(10,'min')],
    'phrygian_dominant':[(0,'maj'),(1,'maj'),(3,'dim'),(5,'min'),(7,'dim'),(8,'maj'),(10,'min')],
    'mixolydian':[(0,'maj'),(2,'min'),(4,'dim'),(5,'maj'),(7,'min'),(9,'min'),(10,'maj')],
    'lydian':[(0,'maj'),(2,'maj'),(4,'min'),(6,'dim'),(7,'maj'),(9,'min'),(11,'min')],
}

# Prioridad de calidades: cuando hay empate en notas explicadas, preferimos
# tríadas sobre acordes de séptima, que a su vez van antes que acordes raros
QUALITY_PRIORITY: Dict[str, int] = {
    'maj': 10, 'min': 10, 'dom7': 9, 'maj7': 8, 'min7': 8,
    'dim7': 7, 'hdim7': 7, 'dim': 7, 'aug': 6,
    'sus4': 5, 'sus2': 5, 'dom9': 4, 'maj9': 4, 'min9': 4,
    'maj6': 3, 'min6': 3,
}

# Tabla de transición armónica por defecto (función → función)
# Construida desde las reglas de voz-leading tonal clásico
# T=tónica(0), S=subdominante(1), D=dominante(2)
# Codificamos: (calidad_src, raíz_relativa_src) → probabilidades de siguiente acorde
# Simplificado: usamos distancias en el círculo de quintas
# Distancia en círculo de quintas entre dos pitch classes
def _fifths_dist(a: int, b: int) -> int:
    """Distancia mínima en el círculo de quintas entre dos pitch classes."""
    steps = [0, 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7]  # C=0 en círculo de quintas
    pa = steps[a % 12]
    pb = steps[b % 12]
    d = abs(pa - pb)
    return min(d, 12 - d)

# Función armónica de un acorde dado tónica y modo
def chord_function(root_pc: int, quality: str, tonic_pc: int, mode: str) -> str:
    """Devuelve 'T', 'S', 'D' o 'X' (no diatónico) para un acorde."""
    interval = (root_pc - tonic_pc) % 12
    diatonic = DIATONIC_CHORDS.get(mode, DIATONIC_CHORDS['major'])
    diatonic_roots = [d[0] for d in diatonic]
    if interval not in diatonic_roots:
        return 'X'
    idx = diatonic_roots.index(interval)
    # Asignación funcional por grado (1-indexed)
    tonal_functions = {
        'major':  ['T','S','T','S','D','T','D'],
        'minor':  ['T','D','T','S','D','S','S'],
        'harmonic_minor': ['T','D','T','S','D','S','D'],
        'dorian': ['T','S','T','S','D','D','S'],
        'phrygian':['T','S','T','S','D','S','S'],
        'phrygian_dominant':['D','S','D','S','D','S','S'],
        'mixolydian':['T','S','D','S','D','S','S'],
        'lydian':['T','S','D','D','D','S','S'],
    }
    funcs = tonal_functions.get(mode, tonal_functions['major'])
    return funcs[idx] if idx < len(funcs) else 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA Y ANÁLISIS BÁSICO DEL MIDI
# ═══════════════════════════════════════════════════════════════════════════════

class MidiInfo:
    """Contenedor con toda la información extraída del MIDI de entrada."""
    def __init__(self):
        self.tpb: int = 480              # ticks per beat
        self.tempo_us: int = 500_000     # microsegundos por beat (120 BPM)
        self.bpm: float = 120.0
        self.beats_per_bar: int = 4
        self.beat_unit: int = 4          # denominador del compás
        self.total_ticks: int = 0
        self.total_beats: float = 0.0
        self.total_bars: float = 0.0
        # Notas: lista de (start_beat, pitch_midi, duration_beats, velocity, channel, track_idx)
        self.notes: List[Tuple[float,int,float,int,int,int]] = []
        self.n_tracks: int = 0
        self.n_channels: int = 0
        self.channel_note_count: Dict[int, int] = {}
        self.track_names: Dict[int, str] = {}


def load_midi(path: str) -> MidiInfo:
    """Carga un fichero MIDI y extrae todas las notas con sus offsets en beats."""
    mid = mido.MidiFile(path)
    info = MidiInfo()
    info.tpb = mid.ticks_per_beat
    info.n_tracks = len(mid.tracks)

    # Recolectar tempo y compás del track 0
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            info.tempo_us = msg.tempo
            info.bpm = 60_000_000 / msg.tempo
        elif msg.type == 'time_signature':
            info.beats_per_bar = msg.numerator
            info.beat_unit = msg.denominator

    def ticks_to_beats(t: int) -> float:
        return t / info.tpb

    all_notes = []
    for track_idx, track in enumerate(mid.tracks):
        # Nombre de la pista
        for msg in track:
            if msg.type == 'track_name':
                info.track_names[track_idx] = msg.name
                break

        # Convertir a absoluto
        abs_ticks = 0
        pending: Dict[Tuple[int,int], List[Tuple[int,int]]] = defaultdict(list)  # (ch,note) -> [start_ticks, vel]

        for msg in track:
            abs_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)].append((abs_ticks, msg.velocity))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if pending[key]:
                    start_t, vel = pending[key].pop(0)
                    dur_t = abs_ticks - start_t
                    if dur_t > 0:
                        start_b = ticks_to_beats(start_t)
                        dur_b   = ticks_to_beats(dur_t)
                        all_notes.append((start_b, msg.note, dur_b, vel, msg.channel, track_idx))
                        info.channel_note_count[msg.channel] = \
                            info.channel_note_count.get(msg.channel, 0) + 1
                        if abs_ticks > info.total_ticks:
                            info.total_ticks = abs_ticks

    info.notes = sorted(all_notes, key=lambda n: (n[0], n[1]))
    info.total_beats = ticks_to_beats(info.total_ticks)
    info.total_bars = info.total_beats / info.beats_per_bar
    info.n_channels = len(info.channel_note_count)
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE TONALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def build_chroma_vector(notes: List[Tuple], start_b: float = 0.0,
                        end_b: float = float('inf')) -> np.ndarray:
    """Construye un vector de chroma (12-dim) ponderado por duración."""
    chroma = np.zeros(12)
    for (s, pitch, dur, vel, ch, tr) in notes:
        if s >= end_b or s + dur <= start_b:
            continue
        overlap = min(s + dur, end_b) - max(s, start_b)
        if overlap > 0:
            chroma[pitch % 12] += overlap * (vel / 127.0)
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


# Perfiles de Krumhansl-Schmuckler para correlación de tonalidad
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def detect_key(notes: List[Tuple]) -> Tuple[int, str]:
    """
    Detecta la tonalidad del MIDI usando correlación de Krumhansl-Schmuckler.
    Devuelve (tonic_pc, mode) donde mode es 'major' o 'minor'.
    """
    # Intentar con music21 si está disponible
    if MUSIC21_OK:
        try:
            # Construir un stream temporal desde las notas
            from music21 import stream as m21stream, note as m21note
            s = m21stream.Stream()
            for (sb, pitch, dur, vel, ch, tr) in notes[:200]:  # muestra
                n = m21note.Note(pitch)
                n.quarterLength = max(dur, 0.25)
                s.insert(sb, n)
            k = s.analyze('key')
            tonic_pc = NOTE_PC.get(k.tonic.name, 0)
            mode = k.mode  # 'major' o 'minor'
            return tonic_pc, mode
        except Exception:
            pass

    # Fallback: correlación KS
    chroma = build_chroma_vector(notes)
    best_score = -np.inf
    best_key = (0, 'major')
    for tonic in range(12):
        rotated = np.roll(chroma, -tonic)
        score_maj = np.corrcoef(rotated, _KS_MAJOR)[0, 1]
        score_min = np.corrcoef(rotated, _KS_MINOR)[0, 1]
        if score_maj > best_score:
            best_score = score_maj
            best_key = (tonic, 'major')
        if score_min > best_score:
            best_score = score_min
            best_key = (tonic, 'minor')
    return best_key


def parse_key_string(key_str: str) -> Tuple[int, str]:
    """Parsea 'A minor', 'C major', 'Bb', 'Am', 'F#m'..."""
    s = key_str.strip()
    mode = 'major'
    if ' ' in s:
        parts = s.split()
        root = parts[0]
        if len(parts) > 1 and parts[1].lower() in ('minor', 'min', 'm'):
            mode = 'minor'
        elif len(parts) > 1 and parts[1].lower() in ('major', 'maj'):
            mode = 'major'
    else:
        if s.endswith('m') and len(s) <= 3 and not s.endswith('dim'):
            root = s[:-1]
            mode = 'minor'
        else:
            root = s
    tonic_pc = NOTE_PC.get(root, 0)
    return tonic_pc, mode


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SEPARACIÓN MELODÍA / ARMONÍA
# ═══════════════════════════════════════════════════════════════════════════════

# Tabla de probabilidades de transición interválica para melodías típicas
# Intervalo en semitonos (0-24, donde 0=unísono, 12=octava) → peso de verosimilitud
_MELODIC_INTERVAL_WEIGHT = {
    0:  0.10,   # unísono repetido (bajo en melodía)
    1:  0.95,   # semitono
    2:  1.00,   # tono (más frecuente)
    3:  0.85,   # tercera menor
    4:  0.80,   # tercera mayor
    5:  0.70,   # cuarta justa
    6:  0.30,   # tritono
    7:  0.65,   # quinta justa
    8:  0.40,
    9:  0.45,   # sexta mayor
    10: 0.30,
    11: 0.25,
    12: 0.50,   # octava
    13: 0.20,
    14: 0.20,
    15: 0.15,
    16: 0.15,
    17: 0.20,
    18: 0.10,
    19: 0.10,
    20: 0.08,
    21: 0.08,
    22: 0.06,
    23: 0.06,
    24: 0.20,   # doble octava
}


def melodic_verosimilitude(pitches: List[int]) -> float:
    """
    Calcula la verosimilitud melódica de una secuencia de alturas MIDI.
    Más alta = más parecido a una melodía vocal/instrumental natural.
    """
    if len(pitches) < 2:
        return 0.5
    weights = []
    for i in range(1, len(pitches)):
        interval = abs(pitches[i] - pitches[i-1])
        w = _MELODIC_INTERVAL_WEIGHT.get(min(interval, 24), 0.05)
        weights.append(w)
    return float(np.mean(weights))


def classify_channels(notes: List[Tuple]) -> Dict[Tuple[int,int], str]:
    """
    Clasifica cada voz MIDI como 'melody', 'harmony', 'bass' o 'drums'.

    La clave de clasificación es (track_idx, channel) — no solo channel.
    Esto es esencial para MIDIs tipo 1 donde múltiples tracks comparten el
    mismo número de canal (p.ej. melodía en track 0 canal 0 y bajo en
    track 1 canal 0): mezclarlos antes de analizar destruye el poly ratio
    y el registro de ambas voces.

    Criterios:
    - drums   : canal 9 GM, o p90 < 40 con poca variedad tonal
    - bass    : p90 < 52 (por debajo de E3)
    - harmony : poly_ratio > 0.55 (acordes en bloque o arpegios densos)
    - melody  : monofónico o casi, verosimilitud interválica alta,
                registro medio-agudo

    Devuelve dict {(track_idx, channel): clasificación}.
    """
    voice_notes: Dict[Tuple[int,int], List[Tuple]] = defaultdict(list)
    for n in notes:
        voice_notes[(n[5], n[4])].append(n)   # clave: (track, channel)

    classification: Dict[Tuple[int,int], str] = {}

    for (tr, ch), ch_n in voice_notes.items():
        if not ch_n:
            continue

        # ── Batería ───────────────────────────────────────────────────────────
        if ch == 9:
            classification[(tr, ch)] = 'drums'
            continue

        pitches = [n[1] for n in ch_n]
        median_pitch = float(np.median(pitches))
        p90 = float(np.percentile(pitches, 90))

        # ── Bajo: registro muy bajo ───────────────────────────────────────────
        if p90 < 52:
            classification[(tr, ch)] = 'bass'
            continue

        # ── Ratio de simultaneidad dentro de la misma voz (tr, ch) ───────────
        ch_sorted = sorted(ch_n, key=lambda x: x[0])
        poly_count = 0
        for i, (s, p, d, v, c, t) in enumerate(ch_sorted):
            has_overlap = any(
                abs(s2 - s) < 0.08 and p2 != p
                for (s2, p2, d2, v2, c2, t2) in ch_sorted[max(0,i-3):i+4]
                if (s2, p2) != (s, p)
            )
            if has_overlap:
                poly_count += 1
        poly_ratio = poly_count / len(ch_sorted)

        # ── Verosimilitud melódica ────────────────────────────────────────────
        mel_score = melodic_verosimilitude(pitches)

        # ── Densidad: notas por beat ──────────────────────────────────────────
        total_span = max(n[0] + n[2] for n in ch_n) - min(n[0] for n in ch_n)
        density = len(ch_n) / max(total_span, 1.0)

        # ── Decisión ──────────────────────────────────────────────────────────
        if poly_ratio > 0.55:
            classification[(tr, ch)] = 'harmony'
        elif median_pitch < 50 and density < 4.0:
            classification[(tr, ch)] = 'bass'
        elif poly_ratio < 0.25 and mel_score > 0.45:
            classification[(tr, ch)] = 'melody'
        elif poly_ratio < 0.40 and mel_score > 0.55 and median_pitch > 55:
            classification[(tr, ch)] = 'melody'
        else:
            classification[(tr, ch)] = 'harmony'

    return classification


def extract_skyline(notes: List[Tuple], tonic_pc: int, mode: str,
                    forced_melody_ch: Optional[int] = None,
                    verbose: bool = False) -> Tuple[List[Tuple], Dict[Tuple[int,int], str]]:
    """
    Extrae la voz melódica principal en dos fases.

    Fase A — Clasificación de voces (track, channel):
        Cada combinación (track_idx, channel) se clasifica independientemente,
        lo que permite distinguir p.ej. melodía en track 0 canal 0 de bajo en
        track 1 canal 0, que es el caso habitual en MIDIs tipo 1.

    Fase B — Skyline sobre candidatos melódicos.

    Devuelve (melody_notes, voice_classifications).
    """
    if not notes:
        return [], {}

    # Canal forzado: filtrar por channel ignorando track
    if forced_melody_ch is not None:
        melody = [n for n in notes if n[4] == forced_melody_ch]
        if melody:
            return sorted(melody, key=lambda x: x[0]), {
                (n[5], n[4]): 'melody' for n in melody
            }

    # ── Fase A: clasificar voces (track, channel) ─────────────────────────────
    voice_classification = classify_channels(notes)

    if verbose:
        for (tr, ch), cls in sorted(voice_classification.items()):
            v_n = [n for n in notes if n[5] == tr and n[4] == ch]
            pitches = [n[1] for n in v_n]
            print(f"║    Track {tr} Canal {ch}: {cls:8s}  n={len(v_n):4d}  "
                  f"median={int(np.median(pitches))}  "
                  f"p80={int(np.percentile(pitches,80))}")

    melody_voices = {(tr, ch) for (tr, ch), cls in voice_classification.items()
                     if cls == 'melody'}

    if melody_voices:
        candidate_notes = [n for n in notes if (n[5], n[4]) in melody_voices]
    else:
        non_melody = {(tr, ch) for (tr, ch), cls in voice_classification.items()
                      if cls in ('drums', 'bass')}
        candidate_notes = [n for n in notes if (n[5], n[4]) not in non_melody]
        if not candidate_notes:
            candidate_notes = [n for n in notes if n[4] != 9]

    if not candidate_notes:
        candidate_notes = notes

    # ── Fase B: Skyline sobre candidatos ─────────────────────────────────────
    # Clave única por voz (track, channel)
    candidate_voices = {(n[5], n[4]) for n in candidate_notes}
    if len(candidate_voices) == 1:
        return sorted(candidate_notes, key=lambda x: x[0]), voice_classification

    # Heurístico: voz con mayor p80 de pitch
    voice_p80: Dict[Tuple[int,int], float] = {}
    for v in candidate_voices:
        p = [n[1] for n in candidate_notes if (n[5], n[4]) == v]
        if p:
            voice_p80[v] = float(np.percentile(p, 80))
    top_voice = max(voice_p80, key=lambda v: voice_p80[v]) if voice_p80 else (0, 0)

    resolution  = 0.1
    total_beats = max(n[0] + n[2] for n in candidate_notes)
    melody_notes: List[Tuple] = []
    prev_pitch: Optional[int] = None
    prev_end: float = 0.0

    beat = 0.0
    while beat < total_beats:
        active = [n for n in candidate_notes if n[0] <= beat < n[0] + n[2]]
        if not active:
            beat += resolution
            continue

        # Nota más aguda de cada voz activa
        v_top: Dict[Tuple[int,int], Tuple] = {}
        for n in active:
            v = (n[5], n[4])
            if v not in v_top or n[1] > v_top[v][1]:
                v_top[v] = n

        best_note = None
        best_score = -np.inf
        for v, n in v_top.items():
            pitch = n[1]
            score = 0.0
            score += (pitch - 48) / 60.0
            if prev_pitch is not None:
                interval = abs(pitch - prev_pitch)
                score += _MELODIC_INTERVAL_WEIGHT.get(min(interval, 24), 0.05) * 2.0
                if n[0] - prev_end < 0.5:
                    score += 0.3
            if v == top_voice:
                score += 0.4
            if score > best_score:
                best_score = score
                best_note = n

        if best_note is not None:
            if not melody_notes or melody_notes[-1] != best_note:
                melody_notes.append(best_note)
                prev_pitch = best_note[1]
                prev_end   = best_note[0] + best_note[2]

        beat += resolution

    seen: set = set()
    unique_melody = []
    for n in melody_notes:
        key = (round(n[0], 4), n[1])
        if key not in seen:
            seen.add(key)
            unique_melody.append(n)

    return sorted(unique_melody, key=lambda x: x[0]), voice_classification


def extract_harmony_notes(all_notes: List[Tuple],
                           melody_notes: List[Tuple],
                           voice_classification: Dict[Tuple[int,int], str]) -> List[Tuple]:
    """
    Extrae las notas de armonía usando la clasificación por (track, channel).

    Prioridad:
    1. Voces clasificadas como 'harmony' (+ 'bass' para información de raíz).
    2. Fallback: todo excepto melodía y batería.
    3. Fallback monofónico: toda la señal sin batería.
    """
    harmony_voices = {v for v, cls in voice_classification.items() if cls == 'harmony'}
    bass_voices    = {v for v, cls in voice_classification.items() if cls == 'bass'}
    drum_voices    = {v for v, cls in voice_classification.items() if cls == 'drums'}

    if harmony_voices:
        harmony = [n for n in all_notes
                   if (n[5], n[4]) in harmony_voices or (n[5], n[4]) in bass_voices
                   and (n[5], n[4]) not in drum_voices]
        if harmony:
            return harmony

    # Fallback: excluir melodía y batería
    melody_keys = {(round(n[0], 4), n[1]) for n in melody_notes}
    harmony = [n for n in all_notes
               if (round(n[0], 4), n[1]) not in melody_keys
               and (n[5], n[4]) not in drum_voices]

    if len(harmony) < 0.15 * len(all_notes):
        return [n for n in all_notes if (n[5], n[4]) not in drum_voices]
    return harmony


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — EXTRACCIÓN DE ACORDES (CHROMA + TEMPLATE MATCHING + HMM)
# ═══════════════════════════════════════════════════════════════════════════════

def chroma_window(notes: List[Tuple], start: float, end: float,
                  weight_by_velocity: bool = True) -> np.ndarray:
    """
    Calcula el vector chroma de las notas en la ventana [start, end) beats.
    Pondera por la duración de la nota dentro de la ventana y opcionalmente por velocity.
    """
    chroma = np.zeros(12)
    for (s, pitch, dur, vel, ch, tr) in notes:
        note_end = s + dur
        overlap_start = max(s, start)
        overlap_end   = min(note_end, end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        weight = overlap * (vel / 127.0 if weight_by_velocity else 1.0)
        chroma[pitch % 12] += weight
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


def score_chord_template(chroma: np.ndarray, root_pc: int, quality: str) -> float:
    """
    Puntúa qué tan bien explica un acorde (root_pc, quality) el vector chroma dado.
    Devuelve [0, 1].
    """
    intervals = CHORD_INTERVALS[quality]
    chord_pcs = set((root_pc + iv) % 12 for iv in intervals)

    # Notas presentes con peso significativo (>2% del chroma)
    present = {pc for pc in range(12) if chroma[pc] > 0.02}

    if not present:
        return 0.0

    # Score: proporción de peso del chroma que cae en notas del acorde
    chord_weight = sum(chroma[pc] for pc in chord_pcs)
    non_chord_weight = sum(chroma[pc] for pc in present - chord_pcs)

    # Penalización por notas fuera del acorde
    penalty = non_chord_weight * 0.5

    # Bonus por presencia de la raíz
    root_bonus = chroma[root_pc] * 0.3

    score = chord_weight - penalty + root_bonus
    return float(np.clip(score, 0.0, 1.0))


def template_match_window(chroma: np.ndarray,
                          tonic_pc: int, mode: str,
                          diatonic_bonus: float = 0.15) -> Tuple[int, str, float]:
    """
    Encuentra el acorde que mejor explica el chroma dado.
    Aplica un bonus a acordes diatónicos en la tonalidad detectada.
    Devuelve (root_pc, quality, score).
    """
    diatonic_set = set()
    for (interval, qual) in DIATONIC_CHORDS.get(mode, DIATONIC_CHORDS['major']):
        diatonic_set.add(((tonic_pc + interval) % 12, qual))

    best_root, best_qual, best_score = 0, 'maj', -1.0

    for (root_pc, quality) in ALL_CHORD_TEMPLATES:
        s = score_chord_template(chroma, root_pc, quality)
        # Bonus diatónico
        if (root_pc, quality) in diatonic_set:
            s += diatonic_bonus
        # Prioridad de calidad (para desempate)
        s += QUALITY_PRIORITY.get(quality, 1) * 0.001

        if s > best_score:
            best_score = s
            best_root = root_pc
            best_qual = quality

    return best_root, best_qual, best_score


# ─── HMM de funciones armónicas ───────────────────────────────────────────────

def build_default_transition_matrix(tonic_pc: int, mode: str) -> np.ndarray:
    """
    Construye una matriz de transición entre acordes basada en reglas armónicas.
    Dimensión: N_TEMPLATES × N_TEMPLATES donde N_TEMPLATES = 12 × len(CHORD_INTERVALS).

    La probabilidad de transición se basa en:
    1. Distancia de voice-leading (cuántos semitonos cambian las voces)
    2. Función armónica (T→S, S→D, D→T son naturales)
    3. Movimiento en el círculo de quintas
    """
    n = len(ALL_CHORD_TEMPLATES)
    trans = np.zeros((n, n))

    # Secuencias de funciones armónicas permitidas con sus pesos
    func_flow = {
        ('T', 'T'): 0.3, ('T', 'S'): 0.5, ('T', 'D'): 0.4, ('T', 'X'): 0.1,
        ('S', 'T'): 0.3, ('S', 'S'): 0.2, ('S', 'D'): 0.6, ('S', 'X'): 0.1,
        ('D', 'T'): 0.7, ('D', 'S'): 0.2, ('D', 'D'): 0.2, ('D', 'X'): 0.1,
        ('X', 'T'): 0.3, ('X', 'S'): 0.3, ('X', 'D'): 0.3, ('X', 'X'): 0.2,
    }

    for i, (r1, q1) in enumerate(ALL_CHORD_TEMPLATES):
        f1 = chord_function(r1, q1, tonic_pc, mode)
        for j, (r2, q2) in enumerate(ALL_CHORD_TEMPLATES):
            f2 = chord_function(r2, q2, tonic_pc, mode)

            # Peso base desde flujo funcional
            w = func_flow.get((f1, f2), 0.1)

            # Bonus por voz-leading: cuántas notas comparten ambos acordes
            pcs1 = set((r1 + iv) % 12 for iv in CHORD_INTERVALS[q1])
            pcs2 = set((r2 + iv) % 12 for iv in CHORD_INTERVALS[q2])
            shared = len(pcs1 & pcs2)
            w += shared * 0.05

            # Bonus por movimiento en círculo de quintas (movimiento natural)
            fd = _fifths_dist(r1, r2)
            if fd <= 2:
                w += 0.1
            elif fd >= 5:
                w -= 0.05

            trans[i, j] = max(w, 0.001)

        # Normalizar fila
        row_sum = trans[i].sum()
        if row_sum > 0:
            trans[i] /= row_sum

    return trans


def build_emission_matrix(tonic_pc: int, mode: str) -> np.ndarray:
    """
    Construye la matriz de emisión: P(chroma_bin | chord).
    Simplificado: para cada acorde, la probabilidad de observar un chroma
    es proporcional a qué notas del acorde están presentes (modelo ideal).
    Dimensión: N_TEMPLATES × 12 (chroma bins)
    """
    n = len(ALL_CHORD_TEMPLATES)
    emit = np.zeros((n, 12))

    for i, (root_pc, quality) in enumerate(ALL_CHORD_TEMPLATES):
        intervals = CHORD_INTERVALS[quality]
        pcs = [(root_pc + iv) % 12 for iv in intervals]
        # Pesos por posición en el acorde: fundamental > quinta > tercera > extensiones
        weights = [1.0, 0.7, 0.8] + [0.4] * (len(pcs) - 3)
        for pc, w in zip(pcs, weights[:len(pcs)]):
            emit[i, pc] += w

        # Normalizar
        row_sum = emit[i].sum()
        if row_sum > 0:
            emit[i] /= row_sum

    return emit


class HarmonicHMM:
    """
    HMM de funciones armónicas para decodificar la secuencia de acordes óptima.

    Estados: acordes (root_pc, quality) = 12 × len(CHORD_INTERVALS) estados
    Observaciones: vectores chroma de 12 dimensiones discretizados
    """

    def __init__(self, tonic_pc: int, mode: str,
                 transition: Optional[np.ndarray] = None,
                 emission: Optional[np.ndarray] = None):
        self.tonic_pc = tonic_pc
        self.mode = mode
        n = len(ALL_CHORD_TEMPLATES)

        self.transition = transition if transition is not None else \
            build_default_transition_matrix(tonic_pc, mode)

        self.emission = emission if emission is not None else \
            build_emission_matrix(tonic_pc, mode)

        # Prior: acordes diatónicos son más probables a priori
        self.prior = np.ones(n) * 0.5
        diatonic = DIATONIC_CHORDS.get(mode, DIATONIC_CHORDS['major'])
        for (interval, qual) in diatonic:
            root = (tonic_pc + interval) % 12
            for j, (r, q) in enumerate(ALL_CHORD_TEMPLATES):
                if r == root and q == qual:
                    self.prior[j] = 2.0
        self.prior /= self.prior.sum()

    def observation_likelihood(self, chroma: np.ndarray, state_idx: int) -> float:
        """P(chroma | state) usando correlación con el perfil de emisión."""
        emit_profile = self.emission[state_idx]
        # Correlación de Pearson entre el chroma observado y el perfil ideal
        if chroma.sum() < 0.01:
            return 1.0 / len(ALL_CHORD_TEMPLATES)
        corr = np.dot(chroma, emit_profile)
        return max(corr, 1e-9)

    def viterbi(self, observations: List[np.ndarray]) -> List[int]:
        """
        Algoritmo de Viterbi para encontrar la secuencia de estados más probable.
        observations: lista de vectores chroma (12-dim) para cada ventana temporal.
        Devuelve: lista de índices de estado (acordes) de la misma longitud.
        """
        T = len(observations)
        if T == 0:
            return []
        n = len(ALL_CHORD_TEMPLATES)

        # Inicializar con log-probabilidades para evitar underflow
        log_prior = np.log(self.prior + 1e-9)
        log_trans  = np.log(self.transition + 1e-9)

        # delta[t, s] = log P de la secuencia hasta t terminando en estado s
        delta = np.full((T, n), -np.inf)
        psi   = np.zeros((T, n), dtype=int)

        # t=0
        for s in range(n):
            obs_ll = self.observation_likelihood(observations[0], s)
            delta[0, s] = log_prior[s] + math.log(obs_ll + 1e-9)

        # Recursión
        for t in range(1, T):
            for s in range(n):
                obs_ll = self.observation_likelihood(observations[t], s)
                trans_scores = delta[t-1] + log_trans[:, s]
                best_prev = int(np.argmax(trans_scores))
                delta[t, s] = trans_scores[best_prev] + math.log(obs_ll + 1e-9)
                psi[t, s] = best_prev

        # Backtrack
        path = [0] * T
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path


def extract_chords_hmm(harmony_notes: List[Tuple],
                       total_beats: float,
                       beats_per_bar: int,
                       tonic_pc: int, mode: str,
                       chord_resolution: float = 1.0,
                       hmm: Optional[HarmonicHMM] = None,
                       verbose: bool = False) -> List[Dict]:
    """
    Extrae la secuencia de acordes mediante análisis de chroma + HMM.

    Devuelve lista de dicts:
        {'start_beat': float, 'duration_beats': float,
         'root_pc': int, 'quality': str, 'name': str, 'score': float}
    """
    if not harmony_notes or total_beats <= 0:
        return []

    # Construir el HMM si no se proporciona
    if hmm is None:
        hmm = HarmonicHMM(tonic_pc, mode)

    # Ventana principal: chord_resolution beats
    # Ventanas adicionales solapadas para votar (×0.5 y ×2)
    window_sizes = [chord_resolution * 0.5, chord_resolution, chord_resolution * 2.0]
    votes: Dict[int, Dict[Tuple[int, str], float]] = defaultdict(lambda: defaultdict(float))

    n_windows = max(1, int(total_beats / chord_resolution))

    for ws in window_sizes:
        step = chord_resolution * 0.5
        t = 0.0
        while t < total_beats:
            chroma = chroma_window(harmony_notes, t, t + ws)
            root, qual, score = template_match_window(chroma, tonic_pc, mode)
            # Distribuir el voto a las ventanas de chord_resolution que solapa
            slot_start = int(t / chord_resolution)
            slot_end   = int(min(t + ws, total_beats) / chord_resolution)
            for slot in range(slot_start, slot_end + 1):
                votes[slot][(root, qual)] += score * (ws / chord_resolution)
            t += step

    # Para cada slot, elegir el acorde con más votos
    slot_chromàs: List[np.ndarray] = []
    slot_chords_voted: List[Tuple[int, str, float]] = []

    for slot in range(n_windows):
        t_start = slot * chord_resolution
        t_end   = t_start + chord_resolution
        chroma = chroma_window(harmony_notes, t_start, t_end)
        slot_chromàs.append(chroma)

        if votes[slot]:
            best = max(votes[slot], key=lambda k: votes[slot][k])
            slot_chords_voted.append((best[0], best[1], votes[slot][best]))
        else:
            # Fallback: template matching directo
            root, qual, score = template_match_window(chroma, tonic_pc, mode)
            slot_chords_voted.append((root, qual, score))

    # Refinar con Viterbi del HMM
    viterbi_path = hmm.viterbi(slot_chromàs)

    # Mezclar: el Viterbi refina la secuencia pero respeta el voto cuando la
    # confianza del voto es alta (score > 0.7)
    final_chords: List[Tuple[int, str]] = []
    for slot, (v_root, v_qual, v_score) in enumerate(slot_chords_voted):
        hmm_state = viterbi_path[slot]
        hmm_root, hmm_qual = ALL_CHORD_TEMPLATES[hmm_state]

        if v_score >= 0.65:
            # Alta confianza en el voto → usar voto
            final_chords.append((v_root, v_qual))
        else:
            # Baja confianza → combinar: si voto y HMM comparten raíz, usar voto
            if v_root == hmm_root:
                final_chords.append((v_root, v_qual))
            else:
                final_chords.append((hmm_root, hmm_qual))

    # Construir la lista de acordes con sus duraciones
    chord_list: List[Dict] = []
    for slot, (root_pc, quality) in enumerate(final_chords):
        start_beat = slot * chord_resolution
        duration   = chord_resolution
        # Ajustar duración del último acorde
        if start_beat + duration > total_beats:
            duration = total_beats - start_beat
        if duration <= 0:
            continue
        name = PITCH_NAMES[root_pc] + quality
        chroma = slot_chromàs[slot]
        score  = score_chord_template(chroma, root_pc, quality)
        chord_list.append({
            'start_beat':    start_beat,
            'duration_beats': duration,
            'root_pc':       root_pc,
            'quality':       quality,
            'name':          name,
            'score':         score,
        })

    return chord_list


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4 — POST-LIMPIEZA Y CONSOLIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def merge_short_chords(chords: List[Dict], min_beats: float = 1.0) -> List[Dict]:
    """
    Fusiona acordes demasiado cortos con el acorde anterior.
    Esto elimina cambios armónicos espurios producto de notas ornamentales.
    """
    if not chords:
        return chords
    result = [chords[0].copy()]
    for chord in chords[1:]:
        if chord['duration_beats'] < min_beats and result:
            # Fusionar con el anterior
            result[-1]['duration_beats'] += chord['duration_beats']
        else:
            result.append(chord.copy())
    return result


def smooth_chord_sequence(chords: List[Dict],
                           tonic_pc: int, mode: str) -> List[Dict]:
    """
    Suaviza la secuencia de acordes usando distancia de voice-leading.
    Si dos acordes adyacentes son muy similares (difieren solo en la calidad
    con la misma raíz), se unifica en el más probable dado el contexto.
    """
    if len(chords) < 2:
        return chords

    result = list(chords)
    changed = True
    max_passes = 3

    while changed and max_passes > 0:
        changed = False
        max_passes -= 1
        for i in range(1, len(result) - 1):
            prev_chord = result[i-1]
            curr_chord = result[i]
            next_chord = result[i+1] if i + 1 < len(result) else None

            # Si el acorde central es ambiguo (score bajo) y los vecinos son iguales
            if curr_chord['score'] < 0.35:
                if (next_chord and
                        prev_chord['root_pc'] == next_chord['root_pc'] and
                        prev_chord['quality'] == next_chord['quality']):
                    # El central es probablemente un error — usar el de los vecinos
                    result[i] = {
                        **curr_chord,
                        'root_pc': prev_chord['root_pc'],
                        'quality': prev_chord['quality'],
                        'name':    prev_chord['name'],
                        'score':   prev_chord['score'] * 0.8,
                    }
                    changed = True

    return result


def quantize_chords_to_grid(chords: List[Dict],
                             beats_per_bar: int,
                             chord_resolution: float) -> List[Dict]:
    """
    Ajusta los inicios de los acordes al grid de chord_resolution beats.
    Asegura que los acordes estén alineados con la métrica.
    """
    result = []
    for chord in chords:
        snapped_start = round(chord['start_beat'] / chord_resolution) * chord_resolution
        result.append({**chord, 'start_beat': snapped_start})

    # Reajustar duraciones para que no se solapen
    result.sort(key=lambda c: c['start_beat'])
    for i in range(len(result) - 1):
        gap = result[i+1]['start_beat'] - result[i]['start_beat']
        if gap > 0:
            result[i] = {**result[i], 'duration_beats': gap}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODO CORPUS — APRENDIZAJE DE MATRICES HMM
# ═══════════════════════════════════════════════════════════════════════════════

def score_midi_quality(info: MidiInfo, tonic_pc: int, mode: str) -> float:
    """
    Puntúa la calidad armónica de un MIDI de 0 a 1.
    Criterios:
    - Proporción de notas explicables por acordes diatónicos
    - Densidad polifónica (ni mono puro ni caótico)
    - Presencia de estructura métrica
    """
    if not info.notes:
        return 0.0

    scale = SCALES.get(mode, SCALES['major'])
    scale_pcs = set((tonic_pc + iv) % 12 for iv in scale)

    in_scale = sum(1 for n in info.notes if n[1] % 12 in scale_pcs)
    scale_ratio = in_scale / len(info.notes)

    # Densidad: notas simultáneas promedio
    sample_points = np.arange(0, min(info.total_beats, 32), 0.5)
    densities = []
    for t in sample_points:
        active = sum(1 for n in info.notes if n[0] <= t < n[0] + n[2])
        densities.append(active)
    avg_density = float(np.mean(densities)) if densities else 1.0
    density_score = 1.0 - abs(avg_density - 3.0) / 5.0  # óptimo ~3 voces

    # Score total
    return float(np.clip(0.7 * scale_ratio + 0.3 * density_score, 0.0, 1.0))


def learn_from_corpus(corpus_dir: str,
                       tonic_pc: int, mode: str,
                       min_score: float = 0.55,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aprende matrices de transición y emisión desde un directorio de MIDIs.
    Solo usa MIDIs con puntuación de calidad >= min_score.

    Devuelve (transition_matrix, emission_matrix).
    """
    midi_files = list(Path(corpus_dir).rglob('*.mid')) + \
                 list(Path(corpus_dir).rglob('*.midi'))

    if not midi_files:
        if verbose:
            print(f"  [corpus] No se encontraron MIDIs en {corpus_dir}")
        return (build_default_transition_matrix(tonic_pc, mode),
                build_emission_matrix(tonic_pc, mode))

    n = len(ALL_CHORD_TEMPLATES)
    # Acumuladores
    trans_counts = np.zeros((n, n)) + 0.1   # suavizado de Laplace
    emit_counts  = np.zeros((n, 12)) + 0.01

    used = 0
    for midi_path in midi_files:
        try:
            info = load_midi(str(midi_path))
            if len(info.notes) < 10:
                continue

            # Detectar tonalidad del MIDI del corpus
            corp_tonic, corp_mode = detect_key(info.notes)

            # Puntuar calidad
            q = score_midi_quality(info, corp_tonic, corp_mode)
            if q < min_score:
                continue

            # Extraer acordes con template matching simple (sin HMM para bootstrap)
            res = max(0.5, info.beats_per_bar / 2)
            n_slots = max(1, int(info.total_beats / res))
            prev_state = None

            for slot in range(n_slots):
                t0 = slot * res
                t1 = t0 + res
                chroma = chroma_window(info.notes, t0, t1)
                if chroma.sum() < 0.01:
                    continue

                # Transponer al tonic de referencia para unificar
                shift = (tonic_pc - corp_tonic) % 12
                chroma_shifted = np.roll(chroma, shift)

                root, qual, score = template_match_window(
                    chroma_shifted, tonic_pc, mode, diatonic_bonus=0.1)

                # Encontrar índice del estado
                try:
                    state_idx = ALL_CHORD_TEMPLATES.index((root, qual))
                except ValueError:
                    continue

                # Acumular emisión
                emit_counts[state_idx] += chroma_shifted

                # Acumular transición
                if prev_state is not None:
                    trans_counts[prev_state, state_idx] += 1.0

                prev_state = state_idx

            used += 1
            if verbose and used % 20 == 0:
                print(f"  [corpus] {used} MIDIs procesados de {len(midi_files)}...")

        except Exception:
            continue

    if verbose:
        print(f"  [corpus] Total usados para entrenamiento: {used}/{len(midi_files)}")

    # Normalizar
    for i in range(n):
        row_sum = trans_counts[i].sum()
        if row_sum > 0:
            trans_counts[i] /= row_sum
        emit_sum = emit_counts[i].sum()
        if emit_sum > 0:
            emit_counts[i] /= emit_sum

    # Interpolar con el modelo por defecto para regularizar
    default_trans = build_default_transition_matrix(tonic_pc, mode)
    default_emit  = build_emission_matrix(tonic_pc, mode)
    alpha = min(0.7, used / 50.0)  # más corpus → más peso al aprendido
    final_trans = alpha * trans_counts + (1 - alpha) * default_trans
    final_emit  = alpha * emit_counts  + (1 - alpha) * default_emit

    return final_trans, final_emit


def save_model(path: str, transition: np.ndarray, emission: np.ndarray,
               tonic_pc: int, mode: str) -> None:
    """Guarda el modelo HMM en JSON."""
    data = {
        'tonic_pc': tonic_pc,
        'mode': mode,
        'chord_templates': [[r, q] for r, q in ALL_CHORD_TEMPLATES],
        'transition': transition.tolist(),
        'emission': emission.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_model(path: str) -> Tuple[int, str, np.ndarray, np.ndarray]:
    """Carga un modelo HMM desde JSON. Devuelve (tonic_pc, mode, transition, emission)."""
    with open(path) as f:
        data = json.load(f)
    return (
        data['tonic_pc'],
        data['mode'],
        np.array(data['transition']),
        np.array(data['emission']),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 5 — EXPORTACIÓN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def _beats_to_ticks(beats: float, tpb: int) -> int:
    return max(0, int(round(beats * tpb)))


def _chord_to_midi_notes(root_pc: int, quality: str,
                          octave: int = 4) -> List[int]:
    """Convierte un acorde a notas MIDI en la octava especificada."""
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS['maj'])
    notes = []
    for iv in intervals:
        pitch = octave * 12 + root_pc + iv
        # Mantener en rango idiomático [36, 84]
        while pitch < 36:
            pitch += 12
        while pitch > 84:
            pitch -= 12
        notes.append(pitch)
    return notes


def monophonize_melody(notes: List[Tuple]) -> List[Tuple]:
    """
    Reduce una lista de notas de melodía a estrictamente monofónica.

    En cada instante con dos o más notas simultáneas conserva únicamente
    la más aguda (voz principal), truncando o eliminando las demás.
    Las notas que empiezan antes pero siguen sonando cuando arranca una
    nueva nota más aguda se truncan al onset de la nueva nota.
    """
    if not notes:
        return notes

    result = sorted(notes, key=lambda n: (n[0], -n[1]))  # orden: tiempo asc, pitch desc
    mono: List[Tuple] = []

    for note in result:
        s, pitch, dur, vel, ch, tr = note
        end = s + dur

        # Truncar la última nota añadida si solapa con esta
        if mono:
            ps, pp, pd, pv, pch, ptr = mono[-1]
            p_end = ps + pd
            if p_end > s:
                # La nota anterior solapa: truncarla al inicio de esta
                new_dur = s - ps
                if new_dur > 0.01:  # conservar solo si queda duración mínima
                    mono[-1] = (ps, pp, new_dur, pv, pch, ptr)
                else:
                    mono.pop()  # demasiado corta: eliminar directamente

        mono.append((s, pitch, dur, vel, ch, tr))

    return mono


def build_melody_track(melody_notes: List[Tuple],
                        tpb: int,
                        velocity: int = 90,
                        target_octave: Optional[int] = None) -> mido.MidiTrack:
    """Construye una pista MIDI de melodía."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Melody', time=0))
    track.append(mido.Message('program_change', channel=0, program=40, time=0))  # violin

    if not melody_notes:
        return track

    # Ordenar y construir eventos absolutos
    events: List[Tuple[int, str, int, int]] = []  # (abs_tick, type, note, vel)
    for (s, pitch, dur, vel, ch, tr) in melody_notes:
        # Transponer a la octava objetivo si se especifica
        if target_octave is not None:
            current_octave = pitch // 12
            if current_octave != target_octave:
                pitch = target_octave * 12 + (pitch % 12)
                # Mantener en rango
                while pitch < 36:  pitch += 12
                while pitch > 96:  pitch -= 12

        on_tick  = _beats_to_ticks(s, tpb)
        off_tick = _beats_to_ticks(s + dur, tpb)
        events.append((on_tick,  'on',  pitch, min(velocity, 127)))
        events.append((off_tick, 'off', pitch, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    prev_tick = 0
    for (tick, etype, pitch, vel) in events:
        delta = max(0, tick - prev_tick)
        if etype == 'on':
            track.append(mido.Message('note_on', channel=0,
                                       note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', channel=0,
                                       note=pitch, velocity=0, time=delta))
        prev_tick = tick

    return track


def build_chords_track(chords: List[Dict],
                        tpb: int,
                        channel: int = 1,
                        octave: int = 4,
                        velocity: int = 70) -> mido.MidiTrack:
    """Construye una pista MIDI de acordes en bloque."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Chords', time=0))
    track.append(mido.Message('program_change', channel=channel, program=0, time=0))  # piano

    if not chords:
        return track

    events: List[Tuple[int, str, int, int]] = []
    for chord in chords:
        notes = _chord_to_midi_notes(chord['root_pc'], chord['quality'], octave)
        on_tick  = _beats_to_ticks(chord['start_beat'], tpb)
        off_tick = _beats_to_ticks(chord['start_beat'] + chord['duration_beats'], tpb)
        for note in notes:
            events.append((on_tick,  'on',  note, velocity))
            events.append((off_tick, 'off', note, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    prev_tick = 0
    for (tick, etype, pitch, vel) in events:
        delta = max(0, tick - prev_tick)
        if etype == 'on':
            track.append(mido.Message('note_on', channel=channel,
                                       note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', channel=channel,
                                       note=pitch, velocity=0, time=delta))
        prev_tick = tick

    return track


def export_split_file(melody_notes: List[Tuple],
                       chords: List[Dict],
                       tpb: int, tempo_us: int,
                       output_path: str,
                       melody_octave: Optional[int],
                       chord_octave: int,
                       velocity_melody: int, velocity_chords: int) -> None:
    """Exporta un único MIDI con dos pistas: ch0=melodía, ch1=acordes."""
    mid = mido.MidiFile(ticks_per_beat=tpb)

    # Track 0: tempo y firma de tiempo
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    mid.tracks.append(meta_track)

    melody_track = build_melody_track(melody_notes, tpb,
                                       velocity=velocity_melody,
                                       target_octave=melody_octave)
    chords_track  = build_chords_track(chords, tpb, channel=1,
                                        octave=chord_octave,
                                        velocity=velocity_chords)
    mid.tracks.append(melody_track)
    mid.tracks.append(chords_track)
    mid.save(output_path)


def export_two_files(melody_notes: List[Tuple],
                      chords: List[Dict],
                      tpb: int, tempo_us: int,
                      melody_path: str, chords_path: str,
                      melody_octave: Optional[int],
                      chord_octave: int,
                      velocity_melody: int, velocity_chords: int) -> None:
    """Exporta dos ficheros MIDI separados: uno para melodía y otro para acordes."""
    # MIDI de melodía
    mid_mel = mido.MidiFile(ticks_per_beat=tpb)
    meta_mel = mido.MidiTrack()
    meta_mel.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    mid_mel.tracks.append(meta_mel)
    mid_mel.tracks.append(build_melody_track(melody_notes, tpb,
                                              velocity=velocity_melody,
                                              target_octave=melody_octave))
    mid_mel.save(melody_path)

    # MIDI de acordes
    mid_ch = mido.MidiFile(ticks_per_beat=tpb)
    meta_ch = mido.MidiTrack()
    meta_ch.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    mid_ch.tracks.append(meta_ch)
    mid_ch.tracks.append(build_chords_track(chords, tpb, channel=0,
                                             octave=chord_octave,
                                             velocity=velocity_chords))
    mid_ch.save(chords_path)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTE
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(info: MidiInfo, tonic_pc: int, mode: str,
                  melody_notes: List[Tuple], chords: List[Dict],
                  corpus_used: bool) -> Dict:
    """Genera un reporte JSON con el análisis detallado."""
    chord_names = [f"{c['name']} ({c['start_beat']:.1f}b, {c['duration_beats']:.1f}b)"
                   for c in chords]
    avg_score = float(np.mean([c['score'] for c in chords])) if chords else 0.0

    return {
        'input': {
            'bpm':           info.bpm,
            'beats_per_bar': info.beats_per_bar,
            'total_beats':   info.total_beats,
            'total_bars':    info.total_bars,
            'n_tracks':      info.n_tracks,
            'n_channels':    info.n_channels,
            'n_notes':       len(info.notes),
        },
        'key': {
            'tonic':  PITCH_NAMES[tonic_pc],
            'mode':   mode,
            'tonic_pc': tonic_pc,
        },
        'melody': {
            'n_notes':      len(melody_notes),
            'pitch_range':  [min(n[1] for n in melody_notes),
                             max(n[1] for n in melody_notes)] if melody_notes else [],
            'avg_interval': float(np.mean([abs(melody_notes[i][1] - melody_notes[i-1][1])
                                           for i in range(1, len(melody_notes))]))
                             if len(melody_notes) > 1 else 0.0,
        },
        'chords': {
            'n_chords':     len(chords),
            'avg_score':    avg_score,
            'sequence':     chord_names,
            'unique_chords': list({c['name'] for c in chords}),
        },
        'corpus_mode': corpus_used,
        'generator': 'midi_split_harmony.py v1.0',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Separa melodía y acordes de un MIDI de entrada.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Argumento posicional: fichero MIDI de entrada (opcional si --train-corpus) ──
    parser.add_argument('input', nargs='?', default=None,
                        help='MIDI de entrada a separar')

    # ── Opciones de análisis ──────────────────────────────────────────────────
    parser.add_argument('--out-dir',         type=str,   default=None,
                        help='Carpeta de salida (default: junto al MIDI de entrada)')
    parser.add_argument('--output',          type=str,   default=None,
                        help='Nombre base de salida (default: stem del fichero)')
    parser.add_argument('--mono-melody',     action='store_true',
                        help='Reducir la melodía a monofónica estricta: en cada '
                             'instante con varias notas simultáneas conserva solo '
                             'la más aguda')
    parser.add_argument('--split-files',     action='store_true',
                        help='Genera dos ficheros separados en vez de uno con dos pistas')
    parser.add_argument('--key',             type=str,   default=None, metavar='KEY',
                        help='Tonalidad, e.g. "A minor" o "C" '
                             '(default: autodetectada; también fija la tónica de '
                             'normalización en --train-corpus)')
    parser.add_argument('--beats-per-bar',   type=int,   default=None,
                        help='Pulsos por compás (default: detectado del MIDI)')
    parser.add_argument('--tempo',           type=float, default=None,
                        help='Tempo BPM de salida (default: detectado)')
    parser.add_argument('--chord-res',       type=float, default=1.0, metavar='BEATS',
                        help='Resolución mínima de acorde en beats (default: 1.0)')
    parser.add_argument('--melody-ch',       type=int,   default=None, metavar='N',
                        help='Canal MIDI forzado para melodía (0-15)')
    parser.add_argument('--harmony-ch',      type=int,   default=None, metavar='N',
                        help='Canal MIDI forzado para armonía (0-15)')
    parser.add_argument('--chord-octave',    type=int,   default=4,   metavar='N',
                        help='Octava base de acordes de salida (default: 4)')
    parser.add_argument('--melody-octave',   type=int,   default=None, metavar='N',
                        help='Octava de transposición de melodía (default: sin cambio)')
    parser.add_argument('--velocity-melody', type=int,   default=90,  metavar='N',
                        help='Velocity de notas de melodía (default: 90)')
    parser.add_argument('--velocity-chords', type=int,   default=70,  metavar='N',
                        help='Velocity de acordes (default: 70)')

    # ── Opciones de modelo HMM ────────────────────────────────────────────────
    parser.add_argument('--train-corpus',    type=str,   default=None, metavar='DIR',
                        help='Entrenar el modelo HMM desde este directorio de MIDIs '
                             'y guardarlo con --save-model. No requiere fichero de entrada.')
    parser.add_argument('--save-model',      type=str,   default=None, metavar='FILE',
                        help='Guardar el modelo HMM entrenado en este fichero JSON')
    parser.add_argument('--load-model',      type=str,   default=None, metavar='FILE',
                        help='Usar este modelo HMM (JSON) en lugar del determinista')
    parser.add_argument('--corpus',          type=str,   default=None, metavar='DIR',
                        help='Entrenar y aplicar en un solo paso: aprende desde este '
                             'directorio y analiza el fichero de entrada')
    parser.add_argument('--min-corpus-score', type=float, default=0.55, metavar='F',
                        help='Umbral de calidad armónica para incluir un MIDI del '
                             'corpus en el entrenamiento (default: 0.55)')

    # ── Control ───────────────────────────────────────────────────────────────
    parser.add_argument('--report',  action='store_true',
                        help='Guardar reporte JSON con análisis detallado')
    parser.add_argument('--verbose', action='store_true',
                        help='Información detallada del proceso')

    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════════
    # MODO --train-corpus : entrenar y guardar el modelo sin analizar ningún MIDI
    # ══════════════════════════════════════════════════════════════════════════

    if args.train_corpus:
        if not os.path.isdir(args.train_corpus):
            print(f"[ERROR] El directorio de corpus no existe: {args.train_corpus}")
            sys.exit(1)
        if not args.save_model:
            print("[ERROR] --train-corpus requiere --save-model <fichero.json>")
            sys.exit(1)

        tonic_pc, mode = parse_key_string(args.key) if args.key else (0, 'major')

        if args.verbose:
            print(f"\n╔══ MIDI SPLIT HARMONY v1.0 — ENTRENAMIENTO ═════════════════╗")
            print(f"║  Corpus:         {args.train_corpus}")
            print(f"║  Modelo salida:  {args.save_model}")
            print(f"║  Tónica ref:     {PITCH_NAMES[tonic_pc]} {mode}")
            print(f"║  Umbral calidad: {args.min_corpus_score}")
            print(f"╠══ Entrenando...")
        else:
            print(f"Entrenando modelo desde: {args.train_corpus}")

        transition, emission = learn_from_corpus(
            args.train_corpus,
            tonic_pc=tonic_pc,
            mode=mode,
            min_score=args.min_corpus_score,
            verbose=args.verbose,
        )
        save_model(args.save_model, transition, emission, tonic_pc, mode)

        if args.verbose:
            print(f"╚══ Modelo guardado en: {args.save_model} ✓")
        else:
            print(f"  → Modelo guardado en: {args.save_model}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODO ANÁLISIS : separar melodía y acordes de un MIDI de entrada
    # ══════════════════════════════════════════════════════════════════════════

    if not args.input:
        parser.print_help()
        print("\n[ERROR] Proporciona un MIDI de entrada, "
              "o usa --train-corpus DIR --save-model FILE para entrenar un modelo.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERROR] No se encuentra el fichero de entrada: {args.input}")
        sys.exit(1)

    input_path = Path(args.input)
    stem = args.output or input_path.stem

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] PRE-ANÁLISIS ─────────────────────────────────────────────────────

    if args.verbose:
        print(f"\n╔══ MIDI SPLIT HARMONY v1.0 ══════════════════════════════════╗")
        print(f"║  Entrada: {input_path.name}")

    info = load_midi(str(input_path))

    if args.beats_per_bar:
        info.beats_per_bar = args.beats_per_bar
    if args.tempo:
        info.bpm = args.tempo
        info.tempo_us = int(60_000_000 / args.tempo)

    if args.verbose:
        print(f"║  Pistas: {info.n_tracks}  |  Canales: {info.n_channels}  "
              f"|  Notas: {len(info.notes)}")
        print(f"║  Tempo: {info.bpm:.1f} BPM  |  Compás: {info.beats_per_bar}/4  "
              f"|  Duración: {info.total_bars:.1f} compases")

    if not info.notes:
        print("[ERROR] El MIDI no contiene notas.")
        sys.exit(1)

    # ── Detección de tonalidad ────────────────────────────────────────────────

    if args.key:
        tonic_pc, mode = parse_key_string(args.key)
        if args.verbose:
            print(f"║  Tonalidad (forzada): {PITCH_NAMES[tonic_pc]} {mode}")
    else:
        tonic_pc, mode = detect_key(info.notes)
        if args.verbose:
            print(f"║  Tonalidad (detectada): {PITCH_NAMES[tonic_pc]} {mode}")

    # ── [2] SEPARACIÓN MELODÍA / ARMONÍA ─────────────────────────────────────

    if args.verbose:
        print(f"╠══ [2] Separando melodía / armonía...")

    melody_notes, ch_classification = extract_skyline(
        info.notes, tonic_pc, mode,
        forced_melody_ch=args.melody_ch,
        verbose=args.verbose,
    )
    harmony_notes = extract_harmony_notes(info.notes, melody_notes, ch_classification)

    if args.verbose:
        print(f"║  Melodía: {len(melody_notes)} notas  |  "
              f"Armonía: {len(harmony_notes)} notas")

    # ── [3] MODELO HMM ───────────────────────────────────────────────────────

    corpus_used    = False
    hmm_transition = None
    hmm_emission   = None

    if args.load_model:
        if os.path.exists(args.load_model):
            if args.verbose:
                print(f"╠══ [3a] Cargando modelo HMM: {args.load_model}")
            _, _, hmm_transition, hmm_emission = load_model(args.load_model)
            corpus_used = True
        else:
            print(f"[AVISO] No se encuentra el modelo: {args.load_model}. "
                  f"Usando modo determinista.")

    elif args.corpus:
        # --corpus DIR: entrenar y analizar en un solo paso
        if os.path.isdir(args.corpus):
            if args.verbose:
                print(f"╠══ [3a] Aprendiendo desde corpus: {args.corpus}")
            hmm_transition, hmm_emission = learn_from_corpus(
                args.corpus, tonic_pc, mode,
                min_score=args.min_corpus_score,
                verbose=args.verbose,
            )
            corpus_used = True
            if args.save_model:
                save_model(args.save_model, hmm_transition, hmm_emission,
                           tonic_pc, mode)
                if args.verbose:
                    print(f"║  Modelo guardado en: {args.save_model}")
        else:
            print(f"[AVISO] El directorio de corpus no existe: {args.corpus}. "
                  f"Usando modo determinista.")

    hmm = HarmonicHMM(tonic_pc, mode, hmm_transition, hmm_emission)

    # ── [3b] EXTRACCIÓN DE ACORDES ────────────────────────────────────────────

    if args.verbose:
        mode_label = "corpus+HMM" if corpus_used else "determinista+HMM"
        print(f"╠══ [3b] Extrayendo acordes ({mode_label})...")

    chords = extract_chords_hmm(
        harmony_notes,
        total_beats=info.total_beats,
        beats_per_bar=info.beats_per_bar,
        tonic_pc=tonic_pc,
        mode=mode,
        chord_resolution=args.chord_res,
        hmm=hmm,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"║  Acordes extraídos: {len(chords)} slots")

    # ── [4] POST-LIMPIEZA ─────────────────────────────────────────────────────

    if args.verbose:
        print(f"╠══ [4] Post-limpieza...")

    chords = merge_short_chords(chords, min_beats=args.chord_res * 0.5)
    chords = smooth_chord_sequence(chords, tonic_pc, mode)
    chords = quantize_chords_to_grid(chords, info.beats_per_bar, args.chord_res)

    if args.mono_melody:
        n_before = len(melody_notes)
        melody_notes = monophonize_melody(melody_notes)
        if args.verbose:
            print(f"║  Monofónica: {n_before} → {len(melody_notes)} notas "
                  f"({n_before - len(melody_notes)} voces dobladas eliminadas)")

    if args.verbose:
        avg_score = float(np.mean([c['score'] for c in chords])) if chords else 0.0
        print(f"║  Acordes finales: {len(chords)}  |  Score medio: {avg_score:.3f}")
        sample = chords[:8]
        names = ' → '.join(c['name'] for c in sample)
        print(f"║  Muestra: {names}{'...' if len(chords) > 8 else ''}")

    # ── [5] EXPORTACIÓN ──────────────────────────────────────────────────────

    if args.verbose:
        print(f"╠══ [5] Exportando resultados...")

    if args.split_files:
        melody_path = str(out_dir / f"{stem}_melody.mid")
        chords_path = str(out_dir / f"{stem}_chords.mid")
        export_two_files(
            melody_notes, chords,
            info.tpb, info.tempo_us,
            melody_path, chords_path,
            melody_octave=args.melody_octave,
            chord_octave=args.chord_octave,
            velocity_melody=args.velocity_melody,
            velocity_chords=args.velocity_chords,
        )
        print(f"  → Melodía: {melody_path}")
        print(f"  → Acordes: {chords_path}")
    else:
        split_path = str(out_dir / f"{stem}_split.mid")
        export_split_file(
            melody_notes, chords,
            info.tpb, info.tempo_us,
            split_path,
            melody_octave=args.melody_octave,
            chord_octave=args.chord_octave,
            velocity_melody=args.velocity_melody,
            velocity_chords=args.velocity_chords,
        )
        print(f"  → Split MIDI: {split_path}")

    # ── Reporte opcional ─────────────────────────────────────────────────────

    if args.report:
        report = build_report(info, tonic_pc, mode, melody_notes, chords, corpus_used)
        report_path = str(out_dir / f"{stem}_split_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  → Reporte: {report_path}")

    if args.verbose:
        print(f"╚══ Completado ✓")


if __name__ == '__main__':
    main()
