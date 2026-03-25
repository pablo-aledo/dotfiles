#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SECTION SCORE  v1.0                                  ║
║     Detección de secciones musicalmente incoherentes en ficheros MIDI       ║
║                                                                              ║
║  Dado un MIDI de entrada, segmenta la obra, construye un perfil global      ║
║  y puntúa cada segmento en cuatro ejes de coherencia musical:               ║
║    · Tonal     — compatibilidad melodía/armonía con la tonalidad global     ║
║    · Estructural — distancia del segmento al perfil medio de la obra        ║
║    · Vecinal    — incompatibilidad con los segmentos adyacentes             ║
║    · Rítmico   — desviación del patrón rítmico dominante                   ║
║                                                                              ║
║  SEGMENTACIÓN (--granularity):                                               ║
║    fixed N    Ventanas fijas de N compases (default: fixed 4)               ║
║    form       Fronteras detectadas por SSM + novelty (harvester)            ║
║    texture    Fronteras por cambios de textura (harvester)                  ║
║    cadence    Fronteras en cadencias detectadas con music21                 ║
║    hybrid     Unión de fronteras form + texture                             ║
║                                                                              ║
║  PESOS (--weights):                                                          ║
║    balanced   Tonal 0.30 · Estructural 0.25 · Vecinal 0.25 · Rítmico 0.20 ║
║    harmonic   Tonal 0.50 · Estructural 0.20 · Vecinal 0.20 · Rítmico 0.10 ║
║    neighbor   Tonal 0.20 · Estructural 0.15 · Vecinal 0.50 · Rítmico 0.15 ║
║    structural Tonal 0.15 · Estructural 0.50 · Vecinal 0.20 · Rítmico 0.15 ║
║    rhythmic   Tonal 0.15 · Estructural 0.20 · Vecinal 0.20 · Rítmico 0.45 ║
║    custom W1,W2,W3,W4   (cuatro floats que suman 1.0)                      ║
║                                                                              ║
║  TRACKS (--track-mode):                                                      ║
║    all        Todas las pistas combinadas (default)                         ║
║    melody     Solo la pista con más notas / pitch más agudo                 ║
║    auto       Detección por programa MIDI (excluye percusión/bajo)          ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    --out-json FILE    JSON detallado por segmento                           ║
║    --out-text FILE    Informe legible (también por stdout si --verbose)     ║
║    --out-midi FILE    MIDI original con markers en segmentos anómalos       ║
║    --remove N         Genera un MIDI con los N segmentos más anómalos       ║
║                       eliminados; los segmentos restantes se concatenan     ║
║                       sin huecos. El fichero se guarda junto al MIDI de    ║
║                       entrada con sufijo _pruned_<N>.mid (o en la ruta     ║
║                       indicada con --remove-out FILE)                       ║
║    --remove-out FILE  Ruta de salida del MIDI podado (opcional)            ║
║    --top N            Nº de segmentos a reportar (default: 5)               ║
║    --threshold F      Score mínimo de anomalía 0-1 (default: 0.40)         ║
║                                                                              ║
║  OTROS:                                                                      ║
║    --chords FILE      Fichero JSON/MIDI de acordes (opcional; si no se      ║
║                       indica, se infieren automáticamente del MIDI)         ║
║    --novelty-threshold F  Sensibilidad de detección de fronteras (0-1)     ║
║    --verbose          Detalle completo por criterio                          ║
║                                                                              ║
║  USO:                                                                        ║
║    python section_score.py obra.mid                                          ║
║    python section_score.py obra.mid --granularity form --weights harmonic   ║
║    python section_score.py obra.mid --granularity fixed 8 --top 3          ║
║    python section_score.py obra.mid --granularity hybrid --weights custom 0.4,0.2,0.3,0.1
║    python section_score.py obra.mid --out-json r.json --out-midi r.mid     ║
║    python section_score.py obra.mid --track-mode melody --verbose           ║
║    python section_score.py obra.mid --chords acordes.json                  ║
║    python section_score.py obra.mid --remove 3                              ║
║    python section_score.py obra.mid --remove 2 --remove-out limpia.mid     ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21, numpy, scipy, sklearn                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import textwrap
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── mido ─────────────────────────────────────────────────────────────────────
try:
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[ERROR] mido no disponible. Instálalo con: pip install mido")
    sys.exit(1)

# ── music21 ───────────────────────────────────────────────────────────────────
try:
    import music21 as m21
    from music21 import converter as m21converter, roman, stream
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Los modos 'cadence' y el análisis armónico profundo estarán limitados.")

# ── scipy / sklearn ───────────────────────────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DINÁMICA DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).parent

def _load_module(name: str):
    """Carga un módulo del ecosistema situado en el mismo directorio."""
    path = _SCRIPT_DIR / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  [AVISO] No se pudo cargar {name}.py: {e}")
        return None

_harvester    = _load_module("harvester")
_mscz2vec     = _load_module("mscz2vec")
_adapter      = _load_module("melody_adapter")
_completer    = _load_module("completer")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

WEIGHT_PRESETS = {
    "balanced":   (0.30, 0.25, 0.25, 0.20),
    "harmonic":   (0.50, 0.20, 0.20, 0.10),
    "neighbor":   (0.20, 0.15, 0.50, 0.15),
    "structural": (0.15, 0.50, 0.20, 0.15),
    "rhythmic":   (0.15, 0.20, 0.20, 0.45),
}

# Programas MIDI considerados "percusión" o "bajo" (para track-mode auto)
_PERCUSSION_PROGRAMS = set(range(112, 120))   # soundfx / percus sintetizada
_BASS_PROGRAMS       = set(range(32, 40))     # bass

# Canal GM de percusión
_DRUM_CHANNEL = 9


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI BÁSICAS (independientes de harvester)
# ══════════════════════════════════════════════════════════════════════════════

def _midi_to_absolute(mid: MidiFile) -> list:
    """Convierte todos los tracks a lista (abs_tick, track_idx, msg)."""
    events = []
    for ti, track in enumerate(mid.tracks):
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            events.append((abs_t, ti, msg))
    events.sort(key=lambda x: x[0])
    return events


def _get_tempo(mid: MidiFile) -> int:
    """Devuelve el primer tempo encontrado en microsegundos (default 500000)."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500_000


def _get_time_signature(mid: MidiFile) -> Tuple[int, int]:
    """Devuelve (numerator, denominator) de la primera firma de compás."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def _estimate_bar_ticks(mid: MidiFile) -> int:
    """Estima la duración de un compás en ticks."""
    num, den = _get_time_signature(mid)
    beats_per_bar = num * (4 / den)
    return max(1, int(beats_per_bar * mid.ticks_per_beat))


def _get_total_ticks(mid: MidiFile) -> int:
    events = _midi_to_absolute(mid)
    return max((e[0] for e in events), default=0)


def _extract_note_events(mid: MidiFile, channels: Optional[set] = None,
                         tracks: Optional[set] = None) -> list:
    """
    Extrae notas como (start_tick, end_tick, pitch, velocity, channel, track_idx).
    Filtra opcionalmente por canal o índice de track.
    """
    notes = []
    active = {}
    for abs_t, ti, msg in _midi_to_absolute(mid):
        if tracks is not None and ti not in tracks:
            continue
        if not hasattr(msg, 'channel'):
            continue
        if channels is not None and msg.channel not in channels:
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            active[(ti, msg.channel, msg.note)] = (abs_t, msg.velocity)
        elif msg.type in ('note_off', 'note_on') and \
                (msg.type == 'note_off' or msg.velocity == 0):
            key = (ti, msg.channel, msg.note)
            if key in active:
                start_t, vel = active.pop(key)
                notes.append((start_t, abs_t, msg.note, vel, msg.channel, ti))
    total = _get_total_ticks(mid)
    for (ti, ch, pitch), (start_t, vel) in active.items():
        notes.append((start_t, total, pitch, vel, ch, ti))
    notes.sort(key=lambda x: x[0])
    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN DE TRACKS SEGÚN --track-mode
# ══════════════════════════════════════════════════════════════════════════════

def _select_tracks(mid: MidiFile, track_mode: str) -> set:
    """
    Devuelve el conjunto de índices de track a analizar.
    """
    n = len(mid.tracks)
    all_tracks = set(range(n))

    if track_mode == "all":
        return all_tracks

    # Recopilar info por track: programa, canal, nº notas, pitch medio
    track_info = {}
    for ti, track in enumerate(mid.tracks):
        program = 0
        channels = set()
        pitches = []
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'program_change':
                program = msg.program
                channels.add(msg.channel)
            elif msg.type == 'note_on' and msg.velocity > 0:
                channels.add(msg.channel)
                pitches.append(msg.note)
        is_drum = _DRUM_CHANNEL in channels
        track_info[ti] = {
            'program': program,
            'channels': channels,
            'is_drum': is_drum,
            'n_notes': len(pitches),
            'mean_pitch': float(np.mean(pitches)) if pitches else 0.0,
        }

    if track_mode == "melody":
        # Track con mayor pitch medio entre los que tienen notas
        candidates = [(ti, info) for ti, info in track_info.items()
                      if info['n_notes'] > 0 and not info['is_drum']]
        if not candidates:
            return all_tracks
        best = max(candidates, key=lambda x: x[1]['mean_pitch'])
        return {best[0]}

    if track_mode == "auto":
        # Excluir percusión y canales de bajo; conservar el resto
        result = set()
        for ti, info in track_info.items():
            if info['is_drum']:
                continue
            if info['program'] in _PERCUSSION_PROGRAMS:
                continue
            if info['program'] in _BASS_PROGRAMS:
                continue
            if info['n_notes'] > 0:
                result.add(ti)
        return result if result else all_tracks

    return all_tracks


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _segment_fixed(mid: MidiFile, window_bars: int) -> List[Tuple[int, int]]:
    """Ventanas fijas de window_bars compases."""
    bar_ticks = _estimate_bar_ticks(mid)
    total = _get_total_ticks(mid)
    segments = []
    start = 0
    step = bar_ticks * window_bars
    while start < total:
        end = min(start + step, total)
        segments.append((start, end))
        start += step
    return segments


def _segment_form(mid: MidiFile, notes: list, novelty_threshold: float) -> List[Tuple[int, int]]:
    """Fronteras por SSM + novelty (harvester)."""
    if _harvester is None or not SCIPY_OK:
        print("  [AVISO] harvester/scipy no disponibles; usando segmentación fixed 4.")
        return _segment_fixed(mid, 4)

    bar_ticks  = _harvester.estimate_bar_ticks(mid)
    total      = _harvester.get_total_ticks(mid)

    try:
        result = _harvester.detect_form_boundaries(
            notes, bar_ticks, total,
            kernel_size=8,
            threshold=novelty_threshold
        )
        boundaries = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        print(f"  [AVISO] detect_form_boundaries falló ({e}); usando fixed 4.")
        return _segment_fixed(mid, 4)

    boundaries = sorted(set(boundaries))
    segments = [(boundaries[i], boundaries[i+1])
                for i in range(len(boundaries) - 1)
                if boundaries[i+1] > boundaries[i]]
    return segments or _segment_fixed(mid, 4)


def _segment_texture(mid: MidiFile, notes: list, novelty_threshold: float) -> List[Tuple[int, int]]:
    """Fronteras por cambios de textura (harvester)."""
    if _harvester is None or not SCIPY_OK:
        print("  [AVISO] harvester no disponible; usando fixed 4.")
        return _segment_fixed(mid, 4)

    bar_ticks = _harvester.estimate_bar_ticks(mid)
    total     = _harvester.get_total_ticks(mid)
    tpb       = mid.ticks_per_beat

    try:
        boundaries = _harvester.detect_texture_changes(
            notes, bar_ticks, total, tpb,
            threshold=novelty_threshold
        )
    except Exception as e:
        print(f"  [AVISO] detect_texture_changes falló ({e}); usando fixed 4.")
        return _segment_fixed(mid, 4)

    boundaries = sorted(set(boundaries))
    segments = [(boundaries[i], boundaries[i+1])
                for i in range(len(boundaries) - 1)
                if boundaries[i+1] > boundaries[i]]
    return segments or _segment_fixed(mid, 4)


def _segment_cadence(mid: MidiFile, midi_path: str) -> List[Tuple[int, int]]:
    """Fronteras en cadencias detectadas por music21."""
    if not MUSIC21_OK:
        print("  [AVISO] music21 no disponible; usando fixed 4.")
        return _segment_fixed(mid, 4)

    try:
        score = m21converter.parse(midi_path)
        cadences = _harvester.detect_cadences_m21(score) if _harvester else []
    except Exception as e:
        print(f"  [AVISO] detect_cadences_m21 falló ({e}); usando fixed 4.")
        return _segment_fixed(mid, 4)

    tpb   = mid.ticks_per_beat
    total = _get_total_ticks(mid)

    boundary_ticks = sorted({0} |
        {int(offset * tpb) for offset, *_ in cadences} |
        {total})

    segments = [(boundary_ticks[i], boundary_ticks[i+1])
                for i in range(len(boundary_ticks) - 1)
                if boundary_ticks[i+1] > boundary_ticks[i]]
    return segments or _segment_fixed(mid, 4)


def _merge_boundaries(*boundary_lists) -> List[int]:
    """Une varias listas de ticks de frontera y elimina duplicados cercanos."""
    merged = sorted(set(t for lst in boundary_lists for t in lst))
    # Eliminar fronteras demasiado próximas (< 480 ticks = 1 beat a 120 BPM)
    MIN_GAP = 480
    clean = [merged[0]]
    for t in merged[1:]:
        if t - clean[-1] >= MIN_GAP:
            clean.append(t)
    return clean


def build_segments(mid: MidiFile, midi_path: str, notes: list,
                   granularity: str, window_bars: int,
                   novelty_threshold: float) -> List[Tuple[int, int]]:
    """
    Devuelve lista de (start_tick, end_tick) según la granularidad elegida.
    """
    g = granularity
    if g == "fixed":
        segs = _segment_fixed(mid, window_bars)
    elif g == "form":
        segs = _segment_form(mid, notes, novelty_threshold)
    elif g == "texture":
        segs = _segment_texture(mid, notes, novelty_threshold)
    elif g == "cadence":
        segs = _segment_cadence(mid, midi_path)
    elif g == "hybrid":
        form_segs    = _segment_form(mid, notes, novelty_threshold)
        texture_segs = _segment_texture(mid, notes, novelty_threshold)
        form_b    = [s for s, _ in form_segs] + [form_segs[-1][1]] if form_segs else [0]
        texture_b = [s for s, _ in texture_segs] + [texture_segs[-1][1]] if texture_segs else [0]
        merged = _merge_boundaries(form_b, texture_b)
        segs = [(merged[i], merged[i+1]) for i in range(len(merged)-1)
                if merged[i+1] > merged[i]]
    else:
        segs = _segment_fixed(mid, 4)

    return segs


# ══════════════════════════════════════════════════════════════════════════════
#  DESCRIPTOR DE COMPÁS / SEGMENTO (propio, no depende de harvester)
# ══════════════════════════════════════════════════════════════════════════════

def _descriptor_for_notes(notes_in_seg: list, bar_ticks: int) -> np.ndarray:
    """
    Vector de 8 características para un conjunto de notas.
    Compatible con harvester.descriptor_for_window pero autónomo.
    """
    if not notes_in_seg:
        return np.zeros(8)

    pitches    = [n[2] for n in notes_in_seg]
    velocities = [n[3] for n in notes_in_seg]
    durations  = [n[1] - n[0] for n in notes_in_seg]
    intervals  = np.diff(pitches) if len(pitches) > 1 else [0]

    density    = len(notes_in_seg) / max(bar_ticks, 1)
    mean_pitch = np.mean(pitches)
    pitch_std  = np.std(pitches)
    mean_vel   = np.mean(velocities)
    vel_var    = np.var(velocities)
    mean_dur   = np.mean(durations)
    prop_asc   = np.sum(np.array(intervals) > 0) / max(len(intervals), 1)
    prop_jump  = np.sum(np.abs(np.array(intervals)) > 5) / max(len(intervals), 1)

    return np.array([
        density,
        mean_pitch / 127,
        pitch_std / 64,
        mean_vel / 127,
        vel_var / (127**2),
        mean_dur / max(bar_ticks, 1),
        prop_asc,
        prop_jump,
    ])


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia coseno entre dos vectores (0=idénticos, 1=ortogonales)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ══════════════════════════════════════════════════════════════════════════════
#  PERFIL GLOBAL DE LA OBRA
# ══════════════════════════════════════════════════════════════════════════════

def build_global_profile(mid: MidiFile, midi_path: str, notes: list,
                          verbose: bool = False) -> dict:
    """
    Extrae el perfil global de la pieza:
      - ADN (tonalidad, tempo, escala) vía completer.extract_dna_from_midi
      - Vector de descriptor simple (harvester-style) del total de notas
      - Vector rítmico global (mscz2vec.rhythmic_features) si disponible
      - Chord timeline global (melody_adapter.extract_chords_from_midi)
      - Melody list en formato adapter (offset_beat, pitch, dur, vel)
    """
    profile = {
        'dna': None,
        'descriptor': np.zeros(8),
        'rhythmic_vec': None,
        'chord_timeline': [],
        'melody': [],           # [(offset_beat, pitch, dur_beats, vel), ...]
        'tempo_bpm': 120.0,
        'beats_per_bar': 4,
        'tpb': mid.ticks_per_beat,
        'bar_ticks': _estimate_bar_ticks(mid),
        'total_ticks': _get_total_ticks(mid),
        'key_root_pc': 0,
        'key_mode': 'major',
        'scale_pitches': set(),
    }

    bar_ticks = profile['bar_ticks']
    tpb       = profile['tpb']

    # ── Descriptor simple global ──────────────────────────────────────────────
    profile['descriptor'] = _descriptor_for_notes(notes, bar_ticks)

    # ── Melody list (beats) ───────────────────────────────────────────────────
    tempo_us = _get_tempo(mid)
    profile['tempo_bpm'] = 60_000_000 / tempo_us
    num, _ = _get_time_signature(mid)
    profile['beats_per_bar'] = num

    melody_list = []
    for start_t, end_t, pitch, vel, ch, ti in notes:
        offset_b = start_t / tpb
        dur_b    = max(0.05, (end_t - start_t) / tpb)
        melody_list.append((offset_b, pitch, dur_b, vel))
    melody_list.sort(key=lambda x: x[0])
    profile['melody'] = melody_list

    # ── ADN global (completer) ────────────────────────────────────────────────
    if _completer is not None:
        try:
            dna = _completer.extract_dna_from_midi(midi_path, verbose=verbose)
            profile['dna'] = dna
            profile['key_root_pc'] = dna.key_root % 12
            profile['key_mode']    = dna.key_mode
            profile['scale_pitches'] = set(dna.scale_pitches) if dna.scale_pitches else set()
            if verbose:
                print(f"  [ADN] Tonalidad: {PITCH_NAMES[profile['key_root_pc']]} "
                      f"{profile['key_mode']}, tempo: {profile['tempo_bpm']:.0f} BPM")
        except Exception as e:
            if verbose:
                print(f"  [ADN] extract_dna_from_midi falló ({e}); usando detección propia.")
            _detect_key_simple(profile, notes)
    else:
        _detect_key_simple(profile, notes)

    # ── Vector rítmico global (mscz2vec) ─────────────────────────────────────
    if _mscz2vec is not None and MUSIC21_OK:
        try:
            score = m21converter.parse(midi_path)
            profile['rhythmic_vec'] = _mscz2vec.rhythmic_features(score)
            if verbose:
                print(f"  [RITMO] Vector rítmico global: dim={len(profile['rhythmic_vec'])}")
        except Exception as e:
            if verbose:
                print(f"  [RITMO] rhythmic_features global falló ({e}).")

    return profile


def _detect_key_simple(profile: dict, notes: list):
    """Detección de tonalidad simplificada por Krumhansl si ADN no disponible."""
    if not notes:
        return
    major_profile = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor_profile = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    pitches = [n[2] for n in notes]
    counts = np.zeros(12)
    for p in pitches:
        counts[p % 12] += 1
    counts = counts / (counts.sum() + 1e-9)
    best_score = -1
    best_root, best_mode = 0, 'major'
    for root in range(12):
        for mode, prof in [('major', major_profile), ('minor', minor_profile)]:
            rotated = [prof[(i - root) % 12] for i in range(12)]
            score = float(np.corrcoef(counts, rotated)[0, 1])
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    profile['key_root_pc'] = best_root
    profile['key_mode']    = best_mode

    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
    intervals = MAJOR_SCALE if best_mode == 'major' else MINOR_SCALE
    profile['scale_pitches'] = {(best_root + i) % 12 for i in intervals}


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA / INFERENCIA DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

def load_or_infer_chords(midi_path: str, chords_file: Optional[str],
                          global_profile: dict, verbose: bool) -> list:
    """
    Devuelve chord_timeline: [(beat_start, beat_end, root_pc, quality), ...]
    Prioridad: --chords FILE → inferencia automática de melody_adapter.
    """
    bpb = global_profile['beats_per_bar']

    # ── Desde fichero externo ─────────────────────────────────────────────────
    if chords_file:
        ext = Path(chords_file).suffix.lower()
        if ext == '.json':
            try:
                with open(chords_file) as f:
                    data = json.load(f)
                # Formato esperado: lista de {"root":str, "quality":str, "duration_beats":float}
                # o lista de [root_pc, quality, dur]
                chords_raw = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            rname  = item.get('root', 'C')
                            rpc    = PITCH_NAMES.index(rname) if rname in PITCH_NAMES else 0
                            quality = item.get('quality', 'M')
                            dur     = float(item.get('duration_beats', bpb))
                            chords_raw.append((rpc, quality, dur))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            chords_raw.append((int(item[0]), str(item[1]), float(item[2]) if len(item) > 2 else bpb))
                if chords_raw and _adapter:
                    total_beats = global_profile['total_ticks'] / global_profile['tpb']
                    timeline = _adapter.build_chord_timeline(chords_raw, total_beats)
                    if verbose:
                        print(f"  [ACORDES] Cargados desde {chords_file}: {len(timeline)} entradas")
                    return timeline
            except Exception as e:
                print(f"  [AVISO] No se pudo leer {chords_file}: {e}. Inferencia automática.")
        elif ext in ('.mid', '.midi'):
            try:
                if _adapter:
                    chords_raw = _adapter.extract_chords_from_midi(chords_file, bpb)
                    total_beats = global_profile['total_ticks'] / global_profile['tpb']
                    timeline = _adapter.build_chord_timeline(chords_raw, total_beats)
                    if verbose:
                        print(f"  [ACORDES] Extraídos del MIDI {chords_file}: {len(timeline)} entradas")
                    return timeline
            except Exception as e:
                print(f"  [AVISO] No se pudo extraer acordes de {chords_file}: {e}. Inferencia automática.")

    # ── Inferencia automática ─────────────────────────────────────────────────
    if _adapter:
        try:
            chords_raw = _adapter.extract_chords_from_midi(midi_path, bpb)
            total_beats = global_profile['total_ticks'] / global_profile['tpb']
            timeline = _adapter.build_chord_timeline(chords_raw, total_beats)
            if verbose:
                print(f"  [ACORDES] Inferidos automáticamente: {len(timeline)} entradas")
            return timeline
        except Exception as e:
            if verbose:
                print(f"  [ACORDES] Inferencia fallida ({e}). Sin timeline de acordes.")

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING POR CRITERIO
# ══════════════════════════════════════════════════════════════════════════════

def _score_tonal(seg_notes: list, seg_melody: list, chord_timeline: list,
                 global_profile: dict, bar_ticks: int) -> Tuple[float, List[str]]:
    """
    Anomalía tonal [0,1]: 1 = máxima incoherencia.
    Combina:
      a) Porcentaje de notas fuera de la escala global.
      b) Score de compatibilidad melodía-acordes (melody_adapter) si disponible.
    """
    reasons = []
    scores = []

    # a) Escala global
    scale_pcs = global_profile['scale_pitches']
    if scale_pcs and seg_notes:
        pitches = [n[2] for n in seg_notes]
        out_of_scale = sum(1 for p in pitches if (p % 12) not in scale_pcs)
        oos_ratio = out_of_scale / len(pitches)
        scores.append(oos_ratio)
        if oos_ratio > 0.3:
            reasons.append(f"fuera de escala: {oos_ratio:.0%} de notas")

    # b) Compatibilidad melodía-acordes
    if chord_timeline and seg_melody and _adapter:
        bpb = global_profile['beats_per_bar']
        # Filtrar chord_timeline al rango del segmento
        if seg_melody:
            seg_start_b = seg_melody[0][0]
            seg_end_b   = seg_melody[-1][0] + seg_melody[-1][2]
            seg_timeline = [(s, e, r, q) for s, e, r, q in chord_timeline
                            if s < seg_end_b and e > seg_start_b]
            if seg_timeline:
                try:
                    result = _adapter.score_melody_vs_progression(seg_melody, seg_timeline, bpb)
                    compat = result.get('global_score', 0.5)
                    anomaly = 1.0 - compat
                    scores.append(anomaly)
                    collisions = result.get('collisions', [])
                    if collisions:
                        reasons.append(f"colisiones armónicas: {len(collisions)}")
                except Exception:
                    pass

    if not scores:
        return 0.0, reasons

    final = float(np.mean(scores))
    return min(1.0, max(0.0, final)), reasons


def _score_structural(seg_descriptor: np.ndarray,
                       global_descriptor: np.ndarray,
                       all_descriptors: List[np.ndarray]) -> Tuple[float, List[str]]:
    """
    Anomalía estructural [0,1]: distancia del segmento al centroide global.
    """
    reasons = []
    dist = _cosine_distance(seg_descriptor, global_descriptor)

    # Normalizar: si la distancia es > umbral respecto a la media de distancias,
    # es anómalo
    if len(all_descriptors) >= 2:
        all_dists = [_cosine_distance(d, global_descriptor) for d in all_descriptors]
        mean_d = np.mean(all_dists)
        std_d  = np.std(all_dists)
        if std_d > 0:
            z = (dist - mean_d) / std_d
            anomaly = min(1.0, max(0.0, (z + 2) / 4))  # z=-2→0, z=+2→1
        else:
            anomaly = dist
    else:
        anomaly = dist

    if anomaly > 0.65:
        reasons.append(f"descriptor muy alejado del perfil global (dist={dist:.3f})")

    return float(anomaly), reasons


def _score_neighbor(seg_idx: int, all_descriptors: List[np.ndarray]) -> Tuple[float, List[str]]:
    """
    Anomalía vecinal [0,1]: delta con segmentos adyacentes.
    """
    reasons = []
    desc = all_descriptors[seg_idx]
    n    = len(all_descriptors)

    neighbor_dists = []
    if seg_idx > 0:
        d = _cosine_distance(desc, all_descriptors[seg_idx - 1])
        neighbor_dists.append(d)
    if seg_idx < n - 1:
        d = _cosine_distance(desc, all_descriptors[seg_idx + 1])
        neighbor_dists.append(d)

    if not neighbor_dists:
        return 0.0, reasons

    # Comparar con la media de distancias globales entre vecinos consecutivos
    all_consec_dists = []
    for i in range(n - 1):
        all_consec_dists.append(_cosine_distance(all_descriptors[i], all_descriptors[i+1]))

    mean_consec = np.mean(all_consec_dists) if all_consec_dists else 0.3
    std_consec  = np.std(all_consec_dists) if all_consec_dists else 0.1

    seg_mean_dist = float(np.mean(neighbor_dists))

    if std_consec > 0:
        z = (seg_mean_dist - mean_consec) / std_consec
        anomaly = min(1.0, max(0.0, (z + 2) / 4))
    else:
        anomaly = min(1.0, seg_mean_dist / 0.5)

    if anomaly > 0.65:
        reasons.append(f"ruptura con segmentos adyacentes (dist_vecinos={seg_mean_dist:.3f})")

    return float(anomaly), reasons


def _score_rhythmic(midi_path: str, seg_start_tick: int, seg_end_tick: int,
                     global_rhythmic_vec: Optional[np.ndarray],
                     mid: MidiFile, verbose: bool) -> Tuple[float, List[str]]:
    """
    Anomalía rítmica [0,1]: distancia del vector rítmico del segmento al global.
    Requiere music21 + mscz2vec.
    """
    reasons = []

    if global_rhythmic_vec is None or _mscz2vec is None or not MUSIC21_OK:
        return 0.0, reasons

    try:
        # Extraer segmento como score music21 temporal
        bar_ticks = _estimate_bar_ticks(mid)
        tpb       = mid.ticks_per_beat

        # Construir un MidiFile temporal del segmento
        seg_mid = _slice_midi_simple(mid, seg_start_tick, seg_end_tick)
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = tmp.name
        seg_mid.save(tmp_path)

        score_seg = m21converter.parse(tmp_path)
        seg_rhythmic = _mscz2vec.rhythmic_features(score_seg)
        _os.unlink(tmp_path)

        dist = _cosine_distance(seg_rhythmic, global_rhythmic_vec)
        # Normalizar directamente (0=idéntico, 1=máx distancia)
        anomaly = min(1.0, dist * 1.5)

        if anomaly > 0.6:
            reasons.append(f"patrón rítmico divergente (dist={dist:.3f})")

        return float(anomaly), reasons

    except Exception as e:
        if verbose:
            print(f"    [RITMO] score_rhythmic falló en segmento: {e}")
        return 0.0, reasons


def _slice_midi_simple(mid: MidiFile, start_tick: int, end_tick: int) -> MidiFile:
    """Extrae un segmento de MIDI (versión ligera, sin dependencia de harvester)."""
    out = MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        new_track = MidiTrack()
        abs_t = 0
        preamble = []
        events_in = []
        for msg in track:
            abs_t += msg.time
            if msg.is_meta:
                if abs_t <= start_tick:
                    preamble.append(msg.copy(time=0))
                elif abs_t < end_tick:
                    events_in.append((abs_t, msg))
            else:
                if start_tick <= abs_t < end_tick:
                    events_in.append((abs_t, msg))
        for msg in preamble:
            new_track.append(msg)
        prev = start_tick
        for abs_t2, msg in sorted(events_in, key=lambda x: x[0]):
            dt = abs_t2 - prev
            new_track.append(msg.copy(time=max(0, dt)))
            prev = abs_t2
        new_track.append(MetaMessage('end_of_track', time=0))
        out.tracks.append(new_track)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL DE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_segments(mid: MidiFile, midi_path: str,
                   segments: List[Tuple[int, int]],
                   all_notes: list,
                   global_profile: dict,
                   chord_timeline: list,
                   weights: Tuple[float, float, float, float],
                   verbose: bool) -> List[dict]:
    """
    Para cada segmento, calcula los cuatro scores y el score compuesto.
    Devuelve lista de dicts ordenada por score_total descendente.
    """
    bar_ticks   = global_profile['bar_ticks']
    tpb         = global_profile['tpb']
    total_ticks = global_profile['total_ticks']
    global_desc = global_profile['descriptor']
    global_rhy  = global_profile.get('rhythmic_vec')
    bpb         = global_profile['beats_per_bar']
    melody_all  = global_profile['melody']

    w_tonal, w_struct, w_neighbor, w_rhythmic = weights

    # Pre-calcular descriptores de todos los segmentos
    all_descs = []
    for start_t, end_t in segments:
        seg_notes = [n for n in all_notes if start_t <= n[0] < end_t]
        all_descs.append(_descriptor_for_notes(seg_notes, bar_ticks))

    results = []

    for idx, (start_t, end_t) in enumerate(segments):
        seg_notes  = [n for n in all_notes if start_t <= n[0] < end_t]
        seg_melody = [(ob, p, d, v) for ob, p, d, v in melody_all
                      if ob < end_t / tpb and ob + d > start_t / tpb]

        bar_start = int(start_t / bar_ticks) + 1
        bar_end   = max(bar_start, int(end_t / bar_ticks))
        n_bars    = max(1, bar_end - bar_start + 1)

        if verbose:
            print(f"\n  ── Segmento {idx+1:02d}  compases {bar_start}-{bar_end} "
                  f"({len(seg_notes)} notas) ──")

        # ── Criterio 1: Tonal ──────────────────────────────────────────────
        s_tonal, r_tonal = _score_tonal(
            seg_notes, seg_melody, chord_timeline, global_profile, bar_ticks)
        if verbose:
            print(f"    Tonal:      {s_tonal:.3f}  {r_tonal}")

        # ── Criterio 2: Estructural ────────────────────────────────────────
        s_struct, r_struct = _score_structural(
            all_descs[idx], global_desc, all_descs)
        if verbose:
            print(f"    Estructural:{s_struct:.3f}  {r_struct}")

        # ── Criterio 3: Vecinal ────────────────────────────────────────────
        s_neighbor, r_neighbor = _score_neighbor(idx, all_descs)
        if verbose:
            print(f"    Vecinal:    {s_neighbor:.3f}  {r_neighbor}")

        # ── Criterio 4: Rítmico ────────────────────────────────────────────
        s_rhythmic, r_rhythmic = _score_rhythmic(
            midi_path, start_t, end_t, global_rhy, mid, verbose)
        if verbose:
            print(f"    Rítmico:    {s_rhythmic:.3f}  {r_rhythmic}")

        # ── Score compuesto ────────────────────────────────────────────────
        score_total = (w_tonal    * s_tonal    +
                       w_struct   * s_struct   +
                       w_neighbor * s_neighbor +
                       w_rhythmic * s_rhythmic)

        all_reasons = (
            [f"[tonal] {r}"      for r in r_tonal]    +
            [f"[estructural] {r}" for r in r_struct]  +
            [f"[vecinal] {r}"    for r in r_neighbor] +
            [f"[rítmico] {r}"    for r in r_rhythmic]
        )

        results.append({
            'segment_idx':      idx + 1,
            'bar_start':        bar_start,
            'bar_end':          bar_end,
            'n_bars':           n_bars,
            'tick_start':       start_t,
            'tick_end':         end_t,
            'time_start_s':     round(start_t / tpb / (global_profile['tempo_bpm'] / 60), 2),
            'time_end_s':       round(end_t   / tpb / (global_profile['tempo_bpm'] / 60), 2),
            'n_notes':          len(seg_notes),
            'score_total':      round(score_total, 4),
            'score_tonal':      round(s_tonal, 4),
            'score_structural': round(s_struct, 4),
            'score_neighbor':   round(s_neighbor, 4),
            'score_rhythmic':   round(s_rhythmic, 4),
            'anomaly_reasons':  all_reasons,
        })

    results.sort(key=lambda x: -x['score_total'])
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA: JSON
# ══════════════════════════════════════════════════════════════════════════════

def write_json(results: List[dict], global_profile: dict,
               args_dict: dict, out_path: str):
    payload = {
        'source':        args_dict.get('midi_path', ''),
        'granularity':   args_dict.get('granularity', 'fixed'),
        'weights':       args_dict.get('weights_label', 'balanced'),
        'weight_values': args_dict.get('weight_values', []),
        'track_mode':    args_dict.get('track_mode', 'all'),
        'total_segments': len(results),
        'tonality': {
            'key':  PITCH_NAMES[global_profile['key_root_pc']],
            'mode': global_profile['key_mode'],
        },
        'tempo_bpm':     round(global_profile['tempo_bpm'], 1),
        'segments':      results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  → JSON guardado en: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA: TEXTO
# ══════════════════════════════════════════════════════════════════════════════

_BAR = "═" * 72

def _score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.3f}"


def write_text(results: List[dict], global_profile: dict,
               args_dict: dict, top_n: int, threshold: float,
               out_path: Optional[str] = None, verbose_stdout: bool = False):
    lines = []
    lines.append(_BAR)
    lines.append("  SECTION SCORE  — Análisis de incoherencias musicales")
    lines.append(_BAR)
    lines.append(f"  Fuente         : {args_dict.get('midi_path','')}")
    lines.append(f"  Granularidad   : {args_dict.get('granularity','')}")
    lines.append(f"  Pesos          : {args_dict.get('weights_label','')}  "
                 f"{tuple(round(w,2) for w in args_dict.get('weight_values',[]))}")
    lines.append(f"  Modo tracks    : {args_dict.get('track_mode','')}")
    lines.append(f"  Tonalidad      : {PITCH_NAMES[global_profile['key_root_pc']]} "
                 f"{global_profile['key_mode']}")
    lines.append(f"  Tempo          : {global_profile['tempo_bpm']:.0f} BPM")
    lines.append(f"  Segmentos tot. : {len(results)}")
    lines.append(_BAR)
    lines.append(f"  TOP {top_n} SEGMENTOS MÁS ANÓMALOS  (umbral ≥ {threshold:.2f})")
    lines.append(_BAR)

    shown = 0
    for r in results:
        if r['score_total'] < threshold:
            continue
        if shown >= top_n:
            break
        shown += 1

        lines.append(f"\n  #{shown:02d}  Compases {r['bar_start']}–{r['bar_end']}  "
                     f"({r['n_bars']} comp. · {r['n_notes']} notas · "
                     f"{r['time_start_s']}s–{r['time_end_s']}s)")
        lines.append(f"  Score total   {_score_bar(r['score_total'])}")
        lines.append(f"    Tonal       {_score_bar(r['score_tonal'])}")
        lines.append(f"    Estructural {_score_bar(r['score_structural'])}")
        lines.append(f"    Vecinal     {_score_bar(r['score_neighbor'])}")
        lines.append(f"    Rítmico     {_score_bar(r['score_rhythmic'])}")
        if r['anomaly_reasons']:
            lines.append("  Motivos:")
            for reason in r['anomaly_reasons']:
                lines.append(f"    · {reason}")

    if shown == 0:
        lines.append(f"\n  Ningún segmento supera el umbral de anomalía {threshold:.2f}.")
        lines.append("  Considera reducir --threshold o cambiar la granularidad.")

    lines.append(f"\n{_BAR}")
    text = "\n".join(lines)

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  → Informe de texto guardado en: {out_path}")

    if verbose_stdout or out_path is None:
        print(text)


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA: MIDI ANOTADO
# ══════════════════════════════════════════════════════════════════════════════

def write_midi_annotated(mid: MidiFile, results: List[dict],
                          top_n: int, threshold: float,
                          out_path: str):
    """
    Copia el MIDI original e inserta MetaMessage('marker') al inicio de
    cada segmento anómalo. El texto del marker incluye score y motivos,
    visible en cualquier DAW que soporte markers de MIDI.
    """
    import copy as _copy
    out_mid = _copy.deepcopy(mid)

    # Recogemos los marcadores que hay que insertar: {tick → texto}
    markers: Dict[int, str] = {}
    shown = 0
    for rank, r in enumerate(results, 1):
        if r['score_total'] < threshold:
            continue
        if shown >= top_n:
            break
        shown += 1
        tick = r['tick_start']
        label = (f"[ANOMALY#{rank}] bars {r['bar_start']}-{r['bar_end']} "
                 f"score={r['score_total']:.3f} "
                 f"T={r['score_tonal']:.2f} "
                 f"S={r['score_structural']:.2f} "
                 f"N={r['score_neighbor']:.2f} "
                 f"R={r['score_rhythmic']:.2f}")
        reasons_short = "; ".join(r['anomaly_reasons'][:3])
        if reasons_short:
            label += f" | {reasons_short}"
        markers[tick] = label

    if not markers:
        print("  [MIDI] Ningún segmento supera el umbral; MIDI no generado.")
        return

    # Insertar markers en la primera pista (track 0 suele ser el header en type=1)
    # Si el MIDI es type=0 usamos la única pista; si type=1 usamos track[0]
    target_track_idx = 0

    for track_idx, track in enumerate(out_mid.tracks):
        if track_idx != target_track_idx:
            continue

        # Reconstruir el track insertando los markers en el lugar correcto
        new_msgs = []
        abs_t = 0
        inserted = set()

        # Primero convertimos el track a lista con tiempos absolutos
        abs_events = []
        cur = 0
        for msg in track:
            cur += msg.time
            abs_events.append((cur, msg))

        # Añadir markers como eventos extra
        marker_events = [(tick, MetaMessage('marker', text=text, time=0))
                         for tick, text in markers.items()]
        all_events = sorted(abs_events + marker_events, key=lambda x: x[0])

        # Reconstruir con tiempos relativos
        prev_t = 0
        rebuilt = []
        for abs_tick, msg in all_events:
            dt = abs_tick - prev_t
            rebuilt.append(msg.copy(time=max(0, dt)))
            prev_t = abs_tick

        out_mid.tracks[track_idx] = MidiTrack(rebuilt)
        break

    try:
        out_mid.save(out_path)
        print(f"  → MIDI anotado guardado en: {out_path}")
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar el MIDI anotado: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA: MIDI PODADO (--remove)
# ══════════════════════════════════════════════════════════════════════════════

def write_midi_pruned(mid: MidiFile, results: List[dict], n_remove: int,
                      out_path: str, verbose: bool = False):
    """
    Genera un MIDI con los n_remove segmentos más anómalos eliminados.

    Estrategia:
      1. Tomar los n_remove primeros de `results` (ya ordenados por score desc).
      2. Calcular los rangos de ticks a conservar (inverso de los eliminados).
      3. Para cada track, recorrer los eventos y copiar solo los que caen en
         rangos conservados, reescribiendo los tiempos para que no queden huecos.
      4. Los meta-mensajes de tempo/firma se preservan siempre; los de tipo
         'end_of_track' se reescriben al final.

    Notas sobre la continuidad:
      - Las notas activas (note_on sin note_off) que cruzan un límite de
        eliminación se cierran con note_off justo antes del corte.
      - Los segmentos eliminados que caigan en medio de una nota larga reciben
        un note_on al inicio del siguiente segmento conservado.
    """
    import copy as _copy

    # ── 1. Segmentos a eliminar (ordenados por tick_start) ────────────────────
    to_remove = sorted(results[:n_remove], key=lambda r: r['tick_start'])

    if not to_remove:
        print("  [REMOVE] No hay segmentos que eliminar.")
        return

    if verbose:
        for r in to_remove:
            print(f"  [REMOVE] Eliminando compases {r['bar_start']}–{r['bar_end']} "
                  f"(score={r['score_total']:.3f}, "
                  f"ticks {r['tick_start']}–{r['tick_end']})")

    total_ticks = _get_total_ticks(mid)

    # ── 2. Calcular rangos conservados ────────────────────────────────────────
    # Construir lista de intervalos eliminados sin solapamiento
    removed_ranges: List[Tuple[int, int]] = []
    for r in to_remove:
        s, e = r['tick_start'], r['tick_end']
        # Fusionar con el rango anterior si se solapan
        if removed_ranges and s <= removed_ranges[-1][1]:
            removed_ranges[-1] = (removed_ranges[-1][0], max(removed_ranges[-1][1], e))
        else:
            removed_ranges.append((s, e))

    # Rangos conservados: huecos entre los eliminados
    kept_ranges: List[Tuple[int, int]] = []
    cursor = 0
    for rs, re in removed_ranges:
        if cursor < rs:
            kept_ranges.append((cursor, rs))
        cursor = re
    if cursor < total_ticks:
        kept_ranges.append((cursor, total_ticks))

    if not kept_ranges:
        print("  [REMOVE] Advertencia: no quedaría ningún contenido. Operación cancelada.")
        return

    if verbose:
        print(f"  [REMOVE] Rangos conservados: {len(kept_ranges)}")
        for ks, ke in kept_ranges:
            print(f"    [{ks} – {ke})  ({ke - ks} ticks)")

    # ── 3. Función auxiliar: ¿está un tick en un rango conservado? ────────────
    def _in_kept(tick: int) -> bool:
        for ks, ke in kept_ranges:
            if ks <= tick < ke:
                return True
        return False

    def _kept_range_for(tick: int) -> Optional[Tuple[int, int]]:
        for ks, ke in kept_ranges:
            if ks <= tick < ke:
                return (ks, ke)
        return None

    # Offset acumulado que hay que restar para eliminar los huecos
    # Para un tick `t` en kept_range[i], el offset es la suma de duraciones
    # de todos los rangos eliminados anteriores a `t`.
    def _adjusted_tick(tick: int) -> int:
        removed_before = 0
        for rs, re in removed_ranges:
            if re <= tick:
                removed_before += re - rs
            elif rs < tick:
                removed_before += tick - rs
                break
        return tick - removed_before

    # ── 4. Reconstruir cada track ─────────────────────────────────────────────
    out_mid = MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = MidiTrack()

        # Convertir a eventos absolutos
        abs_events: List[Tuple[int, object]] = []
        cur = 0
        for msg in track:
            cur += msg.time
            abs_events.append((cur, msg))

        # Rastrear notas activas para cerrarlas en los límites de corte
        # active: (channel, note) → abs_tick de note_on
        active_notes: dict = {}

        # Eventos de salida con ticks absolutos ya ajustados
        out_events: List[Tuple[int, object]] = []

        # Siempre incluir los meta de cabecera (tempo, time_sig, etc.)
        # que ocurren antes de cualquier nota
        for abs_t, msg in abs_events:
            if msg.is_meta and msg.type not in ('end_of_track',):
                # Meta globales: incluir siempre en tick ajustado
                # Si caen en zona eliminada, los reubicamos al inicio del
                # siguiente rango conservado
                if _in_kept(abs_t):
                    out_events.append((_adjusted_tick(abs_t), msg))
                else:
                    # Buscar el próximo rango conservado
                    next_kept_start = None
                    for ks, ke in kept_ranges:
                        if ks >= abs_t:
                            next_kept_start = ks
                            break
                    if next_kept_start is not None:
                        out_events.append((_adjusted_tick(next_kept_start), msg))
                    # Si no hay rango conservado posterior, descartar
            elif not msg.is_meta:
                # Mensajes de canal que no son notas (program_change, control_change,
                # pitchwheel, aftertouch…): se conservan si caen en zona kept; si caen
                # en zona eliminada se reubican al inicio del siguiente rango conservado
                # para que el instrumento/estado del canal no se pierda.
                if msg.type not in ('note_on', 'note_off'):
                    if _in_kept(abs_t):
                        out_events.append((_adjusted_tick(abs_t), msg))
                    else:
                        next_kept_start = None
                        for ks, ke in kept_ranges:
                            if ks >= abs_t:
                                next_kept_start = ks
                                break
                        if next_kept_start is not None:
                            out_events.append((_adjusted_tick(next_kept_start), msg))

                # Eventos de nota
                elif msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                    if _in_kept(abs_t):
                        active_notes[(msg.channel, msg.note)] = abs_t
                        out_events.append((_adjusted_tick(abs_t), msg))
                    # Si cae en zona eliminada: no emitir, pero registrar si
                    # la nota cruza hacia un rango conservado
                    else:
                        # Comprobar si la duración cubre el siguiente rango conservado
                        # (no podemos saberlo aquí sin el note_off; lo manejamos en note_off)
                        active_notes[(msg.channel, msg.note)] = abs_t  # fuera de kept

                elif msg.type in ('note_off',) or \
                        (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0):
                    key = (msg.channel, msg.note)
                    on_tick = active_notes.pop(key, None)

                    if on_tick is None:
                        continue  # note_off huérfano

                    on_in_kept  = _in_kept(on_tick)
                    off_in_kept = _in_kept(abs_t)

                    if on_in_kept and off_in_kept:
                        # Caso normal: nota completamente en zona conservada
                        out_events.append((_adjusted_tick(abs_t), msg))

                    elif on_in_kept and not off_in_kept:
                        # Nota que empieza en kept y termina en eliminado:
                        # cerrarla en el tick de inicio del corte
                        kr = _kept_range_for(on_tick)
                        if kr:
                            close_tick = kr[1]  # fin del rango conservado
                            out_events.append((_adjusted_tick(close_tick - 1),
                                               msg.copy(time=0)))

                    elif not on_in_kept and off_in_kept:
                        # Nota que empieza en eliminado y termina en kept:
                        # emitir note_on al inicio del rango conservado donde termina
                        kr = _kept_range_for(abs_t)
                        if kr:
                            note_on_msg = mido.Message(
                                'note_on', channel=msg.channel,
                                note=msg.note, velocity=64, time=0)
                            out_events.append((_adjusted_tick(kr[0]), note_on_msg))
                            out_events.append((_adjusted_tick(abs_t), msg))

                    # else: nota completamente en zona eliminada → descartar

        # Notas que quedaron abiertas al final (sin note_off):
        for (ch, note), on_tick in active_notes.items():
            if _in_kept(on_tick):
                # Cerrar al final del último rango conservado
                last_kept_end = kept_ranges[-1][1]
                close_msg = mido.Message('note_off', channel=ch, note=note,
                                         velocity=0, time=0)
                out_events.append((_adjusted_tick(last_kept_end - 1), close_msg))

        # Ordenar por tick ajustado
        out_events.sort(key=lambda x: x[0])

        # Convertir a tiempos relativos y añadir al track
        prev_adj = 0
        for adj_tick, msg in out_events:
            dt = max(0, adj_tick - prev_adj)
            new_track.append(msg.copy(time=dt))
            prev_adj = adj_tick

        new_track.append(MetaMessage('end_of_track', time=0))
        out_mid.tracks.append(new_track)

    # ── 5. Guardar ────────────────────────────────────────────────────────────
    try:
        out_mid.save(out_path)
        removed_bars = [(r['bar_start'], r['bar_end']) for r in to_remove]
        print(f"  → MIDI podado guardado en: {out_path}")
        print(f"    Segmentos eliminados: "
              f"{', '.join(f'compases {s}-{e}' for s, e in removed_bars)}")
        # Duración original vs. resultado
        orig_ticks   = total_ticks
        removed_ticks = sum(re - rs for rs, re in removed_ranges)
        kept_ticks    = orig_ticks - removed_ticks
        tpb           = mid.ticks_per_beat
        tempo_us      = _get_tempo(mid)
        bpm           = 60_000_000 / tempo_us
        def _ticks_to_s(t): return round(t / tpb / (bpm / 60), 1)
        print(f"    Duración original : {_ticks_to_s(orig_ticks)}s  →  "
              f"podada: {_ticks_to_s(kept_ticks)}s  "
              f"(−{_ticks_to_s(removed_ticks)}s)")
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar el MIDI podado: {e}")
        if verbose:
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="section_score",
        description="Detecta secciones musicalmente incoherentes en un MIDI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("midi_path", help="Ruta al fichero MIDI de entrada.")

    # Segmentación
    p.add_argument(
        "--granularity", nargs="+", default=["fixed", "4"],
        metavar=("TIPO", "N"),
        help=textwrap.dedent("""\
            Modo de segmentación:
              fixed N    Ventanas fijas de N compases (default: fixed 4)
              form       Fronteras por SSM + novelty
              texture    Fronteras por cambios de textura
              cadence    Fronteras en cadencias (requiere music21)
              hybrid     Combinación form + texture
        """),
    )
    p.add_argument("--novelty-threshold", type=float, default=0.15,
                   help="Sensibilidad de detección de fronteras 0-1 (default: 0.15).")

    # Pesos
    p.add_argument(
        "--weights", nargs="+", default=["balanced"],
        metavar="PRESET_O_CUSTOM",
        help=textwrap.dedent("""\
            Preset de pesos (tonal · estructural · vecinal · rítmico):
              balanced   0.30 · 0.25 · 0.25 · 0.20  (default)
              harmonic   0.50 · 0.20 · 0.20 · 0.10
              neighbor   0.20 · 0.15 · 0.50 · 0.15
              structural 0.15 · 0.50 · 0.20 · 0.15
              rhythmic   0.15 · 0.20 · 0.20 · 0.45
              custom W1,W2,W3,W4  (floats separados por coma, suma=1)
        """),
    )

    # Tracks
    p.add_argument(
        "--track-mode", choices=["all", "melody", "auto"], default="all",
        help="Pistas a analizar: all (default) | melody | auto.",
    )

    # Acordes
    p.add_argument(
        "--chords", metavar="FILE", default=None,
        help="JSON o MIDI con acordes externos. Si no se indica, se infieren.",
    )

    # Output
    p.add_argument("--out-json", metavar="FILE", default=None)
    p.add_argument("--out-text", metavar="FILE", default=None)
    p.add_argument("--out-midi", metavar="FILE", default=None)
    p.add_argument("--remove", type=int, default=None, metavar="N",
                   help=textwrap.dedent("""\
                       Genera un MIDI con los N segmentos más anómalos eliminados.
                       Los segmentos restantes se concatenan sin huecos.
                       Se guarda con sufijo _pruned_<N>.mid junto al fichero de
                       entrada, salvo que se indique --remove-out.
                   """))
    p.add_argument("--remove-out", metavar="FILE", default=None,
                   help="Ruta de salida del MIDI podado (requiere --remove).")
    p.add_argument("--top",   type=int,   default=5,
                   help="Número de segmentos a reportar (default: 5).")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Score mínimo de anomalía 0-1 (default: 0.40).")
    p.add_argument("--verbose", action="store_true",
                   help="Detalle completo por criterio.")

    return p


def parse_granularity(args_granularity: list) -> Tuple[str, int]:
    """Parsea --granularity [tipo] [N]."""
    tokens = args_granularity
    gtype = tokens[0].lower()
    window = 4
    if gtype == "fixed":
        if len(tokens) >= 2:
            try:
                window = int(tokens[1])
            except ValueError:
                pass
    valid = {"fixed", "form", "texture", "cadence", "hybrid"}
    if gtype not in valid:
        print(f"[AVISO] Granularidad '{gtype}' no reconocida; usando 'fixed 4'.")
        gtype, window = "fixed", 4
    return gtype, window


def parse_weights(args_weights: list) -> Tuple[Tuple[float,float,float,float], str]:
    """Parsea --weights [preset | custom W1,W2,W3,W4]."""
    tokens = args_weights
    label = tokens[0].lower()

    if label in WEIGHT_PRESETS:
        return WEIGHT_PRESETS[label], label

    if label == "custom":
        if len(tokens) >= 2:
            try:
                vals = [float(x) for x in tokens[1].split(",")]
                if len(vals) == 4:
                    s = sum(vals)
                    normalized = tuple(v / s for v in vals)
                    return normalized, f"custom({tokens[1]})"
            except ValueError:
                pass
        print("[AVISO] Formato custom inválido; usando 'balanced'.")
        return WEIGHT_PRESETS["balanced"], "balanced"

    print(f"[AVISO] Preset '{label}' no reconocido; usando 'balanced'.")
    return WEIGHT_PRESETS["balanced"], "balanced"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    midi_path = args.midi_path
    if not Path(midi_path).exists():
        print(f"[ERROR] Fichero no encontrado: {midi_path}")
        sys.exit(1)

    verbose = args.verbose

    # ── Cargar MIDI ───────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SECTION SCORE  ·  {Path(midi_path).name}")
    print(f"{'═'*72}")

    try:
        mid = MidiFile(midi_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el MIDI: {e}")
        sys.exit(1)

    # ── Parsear opciones ──────────────────────────────────────────────────────
    granularity, window_bars = parse_granularity(args.granularity)
    weights, weights_label   = parse_weights(args.weights)

    print(f"  Granularidad   : {granularity}" +
          (f" ({window_bars} compases)" if granularity == "fixed" else ""))
    print(f"  Pesos          : {weights_label}  {tuple(round(w,2) for w in weights)}")
    print(f"  Modo tracks    : {args.track_mode}")
    print(f"  Top / umbral   : {args.top} segmentos / score ≥ {args.threshold}")

    args_dict = {
        'midi_path':     midi_path,
        'granularity':   granularity + (f" {window_bars}" if granularity == "fixed" else ""),
        'weights_label': weights_label,
        'weight_values': list(weights),
        'track_mode':    args.track_mode,
    }

    # ── Selección de tracks ───────────────────────────────────────────────────
    selected_tracks = _select_tracks(mid, args.track_mode)
    if verbose:
        print(f"  Tracks seleccionados: {sorted(selected_tracks)}")

    # ── Extraer notas ─────────────────────────────────────────────────────────
    all_notes = _extract_note_events(mid, tracks=selected_tracks)
    print(f"  Notas extraídas: {len(all_notes)}")

    if len(all_notes) < 4:
        print("[ERROR] El MIDI no contiene suficientes notas para analizar.")
        sys.exit(1)

    # ── Perfil global ─────────────────────────────────────────────────────────
    print("\n  Construyendo perfil global de la obra…")
    global_profile = build_global_profile(mid, midi_path, all_notes, verbose=verbose)
    print(f"  Tonalidad detectada: {PITCH_NAMES[global_profile['key_root_pc']]} "
          f"{global_profile['key_mode']}")

    # ── Acordes ───────────────────────────────────────────────────────────────
    print("  Cargando/infiriendo acordes…")
    chord_timeline = load_or_infer_chords(
        midi_path, args.chords, global_profile, verbose=verbose)

    # ── Segmentación ──────────────────────────────────────────────────────────
    print(f"  Segmentando ({granularity})…")
    segments = build_segments(
        mid, midi_path, all_notes,
        granularity, window_bars,
        args.novelty_threshold,
    )
    print(f"  Segmentos detectados: {len(segments)}")

    if len(segments) < 2:
        print("[AVISO] Muy pocos segmentos para analizar. "
              "Prueba con --granularity fixed 2 o un MIDI más largo.")

    # ── Scoring ───────────────────────────────────────────────────────────────
    print(f"\n  Calculando scores ({len(segments)} segmentos)…")
    results = score_segments(
        mid, midi_path, segments, all_notes,
        global_profile, chord_timeline,
        weights, verbose,
    )

    # ── Salidas ───────────────────────────────────────────────────────────────
    print()

    any_output = args.out_json or args.out_text or args.out_midi
    show_text  = not any_output or verbose

    write_text(results, global_profile, args_dict,
               args.top, args.threshold,
               out_path=args.out_text,
               verbose_stdout=show_text)

    if args.out_json:
        write_json(results, global_profile, args_dict, args.out_json)

    if args.out_midi:
        write_midi_annotated(mid, results, args.top, args.threshold, args.out_midi)

    if args.remove is not None:
        n_remove = args.remove
        if n_remove < 1:
            print("[AVISO] --remove debe ser ≥ 1. Ignorado.")
        elif n_remove >= len(results):
            print(f"[AVISO] --remove {n_remove} ≥ número de segmentos ({len(results)}). "
                  "Se eliminarían todos. Operación cancelada.")
        else:
            # Ruta de salida
            if args.remove_out:
                pruned_path = args.remove_out
            else:
                stem = Path(midi_path).stem
                suffix = Path(midi_path).suffix or '.mid'
                pruned_path = str(Path(midi_path).parent / f"{stem}_pruned_{n_remove}{suffix}")

            write_midi_pruned(mid, results, n_remove, pruned_path, verbose=verbose)

    print(f"\n{'═'*72}\n")


if __name__ == "__main__":
    main()
