#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MELODY ADAPTER  v1.0                                   ║
║         Adaptación de melodías a progresiones de acordes externas           ║
║                                                                              ║
║  Dado un MIDI de melodía y una progresión de acordes (en texto o en MIDI),  ║
║  ajusta las notas de la melodía para que sean compatibles con cada acorde,  ║
║  respetando el contorno melódico original y distinguiendo entre notas        ║
║  estructurales y notas de paso/ornamentales.                                 ║
║                                                                              ║
║  MODOS DE ADAPTACIÓN (--mode):                                               ║
║    strict    — solo mueve avoid notes en tiempos fuertes; mínima           ║
║               intervención. Ideal para preservar la melodía casi intacta.   ║
║    smooth    — strict + añade notas de paso para suavizar saltos            ║
║               introducidos por la adaptación. Sonido más cantábile.         ║
║    jazz      — trata 9ª, 11ª y 13ª como tensiones válidas; solo mueve      ║
║               las avoid notes más disonantes (b9, b13, ♮11 sobre maj).     ║
║    rewrite   — reescribe el ritmo y las alturas manteniendo el contorno     ║
║               general. Máxima compatibilidad armónica.                      ║
║                                                                              ║
║  FORMATOS DE PROGRESIÓN:                                                     ║
║    --chords "Cm G7 Fm Bb7"           un acorde por compás                  ║
║    --chords "Cm:2 G7:2 Fm:1 Bb7:1"  duración en beats tras ':'             ║
║    --chords-midi acordes.mid         pista de acordes en MIDI               ║
║                                                                              ║
║  MÉTRICAS DE COMPATIBILIDAD (--report):                                      ║
║    · notas desplazadas / total                                               ║
║    · score de compatibilidad antes y después (como reharmonizer.py)         ║
║    · lista de colisiones resueltas y sin resolver                            ║
║                                                                              ║
║  USO:                                                                        ║
║    python melody_adapter.py melodia.mid --chords "C Am F G"                ║
║    python melody_adapter.py melodia.mid --chords "C:2 F:2 G:4" --mode smooth║
║    python melody_adapter.py melodia.mid --chords-midi acomp.mid            ║
║    python melody_adapter.py melodia.mid --chords "Dm7 G7 Cmaj7" --mode jazz║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --mode rewrite ║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --report       ║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --acc-style arpeggio║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --bars 8       ║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --key "D major"║
║    python melody_adapter.py melodia.mid --chords "C Am F G" --strength 0.7 ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --chords PROG       Progresión en texto (ver formato arriba)              ║
║    --chords-midi FILE  Pista de acordes en MIDI                              ║
║    --mode MODE         Modo de adaptación (default: smooth)                 ║
║    --strength F        Agresividad 0-1 (default: 0.8; 0=mínimo, 1=máximo)  ║
║    --bars N            Compases de salida (default: auto)                   ║
║    --key KEY           Tonalidad forzada, e.g. "D minor" (default: auto)    ║
║    --beats-per-bar N   Pulsos por compás (default: detectado del MIDI)      ║
║    --acc-style S       Acompañamiento: block|arpeggio|alberti|waltz|auto    ║
║    --no-acc            No generar acompañamiento, solo la melodía adaptada  ║
║    --out-dir DIR       Carpeta de salida (default: misma que el MIDI)       ║
║    --report            Guardar JSON con análisis de compatibilidad          ║
║    --verbose           Informe detallado por stdout                          ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    melodia.adapted_smooth.mid                                                ║
║    melodia.adapted_report.json  (con --report)                              ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21, numpy  (+ midi_dna_unified en mismo dir)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import re
import json
import copy
import math
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import mido

# ── music21 ───────────────────────────────────────────────────────────────────
try:
    from music21 import converter, pitch as m21pitch, key as m21key
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Se usará detección de tonalidad simplificada.")

# ── midi_dna_unified ──────────────────────────────────────────────────────────
_DNA_DIR = os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else os.getcwd()))
sys.path.insert(0, _DNA_DIR)
try:
    from midi_dna_unified import (
        _snap_to_scale, _get_scale_pcs,
        _quarter_to_ticks,
        generate_accompaniment, generate_bass, build_midi,
        EmotionalController, FormGenerator,
        MAJOR_SCALE_DEGREES, MINOR_SCALE_DEGREES,
    )
    DNA_OK = True
except ImportError as e:
    DNA_OK = False
    print(f"[AVISO] midi_dna_unified no encontrado: {e}\n"
          "El acompañamiento generado será simplificado.")

# ── reharmonizer (reutilizamos su lógica de clasificación) ───────────────────
try:
    from reharmonizer import (
        chord_tone_category, score_chord_vs_melody,
        CHORD_INTERVALS, NOTE_CATEGORY,
        load_melody, melody_pcs_in_window,
        tonic_pc, parse_key_string, make_key,
        build_chord_pitches, _write_midi_direct,
        ACC_STYLE_PATTERNS,
        PITCH_NAMES,
    )
    REHAR_OK = True
except ImportError:
    REHAR_OK = False
    # Definiciones mínimas de fallback si reharmonizer no está disponible
    PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    CHORD_INTERVALS = {
        'M':   [0, 4, 7],       'm':   [0, 3, 7],
        'd':   [0, 3, 6],       'A':   [0, 4, 8],
        'M7':  [0, 4, 7, 11],   'm7':  [0, 3, 7, 10],
        'Mm7': [0, 4, 7, 10],   'd7':  [0, 3, 6, 9],
        'hd7': [0, 3, 6, 10],   'sus4':[0, 5, 7],
        'sus2':[0, 2, 7],       '6':   [0, 4, 7, 9],
        'm6':  [0, 3, 7, 9],    'M9':  [0, 4, 7, 11, 14],
        'm9':  [0, 3, 7, 10, 14], 'Mm9': [0, 4, 7, 10, 14],
    }

    def chord_tone_category(mel_pc, root_pc, quality):
        interval = (mel_pc - root_pc) % 12
        cats = {0:'chord_tone',2:'tension',3:'chord_tone',4:'chord_tone',
                5:'tension',6:'avoid',7:'chord_tone',9:'tension',
                10:'tension',11:'tension',1:'avoid',8:'avoid'}
        base = cats.get(interval, 'tension')
        if quality in ('Mm7','M7','d7','hd7'):
            if interval == 11: return 'avoid'
            if interval == 6:  return 'tension'
        if quality in ('M','M7'):
            if interval == 5:  return 'avoid'
        return base

    def score_chord_vs_melody(mel_pcs, root_pc, quality):
        if not mel_pcs: return 0.8
        scores = [{'chord_tone':1.0,'tension':0.6,'avoid':0.0}[
            chord_tone_category(pc, root_pc, quality)] for pc in mel_pcs]
        return float(np.mean(scores))

    def tonic_pc(key_obj):
        name = key_obj.tonic.name if hasattr(key_obj, 'tonic') else 'C'
        return PITCH_NAMES.index(name) if name in PITCH_NAMES else 0

    def melody_pcs_in_window(melody, beat_start, beat_end):
        pcs = set()
        for offset, pitch, dur, vel in melody:
            if offset < beat_end and offset + dur > beat_start:
                pcs.add(pitch % 12)
        return pcs


# ═══════════════════════════════════════════════════════════
# PARSER DE PROGRESIÓN EN TEXTO
# ═══════════════════════════════════════════════════════════

# Mapeo de nombres de nota a pitch class
_NOTE_TO_PC = {
    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'Fb':4,
    'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,
    'Bb':10,'B':11,'Cb':11,
}

# Sufijos de calidad a nombre interno
_QUALITY_MAP = [
    # Sufijos largos primero para evitar coincidencias parciales
    ('maj9',  'M9'),  ('maj7',  'M7'),  ('min9',  'm9'),  ('min7',  'm7'),
    ('m7b5',  'hd7'), ('ø7',    'hd7'), ('ø',     'hd7'), ('dim7',  'd7'),
    ('dim',   'd'),   ('aug',   'A'),
    # Variantes con m antes del número (Dm7, Fm9…)
    ('m9',    'm9'),  ('m7',    'm7'),  ('m6',    'm6'),
    ('m',     'm'),   ('min',   'm'),
    ('maj',   'M'),   ('M',     'M'),
    ('sus4',  'sus4'),('sus2',  'sus2'),
    ('9',     'Mm9'), ('7',     'Mm7'), ('6',     '6'),
    ('+',     'A'),   ('°',     'd'),
    ('',      'M'),   # '' = mayor por defecto
]


def parse_chord_name(name):
    """
    Parsea un nombre de acorde textual (e.g. 'Cm7', 'G7', 'Fmaj7', 'Bb') y
    devuelve (root_pc, quality_str).
    """
    name = name.strip()
    # Extraer nota raíz (puede tener sostenido/bemol)
    m = re.match(r'^([A-G][b#]?)', name)
    if not m:
        raise ValueError(f"Nombre de acorde no reconocido: '{name}'")
    root_str = m.group(1)
    root_pc  = _NOTE_TO_PC.get(root_str, 0)
    suffix   = name[len(root_str):]

    # Eliminar inversiones (/bajo) que no nos interesan para la melodía
    slash = suffix.find('/')
    if slash != -1:
        suffix = suffix[:slash]

    # Buscar calidad
    for pattern, quality in _QUALITY_MAP:
        if suffix.lower() == pattern.lower() or suffix == pattern:
            return root_pc, quality

    # Fallback inteligente por longitud
    if suffix:
        return root_pc, 'Mm7' if '7' in suffix else 'M'
    return root_pc, 'M'


def parse_chord_string(chord_str, beats_per_bar=4):
    """
    Parsea una cadena de acordes en formato:
      "C Am F G"              → un acorde por compás (beats_per_bar pulsos c/u)
      "C:2 Am:2 F:1 G:3"     → duración explícita en beats tras ':'
      "C Am | F G"            → '|' como separador de compás (decorativo)

    Devuelve lista de (root_pc, quality, duration_beats).
    """
    # Eliminar separadores de compás
    chord_str = chord_str.replace('|', ' ')
    tokens = chord_str.split()

    result = []
    for tok in tokens:
        if not tok:
            continue
        if ':' in tok:
            chord_part, dur_part = tok.rsplit(':', 1)
            try:
                dur = float(dur_part)
            except ValueError:
                dur = float(beats_per_bar)
        else:
            chord_part = tok
            dur = float(beats_per_bar)

        try:
            root_pc, quality = parse_chord_name(chord_part)
        except ValueError as e:
            print(f"  [AVISO] {e}  →  se omite '{tok}'")
            continue

        result.append((root_pc, quality, dur))

    return result


def extract_chords_from_midi(midi_path, beats_per_bar=4):
    """
    Extrae acordes de una pista MIDI (canal o pista con mayor densidad armónica).
    Devuelve lista de (root_pc, quality, duration_beats) basada en
    detección de pitch class sets por ventana temporal.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    notes_by_ch = defaultdict(list)
    pending = {}

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
            elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                key = (msg.channel, msg.note)
                if key in pending:
                    on_t, vel = pending.pop(key)
                    beat_on  = on_t / tpb
                    beat_dur = max(0.1, (abs_t - on_t) / tpb)
                    if msg.channel != 9:
                        notes_by_ch[msg.channel].append(
                            (beat_on, msg.note, beat_dur, vel))

    if not notes_by_ch:
        raise RuntimeError("No se encontraron notas en el MIDI de acordes.")

    # Canal con menor pitch medio → acordes (el más grave)
    def _mean_p(notes):
        return np.mean([p for _, p, _, _ in notes]) if notes else 999
    chord_ch = min(notes_by_ch, key=lambda c: _mean_p(notes_by_ch[c]))
    chord_notes = sorted(notes_by_ch[chord_ch], key=lambda x: x[0])

    if not chord_notes:
        raise RuntimeError("No se encontraron notas en el canal de acordes.")

    total_beats = max(o + d for o, _, d, _ in chord_notes)
    n_bars = max(1, math.ceil(total_beats / beats_per_bar))

    result = []
    for bar in range(n_bars):
        bar_start = bar * beats_per_bar
        bar_end   = bar_start + beats_per_bar
        pcs_in_bar = set()
        lowest = 127
        for beat_on, pitch, beat_dur, vel in chord_notes:
            if beat_on < bar_end and beat_on + beat_dur > bar_start:
                pcs_in_bar.add(pitch % 12)
                if pitch < lowest:
                    lowest = pitch

        if not pcs_in_bar:
            # Compás vacío: usar el último acorde detectado
            if result:
                result.append(result[-1])
            continue

        root_pc = lowest % 12
        # Inferir calidad desde pitch classes presentes
        quality = _infer_quality(root_pc, pcs_in_bar)
        result.append((root_pc, quality, float(beats_per_bar)))

    return result


def _infer_quality(root_pc, pcs_set):
    """Infiere la calidad de un acorde a partir de su root y los pitch classes."""
    intervals = sorted((pc - root_pc) % 12 for pc in pcs_set)
    has = set(intervals)

    if 3 in has and 6 in has and 10 in has: return 'hd7'
    if 3 in has and 6 in has and 9 in has:  return 'd7'
    if 3 in has and 6 in has:               return 'd'
    if 4 in has and 8 in has:               return 'A'
    if 4 in has and 7 in has and 11 in has: return 'M7'
    if 4 in has and 7 in has and 10 in has: return 'Mm7'
    if 3 in has and 7 in has and 10 in has: return 'm7'
    if 3 in has and 7 in has and 11 in has: return 'm7'  # mM7 → tratar como m7
    if 4 in has and 7 in has:               return 'M'
    if 3 in has and 7 in has:               return 'm'
    if 5 in has and 7 in has:               return 'sus4'
    if 2 in has and 7 in has:               return 'sus2'
    if 7 in has:                            return 'M'  # solo 5ª: asumir mayor
    return 'M'


# ═══════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE TIMELINE DE ACORDES
# ═══════════════════════════════════════════════════════════

def build_chord_timeline(chords, total_beats):
    """
    Convierte lista de (root_pc, quality, dur_beats) en una timeline continua
    que cubre exactamente total_beats.
    Devuelve lista de (beat_start, beat_end, root_pc, quality).
    """
    timeline = []
    cursor = 0.0

    # Repetir la progresión si es más corta que la melodía
    pattern = list(chords)
    if not pattern:
        return []

    while cursor < total_beats - 0.01:
        for root_pc, quality, dur in pattern:
            if cursor >= total_beats:
                break
            actual_dur = min(dur, total_beats - cursor)
            timeline.append((cursor, cursor + actual_dur, root_pc, quality))
            cursor += actual_dur

    return timeline


def chord_at_beat(timeline, beat):
    """Devuelve (root_pc, quality) del acorde activo en un beat dado."""
    for start, end, root_pc, quality in timeline:
        if start <= beat < end:
            return root_pc, quality
    # Fuera de rango: usar el último
    if timeline:
        return timeline[-1][2], timeline[-1][3]
    return 0, 'M'


# ═══════════════════════════════════════════════════════════
# CLASIFICACIÓN DE NOTAS
# ═══════════════════════════════════════════════════════════

def is_strong_beat(beat_offset, beats_per_bar):
    """True si el beat cae en tiempo fuerte (1er o 3er tiempo en 4/4, 1er en 3/4)."""
    pos = beat_offset % beats_per_bar
    if pos < 0.25:
        return True
    if beats_per_bar >= 4 and abs(pos - beats_per_bar / 2) < 0.25:
        return True
    return False


def is_passing_note(melody, idx, timeline, beats_per_bar):
    """
    Heurística: una nota es de paso si:
    - Está en tiempo débil
    - Las notas anterior y siguiente se mueven por grado conjunto en la misma dirección
    - No es la primera ni la última nota
    """
    if idx == 0 or idx >= len(melody) - 1:
        return False

    prev_p = melody[idx - 1][1]
    curr_p = melody[idx][1]
    next_p = melody[idx + 1][1]

    # Movimiento por grado conjunto en la misma dirección
    step_in  = curr_p - prev_p
    step_out = next_p - curr_p
    if abs(step_in) > 2 or abs(step_out) > 2:
        return False
    if step_in == 0 or step_out == 0:
        return False
    if (step_in > 0) != (step_out > 0):
        return False  # cambia de dirección → no es de paso

    # Tiempo débil
    offset = melody[idx][0]
    return not is_strong_beat(offset, beats_per_bar)


def is_neighbor_note(melody, idx, timeline, beats_per_bar):
    """
    Heurística: nota auxiliar (neighboring note) si:
    - El intervalo de entrada y salida son ambos pasos de semitono o tono
    - La nota anterior y la siguiente son iguales o muy cercanas
    - Está en tiempo débil
    """
    if idx == 0 or idx >= len(melody) - 1:
        return False

    prev_p = melody[idx - 1][1]
    curr_p = melody[idx][1]
    next_p = melody[idx + 1][1]

    if abs(curr_p - prev_p) > 2 or abs(next_p - curr_p) > 2:
        return False
    if abs(next_p - prev_p) > 2:
        return False

    offset = melody[idx][0]
    return not is_strong_beat(offset, beats_per_bar)


# ═══════════════════════════════════════════════════════════
# MOTOR DE ADAPTACIÓN
# ═══════════════════════════════════════════════════════════

def nearest_chord_tone(pitch, root_pc, quality, direction_hint=0):
    """
    Encuentra el chord tone más cercano al pitch dado.
    direction_hint: +1 preferir arriba, -1 preferir abajo, 0 = más cercano.
    """
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
    candidates = []
    for octave in range(2, 8):
        for iv in intervals:
            candidate = root_pc + iv + octave * 12
            candidates.append(candidate)

    if not candidates:
        return pitch

    # Filtrar candidatos en rango razonable (±12 semitonos)
    in_range = [c for c in candidates if abs(c - pitch) <= 12]
    pool = in_range if in_range else candidates

    if direction_hint == 0:
        return min(pool, key=lambda c: abs(c - pitch))
    elif direction_hint > 0:
        above = [c for c in pool if c >= pitch]
        return min(above, key=lambda c: c - pitch) if above else min(pool, key=lambda c: abs(c - pitch))
    else:
        below = [c for c in pool if c <= pitch]
        return max(below, key=lambda c: pitch - c) if below else min(pool, key=lambda c: abs(c - pitch))


def nearest_non_avoid(pitch, root_pc, quality, key_pc, mode):
    """
    Encuentra la nota más cercana que no sea avoid note respecto al acorde.
    Busca primero chord tones, luego tensiones.
    """
    intervals = CHORD_INTERVALS.get(quality, [0, 4, 7])
    # Extensiones permitidas (tensiones)
    tension_ivs = [2, 9, 10]  # 9ª, 13ª, 7ª menor
    all_ivs = list(intervals) + tension_ivs

    candidates = []
    for octave in range(2, 8):
        for iv in all_ivs:
            c = root_pc + iv + octave * 12
            cat = chord_tone_category(c % 12, root_pc, quality)
            if cat != 'avoid':
                candidates.append((c, cat))

    if not candidates:
        return pitch

    in_range = [(c, cat) for c, cat in candidates if abs(c - pitch) <= 12]
    pool = in_range if in_range else candidates

    # Preferir chord_tones primero
    chord_tones = [(c, cat) for c, cat in pool if cat == 'chord_tone']
    target_pool = chord_tones if chord_tones else pool

    return min(target_pool, key=lambda x: abs(x[0] - pitch))[0]


def contour_direction(melody, idx, window=2):
    """
    Estima la dirección del contorno melódico local.
    Devuelve +1 (ascendente), -1 (descendente) o 0 (estático).
    """
    lo = max(0, idx - window)
    hi = min(len(melody) - 1, idx + window)
    if lo == hi:
        return 0
    pitch_lo = melody[lo][1]
    pitch_hi = melody[hi][1]
    diff = pitch_hi - pitch_lo
    if diff > 1:   return 1
    if diff < -1:  return -1
    return 0


def adapt_melody_strict(melody, chord_timeline, beats_per_bar, strength=0.8,
                         key_pc=0, mode='major', verbose=False):
    """
    Modo strict: mueve solo avoid notes en tiempos fuertes.
    Las notas de paso y ornamentos se respetan siempre.
    strength controla la probabilidad de aplicar cada corrección.
    """
    adapted = list(melody)
    log = []

    for i, (offset, pitch, dur, vel) in enumerate(adapted):
        root_pc, quality = chord_at_beat(chord_timeline, offset)
        cat = chord_tone_category(pitch % 12, root_pc, quality)

        # Respetar notas de paso y ornamentos
        if is_passing_note(adapted, i, chord_timeline, beats_per_bar):
            continue
        if is_neighbor_note(adapted, i, chord_timeline, beats_per_bar):
            continue

        strong = is_strong_beat(offset, beats_per_bar)

        # En strict: solo mover avoid notes en tiempos fuertes
        if cat == 'avoid' and strong:
            if np.random.random() > strength:
                continue  # strength controla cuántos aplicamos
            direction = contour_direction(adapted, i)
            new_pitch = nearest_non_avoid(pitch, root_pc, quality, key_pc, mode)
            if new_pitch != pitch:
                adapted[i] = (offset, new_pitch, dur, vel)
                log.append({
                    'note_idx': i, 'offset': round(offset, 3),
                    'original': pitch, 'adapted': new_pitch,
                    'reason': 'avoid_on_strong_beat',
                    'chord': f"{PITCH_NAMES[root_pc]}{quality}",
                })
                if verbose:
                    print(f"    [{i:3d}] beat={offset:.2f}  "
                          f"{PITCH_NAMES[pitch%12]}({pitch})→{PITCH_NAMES[new_pitch%12]}({new_pitch})  "
                          f"[avoid→chord_tone | {PITCH_NAMES[root_pc]}{quality}]")

    return adapted, log


def adapt_melody_smooth(melody, chord_timeline, beats_per_bar, strength=0.8,
                         key_pc=0, mode='major', verbose=False):
    """
    Modo smooth: strict + corrección suave de avoid notes en tiempos débiles
    + inserción de notas de paso para suavizar los saltos introducidos.
    """
    adapted, log = adapt_melody_strict(
        melody, chord_timeline, beats_per_bar, strength, key_pc, mode, verbose)

    # Segunda pasada: avoid notes en tiempos débiles con menor agresividad
    for i, (offset, pitch, dur, vel) in enumerate(adapted):
        root_pc, quality = chord_at_beat(chord_timeline, offset)
        cat = chord_tone_category(pitch % 12, root_pc, quality)
        strong = is_strong_beat(offset, beats_per_bar)

        if cat == 'avoid' and not strong:
            if np.random.random() > strength * 0.5:  # mitad de agresividad
                continue
            if is_passing_note(adapted, i, chord_timeline, beats_per_bar):
                continue
            new_pitch = nearest_non_avoid(pitch, root_pc, quality, key_pc, mode)
            if new_pitch != pitch:
                adapted[i] = (offset, new_pitch, dur, vel)
                log.append({
                    'note_idx': i, 'offset': round(offset, 3),
                    'original': pitch, 'adapted': new_pitch,
                    'reason': 'avoid_on_weak_beat',
                    'chord': f"{PITCH_NAMES[root_pc]}{quality}",
                })

    return adapted, log


def adapt_melody_jazz(melody, chord_timeline, beats_per_bar, strength=0.8,
                       key_pc=0, mode='major', verbose=False):
    """
    Modo jazz: trata 9ª, 11ª# y 13ª como tensiones válidas.
    Solo mueve b9 y b13 en tiempos fuertes (las avoid notes más disonantes).
    """
    adapted = list(melody)
    log = []

    # En jazz, solo b9 (intervalo 1) y b13 (intervalo 8) sobre acordes mayores
    # son verdaderas avoid notes intocables
    JAZZ_AVOID = {1, 8}  # b9, b13

    for i, (offset, pitch, dur, vel) in enumerate(adapted):
        root_pc, quality = chord_at_beat(chord_timeline, offset)
        interval = (pitch % 12 - root_pc) % 12
        strong = is_strong_beat(offset, beats_per_bar)

        if interval in JAZZ_AVOID and strong and quality in ('M', 'M7'):
            if is_passing_note(adapted, i, chord_timeline, beats_per_bar):
                continue
            if np.random.random() > strength:
                continue
            new_pitch = nearest_chord_tone(pitch, root_pc, quality)
            if new_pitch != pitch:
                adapted[i] = (offset, new_pitch, dur, vel)
                log.append({
                    'note_idx': i, 'offset': round(offset, 3),
                    'original': pitch, 'adapted': new_pitch,
                    'reason': 'jazz_avoid_b9_b13',
                    'chord': f"{PITCH_NAMES[root_pc]}{quality}",
                })
                if verbose:
                    print(f"    [{i:3d}] beat={offset:.2f}  jazz: "
                          f"{PITCH_NAMES[pitch%12]}→{PITCH_NAMES[new_pitch%12]}  "
                          f"[b9/b13 | {PITCH_NAMES[root_pc]}{quality}]")

    return adapted, log


def adapt_melody_rewrite(melody, chord_timeline, beats_per_bar, strength=0.8,
                          key_pc=0, mode='major', verbose=False):
    """
    Modo rewrite: reescribe la melodía compás a compás manteniendo el contorno
    general pero ajustando activamente las notas al acorde vigente.
    Agrupa notas por compás y redistribuye usando chord tones + tensiones.
    """
    if not melody:
        return [], []

    adapted = []
    log = []
    total_beats = max(o + d for o, _, d, _ in melody)
    n_bars = max(1, math.ceil(total_beats / beats_per_bar))

    for bar in range(n_bars):
        bar_start = bar * beats_per_bar
        bar_end   = bar_start + beats_per_bar

        # Notas de este compás
        bar_notes = [(o, p, d, v) for o, p, d, v in melody
                     if o >= bar_start and o < bar_end]
        if not bar_notes:
            continue

        root_pc, quality = chord_at_beat(chord_timeline, bar_start)
        ivs = CHORD_INTERVALS.get(quality, [0, 4, 7])

        # Contorno relativo dentro del compás
        pitches = [p for _, p, _, _ in bar_notes]
        if not pitches:
            continue

        p_min, p_max = min(pitches), max(pitches)
        p_range = max(1, p_max - p_min)

        # Construir escala de chord tones en el rango de la melodía original
        target_pitches = []
        for octave in range(3, 7):
            for iv in ivs:
                cp = root_pc + iv + octave * 12
                if p_min - 6 <= cp <= p_max + 6:
                    target_pitches.append(cp)
        target_pitches = sorted(set(target_pitches))

        if not target_pitches:
            adapted.extend(bar_notes)
            continue

        for o, p, d, v in bar_notes:
            root_at, quality_at = chord_at_beat(chord_timeline, o)
            cat = chord_tone_category(p % 12, root_at, quality_at)

            if cat == 'chord_tone' or np.random.random() > strength:
                adapted.append((o, p, d, v))
                continue

            if is_passing_note(adapted + [(o, p, d, v)],
                                len(adapted), chord_timeline, beats_per_bar):
                adapted.append((o, p, d, v))
                continue

            # Mapear el pitch original al chord tone más cercano
            new_p = nearest_chord_tone(p, root_at, quality_at,
                                        direction_hint=contour_direction(
                                            [(oo,pp,dd,vv) for oo,pp,dd,vv in melody], 0))
            if new_p != p:
                log.append({
                    'note_idx': len(adapted), 'offset': round(o, 3),
                    'original': p, 'adapted': new_p,
                    'reason': 'rewrite',
                    'chord': f"{PITCH_NAMES[root_at%12]}{quality_at}",
                })
                if verbose:
                    print(f"    [rewrite] beat={o:.2f}  "
                          f"{PITCH_NAMES[p%12]}({p})→{PITCH_NAMES[new_p%12]}({new_p})  "
                          f"[{PITCH_NAMES[root_at%12]}{quality_at}]")
            adapted.append((o, new_p, d, v))

    return adapted, log


ADAPTATION_MODES = {
    'strict':  adapt_melody_strict,
    'smooth':  adapt_melody_smooth,
    'jazz':    adapt_melody_jazz,
    'rewrite': adapt_melody_rewrite,
}


# ═══════════════════════════════════════════════════════════
# SCORING DE COMPATIBILIDAD
# ═══════════════════════════════════════════════════════════

def score_melody_vs_progression(melody, chord_timeline, beats_per_bar):
    """
    Calcula score global de compatibilidad entre melodía y progresión.
    Devuelve dict con score global, por acorde y lista de colisiones.
    """
    scores_per_chord = []
    collisions = []

    for start, end, root_pc, quality in chord_timeline:
        mel_pcs = melody_pcs_in_window(melody, start, end)
        s = score_chord_vs_melody(mel_pcs, root_pc, quality)
        chord_name = f"{PITCH_NAMES[root_pc%12]}{quality}"
        scores_per_chord.append({'chord': chord_name, 'start': round(start, 2),
                                  'end': round(end, 2), 'score': round(s, 4)})

        # Detectar colisiones (avoid notes en tiempos fuertes)
        for offset, pitch, dur, vel in melody:
            if offset >= start and offset < end:
                cat = chord_tone_category(pitch % 12, root_pc, quality)
                if cat == 'avoid' and is_strong_beat(offset, beats_per_bar):
                    collisions.append({
                        'offset': round(offset, 3),
                        'pitch': pitch,
                        'note_name': PITCH_NAMES[pitch % 12],
                        'chord': chord_name,
                        'category': cat,
                    })

    global_score = float(np.mean([s['score'] for s in scores_per_chord])) if scores_per_chord else 0.0
    return {
        'global_score': round(global_score, 4),
        'by_chord': scores_per_chord,
        'collisions': collisions,
    }


# ═══════════════════════════════════════════════════════════
# ESCRITURA MIDI CON ACORDES
# ═══════════════════════════════════════════════════════════

def build_adapted_midi(melody, chord_timeline, key_obj, tempo_bpm, time_sig,
                        n_bars, acc_style='block', complexity=0.5,
                        no_acc=False, output_path='adapted.mid'):
    """
    Escribe el MIDI con la melodía adaptada + acompañamiento opcional.
    """
    tpb = 480
    beats_per_bar = time_sig[0]
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(1, int(round(beats * tpb)))

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    # ── Track de cabecera ──────────────────────────────────────────────────────
    header = mido.MidiTrack()
    header.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    header.append(mido.MetaMessage('time_signature',
                                   numerator=beats_per_bar, denominator=4,
                                   clocks_per_click=24,
                                   notated_32nd_notes_per_beat=8, time=0))
    header.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(header)

    def notes_to_track(notes_list, ch, program, track_name):
        events = []
        for offset, pitch, dur, vel in notes_list:
            p = max(0, min(127, int(pitch)))
            v = max(1, min(127, int(vel)))
            t_on  = to_ticks(float(offset))
            t_off = to_ticks(float(offset) + max(0.05, float(dur)))
            events.append((t_on,  'on',  p, v))
            events.append((t_off, 'off', p, 0))
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=track_name, time=0))
        trk.append(mido.Message('program_change', channel=ch, program=program, time=0))
        prev_t = 0
        for abs_t, kind, p, v in events:
            dt = max(0, abs_t - prev_t)
            msg_type = 'note_on' if kind == 'on' else 'note_off'
            trk.append(mido.Message(msg_type, channel=ch, note=p,
                                    velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    mid.tracks.append(notes_to_track(melody, ch=0, program=0,
                                     track_name='Melody (adapted)'))

    if not no_acc:
        # Construir notas de acordes y bajo desde la timeline
        acc_notes  = []
        bass_notes = []

        pat_fn = ACC_STYLE_PATTERNS.get(acc_style, ACC_STYLE_PATTERNS.get('block', lambda b: [(0, b)]))

        prev_pitches = None
        key_pc_val = tonic_pc(key_obj) if key_obj else 0
        mode_val = getattr(key_obj, 'mode', 'major') if key_obj else 'major'

        for start, end, root_pc, quality in chord_timeline:
            dur = end - start
            # Construir pitches del acorde con voice-leading
            ints = CHORD_INTERVALS.get(quality, [0, 4, 7])
            if prev_pitches:
                from reharmonizer import _voice_lead_simple
                chord_pitches = _voice_lead_simple(prev_pitches, root_pc, ints, 48, 76)
            else:
                base = root_pc + 48
                while base < 48: base += 12
                chord_pitches = sorted(set([base + i for i in ints
                                             if 44 <= base + i <= 80]))
                if not chord_pitches:
                    chord_pitches = [base]
            prev_pitches = chord_pitches

            # Bajo
            root_bass = root_pc + 36
            while root_bass < 28: root_bass += 12
            while root_bass > 52: root_bass -= 12
            bass_notes.append((start, root_bass, dur * 0.9, 72))

            # Acompañamiento
            sub_pattern = pat_fn(dur)
            for rel_off, sub_dur in sub_pattern:
                abs_off = start + rel_off
                if abs_off + sub_dur > end:
                    sub_dur = max(0.05, end - abs_off)
                for p in chord_pitches:
                    acc_notes.append((abs_off, p, sub_dur * 0.85, 62))

        mid.tracks.append(notes_to_track(acc_notes,  ch=1, program=0,
                                         track_name='Chords'))
        mid.tracks.append(notes_to_track(bass_notes, ch=2, program=32,
                                         track_name='Bass'))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    mid.save(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def adapt(midi_path: str,
          chord_str: str = None,
          chords_midi: str = None,
          mode: str = 'smooth',
          strength: float = 0.8,
          n_bars: int = None,
          key_override: str = None,
          beats_per_bar_override: int = None,
          acc_style: str = 'auto',
          no_acc: bool = False,
          out_dir: str = None,
          report: bool = False,
          verbose: bool = False,
          seed: int = 42):
    """
    Pipeline principal de adaptación melódica.
    """
    np.random.seed(seed)

    midi_path = str(midi_path)
    stem = Path(midi_path).stem
    if out_dir is None:
        out_dir = str(Path(midi_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*64}")
    print(f"  MELODY ADAPTER  ·  {os.path.basename(midi_path)}")
    print(f"{'═'*64}")

    # ── Cargar melodía ──────────────────────────────────────────────────────
    if REHAR_OK:
        try:
            melody, tempo_bpm, key_obj, tpb, time_sig = load_melody(midi_path, verbose=verbose)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {}
    else:
        # Fallback: cargar con mido directamente
        try:
            mid = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"[ERROR] No se pudo abrir {midi_path}: {e}")
            return {}
        tpb = mid.ticks_per_beat or 480
        tempo_us = 500_000
        ts_num, ts_den = 4, 4
        notes_by_ch = defaultdict(list)
        pending = {}
        for track in mid.tracks:
            abs_t = 0
            for msg in track:
                abs_t += msg.time
                if msg.type == 'set_tempo': tempo_us = msg.tempo
                elif msg.type == 'time_signature': ts_num, ts_den = msg.numerator, msg.denominator
                elif msg.type == 'note_on' and msg.velocity > 0:
                    pending[(msg.channel, msg.note)] = (abs_t / tpb, msg.velocity)
                elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                    k = (msg.channel, msg.note)
                    if k in pending:
                        on_b, vel = pending.pop(k)
                        notes_by_ch[msg.channel].append(
                            (on_b, msg.note, max(0.1, abs_t / tpb - on_b), vel))
        if not notes_by_ch:
            print("[ERROR] No se encontraron notas en el MIDI.")
            return {}
        mel_ch = max(notes_by_ch, key=lambda c: np.mean([p for _, p, _, _ in notes_by_ch[c]]))
        melody = sorted(notes_by_ch[mel_ch])
        tempo_bpm = round(60_000_000 / tempo_us, 2)
        key_obj = None
        time_sig = (ts_num, ts_den)

    if key_override:
        key_obj = parse_key_string(key_override) if REHAR_OK else None
        print(f"  Tonalidad forzada: {key_override}")

    beats_per_bar = beats_per_bar_override or time_sig[0]
    total_beats = max(o + d for o, _, d, _ in melody) if melody else 0

    if n_bars is None:
        n_bars = max(4, math.ceil(total_beats / beats_per_bar))

    key_pc_val = tonic_pc(key_obj) if key_obj else 0
    mode_val   = getattr(key_obj, 'mode', 'major') if key_obj else 'major'

    if acc_style == 'auto':
        acc_style = 'arpeggio' if tempo_bpm >= 100 else 'block'

    print(f"  Tonalidad  : {key_obj or 'desconocida'}  (PC={key_pc_val}, modo={mode_val})")
    print(f"  Tempo      : {tempo_bpm} BPM")
    print(f"  Compases   : {n_bars}  ({beats_per_bar}/4)")
    print(f"  Notas mel. : {len(melody)}")
    print(f"  Modo       : {mode}  |  strength={strength}")

    # ── Parsear progresión ─────────────────────────────────────────────────
    if chord_str:
        print(f"\n  Progresión : \"{chord_str}\"")
        chords = parse_chord_string(chord_str, beats_per_bar)
    elif chords_midi:
        print(f"\n  Acordes desde MIDI: {os.path.basename(chords_midi)}")
        try:
            chords = extract_chords_from_midi(chords_midi, beats_per_bar)
        except Exception as e:
            print(f"[ERROR] {e}")
            return {}
    else:
        print("[ERROR] Debes especificar --chords o --chords-midi.")
        return {}

    if not chords:
        print("[ERROR] No se pudo parsear ningún acorde.")
        return {}

    # Resumen de la progresión
    prog_summary = '  '.join(
        f"{PITCH_NAMES[r]}{q}({d:.0f}b)" for r, q, d in chords[:8])
    if len(chords) > 8:
        prog_summary += ' …'
    print(f"  Acordes    : {prog_summary}")

    # ── Construir timeline ─────────────────────────────────────────────────
    chord_timeline = build_chord_timeline(chords, total_beats)

    # ── Score antes ────────────────────────────────────────────────────────
    score_before = score_melody_vs_progression(melody, chord_timeline, beats_per_bar)
    print(f"\n  Score antes : {score_before['global_score']:.3f}  "
          f"({len(score_before['collisions'])} colisiones en tiempos fuertes)")

    if verbose and score_before['collisions']:
        print("  Colisiones detectadas:")
        for col in score_before['collisions']:
            print(f"    beat={col['offset']:.2f}  "
                  f"{col['note_name']}({col['pitch']})  sobre  {col['chord']}")

    # ── Adaptar ───────────────────────────────────────────────────────────
    adapt_fn = ADAPTATION_MODES.get(mode, adapt_melody_smooth)
    print(f"\n  Adaptando…")
    adapted_melody, adaptation_log = adapt_fn(
        melody, chord_timeline, beats_per_bar,
        strength=strength, key_pc=key_pc_val, mode=mode_val, verbose=verbose
    )

    # ── Score después ──────────────────────────────────────────────────────
    score_after = score_melody_vs_progression(adapted_melody, chord_timeline, beats_per_bar)
    improvement = score_after['global_score'] - score_before['global_score']
    collisions_resolved = len(score_before['collisions']) - len(score_after['collisions'])

    print(f"  Score después: {score_after['global_score']:.3f}  "
          f"(+{improvement:+.3f})  "
          f"{len(score_after['collisions'])} colisiones restantes")
    print(f"  Notas modificadas: {len(adaptation_log)} / {len(melody)}")
    print(f"  Colisiones resueltas: {collisions_resolved} / "
          f"{len(score_before['collisions'])}")

    # ── Escribir MIDI ──────────────────────────────────────────────────────
    out_name = f"{stem}.adapted_{mode}.mid"
    out_path = os.path.join(out_dir, out_name)

    try:
        build_adapted_midi(
            melody=adapted_melody,
            chord_timeline=chord_timeline,
            key_obj=key_obj,
            tempo_bpm=tempo_bpm,
            time_sig=time_sig,
            n_bars=n_bars,
            acc_style=acc_style,
            no_acc=no_acc,
            output_path=out_path,
        )
        print(f"\n  → {out_name}")
    except Exception as e:
        print(f"  [ERROR] No se pudo escribir el MIDI: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    # ── Reporte ────────────────────────────────────────────────────────────
    report_data = {
        'source': midi_path,
        'mode': mode,
        'strength': strength,
        'key': str(key_obj) if key_obj else 'unknown',
        'tempo_bpm': tempo_bpm,
        'n_bars': n_bars,
        'beats_per_bar': beats_per_bar,
        'chords': [
            {'root': PITCH_NAMES[r], 'quality': q, 'duration_beats': d}
            for r, q, d in chords
        ],
        'score_before': score_before,
        'score_after': score_after,
        'improvement': round(improvement, 4),
        'notes_modified': len(adaptation_log),
        'total_notes': len(melody),
        'collisions_before': len(score_before['collisions']),
        'collisions_after': len(score_after['collisions']),
        'collisions_resolved': collisions_resolved,
        'adaptation_log': adaptation_log,
        'output': out_path,
    }

    if report:
        report_path = os.path.join(out_dir, f"{stem}.adapted_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  Reporte: {os.path.basename(report_path)}")

    print(f"{'═'*64}\n")
    return report_data


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='MELODY ADAPTER — Adaptación de melodías a progresiones de acordes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de adaptación:
  strict    Mueve solo avoid notes en tiempos fuertes. Mínima intervención.
  smooth    strict + corrección suave en tiempos débiles. (default)
  jazz      Solo mueve b9/b13; trata 9ª, 11ª# y 13ª como tensiones válidas.
  rewrite   Reescribe activamente la melodía para maximizar compatibilidad.

Formato de progresión:
  "C Am F G"              → un acorde por compás (beats_per_bar pulsos c/u)
  "C:2 Am:2 F:1 G:3"     → duración en beats tras ':'
  "Cmaj7 Am7 Fmaj7 G7"   → acordes con calidad explícita
  "Dm7:2 G7:2 Cmaj7:4"   → ii-V-I en compás de 4/4

Nombres de acordes soportados:
  C Dm Em F G Am Bdim      (tríadas)
  Cmaj7 Dm7 G7 Fm7b5      (7ª)
  Caug Cdim Csus4 Csus2   (otros)
  C/E Dm/F                 (inversiones — se ignora el bajo)

Ejemplos:
  python melody_adapter.py melodia.mid --chords "C Am F G"
  python melody_adapter.py melodia.mid --chords "Dm7:2 G7:2 Cmaj7:4" --mode jazz
  python melody_adapter.py melodia.mid --chords-midi acomp.mid --mode smooth
  python melody_adapter.py melodia.mid --chords "C Am F G" --strength 0.5 --report
  python melody_adapter.py melodia.mid --chords "C Am F G" --no-acc --out-dir ./out
        """
    )
    p.add_argument('input',
                   help='MIDI de melodía a adaptar')
    p.add_argument('--chords', default=None, metavar='PROG',
                   help='Progresión de acordes en texto (ver formato arriba)')
    p.add_argument('--chords-midi', default=None, metavar='FILE',
                   help='Pista de acordes en MIDI')
    p.add_argument('--mode', default='smooth',
                   choices=['strict', 'smooth', 'jazz', 'rewrite'],
                   help='Modo de adaptación (default: smooth)')
    p.add_argument('--strength', type=float, default=0.8,
                   help='Agresividad de la adaptación 0-1 (default: 0.8)')
    p.add_argument('--bars', type=int, default=None,
                   help='Compases de salida (default: auto)')
    p.add_argument('--key', default=None, metavar='KEY',
                   help='Tonalidad forzada, e.g. "D minor" o "F# major"')
    p.add_argument('--beats-per-bar', type=int, default=None,
                   help='Pulsos por compás (default: detectado del MIDI)')
    p.add_argument('--acc-style', default='auto',
                   choices=['auto', 'block', 'arpeggio', 'alberti', 'waltz', 'jazz_voicing'],
                   help='Estilo de acompañamiento (default: auto)')
    p.add_argument('--no-acc', action='store_true',
                   help='No generar acompañamiento, solo la melodía adaptada')
    p.add_argument('--out-dir', default=None,
                   help='Carpeta de salida (default: misma carpeta que el MIDI)')
    p.add_argument('--report', action='store_true',
                   help='Guardar JSON con análisis de compatibilidad')
    p.add_argument('--seed', type=int, default=42,
                   help='Semilla aleatoria (default: 42)')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] No encontrado: {args.input}")
        sys.exit(1)

    if not args.chords and not args.chords_midi:
        print("[ERROR] Debes especificar --chords o --chords-midi.")
        parser.print_help()
        sys.exit(1)

    adapt(
        midi_path=args.input,
        chord_str=args.chords,
        chords_midi=args.chords_midi,
        mode=args.mode,
        strength=args.strength,
        n_bars=args.bars,
        key_override=args.key,
        beats_per_bar_override=args.beats_per_bar,
        acc_style=args.acc_style,
        no_acc=args.no_acc,
        out_dir=args.out_dir,
        report=args.report,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
