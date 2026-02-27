#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       REHARMONIZER  v1.0                                    ║
║         Reharmonización guiada de melodías con progresiones alternativas    ║
║                                                                              ║
║  Toma una melodía fija y genera N versiones con progresiones armónicas      ║
║  alternativas, aplicando técnicas de reharmonización clásica, jazzística    ║
║  y modal mientras valida la compatibilidad de cada nota melódica.           ║
║                                                                              ║
║  ESTRATEGIAS DISPONIBLES (--strategy):                                      ║
║    diatonic       — progresiones diatónicas de la tonalidad detectada       ║
║    tritone        — sustitución de tritono (bII7 en lugar de V7)            ║
║    secondary      — dominantes secundarios (V/ii, V/IV, V/V…)              ║
║    modal_interchange — préstamos del modo paralelo (bVII, bIII, iv…)       ║
║    chromatic_med  — mediante cromática (III, bVI, bII como mediantes)       ║
║    coltrane       — ciclos de Coltrane (terceras mayores equidistantes)     ║
║    neapolitan     — acorde napolitano y submediante                         ║
║    pedal          — pedal de tónica o dominante con armonías sobre él       ║
║    minor_modal    — reharmonización en modos menores (dorico, frigio, etc.) ║
║    baroque        — progresiones funcionales barrocas                       ║
║    impressionist  — progresiones por tonos enteros y cuartas                ║
║    all            — prueba todas las estrategias                             ║
║                                                                              ║
║  COMPATIBILIDAD MELÓDICA:                                                   ║
║  Cada acorde se valida contra las notas melódicas que lo acompañan.        ║
║  Categorías: chord_tone, tension, avoid_note. Los acordes con 'avoid'      ║
║  se penalizan o se voicea-lideran hacia alternativas más compatibles.      ║
║                                                                              ║
║  USO:                                                                        ║
║    python reharmonizer.py melodia.mid                                        ║
║    python reharmonizer.py melodia.mid --strategy tritone secondary          ║
║    python reharmonizer.py melodia.mid --strategy all --out-dir salidas/     ║
║    python reharmonizer.py melodia.mid --key "D minor" --bars 16            ║
║    python reharmonizer.py melodia.mid --strategy coltrane --candidates 5   ║
║    python reharmonizer.py melodia.mid --acc-style block --tempo 90         ║
║    python reharmonizer.py melodia.mid --strategy all --score-threshold 0.4 ║
║    python reharmonizer.py melodia.mid --report                              ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --strategy S [S…]  Estrategias a aplicar (default: diatonic tritone      ║
║                       secondary modal_interchange)                          ║
║    --key "C major"    Tonalidad origen. Si no se especifica, se autodetecta ║
║    --bars N           Compases de salida (default: auto desde melodía)      ║
║    --candidates N     Candidatos por estrategia (default: 3)                ║
║    --acc-style S      Estilo de acompañamiento: block | arpeggio | alberti  ║
║                       waltz | jazz_voicing (default: auto)                  ║
║    --tempo BPM        Tempo de salida (default: detectado del MIDI)         ║
║    --complexity F     Complejidad armónica 0-1 (default: 0.5)              ║
║    --score-threshold  Umbral mínimo de score para guardar (default: 0.25)  ║
║    --out-dir DIR      Carpeta de salida (default: junto al MIDI de entrada) ║
║    --report           Guardar reporte JSON con análisis completo            ║
║    --verbose          Informe detallado por stdout                           ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    melodia.rehar_tritone_01.mid                                              ║
║    melodia.rehar_coltrane_01.mid                                             ║
║    melodia.rehar_report.json  (con --report)                                ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21, numpy  (+ midi_dna_unified en mismo dir)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import copy
import math
import random
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import mido

# ── música21 ──────────────────────────────────────────────────────────────────
try:
    from music21 import converter, pitch as m21pitch, key as m21key
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Se usará detección de tonalidad simplificada.")

# ── midi_dna_unified (generación de acompañamiento/bajo) ─────────────────────
_DNA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DNA_DIR)
try:
    import midi_dna_unified as _dna
    from midi_dna_unified import (
        _snap_to_scale, _get_scale_pcs, _get_scale_midi,
        _build_chord_pitches_from_roman, _voice_lead, voice_lead_next_chord,
        _quarter_to_ticks, _clamp_pitch,
        generate_accompaniment, generate_bass, build_midi,
        MAJOR_SCALE_DEGREES, MINOR_SCALE_DEGREES,
        INSTRUMENT_RANGES,
    )
    DNA_OK = True
except ImportError as e:
    DNA_OK = False
    print(f"[AVISO] midi_dna_unified no encontrado: {e}\n"
          "Los acompañamientos generados serán simplificados.")

# ═══════════════════════════════════════════════════════════
# CONSTANTES DE REHARMONIZACIÓN
# ═══════════════════════════════════════════════════════════

# Notas del círculo de quintas
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Intervalos de cada calidad de acorde
CHORD_INTERVALS = {
    'M':   [0, 4, 7],
    'm':   [0, 3, 7],
    'd':   [0, 3, 6],
    'A':   [0, 4, 8],
    'M7':  [0, 4, 7, 11],
    'm7':  [0, 3, 7, 10],
    'Mm7': [0, 4, 7, 10],   # dominante 7ª
    'd7':  [0, 3, 6, 9],
    'hd7': [0, 3, 6, 10],   # semidisminuido
    'M9':  [0, 4, 7, 11, 14],
    'm9':  [0, 3, 7, 10, 14],
    'Mm9': [0, 4, 7, 10, 14],
    'sus4':[0, 5, 7],
    'sus2':[0, 2, 7],
    '6':   [0, 4, 7, 9],
    'm6':  [0, 3, 7, 9],
}

# Clasificación de nota melódica vs acorde
# (intervalo desde raíz del acorde) → categoría
NOTE_CATEGORY = {
    0:  'chord_tone',   # fundamental
    2:  'tension',      # 9ª
    3:  'chord_tone',   # 3ª menor
    4:  'chord_tone',   # 3ª mayor
    5:  'tension',      # 11ª (avoid en M)
    6:  'avoid',        # tritonus (en dominante es tension, resto avoid)
    7:  'chord_tone',   # 5ª
    9:  'tension',      # 13ª / 6ª
    10: 'tension',      # 7ª menor
    11: 'tension',      # 7ª mayor (avoid en dominante)
    1:  'avoid',        # b9 (evitar sobre acordes mayores)
    8:  'avoid',        # b13 en acordes mayores
}


def chord_tone_category(mel_pc, root_pc, quality):
    """
    Clasifica una nota melódica (pitch class) respecto a un acorde.
    Devuelve: 'chord_tone' | 'tension' | 'avoid'
    """
    interval = (mel_pc - root_pc) % 12
    base_cat = NOTE_CATEGORY.get(interval, 'tension')

    # Ajustes específicos por calidad
    if quality in ('Mm7', 'M7', 'd7', 'hd7'):
        if interval == 11:  # 7ma mayor sobre dominante = avoid
            return 'avoid'
        if interval == 6:   # #11 sobre dominante es tension (lydian dom)
            return 'tension'
    if quality in ('M', 'M7'):
        if interval == 5:   # 11ª natural sobre mayor = avoid
            return 'avoid'
    return base_cat


def score_chord_vs_melody(mel_pcs, root_pc, quality):
    """
    Score de compatibilidad de un acorde con las notas melódicas.
    Devuelve float en [0, 1]. 1 = perfectamente compatible.
    """
    if not mel_pcs:
        return 0.8  # sin notas: neutral
    scores = []
    for pc in mel_pcs:
        cat = chord_tone_category(pc, root_pc, quality)
        scores.append({'chord_tone': 1.0, 'tension': 0.6, 'avoid': 0.0}[cat])
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════
# CARGA DE MELODÍA
# ═══════════════════════════════════════════════════════════

def load_melody(midi_path, verbose=False):
    """
    Carga la melodía principal de un MIDI.
    Devuelve:
        melody  : list of (offset_beats, pitch_midi, duration_beats, velocity)
        tempo   : BPM float
        key_obj : music21 Key o None
        tpb     : ticks_per_beat
        ts      : (numerator, denominator)
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4

    # Recopilar notas por canal con tiempo absoluto
    notes_by_ch = defaultdict(list)
    pending = {}

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
            elif msg.type in ('note_off', 'note_on'):
                key = (msg.channel, msg.note)
                if key in pending:
                    on_t, vel = pending.pop(key)
                    off_t = abs_t
                    on_beat  = on_t  / tpb
                    dur_beat = max(0.1, (off_t - on_t) / tpb)
                    notes_by_ch[msg.channel].append(
                        (on_beat, msg.note, dur_beat, vel)
                    )

    if not notes_by_ch:
        raise RuntimeError("No se encontraron notas en el MIDI.")

    # Canal con pitch medio más alto → melodía
    def mean_pitch(notes):
        return np.mean([p for _, p, _, _ in notes]) if notes else 0

    mel_ch = max(notes_by_ch, key=lambda c: mean_pitch(notes_by_ch[c]))
    melody = sorted(notes_by_ch[mel_ch], key=lambda x: x[0])

    tempo_bpm = round(60_000_000 / tempo_us, 2)
    if verbose:
        print(f"  Melodía: {len(melody)} notas, canal {mel_ch}")
        print(f"  Tempo: {tempo_bpm} BPM, compás: {ts_num}/{ts_den}")

    # Detectar tonalidad
    key_obj = None
    if MUSIC21_OK:
        try:
            score_m21 = converter.parse(midi_path)
            key_obj = score_m21.analyze('key')
            if verbose:
                print(f"  Tonalidad detectada: {key_obj}")
        except Exception:
            pass

    if key_obj is None:
        # Fallback: C major
        if MUSIC21_OK:
            key_obj = m21key.Key('C', 'major')
        if verbose:
            print("  Tonalidad: C major (fallback)")

    return melody, tempo_bpm, key_obj, tpb, (ts_num, ts_den)


def melody_total_beats(melody):
    if not melody:
        return 0.0
    last = max(melody, key=lambda n: n[0] + n[2])
    return last[0] + last[2]


def melody_pcs_in_window(melody, beat_start, beat_end):
    """Pitch classes de notas melódicas en [beat_start, beat_end)."""
    pcs = set()
    for offset, pitch, dur, vel in melody:
        if offset < beat_end and offset + dur > beat_start:
            pcs.add(pitch % 12)
    return pcs


# ═══════════════════════════════════════════════════════════
# TONALIDAD AUXILIAR (cuando no hay music21)
# ═══════════════════════════════════════════════════════════

class SimpleKey:
    """Proxy mínimo que imita music21.Key para los casos sin music21."""
    def __init__(self, tonic_name='C', mode='major'):
        self.tonic = type('T', (), {'name': tonic_name, 'pitchClass': PITCH_NAMES.index(tonic_name)})()
        self.mode = mode

    def __str__(self):
        return f"{self.tonic.name} {self.mode}"


def parse_key_string(key_str):
    """Parsea 'D minor', 'F# major', etc. → key_obj."""
    parts = key_str.strip().split()
    tonic_name = parts[0].replace('b', '#') if len(parts) > 0 else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    if MUSIC21_OK:
        try:
            return m21key.Key(tonic_name, mode)
        except Exception:
            pass
    return SimpleKey(tonic_name, mode)


def tonic_pc(key_obj):
    try:
        return m21pitch.Pitch(key_obj.tonic.name).pitchClass
    except Exception:
        return PITCH_NAMES.index(key_obj.tonic.name) if key_obj.tonic.name in PITCH_NAMES else 0


def make_key(tonic_pc_val, mode='major'):
    name = PITCH_NAMES[tonic_pc_val % 12]
    if MUSIC21_OK:
        try:
            return m21key.Key(name, mode)
        except Exception:
            pass
    return SimpleKey(name, mode)


# ═══════════════════════════════════════════════════════════
# PROGRESIONES: DEFINICIÓN Y EXPANSIÓN
# ═══════════════════════════════════════════════════════════

# Cada progresión es lista de (roman_figure, beats_duration)
# El campo 'mode' permite filtrar por major/minor/both
PROGRESSION_LIBRARY = {
    # ── DIATÓNICAS ──────────────────────────────────────────
    'diatonic': {
        'major': [
            [('I', 2), ('IV', 2), ('V', 2), ('I', 2)],
            [('I', 2), ('vi', 2), ('ii', 2), ('V', 2)],
            [('I', 1), ('V', 1), ('vi', 2), ('IV', 2)],
            [('I', 2), ('iii', 2), ('IV', 2), ('V', 2)],
            [('IV', 2), ('I', 2), ('V', 2), ('I', 2)],
            [('I', 2), ('IV', 1), ('I', 1), ('V', 2), ('I', 2)],
            [('I', 2), ('ii', 2), ('V', 2), ('I', 2)],
            [('vi', 2), ('IV', 2), ('I', 2), ('V', 2)],
            [('I', 1), ('I', 1), ('IV', 2), ('I', 2), ('V', 2)],
        ],
        'minor': [
            [('i', 2), ('iv', 2), ('V', 2), ('i', 2)],
            [('i', 2), ('VII', 2), ('VI', 2), ('VII', 2)],
            [('i', 2), ('VI', 2), ('III', 2), ('VII', 2)],
            [('i', 2), ('iv', 2), ('VII', 2), ('III', 2)],
            [('i', 1), ('VII', 1), ('VI', 2), ('V', 2)],
            [('i', 2), ('ii°', 2), ('V', 2), ('i', 2)],
        ],
    },

    # ── SUSTITUCIÓN DE TRITONO ───────────────────────────────
    'tritone': {
        'major': [
            # bII7 sustituye a V7
            [('I', 2), ('bII', 2), ('I', 2), ('bII', 2)],
            [('I', 1), ('vi', 1), ('bII', 2), ('I', 2)],
            [('I', 2), ('IV', 2), ('bII', 2), ('I', 2)],
            [('ii', 2), ('bII', 2), ('I', 2), ('vi', 2)],
            [('I', 1), ('iii', 1), ('IV', 1), ('bII', 1), ('I', 2)],
            [('I', 2), ('bVII', 2), ('bII', 2), ('I', 2)],
        ],
        'minor': [
            [('i', 2), ('bII', 2), ('i', 2), ('V', 2)],
            [('i', 2), ('VI', 2), ('bII', 2), ('i', 2)],
            [('i', 1), ('iv', 1), ('bII', 2), ('i', 2)],
        ],
    },

    # ── DOMINANTES SECUNDARIOS ───────────────────────────────
    'secondary': {
        'major': [
            [('I', 2), ('V/V', 2), ('V', 2), ('I', 2)],
            [('I', 1), ('V/ii', 1), ('ii', 2), ('V', 2)],
            [('IV', 2), ('V/V', 2), ('V', 2), ('I', 2)],
            [('I', 1), ('V/vi', 1), ('vi', 2), ('V', 2)],
            [('I', 1), ('V/IV', 1), ('IV', 1), ('V', 1), ('I', 2)],
            [('ii', 2), ('V/V', 2), ('V', 2), ('I', 2)],
            [('I', 1), ('V/ii', 1), ('ii', 1), ('V/V', 1), ('V', 2), ('I', 2)],
        ],
        'minor': [
            [('i', 2), ('V/III', 2), ('III', 2), ('V', 2)],
            [('i', 1), ('V/iv', 1), ('iv', 2), ('V', 2)],
            [('i', 2), ('V/VII', 2), ('VII', 2), ('i', 2)],
        ],
    },

    # ── INTERCAMBIO MODAL ────────────────────────────────────
    'modal_interchange': {
        'major': [
            # Préstamos del modo paralelo menor
            [('I', 2), ('bVII', 2), ('IV', 2), ('I', 2)],
            [('I', 2), ('bVI', 2), ('bVII', 2), ('I', 2)],
            [('I', 2), ('iv', 2), ('I', 2), ('V', 2)],
            [('I', 2), ('bIII', 2), ('bVII', 2), ('I', 2)],
            [('IV', 2), ('iv', 2), ('I', 2), ('V', 2)],
            [('I', 1), ('bVI', 1), ('bVII', 2), ('I', 2)],
            [('I', 2), ('bVI', 2), ('bIII', 2), ('bVII', 2)],
            [('I', 1), ('bVII', 1), ('bVI', 2), ('V', 2)],
        ],
        'minor': [
            # Préstamos del mayor paralelo
            [('i', 2), ('IV', 2), ('i', 2), ('V', 2)],
            [('i', 2), ('I', 2), ('IV', 2), ('V', 2)],
            [('i', 2), ('bVII', 2), ('I', 2), ('V', 2)],
        ],
    },

    # ── MEDIANTE CROMÁTICA ───────────────────────────────────
    'chromatic_med': {
        'major': [
            [('I', 2), ('III', 2), ('IV', 2), ('V', 2)],
            [('I', 2), ('bVI', 2), ('IV', 2), ('V', 2)],
            [('I', 2), ('bII', 2), ('bVI', 2), ('I', 2)],
            [('I', 1), ('III', 1), ('vi', 2), ('V', 2)],
            [('I', 2), ('bIII', 2), ('bVI', 2), ('bII', 2)],
        ],
        'minor': [
            [('i', 2), ('bII', 2), ('V', 2), ('i', 2)],
            [('i', 2), ('III', 2), ('bVI', 2), ('V', 2)],
            [('i', 1), ('bII', 1), ('bVI', 2), ('i', 2)],
        ],
    },

    # ── CICLOS DE COLTRANE ───────────────────────────────────
    # Subdivisión en terceras mayores (ciclo de 3 centros tonales separados por M3)
    'coltrane': {
        'major': [
            [('I', 2), ('bVI', 2), ('bIII', 2), ('I', 2)],
            [('I', 1), ('V/bVI', 1), ('bVI', 1), ('V/bIII', 1), ('bIII', 2), ('I', 2)],
            [('I', 2), ('bVI', 1), ('bIII', 1), ('I', 2), ('bVI', 2)],
            [('I', 1), ('bII', 1), ('bVI', 1), ('bIII', 1), ('I', 2)],
        ],
        'minor': [
            [('i', 2), ('VI', 2), ('III', 2), ('i', 2)],
            [('i', 2), ('bVI', 2), ('bIII', 2), ('i', 2)],
        ],
    },

    # ── NAPOLITANA ───────────────────────────────────────────
    'neapolitan': {
        'major': [
            [('I', 2), ('bII', 2), ('V', 2), ('I', 2)],
            [('i', 2), ('bII', 2), ('V7', 2), ('i', 2)],
            [('I', 1), ('ii', 1), ('bII', 2), ('V', 2)],
            [('I', 2), ('IV', 1), ('bII', 1), ('V', 2), ('I', 2)],
        ],
        'minor': [
            [('i', 2), ('bII', 2), ('V', 2), ('i', 2)],
            [('i', 1), ('iv', 1), ('bII', 2), ('V', 2)],
        ],
    },

    # ── PEDAL ────────────────────────────────────────────────
    # La nota de bajo se mantiene fija mientras la armonía cambia
    'pedal': {
        'major': [
            # Pedal de tónica: acorde over I
            [('I', 4), ('IV/I', 2), ('V/I', 2)],        # IV y V sobre bajo I
            [('I', 2), ('ii/I', 2), ('V/I', 2), ('I', 2)],
            [('I', 1), ('bVII/I', 1), ('IV/I', 2), ('I', 2)],
            # Pedal de dominante
            [('V', 2), ('IV/V', 2), ('I/V', 2), ('V', 2)],
        ],
        'minor': [
            [('i', 4), ('iv/i', 2), ('V/i', 2)],
            [('i', 2), ('VII/i', 2), ('iv/i', 2), ('i', 2)],
        ],
    },

    # ── MODOS MENORES ────────────────────────────────────────
    'minor_modal': {
        'major': [
            # Dórico
            [('i', 2), ('IV', 2), ('i', 2), ('VII', 2)],
            # Frigio
            [('i', 2), ('bII', 2), ('i', 2), ('VII', 2)],
            # Lidio
            [('I', 2), ('II', 2), ('VII', 2), ('I', 2)],
            # Mixolidio
            [('I', 2), ('bVII', 2), ('IV', 2), ('I', 2)],
            # Locrio
            [('i°', 2), ('bII', 2), ('bVII', 2), ('i°', 2)],
        ],
        'minor': [
            # Dórico
            [('i', 2), ('IV', 2), ('VII', 2), ('i', 2)],
            # Frigio
            [('i', 2), ('bII', 2), ('VII', 2), ('i', 2)],
            # Eólico natural
            [('i', 2), ('VII', 2), ('VI', 2), ('VII', 2)],
        ],
    },

    # ── BARROCO ──────────────────────────────────────────────
    'baroque': {
        'major': [
            [('I', 1), ('V', 1), ('vi', 1), ('iii', 1), ('IV', 1), ('I', 1), ('V', 2)],
            [('I', 2), ('IV', 2), ('ii', 2), ('V', 2)],
            [('I', 1), ('ii', 1), ('V/V', 1), ('V', 1), ('I', 2)],
            [('I', 1), ('IV', 1), ('V', 1), ('vi', 1), ('IV', 2), ('V', 2)],
        ],
        'minor': [
            [('i', 1), ('V', 1), ('i', 1), ('iv', 1), ('i', 1), ('V', 1), ('i', 2)],
            [('i', 2), ('iv', 2), ('ii°', 2), ('V', 2)],
            [('i', 1), ('VII', 1), ('VI', 1), ('V', 1), ('i', 2)],
        ],
    },

    # ── IMPRESIONISMO ────────────────────────────────────────
    'impressionist': {
        'major': [
            # Paralelismo de acordes (por tonos enteros o semitonos)
            [('I', 2), ('II', 2), ('III', 2), ('II', 2)],
            [('I', 2), ('bIII', 2), ('bV', 2), ('bVII', 2)],
            [('I', 2), ('bVI', 2), ('bIII', 2), ('bVII', 2)],
            [('I', 1), ('bII', 1), ('bIII', 1), ('bII', 1), ('I', 2)],
            [('I', 2), ('IV', 2), ('bVII', 2), ('bIII', 2)],
        ],
        'minor': [
            [('i', 2), ('bII', 2), ('bIII', 2), ('bII', 2)],
            [('i', 2), ('VI', 2), ('bIII', 2), ('bVII', 2)],
        ],
    },
}

# Numeral → semitono desde tónica (para acordes no estándar)
EXTRA_FIGURES = {
    'bV':   6, 'i°': 0, 'II': 2, 'III': 4,
    'V/I':  7, 'IV/I': 5, 'ii/I': 2, 'bVII/I': 10,
    'I/V':  0, 'IV/V': 5, 'V/i': 7, 'iv/i': 5,
    'VII/i':10, 'bII/i':1,
    'V/bVI': 3, 'V/bIII': 8,  # dominantes secundarios de Coltrane
    'V/IV':  0, 'V/ii': 9, 'V/vi': 4, 'V/V': 2,
    'V/III': 11, 'V/VII': 9, 'V/bIII': 8,
    'V7': 7,
}


def figure_to_root_pc(figure, key_pc, mode):
    """
    Convierte un numeral romano a pitch class de la raíz.
    key_pc: pitch class de la tónica. mode: 'major' | 'minor'.
    """
    deg_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
    base = figure.replace('7','').replace('°','').replace('+','').replace('9','')
    if base in deg_map:
        return (key_pc + deg_map[base][0]) % 12
    if figure in EXTRA_FIGURES:
        return (key_pc + EXTRA_FIGURES[figure]) % 12
    # Intentar inferir desde nombre
    return key_pc


def figure_to_quality(figure, mode):
    """Extrae la calidad (M, m, d, Mm7…) desde el numeral romano."""
    deg_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
    base = figure.replace('7','').replace('°','').replace('+','').replace('9','')
    if base in deg_map:
        q = deg_map[base][1]
    else:
        # heurístico
        if figure.startswith('b') or figure[0].isupper():
            q = 'M'
        else:
            q = 'm'
    if '7' in figure and 'Mm7' not in q and 'M7' not in q:
        if q == 'M':   q = 'Mm7'
        elif q == 'm': q = 'm7'
        elif q == 'd': q = 'd7'
    if '°' in figure and '7' not in figure:
        q = 'd'
    return q


def build_chord_pitches(figure, key_pc, mode, prev_pitches=None,
                        complexity=0.5, register_lo=48, register_hi=76):
    """
    Construye lista de pitches MIDI para un acorde dado el numeral romano.
    Aplica voice-leading si se suministran pitches del acorde anterior.
    """
    root = figure_to_root_pc(figure, key_pc, mode)
    quality = figure_to_quality(figure, mode)

    # Añadir 7ª según complejidad
    if complexity > 0.5 and '7' not in quality and quality in ('M', 'm'):
        quality = 'Mm7' if quality == 'M' else 'm7'

    ints = CHORD_INTERVALS.get(quality, [0, 4, 7])

    if prev_pitches:
        return _voice_lead_simple(prev_pitches, root, ints, register_lo, register_hi)

    # Construir acorde en posición fundamental
    base = root + 48
    while base < register_lo:
        base += 12
    while base > register_lo + 12:
        base -= 12
    pitches = [base + i for i in ints]
    pitches = [p for p in pitches if register_lo - 4 <= p <= register_hi + 4]
    return pitches or [base]


def _voice_lead_simple(prev, root_pc, intervals, lo, hi):
    """Voice-leading mínimo: minimiza movimiento de cada voz."""
    candidates = []
    for o in range(3, 7):
        for i in intervals:
            m = root_pc + i + o * 12
            if lo - 4 <= m <= hi + 4:
                candidates.append(m)

    result, used = [], set()
    for pp in sorted(prev):
        best = min((c for c in candidates if c not in used),
                   key=lambda c: abs(c - pp), default=candidates[0] if candidates else pp)
        result.append(best)
        used.add(best)
    return sorted(set(result))


# ═══════════════════════════════════════════════════════════
# EXPANSIÓN DE PROGRESIÓN A COMPASES
# ═══════════════════════════════════════════════════════════

def expand_progression(prog_pattern, n_bars, beats_per_bar):
    """
    Expande un patrón de progresión para cubrir n_bars compases.
    Devuelve lista de (roman_figure, start_beat, duration_beats).
    """
    total_beats = n_bars * beats_per_bar
    pattern_beats = sum(d for _, d in prog_pattern)

    # Repetir patrón para cubrir total_beats
    reps = max(1, math.ceil(total_beats / pattern_beats))
    full_pattern = (list(prog_pattern) * reps)

    # Truncar al número exacto de beats
    result = []
    beat = 0.0
    for figure, dur in full_pattern:
        if beat >= total_beats:
            break
        actual_dur = min(dur, total_beats - beat)
        result.append((figure, beat, actual_dur))
        beat += actual_dur

    return result


def score_progression(progression_expanded, melody, key_pc, mode):
    """
    Evalúa la compatibilidad global de una progresión expandida con la melodía.
    Devuelve score en [0, 1].
    """
    scores = []
    for figure, start_beat, dur in progression_expanded:
        mel_pcs = melody_pcs_in_window(melody, start_beat, start_beat + dur)
        root = figure_to_root_pc(figure, key_pc, mode)
        quality = figure_to_quality(figure, mode)
        s = score_chord_vs_melody(mel_pcs, root, quality)
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════
# SELECCIÓN ÓPTIMA DE PROGRESIÓN DENTRO DE UNA ESTRATEGIA
# ═══════════════════════════════════════════════════════════

def pick_best_progressions(strategy, mode, melody, key_pc, n_bars,
                            beats_per_bar, n_candidates=3, min_score=0.25):
    """
    Para una estrategia dada, evalúa todos los patrones disponibles y
    devuelve los n_candidates mejores (score >= min_score), ordenados.
    """
    lib = PROGRESSION_LIBRARY.get(strategy, {})
    patterns = lib.get(mode, []) + lib.get('major', [] if mode != 'major' else [])
    # Sin duplicados
    unique_patterns = []
    seen = set()
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    scored = []
    for pat in unique_patterns:
        expanded = expand_progression(pat, n_bars, beats_per_bar)
        s = score_progression(expanded, melody, key_pc, mode)
        scored.append((s, pat, expanded))

    scored.sort(key=lambda x: -x[0])
    return [(s, pat, exp) for s, pat, exp in scored if s >= min_score][:n_candidates]


# ═══════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL MIDI DE SALIDA
# ═══════════════════════════════════════════════════════════

ACC_STYLE_PATTERNS = {
    'block':       lambda beats: [(0, beats)],
    'arpeggio':    lambda beats: [(i * beats / 4, beats / 4) for i in range(4)],
    'alberti':     lambda beats: [(0, beats/4), (beats/2, beats/4), (beats/4, beats/4), (beats/2, beats/4)],
    'waltz':       lambda beats: [(0, 1.0), (1.0, 0.5), (2.0, 0.5)] if beats >= 3 else [(0, beats)],
    'jazz_voicing':lambda beats: [(0, beats * 0.9)],
}


def build_rehar_midi(melody, prog_expanded, key_obj, tempo_bpm, time_sig,
                     n_bars, complexity=0.5, acc_style='block',
                     output_path='rehar_out.mid'):
    """
    Escribe el MIDI con melodía original + acompañamiento reharmonizado.
    Usa build_midi de midi_dna_unified si está disponible, si no construye
    un MIDI mínimo con mido directamente.
    """
    beats_per_bar, beat_unit = time_sig
    tpb = 480

    if DNA_OK and key_obj is not None:
        # ── Construir progresión en formato midi_dna_unified ──────────────
        prog_dna = [(fig, dur) for fig, _start, dur in prog_expanded]
        # Normalizar a duración total exacta
        total_beats = n_bars * beats_per_bar

        # Generar acompañamiento
        try:
            from midi_dna_unified import EmotionalController, FormGenerator, GrooveMap
            _neutral = [0.5] * n_bars
            ec = EmotionalController(
                tension_curve=_neutral,
                arousal_curve=_neutral,
                valence_curve=_neutral,
                stability_curve=_neutral,
                activity_curve=_neutral,
                emotional_arc_label='neutral',
                n_bars=n_bars,
            )

            fg = FormGenerator(
                form_string='AABA',
                section_map=[],
                phrase_lengths=[4],
                cadence_positions=[],
                n_bars_out=n_bars,
            )

            acc = generate_accompaniment(
                prog_dna, key_obj, n_bars, ec, fg, beats_per_bar,
                groove_map=None, force_style=acc_style,
                harmony_complexity=complexity,
            )
            bass = generate_bass(prog_dna, key_obj, n_bars, beats_per_bar, groove_map=None)

            build_midi(
                melody_notes=melody,
                acc_notes=acc,
                bass_notes=bass,
                cp_notes=[],
                target_key=key_obj,
                tempo_bpm=tempo_bpm,
                time_sig=time_sig,
                n_bars=n_bars,
                form_gen=fg,
                output_path=output_path,
                percussion_notes=[],
            )
            return True
        except Exception as e:
            print(f"    [AVISO] build_midi falló ({e}), usando escritura directa.")

    # ── Fallback: escritura MIDI directa con mido ─────────────────────────
    _write_midi_direct(melody, prog_expanded, key_obj, tempo_bpm, time_sig,
                       n_bars, complexity, acc_style, output_path, tpb)
    return True


def _write_midi_direct(melody, prog_expanded, key_obj, tempo_bpm, time_sig,
                       n_bars, complexity, acc_style, output_path, tpb=480):
    """
    Escribe MIDI con mido directamente:
    track 0: melodía, track 1: acordes, track 2: bajo.
    """
    beats_per_bar, _ = time_sig
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(1, int(round(beats * tpb)))

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    def make_header_track():
        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        trk.append(mido.MetaMessage('time_signature',
                                    numerator=beats_per_bar, denominator=4,
                                    clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                    time=0))
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    def notes_to_track(notes_list, ch, program, track_name):
        events = []
        for offset, pitch, dur, vel in notes_list:
            p = max(0, min(127, int(pitch)))
            v = max(1, min(127, int(vel)))
            t_on  = to_ticks(offset)
            t_off = to_ticks(offset + max(0.05, dur))
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
            trk.append(mido.Message(msg_type, channel=ch, note=p, velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    # Construir notas de acompañamiento y bajo
    acc_notes = []
    bass_notes = []
    key_pc_val = tonic_pc(key_obj) if key_obj else 0
    mode_val = key_obj.mode if key_obj else 'major'
    pat_fn = ACC_STYLE_PATTERNS.get(acc_style, ACC_STYLE_PATTERNS['block'])
    prev_pitches = None

    for figure, start_beat, dur in prog_expanded:
        chord_pitches = build_chord_pitches(
            figure, key_pc_val, mode_val,
            prev_pitches=prev_pitches,
            complexity=complexity,
            register_lo=48, register_hi=76
        )
        prev_pitches = chord_pitches

        # Bajo: raíz en octava baja
        root_bass = (chord_pitches[0] % 12) + 36
        while root_bass < 28: root_bass += 12
        while root_bass > 52: root_bass -= 12
        bass_notes.append((start_beat, root_bass, dur * 0.9, 72))

        # Acompañamiento
        sub_pattern = pat_fn(dur)
        for rel_offset, sub_dur in sub_pattern:
            abs_off = start_beat + rel_offset
            if abs_off + sub_dur > start_beat + dur:
                sub_dur = max(0.05, start_beat + dur - abs_off)
            for p in chord_pitches:
                acc_notes.append((abs_off, p, sub_dur * 0.85, 65))

    mid.tracks.append(make_header_track())
    mid.tracks.append(notes_to_track(melody,    ch=0, program=0,  track_name='Melody'))
    mid.tracks.append(notes_to_track(acc_notes, ch=1, program=0,  track_name='Chords'))
    mid.tracks.append(notes_to_track(bass_notes,ch=2, program=32, track_name='Bass'))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    mid.save(output_path)


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

ALL_STRATEGIES = [
    'diatonic', 'tritone', 'secondary', 'modal_interchange',
    'chromatic_med', 'coltrane', 'neapolitan', 'pedal',
    'minor_modal', 'baroque', 'impressionist',
]


def reharmonize(midi_path: str,
                strategies: list = None,
                key_override: str = None,
                n_bars: int = None,
                n_candidates: int = 3,
                acc_style: str = 'auto',
                tempo_override: float = None,
                complexity: float = 0.5,
                score_threshold: float = 0.25,
                out_dir: str = None,
                report: bool = False,
                verbose: bool = False):
    """
    Pipeline principal de reharmonización.

    Args:
        midi_path       : Ruta al MIDI con la melodía.
        strategies      : Lista de estrategias a aplicar.
        key_override    : Tonalidad forzada (e.g. 'D minor').
        n_bars          : Compases. None = auto desde melodía.
        n_candidates    : Candidatos por estrategia.
        acc_style       : Estilo acompañamiento ('block','arpeggio','alberti','waltz','jazz_voicing','auto').
        tempo_override  : BPM forzado.
        complexity      : Complejidad armónica 0-1.
        score_threshold : Score mínimo para guardar un resultado.
        out_dir         : Carpeta de salida.
        report          : Guardar JSON de análisis.
        verbose         : Imprimir detalles.
    """
    if strategies is None or 'all' in strategies:
        strategies = ALL_STRATEGIES

    midi_path = str(midi_path)
    stem = Path(midi_path).stem
    if out_dir is None:
        out_dir = str(Path(midi_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*64}")
    print(f"  REHARMONIZER  ·  {os.path.basename(midi_path)}")
    print(f"{'═'*64}")

    # Cargar melodía
    try:
        melody, tempo_bpm, key_obj, tpb, time_sig = load_melody(midi_path, verbose=verbose)
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}

    if tempo_override:
        tempo_bpm = float(tempo_override)
    if key_override:
        key_obj = parse_key_string(key_override)
        print(f"  Tonalidad forzada: {key_obj}")

    beats_per_bar = time_sig[0]
    total_beats = melody_total_beats(melody)

    if n_bars is None:
        n_bars = max(4, math.ceil(total_beats / beats_per_bar))

    if acc_style == 'auto':
        # Heurístico: elegir por tempo
        if tempo_bpm < 80:
            acc_style = 'block'
        elif tempo_bpm < 120:
            acc_style = 'arpeggio'
        else:
            acc_style = 'alberti'

    key_pc_val = tonic_pc(key_obj)
    mode_val = key_obj.mode if key_obj else 'major'

    print(f"  Tonalidad : {key_obj}  (tónica PC={key_pc_val}, modo={mode_val})")
    print(f"  Tempo     : {tempo_bpm} BPM")
    print(f"  Compases  : {n_bars}  (beats_per_bar={beats_per_bar})")
    print(f"  Notas mel.: {len(melody)}")
    print(f"  Estrateg. : {strategies}")
    print(f"  Acomp.    : {acc_style}  complejidad={complexity}")

    report_data = {
        'source': midi_path,
        'key': str(key_obj),
        'mode': mode_val,
        'tempo_bpm': tempo_bpm,
        'n_bars': n_bars,
        'melody_notes': len(melody),
        'strategies_run': strategies,
        'results': [],
    }

    total_saved = 0

    for strategy in strategies:
        print(f"\n  ── Estrategia: {strategy.upper()} ──")

        candidates = pick_best_progressions(
            strategy=strategy,
            mode=mode_val,
            melody=melody,
            key_pc=key_pc_val,
            n_bars=n_bars,
            beats_per_bar=beats_per_bar,
            n_candidates=n_candidates,
            min_score=score_threshold,
        )

        if not candidates:
            print(f"    Sin candidatos con score >= {score_threshold:.2f}")
            report_data['results'].append({
                'strategy': strategy, 'candidates': 0,
                'reason': f'score < {score_threshold}'
            })
            continue

        print(f"    {len(candidates)} candidato(s) seleccionado(s).")

        for ci, (score, pattern, expanded) in enumerate(candidates, 1):
            out_name = f"{stem}.rehar_{strategy}_{ci:02d}.mid"
            out_path = os.path.join(out_dir, out_name)

            # Resumen de la progresión en texto
            prog_str = ' | '.join(
                f"{fig}({dur:.1f}b)" for fig, _s, dur in expanded[:8]
            )
            if len(expanded) > 8:
                prog_str += '…'

            if verbose:
                print(f"    [{ci}] score={score:.3f}  prog=[{prog_str}]")
            else:
                # Mostrar patrón (no expandido)
                pat_str = ' | '.join(f"{f}({d}b)" for f, d in pattern)
                print(f"    [{ci}] score={score:.3f}  {pat_str}")

            try:
                build_rehar_midi(
                    melody=melody,
                    prog_expanded=expanded,
                    key_obj=key_obj,
                    tempo_bpm=tempo_bpm,
                    time_sig=time_sig,
                    n_bars=n_bars,
                    complexity=complexity,
                    acc_style=acc_style,
                    output_path=out_path,
                )
                print(f"      → {out_name}")
                total_saved += 1

                result_info = {
                    'strategy': strategy,
                    'candidate': ci,
                    'score': round(score, 4),
                    'pattern': [[f, d] for f, d in pattern],
                    'path': out_path,
                }
                report_data['results'].append(result_info)

            except Exception as e:
                print(f"      [ERROR] {e}")
                if verbose:
                    traceback.print_exc()

    # Resumen
    print(f"\n{'─'*64}")
    print(f"  Total archivos generados: {total_saved}")

    by_strategy = {}
    for r in report_data['results']:
        s = r.get('strategy', '?')
        by_strategy[s] = by_strategy.get(s, 0) + (1 if 'path' in r else 0)
    for s, c in by_strategy.items():
        if c > 0:
            print(f"    {s:20s}: {c} archivo(s)")

    # Guardar reporte
    if report:
        report_path = os.path.join(out_dir, f"{stem}.rehar_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Reporte: {report_path}")

    print(f"{'═'*64}\n")
    return report_data


# ═══════════════════════════════════════════════════════════
# UTILIDAD: LISTAR ESTRATEGIAS Y SUS PROGRESIONES
# ═══════════════════════════════════════════════════════════

def print_strategy_catalog():
    """Imprime el catálogo completo de estrategias y progresiones disponibles."""
    print("\n╔══ CATÁLOGO DE ESTRATEGIAS ═══════════════════════════════════════╗")
    descriptions = {
        'diatonic':          'Progresiones diatónicas dentro de la tonalidad',
        'tritone':           'Sustitución de tritono (bII7 en lugar de V7)',
        'secondary':         'Dominantes secundarios (V/ii, V/IV, V/V…)',
        'modal_interchange': 'Préstamos del modo paralelo (bVII, bIII, iv…)',
        'chromatic_med':     'Mediante cromática (III, bVI, bII…)',
        'coltrane':          'Ciclos de Coltrane (terceras mayores equidistantes)',
        'neapolitan':        'Acorde napolitano (bII) y variantes',
        'pedal':             'Pedal de tónica o dominante',
        'minor_modal':       'Modos menores: dórico, frigio, lidio, mixolidio',
        'baroque':           'Progresiones funcionales de estilo barroco',
        'impressionist':     'Paralelismo cromático y tonos enteros',
    }
    for strat, desc in descriptions.items():
        lib = PROGRESSION_LIBRARY.get(strat, {})
        n_major = len(lib.get('major', []))
        n_minor = len(lib.get('minor', []))
        print(f"  {strat:20s}  M:{n_major:2d} m:{n_minor:2d}  {desc}")

    print("╚══════════════════════════════════════════════════════════════════╝\n")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='REHARMONIZER — Reharmonización guiada de melodías MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estrategias disponibles:
  diatonic, tritone, secondary, modal_interchange, chromatic_med,
  coltrane, neapolitan, pedal, minor_modal, baroque, impressionist, all

Ejemplos:
  python reharmonizer.py melodia.mid
  python reharmonizer.py melodia.mid --strategy tritone coltrane --candidates 5
  python reharmonizer.py melodia.mid --strategy all --out-dir rehar/ --report
  python reharmonizer.py melodia.mid --key "D minor" --acc-style arpeggio
  python reharmonizer.py melodia.mid --strategy modal_interchange --complexity 0.8
  python reharmonizer.py --catalog
        """
    )
    p.add_argument('input', nargs='?', help='Archivo MIDI de entrada')
    p.add_argument('--catalog', action='store_true',
                   help='Mostrar catálogo de estrategias y salir')
    p.add_argument('--strategy', nargs='+', default=['diatonic', 'tritone', 'secondary', 'modal_interchange'],
                   metavar='S',
                   help='Estrategias a aplicar (default: diatonic tritone secondary modal_interchange)')
    p.add_argument('--key', default=None, metavar='KEY',
                   help='Tonalidad forzada, e.g. "D minor" o "F# major"')
    p.add_argument('--bars', type=int, default=None,
                   help='Compases de salida (default: auto)')
    p.add_argument('--candidates', type=int, default=3,
                   help='Candidatos por estrategia (default: 3)')
    p.add_argument('--acc-style', default='auto',
                   choices=['auto', 'block', 'arpeggio', 'alberti', 'waltz', 'jazz_voicing'],
                   help='Estilo de acompañamiento (default: auto)')
    p.add_argument('--tempo', type=float, default=None,
                   help='BPM forzado (default: detectado del MIDI)')
    p.add_argument('--complexity', type=float, default=0.5,
                   help='Complejidad armónica 0-1 (default: 0.5)')
    p.add_argument('--score-threshold', type=float, default=0.25,
                   help='Score mínimo de compatibilidad melódica (default: 0.25)')
    p.add_argument('--out-dir', default=None,
                   help='Carpeta de salida (default: misma carpeta que el MIDI)')
    p.add_argument('--report', action='store_true',
                   help='Guardar reporte JSON de análisis')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.catalog:
        print_strategy_catalog()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    import glob
    midi_files = glob.glob(args.input) or [args.input]
    midi_files = [f for f in midi_files if f.endswith(('.mid', '.midi'))]

    if not midi_files:
        print(f"[ERROR] No se encontraron archivos MIDI: {args.input}")
        sys.exit(1)

    for midi_path in midi_files:
        reharmonize(
            midi_path=midi_path,
            strategies=args.strategy,
            key_override=args.key,
            n_bars=args.bars,
            n_candidates=args.candidates,
            acc_style=args.acc_style,
            tempo_override=args.tempo,
            complexity=args.complexity,
            score_threshold=args.score_threshold,
            out_dir=args.out_dir,
            report=args.report,
            verbose=args.verbose,
        )


if __name__ == '__main__':
    main()
