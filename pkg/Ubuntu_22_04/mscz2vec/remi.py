#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         REMI  v1.0  (PyTorch)                                ║
║       Pop Music Transformer — generación y continuación de piano pop        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Implementación del paper:                                                   ║
║    "Pop Music Transformer: Beat-Based Modeling and Generation of             ║
║     Expressive Pop Piano Compositions"  (Huang & Yang, ACM MM 2020)         ║
║  Original TF1: https://github.com/YatingMusic/remi                          ║
║                                                                              ║
║  REPRESENTACIÓN REMI:                                                        ║
║    Cada MIDI se convierte a una secuencia de tokens con estructura métrica:  ║
║    Bar → Position → [Chord] → Tempo Class → Tempo Value →                  ║
║           Note Velocity → Note On → Note Duration → …                      ║
║    Los tokens se numeran en un vocabulario JSON (legible, sin pickle).       ║
║                                                                              ║
║  ARQUITECTURA:                                                                ║
║    Transformer-XL: 12 capas, d_model=512, 8 cabezas, d_ff=2048             ║
║    Memoria recurrente de 512 tokens entre segmentos (Transformer-XL)        ║
║    Dos modos: sin acordes (REMI-tempo) y con acordes (REMI-tempo-chord)     ║
║                                                                              ║
║  LIMITACIONES CONOCIDAS:                                                     ║
║    • Piano solo / track 0 (multi-track no soportado)                        ║
║    • Compás 4/4 implícito (otros compases generarán advertencia)            ║
║    • Tempo fuera de 30-210 BPM → se fuerza al extremo del rango             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COMANDOS                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  convert  — MIDI → tokens REMI (construye vocabulario si no existe)         ║
║    python remi.py convert entrada.mid                                        ║
║    python remi.py convert entrada.mid --chord --output tokens.json          ║
║    python remi.py convert corpus/ --chord --vocab vocab_chord.json          ║
║                                                                              ║
║  inspect  — Muestra los tokens REMI de un MIDI (diagnóstico)                ║
║    python remi.py inspect entrada.mid                                        ║
║    python remi.py inspect entrada.mid --chord --max-tokens 200              ║
║                                                                              ║
║  train    — Entrena el Transformer-XL sobre un corpus de MIDIs              ║
║    python remi.py train corpus/ --model-dir remi_model/ --epochs 200        ║
║    python remi.py train corpus/ --model-dir remi_model/ --chord             ║
║    python remi.py train corpus/ --model-dir remi_model/ \                   ║
║        --epochs 300 --batch-size 2 --lr 2e-4 --patience 30                 ║
║    python remi.py train corpus/ --model-dir remi_model/ --resume            ║
║                                                                              ║
║  generate — Genera piano pop desde cero                                      ║
║    python remi.py generate --model-dir remi_model/                          ║
║    python remi.py generate --model-dir remi_model/ \                        ║
║        --bars 32 --temperature 1.2 --topk 5 --output from_scratch.mid      ║
║                                                                              ║
║  continue — Continúa un MIDI de prompt                                       ║
║    python remi.py continue prompt.mid --model-dir remi_model/               ║
║    python remi.py continue prompt.mid --model-dir remi_model/ \             ║
║        --bars 16 --temperature 1.0 --topk 5 --output continuado.mid        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OPCIONES COMUNES:                                                           ║
║    --chord          Activar tokens de acorde (Chord_X:Y)                    ║
║    --vocab FILE     Ruta del vocabulario JSON [default: vocab[_chord].json] ║
║    --model-dir DIR  Directorio del modelo [default: remi_model/]            ║
║    --output FILE    Fichero de salida                                        ║
║    --bars N         Compases a generar [default: 16]                        ║
║    --temperature F  Temperatura de muestreo [default: 1.2]                  ║
║    --topk N         Top-K para muestreo [default: 5]                        ║
║    --seed N         Semilla aleatoria [default: 42]                         ║
║    --verbose        Detalle de la ejecución                                  ║
║                                                                              ║
║  OPCIONES DE TRAIN:                                                          ║
║    --epochs N       Épocas máximas [default: 200]                           ║
║    --batch-size N   Tamaño de batch [default: 4]                            ║
║    --lr F           Learning rate [default: 2e-4]                           ║
║    --x-len N        Longitud de secuencia [default: 512]                    ║
║    --mem-len N      Longitud de memoria XL [default: 512]                   ║
║    --patience N     Early stopping [default: 30]                            ║
║    --resume         Reanudar desde checkpoint                                ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, torch                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import glob
import math
import time
import random
import argparse
import copy
from pathlib import Path

import numpy as np

try:
    import mido
except ImportError:
    print("ERROR: mido no encontrado.  pip install mido")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES REMI
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_RESOLUTION      = 480          # ticks por negra (salida MIDI)
DEFAULT_FRACTION        = 16           # subdivisiones por compás
DEFAULT_VELOCITY_BINS   = np.linspace(0, 128, 32 + 1, dtype=int)
DEFAULT_DURATION_BINS   = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord maps (de chord_recognition.py original)
CHORD_MAPS       = {'maj': [0, 4],     'min': [0, 3],     'dim': [0, 3, 6],
                    'aug': [0, 4, 8],  'dom': [0, 4, 7, 10]}
CHORD_INSIDERS   = {'maj': [7],        'min': [7],         'dim': [9],
                    'aug': [],         'dom': []}
CHORD_OUTSIDERS1 = {'maj': [2, 5, 9],  'min': [2, 5, 8],  'dim': [2, 5, 10],
                    'aug': [2, 5, 9],  'dom': [2, 5, 9]}
CHORD_OUTSIDERS2 = {'maj': [1, 3, 6, 8, 10],  'min': [1, 4, 6, 9, 11],
                    'dim': [1, 4, 7, 8, 11],   'aug': [1, 3, 6, 7, 10],
                    'dom': [1, 3, 6, 8, 11]}


# ══════════════════════════════════════════════════════════════════════════════
#  ITEM  (almacenamiento intermedio unificado)
# ══════════════════════════════════════════════════════════════════════════════

class Item:
    __slots__ = ('name', 'start', 'end', 'velocity', 'pitch')

    def __init__(self, name, start, end=None, velocity=None, pitch=None):
        self.name     = name
        self.start    = start
        self.end      = end
        self.velocity = velocity
        self.pitch    = pitch

    def __repr__(self):
        return (f"Item(name={self.name}, start={self.start}, end={self.end}, "
                f"velocity={self.velocity}, pitch={self.pitch})")


class Event:
    __slots__ = ('name', 'time', 'value', 'text')

    def __init__(self, name, time=None, value=None, text=None):
        self.name  = name
        self.time  = time
        self.value = value
        self.text  = text

    def __repr__(self):
        return (f"Event(name={self.name}, time={self.time}, "
                f"value={self.value}, text={self.text})")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _read_midi(file_path: str):
    """Lee un MIDI con mido y devuelve notas + cambios de tempo del track 0."""
    mid = mido.MidiFile(file_path)
    tpb = mid.ticks_per_beat  # ticks por negra del fichero fuente

    # ── Notas ─────────────────────────────────────────────────────────────────
    note_items = []
    pending    = {}  # (channel, pitch) → (abs_tick, velocity)
    abs_tick   = 0

    # Usamos el primer track con notas (track 0 en type-0, primer instrumento
    # en type-1). Advertimos si hay más de un track con notas.
    tracks_with_notes = [
        t for t in mid.tracks
        if any(m.type in ('note_on', 'note_off') for m in t)
    ]
    if len(tracks_with_notes) > 1:
        print(f"  AVISO: {len(tracks_with_notes)} tracks con notas — "
              "solo se procesa el primero (piano solo).")
    track = tracks_with_notes[0] if tracks_with_notes else mid.tracks[0]

    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            pending[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in pending:
                start_tick, vel = pending.pop(key)
                # Reescalar a DEFAULT_RESOLUTION si el fichero tiene otra TPB
                scale = DEFAULT_RESOLUTION / tpb
                note_items.append(Item(
                    name     = 'Note',
                    start    = int(start_tick * scale),
                    end      = int(abs_tick   * scale),
                    velocity = vel,
                    pitch    = msg.note,
                ))

    note_items.sort(key=lambda x: (x.start, x.pitch))

    # ── Tempo ─────────────────────────────────────────────────────────────────
    # Buscamos en todos los tracks (el tempo suele estar en el track 0 o meta)
    raw_tempos = []
    for t in mid.tracks:
        abs_t = 0
        for msg in t:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                bpm = round(60_000_000 / msg.tempo)
                scale = DEFAULT_RESOLUTION / tpb
                raw_tempos.append(Item(
                    name  = 'Tempo',
                    start = int(abs_t * scale),
                    pitch = bpm,
                ))
    raw_tempos.sort(key=lambda x: x.start)

    if not raw_tempos:
        raw_tempos = [Item('Tempo', 0, pitch=120)]

    # Expandir a cada beat para que item2event tenga tempo en cada pulso
    if note_items:
        max_tick = note_items[-1].end
    else:
        max_tick = raw_tempos[-1].start
    existing = {it.start: it.pitch for it in raw_tempos}
    beats    = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    last_bpm = raw_tempos[0].pitch
    tempo_items = []
    for tick in beats:
        bpm = existing.get(int(tick), last_bpm)
        last_bpm = bpm
        tempo_items.append(Item('Tempo', int(tick), pitch=bpm))

    return note_items, tempo_items


# ══════════════════════════════════════════════════════════════════════════════
#  CUANTIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _quantize_items(items, ticks: int = DEFAULT_RESOLUTION // 4):
    """Ajusta onset/offset al grid más cercano de `ticks` pasos."""
    if not items:
        return items
    max_t  = max(it.start for it in items)
    grids  = np.arange(0, max_t + ticks + 1, ticks, dtype=int)
    for it in items:
        idx       = int(np.argmin(np.abs(grids - it.start)))
        shift     = int(grids[idx]) - it.start
        it.start += shift
        if it.end is not None:
            it.end += shift
    return items


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE ACORDES  (chord_recognition.py integrado)
# ══════════════════════════════════════════════════════════════════════════════

def _notes_to_chroma(notes, t_start: int, t_end: int) -> np.ndarray:
    """Piano-roll cromático (12,) para el segmento [t_start, t_end)."""
    chroma = np.zeros(12, dtype=int)
    for n in notes:
        if n.start < t_end and n.end > t_start:
            chroma[n.pitch % 12] = 1
    return chroma


def _score_chroma(chroma: np.ndarray):
    """Devuelve (root_pc, quality, score) con la mejor puntuación."""
    best_root, best_qual, best_score = 0, 'maj', -999
    candidates = np.where(chroma)[0]
    for root in candidates:
        rolled = np.roll(chroma, -root)
        seq    = list(np.where(rolled)[0])
        if 3 not in seq and 4 not in seq:
            continue
        if 3 in seq and 4 in seq:
            continue
        if 3 in seq:
            qual = 'dim' if 6 in seq else 'min'
        else:
            if 8 in seq:
                qual = 'aug'
            elif 7 in seq and 10 in seq:
                qual = 'dom'
            else:
                qual = 'maj'
        maps  = CHORD_MAPS[qual]
        extra = [n for n in seq if n not in maps]
        score = 0
        for n in extra:
            if   n in CHORD_INSIDERS[qual]:   score += 1
            elif n in CHORD_OUTSIDERS1[qual]: score -= 1
            elif n in CHORD_OUTSIDERS2[qual]: score -= 2
        if score > best_score:
            best_score, best_root, best_qual = score, root, qual
    return best_root, best_qual, best_score


def extract_chords(note_items: list) -> list:
    """Segmentación greedy de acordes, igual que chord_recognition.py original."""
    if not note_items:
        return []
    max_tick = max(n.end for n in note_items)
    tpb      = DEFAULT_RESOLUTION  # ya reescalado

    candidates = {}
    for interval in [4, 2]:
        for start in range(0, max_tick, tpb):
            end    = min(start + tpb * interval, max_tick)
            chroma = _notes_to_chroma(note_items, start, end)
            if np.sum(chroma) == 0:
                root, qual = 0, 'None'
            else:
                root, qual, _ = _score_chroma(chroma)
            candidates.setdefault(start, {})[end] = (root, qual)

    # Greedy
    chords     = []
    start_tick = 0
    while start_tick < max_tick:
        opts = candidates.get(start_tick, {})
        if not opts:
            start_tick += tpb
            continue
        end_tick, (root, qual) = max(opts.items(), key=lambda kv: kv[0])
        if qual == 'None':
            start_tick = end_tick
            continue
        label = f"{PITCH_CLASSES[root]}:{qual}"
        chords.append(Item('Chord', start_tick, end=end_tick, pitch=label))
        start_tick = end_tick

    return chords


# ══════════════════════════════════════════════════════════════════════════════
#  AGRUPACIÓN POR COMPÁS  (4/4 implícito)
# ══════════════════════════════════════════════════════════════════════════════

def _group_items(items: list, max_time: int,
                 ticks_per_bar: int = DEFAULT_RESOLUTION * 4) -> list:
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups    = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [it for it in items if db1 <= it.start < db2]
        groups.append([int(db1)] + insiders + [int(db2)])
    return groups


# ══════════════════════════════════════════════════════════════════════════════
#  ITEMS → EVENTOS REMI
# ══════════════════════════════════════════════════════════════════════════════

def item2event(groups: list) -> list:
    events     = []
    n_downbeat = 0
    for group in groups:
        inner = group[1:-1]
        if not any(it.name == 'Note' for it in inner):
            continue
        bar_st, bar_et = group[0], group[-1]
        n_downbeat += 1
        events.append(Event('Bar', value=None, text=str(n_downbeat)))

        for it in inner:
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            pos   = int(np.argmin(np.abs(flags - it.start)))
            events.append(Event('Position',
                                time=it.start,
                                value=f"{pos + 1}/{DEFAULT_FRACTION}",
                                text=str(it.start)))
            if it.name == 'Note':
                vel_idx = int(np.searchsorted(DEFAULT_VELOCITY_BINS,
                                              it.velocity, side='right')) - 1
                events.append(Event('Note Velocity',
                                    time=it.start, value=vel_idx,
                                    text=f"{it.velocity}/{DEFAULT_VELOCITY_BINS[vel_idx]}"))
                events.append(Event('Note On',
                                    time=it.start, value=it.pitch,
                                    text=str(it.pitch)))
                dur     = it.end - it.start
                dur_idx = int(np.argmin(np.abs(DEFAULT_DURATION_BINS - dur)))
                events.append(Event('Note Duration',
                                    time=it.start, value=dur_idx,
                                    text=f"{dur}/{DEFAULT_DURATION_BINS[dur_idx]}"))
            elif it.name == 'Chord':
                events.append(Event('Chord',
                                    time=it.start, value=it.pitch,
                                    text=str(it.pitch)))
            elif it.name == 'Tempo':
                bpm = it.pitch
                if bpm in DEFAULT_TEMPO_INTERVALS[0]:
                    cls, val = 'slow', bpm - DEFAULT_TEMPO_INTERVALS[0].start
                elif bpm in DEFAULT_TEMPO_INTERVALS[1]:
                    cls, val = 'mid',  bpm - DEFAULT_TEMPO_INTERVALS[1].start
                elif bpm in DEFAULT_TEMPO_INTERVALS[2]:
                    cls, val = 'fast', bpm - DEFAULT_TEMPO_INTERVALS[2].start
                elif bpm < DEFAULT_TEMPO_INTERVALS[0].start:
                    cls, val = 'slow', 0
                else:
                    cls, val = 'fast', 59
                events.append(Event('Tempo Class', time=it.start, value=cls))
                events.append(Event('Tempo Value', time=it.start, value=val))
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE MIDI → TOKENS
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_events(midi_path: str, use_chord: bool = False,
                   verbose: bool = False) -> list:
    """Convierte un MIDI a lista de Event REMI."""
    note_items, tempo_items = _read_midi(midi_path)
    if not note_items:
        raise ValueError(f"No se encontraron notas en {midi_path}")
    note_items = _quantize_items(note_items)
    max_time   = note_items[-1].end
    items      = list(tempo_items) + list(note_items)
    if use_chord:
        chord_items = extract_chords(note_items)
        items       = chord_items + items
        if verbose:
            print(f"    acordes detectados: {len(chord_items)}")
    groups = _group_items(items, max_time)
    return item2event(groups)


def midi_to_words(midi_path: str, event2word: dict,
                  use_chord: bool = False, verbose: bool = False) -> list:
    """MIDI → secuencia de IDs de token."""
    events = midi_to_events(midi_path, use_chord=use_chord, verbose=verbose)
    words  = []
    for ev in events:
        key = f"{ev.name}_{ev.value}"
        if key in event2word:
            words.append(event2word[key])
        else:
            if ev.name == 'Note Velocity':
                words.append(event2word.get('Note Velocity_21', 0))
            elif verbose:
                print(f"    OOV: {key}")
    return words


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE VOCABULARIO
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab(use_chord: bool = False) -> tuple[dict, dict]:
    """
    Construye el vocabulario REMI completo (sin necesidad de corpus).
    Devuelve (event2word, word2event).
    """
    vocab = []

    # Bar
    vocab.append('Bar_None')

    # Position
    for i in range(1, DEFAULT_FRACTION + 1):
        vocab.append(f'Position_{i}/{DEFAULT_FRACTION}')

    # Note Velocity (32 bins)
    for i in range(len(DEFAULT_VELOCITY_BINS) - 1):
        vocab.append(f'Note Velocity_{i}')

    # Note On (pitch 0-127)
    for p in range(128):
        vocab.append(f'Note On_{p}')

    # Note Duration (bins de 60 a 3840 ticks)
    for i in range(len(DEFAULT_DURATION_BINS)):
        vocab.append(f'Note Duration_{i}')

    # Tempo Class + Tempo Value
    for cls in ('slow', 'mid', 'fast'):
        vocab.append(f'Tempo Class_{cls}')
    for v in range(60):
        vocab.append(f'Tempo Value_{v}')

    # Chord (opcional)
    if use_chord:
        for pc in PITCH_CLASSES:
            for q in ('maj', 'min', 'dim', 'aug', 'dom'):
                vocab.append(f'Chord_{pc}:{q}')

    event2word = {e: i for i, e in enumerate(vocab)}
    word2event = {i: e for i, e in enumerate(vocab)}
    return event2word, word2event


def save_vocab(event2word: dict, path: str):
    with open(path, 'w') as f:
        json.dump(event2word, f, indent=2, ensure_ascii=False)
    print(f"  → Vocabulario guardado: {path}  ({len(event2word)} tokens)")


def load_vocab(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        event2word = json.load(f)
    word2event = {int(v): k for k, v in event2word.items()}
    return event2word, word2event


def _default_vocab_path(use_chord: bool) -> str:
    return 'vocab_chord.json' if use_chord else 'vocab.json'


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE MIDI  (words → MIDI)
# ══════════════════════════════════════════════════════════════════════════════

def write_midi(words: list, word2event: dict,
               output_path: str, prompt_path: str | None = None):
    """Reconstruye un fichero MIDI a partir de una secuencia de tokens."""
    tpb          = DEFAULT_RESOLUTION
    ticks_per_bar = tpb * 4

    temp_notes  = []
    temp_chords = []
    temp_tempos = []

    events = [word2event[int(w)].split('_', 1) for w in words]  # [(name, value), …]

    i = 0
    while i < len(events):
        name, val = events[i]
        if name == 'Bar':
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif name == 'Position':
            pos = int(val.split('/')[0]) - 1
            # Note
            if i + 3 < len(events):
                n1, v1 = events[i + 1]
                n2, v2 = events[i + 2]
                n3, v3 = events[i + 3]
                if n1 == 'Note Velocity' and n2 == 'Note On' and n3 == 'Note Duration':
                    velocity = int(DEFAULT_VELOCITY_BINS[int(v1)])
                    pitch    = int(v2)
                    duration = int(DEFAULT_DURATION_BINS[int(v3)])
                    temp_notes.append([pos, velocity, pitch, duration])
            # Chord
            if i + 1 < len(events):
                cn, cv = events[i + 1]
                if cn == 'Chord':
                    temp_chords.append([pos, cv])
            # Tempo
            if i + 2 < len(events):
                tn1, tv1 = events[i + 1]
                tn2, tv2 = events[i + 2]
                if tn1 == 'Tempo Class' and tn2 == 'Tempo Value':
                    if tv1 == 'slow':
                        bpm = DEFAULT_TEMPO_INTERVALS[0].start + int(tv2)
                    elif tv1 == 'mid':
                        bpm = DEFAULT_TEMPO_INTERVALS[1].start + int(tv2)
                    else:
                        bpm = DEFAULT_TEMPO_INTERVALS[2].start + int(tv2)
                    temp_tempos.append([pos, bpm])
        i += 1

    # Resolver ticks
    def _resolve(items_list):
        out      = []
        cur_bar  = 0
        for it in items_list:
            if it == 'Bar':
                cur_bar += 1
            else:
                pos  = it[0]
                rest = it[1:]
                bar_st = cur_bar * ticks_per_bar
                bar_et = (cur_bar + 1) * ticks_per_bar
                flags  = np.linspace(bar_st, bar_et, DEFAULT_FRACTION,
                                     endpoint=False, dtype=int)
                tick   = int(flags[pos])
                out.append([tick] + rest)
        return out

    notes_r  = _resolve([[it[0]] + it[1:] if it != 'Bar' else it for it in temp_notes])
    chords_r = _resolve([[it[0]] + it[1:] if it != 'Bar' else it for it in temp_chords])
    tempos_r = _resolve([[it[0]] + it[1:] if it != 'Bar' else it for it in temp_tempos])

    # Construir objetos mido
    mid = mido.MidiFile(ticks_per_beat=tpb)

    # Track de tempo
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    prev_tick = 0
    for tick, bpm in sorted(tempos_r, key=lambda x: x[0]):
        us = int(60_000_000 / max(bpm, 1))
        tempo_track.append(
            mido.MetaMessage('set_tempo', tempo=us, time=tick - prev_tick))
        prev_tick = tick
    if not tempos_r:
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    # Track de notas
    note_track = mido.MidiTrack()
    mid.tracks.append(note_track)

    prompt_offset = 0
    if prompt_path:
        # Añadir las notas del prompt primero
        prompt_mid = mido.MidiFile(prompt_path)
        pm_tpb     = prompt_mid.ticks_per_beat
        scale      = tpb / pm_tpb
        for t in prompt_mid.tracks:
            abs_t = 0
            for msg in t:
                abs_t += msg.time
                if msg.type in ('note_on', 'note_off'):
                    note_track.append(msg.copy(time=int(abs_t * scale)))
        # Offset = duración del prompt (4 compases, como en el original)
        prompt_offset = tpb * 4 * 4

    # Convertir notas absolutas a delta
    note_msgs = []
    for tick, vel, pitch, dur in notes_r:
        abs_on  = tick + prompt_offset
        abs_off = abs_on + dur
        note_msgs.append((abs_on,  'note_on',  pitch, vel))
        note_msgs.append((abs_off, 'note_off', pitch, 0))
    # Añadir markers de acorde
    chord_msgs = []
    for tick, label in chords_r:
        chord_msgs.append((tick + prompt_offset, 'marker', label))

    all_msgs = sorted(note_msgs + chord_msgs, key=lambda x: x[0])
    prev_t   = 0
    for entry in all_msgs:
        abs_t = entry[0]
        delta = abs_t - prev_t
        prev_t = abs_t
        if entry[1] == 'marker':
            note_track.append(mido.MetaMessage('marker', text=entry[2], time=delta))
        elif entry[1] == 'note_on':
            note_track.append(mido.Message('note_on',  note=entry[2],
                                           velocity=entry[3], time=delta))
        else:
            note_track.append(mido.Message('note_off', note=entry[2],
                                           velocity=0, time=delta))
    note_track.append(mido.MetaMessage('end_of_track', time=0))
    tempo_track.append(mido.MetaMessage('end_of_track', time=0))

    mid.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA: TRANSFORMER-XL (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

def _build_transformer_xl(n_token: int, n_layer: int = 12, d_model: int = 512,
                           n_head: int = 8, d_ff: int = 2048,
                           dropout: float = 0.1, mem_len: int = 512):
    """
    Transformer-XL decoder-only con memoria recurrente entre segmentos.
    Retorna una instancia nn.Module con firma:
        forward(x, mems=None) → (logits, new_mems)
    donde:
        x         : (T, B)  tokens de entrada
        logits    : (T, B, n_token)
        new_mems  : lista de L tensores (mem_len, B, d_model)
    """
    import torch
    import torch.nn as nn

    d_head = d_model // n_head

    class _SinPosEmb(nn.Module):
        def forward(self, length: int, device):
            pos  = torch.arange(length - 1, -1, -1.0, device=device)
            half = d_model // 2
            freq = torch.exp(
                -math.log(10000) * torch.arange(half, device=device) / half)
            args = pos[:, None] * freq[None]
            emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return emb  # (length, d_model)

    class _MultiHeadRelAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
            self.r_w  = nn.Parameter(torch.zeros(n_head, d_head))
            self.r_r  = nn.Parameter(torch.zeros(n_head, d_head))
            self.r_net = nn.Linear(d_model, d_model, bias=False)
            self.out  = nn.Linear(d_model, d_model, bias=False)
            self.drop = nn.Dropout(dropout)
            self.scale = d_head ** -0.5

        def forward(self, x, r, mem=None):
            # x: (T, B, D)   r: (T+mem, D)
            T, B, _ = x.shape
            cat      = torch.cat([mem, x], dim=0) if mem is not None else x
            qkv      = self.qkv(cat)          # (T+M, B, 3D)
            q, k, v  = qkv.chunk(3, dim=-1)
            q        = q[-T:]                 # solo query sobre x
            q  = q.view(T,  B, n_head, d_head)
            k  = k.view(-1, B, n_head, d_head)
            v  = v.view(-1, B, n_head, d_head)
            r_emb  = self.r_net(r).view(-1, n_head, d_head)  # (T+M, H, d)

            # Content-based attention
            ac = torch.einsum('ibhd,jbhd->ijbh', q + self.r_w, k)
            # Position-based attention (simplified relative)
            bd = torch.einsum('ibhd,jhd->ijbh',  q + self.r_r, r_emb)
            # Causal + memory mask
            attn = (ac + bd) * self.scale
            M    = k.shape[0]
            mask = torch.triu(torch.ones(T, M, device=x.device), diagonal=M - T + 1).bool()
            attn.masked_fill_(mask[:, :, None, None], float('-inf'))
            attn = torch.softmax(attn, dim=1)
            attn = self.drop(attn)
            out  = torch.einsum('ijbh,jbhd->ibhd', attn, v).reshape(T, B, d_model)
            return self.out(out)

    class _FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x):
            return self.net(x)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn  = _MultiHeadRelAttn()
            self.ffn   = _FFN()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x, r, mem=None):
            h = self.norm1(x + self.drop(self.attn(x, r, mem)))
            h = self.norm2(h + self.drop(self.ffn(h)))
            return h

    class TransformerXL(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed  = nn.Embedding(n_token, d_model)
            self.drop   = nn.Dropout(dropout)
            self.layers = nn.ModuleList([_Layer() for _ in range(n_layer)])
            self.pos_emb = _SinPosEmb()
            self.head   = nn.Linear(d_model, n_token, bias=False)
            self.n_layer = n_layer
            self.mem_len = mem_len
            self.d_model = d_model
            # Tie input/output embeddings
            self.head.weight = self.embed.weight

        def init_mems(self, batch_size: int, device):
            return [torch.zeros(mem_len, batch_size, d_model, device=device)
                    for _ in range(self.n_layer)]

        def forward(self, x, mems=None):
            # x: (T, B)
            T, B   = x.shape
            device = x.device
            h      = self.drop(self.embed(x))  # (T, B, D)

            if mems is None:
                mems = self.init_mems(B, device)

            M   = mems[0].shape[0] if mems else 0
            r   = self.pos_emb(T + M, device)   # (T+M, D)

            new_mems = []
            for layer, mem in zip(self.layers, mems):
                h = layer(h, r, mem)
                # Actualizar memoria: detach para no propagar gradientes a través
                new_mem = torch.cat([mem, h], dim=0).detach()[-mem_len:]
                new_mems.append(new_mem)

            logits = self.head(self.drop(h))  # (T, B, n_token)
            return logits, new_mems

    return TransformerXL()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _collect_midis(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    return sorted(glob.glob(str(p / '**' / '*.mid'), recursive=True) +
                  glob.glob(str(p / '**' / '*.midi'), recursive=True))


class REMIDataset:
    """
    Carga un corpus de MIDIs, los convierte a tokens y los segmenta en
    ventanas de x_len para el entrenamiento.
    """
    def __init__(self, midi_paths: list[str], event2word: dict,
                 x_len: int = 512, use_chord: bool = False,
                 verbose: bool = False):
        import torch
        self.x_len    = x_len
        self.segments = []   # lista de tensores int64 de longitud x_len+1

        ok = err = 0
        for path in midi_paths:
            try:
                words = midi_to_words(path, event2word,
                                      use_chord=use_chord, verbose=verbose)
                if len(words) < x_len + 1:
                    continue
                for i in range(0, len(words) - x_len, x_len):
                    seg = words[i: i + x_len + 1]
                    self.segments.append(torch.tensor(seg, dtype=torch.long))
                ok += 1
            except Exception as exc:
                if verbose:
                    print(f"    ERROR {path}: {exc}")
                err += 1

        print(f"  Corpus: {ok} ficheros OK, {err} errores, "
              f"{len(self.segments)} segmentos de {x_len} tokens")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        return seg[:-1], seg[1:]   # x, y  (teacher forcing)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(s: float) -> str:
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


class Trainer:
    CHECKPOINT = 'checkpoint.pt'
    BEST       = 'best_model.pt'
    CONFIG     = 'model_config.json'
    HISTORY    = 'history.json'

    def __init__(self, model, optimizer, model_dir: Path, patience: int = 30):
        self.model      = model
        self.optimizer  = optimizer
        self.model_dir  = model_dir
        self.patience   = patience
        self.best_loss  = float('inf')
        self.no_improve = 0
        self.start_ep   = 0
        self.history    = {'train': [], 'val': []}
        self._resume    = False

    def save(self, epoch: int, val_loss: float, is_best: bool):
        import torch
        state = {
            'epoch':      epoch,
            'model':      self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'best_loss':  self.best_loss,
            'no_improve': self.no_improve,
            'history':    self.history,
        }
        torch.save(state, self.model_dir / self.CHECKPOINT)
        if is_best:
            torch.save(state, self.model_dir / self.BEST)
        with open(self.model_dir / self.HISTORY, 'w') as f:
            json.dump(self.history, f, indent=2)

    def resume(self):
        import torch
        path = self.model_dir / self.CHECKPOINT
        if not path.exists():
            print("[train] Entrenando desde cero.")
            return
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_loss  = state['best_loss']
        self.no_improve = state['no_improve']
        self.history    = state['history']
        self.start_ep   = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_ep}  "
              f"(mejor val={self.best_loss:.4f})")

    def _run_epoch(self, loader, train: bool, epoch: int, n_epochs: int) -> float:
        import torch, torch.nn.functional as F
        self.model.train(train)
        total = 0.0
        n     = 0
        phase = 'train' if train else 'val  '
        ctx   = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for x, y in loader:
                device = next(self.model.parameters()).device
                x, y   = x.T.to(device), y.T.to(device)  # (T, B)
                mems   = self.model.init_mems(x.shape[1], device)

                logits, _ = self.model(x, mems)           # (T, B, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1))

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 0.5)
                    self.optimizer.step()

                total += loss.item()
                n     += 1
                print(f"\r  [{phase}] ep {epoch+1}/{n_epochs}  "
                      f"batch {n}  loss={total/n:.4f}   ",
                      end='', flush=True)

        print(' ' * 80, end='\r')
        return total / max(n, 1)

    def train(self, train_loader, val_loader, n_epochs: int):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        print(f"\n{'═'*64}")
        print(f"  REMI — Entrenamiento")
        print(f"  Épocas     : {n_epochs}   Early stopping: {self.patience}")
        print(f"  Dispositivo: {device}")
        print(f"  Modelo dir : {self.model_dir}")
        print(f"{'═'*64}\n")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self._resume:
            self.resume()

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6)

        epoch_times = []
        t_start     = time.time()

        for epoch in range(self.start_ep, n_epochs):
            eta_str = ''
            if epoch_times:
                avg     = sum(epoch_times) / len(epoch_times)
                eta_str = f"  ETA {_fmt_time(avg * (n_epochs - epoch))}"
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  Época {epoch+1:>4}/{n_epochs}  lr={lr:.2e}{eta_str}")

            t0      = time.time()
            tr_loss = self._run_epoch(train_loader, True,  epoch, n_epochs)
            vl_loss = self._run_epoch(val_loader,   False, epoch, n_epochs)
            sched.step()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            if len(epoch_times) > 5:
                epoch_times.pop(0)

            self.history['train'].append(tr_loss)
            self.history['val'].append(vl_loss)

            is_best = vl_loss < self.best_loss
            if is_best:
                self.best_loss  = vl_loss
                self.no_improve = 0
            else:
                self.no_improve += 1

            self.save(epoch, vl_loss, is_best)

            best_mark = ' ◀ mejor' if is_best else ''
            stop_str  = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                         if self.no_improve > 0 else '')
            print(f"         train={tr_loss:.4f}  val={vl_loss:.4f}  "
                  f"{_fmt_time(elapsed)}/época{best_mark}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        total = time.time() - t_start
        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(total)}")
        print(f"  Mejor val_loss : {self.best_loss:.4f}")
        print(f"  Modelos en     : {self.model_dir}")
        print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN  (temperatura + top-k, memoria XL)
# ══════════════════════════════════════════════════════════════════════════════

def _temperature_sample(logits_np: np.ndarray,
                         temperature: float, topk: int) -> int:
    logits_np = logits_np / max(temperature, 1e-6)
    logits_np -= logits_np.max()
    probs = np.exp(logits_np)
    probs /= probs.sum()
    if topk == 1:
        return int(np.argmax(probs))
    idx    = np.argsort(probs)[::-1][:topk]
    subset = probs[idx]
    subset /= subset.sum()
    return int(np.random.choice(idx, p=subset))


def _load_model_for_inference(model_dir: Path, device: str):
    """Carga modelo + config + vocab."""
    import torch

    cfg_path   = model_dir / Trainer.CONFIG
    best_path  = model_dir / Trainer.BEST
    ckpt_path  = model_dir / Trainer.CHECKPOINT

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No se encontró {cfg_path}. ¿Has ejecutado train?")
    model_path = best_path if best_path.exists() else ckpt_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró ningún checkpoint en {model_dir}. "
            "¿Has ejecutado train?")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = _build_transformer_xl(
        n_token  = cfg['n_token'],
        n_layer  = cfg.get('n_layer',  12),
        d_model  = cfg.get('d_model',  512),
        n_head   = cfg.get('n_head',   8),
        d_ff     = cfg.get('d_ff',     2048),
        dropout  = 0.0,           # sin dropout en inferencia
        mem_len  = cfg.get('mem_len', 512),
    )
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    model.to(device)
    return model, cfg


def generate_tokens(model, event2word: dict, word2event: dict,
                    n_target_bar: int = 16,
                    temperature: float = 1.2,
                    topk: int = 5,
                    use_chord: bool = False,
                    prompt_events: list | None = None,
                    device: str = 'cpu',
                    verbose: bool = False) -> list[int]:
    """
    Genera una secuencia de tokens REMI.

    Si `prompt_events` se proporciona, se usa como contexto inicial y se
    genera la continuación. En caso contrario se genera desde cero.
    """
    import torch

    mem_len = model.mem_len

    # ── Inicializar secuencia ─────────────────────────────────────────────────
    if prompt_events is not None:
        words = [event2word[f"{e.name}_{e.value}"]
                 for e in prompt_events
                 if f"{e.name}_{e.value}" in event2word]
        words.append(event2word['Bar_None'])
    else:
        words = [event2word['Bar_None']]
        # Arranque aleatorio: posición + tempo (+ acorde si procede)
        tc_keys = [v for k, v in event2word.items() if k.startswith('Tempo Class')]
        tv_keys = [v for k, v in event2word.items() if k.startswith('Tempo Value')]
        if use_chord:
            ch_keys = [v for k, v in event2word.items()
                       if k.startswith('Chord') and ':' in k]
            words += [event2word['Position_1/16'],
                      random.choice(ch_keys)]
        words += [event2word['Position_1/16'],
                  random.choice(tc_keys),
                  random.choice(tv_keys)]

    # ── Loop autoregresivo ────────────────────────────────────────────────────
    original_len      = len(words)
    current_bar       = 0
    initial_pass      = True
    mems              = model.init_mems(1, device)

    with torch.no_grad():
        while current_bar < n_target_bar:
            if initial_pass:
                x      = torch.tensor(words, dtype=torch.long,
                                      device=device).unsqueeze(1)  # (T,1)
                initial_pass = False
            else:
                x = torch.tensor([words[-1]], dtype=torch.long,
                                  device=device).unsqueeze(1)       # (1,1)

            logits, mems = model(x, mems)
            logit_np     = logits[-1, 0].cpu().numpy()
            word         = _temperature_sample(logit_np, temperature, topk)
            words.append(word)

            if word2event[word] == 'Bar_None':
                current_bar += 1
                if verbose:
                    print(f"\r  Generando compás {current_bar}/{n_target_bar}…",
                          end='', flush=True)

    if verbose:
        print()

    return words[original_len:]   # solo los tokens nuevos


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_convert(args):
    print("═" * 65)
    print("  REMI — CONVERT")
    print("═" * 65)

    vocab_path = args.vocab or _default_vocab_path(args.chord)
    if Path(vocab_path).exists():
        print(f"  Vocabulario : {vocab_path}  (existente)")
        event2word, _ = load_vocab(vocab_path)
    else:
        print(f"  Vocabulario : {vocab_path}  (nuevo)")
        event2word, _ = build_vocab(use_chord=args.chord)
        save_vocab(event2word, vocab_path)

    midis = _collect_midis(args.input)
    print(f"  Ficheros    : {len(midis)}")
    print(f"  Acordes     : {'sí' if args.chord else 'no'}")

    results = {}
    for path in midis:
        try:
            words = midi_to_words(path, event2word,
                                  use_chord=args.chord,
                                  verbose=args.verbose)
            results[path] = words
            print(f"  ✓ {path}  →  {len(words)} tokens")
        except Exception as exc:
            print(f"  ✗ {path}  →  {exc}")

    if args.output and results:
        out = {str(p): w for p, w in results.items()}
        with open(args.output, 'w') as f:
            json.dump(out, f)
        print(f"\n  → Tokens guardados: {args.output}")

    print("\n" + "═" * 65)
    total = sum(len(w) for w in results.values())
    print(f"  {len(results)} ficheros convertidos  |  {total} tokens totales")
    print("═" * 65)


def cmd_inspect(args):
    print("═" * 65)
    print("  REMI — INSPECT")
    print("═" * 65)
    print(f"  MIDI   : {args.input}")
    print(f"  Acordes: {'sí' if args.chord else 'no'}")

    events = midi_to_events(args.input, use_chord=args.chord, verbose=args.verbose)
    shown  = events[:args.max_tokens]

    print(f"\n  {len(events)} eventos totales (mostrando {len(shown)}):\n")
    for i, ev in enumerate(shown):
        print(f"  {i:>5}  {ev.name:<20} {str(ev.value):<20}"
              f"  t={ev.time}  [{ev.text}]")
    if len(events) > args.max_tokens:
        print(f"  … (+{len(events) - args.max_tokens} eventos)")

    # Estadísticas
    names = [ev.name for ev in events]
    print(f"\n  Compases      : {names.count('Bar')}")
    print(f"  Notas         : {names.count('Note On')}")
    if args.chord:
        print(f"  Acordes       : {names.count('Chord')}")
    print(f"  Cambios tempo : {names.count('Tempo Class')}")

    print("\n" + "═" * 65)


def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader, random_split

    print("═" * 65)
    print("  REMI — TRAIN")
    print("═" * 65)
    print(f"  Corpus     : {args.corpus}")
    print(f"  Modelo dir : {args.model_dir}")
    print(f"  Acordes    : {'sí' if args.chord else 'no'}")
    print(f"  Épocas     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  x_len      : {args.x_len}")
    print(f"  mem_len    : {args.mem_len}")

    model_dir  = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.vocab or _default_vocab_path(args.chord)

    # Vocabulario
    if Path(vocab_path).exists():
        print(f"\n  Vocabulario existente: {vocab_path}")
        event2word, word2event = load_vocab(vocab_path)
    else:
        event2word, word2event = build_vocab(use_chord=args.chord)
        save_vocab(event2word, vocab_path)
    n_token = len(event2word)
    print(f"  Vocabulario : {n_token} tokens")

    # Dataset
    midis = _collect_midis(args.corpus)
    print(f"\n[1/3] Convirtiendo {len(midis)} MIDIs…")
    random.Random(args.seed).shuffle(midis)
    dataset = REMIDataset(midis, event2word,
                          x_len=args.x_len,
                          use_chord=args.chord,
                          verbose=args.verbose)
    if len(dataset) == 0:
        print("ERROR: corpus vacío o MIDIs demasiado cortos.")
        sys.exit(1)

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Modelo
    print("\n[2/3] Construyendo Transformer-XL…")
    model = _build_transformer_xl(
        n_token = n_token,
        n_layer = 12,
        d_model = 512,
        n_head  = 8,
        d_ff    = 2048,
        dropout = 0.1,
        mem_len = args.mem_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Guardar config
    cfg = {
        'n_token':   n_token,
        'n_layer':   12,
        'd_model':   512,
        'n_head':    8,
        'd_ff':      2048,
        'mem_len':   args.mem_len,
        'x_len':     args.x_len,
        'use_chord': args.chord,
        'vocab':     vocab_path,
    }
    with open(model_dir / Trainer.CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)

    # Copiar vocabulario al model_dir para autocontenido
    import shutil
    shutil.copy(vocab_path, model_dir / Path(vocab_path).name)

    trainer            = Trainer(model, optimizer, model_dir, patience=args.patience)
    trainer._resume    = args.resume

    print("\n[3/3] Entrenando…")
    trainer.train(train_loader, val_loader, args.epochs)


def cmd_generate(args):
    import torch

    print("═" * 65)
    print("  REMI — GENERATE  (from scratch)")
    print("═" * 65)
    print(f"  Modelo dir  : {args.model_dir}")
    print(f"  Compases    : {args.bars}")
    print(f"  Temperatura : {args.temperature}")
    print(f"  Top-k       : {args.topk}")
    print(f"  Semilla     : {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[1/3] Cargando modelo desde {model_dir}…")
    model, cfg = _load_model_for_inference(model_dir, device)

    use_chord  = cfg.get('use_chord', False)
    vocab_file = model_dir / Path(cfg['vocab']).name
    if not vocab_file.exists():
        vocab_file = Path(cfg['vocab'])
    event2word, word2event = load_vocab(str(vocab_file))
    print(f"  Vocabulario : {len(event2word)} tokens  |  acordes: {use_chord}")

    print(f"\n[2/3] Generando {args.bars} compases…")
    words = generate_tokens(
        model        = model,
        event2word   = event2word,
        word2event   = word2event,
        n_target_bar = args.bars,
        temperature  = args.temperature,
        topk         = args.topk,
        use_chord    = use_chord,
        device       = device,
        verbose      = args.verbose,
    )
    print(f"  Tokens generados: {len(words)}")

    output = args.output or 'remi_from_scratch.mid'
    print(f"\n[3/3] Escribiendo MIDI → {output}…")
    write_midi(words, word2event, output)

    print("\n" + "═" * 65)
    print(f"  Salida: {output}")
    print("═" * 65)


def cmd_continue(args):
    import torch

    print("═" * 65)
    print("  REMI — CONTINUE")
    print("═" * 65)
    print(f"  Prompt      : {args.prompt}")
    print(f"  Modelo dir  : {args.model_dir}")
    print(f"  Compases    : {args.bars}")
    print(f"  Temperatura : {args.temperature}")
    print(f"  Top-k       : {args.topk}")
    print(f"  Semilla     : {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[1/4] Cargando modelo desde {model_dir}…")
    model, cfg = _load_model_for_inference(model_dir, device)

    use_chord  = cfg.get('use_chord', False)
    vocab_file = model_dir / Path(cfg['vocab']).name
    if not vocab_file.exists():
        vocab_file = Path(cfg['vocab'])
    event2word, word2event = load_vocab(str(vocab_file))
    print(f"  Vocabulario : {len(event2word)} tokens  |  acordes: {use_chord}")

    print(f"\n[2/4] Convirtiendo prompt a eventos REMI…")
    prompt_events = midi_to_events(args.prompt,
                                   use_chord=use_chord,
                                   verbose=args.verbose)
    print(f"  {len(prompt_events)} eventos de contexto")

    print(f"\n[3/4] Generando continuación ({args.bars} compases)…")
    words = generate_tokens(
        model          = model,
        event2word     = event2word,
        word2event     = word2event,
        n_target_bar   = args.bars,
        temperature    = args.temperature,
        topk           = args.topk,
        use_chord      = use_chord,
        prompt_events  = prompt_events,
        device         = device,
        verbose        = args.verbose,
    )
    print(f"  Tokens generados: {len(words)}")

    stem   = Path(args.prompt).stem
    output = args.output or f"{stem}_continued.mid"
    print(f"\n[4/4] Escribiendo MIDI → {output}…")
    write_midi(words, word2event, output, prompt_path=args.prompt)

    print("\n" + "═" * 65)
    print(f"  Salida: {output}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="REMI v1.0 — Pop Music Transformer (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True, metavar='COMANDO')

    # ── convert ───────────────────────────────────────────────────────────────
    p = sub.add_parser('convert', help='MIDI(s) → tokens REMI')
    p.add_argument('input',           help='MIDI o directorio de MIDIs')
    p.add_argument('--chord',         action='store_true',
                   help='Incluir tokens de acorde')
    p.add_argument('--vocab',         default=None,
                   help='Ruta del fichero de vocabulario JSON')
    p.add_argument('--output',        default=None,
                   help='Guardar tokens en JSON')
    p.add_argument('--verbose',       action='store_true')

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Mostrar tokens REMI de un MIDI')
    p.add_argument('input',           help='MIDI a inspeccionar')
    p.add_argument('--chord',         action='store_true')
    p.add_argument('--max-tokens',    type=int, default=100,
                   dest='max_tokens', help='Máximo de tokens a mostrar [100]')
    p.add_argument('--verbose',       action='store_true')

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrenar Transformer-XL sobre corpus MIDI')
    p.add_argument('corpus',          help='Directorio (o fichero) de MIDIs')
    p.add_argument('--model-dir',     default='remi_model', dest='model_dir',
                   help='Directorio de salida del modelo [remi_model/]')
    p.add_argument('--chord',         action='store_true')
    p.add_argument('--vocab',         default=None)
    p.add_argument('--epochs',        type=int,   default=200)
    p.add_argument('--batch-size',    type=int,   default=4,    dest='batch_size')
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--x-len',         type=int,   default=512,  dest='x_len')
    p.add_argument('--mem-len',       type=int,   default=512,  dest='mem_len')
    p.add_argument('--patience',      type=int,   default=30)
    p.add_argument('--resume',        action='store_true')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--verbose',       action='store_true')

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser('generate', help='Generar piano pop desde cero')
    p.add_argument('--model-dir',     required=True, dest='model_dir')
    p.add_argument('--bars',          type=int,   default=16)
    p.add_argument('--temperature',   type=float, default=1.2)
    p.add_argument('--topk',          type=int,   default=5)
    p.add_argument('--output',        default=None)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--verbose',       action='store_true')

    # ── continue ──────────────────────────────────────────────────────────────
    p = sub.add_parser('continue', help='Continuar un MIDI de prompt')
    p.add_argument('prompt',          help='MIDI de prompt')
    p.add_argument('--model-dir',     required=True, dest='model_dir')
    p.add_argument('--bars',          type=int,   default=16)
    p.add_argument('--temperature',   type=float, default=1.2)
    p.add_argument('--topk',          type=int,   default=5)
    p.add_argument('--output',        default=None)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--verbose',       action='store_true')

    args = parser.parse_args()
    {
        'convert':  cmd_convert,
        'inspect':  cmd_inspect,
        'train':    cmd_train,
        'generate': cmd_generate,
        'continue': cmd_continue,
    }[args.command](args)


if __name__ == '__main__':
    main()
