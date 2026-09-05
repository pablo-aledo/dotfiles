#!/usr/bin/env python3
r"""
================================================================================
                      REMI DIFFUSION  v2
   Masked/Uniform Diffusion LM aplicado a composicion de piano pop

  NOVEDADES RESPECTO A v1 (remi_diffusion.py):

  1. ARQUITECTURA ENCODER-DECODER (antes: un unico Transformer bidireccional)
     Un encoder pesado procesa la historia (compases ya generados, siempre
     limpios) con atencion BLOCK-CAUSAL: bidireccional dentro de cada
     compas, causal entre compases. Un decoder ligero hace cross-attention
     a esa memoria para denoisar el compas actual (canvas). Como en Gemma
     Diffusion: "heavy encoder... lightweight decoder".

  2. KV-CACHE REAL entre compases (antes: se reprocesaba toda la historia
     en CADA paso de denoising de CADA compas -- muy ineficiente). Ahora:
       - El encoder cachea claves/valores por capa; al anadir un compas
         nuevo solo se calculan SUS tokens, nunca se recalculan los
         compases anteriores (igual que el KV-cache autorregresivo, pero
         por bloques en vez de por token).
       - La memoria para cross-attention del decoder se recalcula UNA VEZ
         por compas (no en cada paso de denoising).
       - Ventana deslizante (--history-bars) con expulsion de los
         compases mas antiguos del cache cuando se supera el limite.

  3. UNIFORM-STATE DIFFUSION (UDLM), seleccionable con --noise-type:
       - 'mask'    (MDLM, como v1): el forward process reemplaza tokens por
         [MASK]; el sampler des-enmascara + remasking (ReMDM).
       - 'uniform' (UDLM, default en v2): el forward process reemplaza
         tokens por tokens ALEATORIOS del vocabulario (nunca hay [MASK]);
         el sampler mantiene un estado "comprometido/no comprometido" por
         posicion y puede "renoisar" (re-aleatorizar) posiciones ya
         comprometidas para corregir errores -- segun el articulo, esto da
         MEJOR CONTROLABILIDAD que MDLM, relevante para el guidance por
         acordes que es el objetivo central de esta herramienta.

    Referencia: Kuleshov Group, "How to Build a Diffusion Language Model"
    (2026) -- https://kuleshov-group.github.io/blog/blog/2026/
               how-to-build-a-diffusion-language-model/
    Secciones: "Architectures: Encoder, Decoder, and Encoder-Decoder",
    "Block Diffusion for Flexible-Length Generation", "Uniform State
    Diffusion".

  LO QUE v2 NO CAMBIA: mismo esquema de tokens REMI, mismo vocabulario
  (compatible con vocab_chord.json de remi.py y v1), mismo canvas de
  longitud fija por compas, misma guidance D-CFG por acordes, misma
  reparacion gramatical greedy tras el sampler.

  COMANDOS: convert, train, generate, continue, inspect  (identicos a v1)
  DEPENDENCIAS: mido, numpy, torch
================================================================================
  EJEMPLOS
================================================================================

  python remi_diffusion_v2.py convert corpus/ --vocab vocab_chord.json

  python remi_diffusion_v2.py train corpus/ --model-dir diff_v2/ \
      --noise-type uniform --epochs 300 --canvas-len 48

  python remi_diffusion_v2.py generate --model-dir diff_v2/ \
      --chords "C:maj G:maj A:min F:maj" --bars 16 --guidance-scale 3.0

  python remi_diffusion_v2.py continue prompt.mid --model-dir diff_v2/ \
      --chords "D:min G:dom C:maj" --bars 8
================================================================================

  OPCIONES NUEVAS EN v2 (respecto a v1):
    --noise-type {mask,uniform}   Tipo de proceso de difusion [uniform]
    --n-layer-enc N                Capas del encoder (pesado) [6]
    --n-layer-dec N                Capas del decoder (ligero) [2]
    (--n-layer de v1 ya no existe; se sustituye por los dos anteriores)

  OPCIONES COMUNES (heredadas de v1):
    --canvas-len N, --history-bars N, --d-model N, --n-head N,
    --cond-dropout F, --epochs N, --batch-size N, --lr F, --patience N,
    --resume, --small, --chords, --bars N, --steps N, --guidance-scale F,
    --remask-max F, --temperature F, --topk N,
    --repair-grammar/--no-repair-grammar, --seed N
"""

import sys
import os
import json
import glob
import math
import time
import random
import argparse
import textwrap
from pathlib import Path

import numpy as np

try:
    import mido
except ImportError:
    print("ERROR: mido no encontrado.  pip install mido")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES REMI  (idénticas a remi.py para compatibilidad de vocabulario)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_RESOLUTION      = 480
DEFAULT_FRACTION        = 16
DEFAULT_VELOCITY_BINS   = np.linspace(0, 128, 32 + 1, dtype=int)
DEFAULT_DURATION_BINS   = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORD_MAPS       = {'maj': [0, 4],     'min': [0, 3],     'dim': [0, 3, 6],
                    'aug': [0, 4, 8],  'dom': [0, 4, 7, 10]}
CHORD_INSIDERS   = {'maj': [7],        'min': [7],         'dim': [9],
                    'aug': [],         'dom': []}
CHORD_OUTSIDERS1 = {'maj': [2, 5, 9],  'min': [2, 5, 8],  'dim': [2, 5, 10],
                    'aug': [2, 5, 9],  'dom': [2, 5, 9]}
CHORD_OUTSIDERS2 = {'maj': [1, 3, 6, 8, 10],  'min': [1, 4, 6, 9, 11],
                    'dim': [1, 4, 7, 8, 11],   'aug': [1, 3, 6, 7, 10],
                    'dom': [1, 3, 6, 8, 11]}

# Tokens especiales de este módulo (no existen en remi.py; se añaden al final
# del vocabulario para no romper compatibilidad de índices existentes).
SPECIAL_TOKENS = ['PAD_None', 'MASK_None', 'NULL_COND_None']

CANVAS_LEN_DEFAULT   = 48
MAX_POS_BUFFER       = 20000  # buffer sinusoidal generoso; sin parámetros
                               # aprendidos, es válido para cualquier longitud
HISTORY_BARS_DEFAULT = 8


# ══════════════════════════════════════════════════════════════════════════════
#  ITEM / EVENT  (idéntico a remi.py)
# ══════════════════════════════════════════════════════════════════════════════

class Item:
    __slots__ = ('name', 'start', 'end', 'velocity', 'pitch')

    def __init__(self, name, start, end=None, velocity=None, pitch=None):
        self.name, self.start, self.end = name, start, end
        self.velocity, self.pitch = velocity, pitch

    def __repr__(self):
        return (f"Item(name={self.name}, start={self.start}, end={self.end}, "
                f"velocity={self.velocity}, pitch={self.pitch})")


class Event:
    __slots__ = ('name', 'time', 'value', 'text')

    def __init__(self, name, time=None, value=None, text=None):
        self.name, self.time, self.value, self.text = name, time, value, text

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, value={self.value})"


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE MIDI, CUANTIZACIÓN, ACORDES, AGRUPACIÓN  (portado de remi.py)
# ══════════════════════════════════════════════════════════════════════════════

def _read_midi(file_path: str):
    mid = mido.MidiFile(file_path)
    tpb = mid.ticks_per_beat

    note_items = []
    pending    = {}
    tracks_with_notes = [t for t in mid.tracks
                          if any(m.type in ('note_on', 'note_off') for m in t)]
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
                scale = DEFAULT_RESOLUTION / tpb
                note_items.append(Item('Note', int(start_tick * scale),
                                        end=int(abs_tick * scale),
                                        velocity=vel, pitch=msg.note))
    note_items.sort(key=lambda x: (x.start, x.pitch))

    raw_tempos = []
    for t in mid.tracks:
        abs_t = 0
        for msg in t:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                bpm   = round(60_000_000 / msg.tempo)
                scale = DEFAULT_RESOLUTION / tpb
                raw_tempos.append(Item('Tempo', int(abs_t * scale), pitch=bpm))
    raw_tempos.sort(key=lambda x: x.start)
    if not raw_tempos:
        raw_tempos = [Item('Tempo', 0, pitch=120)]

    max_tick = note_items[-1].end if note_items else raw_tempos[-1].start
    existing = {it.start: it.pitch for it in raw_tempos}
    beats    = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    last_bpm = raw_tempos[0].pitch
    tempo_items = []
    for tick in beats:
        bpm = existing.get(int(tick), last_bpm)
        last_bpm = bpm
        tempo_items.append(Item('Tempo', int(tick), pitch=bpm))

    return note_items, tempo_items


def _quantize_items(items, ticks: int = DEFAULT_RESOLUTION // 4):
    if not items:
        return items
    max_t = max(it.start for it in items)
    grids = np.arange(0, max_t + ticks + 1, ticks, dtype=int)
    for it in items:
        idx   = int(np.argmin(np.abs(grids - it.start)))
        shift = int(grids[idx]) - it.start
        it.start += shift
        if it.end is not None:
            it.end += shift
    return items


def _notes_to_chroma(notes, t_start, t_end):
    chroma = np.zeros(12, dtype=int)
    for n in notes:
        if n.start < t_end and n.end > t_start:
            chroma[n.pitch % 12] = 1
    return chroma


def _score_chroma(chroma):
    best_root, best_qual, best_score = 0, 'maj', -999
    for root in np.where(chroma)[0]:
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


def extract_chords(note_items):
    if not note_items:
        return []
    max_tick = max(n.end for n in note_items)
    tpb      = DEFAULT_RESOLUTION
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

    chords, start_tick = [], 0
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


def _group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [it for it in items if db1 <= it.start < db2]
        groups.append([int(db1)] + insiders + [int(db2)])
    return groups


def item2event(groups):
    events, n_downbeat = [], 0
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
            events.append(Event('Position', time=it.start,
                                 value=f"{pos + 1}/{DEFAULT_FRACTION}"))
            if it.name == 'Note':
                vel_idx = int(np.searchsorted(DEFAULT_VELOCITY_BINS,
                                               it.velocity, side='right')) - 1
                events.append(Event('Note Velocity', time=it.start, value=vel_idx))
                events.append(Event('Note On', time=it.start, value=it.pitch))
                dur     = it.end - it.start
                dur_idx = int(np.argmin(np.abs(DEFAULT_DURATION_BINS - dur)))
                events.append(Event('Note Duration', time=it.start, value=dur_idx))
            elif it.name == 'Chord':
                events.append(Event('Chord', time=it.start, value=it.pitch))
            elif it.name == 'Tempo':
                bpm = it.pitch
                if bpm in DEFAULT_TEMPO_INTERVALS[0]:
                    cls, val = 'slow', bpm - DEFAULT_TEMPO_INTERVALS[0].start
                elif bpm in DEFAULT_TEMPO_INTERVALS[1]:
                    cls, val = 'mid', bpm - DEFAULT_TEMPO_INTERVALS[1].start
                elif bpm in DEFAULT_TEMPO_INTERVALS[2]:
                    cls, val = 'fast', bpm - DEFAULT_TEMPO_INTERVALS[2].start
                elif bpm < DEFAULT_TEMPO_INTERVALS[0].start:
                    cls, val = 'slow', 0
                else:
                    cls, val = 'fast', 59
                events.append(Event('Tempo Class', time=it.start, value=cls))
                events.append(Event('Tempo Value', time=it.start, value=val))
    return events


def midi_to_events(midi_path, verbose=False):
    """Convierte un MIDI a eventos REMI. Acordes SIEMPRE activados (son la
    señal de guidance de este módulo)."""
    note_items, tempo_items = _read_midi(midi_path)
    if not note_items:
        raise ValueError(f"No se encontraron notas en {midi_path}")
    note_items  = _quantize_items(note_items)
    max_time    = note_items[-1].end
    chord_items = extract_chords(note_items)
    if verbose:
        print(f"    acordes detectados: {len(chord_items)}")
    items  = chord_items + list(tempo_items) + list(note_items)
    groups = _group_items(items, max_time)
    return item2event(groups)


def midi_to_words(midi_path, event2word, verbose=False):
    events = midi_to_events(midi_path, verbose=verbose)
    words  = []
    for ev in events:
        key = f"{ev.name}_{ev.value}"
        if key in event2word:
            words.append(event2word[key])
        elif ev.name == 'Note Velocity':
            words.append(event2word.get('Note Velocity_21', 0))
        elif verbose:
            print(f"    OOV: {key}")
    return words


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIO  (idéntico a remi.py --chord + 3 tokens especiales al final)
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab():
    vocab = ['Bar_None']
    for i in range(1, DEFAULT_FRACTION + 1):
        vocab.append(f'Position_{i}/{DEFAULT_FRACTION}')
    for i in range(len(DEFAULT_VELOCITY_BINS) - 1):
        vocab.append(f'Note Velocity_{i}')
    for p in range(128):
        vocab.append(f'Note On_{p}')
    for i in range(len(DEFAULT_DURATION_BINS)):
        vocab.append(f'Note Duration_{i}')
    for cls in ('slow', 'mid', 'fast'):
        vocab.append(f'Tempo Class_{cls}')
    for v in range(60):
        vocab.append(f'Tempo Value_{v}')
    for pc in PITCH_CLASSES:
        for q in ('maj', 'min', 'dim', 'aug', 'dom'):
            vocab.append(f'Chord_{pc}:{q}')
    vocab.extend(SPECIAL_TOKENS)
    event2word = {e: i for i, e in enumerate(vocab)}
    word2event = {i: e for i, e in enumerate(vocab)}
    return event2word, word2event


def save_vocab(event2word, path):
    with open(path, 'w') as f:
        json.dump(event2word, f, indent=2, ensure_ascii=False)
    print(f"  → Vocabulario guardado: {path}  ({len(event2word)} tokens)")


def load_vocab(path):
    """Carga un vocab.json/vocab_chord.json (de remi.py o de este módulo).
    Si le faltan los tokens especiales (viene de remi.py), los añade al
    final y re-guarda, preservando todos los índices existentes."""
    with open(path) as f:
        event2word = json.load(f)
    missing = [t for t in SPECIAL_TOKENS if t not in event2word]
    if missing:
        next_idx = max(event2word.values()) + 1
        for t in missing:
            event2word[t] = next_idx
            next_idx += 1
        save_vocab(event2word, path)
        print(f"  (vocabulario extendido con: {', '.join(missing)})")
    word2event = {int(v): k for k, v in event2word.items()}
    return event2word, word2event


def _sp(word2event, name):
    """Índice de un token especial dado su nombre corto, ej. 'PAD' → id."""
    return {v: k for k, v in word2event.items()}[f'{name}_None']


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUES POR COMPÁS  (canvas de longitud fija + token de condición)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Cada compás se representa como:
#     [COND]  seguido de  canvas_len tokens (Position/Velocity/NoteOn/
#             Duration/Chord), rellenados con PAD si sobra espacio.
#
#  COND es el token Chord_X:Y detectado para ese compás (el primero que
#  aparezca), o NULL_COND si el compás no tiene acorde reconocible.
#  Este token nunca se enmascara: es la condición de guidance (D-CFG).

def words_to_blocks(words, word2event, canvas_len, verbose=False):
    """Divide una secuencia de palabras REMI (con Bar_None de separador) en
    una lista de bloques.  Cada bloque: dict(cond=int, canvas=list[int])."""
    bar_id  = {v: k for k, v in word2event.items()}['Bar_None']
    pad_id  = _sp(word2event, 'PAD')
    blocks, current = [], []
    for w in words:
        if w == bar_id:
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(w)
    if current:
        blocks.append(current)

    out = []
    n_trunc = 0
    for bar_tokens in blocks:
        cond = None
        for w in bar_tokens:
            if word2event[w].startswith('Chord_'):
                cond = w
                break
        canvas = [w for w in bar_tokens if not word2event[w].startswith('Chord_')]
        if len(canvas) > canvas_len:
            n_trunc += 1
            canvas = canvas[:canvas_len]
        else:
            canvas = canvas + [pad_id] * (canvas_len - len(canvas))
        out.append({'cond': cond, 'canvas': canvas})
    if verbose and n_trunc:
        print(f"  AVISO: {n_trunc} compases truncados a canvas_len={canvas_len} "
              f"(sube --canvas-len si pasa a menudo)")
    return out


def blocks_to_words(blocks, word2event):
    """Reconstruye la secuencia REMI plana (con Bar_None) desde bloques."""
    bar_id = {v: k for k, v in word2event.items()}['Bar_None']
    pad_id = _sp(word2event, 'PAD')
    mask_id = _sp(word2event, 'MASK')
    words = []
    for blk in blocks:
        words.append(bar_id)
        for w in blk['canvas']:
            if w in (pad_id, mask_id):
                continue
            words.append(w)
    return words


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE MIDI  (idéntico a remi.py, opera sobre la secuencia plana)
# ══════════════════════════════════════════════════════════════════════════════

def write_midi(words, word2event, output_path, prompt_path=None):
    tpb, ticks_per_bar = DEFAULT_RESOLUTION, DEFAULT_RESOLUTION * 4
    temp_notes, temp_chords, temp_tempos = [], [], []
    events = [word2event[int(w)].split('_', 1) for w in words]

    i = 0
    while i < len(events):
        name, val = events[i]
        if name == 'Bar':
            temp_notes.append('Bar'); temp_chords.append('Bar'); temp_tempos.append('Bar')
        elif name == 'Position':
            pos = int(val.split('/')[0]) - 1
            if i + 3 < len(events):
                n1, v1 = events[i + 1]; n2, v2 = events[i + 2]; n3, v3 = events[i + 3]
                if n1 == 'Note Velocity' and n2 == 'Note On' and n3 == 'Note Duration':
                    velocity = int(DEFAULT_VELOCITY_BINS[int(v1)])
                    pitch    = int(v2)
                    duration = int(DEFAULT_DURATION_BINS[int(v3)])
                    temp_notes.append([pos, velocity, pitch, duration])
            if i + 1 < len(events):
                cn, cv = events[i + 1]
                if cn == 'Chord':
                    temp_chords.append([pos, cv])
            if i + 2 < len(events):
                tn1, tv1 = events[i + 1]; tn2, tv2 = events[i + 2]
                if tn1 == 'Tempo Class' and tn2 == 'Tempo Value':
                    base = {'slow': DEFAULT_TEMPO_INTERVALS[0].start,
                            'mid':  DEFAULT_TEMPO_INTERVALS[1].start,
                            'fast': DEFAULT_TEMPO_INTERVALS[2].start}[tv1]
                    temp_tempos.append([pos, base + int(tv2)])
        i += 1

    def _resolve(items_list):
        out, cur_bar = [], 0
        for it in items_list:
            if it == 'Bar':
                cur_bar += 1
            else:
                pos, rest = it[0], it[1:]
                bar_st, bar_et = cur_bar * ticks_per_bar, (cur_bar + 1) * ticks_per_bar
                flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION,
                                     endpoint=False, dtype=int)
                out.append([int(flags[pos])] + rest)
        return out

    notes_r  = _resolve(temp_notes)
    chords_r = _resolve(temp_chords)
    tempos_r = _resolve(temp_tempos)

    mid = mido.MidiFile(ticks_per_beat=tpb)
    tempo_track = mido.MidiTrack(); mid.tracks.append(tempo_track)
    prev_tick = 0
    for tick, bpm in sorted(tempos_r, key=lambda x: x[0]):
        us = int(60_000_000 / max(bpm, 1))
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=us, time=tick - prev_tick))
        prev_tick = tick
    if not tempos_r:
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    note_track = mido.MidiTrack(); mid.tracks.append(note_track)
    prompt_offset = 0
    if prompt_path:
        prompt_mid = mido.MidiFile(prompt_path)
        pm_tpb = prompt_mid.ticks_per_beat
        scale  = tpb / pm_tpb
        for t in prompt_mid.tracks:
            abs_t = 0
            for msg in t:
                abs_t += msg.time
                if msg.type in ('note_on', 'note_off'):
                    note_track.append(msg.copy(time=int(abs_t * scale)))
        prompt_offset = tpb * 4 * 4

    note_msgs = []
    for tick, vel, pitch, dur in notes_r:
        abs_on, abs_off = tick + prompt_offset, tick + prompt_offset + dur
        note_msgs.append((abs_on, 'note_on', pitch, vel))
        note_msgs.append((abs_off, 'note_off', pitch, 0))
    chord_msgs = [(tick + prompt_offset, 'marker', label) for tick, label in chords_r]

    all_msgs = sorted(note_msgs + chord_msgs, key=lambda x: x[0])
    prev_t = 0
    for entry in all_msgs:
        delta = entry[0] - prev_t
        prev_t = entry[0]
        if entry[1] == 'marker':
            note_track.append(mido.MetaMessage('marker', text=entry[2], time=delta))
        elif entry[1] == 'note_on':
            note_track.append(mido.Message('note_on', note=entry[2], velocity=entry[3], time=delta))
        else:
            note_track.append(mido.Message('note_off', note=entry[2], velocity=0, time=delta))
    note_track.append(mido.MetaMessage('end_of_track', time=0))
    tempo_track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(output_path)


def repair_canvas_grammar(canvas, word2event, pad_id):
    """
    Reconstruye una secuencia gramaticalmente válida a partir de un canvas
    generado que puede tener tokens en cualquier orden (la difusión enmascarada,
    a diferencia de la generación autorregresiva, no impone por construcción el
    orden Position→Velocity→NoteOn→Duration).

    Estrategia greedy: por cada token 'Position' (en orden de aparición), busca
    hacia delante -entre los tokens aún no usados- la ruta más cercana que
    complete o bien una NOTA (Note Velocity → Note On → Note Duration) o bien
    un TEMPO (Tempo Class → Tempo Value); adjunta la que se completa primero.
    Los tokens que sobran (Position sin núcleo emparejable, o restos sueltos
    de Velocity/On/Duration/TempoClass/TempoValue) se descartan.

    Devuelve la lista de ids reparada (sin PAD).
    """
    idxs  = [w for w in canvas if w != pad_id]
    names = [word2event[w].split('_', 1)[0] for w in idxs]
    n     = len(idxs)
    used  = [False] * n

    def find_seq(target_names, start):
        ptr, found = start, []
        for target in target_names:
            k = None
            for m in range(ptr, n):
                if not used[m] and names[m] == target:
                    k = m
                    break
            if k is None:
                return None
            found.append(k)
            ptr = k + 1
        return found

    out = []
    for i in range(n):
        if used[i] or names[i] != 'Position':
            continue
        used[i] = True
        note_path  = find_seq(['Note Velocity', 'Note On', 'Note Duration'], i + 1)
        tempo_path = find_seq(['Tempo Class', 'Tempo Value'], i + 1)
        candidates = []
        if note_path:
            candidates.append(note_path)
        if tempo_path:
            candidates.append(tempo_path)
        if not candidates:
            continue  # Position huérfano, sin núcleo que emparejar: se descarta
        path = min(candidates, key=lambda p: max(p))
        for k in path:
            used[k] = True
        out.append(idxs[i])
        out.extend(idxs[k] for k in path)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO: ENCODER-DECODER DE DIFUSIÓN POR BLOQUES, CON KV-CACHE
# ══════════════════════════════════════════════════════════════════════════════
#
#  ENCODER (pesado): procesa la historia (compases ya generados, siempre
#  limpios) con atención BLOCK-CAUSAL — bidireccional dentro de cada compás,
#  causal entre compases. Esto es lo que permite un KV-cache real: las claves/
#  valores de un compás ya procesado NUNCA cambian al añadir compases
#  posteriores, así que se computan una sola vez en toda la generación.
#
#  DECODER (ligero): procesa [COND, canvas_ruidoso] del compás actual con
#  auto-atención bidireccional (se recalcula en cada paso de denoising, ya
#  que el canvas cambia) + cross-attention a la memoria del encoder (los
#  estados finales de la historia — se recalcula una vez por compás, no en
#  cada paso de denoising, porque la memoria no cambia durante el denoising
#  de un compás).

def _build_model(n_token, d_model=512, n_layer_enc=6, n_layer_dec=2, n_head=8,
                  d_ff=2048, dropout=0.1, max_len=8192):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    d_head = d_model // n_head

    class SinPosEmb(nn.Module):
        def __init__(self):
            super().__init__()
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(-math.log(10000.0) *
                             torch.arange(0, d_model, 2).float() / d_model)
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe)

        def forward(self, offset, length):
            return self.pe[offset:offset + length]

    class MultiHeadAttnKV(nn.Module):
        """Atención multi-cabeza con soporte opcional de KV-cache incremental.

        - Sin cache (modo entrenamiento, secuencia completa de una vez):
          forward(x_q, x_kv, attn_mask=mask) con mask booleano (B|1,1,Tq,Tk),
          True = permitido.
        - Con cache (modo generación, procesa solo tokens NUEVOS de x_kv):
          forward(x_q, x_kv_new, cache=cache_dict) concatena las K/V nuevas
          a las cacheadas, sin necesidad de máscara (los tokens nuevos deben
          ver todo el pasado + a sí mismos, exactamente lo que da la
          concatenación).
        """
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = dropout

        def forward(self, x_q, x_kv, attn_mask=None, cache=None):
            B, Tq, _ = x_q.shape
            q = self.q_proj(x_q).view(B, Tq, n_head, d_head).transpose(1, 2)
            k_new = self.k_proj(x_kv).view(B, x_kv.size(1), n_head, d_head).transpose(1, 2)
            v_new = self.v_proj(x_kv).view(B, x_kv.size(1), n_head, d_head).transpose(1, 2)
            if cache is not None:
                if cache.get('k') is not None:
                    k = torch.cat([cache['k'], k_new], dim=2)
                    v = torch.cat([cache['v'], v_new], dim=2)
                else:
                    k, v = k_new, v_new
                cache['k'], cache['v'] = k, v
            else:
                k, v = k_new, v_new
            drop_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                  dropout_p=drop_p)
            out = out.transpose(1, 2).contiguous().view(B, Tq, d_model)
            return self.out_proj(out)

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d_ff, d_model))

        def forward(self, x):
            return self.net(x)

    class EncoderLayer(nn.Module):
        """Auto-atención block-causal (con cache opcional) + FFN, pre-LN."""
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadAttnKV()
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = FFN()
            self.drop = nn.Dropout(dropout)

        def forward(self, x, attn_mask=None, cache=None):
            h = self.ln1(x)
            x = x + self.drop(self.attn(h, h, attn_mask=attn_mask, cache=cache))
            x = x + self.drop(self.ffn(self.ln2(x)))
            return x

    class DecoderLayer(nn.Module):
        """Auto-atención bidireccional (sin cache, canvas cambia cada paso) +
        cross-attention a la memoria del encoder (con cache opcional, la
        memoria no cambia durante el denoising de un compás) + FFN."""
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.self_attn = MultiHeadAttnKV()
            self.ln2 = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttnKV()
            self.ln3 = nn.LayerNorm(d_model)
            self.ffn = FFN()
            self.drop = nn.Dropout(dropout)

        def forward(self, x, memory, cross_cache=None, memory_key_padding_mask=None):
            h = self.ln1(x)
            x = x + self.drop(self.self_attn(h, h))
            h = self.ln2(x)
            if cross_cache is not None and cross_cache.get('k') is not None:
                # Memoria ya proyectada y cacheada: reusar K/V, solo proyectar Q.
                B, Tq, _ = h.shape
                q = self.cross_attn.q_proj(h).view(B, Tq, n_head, d_head).transpose(1, 2)
                k, v = cross_cache['k'], cross_cache['v']
                out = F.scaled_dot_product_attention(q, k, v)
                out = out.transpose(1, 2).contiguous().view(B, Tq, d_model)
                cross_out = self.cross_attn.out_proj(out)
            else:
                cache_slot = cross_cache if cross_cache is not None else None
                mask = None
                if memory_key_padding_mask is not None:
                    # (B, Tk) True=pad -> (B,1,1,Tk) aditivo booleano invertido
                    mask = (~memory_key_padding_mask).unsqueeze(1).unsqueeze(1)
                cross_out = self.cross_attn(h, memory, attn_mask=mask, cache=cache_slot)
            x = x + self.drop(cross_out)
            x = x + self.drop(self.ffn(self.ln3(x)))
            return x

    class EncoderDecoderBlockDiffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(n_token, d_model)
            self.pos_emb = SinPosEmb()
            self.drop = nn.Dropout(dropout)
            self.enc_layers = nn.ModuleList(EncoderLayer() for _ in range(n_layer_enc))
            self.dec_layers = nn.ModuleList(DecoderLayer() for _ in range(n_layer_dec))
            self.enc_norm = nn.LayerNorm(d_model)
            self.dec_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_token)
            self.n_layer_enc, self.n_layer_dec = n_layer_enc, n_layer_dec

        # ── Modo entrenamiento: secuencia completa de una vez, sin cache ──
        def forward_train(self, hist_ids, hist_block_causal_mask, dec_ids,
                           hist_key_padding_mask=None):
            """
            hist_ids: (B, Th)              dec_ids: (B, Td)
            hist_block_causal_mask: (B, 1, Th, Th) booleano, True=permitido
            Devuelve logits (B, Td, n_token) para las posiciones del decoder.
            """
            h = self.drop(self.tok_emb(hist_ids) +
                           self.pos_emb(0, hist_ids.size(1)).unsqueeze(0))
            for layer in self.enc_layers:
                h = layer(h, attn_mask=hist_block_causal_mask)
            memory = self.enc_norm(h)

            d = self.drop(self.tok_emb(dec_ids) +
                           self.pos_emb(hist_ids.size(1), dec_ids.size(1)).unsqueeze(0))
            for layer in self.dec_layers:
                d = layer(d, memory, memory_key_padding_mask=hist_key_padding_mask)
            d = self.dec_norm(d)
            return self.head(d)

        # ── Modo generación: procesa SOLO tokens nuevos, extiende el cache ──
        def encode_incremental(self, new_block_ids, enc_cache, pos_offset):
            """new_block_ids: (1, L) tokens del compás recién finalizado.
            enc_cache: lista de dicts {'k':...,'v':...} (uno por capa),
            se actualiza in-place. Devuelve los estados finales (1, L, D)
            de ESTE bloque (para añadir a la memoria acumulada)."""
            h = self.drop(self.tok_emb(new_block_ids) +
                           self.pos_emb(pos_offset, new_block_ids.size(1)).unsqueeze(0))
            for i, layer in enumerate(self.enc_layers):
                h = layer(h, attn_mask=None, cache=enc_cache[i])
            return self.enc_norm(h)

        def decode_step(self, dec_ids, memory, cross_caches, pos_offset):
            """dec_ids: (1, Ld) [COND, canvas_ruidoso] del compás actual.
            memory: (1, Th_total, D) estados acumulados de toda la historia.
            cross_caches: lista de dicts (uno por capa decoder); en la
            PRIMERA llamada de un compás deben venir vacíos ({}), así se
            proyecta y cachea la memoria; en llamadas siguientes del mismo
            compás (pasos de denoising sucesivos) se reutiliza sin
            recalcular la proyección K/V de la memoria."""
            d = self.drop(self.tok_emb(dec_ids) +
                           self.pos_emb(pos_offset, dec_ids.size(1)).unsqueeze(0))
            for i, layer in enumerate(self.dec_layers):
                d = layer(d, memory, cross_cache=cross_caches[i])
            d = self.dec_norm(d)
            return self.head(d)

        def new_encoder_cache(self):
            return [{'k': None, 'v': None} for _ in range(self.n_layer_enc)]

        def new_cross_cache(self):
            return [{'k': None, 'v': None} for _ in range(self.n_layer_dec)]

    return EncoderDecoderBlockDiffusion()


def build_block_causal_mask(block_ids, device):
    """block_ids: (B, T) LongTensor con el índice de bloque de cada token
    (los tokens de historia de un mismo compás comparten índice). Devuelve
    una máscara (B, 1, T, T) booleana: True = la posición i puede atender
    a la posición j (block_ids[j] <= block_ids[i], es decir: presente y
    pasado, nunca futuro)."""
    import torch
    bi = block_ids.unsqueeze(2)   # (B, T, 1)
    bj = block_ids.unsqueeze(1)   # (B, 1, T)
    mask = (bj <= bi).unsqueeze(1)  # (B, 1, T, T)
    return mask.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESOS DE DIFUSIÓN: MASKED (MDLM) Y UNIFORM-STATE (UDLM)
# ══════════════════════════════════════════════════════════════════════════════

def alpha_t(t):
    """Schedule lineal α_t = 1 - t.  t=0 → limpio, t=1 → todo ruido."""
    return 1.0 - t


def _canvas_content_ids(word2event):
    """Ids de tokens que pueden aparecer legítimamente en un canvas (excluye
    PAD/MASK/NULL_COND y Chord_*, que nunca aparecen en el canvas — el acorde
    va aparte, como token de condición)."""
    return [w for w, name in word2event.items()
            if name.startswith(('Position_', 'Note Velocity_', 'Note On_',
                                 'Note Duration_', 'Tempo Class_', 'Tempo Value_'))]


def forward_mask(canvas, t, mask_id, pad_id, rng, **_ignored):
    """Forward process de MDLM (absorbing-state): cada token no-PAD se
    reemplaza por MASK independientemente con prob. (1 - alpha_t(t)).
    Devuelve (canvas_ruidoso, corrupted_bool_list)."""
    noisy, corrupted = [], []
    p = 1.0 - alpha_t(t)
    for w in canvas:
        if w == pad_id:
            noisy.append(w); corrupted.append(False)
        elif rng.random() < p:
            noisy.append(mask_id); corrupted.append(True)
        else:
            noisy.append(w); corrupted.append(False)
    return noisy, corrupted


def forward_uniform(canvas, t, content_ids, pad_id, rng, **_ignored):
    """Forward process de UDLM (uniform-state): cada token no-PAD se
    reemplaza por un token ALEATORIO del vocabulario de contenido
    (nunca hay MASK) independientemente con prob. (1 - alpha_t(t)).
    Devuelve (canvas_ruidoso, corrupted_bool_list)."""
    noisy, corrupted = [], []
    p = 1.0 - alpha_t(t)
    for w in canvas:
        if w == pad_id:
            noisy.append(w); corrupted.append(False)
        elif rng.random() < p:
            noisy.append(rng.choice(content_ids)); corrupted.append(True)
        else:
            noisy.append(w); corrupted.append(False)
    return noisy, corrupted


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET: EJEMPLOS DE ENTRENAMIENTO A PARTIR DE FICHEROS DE TOKENS
# ══════════════════════════════════════════════════════════════════════════════

def _load_token_files(data_dir, event2word, verbose=False):
    """Convierte (o reutiliza) todos los MIDI/.json de un directorio a listas
    de palabras REMI, cacheando en <data_dir>/_tokens_cache/*.json."""
    data_dir = Path(data_dir)
    cache_dir = data_dir / '_tokens_cache'
    cache_dir.mkdir(exist_ok=True)

    midi_files = sorted(list(data_dir.glob('*.mid')) + list(data_dir.glob('*.midi')))
    all_words = []
    for mf in midi_files:
        cache_f = cache_dir / (mf.stem + '.json')
        if cache_f.exists():
            with open(cache_f) as f:
                words = json.load(f)
        else:
            try:
                words = midi_to_words(str(mf), event2word, verbose=verbose)
            except Exception as e:
                print(f"  AVISO: {mf.name} omitido ({e})")
                continue
            with open(cache_f, 'w') as f:
                json.dump(words, f)
        if words:
            all_words.append(words)
    return all_words


def build_examples(all_words, word2event, canvas_len, history_bars):
    """Convierte cada pieza tokenizada en su lista de bloques por compás."""
    examples = []
    for words in all_words:
        blocks = words_to_blocks(words, word2event, canvas_len)
        if len(blocks) >= 1:
            examples.append(blocks)
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def cmd_convert(args):
    print("═" * 65)
    print("  REMI DIFFUSION — CONVERT")
    print("═" * 65)
    vocab_path = args.vocab or 'vocab_chord.json'
    if os.path.exists(vocab_path):
        event2word, word2event = load_vocab(vocab_path)
    else:
        event2word, word2event = build_vocab()
        save_vocab(event2word, vocab_path)

    inp = Path(args.input)
    if not inp.exists():
        print(f"  ERROR: la ruta de entrada '{inp}' no existe.")
        sys.exit(1)
    files = [inp] if inp.is_file() else sorted(
        list(inp.glob('*.mid')) + list(inp.glob('*.midi')))
    print(f"  Ficheros: {len(files)}")
    out = {}
    for f in files:
        try:
            words = midi_to_words(str(f), event2word, verbose=args.verbose)
            out[f.name] = words
            print(f"    {f.name}: {len(words)} tokens")
        except Exception as e:
            print(f"    {f.name}: OMITIDO ({e})")
    if args.output:
        with open(args.output, 'w') as fh:
            json.dump(out, fh)
        print(f"  → Tokens guardados en {args.output}")
    print("═" * 65)


def cmd_train(args):
    import torch
    import torch.nn as nn

    print("═" * 65)
    print("  REMI DIFFUSION v2 — TRAIN")
    print("═" * 65)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}  |  noise_type: {args.noise_type}")

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.vocab or str(model_dir / 'vocab_chord.json')
    if os.path.exists(vocab_path):
        event2word, word2event = load_vocab(vocab_path)
    else:
        event2word, word2event = build_vocab()
        save_vocab(event2word, vocab_path)
    n_token = len(event2word)
    pad_id, mask_id = _sp(word2event, 'PAD'), _sp(word2event, 'MASK')
    null_cond_id = _sp(word2event, 'NULL_COND')
    content_ids = _canvas_content_ids(word2event)

    print(f"\n[1/4] Cargando y tokenizando corpus de {args.data_dir}…")
    all_words = _load_token_files(args.data_dir, event2word, verbose=args.verbose)
    examples  = build_examples(all_words, word2event, args.canvas_len, args.history_bars)
    print(f"  Piezas: {len(examples)}  |  vocab: {n_token} tokens  |  "
          f"canvas_len: {args.canvas_len}")
    if not examples:
        print("  ERROR: no hay ejemplos de entrenamiento. Revisa --data-dir.")
        sys.exit(1)

    if args.small:
        d_model, n_layer_enc, n_layer_dec, n_head = 256, 3, 1, 4
    else:
        d_model, n_layer_enc, n_layer_dec, n_head = (
            args.d_model, args.n_layer_enc, args.n_layer_dec, args.n_head)
    print(f"\n[2/4] Construyendo modelo encoder-decoder (d_model={d_model}, "
          f"n_layer_enc={n_layer_enc}, n_layer_dec={n_layer_dec}, n_head={n_head})…")
    model = _build_model(n_token, d_model=d_model, n_layer_enc=n_layer_enc,
                          n_layer_dec=n_layer_dec, n_head=n_head,
                          max_len=MAX_POS_BUFFER).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch, best_loss, patience_ctr = 0, float('inf'), 0
    ckpt_path = model_dir / 'model.pt'
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_epoch = ck['epoch'] + 1
        best_loss   = ck.get('best_loss', float('inf'))
        print(f"  Reanudado desde época {start_epoch}")

    cfg = {
        'arch': 'v2', 'n_token': n_token, 'd_model': d_model,
        'n_layer_enc': n_layer_enc, 'n_layer_dec': n_layer_dec, 'n_head': n_head,
        'canvas_len': args.canvas_len, 'history_bars': args.history_bars,
        'vocab': vocab_path, 'noise_type': args.noise_type,
    }
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[3/4] Entrenando ({args.epochs} épocas máx., patience={args.patience})…")
    rng = random.Random(args.seed)
    forward_noise = forward_mask if args.noise_type == 'mask' else forward_uniform
    noise_extra = dict(mask_id=mask_id) if args.noise_type == 'mask' else dict(content_ids=content_ids)

    def sample_batch(bs):
        """Construye un batch: para cada ejemplo, elige un compás objetivo al
        azar, toma hasta --history-bars compases previos como historia limpia
        (con su propio índice de bloque, para la máscara block-causal del
        encoder), aplica el forward process (mask o uniform) al canvas
        objetivo con t~U(0,1), y aplica cond-dropout (D-CFG)."""
        hist_seqs, hist_blockids, dec_seqs, dec_targets, dec_lossmasks, ts = \
            [], [], [], [], [], []
        for _ in range(bs):
            blocks = examples[rng.randrange(len(examples))]
            i = rng.randrange(len(blocks))
            hist_blocks = blocks[max(0, i - args.history_bars):i]

            h_tokens, h_blockids = [], []
            for bidx, hb in enumerate(hist_blocks):
                cond_h = hb['cond'] if hb['cond'] is not None else null_cond_id
                h_tokens.append(cond_h); h_blockids.append(bidx)
                h_tokens.extend(hb['canvas']); h_blockids.extend([bidx] * len(hb['canvas']))
            if not h_tokens:  # sin historia (primer compás de la pieza): 1 token neutro
                h_tokens, h_blockids = [null_cond_id], [0]

            cond = blocks[i]['cond'] if blocks[i]['cond'] is not None else null_cond_id
            if rng.random() < args.cond_dropout:
                cond = null_cond_id

            t = rng.random() * 0.999 + 0.001
            noisy_canvas, corrupted = forward_noise(
                blocks[i]['canvas'], t, pad_id=pad_id, rng=rng, **noise_extra)

            dec_seq = [cond] + noisy_canvas
            dec_tgt = [cond] + blocks[i]['canvas']
            dec_lm  = [False] + corrupted

            hist_seqs.append(h_tokens); hist_blockids.append(h_blockids)
            dec_seqs.append(dec_seq); dec_targets.append(dec_tgt)
            dec_lossmasks.append(dec_lm); ts.append(t)

        Hmax = max(len(s) for s in hist_seqs)
        Dmax = max(len(s) for s in dec_seqs)
        hist_x = torch.full((bs, Hmax), pad_id, dtype=torch.long)
        hist_bid = torch.full((bs, Hmax), -1, dtype=torch.long)
        hist_pad_mask = torch.ones((bs, Hmax), dtype=torch.bool)  # True = pad
        dec_x = torch.full((bs, Dmax), pad_id, dtype=torch.long)
        dec_y = torch.full((bs, Dmax), pad_id, dtype=torch.long)
        dec_lm = torch.zeros((bs, Dmax), dtype=torch.bool)
        ts_t = torch.ones((bs,), dtype=torch.float)
        for b in range(bs):
            Lh = len(hist_seqs[b])
            hist_x[b, :Lh] = torch.tensor(hist_seqs[b])
            hist_bid[b, :Lh] = torch.tensor(hist_blockids[b])
            hist_pad_mask[b, :Lh] = False
            Ld = len(dec_seqs[b])
            dec_x[b, :Ld] = torch.tensor(dec_seqs[b])
            dec_y[b, :Ld] = torch.tensor(dec_targets[b])
            dec_lm[b, :Ld] = torch.tensor(dec_lossmasks[b])
            ts_t[b] = ts[b]
        # Posiciones de padding del historial: nunca deben ser atendibles ni
        # atender (se marcan con block_id muy alto para que la máscara
        # block-causal las excluya de facto; ya están fuera de rango del
        # resto igualmente al ser pad_id, pero por seguridad las aislamos).
        hist_bid[hist_pad_mask] = 10**6
        block_mask = build_block_causal_mask(hist_bid, device)
        return (hist_x.to(device), block_mask, dec_x.to(device), dec_y.to(device),
                dec_lm.to(device), ts_t.to(device), hist_pad_mask.to(device))

    steps_per_epoch = max(1, len(examples) // args.batch_size)
    ce = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            hist_x, block_mask, dec_x, dec_y, dec_lm, ts_t, hist_pad = sample_batch(args.batch_size)
            logits = model.forward_train(hist_x, block_mask, dec_x,
                                          hist_key_padding_mask=hist_pad)  # True=PAD
            flat_logits = logits.reshape(-1, n_token)
            flat_y      = dec_y.reshape(-1)
            flat_lm     = dec_lm.reshape(-1)
            if flat_lm.sum() == 0:
                continue
            losses = ce(flat_logits[flat_lm], flat_y[flat_lm])
            t_expand = ts_t.view(-1, 1).expand_as(dec_lm).reshape(-1)[flat_lm]
            weighted = losses / t_expand.clamp(min=1e-3)
            loss = weighted.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        epoch_loss /= steps_per_epoch
        dt = time.time() - t0
        print(f"  Época {epoch+1:4d}/{args.epochs}  loss={epoch_loss:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({dt:.1f}s)")

        improved = epoch_loss < best_loss - 1e-4
        if improved:
            best_loss, patience_ctr = epoch_loss, 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'best_loss': best_loss}, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping en época {epoch+1} (patience={args.patience})")
                break

    print(f"\n[4/4] Modelo guardado en {ckpt_path}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  MUESTREO: DENOISING ITERATIVO + REMASKING + D-CFG
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_for_inference(model_dir, device):
    import torch
    model_dir = Path(model_dir)
    cfg_path = model_dir / 'model_config.json'
    if not cfg_path.exists():
        print(f"  ERROR: no se encontró un modelo entrenado en '{model_dir}' "
              f"(falta {cfg_path.name}). ¿La ruta de --model-dir es correcta?")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get('arch') != 'v2':
        print(f"  ERROR: '{model_dir}' contiene un modelo v1 (arquitectura antigua). "
              f"Usa remi_diffusion.py (v1) para este modelo, o re-entrena con v2.")
        sys.exit(1)
    # Buffer de posiciones generoso: la generación puede recorrer muchos más
    # tokens que los vistos durante una ventana de entrenamiento; el embedding
    # sinusoidal (sin parámetros aprendidos) es válido para cualquier longitud.
    model = _build_model(cfg['n_token'], d_model=cfg['d_model'],
                          n_layer_enc=cfg['n_layer_enc'], n_layer_dec=cfg['n_layer_dec'],
                          n_head=cfg['n_head'], max_len=MAX_POS_BUFFER).to(device)
    ck = torch.load(model_dir / 'model.pt', map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, cfg


def _parse_chords(chord_str, event2word):
    """'C:maj G:maj A:min F:maj' → lista de ids de token Chord_X:Y."""
    ids = []
    for tok in chord_str.split():
        key = f'Chord_{tok}'
        if key not in event2word:
            raise ValueError(f"Acorde desconocido: '{tok}' "
                              f"(formato esperado: Raíz:calidad, ej. C:maj, "
                              f"calidades válidas: maj/min/dim/aug/dom)")
        ids.append(event2word[key])
    return ids


def sample_block_mask(model, memory, cond_id, null_cond_id, canvas_len, pad_id, mask_id,
                       n_token, device, pos_offset, steps, guidance_scale, remask_max,
                       temperature, topk, rng):
    """Sampler MDLM (absorbing-state): denoising iterativo con remasking
    (ReMDM) + D-CFG, sobre la memoria (ya codificada) de la historia."""
    import torch
    import torch.nn.functional as F

    canvas = [mask_id] * canvas_len
    is_masked = [True] * canvas_len
    cross_cache = model.new_cross_cache()

    def dec_logits(cond):
        dec = torch.tensor([[cond] + canvas], device=device)
        with torch.no_grad():
            return model.decode_step(dec, memory, cross_cache, pos_offset)[0, 1:]

    for step in range(steps):
        t = 1.0 - step / steps
        s = max(1.0 - (step + 1) / steps, 0.0)

        logits_cond = dec_logits(cond_id)
        if guidance_scale != 0:
            logits_uncond = dec_logits(null_cond_id)
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        logits = logits / max(temperature, 1e-4)
        if topk and topk > 0:
            topv, topi = torch.topk(logits, k=min(topk, n_token), dim=-1)
            filt = torch.full_like(logits, float('-inf'))
            filt.scatter_(-1, topi, topv)
            logits = filt
        probs = F.softmax(logits, dim=-1)
        x0_pred = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        a_t, a_s = alpha_t(t), alpha_t(s)
        denom = max(1.0 - a_t, 1e-6)
        p_unmask = (a_s - a_t) / denom if a_t < 1.0 else 1.0
        p_unmask = min(max(p_unmask, 0.0), 1.0)
        p_remask = remask_max * s

        for j in range(canvas_len):
            if canvas[j] == pad_id and not is_masked[j]:
                continue
            if is_masked[j]:
                if rng.random() < p_unmask or step == steps - 1:
                    canvas[j] = x0_pred[j]; is_masked[j] = False
            else:
                if rng.random() < p_remask:
                    canvas[j] = mask_id; is_masked[j] = True

    if any(is_masked):
        preds = dec_logits(cond_id).argmax(dim=-1).tolist()
        for j in range(canvas_len):
            if is_masked[j]:
                canvas[j] = preds[j]
    return canvas


def sample_block_uniform(model, memory, cond_id, null_cond_id, canvas_len, pad_id,
                          content_ids, n_token, device, pos_offset, steps, guidance_scale,
                          remask_max, temperature, topk, rng):
    """Sampler UDLM (uniform-state): el canvas nunca contiene [MASK]. Cada
    posición no comprometida recibe un token aleatorio fresco en cada paso;
    con probabilidad p_commit se fija a la predicción x0 del modelo (deja de
    tocarse); posiciones ya comprometidas pueden "renoisarse" (volver a
    aleatorizarse) para permitir corrección de errores — igual que el
    esquema descrito para Gemma Diffusion en el artículo."""
    import torch
    import torch.nn.functional as F

    canvas = [rng.choice(content_ids) for _ in range(canvas_len)]
    committed = [False] * canvas_len
    cross_cache = model.new_cross_cache()

    def dec_logits(cond):
        dec = torch.tensor([[cond] + canvas], device=device)
        with torch.no_grad():
            return model.decode_step(dec, memory, cross_cache, pos_offset)[0, 1:]

    for step in range(steps):
        t = 1.0 - step / steps
        s = max(1.0 - (step + 1) / steps, 0.0)

        logits_cond = dec_logits(cond_id)
        if guidance_scale != 0:
            logits_uncond = dec_logits(null_cond_id)
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_cond

        logits = logits / max(temperature, 1e-4)
        if topk and topk > 0:
            topv, topi = torch.topk(logits, k=min(topk, n_token), dim=-1)
            filt = torch.full_like(logits, float('-inf'))
            filt.scatter_(-1, topi, topv)
            logits = filt
        probs = F.softmax(logits, dim=-1)
        x0_pred = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        a_t, a_s = alpha_t(t), alpha_t(s)
        denom = max(1.0 - a_t, 1e-6)
        p_commit = (a_s - a_t) / denom if a_t < 1.0 else 1.0
        p_commit = min(max(p_commit, 0.0), 1.0)
        p_renoise = remask_max * s

        for j in range(canvas_len):
            if not committed[j]:
                if rng.random() < p_commit or step == steps - 1:
                    canvas[j] = x0_pred[j]; committed[j] = True
                else:
                    canvas[j] = rng.choice(content_ids)
            else:
                if rng.random() < p_renoise:
                    canvas[j] = rng.choice(content_ids); committed[j] = False

    if not all(committed):
        preds = dec_logits(cond_id).argmax(dim=-1).tolist()
        for j in range(canvas_len):
            if not committed[j]:
                canvas[j] = preds[j]
    return canvas


def _generate_common(args, prompt_blocks=None):
    import torch

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = _load_model_for_inference(args.model_dir, device)
    event2word, word2event = load_vocab(cfg['vocab'])
    pad_id, mask_id = _sp(word2event, 'PAD'), _sp(word2event, 'MASK')
    null_cond_id = _sp(word2event, 'NULL_COND')
    canvas_len   = cfg['canvas_len']
    history_bars = cfg['history_bars']
    n_token      = cfg['n_token']
    noise_type   = cfg.get('noise_type', 'uniform')
    content_ids  = _canvas_content_ids(word2event)

    if args.chords:
        try:
            chord_ids = _parse_chords(args.chords, event2word)
        except ValueError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        chord_ids = [null_cond_id]
    rng = random.Random(args.seed)

    max_hist_tokens = max(history_bars, 1) * (canvas_len + 1)
    enc_cache = model.new_encoder_cache()
    state = {'memory': None, 'pos': 0}

    def add_to_history(tokens_1d):
        toks = torch.tensor([tokens_1d], device=device)
        with torch.no_grad():
            out = model.encode_incremental(toks, enc_cache, state['pos'])
        state['pos'] += toks.size(1)
        state['memory'] = out if state['memory'] is None else torch.cat([state['memory'], out], dim=1)
        if state['memory'].size(1) > max_hist_tokens:
            excess = state['memory'].size(1) - max_hist_tokens
            state['memory'] = state['memory'][:, excess:]
            for lc in enc_cache:
                lc['k'] = lc['k'][:, :, excess:, :]
                lc['v'] = lc['v'][:, :, excess:, :]

    blocks = list(prompt_blocks) if prompt_blocks else []
    n_prompt_bars = len(blocks)
    for hb in (blocks[-history_bars:] if history_bars > 0 else []):
        cond_h = hb['cond'] if hb['cond'] is not None else null_cond_id
        add_to_history([cond_h] + list(hb['canvas']))
    if state['memory'] is None:
        add_to_history([null_cond_id])  # semilla neutra, igual que en entrenamiento

    for i in range(args.bars):
        cond_id = chord_ids[i % len(chord_ids)]
        pos_for_block = state['pos']

        if noise_type == 'mask':
            canvas = sample_block_mask(
                model, state['memory'], cond_id, null_cond_id, canvas_len, pad_id, mask_id,
                n_token, device, pos_for_block, args.steps, args.guidance_scale,
                args.remask_max, args.temperature, args.topk, rng)
        else:
            canvas = sample_block_uniform(
                model, state['memory'], cond_id, null_cond_id, canvas_len, pad_id,
                content_ids, n_token, device, pos_for_block, args.steps,
                args.guidance_scale, args.remask_max, args.temperature, args.topk, rng)

        if args.repair_grammar:
            raw_used = sum(1 for w in canvas if w != pad_id)
            canvas = repair_canvas_grammar(canvas, word2event, pad_id)
            canvas = canvas + [pad_id] * (canvas_len - len(canvas))
            kept = sum(1 for w in canvas if w != pad_id)
            tag = f"  [reparación gramatical: {kept}/{raw_used} tokens salvados]" \
                  if kept < raw_used else ""
        else:
            tag = ""

        blocks.append({'cond': cond_id, 'canvas': canvas})
        add_to_history([cond_id] + list(canvas))
        print(f"  Compás {i+1}/{args.bars} generado (acorde objetivo: "
              f"{word2event[cond_id]}, noise_type={noise_type}){tag}")

    words = blocks_to_words(blocks, word2event)
    return words, word2event, n_prompt_bars


def cmd_generate(args):
    print("═" * 65)
    print("  REMI DIFFUSION v2 — GENERATE")
    print("═" * 65)
    words, word2event, _ = _generate_common(args)
    output = args.output or 'generated.mid'
    write_midi(words, word2event, output)
    print(f"\n  → {output}")
    print("═" * 65)


def cmd_continue(args):
    print("═" * 65)
    print("  REMI DIFFUSION v2 — CONTINUE")
    print("═" * 65)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = _load_model_for_inference(args.model_dir, device)
    event2word, word2event = load_vocab(cfg['vocab'])

    print(f"  Tokenizando prompt: {args.prompt}")
    words = midi_to_words(args.prompt, event2word, verbose=args.verbose)
    prompt_blocks = words_to_blocks(words, word2event, cfg['canvas_len'])
    print(f"  Compases de contexto en el prompt: {len(prompt_blocks)}")

    all_words, word2event, n_prompt_bars = _generate_common(args, prompt_blocks=prompt_blocks)
    output = args.output or 'continued.mid'
    write_midi(all_words, word2event, output)
    print(f"\n  → {output}  ({n_prompt_bars} compases de prompt + {args.bars} generados)")
    print("═" * 65)


def cmd_inspect(args):
    print("═" * 65)
    print("  REMI DIFFUSION v2 — INSPECT")
    print("═" * 65)
    vocab_path = args.vocab or 'vocab_chord.json'
    if os.path.exists(vocab_path):
        event2word, word2event = load_vocab(vocab_path)
    else:
        event2word, word2event = build_vocab()
    words = midi_to_words(args.input, event2word, verbose=args.verbose)
    blocks = words_to_blocks(words, word2event, args.canvas_len, verbose=True)
    print(f"  Compases: {len(blocks)}  |  canvas_len: {args.canvas_len}")
    for i, b in enumerate(blocks[:args.bars_show]):
        cond_name = word2event[b['cond']] if b['cond'] is not None else 'NULL_COND'
        filled = sum(1 for w in b['canvas'] if word2event[w] != 'PAD_None')
        print(f"    Compás {i+1:3d}  cond={cond_name:14s}  "
              f"tokens_usados={filled}/{args.canvas_len}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="REMI Diffusion v1 — Masked Diffusion LM para piano pop "
                     "(block diffusion + D-CFG por acordes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest='command', required=True, metavar='COMANDO')

    # ── convert ──────────────────────────────────────────────────────────────
    p = sub.add_parser('convert', help='MIDI(s) → tokens REMI+chord')
    p.add_argument('input', help='MIDI o directorio de MIDIs')
    p.add_argument('--vocab', default=None, help='Ruta del vocabulario JSON')
    p.add_argument('--output', default=None, help='Guardar tokens en JSON')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_convert)

    # ── train ────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrenar el modelo encoder-decoder de difusión por bloques',
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description=textwrap.dedent("""\
                            Entrena el modelo encoder-decoder de block diffusion (v2) sobre un
                            corpus de MIDIs. Cada paso: elige una pieza y un compás al azar,
                            arma la historia (compases previos, limpios, con máscara block-
                            causal en el encoder) + condición de acorde (con dropout para
                            D-CFG) + canvas objetivo corrompido con t~U(0,1) (mask o uniform
                            según --noise-type); pérdida de cross-entropy ponderada por 1/t.
                        """))
    p.add_argument('data_dir', help='Directorio con MIDIs de entrenamiento')
    p.add_argument('--model-dir', required=True, dest='model_dir')
    p.add_argument('--vocab', default=None)
    p.add_argument('--canvas-len', type=int, default=CANVAS_LEN_DEFAULT, dest='canvas_len')
    p.add_argument('--history-bars', type=int, default=HISTORY_BARS_DEFAULT, dest='history_bars')
    p.add_argument('--d-model', type=int, default=512, dest='d_model')
    p.add_argument('--n-layer-enc', type=int, default=6, dest='n_layer_enc',
                   help='Capas del encoder (pesado, procesa la historia) [6]')
    p.add_argument('--n-layer-dec', type=int, default=2, dest='n_layer_dec',
                   help='Capas del decoder (ligero, denoisa el compás actual) [2]')
    p.add_argument('--n-head', type=int, default=8, dest='n_head')
    p.add_argument('--noise-type', choices=['mask', 'uniform'], default='uniform',
                   dest='noise_type',
                   help="'mask'=MDLM (como v1), 'uniform'=UDLM (mejor "
                        "controlabilidad según el artículo) [uniform]")
    p.add_argument('--cond-dropout', type=float, default=0.15, dest='cond_dropout',
                   help='Prob. de sustituir el acorde por NULL_COND (CFG training) [0.15]')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--small', action='store_true',
                   help='Modelo reducido: enc=3/dec=1 capas, d=256 (CPU-friendly)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_train)

    # ── generate ─────────────────────────────────────────────────────────────
    p = sub.add_parser('generate', help='Generar piano pop desde cero, guiado por acordes')
    p.add_argument('--model-dir', required=True, dest='model_dir')
    p.add_argument('--chords', default=None,
                   help='Progresión objetivo, ej. "C:maj G:maj A:min F:maj" '
                        '(se cicla para cubrir --bars; sin este flag, sin guidance)')
    p.add_argument('--bars', type=int, default=16)
    p.add_argument('--steps', type=int, default=20, help='Pasos de denoising por compás')
    p.add_argument('--guidance-scale', type=float, default=3.0, dest='guidance_scale')
    p.add_argument('--remask-max', type=float, default=0.15, dest='remask_max')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--topk', type=int, default=0)
    p.add_argument('--repair-grammar', dest='repair_grammar', action='store_true',
                   default=True, help='Reparar gramática REMI del canvas generado [on]')
    p.add_argument('--no-repair-grammar', dest='repair_grammar', action='store_false',
                   help='Desactivar la reparación gramatical (para depuración)')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_generate)

    # ── continue ─────────────────────────────────────────────────────────────
    p = sub.add_parser('continue', help='Continuar un MIDI de prompt con nueva progresión')
    p.add_argument('prompt', help='MIDI de prompt')
    p.add_argument('--model-dir', required=True, dest='model_dir')
    p.add_argument('--chords', default=None)
    p.add_argument('--bars', type=int, default=16)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--guidance-scale', type=float, default=3.0, dest='guidance_scale')
    p.add_argument('--remask-max', type=float, default=0.15, dest='remask_max')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--topk', type=int, default=0)
    p.add_argument('--repair-grammar', dest='repair_grammar', action='store_true',
                   default=True, help='Reparar gramática REMI del canvas generado [on]')
    p.add_argument('--no-repair-grammar', dest='repair_grammar', action='store_false',
                   help='Desactivar la reparación gramatical (para depuración)')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_continue)

    # ── inspect ──────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Diagnóstico de bloques/canvas de un MIDI')
    p.add_argument('--input', required=True)
    p.add_argument('--vocab', default=None)
    p.add_argument('--canvas-len', type=int, default=CANVAS_LEN_DEFAULT, dest='canvas_len')
    p.add_argument('--bars-show', type=int, default=8, dest='bars_show')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_inspect)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
