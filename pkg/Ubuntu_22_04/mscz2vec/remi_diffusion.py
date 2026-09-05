#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      REMI DIFFUSION  v1                                      ║
║   Masked Diffusion Language Model aplicado a composición de piano pop        ║
║                                                                              ║
║  IDEA:                                                                       ║
║    remi.py genera con un Transformer-XL AUTORREGRESIVO (izq→der, sin        ║
║    corrección de errores). Este módulo usa el MISMO esquema de tokens       ║
║    REMI, pero sustituye la generación autorregresiva por DIFUSIÓN            ║
║    ENMASCARADA (MDLM) sobre bloques de un compás cada uno ("block           ║
║    diffusion"), con guidance sin clasificador (D-CFG) sobre progresiones    ║
║    de acordes/tonalidad y un sampler con remasking (ReMDM) para corregir    ║
║    errores durante el muestreo.                                             ║
║                                                                              ║
║    Referencia: Kuleshov Group, "How to Build a Diffusion Language Model"    ║
║    (2026) — https://kuleshov-group.github.io/blog/blog/2026/               ║
║               how-to-build-a-diffusion-language-model/                      ║
║                                                                              ║
║  ARQUITECTURA (v1, simplificada respecto al artículo):                      ║
║    • Un único Transformer bidireccional (no encoder-decoder separado)       ║
║      procesa  [historia clara] + [token de condición] + [canvas ruidoso]    ║
║      con atención bidireccional completa; la pérdida solo se calcula        ║
║      sobre las posiciones del canvas. El split encoder/decoder pesado/      ║
║      ligero de Gemma Diffusion queda como optimización futura (v2) —        ║
║      aquí se prioriza corrección y legibilidad sobre velocidad.             ║
║    • Block diffusion: unidad de bloque = 1 compás (frontera natural         ║
║      marcada por el token Bar de REMI). Cada bloque se difunde              ║
║      condicionado en los compases ya generados (historia, sin ruido).       ║
║    • Canvas de longitud fija: cada compás se aplana a --canvas-len          ║
║      tokens REMI (Position/Velocity/NoteOn/Duration/Chord), rellenando      ║
║      con PAD si sobra espacio y truncando (con aviso) si falta.             ║
║    • Guidance D-CFG: el primer token del bloque es el token de acorde       ║
║      objetivo (Chord_X:Y, ya definido en remi.py). Durante entrenamiento    ║
║      se sustituye por NULL_COND con probabilidad --cond-dropout, así el     ║
║      mismo modelo aprende la rama condicional y la no-condicional.          ║
║    • Forward process: enmascara tokens del canvas con schedule             ║
║      α_t = 1 − t (lineal, como en el ELBO simplificado del artículo).       ║
║    • Sampler: infilling + remasking (ReMDM) iterativo, con CFG combinando   ║
║      logits condicionados y no condicionados vía --guidance-scale.          ║
║                                                                              ║
║  COMPATIBILIDAD:                                                             ║
║    Usa el mismo vocab_chord.json que remi.py (--chord), extendido con       ║
║    3 tokens especiales (PAD_None, MASK_None, NULL_COND_None) añadidos al    ║
║    final. Si cargas un vocab.json de remi.py sin esos tokens, se extiende   ║
║    automáticamente (y se re-guarda) la primera vez.                        ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    convert   — MIDI(s) → tokens REMI+chord (idéntico a remi.py --chord)     ║
║    train     — Entrena el Transformer de difusión enmascarada por bloques   ║
║    generate  — Genera piano pop desde cero guiado por una progresión        ║
║    continue  — Continúa un MIDI de prompt con una progresión objetivo       ║
║    inspect   — Diagnóstico: bloques/canvas de un MIDI o corpus              ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, torch                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EJEMPLOS                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  # Preparar corpus (idéntico esquema que remi.py, con acordes obligatorio)  ║
║  python remi_diffusion.py convert corpus/ --vocab vocab_chord.json          ║
║                                                                              ║
║  # Entrenar                                                                 ║
║  python remi_diffusion.py train corpus/ --model-dir diff_model/ \           ║
║      --epochs 300 --batch-size 8 --canvas-len 48 --lr 2e-4                  ║
║                                                                              ║
║  # Generar desde cero con progresión de acordes objetivo                    ║
║  python remi_diffusion.py generate --model-dir diff_model/ \                ║
║      --chords "C:maj G:maj A:min F:maj" --bars 16 \                        ║
║      --guidance-scale 3.0 --steps 20 --output nueva.mid                     ║
║                                                                              ║
║  # Continuar un MIDI existente con nueva progresión                        ║
║  python remi_diffusion.py continue prompt.mid --model-dir diff_model/ \     ║
║      --chords "D:min G:dom C:maj" --bars 8 --guidance-scale 4.0             ║
║                                                                              ║
║  # Diagnóstico: ver cómo quedan los bloques/canvas de un MIDI               ║
║  python remi_diffusion.py inspect --input prompt.mid --canvas-len 48        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  OPCIONES COMUNES DE TRAIN:
    --canvas-len N        Tokens por compás en el canvas [default: 48]
    --history-bars N      Compases de historia máx. que ve el modelo [default: 8]
    --d-model N           Dimensión del modelo [default: 512]
    --n-layer N           Capas del Transformer [default: 8]
    --n-head N            Cabezas de atención [default: 8]
    --cond-dropout F      Prob. de sustituir el acorde por NULL_COND [default: 0.15]
    --epochs N            Épocas máximas [default: 300]
    --batch-size N        Tamaño de batch [default: 8]
    --lr F                Learning rate [default: 2e-4]
    --patience N          Early stopping [default: 30]
    --resume              Reanudar desde checkpoint
    --small               Modelo reducido: 4 capas, d=256 (~CPU-friendly)

  OPCIONES COMUNES DE GENERATE / CONTINUE:
    --chords "X:q Y:q …"  Progresión objetivo (se repite/cicla para cubrir --bars)
    --bars N               Compases a generar [default: 16]
    --steps N               Pasos de denoising por compás [default: 20]
    --guidance-scale F      Fuerza de D-CFG, 0=sin guidance [default: 3.0]
    --remask-max F          Fracción máx. de tokens ya rellenados que se
                             re-enmascaran por paso (ReMDM) [default: 0.15]
    --temperature F         Temperatura de muestreo [default: 1.0]
    --topk N                Top-k al muestrear cada posición [default: 0 (off)]
    --seed N                Semilla [default: 42]
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
#  MODELO: TRANSFORMER BIDIRECCIONAL DE DIFUSIÓN ENMASCARADA POR BLOQUES
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(n_token, d_model=512, n_layer=8, n_head=8, d_ff=2048,
                  dropout=0.1, max_len=4096):
    """
    Transformer bidireccional (encoder-only, sin máscara causal).
    Secuencia de entrada por ejemplo de entrenamiento:
        [historia clara (N_hist tokens)] + [COND] + [canvas ruidoso (canvas_len)]
    La pérdida se calcula solo sobre las posiciones del canvas.
    forward(x) -> logits (B, T, n_token)
    """
    import torch
    import torch.nn as nn

    class SinPosEmb(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pos  = torch.arange(max_len).unsqueeze(1).float()
            div  = torch.exp(-math.log(10000.0) *
                              torch.arange(0, d_model, 2).float() / d_model)
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe)

        def forward(self, length):
            return self.pe[:length]

    class BlockDiffusionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(n_token, d_model)
            self.pos_emb = SinPosEmb(d_model, max_len)
            self.drop    = nn.Dropout(dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm    = nn.LayerNorm(d_model)
            self.head    = nn.Linear(d_model, n_token)

        def forward(self, x, key_padding_mask=None):
            # x: (B, T) ids.  key_padding_mask: (B, T) bool, True = ignorar.
            h = self.drop(self.tok_emb(x) + self.pos_emb(x.size(1)).unsqueeze(0))
            h = self.encoder(h, src_key_padding_mask=key_padding_mask)
            h = self.norm(h)
            return self.head(h)

    return BlockDiffusionTransformer()


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESO DE DIFUSIÓN ENMASCARADA  (forward / schedule / pérdida)
# ══════════════════════════════════════════════════════════════════════════════

def alpha_t(t):
    """Schedule lineal α_t = 1 - t.  t=0 → limpio, t=1 → todo enmascarado."""
    return 1.0 - t


def forward_mask(canvas, t, mask_id, pad_id, rng):
    """
    Aplica el forward process de MDLM a un canvas (lista de ints).
    Cada token no-PAD se enmascara independientemente con prob. (1 - alpha_t(t)).
    Los PAD nunca se enmascaran (no aportan señal) pero tampoco cuentan en
    la pérdida — se excluyen aparte.
    Devuelve (canvas_ruidoso, mask_positions_bool_list).
    """
    noisy, masked = [], []
    p_mask = 1.0 - alpha_t(t)
    for w in canvas:
        if w == pad_id:
            noisy.append(w); masked.append(False)
        elif rng.random() < p_mask:
            noisy.append(mask_id); masked.append(True)
        else:
            noisy.append(w); masked.append(False)
    return noisy, masked


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
    print("  REMI DIFFUSION — TRAIN")
    print("═" * 65)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

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

    print(f"\n[1/4] Cargando y tokenizando corpus de {args.data_dir}…")
    all_words = _load_token_files(args.data_dir, event2word, verbose=args.verbose)
    examples  = build_examples(all_words, word2event, args.canvas_len, args.history_bars)
    print(f"  Piezas: {len(examples)}  |  vocab: {n_token} tokens  |  "
          f"canvas_len: {args.canvas_len}")
    if not examples:
        print("  ERROR: no hay ejemplos de entrenamiento. Revisa --data-dir.")
        sys.exit(1)

    d_model, n_layer, n_head = (256, 4, 4) if args.small else (args.d_model, args.n_layer, args.n_head)
    print(f"\n[2/4] Construyendo modelo (d_model={d_model}, n_layer={n_layer}, "
          f"n_head={n_head})…")
    max_seq = args.canvas_len * (args.history_bars + 1) + args.history_bars + 2
    model = _build_model(n_token, d_model=d_model, n_layer=n_layer, n_head=n_head,
                          max_len=max_seq).to(device)
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
        'n_token': n_token, 'd_model': d_model, 'n_layer': n_layer, 'n_head': n_head,
        'canvas_len': args.canvas_len, 'history_bars': args.history_bars,
        'vocab': vocab_path, 'max_seq': max_seq,
    }
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[3/4] Entrenando ({args.epochs} épocas máx., patience={args.patience})…")
    rng = random.Random(args.seed)

    def sample_batch(bs):
        """Construye un batch: para cada ejemplo, elige un compás objetivo
        al azar, toma hasta --history-bars compases previos como historia
        limpia, aplica forward-masking al canvas objetivo con t~U(0,1), y
        aplica cond-dropout (D-CFG) sobre el token de condición."""
        seqs, targets, loss_masks, pad_masks = [], [], [], []
        for _ in range(bs):
            blocks = examples[rng.randrange(len(examples))]
            i = rng.randrange(len(blocks))
            hist_blocks = blocks[max(0, i - args.history_bars):i]
            hist_tokens = []
            for hb in hist_blocks:
                hist_tokens.append(hb['cond'] if hb['cond'] is not None else null_cond_id)
                hist_tokens.extend(hb['canvas'])

            cond = blocks[i]['cond'] if blocks[i]['cond'] is not None else null_cond_id
            if rng.random() < args.cond_dropout:
                cond = null_cond_id

            t = rng.random() * 0.999 + 0.001  # evitar t=0 exacto (1/t indefinido)
            noisy_canvas, mask_flags = forward_mask(
                blocks[i]['canvas'], t, mask_id, pad_id, rng)

            seq = hist_tokens + [cond] + noisy_canvas
            tgt = [pad_id] * len(hist_tokens) + [cond] + blocks[i]['canvas']
            lm  = [False] * len(hist_tokens) + [False] + mask_flags
            seqs.append(seq); targets.append(tgt); loss_masks.append((lm, t))

        maxlen = max(len(s) for s in seqs)
        x  = torch.full((bs, maxlen), pad_id, dtype=torch.long)
        y  = torch.full((bs, maxlen), pad_id, dtype=torch.long)
        lm = torch.zeros((bs, maxlen), dtype=torch.bool)
        pm = torch.ones((bs, maxlen), dtype=torch.bool)  # True = padding (ignorar)
        ts = torch.ones((bs,), dtype=torch.float)
        for b, (seq, tgt, (mflags, t)) in enumerate(zip(seqs, targets, loss_masks)):
            L = len(seq)
            x[b, :L] = torch.tensor(seq)
            y[b, :L] = torch.tensor(tgt)
            lm[b, :L] = torch.tensor(mflags)
            pm[b, :L] = False
            ts[b] = t
        return x.to(device), y.to(device), lm.to(device), pm.to(device), ts.to(device)

    steps_per_epoch = max(1, len(examples) // args.batch_size)
    ce = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            x, y, lm, pm, ts = sample_batch(args.batch_size)
            logits = model(x, key_padding_mask=pm)  # (B, T, V)
            flat_logits = logits.reshape(-1, n_token)
            flat_y      = y.reshape(-1)
            flat_lm     = lm.reshape(-1)
            if flat_lm.sum() == 0:
                continue
            losses = ce(flat_logits[flat_lm], flat_y[flat_lm])
            # Ponderación 1/t del ELBO simplificado (por ejemplo del batch)
            t_per_tok = ts.unsqueeze(1).expand_as(lm)[lm.bool()] if False else None
            # (aplicamos 1/t a nivel de ejemplo, expandido por posiciones enmascaradas)
            t_expand = ts.view(-1, 1).expand(lm.size(0), lm.size(1)).reshape(-1)[flat_lm]
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
    model = _build_model(cfg['n_token'], d_model=cfg['d_model'], n_layer=cfg['n_layer'],
                          n_head=cfg['n_head'], max_len=cfg['max_seq']).to(device)
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


def sample_block(model, history_tokens, cond_id, null_cond_id, canvas_len,
                  mask_id, pad_id, n_token, device, steps=20, guidance_scale=3.0,
                  remask_max=0.15, temperature=1.0, topk=0, rng=None):
    """
    Genera un compás (canvas_len tokens) mediante denoising iterativo con
    remasking (ReMDM) y D-CFG.  Devuelve la lista final de ids del canvas.
    """
    import torch
    import torch.nn.functional as F

    rng = rng or random.Random()
    canvas = [mask_id] * canvas_len
    is_masked = [True] * canvas_len

    def build_seq(cond):
        return history_tokens + [cond] + canvas

    hist_len = len(history_tokens)

    for step in range(steps):
        t = 1.0 - step / steps           # nivel de ruido actual (antes de este paso)
        s = 1.0 - (step + 1) / steps     # nivel de ruido tras este paso
        s = max(s, 0.0)

        seq_cond   = torch.tensor([build_seq(cond_id)], device=device)
        seq_uncond = torch.tensor([build_seq(null_cond_id)], device=device)
        with torch.no_grad():
            logits_cond   = model(seq_cond)[0, hist_len + 1:]    # (canvas_len, V)
            if guidance_scale != 0:
                logits_uncond = model(seq_uncond)[0, hist_len + 1:]
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

        # Reverse process: des-enmascarar cada posición aún enmascarada con
        # prob. (alpha_s - alpha_t) / (1 - alpha_t)  [MDLM]
        a_t, a_s = alpha_t(t), alpha_t(s)
        denom = max(1.0 - a_t, 1e-6)
        p_unmask = (a_s - a_t) / denom if a_t < 1.0 else 1.0
        p_unmask = min(max(p_unmask, 0.0), 1.0)

        # Fracción de remasking decreciente (ReMDM): más agresiva al
        # principio, se apaga cerca del final para converger.
        p_remask = remask_max * s

        for j in range(canvas_len):
            if canvas[j] == pad_id and not is_masked[j]:
                continue  # PAD original, nunca se toca
            if is_masked[j]:
                if rng.random() < p_unmask or step == steps - 1:
                    canvas[j] = x0_pred[j]
                    is_masked[j] = False
            else:
                if rng.random() < p_remask:
                    canvas[j] = mask_id
                    is_masked[j] = True

    # Barrido final: cualquier MASK residual se rellena con la predicción x0
    if any(is_masked):
        seq_cond = torch.tensor([build_seq(cond_id)], device=device)
        with torch.no_grad():
            logits = model(seq_cond)[0, hist_len + 1:]
        preds = logits.argmax(dim=-1).tolist()
        for j in range(canvas_len):
            if is_masked[j]:
                canvas[j] = preds[j]

    return canvas


def _generate_common(args, prompt_blocks=None):
    import torch

    torch.manual_seed(args.seed)  # sample_block usa torch.multinomial (RNG global)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = _load_model_for_inference(args.model_dir, device)
    event2word, word2event = load_vocab(cfg['vocab'])
    pad_id, mask_id = _sp(word2event, 'PAD'), _sp(word2event, 'MASK')
    null_cond_id = _sp(word2event, 'NULL_COND')
    canvas_len   = cfg['canvas_len']
    history_bars = cfg['history_bars']
    n_token      = cfg['n_token']

    if args.chords:
        try:
            chord_ids = _parse_chords(args.chords, event2word)
        except ValueError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        chord_ids = [null_cond_id]
    rng = random.Random(args.seed)

    blocks = list(prompt_blocks) if prompt_blocks else []
    n_prompt_bars = len(blocks)

    for i in range(args.bars):
        cond_id = chord_ids[i % len(chord_ids)]
        hist_blocks = blocks[-history_bars:] if history_bars > 0 else []
        history_tokens = []
        for hb in hist_blocks:
            history_tokens.append(hb['cond'] if hb['cond'] is not None else null_cond_id)
            history_tokens.extend(hb['canvas'])

        canvas = sample_block(
            model, history_tokens, cond_id, null_cond_id, canvas_len,
            mask_id, pad_id, n_token, device,
            steps=args.steps, guidance_scale=args.guidance_scale,
            remask_max=args.remask_max, temperature=args.temperature,
            topk=args.topk, rng=rng)

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
        print(f"  Compás {i+1}/{args.bars} generado (acorde objetivo: "
              f"{word2event[cond_id]}){tag}")

    words = blocks_to_words(blocks, word2event)
    return words, word2event, n_prompt_bars


def cmd_generate(args):
    print("═" * 65)
    print("  REMI DIFFUSION — GENERATE")
    print("═" * 65)
    words, word2event, _ = _generate_common(args)
    output = args.output or 'generated.mid'
    write_midi(words, word2event, output)
    print(f"\n  → {output}")
    print("═" * 65)


def cmd_continue(args):
    print("═" * 65)
    print("  REMI DIFFUSION — CONTINUE")
    print("═" * 65)
    device = 'cuda'
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
    print("  REMI DIFFUSION — INSPECT")
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
    p = sub.add_parser('train', help='Entrenar el Transformer de difusión enmascarada',
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description=textwrap.dedent("""\
                            Entrena el modelo de block diffusion sobre un corpus de MIDIs.
                            Cada paso: elige una pieza y un compás al azar, arma la historia
                            (compases previos, limpios) + condición de acorde (con dropout
                            para D-CFG) + canvas objetivo enmascarado con t~U(0,1); pérdida
                            de cross-entropy ponderada por 1/t (ELBO simplificado de MDLM).
                        """))
    p.add_argument('data_dir', help='Directorio con MIDIs de entrenamiento')
    p.add_argument('--model-dir', required=True, dest='model_dir')
    p.add_argument('--vocab', default=None)
    p.add_argument('--canvas-len', type=int, default=CANVAS_LEN_DEFAULT, dest='canvas_len')
    p.add_argument('--history-bars', type=int, default=HISTORY_BARS_DEFAULT, dest='history_bars')
    p.add_argument('--d-model', type=int, default=512, dest='d_model')
    p.add_argument('--n-layer', type=int, default=8, dest='n_layer')
    p.add_argument('--n-head', type=int, default=8, dest='n_head')
    p.add_argument('--cond-dropout', type=float, default=0.15, dest='cond_dropout',
                   help='Prob. de sustituir el acorde por NULL_COND (CFG training) [0.15]')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--small', action='store_true',
                   help='Modelo reducido: 4 capas, d=256 (CPU-friendly)')
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
