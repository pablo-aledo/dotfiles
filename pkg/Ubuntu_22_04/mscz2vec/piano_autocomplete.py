#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PIANO AUTOCOMPLETE  v1.0                                 ║
║   Autocompletado de interpretaciones de piano en tiempo real                 ║
║   (inspirado en "Training a 125M-parameter Model to Autocomplete Piano")     ║
║                                                                              ║
║  Transformer decoder-only pequeño (RMSNorm + atención causal + SwiGLU)       ║
║  con una representación NOTE(pitch, delta_onset, duration, velocity):        ║
║  una nota = un paso autoregresivo (no un token por campo). Cada nota tiene   ║
║  cinco campos categóricos [event_type, pitch, delta, duration, velocity]     ║
║  con su propio embedding; el token de la nota es la SUMA de los cinco.       ║
║  Un pequeño "nested decoder" predice los campos en cadena (event→pitch→      ║
║  delta→duration→velocity), condicionando cada campo en los anteriores.       ║
║                                                                              ║
║  El pedal de sustain NO se modela como evento aparte: se "hornea" en la      ║
║  duración de la nota durante el preprocesado (nota se extiende hasta que     ║
║  se suelta el pedal o se retriggerea el mismo pitch).                        ║
║                                                                              ║
║  Incluye post-entrenamiento por preferencias (DPO) usando comparaciones      ║
║  por pares de continuaciones, con métricas heurísticas locales como juez     ║
║  automático (sustituto sin conexión externa del "Gemini pairwise judge"      ║
║  del post original).                                                         ║
║                                                                              ║
║  Nota de diseño: usa embeddings posicionales sinusoidales (no RoPE) y        ║
║  atención causal estándar (no relativa) para mantener el fichero simple y    ║
║  autocontenido — una simplificación deliberada frente al post original.      ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    preprocess   — MIDI corpus → pickles de notas (pedal horneado, splits)    ║
║    train        — Entrenar el modelo (cross-entropy + scheduled sampling)    ║
║    continue     — Autocompletar un MIDI dado (el caso de uso real)           ║
║    score        — Métricas heurísticas de una o dos continuaciones           ║
║    make-pairs   — Generar manifest chosen/rejected para DPO                  ║
║    dpo          — Ajuste por preferencias (Direct Preference Optimization)   ║
║    bench        — Medir velocidad de generación (notas/seg)                  ║
║    inspect      — Diagnóstico de dataset y/o checkpoint                      ║
║                                                                              ║
║  FLUJO TÍPICO:                                                               ║
║    # 1. Preparar datos (pedal horneado, split train/val)                     ║
║    python piano_autocomplete.py preprocess --input midis/ --output data/     ║
║                                                                              ║
║    # 2. Entrenar                                                              ║
║    python piano_autocomplete.py train --data data/ --model-dir runs/exp1/    ║
║                                                                              ║
║    # 3. Autocompletar en vivo                                                 ║
║    python piano_autocomplete.py continue --model-dir runs/exp1/ \            ║
║        --prompt fragmento.mid --notes 64 --output continuado.mid             ║
║                                                                              ║
║    # 4. Preferencias: generar candidatos, puntuarlos, hacer DPO              ║
║    python piano_autocomplete.py continue --model-dir runs/exp1/ \            ║
║        --prompt fragmento.mid --notes 32 --temperature 1.1 \                 ║
║        --output cand_a.mid --seed 1                                          ║
║    python piano_autocomplete.py continue --model-dir runs/exp1/ \            ║
║        --prompt fragmento.mid --notes 32 --temperature 1.1 \                 ║
║        --output cand_b.mid --seed 2                                          ║
║    python piano_autocomplete.py make-pairs --prompt fragmento.mid \          ║
║        --candidates cand_a.mid cand_b.mid --output pairs.json                ║
║    python piano_autocomplete.py dpo --model-dir runs/exp1/ \                 ║
║        --pairs pairs.json --beta 0.03 --output-dir runs/exp1_dpo/            ║
║                                                                              ║
║  OPCIONES PRINCIPALES (train):                                               ║
║    --epochs N          Épocas de entrenamiento (default: 50)                 ║
║    --batch-size N      Tamaño de batch (default: 8)                         ║
║    --max-notes N       Notas por secuencia de entrenamiento (default: 256)   ║
║    --layers N          Capas del transformer (default: 4)                    ║
║    --dim N             Dimensión de embeddings (default: 128)                ║
║    --heads N           Cabezas de atención (default: 4)                      ║
║    --sched-start N     Época en que empieza el scheduled sampling (def: 5)   ║
║    --sched-end N       Época en que alcanza el máximo (default: 25)          ║
║    --sched-max-p F     Probabilidad máxima de auto-alimentación (def: 0.5)   ║
║    --resume            Retomar desde el último checkpoint                    ║
║                                                                              ║
║  OPCIONES PRINCIPALES (continue):                                            ║
║    --notes N            Notas nuevas a generar (default: 64)                 ║
║    --temperature F      Temperatura de muestreo (default: 1.0)               ║
║    --top-k N            Muestreo top-k (default: 0 = desactivado)            ║
║    --top-p F            Muestreo nucleus (default: 0.0 = desactivado)        ║
║    --max-context N      Ventana máx. de notas de contexto (default: 512)     ║
║    --keep-notes N       Notas recientes al reconstruir contexto (def: 384)   ║
║    --allow-random-init  Generar con pesos aleatorios (pruebas de fontanería) ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
python piano_autocomplete.py preprocess --input midis_sinteticos/ --output data/
python piano_autocomplete.py inspect --data data/
python piano_autocomplete.py train --data data/ --model-dir runs/prueba/ \
    --dim 64 --layers 2 --epochs 3 --batch-size 4 --max-notes 64
python piano_autocomplete.py continue --model-dir runs/prueba/ \
    --prompt fragmento.mid --notes 32 --output continuado.mid
"""

import sys
import os
import bisect
import math
import time
import random
import pickle
import argparse
import json
import copy
import itertools
from collections import Counter

import numpy as np

# ── mido (encoder/decoder MIDI nativo) ───────────────────────────────────────
try:
    import mido
    _MIDO_OK = True
except ImportError:
    _MIDO_OK = False

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    class _TorchStub:
        Module = object
        def __getattr__(self, name):
            raise ImportError("PyTorch no disponible. Instala con: pip install torch")
    nn = _TorchStub()
    _TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIOS DE CAMPO  (siguiendo el esquema de campos categóricos del post)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Cada nota = [event_type, pitch, delta_onset, duration, velocity]
#  event_type: PAD, BOS, EOS, NOTE, MASK               →  5 categorías
#  pitch:      0=PAD + 128 pitches MIDI                → 129 categorías
#  delta:      0=PAD + 53 valores (0..48, 72,96,144,192) → 54 categorías
#  duration:   0=PAD + 100 valores (1..96, 144,192,288,384) → 101 categorías
#  velocity:   0=PAD + 16 valores (4,12,...,124)        →  17 categorías
#
#  Timing cuantizado a 24 pasos por negra (igual resolución que el post,
#  suficiente para subdivisiones binarias y de tresillo).

EVENT_PAD, EVENT_BOS, EVENT_EOS, EVENT_NOTE, EVENT_MASK = 0, 1, 2, 3, 4
EVENT_VOCAB = 5

STEPS_PER_QUARTER = 24

DELTA_VALUES    = list(range(0, 49)) + [72, 96, 144, 192]          # 53
DURATION_VALUES = list(range(1, 97)) + [144, 192, 288, 384]        # 100
VELOCITY_VALUES = list(range(4, 125, 8))                           # 16

PITCH_VOCAB    = 1 + 128
DELTA_VOCAB    = 1 + len(DELTA_VALUES)
DURATION_VOCAB = 1 + len(DURATION_VALUES)
VELOCITY_VOCAB = 1 + len(VELOCITY_VALUES)

CHECKPOINT_FNAME = 'checkpoint.pt'
CONFIG_FNAME     = 'config.json'


def _nearest_bucket_idx(value: float, table: list) -> int:
    """Índice (0-based) del valor más cercano en una tabla ordenada."""
    pos = bisect.bisect_left(table, value)
    if pos == 0:
        return 0
    if pos == len(table):
        return len(table) - 1
    before, after = table[pos - 1], table[pos]
    return pos - 1 if (value - before) <= (after - value) else pos


def quantize_ticks(ticks: float, ticks_per_beat: int) -> int:
    """Ticks MIDI → unidades de 1/24 de negra."""
    if ticks_per_beat <= 0:
        ticks_per_beat = 480
    units = round(ticks * STEPS_PER_QUARTER / ticks_per_beat)
    return max(0, int(units))


def unquantize_units(units: int, ticks_per_beat: int) -> int:
    """Unidades de 1/24 de negra → ticks MIDI."""
    return max(0, round(units * ticks_per_beat / STEPS_PER_QUARTER))


def delta_to_id(units: int) -> int:
    return 1 + _nearest_bucket_idx(units, DELTA_VALUES)


def duration_to_id(units: int) -> int:
    units = max(1, units)
    return 1 + _nearest_bucket_idx(units, DURATION_VALUES)


def velocity_to_id(vel: int) -> int:
    vel = max(1, min(127, vel))
    return 1 + _nearest_bucket_idx(vel, VELOCITY_VALUES)


def id_to_delta(idx: int) -> int:
    return DELTA_VALUES[idx - 1] if idx > 0 else 0


def id_to_duration(idx: int) -> int:
    return DURATION_VALUES[idx - 1] if idx > 0 else STEPS_PER_QUARTER // 4


def id_to_velocity(idx: int) -> int:
    return VELOCITY_VALUES[idx - 1] if idx > 0 else 64


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI IN  —  fusión de tracks + horneado del pedal de sustain
# ══════════════════════════════════════════════════════════════════════════════

def bake_sustain_notes(path: str) -> list:
    """
    MIDI file → lista de (pitch, onset_tick_abs, duration_ticks, velocity)
    ordenada por (onset_tick, pitch), con el pedal de sustain horneado en
    la duración: si se suelta la tecla con el pedal pisado, la nota se
    extiende hasta que el pedal sube (o hasta que el mismo pitch se
    retriggerea, lo que ocurra antes).
    """
    mid = mido.MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat or 480

    merged = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            merged.append((abs_tick, msg))
    merged.sort(key=lambda x: x[0])

    pedal_down = False
    active = {}            # pitch -> (onset_tick, velocity)
    pending_release = {}   # pitch -> (onset_tick, velocity)   soltado con pedal abajo
    notes = []             # (pitch, onset_tick, duration_ticks, velocity)

    def _finalize(pitch, onset, velocity, off_tick):
        dur = max(1, off_tick - onset)
        notes.append((pitch, onset, dur, velocity))

    for abs_tick, msg in merged:
        if msg.type == 'control_change' and msg.control == 64:
            was_down = pedal_down
            pedal_down = msg.value >= 64
            if was_down and not pedal_down:
                # pedal levantado: cerrar todas las notas pendientes
                for pitch, (onset, vel) in list(pending_release.items()):
                    _finalize(pitch, onset, vel, abs_tick)
                    del pending_release[pitch]

        elif msg.type == 'note_on' and msg.velocity > 0:
            pitch = msg.note
            # si había una nota activa o pendiente del mismo pitch, cortarla
            if pitch in active:
                onset, vel = active.pop(pitch)
                _finalize(pitch, onset, vel, abs_tick)
            if pitch in pending_release:
                onset, vel = pending_release.pop(pitch)
                _finalize(pitch, onset, vel, abs_tick)
            active[pitch] = (abs_tick, msg.velocity)

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            pitch = msg.note
            if pitch in active:
                onset, vel = active.pop(pitch)
                if pedal_down:
                    pending_release[pitch] = (onset, vel)
                else:
                    _finalize(pitch, onset, vel, abs_tick)

    # cerrar cualquier nota que quedó sonando al final del fichero
    end_tick = merged[-1][0] if merged else 0
    for pitch, (onset, vel) in list(active.items()):
        _finalize(pitch, onset, vel, max(end_tick, onset + 1))
    for pitch, (onset, vel) in list(pending_release.items()):
        _finalize(pitch, onset, vel, max(end_tick, onset + 1))

    notes.sort(key=lambda n: (n[1], n[0]))
    return notes, ticks_per_beat


def notes_to_fields(notes: list, ticks_per_beat: int) -> dict:
    """Notas crudas (ticks) → arrays de campos categóricos cuantizados."""
    pitch_ids, delta_ids, dur_ids, vel_ids = [], [], [], []
    prev_onset = None
    for pitch, onset, dur, vel in notes:
        delta_ticks = 0 if prev_onset is None else max(0, onset - prev_onset)
        prev_onset = onset
        pitch_ids.append(pitch + 1)
        delta_ids.append(delta_to_id(quantize_ticks(delta_ticks, ticks_per_beat)))
        dur_ids.append(duration_to_id(quantize_ticks(dur, ticks_per_beat)))
        vel_ids.append(velocity_to_id(vel))
    return {
        'pitch':    np.array(pitch_ids, dtype=np.int32),
        'delta':    np.array(delta_ids, dtype=np.int32),
        'duration': np.array(dur_ids, dtype=np.int32),
        'velocity': np.array(vel_ids, dtype=np.int32),
    }


def fields_to_notes(fields: dict, ticks_per_beat: int) -> list:
    """Arrays de campos categóricos → notas (pitch, onset_tick, duration_ticks, velocity)."""
    notes = []
    onset = 0
    n = len(fields['pitch'])
    for i in range(n):
        pitch = int(fields['pitch'][i]) - 1
        delta_units = id_to_delta(int(fields['delta'][i]))
        dur_units   = id_to_duration(int(fields['duration'][i]))
        vel         = id_to_velocity(int(fields['velocity'][i]))
        if i > 0:
            onset += unquantize_units(delta_units, ticks_per_beat)
        dur_ticks = unquantize_units(dur_units, ticks_per_beat)
        if pitch < 0 or pitch > 127:
            continue
        notes.append((pitch, onset, max(1, dur_ticks), max(1, vel)))
    return notes


def decode_midi(notes: list, file_path: str, ticks_per_beat: int = 480,
                 tempo: int = 500_000):
    """Notas (pitch, onset_tick, duration_ticks, velocity) → fichero MIDI."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    events = []
    for pitch, onset, dur, vel in notes:
        events.append((onset, 1, pitch, vel))          # note_on  (prioridad 1)
        events.append((onset + dur, 0, pitch, 0))       # note_off (prioridad 0, va antes si empata)
    events.sort(key=lambda e: (e[0], e[1]))

    last_tick = 0
    for abs_tick, kind, pitch, vel in events:
        delta = max(0, abs_tick - last_tick)
        last_tick = abs_tick
        if kind == 1:
            track.append(mido.Message('note_on', note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))

    mid.save(file_path)


# ══════════════════════════════════════════════════════════════════════════════
#  AUMENTACIÓN  (transposición, tempo, jitter, dropped prompt notes)
# ══════════════════════════════════════════════════════════════════════════════

def augment_transpose(notes: list, semitones: int) -> list:
    out = []
    for pitch, onset, dur, vel in notes:
        p = pitch + semitones
        if 0 <= p <= 127:
            out.append((p, onset, dur, vel))
    return out


def augment_tempo(notes: list, factor: float) -> list:
    return [(p, round(o * factor), max(1, round(d * factor)), v) for p, o, d, v in notes]


def augment_jitter(notes: list, dur_jitter: float, vel_jitter: int, rng: random.Random) -> list:
    out = []
    for p, o, d, v in notes:
        d2 = max(1, round(d * (1 + rng.uniform(-dur_jitter, dur_jitter))))
        v2 = max(1, min(127, v + rng.randint(-vel_jitter, vel_jitter)))
        out.append((p, o, d2, v2))
    return out


def augment_drop_prompt(notes: list, n_drop: int) -> list:
    return notes[n_drop:] if n_drop < len(notes) else notes


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (pickles por fichero, splits train/val — mismo patrón que
#            music_transformer.py y remi.py del ecosistema)
# ══════════════════════════════════════════════════════════════════════════════

class NoteDataset:
    def __init__(self, data_dir: str, seed: int = 42):
        self.splits = {}
        for split in ('train', 'val'):
            d = os.path.join(data_dir, split)
            if os.path.isdir(d):
                self.splits[split] = sorted(
                    os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pickle')
                )
            else:
                self.splits[split] = []
        if not self.splits['train']:
            raise FileNotFoundError(f"No se encontraron ficheros .pickle en {data_dir}/train/")
        self.rng = random.Random(seed)

    def __repr__(self):
        return (f"<NoteDataset train={len(self.splits['train'])} "
                f"val={len(self.splits['val'])} ficheros>")

    def _load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def batch(self, batch_size: int, max_notes: int, split: str = 'train') -> list:
        pool = self.splits[split] or self.splits['train']
        chosen = self.rng.sample(pool, k=min(batch_size, len(pool)))
        items = []
        for path in chosen:
            data = self._load(path)
            n = len(data['pitch'])
            if n <= 1:
                continue
            length = min(n, max_notes)
            start = self.rng.randrange(0, max(1, n - length + 1))
            items.append({k: data[k][start:start + length] for k in
                          ('pitch', 'delta', 'duration', 'velocity')})
        while len(items) < batch_size and items:
            items.append(self.rng.choice(items))
        return items

    def n_batches(self, batch_size: int, split: str = 'train') -> int:
        return max(1, len(self.splits[split] or self.splits['train']) // batch_size)


def build_token_batch(items: list, device):
    """
    Lista de dicts de campos (notas crudas, longitudes variables) →
    tensores (B, T+1) con BOS al principio y padding a la derecha,
    más una máscara de padding. T = longitud máxima del batch.
    """
    lengths = [len(it['pitch']) for it in items]
    max_len = max(lengths) + 1  # +1 por el BOS
    B = len(items)

    event = torch.full((B, max_len), EVENT_PAD, dtype=torch.long)
    pitch = torch.zeros((B, max_len), dtype=torch.long)
    delta = torch.zeros((B, max_len), dtype=torch.long)
    dur   = torch.zeros((B, max_len), dtype=torch.long)
    vel   = torch.zeros((B, max_len), dtype=torch.long)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)  # True = padding

    for i, it in enumerate(items):
        n = len(it['pitch'])
        event[i, 0] = EVENT_BOS
        event[i, 1:1 + n] = EVENT_NOTE
        pitch[i, 1:1 + n] = torch.from_numpy(it['pitch'].astype(np.int64))
        delta[i, 1:1 + n] = torch.from_numpy(it['delta'].astype(np.int64))
        dur[i, 1:1 + n]   = torch.from_numpy(it['duration'].astype(np.int64))
        vel[i, 1:1 + n]   = torch.from_numpy(it['velocity'].astype(np.int64))
        pad_mask[i, :1 + n] = False

    return {
        'event': event.to(device), 'pitch': pitch.to(device), 'delta': delta.to(device),
        'duration': dur.to(device), 'velocity': vel.to(device),
        'pad_mask': pad_mask.to(device),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO  —  RMSNorm + atención causal + SwiGLU + nested field decoder
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, pad_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # (B, h, T, dh)

        # Máscara causal (True = permitido) combinada con el padding de las claves.
        causal_allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allowed = causal_allowed[None, None, :, :].expand(B, self.h, T, T)
        if pad_mask is not None:
            key_ok = (~pad_mask)[:, None, None, :].expand(B, self.h, T, T)
            allowed = allowed & key_ok

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            is_causal=False, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.up   = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp   = SwiGLU(dim, dim * 4)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


def _sinusoidal_pe(max_len: int, dim: int) -> 'torch.Tensor':
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


class FieldHead(nn.Module):
    def __init__(self, dim: int, vocab: int, hidden: int = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, vocab))

    def forward(self, x):
        return self.net(x)


class PianoAutocompleteModel(nn.Module):
    """
    note = sum(event_emb, pitch_emb, delta_emb, duration_emb, velocity_emb)
    backbone: N bloques causales (RMSNorm + atención + SwiGLU)
    nested heads: event → pitch → delta → duration → velocity, cada campo
    condicionado (por suma) en los embeddings de los campos ya decididos.
    """

    def __init__(self, dim: int = 128, layers: int = 4, heads: int = 4,
                 dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dim, self.layers_n, self.heads, self.max_len = dim, layers, heads, max_len

        self.emb_event = nn.Embedding(EVENT_VOCAB, dim, padding_idx=EVENT_PAD)
        self.emb_pitch = nn.Embedding(PITCH_VOCAB, dim, padding_idx=0)
        self.emb_delta = nn.Embedding(DELTA_VOCAB, dim, padding_idx=0)
        self.emb_dur   = nn.Embedding(DURATION_VOCAB, dim, padding_idx=0)
        self.emb_vel   = nn.Embedding(VELOCITY_VOCAB, dim, padding_idx=0)

        self.register_buffer('pe', _sinusoidal_pe(max_len, dim), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm_out = RMSNorm(dim)

        self.head_event = FieldHead(dim, EVENT_VOCAB)
        self.head_pitch = FieldHead(dim, PITCH_VOCAB)
        self.head_delta = FieldHead(dim, DELTA_VOCAB)
        self.head_dur   = FieldHead(dim, DURATION_VOCAB)
        self.head_vel   = FieldHead(dim, VELOCITY_VOCAB)

    def embed_tokens(self, batch: dict):
        x = (self.emb_event(batch['event']) + self.emb_pitch(batch['pitch']) +
             self.emb_delta(batch['delta']) + self.emb_dur(batch['duration']) +
             self.emb_vel(batch['velocity']))
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)

    def backbone(self, x, pad_mask=None):
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.norm_out(x)

    def forward_train(self, batch: dict, sched_p: float = 0.0):
        """
        Teacher forcing con scheduled sampling opcional SOLO en el pitch
        (igual que el post): con probabilidad sched_p, el pitch usado para
        condicionar delta/duration/velocity es el predicho por el propio
        modelo en vez del real.
        Devuelve logits por campo, alineados para predecir el token t+1
        desde la posición t (estándar next-token).
        """
        h = self.backbone(self.embed_tokens(batch), batch['pad_mask'])
        h = h[:, :-1, :]  # posiciones 0..T-2 predicen tokens 1..T-1

        event_logits = self.head_event(h)
        event_true = batch['event'][:, 1:]
        h2 = h + self.emb_event(event_true)

        pitch_logits = self.head_pitch(h2)
        pitch_true = batch['pitch'][:, 1:]
        if sched_p > 0:
            with torch.no_grad():
                pitch_pred = torch.multinomial(
                    F.softmax(pitch_logits.reshape(-1, PITCH_VOCAB), dim=-1), 1
                ).view(pitch_true.shape)
            use_pred = (torch.rand_like(pitch_true, dtype=torch.float) < sched_p)
            pitch_cond = torch.where(use_pred, pitch_pred, pitch_true)
        else:
            pitch_cond = pitch_true
        h3 = h2 + self.emb_pitch(pitch_cond)

        delta_logits = self.head_delta(h3)
        delta_true = batch['delta'][:, 1:]
        h4 = h3 + self.emb_delta(delta_true)

        dur_logits = self.head_dur(h4)
        dur_true = batch['duration'][:, 1:]
        h5 = h4 + self.emb_dur(dur_true)

        vel_logits = self.head_vel(h5)

        return {
            'event': event_logits, 'pitch': pitch_logits, 'delta': delta_logits,
            'duration': dur_logits, 'velocity': vel_logits,
        }

    def sequence_logprob(self, batch: dict, start_index: int) -> 'torch.Tensor':
        """
        Log-prob teacher-forced (sin scheduled sampling) sumada sobre los
        cinco campos y sobre las posiciones >= start_index (la parte de
        continuación, excluyendo el prompt). Usado por DPO. batch tiene B=1.
        """
        out = self.forward_train(batch, sched_p=0.0)
        targets = {
            'event': batch['event'][:, 1:], 'pitch': batch['pitch'][:, 1:],
            'delta': batch['delta'][:, 1:], 'duration': batch['duration'][:, 1:],
            'velocity': batch['velocity'][:, 1:],
        }
        pad_mask = batch['pad_mask'][:, 1:]
        # posiciones de continuación: índice de token (1-based en la secuencia
        # original) >= start_index, y no-padding
        T = targets['event'].shape[1]
        idx = torch.arange(T, device=targets['event'].device)
        cont_mask = (idx >= (start_index - 1)) & (~pad_mask[0])

        total = 0.0
        for field, vocab in (('event', EVENT_VOCAB), ('pitch', PITCH_VOCAB),
                              ('delta', DELTA_VOCAB), ('duration', DURATION_VOCAB),
                              ('velocity', VELOCITY_VOCAB)):
            logp = F.log_softmax(out[field], dim=-1)
            tgt = targets[field]
            picked = torch.gather(logp, 2, tgt.unsqueeze(-1)).squeeze(-1)
            total = total + (picked[0] * cont_mask.float()).sum()
        return total

    @torch.no_grad()
    def generate(self, prompt_fields: dict, n_notes: int, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 0.0, max_context: int = 512,
                 keep_notes: int = 384, device=None, rng: 'torch.Generator' = None):
        """
        Autocompleta n_notes notas nuevas a partir de prompt_fields (dict de
        arrays de campos). Ventana deslizante: si el contexto excede
        max_context notas, se recorta a las últimas keep_notes antes de
        seguir generando (igual que el "rebuild the context" del post).
        """
        self.eval()
        device = device or next(self.parameters()).device

        # El buffer de posición sinusoidal solo cubre self.max_len posiciones
        # (BOS + notas). Si max_context/keep_notes piden más de lo que el
        # modelo puede indexar, los acotamos aquí en vez de fallar a mitad
        # de generación con un error de tensores poco claro.
        hard_cap = self.max_len - 1
        if max_context > hard_cap:
            max_context = hard_cap
        if keep_notes >= max_context:
            keep_notes = max(1, max_context - 1)

        seq = {
            'pitch': list(prompt_fields['pitch']), 'delta': list(prompt_fields['delta']),
            'duration': list(prompt_fields['duration']), 'velocity': list(prompt_fields['velocity']),
        }
        generated = {'pitch': [], 'delta': [], 'duration': [], 'velocity': []}

        def _sample(logits):
            logits = logits / max(temperature, 1e-8)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[..., -1, None]] = -float('inf')
            if top_p and top_p > 0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum - probs > top_p
                sorted_logits[remove] = -float('inf')
                logits = torch.full_like(logits, -float('inf')).scatter(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1, generator=rng).item()

        for _ in range(n_notes):
            # ventana deslizante
            if len(seq['pitch']) > max_context:
                seq = {k: v[-keep_notes:] for k, v in seq.items()}

            n = len(seq['pitch'])
            batch = {
                'event': torch.full((1, n + 1), EVENT_NOTE, dtype=torch.long, device=device),
                'pitch': torch.zeros((1, n + 1), dtype=torch.long, device=device),
                'delta': torch.zeros((1, n + 1), dtype=torch.long, device=device),
                'duration': torch.zeros((1, n + 1), dtype=torch.long, device=device),
                'velocity': torch.zeros((1, n + 1), dtype=torch.long, device=device),
                'pad_mask': torch.zeros((1, n + 1), dtype=torch.bool, device=device),
            }
            batch['event'][0, 0] = EVENT_BOS
            if n > 0:
                batch['pitch'][0, 1:] = torch.tensor(seq['pitch'], device=device)
                batch['delta'][0, 1:] = torch.tensor(seq['delta'], device=device)
                batch['duration'][0, 1:] = torch.tensor(seq['duration'], device=device)
                batch['velocity'][0, 1:] = torch.tensor(seq['velocity'], device=device)

            h_full = self.backbone(self.embed_tokens(batch), batch['pad_mask'])
            h = h_full[:, -1, :]  # última posición → predice la siguiente nota

            event_id = _sample(self.head_event(h).clone())
            h2 = h + self.emb_event(torch.tensor([event_id], device=device))
            if event_id == EVENT_EOS:
                break

            pitch_id = _sample(self.head_pitch(h2).clone())
            h3 = h2 + self.emb_pitch(torch.tensor([pitch_id], device=device))

            delta_id = _sample(self.head_delta(h3).clone())
            h4 = h3 + self.emb_delta(torch.tensor([delta_id], device=device))

            dur_id = _sample(self.head_dur(h4).clone())
            h5 = h4 + self.emb_dur(torch.tensor([dur_id], device=device))

            vel_id = _sample(self.head_vel(h5).clone())

            for k, v in zip(('pitch', 'delta', 'duration', 'velocity'),
                             (pitch_id, delta_id, dur_id, vel_id)):
                seq[k].append(v)
                generated[k].append(v)

        return {k: np.array(v, dtype=np.int32) for k, v in generated.items()}

    def config_dict(self) -> dict:
        return {'dim': self.dim, 'layers': self.layers_n, 'heads': self.heads,
                'max_len': self.max_len}

    @classmethod
    def from_config(cls, cfg: dict, dropout: float = 0.0) -> 'PianoAutocompleteModel':
        return cls(dim=cfg['dim'], layers=cfg['layers'], heads=cfg['heads'],
                    dropout=dropout, max_len=cfg.get('max_len', 2048))


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDS DE DEPENDENCIAS Y CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _require_mido():
    if not _MIDO_OK:
        print("ERROR: mido no disponible. Instala con:\n  pip install mido")
        sys.exit(1)


def _require_torch():
    if not _TORCH_OK:
        print("ERROR: PyTorch no disponible. Instala con:\n  pip install torch")
        sys.exit(1)


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _save_checkpoint(model_dir, model, optimizer, epoch, best_loss, cfg):
    os.makedirs(model_dir, exist_ok=True)
    torch.save({'epoch': epoch, 'best_loss': best_loss, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(model_dir, CHECKPOINT_FNAME))
    with open(os.path.join(model_dir, CONFIG_FNAME), 'w') as f:
        json.dump(cfg, f, indent=2)


def _load_config(model_dir):
    path = os.path.join(model_dir, CONFIG_FNAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {CONFIG_FNAME} en {model_dir}")
    with open(path) as f:
        return json.load(f)


def _load_model(model_dir, device, dropout=0.0):
    cfg = _load_config(model_dir)
    model = PianoAutocompleteModel.from_config(cfg, dropout=dropout).to(device)
    ckpt_path = os.path.join(model_dir, CHECKPOINT_FNAME)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return model, cfg, ckpt


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: preprocess
# ══════════════════════════════════════════════════════════════════════════════

def cmd_preprocess(args):
    _require_mido()
    midi_exts = ('.mid', '.midi')
    midi_files = [os.path.join(root, f) for root, _, files in os.walk(args.input)
                  for f in files if f.lower().endswith(midi_exts)]
    if not midi_files:
        print(f"ERROR: no se encontraron ficheros MIDI en {args.input}")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(midi_files)
    n_val = max(1, int(len(midi_files) * args.val_split)) if len(midi_files) > 1 else 0
    val_set = set(midi_files[:n_val])

    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.output, split), exist_ok=True)

    ok = errors = skipped = 0
    for path in midi_files:
        split = 'val' if path in val_set else 'train'
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.output, split, name + '.pickle')
        try:
            notes, tpb = bake_sustain_notes(path)
            if len(notes) < args.min_notes:
                skipped += 1
                if args.verbose:
                    print(f"  ⋯  {path}: solo {len(notes)} notas, se omite")
                continue

            variants = [notes]
            if args.augment:
                for semis in (-2, -1, 1, 2):
                    variants.append(augment_transpose(notes, semis))
                aug_rng = random.Random(hash(path) & 0xffffffff)
                variants.append(augment_tempo(notes, aug_rng.uniform(0.9, 1.1)))
                variants.append(augment_jitter(notes, 0.05, 6, aug_rng))

            for vi, var_notes in enumerate(variants):
                if not var_notes:
                    continue
                fields = notes_to_fields(var_notes, tpb)
                out_path = out if vi == 0 else os.path.join(
                    args.output, split, f"{name}__aug{vi}.pickle")
                fields['source'] = path
                fields['ticks_per_beat'] = tpb
                with open(out_path, 'wb') as f:
                    pickle.dump(fields, f)
            ok += 1
            if args.verbose:
                print(f"  ✓  {path}  →  {len(notes)} notas [{split}]"
                      + (f" (+{len(variants)-1} aug)" if args.augment else ""))
        except KeyboardInterrupt:
            print("\nInterrumpido.")
            break
        except Exception as e:
            errors += 1
            print(f"  ✗  {path}: {e}")

    print(f"\nPreprocess completado: {ok} OK, {skipped} omitidos, {errors} errores "
          f"→ {args.output}/{{train,val}}/")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def _sched_p(epoch: int, start: int, end: int, max_p: float) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return max_p
    return max_p * (epoch - start) / max(1, end - start)


def _field_loss(logits, target, pad_mask):
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction='none')
    keep = (~pad_mask).reshape(-1).float()
    return (loss * keep).sum() / keep.sum().clamp(min=1)


def _field_acc(logits, target, pad_mask):
    pred = logits.argmax(-1)
    keep = ~pad_mask
    correct = ((pred == target) & keep).sum().item()
    total = keep.sum().item()
    return correct / max(1, total)


def cmd_train(args):
    _require_torch()
    device = _get_device()
    print(f"Dispositivo: {device}")

    dataset = NoteDataset(args.data, seed=args.seed)
    print(dataset)

    cfg = {'dim': args.dim, 'layers': args.layers, 'heads': args.heads,
           'max_len': args.max_notes + 8}

    model = PianoAutocompleteModel.from_config(cfg, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params:,} parámetros  (dim={args.dim}, layers={args.layers}, "
          f"heads={args.heads})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch, best_loss = 0, float('inf')

    if args.resume:
        ckpt_path = os.path.join(args.model_dir, CHECKPOINT_FNAME)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            print(f"Retomando desde época {start_epoch} (best_loss={best_loss:.4f})")

    n_batches = dataset.n_batches(args.batch_size, 'train')
    fields = ('event', 'pitch', 'delta', 'duration', 'velocity')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        p = _sched_p(epoch, args.sched_start, args.sched_end, args.sched_max_p)
        epoch_loss, epoch_acc = {f: 0.0 for f in fields}, {f: 0.0 for f in fields}

        for _ in range(n_batches):
            items = dataset.batch(args.batch_size, args.max_notes, 'train')
            batch = build_token_batch(items, device)
            out = model.forward_train(batch, sched_p=p)
            pad_mask = batch['pad_mask'][:, 1:]
            targets = {f: batch[f][:, 1:] for f in fields}

            loss = 0.0
            for f in fields:
                fl = _field_loss(out[f], targets[f], pad_mask)
                loss = loss + fl
                epoch_loss[f] += fl.item()
                epoch_acc[f] += _field_acc(out[f], targets[f], pad_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss = sum(epoch_loss.values()) / n_batches
        print(f"[época {epoch:3d}] sched_p={p:.2f}  loss_total={total_loss:.4f}  " +
              "  ".join(f"{f}={epoch_loss[f]/n_batches:.3f}/{epoch_acc[f]/n_batches:.2%}"
                        for f in fields))

        if args.eval_every and dataset.splits['val'] and (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                items = dataset.batch(args.batch_size, args.max_notes, 'val')
                batch = build_token_batch(items, device)
                out = model.forward_train(batch, sched_p=0.0)
                pad_mask = batch['pad_mask'][:, 1:]
                val_loss = sum(_field_loss(out[f], batch[f][:, 1:], pad_mask).item()
                                for f in fields)
                print(f"           val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss

        _save_checkpoint(args.model_dir, model, optimizer, epoch, best_loss, cfg)

    print(f"\nEntrenamiento completado → {args.model_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: continue  (autocompletar)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_continue(args):
    _require_mido()
    _require_torch()
    device = _get_device()

    notes, tpb = bake_sustain_notes(args.prompt)
    if not notes:
        print(f"ERROR: {args.prompt} no contiene notas.")
        sys.exit(1)
    prompt_fields = notes_to_fields(notes, tpb)
    n_prompt = len(prompt_fields['pitch'])

    if args.model_dir:
        model, cfg, _ = _load_model(args.model_dir, device)
        print(f"Modelo cargado desde {args.model_dir}  ({cfg})")
    elif args.allow_random_init:
        cfg = {'dim': args.dim, 'layers': args.layers, 'heads': args.heads,
               'max_len': max(2048, n_prompt + args.notes + 8)}
        model = PianoAutocompleteModel.from_config(cfg).to(device)
        print("⚠  Sin --model-dir: usando pesos ALEATORIOS (--allow-random-init). "
              "El resultado es estructuralmente válido pero musicalmente sin sentido; "
              "solo sirve para verificar la fontanería del pipeline.")
    else:
        print("ERROR: indica --model-dir o --allow-random-init.")
        sys.exit(1)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    t0 = time.time()
    generated = model.generate(
        prompt_fields, n_notes=args.notes, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, max_context=args.max_context,
        keep_notes=args.keep_notes, device=device, rng=rng,
    )
    elapsed = time.time() - t0
    n_gen = len(generated['pitch'])

    full_fields = {k: np.concatenate([prompt_fields[k], generated[k]])
                   for k in ('pitch', 'delta', 'duration', 'velocity')}
    full_notes = fields_to_notes(full_fields, tpb)
    decode_midi(full_notes, args.output, ticks_per_beat=tpb)

    rate = n_gen / elapsed if elapsed > 0 else float('inf')
    print(f"\nPrompt:      {n_prompt} notas ({args.prompt})")
    print(f"Generadas:   {n_gen} notas en {elapsed:.2f}s  ({rate:.1f} notas/seg)")
    print(f"Guardado en: {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: score  —  métricas heurísticas locales (sustituto sin conexión
#  externa del juez pairwise por LLM que usa el post original)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(notes: list) -> dict:
    if not notes:
        return {'n_notes': 0}

    notes = sorted(notes, key=lambda n: n[1])
    pitches = [n[0] for n in notes]
    onsets = [n[1] for n in notes]
    durations = [n[2] for n in notes]

    span = max(onsets) - min(onsets) if len(onsets) > 1 else 1
    density = len(notes) / max(1, span) * STEPS_PER_QUARTER  # notas por negra aprox.

    # entropía de pitch y de clase de pitch
    def entropy(seq):
        c = Counter(seq)
        n = len(seq)
        return -sum((v / n) * math.log2(v / n) for v in c.values())

    pitch_entropy = entropy(pitches)
    pc_entropy = entropy([p % 12 for p in pitches])

    # n-gramas de pitch repetidos (trigramas)
    trigrams = [tuple(pitches[i:i + 3]) for i in range(len(pitches) - 2)]
    repeated_trigram_ratio = 0.0
    if trigrams:
        c = Counter(trigrams)
        repeated = sum(v for v in c.values() if v > 1)
        repeated_trigram_ratio = repeated / len(trigrams)

    # pausas largas: deltas de onset por encima de 2 negras
    long_pauses = 0
    for i in range(1, len(onsets)):
        gap = onsets[i] - onsets[i - 1]
        if gap > STEPS_PER_QUARTER * 2 * (480 // STEPS_PER_QUARTER) * 0 + 2 * STEPS_PER_QUARTER:
            pass  # placeholder para claridad; el cálculo real está abajo con ticks reales
    # (se calcula en unidades ya cuantizadas al puntuar fields, ver score_from_fields)

    chord_groups = Counter()
    for o in onsets:
        chord_groups[o] += 1
    chord_notes = sum(v for v in chord_groups.values() if v > 1)
    chord_density = chord_notes / len(notes)

    return {
        'n_notes': len(notes),
        'pitch_range': max(pitches) - min(pitches),
        'pitch_entropy': pitch_entropy,
        'pitch_class_entropy': pc_entropy,
        'note_density': density,
        'repeated_trigram_ratio': repeated_trigram_ratio,
        'chord_density': chord_density,
    }


def score_from_fields(fields: dict) -> dict:
    """Métricas + pausas largas calculadas directamente sobre campos cuantizados."""
    n = len(fields.get('pitch', []))
    if n == 0:
        return {'n_notes': 0}
    pitches = [p - 1 for p in fields['pitch']]
    deltas_units = [id_to_delta(d) for d in fields['delta']]
    long_pause_ratio = (sum(1 for d in deltas_units if d > STEPS_PER_QUARTER * 2)
                         / max(1, n - 1))
    base = compute_metrics([(pitches[i], sum(deltas_units[:i + 1]),
                              id_to_duration(fields['duration'][i]),
                              id_to_velocity(fields['velocity'][i]))
                             for i in range(n)])
    base['long_pause_ratio'] = long_pause_ratio
    return base


def aggregate_score(m: dict) -> float:
    """
    Escalar heurístico en [0,1] combinando las métricas. Penaliza repetición
    excesiva, pausas largas y rangos de pitch extremos; premia entropía
    moderada y densidad razonable. Pesos elegidos a mano, no aprendidos.
    """
    if m.get('n_notes', 0) < 2:
        return 0.0
    score = 1.0
    score -= 0.4 * m.get('repeated_trigram_ratio', 0.0)
    score -= 0.3 * m.get('long_pause_ratio', 0.0)
    pr = m.get('pitch_range', 0)
    if pr > 48:
        score -= 0.15
    if pr < 3:
        score -= 0.25
    pe = m.get('pitch_entropy', 0.0)
    score -= 0.2 * max(0.0, 1.0 - pe / 3.0)  # penaliza entropía muy baja (monotonía)
    dens = m.get('note_density', 0.0)
    if dens > 12 or dens < 0.3:
        score -= 0.15
    return max(0.0, min(1.0, score))


def continuity_bonus(prompt_fields: dict, cont_fields: dict, window: int = 8) -> float:
    """
    Similitud de registro entre el final del prompt y el arranque de la
    continuación (proxy simple de "sigue del prompt", análogo al
    continuation-score del post frente al sounds-good-score).
    """
    if len(prompt_fields['pitch']) == 0 or len(cont_fields['pitch']) == 0:
        return 0.0
    p_tail = [p - 1 for p in prompt_fields['pitch'][-window:]]
    c_head = [p - 1 for p in cont_fields['pitch'][:window]]
    mean_p = sum(p_tail) / len(p_tail)
    mean_c = sum(c_head) / len(c_head)
    dist = abs(mean_p - mean_c)
    return max(0.0, 1.0 - dist / 24.0)  # 2 octavas de tolerancia


def _load_fields_from_midi(path):
    notes, tpb = bake_sustain_notes(path)
    return notes_to_fields(notes, tpb), tpb, notes


def cmd_score(args):
    _require_mido()
    files = [args.file] + (args.compare or [])
    reports = []
    for f in files:
        fields, tpb, notes = _load_fields_from_midi(f)
        m = score_from_fields(fields)
        s = aggregate_score(m)
        reports.append((f, m, s))

    B, C, Y, G, R = '\033[1m', '\033[36m', '\033[33m', '\033[32m', '\033[0m'
    print(f"\n{B}── MÉTRICAS HEURÍSTICAS ──{R}")
    for f, m, s in reports:
        print(f"\n{C}{f}{R}")
        if m.get('n_notes', 0) == 0:
            print("  (sin notas)")
            continue
        for k, v in m.items():
            print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
        print(f"  {Y}score agregado:{R} {G}{s:.3f}{R}/1.000")

    if len(reports) > 1:
        best = max(reports, key=lambda r: r[2])
        print(f"\n{B}Mejor puntuado:{R} {best[0]}  (score={best[2]:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: make-pairs
# ══════════════════════════════════════════════════════════════════════════════

def cmd_make_pairs(args):
    _require_mido()
    if len(args.candidates) < 2:
        print("ERROR: se necesitan al menos 2 --candidates para formar pares.")
        sys.exit(1)

    prompt_fields, _, _ = _load_fields_from_midi(args.prompt)
    scored = []
    for c in args.candidates:
        fields, _, _ = _load_fields_from_midi(c)
        m = score_from_fields(fields)
        s = aggregate_score(m) + 0.3 * continuity_bonus(prompt_fields, fields)
        scored.append((c, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    print("Ranking de candidatos:")
    for c, s in scored:
        print(f"  {s:.3f}  {c}")

    pairs = []
    k = min(args.top_k_pairs, len(scored) - 1)
    for i in range(k):
        chosen, rejected = scored[i], scored[len(scored) - 1 - i]
        if chosen[0] == rejected[0]:
            continue
        pairs.append({'prompt': args.prompt, 'chosen': chosen[0], 'rejected': rejected[0]})

    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"\n{len(pairs)} par(es) escrito(s) en {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: dpo
# ══════════════════════════════════════════════════════════════════════════════

def _sequence_batch_for_dpo(prompt_notes_fields, cont_notes_fields, device):
    full = {k: np.concatenate([prompt_notes_fields[k], cont_notes_fields[k]])
            for k in ('pitch', 'delta', 'duration', 'velocity')}
    item = {k: full[k] for k in full}
    batch = build_token_batch([item], device)
    start_index = len(prompt_notes_fields['pitch']) + 1  # +1 por el BOS
    return batch, start_index


def cmd_dpo(args):
    _require_mido()
    _require_torch()
    device = _get_device()

    with open(args.pairs) as f:
        pairs = json.load(f)
    if not pairs:
        print("ERROR: el manifest de pares está vacío.")
        sys.exit(1)

    policy, cfg, _ = _load_model(args.model_dir, device)
    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    print(f"DPO: {len(pairs)} pares, β={args.beta}, epochs={args.epochs}")
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        total_loss, n_correct = 0.0, 0
        for pair in pairs:
            prompt_fields, tpb, _ = _load_fields_from_midi(pair['prompt'])
            chosen_fields, _, _ = _load_fields_from_midi(pair['chosen'])
            rejected_fields, _, _ = _load_fields_from_midi(pair['rejected'])

            batch_c, start_c = _sequence_batch_for_dpo(prompt_fields, chosen_fields, device)
            batch_r, start_r = _sequence_batch_for_dpo(prompt_fields, rejected_fields, device)

            policy.train()
            logp_c_pol = policy.sequence_logprob(batch_c, start_c)
            logp_r_pol = policy.sequence_logprob(batch_r, start_r)
            with torch.no_grad():
                logp_c_ref = reference.sequence_logprob(batch_c, start_c)
                logp_r_ref = reference.sequence_logprob(batch_r, start_r)

            diff = (logp_c_pol - logp_c_ref) - (logp_r_pol - logp_r_ref)
            loss = -F.logsigmoid(args.beta * diff)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if diff.item() > 0:
                n_correct += 1

        print(f"[época {epoch:3d}] dpo_loss={total_loss/len(pairs):.4f}  "
              f"pref_acc={n_correct/len(pairs):.2%}")

    out_dir = args.output_dir or args.model_dir
    optimizer_for_save = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    _save_checkpoint(out_dir, policy, optimizer_for_save, epoch=args.epochs - 1,
                      best_loss=total_loss / len(pairs), cfg=cfg)
    print(f"\nModelo ajustado por preferencias guardado en {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: bench
# ══════════════════════════════════════════════════════════════════════════════

def cmd_bench(args):
    _require_torch()
    device = _get_device()

    if args.model_dir:
        model, cfg, _ = _load_model(args.model_dir, device)
    else:
        cfg = {'dim': args.dim, 'layers': args.layers, 'heads': args.heads,
               'max_len': args.notes + 8}
        model = PianoAutocompleteModel.from_config(cfg).to(device)
        print("⚠  Sin --model-dir: benchmark con pesos ALEATORIOS.")

    prompt_fields = {'pitch': np.array([61], dtype=np.int32),
                      'delta': np.array([1], dtype=np.int32),
                      'duration': np.array([13], dtype=np.int32),
                      'velocity': np.array([9], dtype=np.int32)}

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # warm-up
    model.generate(prompt_fields, n_notes=min(8, args.notes), device=device, rng=rng)

    t0 = time.time()
    generated = model.generate(prompt_fields, n_notes=args.notes, device=device, rng=rng)
    elapsed = time.time() - t0
    n_gen = len(generated['pitch'])
    rate = n_gen / elapsed if elapsed > 0 else float('inf')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDispositivo:  {device}")
    print(f"Parámetros:   {n_params:,}")
    print(f"Generadas:    {n_gen} notas en {elapsed:.3f}s")
    print(f"Velocidad:    {rate:.1f} notas/seg")
    print(f"(referencia del post: ~108 notas/seg con 125M parámetros en iPhone 15)")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    _require_torch()

    if args.data:
        print(f"\n── Dataset: {args.data} ──")
        try:
            ds = NoteDataset(args.data)
            print(ds)
            for split, files in ds.splits.items():
                total_notes = 0
                for f in files:
                    with open(f, 'rb') as fh:
                        data = pickle.load(fh)
                    total_notes += len(data['pitch'])
                avg = total_notes // max(len(files), 1)
                print(f"  {split:5s}: {len(files):4d} ficheros, ~{avg} notas/fichero, "
                      f"total {total_notes:,}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if args.model_dir:
        print(f"\n── Modelo: {args.model_dir} ──")
        try:
            cfg = _load_config(args.model_dir)
            print("  Config:")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            model, cfg, ckpt = _load_model(args.model_dir, torch.device('cpu'))
            n_params = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Checkpoint:")
            print(f"    época:      {ckpt.get('epoch', '?')}")
            bl = ckpt.get('best_loss', '?')
            print(f"    best_loss:  {bl:.4f}" if isinstance(bl, float) else f"    best_loss:  {bl}")
            print(f"    parámetros: {n_params:,} totales, {n_train:,} entrenables")
        except FileNotFoundError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not args.data and not args.model_dir:
        print("Indica --data y/o --model-dir para inspeccionar.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='piano_autocomplete',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Piano Autocomplete v1.0\n"
            "Autocompletado de interpretaciones de piano (estilo RollTab).\n\n"
            "Flujo típico:\n"
            "  1. preprocess  — MIDI corpus → pickles de notas\n"
            "  2. train       — entrenar el modelo\n"
            "  3. continue    — autocompletar un MIDI\n"
            "  4. make-pairs / dpo — refinar por preferencias\n"
        ),
    )
    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── preprocess ────────────────────────────────────────────────────────────
    p = sub.add_parser('preprocess', help='MIDI corpus → pickles de notas')
    p.add_argument('--input', required=True, metavar='DIR')
    p.add_argument('--output', required=True, metavar='DIR')
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--min-notes', type=int, default=8)
    p.add_argument('--augment', action='store_true',
                    help='Añade transposición/tempo/jitter (5x ficheros)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_preprocess)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrenar el modelo')
    p.add_argument('--data', required=True, metavar='DIR')
    p.add_argument('--model-dir', required=True, metavar='DIR')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-notes', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--sched-start', type=int, default=5, metavar='N')
    p.add_argument('--sched-end', type=int, default=25, metavar='N')
    p.add_argument('--sched-max-p', type=float, default=0.5, metavar='F')
    p.add_argument('--eval-every', type=int, default=5, metavar='N')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=cmd_train)

    # ── continue ──────────────────────────────────────────────────────────────
    p = sub.add_parser('continue', help='Autocompletar un MIDI dado')
    p.add_argument('--prompt', required=True, metavar='FILE')
    p.add_argument('--model-dir', default=None, metavar='DIR')
    p.add_argument('--allow-random-init', action='store_true',
                    help='Generar con pesos aleatorios si no hay --model-dir')
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--output', default='continuado.mid', metavar='FILE')
    p.add_argument('--notes', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--top-p', type=float, default=0.0)
    p.add_argument('--max-context', type=int, default=512)
    p.add_argument('--keep-notes', type=int, default=384)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=cmd_continue)

    # ── score ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('score', help='Métricas heurísticas de una o varias continuaciones')
    p.add_argument('file', metavar='FILE')
    p.add_argument('--compare', nargs='+', metavar='FILE',
                    help='Ficheros adicionales para comparar')
    p.set_defaults(func=cmd_score)

    # ── make-pairs ────────────────────────────────────────────────────────────
    p = sub.add_parser('make-pairs', help='Generar manifest chosen/rejected para DPO')
    p.add_argument('--prompt', required=True, metavar='FILE')
    p.add_argument('--candidates', nargs='+', required=True, metavar='FILE')
    p.add_argument('--top-k-pairs', type=int, default=1, metavar='N')
    p.add_argument('--output', default='pairs.json', metavar='FILE')
    p.set_defaults(func=cmd_make_pairs)

    # ── dpo ───────────────────────────────────────────────────────────────────
    p = sub.add_parser('dpo', help='Ajuste por preferencias (DPO)')
    p.add_argument('--model-dir', required=True, metavar='DIR')
    p.add_argument('--pairs', required=True, metavar='FILE')
    p.add_argument('--output-dir', default=None, metavar='DIR')
    p.add_argument('--beta', type=float, default=0.03)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=3)
    p.set_defaults(func=cmd_dpo)

    # ── bench ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('bench', help='Medir velocidad de generación (notas/seg)')
    p.add_argument('--model-dir', default=None, metavar='DIR')
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--notes', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=cmd_bench)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Diagnóstico de dataset y/o checkpoint')
    p.add_argument('--data', default=None, metavar='DIR')
    p.add_argument('--model-dir', default=None, metavar='DIR')
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
