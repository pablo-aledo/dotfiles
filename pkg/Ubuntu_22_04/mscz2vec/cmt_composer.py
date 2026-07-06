#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CMT COMPOSER  v1.1                                     ║
║         Generación de melodía condicionada por acordes (Transformer)         ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Chord Encoder (BiLSTM) → Rhythm Decoder (Transformer causal) →           ║
║    Pitch Decoder (Transformer causal)                                        ║
║    Atención relativa multi-cabeza · Entrenamiento en 2 fases                 ║
║                                                                              ║
║  FLUJO COMPLETO:                                                             ║
║    inspect  → convert → prepare → train (rhythm) →                          ║
║    prepare  → train (pitch) → sample                                         ║
║                                                                              ║
║  COMANDOS — PREPARACIÓN DE CORPUS:                                           ║
║    inspect     — muestra pistas de un MIDI y elige melodía automáticamente  ║
║    convert     — MIDIs orquestales → MIDIs de 2 pistas (melodía + acordes)  ║
║    prepare     — MIDIs de 2 pistas → instancias .pkl para entrenar          ║
║                                                                              ║
║  COMANDOS — MODELO:                                                          ║
║    train       — entrena el modelo (fase: rhythm | pitch)                   ║
║    sample      — genera un fragmento de melodía (4 compases)                ║
║    compose     — genera N fragmentos encadenados en un MIDI largo           ║
║    encode      — MIDI → representación pitch/rhythm/chord (.pkl)            ║
║    round-trip  — MIDI → pkl → MIDI sin modelo (diagnóstico)                 ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pretty_midi, numpy, torch, scipy                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣

# ── 1. Inspeccionar un MIDI orquestal ────────────────────────────────────────
python cmt_composer.py inspect --input cancion.mid

# ── 2. Convertir carpeta de MIDIs orquestales a 2 pistas ─────────────────────
# Detección automática de melodía:
python cmt_composer.py convert \
    --input-dir midis_orquestales/ --output-dir midis_2pistas/

# Especificar pista de melodía manualmente (ver índice en inspect):
python cmt_composer.py convert \
    --input-dir midis_orquestales/ --output-dir midis_2pistas/ \
    --melody-track 0

# Con filtros de bajo y percusión, y 3 voces en el acorde:
python cmt_composer.py convert \
    --input-dir midis_orquestales/ --output-dir midis_2pistas/ \
    --exclude-drums --exclude-bass --chord-voices 3

# ── 3. Preparar corpus ────────────────────────────────────────────────────────
# Con transposición a 12 tonalidades (necesario para fase 1):
python cmt_composer.py prepare \
    --input-dir midis_2pistas/ --output-dir data_12keys/ --shift --report

# En clave original (necesario para fase 2):
python cmt_composer.py prepare \
    --input-dir midis_2pistas/ --output-dir data_ckey/ --report

# Con rango de pitch ampliado (128 notas):
python cmt_composer.py prepare \
    --input-dir midis_2pistas/ --output-dir data_full/ --pitch-range 128

# ── 4. Entrenar fase 1: Rhythm Decoder ───────────────────────────────────────
python cmt_composer.py train \
    --data-dir data_12keys/ --model-dir model/ \
    --phase rhythm --epochs 100 --batch-size 64 --lr 1e-4

# ── 5. Entrenar fase 2: Pitch Decoder (RD congelado) ─────────────────────────
python cmt_composer.py train \
    --data-dir data_ckey/ --model-dir model/ \
    --phase pitch --restore-rhythm-epoch 100 \
    --epochs 100 --batch-size 64 --lr 1e-4

# Continuar un entrenamiento interrumpido:
python cmt_composer.py train \
    --data-dir data_ckey/ --model-dir model/ \
    --phase pitch --restore-epoch 50

# ── 6. Generar melodía ────────────────────────────────────────────────────────
python cmt_composer.py sample \
    --model-dir model/ --chord-midi midis_2pistas/cancion_2track.mid \
    --output melodia.mid --topk 5 --bars 8

# Con prime (seed desde un MIDI de referencia):
python cmt_composer.py sample \
    --model-dir model/ --chord-midi midis_2pistas/cancion_2track.mid \
    --prime-midi referencia.mid --prime-bars 2 \
    --output melodia.mid

# ── Componer una pieza larga (N fragmentos encadenados) ──────────────────────
python cmt_composer.py compose     --model-dir model/ --chord-midi midis_2pistas/cancion_2track.mid     --fragments 16 --output pieza.mid --topk 5

# Con instrumento específico (40=violín, 73=flauta, 0=piano):
python cmt_composer.py compose     --model-dir model/ --chord-midi midis_2pistas/cancion_2track.mid     --fragments 8 --program 40 --output pieza_violin.mid

# ── Diagnóstico ───────────────────────────────────────────────────────────────
python cmt_composer.py encode --input cancion.mid --output instancia.pkl
python cmt_composer.py round-trip --input cancion.mid --output vuelta.mid

"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.1"

# Representación de ritmo: 0=silencio, 1=nota sostenida, 2=onset
RHYTHM_CLASSES = 3
RHYTHM_LABELS  = {0: 'rest', 1: 'hold', 2: 'onset'}

# Valores por defecto del modelo (equivalentes a hparams.yaml original)
# RESTRICCIÓN ARQUITECTURAL: hidden_dim debe ser exactamente 2 * pitch_emb_size.
# Esto no es arbitrario: las proporciones internas garantizan que
#   pitch_emb_size + 2*chord_hidden + rhythm_hidden == 2 * pitch_emb_size
# donde chord_hidden = 7*(pitch_emb_size//32) y rhythm_hidden = 9*(pitch_emb_size//16).
# Si cambias pitch_emb_size, actualiza hidden_dim = 2 * pitch_emb_size en consecuencia.
DEFAULT_MODEL = dict(
    num_pitch         = 50,    # 48 alturas + hold + rest
    frame_per_bar     = 16,    # corchea como unidad mínima (4/4)
    num_bars          = 8,
    chord_emb_size    = 128,
    pitch_emb_size    = 256,
    hidden_dim        = 512,   # debe ser 2 * pitch_emb_size
    key_dim           = 512,
    value_dim         = 512,
    input_dropout     = 0.2,
    layer_dropout     = 0.2,
    attention_dropout = 0.2,
    num_layers        = 8,
    num_heads         = 16,
)

# Valores por defecto del entrenamiento
DEFAULT_TRAIN = dict(
    max_epoch   = 101,
    lr          = 1e-4,
    batch_size  = 64,
    topk        = 5,
    num_prime   = 16,
    num_sample  = 5,
)

# Notas raíz para visualización
ROOT_NOTE_LIST = [' C', 'C#', ' D', 'D#', ' E', ' F', 'F#', ' G', 'G#', ' A', 'A#', ' B']


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE MELODÍA (para convert)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_tempo(midi, default=120.0):
    """Extrae BPM del primer evento de tempo; si no hay o falla, usa default."""
    try:
        times, tempos = midi.get_tempo_changes()
        if len(tempos) > 0:
            return float(tempos[0])
    except Exception:
        pass
    try:
        return float(midi.estimate_tempo())
    except Exception:
        pass
    return default


def _mean_polyphony(notes, sample_points=200):
    """Número medio de notas simultáneas muestreando el tiempo."""
    if not notes:
        return 0.0
    t_start = min(n.start for n in notes)
    t_end   = max(n.end   for n in notes)
    if t_end <= t_start:
        return 1.0
    ts   = np.linspace(t_start, t_end, sample_points)
    poly = [sum(1 for n in notes if n.start <= t < n.end) for t in ts]
    return float(np.mean(poly))


def _score_melody_track(inst):
    """
    Puntúa qué tan probable es que una pista sea la melodía.
    Criterios: muchos onsets, pitch medio alto, poca polifonía, no es percusión.
    """
    if inst.is_drum:
        return -1.0
    notes = inst.notes
    if len(notes) < 8:
        return -1.0
    pitches    = np.array([n.pitch for n in notes])
    dur        = max(n.end for n in notes) - min(n.start for n in notes) + 1e-6
    onset_rate = len(notes) / dur
    pitch_mean = pitches.mean()
    pitch_std  = pitches.std()
    polyphony  = _mean_polyphony(notes)
    score = (
          0.4 * (pitch_mean / 127.0)
        + 0.4 * min(onset_rate / 8.0, 1.0)
        + 0.1 * min(pitch_std / 12.0, 1.0)
        - 0.3 * min(polyphony, 4.0)
    )
    if pitch_mean < 48:
        score -= 0.3
    return score


def _detect_melody_track(instruments):
    """Devuelve el índice de la pista más probable como melodía."""
    scores = [(i, _score_melody_track(inst)) for i, inst in enumerate(instruments)]
    scores = [(i, s) for i, s in scores if s >= 0]
    if not scores:
        return 0
    return max(scores, key=lambda x: x[1])[0]


def _build_chord_instrument(instruments, exclude_idxs, chord_voices, bpm,
                             frame_per_bar=16):
    """
    Combina todas las pistas excepto las excluidas en una pista de acordes.
    Los acordes se cuantifican a la rejilla de frames y se limitan a
    `chord_voices` notas (las más graves del momento).
    """
    import pretty_midi

    fps       = (frame_per_bar / 4) * (bpm / 60)
    unit_time = 1.0 / fps

    acc_notes = []
    for i, inst in enumerate(instruments):
        if i in exclude_idxs or inst.is_drum:
            continue
        acc_notes.extend(inst.notes)

    if not acc_notes:
        return pretty_midi.Instrument(program=0, name='chord')

    t_end    = max(n.end for n in acc_notes)
    n_frames = int(np.ceil(t_end / unit_time)) + 1

    acc_inst       = pretty_midi.Instrument(program=0)
    acc_inst.notes = acc_notes
    roll = acc_inst.get_piano_roll(fs=fps)
    if roll.shape[1] < n_frames:
        roll = np.pad(roll, ((0, 0), (0, n_frames - roll.shape[1])))

    chord_inst  = pretty_midi.Instrument(program=0, name='chord')
    active_prev = set()
    prev_t      = 0

    for t in range(n_frames):
        active = set(int(p) for p in roll[:, t].nonzero()[0])
        if len(active) > chord_voices:
            active = set(sorted(active)[:chord_voices])
        if active != active_prev:
            for p in active_prev:
                chord_inst.notes.append(pretty_midi.Note(
                    start=prev_t * unit_time, end=t * unit_time,
                    pitch=p, velocity=70))
            active_prev = active
            prev_t      = t

    for p in active_prev:
        chord_inst.notes.append(pretty_midi.Note(
            start=prev_t * unit_time, end=n_frames * unit_time,
            pitch=p, velocity=70))

    return chord_inst


# ══════════════════════════════════════════════════════════════════════════════
#  CAPAS Y MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(cfg: dict):
    """Importa torch y construye el modelo con la configuración dada."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DynamicPositionEmbedding(nn.Module):
        def __init__(self, hidden_dim, max_len):
            super().__init__()
            import math
            embed = np.array([[[
                math.sin(
                    pos * math.exp(-math.log(10000) * i / hidden_dim) *
                    math.exp(math.log(10000) / hidden_dim * (i % 2)) +
                    0.5 * math.pi * (i % 2)
                )
                for i in range(hidden_dim)]
                for pos in range(max_len)]])
            self.pe = nn.Parameter(
                torch.tensor(embed, dtype=torch.float), requires_grad=False)

        def forward(self, x):
            return x + self.pe[:, :x.shape[1], :].to(x.device)

    class RelativeMultiHeadAttention(nn.Module):
        def __init__(self, input_dim, key_dim, value_dim, output_dim,
                     max_len, num_heads, preceding_only=True, dropout=0.0):
            super().__init__()
            if key_dim % num_heads != 0:
                raise ValueError(f"key_dim ({key_dim}) no divisible por num_heads ({num_heads})")
            if value_dim % num_heads != 0:
                raise ValueError(f"value_dim ({value_dim}) no divisible por num_heads ({num_heads})")
            self.num_heads   = num_heads
            self.query_scale = (key_dim // num_heads) ** -0.5
            self.max_len     = max_len
            self.query_linear  = nn.Linear(input_dim, key_dim,   bias=False)
            self.key_linear    = nn.Linear(input_dim, key_dim,   bias=False)
            self.value_linear  = nn.Linear(input_dim, value_dim, bias=False)
            self.output_linear = nn.Linear(value_dim, output_dim, bias=False)
            rel_len = max_len if preceding_only else 2 * max_len - 1
            self.relative_embedding = nn.Parameter(
                torch.randn(num_heads, key_dim // num_heads, rel_len))
            self.attn_drop = nn.Dropout(dropout)

        def _split_heads(self, x):
            b, t, d = x.shape
            return x.view(b, t, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

        def _merge_heads(self, x):
            b, h, t, d = x.shape
            return x.permute(0, 2, 1, 3).contiguous().view(b, t, d * h)

        @staticmethod
        def _qe_masking(qe):
            lengths = torch.arange(qe.size(-1) - 1, qe.size(-1) - qe.size(-2) - 1, -1)
            mask = torch.arange(qe.size(-1)).unsqueeze(0) >= lengths.unsqueeze(1)
            return mask.float().to(qe.device) * qe

        def _calc_pos_emb(self, queries, mask):
            """Relative position embedding con skew (Music Transformer).

            El truco de pad+reshape del paper solo funciona cuando la
            longitud de secuencia t == max_len.  Aqui rellenamos las queries
            con ceros hasta max_len, aplicamos el skew original y luego
            recortamos a t, de modo que el resultado es correcto para
            cualquier t <= max_len (y se preserva el comportamiento original
            cuando t == max_len).
            """
            b, h, t, d = queries.shape
            pad_amount = self.max_len - t
            if pad_amount < 0:
                raise ValueError(
                    f"sequence length {t} exceeds max_len {self.max_len}")
            if pad_amount > 0:
                # rellenar con ceros para que el skew alinee como en t==max_len
                queries = F.pad(queries, (0, 0, 0, pad_amount))

            if mask is not None:
                emb = torch.matmul(queries, self.relative_embedding[:, :, :self.max_len])
                emb = self._qe_masking(emb)
                emb = F.pad(emb, (1, 0, 0, 0))
                emb = emb.view(-1, emb.size(1), emb.size(3), emb.size(2))
                emb = emb[:, :, 1:, :]
            else:
                emb = torch.matmul(queries, self.relative_embedding)
                emb = F.pad(emb, (1, 0, 0, 0))
                emb = emb.view(emb.size(0), emb.size(1), -1)[:, :, self.max_len:]
                emb = emb.view(emb.size(0), emb.size(1), self.max_len, -1)[:, :, :, :self.max_len]

            if pad_amount > 0:
                # recortar de vuelta a la longitud original
                emb = emb[:, :, :t, :t]
            return emb

        def forward(self, q, k, v, return_weights=False, mask=None):
            q = self._split_heads(self.query_linear(q))
            k = self._split_heads(self.key_linear(k))
            v = self._split_heads(self.value_linear(v))
            logits = torch.matmul(q, k.permute(0, 1, 3, 2))
            logits += self._calc_pos_emb(q, mask)
            if mask is not None:
                logits += mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits).to(logits.device)
            logits *= self.query_scale
            weights = F.softmax(logits, dim=-1)
            weights = self.attn_drop(weights)
            ctx = self._merge_heads(torch.matmul(weights, v))
            out = self.output_linear(ctx)
            return (out, weights) if return_weights else (out, None)

    def _gen_bias_mask(max_len):
        mask = np.triu(np.full([max_len, max_len], -np.inf), 1)
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(1)

    class SelfAttentionBlock(nn.Module):
        def __init__(self, input_dim, hidden_dim, key_dim, value_dim,
                     num_heads, max_len, preceding_only=True,
                     layer_dropout=0.0, attention_dropout=0.0):
            super().__init__()
            self.mask = _gen_bias_mask(max_len)
            self.mha  = RelativeMultiHeadAttention(
                input_dim, key_dim, value_dim, hidden_dim,
                max_len, num_heads, preceding_only, attention_dropout)
            self.ffn1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.ffn2 = nn.Linear(hidden_dim // 2, hidden_dim)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(layer_dropout)
            self.ln1  = nn.LayerNorm(hidden_dim, eps=1e-6)
            self.ln2  = nn.LayerNorm(hidden_dim, eps=1e-6)

        def forward(self, x, return_weights=False, masking=True):
            mask = self.mask if masking else None
            y, w = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), return_weights, mask)
            x = x + self.drop(y)
            y = self.ffn2(self.relu(self.ffn1(self.ln2(x))))
            x = x + self.drop(y)
            return x, w

    class CMT(nn.Module):
        """Chord-conditioned Melody Transformer."""
        def __init__(self, num_pitch=50, frame_per_bar=16, num_bars=8,
                     chord_emb_size=128, pitch_emb_size=256, hidden_dim=512,
                     key_dim=512, value_dim=512, num_layers=8, num_heads=16,
                     input_dropout=0.2, layer_dropout=0.2, attention_dropout=0.2):
            super().__init__()
            self.max_len       = frame_per_bar * num_bars
            self.frame_per_bar = frame_per_bar
            self.num_pitch     = num_pitch
            self.num_chords    = 12
            self.num_rhythm    = RHYTHM_CLASSES

            self.chord_emb_size  = chord_emb_size
            self.rhythm_emb_size = pitch_emb_size // 8
            self.pitch_emb_size  = pitch_emb_size
            self.chord_hidden    = 7 * (pitch_emb_size // 32)
            self.rhythm_hidden   = 9 * (pitch_emb_size // 16)
            self.hidden_dim      = hidden_dim

            # Embeddings
            self.chord_emb  = nn.Parameter(
                torch.randn(self.num_chords, chord_emb_size, requires_grad=True))
            self.rhythm_emb = nn.Embedding(self.num_rhythm,  self.rhythm_emb_size)
            self.pitch_emb  = nn.Embedding(num_pitch,        pitch_emb_size)

            # Chord encoder
            self.chord_lstm = nn.LSTM(
                chord_emb_size, self.chord_hidden,
                num_layers=1, batch_first=True, bidirectional=True)

            # Positional encodings
            # Nota: p_in = 2*pitch_emb_size = pitch_emb + 2*chord_hidden + rhythm_hidden
            # (identidad matemática garantizada por las proporciones de los tamaños)
            self.rhythm_pos = DynamicPositionEmbedding(self.rhythm_hidden,  self.max_len)
            self.pitch_pos  = DynamicPositionEmbedding(2 * pitch_emb_size,  self.max_len)

            self.emb_drop = nn.Dropout(input_dropout)

            # Rhythm decoder
            r_in = 2 * self.chord_hidden + self.rhythm_emb_size
            self.rhythm_decoder = nn.ModuleList([
                SelfAttentionBlock(r_in, self.rhythm_hidden,
                                   key_dim // 4, value_dim // 4,
                                   num_heads, self.max_len,
                                   preceding_only=False,
                                   layer_dropout=layer_dropout,
                                   attention_dropout=attention_dropout)
                for _ in range(num_layers)
            ])
            self.rhythm_out = nn.Linear(self.rhythm_hidden, self.num_rhythm)

            # Pitch decoder
            p_in = 2 * pitch_emb_size
            self.pitch_decoder = nn.ModuleList([
                SelfAttentionBlock(p_in, hidden_dim,
                                   key_dim, value_dim,
                                   num_heads, self.max_len,
                                   preceding_only=True,
                                   layer_dropout=layer_dropout,
                                   attention_dropout=attention_dropout)
                for _ in range(num_layers)
            ])
            self.pitch_out = nn.Linear(hidden_dim, num_pitch)
            self.log_sm    = nn.LogSoftmax(dim=-1)

        def _chord_forward(self, chord):
            b = chord.size(0)
            emb = torch.matmul(chord.float(), self.chord_emb)
            h0  = torch.zeros(2, b, self.chord_hidden, device=chord.device)
            c0  = torch.zeros(2, b, self.chord_hidden, device=chord.device)
            self.chord_lstm.flatten_parameters()
            out, _ = self.chord_lstm(emb, (h0, c0))
            return out[:, 1:, :self.chord_hidden], out[:, 1:, self.chord_hidden:]

        def _rhythm_forward(self, rhythm, chord_fwd, chord_bwd,
                             return_weights=False, masking=True):
            x = self.rhythm_emb(rhythm)
            x = torch.cat([x, chord_fwd, chord_bwd], dim=-1)
            x = x * (self.rhythm_hidden ** 0.5)
            x = self.rhythm_pos(x)
            x = self.emb_drop(x)
            ws = []
            for layer in self.rhythm_decoder:
                x, w = layer(x, return_weights, masking)
                if return_weights and w is not None:
                    ws.append(w)
            return x, ws

        def _pitch_forward(self, emb, return_weights=False, masking=True):
            x = self.pitch_pos(emb)
            x = self.emb_drop(x)
            ws = []
            for layer in self.pitch_decoder:
                x, w = layer(x, return_weights, masking)
                if return_weights and w is not None:
                    ws.append(w)
            out = self.log_sm(self.pitch_out(x))
            return out, ws

        def forward(self, rhythm, pitch, chord, rhythm_only=False):
            cf, cb = self._chord_forward(chord)

            # Rhythm decoder (causal)
            r_dec, _ = self._rhythm_forward(rhythm[:, :-1], cf, cb, masking=True)
            r_out    = self.log_sm(self.rhythm_out(r_dec))
            result   = {'rhythm': r_out}

            if not rhythm_only:
                # Rhythm encoder (no causal: ve toda la secuencia)
                r_enc, _ = self._rhythm_forward(rhythm[:, 1:], cf, cb, masking=False)
                p_emb    = self.pitch_emb(pitch)
                emb      = torch.cat([p_emb, cf, cb, r_enc], dim=-1)
                emb      = emb * (self.hidden_dim ** 0.5)
                p_out, _ = self._pitch_forward(emb, masking=True)
                result['pitch'] = p_out

            return result

        @torch.no_grad()
        def sampling(self, prime_rhythm, prime_pitch, chord, topk=None):
            import torch.nn.functional as F_

            cf, cb = self._chord_forward(chord)
            b      = prime_pitch.size(0)
            pad_r  = self.max_len - prime_rhythm.size(1)
            rhythm = torch.cat([
                prime_rhythm,
                torch.zeros(b, pad_r, dtype=torch.long, device=prime_rhythm.device)
            ], dim=1)

            # Generar ritmo autoregressivamente
            for i in range(prime_rhythm.size(1), self.max_len):
                r_dec, _ = self._rhythm_forward(rhythm, cf, cb, masking=True)
                logits   = self.log_sm(self.rhythm_out(r_dec))[:, i - 1, :]
                if topk is None:
                    idx = logits.argmax(dim=-1)
                else:
                    probs, idxs = torch.topk(logits, min(topk, self.num_rhythm), dim=-1)
                    idx = idxs.gather(1, torch.multinomial(
                        F_.softmax(probs, dim=-1), 1)).squeeze(1)
                rhythm[:, i] = idx

            # Rhythm encoder completo (no causal)
            r_enc_full, _ = self._rhythm_forward(
                torch.cat([rhythm[:, 1:],
                           self.rhythm_out(
                               self._rhythm_forward(rhythm, cf, cb, masking=True)[0]
                           ).argmax(-1)[:, -1:]], dim=1),
                cf, cb, masking=False)

            # Generar pitch autoregressivamente
            pad_p = self.max_len - prime_pitch.size(1)
            pitch = torch.cat([
                prime_pitch,
                torch.full((b, pad_p), self.num_pitch - 1,
                           dtype=torch.long, device=prime_pitch.device)
            ], dim=1)

            for i in range(prime_pitch.size(1), self.max_len):
                p_emb = self.pitch_emb(pitch)
                emb   = torch.cat([p_emb, cf, cb, r_enc_full], dim=-1) * (self.hidden_dim ** 0.5)
                p_out, _ = self._pitch_forward(emb, masking=True)
                logits   = p_out[:, i - 1, :]
                if topk is None:
                    idx = logits.argmax(dim=-1)
                else:
                    probs, idxs = torch.topk(logits, min(topk, self.num_pitch), dim=-1)
                    idx = idxs.gather(1, torch.multinomial(
                        F_.softmax(probs, dim=-1), 1)).squeeze(1)
                pitch[:, i] = idx

            return {'rhythm': rhythm, 'pitch': pitch}

    return CMT(**cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _pitch_rhythm_to_midi(pitch_seq, chord_array, frame_per_bar=16,
                           output_path=None, basis_note=60, bpm=120.0):
    import pretty_midi

    fps       = (frame_per_bar / 4) * (bpm / 60)
    unit_time = 1.0 / fps
    pm_obj    = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    mel_inst  = pretty_midi.Instrument(program=0, name='melody')

    on_pitch = {}
    max_note  = 128 if len(pitch_seq) > 50 else 48

    for t, idx in enumerate(pitch_seq):
        if idx < max_note:
            if on_pitch:
                mel_inst.notes.append(pretty_midi.Note(
                    start=on_pitch['t'], end=t * unit_time,
                    pitch=basis_note + on_pitch['idx'], velocity=100))
                on_pitch = {}
            on_pitch = {'idx': idx, 't': t * unit_time}
        elif idx == max_note + 1 and on_pitch:
            mel_inst.notes.append(pretty_midi.Note(
                start=on_pitch['t'], end=t * unit_time,
                pitch=basis_note + on_pitch['idx'], velocity=100))
            on_pitch = {}

    if on_pitch:
        mel_inst.notes.append(pretty_midi.Note(
            start=on_pitch['t'], end=len(pitch_seq) * unit_time,
            pitch=basis_note + on_pitch['idx'], velocity=100))

    pm_obj.instruments.append(mel_inst)
    pm_obj.instruments.append(_chord_array_to_instrument(chord_array, frame_per_bar, bpm))

    if output_path:
        pm_obj.write(output_path)
        print(f"[MIDI] Guardado: {output_path}")

    return pm_obj


def _chord_array_to_instrument(chord_array, frame_per_bar=16, bpm=120.0):
    import pretty_midi

    fps       = (frame_per_bar / 4) * (bpm / 60)
    unit_time = 1.0 / fps
    inst      = pretty_midi.Instrument(program=0, name='chord')
    prev_chord = chord_array[0]
    prev_t     = 0

    for t in range(1, chord_array.shape[0]):
        if not np.array_equal(chord_array[t], prev_chord):
            for pitch in prev_chord.nonzero()[0]:
                inst.notes.append(pretty_midi.Note(
                    start=prev_t * unit_time, end=t * unit_time,
                    pitch=48 + int(pitch), velocity=70))
            prev_t     = t
            prev_chord = chord_array[t]

    T = chord_array.shape[0]
    for pitch in prev_chord.nonzero()[0]:
        inst.notes.append(pretty_midi.Note(
            start=prev_t * unit_time, end=T * unit_time,
            pitch=48 + int(pitch), velocity=70))

    return inst


def _midi_to_instance(midi_path: str, num_bars=8, frame_per_bar=16,
                       pitch_range=48, shift_k=0, bpm=0.0, verbose=False):
    """
    Lee un MIDI de 2 pistas (pista 0: melodía, pista 1: acordes) y devuelve
    un dict con 'pitch', 'rhythm' y 'chord'. Devuelve None si no pasa filtros.
    bpm=0 → leer BPM del propio MIDI.
    """
    import pretty_midi

    instance_len = frame_per_bar * num_bars

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[encode] Error leyendo {midi_path}: {e}")
        return None

    if len(midi.instruments) < 2:
        print(f"[encode] {midi_path} necesita exactamente 2 pistas (melodía + acordes)")
        return None

    # Usar BPM del archivo si no se fuerza uno externo
    if bpm <= 0:
        bpm = _estimate_tempo(midi, default=120.0)

    fps       = (frame_per_bar / 4) * (bpm / 60)
    unit_time = 1.0 / fps

    if verbose:
        print(f"  bpm={bpm:.1f}  fps={fps:.2f}  unit={unit_time*1000:.1f}ms  "
              f"instance={instance_len} frames ({instance_len*unit_time:.1f}s)")

    note_inst  = midi.instruments[0]
    chord_inst = midi.instruments[1]

    if shift_k != 0:
        for n in note_inst.notes:
            n.pitch += shift_k
        for n in chord_inst.notes:
            n.pitch += shift_k

    onset_inst = pretty_midi.Instrument(program=0)
    for n in note_inst.notes:
        onset_inst.notes.append(pretty_midi.Note(
            start=n.start, end=n.start + min(n.end - n.start, unit_time),
            pitch=n.pitch, velocity=n.velocity))

    pianoroll  = note_inst.get_piano_roll(fs=fps)
    onset_roll = onset_inst.get_piano_roll(fs=fps)

    chord_snap = pretty_midi.Instrument(program=0)
    for n in chord_inst.notes:
        chord_snap.notes.append(pretty_midi.Note(
            start=n.start, end=n.start + unit_time,
            pitch=n.pitch, velocity=n.velocity))
    chord_roll = chord_snap.get_piano_roll(fs=fps)

    def _pad(r, t):
        if r.shape[1] < t:
            r = np.pad(r, ((0, 0), (0, t - r.shape[1])))
        return r[:, :t]

    pianoroll  = _pad(pianoroll,  instance_len + 1)
    onset_roll = _pad(onset_roll, instance_len + 1)
    chord_roll = _pad(chord_roll, instance_len + 1)

    pianoroll[pianoroll > 0]   = 1
    onset_roll[onset_roll > 0] = 1
    chord_roll[chord_roll > 0] = 1

    rhythm_idx = (np.sum(pianoroll.T, axis=1).clip(0, 1) +
                  np.sum(onset_roll.T, axis=1).clip(0, 1)).astype(int)

    onset_count = rhythm_idx[:instance_len].nonzero()[0].size
    if onset_count < (instance_len // 4):
        if verbose: print(f"  ✗ filtro: demasiado silencio ({onset_count} frames activos < {instance_len//4})")
        return None
    if len(chord_roll.nonzero()[1]) < 4:
        if verbose: print(f"  ✗ filtro: sin acordes suficientes")
        return None

    if pitch_range == 128:
        base_note = 0
    else:
        nonzero_pitches = onset_roll.T.nonzero()
        if len(nonzero_pitches[0]) == 0:
            if verbose: print(f"  ✗ filtro: sin onsets detectados en el piano roll")
            return None
        highest   = int(onset_roll.T.nonzero()[1].max())
        lowest    = int(onset_roll.T.nonzero()[1].min())
        base_note = 12 * (lowest // 12)
        span      = highest - base_note
        if verbose: print(f"  rango pitch: lowest={lowest} highest={highest} base={base_note} span={span} (límite={pitch_range})")
        if span >= pitch_range:
            if verbose: print(f"  ✗ filtro: rango de pitch ({span}) >= pitch_range ({pitch_range}) → usa --pitch-range 128")
            return None

    pitch_list = []
    chord_list = []
    prev_chord = np.zeros(12)
    prev_onset = 0
    cont_rest  = 0

    for t in range(instance_len + 1):
        onset_at_t = onset_roll[:, t].nonzero()[0]
        if len(onset_at_t) > 0:
            p = int(onset_at_t[0]) - base_note
            if pitch_list and abs(p - prev_onset) > 24:
                if verbose: print(f"  ✗ filtro: salto de {abs(p - prev_onset)} semitonos en t={t} (límite=24)")
                cont_rest = 30; break
            pitch_list.append(p)
            prev_onset = p
            cont_rest  = 0
        elif rhythm_idx[t] == 1:
            pitch_list.append(pitch_range)
        else:
            pitch_list.append(pitch_range + 1)
            cont_rest += 1
            if cont_rest >= 30:
                break

        chord_at_t = chord_roll[:, t].nonzero()[0]
        if len(chord_at_t) > 0:
            prev_chord = np.zeros(12)
            for note in sorted(chord_at_t[1:] % 12):
                prev_chord[int(note)] = 1
        chord_list.append(prev_chord.copy())

    if cont_rest >= 30:
        if verbose: print(f"  ✗ filtro: salto de pitch > 12 semitonos o silencio prolongado")
        return None
    if len(set(pitch_list)) <= 5:
        if verbose: print(f"  ✗ filtro: pocas alturas únicas ({len(set(pitch_list))})")
        return None

    from scipy.sparse import csc_matrix

    # Garantizar longitud exacta instance_len en todas las secuencias
    rest_val = pitch_range + 1   # índice de silencio
    seq_len = instance_len + 1
    if len(pitch_list) < seq_len:
        pitch_list += [rest_val] * (seq_len - len(pitch_list))
        chord_list += [np.zeros(12)] * (seq_len - len(chord_list))
    pitch_arr  = np.array(pitch_list[:seq_len], dtype=np.int64)
    chord_arr  = np.array(chord_list[:seq_len], dtype=np.float32)
    rhythm_arr = rhythm_idx[:seq_len].astype(np.int64)

    return {
        'pitch':  pitch_arr,
        'rhythm': rhythm_arr,
        'chord':  csc_matrix(chord_arr),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET Y DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

def _get_loader(data_dir, mode='train', batch_size=64):
    import torch
    from torch.utils.data import Dataset, DataLoader
    import pickle, glob

    class PKLDataset(Dataset):
        def __init__(self, root, mode):
            self.files = sorted(glob.glob(os.path.join(root, mode, '*/*.pkl')))
            if not self.files:
                self.files = sorted(glob.glob(os.path.join(root, mode, '*.pkl')))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            with open(self.files[idx], 'rb') as f:
                inst = pickle.load(f)
            inst['chord'] = inst['chord'].toarray()
            return inst

    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            arr = np.array([b[key] for b in batch])
            result[key] = torch.tensor(arr)
        return result

    ds = PKLDataset(data_dir, mode)
    if not ds.files:
        print(f"[data] ⚠  Sin ficheros .pkl en {data_dir}/{mode}/")
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == 'train'),
                      drop_last=(mode == "train"), collate_fn=collate_fn)


# ══════════════════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

def _focal_loss(gamma=2):
    import torch, torch.nn as nn

    class FocalLoss(nn.Module):
        def __init__(self, gamma):
            super().__init__()
            self.gamma = gamma

        def forward(self, logp, target):
            if logp.dim() > 2:
                logp = logp.view(logp.size(0), logp.size(1), -1).transpose(1, 2).contiguous().view(-1, logp.size(1))
            target = target.view(-1, 1)
            lp  = logp.gather(1, target).view(-1)
            pt  = lp.exp()
            return (-1 * (1 - pt) ** self.gamma * lp).mean()

    return FocalLoss(gamma)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    import pretty_midi

    try:
        midi = pretty_midi.PrettyMIDI(args.input)
    except Exception as e:
        print(f"[inspect] Error leyendo {args.input}: {e}"); sys.exit(1)

    bpm_est = _estimate_tempo(midi)
    dur     = midi.get_end_time()
    print(f"\n── {Path(args.input).name} ─────────────────────────────────────")
    print(f"  Duración : {dur:.1f} s  |  BPM estimado: {bpm_est:.1f}")
    print(f"  Pistas   : {len(midi.instruments)}\n")
    print(f"  {'#':>3}  {'Nombre':<28} {'Prog':>4}  {'Notas':>6}  "
          f"{'PitchMed':>8}  {'Polif':>5}  {'Drum':>4}  {'Score':>6}")
    print("  " + "─" * 72)

    scores = [_score_melody_track(inst) for inst in midi.instruments]
    best_s = max(s for s in scores if s >= 0) if any(s >= 0 for s in scores) else -1

    for i, inst in enumerate(midi.instruments):
        notes  = inst.notes
        pitchm = np.mean([n.pitch for n in notes]) if notes else 0.0
        poly   = _mean_polyphony(notes)
        score  = scores[i]
        marker = " ◀ MELODÍA (auto)" if score == best_s and score >= 0 else ""
        print(f"  {i:>3}  {inst.name[:28]:<28} {inst.program:>4}  "
              f"{len(notes):>6}  {pitchm:>8.1f}  {poly:>5.1f}  "
              f"{'sí' if inst.is_drum else 'no':>4}  {score:>6.3f}{marker}")
    print()


def cmd_convert(args):
    import pretty_midi

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midis = sorted(set(
        list(input_dir.glob('*.mid'))   + list(input_dir.glob('*.midi')) +
        list(input_dir.glob('**/*.mid')) + list(input_dir.glob('**/*.midi'))))

    if not midis:
        print(f"[convert] Sin MIDIs en {input_dir}"); sys.exit(1)

    print(f"[convert] {len(midis)} MIDIs  |  "
          f"melody_track={'auto' if args.melody_track < 0 else args.melody_track}  |  "
          f"chord_voices={args.chord_voices}  |  "
          f"exclude_drums={args.exclude_drums}  |  "
          f"exclude_bass={args.exclude_bass}")

    ok = skip = 0

    for midi_path in midis:
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            print(f"  ✗  {midi_path.name}: {e}"); skip += 1; continue

        if not midi.instruments:
            print(f"  ✗  {midi_path.name}: sin pistas"); skip += 1; continue

        mel_idx = (min(args.melody_track, len(midi.instruments) - 1)
                   if args.melody_track >= 0
                   else _detect_melody_track(midi.instruments))

        exclude = {mel_idx}
        if args.exclude_bass:
            for i, inst in enumerate(midi.instruments):
                if not inst.is_drum and inst.notes:
                    if np.mean([n.pitch for n in inst.notes]) < 48:
                        exclude.add(i)

        bpm = args.bpm if args.bpm > 0 else _estimate_tempo(midi)

        chord_inst = _build_chord_instrument(
            midi.instruments, exclude, args.chord_voices, bpm)

        if not chord_inst.notes:
            print(f"  ✗  {midi_path.name}: sin acordes extraíbles"); skip += 1; continue

        out_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        out_midi.instruments.append(midi.instruments[mel_idx])
        out_midi.instruments.append(chord_inst)

        rel      = midi_path.relative_to(input_dir)
        out_path = output_dir / rel.parent / (midi_path.stem + '_2track.mid')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_midi.write(str(out_path))

        mel_n  = len(midi.instruments[mel_idx].notes)
        mel_pm = np.mean([n.pitch for n in midi.instruments[mel_idx].notes]) if mel_n else 0
        print(f"  ✓  {midi_path.name}  →  {out_path.name}  "
              f"(mel pista {mel_idx}: {mel_n} notas, pmean={mel_pm:.0f}  |  "
              f"chord: {len(chord_inst.notes)} notas)")
        ok += 1

    print(f"\n[convert] Convertidos: {ok}  |  Saltados: {skip}")
    if ok > 0:
        print(f"\nSiguiente paso:")
        print(f"  python cmt_composer.py prepare \\")
        print(f"      --input-dir {output_dir}/ --output-dir data_12keys/ --shift --report")


def cmd_prepare(args):
    import pickle, random as _random

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_bars      = args.num_bars
    frame_per_bar = args.frame_per_bar
    pitch_range   = args.pitch_range
    bpm           = args.bpm
    shift         = args.shift

    if not input_dir.exists():
        print(f"[prepare] Error: --input-dir '{input_dir}' no existe"); sys.exit(1)

    midis = sorted(set(
        list(input_dir.glob('*.mid'))    + list(input_dir.glob('*.midi')) +
        list(input_dir.glob('**/*.mid')) + list(input_dir.glob('**/*.midi'))))

    if not midis:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}"); sys.exit(1)

    pitch_shifts = range(-5, 7) if shift else [0]

    _random.seed(42)
    songs   = list({m.parent.name if m.parent != input_dir else m.stem for m in midis})
    n_eval  = max(1, int(len(songs) * 0.1))
    n_test  = max(1, int(len(songs) * 0.1))
    eval_set = set(_random.sample(songs, n_eval))
    test_set = set(_random.sample(list(set(songs) - eval_set), n_test))

    print(f"[prepare] {len(midis)} MIDIs  |  "
          f"pitch_range={pitch_range}  |  bars={num_bars}  |  "
          f"fpb={frame_per_bar}  |  shift={'12 tonos' if shift else 'no'}")

    counts = {'train': 0, 'eval': 0, 'test': 0, 'skip': 0}

    for midi_path in midis:
        song_id = midi_path.parent.name if midi_path.parent != input_dir else midi_path.stem
        mode    = ('eval' if song_id in eval_set else
                   'test' if song_id in test_set else 'train')
        dest = output_dir / mode / song_id
        dest.mkdir(parents=True, exist_ok=True)

        for k in pitch_shifts:
            inst = _midi_to_instance(
                str(midi_path), num_bars, frame_per_bar, pitch_range, k, bpm)
            if inst is None:
                counts['skip'] += 1
                continue
            ps  = ('%d' % k) if k < 0 else ('+%d' % k)
            idx = len(list(dest.glob('*.pkl')))
            out = dest / f"{song_id}_{idx:02d}_{ps}.pkl"
            with open(out, 'wb') as f:
                pickle.dump(inst, f)
            counts[mode] += 1

    total = sum(v for k, v in counts.items() if k != 'skip')
    print(f"[prepare] Instancias generadas : {total}  "
          f"(train={counts['train']}  eval={counts['eval']}  test={counts['test']})")
    print(f"[prepare] Descartadas          : {counts['skip']}")

    if args.report:
        print(f"\n[prepare] Directorio de salida: {output_dir}")
        for m in ['train', 'eval', 'test']:
            n = len(list(output_dir.glob(f"{m}/*/*.pkl")))
            print(f"   {m:5s}: {n} ficheros")


def cmd_train(args):
    import torch
    import torch.nn as nn

    device    = torch.device('cpu') if not torch.cuda.is_available() \
                else torch.device('cuda')
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = model_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    phase       = args.phase
    rhythm_only = (phase == 'rhythm')

    cfg_path = model_dir / 'model_cfg.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            model_cfg = json.load(f)
        print(f"[train] Configuración cargada desde {cfg_path}")
    else:
        model_cfg = dict(DEFAULT_MODEL)
        model_cfg['num_bars']      = args.num_bars
        model_cfg['frame_per_bar'] = args.frame_per_bar
        with open(cfg_path, 'w') as f:
            json.dump(model_cfg, f, indent=2)

    # ── Inferir num_bars / frame_per_bar de los datos reales ──
    # Los .pkl de prepare tienen seq_len = frame_per_bar * num_bars + 1.
    # Si el config no coincide con los datos, el attention relativo falla
    # (max_len != longitud de secuencia).  Lo corregimos aquí.
    try:
        import pickle, glob
        _pkls = sorted(
            glob.glob(os.path.join(str(args.data_dir), 'train', '*/*.pkl')) +
            glob.glob(os.path.join(str(args.data_dir), 'train', '*.pkl')))
        if _pkls:
            with open(_pkls[0], 'rb') as f:
                _sample = pickle.load(f)
            _seq_len = int(_sample['rhythm'].shape[0])  # instance_len + 1
            _fpb = model_cfg.get('frame_per_bar', args.frame_per_bar)
            _inferred_bars = (_seq_len - 1) // _fpb
            if _inferred_bars * _fpb == _seq_len - 1 and _inferred_bars > 0:
                if model_cfg.get('num_bars') != _inferred_bars or \
                   model_cfg.get('frame_per_bar') != _fpb:
                    old_bars = model_cfg.get('num_bars')
                    model_cfg['num_bars']      = _inferred_bars
                    model_cfg['frame_per_bar'] = _fpb
                    with open(cfg_path, 'w') as f:
                        json.dump(model_cfg, f, indent=2)
                    print(f"[train] num_bars ajustado {old_bars} → {_inferred_bars} "
                          f"(seq_len datos={_seq_len}, fpb={_fpb})")
    except Exception:
        pass  # best-effort; si falla, se usa el config tal cual

    model = _build_model(model_cfg).to(device)

    start_epoch = 1
    if args.restore_epoch > 0:
        ckpt_path = ckpt_dir / f'checkpoint_{args.restore_epoch}.pt'
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck['model'])
            start_epoch = ck['epoch'] + 1
            print(f"[train] Restaurado epoch {args.restore_epoch}")
        else:
            print(f"[train] ⚠  Checkpoint epoch {args.restore_epoch} no encontrado")
    elif args.restore_rhythm_epoch > 0:
        ck_path = ckpt_dir / f'checkpoint_{args.restore_rhythm_epoch}.pt'
        if ck_path.exists():
            ck  = torch.load(ck_path, map_location=device)
            sd  = model.state_dict()
            rsd = {k: v for k, v in ck['model'].items() if 'rhythm' in k}
            sd.update(rsd)
            model.load_state_dict(sd)
            print(f"[train] Rhythm decoder cargado desde epoch {args.restore_rhythm_epoch}")
        else:
            print(f"[train] ⚠  Checkpoint rhythm epoch {args.restore_rhythm_epoch} no encontrado")

    if phase == 'pitch' and args.restore_rhythm_epoch > 0:
        rhythm_params = [p for n, p in model.named_parameters() if 'rhythm' in n]
        pitch_params  = [p for n, p in model.named_parameters() if 'rhythm' not in n]
        params = [{'params': rhythm_params, 'lr': 1e-6}, {'params': pitch_params}]
    else:
        params = model.parameters()

    optimizer  = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    nll_loss   = nn.NLLLoss().to(device)
    focal_loss = _focal_loss(gamma=2).to(device)

    train_loader = _get_loader(args.data_dir, 'train', args.batch_size)
    eval_loader  = _get_loader(args.data_dir, 'eval',  args.batch_size)

    if not train_loader.dataset.files:
        print("[train] Error: sin datos de entrenamiento"); sys.exit(1)

    print(f"[train] Fase: {phase}  |  device: {device}  |  "
          f"train={len(train_loader.dataset)}  eval={len(eval_loader.dataset)}")
    print(f"[train] Epochs: {start_epoch} → {args.epochs}  |  "
          f"batch={args.batch_size}  |  lr={args.lr}")

    eval_losses  = []
    best_loss    = float('inf')
    patience     = args.patience
    no_improve   = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t_rloss = t_ploss = t_n = 0
        for data in train_loader:
            for k in data:
                data[k] = data[k].to(device)
            out   = model(data['rhythm'], data['pitch'][:, :-1], data['chord'], rhythm_only)
            r_out = out['rhythm'].view(-1, out['rhythm'].size(-1))
            r_tgt = data['rhythm'][:, 1:].contiguous().view(-1)
            rl    = nll_loss(r_out, r_tgt)
            pl    = 0
            if not rhythm_only:
                p_out = out['pitch'].view(-1, out['pitch'].size(-1))
                p_tgt = data['pitch'][:, 1:].contiguous().view(-1)
                pl    = focal_loss(p_out, p_tgt)
            loss = rl + pl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_rloss += rl.item()
            t_ploss += (pl.item() if not rhythm_only else 0)
            t_n     += 1

        model.eval()
        e_rloss = e_ploss = e_n = 0
        with torch.no_grad():
            for data in eval_loader:
                for k in data:
                    data[k] = data[k].to(device)
                out   = model(data['rhythm'], data['pitch'][:, :-1], data['chord'], rhythm_only)
                r_out = out['rhythm'].view(-1, out['rhythm'].size(-1))
                r_tgt = data['rhythm'][:, 1:].contiguous().view(-1)
                e_rloss += nll_loss(r_out, r_tgt).item()
                if not rhythm_only:
                    p_out = out['pitch'].view(-1, out['pitch'].size(-1))
                    p_tgt = data['pitch'][:, 1:].contiguous().view(-1)
                    e_ploss += focal_loss(p_out, p_tgt).item()
                e_n += 1

        e_total = (e_rloss + e_ploss) / max(e_n, 1)
        eval_losses.append(e_total)

        if len(eval_losses) > 4 and eval_losses[-1] > np.mean(eval_losses[-4:-1]):
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * 0.5, 1e-6)

        print(f"[train] epoch {epoch:4d}  "
              f"train_r={t_rloss/max(t_n,1):.4f}  "
              f"train_p={t_ploss/max(t_n,1):.4f}  "
              f"eval={e_total:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ck_out = ckpt_dir / f'checkpoint_{epoch}.pt'
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'phase': phase}, ck_out)
            print(f"[train] Checkpoint guardado: {ck_out}")
            if e_total < best_loss:
                best_loss  = e_total
                no_improve = 0
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                            'phase': phase}, model_dir / 'best_model.pt')
                print(f"[train] ✓  Mejor modelo (loss={best_loss:.4f})")
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"[train] Early stop en epoch {epoch} (sin mejora en {patience} epochs)")
                    break

    print(f"[train] Entrenamiento completado. Mejor eval loss: {best_loss:.4f}")


def cmd_sample(args):
    import torch

    device    = torch.device('cpu') if not torch.cuda.is_available() \
                else torch.device('cuda')
    model_dir = Path(args.model_dir)

    cfg_path = model_dir / 'model_cfg.json'
    if not cfg_path.exists():
        print(f"[sample] Error: {cfg_path} no encontrado. Entrena primero."); sys.exit(1)
    with open(cfg_path) as f:
        model_cfg = json.load(f)

    model = _build_model(model_cfg).to(device)

    ckpt_path = model_dir / 'best_model.pt'
    if args.checkpoint:
        ckpt_path = model_dir / 'checkpoints' / f'checkpoint_{args.checkpoint}.pt'
    if not ckpt_path.exists():
        print(f"[sample] Error: checkpoint no encontrado en {ckpt_path}"); sys.exit(1)

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"[sample] Modelo cargado (epoch {ck.get('epoch','?')})")

    num_bars      = model_cfg['num_bars']
    frame_per_bar = model_cfg['frame_per_bar']
    pitch_range   = model_cfg.get('num_pitch', 50) - 2

    inst = _midi_to_instance(args.chord_midi, num_bars, frame_per_bar,
                              pitch_range, bpm=args.bpm)
    if inst is None:
        print("[sample] Error: no se pudo procesar el MIDI de acordes"); sys.exit(1)

    from scipy.sparse import issparse
    chord_np = inst['chord'].toarray() if issparse(inst['chord']) else inst['chord']
    chord    = torch.tensor(chord_np[np.newaxis, :num_bars * frame_per_bar + 1],
                             dtype=torch.float32).to(device)

    prime_frames = args.prime_bars * frame_per_bar
    if args.prime_midi:
        p_inst = _midi_to_instance(args.prime_midi, num_bars, frame_per_bar,
                                    pitch_range, bpm=args.bpm)
        if p_inst is None:
            print("[sample] ⚠  Prime MIDI no procesable, usando seed mínimo")
            prime_frames = 1
            prime_pitch  = torch.zeros(1, 1, dtype=torch.long).to(device)
            prime_rhythm = torch.zeros(1, 1, dtype=torch.long).to(device)
        else:
            prime_pitch  = torch.tensor(
                p_inst['pitch'][np.newaxis, :prime_frames], dtype=torch.long).to(device)
            prime_rhythm = torch.tensor(
                p_inst['rhythm'][np.newaxis, :prime_frames], dtype=torch.long).to(device)
    else:
        prime_pitch  = torch.zeros(1, max(1, prime_frames), dtype=torch.long).to(device)
        prime_rhythm = torch.zeros(1, max(1, prime_frames), dtype=torch.long).to(device)

    print(f"[sample] Generando {num_bars} compases  |  topk={args.topk}  |  "
          f"prime={prime_frames} frames")

    with torch.no_grad():
        result = model.sampling(prime_rhythm, prime_pitch, chord, topk=args.topk)

    pitch_seq = result['pitch'][0].cpu().numpy()
    _pitch_rhythm_to_midi(pitch_seq, chord_np[:num_bars * frame_per_bar],
                          frame_per_bar, args.output, bpm=args.bpm)

    sample_str = []
    for idx in pitch_seq[prime_frames:prime_frames + 20]:
        if idx < pitch_range:
            sample_str.append(f"{ROOT_NOTE_LIST[idx % 12].strip()}{idx // 12 + 3}")
        elif idx == pitch_range:
            sample_str.append('~')
        else:
            sample_str.append('.')
    print(f"[sample] Prime+20 frames: {' '.join(sample_str)}")


def cmd_encode(args):
    import pickle

    inst = _midi_to_instance(args.input, args.num_bars, args.frame_per_bar,
                              args.pitch_range, bpm=args.bpm)
    if inst is None:
        print(f"[encode] Error: {args.input} no superó los filtros de calidad"); sys.exit(1)

    out = args.output or (Path(args.input).stem + '.pkl')
    with open(out, 'wb') as f:
        pickle.dump(inst, f)

    from scipy.sparse import issparse
    chord_np = inst['chord'].toarray() if issparse(inst['chord']) else inst['chord']
    print(f"[encode] Guardado: {out}")
    print(f"  pitch  shape : {inst['pitch'].shape}  "
          f"  valores únicos: {len(set(inst['pitch'].tolist()))}")
    print(f"  rhythm shape : {inst['rhythm'].shape}  "
          f"  (0=rest {(inst['rhythm']==0).sum()}  "
          f"1=hold {(inst['rhythm']==1).sum()}  "
          f"2=onset {(inst['rhythm']==2).sum()})")
    print(f"  chord  shape : {chord_np.shape}  "
          f"  frames con cambio: {(np.diff(chord_np, axis=0).any(axis=1)).sum()}")


def cmd_round_trip(args):
    import pickle
    import pretty_midi

    print(f"[round-trip] Procesando: {args.input}")

    # Leer BPM real del archivo
    try:
        _pm = pretty_midi.PrettyMIDI(args.input)
        bpm = _estimate_tempo(_pm, default=120.0)
    except Exception:
        bpm = args.bpm if args.bpm > 0 else 120.0
    print(f"[round-trip] BPM={bpm:.1f}  pitch_range={args.pitch_range}  bars={args.num_bars}")

    inst = _midi_to_instance(args.input, args.num_bars, args.frame_per_bar,
                              args.pitch_range, bpm=bpm, verbose=True)
    if inst is None:
        print("[round-trip] El MIDI no superó los filtros (ver diagnóstico arriba).")
        print("  Sugerencias:")
        print("    --pitch-range 128   (aceptar cualquier rango de pitch)")
        print("    --num-bars 4        (fragmentos más cortos)")
        sys.exit(1)

    from scipy.sparse import issparse
    chord_np = inst['chord'].toarray() if issparse(inst['chord']) else inst['chord']

    _pitch_rhythm_to_midi(inst['pitch'], chord_np, args.frame_per_bar,
                          args.output, bpm=bpm)
    onset_r = (inst['rhythm'] == 2).mean()
    hold_r  = (inst['rhythm'] == 1).mean()
    rest_r  = (inst['rhythm'] == 0).mean()
    print(f"[round-trip] onset={onset_r:.1%}  hold={hold_r:.1%}  rest={rest_r:.1%}")
    print(f"[round-trip] Alturas únicas: {len(set(inst['pitch'].tolist()))}")


def cmd_compose(args):
    """Genera N fragmentos encadenados y los fusiona en un único MIDI."""
    import torch
    import pretty_midi

    device    = torch.device('cpu') if not torch.cuda.is_available()                 else torch.device('cuda')
    model_dir = Path(args.model_dir)

    cfg_path = model_dir / 'model_cfg.json'
    if not cfg_path.exists():
        print(f"[compose] Error: {cfg_path} no encontrado."); sys.exit(1)
    with open(cfg_path) as f_cfg:
        model_cfg = json.load(f_cfg)

    model = _build_model(model_cfg).to(device)
    ckpt_path = model_dir / 'best_model.pt'
    if args.checkpoint:
        ckpt_path = model_dir / 'checkpoints' / f'checkpoint_{args.checkpoint}.pt'
    if not ckpt_path.exists():
        print(f"[compose] Error: checkpoint no encontrado en {ckpt_path}"); sys.exit(1)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"[compose] Modelo cargado (epoch {ck.get('epoch','?')})")

    num_bars      = model_cfg['num_bars']
    frame_per_bar = model_cfg['frame_per_bar']
    pitch_range   = model_cfg.get('num_pitch', 50) - 2
    n_frames      = num_bars * frame_per_bar

    # Cargar acordes del MIDI completo
    inst = _midi_to_instance(args.chord_midi, num_bars, frame_per_bar,
                              pitch_range, bpm=0.0)
    if inst is None:
        print("[compose] Error: no se pudo procesar el MIDI de acordes"); sys.exit(1)

    from scipy.sparse import issparse
    chord_np = inst['chord'].toarray() if issparse(inst['chord']) else inst['chord']

    # Leer BPM real
    try:
        _pm = pretty_midi.PrettyMIDI(args.chord_midi)
        bpm = _estimate_tempo(_pm, default=120.0)
    except Exception:
        bpm = 120.0

    fps       = (frame_per_bar / 4) * (bpm / 60)
    unit_time = 1.0 / fps

    print(f"[compose] {args.fragments} fragmentos × {num_bars} compases  |  "
          f"topk={args.topk}  |  bpm={bpm:.1f}")

    # ── Generar fragmentos encadenados ────────────────────────────────────────
    all_pitch  = []
    all_rhythm = []
    prime_pitch  = torch.zeros(1, 1, dtype=torch.long).to(device)
    prime_rhythm = torch.zeros(1, 1, dtype=torch.long).to(device)

    for frag_i in range(args.fragments):
        # Acordes: rotar por el MIDI si es más corto que lo necesario
        offset    = (frag_i * n_frames) % max(chord_np.shape[0] - 1, 1)
        chord_seg = chord_np[offset:offset + n_frames + 1]
        if chord_seg.shape[0] < n_frames + 1:
            pad = np.zeros((n_frames + 1 - chord_seg.shape[0], 12))
            chord_seg = np.vstack([chord_seg, pad])
        chord_t = torch.tensor(chord_seg[np.newaxis], dtype=torch.float32).to(device)

        with torch.no_grad():
            result = model.sampling(prime_rhythm, prime_pitch, chord_t, topk=args.topk)

        pitch_seq  = result['pitch'][0].cpu().numpy()
        rhythm_seq = result['rhythm'][0].cpu().numpy()

        all_pitch.append(pitch_seq)
        all_rhythm.append(rhythm_seq)

        # Usar los últimos prime_bars compases como seed del siguiente fragmento
        prime_frames = args.prime_bars * frame_per_bar
        prime_pitch  = result['pitch'][:, -prime_frames:].to(device)
        prime_rhythm = result['rhythm'][:, -prime_frames:].to(device)

        # Resumen textual
        notes_str = []
        for idx in pitch_seq[:16]:
            if idx < pitch_range:
                notes_str.append(f"{ROOT_NOTE_LIST[idx % 12].strip()}{idx // 12 + 3}")
            elif idx == pitch_range:
                notes_str.append('~')
            else:
                notes_str.append('.')
        print(f"[compose] frag {frag_i+1:2d}: {' '.join(notes_str)}")

    # ── Fusionar en un único MIDI ─────────────────────────────────────────────
    out_midi  = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    mel_inst  = pretty_midi.Instrument(program=args.program, name='melody')
    chord_inst_out = pretty_midi.Instrument(program=0, name='chord')
    max_note   = 128 if pitch_range > 50 else 48
    basis_note = 60

    for frag_i, (pitch_seq, _) in enumerate(zip(all_pitch, all_rhythm)):
        t_offset = frag_i * n_frames * unit_time
        on_pitch = {}
        for t, idx in enumerate(pitch_seq):
            t_abs = t_offset + t * unit_time
            if idx < max_note:
                if on_pitch:
                    mel_inst.notes.append(pretty_midi.Note(
                        start=on_pitch['t'], end=t_abs,
                        pitch=basis_note + on_pitch['idx'], velocity=100))
                on_pitch = {'idx': idx, 't': t_abs}
            elif idx == max_note + 1 and on_pitch:
                mel_inst.notes.append(pretty_midi.Note(
                    start=on_pitch['t'], end=t_abs,
                    pitch=basis_note + on_pitch['idx'], velocity=100))
                on_pitch = {}
        if on_pitch:
            mel_inst.notes.append(pretty_midi.Note(
                start=on_pitch['t'],
                end=t_offset + n_frames * unit_time,
                pitch=basis_note + on_pitch['idx'], velocity=100))

        # Acordes del fragmento
        offset    = (frag_i * n_frames) % max(chord_np.shape[0] - 1, 1)
        chord_seg = chord_np[offset:offset + n_frames]
        if chord_seg.shape[0] < n_frames:
            pad = np.zeros((n_frames - chord_seg.shape[0], 12))
            chord_seg = np.vstack([chord_seg, pad])
        prev_ch = chord_seg[0]
        prev_t  = 0
        for t in range(1, n_frames):
            if not np.array_equal(chord_seg[t], prev_ch):
                for p in prev_ch.nonzero()[0]:
                    chord_inst_out.notes.append(pretty_midi.Note(
                        start=t_offset + prev_t * unit_time,
                        end=t_offset + t * unit_time,
                        pitch=48 + int(p), velocity=70))
                prev_ch = chord_seg[t]
                prev_t  = t
        for p in prev_ch.nonzero()[0]:
            chord_inst_out.notes.append(pretty_midi.Note(
                start=t_offset + prev_t * unit_time,
                end=t_offset + n_frames * unit_time,
                pitch=48 + int(p), velocity=70))

    out_midi.instruments.append(mel_inst)
    out_midi.instruments.append(chord_inst_out)
    out_midi.write(args.output)

    total_bars = args.fragments * num_bars
    total_time = args.fragments * n_frames * unit_time
    print(f"[compose] Guardado: {args.output}  "
          f"({total_bars} compases  {total_time:.1f}s  {len(mel_inst.notes)} notas)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='cmt_composer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CMT COMPOSER v1.1
            Generación de melodía condicionada por acordes.

            Flujo completo:
              inspect → convert → prepare → train (rhythm) →
              prepare → train (pitch) → sample
        """),
    )
    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect',
        help='Muestra pistas de un MIDI y detecta la melodía automáticamente',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, metavar='FILE')
    p.set_defaults(func=cmd_inspect)

    # ── convert ───────────────────────────────────────────────────────────────
    p = sub.add_parser('convert',
        help='MIDIs orquestales → MIDIs de 2 pistas (melodía + acordes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',     required=True,  metavar='DIR')
    p.add_argument('--output-dir',    required=True,  metavar='DIR')
    p.add_argument('--melody-track',  type=int,   default=-1,   metavar='N',
                   help='Índice de la pista melódica (-1 = detección automática)')
    p.add_argument('--chord-voices',  type=int,   default=4,    metavar='N',
                   help='Máximo de notas simultáneas en el acorde')
    p.add_argument('--bpm',           type=float, default=0.0,
                   help='BPM forzado (0 = estimar desde el MIDI)')
    p.add_argument('--exclude-drums', action='store_true',
                   help='Excluir pistas de percusión del acompañamiento')
    p.add_argument('--exclude-bass',  action='store_true',
                   help='Excluir pistas con pitch medio < 48 (bajo)')
    p.set_defaults(func=cmd_convert)

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('prepare',
        help='MIDIs de 2 pistas → instancias .pkl para entrenar',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',     required=True,  metavar='DIR')
    p.add_argument('--output-dir',    required=True,  metavar='DIR')
    p.add_argument('--num-bars',      type=int,   default=8,    metavar='N')
    p.add_argument('--frame-per-bar', type=int,   default=16,   metavar='N',
                   help='Frames por compás (16 = corchea en 4/4)')
    p.add_argument('--pitch-range',   type=int,   default=48,   metavar='N',
                   help='Rango de alturas (48=4 octavas, 128=todo el teclado)')
    p.add_argument('--bpm',           type=float, default=120.0)
    p.add_argument('--shift',         action='store_true',
                   help='Transponer a las 12 tonalidades (necesario para fase 1)')
    p.add_argument('--report',        action='store_true')
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train',
        help='Entrena el modelo (fase rhythm o pitch)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir',             required=True,  metavar='DIR')
    p.add_argument('--model-dir',            required=True,  metavar='DIR')
    p.add_argument('--phase',                required=True,  choices=['rhythm', 'pitch'],
                   help='rhythm: fase 1 (RD solo)  |  pitch: fase 2 (PD + RD congelado)')
    p.add_argument('--epochs',               type=int,   default=100)
    p.add_argument('--batch-size',           type=int,   default=64)
    p.add_argument('--lr',                   type=float, default=1e-4)
    p.add_argument('--num-bars',             type=int,   default=8,  metavar='N')
    p.add_argument('--frame-per-bar',        type=int,   default=16, metavar='N')
    p.add_argument('--restore-epoch',        type=int,   default=-1, metavar='N',
                   help='Continuar desde este checkpoint (cualquier fase)')
    p.add_argument('--restore-rhythm-epoch', type=int,   default=-1, metavar='N',
                   help='Cargar solo el RD desde este checkpoint (inicio fase 2)')
    p.add_argument('--patience',             type=int,   default=20, metavar='N',
                   help='Early stopping: parar si eval no mejora en N epochs (0=desactivado)')
    p.set_defaults(func=cmd_train)

    # ── sample ────────────────────────────────────────────────────────────────
    p = sub.add_parser('sample',
        help='Genera melodía dado un MIDI de acordes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-dir',   required=True, metavar='DIR')
    p.add_argument('--chord-midi',  required=True, metavar='FILE',
                   help='MIDI con 2 pistas; los acordes se leen de la pista 1')
    p.add_argument('--output',      default='sample_out.mid', metavar='FILE')
    p.add_argument('--topk',        type=int,   default=5,
                   help='Top-k muestreo (None = greedy)')
    p.add_argument('--bars',        type=int,   default=8,   metavar='N')
    p.add_argument('--bpm',         type=float, default=120.0)
    p.add_argument('--prime-midi',  default=None, metavar='FILE',
                   help='MIDI de referencia para usar como seed inicial')
    p.add_argument('--prime-bars',  type=int,   default=2,   metavar='N')
    p.add_argument('--checkpoint',  type=int,   default=None, metavar='N',
                   help='Época concreta del checkpoint (default: best_model.pt)')
    p.set_defaults(func=cmd_sample)

    # ── encode ────────────────────────────────────────────────────────────────
    p = sub.add_parser('encode',
        help='MIDI → representación pitch/rhythm/chord (.pkl)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input',         required=True, metavar='FILE')
    p.add_argument('--output',        default=None,  metavar='FILE')
    p.add_argument('--num-bars',      type=int,   default=8,   metavar='N')
    p.add_argument('--frame-per-bar', type=int,   default=16,  metavar='N')
    p.add_argument('--pitch-range',   type=int,   default=48,  metavar='N')
    p.add_argument('--bpm',           type=float, default=120.0)
    p.set_defaults(func=cmd_encode)

    # ── round-trip ────────────────────────────────────────────────────────────
    p = sub.add_parser('round-trip',
        help='MIDI → pkl → MIDI sin modelo (diagnóstico de preprocesado)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input',         required=True, metavar='FILE')
    p.add_argument('--output',        default='round_trip_out.mid', metavar='FILE')
    p.add_argument('--num-bars',      type=int,   default=8,   metavar='N')
    p.add_argument('--frame-per-bar', type=int,   default=16,  metavar='N')
    p.add_argument('--pitch-range',   type=int,   default=48,  metavar='N')
    p.add_argument('--bpm',           type=float, default=120.0)
    p.set_defaults(func=cmd_round_trip)

    # ── compose ───────────────────────────────────────────────────────────────
    p = sub.add_parser('compose',
        help='Genera N fragmentos encadenados en un único MIDI largo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-dir',   required=True, metavar='DIR')
    p.add_argument('--chord-midi',  required=True, metavar='FILE',
                   help='MIDI de acordes de referencia (se rota si es necesario)')
    p.add_argument('--output',      default='composition.mid', metavar='FILE')
    p.add_argument('--fragments',   type=int,   default=8,   metavar='N',
                   help='Número de fragmentos a generar y encadenar')
    p.add_argument('--prime-bars',  type=int,   default=2,   metavar='N',
                   help='Compases del fragmento anterior usados como seed')
    p.add_argument('--topk',        type=int,   default=5)
    p.add_argument('--program',     type=int,   default=0,   metavar='N',
                   help='Programa MIDI para la melodía (0=piano, 40=violín, 73=flauta...)')
    p.add_argument('--checkpoint',  type=int,   default=None, metavar='N')
    p.set_defaults(func=cmd_compose)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
