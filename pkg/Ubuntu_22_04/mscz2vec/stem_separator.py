#!/usr/bin/env python3
"""
===== stem_separator.py =====
║                      STEM SEPARATOR  v1.0                                   ║
║      Separación de fuentes de audio (voz/batería/bajo/resto) — HTDemucs     ║
║                                                                              ║
║  Reimplementación autocontenida (un solo fichero, dependencias mínimas)     ║
║  de Hybrid Transformer Demucs v4 (Rouard et al. / Meta AI, MIT license).    ║
║  Separa un WAV en 4 stems (drums, bass, other, vocals) cargando el          ║
║  checkpoint preentrenado oficial "htdemucs" — sin depender del paquete      ║
║  `demucs` ni de su cadena de dependencias (dora, omegaconf, einops...).     ║
║                                                                              ║
║  CÓMO FUNCIONA (resumen):                                                    ║
║    mezcla → STFT (representación complex-as-channels) + forma de onda      ║
║    cruda → dos ramas U-Net paralelas (frecuencia / tiempo) → Transformer    ║
║    cross-dominio en el cuello de botella → decodificación dual → máscara    ║
║    + iSTFT (rama freq) + onda directa (rama tiempo) → suma → 4 stems.       ║
║    Ver spec_stem_separator.md para el análisis arquitectónico completo.     ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    install     — descarga el checkpoint preentrenado oficial (~80 MB)       ║
║    separate    — WAV → hasta 4 stems (drums, bass, other, vocals)           ║
║    info        — inspecciona un checkpoint o la arquitectura por defecto    ║
║                                                                              ║
║  USO:                                                                        ║
║    python stem_separator.py install                                         ║
║    python stem_separator.py install --output ~/mis_modelos/htdemucs.th      ║
║                                                                              ║
║    python stem_separator.py separate cancion.wav                            ║
║    python stem_separator.py separate cancion.wav --output-dir stems/        ║
║    python stem_separator.py separate cancion.wav --stems vocals drums       ║
║    python stem_separator.py separate cancion.wav --two-stems vocals         ║
║    python stem_separator.py separate cancion.wav --shifts 2 --overlap 0.25  ║
║    python stem_separator.py separate cancion.wav --model-path mi_ckpt.th    ║
║    python stem_separator.py separate cancion.wav --device cpu               ║
║                                                                              ║
║    python stem_separator.py info                                            ║
║    python stem_separator.py info --model-path htdemucs.th                   ║
║                                                                              ║
║  OPCIONES (separate):                                                       ║
║    --stems S...       subconjunto de {drums,bass,other,vocals} a exportar   ║
║    --two-stems S      modo karaoke: exporta S y "no_S" (suma del resto)     ║
║    --shifts N         nº de desplazamientos aleatorios promediados (def:1)  ║
║    --overlap F        solape entre segmentos, 0-1 (default: 0.25)           ║
║    --segment F        duración de segmento en segundos (default: del ckpt)  ║
║    --model-path FILE  ruta al checkpoint .th (default: caché de `install`)  ║
║    --device cpu|cuda  dispositivo de inferencia (default: cpu)              ║
║    --output-dir DIR   carpeta de salida (default: junto al WAV de entrada)  ║
║    --random-weights   NO cargar checkpoint; pesos aleatorios (solo pruebas) ║
║                                                                              ║
║  SALIDA (convención de nombres del ecosistema, sufijo con punto):           ║
║    cancion.stem_vocals.wav   cancion.stem_drums.wav                         ║
║    cancion.stem_bass.wav     cancion.stem_other.wav                         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:  torch, numpy, soundfile                                        ║
║    (nada de demucs, dora, omegaconf, einops, torchaudio, ffmpeg)            ║
║                                                                              ║
║  NOTAS:                                                                      ║
║    · El resampling a 44100 Hz usa interpolación lineal (no sinc de alta     ║
║      calidad como `julius`), por simplicidad y mínimas dependencias. Si tu  ║
║      WAV ya está a 44100 Hz estéreo esto no importa.                        ║
║    · Solo soporta el modelo base "htdemucs" (un checkpoint). El modo        ║
║      "htdemucs_ft" (ensemble de 4 checkpoints especializados) queda fuera   ║
║      de esta v1 — ver spec_stem_separator.md §7.                            ║
"""

import argparse
import math
import os
import random
import struct
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 0. CONSTANTES DEL MODELO OFICIAL "htdemucs"
# ============================================================================

SOURCES = ["drums", "bass", "other", "vocals"]
SAMPLERATE = 44100
AUDIO_CHANNELS = 2

# Checkpoint oficial (Meta AI), hospedado en dl.fbaipublicfiles.com (MIT license).
OFFICIAL_MODEL_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
DEFAULT_CACHE_DIR = Path(os.environ.get("STEM_SEPARATOR_HOME", Path.home() / ".stem_separator"))
DEFAULT_MODEL_PATH = DEFAULT_CACHE_DIR / "htdemucs.th"


# ============================================================================
# 1. BLOQUES DE RED — mirroring exacto de la jerarquía de módulos original
#    (nombres de atributo idénticos a demucs/htdemucs.py, hdemucs.py,
#    transformer.py y demucs.py, para poder cargar el state_dict oficial
#    tal cual, sin remapeo de claves).
# ============================================================================


class LayerScale(nn.Module):
    """Layer scale (Touvron et al. 2021): reescala una rama residual, casi 0
    al inicio, aprendida después."""

    def __init__(self, channels: int, init: float = 0.0, channel_last: bool = False):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    """GroupNorm que espera (B, T, C) en vez de (B, C, T)."""

    def forward(self, x):
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class ScaledEmbedding(nn.Module):
    """Embedding con lr efectivo escalado; usado para el embedding de frecuencia."""

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class DConv(nn.Module):
    """Rama residual de convoluciones dilatadas con compuerta (GLU) y LayerScale.
    Se inserta dentro de cada capa del encoder (y opcionalmente del decoder)."""

    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4, gelu: bool = True):
        super().__init__()
        assert depth >= 1
        hidden = int(channels / compress)
        act = nn.GELU if gelu else nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(depth):
            dilation = 2 ** d
            padding = dilation
            mods = [
                nn.Conv1d(channels, hidden, 3, dilation=dilation, padding=padding),
                nn.GroupNorm(1, hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                nn.GroupNorm(1, 2 * channels),
                nn.GLU(1),
                LayerScale(channels, init),
            ]
            self.layers.append(nn.Sequential(*mods))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def pad1d(x: torch.Tensor, paddings, mode: str = "constant", value: float = 0.0):
    """Wrapper de F.pad que permite reflect-padding incluso en entradas cortas."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    return out


class HEncLayer(nn.Module):
    """Capa de encoder, usada tanto por la rama de frecuencia como la temporal."""

    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=4, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw=None, pad=True,
                 rewrite=True):
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        pad_amount = kernel_size // 4 if pad else 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad_amount
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_amount = [pad_amount, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad_amount)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)
        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)
        if not self.freq:
            le = x.shape[-1]
            if le % self.stride != 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1]
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv is not None:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite is not None:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    """Capa de decoder — inversa de HEncLayer."""

    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=4, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw=None, pad=True, rewrite=True):
        super().__init__()
        dconv_kw = dconv_kw or {}
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        pad_amount = kernel_size // 4 if pad else 0
        self.pad = pad_amount
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            self.norm1 = norm_fn(2 * chin)
        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        if not self.empty:
            x = x + skip
            if self.rewrite is not None:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv is not None:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
        if not self.last:
            z = F.gelu(z)
        return z, y


# ---- Transformer cross-dominio -------------------------------------------

def create_sin_embedding(length, dim, shift=0, device="cpu", max_period=10000):
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    if d_model % 4 != 0:
        raise ValueError("d_model debe ser múltiplo de 4 para el embedding 2D seno/coseno")
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe[None, :].to(device)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Capa de self-attention estándar + LayerScale + GroupNorm de salida opcional.
    Subclasifica nn.TransformerEncoderLayer para heredar exactamente su árbol de
    submódulos (self_attn, linear1, linear2, norm1, norm2, dropout*)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation=F.gelu,
                 norm_first=True, norm_out=True, layer_norm_eps=1e-5, layer_scale=True,
                 init_values=1e-4, batch_first=True):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first,
            norm_first=norm_first,
        )
        self.norm_out = None
        if norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=1, num_channels=d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))
        return x


class CrossTransformerEncoderLayer(nn.Module):
    """Capa de cross-attention: una rama (freq o tiempo) atiende como query a
    la OTRA rama como key/value."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation=F.gelu,
                 layer_norm_eps=1e-5, layer_scale=True, init_values=1e-4, norm_first=True,
                 norm_out=True, batch_first=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_out = None
        if norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=1, num_channels=d_model)
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, q, k, mask=None):
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))
        return x

    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CrossTransformerEncoder(nn.Module):
    """Alterna capas de self-attention (por rama) y cross-attention (entre
    ramas), con embeddings posicionales seno/coseno (1D para tiempo, 2D para
    frecuencia×tiempo)."""

    def __init__(self, dim, hidden_scale=4.0, num_heads=8, num_layers=5, dropout=0.0,
                 norm_in=True, norm_first=True, norm_out=True, max_period=10000.0,
                 layer_scale=True, gelu=True, weight_pos_embed=1.0):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = 0  # cross_first=False en htdemucs
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = 0

        self.norm_in = nn.LayerNorm(dim) if norm_in else nn.Identity()
        self.norm_in_t = nn.LayerNorm(dim) if norm_in else nn.Identity()

        activation = F.gelu if gelu else F.relu
        common = dict(d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,
                      activation=activation, norm_first=norm_first, norm_out=norm_out,
                      layer_scale=layer_scale, batch_first=True)

        self.layers = nn.ModuleList()
        self.layers_t = nn.ModuleList()
        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**common))
                self.layers_t.append(MyTransformerEncoderLayer(**common))
            else:
                self.layers.append(CrossTransformerEncoderLayer(**common))
                self.layers_t.append(CrossTransformerEncoderLayer(**common))

    def forward(self, x, xt):
        # Rama de frecuencia: aplana (Fr,T1) -> secuencia con T1 "fuera" y Fr
        # "dentro" (equivalente a rearrange 'b c fr t1 -> b (t1 fr) c' del
        # original), y le suma un embedding posicional 2D seno/coseno.
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, x.device, self.max_period)  # [1,C,Fr,T1]
        pos_emb_2d = pos_emb_2d.permute(0, 3, 2, 1).reshape(1, T1 * Fr, C)
        x = x.permute(0, 3, 2, 1).reshape(B, T1 * Fr, C)
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        # Rama temporal: aplana (C,T2) -> (T2,C) y le suma un embedding 1D.
        B2, C2, T2 = xt.shape
        xt = xt.permute(0, 2, 1)  # b t2 c
        # nota: el original consume aquí un random.randrange(sin_random_shift+1)
        # para decidir el desplazamiento del embedding; con sin_random_shift=0
        # (default de htdemucs) el resultado siempre es 0, pero SÍ avanza el
        # generador aleatorio global. Lo replicamos para que el "shift trick"
        # (que también usa random.randint) quede sincronizado con el original
        # cuando se fija una semilla — si no, ambos producen separaciones
        # igualmente válidas pero no bit-a-bit comparables al fijar la misma
        # semilla desde fuera.
        shift = random.randrange(self.sin_random_shift + 1)
        pos_emb = create_sin_embedding(T2, C2, shift=shift, device=x.device, max_period=self.max_period)
        pos_emb = pos_emb.reshape(T2, C2).unsqueeze(0)  # (T,1,C) -> (1,T,C)
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.reshape(B, T1, Fr, C).permute(0, 3, 2, 1)
        xt = xt.permute(0, 2, 1)
        return x, xt


# ---- STFT / iSTFT -----------------------------------------------------------

def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x, n_fft * (1 + pad), hop_length or n_fft // 4,
                   window=torch.hann_window(n_fft).to(x), win_length=n_fft,
                   normalized=True, center=True, return_complex=True, pad_mode="reflect")
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(z, n_fft, hop_length, window=torch.hann_window(win_length).to(z.real),
                     win_length=win_length, normalized=True, length=length, center=True)
    _, length = x.shape
    return x.view(*other, length)


# ---- HTDemucs (contenedor principal) ---------------------------------------

class HTDemucs(nn.Module):
    """Reimplementación de Hybrid Transformer Demucs v4 (modelo base "htdemucs").

    Solo implementa la ruta de ejecución que usa el checkpoint oficial por
    defecto: cac=True (complex-as-channels), wiener_iters=0 (sin Wiener
    filtering), multi_freqs=None, sin atención dispersa, embedding
    posicional "sin" (no CAPE), sin LSTM/atención local en el DConv. Estas
    son precisamente las opciones no usadas por el modelo "htdemucs" base
    (ver spec_stem_separator.md §7), así que no suponen ninguna pérdida de
    fidelidad frente al checkpoint que vamos a cargar.
    """

    def __init__(self, sources, audio_channels=2, channels=48, channels_time=None, growth=2,
                 nfft=4096, wiener_iters=0, end_iters=0, wiener_residual=False, cac=True,
                 depth=4, rewrite=True, multi_freqs=None, multi_freqs_depth=3,
                 freq_emb=0.2, emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2,
                 stride=4, context=1, context_enc=0, norm_starts=4, norm_groups=4,
                 dconv_mode=1, dconv_depth=2, dconv_comp=8, dconv_init=1e-3,
                 bottom_channels=0, t_layers=5, t_emb="sin", t_hidden_scale=4.0, t_heads=8,
                 t_dropout=0.0, t_max_positions=10000, t_norm_in=True, t_norm_in_group=False,
                 t_group_norm=False, t_norm_first=True, t_norm_out=True, t_max_period=10000.0,
                 t_weight_decay=0.0, t_lr=None, t_layer_scale=True, t_gelu=True,
                 t_weight_pos_embed=1.0, t_sin_random_shift=0,
                 # Parámetros de CAPE y atención dispersa: el checkpoint "htdemucs" base los
                 # trae con estos valores por defecto, pero son inertes salvo que t_emb="cape"
                 # o t_sparse_self_attn/t_sparse_cross_attn=True (ninguno de los dos es el caso
                 # en htdemucs base) — se aceptan y se ignoran, no se implementan.
                 t_cape_mean_normalize=True, t_cape_augment=True,
                 t_cape_glob_loc_scale=(5000.0, 1.0, 1.4), t_sparse_self_attn=False,
                 t_sparse_cross_attn=False, t_mask_type="diag", t_mask_random_seed=42,
                 t_sparse_attn_window=500, t_global_window=100, t_sparsity=0.95,
                 t_auto_sparsity=False, t_cross_first=False,
                 rescale=0.1, samplerate=44100, segment=10, use_train_segment=True,
                 **unsupported_kwargs):
        super().__init__()
        # Validación de que no se pide ninguna variante no implementada. Solo estas
        # opciones activan de verdad una rama de código no reimplementada (CAPE,
        # atención dispersa, multi-frecuencia, o kwargs realmente desconocidos); el
        # resto de parámetros t_cape_*/t_mask_*/t_sparse_* aceptados arriba son inertes
        # mientras estas banderas estén en su valor por defecto.
        if multi_freqs or t_sparse_self_attn or t_sparse_cross_attn or t_emb != "sin" \
                or t_group_norm or t_norm_in_group or t_auto_sparsity or unsupported_kwargs:
            raise NotImplementedError(
                "Esta reimplementación solo soporta la configuración por defecto de "
                "'htdemucs' base. Parámetros no soportados con valor no-default: "
                f"multi_freqs={multi_freqs}, t_sparse_self_attn={t_sparse_self_attn}, "
                f"t_sparse_cross_attn={t_sparse_cross_attn}, t_emb={t_emb}, "
                f"t_group_norm={t_group_norm}, t_norm_in_group={t_norm_in_group}, "
                f"t_auto_sparsity={t_auto_sparsity}, extra_desconocidos={unsupported_kwargs}"
            )
        assert wiener_iters == end_iters

        self.cac = cac
        self.audio_channels = audio_channels
        self.sources = sources

        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.wiener_residual = wiener_residual
        self.freq_emb = None

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin * 2 if cac else chin
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = dict(kernel_size=ker, stride=stri, freq=freq, pad=pad, norm=norm,
                      rewrite=rewrite, norm_groups=norm_groups,
                      dconv_kw=dict(depth=dconv_depth, compress=dconv_comp, init=dconv_init, gelu=True))
            kwt = dict(kw); kwt["freq"] = False; kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride; kwt["pad"] = True
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, dconv=bool(dconv_mode & 1), context=context_enc, **kw)
            if freq:
                tenc = HEncLayer(chin, chout, dconv=bool(dconv_mode & 1), context=context_enc,
                                  empty=last_freq, **kwt)
                self.tencoder.append(tenc)
            self.encoder.append(enc)

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * 2 if cac else chin

            dec = HDecLayer(chout_z, chin_z, dconv=bool(dconv_mode & 2), last=index == 0,
                             context=context, **kw_dec)
            if freq:
                tdec = HDecLayer(chout, chin, dconv=bool(dconv_mode & 2), empty=last_freq,
                                  last=index == 0, context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            self._rescale_module(reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = nn.Conv1d(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        self.crosstransformer = None
        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels, hidden_scale=t_hidden_scale, num_heads=t_heads,
                num_layers=t_layers, dropout=t_dropout, norm_in=t_norm_in,
                norm_first=t_norm_first, norm_out=t_norm_out, max_period=t_max_period,
                layer_scale=t_layer_scale, gelu=t_gelu, weight_pos_embed=t_weight_pos_embed,
            )

    @staticmethod
    def _rescale_conv(conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference) ** 0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    def _rescale_module(self, reference):
        for sub in self.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                self._rescale_conv(sub, reference)

    # ---- STFT helpers --------------------------------------------------

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = spectro(x, nfft, hl)[..., :-1, :]
        z = z[..., 2:2 + le]
        return z

    def _ispec(self, z, length=None):
        hl = self.hop_length
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad:pad + length]
        return x

    def _magnitude(self, z):
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, m):
        # cac=True, wiener_iters=0 -> reconstrucción directa (sin Wiener).
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def valid_length(self, length: int):
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(f"Longitud {length} mayor que la de entrenamiento {training_length}")
        return training_length

    def forward(self, mix):
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                mix = F.pad(mix, (0, training_length - length_pre_pad))

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag
        B, C, Fq, T = x.shape

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer is not None:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = self.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt)
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = self.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0 and len(lengths_t) == 0 and len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(x)
        training_length = int(self.segment * self.samplerate)
        out_length = training_length if self.use_train_segment else length
        x = self._ispec(zout, out_length)

        xt = xt.view(B, S, -1, out_length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x


# ============================================================================
# 2. MOTOR DE INFERENCIA — chunking por segmentos con overlap-add triangular
#    y "shift trick" (test-time augmentation). Ver spec_stem_separator.md §3.
# ============================================================================

def _center_trim(tensor, target_length):
    """Recorta `tensor` simétricamente por el centro hasta `target_length`
    muestras (si sobra un número impar, la muestra extra se quita por la
    derecha) — así es como el motor original recorta cada segmento de vuelta
    a su longitud "válida" tras haberlo rellenado con ceros para completar
    el tamaño de entrenamiento."""
    delta = tensor.shape[-1] - target_length
    if delta < 0:
        raise ValueError(f"tensor ({tensor.shape[-1]}) más corto que target ({target_length})")
    if delta:
        tensor = tensor[..., delta // 2: tensor.shape[-1] - (delta - delta // 2)]
    return tensor


def _extract_padded_window(full_tensor, offset, valid_length, target_length):
    """Extrae una ventana de `target_length` muestras centrada en el hueco
    [offset, offset+valid_length) de `full_tensor`, usando audio real como
    contexto a los lados y rellenando con ceros solo en los bordes reales de
    la pista. Así el modelo ve tanto contexto real como sea posible en vez
    de silencio artificial — igual que TensorChunk.padded() en el original."""
    total_length = full_tensor.shape[-1]
    delta = target_length - valid_length
    start = offset - delta // 2
    end = start + target_length
    correct_start = max(0, start)
    correct_end = min(total_length, end)
    pad_left = correct_start - start
    pad_right = end - correct_end
    window = full_tensor[..., correct_start:correct_end]
    if pad_left or pad_right:
        window = F.pad(window, (pad_left, pad_right))
    return window


def _apply_padded(model, full_tensor, offset, valid_length, segment_len):
    """Aplica el modelo a la ventana de `segment_len` muestras centrada en
    [offset, offset+valid_length) DENTRO de `full_tensor`, y recorta el
    resultado de vuelta al centro, a `valid_length` muestras. `offset` es
    siempre absoluto respecto a `full_tensor` (nunca se recorta el tensor de
    referencia de antemano), para poder tirar de contexto real incluso más
    allá de una región lógica anidada — igual que la composición de
    TensorChunk en el original."""
    window = _extract_padded_window(full_tensor, offset, valid_length, segment_len)
    with torch.no_grad():
        out = model(window[None])[0]
    return _center_trim(out, valid_length)


def _apply_chunked(model, full_tensor, offset, length, segment_len, overlap=0.25,
                    transition_power=1.0, progress=True):
    """Núcleo de chunking (sin shift trick): trocea la región lógica
    [offset, offset+length) de `full_tensor` en segmentos solapados de
    `segment_len` muestras y recompone con una ventana triangular
    (overlap-add), exactamente como demucs.apply.apply_model. `full_tensor`
    puede ser más grande que `length` (p.ej. el buffer ya desplazado y
    reforzado con zero-padding del "shift trick"); `offset` siempre es
    absoluto respecto a él."""
    if length <= segment_len:
        return _apply_padded(model, full_tensor, offset, length, segment_len)

    stride = int((1 - overlap) * segment_len)
    local_offsets = list(range(0, length, stride))
    S = len(model.sources)
    device = full_tensor.device
    out_sum = torch.zeros(S, model.audio_channels, length, device=device)
    weight_sum = torch.zeros(length, device=device)

    # Ventana triangular: máxima en el centro, mínima (no cero) en los bordes.
    half = segment_len // 2
    window = torch.cat([
        torch.arange(1, half + 1, device=device, dtype=torch.float32),
        torch.arange(segment_len - half, 0, -1, device=device, dtype=torch.float32),
    ])
    window = (window / window.max()) ** transition_power

    for i, local_offset in enumerate(local_offsets):
        valid = min(segment_len, length - local_offset)
        chunk_out = _apply_padded(model, full_tensor, offset + local_offset, valid, segment_len)
        w = window[:valid]
        out_sum[..., local_offset:local_offset + valid] += w * chunk_out
        weight_sum[local_offset:local_offset + valid] += w
        if progress:
            print(f"\r  segmento {i + 1}/{len(local_offsets)} "
                  f"({local_offset / model.samplerate:5.1f}s / {length / model.samplerate:5.1f}s)",
                  end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)

    assert weight_sum.min() > 0
    out_sum /= weight_sum
    return out_sum

def apply_model(model, mix, segment=None, overlap=0.25, shifts=1, progress=True):
    """Separa una mezcla de longitud arbitraria. `mix` tiene forma (canales, muestras).

    Estructura idéntica al motor original: el "shift trick" envuelve al
    chunking (se desplaza toda la pista, se trocea/recompone la versión
    desplazada, y se deshace el desplazamiento), no al revés.
    """
    model.eval()
    segment_len = int((segment or model.segment) * model.samplerate)
    device = mix.device
    length = mix.shape[-1]

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        padded = F.pad(mix, (max_shift, max_shift))  # longitud total: length + 2*max_shift
        out = None
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            # Región lógica de longitud (length + max_shift - offset), con
            # extremo derecho FIJO en (length + max_shift) dentro de `padded`
            # — igual que TensorChunk(padded_mix, offset, length+max_shift-offset).
            # Operamos sobre el buffer `padded` COMPLETO (sin recortarlo antes),
            # para que el chunking interno pueda tirar de contexto real más
            # allá de esta región si lo necesita, igual que en el original.
            shift_region_len = length + max_shift - offset
            shifted_out = _apply_chunked(model, padded, offset, shift_region_len,
                                          segment_len, overlap=overlap, progress=progress)
            y = shifted_out[..., max_shift - offset:]
            out = y if out is None else out + y
        return out / shifts
    else:
        return _apply_chunked(model, mix, 0, length, segment_len, overlap=overlap, progress=progress)


# ============================================================================
# 3. CARGA DE CHECKPOINT — con "shim" de compatibilidad para leer el .th
#    oficial sin tener el paquete `demucs` instalado (ver spec §8 y análisis
#    de states.py/repo.py: el checkpoint serializa la CLASE real vía pickle,
#    así que interceptamos esa referencia para que apunte a nuestra propia
#    clase HTDemucs, arquitectónicamente idéntica).
# ============================================================================

def _install_compat_shim():
    """Registra módulos falsos 'demucs' / 'demucs.htdemucs' en sys.modules
    para que torch.load pueda deserializar el checkpoint oficial (que
    referencia demucs.htdemucs.HTDemucs) sin necesitar el paquete real."""
    if "demucs" in sys.modules and getattr(sys.modules.get("demucs.htdemucs"), "_stem_separator_shim", False):
        return
    demucs_mod = types.ModuleType("demucs")
    htdemucs_mod = types.ModuleType("demucs.htdemucs")
    htdemucs_mod.HTDemucs = HTDemucs
    htdemucs_mod._stem_separator_shim = True
    demucs_mod.htdemucs = htdemucs_mod
    sys.modules.setdefault("demucs", demucs_mod)
    sys.modules["demucs.htdemucs"] = htdemucs_mod


def load_checkpoint(path, device="cpu"):
    """Carga un checkpoint .th (formato oficial demucs: dict con klass/args/
    kwargs/state) y devuelve un HTDemucs listo para inferencia."""
    _install_compat_shim()
    package = torch.load(path, map_location=device, weights_only=False)
    if isinstance(package, dict) and "state" in package and "kwargs" in package:
        kwargs = dict(package["kwargs"])
        args = package.get("args", ())
        model = HTDemucs(*args, **kwargs) if args or kwargs else HTDemucs(sources=SOURCES)
        state = package["state"]
        # El checkpoint oficial puede guardar los pesos en half precision.
        state = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                 for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    elif isinstance(package, dict):
        # Un state_dict plano guardado por nosotros mismos (--random-weights de prueba).
        model = HTDemucs(sources=SOURCES)
        model.load_state_dict(package, strict=True)
    else:
        raise ValueError(f"Formato de checkpoint no reconocido en {path}")
    model.to(device)
    model.eval()
    return model


def build_random_model(device="cpu"):
    """Construye un HTDemucs con pesos aleatorios (misma arquitectura e
    hiperparámetros que el checkpoint oficial). Solo para pruebas: la
    separación resultante no tiene ningún significado musical."""
    model = HTDemucs(sources=SOURCES)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# 4. DESCARGA DEL CHECKPOINT OFICIAL
# ============================================================================

def download_checkpoint(url=OFFICIAL_MODEL_URL, dest=DEFAULT_MODEL_PATH):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {url}")
    print(f"  -> {dest}")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {pct:3d}%  ({mb:6.1f} / {total_mb:6.1f} MB)", end="", file=sys.stderr)

    tmp = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_report)
        print(file=sys.stderr)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


# ============================================================================
# 5. E/S DE AUDIO — WAV vía soundfile únicamente (sin ffmpeg/torchaudio)
# ============================================================================

def load_wav(path, target_sr=SAMPLERATE, target_channels=AUDIO_CHANNELS):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, ch)
    wav = torch.from_numpy(data.T)  # (ch, frames)

    if wav.shape[0] == 1:
        wav = wav.repeat(target_channels, 1)
    elif wav.shape[0] != target_channels:
        wav = wav.mean(dim=0, keepdim=True).repeat(target_channels, 1)

    if sr != target_sr:
        wav = _resample_linear(wav, sr, target_sr)
    return wav


def _resample_linear(wav, sr_in, sr_out):
    """Resampling por interpolación lineal (dependencia mínima: solo torch).
    Menor calidad que un resampler polifásico/sinc, aceptable dado que la
    mayoría de entradas ya estarán a 44100 Hz."""
    if sr_in == sr_out:
        return wav
    n_in = wav.shape[-1]
    n_out = int(round(n_in * sr_out / sr_in))
    wav = wav.unsqueeze(0)  # (1, C, T)
    wav = F.interpolate(wav, size=n_out, mode="linear", align_corners=False)
    return wav[0]


def save_wav(path, wav, sr=SAMPLERATE):
    import soundfile as sf
    data = wav.detach().cpu().clamp(-1, 1).numpy().T  # (frames, ch)
    sf.write(str(path), data, sr)


# ============================================================================
# 6. SUBCOMANDOS
# ============================================================================

def cmd_install(args):
    dest = Path(args.output) if args.output else DEFAULT_MODEL_PATH
    if dest.exists() and not args.force:
        print(f"Ya existe {dest} (usa --force para volver a descargar).")
        return 0
    download_checkpoint(dest=dest)
    print("Listo.")
    return 0


def cmd_info(args):
    if args.model_path:
        model = load_checkpoint(args.model_path, device="cpu")
        source = str(args.model_path)
    elif args.random_weights or not Path(DEFAULT_MODEL_PATH).exists():
        model = build_random_model(device="cpu")
        source = "(arquitectura por defecto, sin checkpoint cargado)"
    else:
        model = load_checkpoint(DEFAULT_MODEL_PATH, device="cpu")
        source = str(DEFAULT_MODEL_PATH)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint:        {source}")
    print(f"Fuentes:           {model.sources}")
    print(f"Sample rate:       {model.samplerate} Hz")
    print(f"Canales:           {model.audio_channels}")
    print(f"Segmento entrenam.:{model.segment} s ({int(model.segment * model.samplerate)} muestras)")
    print(f"Profundidad:       {model.depth}")
    print(f"Canales base:      {model.channels}")
    print(f"nfft:              {model.nfft} (hop={model.hop_length})")
    print(f"Transformer:       {model.crosstransformer.num_layers} capas cross-dominio"
          if model.crosstransformer else "Transformer:       (ninguno)")
    print(f"Parámetros:        {n_params:,}")
    return 0


def cmd_separate(args):
    device = args.device
    if args.random_weights:
        print("[!] --random-weights: usando pesos aleatorios, la separación NO será musicalmente válida "
              "(solo sirve para probar que el motor funciona).", file=sys.stderr)
        model = build_random_model(device=device)
    else:
        model_path = args.model_path or DEFAULT_MODEL_PATH
        if not Path(model_path).exists():
            print(f"No se encuentra el checkpoint en {model_path}.", file=sys.stderr)
            print("Ejecuta antes: python stem_separator.py install", file=sys.stderr)
            return 1
        model = load_checkpoint(model_path, device=device)

    if args.segment:
        model.segment = args.segment

    wav = load_wav(args.input).to(device)
    print(f"Entrada: {args.input}  ({wav.shape[-1] / SAMPLERATE:.1f}s, {wav.shape[0]} canales @ {SAMPLERATE}Hz)")

    out = apply_model(model, wav, segment=args.segment, overlap=args.overlap,
                       shifts=args.shifts, progress=not args.quiet)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    wanted = set(args.stems) if args.stems else set(model.sources)
    unknown = wanted - set(model.sources)
    if unknown:
        raise ValueError(f"Fuentes desconocidas: {unknown}. Disponibles: {model.sources}")

    written = []
    if args.two_stems:
        if args.two_stems not in model.sources:
            raise ValueError(f"--two-stems debe ser una de {model.sources}")
        idx = model.sources.index(args.two_stems)
        main = out[idx]
        rest = out.sum(dim=0) - main
        p1 = out_dir / f"{stem}.stem_{args.two_stems}.wav"
        p2 = out_dir / f"{stem}.stem_no_{args.two_stems}.wav"
        save_wav(p1, main)
        save_wav(p2, rest)
        written = [p1, p2]
    else:
        for i, src in enumerate(model.sources):
            if src not in wanted:
                continue
            p = out_dir / f"{stem}.stem_{src}.wav"
            save_wav(p, out[i])
            written.append(p)

    for p in written:
        print(f"  -> {p}")
    return 0


# ============================================================================
# 7. CLI
# ============================================================================

def build_parser():
    banner = __doc__
    parser = argparse.ArgumentParser(
        prog="stem_separator.py",
        description=banner,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="descarga el checkpoint preentrenado oficial")
    p_install.add_argument("--output", type=str, default=None, help="ruta de destino del .th")
    p_install.add_argument("--force", action="store_true", help="volver a descargar aunque ya exista")
    p_install.set_defaults(func=cmd_install)

    p_sep = sub.add_parser("separate", help="separa un WAV en stems")
    p_sep.add_argument("input", type=str, help="fichero WAV de entrada")
    p_sep.add_argument("--stems", nargs="+", default=None, choices=SOURCES,
                        help="subconjunto de fuentes a exportar (default: las 4)")
    p_sep.add_argument("--two-stems", type=str, default=None, choices=SOURCES,
                        help="modo karaoke: exporta esta fuente y 'no_<fuente>' (suma del resto)")
    p_sep.add_argument("--shifts", type=int, default=1, help="nº de desplazamientos promediados (default: 1)")
    p_sep.add_argument("--overlap", type=float, default=0.25, help="solape entre segmentos, 0-1 (default: 0.25)")
    p_sep.add_argument("--segment", type=float, default=None, help="duración de segmento en segundos")
    p_sep.add_argument("--model-path", type=str, default=None, help="ruta al checkpoint .th")
    p_sep.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p_sep.add_argument("--output-dir", type=str, default=None)
    p_sep.add_argument("--random-weights", action="store_true",
                        help="no cargar checkpoint; pesos aleatorios (solo para pruebas del motor)")
    p_sep.add_argument("--quiet", action="store_true", help="no mostrar barra de progreso")
    p_sep.set_defaults(func=cmd_separate)

    p_info = sub.add_parser("info", help="inspecciona un checkpoint o la arquitectura por defecto")
    p_info.add_argument("--model-path", type=str, default=None)
    p_info.add_argument("--random-weights", action="store_true")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
