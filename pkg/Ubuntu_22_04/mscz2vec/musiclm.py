#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MUSICLM  v1.0                                        ║
║         Generación de música a partir de texto — pipeline completo           ║
║                                                                              ║
║  Implementación de MusicLM (Agostinelli et al., 2023) en PyTorch.           ║
║  Combina MuLaN (aprendizaje contrastivo audio-texto, estilo CLIP) con        ║
║  AudioLM para generar música a partir de descripciones en lenguaje natural.  ║
║                                                                              ║
║  FLUJO COMPLETO:                                                             ║
║  [1] TRAIN-MULAN  — entrena el modelo de embedding audio↔texto              ║
║  [2] QUANTIZE     — crea el cuantizador residual sobre MuLaN entrenado       ║
║  [3] EMBED        — extrae embeddings de audio o texto a .json               ║
║  [4] GENERATE     — genera música a partir de una descripción de texto       ║
║  [5] INSPECT      — diagnóstico de un checkpoint guardado                    ║
║  [6] ROUND-TRIP   — audio → embed → similitud (diagnóstico, sin generar)     ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare        — construye un dataset de pares (wav, texto) desde carpeta ║
║    train-mulan    — entrena MuLaN con aprendizaje contrastivo                ║
║    quantize       — ajusta el cuantizador vectorial residual sobre MuLaN     ║
║    embed          — extrae embeddings de un fichero de audio o texto         ║
║    generate       — genera música a partir de texto (requiere AudioLM)       ║
║    inspect        — muestra arquitectura y estado de un checkpoint           ║
║    round-trip     — diagnóstico: audio → embedding → similitud con texto     ║
║                                                                              ║
║  USO RÁPIDO:                                                                 ║
║    # 1. Preparar dataset                                                     ║
║    python musiclm.py prepare --audio-dir ./wavs --manifest manifest.csv     ║
║                                                                              ║
║    # 2. Entrenar MuLaN                                                       ║
║    python musiclm.py train-mulan --manifest manifest.csv --steps 50000      ║
║                                                                              ║
║    # 3. Ajustar cuantizador                                                  ║
║    python musiclm.py quantize --mulan mulan.pt --manifest manifest.csv      ║
║                                                                              ║
║    # 4. Generar música                                                       ║
║    python musiclm.py generate "una pieza de piano melancólica"               ║
║        --mulan mulan.pt --quantizer quantizer.pt                             ║
║                                                                              ║
║    # 5. Diagnóstico                                                          ║
║    python musiclm.py round-trip audio.wav "lluvia suave"                    ║
║        --mulan mulan.pt                                                      ║
║                                                                              ║
║  OPCIONES COMUNES:                                                           ║
║    --mulan FILE        Checkpoint de MuLaN (.pt)                             ║
║    --quantizer FILE    Checkpoint del cuantizador residual (.pt)             ║
║    --audio-lm FILE     Checkpoint de AudioLM (.pt) — solo para generate      ║
║    --output FILE       Fichero de salida (wav, json, o pt según comando)     ║
║    --device DEVICE     cpu | cuda | mps (default: auto)                      ║
║    --dim N             Dimensión del transformer (default: 512)              ║
║    --depth N           Capas del transformer (default: 6)                    ║
║    --heads N           Cabezas de atención (default: 8)                      ║
║    --dim-latent N      Dimensión del espacio latente MuLaN (default: 128)    ║
║    --batch-size N      Tamaño de lote (default: 16)                          ║
║    --lr F              Learning rate (default: 3e-4)                         ║
║    --steps N           Pasos de entrenamiento (default: 100000)              ║
║    --seed N            Semilla aleatoria (default: 42)                       ║
║    --verbose           Informe detallado de operaciones                       ║
║                                                                              ║
║  OPCIONES DE train-mulan:                                                    ║
║    --manifest FILE     CSV con columnas: path,text (requerido)               ║
║    --valid-frac F      Fracción de validación (default: 0.05)               ║
║    --save-every N      Guardar checkpoint cada N pasos (default: 1000)       ║
║    --results-dir DIR   Carpeta de checkpoints (default: ./results_mulan)     ║
║    --use-lion          Usar optimizador Lion en lugar de Adam                ║
║    --decoupled-cl      Usar decoupled contrastive learning                   ║
║    --sigmoid-cl        Usar sigmoid contrastive loss (SigLIP)                ║
║    --hierarchical-cl   Añadir pérdida contrastiva jerárquica multi-capa     ║
║                                                                              ║
║  OPCIONES DE quantize:                                                       ║
║    --rq-quantizers N   Número de cuantizadores residuales (default: 8)      ║
║    --codebook-size N   Tamaño del codebook (default: 1024)                   ║
║    --namespaces S      Namespaces separados por coma (default:               ║
║                        semantic,coarse,fine)                                  ║
║    --conditioning-dims D  Dimensiones de condicionamiento por namespace      ║
║                           (default: 1024,1024,1024)                          ║
║                                                                              ║
║  OPCIONES DE generate:                                                       ║
║    --num-samples N     Candidatos a generar (elige el más similar, default:4)║
║    --output FILE       Fichero wav de salida (default: generated.wav)        ║
║    --stub              Usar AudioLM stub (no requiere audiolm-pytorch)       ║
║                                                                              ║
║  OPCIONES DE embed:                                                          ║
║    --mode audio|text   Qué modalidad embeber (default: auto desde extensión) ║
║    --output FILE       JSON con el vector de embedding (default: embed.json) ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch torchaudio einops beartype vector-quantize-pytorch      ║
║                                                                              ║
║  DEPENDENCIAS OPCIONALES:                                                    ║
║    pip install x-clip          → tokenizador de texto (alternativa: basic)  ║
║    pip install lion-pytorch    → optimizador Lion para train-mulan           ║
║    pip install accelerate      → entrenamiento multi-GPU para train-mulan    ║
║    pip install audiolm-pytorch → comando generate en modo completo           ║
║                                                                              ║
║  REFERENCIAS:                                                                ║
║    Agostinelli et al. (2023) — MusicLM: Generating Music From Text          ║
║    Huang et al. (2022)       — MuLan: A Joint Embedding of Music and NL     ║
║    Zhai et al. (2023)        — Sigmoid Loss for Language Image Pre-Training  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import argparse
import textwrap
import csv
from pathlib import Path
from functools import wraps, partial
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS OPCIONALES — se comprueban en cada comando que los necesita
# ══════════════════════════════════════════════════════════════════════════════

def _require(package: str, pip_name: str | None = None, fatal: bool = True):
    """Importa un paquete y aborta con mensaje útil si no está disponible."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package.replace("_", "-")
        msg = f"  [error] Paquete '{package}' no encontrado. Instálalo con:\n  pip install {name}"
        if fatal:
            print(msg)
            sys.exit(1)
        return None

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def first(it):
    return it[0]

def _auto_device() -> str:
    torch = _require("torch", fatal=False)
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _print_header(title: str):
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  MUSICLM v{VERSION}  —  {title}")
    print(sep)

def _step(n: int, total: int, msg: str):
    print(f"  [{n}/{total}] {msg}")

def _ok(msg: str):
    print(f"    ✓  {msg}")

def _warn(msg: str):
    print(f"    ⚠  {msg}")

def _info(msg: str):
    print(f"    ◆  {msg}")

# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZADOR DE TEXTO
#  Intenta usar x-clip; si no está, usa un tokenizador básico de fallback.
# ══════════════════════════════════════════════════════════════════════════════

class _BasicTokenizer:
    """Tokenizador de emergencia: vocabulario BPE muy simple por caracteres."""
    vocab_size = 4096

    def tokenize(self, texts: list[str]):
        torch = _require("torch")
        max_len = 256
        out = []
        for t in texts:
            ids = [ord(c) % self.vocab_size for c in t[:max_len]]
            ids += [0] * (max_len - len(ids))
            out.append(ids)
        return torch.tensor(out, dtype=torch.long)


def _get_tokenizer():
    """Devuelve el tokenizador de x-clip o el básico como fallback."""
    xcl = _require("x_clip", pip_name="x-clip", fatal=False)
    if xcl is not None:
        try:
            from x_clip.tokenizer import tokenizer
            return tokenizer
        except Exception:
            pass
    _warn("x-clip no disponible: usando tokenizador básico (vocabulario reducido).")
    return _BasicTokenizer()


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUES COMUNES DE ARQUITECTURA
# ══════════════════════════════════════════════════════════════════════════════

def _build_modules():
    """
    Construye y devuelve todos los bloques de arquitectura.
    Se llama en diferido para no importar torch en el nivel de módulo
    (permite que `--help` funcione sin torch instalado).
    """
    torch    = _require("torch")
    nn       = torch.nn
    F        = torch.nn.functional
    einops   = _require("einops")
    from einops import rearrange, repeat, reduce, pack, unpack
    from einops.layers.torch import Rearrange
    from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking
    import torch.distributed as dist

    # ── Funciones tensoriales ─────────────────────────────────────────────────

    def log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def l2norm(t):
        return F.normalize(t, p=2, dim=-1)

    def matrix_diag(t):
        device = t.device
        i, j = t.shape[-2:]
        num_diag_el = min(i, j)
        i_range = torch.arange(i, device=device)
        j_range = torch.arange(j, device=device)
        diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
        diag_el = t.masked_select(diag_mask)
        return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)

    def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
        _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
        y, x = torch.meshgrid(torch.arange(h, device=device),
                               torch.arange(w, device=device), indexing='ij')
        assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
        omega = 1. / (temperature ** omega)
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return rearrange(pe.type(dtype), '(h w) d -> h w d', h=h, w=w)

    def round_down_nearest_multiple(n, divisor):
        return n // divisor * divisor

    def _once(fn):
        called = False
        @wraps(fn)
        def inner(x):
            nonlocal called
            if called:
                return
            called = True
            return fn(x)
        return inner

    print_once = _once(print)

    # ── AllGather (entrenamiento distribuido) ─────────────────────────────────

    class AllGather(nn.Module):
        def __init__(self, dim=2, all_reduce_grads=False):
            super().__init__()
            self.dim = dim
            self.all_reduce_grads = all_reduce_grads
            self.is_distributed = dist.is_available() and dist.is_initialized()

        def forward(self, x, mask=None):
            if not self.is_distributed:
                return x, None
            # distribuido: recoger desde todos los procesos
            x = x.contiguous()
            out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(out, x)
            out = torch.cat(out, dim=self.dim)
            return out, None

    # ── Bloques de capas ──────────────────────────────────────────────────────

    class LayerNorm(nn.Module):
        def __init__(self, dim, scale=True):
            super().__init__()
            self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None
            self.register_buffer('gamma', torch.ones(dim), persistent=False)
            self.register_buffer('beta',  torch.zeros(dim), persistent=False)

        def forward(self, x):
            return F.layer_norm(x, x.shape[-1:],
                                default(self.learned_gamma, self.gamma), self.beta)

    class GEGLU(nn.Module):
        def forward(self, x):
            x, gate = x.chunk(2, dim=-1)
            return F.gelu(gate) * x

    def FeedForward(dim, mult=4, dropout=0.):
        dim_hidden = int(dim * mult * 2 / 3)
        return nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim_hidden * 2, bias=False),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias=False),
        )

    class Attention(nn.Module):
        def __init__(self, dim, causal=False, dim_head=64, heads=8,
                     dropout=0., scale=8):
            super().__init__()
            self.heads  = heads
            self.scale  = scale
            self.causal = causal
            inner_dim   = dim_head * heads

            self.norm = LayerNorm(dim)
            self.attn_dropout = nn.Dropout(dropout)
            self.to_q  = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

            # QK RMSNorm — estabiliza entrenamiento de modelos grandes
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim, bias=False),
                nn.Dropout(dropout),
            )

        def forward(self, x, rel_pos_bias=None, mask=None):
            b, n, _, device = *x.shape, x.device
            x = self.norm(x)
            q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                          (q, k, v))
            # QK normalization
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

            from einops import einsum
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

            if exists(rel_pos_bias):
                sim = sim + rel_pos_bias
            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype=torch.bool,
                                         device=device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = self.attn_dropout(sim.softmax(dim=-1))
            out  = einsum(attn, v, 'b h i j, b h j d -> b h i d')
            return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

    class Transformer(nn.Module):
        def __init__(self, dim, depth, dim_head=64, heads=8,
                     attn_dropout=0., ff_mult=4, ff_dropout=0.):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    Attention(dim=dim, dim_head=dim_head, heads=heads,
                              dropout=attn_dropout),
                    FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ])
                for _ in range(depth)
            ])

        def forward(self, x, rel_pos_bias=None, mask=None, return_all_layers=False):
            layers = []
            for attn, ff in self.layers:
                x = attn(x, rel_pos_bias=rel_pos_bias, mask=mask) + x
                x = ff(x) + x
                layers.append(x)
            if not return_all_layers:
                return x
            return x, torch.stack(layers[:-1])

    # ── Pérdidas contrastivas ─────────────────────────────────────────────────

    class SoftmaxContrastiveLearning(nn.Module):
        def __init__(self, *, layers=1, decoupled_contrastive_learning=False,
                     init_temp=10):
            super().__init__()
            self.temperatures = nn.Parameter(
                torch.ones(layers, 1, 1) * math.log(init_temp))
            self.decoupled = decoupled_contrastive_learning
            self.all_gather = AllGather(dim=2)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, audio_latents, text_latents):
            if audio_latents.ndim == 2:
                audio_latents = rearrange(audio_latents, '... -> 1 ...')
            if text_latents.ndim == 2:
                text_latents  = rearrange(text_latents,  '... -> 1 ...')

            batch = audio_latents.shape[1]
            if self.all_gather.is_distributed:
                latents = torch.stack((audio_latents, text_latents))
                latents, _ = self.all_gather(latents)
                audio_latents, text_latents = latents

            from einops import einsum as eins
            sims = eins(audio_latents, text_latents, 'l i d, l j d -> l i j')
            sims = sims * self.temperatures.exp()
            cosine_sims_exp = sims.exp()
            numerator = matrix_diag(cosine_sims_exp)

            if self.decoupled:
                eye = torch.eye(batch, device=self.device, dtype=torch.bool)
                cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

            denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
            denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')
            cl_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))
            return reduce(cl_loss, 'l n -> l', 'mean').sum()

    class SigmoidContrastiveLearning(nn.Module):
        """Sigmoid loss (SigLIP) — https://arxiv.org/abs/2303.15343"""
        def __init__(self, *, layers=1, init_temp=10, init_bias=-10):
            super().__init__()
            self.temperatures = nn.Parameter(
                torch.ones(layers, 1, 1) * math.log(init_temp))
            self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)
            self.all_gather = AllGather(dim=1, all_reduce_grads=True)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, audio_latents, text_latents):
            if audio_latents.ndim == 2:
                audio_latents = rearrange(audio_latents, '... -> 1 ...')
            if text_latents.ndim == 2:
                text_latents  = rearrange(text_latents,  '... -> 1 ...')

            text_latents, rank_sizes = self.all_gather(text_latents)
            n = text_latents.shape[1]
            from einops import einsum as eins
            sims   = eins(audio_latents, text_latents, 'l i d, l j d -> l i j')
            sims   = sims * self.temperatures.exp() + self.bias
            labels = torch.eye(n, device=self.device)

            if exists(rank_sizes):
                labels_by_ranks = labels.split(rank_sizes.tolist(), dim=0)
                labels = labels_by_ranks[dist.get_rank()]

            labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)
            return -F.logsigmoid(labels * sims).sum() / n

    def _interspersed_indices(layers, total_layers):
        import torch
        assert total_layers >= layers
        step = total_layers / layers
        return (torch.arange(0, layers) * step).floor().long()

    class MultiLayerContrastiveLoss(nn.Module):
        def __init__(self, *, audio_dim, text_dim, dim_latent, layers,
                     decoupled_contrastive_learning=False,
                     sigmoid_contrastive_loss=False):
            super().__init__()
            self.layers = layers
            self.audio_norm  = LayerNorm(audio_dim, scale=False)
            self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
            self.audio_latent_weight = nn.Parameter(
                torch.randn(layers, audio_dim, dim_latent))
            self.audio_latent_bias  = nn.Parameter(
                torch.randn(layers, 1, dim_latent))
            self.text_norm  = LayerNorm(text_dim, scale=False)
            self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
            self.text_latent_weight = nn.Parameter(
                torch.randn(layers, text_dim, dim_latent))
            self.text_latent_bias   = nn.Parameter(
                torch.randn(layers, 1, dim_latent))

            klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else \
                    partial(SoftmaxContrastiveLearning,
                            decoupled_contrastive_learning=decoupled_contrastive_learning)
            self.contrast = klass(layers=layers)

        def forward(self, *, audio_layers, text_layers):
            from einops import einsum as eins
            audio_gap    = reduce(audio_layers, 'l b n d -> l b d', 'mean')
            audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
            audio_latents = eins(audio_embeds, self.audio_latent_weight,
                                  'l b d, l d e -> l b e') + self.audio_latent_bias
            audio_latents = l2norm(audio_latents)

            text_cls = text_layers[:, :, 0]
            text_embeds = self.text_norm(text_cls) * self.text_gamma
            text_latents = eins(text_embeds, self.text_latent_weight,
                                 'l b d, l d e -> l b e') + self.text_latent_bias
            text_latents = l2norm(text_latents)

            return self.contrast(audio_latents, text_latents)

    # ── Encoders principales ──────────────────────────────────────────────────

    def _pair(t):
        return (t, t) if not isinstance(t, tuple) else t

    class AudioSpectrogramTransformer(nn.Module):
        """
        Encoder de audio basado en Vision Transformer sobre espectrogramas.
        Referencia: https://arxiv.org/abs/2104.01778
        """
        def __init__(self, dim, depth, patch_size=16, dim_head=64, heads=8,
                     attn_dropout=0., ff_mult=4, ff_dropout=0.,
                     accept_spec=False, accept_spec_time_first=True,
                     spec_n_fft=128, spec_power=2, spec_win_length=24,
                     spec_hop_length=None, spec_pad=0, spec_center=True,
                     spec_pad_mode='reflect',
                     spec_aug_stretch_factor=0.8,
                     spec_aug_freq_mask=80, spec_aug_time_mask=80,
                     patch_dropout_prob=0.25):
            super().__init__()
            self.dim   = dim
            self.depth = depth

            self.patch_size = _pair(patch_size)
            patch_input_dim = self.patch_size[0] * self.patch_size[1]

            self.to_patch_tokens = nn.Sequential(
                Rearrange('b (h p1) (w p2) -> b h w (p1 p2)',
                          p1=self.patch_size[0], p2=self.patch_size[1]),
                nn.LayerNorm(patch_input_dim),
                nn.Linear(patch_input_dim, dim),
                nn.LayerNorm(dim),
            )

            self.accept_spec = accept_spec
            self.accept_spec_time_first = accept_spec_time_first

            self.spec = Spectrogram(
                n_fft=spec_n_fft, power=spec_power, win_length=spec_win_length,
                hop_length=spec_hop_length, pad=spec_pad, center=spec_center,
                pad_mode=spec_pad_mode,
            )
            self.aug = nn.Sequential(
                TimeStretch(spec_aug_stretch_factor, fixed_rate=True),
                FrequencyMasking(freq_mask_param=spec_aug_freq_mask),
                TimeMasking(time_mask_param=spec_aug_time_mask),
            )

            self.transformer = Transformer(
                dim=dim, depth=depth, dim_head=dim_head, heads=heads,
                attn_dropout=attn_dropout, ff_mult=ff_mult, ff_dropout=ff_dropout,
            )
            self.norm = LayerNorm(dim)
            self.patch_dropout_prob = patch_dropout_prob

            # MLP de sesgo posicional dinámico 2D
            mlp_hidden_dim = dim // 4
            self.dynamic_pos_bias_mlp = nn.Sequential(
                nn.Linear(2, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, heads),
                Rearrange('... i j h -> ... h i j'),
            )

        def forward(self, x, force_no_patch_dropout=False, return_all_layers=False):
            batch, device = x.shape[0], x.device
            assert (self.accept_spec and x.ndim == 3) or \
                   (not self.accept_spec and x.ndim == 2)

            if self.accept_spec and self.accept_spec_time_first:
                x = rearrange(x, 'b t f -> b f t')
            if not self.accept_spec:
                x = self.spec(x)
            if self.training:
                x = self.aug(x)

            height, width = x.shape[-2:]
            ph, pw = self.patch_size
            rh = round_down_nearest_multiple(height, ph)
            rw = round_down_nearest_multiple(width,  pw)

            if (height, width) != (rh, rw):
                print_once(
                    f'  [ast] espectrograma {(height,width)} recortado a {(rh,rw)}')

            x = x[..., :rh, :rw]
            x = self.to_patch_tokens(x)
            _, nph, npw, _ = x.shape

            grid = torch.stack(torch.meshgrid(
                torch.arange(nph, device=device),
                torch.arange(npw, device=device),
                indexing='ij'), dim=-1)
            grid = rearrange(grid, '... c -> (...) c')

            x = x + posemb_sincos_2d(x)
            x = rearrange(x, 'b ... c -> b (...) c')

            # Patch dropout
            if self.training and self.patch_dropout_prob > 0. \
                    and not force_no_patch_dropout:
                n = x.shape[1]
                bi  = torch.arange(batch, device=device)
                bi  = rearrange(bi, '... -> ... 1')
                n_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
                keep_idx = torch.randn(batch, n, device=device) \
                                .topk(n_keep, dim=-1).indices
                x    = x[bi, keep_idx]
                grid = repeat(grid, '... -> b ...', b=batch)[bi, keep_idx]

            rel_dist = (rearrange(grid, '... i c -> ... i 1 c') -
                        rearrange(grid, '... j c -> ... 1 j c'))
            rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

            x, all_layers = self.transformer(x, rel_pos_bias=rel_pos_bias,
                                             return_all_layers=True)
            x   = reduce(x, 'b n d -> b d', 'mean')
            out = self.norm(x)

            if not return_all_layers:
                return out
            return out, all_layers

    class TextTransformer(nn.Module):
        def __init__(self, dim, depth, num_tokens=None, max_seq_len=256,
                     dim_head=64, heads=8, attn_dropout=0., ff_dropout=0.,
                     ff_mult=4, pad_id=0):
            super().__init__()
            self.dim = dim
            self._tok = _get_tokenizer()
            if num_tokens is None:
                num_tokens = self._tok.vocab_size
            self.token_emb = nn.Embedding(num_tokens, dim)
            self.pos_emb   = nn.Embedding(max_seq_len, dim)
            self.depth       = depth
            self.max_seq_len = max_seq_len
            self.cls_token   = nn.Parameter(torch.randn(dim))
            self.transformer = Transformer(
                dim=dim, depth=depth, dim_head=dim_head, heads=heads,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
            )
            self.pad_id = pad_id
            self.norm   = LayerNorm(dim)

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(self, x=None, raw_texts=None, mask=None,
                    return_all_layers=False):
            assert exists(x) ^ exists(raw_texts), \
                'pasa tokens (x) o textos crudos (raw_texts), no ambos'

            if exists(raw_texts):
                x = self._tok.tokenize(raw_texts).to(self.device)
            if not exists(mask):
                mask = x != self.pad_id

            b, n, device = *x.shape, x.device
            x = self.token_emb(x)
            assert n <= self.max_seq_len, \
                f'secuencia de longitud {n} supera el máximo de {self.max_seq_len}'
            x = x + self.pos_emb(torch.arange(n, device=device))

            cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
            x, ps = pack([cls_tokens, x], 'b * d')
            mask  = F.pad(mask, (1, 0), value=True)

            x, all_layers = self.transformer(x, mask=mask, return_all_layers=True)
            cls_tokens, _ = unpack(x, ps, 'b * d')
            out = self.norm(cls_tokens)

            if not return_all_layers:
                return out
            return out, all_layers

    # ── MuLaN ─────────────────────────────────────────────────────────────────

    class MuLaN(nn.Module):
        """
        Music-Language Network: espacio de embedding conjunto audio-texto.
        Entrena con aprendizaje contrastivo (softmax o sigmoid).
        """
        def __init__(self, audio_transformer, text_transformer,
                     dim_latent=128,
                     decoupled_contrastive_learning=True,
                     hierarchical_contrastive_loss=False,
                     hierarchical_contrastive_loss_layers=None,
                     sigmoid_contrastive_loss=False):
            super().__init__()
            self.dim_latent = dim_latent
            self.audio = audio_transformer
            self.text  = text_transformer

            self.text_to_latents  = nn.Linear(self.text.dim,  dim_latent)
            self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

            klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else \
                    partial(SoftmaxContrastiveLearning,
                            decoupled_contrastive_learning=decoupled_contrastive_learning)
            self.contrast = klass()

            self.multi_layer_contrastive_learning = None
            if hierarchical_contrastive_loss:
                n_layers = default(
                    hierarchical_contrastive_loss_layers,
                    min(audio_transformer.depth, text_transformer.depth) - 1)
                assert n_layers > 0
                self.register_buffer(
                    'text_layers_indices',
                    _interspersed_indices(n_layers, text_transformer.depth))
                self.register_buffer(
                    'audio_layers_indices',
                    _interspersed_indices(n_layers, audio_transformer.depth))
                self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                    audio_dim=self.audio.dim, text_dim=self.text.dim,
                    dim_latent=dim_latent, layers=n_layers,
                    decoupled_contrastive_learning=decoupled_contrastive_learning,
                    sigmoid_contrastive_loss=sigmoid_contrastive_loss,
                )

        def get_audio_latents(self, wavs, return_all_layers=False):
            audio_embeds, audio_layers = self.audio(wavs, return_all_layers=True)
            audio_latents = self.audio_to_latents(audio_embeds)
            out = l2norm(audio_latents)
            return (out, audio_layers) if return_all_layers else out

        def get_text_latents(self, texts=None, raw_texts=None,
                             return_all_layers=False):
            text_embeds, text_layers = self.text(
                texts, raw_texts=raw_texts, return_all_layers=True)
            text_latents = self.text_to_latents(text_embeds)
            out = l2norm(text_latents)
            return (out, text_layers) if return_all_layers else out

        def forward(self, wavs, texts=None, raw_texts=None,
                    return_latents=False, return_similarities=False,
                    return_pairwise_similarities=False):
            from einops import einsum as eins
            audio_latents, audio_layers = self.get_audio_latents(
                wavs, return_all_layers=True)
            text_latents, text_layers = self.get_text_latents(
                texts, raw_texts=raw_texts, return_all_layers=True)

            if return_latents:
                return audio_latents, text_latents
            if return_similarities:
                return eins(audio_latents, text_latents, 'i d, i d -> i')
            if return_pairwise_similarities:
                return eins(audio_latents, text_latents, 'i d, j d -> i j')

            cl_loss = self.contrast(audio_latents, text_latents)
            if not exists(self.multi_layer_contrastive_learning):
                return cl_loss

            al = audio_layers[self.audio_layers_indices]
            tl = text_layers[self.text_layers_indices]
            hier_loss = self.multi_layer_contrastive_learning(
                audio_layers=al, text_layers=tl)
            return cl_loss + hier_loss

    # ── Cuantizador residual ──────────────────────────────────────────────────

    class MuLaNEmbedQuantizer(nn.Module):
        """
        Cuantiza los embeddings de MuLaN con Residual VQ y los proyecta
        a las dimensiones de condicionamiento de cada transformer de AudioLM.
        """
        def __init__(self, mulan, conditioning_dims, rq_num_quantizers=8,
                     rq_ema_decay=0.9, codebook_size=1024,
                     namespaces=('semantic', 'coarse', 'fine')):
            super().__init__()
            vqmod = _require("vector_quantize_pytorch",
                             pip_name="vector-quantize-pytorch")
            from vector_quantize_pytorch import ResidualVQ

            self.mulan      = mulan
            self.namespaces = namespaces
            self.conditioning_dims = conditioning_dims

            assert len(conditioning_dims) == len(namespaces), \
                'conditioning_dims y namespaces deben tener la misma longitud'

            dim = mulan.dim_latent
            self.rq = ResidualVQ(
                dim=dim, num_quantizers=rq_num_quantizers,
                codebook_size=codebook_size, decay=rq_ema_decay,
                commitment_weight=0, kmeans_init=True,
                threshold_ema_dead_code=2, quantize_dropout=False,
            )
            self.dim          = dim
            self.num_codebooks = rq_num_quantizers

            self.cond_embeddings = nn.ParameterDict({})
            for namespace, cond_dim in zip(namespaces, conditioning_dims):
                cond_emb = nn.Parameter(
                    torch.randn(rq_num_quantizers, codebook_size, cond_dim))
                nn.init.normal_(cond_emb, std=0.02)
                self.cond_embeddings[namespace] = cond_emb

            self._default_namespace = namespaces[0]

        def parameters(self):
            return self.cond_embeddings.parameters()

        def set_default_namespace(self, namespace):
            self._default_namespace = namespace

        def forward(self, wavs=None, texts=None, namespace=None):
            assert exists(wavs) ^ exists(texts), \
                'pasa wavs o texts, no ambos'
            namespace = default(namespace, self._default_namespace)
            assert namespace in self.namespaces, \
                f'namespace {namespace!r} no encontrado'
            cond_emb = self.cond_embeddings[namespace]

            with torch.no_grad():
                self.mulan.eval()
                latents = (self.mulan.get_audio_latents(wavs)
                           if exists(wavs) else
                           self.mulan.get_text_latents(texts=texts))

            _, indices, _ = self.rq(latents)
            batch        = indices.shape[0]
            n_cb, dim    = self.num_codebooks, cond_emb.shape[-1]

            cond_emb = repeat(cond_emb, 'q c d -> b q c d', b=batch)
            indices  = repeat(indices,  'b q -> b q 1 d', q=n_cb, d=dim)
            cond_emb = cond_emb.gather(2, indices)
            return rearrange(cond_emb, 'b q 1 d -> b q d')

    # ── AudioLM stub / wrapper ────────────────────────────────────────────────

    class AudioLMStub(nn.Module):
        """
        Stub de AudioLM para el comando generate sin audiolm-pytorch instalado.
        Genera ruido gaussiano con la forma esperada y muestra el flujo completo.
        En producción, reemplazar por audiolm_pytorch.AudioLM.
        """
        def __init__(self, sample_rate=24000, seconds=10):
            super().__init__()
            self.sample_rate = sample_rate
            self.seconds     = seconds
            _warn("Usando AudioLM STUB — el audio de salida es ruido de referencia.")
            _warn("Instala audiolm-pytorch para generación real.")

        def forward(self, text_embeds=None, **kwargs):
            n_samples = self.sample_rate * self.seconds
            return torch.randn(1, n_samples)

    class MusicLM(nn.Module):
        """
        Orquestador final: texto → embeddings MuLaN → AudioLM → audio.
        Genera `num_samples` candidatos y elige el más similar al texto
        según la puntuación de MuLaN.
        """
        def __init__(self, audio_lm, mulan_embed_quantizer):
            super().__init__()
            self.mulan_embed_quantizer = mulan_embed_quantizer
            self.audio_lm = audio_lm

        @property
        def device(self):
            return next(self.parameters()).device

        @torch.no_grad()
        def forward(self, text: str, num_samples=1, **audio_lm_kwargs):
            self.eval()
            tok = _get_tokenizer()
            texts = tok.tokenize([text]).to(self.device)
            text_embeds = self.mulan_embed_quantizer(texts=texts)

            samples = [
                self.audio_lm(text_embeds=text_embeds, **audio_lm_kwargs)
                for _ in range(num_samples)
            ]

            if num_samples == 1:
                return first(samples)

            mulan = self.mulan_embed_quantizer.mulan
            from einops import einsum as eins
            sims = torch.cat([
                mulan(texts=texts, wavs=music, return_similarities=True)
                for music in samples
            ], dim=0)
            return samples[sims.topk(1, dim=0).indices.item()]

    # Devolver todas las clases en un namespace
    class _NS:
        pass
    ns = _NS()
    ns.MuLaN                       = MuLaN
    ns.AudioSpectrogramTransformer = AudioSpectrogramTransformer
    ns.TextTransformer             = TextTransformer
    ns.MuLaNEmbedQuantizer         = MuLaNEmbedQuantizer
    ns.MusicLM                     = MusicLM
    ns.AudioLMStub                 = AudioLMStub
    ns.SoftmaxContrastiveLearning  = SoftmaxContrastiveLearning
    ns.SigmoidContrastiveLearning  = SigmoidContrastiveLearning
    return ns


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PARA train-mulan
# ══════════════════════════════════════════════════════════════════════════════

def _build_dataset(manifest_path: str, max_length: int | None = None,
                   sample_rate: int = 22050, verbose: bool = False):
    """
    Dataset de pares (wav, texto) desde un CSV con columnas: path,text
    El CSV puede tener cabecera o no.
    """
    torch    = _require("torch")
    torchaudio = _require("torchaudio")
    from torch.utils.data import Dataset

    rows = []
    with open(manifest_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].strip().lower() in ('path', 'audio', 'file'):
                continue  # cabecera
            rows.append((row[0].strip(), row[1].strip()))

    if verbose:
        _info(f"Dataset: {len(rows)} pares (audio, texto) en {manifest_path}")

    class PairDataset(Dataset):
        def __init__(self, rows, sr, max_len):
            self.rows    = rows
            self.sr      = sr
            self.max_len = max_len

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            path, text = self.rows[idx]
            try:
                wav, orig_sr = torchaudio.load(path)
                if orig_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
                    wav = resampler(wav)
                wav = wav.mean(dim=0)  # mono
                if self.max_len:
                    wav = wav[:self.max_len]
            except Exception as e:
                wav = torch.zeros(self.sr)  # silencio si falla la carga
            return wav, text

    return PairDataset(rows, sample_rate, max_length)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT: guardar / cargar
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(path: str, model_state: dict, meta: dict):
    torch = _require("torch")
    pkg = {'state_dict': model_state, 'meta': meta,
           'version': VERSION, 'saved': datetime.now().isoformat()}
    torch.save(pkg, path)

def _load_checkpoint(path: str, verbose: bool = False):
    torch = _require("torch")
    pkg = torch.load(path, map_location='cpu')
    if verbose:
        meta = pkg.get('meta', {})
        _info(f"Checkpoint: {path}")
        _info(f"  versión:  {pkg.get('version','?')}")
        _info(f"  guardado: {pkg.get('saved','?')}")
        for k, v in meta.items():
            _info(f"  {k}: {v}")
    return pkg


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: prepare
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """Construye un manifest CSV a partir de una carpeta de audios y textos."""
    _print_header("prepare — construir manifest de dataset")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"  [error] Carpeta no encontrada: {audio_dir}")
        sys.exit(1)

    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.aiff'}
    audio_files = sorted([
        p for p in audio_dir.rglob('*')
        if p.suffix.lower() in extensions
    ])

    if not audio_files:
        print(f"  [error] No se encontraron ficheros de audio en {audio_dir}")
        sys.exit(1)

    _step(1, 3, f"Encontrados {len(audio_files)} ficheros de audio")

    # Buscar texto asociado: mismo nombre con .txt, o usar el nombre del fichero
    rows = []
    missing_text = 0
    for ap in audio_files:
        txt_path = ap.with_suffix('.txt')
        if txt_path.exists():
            text = txt_path.read_text(encoding='utf-8').strip()
        else:
            text = ap.stem.replace('_', ' ').replace('-', ' ')
            missing_text += 1
        rows.append((str(ap), text))

    _step(2, 3, f"Pares con fichero .txt: {len(rows) - missing_text} "
                f"/ generados del nombre: {missing_text}")

    out = args.manifest or str(audio_dir / 'manifest.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'text'])
        writer.writerows(rows)

    _step(3, 3, f"Manifest guardado: {out}")
    print()
    _info(f"Total de pares: {len(rows)}")
    if missing_text:
        _warn(f"{missing_text} ficheros sin .txt — se usó el nombre del fichero como texto")
    _info("Edita el CSV para mejorar las descripciones de texto antes de entrenar.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train-mulan
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_mulan(args):
    """Entrena MuLaN con aprendizaje contrastivo audio-texto."""
    torch = _require("torch")
    _require("torchaudio", pip_name="torchaudio")

    _print_header("train-mulan — entrenamiento contrastivo audio↔texto")

    if not Path(args.manifest).exists():
        print(f"  [error] Manifest no encontrado: {args.manifest}")
        sys.exit(1)

    device = args.device or _auto_device()
    torch.manual_seed(args.seed)

    _step(1, 6, "Cargando módulos de arquitectura...")
    M = _build_modules()

    _step(2, 6, "Construyendo modelos...")
    audio_transformer = M.AudioSpectrogramTransformer(
        dim=args.dim, depth=args.depth, heads=args.heads,
        dim_head=args.dim // args.heads,
        spec_n_fft=128, spec_win_length=24, spec_aug_stretch_factor=0.8,
    )
    text_transformer = M.TextTransformer(
        dim=args.dim, depth=args.depth, heads=args.heads,
        dim_head=args.dim // args.heads,
    )
    mulan = M.MuLaN(
        audio_transformer=audio_transformer,
        text_transformer=text_transformer,
        dim_latent=args.dim_latent,
        decoupled_contrastive_learning=args.decoupled_cl,
        sigmoid_contrastive_loss=args.sigmoid_cl,
        hierarchical_contrastive_loss=args.hierarchical_cl,
    ).to(device)

    n_params = sum(p.numel() for p in mulan.parameters()) / 1e6
    _info(f"MuLaN: {n_params:.1f} M parámetros")
    _info(f"Audio encoder: dim={args.dim}, depth={args.depth}, heads={args.heads}")
    _info(f"Texto encoder: dim={args.dim}, depth={args.depth}, heads={args.heads}")
    _info(f"Espacio latente: {args.dim_latent}D")
    if args.decoupled_cl: _info("Pérdida: Decoupled Softmax Contrastive")
    if args.sigmoid_cl:   _info("Pérdida: Sigmoid Contrastive (SigLIP)")
    if args.hierarchical_cl: _info("Pérdida jerárquica multi-capa: activada")

    _step(3, 6, f"Preparando dataset desde {args.manifest}...")
    dataset = _build_dataset(
        args.manifest, verbose=args.verbose,
        max_length=args.data_max_length,
    )
    if len(dataset) < 2:
        print("  [error] El dataset necesita al menos 2 muestras.")
        sys.exit(1)

    # Intentar usar MuLaNTrainer con accelerate; si no, bucle manual
    accel = _require("accelerate", fatal=False)
    lion_mod = _require("lion_pytorch", pip_name="lion-pytorch", fatal=False)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _step(4, 6, "Configurando optimizador y dataloader...")
    if args.use_lion and lion_mod is not None:
        from lion_pytorch import Lion
        optimizer = Lion(mulan.parameters(), lr=args.lr)
        _info("Optimizador: Lion")
    else:
        optimizer = torch.optim.Adam(mulan.parameters(), lr=args.lr,
                                     betas=(0.9, 0.99))
        _info("Optimizador: Adam")

    from torch.utils.data import DataLoader, random_split
    from torch.nn.utils.rnn import pad_sequence

    valid_size = max(1, int(args.valid_frac * len(dataset)))
    train_size = len(dataset) - valid_size
    train_ds, valid_ds = random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed))

    def _collate(batch):
        wavs, texts = zip(*batch)
        max_len = max(w.shape[0] for w in wavs)
        wavs_padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
            for w in wavs
        ])
        return wavs_padded, list(texts)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, collate_fn=_collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size,
                          shuffle=False, drop_last=False, collate_fn=_collate)

    _info(f"Train: {train_size} muestras | Valid: {valid_size} muestras")

    _step(5, 6, f"Entrenando {args.steps} pasos...")

    def _cycle(dl):
        while True:
            for x in dl:
                yield x

    dl_iter = _cycle(train_dl)
    best_valid_loss = float('inf')
    log_every = max(1, args.steps // 20)

    for step in range(1, args.steps + 1):
        mulan.train()
        wavs, texts = next(dl_iter)
        wavs = wavs.to(device)

        loss = mulan(wavs, raw_texts=texts)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mulan.parameters(), 0.5)
        optimizer.step()

        if step % log_every == 0 or step == args.steps:
            # Validación rápida
            mulan.eval()
            val_losses = []
            with torch.no_grad():
                for vwavs, vtexts in valid_dl:
                    vwavs = vwavs.to(device)
                    val_losses.append(mulan(vwavs, raw_texts=vtexts).item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
            print(f"  paso {step:>7}/{args.steps}  "
                  f"train={loss.item():.4f}  valid={val_loss:.4f}")

        if step % args.save_every == 0:
            ckpt = str(results_dir / f'mulan.{step:07d}.pt')
            _save_checkpoint(ckpt, mulan.state_dict(), {
                'step': step, 'dim': args.dim, 'depth': args.depth,
                'heads': args.heads, 'dim_latent': args.dim_latent,
                'decoupled_cl': args.decoupled_cl,
                'sigmoid_cl': args.sigmoid_cl,
            })
            if args.verbose:
                _ok(f"Checkpoint guardado: {ckpt}")

    _step(6, 6, "Guardando checkpoint final...")
    final_path = args.output or 'mulan.pt'
    _save_checkpoint(final_path, mulan.state_dict(), {
        'step': args.steps, 'dim': args.dim, 'depth': args.depth,
        'heads': args.heads, 'dim_latent': args.dim_latent,
        'decoupled_cl': args.decoupled_cl, 'sigmoid_cl': args.sigmoid_cl,
    })
    _ok(f"Modelo guardado: {final_path}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: quantize
# ══════════════════════════════════════════════════════════════════════════════

def cmd_quantize(args):
    """
    Carga MuLaN y ajusta el cuantizador residual (ResidualVQ) pasando
    embeddings de audio del manifest. Guarda el cuantizador listo para generate.
    """
    torch = _require("torch")
    _require("vector_quantize_pytorch", pip_name="vector-quantize-pytorch")

    _print_header("quantize — ajuste del cuantizador vectorial residual")

    device = args.device or _auto_device()
    M = _build_modules()

    _step(1, 5, "Cargando checkpoint de MuLaN...")
    if not Path(args.mulan).exists():
        print(f"  [error] Checkpoint no encontrado: {args.mulan}")
        sys.exit(1)
    pkg = _load_checkpoint(args.mulan, verbose=args.verbose)
    meta = pkg.get('meta', {})

    dim       = meta.get('dim',       args.dim)
    depth     = meta.get('depth',     args.depth)
    heads     = meta.get('heads',     args.heads)
    dim_latent = meta.get('dim_latent', args.dim_latent)

    audio_enc = M.AudioSpectrogramTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
        spec_n_fft=128, spec_win_length=24, spec_aug_stretch_factor=0.8,
    )
    text_enc = M.TextTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
    )
    mulan = M.MuLaN(audio_enc, text_enc, dim_latent=dim_latent)
    mulan.load_state_dict(pkg['state_dict'])
    mulan.eval().to(device)
    _ok("MuLaN cargado")

    namespaces = [n.strip() for n in args.namespaces.split(',')]
    cond_dims  = [int(d) for d in args.conditioning_dims.split(',')]
    if len(cond_dims) == 1:
        cond_dims = cond_dims * len(namespaces)

    _step(2, 5, "Construyendo cuantizador residual...")
    quantizer = M.MuLaNEmbedQuantizer(
        mulan=mulan,
        conditioning_dims=tuple(cond_dims),
        rq_num_quantizers=args.rq_quantizers,
        codebook_size=args.codebook_size,
        namespaces=tuple(namespaces),
    ).to(device)
    _info(f"ResidualVQ: {args.rq_quantizers} cuantizadores, "
          f"codebook={args.codebook_size}, namespaces={namespaces}")

    _step(3, 5, f"Extrayendo embeddings del manifest: {args.manifest}...")
    if not Path(args.manifest).exists():
        print(f"  [error] Manifest no encontrado: {args.manifest}")
        sys.exit(1)
    dataset = _build_dataset(args.manifest, verbose=args.verbose)

    from torch.utils.data import DataLoader

    def _collate(batch):
        wavs, texts = zip(*batch)
        max_len = max(w.shape[0] for w in wavs)
        wavs_padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
            for w in wavs
        ])
        return wavs_padded, list(texts)

    dl = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, collate_fn=_collate)

    _step(4, 5, "Inicializando codebooks (k-means sobre embeddings)...")
    mulan.eval()
    n_batches = min(args.quantize_batches, len(dl))
    with torch.no_grad():
        for i, (wavs, _) in enumerate(dl):
            if i >= n_batches:
                break
            wavs = wavs.to(device)
            latents = mulan.get_audio_latents(wavs)
            quantizer.rq(latents)  # dispara la inicialización k-means
            if args.verbose:
                print(f"    batch {i+1}/{n_batches}")

    _step(5, 5, "Guardando cuantizador...")
    out = args.output or 'quantizer.pt'
    _save_checkpoint(out, quantizer.state_dict(), {
        'namespaces': namespaces,
        'conditioning_dims': cond_dims,
        'rq_num_quantizers': args.rq_quantizers,
        'codebook_size': args.codebook_size,
        'dim_latent': dim_latent,
        'mulan_meta': meta,
    })
    _ok(f"Cuantizador guardado: {out}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: embed
# ══════════════════════════════════════════════════════════════════════════════

def cmd_embed(args):
    """Extrae el embedding de MuLaN de un fichero de audio o un texto."""
    torch = _require("torch")

    _print_header("embed — extracción de embeddings MuLaN")

    device = args.device or _auto_device()
    M = _build_modules()

    _step(1, 4, "Cargando checkpoint de MuLaN...")
    pkg  = _load_checkpoint(args.mulan, verbose=args.verbose)
    meta = pkg.get('meta', {})
    dim        = meta.get('dim',       args.dim)
    depth      = meta.get('depth',     args.depth)
    heads      = meta.get('heads',     args.heads)
    dim_latent = meta.get('dim_latent', args.dim_latent)

    audio_enc = M.AudioSpectrogramTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
        spec_n_fft=128, spec_win_length=24, spec_aug_stretch_factor=0.8,
    )
    text_enc = M.TextTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
    )
    mulan = M.MuLaN(audio_enc, text_enc, dim_latent=dim_latent)
    mulan.load_state_dict(pkg['state_dict'])
    mulan.eval().to(device)
    _ok("MuLaN cargado")

    _step(2, 4, "Determinando modalidad...")
    # auto-detect: si el input parece un fichero de audio, modo audio; si no, texto
    input_val = args.input
    mode = args.mode
    if mode == 'auto':
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a'}
        p = Path(input_val)
        mode = 'audio' if p.suffix.lower() in audio_exts and p.exists() else 'text'
    _info(f"Modalidad: {mode}")

    _step(3, 4, "Extrayendo embedding...")
    with torch.no_grad():
        if mode == 'audio':
            torchaudio = _require("torchaudio")
            wav, sr = torchaudio.load(input_val)
            if sr != 22050:
                wav = torchaudio.transforms.Resample(sr, 22050)(wav)
            wav = wav.mean(0, keepdim=True).to(device)
            latent = mulan.get_audio_latents(wav)
        else:
            latent = mulan.get_text_latents(raw_texts=[input_val])

    vec = latent[0].cpu().tolist()

    _step(4, 4, "Guardando embedding...")
    out_data = {
        'mode':    mode,
        'input':   input_val,
        'dim':     len(vec),
        'vector':  vec,
        'mulan_checkpoint': args.mulan,
        'generated': datetime.now().isoformat(),
    }
    out = args.output or 'embed.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
    _ok(f"Embedding guardado: {out}  (dim={len(vec)})")
    if args.verbose:
        _info(f"Norma L2: {sum(v**2 for v in vec)**0.5:.4f} (debe ser ≈1.0)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    """Genera música a partir de una descripción de texto."""
    torch = _require("torch")
    _print_header("generate — síntesis de música desde texto")

    device = args.device or _auto_device()
    M = _build_modules()
    tok = _get_tokenizer()

    _step(1, 5, "Cargando MuLaN...")
    pkg  = _load_checkpoint(args.mulan, verbose=args.verbose)
    meta = pkg.get('meta', {})
    dim        = meta.get('dim',       args.dim)
    depth      = meta.get('depth',     args.depth)
    heads      = meta.get('heads',     args.heads)
    dim_latent = meta.get('dim_latent', args.dim_latent)

    audio_enc = M.AudioSpectrogramTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
        spec_n_fft=128, spec_win_length=24, spec_aug_stretch_factor=0.8,
    )
    text_enc = M.TextTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
    )
    mulan = M.MuLaN(audio_enc, text_enc, dim_latent=dim_latent)
    mulan.load_state_dict(pkg['state_dict'])
    mulan.eval().to(device)
    _ok("MuLaN cargado")

    _step(2, 5, "Cargando cuantizador...")
    q_pkg  = _load_checkpoint(args.quantizer, verbose=args.verbose)
    q_meta = q_pkg.get('meta', {})
    namespaces = q_meta.get('namespaces', ['semantic', 'coarse', 'fine'])
    cond_dims  = q_meta.get('conditioning_dims', [1024, 1024, 1024])
    rq_n       = q_meta.get('rq_num_quantizers', 8)
    cb_size    = q_meta.get('codebook_size', 1024)

    quantizer = M.MuLaNEmbedQuantizer(
        mulan=mulan,
        conditioning_dims=tuple(cond_dims),
        rq_num_quantizers=rq_n,
        codebook_size=cb_size,
        namespaces=tuple(namespaces),
    )
    quantizer.load_state_dict(q_pkg['state_dict'])
    quantizer.to(device)
    _ok("Cuantizador cargado")

    _step(3, 5, "Inicializando AudioLM...")
    use_stub = args.stub

    if not use_stub and exists(args.audio_lm):
        # Intentar cargar AudioLM real
        audiolm_mod = _require("audiolm_pytorch",
                               pip_name="audiolm-pytorch", fatal=False)
        if audiolm_mod is not None:
            _warn("Carga de AudioLM real no implementada en este script de demo.")
            _warn("Integra aquí tu instancia de audiolm_pytorch.AudioLM.")
            use_stub = True
        else:
            _warn("audiolm-pytorch no instalado. Usando stub.")
            use_stub = True
    else:
        use_stub = True

    audio_lm = M.AudioLMStub(sample_rate=24000, seconds=10).to(device)

    musiclm = M.MusicLM(audio_lm=audio_lm,
                        mulan_embed_quantizer=quantizer).to(device)

    _step(4, 5, f"Generando {args.num_samples} candidatos para: \"{args.text}\"")
    _info("Flujo: texto → tokenizar → MuLaN embed → quantize → AudioLM → "
          "selección por similitud MuLaN")

    with torch.no_grad():
        music = musiclm(args.text, num_samples=args.num_samples)

    _step(5, 5, "Guardando audio...")
    try:
        torchaudio = _require("torchaudio", fatal=False)
        out = args.output or 'generated.wav'
        if torchaudio:
            music_cpu = music.cpu()
            if music_cpu.ndim == 1:
                music_cpu = music_cpu.unsqueeze(0)
            torchaudio.save(out, music_cpu, 24000)
            _ok(f"Audio guardado: {out}  "
                f"({music_cpu.shape[-1]/24000:.1f}s @ 24kHz)")
        else:
            _warn("torchaudio no disponible: no se pudo guardar el wav.")
    except Exception as e:
        _warn(f"Error guardando audio: {e}")

    print()
    if use_stub:
        print(textwrap.fill(
            "  NOTA: Este es el flujo completo de MusicLM con un AudioLM STUB. "
            "Para generación real, instala audiolm-pytorch, entrena (o descarga) "
            "un modelo AudioLM, y pásalo con --audio-lm.",
            width=70, subsequent_indent='  '))
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    """Muestra arquitectura y estado de un checkpoint guardado."""
    torch = _require("torch")
    _print_header("inspect — diagnóstico de checkpoint")

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"  [error] Fichero no encontrado: {path}")
        sys.exit(1)

    pkg  = torch.load(str(path), map_location='cpu')
    meta = pkg.get('meta', {})
    sd   = pkg.get('state_dict', {})

    sep = "─" * 58
    print(f"\n  Fichero:    {path}")
    print(f"  Versión:    {pkg.get('version', '?')}")
    print(f"  Guardado:   {pkg.get('saved', '?')}")
    print(f"\n  {sep}")
    print(f"  Metadatos del modelo")
    print(f"  {sep}")
    for k, v in meta.items():
        print(f"    {k:<28} {v}")

    print(f"\n  {sep}")
    print(f"  Parámetros del state_dict ({len(sd)} tensores)")
    print(f"  {sep}")
    total_params = 0
    for name, tensor in sorted(sd.items()):
        n = tensor.numel()
        total_params += n
        if args.verbose:
            print(f"    {name:<55} {list(tensor.shape)}  ({n:,})")
    print(f"\n    Total: {total_params/1e6:.2f} M parámetros  "
          f"({total_params * 4 / 1024**2:.1f} MB en float32)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: round-trip
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    """
    Diagnóstico: carga un audio y un texto, extrae ambos embeddings,
    y muestra su similitud coseno. No genera ningún fichero de audio.
    Útil para verificar que MuLaN está bien entrenado.
    """
    torch = _require("torch")
    torchaudio = _require("torchaudio")

    _print_header("round-trip — diagnóstico audio ↔ texto")

    device = args.device or _auto_device()
    M = _build_modules()

    _step(1, 4, "Cargando MuLaN...")
    pkg  = _load_checkpoint(args.mulan, verbose=args.verbose)
    meta = pkg.get('meta', {})
    dim        = meta.get('dim',       args.dim)
    depth      = meta.get('depth',     args.depth)
    heads      = meta.get('heads',     args.heads)
    dim_latent = meta.get('dim_latent', args.dim_latent)

    audio_enc = M.AudioSpectrogramTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
        spec_n_fft=128, spec_win_length=24, spec_aug_stretch_factor=0.8,
    )
    text_enc = M.TextTransformer(
        dim=dim, depth=depth, heads=heads, dim_head=dim // heads,
    )
    mulan = M.MuLaN(audio_enc, text_enc, dim_latent=dim_latent)
    mulan.load_state_dict(pkg['state_dict'])
    mulan.eval().to(device)
    _ok("MuLaN cargado")

    _step(2, 4, f"Cargando audio: {args.audio}...")
    wav, sr = torchaudio.load(args.audio)
    if sr != 22050:
        wav = torchaudio.transforms.Resample(sr, 22050)(wav)
    wav = wav.mean(0, keepdim=True).to(device)

    _step(3, 4, "Extrayendo embeddings...")
    texts_to_test = [args.text]
    if args.extra_texts:
        texts_to_test += [t.strip() for t in args.extra_texts.split('|')]

    with torch.no_grad():
        audio_latent = mulan.get_audio_latents(wav)
        text_latents = [mulan.get_text_latents(raw_texts=[t]) for t in texts_to_test]

    _step(4, 4, "Resultados de similitud coseno")
    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  Audio:  {args.audio}")
    print(f"  {sep}")

    from torch.nn.functional import cosine_similarity
    for t, tl in zip(texts_to_test, text_latents):
        sim = cosine_similarity(audio_latent, tl, dim=-1).item()
        bar = '█' * int(sim * 30) + '░' * (30 - int(sim * 30))
        mark = " ← consulta" if t == args.text else ""
        print(f"  {sim:+.4f}  {bar}  \"{t}\"{mark}")

    print(f"\n  {sep}")
    al2 = (audio_latent ** 2).sum().sqrt().item()
    print(f"  Norma L2 embedding audio: {al2:.4f}  (ideal: 1.0000)")
    print()

    if abs(al2 - 1.0) > 0.01:
        _warn("La norma del embedding no es ≈1.0 — el modelo puede no estar "
              "correctamente entrenado o el checkpoint está corrupto.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='musiclm.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            MUSICLM v{VERSION} — Generación de música a partir de texto
            ─────────────────────────────────────────────────────────────
            Pipeline completo: MuLaN (contrastivo audio↔texto) + AudioLM.

            Comandos disponibles:
              prepare      Construye un manifest CSV desde una carpeta de audios
              train-mulan  Entrena el modelo de embedding audio-texto (MuLaN)
              quantize     Ajusta el cuantizador vectorial residual sobre MuLaN
              embed        Extrae el embedding de un audio o texto
              generate     Genera música a partir de una descripción de texto
              inspect      Diagnóstico de un checkpoint guardado
              round-trip   Diagnóstico: similitud audio ↔ texto sin generar

            Ejemplos:
              python musiclm.py prepare --audio-dir ./wavs
              python musiclm.py train-mulan --manifest manifest.csv --steps 50000
              python musiclm.py quantize --mulan mulan.pt --manifest manifest.csv
              python musiclm.py generate "piano melancólico" --mulan mulan.pt --quantizer quantizer.pt
              python musiclm.py embed audio.wav --mulan mulan.pt
              python musiclm.py inspect mulan.pt
              python musiclm.py round-trip audio.wav "lluvia suave" --mulan mulan.pt
        """),
    )
    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── Opciones comunes ──────────────────────────────────────────────────────
    def _add_common(p):
        p.add_argument('--mulan',      default='mulan.pt',
                       help='Checkpoint de MuLaN (default: mulan.pt)')
        p.add_argument('--device',     default=None,
                       help='cpu | cuda | mps (default: auto)')
        p.add_argument('--dim',        type=int, default=512,
                       help='Dimensión del transformer (default: 512)')
        p.add_argument('--depth',      type=int, default=6,
                       help='Capas del transformer (default: 6)')
        p.add_argument('--heads',      type=int, default=8,
                       help='Cabezas de atención (default: 8)')
        p.add_argument('--dim-latent', type=int, default=128,
                       dest='dim_latent',
                       help='Dimensión del espacio latente MuLaN (default: 128)')
        p.add_argument('--seed',       type=int, default=42,
                       help='Semilla aleatoria (default: 42)')
        p.add_argument('--verbose', '-v', action='store_true',
                       help='Informe detallado')

    # ── prepare ───────────────────────────────────────────────────────────────
    p_prep = sub.add_parser('prepare',
        help='Construye un manifest CSV desde una carpeta de audios',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_prep.add_argument('--audio-dir', required=True, dest='audio_dir',
                        help='Carpeta con ficheros de audio (.wav, .mp3, .flac…)')
    p_prep.add_argument('--manifest',  default=None,
                        help='Ruta de salida del CSV (default: <audio-dir>/manifest.csv)')
    p_prep.add_argument('--verbose', '-v', action='store_true')
    p_prep.set_defaults(func=cmd_prepare)

    # ── train-mulan ───────────────────────────────────────────────────────────
    p_tr = sub.add_parser('train-mulan',
        help='Entrena MuLaN con aprendizaje contrastivo',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_tr)
    p_tr.add_argument('--manifest',     required=True,
                      help='CSV con columnas: path,text')
    p_tr.add_argument('--steps',        type=int, default=100_000,
                      help='Pasos de entrenamiento (default: 100000)')
    p_tr.add_argument('--batch-size',   type=int, default=16, dest='batch_size',
                      help='Tamaño de lote (default: 16)')
    p_tr.add_argument('--lr',           type=float, default=3e-4,
                      help='Learning rate (default: 3e-4)')
    p_tr.add_argument('--valid-frac',   type=float, default=0.05, dest='valid_frac',
                      help='Fracción de validación (default: 0.05)')
    p_tr.add_argument('--save-every',   type=int, default=1000, dest='save_every',
                      help='Guardar checkpoint cada N pasos (default: 1000)')
    p_tr.add_argument('--results-dir',  default='./results_mulan', dest='results_dir',
                      help='Carpeta de checkpoints (default: ./results_mulan)')
    p_tr.add_argument('--output',       default='mulan.pt',
                      help='Checkpoint final (default: mulan.pt)')
    p_tr.add_argument('--data-max-length', type=int, default=None,
                      dest='data_max_length',
                      help='Longitud máxima del audio en muestras (default: sin límite)')
    p_tr.add_argument('--use-lion',     action='store_true', dest='use_lion',
                      help='Usar optimizador Lion (requiere lion-pytorch)')
    p_tr.add_argument('--decoupled-cl', action='store_true', dest='decoupled_cl',
                      help='Usar decoupled contrastive learning')
    p_tr.add_argument('--sigmoid-cl',   action='store_true', dest='sigmoid_cl',
                      help='Usar sigmoid contrastive loss (SigLIP)')
    p_tr.add_argument('--hierarchical-cl', action='store_true', dest='hierarchical_cl',
                      help='Añadir pérdida contrastiva jerárquica multi-capa')
    p_tr.set_defaults(func=cmd_train_mulan)

    # ── quantize ──────────────────────────────────────────────────────────────
    p_q = sub.add_parser('quantize',
        help='Ajusta el cuantizador residual sobre MuLaN',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_q)
    p_q.add_argument('--manifest',          required=True,
                     help='CSV con columnas: path,text')
    p_q.add_argument('--rq-quantizers',     type=int, default=8,
                     dest='rq_quantizers',
                     help='Número de cuantizadores residuales (default: 8)')
    p_q.add_argument('--codebook-size',     type=int, default=1024,
                     dest='codebook_size',
                     help='Tamaño del codebook (default: 1024)')
    p_q.add_argument('--namespaces',        default='semantic,coarse,fine',
                     help='Namespaces separados por coma (default: semantic,coarse,fine)')
    p_q.add_argument('--conditioning-dims', default='1024,1024,1024',
                     dest='conditioning_dims',
                     help='Dimensiones de condicionamiento por namespace (default: 1024,1024,1024)')
    p_q.add_argument('--batch-size',        type=int, default=16, dest='batch_size',
                     help='Tamaño de lote (default: 16)')
    p_q.add_argument('--quantize-batches',  type=int, default=50,
                     dest='quantize_batches',
                     help='Batches para inicializar k-means (default: 50)')
    p_q.add_argument('--output',            default='quantizer.pt',
                     help='Checkpoint del cuantizador (default: quantizer.pt)')
    p_q.set_defaults(func=cmd_quantize)

    # ── embed ─────────────────────────────────────────────────────────────────
    p_emb = sub.add_parser('embed',
        help='Extrae el embedding de un audio o texto',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_emb)
    p_emb.add_argument('input',
                       help='Fichero de audio (.wav/.mp3/…) o descripción de texto')
    p_emb.add_argument('--mode', choices=['audio', 'text', 'auto'], default='auto',
                       help='Modalidad (default: auto, detectada por extensión)')
    p_emb.add_argument('--output', default='embed.json',
                       help='JSON de salida (default: embed.json)')
    p_emb.set_defaults(func=cmd_embed)

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser('generate',
        help='Genera música a partir de texto',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_gen)
    p_gen.add_argument('text',
                       help='Descripción de texto de la música a generar')
    p_gen.add_argument('--quantizer',   default='quantizer.pt',
                       help='Checkpoint del cuantizador (default: quantizer.pt)')
    p_gen.add_argument('--audio-lm',    default=None, dest='audio_lm',
                       help='Checkpoint de AudioLM (default: usar stub interno)')
    p_gen.add_argument('--num-samples', type=int, default=4, dest='num_samples',
                       help='Candidatos generados (elige el más similar, default: 4)')
    p_gen.add_argument('--output',      default='generated.wav',
                       help='Fichero wav de salida (default: generated.wav)')
    p_gen.add_argument('--stub',        action='store_true',
                       help='Forzar AudioLM stub (no requiere audiolm-pytorch)')
    p_gen.set_defaults(func=cmd_generate)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect',
        help='Diagnóstico de un checkpoint guardado',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_ins.add_argument('checkpoint',
                       help='Fichero .pt a inspeccionar')
    p_ins.add_argument('--verbose', '-v', action='store_true',
                       help='Listar todos los tensores del state_dict')
    p_ins.set_defaults(func=cmd_inspect)

    # ── round-trip ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser('round-trip',
        help='Diagnóstico: similitud audio ↔ texto sin generar',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p_rt)
    p_rt.add_argument('audio',
                      help='Fichero de audio (.wav/.mp3/…)')
    p_rt.add_argument('text',
                      help='Texto de consulta principal')
    p_rt.add_argument('--extra-texts', default=None, dest='extra_texts',
                      help='Textos adicionales separados por | para comparar')
    p_rt.set_defaults(func=cmd_round_trip)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
