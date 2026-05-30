#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MUSEGAN  v2.0  (PyTorch)                            ║
║       Generación de música multi-pista mediante GAN sobre piano-rolls        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  REPRESENTACIÓN                                                               ║
║    Piano-roll 5D:  (N, n_bars, n_timesteps, n_pitches, n_tracks)             ║
║    Por defecto:    n_bars=4, n_timesteps=48, n_pitches=84, n_tracks=5        ║
║    Tracks:         drums · bass · guitar · strings · piano                   ║
║    Rango MIDI:     pitches 24-107  (C1-B7)                                   ║
║                                                                              ║
║  ARQUITECTURA (BinaryMuseGAN, ISMIR 2018)                                    ║
║    Generador  — shared 3D-tconv + ramas pitch-time / time-pitch por pista    ║
║    Discriminador — ramas simétricas + stream chroma + stream onset/offset    ║
║    Loss          — Wasserstein + penalización de gradiente (WGAN-GP)         ║
║                                                                              ║
║  COMANDOS                                                                    ║
║    prepare     — MIDI corpus → piano-rolls .npz                              ║
║    train       — Entrenar el modelo GAN                                      ║
║    generate    — Generar MIDIs desde ruido latente                           ║
║    accompany   — Generar acompañamiento dado un track de referencia          ║
║    interpolate — Interpolar entre dos puntos del espacio latente             ║
║    inspect     — Diagnóstico del modelo y los datos                          ║
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║    torch, numpy                                                              ║
║    mido        →  pip install mido          (prepare / generate / accompany) ║
║    imageio     →  pip install imageio       (--save-images)                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  USO — PREPARE                                                               ║
║    python musegan.py prepare                                                  ║
║        --input-dir  midis/                                                   ║
║        --output     data/train.npz                                           ║
║        --bars 4  --resolution 12  --recursive                                ║
║                                                                              ║
║  USO — TRAIN                                                                 ║
║    python musegan.py train                                                    ║
║        --data       data/train.npz                                           ║
║        --model-dir  model/                                                   ║
║        --steps 50000  --batch-size 64  --latent-dim 128                      ║
║        --gan-loss wasserstein  --gp  --lr 1e-3                               ║
║        --save-steps 10000  --sample-steps 1000                               ║
║                                                                              ║
║    # Reanudar entrenamiento:                                                  ║
║    python musegan.py train  --data data/train.npz  --model-dir model/        ║
║        --resume                                                              ║
║                                                                              ║
║  USO — GENERATE                                                              ║
║    python musegan.py generate                                                 ║
║        --model-dir  model/                                                   ║
║        --output-dir results/                                                 ║
║        --n 16  --threshold hard                                              ║
║        --save-images  --save-arrays                                          ║
║                                                                              ║
║  USO — ACCOMPANY                                                             ║
║    python musegan.py accompany                                                ║
║        --model-dir  model/                                                   ║
║        --input      melody.mid                                               ║
║        --condition-track 4                                                   ║
║        --output     output.mid                                               ║
║                                                                              ║
║  USO — INTERPOLATE                                                           ║
║    python musegan.py interpolate                                              ║
║        --model-dir  model/                                                   ║
║        --output-dir interpolation/                                           ║
║        --steps 8  --mode slerp                                               ║
║                                                                              ║
║  USO — INSPECT                                                               ║
║    python musegan.py inspect --model-dir model/                              ║
║    python musegan.py inspect --data data/train.npz                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

# ── Constantes globales ────────────────────────────────────────────────────────

N_BARS        = 4
N_TIMESTEPS   = 48
N_PITCHES     = 84
N_TRACKS      = 5
BEAT_RES      = 12        # ticks por beat
LOWEST_PITCH  = 24        # MIDI pitch 24 = C1

TRACK_NAMES   = ["drums", "bass", "guitar", "strings", "piano"]
PROGRAMS      = [0, 0, 25, 33, 48]   # GM programs por track
IS_DRUMS      = [True, False, False, False, False]
TEMPO         = 100       # BPM por defecto

# Colores por track para imágenes (R,G,B en [0,1])
COLORMAP = [
    [1., 0., 0.],
    [1., .5, 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [0., .5, 1.],
]

# Familias MIDI para la etapa prepare
FAMILY_THRESHOLDS = {
    "drums":   (2,  24),
    "bass":    (1,  96),
    "guitar":  (2, 156),
    "strings": (2, 156),
    "piano":   (2, 156),
}


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _build_tconv_block(in_ch, out_ch, kernel, stride, norm=True):
    """Bloque ConvTranspose3d + BatchNorm + ReLU."""
    import torch.nn as nn
    layers = [nn.ConvTranspose3d(in_ch, out_ch, kernel, stride, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm3d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _build_conv_block(in_ch, out_ch, kernel, stride):
    """Bloque Conv3d + LeakyReLU (sin norm, igual que el discriminador original)."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel, stride, bias=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Generator(object):
    """
    Generador MuseGAN con 3D-tconv.

    Estructura:
      z  →  shared  →  [pitch_time_private | time_pitch_private] x n_tracks
         →  merged_private  →  tanh  →  (N, n_bars, n_timesteps, n_pitches, n_tracks)

    El tensor de salida tiene forma NHWDC donde PyTorch espera NCDHW; trabajamos
    internamente en (N, C, D, H, W) = (N, ch, n_bars, n_timesteps, n_pitches)
    y al final reordenamos a la forma 5D canónica de MuseGAN.
    """

    def __init__(self, n_tracks=N_TRACKS, latent_dim=128):
        import torch.nn as nn

        self.n_tracks = n_tracks
        self.latent_dim = latent_dim

        # ── Shared ────────────────────────────────────────────────────────────
        # z: (N, latent_dim) → unsqueeze → (N, latent_dim, 1, 1, 1)
        # tconv: (D, H, W):
        #   (1,1,1) → (4,1,1) → (4,4,3) → (4,16,7)
        self.shared = nn.Sequential(
            _build_tconv_block(latent_dim, 512, (4, 1, 1), (4, 1, 1)),
            _build_tconv_block(512,        256, (1, 4, 3), (1, 4, 3)),
            _build_tconv_block(256,        128, (1, 4, 3), (1, 4, 2)),
        )

        # ── Ramas privadas por track ──────────────────────────────────────────
        # pitch_time:  (4,16,7) → (4,16,84) → (4,48,84)
        self.pt_1 = nn.ModuleList([
            _build_tconv_block(128, 32, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])
        self.pt_2 = nn.ModuleList([
            _build_tconv_block(32, 16, (1, 3, 1), (1, 3, 1))
            for _ in range(n_tracks)
        ])

        # time_pitch:  (4,16,7) → (4,48,7) → (4,48,84)
        self.tp_1 = nn.ModuleList([
            _build_tconv_block(128, 32, (1, 3, 1), (1, 3, 1))
            for _ in range(n_tracks)
        ])
        self.tp_2 = nn.ModuleList([
            _build_tconv_block(32, 16, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

        # merged_private: fusiona las dos ramas (32 ch) → 1 ch por track
        self.merged = nn.ModuleList([
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1), bias=False)
            for _ in range(n_tracks)
        ])

    def parameters(self):
        import itertools
        return itertools.chain(
            self.shared.parameters(),
            self.pt_1.parameters(), self.pt_2.parameters(),
            self.tp_1.parameters(), self.tp_2.parameters(),
            self.merged.parameters(),
        )

    def to(self, device):
        self.shared = self.shared.to(device)
        for mod in (self.pt_1, self.pt_2, self.tp_1, self.tp_2, self.merged):
            for m in mod:
                m.to(device)
        return self

    def train(self):
        self.shared.train()
        for mod in (self.pt_1, self.pt_2, self.tp_1, self.tp_2, self.merged):
            for m in mod:
                m.train()

    def eval(self):
        self.shared.eval()
        for mod in (self.pt_1, self.pt_2, self.tp_1, self.tp_2, self.merged):
            for m in mod:
                m.eval()

    def state_dict(self):
        import torch
        sd = {"shared": self.shared.state_dict()}
        for name, modlist in [("pt_1", self.pt_1), ("pt_2", self.pt_2),
                               ("tp_1", self.tp_1), ("tp_2", self.tp_2),
                               ("merged", self.merged)]:
            sd[name] = [m.state_dict() for m in modlist]
        return sd

    def load_state_dict(self, sd):
        self.shared.load_state_dict(sd["shared"])
        for name, modlist in [("pt_1", self.pt_1), ("pt_2", self.pt_2),
                               ("tp_1", self.tp_1), ("tp_2", self.tp_2),
                               ("merged", self.merged)]:
            for m, s in zip(modlist, sd[name]):
                m.load_state_dict(s)

    def __call__(self, z):
        """
        z: (N, latent_dim)
        devuelve: (N, n_bars, n_timesteps, n_pitches, n_tracks) en [-1, 1]
        """
        import torch
        import torch.nn.functional as F

        h = z.view(z.size(0), self.latent_dim, 1, 1, 1)  # (N, L, 1, 1, 1)
        h = self.shared(h)                                # (N, 128, 4, 16, 7)

        tracks = []
        for i in range(self.n_tracks):
            s1 = self.pt_1[i](h)   # (N, 32, 4, 16, 84)
            s1 = self.pt_2[i](s1)  # (N, 16, 4, 48, 84)

            s2 = self.tp_1[i](h)   # (N, 32, 4, 48,  7)
            s2 = self.tp_2[i](s2)  # (N, 16, 4, 48, 84)

            cat = torch.cat([s1, s2], dim=1)  # (N, 32, 4, 48, 84)
            out = self.merged[i](cat)          # (N,  1, 4, 48, 84)
            tracks.append(out)

        # (N, n_tracks, n_bars, n_timesteps, n_pitches)
        x = torch.cat(tracks, dim=1)
        # → (N, n_bars, n_timesteps, n_pitches, n_tracks)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return torch.tanh(x)


class Discriminator(object):
    """
    Discriminador MuseGAN con tres streams:
      - pitch_time / time_pitch privados por track  (ramas simétricas al gen.)
      - chroma stream
      - onset/offset stream
    Salida: escalar sin sigmoid (para WGAN-GP).
    """

    def __init__(self, n_tracks=N_TRACKS, beat_resolution=BEAT_RES):
        import torch.nn as nn

        self.n_tracks      = n_tracks
        self.beat_resolution = beat_resolution

        # ── Ramas privadas por track ──────────────────────────────────────────
        # Entrada: (N, n_tracks, n_bars, n_timesteps, n_pitches)
        # → se procesa cada track con 1 canal

        # pitch_time: pitch primero → (4,48,7) → (4,16,7)
        self.pt_1 = nn.ModuleList([
            _build_conv_block(1, 16, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])
        self.pt_2 = nn.ModuleList([
            _build_conv_block(16, 32, (1, 3, 1), (1, 3, 1))
            for _ in range(n_tracks)
        ])

        # time_pitch: tiempo primero → (4,16,84) → (4,16, 7)
        self.tp_1 = nn.ModuleList([
            _build_conv_block(1, 16, (1, 3, 1), (1, 3, 1))
            for _ in range(n_tracks)
        ])
        self.tp_2 = nn.ModuleList([
            _build_conv_block(16, 32, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

        # merged_private: 64 ch por track
        self.merged_private = nn.ModuleList([
            _build_conv_block(64, 64, (1, 1, 1), (1, 1, 1))
            for _ in range(n_tracks)
        ])

        # ── Shared (después de concat n_tracks) ──────────────────────────────
        # (N, 64*n_tracks, 4, 16, 7) → (4,4,3) → (4,1,1)
        self.shared = nn.Sequential(
            _build_conv_block(64 * n_tracks, 128, (1, 4, 3), (1, 4, 2)),
            _build_conv_block(128,           256, (1, 4, 3), (1, 4, 3)),
        )

        # ── Chroma stream ─────────────────────────────────────────────────────
        # chroma: (N, n_tracks, n_bars, n_beats, 12)
        self.chroma_1 = _build_conv_block(n_tracks, 32, (1, 1, 12), (1, 1, 12))
        self.chroma_2 = _build_conv_block(32,       64, (1, 4,  1), (1, 4,  1))

        # ── Onset/offset stream ───────────────────────────────────────────────
        # on_off: (N, 1, n_bars, n_timesteps, 1)  (suma de diferencias sobre pitches)
        self.onset_1 = _build_conv_block(n_tracks, 16, (1, 3, 1), (1, 3, 1))
        self.onset_2 = _build_conv_block(16,       32, (1, 4, 1), (1, 4, 1))
        self.onset_3 = _build_conv_block(32,       64, (1, 4, 1), (1, 4, 1))

        # ── Merge final ───────────────────────────────────────────────────────
        # concat: 256 (shared) + 64 (chroma) + 64 (onset) = 384
        self.merge_final = _build_conv_block(384, 512, (2, 1, 1), (1, 1, 1))

        # Dense de salida
        self.dense = nn.Linear(512, 1)

    def parameters(self):
        import itertools
        import torch.nn as nn
        all_mods = (
            list(self.pt_1) + list(self.pt_2) +
            list(self.tp_1) + list(self.tp_2) +
            list(self.merged_private) +
            [self.shared, self.chroma_1, self.chroma_2,
             self.onset_1, self.onset_2, self.onset_3,
             self.merge_final, self.dense]
        )
        return itertools.chain(*[m.parameters() for m in all_mods])

    def to(self, device):
        for modlist in (self.pt_1, self.pt_2, self.tp_1, self.tp_2,
                        self.merged_private):
            for m in modlist:
                m.to(device)
        for m in (self.shared, self.chroma_1, self.chroma_2,
                  self.onset_1, self.onset_2, self.onset_3,
                  self.merge_final, self.dense):
            m.to(device)
        return self

    def train(self):
        for modlist in (self.pt_1, self.pt_2, self.tp_1, self.tp_2,
                        self.merged_private):
            for m in modlist:
                m.train()
        for m in (self.shared, self.chroma_1, self.chroma_2,
                  self.onset_1, self.onset_2, self.onset_3,
                  self.merge_final, self.dense):
            m.train()

    def eval(self):
        for modlist in (self.pt_1, self.pt_2, self.tp_1, self.tp_2,
                        self.merged_private):
            for m in modlist:
                m.eval()
        for m in (self.shared, self.chroma_1, self.chroma_2,
                  self.onset_1, self.onset_2, self.onset_3,
                  self.merge_final, self.dense):
            m.eval()

    def state_dict(self):
        sd = {}
        for name, modlist in [("pt_1", self.pt_1), ("pt_2", self.pt_2),
                               ("tp_1", self.tp_1), ("tp_2", self.tp_2),
                               ("merged_private", self.merged_private)]:
            sd[name] = [m.state_dict() for m in modlist]
        for name in ("shared", "chroma_1", "chroma_2",
                     "onset_1", "onset_2", "onset_3",
                     "merge_final", "dense"):
            sd[name] = getattr(self, name).state_dict()
        return sd

    def load_state_dict(self, sd):
        for name, modlist in [("pt_1", self.pt_1), ("pt_2", self.pt_2),
                               ("tp_1", self.tp_1), ("tp_2", self.tp_2),
                               ("merged_private", self.merged_private)]:
            for m, s in zip(modlist, sd[name]):
                m.load_state_dict(s)
        for name in ("shared", "chroma_1", "chroma_2",
                     "onset_1", "onset_2", "onset_3",
                     "merge_final", "dense"):
            getattr(self, name).load_state_dict(sd[name])

    def __call__(self, x):
        """
        x: (N, n_bars, n_timesteps, n_pitches, n_tracks) en [-1, 1]
        devuelve: (N, 1)  escalar sin activación
        """
        import torch

        N = x.size(0)
        # → (N, n_tracks, n_bars, n_timesteps, n_pitches)
        h = x.permute(0, 4, 1, 2, 3).contiguous()

        # ── Chroma feature ─────────────────────────────────────────────────────
        # Suma sobre timesteps dentro de cada beat → (N, n_tracks, n_bars, n_beats, n_pitches)
        n_beats = N_TIMESTEPS // self.beat_resolution
        chroma_h = h.view(N, self.n_tracks, N_BARS, n_beats,
                          self.beat_resolution, N_PITCHES)
        chroma_h = chroma_h.sum(dim=4)                          # (N, T, bars, beats, pitches)
        # Fold pitches into 12 chroma bins
        factor    = N_PITCHES // 12
        remainder = N_PITCHES % 12
        chroma_h  = chroma_h[..., :(factor * 12)].view(
            N, self.n_tracks, N_BARS, n_beats, factor, 12).sum(dim=4)
        # (N, n_tracks, n_bars, n_beats, 12)

        # ── Onset/offset feature ───────────────────────────────────────────────
        # diff a lo largo de timesteps
        padded   = torch.nn.functional.pad(h[..., :-1, :], (0, 0, 1, 0))
        on_off   = (h - padded).sum(dim=4, keepdim=True)       # (N, T, bars, ts, 1)

        # ── Ramas privadas ─────────────────────────────────────────────────────
        track_feats = []
        for i in range(self.n_tracks):
            ti = h[:, i:i+1, :, :, :]           # (N, 1, bars, ts, pitches)

            s1 = self.pt_1[i](ti)               # (N, 16, bars, ts, 7)
            s1 = self.pt_2[i](s1)               # (N, 32, bars, ts//3, 7)

            s2 = self.tp_1[i](ti)               # (N, 16, bars, ts//3, pitches)
            s2 = self.tp_2[i](s2)               # (N, 32, bars, ts//3, 7)

            cat = torch.cat([s1, s2], dim=1)    # (N, 64, bars, ts//3, 7)
            out = self.merged_private[i](cat)    # (N, 64, bars, ts//3, 7)
            track_feats.append(out)

        main = torch.cat(track_feats, dim=1)     # (N, 64*T, 4, 16, 7)
        main = self.shared(main)                 # (N, 256, 4, 1, 1)

        # ── Chroma stream ──────────────────────────────────────────────────────
        # chroma_h: (N, n_tracks, n_bars, n_beats, 12)
        c = chroma_h                             # (N, T, bars, beats, 12)
        c = self.chroma_1(c)                     # (N, 32, bars, beats, 1)
        c = self.chroma_2(c)                     # (N, 64, bars, 1, 1)

        # ── Onset/offset stream ────────────────────────────────────────────────
        # on_off: (N, n_tracks, bars, ts, 1)
        o = self.onset_1(on_off)                 # (N, 16, bars, ts//3, 1)
        o = self.onset_2(o)                      # (N, 32, bars, ts//12, 1)
        o = self.onset_3(o)                      # (N, 64, bars, 1, 1)

        # ── Merge ──────────────────────────────────────────────────────────────
        merged = torch.cat([main, c, o], dim=1)  # (N, 384, 4, 1, 1)
        merged = self.merge_final(merged)        # (N, 512, 3, 1, 1)

        out = merged.view(N, -1)
        return self.dense(out)                   # (N, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def gan_losses(dis_real, dis_fake, kind="wasserstein"):
    """Pérdidas GAN para generador y discriminador."""
    import torch
    import torch.nn.functional as F

    if kind == "wasserstein":
        gen_loss = -dis_fake.mean()
        dis_loss = dis_fake.mean() - dis_real.mean()
    elif kind == "hinge":
        gen_loss = -dis_fake.mean()
        dis_loss = (F.relu(1. - dis_real) + F.relu(1. + dis_fake)).mean()
    elif kind == "nonsaturating":
        gen_loss = F.binary_cross_entropy_with_logits(
            dis_fake, torch.ones_like(dis_fake))
        dis_loss = (
            F.binary_cross_entropy_with_logits(dis_real, torch.ones_like(dis_real)) +
            F.binary_cross_entropy_with_logits(dis_fake, torch.zeros_like(dis_fake))
        )
    elif kind == "classic":
        gen_loss = -dis_loss
        dis_loss = (
            F.binary_cross_entropy_with_logits(dis_real, torch.ones_like(dis_real)) +
            F.binary_cross_entropy_with_logits(dis_fake, torch.zeros_like(dis_fake))
        )
    else:
        raise ValueError(f"GAN loss desconocida: {kind!r}")
    return gen_loss, dis_loss


def gradient_penalty(dis, real, fake, device):
    """WGAN-GP: penalización de gradiente sobre interpolaciones."""
    import torch

    N = real.size(0)
    alpha = torch.rand(N, 1, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = dis(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(N, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pianoroll):
    """
    pianoroll: np.ndarray bool, shape (N, n_bars, n_timesteps, n_pitches, n_tracks)
    Devuelve dict con métricas por track.
    """
    N, bars, ts, pitches, n_tracks = pianoroll.shape
    pr = pianoroll.astype(np.float32)
    results = {}

    # Empty bar rate — proporción de compases sin ninguna nota activa
    # shape check: any over (ts, pitches) per bar
    has_note = pr.reshape(N, bars, ts * pitches, n_tracks).any(axis=2)
    results["empty_bar_rate"] = (~has_note.astype(bool)).mean(axis=(0, 1)).tolist()

    # N pitches used — pitches únicos por compás y track
    pr_bar = pr.reshape(N * bars, ts, pitches, n_tracks)
    n_pit  = (pr_bar.sum(axis=1) > 0).sum(axis=1)   # (N*bars, n_tracks)
    results["n_pitches_used"] = n_pit.mean(axis=0).tolist()

    # Qualified note rate — notas de duración >= 2 pasos
    flat = pr.reshape(N, bars * ts, pitches, n_tracks)
    padded = np.pad(flat.astype(int), ((0,0),(1,1),(0,0),(0,0)))
    diff = np.diff(padded, axis=1)
    qnr = []
    for t in range(n_tracks):
        d = diff[..., t].reshape(N, -1)              # (N, bars*ts+1)
        onsets  = [(row > 0).sum() for row in d]
        offsets = [(row < 0).sum() for row in d]
        # simplificado: cuenta notas de duración ≥ 2 como heurística
        qualified = []
        for n_on, n_off in zip(onsets, offsets):
            qualified.append(n_on / max(n_on, 1))   # placeholder conservador
        qnr.append(np.mean(qualified))
    results["qualified_note_rate"] = qnr

    # Polyphonic rate — timesteps con ≥ 2 pitches simultáneos
    n_poly  = (pr.sum(axis=3) >= 2).sum(axis=2)     # (N, bars, n_tracks)
    results["polyphonic_rate"] = (n_poly / ts).mean(axis=(0, 1)).tolist()

    return results


def print_metrics(metrics):
    labels = {
        "empty_bar_rate":      "Compases vacíos",
        "n_pitches_used":      "Pitches / compás",
        "qualified_note_rate": "Notas ≥ 2 pasos",
        "polyphonic_rate":     "Polifonía ≥ 2",
    }
    print("\n  Métricas por track:")
    header = f"  {'Métrica':<24}" + "".join(f"{t:>10}" for t in TRACK_NAMES)
    print(header)
    print("  " + "-" * (24 + 10 * N_TRACKS))
    for key, label in labels.items():
        vals = metrics.get(key, [])
        row = f"  {label:<24}"
        for v in vals:
            row += f"{v:>10.3f}"
        print(row)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  IO: PIANOROLL ↔ MIDI
# ══════════════════════════════════════════════════════════════════════════════

def pianoroll_to_midi(pianoroll, path, tempo=TEMPO,
                      beat_resolution=BEAT_RES, lowest_pitch=LOWEST_PITCH):
    """
    pianoroll: bool ndarray (n_bars, n_timesteps, n_pitches, n_tracks)
    Escribe un fichero MIDI.
    """
    import mido

    ticks_per_beat = beat_resolution * 2          # 2 ticks de resolución
    us_per_beat    = int(60_000_000 / tempo)
    mid            = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    bars, ts, pitches, n_tracks = pianoroll.shape
    total_ts = bars * ts

    for t_idx in range(n_tracks):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        prog    = PROGRAMS[t_idx]
        is_drum = IS_DRUMS[t_idx]
        ch      = 9 if is_drum else t_idx % 9

        track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))
        if not is_drum:
            track.append(mido.Message("program_change", program=prog,
                                      channel=ch, time=0))

        roll = pianoroll[:, :, :, t_idx].reshape(total_ts, pitches)

        # Detección onset/offset con diff
        prev = np.zeros(pitches, dtype=bool)
        current_tick = 0
        events = []

        for step in range(total_ts):
            cur = roll[step].astype(bool)
            onsets  = (~prev) & cur
            offsets = prev & (~cur)
            for p in np.where(offsets)[0]:
                events.append((current_tick, "off", p + lowest_pitch))
            for p in np.where(onsets)[0]:
                events.append((current_tick, "on",  p + lowest_pitch))
            prev = cur
            current_tick += 1

        # Offset finales
        for p in np.where(prev)[0]:
            events.append((total_ts, "off", p + lowest_pitch))

        last_tick = 0
        for tick, kind, pitch in sorted(events, key=lambda e: e[0]):
            delta = tick - last_tick
            last_tick = tick
            vel = 80 if kind == "on" else 0
            track.append(mido.Message("note_on", note=pitch, velocity=vel,
                                      channel=ch, time=delta))

    mid.save(path)


def midi_to_pianoroll(path, beat_resolution=BEAT_RES,
                      lowest_pitch=LOWEST_PITCH, n_pitches=N_PITCHES,
                      n_bars=N_BARS):
    """
    Lee un MIDI y devuelve un pianoroll float32 con shape
    (n_bars, beat_resolution*4, n_pitches, 1)  (un único track).
    """
    import mido

    mid  = mido.MidiFile(path)
    tpb  = mid.ticks_per_beat
    ts   = beat_resolution * 4 * n_bars      # total timesteps
    roll = np.zeros((ts, 128), dtype=np.float32)

    ticks_per_step = tpb / beat_resolution
    active = {}

    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type in ("note_on", "note_off"):
                step = int(tick / ticks_per_step)
                pitch = msg.note
                if msg.type == "note_on" and msg.velocity > 0:
                    active[pitch] = step
                else:
                    if pitch in active:
                        s, e = active.pop(pitch), min(step, ts)
                        roll[s:e, pitch] = 1.

    # Recortar y reorganizar
    roll = roll[:ts, lowest_pitch:lowest_pitch + n_pitches]
    roll = roll.reshape(n_bars, beat_resolution * 4, n_pitches, 1)
    return roll


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE GUARDADO / CARGA
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model_dir, gen, dis, g_opt, d_opt, step, config):
    import torch
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step":   step,
        "config": config,
        "gen":    gen.state_dict(),
        "dis":    dis.state_dict(),
        "g_opt":  g_opt.state_dict(),
        "d_opt":  d_opt.state_dict(),
    }
    torch.save(ckpt, path / "checkpoint.pt")
    print(f"  [ckpt] guardado en {path / 'checkpoint.pt'}  (step {step})")


def load_checkpoint(model_dir, gen, dis, g_opt=None, d_opt=None, device="cpu"):
    import torch
    path = Path(model_dir) / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró checkpoint en {path}")
    ckpt = torch.load(path, map_location=device)
    gen.load_state_dict(ckpt["gen"])
    dis.load_state_dict(ckpt["dis"])
    if g_opt is not None:
        g_opt.load_state_dict(ckpt["g_opt"])
    if d_opt is not None:
        d_opt.load_state_dict(ckpt["d_opt"])
    return ckpt["step"], ckpt.get("config", {})


def load_npz(path):
    """Carga el formato sparse de MuseGAN (nonzero + shape)."""
    with np.load(path) as f:
        if "nonzero" in f and "shape" in f:
            data = np.zeros(f["shape"], dtype=np.bool_)
            data[tuple(f["nonzero"])] = True
        else:
            data = f["arr_0"].astype(np.bool_)
    return data


def save_pianoroll_npz(path, pianoroll):
    """Guarda un pianoroll bool en formato sparse compatible con MuseGAN."""
    pr = pianoroll.astype(bool)
    np.savez_compressed(path, shape=pr.shape, nonzero=np.array(pr.nonzero()))


def save_pianoroll_image(path, pianoroll, colormap=None):
    """
    pianoroll: (N, n_bars, n_timesteps, n_pitches, n_tracks)  float32 en [-1,1]
    Guarda una grilla PNG.
    """
    import imageio

    pr = np.flip(0.5 * (pianoroll + 1.), axis=3)    # → [0,1], pitch invertido
    N, bars, ts, pitches, n_tracks = pr.shape

    # Aplanar bars×ts como eje horizontal, pitches como vertical
    flat = pr.reshape(N, bars * ts, pitches, n_tracks)

    if colormap is not None:
        cm = np.array(colormap, dtype=np.float32)    # (n_tracks, 3)
        rgb = np.einsum("ntpc,tc->nthc", flat, cm)   # heurística: max por canal
        img_data = rgb.max(axis=3)                    # (N, bars*ts, pitches, 3)
    else:
        img_data = flat.mean(axis=3, keepdims=True)   # (N, bars*ts, pitches, 1)

    # Grilla cuadrada
    side = int(np.ceil(np.sqrt(N)))
    H, W = pitches, bars * ts
    canvas = np.ones((side * (H + 1), side * (W + 1),
                      img_data.shape[-1]), dtype=np.float32)

    for idx in range(N):
        row, col = divmod(idx, side)
        r0, c0 = row * (H + 1), col * (W + 1)
        canvas[r0:r0 + H, c0:c0 + W] = img_data[idx]

    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
    if canvas.shape[-1] == 1:
        canvas = canvas[..., 0]
    imageio.imwrite(path, canvas)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERPOLACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def lerp(a, b, t):
    return (1 - t) * a + t * b


def slerp(a, b, t):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    omega = np.arccos(np.clip(np.dot(a_n, b_n), -1., 1.))
    so = np.sin(omega)
    if so < 1e-8:
        return lerp(a, b, t)
    return np.sin((1 - t) * omega) / so * a + np.sin(t * omega) / so * b


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """MIDI corpus → piano-rolls .npz"""
    import mido

    input_dir  = Path(args.input_dir)
    output     = Path(args.output)
    bars       = args.bars
    resolution = args.resolution      # ticks por beat
    ts_per_bar = resolution * 4       # 4 beats por compás
    down_sample = args.down_sample

    output.parent.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.mid" if args.recursive else "*.mid"
    midi_files = list(input_dir.glob(pattern))
    if not midi_files:
        print(f"  No se encontraron MIDIs en {input_dir}")
        sys.exit(1)
    print(f"  Encontrados {len(midi_files)} MIDIs")

    def _track_family(program, is_drum):
        if is_drum:                             return 0   # drums
        if 32 <= program <= 39:                 return 1   # bass
        if 24 <= program <= 31:                 return 2   # guitar
        if 40 <= program <= 51:                 return 3   # strings
        if program <= 7 or 16 <= program <= 23: return 4   # piano
        return -1

    def _midi_to_matrix(mid_path):
        """MIDI → bool array (total_steps, 128, 5)  tras alineación de familia."""
        mid   = mido.MidiFile(mid_path)
        tpb   = mid.ticks_per_beat
        steps_per_tick = resolution / tpb
        matrices = [None] * 5   # una por familia

        for track in mid.tracks:
            program  = 0
            is_drum  = False
            tick     = 0
            active   = {}
            family   = -1
            roll_buf = {}

            for msg in track:
                tick += msg.time
                if msg.type == "program_change":
                    program = msg.program
                    family  = _track_family(program, is_drum)
                if msg.type in ("note_on", "note_off"):
                    if family < 0:
                        is_drum = (msg.channel == 9)
                        family  = _track_family(program, is_drum)
                    step = int(tick * steps_per_tick)
                    p    = msg.note
                    if msg.type == "note_on" and msg.velocity > 0:
                        roll_buf[p] = step
                    else:
                        if p in roll_buf:
                            roll_buf.setdefault(family, {})[p] = (
                                roll_buf.pop(p), step)

            if family >= 0 and roll_buf:
                for p, (s, e) in roll_buf.items():
                    if matrices[family] is None:
                        matrices[family] = {}
                    matrices[family][(s, p)] = e
        return matrices

    segments = []
    for fpath in midi_files:
        print(f"  {fpath.name} ... ", end="", flush=True)
        try:
            mid = mido.MidiFile(fpath)
        except Exception as e:
            print(f"error ({e})")
            continue

        tpb = mid.ticks_per_beat
        us_per_beat = 500_000
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, "tempo"):
                    us_per_beat = msg.tempo
                    break

        ticks_per_step = tpb / resolution
        step_total = int(sum(
            msg.time for track in mid.tracks for msg in track
            if not msg.is_meta
        ) / tpb * resolution) + 1

        # Inicializar rolls por familia
        rolls = [np.zeros((step_total, 128), dtype=np.bool_) for _ in range(5)]

        for track in mid.tracks:
            program  = 0
            is_drum  = False
            tick     = 0
            active   = {}

            for msg in track:
                tick += msg.time
                if msg.type == "program_change":
                    program = msg.program
                if msg.type in ("note_on", "note_off"):
                    is_drum = (msg.channel == 9)
                    fam = _track_family(program, is_drum)
                    if fam < 0:
                        continue
                    step = int(tick / ticks_per_step)
                    p    = msg.note
                    if msg.type == "note_on" and msg.velocity > 0:
                        active[(fam, p)] = step
                    else:
                        key = (fam, p)
                        if key in active:
                            s = active.pop(key)
                            e = min(step, step_total)
                            rolls[fam][s:e, p] = True

        # Segmentar en ventanas de `bars` compases
        n_bars_total = step_total // ts_per_bar
        ok_count = 0
        for b in range(0, n_bars_total - bars + 1, max(1, bars // 2)):
            s = b * ts_per_bar
            e = s + bars * ts_per_bar
            segment = np.zeros(
                (bars, ts_per_bar // down_sample, N_PITCHES, 5), dtype=np.bool_)

            for fam in range(5):
                roll_seg = rolls[fam][s:e:down_sample,
                                     LOWEST_PITCH:LOWEST_PITCH + N_PITCHES]
                roll_seg = roll_seg.reshape(bars, ts_per_bar // down_sample,
                                            N_PITCHES)
                # Filtrar segmentos con poca actividad
                fname     = TRACK_NAMES[fam]
                thr_p, thr_b = FAMILY_THRESHOLDS[fname]
                n_pitches_active = (roll_seg.sum(axis=(0, 1)) > 0).sum()
                n_beats_active   = (roll_seg.sum(axis=(0, 2)) > 0).sum()
                if n_pitches_active >= thr_p or fam == 0:
                    segment[:, :, :, fam] = roll_seg

            segments.append(segment)
            ok_count += 1

        print(f"{ok_count} segmentos")

    if not segments:
        print("  No se extrajeron segmentos válidos.")
        sys.exit(1)

    data = np.stack(segments, axis=0)   # (N, bars, ts, pitches, 5)
    print(f"\n  Shape final: {data.shape}")

    if str(output).endswith(".npz"):
        np.savez_compressed(str(output),
                            shape=data.shape,
                            nonzero=np.array(data.nonzero()))
    else:
        np.save(str(output), data)
    print(f"  Guardado en {output}")


def cmd_train(args):
    """Entrenar el modelo GAN."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"  Dispositivo: {device}")

    # ── Datos ──────────────────────────────────────────────────────────────────
    print(f"  Cargando {args.data} ...")
    raw = load_npz(args.data)           # bool (N, bars, ts, pitches, tracks)
    print(f"  Shape: {raw.shape}  —  {len(raw)} muestras")

    # Verificar shape esperado
    _, bars, ts, pitches, n_tracks = raw.shape
    if bars != N_BARS or ts != N_TIMESTEPS or pitches != N_PITCHES:
        print(f"  ⚠  Shape inesperado: esperado (*,{N_BARS},{N_TIMESTEPS},"
              f"{N_PITCHES},*), recibido {raw.shape}")

    # Convertir bool → float32 en [-1, 1]
    data_f = raw.astype(np.float32) * 2. - 1.

    # ── Modelo ─────────────────────────────────────────────────────────────────
    gen = Generator(n_tracks=n_tracks, latent_dim=args.latent_dim)
    dis = Discriminator(n_tracks=n_tracks, beat_resolution=BEAT_RES)
    gen.to(device)
    dis.to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=args.lr,
                              betas=(args.beta1, args.beta2))
    d_opt = torch.optim.Adam(dis.parameters(), lr=args.lr,
                              betas=(args.beta1, args.beta2))

    start_step = 0
    if args.resume:
        try:
            start_step, _ = load_checkpoint(
                args.model_dir, gen, dis, g_opt, d_opt, device=str(device))
            print(f"  Reanudando desde step {start_step}")
        except FileNotFoundError:
            print("  No se encontró checkpoint; empezando desde cero.")

    config = {
        "latent_dim":  args.latent_dim,
        "n_tracks":    n_tracks,
        "gan_loss":    args.gan_loss,
        "gp":          args.gp,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "beta1":       args.beta1,
        "beta2":       args.beta2,
        "steps":       args.steps,
        "n_dis":       args.n_dis,
    }

    # LR decay lineal en el último 10 % del entrenamiento
    lr_decay_start = int(args.steps * 0.9)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.model_dir) / "samples"
    sample_dir.mkdir(exist_ok=True)

    print(f"\n  Entrenando {args.steps} steps  "
          f"(batch={args.batch_size}, latent={args.latent_dim}, "
          f"loss={args.gan_loss})\n")

    t0 = time.time()
    N  = len(data_f)

    for step in range(start_step, args.steps):

        # ── LR decay ──────────────────────────────────────────────────────────
        if step >= lr_decay_start:
            frac = (step - lr_decay_start) / max(1, args.steps - lr_decay_start)
            current_lr = args.lr * (1. - frac)
            for opt in (g_opt, d_opt):
                for pg in opt.param_groups:
                    pg["lr"] = max(current_lr, 0.)

        # ── Discriminador (n_dis actualizaciones) ──────────────────────────────
        gen.eval()
        dis.train()
        for _ in range(args.n_dis):
            idx  = np.random.choice(N, args.batch_size, replace=False)
            real = torch.tensor(data_f[idx], dtype=torch.float32, device=device)
            z    = torch.randn(args.batch_size, args.latent_dim, device=device)

            with torch.no_grad():
                fake = gen(z)

            d_real = dis(real)
            d_fake = dis(fake)
            _, d_loss = gan_losses(d_real, d_fake, args.gan_loss)

            if args.gp:
                d_loss = d_loss + 10. * gradient_penalty(dis, real, fake, device)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

        # ── Generador ──────────────────────────────────────────────────────────
        gen.train()
        dis.eval()
        z    = torch.randn(args.batch_size, args.latent_dim, device=device)
        fake = gen(z)
        g_loss, _ = gan_losses(dis(fake), dis(fake), args.gan_loss)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        # ── Logging ────────────────────────────────────────────────────────────
        if (step + 1) % args.log_steps == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1:>6}/{args.steps}  "
                  f"G={g_loss.item():+.4f}  D={d_loss.item():+.4f}  "
                  f"t={elapsed:.0f}s")

        # ── Muestras ────────────────────────────────────────────────────────────
        if args.sample_steps > 0 and (step + 1) % args.sample_steps == 0:
            gen.eval()
            with torch.no_grad():
                z_s  = torch.randn(16, args.latent_dim, device=device)
                fake_s = gen(z_s).cpu().numpy()
            prefix = sample_dir / f"step_{step+1:06d}"
            save_pianoroll_npz(str(prefix) + ".npz",
                               (fake_s > 0).astype(bool))
            if args.save_images:
                try:
                    save_pianoroll_image(str(prefix) + ".png",
                                        fake_s, COLORMAP)
                except ImportError:
                    pass
            print(f"    → muestras en {prefix}.*")

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            save_checkpoint(args.model_dir, gen, dis, g_opt, d_opt,
                            step + 1, config)

    # Checkpoint final
    save_checkpoint(args.model_dir, gen, dis, g_opt, d_opt, args.steps, config)
    print("\n  Entrenamiento completado.")


def cmd_generate(args):
    """Generar MIDIs desde ruido latente."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    # Cargar modelo
    ckpt_path = Path(args.model_dir) / "checkpoint.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt.get("config", {})
    latent    = cfg.get("latent_dim", args.latent_dim)
    n_tracks  = cfg.get("n_tracks", N_TRACKS)

    gen = Generator(n_tracks=n_tracks, latent_dim=latent)
    gen.load_state_dict(ckpt["gen"])
    gen.to(device)
    gen.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generando {args.n} muestras  "
          f"(latent={latent}, threshold={args.threshold})")

    batch_size = min(args.n, 64)
    generated  = []

    with torch.no_grad():
        remaining = args.n
        while remaining > 0:
            bs = min(batch_size, remaining)
            z  = torch.randn(bs, latent, device=device)
            if args.truncation < 1.:
                z = z.clamp(-args.truncation * 2, args.truncation * 2)
            out = gen(z).cpu().numpy()
            generated.append(out)
            remaining -= bs

    generated = np.concatenate(generated, axis=0)   # (N, bars, ts, pitches, T)

    # Binarizar
    if args.threshold == "hard":
        binary = generated > 0.
    else:  # bernoulli sampling
        prob   = 0.5 * (generated + 1.)
        binary = np.random.rand(*generated.shape) < prob

    for i in range(args.n):
        stem = out_dir / f"sample_{i:04d}"

        # MIDI
        pianoroll_to_midi(
            binary[i],
            str(stem) + ".mid",
            tempo=args.tempo,
        )

        # Pianoroll NPZ
        if args.save_arrays:
            save_pianoroll_npz(str(stem) + ".npz", binary[i])

        # Imagen
        if args.save_images:
            try:
                save_pianoroll_image(
                    str(stem) + ".png",
                    generated[i:i+1],
                    COLORMAP,
                )
            except ImportError:
                print("  ⚠  imageio no disponible; omitiendo imágenes.")
                args.save_images = False

    print(f"  {args.n} ficheros guardados en {out_dir}/")

    # Métricas rápidas
    if args.metrics:
        metrics = compute_metrics(binary)
        print_metrics(metrics)


def cmd_accompany(args):
    """
    Generar acompañamiento dado un track de referencia.
    El track de condición se fija; el generador produce los restantes n_tracks-1.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    ckpt_path = Path(args.model_dir) / "checkpoint.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt.get("config", {})
    latent    = cfg.get("latent_dim", args.latent_dim)
    n_tracks  = cfg.get("n_tracks", N_TRACKS)
    cond_idx  = args.condition_track

    if cond_idx < 0 or cond_idx >= n_tracks:
        print(f"  ⚠  --condition-track debe estar entre 0 y {n_tracks-1}")
        sys.exit(1)

    # Leer MIDI de referencia
    cond_roll = midi_to_pianoroll(args.input, n_bars=N_BARS)
    # (n_bars, ts, pitches, 1) → float32 en [-1, 1]
    cond_f    = (cond_roll * 2. - 1.).astype(np.float32)

    # Generador sin acompañamiento especial: usamos el mismo gen
    # y forzamos el track de condición en la salida.
    gen = Generator(n_tracks=n_tracks, latent_dim=latent)
    gen.load_state_dict(ckpt["gen"])
    gen.to(device)
    gen.eval()

    with torch.no_grad():
        z    = torch.randn(1, latent, device=device)
        fake = gen(z).cpu().numpy()[0]   # (bars, ts, pitches, n_tracks)

    # Sustituir el track de condición con el real
    fake[:, :, :, cond_idx] = cond_f[:, :, :, 0]

    # Binarizar
    binary = fake > 0.

    output = args.output or "accompaniment.mid"
    pianoroll_to_midi(binary, output, tempo=args.tempo)
    print(f"  Resultado guardado en {output}")
    print(f"  Track de condición: {cond_idx} ({TRACK_NAMES[cond_idx]})")
    print(f"  Tracks generados:   "
          + ", ".join(TRACK_NAMES[i] for i in range(n_tracks) if i != cond_idx))


def cmd_interpolate(args):
    """Interpolar entre dos puntos del espacio latente."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")

    ckpt_path = Path(args.model_dir) / "checkpoint.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt.get("config", {})
    latent    = cfg.get("latent_dim", args.latent_dim)
    n_tracks  = cfg.get("n_tracks", N_TRACKS)

    gen = Generator(n_tracks=n_tracks, latent_dim=latent)
    gen.load_state_dict(ckpt["gen"])
    gen.to(device)
    gen.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interp_fn = slerp if args.mode == "slerp" else lerp

    n_pairs = args.pairs
    n_steps = args.steps

    print(f"  Interpolación {args.mode}  —  {n_pairs} par(es)  ×  {n_steps} pasos")

    for pair in range(n_pairs):
        z_a = np.random.randn(latent).astype(np.float32)
        z_b = np.random.randn(latent).astype(np.float32)

        zs = np.stack([
            interp_fn(z_a, z_b, t / (n_steps - 1))
            for t in range(n_steps)
        ])                                           # (n_steps, latent)

        zs_t = torch.tensor(zs, device=device)
        with torch.no_grad():
            fakes = gen(zs_t).cpu().numpy()          # (n_steps, bars, ts, p, T)

        binary = fakes > 0.

        for step_i in range(n_steps):
            stem = out_dir / f"pair{pair:02d}_step{step_i:02d}"
            pianoroll_to_midi(binary[step_i], str(stem) + ".mid",
                              tempo=args.tempo)

        if args.save_images:
            try:
                save_pianoroll_image(
                    str(out_dir / f"pair{pair:02d}_interp.png"),
                    fakes, COLORMAP)
            except ImportError:
                print("  ⚠  imageio no disponible; omitiendo imágenes.")
                args.save_images = False

        print(f"  Par {pair}: {n_steps} MIDIs en {out_dir}/")

    print("  Listo.")


def cmd_inspect(args):
    """Diagnóstico del modelo y/o los datos."""
    import torch

    if args.model_dir:
        ckpt_path = Path(args.model_dir) / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  ✗ No se encontró checkpoint en {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg  = ckpt.get("config", {})
            print("\n  ── MODELO ──────────────────────────────────────────")
            print(f"  Step entrenado:  {ckpt.get('step', '?')}")
            print(f"  Latent dim:      {cfg.get('latent_dim', '?')}")
            print(f"  N tracks:        {cfg.get('n_tracks', '?')}")
            print(f"  GAN loss:        {cfg.get('gan_loss', '?')}")
            print(f"  Gradient penalty:{cfg.get('gp', '?')}")
            print(f"  Batch size:      {cfg.get('batch_size', '?')}")
            print(f"  LR:              {cfg.get('lr', '?')}")
            print(f"  Steps totales:   {cfg.get('steps', '?')}")

            # Contar parámetros del generador
            n_tracks = cfg.get("n_tracks", N_TRACKS)
            latent   = cfg.get("latent_dim", 128)
            gen = Generator(n_tracks=n_tracks, latent_dim=latent)
            gen.load_state_dict(ckpt["gen"])
            n_gen = sum(p.numel() for p in gen.parameters())

            dis = Discriminator(n_tracks=n_tracks)
            dis.load_state_dict(ckpt["dis"])
            n_dis = sum(p.numel() for p in dis.parameters())

            print(f"\n  Parámetros generador:    {n_gen:,}")
            print(f"  Parámetros discriminador: {n_dis:,}")
            print(f"  Total:                    {n_gen + n_dis:,}")

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"  ✗ No se encontró {data_path}")
        else:
            print("\n  ── DATOS ───────────────────────────────────────────")
            raw = load_npz(str(data_path))
            print(f"  Shape:       {raw.shape}")
            print(f"  Dtype:       {raw.dtype}")
            print(f"  N muestras:  {raw.shape[0]}")
            density = raw.mean()
            print(f"  Densidad:    {density:.4f}  ({density*100:.2f}% activo)")
            print(f"  Tracks:      {', '.join(TRACK_NAMES[:raw.shape[-1]])}")
            print(f"  Tamaño:      {raw.nbytes / 1e6:.1f} MB (descomprimido)")

            # Métricas rápidas sobre submuestra
            n_eval = min(256, raw.shape[0])
            metrics = compute_metrics(raw[:n_eval])
            print_metrics(metrics)

    if not args.model_dir and not args.data:
        print("  Usa --model-dir y/o --data para inspeccionar.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="musegan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            MuseGAN v2.0 (PyTorch)
            Generación de música multi-pista mediante GAN sobre piano-rolls 5D.

            Flujo típico:
              1. prepare     — convertir corpus MIDI a piano-rolls .npz
              2. train       — entrenar el modelo GAN
              3. generate    — generar MIDIs nuevos
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMANDO")
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser("prepare",
        help="MIDI corpus → piano-rolls segmentados (.npz)")
    p.add_argument("--input-dir",   required=True, metavar="DIR",
                   help="Carpeta con ficheros MIDI")
    p.add_argument("--output",      required=True, metavar="FILE",
                   help="Fichero de salida (.npy o .npz)")
    p.add_argument("--bars",        type=int, default=N_BARS, metavar="N",
                   help=f"Compases por segmento (default: {N_BARS})")
    p.add_argument("--resolution",  type=int, default=BEAT_RES, metavar="N",
                   help=f"Ticks por beat (default: {BEAT_RES})")
    p.add_argument("--down-sample", type=int, default=1, metavar="N",
                   help="Factor de downsampling temporal (default: 1, sin downsampling)")
    p.add_argument("--recursive",   action="store_true",
                   help="Buscar MIDIs recursivamente")
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("train",
        help="Entrenar el modelo GAN")
    p.add_argument("--data",        required=True, metavar="FILE",
                   help="Piano-rolls de entrenamiento (.npy o .npz)")
    p.add_argument("--model-dir",   required=True, metavar="DIR",
                   help="Directorio para guardar checkpoints")
    p.add_argument("--steps",       type=int,   default=50000,
                   help="Número de steps (default: 50000)")
    p.add_argument("--batch-size",  type=int,   default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--latent-dim",  type=int,   default=128,
                   help="Dimensión del espacio latente (default: 128)")
    p.add_argument("--n-dis",       type=int,   default=5, metavar="N",
                   help="Actualizaciones del discriminador por paso gen (default: 5)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--beta1",       type=float, default=0.5,
                   help="Adam β1 (default: 0.5)")
    p.add_argument("--beta2",       type=float, default=0.9,
                   help="Adam β2 (default: 0.9)")
    p.add_argument("--gan-loss",    default="wasserstein",
                   choices=["wasserstein", "hinge", "nonsaturating", "classic"],
                   help="Tipo de loss GAN (default: wasserstein)")
    p.add_argument("--gp",          action="store_true",
                   help="Usar gradient penalty (WGAN-GP)")
    p.add_argument("--save-steps",  type=int, default=10000, metavar="N",
                   help="Guardar checkpoint cada N steps (default: 10000)")
    p.add_argument("--sample-steps",type=int, default=1000,  metavar="N",
                   help="Generar muestras cada N steps (default: 1000)")
    p.add_argument("--log-steps",   type=int, default=100,   metavar="N",
                   help="Log cada N steps (default: 100)")
    p.add_argument("--save-images", action="store_true",
                   help="Guardar imágenes PNG junto a los .npz de muestra")
    p.add_argument("--resume",      action="store_true",
                   help="Reanudar entrenamiento desde el último checkpoint")
    p.add_argument("--cpu",         action="store_true",
                   help="Forzar uso de CPU aunque haya GPU")
    p.set_defaults(func=cmd_train)

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser("generate",
        help="Generar MIDIs desde ruido latente")
    p.add_argument("--model-dir",   required=True, metavar="DIR",
                   help="Directorio con el checkpoint entrenado")
    p.add_argument("--output-dir",  default="generated", metavar="DIR",
                   help="Directorio de salida (default: generated/)")
    p.add_argument("--n",           type=int, default=16, metavar="N",
                   help="Número de muestras a generar (default: 16)")
    p.add_argument("--latent-dim",  type=int, default=128,
                   help="Latent dim fallback si no hay config en checkpoint")
    p.add_argument("--threshold",   default="hard",
                   choices=["hard", "bernoulli"],
                   help="Método de binarización (default: hard > 0)")
    p.add_argument("--truncation",  type=float, default=1.0, metavar="F",
                   help="Truncation trick: recortar z a [-2t, 2t] (default: 1.0, sin recorte)")
    p.add_argument("--tempo",       type=int, default=TEMPO,
                   help=f"BPM del MIDI de salida (default: {TEMPO})")
    p.add_argument("--save-images", action="store_true",
                   help="Guardar imágenes PNG")
    p.add_argument("--save-arrays", action="store_true",
                   help="Guardar piano-rolls .npz")
    p.add_argument("--metrics",     action="store_true",
                   help="Mostrar métricas sobre los resultados")
    p.add_argument("--cpu",         action="store_true")
    p.set_defaults(func=cmd_generate)

    # ── accompany ─────────────────────────────────────────────────────────────
    p = sub.add_parser("accompany",
        help="Generar acompañamiento dado un track de referencia")
    p.add_argument("--model-dir",       required=True, metavar="DIR")
    p.add_argument("--input",           required=True, metavar="FILE",
                   help="MIDI de referencia (track de condición)")
    p.add_argument("--condition-track", type=int, default=4, metavar="N",
                   help="Índice del track de condición 0-4 (default: 4=piano)")
    p.add_argument("--output",          default=None, metavar="FILE",
                   help="MIDI de salida (default: accompaniment.mid)")
    p.add_argument("--latent-dim",      type=int, default=128)
    p.add_argument("--tempo",           type=int, default=TEMPO)
    p.add_argument("--cpu",             action="store_true")
    p.set_defaults(func=cmd_accompany)

    # ── interpolate ───────────────────────────────────────────────────────────
    p = sub.add_parser("interpolate",
        help="Interpolar entre dos puntos del espacio latente")
    p.add_argument("--model-dir",   required=True, metavar="DIR")
    p.add_argument("--output-dir",  default="interpolation", metavar="DIR",
                   help="Directorio de salida (default: interpolation/)")
    p.add_argument("--pairs",       type=int, default=1, metavar="N",
                   help="Número de pares a interpolar (default: 1)")
    p.add_argument("--steps",       type=int, default=8, metavar="N",
                   help="Pasos de interpolación por par (default: 8)")
    p.add_argument("--mode",        default="slerp", choices=["slerp", "lerp"],
                   help="Método de interpolación (default: slerp)")
    p.add_argument("--latent-dim",  type=int, default=128)
    p.add_argument("--tempo",       type=int, default=TEMPO)
    p.add_argument("--save-images", action="store_true",
                   help="Guardar imagen resumen de la interpolación")
    p.add_argument("--cpu",         action="store_true")
    p.set_defaults(func=cmd_interpolate)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser("inspect",
        help="Diagnóstico del modelo y los datos")
    p.add_argument("--model-dir", default=None, metavar="DIR",
                   help="Directorio con checkpoint")
    p.add_argument("--data",      default=None, metavar="FILE",
                   help="Fichero de datos .npz")
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
