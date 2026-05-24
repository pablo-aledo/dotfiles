#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CYCLEGAN COMPOSER  v1                                    ║
║         Style-transfer end-to-end mediante CycleGAN sobre piano rolls        ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    2 Generadores ResNet (G_A→B, G_B→A) + 2 Discriminadores PatchGAN         ║
║    Entrenamiento unpaired: corpus A y corpus B no necesitan                  ║
║    correspondencia canción-a-canción.                                        ║
║    Losses: adversarial + cycle-consistency (λ=10) + identity (λ=0.5)        ║
║                                                                              ║
║  DIFERENCIAS CLAVE vs diffusion_composer:                                    ║
║    · No hay proceso de ruido/denoising — la transferencia es determinista    ║
║    · El estilo se define implícitamente por el corpus de entrenamiento        ║
║    · Un par de modelos (A↔B) = una sola dirección de transferencia           ║
║    · Inferencia muy rápida (un solo paso forward del generador)              ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare      — MIDI corpus → piano rolls segmentados por rol (.npz)       ║
║    train        — Entrena el par de generadores A↔B (corpus-A y corpus-B)   ║
║    transfer     — Transfiere estilo B a una canción del estilo A              ║
║    transfer-inv — Transfiere estilo A a una canción del estilo B              ║
║    style-corpus — Diagnóstico: estadísticas del espacio latente del corpus   ║
║    round-trip   — Diagnóstico: MIDI → piano roll → MIDI sin modelo           ║
║    inspect      — Diagnóstico del modelo y los datos                         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ── Preparar datos ─────────────────────────────────────────────────────────────
# Preparar corpus A (estilo origen) y corpus B (estilo destino) por separado.
python cyclegan_composer.py prepare --input-dir midis_A/ --output-dir data_A/
python cyclegan_composer.py prepare --input-dir midis_B/ --output-dir data_B/

# Con roles reducidos y rango de pitch limitado:
python cyclegan_composer.py prepare \
    --input-dir midis_A/ --output-dir data_A_small/ \
    --disable-roles percussion counterpoint \
    --pitch-range 48

# ── Entrenar el par de generadores A↔B ────────────────────────────────────────
python cyclegan_composer.py train \
    --data-dir-a data_A/ \
    --data-dir-b data_B/ \
    --model-dir  model_cycle/ \
    --epochs 200 \
    --batch-size 4 \
    --lr 2e-4 \
    --lambda-cycle 10.0 \
    --lambda-identity 0.5 \
    --patience 40

# Con roles reducidos:
python cyclegan_composer.py train \
    --data-dir-a data_A_small/ \
    --data-dir-b data_B_small/ \
    --model-dir  model_small/ \
    --disable-roles percussion counterpoint \
    --epochs 200 --batch-size 8

# ── Style transfer A→B ────────────────────────────────────────────────────────
python cyclegan_composer.py transfer \
    --input midis_A/cancion.mid \
    --model-dir model_cycle/ \
    --output resultado_B.mid \
    --threshold-pct 99.0

# ── Style transfer B→A (inverso) ─────────────────────────────────────────────
python cyclegan_composer.py transfer-inv \
    --input midis_B/cancion.mid \
    --model-dir model_cycle/ \
    --output resultado_A.mid \
    --threshold-pct 99.0

# ── Diagnóstico del corpus ────────────────────────────────────────────────────
python cyclegan_composer.py style-corpus \
    --input-dir midis_A/ \
    --model-dir model_cycle/ \
    --direction a2b \
    --output stats_A.json

# ── Diagnóstico básico ────────────────────────────────────────────────────────
python cyclegan_composer.py round-trip --input midis_A/cancion.mid
python cyclegan_composer.py inspect --data-dir data_A/ --model-dir model_cycle/
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES  (idénticas a diffusion_composer)
# ══════════════════════════════════════════════════════════════════════════════

ROLES = ['melody', 'counterpoint', 'accompaniment', 'bass', 'percussion']

ROLE_RANGES = {
    'melody':        (60, 96),
    'counterpoint':  (52, 84),
    'accompaniment': (48, 84),
    'bass':          (28, 55),
    'percussion':    (0,  127),
}

GM_ROLE_HINTS = {
    43: 'bass', 42: 'bass', 58: 'bass', 70: 'bass',
    73: 'melody', 72: 'melody', 56: 'melody', 40: 'melody',
    68: 'counterpoint', 71: 'counterpoint', 41: 'counterpoint',
    48: 'accompaniment', 49: 'accompaniment',
    19: 'accompaniment', 52: 'accompaniment',
    88: 'accompaniment', 89: 'accompaniment',
}

TICKS_PER_BAR_DEFAULT = 48
WINDOW_BARS_DEFAULT   = 4
PITCH_CLASSES         = 128
MIDI_CENTER           = 60


def _pitch_range(n):
    if n is None:
        return None
    half = n // 2
    lo   = max(0,   MIDI_CENTER - half)
    hi   = min(127, lo + n - 1)
    lo   = hi - n + 1
    lo   = max(0, lo)
    return (lo, hi)


def _crop_pitch(roll, pitch_lo, pitch_hi):
    return roll[..., pitch_lo: pitch_hi + 1]


def _pad_pitch(roll, pitch_lo, n_full=128):
    import numpy as np
    n_crop = roll.shape[-1]
    prefix = pitch_lo
    suffix = n_full - pitch_lo - n_crop
    pad_widths = [(0, 0)] * (roll.ndim - 1) + [(prefix, suffix)]
    return np.pad(roll, pad_widths, mode='constant')


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI  (idénticas a diffusion_composer)
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi(path):
    import mido
    return mido.MidiFile(path)


def _extract_note_lists(mid):
    active = {}
    result = {}
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        prog     = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'program_change':
                prog = msg.program
            if msg.type in ('note_on', 'note_off'):
                ch  = msg.channel
                key = (ti, ch, msg.note)
                on  = msg.type == 'note_on' and msg.velocity > 0
                if on:
                    active[key] = (abs_tick, msg.velocity, prog)
                else:
                    if key in active:
                        st, vel, pr = active.pop(key)
                        stream_key  = (ti, ch)
                        result.setdefault(stream_key, []).append(
                            (st, abs_tick, msg.note, vel, pr))
    return result


def _ticks_per_bar(mid):
    return mid.ticks_per_beat * 4


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES  (idéntico a diffusion_composer)
# ══════════════════════════════════════════════════════════════════════════════

class RoleAssigner:
    def assign(self, mid):
        note_lists = _extract_note_lists(mid)
        if not note_lists:
            return {}
        profiles = self._build_profiles(note_lists, mid)
        return self._resolve_roles(profiles)

    def _build_profiles(self, note_lists, mid):
        tpb_raw   = mid.ticks_per_beat
        total_dur = max(n[1] for notes in note_lists.values() for n in notes) if note_lists else 1
        profiles  = []
        for (ti, ch), notes in note_lists.items():
            if not notes:
                continue
            pitches   = [n[2] for n in notes]
            program   = notes[0][4]
            pitch_mean  = sum(pitches) / len(pitches)
            pitch_range = max(pitches) - min(pitches)
            density     = len(notes) / max(total_dur / tpb_raw, 1)
            polyphony   = self._mean_polyphony(notes)
            profiles.append({
                'key': (ti, ch), 'channel': ch, 'program': program,
                'pitch_mean': pitch_mean, 'pitch_range': pitch_range,
                'density': density, 'polyphony': polyphony,
                'n_notes': len(notes),
            })
        return profiles

    @staticmethod
    def _mean_polyphony(notes):
        if len(notes) < 2:
            return 1.0
        events = []
        for (st, en, *_) in notes:
            events.append((st, 1))
            events.append((en, -1))
        events.sort()
        current = 0
        samples = []
        for _, delta in events:
            current += delta
            samples.append(max(current, 0))
        return sum(samples) / len(samples) if samples else 1.0

    def _resolve_roles(self, profiles):
        if not profiles:
            return {}
        assigned   = {}
        unassigned = []
        for p in profiles:
            if p['channel'] == 9:
                if 'percussion' not in assigned:
                    assigned['percussion'] = p['key']
            else:
                unassigned.append(p)

        remaining_roles = [r for r in ROLES if r != 'percussion']

        if len(unassigned) == 1:
            p  = unassigned[0]
            pm = p['pitch_mean']
            if pm >= 60:   role = 'melody'
            elif pm >= 52: role = 'counterpoint'
            elif pm >= 44: role = 'accompaniment'
            else:          role = 'bass'
            assigned[role] = p['key']
            return assigned

        if not unassigned:
            return assigned

        def norm(key):
            vals = [p[key] for p in unassigned]
            lo, hi = min(vals), max(vals)
            span = hi - lo or 1
            return {p['key']: (p[key] - lo) / span for p in unassigned}

        n_pm   = norm('pitch_mean')
        n_pr   = norm('pitch_range')
        n_poly = norm('polyphony')
        n_dens = norm('density')

        def score(p, role):
            k = p['key']
            hint_bonus = 0.25 if GM_ROLE_HINTS.get(p['program']) == role else 0.0
            if role == 'melody':
                return 0.40 * n_pm[k] + 0.35 * n_pr[k] + 0.15 * (1 - n_poly[k]) + hint_bonus
            elif role == 'counterpoint':
                mid_pm = abs(n_pm[k] - 0.65)
                return 0.30 * (1 - mid_pm) + 0.25 * n_pr[k] + 0.20 * (1 - n_poly[k]) + hint_bonus
            elif role == 'accompaniment':
                mid_pm = abs(n_pm[k] - 0.50)
                return 0.40 * n_poly[k] + 0.25 * (1 - mid_pm) + 0.15 * n_dens[k] + hint_bonus
            elif role == 'bass':
                return 0.50 * (1 - n_pm[k]) + 0.25 * (1 - n_pr[k]) + hint_bonus
            return 0.0

        score_matrix = {p['key']: {r: score(p, r) for r in remaining_roles} for p in unassigned}
        taken_keys  = set()
        taken_roles = set()
        pairs = [(score_matrix[p['key']][r], r, p['key']) for p in unassigned for r in remaining_roles]
        pairs.sort(key=lambda x: -x[0])
        for sc, role, key in pairs:
            if role not in taken_roles and key not in taken_keys:
                assigned[role] = key
                taken_roles.add(role)
                taken_keys.add(key)
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  PIANO ROLL CONVERTER  (idéntico a diffusion_composer)
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    def __init__(self, resolution=TICKS_PER_BAR_DEFAULT, window_bars=WINDOW_BARS_DEFAULT):
        self.resolution  = resolution
        self.window_bars = window_bars

    def notes_to_roll(self, notes, tpb_raw, n_bars):
        import numpy as np
        T = n_bars * self.resolution
        roll = np.zeros((T, PITCH_CLASSES), dtype=np.float32)
        scale = self.resolution / tpb_raw
        for (st, en, pitch, vel, _) in notes:
            t0 = int(st * scale)
            t1 = max(t0 + 1, int(en * scale))
            t0 = min(t0, T - 1)
            t1 = min(t1, T)
            roll[t0:t1, pitch] = 1.0
        return roll.reshape(n_bars, self.resolution, PITCH_CLASSES)

    def roll_to_windows(self, roll):
        import numpy as np
        n_bars = roll.shape[0]
        if n_bars < self.window_bars:
            return np.zeros((0, self.window_bars, self.resolution, PITCH_CLASSES),
                            dtype=np.float32)
        windows = []
        for i in range(n_bars - self.window_bars + 1):
            windows.append(roll[i: i + self.window_bars])
        return np.stack(windows, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (un corpus a la vez; para CycleGAN se instancian dos)
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """Dataset de ventanas de piano roll para un corpus.

    Cada item devuelve:
        x : (N_ROLES, resolution, n_pitch)  — compás objetivo
    """
    def __init__(self, data_dir, roles=None):
        import numpy as np
        self.roles   = roles or ROLES
        self.samples = []
        self.n_pitch = None
        self._cache  = {}

        for path in sorted(Path(data_dir).glob('*.npz')):
            try:
                data = np.load(str(path), allow_pickle=True)
                meta = json.loads(str(data['meta_json'][0]))
                if self.n_pitch is None:
                    self.n_pitch = meta.get('n_pitch', PITCH_CLASSES)
                for i in range(meta['n_windows']):
                    self.samples.append((str(path), i, meta))
                self._cache[str(path)] = data
            except Exception:
                continue

        if self.n_pitch is None:
            self.n_pitch = PITCH_CLASSES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        path, widx, meta = self.samples[idx]
        data = self._cache[path]
        resolution = meta['resolution']

        x_parts = []
        for role in self.roles:
            key = f'roll_{role}'
            if key in data:
                window = data[key][widx]      # (window_bars, resolution, n_pitch)
                x_parts.append(window[-1])    # último compás
            else:
                x_parts.append(np.zeros((resolution, self.n_pitch), dtype=np.float32))

        x = torch.tensor(np.stack(x_parts, axis=0))   # (N_ROLES, res, n_pitch)
        return x


def _collate_fn(batch):
    import torch
    return torch.stack(batch)   # (B, N_ROLES, res, n_pitch)


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA CYCLEGAN SOBRE PIANO ROLLS
# ══════════════════════════════════════════════════════════════════════════════
#
#  Entrada/salida: (B, N_ROLES, resolution, n_pitch)  — piano roll multi-rol
#
#  Generadores:
#    G_A2B : convierte estilo A → estilo B
#    G_B2A : convierte estilo B → estilo A  (para cycle-consistency)
#
#  Discriminadores:
#    D_A   : distingue piano rolls reales del dominio A
#    D_B   : distingue piano rolls reales del dominio B
#
#  Arquitectura de los generadores: ResNet de 9 bloques adaptado a 2D.
#    Stem (conv) → Downsampling (x2) → 9× ResBlock → Upsampling (x2) → Head
#
#  Arquitectura del discriminador: PatchGAN (70×70) — clasifica parches
#    en lugar de la imagen completa, lo que estabiliza el entrenamiento.
#
#  Losses:
#    L_GAN   (adversarial LSGAN): MSE en lugar de BCE, más estable
#    L_cycle (consistencia):      |G_B2A(G_A2B(x)) - x|₁ × λ_cycle
#    L_ident (identidad):         |G_A2B(y) - y|₁ × λ_ident (y del dom. B)
# ══════════════════════════════════════════════════════════════════════════════

def _build_generator(n_roles, resolution, n_pitch, n_res_blocks=9, base_ch=64):
    """
    Generador ResNet para transformación de dominio sobre piano rolls.

    Entrada : (B, N_ROLES, resolution, n_pitch)
    Salida  : (B, N_ROLES, resolution, n_pitch)  — en el espacio del otro dominio
    """
    import torch
    import torch.nn as nn

    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, 3),
                nn.InstanceNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, 3),
                nn.InstanceNorm2d(ch),
            )

        def forward(self, x):
            return x + self.net(x)

    class _Generator(nn.Module):
        def __init__(self):
            super().__init__()
            # Stem
            self.stem = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(n_roles, base_ch, 7),
                nn.InstanceNorm2d(base_ch),
                nn.ReLU(inplace=True),
            )
            # Downsampling ×2
            self.down = nn.Sequential(
                nn.Conv2d(base_ch,     base_ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base_ch * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base_ch * 4),
                nn.ReLU(inplace=True),
            )
            # Bloques residuales
            self.res = nn.Sequential(*[_ResBlock(base_ch * 4) for _ in range(n_res_blocks)])
            # Upsampling ×2
            self.up = nn.Sequential(
                nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(base_ch * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_ch * 2, base_ch,     3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(base_ch),
                nn.ReLU(inplace=True),
            )
            # Head: salida en [0,1] (probabilidades de nota activa)
            self.head = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(base_ch, n_roles, 7),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.up(self.res(self.down(self.stem(x)))))

    return _Generator()


def _build_discriminator(n_roles, base_ch=64):
    """
    Discriminador PatchGAN de 3 capas (campo receptivo ~70×70).

    Entrada : (B, N_ROLES, resolution, n_pitch)
    Salida  : (B, 1, H', W')  — mapa de parches real/falso
    """
    import torch.nn as nn

    def _block(in_ch, out_ch, norm=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    return nn.Sequential(
        *_block(n_roles,      base_ch,     norm=False),
        *_block(base_ch,      base_ch * 2),
        *_block(base_ch * 2,  base_ch * 4),
        nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1),
        nn.InstanceNorm2d(base_ch * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER  (estabiliza el entrenamiento del discriminador)
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Almacena los últimos 50 ejemplos generados para entrenar el discriminador."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data     = []

    def push_and_pop(self, data):
        import torch, random
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    to_return.append(tmp)
                else:
                    to_return.append(element)
        return torch.cat(to_return, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR CYCLEGAN
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds):
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60:02d}s"
    if s >= 60:
        return f"{s//60}m {s%60:02d}s"
    return f"{s}s"


class CycleTrainer:
    CONFIG_NAME     = 'model_config.json'
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'

    def __init__(self, G_A2B, G_B2A, D_A, D_B,
                 opt_G, opt_D_A, opt_D_B,
                 model_dir, patience=40,
                 lambda_cycle=10.0, lambda_ident=0.5):
        self.G_A2B       = G_A2B
        self.G_B2A       = G_B2A
        self.D_A         = D_A
        self.D_B         = D_B
        self.opt_G       = opt_G
        self.opt_D_A     = opt_D_A
        self.opt_D_B     = opt_D_B
        self.model_dir   = model_dir
        self.patience    = patience
        self.λ_cycle     = lambda_cycle
        self.λ_ident     = lambda_ident

        self.history     = {'train_G': [], 'train_D': [], 'val_cycle': []}
        self.best_loss   = float('inf')
        self.no_improve  = 0
        self.start_epoch = 0

    def save_checkpoint(self, epoch, val_loss, is_best):
        import torch
        state = {
            'epoch':        epoch,
            'G_A2B':        self.G_A2B.state_dict(),
            'G_B2A':        self.G_B2A.state_dict(),
            'D_A':          self.D_A.state_dict(),
            'D_B':          self.D_B.state_dict(),
            'opt_G':        self.opt_G.state_dict(),
            'opt_D_A':      self.opt_D_A.state_dict(),
            'opt_D_B':      self.opt_D_B.state_dict(),
            'best_loss':    self.best_loss,
            'no_improve':   self.no_improve,
            'history':      self.history,
        }
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        with open(self.model_dir / self.HISTORY_NAME, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self):
        import torch
        path = self.model_dir / self.CHECKPOINT_NAME
        if not path.exists():
            print("[train] Entrenando desde cero.")
            return
        state = torch.load(path, map_location='cpu')
        self.G_A2B.load_state_dict(state['G_A2B'])
        self.G_B2A.load_state_dict(state['G_B2A'])
        self.D_A.load_state_dict(state['D_A'])
        self.D_B.load_state_dict(state['D_B'])
        self.opt_G.load_state_dict(state['opt_G'])
        self.opt_D_A.load_state_dict(state['opt_D_A'])
        self.opt_D_B.load_state_dict(state['opt_D_B'])
        self.best_loss   = state['best_loss']
        self.no_improve  = state['no_improve']
        self.history     = state['history']
        self.start_epoch = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_epoch}  "
              f"(mejor val={self.best_loss:.4f})")

    def _gan_loss(self, pred, target_is_real, device):
        """LSGAN loss: MSE(pred, 1) o MSE(pred, 0)."""
        import torch, torch.nn.functional as F
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)

    def _train_epoch(self, loader_A, loader_B, buf_A, buf_B, device):
        import torch, torch.nn.functional as F
        import torch.nn.utils as nn_utils

        self.G_A2B.train()
        self.G_B2A.train()
        self.D_A.train()
        self.D_B.train()

        sum_G = sum_D = n_batches = 0
        iter_B = iter(loader_B)

        for real_A in loader_A:
            # Tomar un batch de B (reanudar si B es más corto)
            try:
                real_B = next(iter_B)
            except StopIteration:
                iter_B = iter(loader_B)
                real_B = next(iter_B)

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ── Generadores ──────────────────────────────────────────────────
            self.opt_G.zero_grad()

            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)

            # Adversarial
            loss_G_A2B = self._gan_loss(self.D_B(fake_B), True, device)
            loss_G_B2A = self._gan_loss(self.D_A(fake_A), True, device)

            # Cycle consistency  |G_B2A(G_A2B(a)) - a|₁
            rec_A = self.G_B2A(fake_B)
            rec_B = self.G_A2B(fake_A)
            loss_cycle_A = F.l1_loss(rec_A, real_A)
            loss_cycle_B = F.l1_loss(rec_B, real_B)

            # Identity          |G_A2B(b) - b|₁
            idt_B = self.G_A2B(real_B)
            idt_A = self.G_B2A(real_A)
            loss_idt_A = F.l1_loss(idt_B, real_B)
            loss_idt_B = F.l1_loss(idt_A, real_A)

            loss_G = (loss_G_A2B + loss_G_B2A
                      + self.λ_cycle * (loss_cycle_A + loss_cycle_B)
                      + self.λ_ident * (loss_idt_A + loss_idt_B))
            loss_G.backward()
            nn_utils.clip_grad_norm_(
                list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()), 1.0)
            self.opt_G.step()

            # ── Discriminador A ───────────────────────────────────────────────
            self.opt_D_A.zero_grad()
            fake_A_buf = buf_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (self._gan_loss(self.D_A(real_A), True,  device) +
                               self._gan_loss(self.D_A(fake_A_buf), False, device))
            loss_D_A.backward()
            self.opt_D_A.step()

            # ── Discriminador B ───────────────────────────────────────────────
            self.opt_D_B.zero_grad()
            fake_B_buf = buf_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (self._gan_loss(self.D_B(real_B), True,  device) +
                               self._gan_loss(self.D_B(fake_B_buf), False, device))
            loss_D_B.backward()
            self.opt_D_B.step()

            sum_G   += loss_G.item()
            sum_D   += (loss_D_A + loss_D_B).item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return sum_G / n_batches, sum_D / n_batches

    def _val_cycle_loss(self, loader_A, loader_B, device):
        """Estima la pérdida de ciclo en validación (G_B2A(G_A2B(a)) ≈ a)."""
        import torch, torch.nn.functional as F
        self.G_A2B.eval()
        self.G_B2A.eval()
        total = n = 0
        with torch.no_grad():
            for real_A in loader_A:
                real_A = real_A.to(device)
                fake_B = self.G_A2B(real_A)
                rec_A  = self.G_B2A(fake_B)
                total += F.l1_loss(rec_A, real_A).item()
                n     += 1
                if n >= 20:   # bastan 20 batches para estimar
                    break
        return total / max(n, 1)

    def train(self, loader_A_tr, loader_B_tr,
              loader_A_val, loader_B_val, n_epochs):
        import torch, time

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _first = next(self.G_A2B.parameters(), None)
        if _first is not None and hasattr(_first, 'device'):
            device = str(_first.device)
        buf_A  = ReplayBuffer()
        buf_B  = ReplayBuffer()

        # Learning-rate decay lineal en la segunda mitad del entrenamiento
        def _lr_lambda(epoch):
            decay_start = n_epochs // 2
            if epoch < decay_start:
                return 1.0
            return 1.0 - (epoch - decay_start) / max(n_epochs - decay_start, 1)

        import torch.optim.lr_scheduler as sched
        sch_G   = sched.LambdaLR(self.opt_G,   _lr_lambda)
        sch_D_A = sched.LambdaLR(self.opt_D_A, _lr_lambda)
        sch_D_B = sched.LambdaLR(self.opt_D_B, _lr_lambda)

        if hasattr(self, '_resume') and self._resume:
            self.load_checkpoint()

        train_start = time.time()
        bar_w = 30

        print(f"\n{'─'*64}")
        print(f"  Iniciando entrenamiento CycleGAN: {n_epochs} épocas")
        print(f"  λ_cycle={self.λ_cycle}  λ_identity={self.λ_ident}")
        print(f"{'─'*64}\n")

        for epoch in range(self.start_epoch, n_epochs):
            t0 = time.time()

            # Ajustar LR para las épocas reanudadas
            for _ in range(self.start_epoch):
                sch_G.step()
                sch_D_A.step()
                sch_D_B.step()

            loss_G, loss_D = self._train_epoch(
                loader_A_tr, loader_B_tr, buf_A, buf_B, device)
            val_cycle = self._val_cycle_loss(loader_A_val, loader_B_val, device)

            sch_G.step()
            sch_D_A.step()
            sch_D_B.step()

            self.history['train_G'].append(round(loss_G, 5))
            self.history['train_D'].append(round(loss_D, 5))
            self.history['val_cycle'].append(round(val_cycle, 5))

            is_best = val_cycle < self.best_loss
            if is_best:
                self.best_loss = val_cycle
                self.no_improve = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, val_cycle, is_best)

            elapsed   = time.time() - t0
            total_el  = time.time() - train_start
            progress  = int((epoch + 1) / n_epochs * bar_w)
            bar       = '█' * progress + '░' * (bar_w - progress)
            best_str  = ' ◀ mejor' if is_best else ''
            stop_str  = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                         if self.no_improve > 0 else '')

            print(f"  Época {epoch+1:4d}/{n_epochs}  │{bar}│  "
                  f"G={loss_G:.4f}  D={loss_D:.4f}  val_cycle={val_cycle:.4f}"
                  f"  {_fmt_time(elapsed)}/ép{best_str}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        total_elapsed = time.time() - train_start
        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(total_elapsed)}.")
        print(f"  Mejor val_cycle : {self.best_loss:.4f}")
        print(f"  Modelos en      : {self.model_dir}")
        print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARAR DATOS  (idéntico a diffusion_composer, un corpus a la vez)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    (midi_path, output_dir, resolution, window_bars,
     active_roles, pitch_lo, pitch_hi) = args_tuple
    import numpy as np

    stem = midi_path.stem
    stats = {r: 0 for r in ROLES}
    stats.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})

    n_pitch = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES

    try:
        mid = _load_midi(str(midi_path))
    except Exception as e:
        return stem, f"ERROR al cargar: {e}", None, stats

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        stats['files_skipped'] = 1
        return stem, "sin notas — omitido", None, stats

    assigner  = RoleAssigner()
    converter = PianoRollConverter(resolution=resolution, window_bars=window_bars)

    role_assignment = assigner.assign(mid)
    if not role_assignment:
        stats['files_skipped'] = 1
        return stem, "sin asignación de roles — omitido", None, stats

    tpb_raw    = _ticks_per_bar(mid)
    all_ticks  = max((n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpb_raw) + 1)

    role_rolls  = {}
    roles_found = []
    for role, key in role_assignment.items():
        if role not in active_roles:
            continue
        notes = note_lists.get(key, [])
        if not notes:
            continue
        roll = converter.notes_to_roll(notes, tpb_raw, total_bars)
        if pitch_lo is not None:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        role_rolls[role] = roll
        roles_found.append(role)
        stats[role] = 1

    if not role_rolls:
        stats['files_skipped'] = 1
        return stem, "no se pudo construir ningún piano roll — omitido", None, stats

    role_windows = {}
    min_windows  = None
    for role, roll in role_rolls.items():
        windows = converter.roll_to_windows(roll)
        if windows.shape[0] == 0:
            continue
        role_windows[role] = windows
        min_windows = (windows.shape[0] if min_windows is None
                       else min(min_windows, windows.shape[0]))

    if min_windows is None or min_windows == 0:
        stats['files_skipped'] = 1
        return stem, f"demasiado corto ({total_bars} compases) — omitido", None, stats

    for role in role_windows:
        role_windows[role] = role_windows[role][:min_windows]

    save_dict = {}
    for role, windows in role_windows.items():
        save_dict[f'roll_{role}'] = windows

    meta = {
        'source': stem, 'resolution': resolution, 'window_bars': window_bars,
        'total_bars': total_bars, 'n_windows': min_windows,
        'roles': roles_found, 'tpb_raw': tpb_raw,
        'pitch_lo': pitch_lo if pitch_lo is not None else 0,
        'pitch_hi': pitch_hi if pitch_hi is not None else 127,
        'n_pitch':  n_pitch,
    }
    save_dict['meta_json'] = np.array([json.dumps(meta)])
    out_path = Path(output_dir) / f"{stem}.npz"
    np.savez_compressed(str(out_path), **save_dict)

    stats['files_ok']      = 1
    stats['total_windows'] = min_windows
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, "
            f"roles: {', '.join(roles_found)})",
            True, stats)


def cmd_prepare(args):
    from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor, as_completed

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]
    if disabled:
        print(f"[prepare] Roles deshabilitados : {', '.join(sorted(disabled))}")
        print(f"[prepare] Roles activos        : {', '.join(active_roles)}")

    pitch_n  = getattr(args, 'pitch_range', None)
    pr       = _pitch_range(pitch_n)
    pitch_lo = pr[0] if pr else None
    pitch_hi = pr[1] if pr else None
    if pr:
        print(f"[prepare] Rango de pitch       : {pitch_n} notas  "
              f"(MIDI {pitch_lo}–{pitch_hi})")

    midi_files = sorted(
        list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    print(f"[prepare] {len(midi_files)} archivos MIDI → {output_dir}\n")

    job_args = [
        (p, output_dir, args.resolution, args.window_bars,
         active_roles, pitch_lo, pitch_hi)
        for p in midi_files
    ]

    stats_total = {r: 0 for r in ROLES}
    stats_total.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})

    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_prepare_one_midi, a): a[0] for a in job_args}
        for fut in as_completed(futures):
            stem, msg, ok, partial = fut.result()
            print(f"  {'[OK]' if ok else '[--]'} {stem:40s}  {msg}")
            for k in stats_total:
                stats_total[k] += partial.get(k, 0)

    print(f"\n[prepare] Resumen")
    print(f"  Archivos OK      : {stats_total['files_ok']}")
    print(f"  Archivos omitidos: {stats_total['files_skipped']}")
    print(f"  Ventanas totales : {stats_total['total_windows']}")
    for r in ROLES:
        if stats_total[r]:
            print(f"    {r:20s}: {stats_total[r]} archivos")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader, random_split

    dir_A     = Path(args.data_dir_a)
    dir_B     = Path(args.data_dir_b)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]

    print("[train] Cargando datasets ...")
    ds_A = MidiRollDataset(str(dir_A), roles=active_roles)
    ds_B = MidiRollDataset(str(dir_B), roles=active_roles)

    if len(ds_A) == 0 or len(ds_B) == 0:
        print("[train] ERROR: alguno de los datasets está vacío. "
              "Ejecuta primero `prepare` en ambas carpetas.")
        sys.exit(1)

    # Inferir hiperparámetros del primer sample de A
    sample     = ds_A[0]
    n_roles    = sample.shape[0]
    resolution = sample.shape[1]
    n_pitch    = ds_A.n_pitch

    print(f"[train] Corpus A: {len(ds_A)} ventanas  |  "
          f"Corpus B: {len(ds_B)} ventanas")
    print(f"[train] n_roles={n_roles}  resolution={resolution}  n_pitch={n_pitch}")

    # Splits 90/10 para cada dominio
    def _split(ds):
        n_val   = max(1, int(len(ds) * 0.1))
        n_train = len(ds) - n_val
        return random_split(ds, [n_train, n_val])

    tr_A, val_A = _split(ds_A)
    tr_B, val_B = _split(ds_B)

    kw = dict(collate_fn=_collate_fn, num_workers=0, pin_memory=False)
    loader_A_tr  = DataLoader(tr_A,  batch_size=args.batch_size, shuffle=True,  **kw)
    loader_B_tr  = DataLoader(tr_B,  batch_size=args.batch_size, shuffle=True,  **kw)
    loader_A_val = DataLoader(val_A, batch_size=args.batch_size, shuffle=False, **kw)
    loader_B_val = DataLoader(val_B, batch_size=args.batch_size, shuffle=False, **kw)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Dispositivo: {device}\n")

    # Leer pitch_lo/hi desde metadatos
    import numpy as _np
    _npz = sorted(dir_A.glob('*.npz'))[0]
    _meta = json.loads(str(_np.load(str(_npz), allow_pickle=True)['meta_json'][0]))
    pitch_lo = _meta.get('pitch_lo', 0)
    pitch_hi = _meta.get('pitch_hi', 127)

    G_A2B = _build_generator(n_roles, resolution, n_pitch,
                              n_res_blocks=args.n_res_blocks,
                              base_ch=args.base_ch).to(device)
    G_B2A = _build_generator(n_roles, resolution, n_pitch,
                              n_res_blocks=args.n_res_blocks,
                              base_ch=args.base_ch).to(device)
    D_A   = _build_discriminator(n_roles, base_ch=args.base_ch).to(device)
    D_B   = _build_discriminator(n_roles, base_ch=args.base_ch).to(device)

    opt_G   = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    trainer = CycleTrainer(
        G_A2B, G_B2A, D_A, D_B,
        opt_G, opt_D_A, opt_D_B,
        model_dir,
        patience=args.patience,
        lambda_cycle=args.lambda_cycle,
        lambda_ident=args.lambda_identity,
    )
    trainer._resume = args.resume

    # Guardar configuración
    cfg = {
        'n_roles':     n_roles,
        'roles':       active_roles,
        'resolution':  resolution,
        'n_pitch':     n_pitch,
        'pitch_lo':    pitch_lo,
        'pitch_hi':    pitch_hi,
        'window_bars': _meta.get('window_bars', WINDOW_BARS_DEFAULT),
        'base_ch':     args.base_ch,
        'n_res_blocks': args.n_res_blocks,
        'lambda_cycle': args.lambda_cycle,
        'lambda_identity': args.lambda_identity,
        'model_version': 'cyclegan_v1',
    }
    with open(model_dir / CycleTrainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer.train(loader_A_tr, loader_B_tr,
                  loader_A_val, loader_B_val, args.epochs)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

def _load_models_and_config(model_dir: Path):
    import torch

    cfg_path   = model_dir / CycleTrainer.CONFIG_NAME
    model_path = model_dir / CycleTrainer.BEST_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró {cfg_path}. ¿Has ejecutado train?")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró {model_path}. ¿Has ejecutado train?")

    with open(cfg_path) as f:
        cfg = json.load(f)

    G_A2B = _build_generator(cfg['n_roles'], cfg['resolution'], cfg['n_pitch'],
                              n_res_blocks=cfg.get('n_res_blocks', 9),
                              base_ch=cfg.get('base_ch', 64))
    G_B2A = _build_generator(cfg['n_roles'], cfg['resolution'], cfg['n_pitch'],
                              n_res_blocks=cfg.get('n_res_blocks', 9),
                              base_ch=cfg.get('base_ch', 64))

    state = torch.load(str(model_path), map_location='cpu')
    G_A2B.load_state_dict(state['G_A2B'])
    G_B2A.load_state_dict(state['G_B2A'])
    G_A2B.eval()
    G_B2A.eval()
    return G_A2B, G_B2A, cfg


def _midi_to_rolls(midi_path, cfg):
    import mido, numpy as np

    mid         = mido.MidiFile(midi_path)
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise ValueError(f"No se encontraron notas en {midi_path}")

    tpb       = mid.ticks_per_beat
    tpbar     = tpb * 4
    max_tick  = max((e for nl in note_lists.values() for e in [n[1] for n in nl]), default=0)
    total_bars = max(1, int(max_tick / tpbar) + 1)

    active_roles = cfg.get('roles', ROLES)
    pitch_lo     = cfg.get('pitch_lo', 0)
    pitch_hi     = cfg.get('pitch_hi', 127)
    do_crop      = (pitch_lo, pitch_hi) != (0, 127)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    rolls    = {}
    for role, stream_key in role_map.items():
        if role not in active_roles:
            continue
        notes = note_lists[stream_key]
        roll  = conv.notes_to_roll(notes, tpb * 4, total_bars)
        if do_crop:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        rolls[role] = roll

    return rolls


def _load_palette(palette_path, cfg):
    DEFAULT_PALETTE = {
        'melody':        {'program': 0,  'channel': 0, 'velocity': 80},
        'counterpoint':  {'program': 40, 'channel': 1, 'velocity': 70},
        'accompaniment': {'program': 48, 'channel': 2, 'velocity': 65},
        'bass':          {'program': 43, 'channel': 3, 'velocity': 75},
        'percussion':    {'program': 0,  'channel': 9, 'velocity': 90},
    }
    if palette_path is None:
        return DEFAULT_PALETTE
    try:
        with open(palette_path) as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PALETTE


def _adaptive_threshold(roll, percentile=99.0):
    import numpy as np
    flat = roll.flatten()
    frac_near_zero = float((flat < 0.01).mean())
    if frac_near_zero > 0.90:
        thr = float(np.percentile(flat, percentile))
        return max(thr, 1e-4)
    else:
        nonzero = flat[flat > 0.001]
        if len(nonzero) == 0:
            return 0.5
        return float(np.percentile(nonzero, percentile))


def _rolls_to_midi(bars_per_role, cfg, palette, output_path, bpm=120.0,
                   threshold=None, threshold_pct=99.0):
    import mido, numpy as np

    resolution  = cfg['resolution']
    tpb         = 480
    ticks_bar   = tpb * 4
    ticks_tick  = ticks_bar / resolution
    pitch_lo    = cfg.get('pitch_lo', 0)
    pitch_hi    = cfg.get('pitch_hi', 127)
    do_expand   = (pitch_lo, pitch_hi) != (0, 127)

    mid = mido.MidiFile(ticks_per_beat=tpb)
    t0  = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / bpm), time=0))
    mid.tracks.append(t0)

    n_notes_total = 0

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]   # (n_bars, res, n_pitch)
        if do_expand:
            roll = _pad_pitch(roll, pitch_lo, n_full=128)

        thr = threshold if threshold is not None else _adaptive_threshold(roll, threshold_pct)

        pal  = palette.get(role, {})
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        binary = (roll > thr).astype(np.float32)

        # Eliminar notas aisladas de 1 tick
        for b in range(binary.shape[0]):
            for p in range(128):
                col = binary[b, :, p]
                for t in range(1, len(col) - 1):
                    if col[t] == 1 and col[t-1] == 0 and col[t+1] == 0:
                        binary[b, t, p] = 0

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        events = []
        n_bars_r, res_r, _ = binary.shape
        for bar in range(n_bars_r):
            for tick in range(res_r):
                abs_t = int((bar * res_r + tick) * ticks_tick)
                for pitch in range(128):
                    cur  = binary[bar, tick, pitch] > 0
                    prev = binary[bar, tick - 1, pitch] > 0 if tick > 0 \
                           else (binary[bar - 1, -1, pitch] > 0 if bar > 0 else False)
                    if cur and not prev:
                        events.append((abs_t, 'on',  pitch))
                        n_notes_total += 1
                    elif not cur and prev:
                        events.append((abs_t, 'off', pitch))

        events.sort(key=lambda e: e[0])
        prev_t = 0
        for abs_t, etype, pitch in events:
            delta = abs_t - prev_t
            if etype == 'on':
                track.append(mido.Message('note_on',  note=pitch, velocity=vel,
                                          channel=ch, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0,
                                          channel=ch, time=delta))
            prev_t = abs_t

    mid.save(output_path)
    return n_notes_total


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: transfer  (A→B)  y  transfer-inv  (B→A)
# ══════════════════════════════════════════════════════════════════════════════

def _run_transfer(args, direction='a2b'):
    """
    Aplica el generador G_A2B (o G_B2A) compás a compás sobre el MIDI de entrada.

    A diferencia de la difusión, la transferencia CycleGAN es un único paso
    forward determinista: no hay proceso de denoising, no hay parámetros de
    'strength' o 'temperatura'.  El resultado es siempre el mismo para la
    misma entrada.
    """
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[transfer] Cargando modelos desde {model_dir} ...")
    G_A2B, G_B2A, cfg = _load_models_and_config(model_dir)

    generator = G_A2B if direction == 'a2b' else G_B2A
    dir_label  = 'A→B' if direction == 'a2b' else 'B→A'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)

    palette = _load_palette(getattr(args, 'palette', None), cfg)

    # Parsear el MIDI de entrada
    print(f"[transfer] Procesando {args.input} ({dir_label}) ...")
    rolls = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[transfer] ERROR: no se encontraron notas.")
        sys.exit(1)

    n_roles    = cfg['n_roles']
    role_list  = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg.get('n_pitch', 128)
    n_bars     = min(r.shape[0] for r in rolls.values())

    print(f"[transfer] {n_bars} compases  ·  {n_roles} roles  ·  dirección {dir_label}")

    # Construir tensor de entrada compás a compás
    bars_per_role = {role: [] for role in role_list}

    for bar_idx in range(n_bars):
        x_np = np.zeros((n_roles, resolution, n_pitch), dtype=np.float32)
        for ridx, role in enumerate(role_list):
            if role in rolls:
                x_np[ridx] = rolls[role][bar_idx]

        x_t = torch.tensor(x_np).unsqueeze(0).to(device)   # (1, N_ROLES, res, n_pitch)

        with torch.no_grad():
            y_t = generator(x_t)   # (1, N_ROLES, res, n_pitch)

        y_np = y_t[0].cpu().numpy()   # (N_ROLES, res, n_pitch)

        for ridx, role in enumerate(role_list):
            bars_per_role[role].append(y_np[ridx])

        if bar_idx == 0:
            vmax  = float(y_np.max())
            vmean = float(y_np.mean())
            thr   = _adaptive_threshold(y_np, getattr(args, 'threshold_pct', 99.0))
            n_act = int((y_np > thr).sum())
            print(f"\n  [diag] Compás 0 — salida del generador:")
            print(f"         mean={vmean:.4f}  max={vmax:.4f}  "
                  f"umbral={thr:.4f}  notas_activas={n_act}")
            if n_act == 0:
                print("         ⚠  sin notas activas — prueba a bajar --threshold-pct")
            print()

    # Apilar compases
    import numpy as np
    for role in role_list:
        if bars_per_role[role]:
            bars_per_role[role] = np.stack(bars_per_role[role], axis=0)
        else:
            del bars_per_role[role]

    # Renderizar a MIDI
    thr_arg = getattr(args, 'threshold', None)
    thr_pct = getattr(args, 'threshold_pct', 99.0)
    bpm     = getattr(args, 'bpm', 120.0)

    n_notes = _rolls_to_midi(bars_per_role, cfg, palette,
                              args.output, bpm=bpm,
                              threshold=thr_arg, threshold_pct=thr_pct)

    print(f"[transfer] Guardado: {args.output}  ({n_notes} notas, {n_bars} compases)")
    if n_notes == 0:
        print("[transfer] ⚠  MIDI vacío — ajusta --threshold-pct "
              "(prueba con 97.0 ó 95.0)")


def cmd_transfer(args):
    _run_transfer(args, direction='a2b')


def cmd_transfer_inv(args):
    _run_transfer(args, direction='b2a')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: style-corpus  (diagnóstico)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_style_corpus(args):
    """
    Pasa todos los MIDIs de una carpeta por el generador y calcula
    estadísticas de la salida: densidad media, varianza, etc.

    No extrae un vector de estilo como en el modelo de difusión
    (el estilo CycleGAN está implícito en los pesos del generador),
    pero sirve para diagnosticar si el generador converge bien.
    """
    import torch, numpy as np

    model_dir  = Path(args.model_dir)
    input_dir  = Path(args.input_dir)
    direction  = getattr(args, 'direction', 'a2b')

    G_A2B, G_B2A, cfg = _load_models_and_config(model_dir)
    generator = G_A2B if direction == 'a2b' else G_B2A

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)

    midi_files = sorted(
        list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[style-corpus] No se encontraron MIDIs en {input_dir}")
        sys.exit(1)

    n_roles    = cfg['n_roles']
    role_list  = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg.get('n_pitch', 128)
    dir_label  = 'A→B' if direction == 'a2b' else 'B→A'

    print(f"[style-corpus] {len(midi_files)} archivos  ·  dirección {dir_label}\n")

    densities = []
    skipped   = 0

    for midi_path in midi_files:
        try:
            rolls = _midi_to_rolls(str(midi_path), cfg)
            n_bars = min(r.shape[0] for r in rolls.values())
            x_np = np.zeros((n_bars, n_roles, resolution, n_pitch), dtype=np.float32)
            for ridx, role in enumerate(role_list):
                if role in rolls:
                    bars = rolls[role][:n_bars]
                    x_np[:, ridx] = bars

            x_t = torch.tensor(x_np).to(device)   # (n_bars, N_ROLES, res, n_pitch)
            with torch.no_grad():
                y_t = generator(x_t)
            y_np = y_t.cpu().numpy()

            # Densidad: fracción de celdas activas tras umbral
            thr  = _adaptive_threshold(y_np)
            dens = float((y_np > thr).mean())
            densities.append(dens)
            print(f"  [OK] {midi_path.stem:40s}  densidad_out={dens:.4f}")
        except Exception as e:
            print(f"  [SKIP] {midi_path.stem} — {e}")
            skipped += 1

    if not densities:
        print("[style-corpus] No se pudo procesar ningún MIDI.")
        sys.exit(1)

    mean_dens = float(np.mean(densities))
    std_dens  = float(np.std(densities))

    out_path = args.output or f"corpus_stats_{input_dir.stem}_{direction}.json"
    payload = {
        'source':       str(input_dir),
        'model_dir':    str(model_dir),
        'direction':    direction,
        'n_files':      len(densities),
        'n_skipped':    skipped,
        'density_mean': mean_dens,
        'density_std':  std_dens,
        'densities':    densities,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\n[style-corpus] Archivos procesados : {len(densities)}")
    print(f"[style-corpus] Archivos omitidos   : {skipped}")
    print(f"[style-corpus] Densidad media      : {mean_dens:.4f} ± {std_dens:.4f}")
    if mean_dens < 0.005:
        print("[style-corpus] ⚠  densidad muy baja — el generador puede no haber convergido")
    elif mean_dens > 0.3:
        print("[style-corpus] ⚠  densidad muy alta — posible colapso de modo")
    print(f"[style-corpus] Estadísticas en     : {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: round-trip  (diagnóstico)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    """MIDI → piano roll → MIDI sin modelo (diagnóstico del parser)."""
    import numpy as np

    cfg_rt = {
        'resolution':  args.resolution,
        'window_bars': WINDOW_BARS_DEFAULT,
        'roles':       [r for r in ROLES if r not in (args.disable_roles or [])],
        'n_pitch':     PITCH_CLASSES,
        'pitch_lo':    0,
        'pitch_hi':    127,
    }

    if args.model_dir:
        try:
            with open(Path(args.model_dir) / CycleTrainer.CONFIG_NAME) as f:
                cfg_rt = json.load(f)
            print(f"[round-trip] Usando config de {args.model_dir}")
        except Exception:
            pass

    rolls = _midi_to_rolls(args.input, cfg_rt)
    n_bars = min(r.shape[0] for r in rolls.values())
    print(f"[round-trip] {n_bars} compases  ·  roles: {list(rolls.keys())}")

    bars_per_role = {role: roll for role, roll in rolls.items()}
    palette = _load_palette(None, cfg_rt)

    n_notes = _rolls_to_midi(bars_per_role, cfg_rt, palette, args.output,
                              bpm=args.bpm)
    print(f"[round-trip] Guardado: {args.output}  ({n_notes} notas)")
    if n_notes == 0:
        print("[round-trip] ⚠  MIDI vacío — posible problema en el parser")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    import numpy as np

    if 'npz' in args.what and args.data_dir:
        data_dir = Path(args.data_dir)
        npz_files = sorted(data_dir.glob('*.npz'))
        if not npz_files:
            print(f"[inspect] No se encontraron .npz en {data_dir}")
        else:
            target = args.npz_file or npz_files[0].name
            path   = data_dir / target
            if not path.exists():
                print(f"[inspect] No se encontró {path}")
            else:
                data = np.load(str(path), allow_pickle=True)
                meta = json.loads(str(data['meta_json'][0]))
                print(f"[inspect] {path.name}")
                print(f"  resolution   : {meta['resolution']}")
                print(f"  window_bars  : {meta['window_bars']}")
                print(f"  total_bars   : {meta['total_bars']}")
                print(f"  n_windows    : {meta['n_windows']}")
                print(f"  roles        : {meta['roles']}")
                print(f"  n_pitch      : {meta.get('n_pitch', 128)}")
                for role in meta['roles']:
                    key = f'roll_{role}'
                    if key in data:
                        arr = data[key]
                        dens = float(arr.mean())
                        print(f"    {role:20s}: shape={arr.shape}  densidad={dens:.5f}")

    if 'loss_curve' in args.what and args.model_dir:
        hist_path = Path(args.model_dir) / CycleTrainer.HISTORY_NAME
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            print(f"\n[inspect] Historial de entrenamiento ({len(hist.get('train_G',[]))} épocas)")
            for key, vals in hist.items():
                if vals:
                    print(f"  {key:15s}: último={vals[-1]:.4f}  min={min(vals):.4f}")
        else:
            print(f"[inspect] No se encontró historial en {args.model_dir}")

    if args.model_dir:
        cfg_path = Path(args.model_dir) / CycleTrainer.CONFIG_NAME
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"\n[inspect] Configuración del modelo ({cfg_path})")
            for k, v in cfg.items():
                print(f"  {k:20s}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  (espejo de diffusion_composer)
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='cyclegan_composer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ╔══════════════════════════════════════════════════════╗
            ║           CYCLEGAN COMPOSER  v1                      ║
            ║   Style-transfer MIDI mediante CycleGAN              ║
            ╚══════════════════════════════════════════════════════╝

            Flujo de trabajo típico:
              1. prepare  --input-dir midis_A/ --output-dir data_A/
              2. prepare  --input-dir midis_B/ --output-dir data_B/
              3. train    --data-dir-a data_A/ --data-dir-b data_B/ --model-dir model/
              4. transfer --input cancion.mid  --model-dir model/ --output resultado.mid
        """),
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # ── prepare ───────────────────────────────────────────────────────────────
    p_pr = sub.add_parser('prepare',
        help='MIDI corpus → piano rolls segmentados (.npz)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convierte una carpeta de MIDIs en ventanas de piano roll (.npz).
            Ejecutar por separado para el corpus A y el corpus B.

            Ejemplo:
              prepare --input-dir midis_clasico/ --output-dir data_clasico/
              prepare --input-dir midis_jazz/    --output-dir data_jazz/
        """))
    p_pr.add_argument('--input-dir',   required=True, metavar='DIR')
    p_pr.add_argument('--output-dir',  required=True, metavar='DIR')
    p_pr.add_argument('--resolution',  type=int, default=TICKS_PER_BAR_DEFAULT,
        metavar='INT',
        help=f'Ticks por compás (default: {TICKS_PER_BAR_DEFAULT})')
    p_pr.add_argument('--window-bars', type=int, default=WINDOW_BARS_DEFAULT,
        metavar='INT', dest='window_bars',
        help=f'Compases por ventana (default: {WINDOW_BARS_DEFAULT})')
    p_pr.add_argument('--disable-roles', nargs='+', metavar='ROL',
        choices=ROLES, default=[], dest='disable_roles',
        help=f'Roles a excluir. Posibles: {", ".join(ROLES)}')
    p_pr.add_argument('--pitch-range', type=int, default=None, metavar='N',
        dest='pitch_range',
        help='Limitar a N valores MIDI centrados en Do central.')
    p_pr.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p_tr = sub.add_parser('train',
        help='Entrena el par de generadores A↔B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Entrena los generadores G_A2B y G_B2A y sus discriminadores.
            Los corpus A y B no necesitan correspondencia canción-a-canción.

            Pérdidas:
              L_total = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity

            Ejemplo:
              train --data-dir-a data_clasico/ --data-dir-b data_jazz/ \\
                    --model-dir model_clasico_jazz/ \\
                    --epochs 200 --batch-size 4
        """))
    p_tr.add_argument('--data-dir-a',  required=True, metavar='DIR', dest='data_dir_a')
    p_tr.add_argument('--data-dir-b',  required=True, metavar='DIR', dest='data_dir_b')
    p_tr.add_argument('--model-dir',   required=True, metavar='DIR')
    p_tr.add_argument('--epochs',      type=int,   default=200)
    p_tr.add_argument('--batch-size',  type=int,   default=4,    dest='batch_size')
    p_tr.add_argument('--lr',          type=float, default=2e-4)
    p_tr.add_argument('--lambda-cycle', type=float, default=10.0, dest='lambda_cycle',
        help='Peso de la pérdida de ciclo (default: 10.0)')
    p_tr.add_argument('--lambda-identity', type=float, default=0.5, dest='lambda_identity',
        help='Peso de la pérdida de identidad (default: 0.5)')
    p_tr.add_argument('--n-res-blocks', type=int, default=9, dest='n_res_blocks',
        help='Número de bloques residuales del generador (default: 9; '
             'reducir a 6 para entrenar más rápido)')
    p_tr.add_argument('--base-ch',     type=int, default=64, dest='base_ch',
        help='Canales base del generador/discriminador (default: 64; '
             'reducir a 32 para modelos más pequeños)')
    p_tr.add_argument('--patience',    type=int, default=40,
        help='Early stopping: épocas sin mejora (default: 40)')
    p_tr.add_argument('--resume',      action='store_true',
        help='Reanudar desde el último checkpoint')
    p_tr.add_argument('--disable-roles', nargs='+', metavar='ROL',
        choices=ROLES, default=[], dest='disable_roles')
    p_tr.set_defaults(func=cmd_train)

    # ── transfer (A→B) ────────────────────────────────────────────────────────
    p_tf = sub.add_parser('transfer',
        help='Transfiere el estilo B a un MIDI del estilo A',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Aplica el generador G_A2B compás a compás sobre el MIDI de entrada.

            La transferencia es determinista y en un solo paso forward:
            no hay parámetros de intensidad ni temperatura.

            Ejemplo:
              transfer --input cancion_clasica.mid \\
                       --model-dir model_clasico_jazz/ \\
                       --output cancion_jazz.mid
        """))
    p_tf.add_argument('--input',        required=True,  metavar='FILE',
        help='MIDI de entrada (estilo A)')
    p_tf.add_argument('--model-dir',    required=True,  metavar='DIR')
    p_tf.add_argument('--output',       default='transfer_A2B.mid', metavar='FILE')
    p_tf.add_argument('--palette',      default=None,   metavar='FILE',
        help='Paleta de instrumentos JSON (opcional)')
    p_tf.add_argument('--bpm',          type=float, default=120.0)
    p_tf.add_argument('--threshold',    type=float, default=None, metavar='FLOAT',
        help='Umbral fijo de binarización (default: automático)')
    p_tf.add_argument('--threshold-pct', type=float, default=99.0, metavar='FLOAT',
        dest='threshold_pct',
        help='Percentil para umbral adaptativo (default: 99.0; '
             'bajar a 95–97 si el MIDI sale vacío)')
    p_tf.set_defaults(func=cmd_transfer)

    # ── transfer-inv (B→A) ────────────────────────────────────────────────────
    p_ti = sub.add_parser('transfer-inv',
        help='Transfiere el estilo A a un MIDI del estilo B (generador inverso)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Aplica el generador G_B2A compás a compás sobre el MIDI de entrada.

            Ejemplo:
              transfer-inv --input cancion_jazz.mid \\
                           --model-dir model_clasico_jazz/ \\
                           --output cancion_clasica.mid
        """))
    p_ti.add_argument('--input',        required=True,  metavar='FILE')
    p_ti.add_argument('--model-dir',    required=True,  metavar='DIR')
    p_ti.add_argument('--output',       default='transfer_B2A.mid', metavar='FILE')
    p_ti.add_argument('--palette',      default=None,   metavar='FILE')
    p_ti.add_argument('--bpm',          type=float, default=120.0)
    p_ti.add_argument('--threshold',    type=float, default=None, metavar='FLOAT')
    p_ti.add_argument('--threshold-pct', type=float, default=99.0, metavar='FLOAT',
        dest='threshold_pct')
    p_ti.set_defaults(func=cmd_transfer_inv)

    # ── style-corpus ──────────────────────────────────────────────────────────
    p_sc = sub.add_parser('style-corpus',
        help='Estadísticas de salida del generador sobre un corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Pasa todos los MIDIs de una carpeta por el generador y calcula
            métricas de densidad (diagnóstico de convergencia).

            A diferencia del modelo de difusión, no existe un vector de estilo
            explícito: el estilo CycleGAN está codificado en los pesos del generador.

            Ejemplo:
              style-corpus --input-dir midis_A/ --model-dir model/ --direction a2b
        """))
    p_sc.add_argument('--input-dir',  required=True, metavar='DIR')
    p_sc.add_argument('--model-dir',  required=True, metavar='DIR')
    p_sc.add_argument('--direction',  default='a2b', choices=['a2b', 'b2a'],
        help='Dirección del generador a aplicar (default: a2b)')
    p_sc.add_argument('--output',     default=None, metavar='FILE',
        help='Ruta del JSON de salida (default: corpus_stats_<carpeta>_<dir>.json)')
    p_sc.set_defaults(func=cmd_style_corpus)

    # ── round-trip ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser('round-trip',
        help='MIDI → piano roll → MIDI sin modelo (diagnóstico del parser)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convierte un MIDI a piano roll y de vuelta sin usar el modelo.
            Útil para aislar problemas en el pipeline de parseo.

            Ejemplo:
              round-trip --input cancion.mid
              round-trip --input cancion.mid --model-dir model/
        """))
    p_rt.add_argument('--input',      required=True, metavar='FILE')
    p_rt.add_argument('--model-dir',  default=None,  metavar='DIR',
        help='Si se indica, lee la configuración del modelo.')
    p_rt.add_argument('--resolution', type=int, default=TICKS_PER_BAR_DEFAULT,
        metavar='INT')
    p_rt.add_argument('--disable-roles', nargs='+', metavar='ROL',
        choices=ROLES, default=[], dest='disable_roles')
    p_rt.add_argument('--output',     default='output_roundtrip.mid', metavar='FILE')
    p_rt.add_argument('--bpm',        type=float, default=120.0)
    p_rt.set_defaults(func=cmd_round_trip)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect',
        help='Diagnóstico del modelo y los datos')
    p_ins.add_argument('--what', nargs='+',
        choices=['npz', 'loss_curve'], default=['npz'])
    p_ins.add_argument('--data-dir',  metavar='DIR', default=None, dest='data_dir')
    p_ins.add_argument('--model-dir', metavar='DIR', default=None)
    p_ins.add_argument('--file',      dest='npz_file', metavar='NAME', default=None)
    p_ins.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
