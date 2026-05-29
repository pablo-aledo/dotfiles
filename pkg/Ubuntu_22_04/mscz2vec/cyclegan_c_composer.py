#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CYCLE-GAN COMPOSER  v1                                   ║
║       Transferencia de estilo musical mediante CycleGAN sobre piano rolls    ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Dos generadores ResNet + dos discriminadores PatchGAN                     ║
║    Pérdidas: adversarial (LSGAN) + cycle-consistency + identity              ║
║    Representación: piano roll multi-rol (melody/counterpoint/bass/etc.)      ║
║    Condicionamiento: tensión armónica (8D) inyectada vía AdaIN               ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare      — MIDI corpus → piano rolls segmentados por rol (.npz)       ║
║    train        — Entrena el modelo CycleGAN (dominio A ↔ dominio B)         ║
║    transfer     — Transfiere el estilo de B a un MIDI concreto del dominio A ║
║    encode       — MIDI → vector de estilo latente (.json)                    ║
║    style-corpus — Centroide de estilo de una carpeta de MIDIs                ║
║    inspect      — Diagnóstico del modelo y los datos                         ║
║    round-trip   — MIDI → piano roll → MIDI (sin modelo, diagnóstico)         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣

# ── Preparar dos corpus ───────────────────────────────────────────────────────
python cycle_gan_composer.py prepare \
    --input-dir midis_tango/  --output-dir data_A/ --report

python cycle_gan_composer.py prepare \
    --input-dir midis_jazz/   --output-dir data_B/ --report

# Con rango de pitch reducido (CPU / GPU modesta):
python cycle_gan_composer.py prepare \
    --input-dir midis_tango/  --output-dir data_A_small/ \
    --pitch-range 48 --disable-roles percussion counterpoint

# ── Entrenar ──────────────────────────────────────────────────────────────────
python cycle_gan_composer.py train \
    --data-dir-a data_A/ --data-dir-b data_B/ \
    --model-dir model_cyclegan/ \
    --epochs 200 --batch-size 8 --lr 2e-4 \
    --lambda-cycle 10.0 --lambda-identity 5.0 \
    --patience 40

# GPU modesta (menos canales, menos bloques):
python cycle_gan_composer.py train \
    --data-dir-a data_A_small/ --data-dir-b data_B_small/ \
    --model-dir model_small/ \
    --base-ch 32 --n-res-blocks 4 \
    --epochs 200 --batch-size 16

# ── Transferir estilo B a un MIDI del dominio A ───────────────────────────────
python cycle_gan_composer.py transfer \
    --model-dir model_cyclegan/ --palette palette.json \
    --input midis_tango/cancion.mid \
    --direction A2B \
    --output resultado_jazz.mid \
    --threshold-pct 99.0

# Dirección inversa (B → A):
python cycle_gan_composer.py transfer \
    --model-dir model_cyclegan/ \
    --input midis_jazz/tema.mid \
    --direction B2A \
    --output resultado_tango.mid

# ── Encode / style-corpus ─────────────────────────────────────────────────────
python cycle_gan_composer.py encode \
    --input midis_tango/cancion.mid --model-dir model_cyclegan/ \
    --output z_tango.json

python cycle_gan_composer.py style-corpus \
    --input-dir midis_jazz/ --model-dir model_cyclegan/ \
    --output z_jazz_centroide.json

# ── Diagnóstico ───────────────────────────────────────────────────────────────
python cycle_gan_composer.py round-trip --input midis_tango/cancion.mid
python cycle_gan_composer.py inspect --model-dir model_cyclegan/
python cycle_gan_composer.py inspect --what npz --data-dir data_A/

"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

ROLES = ['melody', 'counterpoint', 'accompaniment', 'bass', 'percussion']

GM_ROLE_HINTS = {
    43: 'bass', 42: 'bass', 58: 'bass', 70: 'bass',
    73: 'melody', 72: 'melody', 56: 'melody', 40: 'melody',
    68: 'counterpoint', 71: 'counterpoint', 41: 'counterpoint',
    48: 'accompaniment', 49: 'accompaniment',
    19: 'accompaniment', 52: 'accompaniment',
}

TICKS_PER_BAR_DEFAULT = 48
WINDOW_BARS_DEFAULT   = 4
PITCH_CLASSES         = 128
MIDI_CENTER           = 60

DEFAULT_PALETTE = {
    'melody':        {'program': 73, 'channel': 0, 'velocity': 90},
    'counterpoint':  {'program': 68, 'channel': 1, 'velocity': 80},
    'accompaniment': {'program': 48, 'channel': 2, 'velocity': 70},
    'bass':          {'program': 43, 'channel': 3, 'velocity': 85},
    'percussion':    {'program':  0, 'channel': 9, 'velocity': 90},
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE PITCH
# ══════════════════════════════════════════════════════════════════════════════

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
#  UTILIDADES MIDI
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


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES
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
            pitch_range_v = max(pitches) - min(pitches)
            density     = len(notes) / max(total_dur / tpb_raw, 1)
            polyphony   = self._mean_polyphony(notes)
            profiles.append({
                'key': (ti, ch), 'channel': ch, 'program': program,
                'pitch_mean': pitch_mean, 'pitch_range': pitch_range_v,
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

        def norm(lst, key):
            vals = [p[key] for p in lst]
            lo, hi = min(vals), max(vals)
            span = hi - lo or 1
            return {p['key']: (p[key] - lo) / span for p in lst}

        n_pm   = norm(unassigned, 'pitch_mean')
        n_pr   = norm(unassigned, 'pitch_range')
        n_poly = norm(unassigned, 'polyphony')
        n_dens = norm(unassigned, 'density')

        def score(p, role):
            k = p['key']
            hint = 0.25 if GM_ROLE_HINTS.get(p['program']) == role else 0.0
            if role == 'melody':
                return 0.40 * n_pm[k] + 0.35 * n_pr[k] + 0.15 * (1 - n_poly[k]) + hint
            elif role == 'counterpoint':
                mid_pm = abs(n_pm[k] - 0.65)
                return 0.30 * (1 - mid_pm) + 0.25 * n_pr[k] + 0.20 * (1 - n_poly[k]) + hint
            elif role == 'accompaniment':
                mid_pm = abs(n_pm[k] - 0.50)
                return 0.40 * n_poly[k] + 0.25 * (1 - mid_pm) + 0.15 * n_dens[k] + hint
            elif role == 'bass':
                return 0.50 * (1 - n_pm[k]) + 0.25 * (1 - n_pr[k]) + hint
            return 0.0

        score_matrix = {p['key']: {r: score(p, r) for r in remaining_roles}
                        for p in unassigned}
        taken_keys  = set()
        taken_roles = set()
        pairs = [(score_matrix[p['key']][r], r, p['key'])
                 for p in unassigned for r in remaining_roles]
        pairs.sort(key=lambda x: -x[0])
        for sc, role, key in pairs:
            if role not in taken_roles and key not in taken_keys:
                assigned[role] = key
                taken_roles.add(role)
                taken_keys.add(key)
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  PIANO ROLL CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    def __init__(self, resolution=TICKS_PER_BAR_DEFAULT,
                 window_bars=WINDOW_BARS_DEFAULT):
        self.resolution  = resolution
        self.window_bars = window_bars

    def notes_to_roll(self, notes, tpb_raw, n_bars):
        import numpy as np
        roll = np.zeros((n_bars, self.resolution, PITCH_CLASSES), dtype=np.float32)
        ticks_per_internal = tpb_raw / self.resolution
        for (start, end, pitch, vel, _) in notes:
            bar_s  = int(start / tpb_raw)
            tick_s = int((start % tpb_raw) / ticks_per_internal)
            bar_e  = int(end   / tpb_raw)
            tick_e = int((end   % tpb_raw) / ticks_per_internal)
            if bar_s >= n_bars:
                continue
            if bar_s == bar_e:
                roll[bar_s, tick_s:min(tick_e, self.resolution), pitch] = 1.0
            else:
                roll[bar_s, tick_s:, pitch] = 1.0
                for b in range(bar_s + 1, min(bar_e, n_bars)):
                    roll[b, :, pitch] = 1.0
                if bar_e < n_bars:
                    roll[bar_e, :tick_e, pitch] = 1.0
        return roll

    def roll_to_windows(self, roll):
        import numpy as np
        n_bars  = roll.shape[0]
        n_pitch = roll.shape[2]
        if n_bars < self.window_bars:
            return np.zeros((0, self.window_bars, self.resolution, n_pitch),
                            dtype=np.float32)
        n_windows = n_bars - self.window_bars + 1
        return np.stack([roll[i:i + self.window_bars] for i in range(n_windows)])


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTOR DE TENSIÓN  (8D por compás)
# ══════════════════════════════════════════════════════════════════════════════

class TensionExtractor:
    TENSION_DIM = 8

    def extract_bar_vectors(self, role_rolls, bars):
        import numpy as np
        n_pitch  = next(iter(role_rolls.values())).shape[-1] if role_rolls else PITCH_CLASSES
        vectors  = np.zeros((bars, self.TENSION_DIM), dtype=np.float32)
        for bar in range(bars):
            combined     = np.zeros((n_pitch,), dtype=np.float32)
            total_events = 0
            resolution   = None
            for role, roll in role_rolls.items():
                if bar >= roll.shape[0]:
                    continue
                bar_roll   = roll[bar]
                resolution = bar_roll.shape[0]
                active     = bar_roll.max(axis=0)
                combined   = np.maximum(combined, active)
                total_events += bar_roll.sum()
            if resolution is None or resolution == 0:
                continue
            pitches_active = np.where(combined > 0)[0]
            n_active       = len(pitches_active)
            capacity       = resolution * n_pitch
            tension        = self._lerdahl_proxy(pitches_active)
            density        = min(float(total_events) / max(capacity * len(role_rolls), 1) * 20, 1.0)
            poly           = min(n_active / 12.0, 1.0)
            reg_mean       = float(np.mean(pitches_active)) / max(n_pitch - 1, 1) if n_active > 0 else 0.5
            reg_spread     = float(np.ptp(pitches_active)) / max(n_pitch - 1, 1) if n_active > 1 else 0.0
            rhythm_density = 0.0
            if 'melody' in role_rolls and bar < role_rolls['melody'].shape[0]:
                mel = role_rolls['melody'][bar]
                act = mel.sum(axis=1)
                rhythm_density = float((abs(act[1:] - act[:-1]) > 0).sum()) / max(resolution - 1, 1)
            arousal = 0.5 * min(density * 2, 1.0) + 0.5 * rhythm_density
            vectors[bar] = [tension, density, poly, reg_mean,
                            reg_spread, 0.5, rhythm_density, arousal]
        return vectors

    @staticmethod
    def _lerdahl_proxy(pitches_active):
        if len(pitches_active) < 2:
            return 0.0
        DISSONANT = {1, 2, 6, 10, 11}
        count = 0
        pairs = 0
        pcs   = pitches_active % 12
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                iv = abs(int(pcs[i]) - int(pcs[j])) % 12
                iv = min(iv, 12 - iv)
                if iv in DISSONANT:
                    count += 1
                pairs += 1
        return count / pairs if pairs > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """
    Cada muestra:
        'x'       : Tensor (N_ROLES, T, n_pitch)  — T = window_bars * resolution
        'tension' : Tensor (TENSION_DIM,)
        'mask'    : Tensor (N_ROLES,) bool
    """
    def __init__(self, data_dir, roles=None, augment=False):
        import numpy as np
        self.samples = []
        self.roles   = roles or ROLES
        self.n_roles = len(self.roles)
        self.augment = augment
        self._cache  = {}
        self.n_pitch = None

        npz_files = sorted(Path(data_dir).glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No hay .npz en {data_dir}")

        for path in npz_files:
            try:
                data = dict(np.load(str(path), allow_pickle=True))
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

        resolution  = meta['resolution']
        window_bars = meta['window_bars']

        x_parts = []
        mask    = []

        for role in self.roles:
            key = f'roll_{role}'
            if key in data:
                window = data[key][widx]                             # (window_bars, res, n_pitch)
                x_parts.append(window.reshape(-1, self.n_pitch))    # (T, n_pitch)
                mask.append(True)
            else:
                x_parts.append(
                    np.zeros((window_bars * resolution, self.n_pitch), dtype=np.float32))
                mask.append(False)

        x         = torch.tensor(np.stack(x_parts, axis=0))     # (N_ROLES, T, n_pitch)
        tension   = torch.tensor(data['tension'][widx])
        role_mask = torch.tensor(mask, dtype=torch.bool)

        if self.augment:
            import torch as th
            if th.rand(1).item() < 0.5:
                x = th.flip(x, dims=[2])        # espejo de pitch
            if th.rand(1).item() < 0.3:
                shift = int(th.randint(-3, 4, (1,)).item())
                x = th.roll(x, shift, dims=2)   # transposición aleatoria ±3 semitonos

        return {'x': x, 'tension': tension, 'mask': role_mask}


def _collate_fn(batch):
    import torch
    return {
        'x':       torch.stack([b['x']       for b in batch]),
        'tension': torch.stack([b['tension'] for b in batch]),
        'mask':    torch.stack([b['mask']    for b in batch]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA CYCLEGAN
#
#  G_A2B : Generador A → B   (ResNet + AdaIN)
#  G_B2A : Generador B → A
#  D_A   : Discriminador PatchGAN para el dominio A
#  D_B   : Discriminador PatchGAN para el dominio B
#
#  Pérdidas:
#    L_adv  = LSGAN (más estable que BCE para datos continuos)
#    L_cyc  = ||G_B2A(G_A2B(x_a)) - x_a||_1  * lambda_cycle
#    L_idt  = ||G_A2B(x_b) - x_b||_1          * lambda_identity
#
#  Formato del tensor: (B, N_ROLES, T, n_pitch)
#    T = window_bars * resolution  (dimensión temporal aplanada)
# ══════════════════════════════════════════════════════════════════════════════

def _build_cyclegan(n_roles, seq_len, n_pitch, tension_dim,
                    base_ch=64, n_res_blocks=6, n_downsamples=2):
    """
    Construye (G_A2B, G_B2A, D_A, D_B) para CycleGAN musical.

    Los generadores son ResNets 2D que operan sobre (N_ROLES, T, n_pitch).
    El vector de tensión (8D) condiciona cada bloque ResNet vía AdaIN.
    Los discriminadores son PatchGAN (3 capas, campo receptivo ~70px).
    """
    import torch
    import torch.nn as nn

    COND_DIM = 64   # dimensión del vector de condicionamiento proyectado

    # ── AdaIN ─────────────────────────────────────────────────────────────────
    class _AdaIN(nn.Module):
        """Adaptive Instance Normalization: escala/shift aprendidos desde la tensión."""
        def __init__(self, ch):
            super().__init__()
            self.norm = nn.InstanceNorm2d(ch, affine=False)
            self.proj = nn.Linear(COND_DIM, ch * 2)

        def forward(self, x, cond):
            # x: (B, ch, H, W),  cond: (B, COND_DIM)
            h  = self.norm(x)
            st = self.proj(cond)[:, :, None, None]
            scale, shift = st.chunk(2, dim=1)
            return h * (1 + scale) + shift

    # ── Bloque ResNet con AdaIN ───────────────────────────────────────────────
    class _ResBlock(nn.Module):
        def __init__(self, ch, dropout=0.0):
            super().__init__()
            self.conv1  = nn.Conv2d(ch, ch, 3, padding=1)
            self.adain1 = _AdaIN(ch)
            self.act    = nn.ReLU(inplace=True)
            self.drop   = nn.Dropout2d(dropout)
            self.conv2  = nn.Conv2d(ch, ch, 3, padding=1)
            self.adain2 = _AdaIN(ch)

        def forward(self, x, cond):
            h = self.act(self.adain1(self.conv1(x), cond))
            h = self.drop(h)
            h = self.adain2(self.conv2(h), cond)
            return x + h   # conexión residual

    # ── Generador ─────────────────────────────────────────────────────────────
    class _Generator(nn.Module):
        """
        Stem → Encoder (downsampling) → ResBlocks → Decoder (upsampling) → Head.
        El condicionamiento de tensión se inyecta vía AdaIN en cada ResBlock.
        Conexión residual escalada: el generador aprende el *delta* de estilo.
        """
        def __init__(self):
            super().__init__()

            # Proyector de tensión → COND_DIM
            self.cond_proj = nn.Sequential(
                nn.Linear(tension_dim, COND_DIM),
                nn.ReLU(),
                nn.Linear(COND_DIM, COND_DIM),
            )

            # Stem: n_roles → base_ch
            self.stem = nn.Sequential(
                nn.Conv2d(n_roles, base_ch, 7, padding=3),
                nn.InstanceNorm2d(base_ch),
                nn.ReLU(inplace=True),
            )

            # Encoder: n_downsamples niveles de downsampling 2×
            enc_layers = []
            ch_in = base_ch
            self._enc_out_ch = base_ch
            for i in range(n_downsamples):
                ch_out = min(ch_in * 2, base_ch * 8)
                enc_layers += [
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ]
                ch_in = ch_out
                self._enc_out_ch = ch_out
            self.encoder = nn.Sequential(*enc_layers)

            # Cuello de botella: n_res_blocks ResBlocks con AdaIN
            self.res_blocks = nn.ModuleList(
                [_ResBlock(self._enc_out_ch) for _ in range(n_res_blocks)]
            )

            # Decoder: n_downsamples niveles de upsampling 2×
            dec_layers = []
            ch_in = self._enc_out_ch
            for i in range(n_downsamples):
                ch_out = max(ch_in // 2, base_ch)
                dec_layers += [
                    nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ]
                ch_in = ch_out
            self.decoder = nn.Sequential(*dec_layers)

            # Head: ch_in → n_roles, Tanh → [−1, 1]
            self.head = nn.Sequential(
                nn.Conv2d(ch_in, n_roles, 7, padding=3),
                nn.Tanh(),
            )

        def forward(self, x, tension):
            # x: (B, N_ROLES, T, n_pitch),  tension: (B, tension_dim)
            cond = self.cond_proj(tension)           # (B, COND_DIM)
            h    = self.stem(x)
            h    = self.encoder(h)
            for blk in self.res_blocks:
                h = blk(h, cond)
            h = self.decoder(h)
            delta = self.head(h)
            # Residual escalado: preserva el contenido, aprende la diferencia de estilo
            return (x + delta * 0.5).clamp(-1, 1)

    # ── Discriminador PatchGAN ────────────────────────────────────────────────
    class _PatchDiscriminator(nn.Module):
        """
        Clasifica parches locales (no la secuencia completa).
        n_layers=3 → campo receptivo aproximado de 70 ticks × 70 pitches.
        """
        def __init__(self, n_layers=3):
            super().__init__()
            layers = [
                nn.Conv2d(n_roles, base_ch, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch_in = base_ch
            for i in range(1, n_layers):
                ch_out = min(ch_in * 2, base_ch * 8)
                s      = 2 if i < n_layers - 1 else 1
                layers += [
                    nn.Conv2d(ch_in, ch_out, 4, stride=s, padding=1),
                    nn.InstanceNorm2d(ch_out),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
                ch_in = ch_out
            ch_out = min(ch_in * 2, base_ch * 8)
            layers += [
                nn.Conv2d(ch_in, ch_out, 4, stride=1, padding=1),
                nn.InstanceNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ch_out, 1, 4, stride=1, padding=1),   # sin activación (LSGAN)
            ]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return {
        'G_A2B': _Generator(),
        'G_B2A': _Generator(),
        'D_A':   _PatchDiscriminator(),
        'D_B':   _PatchDiscriminator(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE POOL
# ══════════════════════════════════════════════════════════════════════════════

class _ImagePool:
    """Buffer de imágenes generadas para estabilizar el discriminador."""
    def __init__(self, size=50):
        self.size = size
        self.pool = []

    def query(self, imgs):
        import torch
        if self.size == 0:
            return imgs
        result = []
        for img in imgs:
            img = img.detach()
            if len(self.pool) < self.size:
                self.pool.append(img)
                result.append(img)
            elif torch.rand(1).item() < 0.5:
                idx = int(torch.randint(0, len(self.pool), (1,)).item())
                old = self.pool[idx].clone()
                self.pool[idx] = img
                result.append(old)
            else:
                result.append(img)
        return torch.stack(result)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR CYCLEGAN
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANTrainer:
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'
    CONFIG_NAME     = 'model_config.json'

    def __init__(self, nets, optimizers, model_dir,
                 lambda_cycle=10.0, lambda_identity=5.0, patience=40):
        self.G_A2B = nets['G_A2B']
        self.G_B2A = nets['G_B2A']
        self.D_A   = nets['D_A']
        self.D_B   = nets['D_B']
        self.opt_G = optimizers['opt_G']
        self.opt_D = optimizers['opt_D']
        self.model_dir       = Path(model_dir)
        self.lambda_cycle    = lambda_cycle
        self.lambda_identity = lambda_identity
        self.patience        = patience

        self.pool_A = _ImagePool(50)
        self.pool_B = _ImagePool(50)

        self.history     = {'train_G': [], 'train_D': [], 'val_G': [], 'val_D': [],
                             'cyc': [], 'idt': []}
        self.best_val    = float('inf')
        self.no_improve  = 0
        self.start_epoch = 0

    @staticmethod
    def _lsgan(pred, real):
        import torch.nn.functional as F
        import torch
        t = torch.ones_like(pred) if real else torch.zeros_like(pred)
        return F.mse_loss(pred, t)

    def _step(self, batch_a, batch_b, training):
        import torch
        import torch.nn.functional as F

        x_a = batch_a['x']
        x_b = batch_b['x']
        t_a = batch_a['tension']
        t_b = batch_b['tension']

        # ── Generadores ───────────────────────────────────────────────────────
        if training:
            self.opt_G.zero_grad()

        fake_b = self.G_A2B(x_a, t_a)
        fake_a = self.G_B2A(x_b, t_b)

        loss_adv_A2B = self._lsgan(self.D_B(fake_b), True)
        loss_adv_B2A = self._lsgan(self.D_A(fake_a), True)

        rec_a = self.G_B2A(fake_b, t_a)
        rec_b = self.G_A2B(fake_a, t_b)
        loss_cyc = (F.l1_loss(rec_a, x_a) + F.l1_loss(rec_b, x_b)) * self.lambda_cycle

        idt_b = self.G_A2B(x_b, t_b)
        idt_a = self.G_B2A(x_a, t_a)
        loss_idt = (F.l1_loss(idt_b, x_b) + F.l1_loss(idt_a, x_a)) * self.lambda_identity

        loss_G = loss_adv_A2B + loss_adv_B2A + loss_cyc + loss_idt

        if training:
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()), 1.0)
            self.opt_G.step()

        # ── Discriminadores ───────────────────────────────────────────────────
        if training:
            self.opt_D.zero_grad()

        # D_B: real=x_b, fake=fake_b (del pool)
        loss_D_B = 0.5 * (
            self._lsgan(self.D_B(x_b), True) +
            self._lsgan(self.D_B(self.pool_B.query(fake_b.unsqueeze(0).squeeze(0)
                                                    if fake_b.dim() == 4 else fake_b).detach()), False)
        )
        # D_A: real=x_a, fake=fake_a (del pool)
        loss_D_A = 0.5 * (
            self._lsgan(self.D_A(x_a), True) +
            self._lsgan(self.D_A(self.pool_A.query(fake_a.unsqueeze(0).squeeze(0)
                                                    if fake_a.dim() == 4 else fake_a).detach()), False)
        )
        loss_D = loss_D_A + loss_D_B

        if training:
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.D_A.parameters()) + list(self.D_B.parameters()), 1.0)
            self.opt_D.step()

        return {
            'G':   loss_G.item(),
            'D':   loss_D.item(),
            'cyc': loss_cyc.item(),
            'idt': loss_idt.item(),
        }

    def _run_epoch(self, loader_a, loader_b, training, epoch=0, n_epochs=0):
        import torch, time, math
        for net in [self.G_A2B, self.G_B2A, self.D_A, self.D_B]:
            net.train(training)

        totals = {'G': 0.0, 'D': 0.0, 'cyc': 0.0, 'idt': 0.0}
        n      = 0
        phase  = 'train' if training else 'val  '
        t0_ep  = time.time()

        ctx      = torch.enable_grad() if training else torch.no_grad()
        iter_b   = iter(loader_b)
        device   = next(self.G_A2B.parameters()).device

        with ctx:
            for batch_a in loader_a:
                try:
                    batch_b = next(iter_b)
                except StopIteration:
                    iter_b  = iter(loader_b)
                    batch_b = next(iter_b)

                batch_a = {k: v.to(device, non_blocking=True) for k, v in batch_a.items()}
                batch_b = {k: v.to(device, non_blocking=True) for k, v in batch_b.items()}

                m = self._step(batch_a, batch_b, training=training)
                if math.isnan(m['G']) or math.isnan(m['D']):
                    continue
                for k in totals:
                    totals[k] += m[k]
                n += 1

                if n % 20 == 0 or n == 1:
                    elapsed = time.time() - t0_ep
                    print(f"\r  [{phase}] "
                          f"época {epoch+1}/{n_epochs}  "
                          f"batch {n}  "
                          f"G={totals['G']/n:.4f}  "
                          f"D={totals['D']/n:.4f}  "
                          f"cyc={totals['cyc']/n:.4f}  "
                          f"idt={totals['idt']/n:.4f}  "
                          f"({elapsed:.0f}s)", end='', flush=True)
        print()
        return {k: (v / n if n > 0 else 0.0) for k, v in totals.items()}

    def save_checkpoint(self, epoch, is_best):
        import torch
        state = {
            'epoch':      epoch,
            'G_A2B':      self.G_A2B.state_dict(),
            'G_B2A':      self.G_B2A.state_dict(),
            'D_A':        self.D_A.state_dict(),
            'D_B':        self.D_B.state_dict(),
            'opt_G':      self.opt_G.state_dict(),
            'opt_D':      self.opt_D.state_dict(),
            'best_val':   self.best_val,
            'no_improve': self.no_improve,
            'history':    self.history,
        }
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        with open(self.model_dir / self.HISTORY_NAME, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path=None):
        import torch
        ckpt = Path(path) if path else (self.model_dir / self.CHECKPOINT_NAME)
        if not ckpt.exists():
            print("[train] Entrenando desde cero.")
            return
        state = torch.load(str(ckpt), map_location='cpu')
        self.G_A2B.load_state_dict(state['G_A2B'])
        self.G_B2A.load_state_dict(state['G_B2A'])
        self.D_A.load_state_dict(state['D_A'])
        self.D_B.load_state_dict(state['D_B'])
        self.opt_G.load_state_dict(state['opt_G'])
        self.opt_D.load_state_dict(state['opt_D'])
        self.best_val    = state['best_val']
        self.no_improve  = state['no_improve']
        self.history     = state['history']
        self.start_epoch = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_epoch}  "
              f"(mejor val_G={self.best_val:.4f})")

    def train(self, loader_a, loader_b, val_a, val_b, n_epochs):
        import torch

        # LR decay lineal en la segunda mitad (estándar de CycleGAN)
        def _lr_lambda(epoch):
            start = n_epochs // 2
            if epoch < start:
                return 1.0
            return max(0.0, 1.0 - (epoch - start) / max(n_epochs - start, 1))

        sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, _lr_lambda)
        sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, _lr_lambda)
        for _ in range(self.start_epoch):
            sched_G.step()
            sched_D.step()

        for epoch in range(self.start_epoch, n_epochs):
            tr  = self._run_epoch(loader_a, loader_b, training=True,
                                   epoch=epoch, n_epochs=n_epochs)
            val = self._run_epoch(val_a,    val_b,    training=False,
                                   epoch=epoch, n_epochs=n_epochs)
            sched_G.step()
            sched_D.step()

            for k in ('G', 'D'):
                self.history[f'train_{k}'].append(tr[k])
                self.history[f'val_{k}'].append(val[k])
            self.history['cyc'].append(tr['cyc'])
            self.history['idt'].append(tr['idt'])

            val_loss = val['G']
            is_best  = val_loss < self.best_val
            if is_best:
                self.best_val   = val_loss
                self.no_improve = 0
                print(f"  ✓ Nuevo mejor modelo  val_G={val_loss:.4f}")
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, is_best)

            lr_now = self.opt_G.param_groups[0]['lr']
            print(f"  Época {epoch+1}/{n_epochs}  "
                  f"train_G={tr['G']:.4f}  train_D={tr['D']:.4f}  "
                  f"val_G={val['G']:.4f}  "
                  f"cyc={tr['cyc']:.4f}  idt={tr['idt']:.4f}  "
                  f"lr={lr_now:.2e}  sin_mejora={self.no_improve}/{self.patience}")

            if self.no_improve >= self.patience:
                print(f"\n[train] Early stopping en época {epoch+1}.")
                break

        print(f"\n[train] Entrenamiento completado. Mejor val_G={self.best_val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES: CARGA DE MODELO / MIDI ↔ ROLL
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_and_config(model_dir):
    import torch
    model_dir = Path(model_dir)
    cfg_path  = model_dir / CycleGANTrainer.CONFIG_NAME
    ckpt_path = model_dir / CycleGANTrainer.BEST_NAME

    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró {cfg_path}. ¿Has ejecutado train?")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró {ckpt_path}. ¿Has ejecutado train?")

    with open(cfg_path) as f:
        cfg = json.load(f)

    nets = _build_cyclegan(
        n_roles       = cfg['n_roles'],
        seq_len       = cfg['seq_len'],
        n_pitch       = cfg.get('n_pitch', PITCH_CLASSES),
        tension_dim   = cfg['tension_dim'],
        base_ch       = cfg.get('base_ch', 64),
        n_res_blocks  = cfg.get('n_res_blocks', 6),
        n_downsamples = cfg.get('n_downsamples', 2),
    )

    state = torch.load(str(ckpt_path), map_location='cpu')
    nets['G_A2B'].load_state_dict(state['G_A2B'])
    nets['G_B2A'].load_state_dict(state['G_B2A'])
    nets['G_A2B'].eval()
    nets['G_B2A'].eval()
    return nets, cfg


def _midi_to_rolls(midi_path, cfg):
    import mido, numpy as np

    mid = mido.MidiFile(midi_path)
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise ValueError(f"No se encontraron notas en {midi_path}")

    tpb       = mid.ticks_per_beat
    tpbar     = tpb * 4
    max_tick  = max(n[1] for nl in note_lists.values() for n in nl)
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
        notes = note_lists.get(stream_key, [])
        if not notes:
            continue
        roll = conv.notes_to_roll(notes, tpbar, total_bars)
        if do_crop:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        rolls[role] = roll
    return rolls


def _rolls_to_tensor(rolls, cfg):
    """rolls {role: (n_bars, res, n_pitch)} → tensor (1, N_ROLES, T, n_pitch) en [-1, 1]."""
    import numpy as np, torch

    n_roles    = cfg['n_roles']
    role_list  = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg.get('n_pitch', PITCH_CLASSES)

    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 1
    T      = n_bars * resolution

    x = np.zeros((n_roles, T, n_pitch), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls:
            flat = rolls[role].reshape(-1, n_pitch)
            L    = min(len(flat), T)
            x[ridx, :L] = flat[:L]

    x = x * 2.0 - 1.0   # [0, 1] → [-1, 1]
    return torch.tensor(x).unsqueeze(0)   # (1, N_ROLES, T, n_pitch)


def _tensor_to_rolls(tensor, cfg):
    """tensor (1, N_ROLES, T, n_pitch) en [-1, 1] → rolls {role: (n_bars, res, n_pitch)}."""
    import numpy as np

    role_list  = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg.get('n_pitch', PITCH_CLASSES)

    x = (tensor[0].cpu().numpy() + 1.0) * 0.5   # [-1, 1] → [0, 1]
    x = x.clip(0.0, 1.0)   # (N_ROLES, T, n_pitch)

    rolls = {}
    for ridx, role in enumerate(role_list):
        flat   = x[ridx]   # (T, n_pitch)
        n_bars = max(len(flat) // resolution, 1)
        rolls[role] = flat[:n_bars * resolution].reshape(n_bars, resolution, n_pitch)
    return rolls


def _adaptive_threshold(roll, percentile=99.0):
    import numpy as np
    vals = roll.flatten()
    nonzero = vals[vals > 1e-4]
    if len(nonzero) == 0:
        return 0.5
    return float(np.percentile(nonzero, percentile))


def _rolls_to_midi(bars_per_role, cfg, palette, output_path,
                   bpm=120.0, threshold=None, adaptive_per_bar=False):
    import mido, numpy as np

    resolution = cfg['resolution']
    tpb        = 480
    ticks_bar  = tpb * 4
    ticks_tick = ticks_bar / resolution

    pitch_lo  = cfg.get('pitch_lo', 0)
    pitch_hi  = cfg.get('pitch_hi', 127)
    do_expand = (pitch_lo, pitch_hi) != (0, 127)

    mid  = mido.MidiFile(ticks_per_beat=tpb)
    t0   = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / bpm), time=0))
    mid.tracks.append(t0)
    n_notes_total = 0

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]   # (n_bars, res, n_pitch)

        if do_expand:
            roll = _pad_pitch(roll, pitch_lo, n_full=128)

        thr  = threshold if threshold is not None else _adaptive_threshold(roll)
        pal  = palette.get(role, {})
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        if adaptive_per_bar:
            binary = np.zeros_like(roll)
            for b in range(roll.shape[0]):
                bmax = roll[b].max()
                if bmax >= thr:
                    bar_thr = thr
                elif bmax >= 0.1:
                    bar_thr = bmax * 0.5
                else:
                    bar_thr = 1.1
                binary[b] = (roll[b] > bar_thr).astype(np.float32)
        else:
            binary = (roll > thr).astype(np.float32)

        # Eliminar notas aisladas de 1 tick
        for b in range(binary.shape[0]):
            for p in range(128):
                col = binary[b, :, p]
                for t in range(1, len(col) - 1):
                    if col[t] == 1 and col[t-1] == 0 and col[t+1] == 0:
                        binary[b, t, p] = 0

        events = []
        n_bars_r, res_r, _ = binary.shape
        for bar in range(n_bars_r):
            for tick in range(res_r):
                abs_tick = int((bar * res_r + tick) * ticks_tick)
                for pitch in range(128):
                    cur  = binary[bar, tick, pitch] > 0
                    prev = binary[bar, tick-1, pitch] > 0 if tick > 0 else \
                           (binary[bar-1, -1, pitch] > 0 if bar > 0 else False)
                    if cur and not prev:
                        events.append((abs_tick, 'on', pitch))
                    elif not cur and prev:
                        events.append((abs_tick, 'off', pitch))

        last_tick = int(n_bars_r * res_r * ticks_tick)
        for pitch in range(128):
            if binary[-1, -1, pitch] > 0:
                events.append((last_tick, 'off', pitch))

        n_notes_total += sum(1 for e in events if e[1] == 'on')
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

        prev_tick = 0
        for abs_tick, etype, pitch in events:
            delta = abs_tick - prev_tick
            if etype == 'on':
                track.append(mido.Message('note_on',  channel=ch, note=pitch,
                                           velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', channel=ch, note=pitch,
                                           velocity=0,   time=delta))
            prev_tick = abs_tick

        remaining = last_tick - prev_tick
        if remaining > 0:
            track.append(mido.MetaMessage('end_of_track', time=remaining))

    mid.save(output_path)
    return n_notes_total


def _load_palette(palette_path, cfg):
    if not palette_path or not Path(palette_path).exists():
        return dict(DEFAULT_PALETTE)
    with open(palette_path) as f:
        user = json.load(f)
    palette = dict(DEFAULT_PALETTE)
    for role, params in user.items():
        palette[role] = {**DEFAULT_PALETTE.get(role, {}), **params}
    return palette


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    (midi_path, output_dir, resolution, window_bars,
     active_roles, pitch_lo, pitch_hi) = args_tuple
    import numpy as np

    stem          = midi_path.stem
    stats_partial = {r: 0 for r in ROLES}
    stats_partial.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})
    n_pitch = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES

    try:
        mid = _load_midi(str(midi_path))
    except Exception as e:
        return stem, f"ERROR al cargar: {e}", None, stats_partial

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        stats_partial['files_skipped'] = 1
        return stem, "sin notas — omitido", None, stats_partial

    assigner  = RoleAssigner()
    converter = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    extractor = TensionExtractor()

    role_assignment = assigner.assign(mid)
    if not role_assignment:
        stats_partial['files_skipped'] = 1
        return stem, "sin asignación de roles — omitido", None, stats_partial

    tpb_raw    = _ticks_per_bar(mid)
    all_ticks  = max((n[1] for nl in note_lists.values() for n in nl), default=0)
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
        stats_partial[role] = 1

    if not role_rolls:
        stats_partial['files_skipped'] = 1
        return stem, "no se pudo construir ningún piano roll — omitido", None, stats_partial

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
        stats_partial['files_skipped'] = 1
        return stem, f"demasiado corto ({total_bars} compases) — omitido", None, stats_partial

    for role in role_windows:
        role_windows[role] = role_windows[role][:min_windows]

    tension_bars    = extractor.extract_bar_vectors(role_rolls, total_bars)
    mid_offset      = window_bars // 2
    tension_windows = tension_bars[mid_offset: mid_offset + min_windows]
    if len(tension_windows) < min_windows:
        pad = np.zeros((min_windows - len(tension_windows),
                        TensionExtractor.TENSION_DIM), dtype=np.float32)
        tension_windows = np.concatenate([tension_windows, pad], axis=0)

    save_dict = {'tension': tension_windows}
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
    np.savez_compressed(str(Path(output_dir) / f"{stem}.npz"), **save_dict)

    stats_partial['files_ok']      = 1
    stats_partial['total_windows'] = min_windows
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, "
            f"roles: {', '.join(roles_found)})",
            True, stats_partial)


def cmd_prepare(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

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
        print(f"[prepare] Rango de pitch       : {pitch_n} notas "
              f"(MIDI {pitch_lo}–{pitch_hi})")

    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    n_workers = min(multiprocessing.cpu_count(), len(midi_files), 8)
    print(f"[prepare] {len(midi_files)} archivos MIDI  |  "
          f"resolución {args.resolution}  |  ventana {args.window_bars} compases  |  "
          f"{n_workers} procesos\n")

    stats = {r: 0 for r in ROLES}
    stats.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})
    task_args = [
        (midi_path, str(output_dir), args.resolution, args.window_bars,
         active_roles, pitch_lo, pitch_hi)
        for midi_path in midi_files
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prepare_one_midi, a): a[0] for a in task_args}
        for future in futures:
            try:
                stem, msg, ok, partial = future.result()
            except Exception as e:
                print(f"  [{Path(futures[future]).stem}] EXCEPCIÓN: {e}")
                stats['files_skipped'] += 1
                continue
            print(f"  [{stem}] {msg}")
            for role in ROLES:
                stats[role] += partial[role]
            stats['files_ok']      += partial['files_ok']
            stats['files_skipped'] += partial['files_skipped']
            stats['total_windows'] += partial['total_windows']

    print()
    print("═" * 60)
    print("  RESUMEN PREPARE")
    print("═" * 60)
    print(f"  Archivos procesados : {stats['files_ok']}")
    print(f"  Archivos omitidos   : {stats['files_skipped']}")
    print(f"  Ventanas totales    : {stats['total_windows']}")
    print("\n  Cobertura de roles:")
    for role in ROLES:
        print(f"    {role:<16} {stats[role]} archivos")
    print("═" * 60)

    if getattr(args, 'report', False):
        rp = output_dir / 'prepare_report.json'
        with open(rp, 'w') as f:
            json.dump({'input_dir': str(input_dir), 'output_dir': str(output_dir),
                       'resolution': args.resolution, 'window_bars': args.window_bars,
                       'active_roles': active_roles, 'stats': stats}, f, indent=2)
        print(f"\n  Reporte guardado en {rp}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader, random_split
    import numpy as _np

    data_dir_a = Path(args.data_dir_a)
    data_dir_b = Path(args.data_dir_b)
    model_dir  = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]
    if disabled:
        print(f"[train] Roles deshabilitados : {', '.join(sorted(disabled))}")

    print("[train] Cargando datasets ...")
    ds_a = MidiRollDataset(str(data_dir_a), roles=active_roles, augment=True)
    ds_b = MidiRollDataset(str(data_dir_b), roles=active_roles, augment=True)

    if ds_a.n_pitch != ds_b.n_pitch:
        print(f"[train] ⚠  n_pitch incompatible: A={ds_a.n_pitch} vs B={ds_b.n_pitch}")
        print("[train]    Ambos corpus deben prepararse con el mismo --pitch-range")
        sys.exit(1)

    n_val_a = max(1, int(len(ds_a) * 0.1))
    n_val_b = max(1, int(len(ds_b) * 0.1))
    tr_a, val_a = random_split(ds_a, [len(ds_a) - n_val_a, n_val_a])
    tr_b, val_b = random_split(ds_b, [len(ds_b) - n_val_b, n_val_b])

    loader_tr_a  = DataLoader(tr_a,  batch_size=args.batch_size, shuffle=True,
                               collate_fn=_collate_fn, num_workers=0)
    loader_tr_b  = DataLoader(tr_b,  batch_size=args.batch_size, shuffle=True,
                               collate_fn=_collate_fn, num_workers=0)
    loader_val_a = DataLoader(val_a, batch_size=args.batch_size, shuffle=False,
                               collate_fn=_collate_fn, num_workers=0)
    loader_val_b = DataLoader(val_b, batch_size=args.batch_size, shuffle=False,
                               collate_fn=_collate_fn, num_workers=0)

    sample      = ds_a[0]
    n_roles     = sample['x'].shape[0]
    seq_len     = sample['x'].shape[1]
    n_pitch     = ds_a.n_pitch
    tension_dim = sample['tension'].shape[0]

    _first = sorted(data_dir_a.glob('*.npz'))[0]
    _meta  = json.loads(str(_np.load(str(_first), allow_pickle=True)['meta_json'][0]))
    resolution  = _meta['resolution']
    window_bars = _meta['window_bars']
    pitch_lo    = _meta.get('pitch_lo', 0)
    pitch_hi    = _meta.get('pitch_hi', 127)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Dominio A: {len(ds_a)} muestras  |  Dominio B: {len(ds_b)} muestras")
    print(f"[train] n_roles={n_roles}  seq_len={seq_len}  n_pitch={n_pitch}  "
          f"tension_dim={tension_dim}")
    print(f"[train] base_ch={args.base_ch}  n_res_blocks={args.n_res_blocks}  "
          f"n_downsamples={args.n_downsamples}")
    print(f"[train] lambda_cycle={args.lambda_cycle}  "
          f"lambda_identity={args.lambda_identity}")
    print(f"[train] Dispositivo: {device}")

    nets = _build_cyclegan(
        n_roles       = n_roles,
        seq_len       = seq_len,
        n_pitch       = n_pitch,
        tension_dim   = tension_dim,
        base_ch       = args.base_ch,
        n_res_blocks  = args.n_res_blocks,
        n_downsamples = args.n_downsamples,
    )
    for net in nets.values():
        net.to(device)

    params_G = (list(nets['G_A2B'].parameters()) + list(nets['G_B2A'].parameters()))
    params_D = (list(nets['D_A'].parameters())   + list(nets['D_B'].parameters()))
    opt_G = torch.optim.Adam(params_G, lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(params_D, lr=args.lr, betas=(0.5, 0.999))

    trainer = CycleGANTrainer(
        nets, {'opt_G': opt_G, 'opt_D': opt_D},
        model_dir,
        lambda_cycle    = args.lambda_cycle,
        lambda_identity = args.lambda_identity,
        patience        = args.patience,
    )

    if args.resume:
        trainer.load_checkpoint()

    cfg = {
        'n_roles':        n_roles,
        'roles':          active_roles,
        'seq_len':        seq_len,
        'resolution':     resolution,
        'window_bars':    window_bars,
        'n_pitch':        n_pitch,
        'pitch_lo':       pitch_lo,
        'pitch_hi':       pitch_hi,
        'tension_dim':    tension_dim,
        'base_ch':        args.base_ch,
        'n_res_blocks':   args.n_res_blocks,
        'n_downsamples':  args.n_downsamples,
        'lambda_cycle':   args.lambda_cycle,
        'lambda_identity': args.lambda_identity,
        'model_version':  'cyclegan_v1',
    }
    with open(model_dir / CycleGANTrainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer.train(loader_tr_a, loader_tr_b, loader_val_a, loader_val_b, args.epochs)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFER
# ══════════════════════════════════════════════════════════════════════════════

def cmd_transfer(args):
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[transfer] Cargando modelo desde {model_dir} ...")
    nets, cfg = _load_model_and_config(model_dir)

    direction = args.direction.upper()
    if direction not in ('A2B', 'B2A'):
        print(f"[transfer] Dirección inválida: {direction}  (use A2B o B2A)")
        sys.exit(1)

    G       = nets[f'G_{direction}']
    palette = _load_palette(args.palette, cfg)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    G.to(device)

    print(f"[transfer] Procesando {args.input}  (dirección: {direction}) ...")
    rolls = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[transfer] ERROR: no se pudieron extraer rolls del MIDI")
        sys.exit(1)

    x_tensor = _rolls_to_tensor(rolls, cfg).to(device)

    # Tensión media del MIDI de entrada para condicionar el generador
    n_bars       = min(r.shape[0] for r in rolls.values())
    tension_bars = TensionExtractor().extract_bar_vectors(rolls, n_bars)
    tension_t    = torch.tensor(tension_bars.mean(axis=0)).unsqueeze(0).to(device)

    print(f"[transfer] Generando  ({n_bars} compases) ...")
    with torch.no_grad():
        out_tensor = G(x_tensor, tension_t)

    out_rolls = _tensor_to_rolls(out_tensor, cfg)

    # Diagnóstico del primer compás
    all_vals = np.concatenate([r.flatten() for r in out_rolls.values()])
    if getattr(args, 'threshold', None):
        thr        = args.threshold
        thr_method = f'fijo ({thr:.3f})'
    else:
        thr_pct    = getattr(args, 'threshold_pct', 99.0)
        thr        = _adaptive_threshold(all_vals, thr_pct)
        thr_method = f'p{thr_pct}'

    n_active = int((all_vals > thr).sum())
    density  = 100 * n_active / max(len(all_vals), 1)
    print(f"\n  [diag] Umbral {thr_method}: {thr:.4f}  →  "
          f"{n_active} píxeles activos ({density:.2f}%)")

    if density < 0.2:
        print("  [diag] ⚠  Densidad baja — prueba --threshold-pct 99.5 "
              "o más épocas de entrenamiento")
    elif density > 10.0:
        print("  [diag] ⚠  Densidad alta — prueba --threshold-pct 98")
    elif all_vals.max() < 0.05:
        print("  [diag] ⚠  Activaciones muy bajas — el modelo necesita más entrenamiento")

    n_notes = _rolls_to_midi(out_rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=thr,
                              adaptive_per_bar=getattr(args, 'adaptive_per_bar', False))
    print(f"[transfer] MIDI guardado en {args.output}  "
          f"({n_notes} notas, umbral={thr:.3f})")
    if n_notes == 0:
        print("[transfer] ⚠  MIDI vacío. "
              "Prueba --threshold-pct 70 o revisa el entrenamiento.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_encode(args):
    import torch, numpy as np

    nets, cfg = _load_model_and_config(args.model_dir)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    G_A2B     = nets['G_A2B'].to(device)

    rolls = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[encode] ERROR: no se pudieron extraer rolls")
        sys.exit(1)

    x_t      = _rolls_to_tensor(rolls, cfg).to(device)
    n_bars   = min(r.shape[0] for r in rolls.values())
    t_bars   = TensionExtractor().extract_bar_vectors(rolls, n_bars)
    tension_t = torch.tensor(t_bars.mean(axis=0)).unsqueeze(0).to(device)

    # Extraemos la activación del último ResBlock como proxy del estilo aprendido
    acts = {}
    def hook(m, i, o): acts['z'] = o.detach().cpu()
    handle = G_A2B.res_blocks[-1].register_forward_hook(hook)
    with torch.no_grad():
        G_A2B(x_t, tension_t)
    handle.remove()

    z = acts['z'].mean(dim=(2, 3))[0].numpy()
    out_path = args.output or 'z_style.json'
    with open(out_path, 'w') as f:
        json.dump({'z': z.tolist(), 'source': str(args.input),
                   'direction': 'A2B', 'dim': len(z)}, f, indent=2)
    print(f"[encode] Vector de estilo guardado en {out_path}  (dim={len(z)})")


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE-CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_style_corpus(args):
    import torch, numpy as np

    nets, cfg  = _load_model_and_config(args.model_dir)
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    G_A2B      = nets['G_A2B'].to(device)
    input_dir  = Path(args.input_dir)

    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[style-corpus] No se encontraron MIDI en {input_dir}")
        sys.exit(1)

    print(f"[style-corpus] Procesando {len(midi_files)} archivos ...")
    zs = []
    for mf in midi_files:
        try:
            rolls = _midi_to_rolls(str(mf), cfg)
            if not rolls:
                continue
            x_t      = _rolls_to_tensor(rolls, cfg).to(device)
            n_bars   = min(r.shape[0] for r in rolls.values())
            t_bars   = TensionExtractor().extract_bar_vectors(rolls, n_bars)
            tension_t = torch.tensor(t_bars.mean(axis=0)).unsqueeze(0).to(device)

            acts = {}
            def hook(m, i, o): acts['z'] = o.detach().cpu()
            handle = G_A2B.res_blocks[-1].register_forward_hook(hook)
            with torch.no_grad():
                G_A2B(x_t, tension_t)
            handle.remove()

            zs.append(acts['z'].mean(dim=(2, 3))[0].numpy())
            print(f"  {mf.stem}  ✓")
        except Exception as e:
            print(f"  {mf.stem}  ERROR: {e}")

    if not zs:
        print("[style-corpus] No se pudo procesar ningún archivo")
        sys.exit(1)

    centroid = np.stack(zs).mean(axis=0)
    out_path = args.output or 'z_corpus_centroid.json'
    with open(out_path, 'w') as f:
        json.dump({'z': centroid.tolist(), 'n_files': len(zs),
                   'dim': len(centroid)}, f, indent=2)
    print(f"\n[style-corpus] Centroide guardado en {out_path}  "
          f"({len(zs)} archivos, dim={len(centroid)})")


# ══════════════════════════════════════════════════════════════════════════════
#  INSPECT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    import numpy as np

    what = getattr(args, 'what', 'model')

    if 'model' in what:
        model_dir = Path(args.model_dir) if args.model_dir else None
        if model_dir is None:
            print("[inspect] --model-dir requerido para --what model")
            return

        cfg_path  = model_dir / CycleGANTrainer.CONFIG_NAME
        hist_path = model_dir / CycleGANTrainer.HISTORY_NAME
        ckpt_path = model_dir / CycleGANTrainer.BEST_NAME

        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print("\n  CONFIGURACIÓN DEL MODELO")
            print("  " + "─" * 40)
            for k, v in cfg.items():
                print(f"    {k:<24} {v}")

        if ckpt_path.exists():
            import torch
            state = torch.load(str(ckpt_path), map_location='cpu')
            print(f"\n  Mejor val_G    : {state.get('best_val', 'N/A')}")
            print(f"  Épocas totales : {state.get('epoch', 0) + 1}")

        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            n = len(hist.get('train_G', []))
            if n > 0:
                print(f"\n  Historial ({n} épocas registradas)")
                for k in ('train_G', 'val_G', 'cyc', 'idt'):
                    if hist.get(k):
                        print(f"    {k:<14} último={hist[k][-1]:.4f}  "
                              f"mín={min(hist[k]):.4f}")

        try:
            nets, cfg = _load_model_and_config(model_dir)
            for name in ('G_A2B', 'G_B2A', 'D_A', 'D_B'):
                n_params = sum(p.numel() for p in nets[name].parameters())
                print(f"  {name}  parámetros: {n_params:,}")
        except Exception:
            pass

    if 'npz' in what:
        data_dir = Path(args.data_dir) if args.data_dir else None
        if data_dir is None:
            print("[inspect] --data-dir requerido para --what npz")
            return

        npz_files = sorted(data_dir.glob('*.npz'))
        if not npz_files:
            print(f"[inspect] No hay .npz en {data_dir}")
            return

        target = getattr(args, 'npz_file', None)
        if target:
            npz_files = [f for f in npz_files if f.stem == target]

        limit = None if getattr(args, 'no_roll', False) else 5
        for path in (npz_files[:limit] if limit else npz_files):
            try:
                data = dict(np.load(str(path), allow_pickle=True))
                meta = json.loads(str(data['meta_json'][0]))
                print(f"\n  {path.stem}")
                print(f"    compases / ventanas : {meta['total_bars']} / {meta['n_windows']}")
                print(f"    roles               : {', '.join(meta['roles'])}")
                print(f"    resolución / n_pitch: {meta['resolution']} / {meta.get('n_pitch', 128)}")
                if not getattr(args, 'no_roll', False):
                    for role in meta['roles']:
                        key = f'roll_{role}'
                        if key in data:
                            arr     = data[key]
                            density = float(arr.mean()) * 100
                            print(f"    {role:<16} shape={arr.shape}  "
                                  f"densidad={density:.2f}%")
            except Exception as e:
                print(f"  {path.stem}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUND-TRIP
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    import numpy as np

    print(f"[round-trip] {args.input}")
    mid        = _load_midi(args.input)
    note_lists = _extract_note_lists(mid)
    if not note_lists:
        print("[round-trip] ERROR: no se encontraron notas")
        sys.exit(1)

    resolution  = getattr(args, 'resolution', TICKS_PER_BAR_DEFAULT)
    window_bars = getattr(args, 'window_bars', WINDOW_BARS_DEFAULT)
    tpb_raw     = _ticks_per_bar(mid)
    all_ticks   = max(n[1] for nl in note_lists.values() for n in nl)
    total_bars  = max(1, int(all_ticks / tpb_raw) + 1)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    rolls    = {}
    for role, key in role_map.items():
        notes = note_lists.get(key, [])
        if notes:
            rolls[role] = conv.notes_to_roll(notes, tpb_raw, total_bars)

    print(f"  Compases        : {total_bars}")
    print(f"  Roles asignados :")
    for role, roll in rolls.items():
        density = float(roll.mean()) * 100
        print(f"    {role:<16} shape={roll.shape}  densidad={density:.2f}%")

    cfg_dummy = {
        'roles': list(rolls.keys()), 'n_roles': len(rolls),
        'resolution': resolution, 'window_bars': window_bars,
        'seq_len': window_bars * resolution, 'n_pitch': PITCH_CLASSES,
        'pitch_lo': 0, 'pitch_hi': 127,
    }
    out_path = getattr(args, 'output', 'round_trip_output.mid')
    n_notes  = _rolls_to_midi(rolls, cfg_dummy, DEFAULT_PALETTE, out_path,
                               bpm=getattr(args, 'bpm', 120.0))
    print(f"\n[round-trip] MIDI reconstruido en {out_path}  ({n_notes} notas)")
    if n_notes == 0:
        print("[round-trip] ⚠  MIDI vacío — posible problema en la extracción de notas")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='cycle_gan_composer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CYCLE-GAN COMPOSER v1
            Transferencia de estilo musical sobre piano rolls multi-rol.

            Flujo típico:
              1. prepare   — convertir corpus A y corpus B a .npz
              2. train     — entrenar el modelo A ↔ B
              3. transfer  — aplicar transferencia a un MIDI concreto
        """),
    )

    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('prepare',
        help='MIDI corpus → piano rolls segmentados (.npz)')
    p.add_argument('--input-dir',    required=True,  metavar='DIR')
    p.add_argument('--output-dir',   required=True,  metavar='DIR')
    p.add_argument('--resolution',   type=int, default=TICKS_PER_BAR_DEFAULT,
                   metavar='INT',
                   help=f'Ticks por compás (default: {TICKS_PER_BAR_DEFAULT})')
    p.add_argument('--window-bars',  type=int, default=WINDOW_BARS_DEFAULT,
                   metavar='INT',
                   help=f'Compases por ventana (default: {WINDOW_BARS_DEFAULT})')
    p.add_argument('--pitch-range',  type=int, default=None, metavar='INT',
                   help='Limitar a N notas centradas en Do central (ej: 48)')
    p.add_argument('--disable-roles', nargs='+', metavar='ROL', choices=ROLES)
    p.add_argument('--report',       action='store_true')
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train',
        help='Entrena el modelo CycleGAN (dominio A ↔ dominio B)')
    p.add_argument('--data-dir-a',    required=True,  metavar='DIR')
    p.add_argument('--data-dir-b',    required=True,  metavar='DIR')
    p.add_argument('--model-dir',     required=True,  metavar='DIR')
    p.add_argument('--epochs',        type=int,   default=200)
    p.add_argument('--batch-size',    type=int,   default=8)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--lambda-cycle',  type=float, default=10.0,  metavar='F',
                   help='Peso cycle-consistency (default: 10)')
    p.add_argument('--lambda-identity', type=float, default=5.0, metavar='F',
                   help='Peso identity (default: 5)')
    p.add_argument('--base-ch',       type=int,   default=64,
                   help='Canales base del generador (default: 64)')
    p.add_argument('--n-res-blocks',  type=int,   default=6,
                   help='Bloques ResNet en bottleneck (default: 6)')
    p.add_argument('--n-downsamples', type=int,   default=2,
                   help='Niveles de downsampling (default: 2)')
    p.add_argument('--patience',      type=int,   default=40)
    p.add_argument('--disable-roles', nargs='+',  metavar='ROL', choices=ROLES)
    p.add_argument('--resume',        action='store_true')
    p.set_defaults(func=cmd_train)

    # ── transfer ──────────────────────────────────────────────────────────────
    p = sub.add_parser('transfer',
        help='Aplica el generador entrenado a un MIDI concreto')
    p.add_argument('--model-dir',      required=True, metavar='DIR')
    p.add_argument('--input',          required=True, metavar='FILE')
    p.add_argument('--direction',      default='A2B', choices=['A2B', 'B2A'],
                   help='A2B: estilo de A → estilo de B  (default: A2B)')
    p.add_argument('--palette',        default=None,  metavar='FILE')
    p.add_argument('--output',         default='output_transfer.mid', metavar='FILE')
    p.add_argument('--bpm',            type=float, default=120.0)
    p.add_argument('--threshold',      type=float, default=None,  metavar='F')
    p.add_argument('--threshold-pct',  type=float, default=99.0,  metavar='F',
                   help='Percentil para umbral adaptativo (default: 99.0)')
    p.add_argument('--adaptive-per-bar', action='store_true')
    p.set_defaults(func=cmd_transfer)

    # ── encode ────────────────────────────────────────────────────────────────
    p = sub.add_parser('encode',
        help='MIDI → vector de estilo latente (.json)')
    p.add_argument('--input',      required=True, metavar='FILE')
    p.add_argument('--model-dir',  required=True, metavar='DIR')
    p.add_argument('--output',     default='z_style.json', metavar='FILE')
    p.set_defaults(func=cmd_encode)

    # ── style-corpus ──────────────────────────────────────────────────────────
    p = sub.add_parser('style-corpus',
        help='Centroide de estilo de una carpeta de MIDIs')
    p.add_argument('--input-dir',  required=True, metavar='DIR')
    p.add_argument('--model-dir',  required=True, metavar='DIR')
    p.add_argument('--output',     default='z_corpus_centroid.json', metavar='FILE')
    p.set_defaults(func=cmd_style_corpus)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect',
        help='Diagnóstico del modelo y los datos')
    p.add_argument('--model-dir', default=None, metavar='DIR')
    p.add_argument('--data-dir',  default=None, metavar='DIR')
    p.add_argument('--what',      default='model',
                   help='"model", "npz", o "model npz"')
    p.add_argument('--npz-file',  default=None, metavar='STEM')
    p.add_argument('--no-roll',   action='store_true')
    p.set_defaults(func=cmd_inspect)

    # ── round-trip ────────────────────────────────────────────────────────────
    p = sub.add_parser('round-trip',
        help='MIDI → piano roll → MIDI (diagnóstico sin modelo)')
    p.add_argument('--input',       required=True, metavar='FILE')
    p.add_argument('--output',      default='round_trip_output.mid', metavar='FILE')
    p.add_argument('--resolution',  type=int, default=TICKS_PER_BAR_DEFAULT)
    p.add_argument('--window-bars', type=int, default=WINDOW_BARS_DEFAULT)
    p.add_argument('--bpm',         type=float, default=120.0)
    p.set_defaults(func=cmd_round_trip)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
