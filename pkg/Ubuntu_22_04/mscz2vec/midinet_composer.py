#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MIDINET COMPOSER  v1.0                                 ║
║         Generación de melodías condicionada en acordes (GAN bar-a-bar)      ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Generador convolucional + Discriminador condicional (cGAN)                ║
║    Entrada:  ruido z (100d) + acorde y (13d) + compás anterior prev_x       ║
║    Salida:   piano roll binario (1, 16, 128) — un compás de melodía         ║
║    Acorde:   12 clases cromáticas + 1 bit mayor/menor = 13 dimensiones      ║
║    Pérdida G: adversarial + feature matching discriminador + matching medio  ║
║    Pérdida D: BCE con label smoothing (0.9) en ejemplos reales              ║
║    Referencia: MidiNet (Yang et al., 2017) https://arxiv.org/abs/1703.10847 ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare    — MIDIs → piano rolls segmentados + vectores de acorde (.npz) ║
║    train      — Entrena el cGAN sobre los datos preparados                   ║
║    generate   — Genera una melodía bar-a-bar desde ruido + acordes           ║
║    inspect    — Diagnóstico del modelo y los datos                           ║
║    round-trip — MIDI → piano roll → MIDI (sin modelo, diagnóstico)          ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  # ── Preparar corpus ───────────────────────────────────────────────────── ║
║  python midinet_composer.py prepare                                          ║
║      --input-dir  midis/                                                     ║
║      --output-dir data/                                                      ║
║      --report                                                                ║
║                                                                              ║
║  # Limitar rango de pitch (alivia carga en GPU modesta):                    ║
║  python midinet_composer.py prepare                                          ║
║      --input-dir midis/ --output-dir data_small/                            ║
║      --pitch-lo 48 --pitch-hi 96                                            ║
║                                                                              ║
║  # ── Entrenar ──────────────────────────────────────────────────────────── ║
║  python midinet_composer.py train                                            ║
║      --data-dir data/                                                        ║
║      --model-dir model_midinet/                                              ║
║      --epochs 300 --batch-size 72 --lr 2e-4                                 ║
║                                                                              ║
║  # GPU modesta (menos z, menos canales):                                    ║
║  python midinet_composer.py train                                            ║
║      --data-dir data_small/ --model-dir model_small/                        ║
║      --epochs 200 --batch-size 32 --nz 64 --base-ch 32                     ║
║                                                                              ║
║  # ── Generar melodía ───────────────────────────────────────────────────── ║
║  # Con acordes inferidos automáticamente de un MIDI de referencia:          ║
║  python midinet_composer.py generate                                         ║
║      --model-dir model_midinet/                                              ║
║      --chord-ref referencia.mid                                              ║
║      --bars 8                                                                ║
║      --output melodia_generada.mid                                           ║
║                                                                              ║
║  # Con progresión de acordes explícita (notación: raíz:tipo,...):           ║
║  python midinet_composer.py generate                                         ║
║      --model-dir model_midinet/                                              ║
║      --chords "C:maj,F:maj,G:maj,C:maj"                                     ║
║      --bars 8 --bpm 120                                                      ║
║      --output resultado.mid                                                  ║
║      --seed 42                                                               ║
║                                                                              ║
║  # Temperatura alta = más variación armónica, temperatura baja = más fiel:  ║
║  python midinet_composer.py generate                                         ║
║      --model-dir model_midinet/                                              ║
║      --chords "Am:min,F:maj,C:maj,G:maj"                                    ║
║      --bars 16 --temperature 1.4 --output variado.mid                       ║
║                                                                              ║
║  # ── Diagnóstico ───────────────────────────────────────────────────────── ║
║  python midinet_composer.py round-trip --input midis/cancion.mid            ║
║  python midinet_composer.py inspect --model-dir model_midinet/              ║
║  python midinet_composer.py inspect --data-dir data/ --what data            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

STEPS_PER_BAR   = 16          # subdivisiones por compás (semicorcheas)
PITCH_CLASSES   = 128         # rango MIDI completo
CHORD_DIM       = 13          # 12 clases cromáticas + 1 bit mayor/menor
NZ_DEFAULT      = 100         # dimensión del ruido latente
BASE_CH_DEFAULT = 64          # canales base del generador

# Progresiones estándar para pruebas rápidas
CHORD_PRESETS = {
    'I-IV-V-I':   'C:maj,F:maj,G:maj,C:maj',
    'I-V-vi-IV':  'C:maj,G:maj,A:min,F:maj',
    'ii-V-I':     'D:min,G:maj,C:maj,C:maj',
    'andaluza':   'A:min,G:maj,F:maj,E:maj',
    '12bar':      'C:maj,C:maj,C:maj,C:maj,F:maj,F:maj,C:maj,C:maj,G:maj,F:maj,C:maj,G:maj',
}

CHORD_ROOTS = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
               'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS MIDI / PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi(path, fatal=True):
    import mido
    try:
        return mido.MidiFile(str(path))
    except Exception as e:
        if fatal:
            sys.exit(f"ERROR: no se pudo abrir {path}: {e}")
        raise


def _midi_to_pianoroll(mid, pitch_lo=0, pitch_hi=128, steps_per_bar=STEPS_PER_BAR):
    """
    Convierte un MidiFile a una lista de compases en formato piano roll.
    Retorna: list de arrays (1, steps_per_bar, n_pitch)  —  n_pitch = pitch_hi - pitch_lo
    """
    tpb       = mid.ticks_per_beat or 480
    tempo_us  = 500_000   # 120 BPM por defecto
    ticks_per_step = tpb // (steps_per_bar // 4)   # asume 4/4
    n_pitch   = pitch_hi - pitch_lo

    # Acumulamos ticks absolutos
    events = []  # (abs_tick, type, channel, note, velocity)
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type in ('note_on', 'note_off'):
                vel = msg.velocity if msg.type == 'note_on' else 0
                events.append((abs_t, msg.type, msg.channel, msg.note, vel))

    if not events:
        return []

    events.sort(key=lambda x: x[0])
    total_ticks = events[-1][0] + 1
    ticks_per_bar = tpb * 4
    n_bars = max(1, total_ticks // ticks_per_bar)

    roll = np.zeros((n_bars * steps_per_bar, PITCH_CLASSES), dtype=np.float32)
    pending = {}

    for abs_t, etype, ch, note, vel in events:
        if ch == 9:
            continue   # ignorar percusión
        if etype == 'note_on' and vel > 0:
            pending[(ch, note)] = abs_t
        else:
            if (ch, note) in pending:
                onset = pending.pop((ch, note))
                step_on  = onset    // ticks_per_step
                step_off = abs_t    // ticks_per_step
                step_on  = min(step_on,  roll.shape[0] - 1)
                step_off = min(step_off, roll.shape[0])
                if step_off > step_on:
                    roll[step_on:step_off, note] = 1.0

    # Recortar rango de pitch
    roll = roll[:, pitch_lo:pitch_hi]

    # Dividir en compases de steps_per_bar pasos
    bars = []
    for b in range(n_bars):
        sl = roll[b * steps_per_bar:(b + 1) * steps_per_bar, :]
        if sl.shape[0] == steps_per_bar and sl.sum() > 0:
            bars.append(sl.reshape(1, steps_per_bar, n_pitch))

    return bars


def _pianoroll_bar_to_midi(bar, bpm=120.0, pitch_lo=0,
                            program=0, velocity=80,
                            steps_per_bar=STEPS_PER_BAR):
    """
    Convierte un array (steps_per_bar, n_pitch) a una lista de mensajes mido.
    Retorna (messages, ticks_per_beat, ticks_per_step).
    """
    import mido
    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat // (steps_per_bar // 4)

    # Binarizar: nota activa = max de cada paso (sólo la nota más alta)
    # Puede cambiarse por un umbral si se quieren acordes.
    n_steps, n_pitch = bar.shape
    active = []
    for s in range(n_steps):
        row = bar[s]
        if row.max() > 0.3:
            note = int(np.argmax(row)) + pitch_lo
            active.append(note)
        else:
            active.append(None)

    msgs = []
    prev_note = None
    for s, note in enumerate(active):
        if note != prev_note:
            if prev_note is not None:
                msgs.append(mido.Message('note_off', note=prev_note, velocity=0,
                                         time=0))
            if note is not None:
                msgs.append(mido.Message('note_on', note=note, velocity=velocity,
                                         time=0))
            prev_note = note
        if s < n_steps - 1:
            msgs.append(mido.Message('note_on', note=note if note else 60,
                                     velocity=0, time=ticks_per_step))
            # placeholder de tiempo — será reabsorbido en _bars_to_midi

    # Cerrar última nota
    if prev_note is not None:
        msgs.append(mido.Message('note_off', note=prev_note, velocity=0, time=0))

    return msgs, ticks_per_beat, ticks_per_step


def _bars_to_midi(bars, bpm=120.0, pitch_lo=0, program=0, velocity=80,
                  steps_per_bar=STEPS_PER_BAR):
    """
    Convierte una lista de arrays (steps_per_bar, n_pitch) a un MidiFile.
    """
    import mido
    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat // (steps_per_bar // 4)
    tempo_us = mido.bpm2tempo(bpm)

    mid  = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.Message('program_change', program=program, channel=0, time=0))

    for bar in bars:
        arr = bar.reshape(steps_per_bar, -1)
        prev_note = None
        elapsed   = 0
        for s in range(steps_per_bar):
            row  = arr[s]
            note = int(np.argmax(row)) + pitch_lo if row.max() > 0.3 else None

            if note != prev_note:
                if prev_note is not None:
                    track.append(mido.Message('note_off', note=prev_note,
                                              velocity=0, time=elapsed))
                    elapsed = 0
                if note is not None:
                    track.append(mido.Message('note_on', note=note,
                                              velocity=velocity,
                                              channel=0, time=elapsed))
                    elapsed = 0
                prev_note = note

            elapsed += ticks_per_step

        if prev_note is not None:
            track.append(mido.Message('note_off', note=prev_note,
                                      velocity=0, time=elapsed))
            elapsed = 0

    return mid


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODING DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

def _chord_vector(root_pc: int, is_minor: bool) -> np.ndarray:
    """
    Construye el vector de 13 dimensiones:
      [0..11] one-hot de la clase cromática (root_pc)
      [12]    0 = mayor, 1 = menor
    """
    v = np.zeros(CHORD_DIM, dtype=np.float32)
    v[root_pc % 12] = 1.0
    v[12] = 1.0 if is_minor else 0.0
    return v


def _parse_chord_string(chord_str: str) -> list:
    """
    Parsea "C:maj,F:maj,G:maj,C:maj" → lista de vectores (13,).
    Acepta notación 'root:tipo' donde tipo ∈ {maj, min, major, minor, m, M}.
    """
    vectors = []
    for token in chord_str.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            root_s, quality = token.split(':', 1)
        else:
            root_s, quality = token, 'maj'

        root_s = root_s.strip().capitalize()
        # capitalizar sostenidos/bemoles correctamente
        if len(root_s) == 2 and root_s[1] in ('#', 'b'):
            root_s = root_s[0].upper() + root_s[1]

        if root_s not in CHORD_ROOTS:
            sys.exit(f"ERROR: raíz de acorde desconocida: '{root_s}' en '{token}'")

        root_pc = CHORD_ROOTS[root_s]
        is_min  = quality.lower() in ('min', 'minor', 'm')
        vectors.append(_chord_vector(root_pc, is_min))

    return vectors


def _extract_chords_from_midi(mid, n_bars: int) -> list:
    """
    Estima acordes buscando la nota más grave activa en cada compás
    (proxy sencillo: root = lowest note mod 12).
    Devuelve lista de n_bars vectores (13,).
    """
    tpb = mid.ticks_per_beat or 480
    ticks_per_bar = tpb * 4

    # Acumular notas con sus ticks absolutos
    note_events = []
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
                note_events.append((abs_t, msg.note))

    vectors = []
    for b in range(n_bars):
        bar_start = b * ticks_per_bar
        bar_end   = bar_start + ticks_per_bar
        notes_in  = [n for t, n in note_events if bar_start <= t < bar_end]
        if notes_in:
            root_pc = min(notes_in) % 12
            # Heurística mayor/menor: si hay una 3ª menor (3 semitonos encima) → menor
            pcs     = set(n % 12 for n in notes_in)
            is_min  = ((root_pc + 3) % 12) in pcs and ((root_pc + 4) % 12) not in pcs
        else:
            root_pc, is_min = 0, False   # C mayor por defecto
        vectors.append(_chord_vector(root_pc, is_min))

    return vectors


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _build_models(n_pitch: int, nz: int = NZ_DEFAULT, base_ch: int = BASE_CH_DEFAULT):
    """
    Construye y devuelve (Generator, Discriminator).
    Import de torch diferido para no penalizar subcomandos sin GPU.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    y_dim = CHORD_DIM

    class _BN1d(nn.Module):
        """BatchNorm1d instanciado una sola vez y registrado como módulo."""
        def __init__(self, size):
            super().__init__()
            self.bn = nn.BatchNorm1d(size, eps=1e-5, momentum=0.9)
        def forward(self, x):
            return self.bn(x)

    class _BN2d(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.bn = nn.BatchNorm2d(size, eps=1e-5, momentum=0.9)
        def forward(self, x):
            return self.bn(x)

    class Generator(nn.Module):
        """
        Genera un compás (1, steps_per_bar, n_pitch) condicionado en:
          z       — ruido latente (nz,)
          prev_x  — compás anterior (1, steps_per_bar, n_pitch)
          y       — vector de acorde (y_dim,)

        Fiel al paper MidiNet Model-3:
          • Rama prev_x: 4 Conv2d con lrelu → skip connections a cada capa del decoder
          • Rama z+y: dos Linear → reshape → 4 ConvTranspose2d
          • En cada nivel: conv_cond_concat(yb) + conv_prev_concat(prev_feat)
        """
        def __init__(self):
            super().__init__()
            # ── Encoder del compás previo ──────────────────────────────────
            self.prev_enc0 = nn.Conv2d(1,  16, kernel_size=(1, n_pitch), stride=(1, 2))
            self.prev_enc1 = nn.Conv2d(16, 16, kernel_size=(2, 1),       stride=(2, 2))
            self.prev_enc2 = nn.Conv2d(16, 16, kernel_size=(2, 1),       stride=(2, 2))
            self.prev_enc3 = nn.Conv2d(16, 16, kernel_size=(2, 1),       stride=(2, 2))

            self.bn_pe0 = _BN2d(16)
            self.bn_pe1 = _BN2d(16)
            self.bn_pe2 = _BN2d(16)
            self.bn_pe3 = _BN2d(16)

            # ── Proyección z+y ────────────────────────────────────────────
            self.lin1   = nn.Linear(nz + y_dim, 1024)
            self.bn_l1  = _BN1d(1024)
            self.lin2   = nn.Linear(1024 + y_dim, base_ch * 2 * 2 * 1)
            self.bn_l2  = _BN1d(base_ch * 2 * 2 * 1)

            # Canales en la entrada de cada ConvTranspose2d:
            #   base_ch*2 (de lin2) + y_dim (cond) + 16 (prev skip) = base_ch*2 + y_dim + 16
            in_ch = base_ch * 2 + y_dim + 16

            self.dec1 = nn.ConvTranspose2d(in_ch, n_pitch,   kernel_size=(2, 1), stride=(2, 2))
            self.dec2 = nn.ConvTranspose2d(n_pitch + y_dim + 16, n_pitch, kernel_size=(2, 1), stride=(2, 2))
            self.dec3 = nn.ConvTranspose2d(n_pitch + y_dim + 16, n_pitch, kernel_size=(2, 1), stride=(2, 2))
            self.dec4 = nn.ConvTranspose2d(n_pitch + y_dim + 16, 1,       kernel_size=(1, n_pitch), stride=(1, 2))

            self.bn_d1 = _BN2d(n_pitch)
            self.bn_d2 = _BN2d(n_pitch)
            self.bn_d3 = _BN2d(n_pitch)

        def forward(self, z, prev_x, y):
            B = z.size(0)

            # ── Encoder prev_x ────────────────────────────────────────────
            p0 = F.leaky_relu(self.bn_pe0(self.prev_enc0(prev_x)), 0.2)
            p1 = F.leaky_relu(self.bn_pe1(self.prev_enc1(p0)),     0.2)
            p2 = F.leaky_relu(self.bn_pe2(self.prev_enc2(p1)),     0.2)
            p3 = F.leaky_relu(self.bn_pe3(self.prev_enc3(p2)),     0.2)

            yb = y.view(B, y_dim, 1, 1)

            # ── Proyección z+y ────────────────────────────────────────────
            h  = torch.cat([z, y], dim=1)
            h  = F.relu(self.bn_l1(self.lin1(h)))
            h  = torch.cat([h, y], dim=1)
            h  = F.relu(self.bn_l2(self.lin2(h)))
            h  = h.view(B, base_ch * 2, 2, 1)

            # ── Decoder con skip connections ──────────────────────────────
            h = _cond_cat(h, yb)
            h = _prev_cat(h, p3)
            h = F.relu(self.bn_d1(self.dec1(h)))

            h = _cond_cat(h, yb)
            h = _prev_cat(h, p2)
            h = F.relu(self.bn_d2(self.dec2(h)))

            h = _cond_cat(h, yb)
            h = _prev_cat(h, p1)
            h = F.relu(self.bn_d3(self.dec3(h)))

            h = _cond_cat(h, yb)
            h = _prev_cat(h, p0)
            out = torch.sigmoid(self.dec4(h))   # (B, 1, steps_per_bar, n_pitch)

            return out

    class Discriminator(nn.Module):
        """
        Discrimina compases reales de falsos condicionado en acorde y.
        Devuelve (prob, logit, feature_map) — feature_map para feature matching.
        """
        def __init__(self):
            super().__init__()
            # input: x (1, 16, n_pitch) + yb broadcast → (1 + y_dim, 16, n_pitch)
            self.conv0 = nn.Conv2d(1 + y_dim,  1 + y_dim,  kernel_size=(2, n_pitch), stride=(2, 2))
            # → (1 + y_dim, 8, 1) → concat y → (1 + 2*y_dim, 8, 1)
            self.conv1 = nn.Conv2d(1 + 2*y_dim, base_ch + y_dim, kernel_size=(4, 1), stride=(2, 2))
            self.bn_c1 = _BN2d(base_ch + y_dim)
            # flatten + y
            flat_dim   = (base_ch + y_dim) * 3
            self.lin1  = nn.Linear(flat_dim + y_dim, 1024)
            self.bn_l1 = _BN1d(1024)
            self.lin2  = nn.Linear(1024 + y_dim, 1)

        def forward(self, x, y):
            B    = x.size(0)
            yb   = y.view(B, y_dim, 1, 1)

            h    = _cond_cat(x, yb)          # (B, 1+y_dim, 16, n_pitch)
            fm   = self.conv0(h)             # feature map intermedio
            h    = F.leaky_relu(fm, 0.2)
            h    = _cond_cat(h, yb)
            h    = F.leaky_relu(self.bn_c1(self.conv1(h)), 0.2)
            h    = h.view(B, -1)
            h    = torch.cat([h, y], dim=1)
            h    = F.leaky_relu(self.bn_l1(self.lin1(h)), 0.2)
            h    = torch.cat([h, y], dim=1)
            logit = self.lin2(h)
            prob  = torch.sigmoid(logit)
            return prob, logit, fm

    return Generator(), Discriminator()


# ── Helpers de concatenación condicional ─────────────────────────────────────

def _cond_cat(x, y):
    """Expande y hasta la resolución espacial de x y concatena en dim=1."""
    y_exp = y.expand(x.size(0), y.size(1), x.size(2), x.size(3))
    return __import__('torch').cat([x, y_exp], dim=1)


def _prev_cat(x, prev):
    """Concatena prev (skip connection) en dim=1 si las resoluciones coinciden."""
    if x.shape[2:] == prev.shape[2:]:
        return __import__('torch').cat([x, prev], dim=1)
    # Mismatch de resolución: interpolar prev
    import torch.nn.functional as F
    prev_r = F.interpolate(prev, size=x.shape[2:], mode='nearest')
    return __import__('torch').cat([x, prev_r], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class _MidiNetDataset:
    """
    Carga datos preparados desde un directorio con ficheros .npz.
    Cada .npz contiene:
      x       (N_bars, 1, STEPS_PER_BAR, n_pitch)  — compases
      prev_x  (N_bars, 1, STEPS_PER_BAR, n_pitch)  — compás anterior (cero-padded)
      y       (N_bars, CHORD_DIM)                   — vectores de acorde
    """
    def __init__(self, data_dir):
        import torch
        from torch.utils.data import TensorDataset
        xs, prevs, ys = [], [], []
        npz_files = sorted(Path(data_dir).glob('*.npz'))
        if not npz_files:
            sys.exit(f"ERROR: no se encontraron ficheros .npz en {data_dir}")

        for f in npz_files:
            d = np.load(f)
            xs.append(d['x'])
            prevs.append(d['prev_x'])
            ys.append(d['y'])

        X    = torch.tensor(np.concatenate(xs,    axis=0), dtype=torch.float32)
        Prev = torch.tensor(np.concatenate(prevs, axis=0), dtype=torch.float32)
        Y    = torch.tensor(np.concatenate(ys,    axis=0), dtype=torch.float32)
        self.dataset = TensorDataset(X, Prev, Y)
        self.n_pitch = X.shape[-1]

    def get_loader(self, batch_size, shuffle=True):
        from torch.utils.data import DataLoader
        return DataLoader(self.dataset, batch_size=batch_size,
                          shuffle=shuffle, num_workers=2, pin_memory=True,
                          drop_last=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """MIDI corpus → piano rolls + acordes segmentados (.npz)"""
    import mido

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitch_lo = args.pitch_lo
    pitch_hi = args.pitch_hi
    n_pitch  = pitch_hi - pitch_lo

    midi_files = sorted(input_dir.rglob('*.mid')) + sorted(input_dir.rglob('*.midi'))
    if not midi_files:
        sys.exit(f"ERROR: no se encontraron MIDIs en {input_dir}")

    print(f"  Encontrados {len(midi_files)} MIDIs  |  "
          f"pitch [{pitch_lo}, {pitch_hi})  →  {n_pitch} clases")

    saved = skipped = corrupt = total_bars = 0
    for midi_path in midi_files:
        try:
            mid = _load_midi(midi_path, fatal=False)
        except Exception:
            corrupt += 1
            continue

        try:
            bars = _midi_to_pianoroll(mid, pitch_lo=pitch_lo, pitch_hi=pitch_hi,
                                      steps_per_bar=STEPS_PER_BAR)
        except Exception:
            corrupt += 1
            continue

        if len(bars) < 2:
            skipped += 1
            continue

        # Extraer acordes para cada compás
        n_bars = len(bars)
        chord_vecs = _extract_chords_from_midi(mid, n_bars)

        # Construir tensores: x y prev_x (compás anterior, zero para el primero)
        x_arr     = np.stack(bars, axis=0).astype(np.float32)       # (N, 1, 16, n_pitch)
        prev_arr  = np.zeros_like(x_arr)
        prev_arr[1:] = x_arr[:-1]
        y_arr     = np.stack(chord_vecs, axis=0).astype(np.float32) # (N, 13)

        stem = midi_path.stem.replace(' ', '_')
        out  = output_dir / f"{stem}.npz"
        np.savez_compressed(out, x=x_arr, prev_x=prev_arr, y=y_arr)
        saved     += 1
        total_bars += n_bars

    print(f"  Guardados: {saved}  |  Omitidos (poco contenido): {skipped}  |  "
          f"Corruptos/error: {corrupt}  |  Total compases: {total_bars}")

    if args.report:
        print(f"\n  Informe detallado:")
        for f in sorted(output_dir.glob('*.npz'))[:10]:
            d = np.load(f)
            print(f"    {f.name:40s}  {d['x'].shape[0]:>4d} compases")
        if saved > 10:
            print(f"    ... ({saved - 10} más)")


def cmd_train(args):
    """Entrena el cGAN sobre los datos preparados."""
    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    device    = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Dispositivo : {device}")
    print(f"  Datos       : {args.data_dir}")
    print(f"  Modelo      : {model_dir}")

    # ── Datos ────────────────────────────────────────────────────────────────
    ds     = _MidiNetDataset(args.data_dir)
    loader = ds.get_loader(args.batch_size)
    n_pitch = ds.n_pitch
    print(f"  Muestras    : {len(ds.dataset)}  |  n_pitch = {n_pitch}")

    # Inferir pitch_lo desde el argumento opcional o deducirlo del corpus
    if getattr(args, 'pitch_lo', None) is not None:
        _pitch_lo = args.pitch_lo
    else:
        # n_pitch = pitch_hi - pitch_lo; asumimos pitch_hi=128
        _pitch_lo = 128 - n_pitch if n_pitch < 128 else 0
    _pitch_hi = _pitch_lo + n_pitch
    print(f"  pitch_lo    : {_pitch_lo}  pitch_hi: {_pitch_hi}  (inferido del corpus)")

    # ── Modelos ───────────────────────────────────────────────────────────────
    netG, netD = _build_models(n_pitch, nz=args.nz, base_ch=args.base_ch)
    start_epoch = 0

    ckpt_g = model_dir / 'netG_latest.pth'
    ckpt_d = model_dir / 'netD_latest.pth'
    meta_f = model_dir / 'meta.json'

    if args.resume and ckpt_g.exists() and ckpt_d.exists():
        netG.load_state_dict(torch.load(ckpt_g, map_location='cpu'))
        netD.load_state_dict(torch.load(ckpt_d, map_location='cpu'))
        if meta_f.exists():
            meta = json.loads(meta_f.read_text())
            start_epoch = meta.get('epoch', 0) + 1
        print(f"  Reanudando desde época {start_epoch}")

    netG = netG.to(device)
    netD = netD.to(device)
    netG.train(); netD.train()

    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def _bce_logits(logits, target_val):
        """BCE con logits; target_val es un escalar."""
        t = torch.full_like(logits, target_val)
        return F.binary_cross_entropy_with_logits(logits, t)

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    best_lossG = float('inf')
    history = {'lossD': [], 'lossG': []}

    for epoch in range(start_epoch, start_epoch + args.epochs):
        sum_lD = sum_lG = 0.0
        n_batches = 0

        for x_real, prev_x, y in loader:
            x_real = x_real.to(device)
            prev_x = prev_x.to(device)
            y      = y.to(device)
            B      = x_real.size(0)

            # ── Entrenar D ────────────────────────────────────────────────
            netD.zero_grad()
            _, d_real_logit, fm_real = netD(x_real, y)
            loss_d_real = _bce_logits(d_real_logit, 0.9)   # label smoothing

            z = torch.randn(B, args.nz, device=device)
            with torch.no_grad():
                fake_d = netG(z, prev_x, y)
            _, d_fake_logit, _ = netD(fake_d, y)
            loss_d_fake = _bce_logits(d_fake_logit, 0.0)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # Guardar fm_real sin grafo para feature matching en G
            fm_real_det = fm_real.detach()

            # ── Entrenar G (primera pasada) ───────────────────────────────
            netG.zero_grad()
            fake_g = netG(z, prev_x, y)
            _, g_logit, fm_fake = netD(fake_g, y)
            loss_g_adv = _bce_logits(g_logit, 1.0)
            fm_g_loss  = F.mse_loss(fm_fake.mean(0), fm_real_det.mean(0)) * 0.1
            img_g_loss = F.mse_loss(fake_g.mean(0), x_real.mean(0)) * 0.01
            loss_g = loss_g_adv + fm_g_loss + img_g_loss
            loss_g.backward()
            optG.step()

            # ── Entrenar G (segunda pasada, estabilización) ───────────────
            netG.zero_grad()
            z2    = torch.randn(B, args.nz, device=device)
            fake2 = netG(z2, prev_x, y)
            _, g_logit2, fm_fake2 = netD(fake2, y)
            loss_g2 = (_bce_logits(g_logit2, 1.0)
                       + F.mse_loss(fm_fake2.mean(0), fm_real_det.mean(0)) * 0.1
                       + F.mse_loss(fake2.mean(0), x_real.mean(0)) * 0.01)
            loss_g2.backward()
            optG.step()

            sum_lD += loss_d.item()
            sum_lG += loss_g2.item()
            n_batches += 1

        avg_lD = sum_lD / max(n_batches, 1)
        avg_lG = sum_lG / max(n_batches, 1)
        history['lossD'].append(avg_lD)
        history['lossG'].append(avg_lG)

        if (epoch - start_epoch) % max(1, args.epochs // 20) == 0 or epoch == start_epoch:
            print(f"  Época {epoch:4d}/{start_epoch + args.epochs - 1}  "
                  f"lossD={avg_lD:.4f}  lossG={avg_lG:.4f}")

        # Checkpoint latest al final de cada época (permite reanudar tras Ctrl+C)
        torch.save(netG.state_dict(), ckpt_g)
        torch.save(netD.state_dict(), ckpt_d)
        meta_epoch = {
            'epoch':    epoch,
            'n_pitch':  n_pitch,
            'nz':       args.nz,
            'base_ch':  args.base_ch,
            'pitch_lo': _pitch_lo,
            'pitch_hi': _pitch_hi,
        }
        meta_f.write_text(json.dumps(meta_epoch, indent=2))
        np.save(model_dir / 'history.npy', history)

        # Checkpoint periódico numerado
        if (epoch + 1) % args.save_every == 0:
            torch.save(netG.state_dict(), model_dir / f'netG_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), model_dir / f'netD_epoch_{epoch}.pth')
            print(f"  [ckpt] guardado en época {epoch}")

    # ── Guardar modelo final ──────────────────────────────────────────────────
    torch.save(netG.state_dict(), ckpt_g)
    torch.save(netD.state_dict(), ckpt_d)

    meta = {
        'epoch':   start_epoch + args.epochs - 1,
        'n_pitch': n_pitch,
        'nz':      args.nz,
        'base_ch': args.base_ch,
        'pitch_lo': _pitch_lo,
        'pitch_hi': _pitch_hi,
    }
    meta_f.write_text(json.dumps(meta, indent=2))
    np.save(model_dir / 'history.npy', history)
    print(f"\n  Modelo guardado en {model_dir}")
    print(f"  Mejor lossG: {min(history['lossG']):.4f}")


def cmd_generate(args):
    """Genera una melodía bar-a-bar con el modelo entrenado."""
    import torch

    model_dir = Path(args.model_dir)
    meta_f    = model_dir / 'meta.json'
    if not meta_f.exists():
        sys.exit(f"ERROR: {meta_f} no encontrado — ¿se ha entrenado el modelo?")

    meta    = json.loads(meta_f.read_text())
    n_pitch = meta['n_pitch']
    nz      = meta.get('nz', NZ_DEFAULT)
    base_ch = meta.get('base_ch', BASE_CH_DEFAULT)
    pitch_lo = meta.get('pitch_lo', 0)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    netG, _ = _build_models(n_pitch, nz=nz, base_ch=base_ch)
    ckpt    = model_dir / 'netG_latest.pth'
    if not ckpt.exists():
        sys.exit(f"ERROR: {ckpt} no encontrado")
    netG.load_state_dict(torch.load(ckpt, map_location='cpu'))
    netG = netG.to(device)
    netG.eval()

    # ── Acordes ──────────────────────────────────────────────────────────────
    if args.chords:
        chord_token = args.chords
        if chord_token in CHORD_PRESETS:
            chord_token = CHORD_PRESETS[chord_token]
        chord_vecs = _parse_chord_string(chord_token)
    elif args.chord_ref:
        mid_ref    = _load_midi(args.chord_ref)
        chord_vecs = _extract_chords_from_midi(mid_ref, args.bars)
    else:
        # Fallback: I-IV-V-I ciclado
        chord_vecs = _parse_chord_string(CHORD_PRESETS['I-IV-V-I'])

    # Ciclar o recortar según el número de compases pedido
    n_bars = args.bars
    chord_vecs = [chord_vecs[i % len(chord_vecs)] for i in range(n_bars)]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"  Generando {n_bars} compases  |  temperatura={args.temperature}  "
          f"|  pitch_lo={pitch_lo}")

    # ── Generación autoregresiva ──────────────────────────────────────────────
    generated_bars = []
    prev = torch.zeros(1, 1, STEPS_PER_BAR, n_pitch, device=device)

    with torch.no_grad():
        for b, y_np in enumerate(chord_vecs):
            z = torch.randn(1, nz, device=device) * args.temperature
            y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)
            bar = netG(z, prev, y)          # (1, 1, steps_per_bar, n_pitch)
            generated_bars.append(bar.squeeze(0).cpu().numpy())   # (1, 16, n_pitch)
            prev = bar

    # ── Piano roll → MIDI ────────────────────────────────────────────────────
    arrays = [b.reshape(STEPS_PER_BAR, n_pitch) for b in generated_bars]
    mid    = _bars_to_midi(arrays, bpm=args.bpm, pitch_lo=pitch_lo,
                           program=args.program, velocity=args.velocity)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out))
    print(f"  MIDI guardado: {out}")

    if args.verbose:
        # Estadísticas de densidad por compás
        print("\n  Densidad por compás (notas activas / steps):")
        for i, arr in enumerate(arrays):
            dens = (arr.max(axis=1) > 0.3).sum()
            bar = '█' * dens + '░' * (STEPS_PER_BAR - dens)
            print(f"    [{i:3d}] {bar}  {dens:2d}/{STEPS_PER_BAR}")


def cmd_round_trip(args):
    """MIDI → piano roll → MIDI, sin modelo (diagnóstico de representación)."""
    import mido

    mid_in   = _load_midi(args.input)
    pitch_lo = args.pitch_lo
    pitch_hi = args.pitch_hi

    print(f"  Leyendo {args.input}")
    print(f"  Rango pitch: [{pitch_lo}, {pitch_hi})  →  {pitch_hi - pitch_lo} clases")

    bars = _midi_to_pianoroll(mid_in, pitch_lo=pitch_lo, pitch_hi=pitch_hi)
    if not bars:
        sys.exit("ERROR: no se obtuvieron compases — ¿hay notas en el MIDI?")

    print(f"  Compases extraídos: {len(bars)}")

    arrays  = [b.reshape(STEPS_PER_BAR, pitch_hi - pitch_lo) for b in bars]
    mid_out = _bars_to_midi(arrays, bpm=args.bpm, pitch_lo=pitch_lo)
    out     = Path(args.output)
    mid_out.save(str(out))
    print(f"  MIDI reconstruido: {out}")

    # Estadísticas
    densities = [(a.max(axis=1) > 0.3).sum() for a in arrays]
    print(f"  Densidad media: {np.mean(densities):.1f} / {STEPS_PER_BAR} pasos")
    print(f"  Compases vacíos: {sum(1 for d in densities if d == 0)}")


def cmd_inspect(args):
    """Diagnóstico del modelo entrenado y/o de los datos preparados."""
    if args.what in ('model', 'all') and args.model_dir:
        import torch
        model_dir = Path(args.model_dir)
        meta_f    = model_dir / 'meta.json'
        ckpt_g    = model_dir / 'netG_latest.pth'

        if not meta_f.exists():
            print(f"  [modelo] {model_dir} — sin meta.json (modelo no entrenado aún)")
        else:
            meta = json.loads(meta_f.read_text())
            print(f"  ╔═══ Modelo: {model_dir} ════")
            for k, v in meta.items():
                print(f"  ║  {k:15s}: {v}")

            if ckpt_g.exists():
                sd = torch.load(ckpt_g, map_location='cpu')
                n_params = sum(v.numel() for v in sd.values())
                print(f"  ║  parámetros G : {n_params:,}")
                print(f"  ╚{'═' * 40}")

            hist_f = model_dir / 'history.npy'
            if hist_f.exists():
                h = np.load(str(hist_f), allow_pickle=True).item()
                print(f"\n  Historial de pérdidas ({len(h['lossD'])} épocas):")
                print(f"    lossD final : {h['lossD'][-1]:.4f}  "
                      f"(mejor: {min(h['lossD']):.4f})")
                print(f"    lossG final : {h['lossG'][-1]:.4f}  "
                      f"(mejor: {min(h['lossG']):.4f})")

    if args.what in ('data', 'all') and args.data_dir:
        data_dir  = Path(args.data_dir)
        npz_files = sorted(data_dir.glob('*.npz'))
        print(f"\n  ╔═══ Datos: {data_dir} ════")
        print(f"  ║  Ficheros .npz: {len(npz_files)}")
        total_bars = 0
        n_pitch_set = set()
        for f in npz_files:
            d = np.load(f)
            total_bars  += d['x'].shape[0]
            n_pitch_set.add(d['x'].shape[-1])
        print(f"  ║  Total compases : {total_bars}")
        print(f"  ║  n_pitch        : {n_pitch_set}")
        print(f"  ╚{'═' * 40}")

        if args.verbose and npz_files:
            print(f"\n  Muestra de los primeros 5 ficheros:")
            for f in npz_files[:5]:
                d = np.load(f)
                x = d['x']
                density = (x.reshape(x.shape[0], -1).max(axis=1) > 0.3).mean()
                print(f"    {f.name:45s}  {x.shape[0]:4d} bars  "
                      f"densidad={density:.2f}")

    if args.what == 'presets':
        print("  Progresiones de acordes predefinidas (--chords <nombre>):")
        for name, prog in CHORD_PRESETS.items():
            print(f"    {name:15s}: {prog}")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='midinet_composer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            MIDINET COMPOSER v1.0
            Generación de melodías condicionada en acordes mediante un cGAN bar-a-bar.

            Flujo típico:
              1. prepare   — convertir corpus de MIDIs a piano rolls (.npz)
              2. train     — entrenar generador y discriminador
              3. generate  — generar melodías con acordes especificados
        """),
    )

    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('prepare',
        help='MIDIs → piano rolls + acordes (.npz)')
    p.add_argument('--input-dir',  required=True,  metavar='DIR',
                   help='Carpeta con ficheros .mid / .midi')
    p.add_argument('--output-dir', required=True,  metavar='DIR',
                   help='Carpeta de salida para los .npz')
    p.add_argument('--pitch-lo',   type=int, default=0,   metavar='N',
                   help='MIDI pitch mínimo incluido (default: 0)')
    p.add_argument('--pitch-hi',   type=int, default=PITCH_CLASSES, metavar='N',
                   help=f'MIDI pitch máximo excluido (default: {PITCH_CLASSES})')
    p.add_argument('--report',     action='store_true',
                   help='Mostrar muestra de los ficheros generados')
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train',
        help='Entrena el cGAN (Generador + Discriminador)')
    p.add_argument('--data-dir',   required=True, metavar='DIR',
                   help='Carpeta con .npz generados por prepare')
    p.add_argument('--model-dir',  required=True, metavar='DIR',
                   help='Carpeta donde guardar el modelo')
    p.add_argument('--epochs',     type=int,   default=300,
                   help='Épocas de entrenamiento (default: 300)')
    p.add_argument('--batch-size', type=int,   default=72,
                   help='Tamaño de batch (default: 72)')
    p.add_argument('--lr',         type=float, default=2e-4,
                   help='Learning rate Adam (default: 2e-4)')
    p.add_argument('--nz',         type=int,   default=NZ_DEFAULT,
                   help=f'Dimensión del ruido latente (default: {NZ_DEFAULT})')
    p.add_argument('--base-ch',    type=int,   default=BASE_CH_DEFAULT,
                   help=f'Canales base del generador (default: {BASE_CH_DEFAULT})')
    p.add_argument('--save-every', type=int,   default=50,  metavar='N',
                   help='Guardar checkpoint cada N épocas (default: 50)')
    p.add_argument('--pitch-lo',   type=int,   default=None, metavar='N',
                   help='Pitch mínimo usado en prepare (se infiere si no se indica)')
    p.add_argument('--resume',     action='store_true',
                   help='Reanudar desde el último checkpoint')
    p.add_argument('--cpu',        action='store_true',
                   help='Forzar CPU aunque haya GPU disponible')
    p.set_defaults(func=cmd_train)

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser('generate',
        help='Genera una melodía bar-a-bar con el modelo entrenado')
    p.add_argument('--model-dir',  required=True, metavar='DIR',
                   help='Carpeta del modelo entrenado')
    p.add_argument('--bars',       type=int,   default=8,
                   help='Número de compases a generar (default: 8)')
    p.add_argument('--chords',     default=None, metavar='PROG',
                   help='Progresión de acordes: "C:maj,F:maj,G:maj,C:maj"  '
                        'o nombre de preset: I-IV-V-I | I-V-vi-IV | ii-V-I | '
                        'andaluza | 12bar')
    p.add_argument('--chord-ref',  default=None, metavar='FILE',
                   help='MIDI de referencia del que extraer la progresión de acordes')
    p.add_argument('--bpm',        type=float, default=120.0,
                   help='Tempo BPM del MIDI de salida (default: 120)')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Temperatura del muestreo (default: 1.0; >1 = más variación)')
    p.add_argument('--program',    type=int,   default=0,
                   help='Programa MIDI General (instrumento, 0-127; default: 0 = piano)')
    p.add_argument('--velocity',   type=int,   default=80,
                   help='Velocidad MIDI de las notas (default: 80)')
    p.add_argument('--output',     default='midinet_out.mid', metavar='FILE',
                   help='Fichero MIDI de salida (default: midinet_out.mid)')
    p.add_argument('--seed',       type=int,   default=None,
                   help='Semilla aleatoria para reproducibilidad')
    p.add_argument('--cpu',        action='store_true',
                   help='Forzar CPU aunque haya GPU disponible')
    p.add_argument('--verbose',    action='store_true',
                   help='Mostrar densidad de notas por compás')
    p.set_defaults(func=cmd_generate)

    # ── round-trip ────────────────────────────────────────────────────────────
    p = sub.add_parser('round-trip',
        help='MIDI → piano roll → MIDI (diagnóstico, sin modelo)')
    p.add_argument('--input',    required=True,  metavar='FILE')
    p.add_argument('--output',   default='round_trip_out.mid', metavar='FILE')
    p.add_argument('--bpm',      type=float, default=120.0)
    p.add_argument('--pitch-lo', type=int,   default=0)
    p.add_argument('--pitch-hi', type=int,   default=PITCH_CLASSES)
    p.set_defaults(func=cmd_round_trip)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect',
        help='Diagnóstico del modelo entrenado y/o los datos preparados')
    p.add_argument('--model-dir', default=None, metavar='DIR',
                   help='Carpeta del modelo a inspeccionar')
    p.add_argument('--data-dir',  default=None, metavar='DIR',
                   help='Carpeta de datos .npz a inspeccionar')
    p.add_argument('--what',      default='all',
                   choices=['model', 'data', 'all', 'presets'],
                   help='Qué inspeccionar: model | data | all | presets')
    p.add_argument('--verbose',   action='store_true',
                   help='Mostrar detalle por fichero en el caso de datos')
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
