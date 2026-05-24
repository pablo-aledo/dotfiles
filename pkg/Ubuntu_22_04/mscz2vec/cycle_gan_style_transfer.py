#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CYCLE-GAN STYLE TRANSFER  v2.0  (piano-roll)                    ║
║        Transferencia de estilo bidireccional inspirada en CycleGAN          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CAMBIO RESPECTO A v1                                                        ║
║    v1: feature space = vector de 22 estadísticas (DNA comprimido)           ║
║    v2: feature space = piano roll multi-rol aplanado + PCA opcional         ║
║                                                                              ║
║    El piano roll preserva estructura temporal concreta (qué notas suenan    ║
║    en qué posición dentro del compás) en lugar de promedios globales.       ║
║    La dimensión efectiva depende de resolution × n_pitch × n_roles,        ║
║    reducida opcionalmente con PCA para entrenar G y F.                      ║
║                                                                              ║
║  CONCEPTO                                                                    ║
║    CycleGAN original aprende dos mappings: G: A→B y F: B→A, más dos        ║
║    discriminadores D_A y D_B, entrenados con pérdida de ciclo:             ║
║        F(G(a)) ≈ a   y   G(F(b)) ≈ b                                       ║
║                                                                              ║
║    Aquí adaptamos ese framework al dominio MIDI sin redes neuronales:       ║
║                                                                              ║
║    • G  (style_A → style_B): Ridge/KNN sobre piano rolls aplanados+PCA     ║
║    • F  (style_B → style_A): la inversa aprendida del mismo modo.           ║
║    • D_B: distancia Mahalanobis al centroide del dominio B (en piano roll)  ║
║    • D_A: ídem para dominio A.                                              ║
║    • Pérdida de ciclo: ||F(G(roll_a)) - roll_a||₂ + viceversa.            ║
║    • Pérdida de identidad: ||G(roll_b) - roll_b||₂                        ║
║                                                                              ║
║  REPRESENTACIÓN MUSICAL                                                      ║
║    Cada MIDI se convierte a rolls por rol:                                  ║
║        (n_bars, resolution, n_pitch)  para cada rol activo                  ║
║    Luego se computa el centroide temporal (media sobre compases):            ║
║        (resolution, n_pitch)  por rol                                        ║
║    y se concatenan todos los roles y se aplana a un vector 1D:              ║
║        n_roles × resolution × n_pitch  dimensiones                          ║
║    Opcionalmente reducido con PCA a --pca-dim dimensiones.                  ║
║                                                                              ║
║    Roles: melody, counterpoint, accompaniment, bass, percussion             ║
║    Resolución por defecto: 48 ticks/compás                                  ║
║    Rango de pitch por defecto: 128 (completo)                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — TRAIN                                                                 ║
║                                                                              ║
║    python cycle_gan_style_transfer.py train                                  ║
║        --domain-a  corpus_tango/                                             ║
║        --domain-b  corpus_epico/                                             ║
║        --model     tango2epico.cgan.pkl                                      ║
║                                                                              ║
║    # Con PCA y resolución reducida:                                         ║
║    python cycle_gan_style_transfer.py train                                  ║
║        --domain-a  tango/  --domain-b  epico/                               ║
║        --model     t2e.cgan.pkl                                              ║
║        --resolution 16  --pitch-range 48  --pca-dim 64                     ║
║        --lambda-cycle 10.0  --lambda-identity 5.0  --iters 200  --verbose  ║
║                                                                              ║
║    Opciones de train:                                                        ║
║      --domain-a DIR       Carpeta con MIDIs del dominio A                   ║
║      --domain-b DIR       Carpeta con MIDIs del dominio B                   ║
║      --model FILE         Ruta del modelo a guardar [default: cycle_gan.pkl]║
║      --resolution N       Ticks por compás [default: 48]                    ║
║      --pitch-range N      Notas MIDI activas centradas en C4 [default: 128] ║
║      --disable-roles ROL  Roles a ignorar (melody/bass/accompaniment/…)    ║
║      --pca-dim N          Reducir con PCA a N dims (0 = sin PCA)            ║
║      --solver STR         ridge|knn [default: ridge]                        ║
║      --lambda-cycle F     Peso de la pérdida de ciclo [default: 10.0]      ║
║      --lambda-identity F  Peso de la pérdida de identidad [default: 5.0]   ║
║      --iters N            Iteraciones de refinamiento [default: 100]        ║
║      --alpha F            Regularización ridge [default: 1.0]              ║
║      --k-neighbors N      Vecinos para solver KNN [default: 5]             ║
║      --verbose            Mostrar pérdidas por iteración                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — TRANSFORM                                                             ║
║                                                                              ║
║    python cycle_gan_style_transfer.py transform entrada.mid                  ║
║        --model t2e.cgan.pkl  --direction AtoB  --output salida.mid          ║
║                                                                              ║
║    python cycle_gan_style_transfer.py transform entrada.mid                  ║
║        --model t2e.cgan.pkl  --intensity 0.6  --output salida.mid           ║
║                                                                              ║
║    Opciones de transform:                                                    ║
║      --model FILE         Modelo entrenado (.cgan.pkl)                      ║
║      --direction STR      AtoB | BtoA [default: AtoB]                       ║
║      --intensity F        0.0=sin cambio, 1.0=transformación total [1.0]   ║
║      --output FILE        MIDI de salida                                    ║
║      --bars N             Compases de salida [default: auto]                ║
║      --candidates N       Candidatos a evaluar [default: 3]                 ║
║      --no-percussion      No generar percusión                              ║
║      --export-fingerprint Exportar fingerprint del resultado                ║
║      --seed N             Semilla aleatoria [default: 42]                   ║
║      --verbose            Detalle de la transformación                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — CYCLE / ANALYZE                                                       ║
║    (mismos flags que v1)                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS: midi_dna_unified.py, mido, numpy, scikit-learn               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import glob
import json
import pickle
import argparse
import random
import copy
import math
from pathlib import Path

import numpy as np

# ── Importar ecosistema ───────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import (
        UnifiedDNA, EmotionalController, FormGenerator,
        MarkovMelody, GrooveMap,
        generate_accompaniment, generate_bass, generate_counterpoint,
        generate_percussion, add_ornamentation, humanize,
        humanize_with_swing, build_midi, score_candidate,
        _snap_to_scale, _get_scale_midi, _get_scale_pcs,
        _quarter_to_ticks, INSTRUMENT_RANGES,
    )
    from music21 import pitch as m21pitch, key as m21key
except ImportError as e:
    print(f"ERROR: {e}\nmidi_dna_unified.py no encontrado en el mismo directorio.")
    sys.exit(1)

try:
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("AVISO: scikit-learn no encontrado. Solver 'ridge'/'knn' y PCA no disponibles.")

import mido

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE PIANO ROLL  (tomadas de diffusion_composer_v4.py)
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

PITCH_CLASSES  = 128
MIDI_CENTER    = 60   # Do central
TICKS_PER_BAR  = 48   # resolución por defecto


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE RANGO DE PITCH  (de diffusion_composer_v4.py)
# ══════════════════════════════════════════════════════════════════════════════

def _pitch_range(n):
    """Devuelve (pitch_lo, pitch_hi) centrado en Do central, o None si n es None."""
    if n is None:
        return None
    half = n // 2
    lo   = max(0,   MIDI_CENTER - half)
    hi   = min(127, lo + n - 1)
    lo   = max(0, hi - n + 1)
    return (lo, hi)


def _crop_pitch(roll, pitch_lo, pitch_hi):
    """roll shape: (..., 128) → (..., pitch_hi - pitch_lo + 1)"""
    return roll[..., pitch_lo: pitch_hi + 1]


def _pad_pitch(roll, pitch_lo, n_full=128):
    """roll shape: (..., n_crop) → (..., 128)"""
    n_crop = roll.shape[-1]
    suffix = n_full - pitch_lo - n_crop
    pad_widths = [(0, 0)] * (roll.ndim - 1) + [(pitch_lo, suffix)]
    return np.pad(roll, pad_widths, mode='constant')


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE NOTAS POR STREAM  (de diffusion_composer_v4.py)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_note_lists(mid):
    """Retorna dict {(track_idx, channel): [(start_tick, end_tick, note, vel, prog)]}"""
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


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNADOR DE ROLES  (de diffusion_composer_v4.py)
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
            durations = [n[1] - n[0] for n in notes]
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

        pairs = [(score(p, r), r, p['key']) for p in unassigned for r in remaining_roles]
        pairs.sort(key=lambda x: -x[0])
        taken_keys  = set()
        taken_roles = set()
        for sc, role, key in pairs:
            if role not in taken_roles and key not in taken_keys:
                assigned[role] = key
                taken_roles.add(role)
                taken_keys.add(key)
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  PIANO ROLL CONVERTER  (de diffusion_composer_v4.py)
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    def __init__(self, resolution=TICKS_PER_BAR):
        self.resolution = resolution

    def notes_to_roll(self, notes, tpb_raw, n_bars):
        """
        Convierte lista de (start_tick, end_tick, note, vel, prog) a
        piano roll de shape (n_bars, resolution, PITCH_CLASSES) con valores en {0, 1}.
        """
        roll = np.zeros((n_bars, self.resolution, PITCH_CLASSES), dtype=np.float32)
        ticks_per_internal = tpb_raw * 4 / self.resolution
        for start, end, pitch, vel, prog in notes:
            bar_s  = int(start / (tpb_raw * 4))
            bar_e  = int(end   / (tpb_raw * 4))
            tick_s = int((start % (tpb_raw * 4)) / ticks_per_internal)
            tick_e = int((end   % (tpb_raw * 4)) / ticks_per_internal)
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


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MELODÍA (para el motor de generación, igual que v1)
# ══════════════════════════════════════════════════════════════════════════════

def extract_raw_melody(midi_path, verbose=False):
    """
    Extrae la línea melódica del MIDI como lista de
    (offset_beats, pitch_midi, duration_beats, velocity).
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb      = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    notes_by_channel = {}
    pending = {}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            abs_beats  = abs_ticks / tpb
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_beats, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    onset, vel = pending.pop(key_)
                    dur = max(0.1, abs_beats - onset)
                    if msg.channel != 9:
                        notes_by_channel.setdefault(msg.channel, []).append(
                            (onset, msg.note, dur, vel))

    tempo_bpm = 60_000_000 / max(tempo_us, 1)
    if not notes_by_channel:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    def _mean_pitch(notes):
        return sum(p for _, p, _, _ in notes) / len(notes) if notes else 0

    melody_ch = max(notes_by_channel.keys(),
                    key=lambda ch: _mean_pitch(notes_by_channel[ch]))
    melody = sorted(notes_by_channel[melody_ch], key=lambda x: x[0])

    if verbose:
        print(f"    Melodía: {len(melody)} notas | ch={melody_ch} | "
              f"{tempo_bpm:.1f} BPM | {ts_num}/{ts_den}")

    return melody, tempo_bpm, (ts_num, ts_den)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN MIDI → VECTOR PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_roll_vector(midi_path: str,
                        resolution: int = TICKS_PER_BAR,
                        active_roles: list = None,
                        pitch_lo: int = None,
                        pitch_hi: int = None,
                        verbose: bool = False) -> np.ndarray:
    """
    Convierte un MIDI a un vector de piano roll concatenado y aplanado.

    Flujo:
      1. Extraer note_lists por stream.
      2. Asignar roles con RoleAssigner.
      3. Construir roll (n_bars, resolution, n_pitch) por rol.
      4. Calcular centroide temporal: mean sobre el eje de compases
         → (resolution, n_pitch) por rol.
      5. Concatenar todos los roles → (n_roles, resolution, n_pitch).
      6. Aplanar → vector 1D de longitud n_roles × resolution × n_pitch.

    El centroide temporal captura el "ritmo e interválica promedios" del MIDI
    sin depender de su longitud, manteniendo estructura interna del compás.
    """
    if active_roles is None:
        active_roles = ROLES

    n_pitch = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES

    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    tpb_raw   = mid.ticks_per_beat
    tpbar     = tpb_raw * 4
    all_ticks = max((n[1] for nl in note_lists.values() for n in nl), default=0)
    n_bars    = max(1, int(all_ticks / tpbar) + 1)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution)

    # Un slot por rol activo, incluso si no hay datos (roll de ceros)
    centroid_parts = []
    for role in active_roles:
        stream_key = role_map.get(role)
        if stream_key and stream_key in note_lists:
            notes = note_lists[stream_key]
            roll  = conv.notes_to_roll(notes, tpbar, n_bars)
            if pitch_lo is not None:
                roll = _crop_pitch(roll, pitch_lo, pitch_hi)
            # Centroide temporal: media sobre compases → (resolution, n_pitch)
            centroid = roll.mean(axis=0)
        else:
            centroid = np.zeros((resolution, n_pitch), dtype=np.float32)
        centroid_parts.append(centroid)

    # Concatenar roles y aplanar
    # shape: (n_roles, resolution, n_pitch) → 1D
    vec = np.concatenate([c.flatten() for c in centroid_parts]).astype(np.float32)

    if verbose:
        n_active = sum(1 for r in active_roles if role_map.get(r))
        print(f"    Roll: {n_bars} compases | {n_active}/{len(active_roles)} roles | "
              f"vector {len(vec)}D")

    return vec


def roll_vector_dim(n_roles: int, resolution: int, n_pitch: int) -> int:
    """Dimensión del vector de piano roll para los parámetros dados."""
    return n_roles * resolution * n_pitch


# ══════════════════════════════════════════════════════════════════════════════
#  DISCRIMINADORES SIMBÓLICOS (Mahalanobis en espacio de piano roll / PCA)
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicDiscriminator:
    """
    D_X: mide distancia de un vector de piano roll al centroide del dominio X.
    Idéntico en interfaz a v1, pero opera sobre el espacio de rolls (o PCA).
    """

    def __init__(self, name: str):
        self.name     = name
        self.centroid = None
        self.inv_cov  = None
        self.scaler   = None

    def fit(self, features: np.ndarray):
        """features: (N, D)"""
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            fs = self.scaler.fit_transform(features)
        else:
            fs = features
        self.centroid = np.mean(fs, axis=0)
        try:
            cov = np.cov(fs.T)
            self.inv_cov = np.linalg.pinv(cov)
        except Exception:
            self.inv_cov = None

    def score(self, feat: np.ndarray) -> float:
        if self.centroid is None:
            return 1.0
        if HAS_SKLEARN and self.scaler:
            f = self.scaler.transform(feat.reshape(1, -1)).flatten()
        else:
            f = feat
        diff = f - self.centroid
        if self.inv_cov is not None:
            dist = float(np.sqrt(np.maximum(diff @ self.inv_cov @ diff, 0.0)))
        else:
            dist = float(np.linalg.norm(diff))
        return dist

    def domain_probability(self, feat: np.ndarray) -> float:
        d = self.score(feat)
        return float(np.exp(-d * 0.5))


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES G y F  (regresión en espacio de piano roll / PCA)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMapper:
    """
    G o F en el CycleGAN simbólico.
    Opera sobre vectores de piano roll (posiblemente reducidos con PCA).
    """

    def __init__(self, name: str, solver: str = 'ridge',
                 alpha: float = 1.0, k: int = 5):
        self.name    = name
        self.solver  = solver
        self.alpha   = alpha
        self.k       = k
        self.model   = None
        self.scaler_in  = None
        self.scaler_out = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (N, D) source features  |  Y: (N, D) target features"""
        if not HAS_SKLEARN:
            self._mean_dst = np.mean(Y, axis=0)
            self._mean_src = np.mean(X, axis=0)
            return

        self.scaler_in  = StandardScaler().fit(X)
        self.scaler_out = StandardScaler().fit(Y)
        Xs = self.scaler_in.transform(X)
        Ys = self.scaler_out.transform(Y)

        if self.solver == 'knn':
            self.model = KNeighborsRegressor(n_neighbors=min(self.k, len(X)))
        else:
            self.model = Ridge(alpha=self.alpha)
        self.model.fit(Xs, Ys)

    def transform(self, feat: np.ndarray) -> np.ndarray:
        if not HAS_SKLEARN or self.model is None:
            if hasattr(self, '_mean_dst'):
                delta = self._mean_dst - self._mean_src
                return np.clip(feat + delta, 0.0, 1.0)
            return feat.copy()

        f_in  = self.scaler_in.transform(feat.reshape(1, -1))
        f_out = self.model.predict(f_in)[0]
        return self.scaler_out.inverse_transform(f_out.reshape(1, -1))[0]

    def cycle_loss(self, feats_a: np.ndarray, mapper_back: "FeatureMapper") -> float:
        losses = []
        for f in feats_a:
            ab  = self.transform(f)
            aba = mapper_back.transform(ab)
            losses.append(np.linalg.norm(aba - f))
        return float(np.mean(losses))

    def identity_loss(self, feats_dst: np.ndarray) -> float:
        losses = []
        for f in feats_dst:
            gf = self.transform(f)
            losses.append(np.linalg.norm(gf - f))
        return float(np.mean(losses))


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO CYCLE-GAN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANModel:
    """
    Encapsula G, F, D_A, D_B y las pérdidas de ciclo e identidad.
    Ahora trabaja en el espacio de piano roll (con PCA opcional).
    Se serializa a .cgan.pkl.
    """

    VERSION = "2.0"

    def __init__(self,
                 solver: str = 'ridge',
                 alpha: float = 1.0,
                 k_neighbors: int = 5,
                 lambda_cycle: float = 10.0,
                 lambda_identity: float = 5.0,
                 # parámetros de piano roll
                 resolution: int = TICKS_PER_BAR,
                 active_roles: list = None,
                 pitch_lo: int = None,
                 pitch_hi: int = None,
                 pca_dim: int = 0):

        self.solver          = solver
        self.alpha           = alpha
        self.k_neighbors     = k_neighbors
        self.lambda_cycle    = lambda_cycle
        self.lambda_identity = lambda_identity

        # Parámetros de representación de piano roll
        self.resolution  = resolution
        self.active_roles = active_roles or ROLES
        self.pitch_lo    = pitch_lo
        self.pitch_hi    = pitch_hi
        self.pca_dim     = pca_dim

        n_pitch  = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES
        self.raw_dim = roll_vector_dim(len(self.active_roles), resolution, n_pitch)

        self.G   = FeatureMapper("G_AtoB", solver, alpha, k_neighbors)
        self.F   = FeatureMapper("F_BtoA", solver, alpha, k_neighbors)
        self.D_A = SymbolicDiscriminator("D_A")
        self.D_B = SymbolicDiscriminator("D_B")

        # PCA compartido (se ajusta sobre la unión de ambos dominios)
        self.pca: "PCA | None" = None

        self.centroid_a:     np.ndarray | None = None
        self.centroid_b:     np.ndarray | None = None
        self.losses_history: list[dict] = []
        self.domain_a_name:  str = "A"
        self.domain_b_name:  str = "B"

    # ── API de vectorización (encapsula los parámetros del modelo) ──────────

    def midi_to_vec(self, midi_path: str, verbose: bool = False) -> np.ndarray:
        """Convierte un MIDI al vector de features del modelo (roll + PCA si aplica)."""
        vec = midi_to_roll_vector(
            midi_path,
            resolution   = self.resolution,
            active_roles = self.active_roles,
            pitch_lo     = self.pitch_lo,
            pitch_hi     = self.pitch_hi,
            verbose      = verbose,
        )
        if self.pca is not None:
            vec = self.pca.transform(vec.reshape(1, -1))[0]
        return vec

    def _apply_pca(self, vecs: np.ndarray) -> np.ndarray:
        """Aplica la PCA ya ajustada a un conjunto de vectores."""
        if self.pca is None:
            return vecs
        return self.pca.transform(vecs)

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def fit(self, feats_a: np.ndarray, feats_b: np.ndarray,
            iters: int = 100, verbose: bool = False):
        """
        feats_a, feats_b: arrays (N, raw_dim) de vectores de piano roll crudos.
        Si pca_dim > 0, ajusta PCA sobre la unión y proyecta antes de entrenar.
        """
        if len(feats_a) == 0 or len(feats_b) == 0:
            raise ValueError("Se necesitan MIDIs en ambos dominios.")

        # ── PCA (opcional) ─────────────────────────────────────────────────
        if self.pca_dim > 0 and HAS_SKLEARN:
            all_vecs = np.vstack([feats_a, feats_b])
            n_components = min(self.pca_dim, all_vecs.shape[0], all_vecs.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(all_vecs)
            feats_a = self.pca.transform(feats_a)
            feats_b = self.pca.transform(feats_b)
            if verbose:
                var = self.pca.explained_variance_ratio_.sum()
                print(f"  PCA: {n_components} componentes | varianza explicada: {var:.1%}")
        else:
            self.pca = None

        self.centroid_a = np.mean(feats_a, axis=0)
        self.centroid_b = np.mean(feats_b, axis=0)

        self.D_A.fit(feats_a)
        self.D_B.fit(feats_b)

        # Pares iniciales por vecino más cercano
        pairs_ab = _nearest_neighbor_pairs(feats_a, feats_b)
        pairs_ba = _nearest_neighbor_pairs(feats_b, feats_a)

        Y_b = np.array([feats_b[j] for j in pairs_ab])
        Y_a = np.array([feats_a[j] for j in pairs_ba])

        self.G.fit(feats_a, Y_b)
        self.F.fit(feats_b, Y_a)

        if verbose:
            print(f"  {'Iter':>4}  {'CycLoss_A':>10}  {'CycLoss_B':>10}  "
                  f"{'IdLoss_G':>9}  {'IdLoss_F':>9}  {'Total':>10}")

        for it in range(iters):
            lc_a  = self.G.cycle_loss(feats_a, self.F)
            lc_b  = self.F.cycle_loss(feats_b, self.G)
            li_g  = self.G.identity_loss(feats_b)
            li_f  = self.F.identity_loss(feats_a)
            total = (self.lambda_cycle * (lc_a + lc_b) +
                     self.lambda_identity * (li_g + li_f))

            self.losses_history.append({
                'iter': it, 'cycle_a': lc_a, 'cycle_b': lc_b,
                'identity_g': li_g, 'identity_f': li_f, 'total': total
            })

            if verbose and (it % max(1, iters // 20) == 0 or it == iters - 1):
                print(f"  {it:>4}  {lc_a:>10.4f}  {lc_b:>10.4f}  "
                      f"{li_g:>9.4f}  {li_f:>9.4f}  {total:>10.4f}")

            # Refinamiento con datos pseudo-ciclados
            G_feats_a = np.array([self.G.transform(f) for f in feats_a])
            aug_X_a   = np.vstack([feats_a, np.array([self.F.transform(f) for f in feats_b])])
            aug_Y_b   = np.vstack([G_feats_a, feats_b])
            self.G.fit(aug_X_a, aug_Y_b)

            F_feats_b = np.array([self.F.transform(f) for f in feats_b])
            aug_X_b   = np.vstack([feats_b, np.array([self.G.transform(f) for f in feats_a])])
            aug_Y_a   = np.vstack([F_feats_b, feats_a])
            self.F.fit(aug_X_b, aug_Y_a)

        if verbose:
            print(f"\n  Entrenamiento completado. Pérdida final total: {total:.4f}")

    # ── Inferencia ─────────────────────────────────────────────────────────

    def map_features(self, feat: np.ndarray, direction: str = 'AtoB') -> np.ndarray:
        if direction == 'AtoB':
            return self.G.transform(feat)
        elif direction == 'BtoA':
            return self.F.transform(feat)
        else:
            raise ValueError(f"direction debe ser 'AtoB' o 'BtoA', no '{direction}'")

    def discriminate(self, feat: np.ndarray) -> dict:
        return {
            'D_A_dist':  self.D_A.score(feat),
            'D_B_dist':  self.D_B.score(feat),
            'D_A_prob':  self.D_A.domain_probability(feat),
            'D_B_prob':  self.D_B.domain_probability(feat),
        }

    # ── Serialización ────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado → {path}")

    @staticmethod
    def load(path: str) -> "CycleGANModel":
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, CycleGANModel):
            raise TypeError("El fichero no contiene un CycleGANModel válido.")
        return model


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN: vector de roll transformado → parámetros para el generador DNA
# ══════════════════════════════════════════════════════════════════════════════

def roll_vector_to_dna_overrides(
        src_vec: np.ndarray,
        dst_vec: np.ndarray,
        source_dna: UnifiedDNA,
        model: CycleGANModel,
        intensity: float = 1.0) -> UnifiedDNA:
    """
    Deriva parámetros musicales del desplazamiento en el espacio de piano roll
    para guiar el motor generador DNA.

    En lugar de operar directamente sobre un vector de 22 estadísticas como
    en v1, aquí se extraen las dimensiones relevantes del roll transformado:

      - Densidad rítmica: fracción de celdas activas en el centroide de melody
      - Registro medio de pitch: centro de masa del eje de pitch en melody
      - Complejidad armónica: densidad en accompaniment vs melody
      - Tensión emocional: actividad en notas disonantes (mod 12)

    El resultado es un UnifiedDNA copia del source con solo esos campos
    sobreescritos. El resto de parámetros (tonalidad, forma, markov, groove)
    se heredan del source_dna sin modificación, lo que preserva más estructura
    original que el enfoque de v1.
    """
    out = copy.deepcopy(source_dna)

    n_roles    = len(model.active_roles)
    resolution = model.resolution
    n_pitch    = (model.pitch_hi - model.pitch_lo + 1) if model.pitch_lo is not None else PITCH_CLASSES
    slot_size  = resolution * n_pitch

    # Si hay PCA, revertir al espacio de roll (aproximado)
    def _get_raw(vec):
        if model.pca is not None:
            return model.pca.inverse_transform(vec.reshape(1, -1))[0]
        return vec

    src_raw = _get_raw(src_vec)
    dst_raw = _get_raw(dst_vec)

    # Interpolar según intensidad
    interp = src_raw * (1.0 - intensity) + dst_raw * intensity

    def _slot(vec, role_name):
        """Extrae el slot (resolution, n_pitch) del rol dado."""
        idx = model.active_roles.index(role_name) if role_name in model.active_roles else None
        if idx is None:
            return None
        start = idx * slot_size
        return vec[start: start + slot_size].reshape(resolution, n_pitch)

    mel_slot  = _slot(interp, 'melody')
    acc_slot  = _slot(interp, 'accompaniment')
    bass_slot = _slot(interp, 'bass')

    # ── Densidad rítmica (notas activas en melodía) ────────────────────────
    if mel_slot is not None:
        density = float(mel_slot.mean())
        # Mapear [0, 1] a una escala de densidad de notas por compás
        # Un centroide con densidad 0.1 equivale aprox. a corcheas (~8 notas/compás)
        out.harmony_complexity = float(
            np.clip(getattr(source_dna, 'harmony_complexity', 0.3)
                    + (density - float(_slot(src_raw, 'melody').mean())),
                    0.0, 1.0)
        ) if _slot(src_raw, 'melody') is not None else out.harmony_complexity

    # ── Registro medio de pitch ────────────────────────────────────────────
    if mel_slot is not None and mel_slot.sum() > 0:
        pitch_axis = mel_slot.sum(axis=0)  # (n_pitch,)
        mean_pitch_idx = float(np.average(np.arange(n_pitch), weights=pitch_axis + 1e-9))
        # Convertir índice de pitch al espacio MIDI
        pitch_lo_eff = model.pitch_lo if model.pitch_lo is not None else 0
        mean_pitch_midi = pitch_lo_eff + mean_pitch_idx
        # Ajustar el registro del source_dna si el desplazamiento es significativo
        src_mel_slot = _slot(src_raw, 'melody')
        if src_mel_slot is not None and src_mel_slot.sum() > 0:
            src_pitch_ax = src_mel_slot.sum(axis=0)
            src_mean_idx = float(np.average(np.arange(n_pitch), weights=src_pitch_ax + 1e-9))
            pitch_delta  = (mean_pitch_idx - src_mean_idx) * intensity
            # Solo guardar si hay desplazamiento real
            if abs(pitch_delta) > 1.0:
                contour = list(getattr(source_dna, 'pitch_contour', []) or [])
                if contour:
                    shift = int(round(pitch_delta))
                    out.pitch_contour = [
                        int(np.clip(p + shift, 21, 108)) for p in contour
                    ]

    # ── Complejidad armónica relativa acc/melody ───────────────────────────
    if acc_slot is not None and mel_slot is not None:
        acc_density = float(acc_slot.mean())
        mel_density = float(mel_slot.mean()) + 1e-9
        ratio       = min(acc_density / mel_density, 2.0)
        src_acc = _slot(src_raw, 'accompaniment')
        src_mel = _slot(src_raw, 'melody')
        if src_acc is not None and src_mel is not None:
            src_ratio = float(src_acc.mean()) / (float(src_mel.mean()) + 1e-9)
            delta_ratio = (ratio - src_ratio) * intensity * 0.3
            out.harmony_complexity = float(
                np.clip(out.harmony_complexity + delta_ratio, 0.0, 1.0))

    # ── Tensión emocional: presencia de intervalos disonantes ────────────
    if mel_slot is not None:
        # Proyectar al espacio de pitch-class
        pitch_lo_eff = model.pitch_lo if model.pitch_lo is not None else 0
        pc_profile   = np.zeros(12, dtype=np.float32)
        for pc in range(12):
            # Sumar todas las posiciones pitch que correspondan a esta clase
            idxs = [i for i in range(n_pitch)
                    if (pitch_lo_eff + i) % 12 == pc]
            if idxs:
                pc_profile[pc] = mel_slot[:, idxs].mean()

        DISSONANT_PCS = {1, 2, 6, 10, 11}
        total_act = pc_profile.sum() + 1e-9
        diss_frac = sum(pc_profile[pc] for pc in DISSONANT_PCS) / total_act

        tc = list(getattr(source_dna, 'tension_curve', [0.5]) or [0.5])
        src_mel_slot2 = _slot(src_raw, 'melody')
        if src_mel_slot2 is not None:
            src_pc = np.zeros(12, dtype=np.float32)
            for pc in range(12):
                idxs = [i for i in range(n_pitch)
                        if (pitch_lo_eff + i) % 12 == pc]
                if idxs:
                    src_pc[pc] = src_mel_slot2[:, idxs].mean()
            src_diss = sum(src_pc[pc] for pc in DISSONANT_PCS) / (src_pc.sum() + 1e-9)
            delta_tension = (diss_frac - src_diss) * intensity
            out.tension_curve = [
                float(np.clip(t + delta_tension, 0.0, 1.0)) for t in tc
            ]

    return out


def _nearest_neighbor_pairs(src: np.ndarray, dst: np.ndarray) -> list:
    indices = []
    for f in src:
        dists = [np.linalg.norm(f - g) for g in dst]
        indices.append(int(np.argmin(dists)))
    return indices


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus_rolls(directory: str,
                      resolution: int = TICKS_PER_BAR,
                      active_roles: list = None,
                      pitch_lo: int = None,
                      pitch_hi: int = None,
                      verbose: bool = False) -> np.ndarray:
    """
    Carga todos los MIDIs de un directorio y los convierte a vectores de piano roll.
    Retorna array (N, D).
    """
    midi_files = (
        glob.glob(os.path.join(directory, "*.mid")) +
        glob.glob(os.path.join(directory, "*.midi"))
    )
    if not midi_files:
        raise RuntimeError(f"No se encontraron MIDIs en {directory}")

    vecs = []
    for path in midi_files:
        try:
            v = midi_to_roll_vector(
                path,
                resolution   = resolution,
                active_roles = active_roles or ROLES,
                pitch_lo     = pitch_lo,
                pitch_hi     = pitch_hi,
                verbose      = False,
            )
            vecs.append(v)
            if verbose:
                print(f"  ✓ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: {e}")

    if not vecs:
        raise RuntimeError(f"No se pudo procesar ningún MIDI en {directory}")

    arr = np.array(vecs, dtype=np.float32)
    print(f"  {len(arr)} MIDIs cargados desde '{directory}' | vector dim = {arr.shape[1]}")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE GENERACIÓN (igual que v1 — el DNA sigue siendo el generador)
# ══════════════════════════════════════════════════════════════════════════════

def _snap_melody_to_key(melody, key_obj):
    return [(o, _snap_to_scale(p, key_obj), d, v) for o, p, d, v in melody]


def run_cycle_gan_transform(
    source_dna: UnifiedDNA,
    transformed_dna: UnifiedDNA,
    original_melody,
    intensity: float = 1.0,
    n_bars: int = 16,
    candidates: int = 3,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Motor de generación: dado el DNA transformado por el CycleGAN,
    genera el MIDI de salida preservando la esencia melódica del source.
    (Idéntico a v1; el DNA transformado ahora viene de roll_vector_to_dna_overrides.)
    """
    random.seed(seed)
    np.random.seed(seed)

    target_key = transformed_dna.key_obj or source_dna.key_obj
    tempo_bpm  = transformed_dna.tempo_bpm
    time_sig   = transformed_dna.time_sig or source_dna.time_sig
    bpb = time_sig[0]

    prog_s = source_dna.harmony_prog or [('I', 2.0)]
    prog_t = transformed_dna.harmony_prog or [('I', 2.0)]
    n_prog = max(len(prog_s), len(prog_t))
    prog_s = (prog_s * (n_prog // max(len(prog_s), 1) + 1))[:n_prog]
    prog_t = (prog_t * (n_prog // max(len(prog_t), 1) + 1))[:n_prog]
    mixed_prog = []
    for (fs, ds), (ft, dt) in zip(prog_s, prog_t):
        if random.random() < intensity:
            mixed_prog.append((ft, dt))
        else:
            mixed_prog.append((fs, ds))

    ec = EmotionalController(
        tension_curve   = transformed_dna.tension_curve   or [0.5],
        arousal_curve   = transformed_dna.arousal_curve   or [0.0],
        valence_curve   = transformed_dna.valence_curve   or [0.0],
        stability_curve = getattr(transformed_dna, 'stability_curve', None) or [0.7],
        activity_curve  = getattr(transformed_dna, 'activity_curve',  None) or [0.5],
        emotional_arc_label = getattr(transformed_dna, 'emotional_arc_label', ''),
        n_bars = n_bars,
    )

    fg = FormGenerator(
        form_string       = getattr(transformed_dna, 'form_string',       'AABA'),
        section_map       = getattr(transformed_dna, 'section_map',       {}),
        phrase_lengths    = getattr(transformed_dna, 'phrase_lengths',    [4]),
        cadence_positions = getattr(transformed_dna, 'cadence_positions', []),
        n_bars_out        = n_bars,
    )

    total_beats_out = n_bars * bpb
    if original_melody:
        total_beats_in = max(o + d for o, _, d, _ in original_melody)
        scale = total_beats_out / max(total_beats_in, 1e-9)
        scaled_mel = [(o * scale, p, d * scale, v) for o, p, d, v in original_melody]
        scaled_mel = _snap_melody_to_key(scaled_mel, target_key)
    else:
        scaled_mel = []

    groove = (transformed_dna.groove_map
              if getattr(transformed_dna, 'groove_map', None)
              and transformed_dna.groove_map.trained
              else None)

    best_score, best_result = -1.0, None

    for ci in range(max(1, candidates)):
        random.seed(seed + ci * 17)
        np.random.seed(seed + ci * 17)

        generated = dna_mod._generate_melody_with_modulation(
            h_prog       = mixed_prog,
            target_key   = target_key,
            r_pat        = transformed_dna.rhythm_pattern or source_dna.rhythm_pattern,
            contour      = transformed_dna.pitch_contour  or source_dna.pitch_contour,
            reg          = source_dna.pitch_register,
            motif        = source_dna.motif_intervals,
            n_bars       = n_bars,
            ec           = ec,
            fg           = fg,
            bpb          = bpb,
            rhythm_strength    = intensity,
            markov             = transformed_dna.markov,
            seq_phrases        = getattr(transformed_dna, 'sequitur_phrases', []),
            melody_mode        = 'markov',
            surprise_rate      = 0.08,
            use_motif_coherence= True,
            use_tension_markov = True,
        )

        if scaled_mel and generated:
            n     = min(len(scaled_mel), len(generated))
            alpha = 1.0 - intensity
            gen_sorted = sorted(generated, key=lambda x: x[0])
            src_sorted = sorted(scaled_mel, key=lambda x: x[0])
            mel = []
            for i in range(n):
                oa, pa, da, va = src_sorted[i]
                ob, pb, db, vb = gen_sorted[i]
                mp = int(round(pa * alpha + pb * (1 - alpha)))
                mp = _snap_to_scale(mp, target_key)
                mel.append((oa, mp, da * alpha + db * (1 - alpha),
                             int(va * alpha + vb * (1 - alpha))))
        elif scaled_mel:
            mel = scaled_mel
        else:
            mel = generated

        mel = add_ornamentation(mel, target_key,
                                getattr(transformed_dna, 'style', 'generic'))
        if groove:
            mel = humanize(mel, groove, bpb)

        acc  = generate_accompaniment(
            mixed_prog, target_key, n_bars, ec, fg, bpb,
            groove_map=groove,
            harmony_complexity=transformed_dna.harmony_complexity)
        bass = generate_bass(mixed_prog, target_key, n_bars, bpb, groove)
        cp   = generate_counterpoint(mel, mixed_prog, target_key, n_bars, bpb, ec)

        sc = score_candidate(mel, acc, target_key)
        if verbose:
            print(f"    Candidato {ci+1}/{candidates}: score={sc:.3f}")
        if sc > best_score:
            best_score, best_result = sc, (mel, acc, bass, cp)

    mel, acc, bass, cp = best_result

    perc = generate_percussion(
        transformed_dna.rhythm_grid,
        transformed_dna.rhythm_accent_grid,
        n_bars, bpb,
        groove_map=groove,
        style=getattr(transformed_dna, 'style', 'generic'),
    )

    return mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_roll_args(args) -> tuple:
    """Extrae parámetros de piano roll de los args del CLI."""
    active_roles = [r for r in ROLES
                    if r not in set(getattr(args, 'disable_roles', None) or [])]
    pr = _pitch_range(getattr(args, 'pitch_range', None))
    pitch_lo = pr[0] if pr else None
    pitch_hi = pr[1] if pr else None
    resolution = getattr(args, 'resolution', TICKS_PER_BAR)
    pca_dim    = getattr(args, 'pca_dim', 0)
    return active_roles, pitch_lo, pitch_hi, resolution, pca_dim


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v2 — TRAIN  (piano roll)")
    print("═" * 65)
    print(f"  Dominio A    : {args.domain_a}")
    print(f"  Dominio B    : {args.domain_b}")
    print(f"  Modelo       : {args.model}")
    print(f"  Solver       : {args.solver}")
    print(f"  Resolución   : {args.resolution} ticks/compás")
    print(f"  Rango pitch  : {args.pitch_range or 128} notas")
    print(f"  PCA dim      : {args.pca_dim or 'sin PCA'}")
    print(f"  λ ciclo      : {args.lambda_cycle}")
    print(f"  λ identidad  : {args.lambda_identity}")
    print(f"  Iteraciones  : {args.iters}")

    active_roles, pitch_lo, pitch_hi, resolution, pca_dim = _parse_roll_args(args)

    if args.verbose and active_roles != ROLES:
        print(f"  Roles activos: {', '.join(active_roles)}")

    print("\n[1/3] Cargando corpus A…")
    feats_a = load_corpus_rolls(
        args.domain_a, resolution, active_roles, pitch_lo, pitch_hi, args.verbose)

    print("\n[2/3] Cargando corpus B…")
    feats_b = load_corpus_rolls(
        args.domain_b, resolution, active_roles, pitch_lo, pitch_hi, args.verbose)

    print("\n[3/3] Entrenando CycleGAN…")
    model = CycleGANModel(
        solver          = args.solver,
        alpha           = args.alpha,
        k_neighbors     = args.k_neighbors,
        lambda_cycle    = args.lambda_cycle,
        lambda_identity = args.lambda_identity,
        resolution      = resolution,
        active_roles    = active_roles,
        pitch_lo        = pitch_lo,
        pitch_hi        = pitch_hi,
        pca_dim         = pca_dim,
    )
    model.domain_a_name = os.path.basename(args.domain_a.rstrip('/'))
    model.domain_b_name = os.path.basename(args.domain_b.rstrip('/'))
    model.fit(feats_a, feats_b, iters=args.iters, verbose=args.verbose)

    model.save(args.model)

    if model.losses_history:
        last = model.losses_history[-1]
        print(f"\n  Pérdida de ciclo A   : {last['cycle_a']:.4f}")
        print(f"  Pérdida de ciclo B   : {last['cycle_b']:.4f}")
        print(f"  Pérdida identidad G  : {last['identity_g']:.4f}")
        print(f"  Pérdida identidad F  : {last['identity_f']:.4f}")

    print("\n" + "═" * 65)
    print(f"  Modelo guardado: {args.model}")
    print("═" * 65)


# ── TRANSFORM ─────────────────────────────────────────────────────────────────

def cmd_transform(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v2 — TRANSFORM  (piano roll)")
    print("═" * 65)
    print(f"  Entrada   : {args.input}")
    print(f"  Modelo    : {args.model}")
    print(f"  Dirección : {args.direction}")
    print(f"  Intensidad: {args.intensity:.2f}")

    model = CycleGANModel.load(args.model)

    print("\n[1/5] Extrayendo melodía…")
    original_melody, _, _ = extract_raw_melody(args.input, verbose=args.verbose)
    print(f"  ✓ {len(original_melody)} notas")

    print("\n[2/5] Convirtiendo a piano roll…")
    feat_src = model.midi_to_vec(args.input, verbose=args.verbose)
    print(f"  ✓ vector {len(feat_src)}D")

    print("\n[3/5] Aplicando transformación CycleGAN…")
    feat_dst = model.map_features(feat_src, direction=args.direction)
    feat_interp = feat_src * (1 - args.intensity) + feat_dst * args.intensity

    disc = model.discriminate(feat_interp)
    if args.verbose:
        print(f"  D_A prob: {disc['D_A_prob']:.3f}  |  D_B prob: {disc['D_B_prob']:.3f}")

    print("\n[4/5] Derivando parámetros DNA desde roll transformado…")
    source_dna = UnifiedDNA(args.input)
    source_dna.extract(verbose=args.verbose)
    transformed_dna = roll_vector_to_dna_overrides(
        feat_src, feat_interp, source_dna, model, intensity=args.intensity)

    if args.bars:
        n_bars = args.bars
    else:
        bpb = source_dna.time_sig[0]
        if original_melody:
            total_beats = max(o + d for o, _, d, _ in original_melody)
            n_bars = max(4, (int(math.ceil(total_beats / bpb)) // 4) * 4)
        else:
            n_bars = 16

    print(f"\n[5/5] Generando MIDI ({n_bars} compases, {args.candidates} candidatos)…")
    mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg = run_cycle_gan_transform(
        source_dna=source_dna,
        transformed_dna=transformed_dna,
        original_melody=original_melody,
        intensity=args.intensity,
        n_bars=n_bars,
        candidates=args.candidates,
        seed=args.seed,
        verbose=args.verbose,
    )

    out_path = args.output or (
        os.path.splitext(args.input)[0] + f"_cgan_{args.direction}.mid")

    build_midi(mel, acc, bass, cp,
               target_key, tempo_bpm, time_sig, n_bars,
               form_gen=fg, output_path=out_path,
               percussion_notes=None if args.no_percussion else perc)

    if args.export_fingerprint:
        fp = dna_mod.extract_fingerprint(
            mel, bass, acc, target_key, tempo_bpm, n_bars, time_sig,
            fg, transformed_dna, out_path)
        json_path = dna_mod.export_fingerprint(fp, out_path)
        print(f"  → Fingerprint: {json_path}")

    print("\n" + "═" * 65)
    print(f"  Salida: {out_path}")
    print(f"  Tonalidad: {target_key.tonic.name} {target_key.mode}  |  {tempo_bpm:.0f} BPM")
    print(f"  D_A prob: {disc['D_A_prob']:.3f}  |  D_B prob: {disc['D_B_prob']:.3f}")
    print("═" * 65)


# ── CYCLE ─────────────────────────────────────────────────────────────────────

def cmd_cycle(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v2 — CYCLE CONSISTENCY  (piano roll)")
    print("═" * 65)

    model = CycleGANModel.load(args.model)

    print("\n[1/5] Extrayendo melodía y piano roll…")
    original_melody, _, _ = extract_raw_melody(args.input, verbose=args.verbose)
    feat_src = model.midi_to_vec(args.input, verbose=args.verbose)

    source_dna = UnifiedDNA(args.input)
    source_dna.extract(verbose=args.verbose)

    dir1 = args.direction
    dir2 = 'BtoA' if dir1 == 'AtoB' else 'AtoB'

    print(f"\n[2/5] Aplicando {dir1} (G)…")
    feat_ab    = model.map_features(feat_src, direction=dir1)
    feat_ab_i  = feat_src * (1 - args.intensity) + feat_ab * args.intensity
    dna_ab     = roll_vector_to_dna_overrides(
        feat_src, feat_ab_i, source_dna, model, intensity=args.intensity)

    bpb = source_dna.time_sig[0]
    if original_melody:
        total_beats = max(o + d for o, _, d, _ in original_melody)
        n_bars = max(4, (int(math.ceil(total_beats / bpb)) // 4) * 4)
    else:
        n_bars = 16
    if args.bars:
        n_bars = args.bars

    print(f"\n[3/5] Generando MIDI intermedio ({n_bars} compases)…")
    mel_ab, acc_ab, bass_ab, cp_ab, perc_ab, tk_ab, tmp_ab, ts_ab, fg_ab = \
        run_cycle_gan_transform(source_dna, dna_ab, original_melody,
                                intensity=args.intensity,
                                n_bars=n_bars, candidates=args.candidates,
                                seed=args.seed, verbose=args.verbose)

    out_ab = args.output_ab or (os.path.splitext(args.input)[0] + "_ab.mid")
    build_midi(mel_ab, acc_ab, bass_ab, cp_ab, tk_ab, tmp_ab, ts_ab, n_bars,
               form_gen=fg_ab, output_path=out_ab,
               percussion_notes=None if args.no_percussion else perc_ab)
    print(f"  ✓ Salida G: {out_ab}")

    print(f"\n[4/5] Aplicando {dir2} (F) sobre el resultado intermedio…")
    # Extraer roll del MIDI intermedio generado
    feat_ab_real = model.midi_to_vec(out_ab, verbose=False)

    feat_aba   = model.map_features(feat_ab_real, direction=dir2)
    feat_aba_i = feat_ab_real * (1 - args.intensity) + feat_aba * args.intensity

    dna_ab_file = UnifiedDNA(out_ab)
    dna_ab_file.extract(verbose=False)
    dna_aba = roll_vector_to_dna_overrides(
        feat_ab_real, feat_aba_i, dna_ab_file, model, intensity=args.intensity)

    mel_aba_src, _, _ = extract_raw_melody(out_ab, verbose=False)

    print(f"\n[5/5] Generando MIDI reconstruido ({n_bars} compases)…")
    mel_aba, acc_aba, bass_aba, cp_aba, perc_aba, tk_aba, tmp_aba, ts_aba, fg_aba = \
        run_cycle_gan_transform(dna_ab_file, dna_aba, mel_aba_src,
                                intensity=args.intensity,
                                n_bars=n_bars, candidates=args.candidates,
                                seed=args.seed + 100, verbose=args.verbose)

    out_aba = args.output_aba or (os.path.splitext(args.input)[0] + "_aba.mid")
    build_midi(mel_aba, acc_aba, bass_aba, cp_aba, tk_aba, tmp_aba, ts_aba, n_bars,
               form_gen=fg_aba, output_path=out_aba,
               percussion_notes=None if args.no_percussion else perc_aba)

    # Pérdida de ciclo en el espacio de roll (con PCA si aplica)
    feat_aba_check = model.map_features(feat_ab_real, direction=dir2)
    cycle_loss = float(np.linalg.norm(feat_aba_check - feat_src))

    print("\n" + "═" * 65)
    print(f"  Salida G   : {out_ab}")
    print(f"  Salida F∘G : {out_aba}")
    print(f"  Pérdida de ciclo (‖F(G(a)) - a‖₂): {cycle_loss:.4f}")
    print(f"  Espacio: {'PCA ' + str(model.pca_dim) + 'D' if model.pca else 'roll ' + str(len(feat_src)) + 'D'}")
    print("═" * 65)


# ── ANALYZE ───────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v2 — ANALYZE  (piano roll)")
    print("═" * 65)
    print(f"  Entrada : {args.input}")
    print(f"  Modelo  : {args.model}")

    model = CycleGANModel.load(args.model)

    feat = model.midi_to_vec(args.input, verbose=args.verbose)

    disc    = model.discriminate(feat)
    feat_ab = model.map_features(feat, 'AtoB')
    feat_ba = model.map_features(feat, 'BtoA')

    print(f"\n  Dominio A ({model.domain_a_name}):")
    print(f"    Distancia discriminativa : {disc['D_A_dist']:.4f}")
    print(f"    Probabilidad pertenencia : {disc['D_A_prob']:.4f}")
    print(f"\n  Dominio B ({model.domain_b_name}):")
    print(f"    Distancia discriminativa : {disc['D_B_dist']:.4f}")
    print(f"    Probabilidad pertenencia : {disc['D_B_prob']:.4f}")

    total_prob = disc['D_A_prob'] + disc['D_B_prob'] + 1e-9
    pct_a = disc['D_A_prob'] / total_prob * 100
    pct_b = disc['D_B_prob'] / total_prob * 100
    print(f"\n  Posición relativa: {pct_a:.1f}% {model.domain_a_name} / "
          f"{pct_b:.1f}% {model.domain_b_name}")

    print(f"\n  Espacio: {'PCA ' + str(model.pca_dim) + 'D (roll ' + str(model.raw_dim) + 'D crudo)' if model.pca else 'roll ' + str(len(feat)) + 'D'}")

    # Mostrar primeros 10 componentes del feature vector (roll o PCA)
    n_show = min(10, len(feat))
    print(f"\n  Vector (primeros {n_show} componentes):")
    for i in range(n_show):
        print(f"    [{i:>3}]  src={feat[i]:.4f}  →A→B={feat_ab[i]:.4f}  →B→A={feat_ba[i]:.4f}")

    # Pérdida de ciclo
    feat_aba = model.map_features(feat_ab, 'BtoA')
    cl = float(np.linalg.norm(feat_aba - feat))
    print(f"\n  Pérdida de ciclo puntual (A→B→A): {cl:.4f}")

    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CYCLE-GAN STYLE TRANSFER v2 — Transferencia bidireccional (piano roll)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Entrenar modelo CycleGAN')
    p_train.add_argument('--domain-a',        required=True)
    p_train.add_argument('--domain-b',        required=True)
    p_train.add_argument('--model',           default='cycle_gan.pkl')
    p_train.add_argument('--resolution',      type=int,   default=TICKS_PER_BAR,
                         help=f'Ticks por compás [default: {TICKS_PER_BAR}]')
    p_train.add_argument('--pitch-range',     type=int,   default=None, dest='pitch_range',
                         help='Notas MIDI centradas en C4 (None = 128 completo)')
    p_train.add_argument('--disable-roles',   nargs='+',  default=[], dest='disable_roles',
                         choices=ROLES, metavar='ROL')
    p_train.add_argument('--pca-dim',         type=int,   default=0, dest='pca_dim',
                         help='Reducir con PCA (0 = sin PCA)')
    p_train.add_argument('--solver',          default='ridge', choices=['ridge', 'knn'])
    p_train.add_argument('--alpha',           type=float, default=1.0)
    p_train.add_argument('--k-neighbors',     type=int,   default=5)
    p_train.add_argument('--lambda-cycle',    type=float, default=10.0)
    p_train.add_argument('--lambda-identity', type=float, default=5.0)
    p_train.add_argument('--iters',           type=int,   default=100)
    p_train.add_argument('--verbose',         action='store_true')

    # ── transform ─────────────────────────────────────────────────────────────
    p_tf = sub.add_parser('transform', help='Transformar un MIDI con el modelo')
    p_tf.add_argument('input',               help='MIDI de entrada')
    p_tf.add_argument('--model',             required=True)
    p_tf.add_argument('--direction',         default='AtoB', choices=['AtoB', 'BtoA'])
    p_tf.add_argument('--intensity',         type=float, default=1.0)
    p_tf.add_argument('--output',            default=None)
    p_tf.add_argument('--bars',              type=int,   default=None)
    p_tf.add_argument('--candidates',        type=int,   default=3)
    p_tf.add_argument('--no-percussion',     action='store_true')
    p_tf.add_argument('--export-fingerprint',action='store_true')
    p_tf.add_argument('--seed',              type=int,   default=42)
    p_tf.add_argument('--verbose',           action='store_true')

    # ── cycle ─────────────────────────────────────────────────────────────────
    p_cy = sub.add_parser('cycle', help='Verificar consistencia de ciclo A→B→A')
    p_cy.add_argument('input',               help='MIDI de entrada')
    p_cy.add_argument('--model',             required=True)
    p_cy.add_argument('--direction',         default='AtoB', choices=['AtoB', 'BtoA'])
    p_cy.add_argument('--intensity',         type=float, default=1.0)
    p_cy.add_argument('--output-ab',         default=None)
    p_cy.add_argument('--output-aba',        default=None)
    p_cy.add_argument('--bars',              type=int,   default=None)
    p_cy.add_argument('--candidates',        type=int,   default=2)
    p_cy.add_argument('--no-percussion',     action='store_true')
    p_cy.add_argument('--seed',              type=int,   default=42)
    p_cy.add_argument('--verbose',           action='store_true')

    # ── analyze ───────────────────────────────────────────────────────────────
    p_an = sub.add_parser('analyze', help='Analizar posición de un MIDI en el espacio A/B')
    p_an.add_argument('input',    help='MIDI a analizar')
    p_an.add_argument('--model',  required=True)
    p_an.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    {
        'train':     cmd_train,
        'transform': cmd_transform,
        'cycle':     cmd_cycle,
        'analyze':   cmd_analyze,
    }[args.command](args)


if __name__ == '__main__':
    main()
