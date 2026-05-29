#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       cyclegan_composer.py                                  ║
║       Transferencia de estilo musical mediante CycleGAN + Piano Roll        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DESCRIPCIÓN                                                                 ║
║  Aplica la arquitectura CycleGAN (dos generadores G_AB/G_BA + dos           ║
║  discriminadores D_A/D_B) sobre representaciones piano roll de MIDI.        ║
║  Aprende a transformar el estilo musical de un corpus A hacia un corpus B   ║
║  (y viceversa) sin necesidad de pares supervisados.                          ║
║                                                                              ║
║  ARQUITECTURA                                                                ║
║  • Generadores : ResNet 2D (6 bloques residuales) + InstanceNorm            ║
║  • Discriminadores: PatchGAN 70×70 (3 capas de convolución)                 ║
║  • Piano roll  : (N_ROLES, resolution, 128) — mismo formato que             ║
║                  diffusion_composer_v4                                       ║
║  • Condicionamiento de tensión inyectado en cada bloque residual             ║
║    via Adaptive Instance Normalization (AdaIN)                               ║
║                                                                              ║
║  COMANDOS                                                                    ║
║  prepare    MIDI corpus → .npz  (roles + piano roll + tensión)              ║
║  train      Entrena CycleGAN entre corpus A y corpus B                      ║
║  transfer   Transfiere el estilo A→B sobre un MIDI de entrada               ║
║  round-trip MIDI → piano roll → MIDI sin modelo (diagnóstico)               ║
║  inspect    Muestra contenido de un .npz                                    ║
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║  pip install torch mido numpy                                                ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║  # 1. Preparar ambos corpus                                                 ║
║  python cyclegan_composer.py prepare --input-dir midi_jazz/ --output-dir data_A/
║  python cyclegan_composer.py prepare --input-dir midi_clasic/ --output-dir data_B/
║                                                                              ║
║  # 2. Entrenar                                                               ║
║  python cyclegan_composer.py train --data-dir-a data_A/ --data-dir-b data_B/ \\
║      --model-dir model_jazz2classic/ --epochs 200                           ║
║                                                                              ║
║  # 3. Transferir estilo A→B                                                  ║
║  python cyclegan_composer.py transfer --input cancion.mid \                  ║
║      --model-dir model_jazz2classic/ --direction AB --output salida.mid     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse, json, sys, textwrap
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
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

DEFAULT_PALETTE = {
    'melody':        {'program': 73, 'channel': 0, 'velocity': 90},
    'counterpoint':  {'program': 68, 'channel': 1, 'velocity': 80},
    'accompaniment': {'program': 48, 'channel': 2, 'velocity': 70},
    'bass':          {'program': 43, 'channel': 3, 'velocity': 85},
    'percussion':    {'program':  0, 'channel': 9, 'velocity': 90},
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi(path: str):
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


def _ticks_per_bar(mid) -> int:
    return mid.ticks_per_beat * 4


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _adaptive_threshold(roll, pct: float = 90.0) -> float:
    """
    Calcula un umbral de binarización automático a partir del percentil `pct`
    de la distribución de activaciones del piano roll generado.

    A diferencia del umbral fijo, este método se adapta al rango real de
    salida del generador en cada etapa del entrenamiento:
      - Modelo recién iniciado (valores bajos, ~0.05–0.1): umbral ~p90
      - Modelo convergido (valores distribuidos en [0,1]): umbral similar
    Usar p90 mantiene ~10% de celdas activas, densidad musicalmenteusable.
    """
    import numpy as np
    flat = roll.flatten()
    vmax = float(flat.max())
    if vmax < 1e-4:
        return 0.5           # roll vacío: threshold imposible → silencio
    return float(np.percentile(flat, pct))


def _pitch_range(n: int | None):
    if n is None:
        return None
    half = n // 2
    lo   = max(0,   MIDI_CENTER - half)
    hi   = min(127, lo + n - 1)
    lo   = hi - n + 1
    lo   = max(0, lo)
    return (lo, hi)


def _crop_pitch(roll, pitch_lo: int, pitch_hi: int):
    return roll[..., pitch_lo: pitch_hi + 1]


def _pad_pitch(roll, pitch_lo: int, n_full: int = 128):
    import numpy as np
    n_crop     = roll.shape[-1]
    suffix     = n_full - pitch_lo - n_crop
    pad_widths = [(0, 0)] * (roll.ndim - 1) + [(pitch_lo, suffix)]
    return np.pad(roll, pad_widths, mode='constant')


def _fmt_time(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES
# ══════════════════════════════════════════════════════════════════════════════

class RoleAssigner:
    def assign(self, mid) -> dict:
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
            pitches     = [n[2] for n in notes]
            program     = notes[0][4]
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

        score_matrix = {p['key']: {r: score(p, r) for r in remaining_roles} for p in unassigned}
        taken_keys   = set()
        taken_roles  = set()
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
    def __init__(self, resolution: int = TICKS_PER_BAR_DEFAULT,
                 window_bars: int = WINDOW_BARS_DEFAULT):
        self.resolution  = resolution
        self.window_bars = window_bars

    def notes_to_roll(self, notes, tpb_raw, n_bars):
        import numpy as np
        roll = np.zeros((n_bars, self.resolution, PITCH_CLASSES), dtype=np.float32)
        ticks_per_internal = tpb_raw * 4 / self.resolution
        for (start, end, pitch, vel, _) in notes:
            bar_s  = int(start / (tpb_raw * 4))
            tick_s = int((start % (tpb_raw * 4)) / ticks_per_internal)
            bar_e  = int(end   / (tpb_raw * 4))
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
#  EXTRACTOR DE TENSIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TensionExtractor:
    TENSION_DIM = 8

    def extract_bar_vectors(self, role_rolls: dict, bars: int):
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
                mel    = role_rolls['melody'][bar]
                active_per_tick = mel.sum(axis=1)
                changes = float(np.sum(np.diff(active_per_tick) != 0))
                rhythm_density  = changes / max(resolution - 1, 1)
            arousal = 0.5 * min(density * 2, 1.0) + 0.5 * rhythm_density
            vectors[bar] = [tension, density, poly, reg_mean,
                            reg_spread, 0.5, rhythm_density, arousal]
        return vectors

    @staticmethod
    def _lerdahl_proxy(pitches_active) -> float:
        import numpy as np
        if len(pitches_active) < 2:
            return 0.0
        DISSONANT = {1, 2, 6, 10, 11}
        count = 0; pairs = 0
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
#  MIDI OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _rolls_to_midi(bars_per_role: dict, cfg: dict, palette: dict,
                   output_path: str, bpm: float = 120.0,
                   threshold: float = None, adaptive_per_bar: bool = False,
                   threshold_percentile: float = 90.0):
    import mido, numpy as np

    resolution  = cfg['resolution']
    tpb         = 480
    ticks_bar   = tpb * 4
    ticks_tick  = ticks_bar / resolution
    pitch_lo    = cfg.get('pitch_lo', 0)
    pitch_hi    = cfg.get('pitch_hi', 127)
    do_expand   = (pitch_lo, pitch_hi) != (0, 127)

    mid       = mido.MidiFile(ticks_per_beat=tpb)
    tempo_val = int(60_000_000 / bpm)
    t0        = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=tempo_val, time=0))
    mid.tracks.append(t0)

    n_notes_total = 0

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]   # (n_bars, res, n_pitch)

        if do_expand:
            roll = _pad_pitch(roll, pitch_lo, n_full=128)

        thr  = threshold if threshold is not None else _adaptive_threshold(roll, pct=threshold_percentile)
        pal  = palette.get(role, DEFAULT_PALETTE.get(role, {}))
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        if adaptive_per_bar:
            binary = np.zeros_like(roll)
            for b in range(roll.shape[0]):
                bar_slice = roll[b]
                bar_max   = bar_slice.max()
                if bar_max < 1e-4:
                    continue   # compás vacío
                bar_thr = _adaptive_threshold(bar_slice, pct=threshold_percentile) \
                          if threshold is None else threshold
                binary[b] = (bar_slice > bar_thr).astype(np.float32)
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
                    prev = binary[bar, tick - 1, pitch] > 0 if tick > 0 \
                           else (binary[bar - 1, -1, pitch] > 0 if bar > 0 else False)
                    if cur and not prev:
                        events.append((abs_tick, 'on',  pitch))
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
                                          velocity=0, time=delta))
            prev_tick = abs_tick

        remaining = last_tick - prev_tick
        if remaining > 0:
            track.append(mido.MetaMessage('end_of_track', time=remaining))

    mid.save(output_path)
    return n_notes_total


def _load_palette(palette_path: str) -> dict:
    with open(palette_path) as f:
        user = json.load(f)
    palette = {k: dict(v) for k, v in DEFAULT_PALETTE.items()}
    for role, params in user.items():
        palette[role] = {**DEFAULT_PALETTE.get(role, {}), **params}
    return palette


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARACIÓN DE UN MIDI (función de proceso)
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

    assigner        = RoleAssigner()
    converter       = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    extractor       = TensionExtractor()
    role_assignment = assigner.assign(mid)
    if not role_assignment:
        stats_partial['files_skipped'] = 1
        return stem, "sin asignación de roles — omitido", None, stats_partial

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
        min_windows = windows.shape[0] if min_windows is None else min(min_windows, windows.shape[0])

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
    save_dict['meta_json'] = __import__('numpy').array([json.dumps(meta)])
    np.savez_compressed(str(Path(output_dir) / f"{stem}.npz"), **save_dict)

    stats_partial.update({'files_ok': 1, 'total_windows': min_windows})
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, roles: {', '.join(roles_found)})",
            True, stats_partial)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """
    Una muestra: dict con
        'x'       : Tensor (N_ROLES, resolution, n_pitch)
        'tension' : Tensor (TENSION_DIM,)
    """
    def __init__(self, data_dir: str, roles: list = None):
        import numpy as np
        self.samples  = []
        self.roles    = roles or ROLES
        self.n_roles  = len(self.roles)
        self._cache   = {}
        self.n_pitch  = None

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
                    # Tomar el último compás de cada ventana
                    self.samples.append((str(path), i, meta))
                self._cache[str(path)] = data
            except Exception:
                continue

        if self.n_pitch is None:
            self.n_pitch = PITCH_CLASSES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np, torch

        path, widx, meta = self.samples[idx]
        data             = self._cache[path]
        resolution       = meta['resolution']
        n_pitch          = self.n_pitch

        x_parts = []
        mask    = []
        for role in self.roles:
            key = f'roll_{role}'
            if key in data:
                window = data[key][widx]     # (window_bars, resolution, n_pitch)
                x_parts.append(window[-1])   # último compás
                mask.append(True)
            else:
                x_parts.append(np.zeros((resolution, n_pitch), dtype=np.float32))
                mask.append(False)

        x        = torch.tensor(np.stack(x_parts, axis=0))   # (N_ROLES, res, n_pitch)
        tension  = torch.tensor(data['tension'][widx])
        role_mask = torch.tensor(mask, dtype=torch.bool)
        return {'x': x, 'tension': tension, 'role_mask': role_mask}


def _collate(batch):
    import torch
    return {
        'x':         torch.stack([b['x']        for b in batch]),
        'tension':   torch.stack([b['tension']  for b in batch]),
        'role_mask': torch.stack([b['role_mask'] for b in batch]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA CYCLEGAN
#
#  Generadores: ResNet 2D con 6 bloques residuales + AdaIN de tensión
#  Discriminadores: PatchGAN 70×70 (4 capas Conv → parche de realidad/falsedad)
#
#  El piano roll tiene forma (B, N_ROLES, resolution, n_pitch).
#  Cada generador recibe además un vector de tensión (B, TENSION_DIM) que
#  modula las capas normales vía Adaptive Instance Normalization.
# ══════════════════════════════════════════════════════════════════════════════

def _build_cyclegan(n_roles: int, resolution: int, n_pitch: int,
                    tension_dim: int, base_ch: int = 64,
                    n_res_blocks: int = 6):
    """
    Devuelve un nn.Module con:
        .G_AB   Generador A→B
        .G_BA   Generador B→A
        .D_A    Discriminador de dominio A
        .D_B    Discriminador de dominio B

    Entrada de generadores: (B, N_ROLES, resolution, n_pitch) + (B, TENSION_DIM)
    Entrada de discriminadores: (B, N_ROLES, resolution, n_pitch)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── AdaIN: modula mean/std de canales según vector de tensión ───────────
    class AdaIN(nn.Module):
        """
        Adaptive Instance Normalization condicionada en el vector de tensión.
        Aplica IN al activación y luego le da scale/shift aprendidos desde
        el vector de condicionamiento.
        """
        def __init__(self, n_ch: int, cond_dim: int):
            super().__init__()
            self.norm  = nn.InstanceNorm2d(n_ch, affine=False)
            self.scale = nn.Linear(cond_dim, n_ch)
            self.shift = nn.Linear(cond_dim, n_ch)

        def forward(self, x, cond):
            # x: (B, C, H, W)  cond: (B, cond_dim)
            h     = self.norm(x)
            scale = self.scale(cond).unsqueeze(-1).unsqueeze(-1)   # (B,C,1,1)
            shift = self.shift(cond).unsqueeze(-1).unsqueeze(-1)
            return h * (1 + scale) + shift

    # ── Bloque residual con AdaIN ────────────────────────────────────────────
    class ResBlock(nn.Module):
        def __init__(self, n_ch: int, cond_dim: int):
            super().__init__()
            self.pad1  = nn.ReflectionPad2d(1)
            self.conv1 = nn.Conv2d(n_ch, n_ch, 3)
            self.ada1  = AdaIN(n_ch, cond_dim)
            self.pad2  = nn.ReflectionPad2d(1)
            self.conv2 = nn.Conv2d(n_ch, n_ch, 3)
            self.ada2  = AdaIN(n_ch, cond_dim)

        def forward(self, x, cond):
            h = F.relu(self.ada1(self.conv1(self.pad1(x)), cond), inplace=True)
            h = self.ada2(self.conv2(self.pad2(h)), cond)
            return x + h

    # ── Generador ResNet ─────────────────────────────────────────────────────
    class Generator(nn.Module):
        """
        Arquitectura:
          Stem (7×7 ReflectionPad) → Downsample ×2 → ResBlocks ×n_res_blocks
          → Upsample ×2 → Head (7×7 ReflectionPad + Tanh)
        Cada ResBlock recibe el vector de tensión vía AdaIN.
        La salida está en [-1, 1]; se convierte a [0,1] con (out+1)/2.
        """
        def __init__(self):
            super().__init__()
            c0, c1, c2 = base_ch, base_ch * 2, base_ch * 4
            in_ch = n_roles

            # Stem
            self.stem = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, c0, 7),
                nn.InstanceNorm2d(c0, affine=True),
                nn.ReLU(True),
            )
            # Encoder (downsampling)
            self.enc = nn.Sequential(
                nn.Conv2d(c0, c1, 3, stride=2, padding=1),
                nn.InstanceNorm2d(c1, affine=True),
                nn.ReLU(True),
                nn.Conv2d(c1, c2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(c2, affine=True),
                nn.ReLU(True),
            )
            # Bloques residuales con condicionamiento AdaIN
            self.res_blocks = nn.ModuleList(
                [ResBlock(c2, tension_dim) for _ in range(n_res_blocks)]
            )
            # Decoder (upsampling)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c1, affine=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(c1, c0, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c0, affine=True),
                nn.ReLU(True),
            )
            # Head
            self.head = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(c0, n_roles, 7),
                nn.Tanh(),
            )

        def forward(self, x, tension):
            # x: (B, N_ROLES, res, n_pitch)  tension: (B, TENSION_DIM)
            h = self.stem(x)
            h = self.enc(h)
            for blk in self.res_blocks:
                h = blk(h, tension)
            h = self.dec(h)
            return self.head(h)   # (B, N_ROLES, res, n_pitch) en [-1,1]

    # ── Discriminador PatchGAN ───────────────────────────────────────────────
    class Discriminator(nn.Module):
        """
        PatchGAN: clasifica si parches locales del piano roll son reales o falsos.
        4 capas de Conv(stride=2) + Conv final → mapa de predicciones.
        """
        def __init__(self):
            super().__init__()

            def block(in_c, out_c, norm=True):
                layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
                if norm:
                    layers.append(nn.InstanceNorm2d(out_c, affine=True))
                layers.append(nn.LeakyReLU(0.2, True))
                return layers

            self.model = nn.Sequential(
                *block(n_roles, base_ch,     norm=False),
                *block(base_ch,   base_ch*2),
                *block(base_ch*2, base_ch*4),
                *block(base_ch*4, base_ch*8),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(base_ch*8, 1, 4, padding=1),
            )

        def forward(self, x):
            return self.model(x)

    # ── Módulo raíz CycleGAN ─────────────────────────────────────────────────
    class CycleGAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.G_AB = Generator()
            self.G_BA = Generator()
            self.D_A  = Discriminator()
            self.D_B  = Discriminator()

        def generate_AB(self, x_a, tension):
            """Genera piano roll dominio B a partir de dominio A."""
            raw = self.G_AB(x_a, tension)
            return (raw + 1.0) / 2.0   # [-1,1] → [0,1]

        def generate_BA(self, x_b, tension):
            """Genera piano roll dominio A a partir de dominio B."""
            raw = self.G_BA(x_b, tension)
            return (raw + 1.0) / 2.0

        def _to_g_input(self, x):
            """Convierte [0,1] → [-1,1] para la entrada del generador."""
            return x * 2.0 - 1.0

        def generator_loss(self, x_a, x_b, tension,
                           lambda_cycle: float = 10.0,
                           lambda_identity: float = 0.5):
            """
            Pérdida completa del generador (se usa solo para entrenar G_AB y G_BA):
              L_GAN(G_AB) + L_GAN(G_BA)
            + lambda_cycle * (L_cycle_A + L_cycle_B)
            + lambda_identity * (L_id_A + L_id_B)
            """
            import torch

            xa_g = self._to_g_input(x_a)
            xb_g = self._to_g_input(x_b)

            # Generar
            fake_b_raw = self.G_AB(xa_g, tension)
            fake_a_raw = self.G_BA(xb_g, tension)

            # GAN loss (LS-GAN: MSE contra unos)
            loss_gan_ab = F.mse_loss(self.D_B(fake_b_raw), torch.ones_like(self.D_B(fake_b_raw)))
            loss_gan_ba = F.mse_loss(self.D_A(fake_a_raw), torch.ones_like(self.D_A(fake_a_raw)))

            # Cycle consistency: A→B→A y B→A→B
            rec_a_raw = self.G_BA(fake_b_raw, tension)
            rec_b_raw = self.G_AB(fake_a_raw, tension)
            loss_cycle_a = F.l1_loss(rec_a_raw, xa_g)
            loss_cycle_b = F.l1_loss(rec_b_raw, xb_g)

            # Identity: G_AB(B) ≈ B, G_BA(A) ≈ A  (estabiliza colores/tonos)
            id_b_raw = self.G_AB(xb_g, tension)
            id_a_raw = self.G_BA(xa_g, tension)
            loss_id_a = F.l1_loss(id_a_raw, xa_g)
            loss_id_b = F.l1_loss(id_b_raw, xb_g)

            total = (loss_gan_ab + loss_gan_ba
                     + lambda_cycle  * (loss_cycle_a + loss_cycle_b)
                     + lambda_identity * (loss_id_a + loss_id_b))

            return total, {
                'gan_AB':   loss_gan_ab.item(),
                'gan_BA':   loss_gan_ba.item(),
                'cycle_A':  loss_cycle_a.item(),
                'cycle_B':  loss_cycle_b.item(),
                'id_A':     loss_id_a.item(),
                'id_B':     loss_id_b.item(),
            }

        def discriminator_loss(self, x_a, x_b, tension,
                               fake_a=None, fake_b=None):
            """
            Pérdida de los discriminadores D_A y D_B (LS-GAN).
            Se pasan fake_a y fake_b precalculados (del replay buffer).
            """
            import torch

            xa_g = self._to_g_input(x_a)
            xb_g = self._to_g_input(x_b)

            with torch.no_grad():
                if fake_b is None:
                    fake_b = self.G_AB(xa_g, tension)
                if fake_a is None:
                    fake_a = self.G_BA(xb_g, tension)

            # D_B: real B → 1,  fake B → 0
            loss_db = 0.5 * (
                F.mse_loss(self.D_B(xb_g),   torch.ones_like(self.D_B(xb_g))) +
                F.mse_loss(self.D_B(fake_b.detach()), torch.zeros_like(self.D_B(fake_b.detach())))
            )
            # D_A: real A → 1, fake A → 0
            loss_da = 0.5 * (
                F.mse_loss(self.D_A(xa_g),   torch.ones_like(self.D_A(xa_g))) +
                F.mse_loss(self.D_A(fake_a.detach()), torch.zeros_like(self.D_A(fake_a.detach())))
            )
            return loss_da + loss_db, {'D_A': loss_da.item(), 'D_B': loss_db.item()}

        @torch.no_grad()
        def transfer(self, x_a, tension, direction: str = 'AB'):
            """Inferencia: transfiere un piano roll de dominio A a B (o B→A)."""
            xa_g = self._to_g_input(x_a)
            if direction == 'AB':
                raw = self.G_AB(xa_g, tension)
            else:
                raw = self.G_BA(xa_g, tension)
            return ((raw + 1.0) / 2.0).clamp(0, 1)

    return CycleGAN()


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER  (estabiliza el entrenamiento GAN)
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Buffer de historial de imágenes generadas (capacity=50).
    Devuelve con probabilidad 0.5 un elemento aleatorio del buffer
    en lugar del elemento recién generado, evitando oscilaciones.
    """
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer   = []

    def push_and_sample(self, items):
        import torch, random
        out = []
        for item in items:
            item_cpu = item.detach().cpu()
            if len(self.buffer) < self.capacity:
                self.buffer.append(item_cpu)
                out.append(item)
            else:
                if random.random() > 0.5:
                    idx = random.randrange(self.capacity)
                    old = self.buffer[idx].to(item.device)
                    self.buffer[idx] = item_cpu
                    out.append(old)
                else:
                    out.append(item)
        return torch.stack(out)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANTrainer:
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'
    CONFIG_NAME     = 'model_config.json'

    def __init__(self, model, opt_g, opt_d, model_dir: Path,
                 patience: int = 50,
                 lambda_cycle: float = 10.0,
                 lambda_identity: float = 0.5):
        self.model    = model
        self.opt_g    = opt_g
        self.opt_d    = opt_d
        self.model_dir = model_dir
        self.patience  = patience
        self.lambda_cycle    = lambda_cycle
        self.lambda_identity = lambda_identity

        self.history       = {'loss_G': [], 'loss_D': [], 'cycle_A': [], 'cycle_B': []}
        self.best_loss_g   = float('inf')
        self.no_improve    = 0
        self.start_epoch   = 0
        self._resume       = False

    def save_checkpoint(self, epoch: int, loss_g: float, is_best: bool):
        import torch
        state = {
            'epoch':      epoch,
            'model':      self.model.state_dict(),
            'opt_g':      self.opt_g.state_dict(),
            'opt_d':      self.opt_d.state_dict(),
            'best_loss_g': self.best_loss_g,
            'no_improve':  self.no_improve,
            'history':     self.history,
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
        self.model.load_state_dict(state['model'])
        self.opt_g.load_state_dict(state['opt_g'])
        self.opt_d.load_state_dict(state['opt_d'])
        self.best_loss_g = state['best_loss_g']
        self.no_improve  = state['no_improve']
        self.history     = state['history']
        self.start_epoch = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_epoch}  "
              f"(mejor loss_G={self.best_loss_g:.4f})")

    def train(self, loader_a, loader_b, n_epochs: int):
        import torch, time, math

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self._resume:
            self.load_checkpoint()

        # Schedulers lineales: LR constante hasta epoch 100, luego decae a 0
        def lr_lambda(epoch):
            decay_start = max(n_epochs // 2, 1)
            if epoch < decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start) / max(n_epochs - decay_start, 1))

        sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, lr_lambda)
        sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda)

        buf_a = ReplayBuffer()
        buf_b = ReplayBuffer()

        print(f"\n{'═'*64}")
        print(f"  CYCLEGAN COMPOSER — Entrenamiento")
        print(f"  Épocas máx. : {n_epochs}   Early stopping: {self.patience} sin mejora")
        print(f"  Dispositivo : {device}")
        print(f"  λ_cycle={self.lambda_cycle}  λ_id={self.lambda_identity}")
        print(f"{'═'*64}\n")

        epoch_times = []
        train_start = time.time()

        for epoch in range(self.start_epoch, n_epochs):
            if epoch_times:
                avg_ep  = sum(epoch_times) / len(epoch_times)
                eta_str = f"  ETA {_fmt_time(avg_ep * (n_epochs - epoch))}"
            else:
                eta_str = ""

            lr_cur = self.opt_g.param_groups[0]['lr']
            print(f"  Época {epoch+1:>4}/{n_epochs}  lr={lr_cur:.2e}{eta_str}",
                  flush=True)

            epoch_t0 = time.time()
            self.model.train()

            sum_g = sum_d = sum_ca = sum_cb = n_b = 0

            iter_b = iter(loader_b)
            for batch_a in loader_a:
                try:
                    batch_b = next(iter_b)
                except StopIteration:
                    iter_b  = iter(loader_b)
                    batch_b = next(iter_b)

                x_a     = batch_a['x'].to(device)
                x_b     = batch_b['x'].to(device)
                tension = batch_a['tension'].to(device)

                # ── Entrenar generadores ─────────────────────────────────────
                self.opt_g.zero_grad()
                loss_g, metrics_g = self.model.generator_loss(
                    x_a, x_b, tension,
                    lambda_cycle    = self.lambda_cycle,
                    lambda_identity = self.lambda_identity,
                )
                if math.isnan(loss_g.item()):
                    continue
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.G_AB.parameters()) +
                    list(self.model.G_BA.parameters()), 1.0)
                self.opt_g.step()

                # ── Generar fakes (con detach) para discriminadores ──────────
                with torch.no_grad():
                    xa_g   = self.model._to_g_input(x_a)
                    xb_g   = self.model._to_g_input(x_b)
                    fake_b = self.model.G_AB(xa_g, tension)
                    fake_a = self.model.G_BA(xb_g, tension)
                fake_b = buf_b.push_and_sample(list(fake_b))
                fake_a = buf_a.push_and_sample(list(fake_a))

                # ── Entrenar discriminadores ─────────────────────────────────
                self.opt_d.zero_grad()
                loss_d, metrics_d = self.model.discriminator_loss(
                    x_a, x_b, tension, fake_a=fake_a, fake_b=fake_b)
                if math.isnan(loss_d.item()):
                    continue
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.D_A.parameters()) +
                    list(self.model.D_B.parameters()), 1.0)
                self.opt_d.step()

                sum_g  += loss_g.item()
                sum_d  += loss_d.item()
                sum_ca += metrics_g['cycle_A']
                sum_cb += metrics_g['cycle_B']
                n_b    += 1

                print(f"\r    batch {n_b:>4}  "
                      f"G={sum_g/n_b:.4f}  D={sum_d/n_b:.4f}  "
                      f"cycA={sum_ca/n_b:.4f}  cycB={sum_cb/n_b:.4f}   ",
                      end='', flush=True)

            print(' ' * 80, end='\r')

            sched_g.step()
            sched_d.step()

            avg_g = sum_g / max(n_b, 1)
            avg_d = sum_d / max(n_b, 1)
            avg_ca = sum_ca / max(n_b, 1)
            avg_cb = sum_cb / max(n_b, 1)

            self.history['loss_G'].append(avg_g)
            self.history['loss_D'].append(avg_d)
            self.history['cycle_A'].append(avg_ca)
            self.history['cycle_B'].append(avg_cb)

            is_best = avg_g < self.best_loss_g
            if is_best:
                self.best_loss_g = avg_g
                self.no_improve  = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, avg_g, is_best)

            elapsed = time.time() - epoch_t0
            epoch_times.append(elapsed)
            if len(epoch_times) > 5:
                epoch_times.pop(0)

            best_mark = ' ◀ mejor' if is_best else ''
            stop_str  = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                         if self.no_improve > 0 else '')
            print(f"         G={avg_g:.4f}  D={avg_d:.4f}  "
                  f"cycA={avg_ca:.4f}  cycB={avg_cb:.4f}  "
                  f"{_fmt_time(elapsed)}/época{best_mark}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        total = time.time() - train_start
        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(total)}.")
        print(f"  Mejor loss_G : {self.best_loss_g:.4f}")
        print(f"  Modelos en   : {self.model_dir}")
        print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_and_config(model_dir: Path):
    import torch

    cfg_path   = model_dir / CycleGANTrainer.CONFIG_NAME
    model_path = model_dir / CycleGANTrainer.BEST_NAME
    if not cfg_path.exists():
        print(f"[ERROR] No se encontró {cfg_path}. ¿Ha entrenado el modelo?")
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] No se encontró {model_path}. ¿Ha terminado el entrenamiento?")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = _build_cyclegan(
        n_roles      = cfg['n_roles'],
        resolution   = cfg['resolution'],
        n_pitch      = cfg['n_pitch'],
        tension_dim  = cfg['tension_dim'],
        base_ch      = cfg.get('base_ch', 64),
        n_res_blocks = cfg.get('n_res_blocks', 6),
    )
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI → PIANO ROLL (inferencia de un solo archivo)
# ══════════════════════════════════════════════════════════════════════════════

def _midi_to_rolls(midi_path: str, cfg: dict) -> tuple[dict, int]:
    """
    Carga un MIDI y devuelve (role_rolls, total_bars).
    role_rolls: {role: np.ndarray (n_bars, resolution, n_pitch)}
    """
    import numpy as np

    mid = _load_midi(midi_path)
    note_lists = _extract_note_lists(mid)
    if not note_lists:
        print(f"[transfer] ADVERTENCIA: {midi_path} no tiene notas.")
        return {}, 0

    roles      = cfg.get('roles', ROLES)
    resolution = cfg['resolution']
    pitch_lo   = cfg.get('pitch_lo', 0)
    pitch_hi   = cfg.get('pitch_hi', 127)
    n_pitch    = cfg['n_pitch']

    assigner   = RoleAssigner()
    converter  = PianoRollConverter(resolution=resolution, window_bars=1)
    tpb_raw    = _ticks_per_bar(mid)
    all_ticks  = max((n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpb_raw) + 1)

    role_assignment = assigner.assign(mid)
    role_rolls = {}
    for role, key in role_assignment.items():
        if role not in roles:
            continue
        notes = note_lists.get(key, [])
        if not notes:
            continue
        roll = converter.notes_to_roll(notes, tpb_raw, total_bars)
        if (pitch_lo, pitch_hi) != (0, 127):
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        role_rolls[role] = roll   # (n_bars, resolution, n_pitch)

    return role_rolls, total_bars


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: prepare
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]

    pitch_n  = getattr(args, 'pitch_range', None)
    pr       = _pitch_range(pitch_n)
    pitch_lo = pr[0] if pr else None
    pitch_hi = pr[1] if pr else None

    if disabled:
        print(f"[prepare] Roles deshabilitados : {', '.join(sorted(disabled))}")
        print(f"[prepare] Roles activos        : {', '.join(active_roles)}")
    if pr:
        n_p = pitch_hi - pitch_lo + 1
        print(f"[prepare] Rango de pitch       : {pitch_n} notas  "
              f"(MIDI {pitch_lo}–{pitch_hi}, n_pitch={n_p})")

    midi_files = sorted(list(input_dir.glob('*.mid')) +
                        list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    n_workers = min(multiprocessing.cpu_count(), len(midi_files))
    print(f"[prepare] {len(midi_files)} archivos MIDI encontrados")
    print(f"[prepare] Resolución: {args.resolution} ticks/compás  |  "
          f"Ventana: {args.window_bars} compases")
    print(f"[prepare] Paralelizando con {n_workers} procesos\n")

    task_args = [
        (midi_path, str(output_dir), args.resolution, args.window_bars,
         active_roles, pitch_lo, pitch_hi)
        for midi_path in midi_files
    ]

    stats = {r: 0 for r in ROLES}
    stats.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prepare_one_midi, a): a[0] for a in task_args}
        for future in futures:
            midi_path = futures[future]
            try:
                stem, msg, ok, partial = future.result()
            except Exception as e:
                print(f"  [{Path(midi_path).stem}] EXCEPCIÓN: {e}")
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


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader

    dir_a     = Path(args.data_dir_a)
    dir_b     = Path(args.data_dir_b)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]

    print("[train] Cargando datasets ...")
    ds_a = MidiRollDataset(str(dir_a), roles=active_roles)
    ds_b = MidiRollDataset(str(dir_b), roles=active_roles)
    print(f"[train] Corpus A: {len(ds_a)} muestras  |  Corpus B: {len(ds_b)} muestras")

    loader_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    loader_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=0)

    sample      = ds_a[0]
    n_roles     = sample['x'].shape[0]
    resolution  = sample['x'].shape[1]
    n_pitch     = ds_a.n_pitch
    tension_dim = sample['tension'].shape[0]

    import numpy as _np
    _first_npz = sorted(dir_a.glob('*.npz'))[0]
    _meta      = json.loads(str(_np.load(str(_first_npz), allow_pickle=True)['meta_json'][0]))
    pitch_lo   = _meta.get('pitch_lo', 0)
    pitch_hi   = _meta.get('pitch_hi', 127)

    print(f"[train] n_roles={n_roles}  resolution={resolution}  "
          f"n_pitch={n_pitch}  tension_dim={tension_dim}")
    print(f"[train] base_ch={args.base_ch}  n_res_blocks={args.n_res_blocks}  "
          f"lambda_cycle={args.lambda_cycle}  lambda_identity={args.lambda_identity}")

    model = _build_cyclegan(
        n_roles      = n_roles,
        resolution   = resolution,
        n_pitch      = n_pitch,
        tension_dim  = tension_dim,
        base_ch      = args.base_ch,
        n_res_blocks = args.n_res_blocks,
    )

    # Dos optimizadores independientes: uno para G, uno para D
    opt_g = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    cfg = {
        'n_roles':       n_roles,
        'roles':         active_roles,
        'resolution':    resolution,
        'n_pitch':       n_pitch,
        'pitch_lo':      pitch_lo,
        'pitch_hi':      pitch_hi,
        'tension_dim':   tension_dim,
        'base_ch':       args.base_ch,
        'n_res_blocks':  args.n_res_blocks,
        'lambda_cycle':  args.lambda_cycle,
        'lambda_identity': args.lambda_identity,
        'model_version': 'cyclegan_v1',
    }
    with open(model_dir / CycleGANTrainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer = CycleGANTrainer(
        model, opt_g, opt_d, model_dir,
        patience         = args.patience,
        lambda_cycle     = args.lambda_cycle,
        lambda_identity  = args.lambda_identity,
    )
    trainer._resume = args.resume
    trainer.train(loader_a, loader_b, args.epochs)


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: transfer
# ══════════════════════════════════════════════════════════════════════════════

def cmd_transfer(args):
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[transfer] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"[transfer] Cargando MIDI: {args.input}")
    role_rolls, total_bars = _midi_to_rolls(args.input, cfg)
    if not role_rolls:
        print("[transfer] ERROR: no se pudo extraer ningún piano roll.")
        sys.exit(1)

    # Calcular vectores de tensión para todos los compases
    extractor   = TensionExtractor()
    tension_arr = extractor.extract_bar_vectors(role_rolls, total_bars)
    # tension_arr: (total_bars, TENSION_DIM)

    roles      = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg['n_pitch']
    n_roles    = cfg['n_roles']

    # Apilar roles en un tensor (total_bars, N_ROLES, resolution, n_pitch)
    roll_stack = np.zeros((total_bars, n_roles, resolution, n_pitch), dtype=np.float32)
    for ri, role in enumerate(roles):
        if role in role_rolls:
            rr = role_rolls[role]
            n  = min(rr.shape[0], total_bars)
            roll_stack[:n, ri] = rr[:n]

    print(f"[transfer] {total_bars} compases  |  dirección: {args.direction}  |  "
          f"batch_size: {args.batch_size}")

    # Procesar compás a compás en batches
    out_stack = np.zeros_like(roll_stack)
    bs = args.batch_size

    for start in range(0, total_bars, bs):
        end    = min(start + bs, total_bars)
        x_np   = roll_stack[start:end]
        t_np   = tension_arr[start:end]

        x_t    = torch.tensor(x_np).to(device)
        t_t    = torch.tensor(t_np).to(device)

        y_t    = model.transfer(x_t, t_t, direction=args.direction)
        out_stack[start:end] = y_t.cpu().numpy()

        print(f"\r  Compases {end}/{total_bars}   ", end='', flush=True)

    print()

    # ── Diagnóstico de calibración del umbral ────────────────────────────────
    flat     = out_stack.flatten()
    vmin, vmax, vmean = float(flat.min()), float(flat.max()), float(flat.mean())
    thr_pct  = args.threshold_percentile
    auto_thr = _adaptive_threshold(out_stack, pct=thr_pct)
    used_thr = args.threshold if args.threshold is not None else auto_thr
    frac_on  = float((flat > used_thr).mean())
    print(f"[transfer] Salida del generador — min={vmin:.4f}  max={vmax:.4f}  "
          f"mean={vmean:.4f}")
    print(f"[transfer] Umbral {'fijo' if args.threshold is not None else f'auto (p{thr_pct:.0f})'}"
          f" = {used_thr:.5f}  →  {frac_on*100:.1f}% celdas activas"
          + ("  ⚠ modelo poco convergido, considera más épocas"
             if vmax < 0.2 and args.threshold is None else ""))

    # Convertir de vuelta a dict {role: (n_bars, res, n_pitch)}
    bars_per_role = {}
    for ri, role in enumerate(roles):
        bars_per_role[role] = out_stack[:, ri]

    # Paleta de instrumentos
    palette = DEFAULT_PALETTE
    if args.palette:
        palette = _load_palette(args.palette)

    n_notes = _rolls_to_midi(
        bars_per_role, cfg, palette,
        output_path          = args.output,
        bpm                  = args.bpm,
        threshold            = args.threshold,
        adaptive_per_bar     = args.adaptive_per_bar,
        threshold_percentile = args.threshold_percentile,
    )
    print(f"[transfer] Guardado: {args.output}  ({n_notes} notas, {total_bars} compases)")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: round-trip
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    import numpy as np

    cfg_override = {
        'resolution': args.resolution,
        'roles':      ROLES,
        'pitch_lo':   0,
        'pitch_hi':   127,
        'n_pitch':    PITCH_CLASSES,
    }

    if args.model_dir:
        model_dir = Path(args.model_dir)
        cfg_path  = model_dir / CycleGANTrainer.CONFIG_NAME
        if cfg_path.exists():
            with open(cfg_path) as f:
                loaded = json.load(f)
            cfg_override.update(loaded)
            print(f"[round-trip] Config leída de {cfg_path}")

    palette = DEFAULT_PALETTE
    if args.palette:
        palette = _load_palette(args.palette)

    print(f"[round-trip] Cargando: {args.input}")
    role_rolls, total_bars = _midi_to_rolls(args.input, cfg_override)
    if not role_rolls:
        print("[round-trip] No se encontraron notas.")
        sys.exit(1)

    print(f"[round-trip] {total_bars} compases, roles: {list(role_rolls.keys())}")

    n_notes = _rolls_to_midi(
        role_rolls, cfg_override, palette,
        output_path = args.output,
        bpm         = args.bpm,
    )
    print(f"[round-trip] Guardado: {args.output}  ({n_notes} notas)")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    import numpy as np

    if 'npz' in args.what:
        data_dir = Path(args.data_dir) if args.data_dir else Path('.')
        if args.npz_file:
            npz_files = [data_dir / args.npz_file]
        else:
            npz_files = sorted(data_dir.glob('*.npz'))[:5]

        for path in npz_files:
            if not path.exists():
                print(f"[inspect] No encontrado: {path}")
                continue
            data = dict(np.load(str(path), allow_pickle=True))
            meta = json.loads(str(data['meta_json'][0]))
            print(f"\n{'─'*56}")
            print(f"  {path.name}")
            print(f"{'─'*56}")
            print(f"  source      : {meta['source']}")
            print(f"  resolution  : {meta['resolution']}")
            print(f"  window_bars : {meta['window_bars']}")
            print(f"  total_bars  : {meta['total_bars']}")
            print(f"  n_windows   : {meta['n_windows']}")
            print(f"  n_pitch     : {meta.get('n_pitch', 128)}  "
                  f"(MIDI {meta.get('pitch_lo',0)}–{meta.get('pitch_hi',127)})")
            print(f"  roles       : {meta['roles']}")
            print(f"  tensión     : {data['tension'].shape}  "
                  f"mean={data['tension'].mean():.3f}")
            for role in ROLES:
                key = f'roll_{role}'
                if key in data:
                    r = data[key]
                    density = r.mean() * 100
                    print(f"  roll_{role:<16} {r.shape}  "
                          f"densidad={density:.2f}%")

    if 'model' in args.what and args.model_dir:
        model_dir = Path(args.model_dir)
        cfg_path  = model_dir / CycleGANTrainer.CONFIG_NAME
        hist_path = model_dir / CycleGANTrainer.HISTORY_NAME
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"\n{'─'*56}")
            print(f"  Configuración del modelo")
            print(f"{'─'*56}")
            for k, v in cfg.items():
                print(f"  {k:<20} {v}")
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            n = len(hist.get('loss_G', []))
            if n > 0:
                print(f"\n  Historial: {n} épocas")
                print(f"  Última    loss_G={hist['loss_G'][-1]:.4f}  "
                      f"loss_D={hist['loss_D'][-1]:.4f}")
                print(f"  Mejor     loss_G={min(hist['loss_G']):.4f}  "
                      f"(época {hist['loss_G'].index(min(hist['loss_G']))+1})")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='cyclegan_composer',
        description='Transferencia de estilo musical mediante CycleGAN + Piano Roll',
    )
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p_prep = sub.add_parser('prepare',
        help='Convierte un corpus MIDI en .npz (piano roll + tensión)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Prepara un corpus de archivos MIDI para el entrenamiento.
            Cada MIDI se divide en ventanas de compases y se guarda como .npz.

            Ejecutar una vez para cada corpus (A y B) antes de entrenar:
              prepare --input-dir midi_jazz/    --output-dir data_jazz/
              prepare --input-dir midi_clasico/ --output-dir data_clasico/
        """))
    p_prep.add_argument('--input-dir',   required=True, metavar='DIR')
    p_prep.add_argument('--output-dir',  required=True, metavar='DIR')
    p_prep.add_argument('--resolution',  type=int, default=TICKS_PER_BAR_DEFAULT, metavar='INT',
        help=f'Ticks por compás (default: {TICKS_PER_BAR_DEFAULT})')
    p_prep.add_argument('--window-bars', type=int, default=WINDOW_BARS_DEFAULT, metavar='INT',
        help=f'Compases por ventana (default: {WINDOW_BARS_DEFAULT})')
    p_prep.add_argument('--disable-roles', nargs='+', metavar='ROL',
        choices=ROLES, default=[], dest='disable_roles',
        help=f'Roles a excluir. Posibles: {", ".join(ROLES)}')
    p_prep.add_argument('--pitch-range', type=int, default=None, metavar='N',
        dest='pitch_range',
        help='Limitar a N valores MIDI centrados en Do central (MIDI 60). '
             'Ej: --pitch-range 48 → MIDI 36–83')
    p_prep.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train',
        help='Entrena el CycleGAN entre corpus A y corpus B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Entrena dos generadores G_AB (A→B) y G_BA (B→A) junto con sus
            discriminadores D_A y D_B mediante pérdidas GAN + cycle + identity.

            Ejemplo:
              train --data-dir-a data_jazz/ --data-dir-b data_clasico/ \\
                    --model-dir model_jazz2clasico/ --epochs 200
        """))
    p_train.add_argument('--data-dir-a',  required=True, metavar='DIR',
        help='Corpus A (preprocesado con prepare)')
    p_train.add_argument('--data-dir-b',  required=True, metavar='DIR',
        help='Corpus B (preprocesado con prepare)')
    p_train.add_argument('--model-dir',   required=True, metavar='DIR')
    p_train.add_argument('--epochs',      type=int,   default=200)
    p_train.add_argument('--batch-size',  type=int,   default=8)
    p_train.add_argument('--lr',          type=float, default=2e-4)
    p_train.add_argument('--base-ch',     type=int,   default=64, metavar='INT',
        help='Canales base del generador/discriminador (default: 64). '
             'Bajar a 32 para reducir memoria en piano rolls grandes.')
    p_train.add_argument('--n-res-blocks', type=int,  default=6, metavar='INT',
        dest='n_res_blocks',
        help='Bloques residuales en el generador (default: 6; paper usa 9 para 256px)')
    p_train.add_argument('--lambda-cycle', type=float, default=10.0, metavar='FLOAT',
        dest='lambda_cycle',
        help='Peso de la pérdida de consistencia cíclica (default: 10.0)')
    p_train.add_argument('--lambda-identity', type=float, default=0.5, metavar='FLOAT',
        dest='lambda_identity',
        help='Peso de la pérdida de identidad (default: 0.5). '
             'Poner a 0 para desactivarla.')
    p_train.add_argument('--patience',    type=int,   default=50)
    p_train.add_argument('--resume',      action='store_true',
        help='Reanudar entrenamiento desde el último checkpoint')
    p_train.add_argument('--disable-roles', nargs='+', metavar='ROL',
        choices=ROLES, default=[], dest='disable_roles',
        help=f'Roles a excluir del entrenamiento. Debe coincidir con --disable-roles '
             f'usado en prepare.')
    p_train.set_defaults(func=cmd_train)

    # ── transfer ──────────────────────────────────────────────────────────────
    p_tr = sub.add_parser('transfer',
        help='Transfiere el estilo A→B (o B→A) sobre un MIDI de entrada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Aplica el generador entrenado compás a compás sobre un MIDI de entrada.

            Direcciones:
              AB  Transforma del dominio A al dominio B (aprendido como corpus A→B)
              BA  Transforma del dominio B al dominio A (generador inverso)

            El vector de tensión se calcula automáticamente del MIDI de entrada
            y se usa para condicionar el generador en cada compás.

            Umbral de binarización:
              Por defecto se auto-calibra al percentil 90 de la distribución de
              activación del generador (--threshold-percentile 90). Esto hace que
              el umbral se adapte al rango real de salida en cualquier etapa del
              entrenamiento, sin necesidad de conocer ese rango a priori.

              Con modelos poco entrenados (max < 0.2) el comando muestra un aviso
              y sugiere más épocas de entrenamiento.

            Ejemplos:
              transfer --input cancion_jazz.mid --model-dir model_jazz2clasico/ \\
                       --direction AB --output cancion_clasica.mid

              # Más notas (umbral más bajo):
              transfer --input cancion.mid --model-dir model/ \\
                       --threshold-percentile 80 --adaptive-per-bar

              # Umbral fijo explícito:
              transfer --input cancion.mid --model-dir model/ --threshold 0.35
        """))
    p_tr.add_argument('--input',      required=True, metavar='FILE')
    p_tr.add_argument('--model-dir',  required=True, metavar='DIR')
    p_tr.add_argument('--direction',  choices=['AB', 'BA'], default='AB',
        help='Dirección de transferencia: AB (A→B) o BA (B→A) (default: AB)')
    p_tr.add_argument('--output',     default='transfer_output.mid', metavar='FILE')
    p_tr.add_argument('--palette',    default=None,  metavar='FILE',
        help='JSON con programas MIDI por rol (opcional)')
    p_tr.add_argument('--bpm',        type=float, default=120.0)
    p_tr.add_argument('--batch-size', type=int,   default=16, metavar='INT',
        help='Compases procesados a la vez en inferencia (default: 16)')
    p_tr.add_argument('--threshold',  type=float, default=None, metavar='FLOAT',
        help='Umbral fijo de binarización [0,1]. Si se omite, se calcula '
             'automáticamente según --threshold-percentile.')
    p_tr.add_argument('--threshold-percentile', type=float, default=90.0,
        metavar='FLOAT', dest='threshold_percentile',
        help='Percentil de la distribución de activación usado para calcular '
             'el umbral automático (default: 90). '
             'Valores bajos → más notas; valores altos → menos notas. '
             'Se ignora si se pasa --threshold. '
             'Recomendado: 85–92 con modelos convergidos, 88–95 con modelos '
             'poco entrenados.')
    p_tr.add_argument('--adaptive-per-bar', action='store_true',
        dest='adaptive_per_bar',
        help='Calcula el umbral independientemente para cada compás en lugar '
             'de una vez sobre todo el roll. Evita compases vacíos cuando '
             'hay variación grande de activación entre compases.')
    p_tr.set_defaults(func=cmd_transfer)

    # ── round-trip ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser('round-trip',
        help='MIDI → piano roll → MIDI sin modelo (diagnóstico del parser)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convierte un MIDI a piano roll y de vuelta a MIDI sin usar el modelo.
            Útil para verificar que el pipeline de parsing y renderizado funciona.

              round-trip OK, transfer mal → problema en el modelo
              round-trip mal              → problema en el parser MIDI

            Ejemplos:
              round-trip --input ref.mid
              round-trip --input ref.mid --model-dir model_jazz2clasico/
        """))
    p_rt.add_argument('--input',      required=True, metavar='FILE')
    p_rt.add_argument('--model-dir',  default=None,  metavar='DIR',
        help='Si se indica, lee configuración del modelo (resolución, roles, pitch)')
    p_rt.add_argument('--resolution', type=int, default=TICKS_PER_BAR_DEFAULT, metavar='INT')
    p_rt.add_argument('--palette',    default=None,  metavar='FILE')
    p_rt.add_argument('--output',     default='output_roundtrip.mid', metavar='FILE')
    p_rt.add_argument('--bpm',        type=float, default=120.0)
    p_rt.set_defaults(func=cmd_round_trip)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect',
        help='Diagnóstico de archivos .npz y del modelo entrenado')
    p_ins.add_argument('--what', nargs='+',
        choices=['npz', 'model'],
        default=['npz'])
    p_ins.add_argument('--data-dir',  metavar='DIR', default=None)
    p_ins.add_argument('--model-dir', metavar='DIR', default=None)
    p_ins.add_argument('--file',      dest='npz_file', metavar='NAME', default=None,
        help='Nombre de un .npz específico a inspeccionar')
    p_ins.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
