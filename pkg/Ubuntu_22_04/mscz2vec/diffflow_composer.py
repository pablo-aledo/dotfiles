#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      DIFFFLOW COMPOSER  v1                                   ║
║   Composición end-to-end mediante Diffusion Normalizing Flow (SDE dual)      ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Dos SDEs neuronales entrenadas conjuntamente sobre el piano roll:         ║
║      forward : dx = f_θ(x,t,cond) dt + g dw   (drift APRENDIDO, no fijo)     ║
║      backward: dx = [f_θ(x,t,cond) - g²·s_θ(x,t,cond)] dt + g dw             ║
║    Se entrena minimizando KL(p_forward‖p_backward) sobre la trayectoria      ║
║    completa (Zhang & Chen, "Diffusion Normalizing Flow", NeurIPS 2021).      ║
║    A medio camino entre flow_composer.py (biyección determinista, sin        ║
║    ruido) y diffusion_composer_v4.py (drift fijo, solo el backward aprende): ║
║    aquí AMBAS direcciones son redes entrenables, lo que permite menos pasos  ║
║    de discretización que una difusión estándar manteniendo capacidad de      ║
║    modelar distribuciones multimodales que un flow puro no puede.            ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare      — MIDI corpus → piano rolls segmentados por rol (.npz)       ║
║    train        — Entrena las dos SDEs (forward + backward) conjuntamente    ║
║    encode       — MIDI referencia → z_style (.json)                          ║
║    style-corpus — Centroide de estilo de una carpeta de MIDIs                ║
║    compose      — Genera obra nueva (modos: sample/blend/sweep/transfer/     ║
║                    reconstruct)                                              ║
║    round-trip   — Diagnóstico: MIDI → piano roll → MIDI sin modelo           ║
║    inspect      — Diagnóstico del modelo y los datos                         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ── Preparar datos (idéntico al resto de composers) ───────────────────────────
python diffflow_composer.py prepare --input-dir midis/ --output-dir data/ --report

python diffflow_composer.py prepare \
    --input-dir midis/ --output-dir data_small/ \
    --disable-roles percussion counterpoint \
    --pitch-range 48

# ── Entrenar ──────────────────────────────────────────────────────────────────
python diffflow_composer.py train \
    --data-dir data/ --model-dir model_diffflow/ \
    --epochs 300 --batch-size 8 --lr 1e-4 \
    --style-dim 16 --hidden-channels 64 \
    --sde-steps 24 --g-coef 1.0 --patience 50

# Reanudar entrenamiento interrumpido:
python diffflow_composer.py train \
    --data-dir data/ --model-dir model_diffflow/ --epochs 300 --resume

# ── Codificar un MIDI de referencia → estilo ──────────────────────────────────
python diffflow_composer.py encode --input midis/005505b_.mid \
    --model-dir model_diffflow/ --output z_estilo_A.json

python diffflow_composer.py style-corpus --input-dir midis_A/ \
    --model-dir model_diffflow/ --output z_estilo_A.json

# ── Generar (sample libre condicionado en estilo+tensión) ─────────────────────
python diffflow_composer.py compose \
    --model-dir model_diffflow/ --palette palette.json \
    --mode sample --input midis/005505b_.mid \
    --temperature 0.8 --bars 16 --tension arch \
    --gen-steps 30 --threshold-pct 99.0

# ── Morphing gradual entre dos canciones (sweep) ───────────────────────────────
python diffflow_composer.py compose \
    --model-dir model_diffflow/ --palette palette.json \
    --mode sweep --inputs midis/005505b_.mid midis/008906b_.mid \
    --bars 32 --temperature 0.8 --threshold-pct 99.0

# ── Mezcla estática entre dos estilos (blend) ──────────────────────────────────
python diffflow_composer.py compose \
    --model-dir model_diffflow/ --palette palette.json \
    --mode blend --inputs midis/005505b_.mid midis/008906b_.mid \
    --weights 0.5 0.5 --bars 16 --threshold-pct 99.0

# ── Transferencia de estilo ─────────────────────────────────────────────────────
python diffflow_composer.py compose \
    --model-dir model_diffflow/ --palette palette.json \
    --mode transfer --input midis/005505b_.mid \
    --style-from z_estilo_A.json --output resultado.mid \
    --threshold-pct 99.0

# ── Round-trip vía las dos SDEs (diagnóstico, NO es una biyección exacta         )
#    a diferencia de flow_composer.py: aquí el error de reconstrucción > 0        ─
python diffflow_composer.py compose \
    --model-dir model_diffflow/ --palette palette.json \
    --mode reconstruct --input midis/005505b_.mid --output recon.mid

# ── Diagnóstico ──────────────────────────────────────────────────────────────────
python diffflow_composer.py round-trip --input midis/005505b_.mid
python diffflow_composer.py inspect --model-dir model_diffflow/ --data-dir data/

"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES  (idénticas a diffusion_composer_v4.py / latent_composer.py)
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
MIDI_CENTER           = 60   # Do central


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
#  UTILIDADES MIDI COMUNES  (idénticas a diffusion_composer_v4.py)
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


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES  (RoleAssigner) — idéntico a diffusion_composer_v4.py
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

        def norm(lst, key):
            vals = [p[key] for p in unassigned]
            lo, hi = min(vals), max(vals)
            span = hi - lo or 1
            return {p['key']: (p[key] - lo) / span for p in unassigned}

        if not unassigned:
            return assigned

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
        pairs = [(score_matrix[p['key']][r], r, p['key']) for p in unassigned for r in remaining_roles]
        pairs.sort(key=lambda x: -x[0])
        for sc, role, key in pairs:
            if role not in taken_roles and key not in taken_keys:
                assigned[role] = key
                taken_roles.add(role)
                taken_keys.add(key)
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  PIANO ROLL CONVERTER  — idéntico a diffusion_composer_v4.py
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
            bar_s    = int(start / (tpb_raw * 4))
            tick_s   = int((start % (tpb_raw * 4)) / ticks_per_internal)
            bar_e    = int(end   / (tpb_raw * 4))
            tick_e   = int((end   % (tpb_raw * 4)) / ticks_per_internal)
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
        windows   = np.stack([roll[i:i + self.window_bars]
                               for i in range(n_windows)])
        return windows


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTOR DE TENSIÓN  — idéntico a diffusion_composer_v4.py (sin enriquecimiento
#  mscz2vec opcional, omitido aquí para mantener el archivo autocontenido)
# ══════════════════════════════════════════════════════════════════════════════

class TensionExtractor:
    TENSION_DIM = 8

    def extract_bar_vectors(self, role_rolls: dict, bars: int):
        import numpy as np
        n_pitch = next(iter(role_rolls.values())).shape[-1] if role_rolls else PITCH_CLASSES
        vectors = np.zeros((bars, self.TENSION_DIM), dtype=np.float32)
        for bar in range(bars):
            combined    = np.zeros((n_pitch,), dtype=np.float32)
            total_events = 0
            resolution   = None
            for role, roll in role_rolls.items():
                if bar >= roll.shape[0]:
                    continue
                bar_roll  = roll[bar]
                resolution = bar_roll.shape[0]
                active    = bar_roll.max(axis=0)
                combined  = np.maximum(combined, active)
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
            vel_mean       = 0.5
            rhythm_density = 0.0
            if 'melody' in role_rolls and bar < role_rolls['melody'].shape[0]:
                mel = role_rolls['melody'][bar]
                active_per_tick = mel.sum(axis=1)
                changes = float(np.sum(np.diff(active_per_tick) != 0))
                rhythm_density = changes / max(resolution - 1, 1)
            arousal = 0.5 * min(density * 2, 1.0) + 0.5 * rhythm_density
            vectors[bar] = [tension, density, poly, reg_mean,
                            reg_spread, vel_mean, rhythm_density, arousal]
        return vectors

    @staticmethod
    def _lerdahl_proxy(pitches_active) -> float:
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
#  PREPARE: MIDI corpus → piano rolls (.npz)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    (midi_path, output_dir, resolution, window_bars,
     active_roles, pitch_lo, pitch_hi) = args_tuple
    import numpy as np

    stem = midi_path.stem
    stats_partial = {r: 0 for r in ROLES}
    stats_partial['files_ok']      = 0
    stats_partial['files_skipped'] = 0
    stats_partial['total_windows'] = 0

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

    tpb_beat   = mid.ticks_per_beat          # ticks por negra — el que espera notes_to_roll
    tpb_raw    = _ticks_per_bar(mid)         # ticks por compás (= tpb_beat * 4) — para total_bars
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
        roll = converter.notes_to_roll(notes, tpb_beat, total_bars)
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
    out_path = Path(output_dir) / f"{stem}.npz"
    np.savez_compressed(str(out_path), **save_dict)

    stats_partial['files_ok']      = 1
    stats_partial['total_windows'] = min_windows

    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, roles: {', '.join(roles_found)})",
            True, stats_partial)


def cmd_prepare(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution  = args.resolution
    window_bars = args.window_bars

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
        n_pitch = pitch_hi - pitch_lo + 1
        print(f"[prepare] Rango de pitch       : {pitch_n} notas  "
              f"(MIDI {pitch_lo}–{pitch_hi}, n_pitch={n_pitch})")

    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    n_workers = min(multiprocessing.cpu_count(), len(midi_files))

    print(f"[prepare] {len(midi_files)} archivos MIDI encontrados")
    print(f"[prepare] Resolución: {resolution} ticks/compás  |  Ventana: {window_bars} compases")
    print(f"[prepare] Paralelizando con {n_workers} procesos\n")

    stats = {r: 0 for r in ROLES}
    stats['files_ok'] = stats['files_skipped'] = stats['total_windows'] = 0

    task_args = [
        (midi_path, str(output_dir), resolution, window_bars,
         active_roles, pitch_lo, pitch_hi)
        for midi_path in midi_files
    ]

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
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """
    Cada muestra:
        'x'        : Tensor (N_ROLES, resolution, n_pitch)  — compás objetivo
        'context'  : Tensor (N_ROLES, ctx_bars, resolution, n_pitch)  — contexto previo
        'tension'  : Tensor (TENSION_DIM,)
        'role_mask': Tensor (N_ROLES,) bool
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
        ctx_bars    = window_bars - 1

        x_parts   = []
        ctx_parts = []
        mask      = []

        for role in self.roles:
            key = f'roll_{role}'
            if key in data:
                window = data[key][widx]
                x_parts.append(window[-1])
                ctx_parts.append(window[:ctx_bars])
                mask.append(True)
            else:
                x_parts.append(np.zeros((resolution, self.n_pitch), dtype=np.float32))
                ctx_parts.append(np.zeros((ctx_bars, resolution, self.n_pitch), dtype=np.float32))
                mask.append(False)

        x       = torch.tensor(np.stack(x_parts,   axis=0))
        context = torch.tensor(np.stack(ctx_parts,  axis=0))
        tension = torch.tensor(data['tension'][widx])
        role_mask = torch.tensor(mask, dtype=torch.bool)

        return {'x': x, 'context': context, 'tension': tension, 'role_mask': role_mask}


def _collate_fn(batch):
    import torch
    return {
        'x':         torch.stack([b['x']         for b in batch]),
        'context':   torch.stack([b['context']   for b in batch]),
        'tension':   torch.stack([b['tension']   for b in batch]),
        'role_mask': torch.stack([b['role_mask'] for b in batch]),
    }
# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA: DIFFUSION NORMALIZING FLOW (SDE dual) SOBRE EL PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════
#
#  x (N_ROLES, res, n_pitch) ∈ {0,1}   — compás objetivo, binario
#
#  1) Dequantización uniforme + transformación logit (idéntico a flow_composer.py,
#     solo como preprocesado a espacio continuo — aquí NO se usa su log-det,
#     porque la pérdida de DiffFlow no es una NLL exacta por cambio de variable
#     sino la cota KL(p_forward‖p_backward) de Zhang & Chen, Ec. 13-15).
#
#  2) SDE forward   (dato → ruido), drift APRENDIDO:
#         dx = f_θ(x,t,cond) dt + g·dw
#     SDE backward  (ruido → dato), con corrección de score:
#         dx = [f_θ(x,t,cond) - g²·s_θ(x,t,cond)] dt + g·dw
#     Discretizadas con Euler-Maruyama en n_sde_steps pasos (Ec. 11-12).
#
#  3) Entrenamiento — Algoritmo 1 del paper: se simula UNA trayectoria forward
#     por muestra (con ruido δ_i^F fresco), se reconstruye δ_i^B con la Ec. 14
#     evaluando f y s en cada x_{i+1}, y se minimiza
#         L = E[ -log p_B(x_N) + Σ_i ½(δ_i^B)² ]
#     por retropropagación directa a través de los n_sde_steps pasos
#     (equivalente al "reparameterization trick" de la Ec. 15). El paper
#     propone además un "stochastic adjoint" (su Algoritmo 2) para entrenar
#     con memoria O(1) en el número de pasos; se omite aquí para mantener el
#     fichero autocontenido y simple — con n_sde_steps moderado (16-32) y los
#     piano rolls pequeños de mutopia, el backprop directo es viable.
#
#  4) Condicionamiento (contexto + tensión + estilo) vía la misma
#     ConditionEncoder de flow_composer.py, modulando drift y score por FiLM
#     junto con el embedding temporal — igual función que el AdaGN de la
#     U-Net de difusión, pero aquí con dos redes (forward y backward) en vez
#     de una sola.
#
#  COSTE: cada paso de entrenamiento evalúa drift_net ~2×n_sde_steps veces y
#  score_net ~1×n_sde_steps veces (vs. 1 sola evaluación por paso en
#  diffusion_composer_v4.py). Si el entrenamiento va lento, baja --sde-steps,
#  --hidden-channels o --batch-size antes que nada.
# ══════════════════════════════════════════════════════════════════════════════

def _build_diffflow_modules():
    """Importa torch de forma perezosa y devuelve las clases de DiffFlow."""
    import math
    import torch
    import torch.nn as nn

    ALPHA = 0.05  # margen de la transformación logit (evita ±inf) — igual que flow_composer.py

    def dequantize_and_logit(x):
        """x ∈ {0,1}^(...)  →  x_logit  (preprocesado a espacio continuo, sin log-det)."""
        u = torch.rand_like(x)
        x_deq  = x + u
        x_norm = x_deq / 2.0
        s = ALPHA + (1 - 2 * ALPHA) * x_norm
        x_logit = torch.log(s) - torch.log1p(-s)
        return x_logit, None

    def invert_logit(x_logit):
        """x_logit → x_cont ∈ [0,2) aprox — inversa de la transformación anterior."""
        s = torch.sigmoid(x_logit)
        x_norm = (s - ALPHA) / (1 - 2 * ALPHA)
        return x_norm * 2.0

    class ConditionEncoder(nn.Module):
        """Idéntico en espíritu a flow_composer.py: contexto+tensión+estilo → cond.
        Expone style_head por separado para encode / style-corpus / transfer."""
        def __init__(self, n_roles, ctx_bars, resolution, n_pitch,
                     tension_dim, style_dim, cond_dim):
            super().__init__()
            self.style_dim = style_dim
            self.cond_dim  = cond_dim

            in_feats = n_roles * ctx_bars * resolution * n_pitch
            trunk_hidden = min(512, max(128, cond_dim * 2))

            self.trunk = nn.Sequential(
                nn.Linear(in_feats, trunk_hidden), nn.ReLU(inplace=True),
                nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(inplace=True),
            )
            self.style_head   = nn.Linear(trunk_hidden, style_dim)
            self.content_head = nn.Linear(trunk_hidden, cond_dim - style_dim - tension_dim)
            self.tension_proj = nn.Linear(tension_dim, tension_dim)

        def encode_style(self, context):
            b = context.shape[0]
            h = self.trunk(context.reshape(b, -1))
            return self.style_head(h)

        def forward(self, context, tension, style_override=None):
            b = context.shape[0]
            h = self.trunk(context.reshape(b, -1))
            style   = self.style_head(h) if style_override is None else style_override
            content = self.content_head(h)
            ten     = self.tension_proj(tension)
            return torch.cat([content, style, ten], dim=1)  # (B, cond_dim)

    class _TimeEmb(nn.Module):
        """Embedding sinusoidal para t ∈ [0,1] continuo (pseudo-tiempo de la SDE)."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.proj = nn.Sequential(
                nn.Linear(dim, dim * 2), nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, t):
            half   = self.dim // 2
            freqs  = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
            args   = t[:, None].float() * 1000.0 * freqs[None]   # escala t∈[0,1] a rango útil
            emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return self.proj(emb)

    class _FiLMConvNet(nn.Module):
        """Red conv ligera (sin downsampling) condicionada por FiLM(cond, t_emb).
        Usada tanto para el drift (forward) como para el score (backward) —
        pesos independientes, misma arquitectura. Salida con init a cero para
        que al empezar el entrenamiento f≈0, s≈0 y la SDE forward arranque
        como un movimiento browniano puro (comportamiento estable)."""
        def __init__(self, n_roles, cond_dim, t_dim, hidden):
            super().__init__()
            self.in_conv  = nn.Conv2d(n_roles, hidden, 3, padding=1)
            self.film1    = nn.Linear(cond_dim + t_dim, hidden * 2)
            self.mid_conv = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.film2    = nn.Linear(cond_dim + t_dim, hidden * 2)
            self.out_conv = nn.Conv2d(hidden, n_roles, 3, padding=1)
            self.act      = nn.SiLU()
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

        def forward(self, x, t_emb, cond):
            ct = torch.cat([cond, t_emb], dim=1)
            h  = self.act(self.in_conv(x))
            g1, b1 = self.film1(ct).chunk(2, dim=1)
            h = h * (1 + g1[:, :, None, None]) + b1[:, :, None, None]
            h = self.act(self.mid_conv(h))
            g2, b2 = self.film2(ct).chunk(2, dim=1)
            h = h * (1 + g2[:, :, None, None]) + b2[:, :, None, None]
            return self.out_conv(h)

    class DiffFlowModel(nn.Module):
        def __init__(self, n_roles, resolution, n_pitch, ctx_bars,
                     tension_dim, style_dim, hidden_channels=64,
                     cond_dim=128, t_dim=64, n_sde_steps=24, g_coef=1.0):
            super().__init__()
            self.n_roles     = n_roles
            self.resolution  = resolution
            self.n_pitch     = n_pitch
            self.ctx_bars    = ctx_bars
            self.tension_dim = tension_dim
            self.style_dim   = style_dim
            self.cond_dim    = cond_dim
            self.t_dim       = t_dim
            self.n_steps     = n_sde_steps
            self.g_coef      = g_coef
            self.T           = 1.0
            self.dim         = n_roles * resolution * n_pitch

            self.cond_encoder = ConditionEncoder(
                n_roles, ctx_bars, resolution, n_pitch,
                tension_dim, style_dim, cond_dim)
            self.time_emb  = _TimeEmb(t_dim)
            self.drift_net = _FiLMConvNet(n_roles, cond_dim, t_dim, hidden_channels)
            self.score_net = _FiLMConvNet(n_roles, cond_dim, t_dim, hidden_channels)

        def _get_style(self, context):
            return self.cond_encoder.encode_style(context)

        # ── SDE forward: dato → ruido (Ec. 11) ─────────────────────────────
        def simulate_forward(self, x0_logit, cond, n_steps=None):
            n_steps = n_steps or self.n_steps
            dt = self.T / n_steps
            x  = x0_logit
            B, device = x.shape[0], x.device
            for i in range(n_steps):
                t     = torch.full((B,), i / n_steps, device=device)
                t_emb = self.time_emb(t)
                f     = self.drift_net(x, t_emb, cond)
                noise = torch.randn_like(x)
                x = x + f * dt + self.g_coef * math.sqrt(dt) * noise
            return x

        # ── SDE backward: ruido → dato (Ec. 12) ────────────────────────────
        def simulate_backward(self, x_N, cond, n_steps=None, temperature=1.0):
            n_steps = n_steps or self.n_steps
            dt = self.T / n_steps
            x  = x_N
            B, device = x.shape[0], x.device
            for i in range(n_steps, 0, -1):
                t     = torch.full((B,), i / n_steps, device=device)
                t_emb = self.time_emb(t)
                f     = self.drift_net(x, t_emb, cond)
                s     = self.score_net(x, t_emb, cond)
                noise = torch.randn_like(x) * temperature
                x = x - (f - (self.g_coef ** 2) * s) * dt + self.g_coef * math.sqrt(dt) * noise
            return x

        def sample(self, context, tension, style_override=None,
                   temperature=1.0, n_steps=None):
            cond = self.cond_encoder(context, tension, style_override=style_override)
            B, device = context.shape[0], context.device
            x_N = torch.randn(B, self.n_roles, self.resolution, self.n_pitch,
                               device=device) * temperature
            x0_logit = self.simulate_backward(x_N, cond, n_steps=n_steps, temperature=temperature)
            return torch.clamp(invert_logit(x0_logit), 0.0, 1.0)

        def encode_stochastic(self, x, context, tension, n_steps=None):
            """x → x_N vía la SDE forward entrenada. NO es una biyección exacta
            (a diferencia de flow_composer.py): dos llamadas con el mismo x dan
            x_N distintos por el ruido de la propia SDE. Diagnóstico en 'reconstruct'."""
            x_logit, _ = dequantize_and_logit(x)
            cond = self.cond_encoder(context, tension)
            x_N  = self.simulate_forward(x_logit, cond, n_steps=n_steps)
            return x_N, cond

        def decode(self, x_N, cond, n_steps=None, temperature=1.0):
            x0_logit = self.simulate_backward(x_N, cond, n_steps=n_steps, temperature=temperature)
            return torch.clamp(invert_logit(x0_logit), 0.0, 1.0)

        # ── Entrenamiento: Algoritmo 1 (backprop directo por reparametrización) ──
        def forward(self, x, context, tension):
            """Devuelve (loss, metrics). loss ≈ KL(p_forward‖p_backward)/dim (Ec. 15)."""
            x_logit, _ = dequantize_and_logit(x)
            cond = self.cond_encoder(context, tension)
            B, device = x.shape[0], x.device
            n, dt = self.n_steps, self.T / self.n_steps

            x_i, saved = x_logit, []
            for i in range(n):
                t     = torch.full((B,), i / n, device=device)
                t_emb = self.time_emb(t)
                f_i   = self.drift_net(x_i, t_emb, cond)
                noise = torch.randn_like(x_i)
                x_next = x_i + f_i * dt + self.g_coef * math.sqrt(dt) * noise
                saved.append((x_i, x_next, i))
                x_i = x_next
            x_N = x_i

            loss_terms = torch.zeros(B, device=device)
            for (xa, xb, i) in saved:
                t_next    = torch.full((B,), (i + 1) / n, device=device)
                temb_next = self.time_emb(t_next)
                f_next = self.drift_net(xb, temb_next, cond)
                s_next = self.score_net(xb, temb_next, cond)
                bracket  = xa - xb + (f_next - (self.g_coef ** 2) * s_next) * dt
                delta_B  = bracket / (self.g_coef * math.sqrt(dt))
                loss_terms = loss_terms + 0.5 * (delta_B ** 2).flatten(1).sum(dim=1)

            neg_log_pB_xN = 0.5 * (x_N ** 2).flatten(1).sum(dim=1)   # prior N(0,I), salvo constante
            total = neg_log_pB_xN + loss_terms
            loss  = total.mean() / self.dim
            bpd   = loss / math.log(2.0)
            return loss, {'loss': loss.item(), 'bpd': bpd.item(),
                          'prior': (neg_log_pB_xN.mean() / self.dim).item()}

    return {
        'DiffFlowModel': DiffFlowModel,
        'dequantize_and_logit': dequantize_and_logit,
        'invert_logit': invert_logit,
    }


def _build_full_model(n_roles, resolution, n_pitch, ctx_bars,
                       tension_dim, style_dim, hidden_channels,
                       cond_dim=128, t_dim=64, n_sde_steps=24, g_coef=1.0):
    mods = _build_diffflow_modules()
    return mods['DiffFlowModel'](
        n_roles=n_roles, resolution=resolution, n_pitch=n_pitch, ctx_bars=ctx_bars,
        tension_dim=tension_dim, style_dim=style_dim, hidden_channels=hidden_channels,
        cond_dim=cond_dim, t_dim=t_dim, n_sde_steps=n_sde_steps, g_coef=g_coef)


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'
    CONFIG_NAME     = 'model_config.json'

    def __init__(self, model, optimizer, model_dir: Path, patience: int = 50):
        self.model      = model
        self.optimizer  = optimizer
        self.model_dir  = model_dir
        self.patience   = patience

        self.history       = {'train': [], 'val': [], 'val_bpd': []}
        self.best_val_loss = float('inf')
        self.no_improve    = 0
        self.start_epoch   = 0
        self._resume        = False

    def save_checkpoint(self, epoch, val_loss, is_best):
        import torch
        state = {
            'epoch': epoch, 'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss, 'no_improve': self.no_improve,
            'history': self.history,
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
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.best_val_loss = state['best_val_loss']
        self.no_improve    = state['no_improve']
        self.history       = state['history']
        self.start_epoch   = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_epoch}  "
              f"(mejor val={self.best_val_loss:.4f})")

    def _run_epoch(self, loader, training, epoch=0, n_epochs=0):
        import torch, time, math
        self.model.train(training)
        total_loss = bpd_sum = 0.0
        n_batches  = 0
        phase = 'train' if training else 'val  '
        n_total = (len(loader.dataset) // max(loader.batch_size, 1) + 1
                   if hasattr(loader, 'dataset') and hasattr(loader, 'batch_size') else None)

        batch_times = []
        n_skipped = 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                t0     = time.time()
                device = next(self.model.parameters()).device
                x       = batch['x'].to(device, non_blocking=True)
                context = batch['context'].to(device, non_blocking=True)
                tension = batch['tension'].to(device, non_blocking=True)

                loss, metrics = self.model(x, context, tension)

                if not math.isfinite(loss.item()):
                    n_skipped += 1
                    if training:
                        self.optimizer.zero_grad()
                    batch_times.append(time.time() - t0)
                    continue

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                total_loss += loss.item()
                bpd_sum    += metrics['bpd']
                n_batches  += 1

                batch_times.append(time.time() - t0)
                if len(batch_times) > 20:
                    batch_times.pop(0)

                avg_loss = total_loss / n_batches
                if n_total:
                    pct    = n_batches / n_total
                    bar_w  = 18
                    filled = int(pct * bar_w)
                    bar    = '█' * filled + '░' * (bar_w - filled)
                    prog   = f"[{bar}] {n_batches}/{n_total}"
                    if batch_times:
                        avg_bt  = sum(batch_times) / len(batch_times)
                        rem_bt  = avg_bt * (n_total - n_batches)
                        eta_str = f"  ~{_fmt_time(rem_bt)}"
                    else:
                        eta_str = ""
                else:
                    prog, eta_str = f"batch {n_batches}", ""

                print(f"\r  [{phase}] ep {epoch+1}/{n_epochs}  {prog}"
                      f"  loss={avg_loss:.4f}  bpd={bpd_sum/n_batches:.4f}{eta_str}   ",
                      end='', flush=True)

        print(' ' * 120, end='\r')
        if n_batches == 0 and n_skipped > 0:
            raise RuntimeError(
                f"[{phase.strip()}] Los {n_skipped} batches de esta época dieron loss "
                f"no finita (NaN/Inf) — el modelo no aprendió nada en esta época. "
                f"Revisa --g-coef (debe ser > 0), --lr (prueba a bajarlo) y --sde-steps.")
        if n_batches == 0:
            raise RuntimeError(
                f"[{phase.strip()}] Esta época no produjo ni un solo batch — el split de "
                f"datos ({len(loader.dataset) if hasattr(loader, 'dataset') else '?'} muestras) "
                f"es más pequeño que --batch-size. Baja --batch-size o añade más datos a 'prepare'.")
        if n_skipped:
            print(f"  [{phase}] ⚠ {n_skipped} batch(es) con loss no finita, omitidos "
                  f"(de {n_batches + n_skipped} totales).")
        return total_loss / n_batches, bpd_sum / n_batches

    def train(self, train_loader, val_loader, n_epochs: int):
        import torch, time

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        print(f"\n{'═'*64}")
        print(f"  DIFFFLOW COMPOSER — Entrenamiento (SDE dual, Algoritmo 1)")
        print(f"  Épocas máx. : {n_epochs}   Early stopping: {self.patience} sin mejora")
        print(f"  Dispositivo : {device}")
        print(f"  Modelo dir  : {self.model_dir}")
        print(f"{'═'*64}\n")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self._resume:
            self.load_checkpoint()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6)

        epoch_times = []
        train_start = time.time()

        for epoch in range(self.start_epoch, n_epochs):
            if epoch_times:
                avg_ep  = sum(epoch_times) / len(epoch_times)
                eta_sec = avg_ep * (n_epochs - epoch)
                eta_str = f"  ETA {_fmt_time(eta_sec)}"
            else:
                eta_str = ""

            lr_current = self.optimizer.param_groups[0]['lr']
            print(f"  Época {epoch+1:>4}/{n_epochs}  lr={lr_current:.2e}{eta_str}", flush=True)

            epoch_t0 = time.time()
            tr_loss, tr_bpd = self._run_epoch(train_loader, True, epoch, n_epochs)
            vl_loss, vl_bpd = self._run_epoch(val_loader, False, epoch, n_epochs)
            scheduler.step()

            epoch_elapsed = time.time() - epoch_t0
            epoch_times.append(epoch_elapsed)
            if len(epoch_times) > 5:
                epoch_times.pop(0)

            self.history['train'].append(tr_loss)
            self.history['val'].append(vl_loss)
            self.history['val_bpd'].append(vl_bpd)

            is_best = vl_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = vl_loss
                self.no_improve = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, vl_loss, is_best)

            best_marker = ' ◀ mejor' if is_best else ''
            stop_str    = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                           if self.no_improve > 0 else '')
            print(f"         train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}"
                  f"  (val_bpd={vl_bpd:.4f})"
                  f"  {_fmt_time(epoch_elapsed)}/época{best_marker}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        total_elapsed = time.time() - train_start
        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(total_elapsed)}.")
        print(f"  Mejor val_loss (bits/dim aprox. = /ln2) : {self.best_val_loss:.4f}")
        print(f"  Modelos en : {self.model_dir}")
        print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader

    data_dir  = args.data_dir
    model_dir = Path(args.model_dir)

    disabled = set(getattr(args, 'disable_roles', None) or [])
    roles    = [r for r in ROLES if r not in disabled]
    if not roles:
        print("[train] --disable-roles ha deshabilitado los 5 roles — no queda nada que entrenar.")
        sys.exit(1)

    if args.sde_steps < 1:
        print(f"[train] --sde-steps debe ser ≥ 1 (recibido: {args.sde_steps}).")
        sys.exit(1)
    if args.g_coef <= 0:
        print(f"[train] --g-coef debe ser > 0 (recibido: {args.g_coef}). Con g=0 la SDE se "
              f"degenera en una ODE determinista y la reconstrucción de δ_i^B (Ec. 14) divide "
              f"entre cero — el entrenamiento produciría NaN/Inf silenciosos en vez de aprender.")
        sys.exit(1)

    print(f"[train] Cargando dataset desde {data_dir} ...")
    try:
        dataset = MidiRollDataset(data_dir, roles=roles)
    except FileNotFoundError as e:
        print(f"[train] {e}")
        print("[train] Ejecuta 'prepare' primero, o revisa --data-dir / --disable-roles.")
        sys.exit(1)
    print(f"[train] {len(dataset)} muestras  |  n_pitch={dataset.n_pitch}")
    if len(dataset) < 10:
        print("[train] Muy pocas muestras — ejecuta 'prepare' con más MIDIs.")
        sys.exit(1)

    # Cobertura real de roles en el corpus preparado, para detectar un
    # --disable-roles que deja solo roles ausentes del corpus (entrenaría
    # en silencio sobre canales vacíos).
    roles_in_corpus = set()
    for _, _, meta in dataset.samples:
        roles_in_corpus.update(meta.get('roles', []))
    active_with_data = [r for r in roles if r in roles_in_corpus]
    active_without_data = [r for r in roles if r not in roles_in_corpus]
    if not active_with_data:
        print(f"[train] Ninguno de los roles activos {roles} aparece en el corpus "
              f"preparado (roles presentes: {sorted(roles_in_corpus)}). Entrenar así "
              f"solo aprendería sobre canales vacíos. Revisa --disable-roles o "
              f"vuelve a ejecutar 'prepare' con los roles que necesitas.")
        sys.exit(1)
    if active_without_data:
        print(f"[train] ⚠ Los roles {active_without_data} están activos pero no aparecen "
              f"en ningún fichero del corpus — se entrenarán como canales siempre vacíos.")

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    if n_train < args.batch_size:
        print(f"[train] --batch-size ({args.batch_size}) es mayor que las muestras de "
              f"entrenamiento tras el split 90/10 ({n_train} de {len(dataset)}) — no "
              f"quedaría ningún batch. Baja --batch-size a ≤ {n_train} o añade más datos.")
        sys.exit(1)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               collate_fn=_collate_fn, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                               collate_fn=_collate_fn)

    sample = dataset[0]
    n_roles     = sample['x'].shape[0]
    resolution  = sample['x'].shape[1]
    n_pitch     = sample['x'].shape[2]
    ctx_bars    = sample['context'].shape[1]
    tension_dim = sample['tension'].shape[0]

    model = _build_full_model(
        n_roles=n_roles, resolution=resolution, n_pitch=n_pitch, ctx_bars=ctx_bars,
        tension_dim=tension_dim, style_dim=args.style_dim,
        hidden_channels=args.hidden_channels, cond_dim=args.cond_dim,
        n_sde_steps=args.sde_steps, g_coef=args.g_coef)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Modelo: {args.sde_steps} pasos de SDE (g={args.g_coef}), "
          f"{args.hidden_channels} canales ocultos, {n_params:,} parámetros")
    print(f"[train] Coste aprox. por paso: ~{3*args.sde_steps} evaluaciones de red "
          f"(2×drift + 1×score por paso de discretización)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, model_dir, patience=args.patience)
    trainer._resume = getattr(args, 'resume', False)

    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        'n_roles': n_roles, 'roles': roles, 'resolution': resolution,
        'n_pitch': n_pitch, 'window_bars': ctx_bars + 1, 'ctx_bars': ctx_bars,
        'tension_dim': tension_dim, 'style_dim': args.style_dim,
        'hidden_channels': args.hidden_channels, 'cond_dim': args.cond_dim,
        't_dim': 64, 'sde_steps': args.sde_steps, 'g_coef': args.g_coef,
        'pitch_lo': 0, 'pitch_hi': 127,
    }
    first_meta = dataset.samples[0][2] if dataset.samples else {}
    cfg['pitch_lo'] = first_meta.get('pitch_lo', 0)
    cfg['pitch_hi'] = first_meta.get('pitch_hi', 127)
    with open(model_dir / Trainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer.train(train_loader, val_loader, n_epochs=args.epochs)

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELO Y MIDI DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_and_config(model_dir: Path):
    import torch
    cfg_path   = model_dir / Trainer.CONFIG_NAME
    model_path = model_dir / Trainer.BEST_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró {cfg_path}. ¿Has ejecutado train?")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró {model_path}. ¿Has ejecutado train?")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = _build_full_model(
        n_roles=cfg['n_roles'], resolution=cfg['resolution'], n_pitch=cfg['n_pitch'],
        ctx_bars=cfg['ctx_bars'], tension_dim=cfg['tension_dim'], style_dim=cfg['style_dim'],
        hidden_channels=cfg['hidden_channels'], cond_dim=cfg['cond_dim'],
        t_dim=cfg.get('t_dim', 64), n_sde_steps=cfg['sde_steps'], g_coef=cfg['g_coef'])
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state['model_state'])
    model.train(False)
    return model, cfg


def _midi_to_rolls(midi_path: str, cfg: dict) -> dict:
    import mido
    mid        = mido.MidiFile(midi_path)
    resolution = cfg['resolution']
    window_bars = cfg['ctx_bars'] + 1

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise ValueError(f"No se encontraron notas en {midi_path}")

    tpb        = mid.ticks_per_beat
    tpbar      = tpb * 4
    max_tick   = max((e for nl in note_lists.values() for e in [n[1] for n in nl]), default=0)
    total_bars = max(1, int(max_tick / tpbar) + 1)

    active_roles = cfg.get('roles', ROLES)
    pitch_lo = cfg.get('pitch_lo', 0)
    pitch_hi = cfg.get('pitch_hi', 127)
    do_crop  = (pitch_lo, pitch_hi) != (0, 127)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    rolls    = {}
    for role, stream_key in role_map.items():
        if role not in active_roles:
            continue
        notes = note_lists[stream_key]
        roll  = conv.notes_to_roll(notes, tpb, total_bars)
        if do_crop:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        rolls[role] = roll
    return rolls


def _rolls_to_context_tensor(rolls: dict, cfg: dict):
    import numpy as np
    ctx_bars   = cfg['ctx_bars']
    resolution = cfg['resolution']
    n_roles    = cfg['n_roles']
    role_list  = cfg['roles']
    n_pitch    = cfg['n_pitch']

    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 0
    if n_bars < ctx_bars:
        raise ValueError(f"MIDI demasiado corto: {n_bars} compases, se necesitan ≥ {ctx_bars}")

    ctx = np.zeros((n_roles, ctx_bars, resolution, n_pitch), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls:
            ctx[ridx] = rolls[role][:ctx_bars]
    return ctx


def _bar_tensor_from_rolls(rolls: dict, cfg: dict, bar_idx: int):
    """Devuelve la barra bar_idx de todos los roles como tensor (N_ROLES,res,n_pitch)."""
    import numpy as np
    n_roles    = cfg['n_roles']
    role_list  = cfg['roles']
    resolution = cfg['resolution']
    n_pitch    = cfg['n_pitch']
    x = np.zeros((n_roles, resolution, n_pitch), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls and bar_idx < rolls[role].shape[0]:
            x[ridx] = rolls[role][bar_idx]
    return x


def _encode_ref(midi_path: str, model, cfg: dict):
    """Devuelve (z_context_flat, z_style) para un MIDI de referencia."""
    import torch, numpy as np

    rolls  = _midi_to_rolls(midi_path, cfg)
    ctx_np = _rolls_to_context_tensor(rolls, cfg)
    ctx_t  = torch.tensor(ctx_np).unsqueeze(0)

    with torch.no_grad():
        z_style = model._get_style(ctx_t)

    ctx_bars = cfg['ctx_bars']
    x_np = _bar_tensor_from_rolls(rolls, cfg, ctx_bars)
    return x_np.flatten(), z_style[0].numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: encode
# ══════════════════════════════════════════════════════════════════════════════

def cmd_encode(args):
    model_dir = Path(args.model_dir)
    print(f"[encode] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    print(f"[encode] Procesando {args.input} ...")
    z_context, z_style = _encode_ref(args.input, model, cfg)

    rolls = _midi_to_rolls(args.input, cfg)
    tension_vecs = TensionExtractor().extract_bar_vectors(
        rolls, min(r.shape[0] for r in rolls.values()))
    tension_mean = tension_vecs.mean(axis=0).tolist()

    out_path = args.output or (Path(args.input).stem + '.style.json')
    payload = {
        'source': args.input, 'model_dir': str(model_dir),
        'style_dim': cfg['style_dim'],
        'z_style': z_style.tolist(),
        'z_context_flat_head': z_context[:64].tolist(),
        'tension_mean': tension_mean,
        'roles_found': list(rolls.keys()),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[encode] Guardado en {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: style-corpus
# ══════════════════════════════════════════════════════════════════════════════

def cmd_style_corpus(args):
    import numpy as np
    model_dir = Path(args.model_dir)
    print(f"[style-corpus] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    input_dir = Path(args.input_dir)
    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[style-corpus] No se encontraron MIDIs en {input_dir}")
        sys.exit(1)

    z_styles = []
    for mf in midi_files:
        try:
            _, zs = _encode_ref(str(mf), model, cfg)
            z_styles.append(zs)
            print(f"  [{mf.stem}] OK")
        except Exception as e:
            print(f"  [{mf.stem}] omitido: {e}")

    if not z_styles:
        print("[style-corpus] No se pudo codificar ningún MIDI.")
        sys.exit(1)

    z_centroid = np.mean(np.stack(z_styles), axis=0)
    out_path = args.output or 'z_style_corpus.json'
    payload = {
        'source_dir': str(input_dir), 'model_dir': str(model_dir),
        'n_files': len(z_styles), 'style_dim': cfg['style_dim'],
        'z_style': z_centroid.tolist(),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[style-corpus] Centroide de {len(z_styles)} MIDIs guardado en {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PERFILES DE TENSIÓN  (idéntico al resto de composers)
# ══════════════════════════════════════════════════════════════════════════════

def _tension_profile(profile: str, n_bars: int, tension_dim: int):
    import numpy as np
    t = np.linspace(0, 1, n_bars)
    if profile == 'flat':
        curve = np.full(n_bars, 0.5)
    elif profile == 'arch':
        curve = np.sin(t * np.pi)
    elif profile == 'rise':
        curve = t
    elif profile == 'fall':
        curve = 1.0 - t
    elif Path(profile).exists():
        with open(profile) as f:
            raw = json.load(f)
        curve = np.interp(t, np.linspace(0, 1, len(raw)), raw)
    else:
        print(f"[compose] Perfil '{profile}' desconocido — usando arch")
        curve = np.sin(t * np.pi)

    out = np.zeros((n_bars, tension_dim), dtype=np.float32)
    out[:, 0] = curve
    out[:, 1] = curve * 0.7
    if tension_dim > 7:
        out[:, 7] = curve
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERER: rolls → MIDI  (idéntico al resto de composers)
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(roll, percentile: float = 85.0) -> float:
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


def _rolls_to_midi(bars_per_role: dict, cfg: dict, palette: dict,
                    output_path: str, bpm: float = 120.0, threshold: float = None):
    import mido, numpy as np

    resolution = cfg['resolution']
    tpb        = 480
    ticks_bar  = tpb * 4
    ticks_tick = ticks_bar / resolution

    mid = mido.MidiFile(ticks_per_beat=tpb)
    tempo_val = int(60_000_000 / bpm)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=tempo_val, time=0))
    mid.tracks.append(t0)

    n_notes_total = 0
    pitch_lo = cfg.get('pitch_lo', 0)
    pitch_hi = cfg.get('pitch_hi', 127)
    do_expand = (pitch_lo, pitch_hi) != (0, 127)

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]
        if do_expand:
            roll = _pad_pitch(roll, pitch_lo, n_full=128)

        thr = threshold if threshold is not None else _adaptive_threshold(roll)
        roll_max = float(roll.max()) if roll.size else 0.0
        if thr >= roll_max:
            # Caso degenerado: la salida está saturada contra el límite del clamp
            # (típico con --temperature alta en un modelo poco entrenado) y el
            # percentil calculado coincide con el máximo — sin este ajuste,
            # ningún píxel superaría el umbral y el MIDI saldría vacío en silencio.
            thr = roll_max - 1e-6

        pal  = palette.get(role, {})
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        binary = (roll > thr).astype(np.float32)

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
                track.append(mido.Message('note_on', channel=ch, note=pitch, velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', channel=ch, note=pitch, velocity=0, time=delta))
            prev_tick = abs_tick

        remaining = last_tick - prev_tick
        if remaining > 0:
            track.append(mido.MetaMessage('end_of_track', time=remaining))

    mid.save(output_path)
    return n_notes_total


def _load_palette(palette_path, cfg: dict) -> dict:
    DEFAULT_PALETTE = {
        'melody':        {'program': 73, 'channel': 0, 'velocity': 90},
        'counterpoint':  {'program': 68, 'channel': 1, 'velocity': 80},
        'accompaniment': {'program': 48, 'channel': 2, 'velocity': 70},
        'bass':          {'program': 43, 'channel': 3, 'velocity': 85},
        'percussion':    {'program': 0,  'channel': 9, 'velocity': 90},
    }
    if not palette_path:
        return DEFAULT_PALETTE
    with open(palette_path) as f:
        user = json.load(f)
    palette = {**DEFAULT_PALETTE}
    for role, params in user.items():
        palette[role] = {**DEFAULT_PALETTE.get(role, {}), **params}
    return palette

# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: compose
# ══════════════════════════════════════════════════════════════════════════════

def cmd_compose(args):
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[compose] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    palette = _load_palette(args.palette, cfg)
    n_bars      = args.bars
    tension_dim = cfg['tension_dim']
    mode        = args.mode
    role_list   = cfg['roles']
    n_roles     = cfg['n_roles']
    resolution  = cfg['resolution']
    ctx_bars    = cfg['ctx_bars']
    n_pitch     = cfg['n_pitch']
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    gen_steps   = args.gen_steps
    if gen_steps < 1:
        print(f"[compose] --gen-steps debe ser ≥ 1 (recibido: {gen_steps})."); sys.exit(1)
    if n_bars < 1:
        print(f"[compose] --bars debe ser ≥ 1 (recibido: {n_bars})."); sys.exit(1)

    tension_matrix = _tension_profile(args.tension, n_bars, tension_dim)

    # ── Modo RECONSTRUCT: ida y vuelta por las dos SDEs — diagnóstico ─────────
    # A diferencia de flow_composer.py (biyección exacta), aquí el forward es
    # estocástico: dos pasadas con el mismo x dan x_N distintos, así que el
    # error de reconstrucción NO tiende a 0. Sirve para medir qué tan bien
    # aprendieron a "deshacerse" mutuamente el drift y el score.
    if mode == 'reconstruct':
        if not args.input:
            print("[compose] El modo 'reconstruct' requiere --input"); sys.exit(1)
        rolls = _midi_to_rolls(args.input, cfg)
        n_bars_ref = min(r.shape[0] for r in rolls.values())
        if n_bars_ref < ctx_bars + 1:
            raise ValueError(f"MIDI demasiado corto para 'reconstruct': {n_bars_ref} compases, "
                              f"se necesitan ≥ {ctx_bars + 1} (contexto de {ctx_bars} + 1 a reconstruir)")
        n_windows  = n_bars_ref - ctx_bars
        n_bars     = min(n_bars, n_windows)

        bars_per_role = {role: [] for role in role_list}
        errs = []
        print(f"[compose] Reconstruyendo {n_bars} compases vía x → x_N → x̂ "
              f"({gen_steps} pasos de SDE, estocástico) ...")
        for bar_idx in range(n_bars):
            ctx_np = np.zeros((n_roles, ctx_bars, resolution, n_pitch), dtype=np.float32)
            for ridx, role in enumerate(role_list):
                if role in rolls:
                    ctx_np[ridx] = rolls[role][bar_idx: bar_idx + ctx_bars]
            x_np = _bar_tensor_from_rolls(rolls, cfg, bar_idx + ctx_bars)

            ctx_t = torch.tensor(ctx_np).unsqueeze(0).to(device)
            x_t   = torch.tensor(x_np).unsqueeze(0).to(device)
            tension_t = torch.tensor(tension_matrix[bar_idx]).unsqueeze(0).to(device)

            with torch.no_grad():
                x_N, cond = model.encode_stochastic(x_t, ctx_t, tension_t, n_steps=gen_steps)
                x_hat = model.decode(x_N, cond, n_steps=gen_steps, temperature=1.0)

            err = float((x_hat[0].cpu().numpy() - x_np).__abs__().mean())
            errs.append(err)
            bar_np = x_hat[0].cpu().numpy()
            for ridx, role in enumerate(role_list):
                bars_per_role[role].append(bar_np[ridx])
            print(f"\r  Compás {bar_idx + 1}/{n_bars}  |x̂-x|₁ medio={err:.5f}", end='', flush=True)
        print()
        print(f"[compose] Error medio de reconstrucción = {np.mean(errs):.6f} "
              f"— NO se espera ~0 (SDE estocástica, no biyección). Sirve como "
              f"medida de fidelidad relativa: cuanto más bajo, mejor convergió el par forward/backward.")
        final_rolls = {r: np.stack(v, axis=0) for r, v in bars_per_role.items() if v}
        arg_thr = getattr(args, 'threshold', None)
        thr = arg_thr if arg_thr is not None else 0.5
        n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output, bpm=args.bpm, threshold=thr)
        print(f"[compose] MIDI guardado en {args.output}  ({n_notes} notas)")
        return

    # ── Preparar z_style / contexto según el modo ──────────────────────────
    if mode in ('sample', 'blend'):
        midi_sources = [args.input] if args.input else (args.inputs or [])
        if not midi_sources:
            print(f"[compose] El modo '{mode}' requiere --input o --inputs"); sys.exit(1)
        z_styles, contexts = [], []
        for src in midi_sources:
            _, zs = _encode_ref(src, model, cfg)
            z_styles.append(zs)
            rolls = _midi_to_rolls(src, cfg)
            contexts.append(_rolls_to_context_tensor(rolls, cfg))

        if mode == 'blend' and len(z_styles) > 1:
            weights = args.weights or [1.0 / len(z_styles)] * len(z_styles)
            if len(weights) != len(z_styles):
                print(f"[compose] --weights tiene {len(weights)} valores pero --inputs "
                      f"tiene {len(z_styles)} — deben coincidir exactamente.")
                sys.exit(1)
            s = sum(weights)
            if s == 0:
                print("[compose] --weights no puede sumar 0."); sys.exit(1)
            weights = [w / s for w in weights]
            z_style_np = sum(w * z for w, z in zip(weights, z_styles))
            ctx_np = contexts[0]
        else:
            z_style_np, ctx_np = z_styles[0], contexts[0]

    elif mode == 'transfer':
        if not args.input or not args.style_from:
            print("[compose] El modo 'transfer' requiere --input y --style-from"); sys.exit(1)
        with open(args.style_from) as f:
            style_payload = json.load(f)
        z_style_np = np.array(style_payload['z_style'], dtype=np.float32)
        rolls  = _midi_to_rolls(args.input, cfg)
        ctx_np = _rolls_to_context_tensor(rolls, cfg)

    elif mode == 'sweep':
        if not args.inputs or len(args.inputs) < 2:
            print("[compose] sweep requiere al menos 2 --inputs"); sys.exit(1)
        all_zs, all_ctx = [], []
        for src in args.inputs:
            _, zs = _encode_ref(src, model, cfg)
            all_zs.append(zs)
            rolls = _midi_to_rolls(src, cfg)
            all_ctx.append(_rolls_to_context_tensor(rolls, cfg))
        sweep_styles = np.stack(all_zs, axis=0)

    else:
        print(f"[compose] Modo desconocido: {mode}"); sys.exit(1)

    bars_per_role = {role: [] for role in role_list}
    if mode != 'sweep':
        ctx_buffer = torch.tensor(ctx_np).unsqueeze(0).to(device)
    else:
        ctx_buffer = None

    print(f"[compose] Generando {n_bars} compases (modo={mode}, "
          f"temperatura={args.temperature}, {gen_steps} pasos de SDE backward) ...")
    adaptive_thr = None

    for bar_idx in range(n_bars):
        v_ten = torch.tensor(tension_matrix[bar_idx]).unsqueeze(0).to(device)

        if mode == 'sweep':
            alpha = bar_idx / max(n_bars - 1, 1)
            n_src = len(sweep_styles)
            seg = alpha * (n_src - 1)
            i0  = min(int(seg), n_src - 2)
            lam = seg - i0
            zs_np = (1 - lam) * sweep_styles[i0] + lam * sweep_styles[i0 + 1]
            v_sty = torch.tensor(zs_np).unsqueeze(0).to(device)
            if bar_idx == 0:
                ctx_buffer = torch.tensor(all_ctx[i0]).unsqueeze(0).to(device)
        else:
            v_sty = torch.tensor(z_style_np).unsqueeze(0).to(device)

        with torch.no_grad():
            roll_bar = model.sample(ctx_buffer, v_ten, style_override=v_sty,
                                     temperature=args.temperature, n_steps=gen_steps)

        bar_np = roll_bar[0].cpu().numpy()

        if bar_idx == 0:
            vmin, vmax, vmean = float(bar_np.min()), float(bar_np.max()), float(bar_np.mean())
            print(f"\n  [diag] Compás 0 — salida de la SDE backward (tras clamp a [0,1]):")
            print(f"         min={vmin:.4f}  mean={vmean:.4f}  max={vmax:.4f}")
            if getattr(args, 'threshold', None) is not None:
                adaptive_thr = args.threshold
                thr_method = f'fijo ({args.threshold})'
            else:
                thr_pct = getattr(args, 'threshold_pct', 99.0)
                adaptive_thr = _adaptive_threshold(bar_np, percentile=thr_pct)
                bar_max = float(bar_np.max()) if bar_np.size else 0.0
                if adaptive_thr >= bar_max:
                    # Mismo caso degenerado que en _rolls_to_midi (saturación
                    # contra el clamp) — se ajusta para que el diagnóstico
                    # coincida con lo que realmente terminará en el MIDI.
                    adaptive_thr = bar_max - 1e-6
                thr_method = f'p{thr_pct}'
            n_active = int((bar_np > adaptive_thr).sum())
            density  = 100 * n_active / bar_np.size
            print(f"         Umbral {thr_method}: {adaptive_thr:.4f}  →  "
                  f"{n_active} píxeles activos ({density:.2f}%)")

        for ridx, role in enumerate(role_list):
            bars_per_role[role].append(bar_np[ridx])

        bar_binary = (bar_np > (adaptive_thr if adaptive_thr is not None else 0.3)).astype(np.float32)
        new_bar = torch.tensor(bar_binary).unsqueeze(0).unsqueeze(2).to(device)
        if ctx_bars > 1:
            ctx_buffer = torch.cat([ctx_buffer[:, :, 1:, :, :], new_bar], dim=2)
        else:
            ctx_buffer = new_bar

        print(f"\r  Compás {bar_idx + 1}/{n_bars}", end='', flush=True)
    print()

    final_rolls = {r: np.stack(v, axis=0) for r, v in bars_per_role.items() if v}
    arg_thr = getattr(args, 'threshold', None)
    final_thr = arg_thr if arg_thr is not None else adaptive_thr
    n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output, bpm=args.bpm, threshold=final_thr)
    print(f"[compose] MIDI guardado en {args.output}  ({n_notes} notas, umbral={final_thr:.3f})")
    if n_notes == 0:
        print("[compose] ⚠  MIDI vacío. Prueba --threshold-pct 90, sube --gen-steps o revisa el entrenamiento.")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: round-trip  (diagnóstico sin modelo — idéntico al resto de composers)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    import numpy as np
    mid = _load_midi(args.input)
    note_lists = _extract_note_lists(mid)
    if not note_lists:
        print("[round-trip] Sin notas."); sys.exit(1)

    role_map = RoleAssigner().assign(mid)
    tpb_raw  = _ticks_per_bar(mid)
    all_ticks = max((n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpb_raw) + 1)

    conv = PianoRollConverter(resolution=args.resolution, window_bars=1)
    rolls = {}
    for role, key in role_map.items():
        notes = note_lists.get(key, [])
        if not notes:
            continue
        rolls[role] = conv.notes_to_roll(notes, tpb_raw, total_bars)

    cfg = {'roles': list(rolls.keys()), 'resolution': args.resolution,
           'pitch_lo': 0, 'pitch_hi': 127}
    palette = _load_palette(None, cfg)
    n_notes = _rolls_to_midi(rolls, cfg, palette, args.output, bpm=args.bpm, threshold=0.5)
    print(f"[round-trip] {args.input} → {args.output}  ({n_notes} notas, roles: {list(rolls.keys())})")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    if args.model_dir:
        model_dir = Path(args.model_dir)
        print(f"[inspect] Modelo en {model_dir}")
        model, cfg = _load_model_and_config(model_dir)
        n_params = sum(p.numel() for p in model.parameters())
        print("─" * 60)
        print(f"  Roles              : {cfg['roles']}")
        print(f"  Resolución         : {cfg['resolution']} ticks/compás")
        print(f"  n_pitch            : {cfg['n_pitch']}  (MIDI {cfg['pitch_lo']}–{cfg['pitch_hi']})")
        print(f"  Compases de ctx    : {cfg['ctx_bars']}")
        print(f"  Dim. tensión       : {cfg['tension_dim']}")
        print(f"  Dim. estilo        : {cfg['style_dim']}")
        print(f"  Dim. condicionam.  : {cfg['cond_dim']}")
        print(f"  Pasos de SDE (N)   : {cfg['sde_steps']}")
        print(f"  Coef. difusión (g) : {cfg['g_coef']}")
        print(f"  Canales ocultos    : {cfg['hidden_channels']}")
        print(f"  Parámetros totales : {n_params:,}")
        hist_path = model_dir / Trainer.HISTORY_NAME
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            if hist.get('val'):
                print(f"  Mejor val_loss     : {min(hist['val']):.4f}")
                print(f"  Épocas entrenadas  : {len(hist['train'])}")
        print("─" * 60)

    if args.data_dir:
        npz_files = sorted(Path(args.data_dir).glob('*.npz'))
        print(f"\n[inspect] {len(npz_files)} archivos .npz en {args.data_dir}")
        total_windows = 0
        for p in npz_files[:5]:
            import numpy as np
            data = dict(np.load(str(p), allow_pickle=True))
            meta = json.loads(str(data['meta_json'][0]))
            total_windows += meta['n_windows']
            print(f"  [{p.stem}] {meta['n_windows']} ventanas, roles: {meta['roles']}")
        if len(npz_files) > 5:
            print(f"  ... y {len(npz_files) - 5} más")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description='DiffFlow Composer — composición generativa vía Diffusion Normalizing Flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # prepare
    p_prep = sub.add_parser('prepare', help='MIDI corpus → piano rolls segmentados (.npz)')
    p_prep.add_argument('--input-dir', required=True)
    p_prep.add_argument('--output-dir', required=True)
    p_prep.add_argument('--resolution', type=int, default=TICKS_PER_BAR_DEFAULT)
    p_prep.add_argument('--window-bars', type=int, default=WINDOW_BARS_DEFAULT)
    p_prep.add_argument('--disable-roles', nargs='+', default=None)
    p_prep.add_argument('--pitch-range', type=int, default=None)
    p_prep.add_argument('--report', action='store_true')
    p_prep.set_defaults(func=cmd_prepare)

    # train
    p_train = sub.add_parser('train', help='Entrena las dos SDEs (forward + backward)')
    p_train.add_argument('--data-dir', required=True)
    p_train.add_argument('--model-dir', required=True)
    p_train.add_argument('--epochs', type=int, default=300)
    p_train.add_argument('--batch-size', type=int, default=8)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--style-dim', type=int, default=16)
    p_train.add_argument('--cond-dim', type=int, default=128)
    p_train.add_argument('--hidden-channels', type=int, default=64)
    p_train.add_argument('--sde-steps', type=int, default=24,
                          help='Pasos de discretización de la SDE en entrenamiento (Ec. 11-12)')
    p_train.add_argument('--g-coef', type=float, default=1.0,
                          help='Coeficiente de difusión g (constante, como en el paper)')
    p_train.add_argument('--patience', type=int, default=50)
    p_train.add_argument('--disable-roles', nargs='+', default=None)
    p_train.add_argument('--resume', action='store_true')
    p_train.set_defaults(func=cmd_train)

    # encode
    p_enc = sub.add_parser('encode', help='MIDI referencia → z_style (.json)')
    p_enc.add_argument('--input', required=True)
    p_enc.add_argument('--model-dir', required=True)
    p_enc.add_argument('--output', default=None)
    p_enc.set_defaults(func=cmd_encode)

    # style-corpus
    p_sc = sub.add_parser('style-corpus', help='Centroide de estilo de una carpeta de MIDIs')
    p_sc.add_argument('--input-dir', required=True)
    p_sc.add_argument('--model-dir', required=True)
    p_sc.add_argument('--output', default=None)
    p_sc.set_defaults(func=cmd_style_corpus)

    # compose
    p_comp = sub.add_parser(
        'compose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Genera una obra nueva',
        epilog=textwrap.dedent("""
            Modos disponibles:
              sample       — generación libre condicionada en --input (estilo) + tensión
              blend        — mezcla estática del estilo de --inputs (usa --weights)
              sweep        — morphing gradual de estilo entre --inputs a lo largo de la pieza
              transfer     — aplica el estilo de --style-from al contenido de --input
              reconstruct  — ida y vuelta x→x_N→x̂ sobre --input (diagnóstico, NO exacto)
        """))
    p_comp.add_argument('--model-dir', required=True)
    p_comp.add_argument('--palette', default=None)
    p_comp.add_argument('--mode', required=True,
                         choices=['sample', 'blend', 'sweep', 'transfer', 'reconstruct'])
    p_comp.add_argument('--input', default=None)
    p_comp.add_argument('--inputs', nargs='+', default=None)
    p_comp.add_argument('--weights', nargs='+', type=float, default=None)
    p_comp.add_argument('--style-from', default=None)
    p_comp.add_argument('--bars', type=int, default=16)
    p_comp.add_argument('--tension', default='arch')
    p_comp.add_argument('--temperature', type=float, default=1.0)
    p_comp.add_argument('--gen-steps', type=int, default=30,
                         help='Pasos de la SDE backward al generar (puede diferir de --sde-steps de train)')
    p_comp.add_argument('--threshold', type=float, default=None)
    p_comp.add_argument('--threshold-pct', type=float, default=99.0)
    p_comp.add_argument('--bpm', type=float, default=120.0)
    p_comp.add_argument('--output', required=True)
    p_comp.set_defaults(func=cmd_compose)

    # round-trip
    p_rt = sub.add_parser('round-trip', help='Diagnóstico: MIDI → piano roll → MIDI (sin modelo)')
    p_rt.add_argument('--input', required=True)
    p_rt.add_argument('--output', default='round_trip.mid')
    p_rt.add_argument('--resolution', type=int, default=TICKS_PER_BAR_DEFAULT)
    p_rt.add_argument('--bpm', type=float, default=120.0)
    p_rt.set_defaults(func=cmd_round_trip)

    # inspect
    p_ins = sub.add_parser('inspect', help='Diagnóstico del modelo y/o los datos')
    p_ins.add_argument('--model-dir', default=None)
    p_ins.add_argument('--data-dir', default=None)
    p_ins.set_defaults(func=cmd_inspect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        # Errores esperables (MIDI corto/vacío, modelo/fichero no encontrado, etc.)
        # ya traen un mensaje claro desde donde se lanzan — se muestran tal cual
        # en vez de como traceback crudo.
        print(f"[{args.command}] Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
