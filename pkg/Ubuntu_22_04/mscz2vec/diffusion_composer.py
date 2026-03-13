#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DIFFUSION COMPOSER  v0.1                                 ║
║         Composición end-to-end mediante Difusión Latente multi-rol           ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Encoder CNN  →  espacio latente continuo                                  ║
║    DDPM (Denoising Diffusion Probabilistic Model)                            ║
║       condicionado en tensión + estilo                                       ║
║    Decoder CNN  →  piano roll multi-rol                                      ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare   — MIDI corpus → piano rolls segmentados por rol (.npz)          ║
║    train     — Entrena el modelo de difusión latente                         ║
║    encode    — MIDI referencia → z_style (.json)                             ║
║    compose   — Genera obra nueva (modos: sample/denoise/blend/sweep)        ║
║    inspect   — Diagnóstico del modelo y espacio latente                      ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch, scipy (opcional para sweep --smooth)                  ║
║                                                                              ║
║  USO RÁPIDO:                                                                 ║
║    python diffusion_composer.py prepare --input-dir midis/ --output-dir data/║
║    python diffusion_composer.py train   --data-dir data/  --model-dir model/ ║
║    python diffusion_composer.py encode  --input ref.mid   --model-dir model/ ║
║    python diffusion_composer.py compose --mode sample --model-dir model/     ║
║                                         --palette palette.json               ║
╚══════════════════════════════════════════════════════════════════════════════╝

python diffusion_composer.py prepare --input-dir midis/ --output-dir data/ --report

python diffusion_composer.py train \
    --data-dir data/ \
    --model-dir model_diff/ \
    --epochs 300 \
    --batch-size 16 \
    --lr 1e-4 \
    --latent-dim 128 \
    --style-dim 16 \
    --diffusion-steps 1000 \
    --pos-weight 5.0 \
    --decoder-dropout 0.2 \
    --patience 50

python diffusion_composer.py compose \
    --model-dir model_diff/ \
    --palette palette.json \
    --mode denoise \
    --input ref.mid \
    --strength 0.2 \
    --ddim-steps 200 \
    --eta 0.0 \
    --temperature 0.5 \
    --bars 16 \
    --threshold-pct 99


python diffusion_composer.py style-corpus \
    --input-dir midis_A/ --model-dir model/ --output z_A.json

python diffusion_composer.py style-corpus \
    --input-dir midis_B/ --model-dir model/ --output z_B.json

python diffusion_composer.py transfer \
    --input cancion.mid --model-dir model/ --palette palette.json \
    --style-from z_A.json --style-to z_B.json \
    --strength 0.8 --output resultado.mid

python diffusion_composer.py transfer ... --progressive


train
-----

**`val_recon` sube mientras `train_recon` baja (sobreajuste del decoder)**

- Brecha < 0.005 → normal, ignorar
- Brecha 0.005–0.010 → vigilar
- Brecha > 0.010 sostenida 5+ épocas → subir `--decoder-dropout` en 0.05 y reiniciar

---

**`loss=nan` en los primeros batches**

- Pocos batches NaN → se saltan automáticamente, ignorar
- Mayoría de batches NaN → bajar `--lr` a la mitad y reiniciar

---

**Spikes periódicos en `val_loss`**

- `recon` y `sparse` suben juntos → batch de validación atípico, ignorar
- Ocurren cada 5–8 épocas regularmente → varianza estadística, ignorar
- `val_diff` también sube en el spike → problema real, vigilar tendencia

---

**Generación ruidosa (`p90 < 0.01` en el diagnóstico)**

- `reconstruct` también ruidoso → decoder no ha convergido, esperar más épocas
- `reconstruct` limpio pero `compose` ruidoso → denoiser no ha convergido, esperar más épocas
- `round-trip` ruidoso → problema en el parser MIDI, revisar los datos

---

**Convergencia lenta (val_loss apenas baja)**

- `val_diff` estancado → el denoiser ha llegado a su límite con esta arquitectura
- `val_recon` estancado alto → el decoder necesita más capacidad (`H_FEAT`) o menos regularización
- Ambos estancados → considerar aumentar `--latent-dim` o cambiar arquitectura

---

**Early stopping activo (sin mejora X/50)**

- < 25/50 → normal
- 25–40/50 → valorar si `val_diff` sigue bajando; si sí, el modelo mejora aunque `val_loss` no baje
- > 40/50 → prepararse para usar el `best_model.pt` y parar

---

**Referencia de valores objetivo para generación usable**

- `val_diff` < 0.004 → generación probablemente usable
- `p90` > 0.05 en `compose` → separación ruido/notas aceptable
- `p90` > 0.01 → mejora respecto a versiones anteriores pero aún ruidoso

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


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI COMUNES  (idénticas a latent_composer.py)
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
#  ASIGNACIÓN DE ROLES  (RoleAssigner) — idéntico a latent_composer.py
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
#  PIANO ROLL CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    def __init__(self, resolution: int = TICKS_PER_BAR_DEFAULT,
                 window_bars: int = WINDOW_BARS_DEFAULT):
        self.resolution  = resolution
        self.window_bars = window_bars

    def notes_to_roll(self, notes, tpb_raw, n_bars) -> 'np.ndarray':
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

    def roll_to_windows(self, roll) -> 'np.ndarray':
        import numpy as np
        n_bars = roll.shape[0]
        if n_bars < self.window_bars:
            return np.zeros((0, self.window_bars, self.resolution, PITCH_CLASSES),
                            dtype=np.float32)
        n_windows = n_bars - self.window_bars + 1
        windows   = np.stack([roll[i:i + self.window_bars]
                               for i in range(n_windows)])
        return windows


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTOR DE TENSIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TensionExtractor:
    TENSION_DIM = 8

    def extract_bar_vectors(self, role_rolls: dict, bars: int) -> 'np.ndarray':
        import numpy as np
        vectors = np.zeros((bars, self.TENSION_DIM), dtype=np.float32)
        for bar in range(bars):
            combined    = np.zeros((PITCH_CLASSES,), dtype=np.float32)
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
            capacity       = resolution * PITCH_CLASSES
            tension        = self._lerdahl_proxy(pitches_active)
            density        = min(float(total_events) / max(capacity * len(role_rolls), 1) * 20, 1.0)
            poly           = min(n_active / 12.0, 1.0)
            reg_mean       = float(np.mean(pitches_active)) / 127 if n_active > 0 else 0.5
            reg_spread     = float(np.ptp(pitches_active)) / 127 if n_active > 1 else 0.0
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
        import numpy as np
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
#  COMANDO: prepare  (idéntico a latent_composer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    midi_path, output_dir, resolution, window_bars = args_tuple
    import numpy as np

    stem = midi_path.stem
    stats_partial = {r: 0 for r in ROLES}
    stats_partial['files_ok']      = 0
    stats_partial['files_skipped'] = 0
    stats_partial['total_windows'] = 0

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
    all_ticks  = max((n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpb_raw) + 1)

    role_rolls  = {}
    roles_found = []
    for role, key in role_assignment.items():
        notes = note_lists.get(key, [])
        if not notes:
            continue
        roll = converter.notes_to_roll(notes, tpb_raw, total_bars)
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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution  = args.resolution
    window_bars = args.window_bars

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

    task_args = [(midi_path, str(output_dir), resolution, window_bars) for midi_path in midi_files]

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
        'x'        : Tensor (N_ROLES, resolution, 128)  — compás a difundir (target)
        'context'  : Tensor (N_ROLES, ctx_bars, resolution, 128)  — contexto previo
        'tension'  : Tensor (TENSION_DIM,)
        'role_mask': Tensor (N_ROLES,) bool
    """
    def __init__(self, data_dir: str, roles: list = None, augment: bool = False):
        import numpy as np
        self.samples  = []
        self.roles    = roles or ROLES
        self.n_roles  = len(self.roles)
        self.augment  = augment
        self._cache   = {}

        npz_files = sorted(Path(data_dir).glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No hay .npz en {data_dir}")

        for path in npz_files:
            try:
                data = dict(np.load(str(path), allow_pickle=True))
                meta = json.loads(str(data['meta_json'][0]))
                for i in range(meta['n_windows']):
                    self.samples.append((str(path), i, meta))
                self._cache[str(path)] = data
            except Exception:
                continue

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

        # x = último compás de la ventana (el "target" de difusión)
        # context = compases anteriores (condicionamiento temporal)
        x_parts   = []
        ctx_parts = []
        mask      = []

        for role in self.roles:
            key = f'roll_{role}'
            if key in data:
                window = data[key][widx]                  # (window_bars, resolution, 128)
                x_parts.append(window[-1])                # último compás
                ctx_parts.append(window[:ctx_bars])       # compases de contexto
                mask.append(True)
            else:
                x_parts.append(np.zeros((resolution, 128), dtype=np.float32))
                ctx_parts.append(np.zeros((ctx_bars, resolution, 128), dtype=np.float32))
                mask.append(False)

        x       = torch.tensor(np.stack(x_parts,   axis=0))   # (N_ROLES, res, 128)
        context = torch.tensor(np.stack(ctx_parts,  axis=0))   # (N_ROLES, ctx_bars, res, 128)
        tension = torch.tensor(data['tension'][widx])           # (TENSION_DIM,)
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
#  ARQUITECTURA: ENCODER / DECODER CNN (espacio latente comprimido)
# ══════════════════════════════════════════════════════════════════════════════

def _build_codec(latent_dim: int, n_roles: int, resolution: int,
                 decoder_dropout: float = 0.1):
    """
    Construye el par Encoder / Decoder CNN que mapea:
        piano_roll (N_ROLES, resolution, 128) ↔ z (latent_dim,)

    v2 — cambios respecto a v1:
      · Encoder más profundo: 3 bloques Conv con BatchNorm en lugar de 2
      · H_FEAT aumentado a 256 para mayor capacidad de representación
      · Decoder con Dropout entre bloques para regularización
        (frena el sobreajuste que aparece ~época 85 en v1)
      · Skip connections en el decoder: concatena features del encoder
        para recuperar detalles finos perdidos en la compresión
      · latent_dim default 128 (era 64) para mayor capacidad del espacio latente
    """
    import torch
    import torch.nn as nn

    PITCH   = 128
    H_FEAT  = 128   # igual que v1 — con resolution=48 subir esto dispara el modelo a >1GB
                    # La mejora real viene de BatchNorm + Dropout, no de más canales
    res4    = resolution // 4
    p4      = PITCH // 4
    flat    = n_roles * H_FEAT * res4 * p4

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.per_role = nn.ModuleList([
                nn.Sequential(
                    # Bloque 1: stride 1, mantiene resolución
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # Bloque 2: stride 2 → /2
                    nn.Conv2d(64, H_FEAT, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(H_FEAT),
                    nn.ReLU(),
                    # Bloque 3: stride 2 → /4
                    nn.Conv2d(H_FEAT, H_FEAT, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(H_FEAT),
                    nn.ReLU(),
                )
                for _ in range(n_roles)
            ])
            self.fc_mu     = nn.Linear(flat, latent_dim)
            self.fc_logvar = nn.Linear(flat, latent_dim)

        def forward(self, x):
            # x: (B, N_ROLES, resolution, 128)
            B = x.size(0)
            feats = []
            for r, conv in enumerate(self.per_role):
                xr = x[:, r, :, :].unsqueeze(1)   # (B, 1, res, 128)
                feats.append(conv(xr))             # (B, H_FEAT, res/4, 128/4)
            h = torch.cat(feats, dim=1).reshape(B, -1)  # (B, flat)
            return self.fc_mu(h), self.fc_logvar(h)

    class _Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(latent_dim, flat)
            self.per_role = nn.ModuleList([
                nn.Sequential(
                    # Bloque 1: upsample ×2
                    nn.ConvTranspose2d(H_FEAT, H_FEAT, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(H_FEAT),
                    nn.ReLU(),
                    nn.Dropout2d(decoder_dropout),
                    # Bloque 2: upsample ×2 → resolución original
                    nn.ConvTranspose2d(H_FEAT, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(decoder_dropout),
                    # Bloque 3: refinamiento sin cambio de escala
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    # Salida: sin Sigmoid — usamos BCEWithLogits en el loss
                    nn.Conv2d(32, 1, kernel_size=3, padding=1),
                )
                for _ in range(n_roles)
            ])

        def forward(self, z):
            B = z.size(0)
            h = self.fc(z).reshape(B, n_roles, H_FEAT, res4, p4)
            rolls = []
            for r, deconv in enumerate(self.per_role):
                xr = deconv(h[:, r])          # (B, 1, resolution, 128) — logits
                rolls.append(xr.squeeze(1))   # (B, res, 128)
            return torch.stack(rolls, dim=1)  # (B, N_ROLES, res, 128) — logits

        def decode_probs(self, z):
            """Versión con Sigmoid para inferencia (produce probabilidades [0,1])."""
            import torch
            return torch.sigmoid(self.forward(z))

    return _Encoder(), _Decoder()


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA: DDPM  (Denoising Diffusion Probabilistic Model)
# ══════════════════════════════════════════════════════════════════════════════

def _build_diffusion_model(latent_dim: int, style_dim: int, tension_dim: int,
                           n_roles: int, resolution: int,
                           n_steps: int = 1000):
    """
    Construye:
      1. CosineSchedule  — schedule de ruido β(t)
      2. DenoisingNet    — red U-Net-like 1D que predice ε(z_t, t, cond)
      3. DiffusionModel  — wrapper con forward / sample / ddim_sample

    Condicionamiento:
        cond = Linear(context_feat + tension + style) → cond_dim
        Inyectado en cada bloque temporal vía cross-attention o concat.

    Arquitectura del DenoisingNet (simple pero efectivo):
        z_t ∈ R^{latent_dim}  +  t_emb ∈ R^{128}  +  cond ∈ R^{cond_dim}
        → MLP ResNet de 4 capas → ε̂ ∈ R^{latent_dim}
    """
    import torch
    import torch.nn as nn
    import math

    COND_DIM   = 256
    H_DIM      = 512
    T_EMB_DIM  = 128

    # ── Context encoder (para condicionamiento) ────────────────────────────
    # Toma el contexto (N_ROLES, ctx_bars, res, 128) y lo comprime a cond_dim
    ctx_flat = n_roles * resolution * 128   # puede ser enorme; usamos avg pool

    class _ContextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Compresión espacial: avgpool sobre (res, 128) → escalar por rol y bar
            # Luego MLP ligero
            self.pool = nn.AdaptiveAvgPool2d((4, 8))   # → n_roles * ctx_bars * 4 * 8
            # Dimensión dinámica: se calcula en forward
            self.fc = nn.LazyLinear(COND_DIM)

        def forward(self, ctx, tension, z_style):
            # ctx: (B, N_ROLES, ctx_bars, res, 128)
            B, R, C, Res, P = ctx.shape
            flat_ctx = ctx.reshape(B * R * C, 1, Res, P)
            pooled   = self.pool(flat_ctx).reshape(B, -1)   # (B, R*C*4*8)
            cond_raw = torch.cat([pooled, tension, z_style], dim=-1)
            return self.fc(cond_raw)   # (B, COND_DIM)

    # ── Sinusoidal time embedding ──────────────────────────────────────────
    class _TimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.proj = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, t):
            # t: (B,) enteros [0, n_steps-1]
            half  = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / half
            )
            args  = t[:, None].float() * freqs[None]
            emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return self.proj(emb)   # (B, T_EMB_DIM)

    # ── ResBlock para el DenoisingNet ──────────────────────────────────────
    class _ResBlock(nn.Module):
        def __init__(self, dim: int, cond_dim: int, t_dim: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.fc1   = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.fc2   = nn.Linear(dim, dim)
            self.act   = nn.SiLU()
            # Proyecciones de condicionamiento (tiempo + contexto)
            self.t_proj    = nn.Linear(t_dim, dim)
            self.cond_proj = nn.Linear(cond_dim, dim)

        def forward(self, x, t_emb, cond):
            h = self.norm1(x)
            h = self.fc1(h) + self.t_proj(t_emb) + self.cond_proj(cond)
            h = self.act(h)
            h = self.norm2(h)
            h = self.fc2(h)
            return x + h   # skip connection

    # ── Red de denoising ───────────────────────────────────────────────────
    class _DenoisingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(latent_dim, H_DIM)
            self.t_emb      = _TimeEmbedding(T_EMB_DIM)
            self.ctx_enc    = _ContextEncoder()
            self.blocks     = nn.ModuleList([
                _ResBlock(H_DIM, COND_DIM, T_EMB_DIM) for _ in range(6)
            ])
            self.out_norm   = nn.LayerNorm(H_DIM)
            self.out_proj   = nn.Linear(H_DIM, latent_dim)

        def forward(self, z_t, t, context, tension, z_style):
            """
            z_t     : (B, latent_dim)   — latente ruidoso
            t       : (B,)              — paso de difusión
            context : (B, N_ROLES, ctx_bars, res, 128)
            tension : (B, tension_dim)
            z_style : (B, style_dim)
            → ε̂    : (B, latent_dim)
            """
            cond  = self.ctx_enc(context, tension, z_style)  # (B, COND_DIM)
            t_emb = self.t_emb(t)                             # (B, T_EMB_DIM)
            h     = self.input_proj(z_t)                      # (B, H_DIM)
            for block in self.blocks:
                h = block(h, t_emb, cond)
            h = self.out_norm(h)
            return self.out_proj(h)   # (B, latent_dim)

    # ── Schedule coseno (mejor que lineal) ────────────────────────────────
    class _CosineSchedule:
        """
        Schedule coseno de Nichol & Dhariwal 2021.
        α_t = cos((t/T + s)/(1 + s) · π/2)²  con s = 0.008
        """
        def __init__(self, n_steps: int = 1000, s: float = 0.008):
            import torch, math
            self.n_steps = n_steps
            steps  = torch.arange(n_steps + 1, dtype=torch.float64)
            alphas = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas  = 1.0 - alphas[1:] / alphas[:-1]
            betas  = betas.clamp(0.0, 0.999).float()

            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
            self.register = {
                'betas':               betas,
                'alphas_cumprod':      alphas_cumprod,
                'sqrt_alphas_cumprod': alphas_cumprod.sqrt(),
                'sqrt_one_minus_alphas_cumprod': (1.0 - alphas_cumprod).sqrt(),
            }

        def to(self, device):
            self.register = {k: v.to(device) for k, v in self.register.items()}
            return self

        def q_sample(self, x0, t, noise=None):
            """Añade ruido a x0 en el paso t."""
            import torch
            if noise is None:
                noise = torch.randn_like(x0)
            sqrt_a = self.register['sqrt_alphas_cumprod'][t][:, None]
            sqrt_b = self.register['sqrt_one_minus_alphas_cumprod'][t][:, None]
            return sqrt_a * x0 + sqrt_b * noise, noise

        def p_sample(self, net, z_t, t_scalar, context, tension, z_style):
            """Un paso de denoising DDPM (inverso)."""
            import torch
            B     = z_t.size(0)
            t_ten = torch.full((B,), t_scalar, device=z_t.device, dtype=torch.long)
            with torch.no_grad():
                eps_pred = net(z_t, t_ten, context, tension, z_style)

            alpha_t    = self.register['alphas_cumprod'][t_scalar]
            alpha_tm1  = self.register['alphas_cumprod'][t_scalar - 1] if t_scalar > 0 else torch.tensor(1.0)
            beta_t     = 1 - alpha_t / alpha_tm1

            coef1 = (1.0 / (1 - alpha_t).sqrt())
            coef2 = beta_t / (1 - alpha_t).sqrt()
            mean  = coef1 * (z_t - coef2 * eps_pred)

            if t_scalar > 0:
                noise = torch.randn_like(z_t)
                var   = beta_t.sqrt().to(z_t.device) * noise
            else:
                var = 0.0
            return mean + var

        def ddim_sample(self, net, z_t, t_scalar, t_prev,
                        context, tension, z_style, eta: float = 0.0):
            """Un paso DDIM (más rápido, menos pasos en inference)."""
            import torch
            B     = z_t.size(0)
            t_ten = torch.full((B,), t_scalar, device=z_t.device, dtype=torch.long)
            with torch.no_grad():
                eps_pred = net(z_t, t_ten, context, tension, z_style)

            alpha_t  = self.register['alphas_cumprod'][t_scalar]
            alpha_tp = self.register['alphas_cumprod'][t_prev] if t_prev >= 0 else torch.tensor(1.0)

            x0_pred  = (z_t - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
            x0_pred  = x0_pred.clamp(-5, 5)

            sigma    = eta * ((1 - alpha_tp) / (1 - alpha_t) * (1 - alpha_t / alpha_tp)).sqrt()
            dir_xt   = (1 - alpha_tp - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            noise    = sigma * torch.randn_like(z_t) if eta > 0 else 0.0
            return alpha_tp.sqrt() * x0_pred + dir_xt + noise

    schedule = _CosineSchedule(n_steps=n_steps)
    net      = _DenoisingNet()
    return net, schedule


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO COMPLETO: DiffusionComposer
# ══════════════════════════════════════════════════════════════════════════════

def _build_full_model(latent_dim: int, style_dim: int, tension_dim: int,
                      n_roles: int, window_bars: int, resolution: int,
                      n_diffusion_steps: int = 1000,
                      pos_weight: float = 5.0,
                      decoder_dropout: float = 0.1):
    """
    Integra Encoder + DenoisingNet + Schedule + Decoder en un módulo único.
    """
    import torch
    import torch.nn as nn

    encoder, decoder = _build_codec(latent_dim, n_roles, resolution,
                                    decoder_dropout=decoder_dropout)
    denoiser, schedule = _build_diffusion_model(
        latent_dim, style_dim, tension_dim, n_roles, resolution, n_diffusion_steps)

    class _StyleProjector(nn.Module):
        """Mapea z latente → z_style condicionante."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, style_dim),
                nn.Tanh(),
            )
        def forward(self, z):
            return self.proj(z)

    class _DiffusionComposer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder      = encoder
            self.decoder      = decoder
            self.denoiser     = denoiser
            self.schedule     = schedule
            self.style_proj   = _StyleProjector()
            self.n_steps      = n_diffusion_steps
            self.latent_dim   = latent_dim
            self.style_dim    = style_dim
            self.tension_dim  = tension_dim
            self.n_roles      = n_roles
            self.window_bars  = window_bars
            self.resolution   = resolution
            self.pos_weight      = pos_weight
            self.decoder_dropout = decoder_dropout

        def reparametrize(self, mu, logvar):
            if self.training:
                std = (0.5 * logvar).exp()
                return mu + std * torch.randn_like(std)
            return mu

        def forward(self, x, context, tension, t=None):
            """
            x        : (B, N_ROLES, res, 128)  — compás target
            context  : (B, N_ROLES, ctx_bars, res, 128)
            tension  : (B, tension_dim)
            t        : (B,) opcional; si None se muestrea aleatoriamente

            Devuelve : loss (escalar), dict de métricas
            """
            import torch
            B      = x.size(0)
            device = x.device

            # 1. Codificar → z₀
            mu, logvar = self.encoder(x)
            z0 = self.reparametrize(mu, logvar)

            # 2. Calcular z_style desde z₀
            z_style = self.style_proj(mu.detach())

            # 3. Muestrear t aleatorio y añadir ruido
            if t is None:
                t = torch.randint(0, self.n_steps, (B,), device=device)
            self.schedule.to(device)
            z_t, eps_real = self.schedule.q_sample(z0, t)

            # 4. Red de denoising predice el ruido
            eps_pred = self.denoiser(z_t, t, context, tension, z_style)

            # 5. Loss de difusión y KL
            diff_loss = torch.nn.functional.mse_loss(eps_pred, eps_real)
            # KL clampado: evita explosiones en las primeras épocas cuando el
            # encoder aún no ha aprendido a producir varianzas estables.
            kl_loss   = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=-1).mean()
            kl_loss   = kl_loss.clamp(max=100.0)   # nunca más de 100 por batch

            # 6. Loss de reconstrucción con pos_weight moderado.
            # El decoder ahora produce logits directamente (sin Sigmoid final),
            # por lo que usamos BCEWithLogits sin necesidad de invertir sigmoid.
            recon_logits = self.decoder(z0)   # logits (B, N_ROLES, res, 128)
            pw           = torch.tensor(self.pos_weight, device=device)
            recon_loss   = torch.nn.functional.binary_cross_entropy_with_logits(
                recon_logits, x, pos_weight=pw, reduction='mean')

            # 7. Sparsity loss sobre probabilidades (sigmoid de logits)
            recon_probs    = torch.sigmoid(recon_logits)
            density_target = x.mean()
            density_pred   = recon_probs.mean()
            sparsity_loss  = torch.relu(density_pred - density_target * 2.0).mean()

            loss = diff_loss + 0.0001 * kl_loss + 2.0 * recon_loss + 5.0 * sparsity_loss

            return loss, {'diff': diff_loss.item(),
                          'kl':   kl_loss.item(),
                          'recon': recon_loss.item(),
                          'sparse': sparsity_loss.item()}

        @torch.no_grad()
        def sample(self, context, tension, z_style,
                   temperature: float = 1.0,
                   n_ddim_steps: int  = 50,
                   eta: float         = 0.0):
            """
            Genera un compás nuevo via DDIM (rápido) o DDPM completo.

            context  : (B, N_ROLES, ctx_bars, res, 128)
            tension  : (B, tension_dim)
            z_style  : (B, style_dim)
            → roll   : (B, N_ROLES, res, 128)
            """
            import torch
            B      = context.size(0)
            device = context.device
            self.schedule.to(device)

            # Ruido inicial escalado por temperatura
            z = torch.randn(B, self.latent_dim, device=device) * temperature

            # DDIM steps: subsecuencia de pasos
            total   = self.n_steps
            step_sz = max(total // n_ddim_steps, 1)
            ts      = list(range(total - 1, -1, -step_sz))

            for i, t_cur in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                z = self.schedule.ddim_sample(
                    self.denoiser, z, t_cur, t_prev,
                    context, tension, z_style, eta=eta)

            return self.decoder.decode_probs(z)

        @torch.no_grad()
        def denoise_from_ref(self, x_ref, context, tension, z_style,
                             strength: float = 0.7, n_ddim_steps: int = 50, eta: float = 0.0):
            """
            Inicia el denoising a partir de x_ref parcialmente ruidificado.
            strength = 0 → reconstrucción exacta; 1 → generación libre.
            """
            import torch
            B      = x_ref.size(0)
            device = x_ref.device
            self.schedule.to(device)

            # Codificar ref → z₀
            mu, logvar = self.encoder(x_ref)
            z0 = self.reparametrize(mu, logvar)

            # Añadir ruido hasta t = int(strength * n_steps)
            t_start = int(strength * self.n_steps)
            t_ten   = torch.full((B,), t_start, device=device, dtype=torch.long)
            z_t, _  = self.schedule.q_sample(z0, t_ten)

            # DDIM desde t_start → 0
            total   = t_start
            step_sz = max(total // n_ddim_steps, 1)
            ts      = list(range(t_start - 1, -1, -step_sz))

            z = z_t
            for i, t_cur in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                z = self.schedule.ddim_sample(
                    self.denoiser, z, t_cur, t_prev,
                    context, tension, z_style, eta=eta)

            return self.decoder.decode_probs(z)

    return _DiffusionComposer()


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE TIEMPO
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    """Formatea segundos como string legible: 1h 23m 45s / 4m 12s / 38s."""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60:02d}s"
    if s >= 60:
        return f"{s//60}m {s%60:02d}s"
    return f"{s}s"


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'
    CONFIG_NAME     = 'model_config.json'

    def __init__(self, model, optimizer, model_dir: Path,
                 patience: int = 50):
        self.model      = model
        self.optimizer  = optimizer
        self.model_dir  = model_dir
        self.patience   = patience

        self.history       = {'train': [], 'val': [], 'val_diff': [], 'val_recon': []}
        self.best_val_loss = float('inf')
        self.no_improve    = 0
        self.start_epoch   = 0

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        import torch
        state = {
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss':   self.best_val_loss,
            'no_improve':      self.no_improve,
            'history':         self.history,
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

    def _run_epoch(self, loader, training: bool, epoch: int = 0, n_epochs: int = 0):
        import torch, time
        self.model.train(training)
        total_loss = diff_sum = recon_sum = sparse_sum = 0.0
        n_batches  = 0
        phase      = 'train' if training else 'val  '

        # Total de batches para la barra de progreso intra-época
        n_total = (len(loader.dataset) // max(loader.batch_size, 1) + 1
                   if hasattr(loader, 'dataset') and hasattr(loader, 'batch_size')
                   else None)

        batch_times = []
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                t0      = time.time()
                device  = next(self.model.parameters()).device
                x       = batch['x'].to(device, non_blocking=True)
                context = batch['context'].to(device, non_blocking=True)
                tension = batch['tension'].to(device, non_blocking=True)

                loss, metrics = self.model(x, context, tension)

                # Saltar batch si el loss es NaN (puede ocurrir con dropout alto
                # y BatchNorm en las primeras iteraciones)
                import math
                if math.isnan(loss.item()):
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
                diff_sum   += metrics['diff']
                recon_sum  += metrics['recon']
                sparse_sum += metrics.get('sparse', 0.0)
                n_batches  += 1

                batch_times.append(time.time() - t0)
                if len(batch_times) > 20:
                    batch_times.pop(0)

                # ── Barra de progreso intra-época ────────────────────────────
                avg_loss = total_loss / n_batches
                if n_total:
                    pct    = n_batches / n_total
                    bar_w  = 18
                    filled = int(pct * bar_w)
                    bar    = '█' * filled + '░' * (bar_w - filled)
                    prog   = f"[{bar}] {n_batches}/{n_total}"
                    # ETA intra-época
                    if batch_times:
                        avg_bt   = sum(batch_times) / len(batch_times)
                        rem_bt   = avg_bt * (n_total - n_batches)
                        eta_str  = f"  ~{_fmt_time(rem_bt)}"
                    else:
                        eta_str = ""
                else:
                    prog    = f"batch {n_batches}"
                    eta_str = ""

                print(f"\r  [{phase}] ep {epoch+1}/{n_epochs}  {prog}"
                      f"  loss={avg_loss:.4f}"
                      f"  diff={diff_sum/n_batches:.4f}"
                      f"  recon={recon_sum/n_batches:.4f}"
                      f"  sparse={sparse_sum/n_batches:.4f}"
                      f"{eta_str}   ",
                      end='', flush=True)

        # Limpiar la línea de progreso al terminar la fase
        print(' ' * 120, end='\r')

        n = max(n_batches, 1)
        return total_loss / n, diff_sum / n, recon_sum / n

    def train(self, train_loader, val_loader, n_epochs: int):
        import torch, time

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        print(f"\n{'═'*64}")
        print(f"  DIFFUSION COMPOSER — Entrenamiento")
        print(f"  Épocas máx. : {n_epochs}   Early stopping: {self.patience} sin mejora")
        print(f"  Dispositivo : {device}")
        print(f"  Modelo dir  : {self.model_dir}")
        print(f"{'═'*64}\n")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, '_resume', False):
            self.load_checkpoint()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6)

        epoch_times = []
        train_start = time.time()
        total_epochs = n_epochs   # total desde 0 (para ETA global correcta)

        for epoch in range(self.start_epoch, n_epochs):

            # ── ETA global ───────────────────────────────────────────────────
            if epoch_times:
                avg_ep  = sum(epoch_times) / len(epoch_times)
                eta_sec = avg_ep * (n_epochs - epoch)
                eta_str = f"  ETA {_fmt_time(eta_sec)}"
            else:
                eta_str = ""

            lr_current = self.optimizer.param_groups[0]['lr']
            print(f"  Época {epoch+1:>4}/{n_epochs}"
                  f"  lr={lr_current:.2e}"
                  f"{eta_str}",
                  flush=True)

            epoch_t0 = time.time()

            tr_loss, tr_diff, tr_recon = self._run_epoch(
                train_loader, True, epoch, n_epochs)
            vl_loss, vl_diff, vl_recon = self._run_epoch(
                val_loader, False, epoch, n_epochs)

            scheduler.step()

            epoch_elapsed = time.time() - epoch_t0
            epoch_times.append(epoch_elapsed)
            if len(epoch_times) > 5:   # ventana deslizante de 5 épocas
                epoch_times.pop(0)

            self.history['train'].append(tr_loss)
            self.history['val'].append(vl_loss)
            self.history['val_diff'].append(vl_diff)
            self.history['val_recon'].append(vl_recon)

            is_best = vl_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = vl_loss
                self.no_improve    = 0
            else:
                self.no_improve   += 1

            self.save_checkpoint(epoch, vl_loss, is_best)

            # ── Barra de progreso basada en val_recon ─────────────────────
            ref_recon = self.history['val_recon'][0] if self.history['val_recon'] else vl_recon
            bar_w     = 24
            import math
            if math.isnan(vl_recon) or math.isnan(ref_recon) or ref_recon < 1e-6:
                progress = 0
            else:
                progress = min(int((1 - vl_recon / ref_recon) * bar_w), bar_w)
            bar       = '█' * max(progress, 0) + '░' * (bar_w - max(progress, 0))

            best_marker = ' ◀ mejor' if is_best else ''
            stop_str    = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                           if self.no_improve > 0 else '')

            print(f"         train={tr_loss:.4f}  val={vl_loss:.4f}"
                  f"  (diff={vl_diff:.4f}  recon={vl_recon:.4f})"
                  f"  │{bar}│  {_fmt_time(epoch_elapsed)}/época"
                  f"{best_marker}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        total_elapsed = time.time() - train_start
        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(total_elapsed)}.")
        print(f"  Mejor val_loss : {self.best_val_loss:.4f}")
        print(f"  Modelos en     : {self.model_dir}")
        print(f"{'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader, random_split

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("[train] Cargando dataset ...")
    dataset    = MidiRollDataset(str(data_dir))
    n_val      = max(1, int(len(dataset) * 0.1))
    n_train    = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate_fn, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate_fn, num_workers=0, pin_memory=False)

    # Inferir hiperparámetros del corpus
    sample     = dataset[0]
    n_roles    = sample['x'].shape[0]
    resolution = sample['x'].shape[1]
    ctx_bars   = sample['context'].shape[1]
    window_bars = ctx_bars + 1
    tension_dim = sample['tension'].shape[0]

    print(f"[train] n_roles={n_roles}  resolution={resolution}  "
          f"window_bars={window_bars}  tension_dim={tension_dim}")
    print(f"[train] latent_dim={args.latent_dim}  style_dim={args.style_dim}  "
          f"diffusion_steps={args.diffusion_steps}")

    model = _build_full_model(
        latent_dim        = args.latent_dim,
        style_dim         = args.style_dim,
        tension_dim       = tension_dim,
        n_roles           = n_roles,
        window_bars       = window_bars,
        resolution        = resolution,
        n_diffusion_steps = args.diffusion_steps,
        pos_weight        = args.pos_weight,
        decoder_dropout   = args.decoder_dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer   = Trainer(model, optimizer, model_dir, patience=args.patience)
    trainer._resume = args.resume

    # Guardar configuración
    cfg = {
        'latent_dim':       args.latent_dim,
        'style_dim':        args.style_dim,
        'tension_dim':      tension_dim,
        'n_roles':          n_roles,
        'roles':            dataset.roles,
        'window_bars':      window_bars,
        'resolution':       resolution,
        'diffusion_steps':  args.diffusion_steps,
        'pos_weight':       args.pos_weight,
        'decoder_dropout':  args.decoder_dropout,
    }
    with open(model_dir / Trainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer.train(train_loader, val_loader, args.epochs)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE CARGA
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
        latent_dim        = cfg['latent_dim'],
        style_dim         = cfg['style_dim'],
        tension_dim       = cfg['tension_dim'],
        n_roles           = cfg['n_roles'],
        window_bars       = cfg['window_bars'],
        resolution        = cfg['resolution'],
        n_diffusion_steps = cfg.get('diffusion_steps', 1000),
        pos_weight        = cfg.get('pos_weight', 5.0),
        decoder_dropout   = cfg.get('decoder_dropout', 0.1),
    )
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state['model_state'])
    model.train(False)
    return model, cfg


def _midi_to_rolls(midi_path: str, cfg: dict) -> dict:
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

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    rolls    = {}
    for role, stream_key in role_map.items():
        notes = note_lists[stream_key]
        roll  = conv.notes_to_roll(notes, tpb * 4, total_bars)
        rolls[role] = roll

    return rolls


def _rolls_to_context_tensor(rolls: dict, cfg: dict) -> 'np.ndarray':
    """Convierte rolls → tensor de contexto (N_ROLES, ctx_bars, res, 128)."""
    import numpy as np

    window_bars = cfg['window_bars']
    resolution  = cfg['resolution']
    n_roles     = cfg['n_roles']
    role_list   = cfg['roles']
    ctx_bars    = window_bars - 1

    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 0
    if n_bars < ctx_bars:
        raise ValueError(f"MIDI demasiado corto: {n_bars} compases, se necesitan ≥ {ctx_bars}")

    ctx = np.zeros((n_roles, ctx_bars, resolution, 128), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls:
            ctx[ridx] = rolls[role][:ctx_bars]

    return ctx   # (N_ROLES, ctx_bars, res, 128)


def _encode_ref(midi_path: str, model, cfg: dict) -> tuple:
    """Devuelve (z_mean, z_style) para un MIDI de referencia."""
    import torch, numpy as np

    rolls = _midi_to_rolls(midi_path, cfg)
    # Usar el último segmento disponible para construir x y ctx
    window_bars = cfg['window_bars']
    resolution  = cfg['resolution']
    n_roles     = cfg['n_roles']
    role_list   = cfg['roles']
    ctx_bars    = window_bars - 1

    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 0
    x_np   = np.zeros((n_roles, resolution, 128), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls and n_bars > 0:
            x_np[ridx] = rolls[role][min(n_bars - 1, ctx_bars)]

    x = torch.tensor(x_np).unsqueeze(0)   # (1, N_ROLES, res, 128)
    with torch.no_grad():
        mu, logvar = model.encoder(x)
        z_style    = model.style_proj(mu)
    return mu[0].numpy(), z_style[0].numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  PERFILES DE TENSIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _tension_profile(profile: str, n_bars: int, tension_dim: int) -> 'np.ndarray':
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
            raw   = json.load(f)
        raw_t = np.interp(t, np.linspace(0, 1, len(raw)), raw)
        curve = raw_t
    else:
        print(f"[compose] Perfil '{profile}' desconocido — usando arch")
        curve = np.sin(t * np.pi)

    # Expandir curva escalar → vector de tensión completo
    out = np.zeros((n_bars, tension_dim), dtype=np.float32)
    out[:, 0] = curve          # tension_lerdahl
    out[:, 1] = curve * 0.7   # density
    out[:, 7] = curve          # arousal
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERER: rolls → MIDI  (idéntico a latent_composer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(roll: 'np.ndarray', percentile: float = 85.0) -> float:
    """
    Calcula un umbral adaptativo para binarizar el piano roll del decoder.

    Toma el percentil 'percentile' de todos los valores > 0.001.
    Un percentil alto (95–99) = umbral alto = pocas notas (menos ruido).
    Un percentil bajo (70–80) = umbral bajo = más notas (más densidad).

    El rango útil típico es 75–97 dependiendo del estado del entrenamiento.
    """
    import numpy as np
    flat    = roll.flatten()
    nonzero = flat[flat > 0.001]
    if len(nonzero) == 0:
        return 0.5   # señal nula: umbral alto → MIDI vacío con aviso
    return float(np.percentile(nonzero, percentile))


def _rolls_to_midi(bars_per_role: dict, cfg: dict, palette: dict,
                   output_path: str, bpm: float = 120.0,
                   threshold: float = None):
    """
    Convierte rolls continuos [0,1] → MIDI.

    threshold: umbral de binarización. Si None, se calcula de forma adaptativa
               por rol usando el percentil 85 de las activaciones.
    """
    import mido, numpy as np

    resolution  = cfg['resolution']
    tpb         = 480
    ticks_bar   = tpb * 4
    ticks_tick  = ticks_bar / resolution

    mid        = mido.MidiFile(ticks_per_beat=tpb)
    tempo_val  = int(60_000_000 / bpm)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=tempo_val, time=0))
    mid.tracks.append(t0)

    n_notes_total = 0

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]   # (n_bars, res, 128)

        # Umbral adaptativo por rol
        thr = threshold if threshold is not None else _adaptive_threshold(roll)

        pal  = palette.get(role, {})
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        # Binarizar con umbral adaptativo
        binary = (roll > thr).astype(np.float32)

        # Suavizado temporal mínimo: eliminar notas aisladas de 1 tick
        # (artefacto común del decoder: activa/desactiva en ticks alternos)
        for b in range(binary.shape[0]):
            for p in range(128):
                col = binary[b, :, p]
                for t in range(1, len(col) - 1):
                    if col[t] == 1 and col[t-1] == 0 and col[t+1] == 0:
                        binary[b, t, p] = 0   # nota de 1 tick → silencio

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
                track.append(mido.Message('note_on',  channel=ch, note=pitch, velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', channel=ch, note=pitch, velocity=0,   time=delta))
            prev_tick = abs_tick

    mid.save(output_path)
    return n_notes_total


def _load_palette(palette_path: str, cfg: dict) -> dict:
    DEFAULT_PALETTE = {
        'melody':        {'program': 73, 'channel': 0, 'velocity': 90},
        'counterpoint':  {'program': 68, 'channel': 1, 'velocity': 80},
        'accompaniment': {'program': 48, 'channel': 2, 'velocity': 70},
        'bass':          {'program': 43, 'channel': 3, 'velocity': 85},
        'percussion':    {'program':  0, 'channel': 9, 'velocity': 90},
    }
    with open(palette_path) as f:
        user = json.load(f)
    palette = {**DEFAULT_PALETTE}
    for role, params in user.items():
        palette[role] = {**DEFAULT_PALETTE.get(role, {}), **params}
    return palette


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: encode
# ══════════════════════════════════════════════════════════════════════════════

def cmd_encode(args):
    model_dir = Path(args.model_dir)
    print(f"[encode] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    print(f"[encode] Procesando {args.input} ...")
    z_mean, z_style = _encode_ref(args.input, model, cfg)

    rolls = _midi_to_rolls(args.input, cfg)
    tension_vecs = TensionExtractor().extract_bar_vectors(
        rolls, min(r.shape[0] for r in rolls.values()))
    tension_mean = tension_vecs.mean(axis=0).tolist()

    out_path = args.output or (Path(args.input).stem + '.style.json')
    payload  = {
        'source':       args.input,
        'model_dir':    str(model_dir),
        'latent_dim':   cfg['latent_dim'],
        'style_dim':    cfg['style_dim'],
        'z_context':    z_mean.tolist(),
        'z_style':      z_style.tolist(),
        'tension_mean': tension_mean,
        'roles_found':  list(rolls.keys()),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[encode] Guardado en {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: style-corpus
# ══════════════════════════════════════════════════════════════════════════════

def cmd_style_corpus(args):
    """
    Encoda todos los MIDIs de una carpeta y calcula el centroide de z_style.
    El resultado es un .json compatible con el formato de `encode`.

    Uso:
        python diffusion_composer.py style-corpus \\
            --input-dir midis_A/ \\
            --model-dir model/ \\
            --output z_corpus_A.json
    """
    import numpy as np

    model_dir = Path(args.model_dir)
    input_dir = Path(args.input_dir)

    print(f"[style-corpus] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    midi_files = sorted(
        list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi'))
    )
    if not midi_files:
        print(f"[style-corpus] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    print(f"[style-corpus] {len(midi_files)} archivos MIDI encontrados\n")

    z_styles  = []
    z_contexts = []
    tensions  = []
    skipped   = 0

    for midi_path in midi_files:
        try:
            z_mean, z_style = _encode_ref(str(midi_path), model, cfg)
            rolls = _midi_to_rolls(str(midi_path), cfg)
            n_bars = min(r.shape[0] for r in rolls.values())
            tension_vecs = TensionExtractor().extract_bar_vectors(rolls, n_bars)
            tension_mean = tension_vecs.mean(axis=0)

            z_styles.append(z_style)
            z_contexts.append(z_mean)
            tensions.append(tension_mean)
            print(f"  [OK] {midi_path.stem}")
        except Exception as e:
            print(f"  [SKIP] {midi_path.stem} — {e}")
            skipped += 1

    if not z_styles:
        print("[style-corpus] No se pudo encodear ningún MIDI.")
        sys.exit(1)

    # Calcular centroides
    z_style_mean   = np.mean(z_styles,   axis=0)
    z_context_mean = np.mean(z_contexts, axis=0)
    tension_mean   = np.mean(tensions,   axis=0)

    # Calcular desviación estándar (diagnóstico de cohesión del corpus)
    z_style_std = float(np.std(z_styles))

    out_path = args.output or f"z_corpus_{input_dir.stem}.json"
    payload = {
        'source':         str(input_dir),
        'model_dir':      str(model_dir),
        'n_files':        len(z_styles),
        'n_skipped':      skipped,
        'latent_dim':     cfg['latent_dim'],
        'style_dim':      cfg['style_dim'],
        'z_context':      z_context_mean.tolist(),
        'z_style':        z_style_mean.tolist(),
        'z_style_std':    z_style_std,
        'tension_mean':   tension_mean.tolist(),
        'roles_found':    cfg['roles'],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\n[style-corpus] Archivos procesados : {len(z_styles)}")
    print(f"[style-corpus] Archivos omitidos   : {skipped}")
    print(f"[style-corpus] Cohesión del corpus (std z_style): {z_style_std:.4f}")
    if z_style_std > 1.0:
        print("[style-corpus] ⚠  std alto — el corpus puede ser heterogéneo. "
              "Considera usar subcarpetas más homogéneas.")
    print(f"[style-corpus] Centroide guardado en {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: transfer
# ══════════════════════════════════════════════════════════════════════════════

def cmd_transfer(args):
    """
    Transfiere el estilo de un corpus B a una canción del corpus A,
    manteniendo el contenido musical (ritmo, melodía) del input.

    Flujo:
        1. Encodear input → z_style_input
        2. Leer centroides z_A y z_B de los corpus
        3. Calcular desplazamiento: Δz = z_B - z_A
        4. Aplicar:  z_style_nuevo = z_style_input + strength * Δz
           (con --progressive, strength varía 0→1 a lo largo de los compases)
        5. Denoising compás a compás usando z_style_nuevo como condicionamiento

    Uso:
        python diffusion_composer.py transfer \\
            --input cancion.mid \\
            --model-dir model/ \\
            --style-from z_corpus_A.json \\
            --style-to   z_corpus_B.json \\
            --strength 0.8 \\
            --output resultado.mid

        # Interpolación progresiva A→B a lo largo de la canción:
        python diffusion_composer.py transfer ... --progressive
    """
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[transfer] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    palette = _load_palette(args.palette, cfg)

    # ── Cargar centroides de estilo ───────────────────────────────────────
    with open(args.style_from) as f:
        data_A = json.load(f)
    with open(args.style_to) as f:
        data_B = json.load(f)

    z_A = np.array(data_A['z_style'], dtype=np.float32)
    z_B = np.array(data_B['z_style'], dtype=np.float32)
    delta_z = z_B - z_A   # vector de traslación de estilo

    print(f"[transfer] Estilo origen : {args.style_from}  "
          f"(n={data_A.get('n_files', '?')} MIDIs)")
    print(f"[transfer] Estilo destino: {args.style_to}  "
          f"(n={data_B.get('n_files', '?')} MIDIs)")
    print(f"[transfer] |Δz| = {float(np.linalg.norm(delta_z)):.4f}  "
          f"strength={args.strength}  progressive={args.progressive}")

    # ── Encodear el MIDI de input ─────────────────────────────────────────
    print(f"[transfer] Procesando input: {args.input} ...")
    rolls_ref = _midi_to_rolls(args.input, cfg)
    z_mean_input, z_style_input = _encode_ref(args.input, model, cfg)
    ctx_np = _rolls_to_context_tensor(rolls_ref, cfg)

    # Número de compases = los que tiene el input (misma duración)
    n_bars     = min(r.shape[0] for r in rolls_ref.values())
    tension_dim = cfg['tension_dim']
    role_list   = cfg['roles']
    n_roles     = cfg['n_roles']
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']
    ctx_bars    = window_bars - 1

    # Tensión: extraída del propio input (preserva la dinámica original)
    tension_matrix = TensionExtractor().extract_bar_vectors(rolls_ref, n_bars)
    # Rellenar hasta n_bars si hace falta
    if tension_matrix.shape[0] < n_bars:
        pad = np.zeros((n_bars - tension_matrix.shape[0], tension_dim), dtype=np.float32)
        tension_matrix = np.concatenate([tension_matrix, pad], axis=0)

    # ── Generación compás a compás ────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ctx_buffer    = torch.tensor(ctx_np).unsqueeze(0).to(device)
    bars_per_role = {role: [] for role in role_list}
    adaptive_thr  = None

    print(f"[transfer] Generando {n_bars} compases ...")

    for bar_idx in range(n_bars):
        # Tensión de este compás
        v_ten = torch.tensor(tension_matrix[bar_idx]).unsqueeze(0).to(device)

        # ── Calcular z_style para este compás ────────────────────────────
        if args.progressive:
            # strength crece linealmente de 0 → strength a lo largo de la canción
            bar_strength = args.strength * (bar_idx / max(n_bars - 1, 1))
        else:
            bar_strength = args.strength

        z_style_bar = z_style_input + bar_strength * delta_z
        v_sty = torch.tensor(z_style_bar).unsqueeze(0).to(device)

        # ── Denoising desde el compás de referencia ───────────────────────
        # Usamos el compás correspondiente del input como punto de partida,
        # igual que el modo `denoise` de compose, pero con z_style sustituido.
        ref_bar_idx = min(bar_idx, min(r.shape[0] for r in rolls_ref.values()) - 1)
        xr_np = np.zeros((n_roles, resolution, 128), dtype=np.float32)
        for ridx, role in enumerate(role_list):
            if role in rolls_ref:
                xr_np[ridx] = rolls_ref[role][ref_bar_idx]
        x_ref = torch.tensor(xr_np).unsqueeze(0).to(device)

        with torch.no_grad():
            roll_bar = model.denoise_from_ref(
                x_ref, ctx_buffer, v_ten, v_sty,
                strength=args.denoise_strength,
                n_ddim_steps=args.ddim_steps,
                eta=args.eta,
            )

        bar_np = roll_bar[0].cpu().numpy()   # (N_ROLES, res, 128)

        # Diagnóstico en el primer compás
        if bar_idx == 0:
            vmax  = float(bar_np.max())
            vmean = float(bar_np.mean())
            p90   = float(np.percentile(bar_np, 90))
            adaptive_thr = _adaptive_threshold(bar_np, percentile=args.threshold_pct)
            n_active = int((bar_np > adaptive_thr).sum())
            density  = 100 * n_active / bar_np.size
            print(f"\n  [diag] Compás 0 — decoder output:")
            print(f"         mean={vmean:.4f}  p90={p90:.4f}  max={vmax:.4f}")
            print(f"         Umbral adaptativo: {adaptive_thr:.4f}  →  "
                  f"{n_active} píxeles activos ({density:.2f}%)")
            if vmax < 0.05:
                print(f"  [diag] ⚠  Activaciones muy bajas. "
                      f"Prueba con --denoise-strength más alto.")

        for ridx, role in enumerate(role_list):
            bars_per_role[role].append(bar_np[ridx])

        # Actualizar ctx_buffer con el compás generado
        bar_binary = (bar_np > (adaptive_thr or 0.3)).astype(np.float32)
        new_bar    = torch.tensor(bar_binary).unsqueeze(0).unsqueeze(2).to(device)
        if ctx_bars > 1:
            ctx_buffer = torch.cat([ctx_buffer[:, :, 1:, :, :], new_bar], dim=2)
        else:
            ctx_buffer = new_bar

        print(f"\r  Compás {bar_idx + 1}/{n_bars}", end='', flush=True)

    print()

    # Convertir listas → arrays y guardar MIDI
    final_rolls = {
        role: np.stack(bars, axis=0)
        for role, bars in bars_per_role.items()
        if bars
    }

    n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=adaptive_thr)
    print(f"[transfer] MIDI guardado en {args.output}  "
          f"({n_notes} notas, umbral={adaptive_thr:.3f})")
    if n_notes == 0:
        print("[transfer] ⚠  MIDI vacío. Prueba con --threshold-pct 70 "
              "o aumenta --denoise-strength.")


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

    # ── Calcular z_style condicionante ────────────────────────────────────
    if mode in ('sample', 'denoise', 'blend'):
        midi_sources = [args.input] if args.input else (args.inputs or [])
        if not midi_sources:
            print(f"[compose] El modo '{mode}' requiere --input o --inputs")
            sys.exit(1)

        z_styles = []
        contexts = []
        for src in midi_sources:
            zm, zs = _encode_ref(src, model, cfg)
            z_styles.append(zs)
            rolls = _midi_to_rolls(src, cfg)
            ctx   = _rolls_to_context_tensor(rolls, cfg)
            contexts.append(ctx)

        if mode == 'blend' and len(z_styles) > 1:
            weights = args.weights or [1.0 / len(z_styles)] * len(z_styles)
            s = sum(weights)
            weights = [w / s for w in weights]
            z_style_np = sum(w * z for w, z in zip(weights, z_styles))
            ctx_np     = contexts[0]   # usamos contexto del primer MIDI
        else:
            z_style_np = z_styles[0]
            ctx_np     = contexts[0]

    elif mode == 'sweep':
        if not args.inputs or len(args.inputs) < 2:
            print("[compose] sweep requiere al menos 2 --inputs")
            sys.exit(1)
        all_zs  = []
        all_ctx = []
        for src in args.inputs:
            zm, zs = _encode_ref(src, model, cfg)
            all_zs.append(zs)
            rolls = _midi_to_rolls(src, cfg)
            ctx   = _rolls_to_context_tensor(rolls, cfg)
            all_ctx.append(ctx)
        # z_style se interpola por barra
        sweep_styles = np.stack(all_zs, axis=0)   # (n_src, style_dim)
        sweep_ctx    = all_ctx                      # lista de ctx arrays

    else:
        print(f"[compose] Modo desconocido: {mode}")
        sys.exit(1)

    # ── Perfil de tensión ─────────────────────────────────────────────────
    tension_matrix = _tension_profile(args.tension, n_bars, tension_dim)

    # ── Generación barra a barra ──────────────────────────────────────────
    role_list   = cfg['roles']
    n_roles     = cfg['n_roles']
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']
    ctx_bars    = window_bars - 1

    bars_per_role = {role: [] for role in role_list}
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Buffer de contexto: se actualiza progresivamente con las barras generadas
    # Inicializado con el contexto del MIDI de referencia
    if mode != 'sweep':
        ctx_buffer = torch.tensor(ctx_np).unsqueeze(0).to(device)   # (1, N_ROLES, ctx_bars, res, 128)
    else:
        # sweep: iniciar con contexto del primer MIDI
        ctx_buffer = torch.tensor(all_ctx[0]).unsqueeze(0).to(device)

    # ── Pre-cargar rolls de referencia para modo denoise ─────────────────
    rolls_ref = None
    if mode == 'denoise' and args.input:
        rolls_ref = _midi_to_rolls(args.input, cfg)
        print(f"[compose] Referencia cargada: {list(rolls_ref.keys())}")

    print(f"[compose] Generando {n_bars} compases (modo={mode}, ddim_steps={args.ddim_steps}) ...")

    # Umbral adaptativo: se estima tras el primer compás y se reutiliza
    adaptive_thr = None

    for bar_idx in range(n_bars):
        # Tensión de este compás
        v_ten = torch.tensor(tension_matrix[bar_idx]).unsqueeze(0).to(device)

        # z_style: fijo o interpolado (sweep)
        if mode == 'sweep':
            alpha = bar_idx / max(n_bars - 1, 1)
            n_src = len(sweep_styles)
            seg   = alpha * (n_src - 1)
            i0    = min(int(seg), n_src - 2)
            lam   = seg - i0
            zs_np    = (1 - lam) * sweep_styles[i0] + lam * sweep_styles[i0 + 1]
            v_sty    = torch.tensor(zs_np).unsqueeze(0).to(device)
            ctx_np_i = (1 - lam) * all_ctx[i0] + lam * all_ctx[i0 + 1]
            ctx_buffer = torch.tensor(ctx_np_i).unsqueeze(0).to(device)
        else:
            v_sty = torch.tensor(z_style_np).unsqueeze(0).to(device)

        # Generar un compás
        with torch.no_grad():
            if mode == 'denoise' and rolls_ref is not None:
                # En modo denoise usamos el compás correspondiente del MIDI
                # de referencia (o el último si la ref es más corta) como
                # punto de partida del denoising, para TODOS los compases.
                ref_bar_idx = min(bar_idx, min(r.shape[0] for r in rolls_ref.values()) - 1)
                xr_np = np.zeros((n_roles, resolution, 128), dtype=np.float32)
                for ridx, role in enumerate(role_list):
                    if role in rolls_ref:
                        xr_np[ridx] = rolls_ref[role][ref_bar_idx]
                x_ref = torch.tensor(xr_np).unsqueeze(0).to(device)
                roll_bar = model.denoise_from_ref(
                    x_ref, ctx_buffer, v_ten, v_sty,
                    strength=args.strength,
                    n_ddim_steps=args.ddim_steps,
                    eta=args.eta,
                )
            else:
                roll_bar = model.sample(
                    ctx_buffer, v_ten, v_sty,
                    temperature=args.temperature,
                    n_ddim_steps=args.ddim_steps,
                    eta=args.eta,
                )

        # roll_bar: (1, N_ROLES, res, 128) — valores continuos [0,1]
        bar_np = roll_bar[0].cpu().numpy()   # (N_ROLES, res, 128)

        # Diagnóstico en el primer compás
        if bar_idx == 0:
            import numpy as np
            vmin  = float(bar_np.min())
            vmax  = float(bar_np.max())
            vmean = float(bar_np.mean())
            p50   = float(np.percentile(bar_np, 50))
            p90   = float(np.percentile(bar_np, 90))
            p99   = float(np.percentile(bar_np, 99))
            print(f"\n  [diag] Compás 0 — decoder output:")
            print(f"         min={vmin:.4f}  mean={vmean:.4f}  "
                  f"p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}  max={vmax:.4f}")
            # Estimar umbral adaptativo global desde el primer compás
            adaptive_thr = _adaptive_threshold(bar_np, percentile=getattr(args, 'threshold_pct', 85.0))
            n_active = int((bar_np > adaptive_thr).sum())
            density  = 100 * n_active / bar_np.size
            print(f"         Umbral adaptativo: {adaptive_thr:.4f}  →  "
                  f"{n_active} píxeles activos ({density:.2f}%)")
            if vmax < 0.05:
                print(f"  [diag] ⚠  Activaciones muy bajas (max={vmax:.4f}). "
                      f"El modelo necesita más entrenamiento.")
                print(f"         Generando con umbral relativo al máximo — "
                      f"el resultado tendrá estructura pero puede sonar raro.")

        for ridx, role in enumerate(role_list):
            bars_per_role[role].append(bar_np[ridx])   # (res, 128)

        # Actualizar ctx_buffer con el compás generado (binarizado para
        # evitar acumulación de ruido continuo en el contexto)
        bar_binary = (bar_np > (adaptive_thr or 0.3)).astype(np.float32)
        new_bar = torch.tensor(bar_binary).unsqueeze(0).unsqueeze(2).to(device)
        # new_bar: (1, N_ROLES, 1, res, 128)
        if ctx_bars > 1:
            ctx_buffer = torch.cat([ctx_buffer[:, :, 1:, :, :], new_bar], dim=2)
        else:
            ctx_buffer = new_bar

        print(f"\r  Compás {bar_idx + 1}/{n_bars}", end='', flush=True)

    print()

    # Convertir listas → arrays
    final_rolls = {}
    for role in role_list:
        if bars_per_role[role]:
            final_rolls[role] = np.stack(bars_per_role[role], axis=0)  # (n_bars, res, 128)

    # Guardar con umbral adaptativo (ya estimado)
    n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=adaptive_thr)
    print(f"[compose] MIDI guardado en {args.output}  ({n_notes} notas, umbral={adaptive_thr:.3f})")
    if n_notes == 0:
        print("[compose] ⚠  ADVERTENCIA: MIDI vacío. Prueba con --threshold-pct 70 "
              "o revisa que el modelo haya entrenado suficientes épocas.")
        print("          También puedes probar: --eta 0.5  --temperature 0.8  --ddim-steps 100")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    import numpy as np

    what = args.what

    if 'npz' in what:
        data_dir = Path(args.data_dir) if args.data_dir else None
        if data_dir is None:
            print("[inspect] --data-dir requerido para --what npz")
            return
        npz_files = sorted(data_dir.glob('*.npz'))
        if not npz_files:
            print(f"[inspect] No hay .npz en {data_dir}")
            return

        target = args.npz_file
        if target:
            npz_files = [f for f in npz_files if f.stem == target]

        # Con --no-roll mostramos todos los archivos con densidad.
        # Sin --no-roll limitamos a 5 para no inundar el terminal con piano rolls.
        files_to_show = npz_files if args.no_roll else npz_files[:5]

        for f in files_to_show:
            data = dict(np.load(str(f), allow_pickle=True))
            meta = json.loads(str(data['meta_json'][0]))
            print(f"\n  {f.name}")
            print(f"    Compases   : {meta['total_bars']}")
            print(f"    Ventanas   : {meta['n_windows']}")
            print(f"    Roles      : {', '.join(meta['roles'])}")
            if args.no_roll:
                # Calcular densidad media de todos los roles
                densities = []
                for role in meta['roles']:
                    key = f'roll_{role}'
                    if key in data:
                        densities.append(float(data[key].mean()))
                if densities:
                    mean_density = sum(densities) / len(densities)
                    flag = '  ⚠ DENSO' if mean_density > 0.15 else ''
                    print(f"    Densidad   : {mean_density*100:.1f}%{flag}")
            if not args.no_roll:
                widx  = args.window if args.window >= 0 else meta['n_windows'] + args.window
                widx  = max(0, min(widx, meta['n_windows'] - 1))
                for role in (args.roles or meta['roles']):
                    key = f'roll_{role}'
                    if key not in data:
                        continue
                    window = data[key][widx]   # (window_bars, res, 128)
                    bars_show = min(args.bars_show, window.shape[0])
                    print(f"\n    Piano roll  [{role}]  ventana {widx}  "
                          f"({bars_show} compases de {window.shape[0]}):")
                    for b in range(bars_show):
                        bar = window[b]   # (res, 128)
                        active_pitches = np.where(bar.sum(axis=0) > 0)[0]
                        if len(active_pitches) == 0:
                            row = '    ·'
                        else:
                            lo, hi = active_pitches.min(), active_pitches.max()
                            row    = f'    pitches [{lo}–{hi}]  n_active={len(active_pitches)}'
                        print(f"      bar {b}: {row}")

    if 'loss_curve' in what:
        model_dir = Path(args.model_dir) if args.model_dir else None
        if model_dir is None:
            print("[inspect] --model-dir requerido para --what loss_curve")
            return
        hist_path = model_dir / Trainer.HISTORY_NAME
        if not hist_path.exists():
            print(f"[inspect] No se encontró {hist_path}")
            return
        with open(hist_path) as f:
            hist = json.load(f)
        print(f"\n  Curvas de loss ({len(hist['train'])} épocas):")
        for ep, (tr, vl) in enumerate(zip(hist['train'], hist['val'])):
            bar_len = int((1 - min(vl, 1)) * 20)
            bar     = '█' * bar_len + '░' * (20 - bar_len)
            print(f"    ep {ep+1:>4}  train={tr:.4f}  val={vl:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: round-trip
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    """
    MIDI → piano roll → MIDI sin pasar por el modelo.

    Útil para aislar la pérdida de información del parser/renderer
    antes de culpar al encoder o al decoder.

    Si round-trip suena mal → el problema está en el parseo MIDI.
    Si round-trip suena bien pero reconstruct suena mal → problema en el codec.
    Si reconstruct suena bien pero compose suena mal → problema en el denoiser.
    """
    import numpy as np

    # Construir cfg mínimo: desde model_config.json o desde args directos
    if args.model_dir:
        config_path = Path(args.model_dir) / 'model_config.json'
        if not config_path.exists():
            print(f"[round-trip] ERROR: no se encontró {config_path}")
            import sys; sys.exit(1)
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"[round-trip] Config cargada desde {config_path}")
    else:
        cfg = {
            'resolution':   args.resolution,
            'window_bars':  4,
            'roles':        ROLES,
            'n_roles':      len(ROLES),
            'tension_dim':  8,
        }
        print(f"[round-trip] Config manual: resolution={args.resolution}, "
              f"roles={ROLES}")

    palette = {}
    if args.palette:
        palette = _load_palette(args.palette, cfg)

    print(f"[round-trip] Procesando {args.input} ...")
    rolls = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[round-trip] ERROR: no se pudieron extraer rolls del MIDI")
        import sys; sys.exit(1)

    n_bars     = min(r.shape[0] for r in rolls.values())
    resolution = cfg['resolution']
    print(f"[round-trip] Roles: {list(rolls.keys())}  |  "
          f"Compases: {n_bars}  |  Resolución: {resolution}")

    all_vals = np.concatenate([r.flatten() for r in rolls.values()])
    density  = float(all_vals.mean())
    print(f"[round-trip] Densidad media del piano roll: {density*100:.2f}% "
          f"({int(all_vals.sum())} píxeles activos de {len(all_vals)})")

    n_notes = _rolls_to_midi(rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=0.5)
    print(f"[round-trip] MIDI guardado en {args.output}  ({n_notes} notas)")
    if n_notes == 0:
        print("[round-trip] ⚠  MIDI vacío — problema en el parser MIDI")
    else:
        print("[round-trip] ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: reconstruct
# ══════════════════════════════════════════════════════════════════════════════

def cmd_reconstruct(args):
    """
    Encode → decode puro sin ningún paso de difusión.

    Útil para:
      · Diagnosticar si el decoder ha convergido (p90 > 0.05 = bien).
      · Escuchar cómo suena el espacio latente antes de que el denoiser madure.
      · Verificar que la arquitectura funciona antes de usar compose.

    Si output_recon.mid suena musical → el decoder está bien, esperar más épocas.
    Si suena ruidoso → el decoder aún no ha convergido.
    """
    import torch, numpy as np

    model_dir = Path(args.model_dir)
    print(f"[reconstruct] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)
    model.eval()

    palette = _load_palette(args.palette, cfg) if args.palette else {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"[reconstruct] Procesando {args.input} ...")
    rolls = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[reconstruct] ERROR: no se pudieron extraer rolls del MIDI")
        import sys; sys.exit(1)

    role_list  = cfg['roles']
    n_roles    = cfg['n_roles']
    resolution = cfg['resolution']
    n_bars     = min(r.shape[0] for r in rolls.values())
    print(f"[reconstruct] Roles: {list(rolls.keys())}  |  "
          f"Compases: {n_bars}  |  Resolución: {resolution}")

    bars_recon = {role: [] for role in role_list}

    with torch.no_grad():
        for bar_idx in range(n_bars):
            bar = np.zeros((n_roles, resolution, 128), dtype=np.float32)
            for ri, role in enumerate(role_list):
                if role in rolls and bar_idx < rolls[role].shape[0]:
                    bar[ri] = rolls[role][bar_idx]

            x   = torch.tensor(bar).unsqueeze(0).to(device)
            mu, _ = model.encoder(x)
            recon = model.decoder.decode_probs(mu)   # sin ruido, sin difusión
            recon_np = recon[0].cpu().numpy()         # (N_ROLES, res, 128)

            if bar_idx == 0:
                flat = recon_np.flatten()
                nz   = flat[flat > 0.001]
                p90  = float(np.percentile(nz, 90)) if len(nz) > 0 else 0.0
                p99  = float(np.percentile(nz, 99)) if len(nz) > 0 else 0.0
                thr_diag = args.threshold if args.threshold else p99 * 0.5
                n_active = int((flat > thr_diag).sum())
                print(f"\n[diag] Compás 0 — reconstrucción encoder/decoder:")
                print(f"       min={flat.min():.4f}  mean={flat.mean():.4f}  "
                      f"p50={float(np.percentile(flat,50)):.4f}  "
                      f"p90={p90:.4f}  p99={p99:.4f}  max={float(flat.max()):.4f}")
                print(f"       Umbral: {thr_diag:.4f}  →  {n_active} píxeles activos "
                      f"({100*n_active/len(flat):.2f}%)")
                if p90 < 0.01:
                    print(f"       ⚠  p90={p90:.4f} — decoder aún ruidoso, necesita más épocas")
                elif p90 < 0.05:
                    print(f"       ~  p90={p90:.4f} — decoder en progreso")
                else:
                    print(f"       ✓  p90={p90:.4f} — separación ruido/notas aceptable")

            for ri, role in enumerate(role_list):
                bars_recon[role].append(recon_np[ri])

    final_rolls = {
        role: np.stack(bars, axis=0)
        for role, bars in bars_recon.items()
    }

    # Umbral global
    all_vals = np.concatenate([r.flatten() for r in final_rolls.values()])
    nz_all   = all_vals[all_vals > 0.001]
    if args.threshold:
        thr = args.threshold
        method = 'fijo'
    elif len(nz_all) > 0:
        thr    = float(np.percentile(nz_all, 99)) * 0.5
        method = 'p99×0.5'
    else:
        thr    = 0.5
        method = 'default'

    n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=thr)
    print(f"\n[reconstruct] MIDI guardado en {args.output}  "
          f"({n_notes} notas, umbral={thr:.4f} [{method}])")
    if n_notes == 0:
        print("[reconstruct] ⚠  MIDI vacío — prueba --threshold 0.1")
    elif n_notes > 10000:
        print(f"[reconstruct] ⚠  Muchas notas ({n_notes}) — prueba --threshold 0.4")
    else:
        print("[reconstruct] ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='diffusion_composer',
        description='Composición musical por difusión latente multi-rol',
    )
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p_prep = sub.add_parser('prepare', help='MIDI corpus → .npz')
    p_prep.add_argument('--input-dir',   required=True, metavar='DIR')
    p_prep.add_argument('--output-dir',  required=True, metavar='DIR')
    p_prep.add_argument('--resolution',  type=int, default=TICKS_PER_BAR_DEFAULT, metavar='INT',
        help=f'Ticks por compás (default: {TICKS_PER_BAR_DEFAULT})')
    p_prep.add_argument('--window-bars', type=int, default=WINDOW_BARS_DEFAULT, metavar='INT',
        help=f'Compases por ventana (default: {WINDOW_BARS_DEFAULT})')
    p_prep.add_argument('--report',      action='store_true')
    p_prep.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Entrena el modelo de difusión latente')
    p_train.add_argument('--data-dir',         required=True, metavar='DIR')
    p_train.add_argument('--model-dir',        required=True, metavar='DIR')
    p_train.add_argument('--epochs',           type=int,   default=300)
    p_train.add_argument('--batch-size',       type=int,   default=16)
    p_train.add_argument('--lr',               type=float, default=1e-4)
    p_train.add_argument('--latent-dim',       type=int,   default=128,
        help='Dimensión del espacio latente (default: 128; era 64 en v1)')
    p_train.add_argument('--style-dim',        type=int,   default=16)
    p_train.add_argument('--diffusion-steps',  type=int,   default=1000,
        help='Pasos T del proceso de difusión (default: 1000)')
    p_train.add_argument('--pos-weight',       type=float, default=5.0,
        help='Peso de notas activas en BCE (default: 5.0).')
    p_train.add_argument('--decoder-dropout',  type=float, default=0.1,
        dest='decoder_dropout',
        help='Dropout entre bloques del decoder (default: 0.1). '
             'Aumenta a 0.2–0.3 si aparece sobreajuste (val_recon sube mientras train_recon baja).')
    p_train.add_argument('--patience',         type=int,   default=50)
    p_train.add_argument('--resume',           action='store_true',
        help='Reanudar entrenamiento desde el último checkpoint')
    p_train.set_defaults(func=cmd_train)

    # ── encode ────────────────────────────────────────────────────────────────
    p_enc = sub.add_parser('encode', help='MIDI referencia → z_style.json')
    p_enc.add_argument('--input',     required=True, metavar='FILE')
    p_enc.add_argument('--model-dir', required=True, metavar='DIR')
    p_enc.add_argument('--output',    metavar='FILE')
    p_enc.set_defaults(func=cmd_encode)

    # ── compose ───────────────────────────────────────────────────────────────
    p_comp = sub.add_parser('compose',
        help='Genera una obra nueva mediante difusión',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Modos de composición (--mode):

              sample    Generación libre desde ruido gaussiano condicionada en ref.
              denoise   Parte del primer compás del ref. y varía progresivamente.
              blend     Interpolación ponderada entre varios MIDIs de referencia.
              sweep     Barrido suave entre varios MIDIs a lo largo de la obra.

            Parámetros de difusión:
              --ddim-steps INT    Pasos DDIM de inferencia (default: 50).
                                  Menos pasos = más rápido pero menos calidad.
              --eta FLOAT         Ruido en DDIM: 0.0=determinista, 1.0=DDPM.
              --temperature FLOAT Escala del ruido inicial (default: 1.0).
              --strength FLOAT    [denoise] Cuánto alejarse de la ref (default: 0.7).

            Perfiles de tensión (--tension):
              flat | arch | rise | fall | archivo.json

            Ejemplos:
              compose --model-dir model/ --palette p.json --mode sample --input ref.mid
              compose --model-dir model/ --palette p.json --mode denoise --input ref.mid --strength 0.5
              compose --model-dir model/ --palette p.json --mode sweep --inputs a.mid b.mid --bars 64
              compose --model-dir model/ --palette p.json --mode blend --inputs a.mid b.mid --weights 0.6 0.4
        """))
    p_comp.add_argument('--model-dir',   required=True, metavar='DIR')
    p_comp.add_argument('--palette',     required=True, metavar='FILE')
    p_comp.add_argument('--mode',
        choices=['sample', 'denoise', 'blend', 'sweep'],
        default='sample')
    p_comp.add_argument('--bars',        type=int,   default=32)
    p_comp.add_argument('--tension',     default='arch')
    p_comp.add_argument('--output',      default='output.mid')
    p_comp.add_argument('--bpm',         type=float, default=120.0)
    p_comp.add_argument('--temperature', type=float, default=1.0)
    p_comp.add_argument('--ddim-steps',  type=int,   default=50,
        help='Pasos de inferencia DDIM (default: 50; más = mejor calidad)')
    p_comp.add_argument('--eta',         type=float, default=0.0,
        help='Estocasticidad DDIM: 0=determinista, 1=DDPM completo (default: 0.0)')
    p_comp.add_argument('--strength',    type=float, default=0.7,
        help='[denoise] Fuerza de variación respecto a ref: 0=copia, 1=libre (default: 0.7)')
    p_comp.add_argument('--input',         metavar='FILE', default=None)
    p_comp.add_argument('--inputs',        nargs='+', metavar='FILE', default=None)
    p_comp.add_argument('--weights',       nargs='+', type=float, default=None)
    p_comp.add_argument('--threshold-pct', type=float, default=85.0, metavar='FLOAT',
        dest='threshold_pct',
        help='Percentil para umbral adaptativo de binarización (default: 85). '
             'Bájalo (ej. 70) si el MIDI sale vacío; súbelo (ej. 95) si hay demasiado ruido.')
    p_comp.set_defaults(func=cmd_compose)

    # ── style-corpus ──────────────────────────────────────────────────────────
    p_sc = sub.add_parser('style-corpus',
        help='Encoda una carpeta de MIDIs y calcula el centroide de estilo')
    p_sc.add_argument('--input-dir',  required=True, metavar='DIR',
        help='Carpeta con los MIDIs del corpus')
    p_sc.add_argument('--model-dir',  required=True, metavar='DIR')
    p_sc.add_argument('--output',     metavar='FILE',
        help='Ruta del .json de salida (default: z_corpus_<carpeta>.json)')
    p_sc.set_defaults(func=cmd_style_corpus)

    # ── transfer ──────────────────────────────────────────────────────────────
    p_tr = sub.add_parser('transfer',
        help='Transfiere el estilo de un corpus a una canción',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Transfiere el estilo del corpus B a una canción del corpus A.

            Flujo:
              1. Encodear input → z_style_input
              2. Calcular Δz = z_B_mean - z_A_mean
              3. z_style_nuevo = z_style_input + strength * Δz
              4. Denoising compás a compás (misma duración que el input)

            Modos:
              Uniforme   : mismo desplazamiento en todos los compases
              Progresivo : strength varía 0→strength a lo largo de la canción (--progressive)

            Ejemplos:
              transfer --input cancion.mid --model-dir model/ --palette p.json \\
                       --style-from z_A.json --style-to z_B.json --strength 0.8

              transfer --input cancion.mid --model-dir model/ --palette p.json \\
                       --style-from z_A.json --style-to z_B.json --strength 1.0 --progressive
        """))
    p_tr.add_argument('--input',            required=True,  metavar='FILE',
        help='MIDI de entrada (estilo A)')
    p_tr.add_argument('--model-dir',        required=True,  metavar='DIR')
    p_tr.add_argument('--palette',          required=True,  metavar='FILE')
    p_tr.add_argument('--style-from',       required=True,  metavar='FILE',
        help='Centroide del corpus origen A (.json de style-corpus o encode)')
    p_tr.add_argument('--style-to',         required=True,  metavar='FILE',
        help='Centroide del corpus destino B (.json de style-corpus o encode)')
    p_tr.add_argument('--strength',         type=float, default=0.8, metavar='FLOAT',
        help='Intensidad del desplazamiento de estilo: 0=sin cambio, 1=traslación completa (default: 0.8)')
    p_tr.add_argument('--progressive',      action='store_true',
        help='Interpolar progresivamente A→B a lo largo de los compases')
    p_tr.add_argument('--denoise-strength', type=float, default=0.6, metavar='FLOAT',
        dest='denoise_strength',
        help='Cuánto alejarse del MIDI de referencia en el denoising: '
             '0=copia exacta, 1=libre (default: 0.6)')
    p_tr.add_argument('--ddim-steps',       type=int,   default=50)
    p_tr.add_argument('--eta',              type=float, default=0.0)
    p_tr.add_argument('--bpm',              type=float, default=120.0)
    p_tr.add_argument('--threshold-pct',    type=float, default=85.0, metavar='FLOAT',
        dest='threshold_pct')
    p_tr.add_argument('--output',           default='transfer_output.mid', metavar='FILE')
    p_tr.set_defaults(func=cmd_transfer)

    # ── round-trip ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser('round-trip',
        help='MIDI → piano roll → MIDI sin modelo (diagnóstico del parser)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convierte un MIDI a piano roll y de vuelta a MIDI sin usar el modelo.
            Útil para aislar problemas en el pipeline de parseo/renderizado.

              round-trip OK, reconstruct mal  →  problema en el codec
              round-trip mal                  →  problema en el parser MIDI

            Ejemplos:
              round-trip --input ref.mid
              round-trip --input ref.mid --model-dir model_diff_v2/
        """))
    p_rt.add_argument('--input',      required=True, metavar='FILE')
    p_rt.add_argument('--model-dir',  default=None,  metavar='DIR',
        help='Si se indica, lee resolución y roles desde model_config.json')
    p_rt.add_argument('--resolution', type=int, default=TICKS_PER_BAR_DEFAULT,
        metavar='INT',
        help=f'Ticks por compás (solo si no se usa --model-dir; '
             f'default: {TICKS_PER_BAR_DEFAULT})')
    p_rt.add_argument('--palette',    default=None,  metavar='FILE',
        help='Paleta de instrumentos (opcional)')
    p_rt.add_argument('--output',     default='output_roundtrip.mid', metavar='FILE')
    p_rt.add_argument('--bpm',        type=float, default=120.0)
    p_rt.set_defaults(func=cmd_round_trip)

    # ── reconstruct ───────────────────────────────────────────────────────────
    p_rec = sub.add_parser('reconstruct',
        help='Encode→decode sin difusión (diagnóstico del decoder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Reconstruye un MIDI mediante encode→decode puro, sin ningún paso
            de difusión. Útil para diagnosticar si el decoder ha convergido.

              p90 > 0.05  →  decoder OK, el problema está en el denoiser
              p90 < 0.01  →  decoder aún ruidoso, esperar más épocas

            Ejemplo:
              reconstruct --model-dir model_diff_v2/ --input ref.mid
        """))
    p_rec.add_argument('--model-dir', required=True, metavar='DIR')
    p_rec.add_argument('--input',     required=True, metavar='FILE')
    p_rec.add_argument('--palette',   default=None,  metavar='FILE',
        help='Paleta de instrumentos (opcional)')
    p_rec.add_argument('--output',    default='output_recon.mid', metavar='FILE')
    p_rec.add_argument('--bpm',       type=float, default=120.0)
    p_rec.add_argument('--threshold', type=float, default=None, metavar='FLOAT',
        help='Umbral fijo de binarización. Sin valor: automático p99×0.5')
    p_rec.set_defaults(func=cmd_reconstruct)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect', help='Diagnóstico del modelo y los datos')
    p_ins.add_argument('--what', nargs='+',
        choices=['npz', 'loss_curve'],
        default=['npz'])
    p_ins.add_argument('--data-dir',   metavar='DIR', default=None)
    p_ins.add_argument('--model-dir',  metavar='DIR', default=None)
    p_ins.add_argument('--file',       dest='npz_file', metavar='NAME', default=None)
    p_ins.add_argument('--window',     type=int, default=0)
    p_ins.add_argument('--bars-show',  type=int, default=4, dest='bars_show')
    p_ins.add_argument('--roles',      nargs='+', metavar='ROL', choices=ROLES, default=None)
    p_ins.add_argument('--no-roll',    action='store_true', dest='no_roll', default=False)
    p_ins.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
