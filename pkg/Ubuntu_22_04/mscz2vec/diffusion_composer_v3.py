#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DIFFUSION COMPOSER  v3                                   ║
║         Composición end-to-end mediante Difusión Directa multi-rol           ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    U-Net 2D → difusión directa sobre el piano roll (sin VAE)                 ║
║    DDPM (x0-parametrization) condicionado en tensión + estilo                ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare      — MIDI corpus → piano rolls segmentados por rol (.npz)       ║
║    train        — Entrena el modelo de difusión                              ║
║    encode       — MIDI referencia → z_style (.json)                          ║
║    compose      — Genera obra nueva (modos: sample/denoise/blend/sweep)      ║
║    transfer     — Transfiere el estilo de un corpus a una canción            ║
║    style-corpus — Calcula el centroide de estilo de una carpeta de MIDIs     ║
║    reconstruct  — Diagnóstico: denoising leve sobre el input                 ║
║    round-trip   — Diagnóstico: MIDI → piano roll → MIDI sin modelo           ║
║    inspect      — Diagnóstico del modelo y los datos                         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ── Preparar datos ────────────────────────────────────────────────────────────
python diffusion_composer.py prepare --input-dir midis/ --output-dir data/ --report

# ── Entrenar ──────────────────────────────────────────────────────────────────
python diffusion_composer_v3.py train \
    --data-dir data/ \
    --model-dir model_diff_v3/ \
    --epochs 300 \
    --batch-size 8 \
    --lr 1e-4 \
    --style-dim 16 \
    --diffusion-steps 1000 \
    --patience 50

# ── Generar (denoising desde referencia) ─────────────────────────────────────
python diffusion_composer_v3.py compose \                         
    --model-dir model_diff_v3/ --palette palette.json \
    --mode denoise --input midis/005505b_.mid \
    --strength 0.1 --ddim-steps 50 --eta 0.0 \
    --temperature 1.0 --bars 16 \
    --threshold 0.3

python diffusion_composer_v3.py compose \
    --model-dir model_diff_v3/ --palette palette.json \
    --mode denoise --input midis/005505b_.mid \
    --strength 0.3 --ddim-steps 50 --eta 0.0 \
    --temperature 1.0 --bars 16 \
    --threshold-pct 99.0

# ── Morphing gradual entre dos canciones (sweep) ─────────────────────────────
python diffusion_composer_v3.py compose \
    --model-dir model_diff_v3/ --palette palette.json \
    --mode sweep \
    --inputs midis/005505b_.mid midis/008906b_.mid \
    --bars 32 \
    --ddim-steps 50 --eta 0.0 \
    --temperature 1.0 \
    --threshold-pct 99.0

# ── Mezcla estática entre dos estilos (blend) ────────────────────────────────
python diffusion_composer_v3.py compose \
    --model-dir model_diff_v3/ --palette palette.json \
    --mode blend \
    --inputs midis/005505b_.mid midis/008906b_.mid \
    --weights 0.5 0.5 \
    --bars 16 \
    --ddim-steps 50 --eta 0.0 \
    --threshold-pct 99.0

# ── Style transfer ────────────────────────────────────────────────────────────
# Estilo A: carpeta con MIDIs del estilo origen
python diffusion_composer_v3.py style-corpus \
    --input-dir midis_A/ \
    --model-dir model_diff_v3/ \
    --output z_estilo_A.json

# Estilo B: carpeta con MIDIs del estilo destino
python diffusion_composer_v3.py style-corpus \
    --input-dir midis_B/ \
    --model-dir model_diff_v3/ \
    --output z_estilo_B.json

python diffusion_composer_v3.py encode \
    --input midis/005505b_.mid \
    --model-dir model_diff_v3/ \
    --output z_estilo_A.json

python diffusion_composer_v3.py transfer \
    --input midis/005505b_.mid \
    --model-dir model_diff_v3/ \
    --palette palette.json \
    --style-from z_estilo_A.json \
    --style-to z_estilo_B.json \
    --strength 0.8 \
    --output resultado.mid \
    --threshold-pct 99.0

# ── Diagnóstico ───────────────────────────────────────────────────────────────
python diffusion_composer_v3.py round-trip --input midis/005505b_.mid
python diffusion_composer_v3.py reconstruct --model-dir model_diff_v3/ --input midis/005505b_.mid

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
#  ENRIQUECIMIENTO CON MSCZ2VEC (opcional, requiere music21)
# ══════════════════════════════════════════════════════════════════════════════

# Dimensión del vector de tensión enriquecido:
#   8  (TensionExtractor básico)
# + 1  (tensión armónica por compás, de harmonic_tension_profile)
# + 1  (estabilidad tonal por compás, de tonal_stability_profile)
# + 12 (tonalidad: one-hot de las 12 clases de pitch)
# + 1  (modo: 1.0=mayor, 0.0=menor)
# = 23 dimensiones
ENRICHED_TENSION_DIM = 23


def _enrich_with_mscz2vec(midi_path: str, total_bars: int,
                           mscz2vec_path: str = None) -> 'np.ndarray | None':
    """
    Extrae vectores de condicionamiento enriquecidos usando mscz2vec.

    Devuelve array (total_bars, 15) con:
        [0]    tensión armónica por compás (harmonic_tension_profile)
        [1]    estabilidad tonal por compás (tonal_stability_profile)
        [2:14] tonalidad: one-hot 12 clases de pitch (Do=0 ... Si=11)
        [14]   modo: 1.0=mayor, 0.0=menor

    Si music21 o mscz2vec no están disponibles, devuelve None silenciosamente.
    """
    import numpy as np

    ENRICH_DIM = 15  # solo la parte añadida (sin los 8 básicos)

    try:
        # Intentar importar music21
        import music21
        from music21 import converter as m21_converter
    except ImportError:
        return None

    # Importar mscz2vec
    try:
        if mscz2vec_path:
            import importlib.util
            spec = importlib.util.spec_from_file_location('mscz2vec', mscz2vec_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            import mscz2vec as mod
    except (ImportError, Exception):
        return None

    try:
        # Suprimir prints de debug de mscz2vec
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            score = m21_converter.parse(midi_path)

            # 1. Tensión armónica por compás
            tension_data = mod.harmonic_tension_profile(score)
            harm_tension = tension_data.get('tension_values', []) if tension_data else []

            # 2. Estabilidad tonal por compás
            stability_data = mod.tonal_stability_profile(score)
            tonal_stab = (stability_data.get('stability_smoothed', [])
                         if stability_data else [])

            # 3. Tonalidad global → one-hot + modo
            try:
                key = score.analyze('key')
                tonic_pc  = key.tonic.pitchClass        # 0–11
                is_major  = 1.0 if key.mode == 'major' else 0.0
            except Exception:
                tonic_pc  = 0
                is_major  = 0.5

        # Construir array (total_bars, ENRICH_DIM)
        result = np.zeros((total_bars, ENRICH_DIM), dtype=np.float32)

        for bar in range(total_bars):
            # Tensión armónica (interpolar si hay menos compases que total_bars)
            if harm_tension:
                idx = min(bar, len(harm_tension) - 1)
                result[bar, 0] = float(harm_tension[idx])

            # Estabilidad tonal
            if tonal_stab:
                idx = min(bar, len(tonal_stab) - 1)
                result[bar, 1] = float(tonal_stab[idx])

            # Tonalidad: one-hot
            result[bar, 2 + tonic_pc] = 1.0

            # Modo
            result[bar, 14] = is_major

        return result

    except Exception:
        return None




def _load_external_vec(midi_path: 'Path') -> 'np.ndarray | None':
    """
    Busca un archivo .vec junto al MIDI (mismo nombre, extensión .vec)
    y lo carga como array numpy 1D.

    El vector se replica para todos los compases durante el preprocesado,
    añadiéndose al vector de condicionamiento como embedding global del MIDI.

    Formato esperado: array numpy guardado con np.save() → archivo .npy,
    o bien archivo de texto con valores separados por espacios/líneas.
    """
    import numpy as np

    vec_path = midi_path.with_suffix('.vec')
    if not vec_path.exists():
        return None
    try:
        # Intentar cargar como numpy binario primero
        try:
            vec = np.load(str(vec_path))
        except Exception:
            # Fallback: texto plano (valores separados por espacios o líneas)
            vec = np.loadtxt(str(vec_path), dtype=np.float32)
        vec = np.array(vec, dtype=np.float32).flatten()
        return vec
    except Exception:
        return None


def _prepare_one_midi(args_tuple):
    midi_path, output_dir, resolution, window_bars, enrich, mscz2vec_path, enrich_external = args_tuple
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

    # Enriquecimiento opcional con mscz2vec
    enrich_ok = False
    if enrich:
        enrich_bars = _enrich_with_mscz2vec(str(midi_path), total_bars, mscz2vec_path)
        if enrich_bars is not None:
            # Combinar: [tensión básica (8D) | enriquecida (15D)] = 23D por compás
            tension_full = np.concatenate([tension_bars, enrich_bars], axis=1)
            tension_full_windows = tension_full[mid_offset: mid_offset + min_windows]
            if len(tension_full_windows) < min_windows:
                pad = np.zeros((min_windows - len(tension_full_windows),
                                ENRICHED_TENSION_DIM), dtype=np.float32)
                tension_full_windows = np.concatenate([tension_full_windows, pad], axis=0)
            save_dict['tension_enriched'] = tension_full_windows
            enrich_ok = True
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
    enrich_str = ' +enrich' if enrich_ok else (' enrich:FAIL' if enrich else '')

    # Enriquecimiento externo: cargar .vec junto al MIDI
    ext_vec_ok = False
    if enrich_external:
        ext_vec = _load_external_vec(midi_path)
        if ext_vec is not None:
            ext_dim = len(ext_vec)
            # Replicar el vector para todos los compases: (total_bars, ext_dim)
            ext_bars = np.tile(ext_vec, (total_bars, 1))
            # Ventanas del vector externo (mismo offset que la tensión)
            ext_windows = ext_bars[mid_offset: mid_offset + min_windows]
            if len(ext_windows) < min_windows:
                pad = np.zeros((min_windows - len(ext_windows), ext_dim), dtype=np.float32)
                ext_windows = np.concatenate([ext_windows, pad], axis=0)

            # Concatenar al vector de tensión ya disponible
            base_key = 'tension_enriched' if 'tension_enriched' in save_dict else 'tension'
            save_dict['tension_enriched'] = np.concatenate(
                [save_dict[base_key], ext_windows], axis=1)
            # Guardar dimensión del vec externo en meta para poder reconstruirla
            save_dict['ext_vec_dim'] = np.array([ext_dim], dtype=np.int32)
            ext_vec_ok = True
            enrich_str += f' +vec({ext_dim}D)'
        else:
            if enrich_external:
                enrich_str += ' vec:MISSING'
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, roles: {', '.join(roles_found)}){enrich_str}",
            True, stats_partial)


def cmd_prepare(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution  = args.resolution
    window_bars = args.window_bars
    enrich      = getattr(args, 'enrich', False)
    mscz2vec_path = getattr(args, 'mscz2vec_path', None)
    enrich_external = getattr(args, 'enrich_external', False)

    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    if enrich:
        # Verificar disponibilidad de music21 y mscz2vec antes de empezar
        try:
            import music21
            enrich_available = True
        except ImportError:
            print("[prepare] ⚠  --enrich ignorado: music21 no está instalado.")
            print("           Instala con: pip install music21")
            enrich_available = False
            enrich = False

        if enrich_available and mscz2vec_path:
            if not Path(mscz2vec_path).exists():
                print(f"[prepare] ⚠  --mscz2vec-path '{mscz2vec_path}' no encontrado. "
                      f"Enriquecimiento desactivado.")
                enrich = False

        if enrich:
            print(f"[prepare] ✓  Enriquecimiento mscz2vec ACTIVO "
                  f"(tensión armónica + estabilidad tonal + tonalidad → {ENRICHED_TENSION_DIM}D)")
            print(f"[prepare]    Nota: el preprocesado será más lento (~5-30s por archivo)")

    if enrich_external:
        print(f"[prepare] ✓  Enriquecimiento externo ACTIVO "
              f"(buscando .vec junto a cada MIDI)")

    n_workers = min(multiprocessing.cpu_count(), len(midi_files))
    # Con enriquecimiento, limitar workers para no saturar music21
    if enrich:
        n_workers = min(n_workers, 4)

    print(f"[prepare] {len(midi_files)} archivos MIDI encontrados")
    print(f"[prepare] Resolución: {resolution} ticks/compás  |  Ventana: {window_bars} compases")
    print(f"[prepare] Paralelizando con {n_workers} procesos\n")

    stats = {r: 0 for r in ROLES}
    stats['files_ok'] = stats['files_skipped'] = stats['total_windows'] = 0

    task_args = [
        (midi_path, str(output_dir), resolution, window_bars, enrich, mscz2vec_path, enrich_external)
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
        # Usar tensión enriquecida si está disponible, si no la básica
        if 'tension_enriched' in data:
            tension = torch.tensor(data['tension_enriched'][widx])
        else:
            tension = torch.tensor(data['tension'][widx])       # (TENSION_DIM,)
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
#  ARQUITECTURA v3: DIFUSIÓN DIRECTA EN EL PIANO ROLL (sin VAE)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Cambio respecto a v1/v2:
#    v1/v2: piano_roll → Encoder → z (128d) → DDPM en z → Decoder → piano_roll
#    v3:    piano_roll (N_ROLES, res, 128) → U-Net DDPM → piano_roll
#
#  El denoiser es una U-Net 2D que opera sobre el piano roll completo.
#  El proceso inverso converge directamente a notas sin espacio latente.
#
#  Condicionamiento: context (compases anteriores) + tensión + estilo
#  inyectados via AdaGN (Adaptive Group Normalization) en cada bloque.
# ══════════════════════════════════════════════════════════════════════════════


def _build_unet_diffusion(n_roles: int, resolution: int, style_dim: int,
                          tension_dim: int, n_steps: int = 1000,
                          base_ch: int = 64):
    """
    Construye la U-Net 2D para difusión directa en el piano roll.

    Entrada : x_t  (B, N_ROLES, resolution, 128)  — piano roll ruidoso
    Salida  : ε̂   (B, N_ROLES, resolution, 128)  — ruido predicho

    Condicionamiento:
        cond = f(context_avg, tension, style) → vector (COND_DIM,)
        Inyectado en cada bloque via AdaGN: scale/shift aprendidos.

    Arquitectura U-Net:
        Encoder: [res×128] → [res/2×64] → [res/4×32]  (stride 2)
        Bottleneck: [res/4×32]
        Decoder: [res/4×32] → [res/2×64] → [res×128]  (upsample + skip)
    """
    import torch
    import torch.nn as nn
    import math

    PITCH    = 128
    COND_DIM = 256
    T_DIM    = 128

    # Canales en cada nivel de la U-Net
    ch = [base_ch, base_ch * 2, base_ch * 4]   # [64, 128, 256]

    # ── Time embedding sinusoidal ──────────────────────────────────────────
    class _TimeEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(T_DIM, T_DIM * 2), nn.SiLU(),
                nn.Linear(T_DIM * 2, T_DIM),
            )

        def forward(self, t):
            half  = T_DIM // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / half)
            args  = t[:, None].float() * freqs[None]
            emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return self.proj(emb)   # (B, T_DIM)

    # ── Context encoder ────────────────────────────────────────────────────
    class _ContextEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((4, 8))
            self.fc   = nn.LazyLinear(COND_DIM)

        def forward(self, ctx, tension, style):
            # ctx: (B, N_ROLES, ctx_bars, res, 128)
            B, R, C, H, W = ctx.shape
            pooled = self.pool(ctx.reshape(B * R * C, 1, H, W)).reshape(B, -1)
            return self.fc(torch.cat([pooled, tension, style], dim=-1))

    # ── AdaGN block: Conv2D + Adaptive Group Norm condicionado ────────────
    class _AdaGNBlock(nn.Module):
        """ResBlock con AdaGN: el condicionamiento escala y desplaza la norma."""
        def __init__(self, in_ch, out_ch, cond_dim, t_dim, dropout=0.1):
            super().__init__()
            self.norm1  = nn.GroupNorm(min(8, in_ch), in_ch)
            self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.drop   = nn.Dropout2d(dropout)
            self.act    = nn.SiLU()
            # Proyecciones AdaGN: scale + shift desde cond y t
            self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
            self.t_proj    = nn.Linear(t_dim,    out_ch * 2)
            # Skip connection si in_ch != out_ch
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x, t_emb, cond):
            h = self.act(self.norm1(x))
            h = self.conv1(h)
            # AdaGN: escala y desplazamiento aprendidos desde cond y t
            st_c = self.cond_proj(cond)[:, :, None, None]   # (B, out*2, 1, 1)
            st_t = self.t_proj(t_emb)[:, :, None, None]
            st   = st_c + st_t
            scale, shift = st.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
            h = self.act(h)
            h = self.drop(self.conv2(h))
            return h + self.skip(x)

    # ── U-Net completa ─────────────────────────────────────────────────────
    class _UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.t_emb   = _TimeEmb()
            self.ctx_enc = _ContextEnc()

            # Stem: proyecta N_ROLES canales a ch[0]
            self.stem = nn.Conv2d(n_roles, ch[0], 3, padding=1)

            # Encoder
            self.enc0 = _AdaGNBlock(ch[0], ch[0], COND_DIM, T_DIM)
            self.down0 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)  # /2

            self.enc1 = _AdaGNBlock(ch[0], ch[1], COND_DIM, T_DIM)
            self.down1 = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)  # /4

            # Bottleneck
            self.mid0 = _AdaGNBlock(ch[1], ch[2], COND_DIM, T_DIM)
            self.mid1 = _AdaGNBlock(ch[2], ch[2], COND_DIM, T_DIM)

            # Decoder (con skip connections del encoder)
            self.up1   = nn.ConvTranspose2d(ch[2], ch[1], 4, stride=2, padding=1)
            self.dec1  = _AdaGNBlock(ch[1] * 2, ch[1], COND_DIM, T_DIM)  # *2 por skip

            self.up0   = nn.ConvTranspose2d(ch[1], ch[0], 4, stride=2, padding=1)
            self.dec0  = _AdaGNBlock(ch[0] * 2, ch[0], COND_DIM, T_DIM)  # *2 por skip

            # Head: proyecta ch[0] → N_ROLES (predicción de ruido)
            self.head = nn.Sequential(
                nn.GroupNorm(min(8, ch[0]), ch[0]),
                nn.SiLU(),
                nn.Conv2d(ch[0], n_roles, 3, padding=1),
            )

        def forward(self, x_t, t, context, tension, z_style):
            """
            x_t     : (B, N_ROLES, res, 128)
            t       : (B,) enteros
            context : (B, N_ROLES, ctx_bars, res, 128)
            tension : (B, tension_dim)
            z_style : (B, style_dim)
            → ε̂    : (B, N_ROLES, res, 128)
            """
            cond  = self.ctx_enc(context, tension, z_style)  # (B, COND_DIM)
            t_emb = self.t_emb(t)                             # (B, T_DIM)

            # Stem
            h = self.stem(x_t)                    # (B, ch0, res, 128)

            # Encoder
            h0 = self.enc0(h,  t_emb, cond)       # (B, ch0, res, 128)
            h  = self.down0(h0)                    # (B, ch0, res/2, 64)
            h1 = self.enc1(h,  t_emb, cond)       # (B, ch1, res/2, 64)
            h  = self.down1(h1)                    # (B, ch1, res/4, 32)

            # Bottleneck
            h  = self.mid0(h,  t_emb, cond)
            h  = self.mid1(h,  t_emb, cond)       # (B, ch2, res/4, 32)

            # Decoder con skips
            h  = self.up1(h)                       # (B, ch1, res/2, 64)
            h  = self.dec1(torch.cat([h, h1], 1), t_emb, cond)

            h  = self.up0(h)                       # (B, ch0, res, 128)
            h  = self.dec0(torch.cat([h, h0], 1), t_emb, cond)

            return self.head(h)                    # (B, N_ROLES, res, 128)

    # ── Schedule coseno ────────────────────────────────────────────────────
    class _CosineSchedule:
        def __init__(self, n_steps=1000, s=0.008):
            import torch, math
            self.n_steps = n_steps
            steps  = torch.arange(n_steps + 1, dtype=torch.float64)
            alphas = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas  = (1.0 - alphas[1:] / alphas[:-1]).clamp(0, 0.999).float()
            acp    = torch.cumprod(1.0 - betas, dim=0)
            self.register = {
                'betas':               betas,
                'alphas_cumprod':      acp,
                'sqrt_alphas_cumprod': acp.sqrt(),
                'sqrt_one_minus_acp':  (1.0 - acp).sqrt(),
            }

        def to(self, device):
            self.register = {k: v.to(device) for k, v in self.register.items()}
            return self

        def q_sample(self, x0, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x0)
            sa = self.register['sqrt_alphas_cumprod'][t]
            sb = self.register['sqrt_one_minus_acp'][t]
            # Broadcast para tensores 4D (B, N_ROLES, res, 128)
            while sa.dim() < x0.dim():
                sa = sa.unsqueeze(-1)
                sb = sb.unsqueeze(-1)
            return sa * x0 + sb * noise, noise

        def ddim_sample(self, net, x_t, t_scalar, t_prev,
                        context, tension, z_style, eta=0.0):
            import torch
            B     = x_t.size(0)
            t_ten = torch.full((B,), t_scalar, device=x_t.device, dtype=torch.long)
            with torch.no_grad():
                # La U-Net predice x_0 directamente
                x0_logits = net(x_t, t_ten, context, tension, z_style)
                x0_pred   = torch.sigmoid(x0_logits)   # → [0,1]

            at  = self.register['alphas_cumprod'][t_scalar]
            atp = self.register['alphas_cumprod'][t_prev] if t_prev >= 0 else torch.tensor(1.0)

            while at.dim() < x_t.dim():
                at  = at.unsqueeze(-1)
                atp = atp.unsqueeze(-1)

            # Reconstruir eps desde x0_pred
            eps = (x_t - at.sqrt() * x0_pred) / (1 - at).sqrt().clamp(min=1e-6)

            sigma   = eta * ((1 - atp) / (1 - at) * (1 - at / atp)).clamp(min=0).sqrt()
            dir_xt  = (1 - atp - sigma ** 2).clamp(min=0).sqrt() * eps
            noise   = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
            return atp.sqrt() * x0_pred + dir_xt + noise

    # ── Modelo completo ────────────────────────────────────────────────────
    class _DirectDiffusionComposer(nn.Module):
        """
        v3: difusión directa en el piano roll — sin encoder/decoder VAE.

        forward(x, context, tension) → loss, metrics
        sample(context, tension, z_style) → piano_roll [0,1]
        denoise_from_ref(x_ref, context, tension, z_style, strength) → piano_roll [0,1]
        """
        def __init__(self):
            super().__init__()
            self.unet        = _UNet()
            self.schedule    = _CosineSchedule(n_steps=n_steps)
            self.n_steps     = n_steps
            self.n_roles     = n_roles
            self.resolution  = resolution
            self.style_dim   = style_dim
            self.tension_dim = tension_dim
            # Proyector de estilo: promedia z del contexto → style vector
            self.style_proj = nn.Sequential(
                nn.Linear(n_roles * 4 * 8, style_dim),
                nn.Tanh(),
            )

        def _get_style(self, context):
            """Extrae z_style desde el contexto (promedio espacial)."""
            import torch
            B, R, C, H, W = context.shape
            pool = torch.nn.functional.adaptive_avg_pool2d(
                context.reshape(B * R * C, 1, H, W), (4, 8)
            ).reshape(B, -1)[:, :self.n_roles * 4 * 8]
            return self.style_proj(pool.reshape(B, self.n_roles * 4 * 8))

        def forward(self, x, context, tension, t=None):
            """
            x0-parametrization: la U-Net predice x_0 directamente en lugar
            del ruido ε. Esto funciona mejor con datos binarios (piano rolls)
            porque el modelo aprende directamente la distribución objetivo.
            """
            import torch
            B      = x.size(0)
            device = x.device
            self.schedule.to(device)

            z_style = self._get_style(context)

            if t is None:
                t = torch.randint(0, self.n_steps, (B,), device=device)

            x_t, eps_real = self.schedule.q_sample(x, t)

            # La U-Net predice x_0 directamente (no el ruido)
            x0_pred = self.unet(x_t, t, context, tension, z_style)
            x0_pred_prob = torch.sigmoid(x0_pred)   # → [0,1]

            # Loss principal: BCE entre x0 predicho y x0 real
            recon_loss = torch.nn.functional.binary_cross_entropy(
                x0_pred_prob.clamp(1e-6, 1 - 1e-6), x, reduction='mean')

            # Loss auxiliar: reconstruir el ruido desde x0_pred para
            # mantener coherencia con el schedule
            at = self.schedule.register['alphas_cumprod'][t]
            while at.dim() < x.dim():
                at = at.unsqueeze(-1)
            eps_from_x0 = (x_t - at.sqrt() * x0_pred_prob) / (1 - at).sqrt().clamp(min=1e-6)
            diff_loss = torch.nn.functional.mse_loss(eps_from_x0, eps_real)

            loss = recon_loss + 0.5 * diff_loss

            return loss, {'diff': diff_loss.item(), 'recon': recon_loss.item(),
                          'kl': 0.0, 'sparse': 0.0}

        @torch.no_grad()
        def sample(self, context, tension, z_style,
                   temperature=1.0, n_ddim_steps=50, eta=0.0):
            import torch
            B      = context.size(0)
            device = context.device
            self.schedule.to(device)

            x = torch.randn(B, self.n_roles, self.resolution, 128,
                            device=device) * temperature

            total   = self.n_steps
            step_sz = max(total // n_ddim_steps, 1)
            ts      = list(range(total - 1, -1, -step_sz))

            for i, t_cur in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                x = self.schedule.ddim_sample(
                    self.unet, x, t_cur, t_prev,
                    context, tension, z_style, eta=eta)

            return x.clamp(0, 1)   # DDIM converge directamente al piano roll

        @torch.no_grad()
        def denoise_from_ref(self, x_ref, context, tension, z_style,
                             strength=0.7, n_ddim_steps=50, eta=0.0):
            import torch
            B      = x_ref.size(0)
            device = x_ref.device
            self.schedule.to(device)

            t_start = int(strength * self.n_steps)
            t_ten   = torch.full((B,), t_start, device=device, dtype=torch.long)
            x_t, _  = self.schedule.q_sample(x_ref, t_ten)

            total   = t_start
            step_sz = max(total // n_ddim_steps, 1)
            ts      = list(range(t_start - 1, -1, -step_sz))

            x = x_t
            for i, t_cur in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                x = self.schedule.ddim_sample(
                    self.unet, x, t_cur, t_prev,
                    context, tension, z_style, eta=eta)

            return x.clamp(0, 1)

    return _DirectDiffusionComposer()


def _build_full_model(latent_dim: int, style_dim: int, tension_dim: int,
                      n_roles: int, window_bars: int, resolution: int,
                      n_diffusion_steps: int = 1000,
                      pos_weight: float = 5.0,
                      decoder_dropout: float = 0.1):
    """
    v3: wrapper que construye el modelo de difusión directa.
    Los parámetros latent_dim, pos_weight y decoder_dropout se ignoran
    (mantenidos por compatibilidad con el CLI).
    """
    return _build_unet_diffusion(
        n_roles     = n_roles,
        resolution  = resolution,
        style_dim   = style_dim,
        tension_dim = tension_dim,
        n_steps     = n_diffusion_steps,
    )

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
        'latent_dim':       args.latent_dim,   # informativo, no usado en v3
        'style_dim':        args.style_dim,
        'tension_dim':      tension_dim,
        'n_roles':          n_roles,
        'roles':            dataset.roles,
        'window_bars':      window_bars,
        'resolution':       resolution,
        'diffusion_steps':  args.diffusion_steps,
        'model_version':    'v3_direct',
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
        latent_dim        = cfg.get('latent_dim', 128),
        style_dim         = cfg['style_dim'],
        tension_dim       = cfg['tension_dim'],
        n_roles           = cfg['n_roles'],
        window_bars       = cfg['window_bars'],
        resolution        = cfg['resolution'],
        n_diffusion_steps = cfg.get('diffusion_steps', 1000),
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
    """
    v3: extrae z_style desde el contexto del MIDI (sin encoder VAE).
    Devuelve (z_context, z_style) para compatibilidad con el CLI.
    """
    import torch, numpy as np

    rolls       = _midi_to_rolls(midi_path, cfg)
    ctx_np      = _rolls_to_context_tensor(rolls, cfg)
    ctx_t       = torch.tensor(ctx_np).unsqueeze(0)   # (1, N_ROLES, ctx_bars, res, 128)

    # tension dummy (cero) solo para extraer estilo
    tension_dim = cfg['tension_dim']
    tension     = torch.zeros(1, tension_dim)

    with torch.no_grad():
        z_style = model._get_style(ctx_t)

    # z_context = promedio de todos los compases del piano roll
    n_roles   = cfg['n_roles']
    role_list = cfg['roles']
    resolution = cfg['resolution']
    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 0
    x_np   = np.zeros((n_roles, resolution, 128), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls and n_bars > 0:
            x_np[ridx] = rolls[role].mean(axis=0)

    return x_np.flatten()[:cfg.get('latent_dim', 128)], z_style[0].numpy()


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

    Para distribuciones bimodales (v3: casi todo cero, picos en 1.0):
        Usa el percentil sobre TODOS los valores, no solo los >0.001.
        Con percentile=99 y distribución 99% ceros → umbral ~0.0, deja pasar las notas.
        Con percentile=99.9 → umbral más alto → menos notas.

    Para distribuciones continuas (v1/v2: valores dispersos entre 0 y 1):
        Usa el percentil sobre valores >0.001 como antes.

    El rango útil:
        Distribución bimodal (v3): percentile=99.0–99.9
        Distribución continua (v1/v2): percentile=85–97
    """
    import numpy as np
    flat = roll.flatten()

    # Detectar si la distribución es bimodal:
    # si más del 90% de valores son < 0.01, es bimodal
    frac_near_zero = float((flat < 0.01).mean())
    if frac_near_zero > 0.90:
        # Bimodal: usar percentil sobre todos los valores
        thr = float(np.percentile(flat, percentile))
        return max(thr, 1e-4)   # nunca cero
    else:
        # Continua: usar percentil sobre valores activos
        nonzero = flat[flat > 0.001]
        if len(nonzero) == 0:
            return 0.5
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

            # Umbral: --threshold fijo tiene prioridad sobre adaptativo
            if getattr(args, 'threshold', None):
                adaptive_thr = args.threshold
                thr_method   = f'fijo ({args.threshold})'
            else:
                thr_pct      = getattr(args, 'threshold_pct', 99.0)
                adaptive_thr = _adaptive_threshold(bar_np, percentile=thr_pct)
                thr_method   = f'p{thr_pct}'

            n_active = int((bar_np > adaptive_thr).sum())
            density  = 100 * n_active / bar_np.size
            print(f"         Umbral {thr_method}: {adaptive_thr:.4f}  →  "
                  f"{n_active} píxeles activos ({density:.2f}%)")

            if density < 0.2:
                print(f"  [diag] ⚠  Densidad baja ({density:.2f}%) — "
                      f"prueba --threshold-pct 99.5 o --threshold-pct 99.9")
            elif density > 8.0:
                print(f"  [diag] ⚠  Densidad alta ({density:.2f}%) — "
                      f"prueba --threshold-pct 98 o --threshold-pct 95")
            elif vmean > 0.4:
                print(f"  [diag] ⚠  mean={vmean:.3f} — modelo aún indeciso (cerca de 0.5).")
                print(f"         Necesita más épocas para separar notas de silencio.")
            elif vmax < 0.05:
                print(f"  [diag] ⚠  Activaciones muy bajas (max={vmax:.4f}). "
                      f"El modelo necesita más entrenamiento.")

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

    # Umbral final: --threshold fijo tiene prioridad
    final_thr = getattr(args, 'threshold', None) or adaptive_thr
    n_notes = _rolls_to_midi(final_rolls, cfg, palette, args.output,
                              bpm=args.bpm, threshold=final_thr)
    print(f"[compose] MIDI guardado en {args.output}  ({n_notes} notas, umbral={final_thr:.3f})")
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
    v3: denoising con strength muy bajo (~0.05) compás a compás.

    En v3 no existe encoder/decoder separado — el modelo opera directamente
    sobre el piano roll. Este comando hace el equivalente: añade poco ruido
    al input y lo elimina con el denoiser, verificando que la U-Net funciona.

    Diagnóstico:
      p90 > 0.3  → U-Net OK, resultado musical
      p90 < 0.05 → U-Net aún no ha convergido, esperar más épocas
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
    rolls    = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("[reconstruct] ERROR: no se pudieron extraer rolls del MIDI")
        import sys; sys.exit(1)

    role_list   = cfg['roles']
    n_roles     = cfg['n_roles']
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']
    ctx_bars    = window_bars - 1
    tension_dim = cfg['tension_dim']
    n_bars      = min(r.shape[0] for r in rolls.values())
    print(f"[reconstruct] Roles: {list(rolls.keys())}  |  "
          f"Compases: {n_bars}  |  Resolución: {resolution}")

    ctx_np      = _rolls_to_context_tensor(rolls, cfg)
    ctx_buffer  = torch.tensor(ctx_np).unsqueeze(0).to(device)
    z_style_t   = model._get_style(ctx_buffer)

    strength    = args.strength if hasattr(args, 'strength') else 0.05
    bars_out    = {role: [] for role in role_list}
    adaptive_thr = None

    with torch.no_grad():
        for bar_idx in range(n_bars):
            bar = np.zeros((n_roles, resolution, 128), dtype=np.float32)
            for ri, role in enumerate(role_list):
                if role in rolls and bar_idx < rolls[role].shape[0]:
                    bar[ri] = rolls[role][bar_idx]

            x_ref   = torch.tensor(bar).unsqueeze(0).to(device)
            tension = torch.zeros(1, tension_dim).to(device)

            out = model.denoise_from_ref(
                x_ref, ctx_buffer, tension, z_style_t,
                strength=strength, n_ddim_steps=20, eta=0.0)
            out_np = out[0].cpu().numpy()   # (N_ROLES, res, 128)

            if bar_idx == 0:
                flat = out_np.flatten()
                nz   = flat[flat > 0.001]
                p90  = float(np.percentile(nz, 90)) if len(nz) > 0 else 0.0
                p99  = float(np.percentile(nz, 99)) if len(nz) > 0 else 0.0
                thr_diag = args.threshold if args.threshold else p99 * 0.5
                n_active = int((flat > thr_diag).sum())
                print(f"\n[diag] Compás 0 — denoising leve (strength={strength}):")
                print(f"       min={flat.min():.4f}  mean={flat.mean():.4f}  "
                      f"p50={float(np.percentile(flat,50)):.4f}  "
                      f"p90={p90:.4f}  p99={p99:.4f}  max={float(flat.max()):.4f}")
                print(f"       Umbral: {thr_diag:.4f}  →  {n_active} píxeles activos "
                      f"({100*n_active/len(flat):.2f}%)")
                if p90 < 0.05:
                    print(f"       ⚠  p90={p90:.4f} — U-Net aún no ha convergido")
                elif p90 < 0.20:
                    print(f"       ~  p90={p90:.4f} — U-Net en progreso")
                else:
                    print(f"       ✓  p90={p90:.4f} — U-Net funcionando correctamente")
                adaptive_thr = thr_diag

            # Actualizar contexto
            bar_bin = (out_np > adaptive_thr).astype(np.float32)
            bar_t   = torch.tensor(bar_bin).unsqueeze(0).to(device)
            ctx_buffer = torch.cat([
                ctx_buffer[:, :, 1:],
                bar_t.unsqueeze(2)
            ], dim=2)
            z_style_t = model._get_style(ctx_buffer)

            for ri, role in enumerate(role_list):
                bars_out[role].append(out_np[ri])

    final_rolls = {role: np.stack(b, axis=0) for role, b in bars_out.items()}
    all_vals    = np.concatenate([r.flatten() for r in final_rolls.values()])
    nz_all      = all_vals[all_vals > 0.001]
    thr   = args.threshold if args.threshold else (
        float(np.percentile(nz_all, 99)) * 0.5 if len(nz_all) > 0 else 0.5)
    method = 'fijo' if args.threshold else 'p99×0.5'

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
    p_prep.add_argument('--enrich',      action='store_true',
        help='Enriquecer el vector de condicionamiento con mscz2vec '
             '(tensión armónica + estabilidad tonal + tonalidad → 23D). '
             'Requiere music21 y mscz2vec. Se desactiva automáticamente si no están disponibles.')
    p_prep.add_argument('--mscz2vec-path', default=None, metavar='FILE',
        dest='mscz2vec_path',
        help='Ruta al script mscz2vec.py (opcional si está en el PATH o en el mismo directorio)')
    p_prep.add_argument('--enrich-external', action='store_true',
        dest='enrich_external',
        help='Cargar vector externo .vec junto a cada MIDI (mismo nombre, extensión .vec). '
             'El vector se replica para todos los compases y se añade al condicionamiento. '
             'Los MIDIs sin .vec se procesan sin este vector (marcado como vec:MISSING en el log).')
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
    p_comp.add_argument('--threshold-pct', type=float, default=99.0, metavar='FLOAT',
        dest='threshold_pct',
        help='Percentil para umbral adaptativo (default: 99.0 para distribuciones bimodales v3). '
             'Bájalo (ej. 99.5→99.9) si el MIDI sale vacío; súbelo (ej. 98→95) si hay demasiadas notas.')
    p_comp.add_argument('--threshold', type=float, default=None, metavar='FLOAT',
        help='Umbral fijo de binarización (0.0–1.0). Tiene prioridad sobre --threshold-pct. '
             'Para distribuciones bimodales (v3) prueba 0.5.')
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
