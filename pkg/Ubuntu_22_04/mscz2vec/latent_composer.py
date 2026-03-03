#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       LATENT COMPOSER  v0.1                                  ║
║         Composición end-to-end mediante VAE multi-rol                        ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare   — MIDI corpus → piano rolls segmentados por rol (.npz)          ║
║    train     — Entrena el VAE multi-rol                                      ║
║    encode    — MIDI referencia → z_style (.json)                             ║
║    compose   — Genera obra nueva (modos: reconstruct/z-noise/blend/sweep)   ║
║               Recuperación: --retrieval decoder (VAE) | nnr (vecino latente)║
║    inspect   — Diagnóstico del modelo y espacio latente                      ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    mido, numpy, torch, scipy                                                 ║
║    (scipy opcional: solo para sweep con --smooth)                            ║
║                                                                              ║
║  USO RÁPIDO:                                                                 ║
║    python latent_composer.py prepare --input-dir midis/ --output-dir data/  ║
║    python latent_composer.py train   --data-dir data/  --model-dir model/   ║
║    python latent_composer.py encode  --input ref.mid   --model-dir model/   ║
║    python latent_composer.py compose --mode z-noise --input ref.mid         ║
║                               --model-dir model/ --palette palette.json     ║
╚══════════════════════════════════════════════════════════════════════════════╝

python latent_composer.py train \
    --data-dir data/ \
    --model-dir model_small2/ \
    --epochs 300 \
    --batch-size 16 \
    --lr 1e-3 \
    --beta 1.0 \
    --beta-warmup 40 \
    --latent-dim 32 \
    --style-dim 16 \
    --pos-weight 5.0 \
    --patience 50 \
    --spatial-reg interval \
    --lambda-spatial 0.03 \
    --kl-threshold 1.0 \
    --kl-warmup-window 2 \
    --decoder-lr-factor 0.5 \
    --free-bits 0.5

python latent_composer.py compose --model-dir model/ --palette palette.json --mode z-noise --input ref.mid --temperature 0.5 --noise 0.1 

python latent_composer.py compose --model-dir model/ --palette palette.json --mode sweep --inputs ref.mid ref.mid --bars 64 --tension arch --retrieval nnr --data-dir data/ --nnr-temperature 0.5 
python latent_composer.py compose --model-dir model/ --palette palette.json --mode z-noise --input ref.mid --retrieval nnr --data-dir data/ --nnr-temperature 0.5

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

# Rangos MIDI canónicos por rol (lo, hi) — basados en INSTRUMENT_RANGES
# del sistema existente, extendidos para los 5 roles
ROLE_RANGES = {
    'melody':        (60, 96),   # soprano/melodía principal
    'counterpoint':  (52, 84),   # voz intermedia alta
    'accompaniment': (48, 84),   # voces intermedias / acordes
    'bass':          (28, 55),   # bajo
    'percussion':    (0,  127),  # cualquier pitch (canal 9)
}

# Programs GM que, sin canal 9, sugieren fuertemente un rol
# Usados como señal secundaria en la heurística
GM_ROLE_HINTS = {
    # bass
    43: 'bass',   # contrabass
    42: 'bass',   # cello (frecuentemente bass en conjuntos pequeños)
    58: 'bass',   # tuba
    70: 'bass',   # bassoon
    # melody / counterpoint  (flautas, oboe, trompeta, violín)
    73: 'melody', 72: 'melody', 56: 'melody', 40: 'melody',
    68: 'counterpoint', 71: 'counterpoint', 41: 'counterpoint',
    # accompaniment / pad
    48: 'accompaniment', 49: 'accompaniment',  # strings ensemble
    19: 'accompaniment', 52: 'accompaniment',  # organ, choir
    88: 'accompaniment', 89: 'accompaniment',  # pad
    # percusión ya la detectamos por canal
}

TICKS_PER_BAR_DEFAULT = 48   # resolución interna: 48 ticks = 1 compás 4/4
WINDOW_BARS_DEFAULT   = 4
PITCH_CLASSES         = 128


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MIDI COMUNES
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi(path: str):
    """Carga un fichero MIDI y devuelve el objeto mido.MidiFile."""
    import mido
    return mido.MidiFile(path)


def _midi_to_absolute_events(mid):
    """
    Convierte todas las pistas a una lista plana de eventos con tiempo
    absoluto en ticks, agrupados por (track_idx, channel).

    Devuelve:
        List[ dict(abs_tick, type, channel, track_idx, **msg_fields) ]
    """
    events = []
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        current_program = {ch: 0 for ch in range(16)}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'program_change':
                current_program[msg.channel] = msg.program
            ev = {'abs_tick': abs_tick, 'type': msg.type,
                  'track_idx': ti}
            if hasattr(msg, 'channel'):
                ev['channel'] = msg.channel
                ev['program'] = current_program.get(msg.channel, 0)
            if msg.type in ('note_on', 'note_off'):
                ev['note']     = msg.note
                ev['velocity'] = msg.velocity
            events.append(ev)
    events.sort(key=lambda e: e['abs_tick'])
    return events


def _extract_note_lists(mid):
    """
    Extrae notas por (track_idx, channel) como listas de
    (start_tick, end_tick, pitch, velocity, program).

    Devuelve:
        dict[(track_idx, channel)] = list[(start, end, pitch, vel, program)]
    """
    active   = {}   # (ti, ch, pitch) → (start_tick, vel, program)
    result   = {}

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
                            (st, abs_tick, msg.note, vel, pr)
                        )
    return result


def _ticks_per_bar(mid) -> int:
    """
    Calcula los ticks reales de un compás 4/4 desde el tempo map del MIDI.
    Devuelve ticks_per_beat × 4 (asumiendo 4/4).
    Para compases no 4/4 esto es una aproximación aceptable para el VAE.
    """
    return mid.ticks_per_beat * 4


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES  (RoleAssigner)
# ══════════════════════════════════════════════════════════════════════════════

class RoleAssigner:
    """
    Asigna uno de los 5 roles canónicos a cada stream (track_idx, channel)
    de un MIDI usando una heurística multi-señal.

    Señales usadas (en orden de peso):
      1. Canal 9 → percussion (infalible)
      2. Program GM → hint de rol si está en GM_ROLE_HINTS
      3. pitch_mean  — melodía arriba, bajo abajo
      4. pitch_range — melodía tiene rango amplio, bajo estrecho
      5. polyphony   — acompañamiento tiene alta polifonía simultánea
      6. note_density — percusión no-canal-9 tiene densidad alta + pitches bajos fijos

    Si dos streams compiten por el mismo rol, gana el de mayor score.
    El perdedor se mapea al siguiente rol más compatible o se descarta.
    """

    def assign(self, mid) -> dict:
        """
        Parámetros
        ----------
        mid : mido.MidiFile

        Devuelve
        --------
        dict[(track_idx, channel)] → rol (str)
            Solo contiene los streams que se asignan (uno por rol como máximo).
        """
        note_lists = _extract_note_lists(mid)
        if not note_lists:
            return {}

        profiles = self._build_profiles(note_lists, mid)
        return self._resolve_roles(profiles)

    # ── construcción de perfiles ──────────────────────────────────────────────

    def _build_profiles(self, note_lists: dict, mid) -> list:
        tpb_raw    = mid.ticks_per_beat
        total_dur  = max(n[1] for notes in note_lists.values() for n in notes) if note_lists else 1

        profiles = []
        for (ti, ch), notes in note_lists.items():
            if not notes:
                continue

            pitches    = [n[2] for n in notes]
            durations  = [n[1] - n[0] for n in notes]
            starts     = [n[0] for n in notes]
            program    = notes[0][4]   # program del primer evento

            pitch_mean  = sum(pitches) / len(pitches)
            pitch_std   = _std(pitches)
            pitch_range = max(pitches) - min(pitches)
            mean_dur    = sum(durations) / len(durations) if durations else 0
            density     = len(notes) / max(total_dur / tpb_raw, 1)

            # Polifonía media: cuántas notas suenan simultáneamente
            polyphony   = self._mean_polyphony(notes)

            profiles.append({
                'key':         (ti, ch),
                'channel':     ch,
                'program':     program,
                'pitch_mean':  pitch_mean,
                'pitch_std':   pitch_std,
                'pitch_range': pitch_range,
                'mean_dur':    mean_dur,
                'density':     density,
                'polyphony':   polyphony,
                'n_notes':     len(notes),
            })
        return profiles

    @staticmethod
    def _mean_polyphony(notes: list) -> float:
        """Cuántas notas están activas simultáneamente en promedio."""
        if len(notes) < 2:
            return 1.0
        events = []
        for (st, en, *_) in notes:
            events.append((st,  1))
            events.append((en, -1))
        events.sort()
        current = 0
        samples = []
        for _, delta in events:
            current += delta
            samples.append(max(current, 0))
        return sum(samples) / len(samples) if samples else 1.0

    # ── resolución de roles ───────────────────────────────────────────────────

    def _resolve_roles(self, profiles: list) -> dict:
        """
        Puntúa cada perfil para cada rol y asigna de forma greedy:
        el rol va al stream con mayor score; si hay empate posterior,
        el perdedor queda sin rol (descartado).
        """
        if not profiles:
            return {}

        # Paso 1: percusión por canal 9 (regla infalible)
        assigned    = {}   # rol → profile key
        unassigned  = []

        for p in profiles:
            if p['channel'] == 9:
                if 'percussion' not in assigned:
                    assigned['percussion'] = p['key']
                # si ya hay percusión, descartamos la redundante
            else:
                unassigned.append(p)

        # Paso 2: puntuar para los 4 roles restantes
        remaining_roles = [r for r in ROLES if r != 'percussion']

        # Caso especial: un único candidato → asignar directamente al rol
        # más compatible según su pitch_mean absoluto
        if len(unassigned) == 1:
            p = unassigned[0]
            pm = p['pitch_mean']
            if pm >= 60:
                role = 'melody'
            elif pm >= 52:
                role = 'counterpoint'
            elif pm >= 44:
                role = 'accompaniment'
            else:
                role = 'bass'
            assigned[role] = p['key']
            return assigned

        # Normalizar features para scoring
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
            gm_hint = GM_ROLE_HINTS.get(p['program'])
            hint_bonus = 0.25 if gm_hint == role else 0.0

            if role == 'melody':
                # alta pitch_mean + alto pitch_range + baja polifonía
                return (0.40 * n_pm[k] +
                        0.35 * n_pr[k] +
                        0.15 * (1 - n_poly[k]) +
                        hint_bonus)

            elif role == 'counterpoint':
                # pitch_mean media-alta + pitch_range medio
                mid_pm = abs(n_pm[k] - 0.65)   # cerca de 0.65 normalizado
                return (0.30 * (1 - mid_pm) +
                        0.25 * n_pr[k] +
                        0.20 * (1 - n_poly[k]) +
                        hint_bonus)

            elif role == 'accompaniment':
                # alta polifonía + pitch_mean media
                mid_pm = abs(n_pm[k] - 0.50)
                return (0.40 * n_poly[k] +
                        0.25 * (1 - mid_pm) +
                        0.15 * n_dens[k] +
                        hint_bonus)

            elif role == 'bass':
                # baja pitch_mean + bajo pitch_range
                return (0.50 * (1 - n_pm[k]) +
                        0.25 * (1 - n_pr[k]) +
                        hint_bonus)

            return 0.0

        # Construir matriz de scores
        score_matrix = {}
        for p in unassigned:
            score_matrix[p['key']] = {r: score(p, r) for r in remaining_roles}

        # Asignación greedy: iterar hasta que todos los roles estén cubiertos
        # o no queden candidatos
        taken_keys  = set()
        taken_roles = set()

        # Ordenar pares (role, key) por score descendente
        pairs = []
        for p in unassigned:
            for r in remaining_roles:
                pairs.append((score_matrix[p['key']][r], r, p['key']))
        pairs.sort(reverse=True)

        for sc, role, key in pairs:
            if role in taken_roles:
                continue
            if key in taken_keys:
                continue
            assigned[role] = key
            taken_roles.add(role)
            taken_keys.add(key)
            if len(taken_roles) == len(remaining_roles):
                break

        return assigned   # {rol: (track_idx, channel)}


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN A PIANO ROLL  (PianoRollConverter)
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    """
    Convierte un stream de notas MIDI en una secuencia de piano rolls binarios
    segmentados en compases y luego en ventanas de N compases.

    Piano roll de un compás:
        shape = (ticks_per_bar, 128)   — binario 0/1
        ticks_per_bar = TICKS_PER_BAR_DEFAULT (resolución interna, no del MIDI)

    Para convertir de ticks reales del MIDI a ticks internos se hace un
    remuestreo lineal.
    """

    def __init__(self, resolution: int = TICKS_PER_BAR_DEFAULT,
                 window_bars: int = WINDOW_BARS_DEFAULT):
        self.resolution   = resolution    # ticks internos por compás
        self.window_bars  = window_bars

    def notes_to_roll(self, notes: list, total_bars: int,
                      ticks_per_bar_raw: int) -> 'np.ndarray':
        """
        Convierte lista de notas al piano roll completo.

        Parámetros
        ----------
        notes           : [(start_tick, end_tick, pitch, vel, program), ...]
                          en ticks REALES del MIDI
        total_bars      : número de compases a representar
        ticks_per_bar_raw : ticks reales por compás del MIDI fuente

        Devuelve
        --------
        np.ndarray shape (total_bars, resolution, 128)  dtype float32
        """
        import numpy as np

        roll = np.zeros((total_bars, self.resolution, PITCH_CLASSES),
                        dtype=np.float32)

        scale = self.resolution / ticks_per_bar_raw  # factor de remuestreo

        for (st, en, pitch, vel, _) in notes:
            if not (0 <= pitch < PITCH_CLASSES):
                continue
            # convertir a ticks internos
            st_i = st * scale
            en_i = en * scale

            bar_start = int(st_i // self.resolution)
            bar_end   = int((en_i - 1e-6) // self.resolution)

            for bar in range(max(0, bar_start),
                             min(total_bars, bar_end + 1)):
                bar_st_tick = bar * self.resolution
                t0 = max(0, int(st_i - bar_st_tick))
                t1 = min(self.resolution, int(en_i - bar_st_tick))
                if t1 > t0:
                    roll[bar, t0:t1, pitch] = 1.0

        return roll

    def roll_to_windows(self, roll: 'np.ndarray') -> 'np.ndarray':
        """
        Segmenta el piano roll en ventanas deslizantes de window_bars compases,
        con paso de 1 compás (produce n_bars - window_bars + 1 ventanas).

        Devuelve
        --------
        np.ndarray shape (n_windows, window_bars, resolution, 128)
        """
        import numpy as np

        n_bars = roll.shape[0]
        if n_bars < self.window_bars:
            return np.zeros((0, self.window_bars, self.resolution,
                             PITCH_CLASSES), dtype=np.float32)

        n_windows = n_bars - self.window_bars + 1
        windows   = np.stack([roll[i:i + self.window_bars]
                               for i in range(n_windows)])
        return windows   # (n_windows, window_bars, resolution, 128)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTOR DE TENSIÓN  (TensionExtractor)
# ══════════════════════════════════════════════════════════════════════════════

class TensionExtractor:
    """
    Calcula un vector de tensión por compás a partir del piano roll multi-rol.

    Vector de tensión v_tension ∈ ℝ⁸:
        [0] tension_lerdahl  — disonancia armónica (semejante a tension_designer)
        [1] density          — notas activas / capacidad máxima
        [2] polyphony        — media de voces simultáneas normalizadas
        [3] register_mean    — pitch_mean global normalizado [0,1]
        [4] register_spread  — rango de pitches activos normalizado
        [5] velocity_mean    — dinámica media (si está disponible, sino 0.5)
        [6] rhythmic_density — cambios de estado / tick
        [7] arousal_proxy    — combinación de density + rhythmic_density
    """

    TENSION_DIM = 8

    def extract_bar_vectors(self, role_rolls: dict,
                            bars: int) -> 'np.ndarray':
        """
        Parámetros
        ----------
        role_rolls : dict[rol → np.ndarray(n_bars, resolution, 128)]
        bars       : número de compases

        Devuelve
        --------
        np.ndarray shape (bars, 8)
        """
        import numpy as np

        vectors = np.zeros((bars, self.TENSION_DIM), dtype=np.float32)

        for bar in range(bars):
            # Combinar todos los roles en un piano roll agregado para este compás
            combined = np.zeros((PITCH_CLASSES,), dtype=np.float32)
            total_events = 0
            resolution   = None

            for role, roll in role_rolls.items():
                if bar >= roll.shape[0]:
                    continue
                bar_roll  = roll[bar]          # (resolution, 128)
                resolution = bar_roll.shape[0]
                active    = bar_roll.max(axis=0)  # (128,) — 1 si suena en algún tick
                combined  = np.maximum(combined, active)
                total_events += bar_roll.sum()

            if resolution is None or resolution == 0:
                continue

            pitches_active = np.where(combined > 0)[0]
            n_active       = len(pitches_active)
            capacity       = resolution * PITCH_CLASSES

            # [0] tension_lerdahl: proporción de intervalos disonantes
            tension = self._lerdahl_proxy(pitches_active)

            # [1] density
            density = float(total_events) / max(capacity * len(role_rolls), 1)
            density = min(density * 20, 1.0)  # escalar a rango útil

            # [2] polyphony normalizada
            poly = n_active / 12.0   # 12 voces = saturación
            poly = min(poly, 1.0)

            # [3] register_mean
            reg_mean = float(np.mean(pitches_active)) / 127 if n_active > 0 else 0.5

            # [4] register_spread
            reg_spread = float(np.ptp(pitches_active)) / 127 if n_active > 1 else 0.0

            # [5] velocity_mean — no disponible en el piano roll binario
            vel_mean = 0.5

            # [6] rhythmic_density: cambios de estado por tick en melodía
            rhythm_density = 0.0
            if 'melody' in role_rolls and bar < role_rolls['melody'].shape[0]:
                mel = role_rolls['melody'][bar]      # (resolution, 128)
                active_per_tick = mel.sum(axis=1)    # (resolution,)
                changes = float(np.sum(np.diff(active_per_tick) != 0))
                rhythm_density = changes / max(resolution - 1, 1)

            # [7] arousal_proxy
            arousal = 0.5 * min(density * 2, 1.0) + 0.5 * rhythm_density

            vectors[bar] = [tension, density, poly, reg_mean,
                            reg_spread, vel_mean, rhythm_density, arousal]

        return vectors

    @staticmethod
    def _lerdahl_proxy(pitches_active: 'np.ndarray') -> float:
        """
        Proxy de tensión armónica de Lerdahl: cuenta intervalos disonantes
        (2m, 7M, 7m, tritono) entre las notas activas y los normaliza.
        """
        import numpy as np

        if len(pitches_active) < 2:
            return 0.0
        DISSONANT = {1, 2, 6, 10, 11}   # semitonos disonantes (mod 12)
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
#  UTILIDAD ESTADÍSTICA
# ══════════════════════════════════════════════════════════════════════════════

def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: prepare
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    """
    Worker function para paralelizar cmd_prepare.
    Procesa un único archivo MIDI y devuelve (stem, status, save_dict, stats_partial).
    Diseñada para ser llamada desde ProcessPoolExecutor.
    """
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
    all_ticks  = max(
        (n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpb_raw) + 1)

    role_rolls  = {}
    roles_found = []
    for role, key in role_assignment.items():
        notes = note_lists.get(key, [])
        if not notes:
            continue
        roll = converter.notes_to_roll(notes, total_bars, tpb_raw)
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
        'source':      stem,
        'resolution':  resolution,
        'window_bars': window_bars,
        'total_bars':  total_bars,
        'n_windows':   min_windows,
        'roles':       roles_found,
        'tpb_raw':     tpb_raw,
    }
    save_dict['meta_json'] = np.array([json.dumps(meta)])

    out_path = Path(output_dir) / f"{stem}.npz"
    np.savez_compressed(str(out_path), **save_dict)

    stats_partial['files_ok']      = 1
    stats_partial['total_windows'] = min_windows
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, roles: {', '.join(roles_found)})",
            True,
            stats_partial)


def cmd_prepare(args):
    """
    Procesa un directorio de MIDIs y genera archivos .npz con:
        - piano rolls segmentados por rol y ventana
        - vectores de tensión por compás
        - metadatos de asignación de roles

    Paralelizado con ProcessPoolExecutor para aprovechar todos los cores.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution  = args.resolution
    window_bars = args.window_bars

    midi_files = sorted(list(input_dir.glob('*.mid')) +
                        list(input_dir.glob('*.midi')))

    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    n_workers = min(multiprocessing.cpu_count(), len(midi_files))
    print(f"[prepare] {len(midi_files)} archivos MIDI encontrados")
    print(f"[prepare] Resolución: {resolution} ticks/compás  |  "
          f"Ventana: {window_bars} compases")
    print(f"[prepare] Paralelizando con {n_workers} procesos")
    print()

    stats = {r: 0 for r in ROLES}
    stats['files_ok']      = 0
    stats['files_skipped'] = 0
    stats['total_windows'] = 0

    task_args = [
        (midi_path, str(output_dir), resolution, window_bars)
        for midi_path in midi_files
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prepare_one_midi, a): a[0] for a in task_args}
        for future in as_completed(futures):
            midi_path = futures[future]
            try:
                stem, msg, ok, partial = future.result()
            except Exception as e:
                stem = Path(midi_path).stem
                print(f"  [{stem}] EXCEPCIÓN: {e}")
                stats['files_skipped'] += 1
                continue

            print(f"  [{stem}] {msg}")
            for role in ROLES:
                stats[role] += partial[role]
            stats['files_ok']      += partial['files_ok']
            stats['files_skipped'] += partial['files_skipped']
            stats['total_windows'] += partial['total_windows']

    # ── informe final ────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  RESUMEN PREPARE")
    print("═" * 60)
    print(f"  Archivos procesados : {stats['files_ok']}")
    print(f"  Archivos omitidos   : {stats['files_skipped']}")
    print(f"  Ventanas totales    : {stats['total_windows']}")
    print()
    print("  Cobertura de roles:")
    for role in ROLES:
        print(f"    {role:<16} {stats[role]} archivos")
    print("═" * 60)

    if args.report:
        _prepare_report(output_dir)


def _prepare_report(output_dir: Path):
    """Muestra estadísticas adicionales del corpus preparado."""
    import numpy as np

    npz_files = list(output_dir.glob('*.npz'))
    if not npz_files:
        return

    print()
    print("  INFORME DETALLADO DEL CORPUS")
    print("─" * 60)

    all_windows  = []
    role_counts  = {r: 0 for r in ROLES}
    tension_all  = []

    for f in npz_files:
        data = np.load(str(f), allow_pickle=True)
        meta = json.loads(str(data['meta_json'][0]))
        n    = meta['n_windows']
        all_windows.append(n)
        for role in meta['roles']:
            role_counts[role] += n
        if 'tension' in data:
            tension_all.append(data['tension'])

    print(f"  Ficheros .npz       : {len(npz_files)}")
    print(f"  Ventanas por fichero: min={min(all_windows)}  "
          f"max={max(all_windows)}  "
          f"media={sum(all_windows)/len(all_windows):.1f}")
    print()
    print("  Ventanas por rol:")
    for role in ROLES:
        bar_len = int(role_counts[role] / max(max(role_counts.values()), 1) * 30)
        bar     = '█' * bar_len
        print(f"    {role:<16} {role_counts[role]:>6}  {bar}")

    if tension_all:
        tension_cat = np.concatenate(tension_all, axis=0)
        labels = ['tension', 'density', 'polyphony', 'reg_mean',
                  'reg_spread', 'vel_mean', 'rhythm_dens', 'arousal']
        print()
        print("  Estadísticas de tensión (media ± std):")
        for i, label in enumerate(labels):
            col  = tension_cat[:, i]
            mean = float(col.mean())
            std  = float(col.std())
            print(f"    {label:<14} {mean:.3f} ± {std:.3f}")

    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  STUBS DE COMANDOS FUTUROS
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """
    Dataset PyTorch que carga los .npz generados por `prepare`.

    Cada muestra es un dict con:
        'context'  : Tensor (N_ROLES, window_bars, resolution, 128)  — ventana de entrada
        'target'   : Tensor (N_ROLES, resolution, 128)               — compás a predecir
        'tension'  : Tensor (TENSION_DIM,)                           — vector de tensión del target
        'role_mask': Tensor (N_ROLES,) bool                          — qué roles están presentes

    La ventana de contexto son los window_bars compases anteriores al target.
    El target es el compás window_bars (índice siguiente a la ventana).
    Por tanto necesitamos ventanas de tamaño window_bars+1 del roll original.
    Como prepare guarda ventanas de window_bars, aquí tratamos cada ventana[0:window_bars-1]
    como contexto y ventana[window_bars-1] como target.
    """

    def __init__(self, data_dir: str, roles: list = None,
                 augment: bool = False):
        import numpy as np
        self.samples  = []   # list of (npz_path, window_idx)
        self.roles    = roles or ROLES
        self.n_roles  = len(self.roles)
        self.augment  = augment
        self._cache   = {}   # path → np.NpzFile cargado en memoria

        npz_files = sorted(Path(data_dir).glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No hay .npz en {data_dir}")

        for path in npz_files:
            # Cargar como dict de arrays planos (no NpzFile lazy).
            # NpzFile mantiene un ZipFile abierto que NO es seguro para fork
            # y corrompe el estado en los workers de DataLoader.
            with np.load(str(path), allow_pickle=True) as raw:
                data = {k: raw[k] for k in raw.files}   # arrays en RAM
            self._cache[path] = data
            meta = json.loads(str(data['meta_json'][0]))
            n    = meta['n_windows']
            roles_present = [r for r in self.roles
                             if f'roll_{r}' in data]
            if not roles_present:
                continue
            for i in range(n):
                self.samples.append((path, i))

        if not self.samples:
            raise ValueError("El dataset está vacío tras filtrar.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        path, widx = self.samples[idx]
        data = self._cache[path]             # ← desde RAM, sin disco
        meta = json.loads(str(data['meta_json'][0]))

        wb  = meta['window_bars']    # e.g. 4
        res = meta['resolution']     # e.g. 48

        # Construir tensor de roles (N_ROLES, window_bars, res, 128)
        context_list = []
        role_mask    = []

        for role in self.roles:
            key = f'roll_{role}'
            if key in data:                          # dict normal, no NpzFile
                win = data[key][widx]    # (window_bars, res, 128)
                context_list.append(win)
                role_mask.append(True)
            else:
                context_list.append(
                    np.zeros((wb, res, 128), dtype=np.float32))
                role_mask.append(False)

        context = np.stack(context_list, axis=0)   # (N_ROLES, wb, res, 128)

        # Context: primeros wb-1 compases; target: último compás
        context_in = context[:, :-1, :, :]          # (N_ROLES, wb-1, res, 128)
        target     = context[:, -1,  :, :]           # (N_ROLES, res, 128)

        # Tensión del compás target
        if 'tension' in data:                        # dict normal
            tension = data['tension'][widx]          # (8,)
        else:
            tension = np.zeros(TensionExtractor.TENSION_DIM, dtype=np.float32)

        # Augmentación: transposición aleatoria ±6 semitonos
        if self.augment:
            shift = np.random.randint(-6, 7)
            if shift != 0:
                context_in = _transpose_roll(context_in, shift)
                target     = _transpose_roll(target,     shift)

        return {
            'context':   torch.tensor(context_in),
            'target':    torch.tensor(target),
            'tension':   torch.tensor(tension),
            'role_mask': torch.tensor(role_mask),
        }


def _transpose_roll(roll: 'np.ndarray', shift: int) -> 'np.ndarray':
    """
    Transpone un piano roll desplazando el eje de pitch.
    roll puede ser (..., 128). Notas que salen del rango [0,127] se descartan.
    """
    import numpy as np
    if shift == 0:
        return roll
    result = np.zeros_like(roll)
    if shift > 0:
        result[..., shift:] = roll[..., :-shift]
    else:
        result[..., :shift] = roll[..., -shift:]
    return result


def collate_fn(batch):
    """Agrupa muestras en un batch, apilando tensors."""
    import torch
    return {
        'context':   torch.stack([b['context']   for b in batch]),
        'target':    torch.stack([b['target']    for b in batch]),
        'tension':   torch.stack([b['tension']   for b in batch]),
        'role_mask': torch.stack([b['role_mask'] for b in batch]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA VAE
# ══════════════════════════════════════════════════════════════════════════════

class RoleEncoder(object):  # se reemplaza por nn.Module al importar torch
    pass

class MultiRoleVAE(object):
    pass

def _build_model(latent_dim: int, style_dim: int, tension_dim: int,
                 n_roles: int, window_bars: int, resolution: int):
    """
    Construye el MultiRoleVAE. Separado en función para poder importar
    torch solo cuando se necesita.

    Arquitectura:
        Encoder:
            Para cada compás i en [0, window_bars-1]:
                Para cada rol r:
                    Linear(resolution*128 → 256) → ReLU  →  h_bar_r   [256]
                Concat de N_ROLES h_bar_r  →  h_bar  [N_ROLES*256]
                Linear(N_ROLES*256 → 256)  →  ReLU   →  h_bar_proj [256]
            GRU(input=256, hidden=512, layers=1) sobre secuencia de compases
            → h_context [512]
            → Linear → μ [latent_dim],  Linear → log_σ² [latent_dim]

        Decoder:
            z_augmented = concat(z [latent_dim], z_style [style_dim],
                                 v_tension [tension_dim])
            Linear(latent_dim+style_dim+tension_dim → 512) → ReLU
            GRU(input=512, hidden=512) desempaquetado en window_bars-1 pasos
            Para cada paso t, para cada rol r:
                Linear(512 → resolution*128) → Sigmoid  →  piano_roll_r

        Style encoder (usado en encode y compose):
            Media de los z del encoder sobre todas las ventanas del MIDI ref.
            → z_style  [style_dim]   (proyectado con Linear(latent_dim → style_dim))
    """
    import torch
    import torch.nn as nn

    PITCH        = 128
    bar_flat     = resolution * PITCH      # 48 × 128 = 6144
    h_bar        = 256
    h_gru        = 512
    # Dividir el espacio latente en global (estilo) y local (contenido del compás)
    # latent_dim debe ser divisible; si no, z_global_dim = latent_dim // 2
    z_global_dim = latent_dim // 2         # p.ej. 64 de 128
    z_local_dim  = latent_dim - z_global_dim  # resto (64)

    class _RoleBarEncoder(nn.Module):
        """Codifica un único compás de un único rol → vector h_bar."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(bar_flat, h_bar),
                nn.ReLU(),
                nn.Linear(h_bar, h_bar),
                nn.ReLU(),
            )
        def forward(self, x):
            return self.fc(x.reshape(x.size(0), -1))   # (B, h_bar)

    class _HierarchicalMultiRoleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.training      = True
            self.n_roles       = n_roles
            self.window_bars   = window_bars
            self.resolution    = resolution
            self.latent_dim    = latent_dim
            self.z_global_dim  = z_global_dim
            self.z_local_dim   = z_local_dim
            self.style_dim     = style_dim
            self.tension_dim   = tension_dim

            # ── Encoder compartido: RoleBarEncoder por rol ────────────────
            self.role_encoders = nn.ModuleList(
                [_RoleBarEncoder() for _ in range(n_roles)]
            )
            self.bar_proj = nn.Sequential(
                nn.Linear(n_roles * h_bar, h_bar),
                nn.ReLU(),
            )

            # ── Nivel global: captura estilo global (tonalidad, densidad) ─
            # GRU sobre secuencia de compases → z_g
            self.gru_global   = nn.GRU(
                input_size=h_bar, hidden_size=h_gru,
                num_layers=1, batch_first=True
            )
            self.fc_mu_g      = nn.Linear(h_gru, z_global_dim)
            self.fc_logvar_g  = nn.Linear(h_gru, z_global_dim)

            # ── Nivel local: captura contenido del compás target ──────────
            # Condicionado en z_g para que aprenda "qué notas dado este estilo"
            self.local_encoder = nn.Sequential(
                nn.Linear(h_bar + z_global_dim, h_bar),
                nn.ReLU(),
            )
            self.fc_mu_l     = nn.Linear(h_bar, z_local_dim)
            self.fc_logvar_l = nn.Linear(h_bar, z_local_dim)

            # ── Style projector (desde z_g, no z_total) ───────────────────
            self.style_proj = nn.Sequential(
                nn.Linear(z_global_dim, style_dim),
                nn.Tanh(),
            )

            # ── Decoder: conductor desde z_g + detalle desde z_l ─────────
            #
            # El conductor procesa z_g (estilo global) junto con tensión
            # y produce h_conductor. El decoder local usa h_conductor + z_l
            # para generar el piano roll.
            #
            # Gradientes: z_g recibe gradiente desde la reconstrucción a través
            # del conductor, y z_l directamente desde las skip connections.
            # Es estructuralmente imposible ignorar ninguno de los dos.
            conductor_in = z_global_dim + style_dim + tension_dim
            self.conductor = nn.Sequential(
                nn.Linear(conductor_in, h_gru),
                nn.ReLU(),
            )
            self.dec_merge = nn.Sequential(
                nn.Linear(h_gru + z_local_dim, h_gru),
                nn.ReLU(),
            )
            self.gru_dec = nn.GRU(
                input_size=h_gru, hidden_size=h_gru,
                num_layers=1, batch_first=True
            )

            # Skip connections de z_g y z_l al rol-decoder
            # z_g → estilo por rol; z_l → contenido específico del compás
            self.zg_skip  = nn.Linear(z_global_dim, h_bar)
            self.zg_gate  = nn.Linear(z_global_dim, h_bar)
            self.zl_skip  = nn.Linear(z_local_dim,  h_bar)
            self.zl_gate  = nn.Linear(z_local_dim,  h_bar)

            self.role_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(h_gru + h_bar + h_bar, h_bar),  # h_out + zg_gated + zl_gated
                    nn.ReLU(),
                    nn.Linear(h_bar, bar_flat),
                    nn.Sigmoid(),
                )
                for _ in range(n_roles)
            ])

        def train(self, mode=True):
            self.training = mode
            return self

        # ── encode ────────────────────────────────────────────────────────
        def encode(self, context):
            """
            context: (B, N_ROLES, context_bars, resolution, 128)
            Devuelve (mu, logvar) cada uno (B, latent_dim)
            donde latent_dim = z_global_dim + z_local_dim
            y mu = concat(mu_g, mu_l), logvar = concat(logvar_g, logvar_l)
            """
            B     = context.size(0)
            n_ctx = context.size(2)

            bar_vecs = []
            for t in range(n_ctx):
                role_vecs = []
                for r in range(self.n_roles):
                    x_r = context[:, r, t, :, :]
                    h_r = self.role_encoders[r](x_r)
                    role_vecs.append(h_r)
                h_multi = torch.cat(role_vecs, dim=-1)
                h_bar_t = self.bar_proj(h_multi)
                bar_vecs.append(h_bar_t)

            seq          = torch.stack(bar_vecs, dim=1)   # (B, n_ctx, h_bar)
            _, h_last    = self.gru_global(seq)            # (1, B, h_gru)
            h_g          = h_last.squeeze(0)               # (B, h_gru)

            mu_g     = self.fc_mu_g(h_g)                  # (B, z_global_dim)
            logvar_g = self.fc_logvar_g(h_g)

            # Nivel local: condicionado en z_g (usamos mu_g en inferencia)
            # El último compás del contexto como representación del target
            h_target = bar_vecs[-1]                        # (B, h_bar)
            z_g_det  = mu_g.detach()                       # no propagar gradiente al global desde local
            h_local  = self.local_encoder(
                torch.cat([h_target, z_g_det], dim=-1))    # (B, h_bar)
            mu_l     = self.fc_mu_l(h_local)               # (B, z_local_dim)
            logvar_l = self.fc_logvar_l(h_local)

            # Concatenar para interfaz uniforme con el resto del código
            mu     = torch.cat([mu_g,     mu_l],     dim=-1)  # (B, latent_dim)
            logvar = torch.cat([logvar_g, logvar_l], dim=-1)
            return mu, logvar

        # ── reparametrización ─────────────────────────────────────────────
        def reparametrize(self, mu, logvar):
            if self.training:
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

        # ── decode ────────────────────────────────────────────────────────
        def decode(self, z, z_style, v_tension):
            """
            z         : (B, latent_dim)  = concat(z_g, z_l)
            z_style   : (B, style_dim)
            v_tension : (B, tension_dim)
            Devuelve  : (B, N_ROLES, resolution, 128)
            """
            B    = z.size(0)
            z_g  = z[:, :self.z_global_dim]   # (B, z_global_dim)
            z_l  = z[:, self.z_global_dim:]   # (B, z_local_dim)

            # Conductor: z_g + estilo + tensión → h_conductor
            cond_in    = torch.cat([z_g, z_style, v_tension], dim=-1)
            h_cond     = self.conductor(cond_in)              # (B, h_gru)

            # Merge con z_l y pasar por GRU
            merge_in   = torch.cat([h_cond, z_l], dim=-1)
            h_merged   = self.dec_merge(merge_in).unsqueeze(1)  # (B, 1, h_gru)
            out, _     = self.gru_dec(h_merged)                  # (B, 1, h_gru)
            h_out      = out[:, 0, :]                            # (B, h_gru)

            # Skip connections gated para z_g y z_l
            zg_proj  = torch.relu(self.zg_skip(z_g))
            zg_gated = torch.sigmoid(self.zg_gate(z_g)) * zg_proj  # (B, h_bar)
            zl_proj  = torch.relu(self.zl_skip(z_l))
            zl_gated = torch.sigmoid(self.zl_gate(z_l)) * zl_proj  # (B, h_bar)

            h_combined = torch.cat([h_out, zg_gated, zl_gated], dim=-1)  # (B, h_gru+2*h_bar)

            rolls = []
            for r in range(self.n_roles):
                flat   = self.role_decoders[r](h_combined)
                roll_r = flat.reshape(B, self.resolution, 128)
                rolls.append(roll_r)

            return torch.stack(rolls, dim=1)   # (B, N_ROLES, res, 128)

        # ── forward completo ──────────────────────────────────────────────
        def forward(self, context, v_tension, z_style=None):
            """
            context   : (B, N_ROLES, context_bars, resolution, 128)
            v_tension : (B, tension_dim)
            z_style   : (B, style_dim) opcional; si None se proyecta desde μ_g

            Devuelve  : recon (B, N_ROLES, res, 128), μ, logvar
            """
            mu, logvar = self.encode(context)
            z          = self.reparametrize(mu, logvar)
            if z_style is None:
                # Proyectar solo desde la parte global de mu
                mu_g    = mu[:, :self.z_global_dim]
                z_style = self.style_proj(mu_g.detach())
            recon = self.decode(z, z_style, v_tension)
            return recon, mu, logvar

    return _HierarchicalMultiRoleVAE()



# ══════════════════════════════════════════════════════════════════════════════
#  REGULARIZACIÓN ESPACIAL DEL PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

def spatial_loss_smoothness(recon):
    """
    Smoothness loss: penaliza diferencias bruscas entre pitches adyacentes.

    La hipótesis musical es que los píxeles vecinos en el eje de pitch tienden
    a estar correlacionados (acordes, cromatismo). Penalizar el gradiente
    discourages al decoder de predecir notas aisladas sin contexto armónico,
    forzándolo a leer z para saber dónde colocar grupos de notas.

    recon : (B, N_ROLES, res, 128)
    return: escalar
    """
    # Diferencia entre cada pitch p y su vecino p+1
    diff = recon[..., 1:] - recon[..., :-1]   # (B, N_ROLES, res, 127)
    return (diff ** 2).mean()


def spatial_loss_pitch(recon, window: int = 2):
    """
    Contrastive pitch loss: penaliza notas activas sin vecinas cercanas.

    Una nota aislada (activa en pitch p, silencio en [p-window, p+window])
    es un artefacto habitual del posterior collapse. Este término penaliza
    exactamente esos casos usando max-pooling como detector de vecindad.

    recon  : (B, N_ROLES, res, 128)
    window : radio en semitonos que define «vecindad» (default: 2)
    return : escalar
    """
    import torch.nn.functional as F

    B, R, T, P = recon.shape
    # Aplanar a (N, 1, P) para usar max_pool1d como detector de vecinos
    flat         = recon.reshape(B * R * T, 1, P)
    neighborhood = F.max_pool1d(
        flat,
        kernel_size = window * 2 + 1,
        stride      = 1,
        padding     = window
    ).reshape(B, R, T, P)

    # Una nota activa en una zona sin vecinos contribuye al loss.
    # Usamos .detach() en neighborhood y recon para que el gradiente
    # solo fluya a través del factor recon (evita bucle de gradiente).
    isolated = recon * (1.0 - neighborhood.detach() + recon.detach())
    return isolated.mean()


def spatial_loss_interval(recon, max_interval: int = 12):
    """
    Interval loss: penaliza coactivaciones entre notas separadas más de
    max_interval semitonos sin notas intermedias.

    Basado en la estadística de intervalos musicales: 2ªs y 3ªs son mucho
    más frecuentes que 7ªs o 9ªs. Los saltos grandes no se prohíben, pero
    se ponderan proporcionalmente a su tamaño, lo que suaviza la distribución
    de intervalos hacia algo musicalmente más plausible.

    recon        : (B, N_ROLES, res, 128)
    max_interval : intervalo máximo penalizado en semitonos (default: 12 = 8va)
    return       : escalar
    """
    import torch

    B, R, T, P = recon.shape
    x    = recon.reshape(B * R * T, P)           # (N, 128)
    loss = torch.tensor(0.0, device=recon.device)

    for interval in range(1, max_interval + 1):
        # co_active[i, p] = probabilidad de que pitch p y pitch p+interval
        # estén activos simultáneamente en el tick i
        co_active = x[:, interval:] * x[:, :-interval]   # (N, 128-interval)
        # Saltos más grandes pesan más: normalizado a [1/max, 1]
        weight    = interval / max_interval
        loss      = loss + weight * co_active.mean()

    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# Modos de regularización espacial disponibles
SPATIAL_REG_MODES = ('none', 'smoothness', 'pitch', 'interval')


def vae_loss(recon, target, mu, logvar, role_mask,
             beta: float = 1.0, pos_weight: float = 10.0,
             spatial_reg: str = 'none',
             lambda_spatial: float = 0.05,
             free_bits: float = 0.1,
             z_global_dim: int = 0):
    """
    ELBO loss para el VAE multi-rol con regularización espacial opcional.

    Para el VAE jerárquico, el KL se calcula por separado para z_g y z_l
    con free_bits independientes. Esto evita que el collapse de un nivel
    destruya el otro.

    recon          : (B, N_ROLES, res, 128)  — salida del decoder (sigmoid aplicado)
    target         : (B, N_ROLES, res, 128)  — piano roll objetivo
    mu, logvar     : (B, latent_dim)  = concat([mu_g, mu_l], [logvar_g, logvar_l])
    role_mask      : (B, N_ROLES) bool       — qué roles están presentes en cada muestra
    beta           : peso del término KL (beta-VAE)
    pos_weight     : peso de los píxeles positivos en BCE (compensa desbalance 0/1)
    spatial_reg    : modo de regularización espacial: 'none' | 'smoothness' |
                     'pitch' | 'interval'
    lambda_spatial : peso del término de regularización espacial
    free_bits      : nats mínimos por dimensión latente (evita posterior collapse)
    z_global_dim   : dims del nivel global (0 = VAE plano, compatibilidad hacia atrás)

    Devuelve  : loss total, recon_loss, kl_loss  (todos escalares)
    """
    import torch
    import torch.nn.functional as F

    B, N_ROLES, res, pitch = recon.shape

    # ── Reconstrucción BCE ────────────────────────────────────────────────────
    # BCE por píxel con pos_weight para compensar la alta esparsidad.
    # Solo consideramos roles presentes en la muestra.
    recon_loss = torch.tensor(0.0, device=recon.device)
    n_active   = 0

    for r in range(N_ROLES):
        mask_r = role_mask[:, r]          # (B,) bool
        if not mask_r.any():
            continue
        r_recon  = recon[mask_r, r]       # (B', res, 128)
        r_target = target[mask_r, r]      # (B', res, 128)

        # Reconstruir logits desde sigmoid output para la BCE
        # sigmoid(x) = p  →  x = log(p/(1-p))
        p     = r_recon
        p_inv = 1.0 - r_recon
        # Clip sin torch.clamp (compatible con mock y torch real)
        p_c  = type(p)(p._d.clip(1e-6, 1-1e-6))   if hasattr(p,   '_d') else p.clamp(1e-6, 1-1e-6)
        pi_c = type(p)(p_inv._d.clip(1e-6, 1-1e-6)) if hasattr(p_inv,'_d') else p_inv.clamp(1e-6,1-1e-6)
        logits = p_c.log() - pi_c.log()

        pw  = torch.tensor(pos_weight, device=recon.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            r_target,
            pos_weight = pw,
            reduction  = 'mean'
        )
        recon_loss = recon_loss + bce
        n_active  += 1

    if n_active > 0:
        recon_loss = recon_loss / n_active

    # ── KL con free bits por dimensión ────────────────────────────────────────
    # Clamp de μ y log σ² para estabilidad numérica.
    mu_c     = mu.clamp(-6.0, 6.0)
    logvar_c = logvar.clamp(-6.0, 4.0)

    kl_per_dim = -0.5 * (1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

    if z_global_dim > 0:
        # ── VAE jerárquico: KL separado por nivel ─────────────────────────
        # z_g (global) y z_l (local) tienen free_bits independientes.
        # Si un nivel colapsa, el otro sigue funcionando.
        # z_g recibe más presión (free_bits x2) para que codifique estilo global.
        kl_g   = kl_per_dim[:, :z_global_dim]
        kl_l   = kl_per_dim[:, z_global_dim:]
        kl_loss = (torch.mean(torch.clamp(kl_g, min=free_bits * 2.0)) +
                   torch.mean(torch.clamp(kl_l, min=free_bits)))
    else:
        # ── VAE plano: compatibilidad hacia atrás ─────────────────────────
        kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits))

    # ── Regularización espacial ───────────────────────────────────────────────
    if spatial_reg == 'smoothness':
        s_loss = spatial_loss_smoothness(recon)
    elif spatial_reg == 'pitch':
        s_loss = spatial_loss_pitch(recon)
    elif spatial_reg == 'interval':
        s_loss = spatial_loss_interval(recon)
    else:
        s_loss = torch.tensor(0.0, device=recon.device)

    total = recon_loss + beta * kl_loss + lambda_spatial * s_loss
    return total, recon_loss, kl_loss


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    """Formatea segundos como string legible: 1h 23m 45s / 4m 12s / 38s."""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60:02d}s"
    if s >= 60:
        return f"{s//60}m {s%60:02d}s"
    return f"{s}s"

class Trainer:
    """
    Gestiona el bucle de entrenamiento del VAE multi-rol.

    Características:
        - Checkpoint tras cada época (guarda mejor modelo por val_loss)
        - Early stopping configurable
        - Warm-up lineal del coeficiente beta del KL
        - Logging en texto + guardado de historial en JSON
        - Reanudación desde checkpoint con --resume
    """

    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'train_history.json'
    CONFIG_NAME     = 'model_config.json'

    def __init__(self, model, optimizer, model_dir: Path,
                 beta_start: float = 0.0, beta_end: float = 1.0,
                 beta_warmup_epochs: int = 20,
                 early_stop_patience: int = 15,
                 pos_weight: float = 10.0,
                 spatial_reg: str = 'none',
                 lambda_spatial: float = 0.05,
                 kl_threshold: float = 5.0,
                 kl_warmup_window: int = 3,
                 freeze_decoder_epochs: int = 0,
                 decoder_lr_factor: float = 0.1,
                 free_bits: float = 0.1):
        self.model             = model
        self.optimizer         = optimizer
        self.model_dir         = model_dir
        self.beta_start        = beta_start
        self.beta_end          = beta_end
        self.beta_warmup       = beta_warmup_epochs
        self.patience          = early_stop_patience
        self.pos_weight        = pos_weight
        self.spatial_reg       = spatial_reg
        self.lambda_spatial    = lambda_spatial
        self.kl_threshold          = kl_threshold
        self.kl_warmup_window      = kl_warmup_window
        self.freeze_decoder_epochs = freeze_decoder_epochs
        # Factor de lr para el decoder al descongelarse (< 1 = más lento que encoder)
        self.decoder_lr_factor     = decoder_lr_factor
        self.free_bits             = free_bits

        self.history        = {'train': [], 'val': [], 'val_recon': [],
                               'val_kl': [], 'beta': []}
        self.best_val_loss  = float('inf')
        self.best_val_recon = float('inf')
        self.no_improve     = 0
        self.start_epoch    = 0
        # Estado interno del warmup adaptativo.
        # Se recalcula desde history['val_kl'] al reanudar con --resume.
        self._kl_above_epochs = 0    # épocas consecutivas con KL > threshold
        self._warmup_started  = False # True cuando β empieza a subir
        self._warmup_start_ep = 0     # época absoluta en que arrancó el warmup

    # Nombres de los módulos del decoder (usados para congelar/descongelar)
    _DECODER_MODULES = (
        # Flat VAE modules (kept for backward compat)
        'dec_input', 'gru_dec', 'role_decoders', 'z_skip_proj', 'z_gate',
        # Hierarchical VAE modules
        'conductor', 'dec_merge', 'zg_skip', 'zg_gate', 'zl_skip', 'zl_gate',
    )

    def _set_decoder_grad(self, requires_grad: bool):
        """Activa o congela los gradientes de todos los parámetros del decoder."""
        raw = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in raw.named_parameters():
            if any(name.startswith(m) for m in self._DECODER_MODULES):
                param.requires_grad = requires_grad

    def _apply_decoder_lr(self):
        """
        Ajusta el learning rate del decoder al descongelarlo.
        Divide el lr del grupo del decoder por decoder_lr_factor respecto
        al lr base del encoder, evitando que el decoder aprenda demasiado
        rápido e ignore z al descongelarse.
        Requiere que el optimizador tenga dos param_groups:
          [0] encoder  (lr base)
          [1] decoder  (lr reducido)
        Si el optimizador solo tiene un grupo (setup antiguo), no hace nada.
        """
        if len(self.optimizer.param_groups) < 2:
            return
        base_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[1]['lr'] = base_lr * self.decoder_lr_factor
        print(f"  [decoder lr={base_lr * self.decoder_lr_factor:.2e}  "
              f"encoder lr={base_lr:.2e}]")

    def _recalc_warmup_state(self):
        """Reconstruye el estado del warmup adaptativo desde el historial."""
        kl_hist = self.history.get('val_kl', [])
        self._kl_above_epochs = 0
        self._warmup_started  = False
        self._warmup_start_ep = 0
        for ep, kl in enumerate(kl_hist):
            if not self._warmup_started:
                if kl > self.kl_threshold:
                    self._kl_above_epochs += 1
                else:
                    self._kl_above_epochs = 0
                if self._kl_above_epochs >= self.kl_warmup_window:
                    self._warmup_started  = True
                    self._warmup_start_ep = ep + 1

    def _beta(self, epoch: int, current_kl: float = None) -> float:
        """
        Beta annealing adaptativo basado en KL.

        β permanece en 0 hasta que el KL supera kl_threshold durante
        kl_warmup_window épocas consecutivas. A partir de ese punto sube
        linealmente durante beta_warmup épocas hasta beta_end.

        Si beta_warmup <= 0 se salta el warmup y se devuelve beta_end
        directamente (comportamiento original).
        """
        if self.beta_warmup <= 0:
            return self.beta_end

        # Actualizar contador de épocas con KL alto si se pasa el KL actual
        if current_kl is not None:
            if current_kl > self.kl_threshold:
                self._kl_above_epochs += 1
            else:
                self._kl_above_epochs = 0
            # Disparar el warmup lineal si se alcanza la ventana
            if (not self._warmup_started and
                    self._kl_above_epochs >= self.kl_warmup_window):
                self._warmup_started  = True
                self._warmup_start_ep = epoch

        if not self._warmup_started:
            return 0.0

        # Warmup lineal desde _warmup_start_ep durante beta_warmup épocas
        t = min((epoch - self._warmup_start_ep) / self.beta_warmup, 1.0)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        import torch
        state = {
            'epoch':              epoch,
            'model_state':        self.model.state_dict(),
            'optimizer_state':    self.optimizer.state_dict(),
            'best_val_loss':      self.best_val_loss,
            'best_val_recon':     self.best_val_recon,
            'no_improve':         self.no_improve,
            'history':            self.history,
            'kl_above_epochs':    self._kl_above_epochs,
            'warmup_started':     self._warmup_started,
            'warmup_start_ep':    self._warmup_start_ep,
            'freeze_done':        (epoch >= self.freeze_decoder_epochs
                                  if self.freeze_decoder_epochs > 0 else True),
        }
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        # Guardar historial en JSON legible
        with open(self.model_dir / self.HISTORY_NAME, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self):
        import torch
        path = self.model_dir / self.CHECKPOINT_NAME
        if not path.exists():
            print("[train] No se encontró checkpoint — entrenando desde cero.")
            return
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.best_val_loss      = state['best_val_loss']
        self.best_val_recon     = state.get('best_val_recon', float('inf'))
        self.no_improve         = state['no_improve']
        self.history            = state['history']
        self.start_epoch        = state['epoch'] + 1
        # Restaurar estado del warmup adaptativo; si no existe en el checkpoint
        # (modelos antiguos), recalcularlo desde el historial de KL.
        if 'warmup_started' in state:
            self._kl_above_epochs = state['kl_above_epochs']
            self._warmup_started  = state['warmup_started']
            self._warmup_start_ep = state['warmup_start_ep']
        else:
            self._recalc_warmup_state()
        # Restaurar estado de congelación del decoder
        freeze_done = state.get('freeze_done', True)
        if self.freeze_decoder_epochs > 0 and not freeze_done:
            self._set_decoder_grad(False)
            print(f'[train] Decoder sigue congelado (reanudando)')
        print(f"[train] Reanudando desde época {self.start_epoch}  "
              f"(mejor val_loss={self.best_val_loss:.4f})")

    def _run_epoch(self, loader, training: bool, beta: float,
                   epoch: int = 0, n_epochs: int = 0):
        import torch
        self.model.train(training)
        total_loss = recon_sum = kl_sum = 0.0
        n_batches  = 0
        phase      = 'train' if training else 'val  '

        # Calcular total de batches para la barra de progreso
        n_total = len(loader.dataset) // max(loader.batch_size, 1) + 1 \
                  if hasattr(loader, 'dataset') and hasattr(loader, 'batch_size') \
                  else None

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                # Mover batch al device (GPU si disponible)
                device = next(self.model.parameters()).device
                context   = batch['context'].to(device, non_blocking=True)
                target    = batch['target'].to(device, non_blocking=True)
                tension   = batch['tension'].to(device, non_blocking=True)
                role_mask = batch['role_mask'].to(device, non_blocking=True)

                recon, mu, logvar = self.model(context, tension)
                # Obtener z_global_dim del modelo si es jerárquico
                raw_m = self.model.module if hasattr(self.model, 'module') else self.model
                z_global_dim = getattr(raw_m, 'z_global_dim', 0)
                loss, r_loss, kl  = vae_loss(recon, target, mu, logvar,
                                             role_mask, beta=beta,
                                             pos_weight=self.pos_weight,
                                             spatial_reg=self.spatial_reg,
                                             lambda_spatial=self.lambda_spatial,
                                             free_bits=self.free_bits,
                                             z_global_dim=z_global_dim)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                recon_sum  += r_loss.item()
                kl_sum     += kl.item()
                n_batches  += 1

                # Progreso intra-época (sobreescribe la misma línea)
                avg = total_loss / n_batches
                if n_total:
                    pct   = n_batches / n_total
                    bar_w = 20
                    filled = int(pct * bar_w)
                    bar   = '█' * filled + '░' * (bar_w - filled)
                    prog  = f"[{bar}] {n_batches}/{n_total}"
                else:
                    prog  = f"batch {n_batches}"

                ep_str = f"época {epoch+1}/{n_epochs}" if n_epochs else ""
                print(f"  {ep_str}  {phase}  {prog}  loss={avg:.4f}  "
                      f"(rec={recon_sum/n_batches:.4f} kl={kl_sum/n_batches:.4f})",
                      end='\r', flush=True)

        # Limpiar la línea de progreso al terminar la fase
        print(' ' * 100, end='\r')

        n = max(n_batches, 1)
        return total_loss / n, recon_sum / n, kl_sum / n

    def train(self, train_loader, val_loader, epochs: int):
        import torch
        import time

        print(f"\n{'═'*60}")
        print(f"  ENTRENAMIENTO  —  {epochs} épocas máx.")
        print(f"  Beta warmup adaptativo: β sube cuando KL>{self.kl_threshold:.1f} "
              f"durante {self.kl_warmup_window} épocas consecutivas")
        print(f"  Duración del warmup lineal: {self.beta_warmup} épocas "
              f"({self.beta_start:.2f} → {self.beta_end:.2f})")
        print(f"  Early stopping: paciencia {self.patience} épocas")
        print(f"  Criterio de mejora: val_rec durante warmup, val_loss después")
        if self.freeze_decoder_epochs > 0:
            print(f"  Decoder congelado: primeras {self.freeze_decoder_epochs} épocas")
        reg_info = (f"{self.spatial_reg}  λ={self.lambda_spatial}"
                    if self.spatial_reg != 'none' else "none")
        print(f"  Regularización espacial: {reg_info}")
        print(f"{'═'*60}\n")

        total_epochs = self.start_epoch + epochs
        epoch_times  = []
        train_start  = time.time()

        # Usar el KL de la última época conocida como punto de partida
        # (relevante solo al reanudar con --resume)
        last_kl = (self.history['val_kl'][-1]
                   if self.history.get('val_kl') else 0.0)

        for epoch in range(self.start_epoch, total_epochs):
            # Congelar / descongelar decoder según época
            if self.freeze_decoder_epochs > 0:
                if epoch == self.start_epoch:
                    # Al arrancar: congelar si todavía estamos en la fase de freeze
                    frozen = epoch < self.freeze_decoder_epochs
                    self._set_decoder_grad(not frozen)
                    if frozen:
                        print(f"  [decoder congelado hasta época {self.freeze_decoder_epochs}]")
                elif epoch == self.freeze_decoder_epochs:
                    # Descongelar el decoder con lr reducido
                    self._set_decoder_grad(True)
                    self._apply_decoder_lr()
                    # Segundo warmup: resetear β y el contador de warmup
                    # para que el decoder aprenda a leer z sin presión KL
                    self._warmup_started  = False
                    self._warmup_start_ep = 0
                    self._kl_above_epochs = 0
                    last_kl = 0.0
                    print(f"\n  [decoder descongelado en época {epoch+1} — "
                          f"β reseteado a 0, segundo warmup activo]")

            # β se calcula ANTES de la época usando el KL de la época anterior.
            # En la época 0 last_kl=0 → β=0 siempre, el encoder aprende libre.
            beta     = self._beta(epoch, current_kl=last_kl)
            epoch_t0 = time.time()

            # Indicador de estado del warmup en la cabecera
            freeze_status = (f"  [decoder frozen {epoch+1}/{self.freeze_decoder_epochs}]"
                             if (self.freeze_decoder_epochs > 0
                                 and epoch < self.freeze_decoder_epochs) else "")
            if not self._warmup_started:
                warmup_status = (f"  [esperando KL>{self.kl_threshold:.1f}  "
                                 f"{self._kl_above_epochs}/{self.kl_warmup_window}]")
            elif beta < self.beta_end:
                ep_since = epoch - self._warmup_start_ep
                warmup_status = f"  [warmup {ep_since}/{self.beta_warmup}]"
            else:
                warmup_status = ""

            # ETA
            if epoch_times:
                avg_epoch   = sum(epoch_times) / len(epoch_times)
                eta_secs    = avg_epoch * (total_epochs - epoch)
                eta_str     = f"  ETA {_fmt_time(eta_secs)}"
            else:
                eta_str = ""

            print(f"  Época {epoch+1:>4}/{total_epochs}  β={beta:.3f}"
                  f"{eta_str}{freeze_status}{warmup_status}", flush=True)

            tr_loss, tr_recon, tr_kl = self._run_epoch(
                train_loader, training=True,  beta=beta,
                epoch=epoch, n_epochs=total_epochs)
            vl_loss, vl_recon, vl_kl = self._run_epoch(
                val_loader,   training=False, beta=beta,
                epoch=epoch, n_epochs=total_epochs)

            last_kl = vl_kl   # se usará en la siguiente iteración

            epoch_elapsed = time.time() - epoch_t0
            epoch_times.append(epoch_elapsed)
            if len(epoch_times) > 5:
                epoch_times.pop(0)

            self.history['train'].append(tr_loss)
            self.history['val'].append(vl_loss)
            self.history['val_recon'].append(vl_recon)
            self.history['val_kl'].append(vl_kl)
            self.history['beta'].append(beta)

            # ── Criterio de mejora dual ───────────────────────────────────────
            # Durante el warmup (β < β_end): vigilar val_recon.
            # Después del warmup: vigilar val_loss total.
            in_warmup  = beta < self.beta_end
            monitor    = vl_recon if in_warmup else vl_loss
            best_sofar = self.best_val_recon if in_warmup else self.best_val_loss
            criterion  = 'rec' if in_warmup else 'loss'

            is_best = monitor < best_sofar
            if is_best:
                self.best_val_recon = vl_recon
                self.best_val_loss  = vl_loss
                self.no_improve     = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, vl_loss, is_best)

            # Barra de progreso basada en val_recon
            ref_recon = self.history['val_recon'][0] if self.history['val_recon'] else vl_recon
            bar_w     = 24
            progress  = min(int((1 - vl_recon / max(ref_recon, 1e-6)) * bar_w), bar_w)
            bar       = '█' * max(progress, 0) + '░' * (bar_w - max(progress, 0))

            best_marker = f' ◀ mejor ({criterion})' if is_best else ''
            stop_marker = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                           if self.no_improve > 0 else '')

            print(f"         train={tr_loss:.4f}  val={vl_loss:.4f} "
                  f"(rec={vl_recon:.4f} kl={vl_kl:.4f})  │{bar}│  "
                  f"{_fmt_time(epoch_elapsed)}/época"
                  f"{best_marker}{stop_marker}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas "
                      f"(criterio: {criterion}).")
                break

        total_elapsed = time.time() - train_start
        print(f"\n{'─'*60}")
        print(f"  Entrenamiento completado en {_fmt_time(total_elapsed)}.")
        print(f"  Mejor val_loss: {self.best_val_loss:.4f}  "
              f"mejor val_rec: {self.best_val_recon:.4f}")
        print(f"  Modelos guardados en: {self.model_dir}")
        print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """
    Carga los .npz de prepare, construye el VAE multi-rol y lo entrena.
    Guarda checkpoints, el mejor modelo y el historial de loss.
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print(f"[train] Cargando datos desde {args.data_dir} ...")
    dataset = MidiRollDataset(args.data_dir, augment=not args.no_augment)
    print(f"[train] {len(dataset)} muestras totales")

    # Split train/val 90/10
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[train] Train: {n_train}  Val: {n_val}")

    # Inferir window_bars y resolution del primer .npz
    first_npz = sorted(Path(args.data_dir).glob('*.npz'))[0]
    first_data = __import__('numpy').load(str(first_npz), allow_pickle=True)
    first_meta = json.loads(str(first_data['meta_json'][0]))
    window_bars = first_meta['window_bars']
    resolution  = first_meta['resolution']
    print(f"[train] window_bars={window_bars}  resolution={resolution}")

    # Usar ~75% de los cores disponibles para carga de datos, mínimo 4
    import multiprocessing
    available_cores = multiprocessing.cpu_count()
    num_workers = min(max(available_cores * 3 // 4, 4), 24)
    print(f"[train] DataLoader workers: {num_workers} (de {available_cores} cores)")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=num_workers,
                              pin_memory=use_pin_memory,
                              persistent_workers=True,
                              prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=max(num_workers // 4, 2),
                              pin_memory=use_pin_memory,
                              persistent_workers=True,
                              prefetch_factor=2)

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    # Configurar PyTorch para usar todos los cores disponibles en CPU
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(n_cpu // 2, 1))

    # Detectar dispositivo: CUDA > CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"[train] GPUs disponibles: {torch.cuda.device_count()}")

    model = _build_model(
        latent_dim  = args.latent_dim,
        style_dim   = args.style_dim,
        tension_dim = TensionExtractor.TENSION_DIM,
        n_roles     = len(ROLES),
        window_bars = window_bars,
        resolution  = resolution,
    )

    # Multi-GPU automático si hay más de 1 GPU
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"[train] Usando DataParallel en {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parámetros del modelo: {n_params:,}")

    # ── 3. Guardar config del modelo ──────────────────────────────────────────
    config = {
        'latent_dim':    args.latent_dim,
        'style_dim':     args.style_dim,
        'tension_dim':   TensionExtractor.TENSION_DIM,
        'n_roles':       len(ROLES),
        'roles':         ROLES,
        'window_bars':   window_bars,
        'resolution':    resolution,
        'z_global_dim':  args.latent_dim // 2,  # guardado para referencia
    }
    with open(model_dir / Trainer.CONFIG_NAME, 'w') as f:
        json.dump(config, f, indent=2)

    # ── 4. Optimizador ────────────────────────────────────────────────────────
    # Acceder a los parámetros del modelo real (desenvuelto si hay DataParallel)
    raw_model = model.module if hasattr(model, 'module') else model

    # Dos param_groups independientes: encoder (lr base) y decoder (lr reducido).
    # El decoder arranca congelado si freeze_decoder_epochs > 0, así que su lr
    # inicial no importa — se ajusta al descongelarse mediante _apply_decoder_lr.
    _DECODER_MODS = Trainer._DECODER_MODULES
    enc_params = [p for n, p in raw_model.named_parameters()
                  if not any(n.startswith(m) for m in _DECODER_MODS)]
    dec_params = [p for n, p in raw_model.named_parameters()
                  if any(n.startswith(m) for m in _DECODER_MODS)]
    optimizer = torch.optim.Adam([
        {'params': enc_params, 'lr': args.lr},
        {'params': dec_params, 'lr': args.lr},   # se reducirá al descongelar
    ])

    # ── 5. Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model              = model,
        optimizer          = optimizer,
        model_dir          = model_dir,
        beta_end           = args.beta,
        beta_warmup_epochs = args.beta_warmup,
        early_stop_patience= args.patience,
        pos_weight         = args.pos_weight,
        spatial_reg        = args.spatial_reg,
        lambda_spatial     = args.lambda_spatial,
        kl_threshold          = args.kl_threshold,
        kl_warmup_window      = args.kl_warmup_window,
        freeze_decoder_epochs = args.freeze_decoder_epochs,
        decoder_lr_factor     = args.decoder_lr_factor,
        free_bits             = args.free_bits,
    )

    if args.resume:
        trainer.load_checkpoint()

    # ── 6. Entrenamiento ──────────────────────────────────────────────────────
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    if args.report:
        _train_report(model_dir)


def _train_report(model_dir: Path):
    """Muestra las curvas de loss en texto tras el entrenamiento."""
    history_path = model_dir / Trainer.HISTORY_NAME
    if not history_path.exists():
        return
    with open(history_path) as f:
        history = json.load(f)

    train_losses = history.get('train', [])
    val_losses   = history.get('val',   [])
    if not train_losses:
        return

    print()
    print("  CURVAS DE LOSS")
    print("─" * 60)

    HEIGHT = 10
    WIDTH  = min(60, len(train_losses))
    step   = max(1, len(train_losses) // WIDTH)

    tr_pts = [train_losses[i*step] for i in range(WIDTH)]
    vl_pts = [val_losses[i*step]   for i in range(WIDTH)]

    all_vals = tr_pts + vl_pts
    lo, hi   = min(all_vals), max(all_vals)
    span     = hi - lo or 1.0

    for row in range(HEIGHT, 0, -1):
        threshold = lo + span * row / HEIGHT
        line = f"  {threshold:6.4f} │"
        for i in range(WIDTH):
            tr = tr_pts[i] >= threshold
            vl = vl_pts[i] >= threshold
            if tr and vl:
                line += '█'
            elif tr:
                line += 'T'
            elif vl:
                line += 'V'
            else:
                line += ' '
        print(line + '│')
    print("         └" + '─' * WIDTH + '┘')
    print(f"          {'épocas':>{WIDTH//2}}    (T=train  V=val  █=ambos)")
    print()
    print(f"  Época final — train: {train_losses[-1]:.4f}  "
          f"val: {val_losses[-1]:.4f}")
    print(f"  Mejor val:   {min(val_losses):.4f}  "
          f"(época {val_losses.index(min(val_losses))+1})")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES COMPARTIDAS: carga de modelo y procesado de MIDI de referencia
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_and_config(model_dir: Path):
    """
    Carga model_config.json + best_model.pt desde model_dir.
    Devuelve (model, config).
    """
    import torch

    cfg_path   = model_dir / Trainer.CONFIG_NAME
    model_path = model_dir / Trainer.BEST_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró {cfg_path}. ¿Has ejecutado train?")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró {model_path}. ¿Has ejecutado train?")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = _build_model(
        latent_dim  = cfg['latent_dim'],
        style_dim   = cfg['style_dim'],
        tension_dim = cfg['tension_dim'],
        n_roles     = cfg['n_roles'],
        window_bars = cfg['window_bars'],
        resolution  = cfg['resolution'],
    )
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state['model_state'])
    model.train(False)
    return model, cfg


def _midi_to_rolls(midi_path: str, cfg: dict) -> dict:
    """
    Convierte un MIDI de referencia a rolls por rol usando la misma
    pipeline que prepare (RoleAssigner + PianoRollConverter).

    Devuelve {role: np.ndarray (n_bars, resolution, 128)}.
    """
    import mido
    import numpy as np

    mid         = mido.MidiFile(midi_path)
    resolution  = cfg['resolution']
    window_bars = cfg['window_bars']

    # Extraer note_lists por stream (misma lógica que cmd_prepare)
    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise ValueError(f"No se encontraron notas en {midi_path}")

    tpb         = mid.ticks_per_beat
    tpbar       = tpb * 4           # asume 4/4
    max_tick    = max(
        (e for nl in note_lists.values() for e in [n[1] for n in nl]),
        default=0
    )
    total_bars  = max(1, int(np.ceil(max_tick / tpbar)))

    role_map    = RoleAssigner().assign(mid)
    conv        = PianoRollConverter(resolution=resolution,
                                     window_bars=window_bars)
    rolls = {}
    for role, stream_key in role_map.items():
        notes = note_lists[stream_key]
        roll  = conv.notes_to_roll(notes, tpb, total_bars)
        rolls[role] = roll   # (total_bars, resolution, 128)

    return rolls


def _rolls_to_windows(rolls: dict, cfg: dict) -> 'np.ndarray':
    """
    Convierte rolls por rol → tensor de contexto para el encoder.
    Devuelve np.ndarray (n_windows, N_ROLES, window_bars-1, resolution, 128).
    """
    import numpy as np

    window_bars = cfg['window_bars']
    resolution  = cfg['resolution']
    n_roles     = cfg['n_roles']
    role_list   = cfg['roles']

    # Calcular n_bars mínimo entre todos los roles
    n_bars = min(r.shape[0] for r in rolls.values()) if rolls else 0
    if n_bars < window_bars:
        raise ValueError(
            f"MIDI demasiado corto: {n_bars} compases, se necesitan ≥ {window_bars}")

    n_windows = n_bars - window_bars + 1
    ctx_bars  = window_bars - 1   # compases de contexto (sin el target)

    windows = np.zeros(
        (n_windows, n_roles, ctx_bars, resolution, 128), dtype=np.float32)

    for widx in range(n_windows):
        for ridx, role in enumerate(role_list):
            if role in rolls:
                windows[widx, ridx] = rolls[role][widx:widx + ctx_bars]

    return windows   # (n_windows, N_ROLES, ctx_bars, res, 128)


def _encode_windows(windows: 'np.ndarray', model, cfg: dict,
                    batch_size: int = 32) -> 'np.ndarray':
    """
    Pasa todas las ventanas por el encoder del VAE.
    Devuelve z_mean promedio: np.ndarray (latent_dim,)
    y también z_style: np.ndarray (style_dim,)
    """
    import torch
    import numpy as np

    all_mu = []
    n      = windows.shape[0]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.tensor(windows[i:i + batch_size])
            mu, _ = model.encode(batch)
            all_mu.append(mu.numpy() if hasattr(mu, 'numpy') else mu._d)

    z_mean    = np.concatenate(all_mu, axis=0).mean(axis=0)   # (latent_dim,)
    with torch.no_grad():
        # Para el VAE jerárquico, style_proj toma solo la parte global de z
        z_global_dim = getattr(model, 'z_global_dim', 0)
        z_for_style  = z_mean[:z_global_dim] if z_global_dim > 0 else z_mean
        z_style_t    = model.style_proj(torch.tensor(z_for_style[None]))  # (1, style_dim)
    z_style   = (z_style_t.detach().numpy() if hasattr(z_style_t, 'numpy')
                 else z_style_t._d)[0]                             # (style_dim,)
    return z_mean, z_style


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: encode
# ══════════════════════════════════════════════════════════════════════════════

def cmd_encode(args):
    """
    MIDI de referencia → z_style.json

    Carga el modelo entrenado, procesa el MIDI con la misma pipeline
    que prepare, pasa todas las ventanas por el encoder y guarda la
    media de z_context + z_style en un JSON.
    """
    model_dir = Path(args.model_dir)
    midi_path = args.input

    print(f"[encode] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    print(f"[encode] Procesando {midi_path} ...")
    rolls   = _midi_to_rolls(midi_path, cfg)
    print(f"[encode] Roles detectados: {list(rolls.keys())}")

    windows = _rolls_to_windows(rolls, cfg)
    print(f"[encode] {windows.shape[0]} ventanas → encoder")

    z_mean, z_style = _encode_windows(windows, model, cfg)

    # Calcular tensión media del MIDI como metadata
    all_rolls_stacked = {}
    for role, roll in rolls.items():
        all_rolls_stacked[role] = roll
    tension_vecs = TensionExtractor().extract_bar_vectors(all_rolls_stacked)
    tension_mean = tension_vecs.mean(axis=0).tolist()

    # Construir payload JSON
    out_path = args.output or (Path(midi_path).stem + '.style.json')
    payload  = {
        'source':       midi_path,
        'model_dir':    str(model_dir),
        'latent_dim':   cfg['latent_dim'],
        'style_dim':    cfg['style_dim'],
        'n_windows':    int(windows.shape[0]),
        'z_context':    z_mean.tolist(),
        'z_style':      z_style.tolist(),
        'tension_mean': tension_mean,
        'roles_found':  list(rolls.keys()),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[encode] Guardado en {out_path}")

    if args.report:
        _encode_report(payload)


def _encode_report(payload: dict):
    """Imprime un resumen visual del estilo codificado."""
    import numpy as np

    print()
    print("  ESTILO CODIFICADO")
    print("─" * 56)
    print(f"  Fuente     : {payload['source']}")
    print(f"  Ventanas   : {payload['n_windows']}")
    print(f"  Roles      : {', '.join(payload['roles_found'])}")

    z   = np.array(payload['z_style'])
    lo, hi, mean = float(z.min()), float(z.max()), float(z.mean())
    std = float(z.std())
    print(f"  z_style    : dim={len(z)}  "
          f"μ={mean:.3f}  σ={std:.3f}  [{lo:.2f}, {hi:.2f}]")

    # Histograma de activaciones de z_style
    bins   = 8
    counts, edges = np.histogram(z, bins=bins, range=(-2, 2))
    max_c  = max(counts) or 1
    bar_w  = 16
    print()
    print("  Distribución de z_style ([-2, 2]):")
    for i in range(bins):
        label  = f"{edges[i]:+.1f}"
        filled = int(counts[i] / max_c * bar_w)
        bar    = '█' * filled + '░' * (bar_w - filled)
        print(f"    {label} │{bar}│ {counts[i]}")

    # Tensión media
    labels = ['tension','density','polyphony','reg_mean',
              'reg_spread','vel_mean','rhythm_dens','arousal']
    tm     = payload['tension_mean']
    print()
    print("  Tensión media del MIDI de referencia:")
    for label, val in zip(labels, tm):
        bar_len = int(val * 20)
        bar     = '█' * bar_len + '░' * (20 - bar_len)
        print(f"    {label:<14} │{bar}│ {val:.3f}")
    print("─" * 56)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR DE TENSIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _build_tension_curve(mode: str, n_bars: int,
                          tension_dim: int = 8) -> 'np.ndarray':
    """
    Genera una matriz (n_bars, tension_dim) según el perfil de tensión.

    mode puede ser:
      'flat'    — tensión constante media (0.5)
      'arch'    — sube hasta el medio y baja (forma de arco)
      'rise'    — sube linealmente de 0.2 a 0.9
      'fall'    — baja linealmente de 0.9 a 0.2
      archivo.json — carga el array directamente del JSON
    """
    import numpy as np

    if mode.endswith('.json'):
        with open(mode) as f:
            data = json.load(f)
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.tile(arr, (n_bars, 1))
        return arr[:n_bars]

    t = np.linspace(0, 1, n_bars)

    if mode == 'flat':
        curve = np.full(n_bars, 0.5)
    elif mode == 'arch':
        curve = np.sin(t * np.pi)           # 0 → 1 → 0
        curve = 0.2 + curve * 0.7           # rango [0.2, 0.9]
    elif mode == 'rise':
        curve = 0.2 + t * 0.7
    elif mode == 'fall':
        curve = 0.9 - t * 0.7
    else:
        raise ValueError(f"Perfil de tensión desconocido: {mode!r}. "
                         f"Usa flat|arch|rise|fall|<archivo.json>")

    # Replicar la curva escalar a todas las dimensiones de tensión,
    # con variación suave por dimensión (evita vectores idénticos)
    offsets = np.linspace(-0.05, 0.05, tension_dim)
    matrix  = np.clip(
        curve[:, None] + offsets[None, :], 0.0, 1.0
    ).astype(np.float32)
    return matrix   # (n_bars, tension_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  MODOS DE COMPOSICIÓN: construcción de z_context por barra
# ══════════════════════════════════════════════════════════════════════════════

def _z_from_style_file(style_path: str, cfg: dict) -> tuple:
    """Carga z_context y z_style desde un .style.json."""
    import numpy as np
    with open(style_path) as f:
        payload = json.load(f)
    z_ctx   = np.array(payload['z_context'],  dtype=np.float32)
    z_style = np.array(payload['z_style'],    dtype=np.float32)
    _check_dims(z_ctx, cfg['latent_dim'], 'z_context')
    _check_dims(z_style, cfg['style_dim'], 'z_style')
    return z_ctx, z_style


def _check_dims(arr, expected, name):
    if arr.shape[0] != expected:
        raise ValueError(
            f"{name}: dimensión {arr.shape[0]} no coincide "
            f"con el modelo ({expected})")


def _mode_reconstruct(args, model, cfg) -> tuple:
    """Usa z_context del MIDI de referencia sin modificar."""
    import numpy as np
    rolls   = _midi_to_rolls(args.input, cfg)
    windows = _rolls_to_windows(rolls, cfg)
    z_ctx, z_style = _encode_windows(windows, model, cfg)
    return z_ctx, z_style


def _mode_z_noise(args, model, cfg) -> tuple:
    """z_context del MIDI de referencia + ruido gaussiano."""
    import numpy as np
    z_ctx, z_style = _mode_reconstruct(args, model, cfg)
    noise   = np.random.randn(cfg['latent_dim']).astype(np.float32)
    z_ctx   = z_ctx + args.noise * noise
    return z_ctx, z_style


def _mode_blend(args, model, cfg) -> tuple:
    """
    Interpolación ponderada de z_context entre varios MIDIs.
    Si se pasan --weights, se normalizan a suma 1.
    Si no, se usa media uniforme.
    """
    import numpy as np

    sources  = args.inputs
    n        = len(sources)
    weights  = args.weights
    if weights is None:
        weights = [1.0 / n] * n
    else:
        s = sum(weights)
        weights = [w / s for w in weights]
        if len(weights) != n:
            raise ValueError("--weights debe tener el mismo número de valores que --inputs")

    z_ctxs   = []
    z_styles = []
    for src in sources:
        rolls   = _midi_to_rolls(src, cfg)
        windows = _rolls_to_windows(rolls, cfg)
        zc, zs  = _encode_windows(windows, model, cfg)
        z_ctxs.append(zc)
        z_styles.append(zs)

    z_ctx   = sum(w * z for w, z in zip(weights, z_ctxs))
    z_style = sum(w * z for w, z in zip(weights, z_styles))
    return z_ctx, z_style


def _mode_sweep(args, model, cfg, n_bars: int) -> 'np.ndarray':
    """
    Interpolación suave entre dos MIDIs a lo largo de n_bars.
    Devuelve array (n_bars, latent_dim) con z_context variando gradualmente.
    Usa scipy.interpolate.CubicSpline si --smooth > 0, lineal si = 0.
    """
    import numpy as np

    sources = args.inputs
    if len(sources) < 2:
        raise ValueError("[sweep] Se necesitan al menos 2 --inputs")

    # Codificar todos los puntos de ancla
    anchors_ctx = []
    for src in sources:
        rolls   = _midi_to_rolls(src, cfg)
        windows = _rolls_to_windows(rolls, cfg)
        zc, _   = _encode_windows(windows, model, cfg)
        anchors_ctx.append(zc)

    anchors = np.stack(anchors_ctx, axis=0)   # (n_anchors, latent_dim)
    n_anch  = len(anchors)
    t_anch  = np.linspace(0, 1, n_anch)
    t_bars  = np.linspace(0, 1, n_bars)

    if args.smooth > 0:
        try:
            from scipy.interpolate import CubicSpline
            cs   = CubicSpline(t_anch, anchors, axis=0)
            path = cs(t_bars).astype(np.float32)
        except ImportError:
            print("[sweep] scipy no disponible, usando interpolación lineal")
            path = np.array(
                [np.interp(t_bars, t_anch, anchors[:, d])
                 for d in range(anchors.shape[1])],
                dtype=np.float32
            ).T
    else:
        path = np.array(
            [np.interp(t_bars, t_anch, anchors[:, d])
             for d in range(anchors.shape[1])],
            dtype=np.float32
        ).T

    return path   # (n_bars, latent_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERER: rolls → MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _rolls_to_midi(bars_per_role: dict, cfg: dict,
                   palette: dict, output_path: str,
                   bpm: float = 120.0):
    """
    Convierte un dict {role: np.ndarray (n_bars, resolution, 128)}
    en un archivo MIDI multi-pista.

    palette: {role: {'program': int, 'channel': int, 'velocity': int}}
    bpm    : tempo en BPM
    """
    import mido
    import numpy as np

    resolution  = cfg['resolution']
    tpb         = 480                           # ticks por beat (estándar)
    ticks_bar   = tpb * 4                       # 4/4
    ticks_tick  = ticks_bar / resolution        # ticks MIDI por tick interno

    mid  = mido.MidiFile(ticks_per_beat=tpb)
    tempo_val = int(60_000_000 / bpm)

    # Pista 0: tempo
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=tempo_val, time=0))
    mid.tracks.append(t0)

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]              # (n_bars, resolution, 128)
        pal  = palette.get(role, {})
        prog = int(pal.get('program', 0))
        ch   = int(pal.get('channel', 0))
        vel  = int(pal.get('velocity', 80))

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change',
                                  program=prog, channel=ch, time=0))

        # Convertir roll binario → eventos note_on / note_off
        events = []   # lista de (abs_tick_midi, type, pitch)

        n_bars_r, res_r, _ = roll.shape
        for bar in range(n_bars_r):
            for tick in range(res_r):
                abs_tick = int((bar * res_r + tick) * ticks_tick)
                for pitch in range(128):
                    cur  = roll[bar, tick, pitch] > 0.5
                    prev = roll[bar, tick - 1, pitch] > 0.5 if tick > 0 \
                           else (roll[bar - 1, -1, pitch] > 0.5 if bar > 0
                                 else False)
                    if cur and not prev:
                        events.append((abs_tick, 'on',  pitch))
                    elif not cur and prev:
                        events.append((abs_tick, 'off', pitch))

        # Asegurar note_off al final para notas que terminan en el último tick
        last_tick = int(n_bars_r * res_r * ticks_tick)
        for pitch in range(128):
            if roll[-1, -1, pitch] > 0.5:
                events.append((last_tick, 'off', pitch))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

        prev_tick = 0
        for abs_tick, etype, pitch in events:
            delta = abs_tick - prev_tick
            if etype == 'on':
                track.append(mido.Message(
                    'note_on', channel=ch,
                    note=pitch, velocity=vel, time=delta))
            else:
                track.append(mido.Message(
                    'note_off', channel=ch,
                    note=pitch, velocity=0, time=delta))
            prev_tick = abs_tick

    mid.save(output_path)


def _load_palette(palette_path: str, cfg: dict) -> dict:
    """
    Carga la paleta de instrumentos desde JSON.
    Si falta algún rol, usa defaults razonables.
    """
    DEFAULT_PALETTE = {
        'melody':        {'program': 73, 'channel': 0, 'velocity': 90},  # flauta
        'counterpoint':  {'program': 68, 'channel': 1, 'velocity': 80},  # oboe
        'accompaniment': {'program': 48, 'channel': 2, 'velocity': 70},  # strings
        'bass':          {'program': 43, 'channel': 3, 'velocity': 85},  # contrabass
        'percussion':    {'program':  0, 'channel': 9, 'velocity': 90},  # canal 9
    }
    with open(palette_path) as f:
        user = json.load(f)
    palette = {**DEFAULT_PALETTE}
    for role, params in user.items():
        if role in palette:
            palette[role].update(params)
    return palette


# ══════════════════════════════════════════════════════════════════════════════
#  NÚCLEO DE GENERACIÓN: z → bars por rol
# ══════════════════════════════════════════════════════════════════════════════

def _generate_bars(z_ctx_seq, z_style, tension_curve,
                   model, cfg, temperature: float = 1.0) -> dict:
    """
    Genera n_bars compases usando el decoder del VAE.

    z_ctx_seq : np.ndarray (n_bars, latent_dim)  ó  (latent_dim,) → broadcast
    z_style   : np.ndarray (style_dim,)
    tension_curve: np.ndarray (n_bars, tension_dim)
    temperature  : escala del muestreo Bernoulli (>1 = más aleatorio)

    Devuelve  {role: np.ndarray (n_bars, resolution, 128)} binario.
    """
    import torch
    import numpy as np

    n_bars      = tension_curve.shape[0]
    latent_dim  = cfg['latent_dim']
    style_dim   = cfg['style_dim']
    resolution  = cfg['resolution']
    n_roles     = cfg['n_roles']
    role_list   = cfg['roles']

    # Si z_ctx_seq es 1-D lo repetimos para cada barra
    if z_ctx_seq.ndim == 1:
        z_ctx_seq = np.tile(z_ctx_seq[None], (n_bars, 1))

    bars_per_role = {role: np.zeros((n_bars, resolution, 128), dtype=np.float32)
                     for role in role_list}

    with torch.no_grad():
        for bar_i in range(n_bars):
            z   = torch.tensor(z_ctx_seq[bar_i][None])          # (1, latent_dim)
            zs  = torch.tensor(z_style[None])                   # (1, style_dim)
            vt  = torch.tensor(tension_curve[bar_i][None])      # (1, tension_dim)

            recon = model.decode(z, zs, vt)                     # (1, N_ROLES, res, 128)
            probs = (recon.numpy() if hasattr(recon, 'numpy')
                     else recon._d)[0]                           # (N_ROLES, res, 128)

            # Muestreo Bernoulli con temperatura.
            #
            # PROBLEMA con potencia directa: probs^(1/T) con T<1 colapsa
            # probabilidades bajas a 0. Por ejemplo probs=0.3, T=0.15:
            #   0.3^(1/0.15) = 0.3^6.67 ≈ 0.0003 → MIDI vacío.
            #
            # Solución: aplicar temperatura en espacio logit (log-odds),
            # que es el espacio natural de las probabilidades Bernoulli.
            # T < 1 agudiza la distribución (más determinista).
            # T > 1 la aplana (más aleatorio).
            # T = 1 no cambia nada.
            eps       = 1e-6
            logits    = np.log(np.clip(probs, eps, 1 - eps) /
                               np.clip(1 - probs, eps, 1 - eps))  # log-odds
            adj_logits = logits / temperature                      # escalar logits
            adj_probs  = 1.0 / (1.0 + np.exp(-adj_logits))        # sigmoid inverso
            adj_probs  = np.clip(adj_probs, 0, 1)
            samples    = (np.random.rand(*adj_probs.shape) < adj_probs).astype(np.float32)

            for ridx, role in enumerate(role_list):
                bars_per_role[role][bar_i] = samples[ridx]

    return bars_per_role


# ══════════════════════════════════════════════════════════════════════════════
#  ÍNDICE LATENTE (NNR): encode del corpus completo
# ══════════════════════════════════════════════════════════════════════════════

def _build_latent_index(data_dir: str, model, cfg: dict,
                        batch_size: int = 64) -> tuple:
    """
    Codifica todos los compases target del corpus y construye un índice de
    búsqueda por vecino más cercano en el espacio latente.

    Para cada .npz y cada ventana, el compás target es el último de la ventana
    (índice window_bars-1). El contexto son los window_bars-1 compases
    anteriores, que se pasan por el encoder para obtener z (usando mu, sin
    muestreo estocástico).

    Devuelve
    --------
    z_index      : np.ndarray (N, latent_dim)
                   Vector latente de cada compás del corpus.
    bar_index    : dict {role: np.ndarray (N, resolution, 128)}
                   Piano roll binario del compás target por rol.
                   Roles ausentes en una muestra contienen ceros.
    tension_index: np.ndarray (N, tension_dim)
                   Vector de tensión del compás target.
    """
    import numpy as np
    import torch

    window_bars = cfg['window_bars']
    resolution  = cfg['resolution']
    n_roles     = cfg['n_roles']
    role_list   = cfg['roles']
    latent_dim  = cfg['latent_dim']
    tension_dim = cfg['tension_dim']

    npz_files = sorted(Path(data_dir).glob('*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No se encontraron .npz en {data_dir}")

    # Acumuladores
    all_contexts  = []   # (N, n_roles, ctx_bars, resolution, 128)
    all_bars      = {role: [] for role in role_list}
    all_tensions  = []
    ctx_bars      = window_bars - 1

    print(f"[nnr] Cargando corpus desde {data_dir} ...")

    for path in npz_files:
        with np.load(str(path), allow_pickle=True) as raw:
            data = {k: raw[k] for k in raw.files}

        meta         = json.loads(str(data['meta_json'][0]))
        n_windows    = meta['n_windows']
        roles_present = [r for r in role_list if f'roll_{r}' in data]

        if not roles_present:
            continue

        for widx in range(n_windows):
            # Construir contexto (N_ROLES, ctx_bars, res, 128)
            ctx = np.zeros((n_roles, ctx_bars, resolution, 128), dtype=np.float32)
            # Compás target (N_ROLES, res, 128)
            tgt = np.zeros((n_roles, resolution, 128), dtype=np.float32)

            for ridx, role in enumerate(role_list):
                key = f'roll_{role}'
                if key in data:
                    win = data[key][widx]        # (window_bars, res, 128)
                    ctx[ridx] = win[:-1]         # primeros ctx_bars
                    tgt[ridx] = win[-1]          # último compás = target

            all_contexts.append(ctx)

            for ridx, role in enumerate(role_list):
                all_bars[role].append(tgt[ridx])

            if 'tension' in data:
                all_tensions.append(data['tension'][widx])
            else:
                all_tensions.append(np.zeros(tension_dim, dtype=np.float32))

    N = len(all_contexts)
    if N == 0:
        raise ValueError("[nnr] El corpus no produjo ninguna muestra válida.")

    print(f"[nnr] {N} compases indexados — codificando con el encoder ...")

    # Codificar en batches
    contexts_arr = np.stack(all_contexts, axis=0)   # (N, n_roles, ctx_bars, res, 128)
    z_index      = np.zeros((N, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.tensor(contexts_arr[i:i + batch_size])
            mu, _ = model.encode(batch)
            z_np  = mu.numpy() if hasattr(mu, 'numpy') else mu._d
            z_index[i:i + len(z_np)] = z_np
            pct = min(i + batch_size, N)
            print(f"  [{pct}/{N}]", end='\r', flush=True)

    print(' ' * 40, end='\r')

    bar_index     = {role: np.stack(all_bars[role], axis=0)
                     for role in role_list}   # cada (N, res, 128)
    tension_index = np.stack(all_tensions, axis=0)  # (N, tension_dim)

    print(f"[nnr] Índice construido: {N} compases × {latent_dim} dims latentes")
    return z_index, bar_index, tension_index


def _generate_bars_nnr(z_ctx_seq, tension_curve,
                       z_index, bar_index, tension_index,
                       cfg, distance: str = 'euclidean',
                       tension_weight: float = 0.0,
                       temperature: float = 0.0) -> dict:
    """
    Genera n_bars compases por recuperación ponderada en el espacio latente.

    Para cada compás a generar calcula la distancia entre el z de contexto y
    todos los z del índice, y muestrea un candidato con probabilidad
    proporcional a exp(-dist / T). Con T=0 equivale a argmin (vecino exacto);
    con T alto la distribucion se aplana y hay mas variedad.

    Parametros
    ----------
    z_ctx_seq      : np.ndarray (n_bars, latent_dim) o (latent_dim,)
    tension_curve  : np.ndarray (n_bars, tension_dim)
    z_index        : np.ndarray (N, latent_dim)   -- indice pre-codificado
    bar_index      : dict {role: np.ndarray (N, res, 128)}
    tension_index  : np.ndarray (N, tension_dim)  -- tension de cada compas
    distance       : 'euclidean' | 'cosine'
    tension_weight : peso del termino de tension en la distancia compuesta
                     (0.0 = solo distancia latente; 1.0 = pesos iguales)
    temperature    : T del muestreo softmax inverso sobre distancias.
                     0.0 = determinista (vecino mas cercano exacto).
                     Valores tipicos: 0.1 (conservador) ... 1.0 (muy variado).

    Devuelve
    --------
    dict {role: np.ndarray (n_bars, resolution, 128)} -- binario (0/1)
    """
    import numpy as np

    n_bars     = tension_curve.shape[0]
    role_list  = cfg['roles']
    resolution = cfg['resolution']

    if z_ctx_seq.ndim == 1:
        z_ctx_seq = np.tile(z_ctx_seq[None], (n_bars, 1))

    # Normalizar z_index una vez si se usa coseno
    if distance == 'cosine':
        norms   = np.linalg.norm(z_index, axis=1, keepdims=True)
        z_idx_n = z_index / np.clip(norms, 1e-8, None)
    else:
        z_idx_n = None

    bars_per_role = {role: np.zeros((n_bars, resolution, 128), dtype=np.float32)
                     for role in role_list}

    for bar_i in range(n_bars):
        z_q = z_ctx_seq[bar_i]   # (latent_dim,)

        # ── Distancia latente ─────────────────────────────────────────────
        if distance == 'cosine':
            nq    = z_q / max(float(np.linalg.norm(z_q)), 1e-8)
            dists = 1.0 - z_idx_n @ nq           # (N,)  cosine distance
        else:
            diff  = z_index - z_q                # (N, latent_dim)
            dists = np.linalg.norm(diff, axis=1)  # (N,)

        # ── Termino de tension (opcional) ─────────────────────────────────
        if tension_weight > 0.0:
            t_q     = tension_curve[bar_i]                         # (tension_dim,)
            t_diff  = tension_index - t_q                          # (N, tension_dim)
            t_dists = np.linalg.norm(t_diff, axis=1)              # (N,)
            # Normalizar ambos terminos a [0, 1] antes de combinar
            d_max   = dists.max()   or 1.0
            t_max   = t_dists.max() or 1.0
            dists   = ((1.0 - tension_weight) * dists   / d_max +
                        tension_weight         * t_dists / t_max)

        # ── Seleccion: determinista o muestreo ponderado ──────────────────
        if temperature <= 0.0:
            # Vecino mas cercano exacto
            selected = int(np.argmin(dists))
        else:
            # Softmax inverso: prob proporcional a exp(-dist / T)
            # Restar el minimo antes de exponenciar para estabilidad numerica
            logits = -(dists - dists.min()) / temperature
            probs  = np.exp(logits)
            probs /= probs.sum()
            selected = int(np.random.choice(len(dists), p=probs))

        for role in role_list:
            bars_per_role[role][bar_i] = bar_index[role][selected]

    return bars_per_role


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: compose
# ══════════════════════════════════════════════════════════════════════════════

def cmd_compose(args):
    """
    Genera una obra nueva en función del modo seleccionado:

      reconstruct  — Decodifica el z_context exacto de un MIDI de referencia
      z-noise      — z_context del MIDI + ruido gaussiano controlado
      blend        — Interpolación ponderada entre varios MIDIs
      sweep        — Barrido suave entre varios MIDIs a lo largo de la obra
    """
    import numpy as np

    model_dir = Path(args.model_dir)
    print(f"[compose] Cargando modelo desde {model_dir} ...")
    model, cfg = _load_model_and_config(model_dir)

    n_bars       = args.bars
    tension_curve = _build_tension_curve(args.tension, n_bars, cfg['tension_dim'])
    print(f"[compose] Perfil de tensión: {args.tension}  ({n_bars} compases)")

    # ── Obtener z_context y z_style según el modo ─────────────────────────
    mode = args.mode
    print(f"[compose] Modo: {mode}")

    if mode == 'reconstruct':
        if not args.input:
            raise ValueError("[reconstruct] Requiere --input <midi>")
        z_ctx, z_style = _mode_reconstruct(args, model, cfg)
        z_ctx_seq = z_ctx   # (latent_dim,) → se broadcast en _generate_bars

    elif mode == 'z-noise':
        if not args.input:
            raise ValueError("[z-noise] Requiere --input <midi>")
        z_ctx, z_style = _mode_z_noise(args, model, cfg)
        z_ctx_seq = z_ctx

    elif mode == 'blend':
        if not args.inputs or len(args.inputs) < 2:
            raise ValueError("[blend] Requiere al menos 2 --inputs")
        z_ctx, z_style = _mode_blend(args, model, cfg)
        z_ctx_seq = z_ctx

    elif mode == 'sweep':
        if not args.inputs or len(args.inputs) < 2:
            raise ValueError("[sweep] Requiere al menos 2 --inputs")
        # sweep devuelve (n_bars, latent_dim) directamente
        z_ctx_seq = _mode_sweep(args, model, cfg, n_bars)
        # z_style: media de todos los inputs
        z_styles = []
        for src in args.inputs:
            rolls   = _midi_to_rolls(src, cfg)
            windows = _rolls_to_windows(rolls, cfg)
            _, zs   = _encode_windows(windows, model, cfg)
            z_styles.append(zs)
        z_style = np.mean(z_styles, axis=0)

    else:
        raise ValueError(f"Modo desconocido: {mode}")

    # ── Generar compases ──────────────────────────────────────────────────
    retrieval = getattr(args, 'retrieval', 'decoder')

    if retrieval == 'nnr':
        # Modo recuperación: buscar el vecino más cercano en el corpus
        if not getattr(args, 'data_dir', None):
            raise ValueError(
                "[nnr] --retrieval nnr requiere --data-dir con los .npz del corpus")
        distance       = getattr(args, 'nnr_distance', 'euclidean')
        tension_weight = getattr(args, 'nnr_tension_weight', 0.0)

        z_index, bar_index, tension_index = _build_latent_index(
            args.data_dir, model, cfg)

        nnr_temperature = getattr(args, 'nnr_temperature', 0.0)
        print(f"[compose] Generando {n_bars} compases  "
              f"[NNR · distancia={distance} · T={nnr_temperature} · tension_w={tension_weight}] ...")
        bars_per_role = _generate_bars_nnr(
            z_ctx_seq, tension_curve,
            z_index, bar_index, tension_index,
            cfg,
            distance       = distance,
            tension_weight = tension_weight,
            temperature    = nnr_temperature,
        )
    else:
        # Modo clásico: decodificar z con el decoder del VAE
        print(f"[compose] Generando {n_bars} compases (temperatura={args.temperature}) ...")
        bars_per_role = _generate_bars(
            z_ctx_seq, z_style, tension_curve,
            model, cfg, temperature=args.temperature
        )

    # ── Cargar paleta y renderizar MIDI ──────────────────────────────────
    palette = _load_palette(args.palette, cfg)
    bpm     = getattr(args, 'bpm', 120.0)

    print(f"[compose] Renderizando MIDI → {args.output} ...")
    _rolls_to_midi(bars_per_role, cfg, palette, args.output, bpm=bpm)

    # Estadísticas rápidas del output
    total_notes = 0
    active_roles = []
    for role, roll in bars_per_role.items():
        n = int((roll.max(axis=(1,2)) > 0).sum())
        if n > 0:
            active_roles.append(role)
            total_notes += int(roll.sum())

    print(f"[compose] ✓  Roles activos: {', '.join(active_roles)}")
    print(f"[compose] ✓  Notas totales aprox.: {total_notes}")
    print(f"[compose] ✓  Guardado en {args.output}")


def cmd_inspect(args):
    """
    Dispatcher de subcomandos de inspección.
    Actualmente implementado: npz (visualización de datos preparados).
    El resto (loss_curve, latent_projection, etc.) se añadirán con train/encode.
    """
    what = set(args.what)

    if 'npz' in what:
        if not args.data_dir:
            print("[inspect npz] Requiere --data-dir con los archivos .npz")
            sys.exit(1)
        _inspect_npz(Path(args.data_dir), args)
        return

    # Subcomandos que necesitan modelo entrenado (stubs por ahora)
    for w in ('loss_curve', 'role_quality', 'latent_projection', 'tension_response'):
        if w in what:
            print(f"[inspect {w}] No implementado aún (disponible tras train).")


# ══════════════════════════════════════════════════════════════════════════════
#  INSPECT: VISUALIZACIÓN DE ARCHIVOS .npz
# ══════════════════════════════════════════════════════════════════════════════

_DENSITY_CHARS = ' ░▒▓█'


def _density_char(value: float, n_ticks: int, tick_width: int) -> str:
    frac = value / max(n_ticks * tick_width, 1)
    frac = min(frac, 1.0)
    idx  = int(frac * (len(_DENSITY_CHARS) - 1))
    return _DENSITY_CHARS[idx]


def _render_pianoroll_bar(bar_roll, pitch_lo: int = 24, pitch_hi: int = 108,
                          width: int = 56) -> list:
    """
    Renderiza un compás (resolution × 128) como lista de líneas de texto.
    Eje vertical: pitch (agudo arriba, grave abajo).
    Eje horizontal: tiempo comprimido a `width` columnas.
    """
    import numpy as np
    resolution = bar_roll.shape[0]
    tick_width = max(1, resolution // width)
    n_cols     = resolution // tick_width

    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    lines = []
    prev_octave = None
    for pitch in range(min(pitch_hi - 1, 127), max(pitch_lo - 1, -1), -1):
        octave    = pitch // 12
        semitone  = pitch % 12
        note_name = NOTE_NAMES[semitone]
        is_black  = '#' in note_name

        if prev_octave is not None and octave != prev_octave:
            lines.append(f"       │{'─' * n_cols}│")
        prev_octave = octave

        row      = bar_roll[:, pitch]
        col_vals = [row[c * tick_width:(c + 1) * tick_width].sum()
                    for c in range(n_cols)]
        rendered = ''.join(_density_char(v, 1, tick_width) for v in col_vals)
        prefix   = '·' if is_black else ' '
        lines.append(f"  {prefix}{note_name:<2}{octave} │{rendered}│")

    return lines


def _render_tension_bar(tension_vec, width: int = 48) -> str:
    """Vector de tensión de un compás como minigráfico de barras en una línea."""
    labels  = ['ten', 'dens', 'poly', 'reg↕', 'spr', 'vel', 'rhy', 'aro']
    seg_w   = max(3, width // len(labels))
    parts   = []
    for i, label in enumerate(labels):
        val     = float(tension_vec[i])
        filled  = int(val * seg_w)
        bar     = '█' * filled + '░' * (seg_w - filled)
        parts.append(f"{label}:{bar}{val:.2f}")
    return '  '.join(parts)


def _render_tension_curve(tension, dim: int, label: str, width: int = 60) -> list:
    """Sparkline vertical de una dimensión de tensión a lo largo de las ventanas."""
    import numpy as np
    n      = tension.shape[0]
    vals   = tension[:, dim]
    lo, hi = float(vals.min()), float(vals.max())
    span   = hi - lo or 1.0
    HEIGHT = 5

    step   = max(1, n // width)
    points = [float(vals[i * step: (i + 1) * step].mean())
              for i in range(min(width, (n + step - 1) // step))]
    w      = len(points)

    lines = [f"  {label}  min={lo:.2f} max={hi:.2f}"]
    for row in range(HEIGHT, 0, -1):
        threshold = lo + span * row / HEIGHT
        line = '  │'
        for v in points:
            line += '█' if v >= threshold else ' '
        lines.append(line + '│')
    lines.append('  └' + '─' * w + '┘')
    return lines


def _inspect_npz(data_dir: Path, args):
    """
    Visualiza en modo texto los .npz generados por `prepare`.
    """
    import numpy as np

    npz_files = sorted(data_dir.glob('*.npz'))
    if not npz_files:
        print(f"[inspect npz] No hay archivos .npz en {data_dir}")
        return

    if args.npz_file:
        npz_files = [f for f in npz_files
                     if f.stem == args.npz_file or f.name == args.npz_file]
        if not npz_files:
            print(f"[inspect npz] No se encontró '{args.npz_file}' en {data_dir}")
            return

    roles_filter = set(args.roles) if args.roles else set(ROLES)
    window_idx   = args.window
    show_roll    = not args.no_roll
    show_tcurve  = args.tension_curve

    TENSION_LABELS = ['tension','density','polyphony','reg_mean',
                      'reg_spread','vel_mean','rhythm_dens','arousal']

    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        meta = json.loads(str(data['meta_json'][0]))

        # ── cabecera ──────────────────────────────────────────────────────
        W = 72
        print()
        print("╔" + "═" * W + "╗")
        print(f"║  {npz_path.name:<{W-2}}║")
        print("╠" + "═" * W + "╣")
        print(f"║  Compases: {meta['total_bars']:<5} │ "
              f"Ventanas: {meta['n_windows']:<5} │ "
              f"Resolución: {meta['resolution']} ticks/compás"
              f"{'':<{W - 54}}║")
        roles_str = ', '.join(meta['roles'])
        print(f"║  Roles: {roles_str:<{W-9}}║")
        print("╚" + "═" * W + "╝")

        # ── inventario de arrays ──────────────────────────────────────────
        print()
        print("  Arrays:")
        for key in sorted(data.files):
            arr = data[key]
            is_numeric = (hasattr(arr, 'dtype') and
                          arr.dtype.kind in ('f', 'i', 'u'))  # float, int, uint
            if is_numeric:
                nnz = int((arr > 0).sum()) if arr.size > 0 else 0
                pct = f"  ({100*nnz/arr.size:.1f}% activo)" if arr.size > 0 else ''
                print(f"    {key:<24} {str(arr.shape):<28} {arr.dtype}{pct}")
            else:
                print(f"    {key:<24} (texto/meta)")

        # ── estadísticas de tensión del fichero completo ──────────────────
        if 'tension' in data:
            tension = data['tension']   # (n_windows, 8)
            print()
            print("  Estadísticas de tensión (media ± std  /  min–max):")
            for dim, label in enumerate(TENSION_LABELS):
                col  = tension[:, dim]
                mean = float(col.mean())
                std  = float(col.std())
                lo   = float(col.min())
                hi   = float(col.max())
                bar_len = int(mean * 30)
                bar  = '█' * bar_len + '░' * (30 - bar_len)
                print(f"    {label:<14} {bar}  {mean:.3f}±{std:.3f}  [{lo:.2f}–{hi:.2f}]")

        # ── curvas de tensión a lo largo de las ventanas ──────────────────
        if show_tcurve and 'tension' in data:
            tension = data['tension']
            print()
            print("  Curvas de tensión (evolución ventana a ventana):")
            print()
            for dim, label in enumerate(TENSION_LABELS):
                for line in _render_tension_curve(tension, dim, f"{label:<12}"):
                    print(line)
                print()

        # ── ventana seleccionada ──────────────────────────────────────────
        n_windows = meta['n_windows']
        widx      = window_idx % n_windows

        print()
        print(f"  ── VENTANA {widx}  (de 0 a {n_windows-1}) " + "─" * 44)

        if 'tension' in data:
            tension = data['tension']
            if widx < tension.shape[0]:
                print()
                print("  Tensión del compás central de esta ventana:")
                print("  " + _render_tension_bar(tension[widx]))
                print()

        if show_roll:
            n_bars_show = min(args.bars_show, meta['window_bars'])
            for role in ROLES:
                if role not in roles_filter:
                    continue
                key = f'roll_{role}'
                if key not in data.files:
                    continue
                roll   = data[key]          # (n_windows, window_bars, res, 128)
                if widx >= roll.shape[0]:
                    continue
                window = roll[widx]          # (window_bars, res, 128)

                print(f"  ┌─ ROL: {role.upper()}")

                for bar_i in range(n_bars_show):
                    bar_roll = window[bar_i]
                    active_p = np.where(bar_roll.max(axis=0) > 0)[0]
                    n_notes  = len(active_p)

                    print(f"  │  Compás {bar_i+1}/{meta['window_bars']}  "
                          f"({n_notes} pitch{'es' if n_notes!=1 else ''} activos)")

                    if n_notes == 0:
                        print("  │  (silencio)")
                    else:
                        p_lo = (int(active_p.min()) // 12) * 12
                        p_hi = ((int(active_p.max()) // 12) + 1) * 12
                        p_lo = max(0, p_lo)
                        p_hi = min(128, p_hi)
                        for line in _render_pianoroll_bar(
                                bar_roll, pitch_lo=p_lo, pitch_hi=p_hi):
                            print("  │" + line)
                    print("  │")
                print("  └" + "─" * 68)
                print()


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='latent_composer',
        description='Composición end-to-end mediante VAE multi-rol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p_prep = sub.add_parser('prepare',
        help='Convierte corpus MIDI en piano rolls segmentados por rol')
    p_prep.add_argument('--input-dir',   required=True,
        metavar='DIR', help='Directorio con archivos .mid / .midi')
    p_prep.add_argument('--output-dir',  required=True,
        metavar='DIR', help='Directorio de salida para archivos .npz')
    p_prep.add_argument('--resolution',  type=int,
        default=TICKS_PER_BAR_DEFAULT,
        metavar='INT', help=f'Ticks internos por compás (default: {TICKS_PER_BAR_DEFAULT})')
    p_prep.add_argument('--window-bars', type=int,
        default=WINDOW_BARS_DEFAULT,
        metavar='INT', help=f'Compases por ventana (default: {WINDOW_BARS_DEFAULT})')
    p_prep.add_argument('--report',      action='store_true',
        help='Muestra estadísticas detalladas del corpus procesado')
    p_prep.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Entrena el VAE multi-rol')
    p_train.add_argument('--data-dir',    required=True, metavar='DIR',
        help='Directorio con .npz de prepare')
    p_train.add_argument('--model-dir',   required=True, metavar='DIR',
        help='Directorio de salida para checkpoints y modelo final')
    p_train.add_argument('--epochs',      type=int, default=100,
        help='Épocas máximas (default: 100)')
    p_train.add_argument('--batch-size',  type=int, default=16, metavar='INT',
        help='Tamaño de batch (default: 16)')
    p_train.add_argument('--latent-dim',  type=int, default=128, metavar='INT',
        help='Dimensión del espacio latente z (default: 128)')
    p_train.add_argument('--style-dim',   type=int, default=64, metavar='INT',
        help='Dimensión del vector de estilo z_style (default: 64)')
    p_train.add_argument('--lr',          type=float, default=1e-3,
        help='Learning rate Adam (default: 1e-3)')
    p_train.add_argument('--beta',        type=float, default=1.0,
        help='Peso final del término KL — beta-VAE (default: 1.0)')
    p_train.add_argument('--beta-warmup', type=int, default=20, metavar='INT',
        dest='beta_warmup',
        help='Épocas de warm-up lineal de beta, 0→beta (default: 20)')
    p_train.add_argument('--patience',    type=int, default=15, metavar='INT',
        help='Early stopping: épocas sin mejora antes de parar (default: 15)')
    p_train.add_argument('--pos-weight',  type=float, default=10.0, metavar='W',
        dest='pos_weight',
        help='Peso de píxeles positivos en BCE (default: 10.0)')
    p_train.add_argument('--no-augment',  action='store_true',
        dest='no_augment',
        help='Desactivar augmentación por transposición')
    p_train.add_argument('--spatial-reg', default='none',
        dest='spatial_reg',
        choices=SPATIAL_REG_MODES,
        help='Regularización espacial del piano roll: '
             'none | smoothness | pitch | interval (default: none)')
    p_train.add_argument('--lambda-spatial', type=float, default=0.05,
        dest='lambda_spatial', metavar='FLOAT',
        help='Peso del término de regularización espacial (default: 0.05)')
    p_train.add_argument('--decoder-lr-factor', type=float, default=0.1,
        dest='decoder_lr_factor', metavar='FLOAT',
        help='Factor de lr del decoder al descongelarse: lr_dec = lr * factor '
             '(default: 0.1 — decoder aprende 10x más lento que encoder)')
    p_train.add_argument('--freeze-decoder-epochs', type=int, default=0,
        dest='freeze_decoder_epochs', metavar='INT',
        help='Congelar el decoder durante N épocas iniciales para forzar '
             'al encoder a usar z (default: 0 = desactivado)')
    p_train.add_argument('--kl-threshold',    type=float, default=5.0,
        dest='kl_threshold', metavar='FLOAT',
        help='KL mínimo para arrancar el warmup de beta (default: 5.0)')
    p_train.add_argument('--kl-warmup-window', type=int, default=3,
        dest='kl_warmup_window', metavar='INT',
        help='Épocas consecutivas con KL>threshold para disparar warmup (default: 3)')
    p_train.add_argument('--free-bits',      type=float, default=0.1,
        dest='free_bits', metavar='FLOAT',
        help='Bits libres por dimensión latente para evitar colapso (default: 0.1)')
    p_train.add_argument('--resume',      action='store_true',
        help='Reanudar desde el último checkpoint')
    p_train.add_argument('--report',      action='store_true',
        help='Mostrar curvas de loss al finalizar')
    p_train.set_defaults(func=cmd_train)

    # ── encode ────────────────────────────────────────────────────────────────
    p_enc = sub.add_parser('encode',
        help='Codifica un MIDI de referencia como z_style (.json)')
    p_enc.add_argument('--input',      required=True, metavar='FILE',
        help='MIDI de referencia')
    p_enc.add_argument('--model-dir',  required=True, metavar='DIR',
        help='Directorio con el modelo entrenado')
    p_enc.add_argument('--output',     metavar='FILE',
        help='Archivo .json de salida (default: <input>.style.json)')
    p_enc.add_argument('--report',     action='store_true',
        help='Mostrar distribución de z_style y tensión media')
    p_enc.set_defaults(func=cmd_encode)

    # ── compose ───────────────────────────────────────────────────────────────
    p_comp = sub.add_parser('compose',
        help='Genera una obra nueva a partir del espacio latente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Modos de composición (--mode):

              reconstruct   Decodifica z_context exacto de un MIDI ref.
              z-noise       z_context del MIDI + ruido gaussiano
              blend         Interpolación ponderada entre varios MIDIs
              sweep         Barrido suave entre varios MIDIs a lo largo de la obra

            Método de recuperación (--retrieval):

              decoder  (default) Genera el piano roll decodificando z con el VAE.
              nnr      Recupera el compás del corpus con z más cercano al query.
                       Requiere --data-dir. No usa el decoder: es más estable
                       y sirve como diagnóstico del espacio latente.

            Opciones NNR:
              --data-dir DIR              Directorio con los .npz del corpus
              --nnr-distance euclidean|cosine  Métrica de distancia (default: euclidean)
              --nnr-tension-weight FLOAT  Peso del término de tensión [0,1] (default: 0.0)
              --nnr-temperature FLOAT     Temperatura de muestreo: 0.0=vecino exacto,
                                          >0 muestrea entre cercanos ponderando por
                                          exp(-dist/T). Rango típico 0.1-1.0 (default: 0.0)

            Perfiles de tensión (--tension):
              flat | arch | rise | fall | archivo.json

            Ejemplos:
              # Modo clásico (decoder)
              compose --model-dir model/ --palette p.json --mode z-noise --input ref.mid

              # Modo NNR (recuperación del corpus)
              compose --model-dir model/ --palette p.json --mode z-noise --input ref.mid \
                      --retrieval nnr --data-dir data/

              # NNR con distancia coseno y condicionamiento de tensión
              compose --model-dir model/ --palette p.json --mode sweep --inputs a.mid b.mid \
                      --retrieval nnr --data-dir data/ \
                      --nnr-distance cosine --nnr-tension-weight 0.3

              # Blend clásico
              compose --model-dir model/ --palette p.json --mode blend --inputs a.mid b.mid --weights 0.6 0.4
        """))
    p_comp.add_argument('--model-dir',   required=True, metavar='DIR',
        help='Directorio con el modelo entrenado')
    p_comp.add_argument('--palette',     required=True, metavar='FILE',
        help='JSON con mapeo rol → {program, channel, velocity}')
    p_comp.add_argument('--mode',
        choices=['reconstruct', 'z-noise', 'blend', 'sweep'],
        default='z-noise',
        help='Modo de generación (default: z-noise)')
    p_comp.add_argument('--bars',        type=int, default=32, metavar='INT',
        help='Compases a generar (default: 32)')
    p_comp.add_argument('--tension',     default='arch', metavar='PERFIL',
        help='Perfil de tensión: flat|arch|rise|fall|<archivo.json> (default: arch)')
    p_comp.add_argument('--output',      default='output.mid', metavar='FILE',
        help='Archivo MIDI de salida (default: output.mid)')
    p_comp.add_argument('--temperature', type=float, default=1.0, metavar='FLOAT',
        help='Temperatura de muestreo: >1 más aleatorio, <1 más determinista (default: 1.0)')
    p_comp.add_argument('--bpm',         type=float, default=120.0, metavar='FLOAT',
        help='Tempo en BPM (default: 120)')
    # argumentos dependientes del modo
    p_comp.add_argument('--input',       metavar='FILE',
        help='[reconstruct, z-noise] MIDI de referencia')
    p_comp.add_argument('--inputs',      nargs='+', metavar='FILE',
        help='[blend, sweep] 2 o más MIDIs de referencia')
    p_comp.add_argument('--weights',     nargs='+', type=float, metavar='W',
        help='[blend] Pesos de mezcla (uno por --inputs, se normalizan)')
    p_comp.add_argument('--noise',       type=float, default=0.3, metavar='FLOAT',
        help='[z-noise] Magnitud del ruido gaussiano (default: 0.3)')
    p_comp.add_argument('--smooth',      type=float, default=0.5, metavar='FLOAT',
        help='[sweep] Suavizado: 0=lineal, >0=spline cúbico (default: 0.5)')
    # ── Modo de recuperación ──────────────────────────────────────────────
    p_comp.add_argument('--retrieval',
        choices=['decoder', 'nnr'],
        default='decoder',
        dest='retrieval',
        help='Método de generación: decoder (VAE clásico) | nnr (vecino más '
             'cercano en el espacio latente del corpus). (default: decoder)')
    p_comp.add_argument('--data-dir',    metavar='DIR',
        dest='data_dir', default=None,
        help='[nnr] Directorio con los .npz del corpus (requerido para --retrieval nnr)')
    p_comp.add_argument('--nnr-distance',
        choices=['euclidean', 'cosine'],
        default='euclidean',
        dest='nnr_distance',
        help='[nnr] Métrica de distancia en el espacio latente (default: euclidean)')
    p_comp.add_argument('--nnr-tension-weight', type=float, default=0.0,
        metavar='FLOAT', dest='nnr_tension_weight',
        help='[nnr] Peso del término de tensión en la distancia compuesta: '
             '0.0=solo latente, 1.0=latente y tensión a partes iguales (default: 0.0)')
    p_comp.add_argument('--nnr-temperature', type=float, default=0.0,
        metavar='FLOAT', dest='nnr_temperature',
        help='[nnr] Temperatura del muestreo ponderado por distancia. '
             '0.0=determinista (vecino exacto); valores mayores aumentan la '
             'variedad muestreando entre los candidatos cercanos con '
             'prob proporcional a exp(-dist/T). Rango típico: 0.1-1.0 (default: 0.0)')
    p_comp.set_defaults(func=cmd_compose,
                        input=None, inputs=None, weights=None)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect',
        help='Diagnóstico del modelo, espacio latente y datos preparados')
    p_ins.add_argument('--what',          nargs='+',
        choices=['npz', 'loss_curve', 'role_quality',
                 'latent_projection', 'tension_response'],
        default=['npz'],
        help='Qué inspeccionar (default: npz)')
    # argumentos para --what npz
    p_ins.add_argument('--data-dir',      metavar='DIR',
        help='Directorio con .npz (requerido para --what npz)')
    p_ins.add_argument('--file',          dest='npz_file', metavar='NAME',
        help='Inspeccionar solo este fichero (sin extensión .npz)')
    p_ins.add_argument('--window',        type=int, default=0, metavar='INT',
        help='Índice de ventana a mostrar (default: 0, soporta negativos)')
    p_ins.add_argument('--bars-show',     type=int, default=4, metavar='INT',
        dest='bars_show',
        help='Compases de la ventana a renderizar (default: todos)')
    p_ins.add_argument('--roles',         nargs='+', metavar='ROL',
        choices=ROLES,
        help='Mostrar solo estos roles (default: todos)')
    p_ins.add_argument('--tension-curve', action='store_true',
        dest='tension_curve',
        help='Mostrar curvas de tensión completas')
    p_ins.add_argument('--no-roll',       action='store_true',
        dest='no_roll',
        help='Omitir piano roll (solo metadatos y tensión)')
    # argumentos para subcomandos futuros (post-train)
    p_ins.add_argument('--model-dir',     metavar='DIR',
        help='Modelo entrenado (requerido para loss_curve, etc.)')
    p_ins.add_argument('--midi',          metavar='FILE',
        help='MIDI a analizar (para latent_projection)')
    p_ins.set_defaults(func=cmd_inspect,
                       npz_file=None, roles=None,
                       tension_curve=False, no_roll=False,
                       bars_show=4, window=0)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
