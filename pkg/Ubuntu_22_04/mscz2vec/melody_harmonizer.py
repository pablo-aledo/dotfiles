#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     MELODY HARMONIZER  v1.0                                  ║
║      Adaptación de melodía a una armonía dada con ornamentación opcional    ║
║                                                                              ║
║  Toma un MIDI con armonía (acordes) y otro con una melodía, y reescribe     ║
║  la melodía para que sea musicalmente correcta sobre la armonía dada.       ║
║  Cada nota se ajusta al acorde vigente priorizando chord tones y usando     ║
║  tensiones admitidas; opcionalmente introduce apoyaturas o notas de paso   ║
║  en compases concretos.                                                      ║
║                                                                              ║
║  ESTRATEGIAS DE ADAPTACIÓN (--adapt-mode):                                  ║
║    nearest     — nota más cercana que sea chord_tone o tension permitida    ║
║    above       — chord tone más próximo por arriba                          ║
║    below       — chord tone más próximo por debajo                          ║
║    contour     — preserva el contorno melódico (subidas/bajadas) relativo  ║
║    reposo      — clasifica notas de reposo (estables) y de paso; las de    ║
║                  reposo se fijan al chord tone más cercano y las de paso   ║
║                  se recalculan por grados conjuntos entre ambas            ║
║    scale       — proyecta sobre la escala del acorde vigente                ║
║                                                                              ║
║  ORNAMENTOS (--ornaments):                                                   ║
║    appoggiatura  — apoyatura: nota disonante en tiempo fuerte → resolución  ║
║    passing       — nota de paso entre dos chord tones                       ║
║    neighbor      — bordadura (nota auxiliar superior o inferior)            ║
║    all           — aplica todos los ornamentos según contexto               ║
║                                                                              ║
║  USO:                                                                        ║
║    python melody_harmonizer.py harmony.mid melody.mid                       ║
║    python melody_harmonizer.py harmony.mid melody.mid --adapt-mode contour  ║
║    python melody_harmonizer.py harmony.mid melody.mid --ornaments appoggiatura passing --ornament-bars 3 7 11 ║
║    python melody_harmonizer.py harmony.mid melody.mid --key "G major"      ║
║    python melody_harmonizer.py harmony.mid melody.mid --verbose --report   ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --adapt-mode M    Estrategia de adaptación melódica (default: contour)   ║
║    --ornaments O [O…] Tipos de ornamento a aplicar (default: ninguno)      ║
║    --ornament-bars N [N…] Compases (1-based) donde insertar ornamentos     ║
║                           Si no se especifica, se aplican a todos si        ║
║                           --ornaments está activo                           ║
║    --ornament-prob F  Probabilidad de ornamentación por nota (0-1, def 0.4) ║
║    --key "C major"   Tonalidad. Si no se indica, se autodetecta             ║
║    --tempo BPM       Tempo de salida (default: detectado del MIDI harmonía) ║
║    --out-dir DIR     Carpeta de salida (default: junto al MIDI melodía)     ║
║    --report          Guardar reporte JSON con análisis completo             ║
║    --verbose         Informe detallado por stdout                            ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    melody.harmonized.mid          (melodía adaptada + armonía original)     ║
║    melody.harmonized_report.json  (con --report)                            ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21, numpy                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import math
import random
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import mido

# ── music21 ───────────────────────────────────────────────────────────────────
try:
    from music21 import converter, pitch as m21pitch, key as m21key
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Se usará detección de tonalidad simplificada.")


# ═══════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Intervalos de cada calidad de acorde (en semitonos desde la raíz)
CHORD_INTERVALS = {
    'M':    [0, 4, 7],
    'm':    [0, 3, 7],
    'd':    [0, 3, 6],
    'A':    [0, 4, 8],
    'M7':   [0, 4, 7, 11],
    'm7':   [0, 3, 7, 10],
    'Mm7':  [0, 4, 7, 10],
    'd7':   [0, 3, 6, 9],
    'hd7':  [0, 3, 6, 10],
    'M9':   [0, 4, 7, 11, 14],
    'm9':   [0, 3, 7, 10, 14],
    'Mm9':  [0, 4, 7, 10, 14],
    'sus4': [0, 5, 7],
    'sus2': [0, 2, 7],
    '6':    [0, 4, 7, 9],
    'm6':   [0, 3, 7, 9],
    'aug':  [0, 4, 8],
    'dim':  [0, 3, 6],
    'dom7': [0, 4, 7, 10],
}

# Grados de escala mayor / menor (para tensiones admitidas sobre cada acorde)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]   # natural menor

# Clasificación: intervalo desde raíz → categoría
NOTE_CATEGORY = {
    0:  'chord_tone',
    2:  'tension',
    3:  'chord_tone',
    4:  'chord_tone',
    5:  'tension',
    6:  'avoid',
    7:  'chord_tone',
    9:  'tension',
    10: 'tension',
    11: 'tension',
    1:  'avoid',
    8:  'avoid',
}


def chord_tone_category(mel_pc, root_pc, quality):
    """Clasifica una nota melódica respecto a un acorde."""
    interval = (mel_pc - root_pc) % 12
    base_cat = NOTE_CATEGORY.get(interval, 'tension')
    if quality in ('Mm7', 'M7', 'd7', 'hd7', 'dom7'):
        if interval == 11:
            return 'avoid'
        if interval == 6:
            return 'tension'
    if quality in ('M', 'M7'):
        if interval == 5:
            return 'avoid'
    return base_cat


def chord_tones_for_quality(root_pc, quality):
    """Devuelve el conjunto de pitch classes que son chord tones de un acorde."""
    ints = CHORD_INTERVALS.get(quality, [0, 4, 7])
    return {(root_pc + i) % 12 for i in ints}


def admissible_pcs(root_pc, quality, scale_pcs=None):
    """
    Devuelve pitch classes admisibles (chord_tone + tension no evitada)
    sobre un acorde. Si se pasa scale_pcs, se intersecta con la escala.
    """
    ok = set()
    for pc in range(12):
        cat = chord_tone_category(pc, root_pc, quality)
        if cat in ('chord_tone', 'tension'):
            ok.add(pc)
    if scale_pcs:
        ok &= set(scale_pcs)
        # siempre incluir chord tones aunque no estén en escala
        ok |= chord_tones_for_quality(root_pc, quality)
    return ok


# ═══════════════════════════════════════════════════════════
# TONALIDAD AUXILIAR
# ═══════════════════════════════════════════════════════════

class SimpleKey:
    def __init__(self, tonic_name='C', mode='major'):
        self.tonic = type('T', (), {
            'name': tonic_name,
            'pitchClass': PITCH_NAMES.index(tonic_name) if tonic_name in PITCH_NAMES else 0
        })()
        self.mode = mode

    def __str__(self):
        return f"{self.tonic.name} {self.mode}"


def parse_key_string(key_str):
    parts = key_str.strip().split()
    tonic_name = parts[0] if parts else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    if MUSIC21_OK:
        try:
            return m21key.Key(tonic_name, mode)
        except Exception:
            pass
    return SimpleKey(tonic_name, mode)


def tonic_pc(key_obj):
    if key_obj is None:
        return 0
    try:
        return m21pitch.Pitch(key_obj.tonic.name).pitchClass
    except Exception:
        return PITCH_NAMES.index(key_obj.tonic.name) if key_obj.tonic.name in PITCH_NAMES else 0


def scale_pcs_for_key(key_obj):
    """Devuelve los pitch classes de la escala de la tonalidad dada."""
    root = tonic_pc(key_obj)
    mode = key_obj.mode if key_obj else 'major'
    intervals = MAJOR_SCALE if mode == 'major' else MINOR_SCALE
    return [(root + i) % 12 for i in intervals]


# ═══════════════════════════════════════════════════════════
# CARGA DE MIDI
# ═══════════════════════════════════════════════════════════

def _load_midi_notes(midi_path, verbose=False):
    """
    Carga todas las notas de un MIDI con tiempos en beats.
    Devuelve:
        notes_by_ch : dict canal → list of (offset_beats, pitch, dur_beats, velocity)
        tempo_bpm   : float
        tpb         : ticks_per_beat
        time_sig    : (numerator, denominator)
        key_obj     : music21 Key o None
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4

    notes_by_ch = defaultdict(list)
    pending = {}

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
            elif msg.type in ('note_off', 'note_on'):
                key = (msg.channel, msg.note)
                if key in pending:
                    on_t, vel = pending.pop(key)
                    dur_beat = max(0.05, (abs_t - on_t) / tpb)
                    notes_by_ch[msg.channel].append(
                        (on_t / tpb, msg.note, dur_beat, vel)
                    )

    tempo_bpm = round(60_000_000 / max(tempo_us, 1), 2)
    if verbose:
        total_notes = sum(len(v) for v in notes_by_ch.values())
        print(f"    Cargado: {total_notes} notas, tempo={tempo_bpm} BPM, "
              f"compás={ts_num}/{ts_den}, canales={list(notes_by_ch.keys())}")

    key_obj = None
    if MUSIC21_OK:
        try:
            sc = converter.parse(midi_path)
            key_obj = sc.analyze('key')
            if verbose:
                print(f"    Tonalidad detectada: {key_obj}")
        except Exception:
            pass
    if key_obj is None and MUSIC21_OK:
        key_obj = m21key.Key('C', 'major')

    return notes_by_ch, tempo_bpm, tpb, (ts_num, ts_den), key_obj


def load_harmony(midi_path, verbose=False):
    """
    Carga el MIDI de armonía y extrae una lista de acordes como:
      list of (start_beat, duration_beats, root_pc, quality, pitches_list)
    Agrupa notas simultáneas (dentro de una ventana) en acordes.
    """
    notes_by_ch, tempo_bpm, tpb, time_sig, key_obj = _load_midi_notes(midi_path, verbose)

    if not notes_by_ch:
        raise RuntimeError("No se encontraron notas en el MIDI de armonía.")

    # Recoger todas las notas de todos los canales
    all_notes = []
    for ch_notes in notes_by_ch.values():
        all_notes.extend(ch_notes)
    all_notes.sort(key=lambda n: n[0])

    if verbose:
        print(f"    Total notas armonía: {len(all_notes)}")

    # Agrupar en «eventos de acorde» por onset cercano (ventana = 0.1 beat)
    chords_raw = []
    window = 0.12

    i = 0
    while i < len(all_notes):
        t0 = all_notes[i][0]
        group = []
        j = i
        while j < len(all_notes) and abs(all_notes[j][0] - t0) <= window:
            group.append(all_notes[j])
            j += 1
        i = j

        if not group:
            continue

        start = min(n[0] for n in group)
        # Duración: mínima duración de las notas del grupo (o hasta el siguiente onset)
        dur = min(n[2] for n in group)
        pitches = sorted(set(n[1] for n in group))
        chords_raw.append((start, dur, pitches))

    # Para cada grupo de pitches, inferir raíz y calidad
    chords = []
    for start, dur, pitches in chords_raw:
        root_pc, quality = infer_chord(pitches)
        chords.append({
            'start': start,
            'dur': dur,
            'pitches': pitches,
            'root_pc': root_pc,
            'quality': quality,
        })

    # Ajustar duración de cada acorde hasta el inicio del siguiente
    for idx in range(len(chords) - 1):
        next_start = chords[idx + 1]['start']
        chords[idx]['dur'] = max(0.05, next_start - chords[idx]['start'])

    if verbose:
        print(f"    Acordes detectados: {len(chords)}")
        for c in chords[:8]:
            pnames = [PITCH_NAMES[p % 12] for p in c['pitches']]
            print(f"      beat={c['start']:.2f} dur={c['dur']:.2f}  "
                  f"root={PITCH_NAMES[c['root_pc']]} {c['quality']}  "
                  f"pitches={pnames}")
        if len(chords) > 8:
            print(f"      … ({len(chords) - 8} más)")

    return chords, tempo_bpm, tpb, time_sig, key_obj


def infer_chord(pitches):
    """
    Intenta inferir la raíz y calidad de un acorde dado un grupo de pitches MIDI.
    Estrategia: evalúa cada posible raíz (pitch class de las notas) y elige la que
    mejor encaja con un patrón de acorde conocido.
    Devuelve (root_pc, quality_str).
    """
    if not pitches:
        return 0, 'M'

    pcs = sorted(set(p % 12 for p in pitches))

    best_root, best_quality, best_score = pcs[0], 'M', -1

    for candidate_root in pcs:
        intervals = sorted((p - candidate_root) % 12 for p in pcs)
        for quality, ints in CHORD_INTERVALS.items():
            ints_set = set(ints)
            pc_set = set(intervals)
            # puntuación: chord tones cubiertos / total intervalos del acorde
            covered = len(pc_set & ints_set)
            extra = len(pc_set - ints_set)
            score = covered / max(len(ints_set), 1) - 0.2 * extra
            if score > best_score:
                best_score = score
                best_root = candidate_root
                best_quality = quality

    return best_root, best_quality


def load_melody_midi(midi_path, verbose=False):
    """
    Carga el MIDI de melodía. Selecciona el canal con el pitch medio más alto
    como melodía principal.
    Devuelve list of (offset_beats, pitch, dur_beats, velocity).
    """
    notes_by_ch, tempo_bpm, tpb, time_sig, key_obj = _load_midi_notes(midi_path, verbose)

    if not notes_by_ch:
        raise RuntimeError("No se encontraron notas en el MIDI de melodía.")

    def mean_pitch(notes):
        return np.mean([p for _, p, _, _ in notes]) if notes else 0

    mel_ch = max(notes_by_ch, key=lambda c: mean_pitch(notes_by_ch[c]))
    melody = sorted(notes_by_ch[mel_ch], key=lambda x: x[0])

    if verbose:
        print(f"    Canal melodía seleccionado: {mel_ch}  "
              f"({len(melody)} notas, pitch medio={mean_pitch(melody):.1f})")

    return melody, tempo_bpm, tpb, time_sig, key_obj


# ═══════════════════════════════════════════════════════════
# OBTENER ACORDE VIGENTE EN UN BEAT DADO
# ═══════════════════════════════════════════════════════════

def chord_at_beat(chords, beat):
    """
    Devuelve el dict de acorde vigente en un beat dado.
    Busca el acorde cuyo rango [start, start+dur) contiene beat.
    Si no hay ninguno, devuelve el más cercano por la izquierda.
    """
    best = None
    for c in chords:
        if c['start'] <= beat < c['start'] + c['dur']:
            return c
        if c['start'] <= beat:
            best = c
    return best or (chords[0] if chords else None)


# ═══════════════════════════════════════════════════════════
# ADAPTACIÓN DE NOTA
# ═══════════════════════════════════════════════════════════

def nearest_admissible(pitch, admissible_pcs_set, direction='nearest'):
    """
    Dado un pitch MIDI y un conjunto de pitch classes admisibles,
    devuelve el pitch MIDI más cercano en esa dirección.
    direction: 'nearest' | 'above' | 'below'
    """
    if not admissible_pcs_set:
        return pitch

    pc = pitch % 12

    if pc in admissible_pcs_set:
        return pitch  # ya es admisible

    # Calcular distancias hacia arriba y hacia abajo
    up_dist = min((ap - pc) % 12 for ap in admissible_pcs_set)
    down_dist = min((pc - ap) % 12 for ap in admissible_pcs_set)

    # Pitch class objetivo
    if direction == 'above':
        target_pc = (pc + up_dist) % 12
    elif direction == 'below':
        target_pc = (pc - down_dist) % 12
    else:  # nearest
        if up_dist <= down_dist:
            target_pc = (pc + up_dist) % 12
        else:
            target_pc = (pc - down_dist) % 12

    # Construir pitch MIDI en la octava más cercana al original
    base = (pitch // 12) * 12 + target_pc
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda p: abs(p - pitch))


def adapt_note_contour(pitch, prev_adapted, chord, admissible, scale_pcs):
    """
    Preserva el contorno melódico: si la nota original subió respecto a la
    anterior, busca la nota admisible más próxima que también suba.
    Fallback a nearest si no hay candidatos en esa dirección.
    """
    if prev_adapted is None:
        return nearest_admissible(pitch, admissible, 'nearest')

    direction_original = pitch - (pitch - 1)  # placeholder; se calcula fuera

    # Se calcula en adapt_melody(); aquí simplemente llamamos nearest
    return nearest_admissible(pitch, admissible, 'nearest')


def adapt_note_scale(pitch, chord, scale_pcs):
    """
    Proyecta la nota sobre la escala del acorde vigente (chord tones + escala).
    """
    adm = admissible_pcs(chord['root_pc'], chord['quality'], scale_pcs)
    return nearest_admissible(pitch, adm, 'nearest')


# ═══════════════════════════════════════════════════════════
# ADAPTACIÓN COMPLETA DE MELODÍA
# ═══════════════════════════════════════════════════════════

def note_metric_strength(offset, beats_per_bar):
    """
    Estima la fuerza métrica de una nota según su posición dentro del compás:
      1.0  → tiempo fuerte (inicio de compás)
      0.6  → tiempo (pulso) pero no el primero
      0.25 → subdivisión fuera de pulso (contratiempo / anacrusa corta)
    """
    pos = offset % beats_per_bar
    if abs(pos) < 1e-6:
        return 1.0
    if abs(pos - round(pos)) < 1e-6:
        return 0.6
    return 0.25


def classify_note_roles(melody, time_sig, verbose=False):
    """
    Clasifica cada nota de la melodía como:
      'reposo' — nota estable/estructural: cae en tiempo fuerte y/o tiene
                 una duración relativamente larga. Son los puntos de anclaje
                 armónico de la frase.
      'paso'   — nota de paso/transitoria: breve, en parte débil del compás,
                 y frecuentemente parte de un movimiento por grados conjuntos
                 (stepwise) entre dos notas más estables.

    Combina tres señales:
      - fuerza métrica (posición en el compás)
      - duración relativa a la mediana de la melodía
      - si la nota participa en un tramo monótono de grados conjuntos
        rodeada de notas de mayor duración (patrón típico de nota de paso)

    Devuelve:
        roles  : list de 'reposo' | 'paso' (uno por nota)
        scores : list de float (score usado internamente, útil para debug)
    """
    beats_per_bar = beats_per_bar_from_ts(time_sig)
    n = len(melody)
    durations = [d for _, _, d, _ in melody]
    median_dur = float(np.median(durations)) if durations else 1.0

    roles, scores = [], []
    for idx, (offset, pitch, dur, vel) in enumerate(melody):
        metric = note_metric_strength(offset, beats_per_bar)
        dur_score = min(1.0, dur / max(median_dur, 1e-6))

        stepwise_transit = False
        if 0 < idx < n - 1:
            prev_pitch, prev_dur = melody[idx - 1][1], melody[idx - 1][2]
            next_pitch, next_dur = melody[idx + 1][1], melody[idx + 1][2]
            d1 = pitch - prev_pitch
            d2 = next_pitch - pitch
            same_dir = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)
            small_steps = 0 < abs(d1) <= 2 and 0 < abs(d2) <= 2
            neighbors_longer = (prev_dur >= dur) and (next_dur >= dur)
            if same_dir and small_steps and neighbors_longer:
                stepwise_transit = True

        score = 0.55 * metric + 0.45 * dur_score
        if stepwise_transit:
            score -= 0.3

        roles.append('reposo' if score >= 0.5 else 'paso')
        scores.append(round(score, 3))

    if verbose:
        n_reposo = roles.count('reposo')
        print(f"    Clasificación reposo/paso: {n_reposo} reposo / {n - n_reposo} paso")

    return roles, scores


def adapt_melody_reposo_paso(melody, chords, key_obj, time_sig, verbose=False):
    """
    Estrategia 'reposo': adapta la melodía en dos pasadas.

      1) Clasifica cada nota como 'reposo' o 'paso' (classify_note_roles).
      2) Las notas de reposo se fijan al chord tone más cercano del acorde
         vigente: son los anclajes armónicos de la frase.
      3) Las notas de paso se recalculan interpolando por grados conjuntos
         (dentro de las notas admisibles del acorde/escala vigente) entre las
         dos notas de reposo ya fijadas que las rodean, de modo que cumplan
         su función de enlace melódico entre chord tones.

    Devuelve (adapted, log) con el mismo formato que adapt_melody().
    """
    scale_pcs = scale_pcs_for_key(key_obj) if key_obj else list(range(12))
    roles, scores = classify_note_roles(melody, time_sig, verbose=verbose)
    n = len(melody)
    adapted = [None] * n
    log = [None] * n

    def _log_entry(ridx, offset, pitch, new_pitch, root_pc, quality, role):
        cat_orig = chord_tone_category(pitch % 12, root_pc, quality) if root_pc is not None else None
        cat_new = chord_tone_category(new_pitch % 12, root_pc, quality) if root_pc is not None else None
        return {
            'idx': ridx,
            'beat': round(offset, 3),
            'role': role,
            'orig': pitch,
            'new': new_pitch,
            'moved': (new_pitch != pitch),
            'semitones': new_pitch - pitch,
            'chord_root': PITCH_NAMES[root_pc] if root_pc is not None else None,
            'chord_quality': quality,
            'cat_orig': cat_orig,
            'cat_new': cat_new,
            'score': scores[ridx],
        }

    # ── Paso 1: fijar notas de reposo sobre chord tones ──────────────────
    for idx, (offset, pitch, dur, vel) in enumerate(melody):
        if roles[idx] != 'reposo':
            continue
        chord = chord_at_beat(chords, offset)
        if chord is None:
            new_pitch, root_pc, quality = pitch, None, None
        else:
            root_pc, quality = chord['root_pc'], chord['quality']
            chord_pcs = chord_tones_for_quality(root_pc, quality)
            new_pitch = nearest_admissible(pitch, chord_pcs, 'nearest')
            new_pitch = max(21, min(108, new_pitch))

        adapted[idx] = (offset, new_pitch, dur, vel)
        log[idx] = _log_entry(idx, offset, pitch, new_pitch, root_pc, quality, 'reposo')

        if verbose and new_pitch != pitch:
            root_str = f"{PITCH_NAMES[root_pc]}{quality}" if root_pc is not None else "—"
            print(f"    [reposo] beat={offset:.2f}  {PITCH_NAMES[pitch%12]}{pitch//12-1}"
                  f" → {PITCH_NAMES[new_pitch%12]}{new_pitch//12-1}  [{root_str}]")

    # ── Paso 2: rellenar las notas de paso entre anclas de reposo ────────
    idx = 0
    while idx < n:
        if adapted[idx] is not None:
            idx += 1
            continue

        start = idx
        while idx < n and adapted[idx] is None:
            idx += 1
        end = idx  # adapted[end] ya fijado, o end == n si es el final de la melodía

        left_pitch = adapted[start - 1][1] if start > 0 else None
        right_pitch = adapted[end][1] if end < n else None
        run = list(range(start, end))
        run_len = len(run)

        for k, ridx in enumerate(run):
            offset, pitch, dur, vel = melody[ridx]
            chord = chord_at_beat(chords, offset)
            root_pc = chord['root_pc'] if chord else None
            quality = chord['quality'] if chord else None
            adm = admissible_pcs(root_pc, quality, scale_pcs) if chord else set(scale_pcs)

            if left_pitch is not None and right_pitch is not None:
                # Interpolación lineal entre las dos notas de reposo vecinas,
                # proyectada sobre las notas admisibles (grados conjuntos reales)
                frac = (k + 1) / (run_len + 1)
                target = left_pitch + frac * (right_pitch - left_pitch)
                new_pitch = nearest_admissible(int(round(target)), adm, 'nearest')
            elif left_pitch is not None:
                # Sin ancla derecha (cola de la melodía): continuar la dirección
                # original respecto al reposo anterior
                direction = 'above' if pitch >= left_pitch else 'below'
                new_pitch = nearest_admissible(pitch, adm, direction)
            elif right_pitch is not None:
                # Sin ancla izquierda (inicio de la melodía)
                direction = 'below' if pitch >= right_pitch else 'above'
                new_pitch = nearest_admissible(pitch, adm, direction)
            else:
                # Melodía sin ninguna nota de reposo detectada
                new_pitch = nearest_admissible(pitch, adm, 'nearest')

            new_pitch = max(21, min(108, new_pitch))
            adapted[ridx] = (offset, new_pitch, dur, vel)
            log[ridx] = _log_entry(ridx, offset, pitch, new_pitch, root_pc, quality, 'paso')

            if verbose and new_pitch != pitch:
                print(f"    [paso]   beat={offset:.2f}  {PITCH_NAMES[pitch%12]}{pitch//12-1}"
                      f" → {PITCH_NAMES[new_pitch%12]}{new_pitch//12-1}")

    return adapted, log


def adapt_melody(melody, chords, key_obj, mode='contour', time_sig=(4, 4), verbose=False):
    """
    Adapta cada nota de la melodía al acorde vigente.

    Args:
        melody   : list of (offset, pitch, dur, vel)
        chords   : list de acordes (dicts con root_pc, quality, start, dur)
        key_obj  : tonalidad
        mode     : 'nearest' | 'above' | 'below' | 'contour' | 'scale' | 'reposo'
        time_sig : (numerador, denominador) — necesario para el modo 'reposo'

    Returns:
        adapted  : list of (offset, new_pitch, dur, vel)
        log      : list of dicts con info por nota
    """
    if mode == 'reposo':
        return adapt_melody_reposo_paso(melody, chords, key_obj, time_sig, verbose=verbose)

    scale_pcs = scale_pcs_for_key(key_obj) if key_obj else list(range(12))
    adapted = []
    log = []

    prev_orig = None
    prev_adapted_pitch = None

    for idx, (offset, pitch, dur, vel) in enumerate(melody):
        chord = chord_at_beat(chords, offset)
        if chord is None:
            adapted.append((offset, pitch, dur, vel))
            log.append({'idx': idx, 'orig': pitch, 'new': pitch,
                        'chord': None, 'category': 'no_chord'})
            prev_orig = pitch
            prev_adapted_pitch = pitch
            continue

        root_pc = chord['root_pc']
        quality = chord['quality']

        adm = admissible_pcs(root_pc, quality, scale_pcs)
        cat_orig = chord_tone_category(pitch % 12, root_pc, quality)

        if cat_orig != 'avoid':
            # Ya es admisible: mantener
            new_pitch = pitch
        else:
            if mode == 'nearest':
                new_pitch = nearest_admissible(pitch, adm, 'nearest')
            elif mode == 'above':
                new_pitch = nearest_admissible(pitch, adm, 'above')
            elif mode == 'below':
                new_pitch = nearest_admissible(pitch, adm, 'below')
            elif mode == 'scale':
                new_pitch = adapt_note_scale(pitch, chord, scale_pcs)
            elif mode == 'contour':
                # Preservar contorno
                if prev_orig is not None and prev_adapted_pitch is not None:
                    delta = pitch - prev_orig
                    if delta > 0:
                        direction = 'above'
                    elif delta < 0:
                        direction = 'below'
                    else:
                        direction = 'nearest'
                    new_pitch = nearest_admissible(pitch, adm, direction)
                    # Si el resultado invierte el contorno, relajar a nearest
                    if delta > 0 and new_pitch < prev_adapted_pitch:
                        new_pitch = nearest_admissible(pitch, adm, 'nearest')
                    elif delta < 0 and new_pitch > prev_adapted_pitch:
                        new_pitch = nearest_admissible(pitch, adm, 'nearest')
                else:
                    new_pitch = nearest_admissible(pitch, adm, 'nearest')
            else:
                new_pitch = nearest_admissible(pitch, adm, 'nearest')

        new_pitch = max(21, min(108, new_pitch))  # rango MIDI de piano
        cat_new = chord_tone_category(new_pitch % 12, root_pc, quality)

        log.append({
            'idx': idx,
            'beat': round(offset, 3),
            'orig': pitch,
            'new': new_pitch,
            'moved': (new_pitch != pitch),
            'semitones': new_pitch - pitch,
            'chord_root': PITCH_NAMES[root_pc],
            'chord_quality': quality,
            'cat_orig': cat_orig,
            'cat_new': cat_new,
        })

        if verbose and new_pitch != pitch:
            print(f"    beat={offset:.2f}  {PITCH_NAMES[pitch%12]}{pitch//12-1}"
                  f" → {PITCH_NAMES[new_pitch%12]}{new_pitch//12-1}"
                  f"  [{PITCH_NAMES[root_pc]}{quality}]"
                  f"  {cat_orig}→{cat_new}")

        adapted.append((offset, new_pitch, dur, vel))
        prev_orig = pitch
        prev_adapted_pitch = new_pitch

    return adapted, log


# ═══════════════════════════════════════════════════════════
# ORNAMENTACIÓN
# ═══════════════════════════════════════════════════════════

def beats_per_bar_from_ts(time_sig):
    return time_sig[0]


def note_bar(offset, beats_per_bar):
    """Devuelve el compás (1-based) al que pertenece un offset en beats."""
    return int(offset // beats_per_bar) + 1


def insert_appoggiatura(note_idx, notes, chords, scale_pcs, min_dur=0.25):
    """
    Inserta una apoyatura antes de la nota indicada.
    La apoyatura es un semitono adyacente disonante que resuelve a la nota objetivo.
    La duración de la apoyatura es la mitad de la nota original (mínimo min_dur beats).
    Devuelve lista de (offset, pitch, dur, vel) que reemplaza la nota original.
    """
    offset, pitch, dur, vel = notes[note_idx]
    chord = chord_at_beat(chords, offset)
    if chord is None or dur < min_dur * 2:
        return [notes[note_idx]]

    root_pc = chord['root_pc']
    quality = chord['quality']
    cat = chord_tone_category(pitch % 12, root_pc, quality)

    # Solo aplicar si la nota objetivo es chord_tone o tension admitida
    if cat == 'avoid':
        return [notes[note_idx]]

    # Elegir nota de apoyatura: semitono superior o inferior disonante
    candidates = []
    for delta in [+1, -1, +2, -2]:
        app_pitch = pitch + delta
        app_cat = chord_tone_category(app_pitch % 12, root_pc, quality)
        if app_cat == 'avoid' or app_cat == 'tension':
            # Verificar que no sea chord tone (queremos disonancia real)
            if app_cat != 'chord_tone':
                candidates.append((app_pitch, abs(delta)))

    if not candidates:
        return [notes[note_idx]]

    # Preferir el más cercano
    app_pitch = min(candidates, key=lambda x: x[1])[0]
    app_pitch = max(21, min(108, app_pitch))

    app_dur = max(min_dur, dur * 0.33)
    main_dur = dur - app_dur

    return [
        (offset, app_pitch, app_dur * 0.9, min(vel, 90)),
        (offset + app_dur, pitch, main_dur * 0.95, vel),
    ]


def insert_passing_note(note_idx, notes, chords, scale_pcs):
    """
    Inserta una nota de paso entre la nota actual y la siguiente,
    si el intervalo es de tercera o mayor.
    Devuelve lista de notas (reemplazando la nota actual + añade nota de paso).
    """
    if note_idx >= len(notes) - 1:
        return [notes[note_idx]]

    offset, pitch, dur, vel = notes[note_idx]
    next_offset, next_pitch, next_dur, next_vel = notes[note_idx + 1]

    interval = abs(next_pitch - pitch)
    if interval < 2 or interval > 7:
        return [notes[note_idx]]

    if dur < 0.4:  # nota demasiado corta para dividir
        return [notes[note_idx]]

    chord = chord_at_beat(chords, offset)
    if chord is None:
        return [notes[note_idx]]

    # Nota de paso: punto intermedio diatónico entre las dos notas
    direction = 1 if next_pitch > pitch else -1
    # Elegir nota de paso que esté en la escala
    passing_pc_candidates = [
        (pitch + direction * step) % 12
        for step in range(1, interval)
        if (pitch + direction * step) % 12 in scale_pcs
    ]

    if not passing_pc_candidates:
        return [notes[note_idx]]

    # La nota de paso más cercana al punto medio
    mid_pc = ((pitch + next_pitch) // 2) % 12
    chosen_pc = min(passing_pc_candidates, key=lambda pc: abs(pc - mid_pc))

    # Pitch MIDI de la nota de paso en la octava correcta
    passing_pitch = (pitch // 12) * 12 + chosen_pc
    if direction > 0 and passing_pitch < pitch:
        passing_pitch += 12
    elif direction < 0 and passing_pitch > pitch:
        passing_pitch -= 12
    passing_pitch = max(21, min(108, passing_pitch))

    # Dividir la duración: nota actual acortada + nota de paso
    note_dur = dur * 0.55
    pass_dur = dur * 0.40

    return [
        (offset, pitch, note_dur * 0.95, vel),
        (offset + note_dur, passing_pitch, pass_dur * 0.9, max(40, vel - 15)),
    ]


def insert_neighbor_note(note_idx, notes, chords, scale_pcs, upper=True):
    """
    Bordadura: nota auxiliar (un tono/semitono diatónico arriba o abajo)
    que retorna a la nota original.
    """
    offset, pitch, dur, vel = notes[note_idx]
    if dur < 0.5:
        return [notes[note_idx]]

    chord = chord_at_beat(chords, offset)
    if chord is None:
        return [notes[note_idx]]

    direction = 1 if upper else -1
    # Buscar nota vecina diatónica
    for delta in [direction, direction * 2]:
        neighbor_pc = (pitch + delta) % 12
        if neighbor_pc in scale_pcs:
            neighbor_pitch = pitch + delta
            neighbor_pitch = max(21, min(108, neighbor_pitch))
            # Dividir: nota → vecina → nota
            seg = dur / 3
            return [
                (offset,         pitch,          seg * 0.95, vel),
                (offset + seg,   neighbor_pitch, seg * 0.9,  max(35, vel - 20)),
                (offset + 2*seg, pitch,          seg * 0.95, vel),
            ]

    return [notes[note_idx]]


def apply_ornaments(adapted_melody, chords, key_obj, time_sig,
                    ornament_types, ornament_bars=None,
                    ornament_prob=0.4, verbose=False):
    """
    Aplica ornamentos a la melodía adaptada.

    Args:
        adapted_melody  : list of (offset, pitch, dur, vel)
        chords          : lista de acordes
        key_obj         : tonalidad
        time_sig        : (numerator, denominator)
        ornament_types  : list de str: 'appoggiatura', 'passing', 'neighbor', 'all'
        ornament_bars   : list de ints (1-based) con compases donde aplicar.
                          None = todos los compases.
        ornament_prob   : probabilidad de ornamentación por nota (0-1)

    Returns:
        ornamented : lista de notas ornamentadas
        orn_log    : registro de ornamentos aplicados
    """
    if not ornament_types:
        return adapted_melody, []

    if 'all' in ornament_types:
        ornament_types = ['appoggiatura', 'passing', 'neighbor']

    scale_pcs = scale_pcs_for_key(key_obj) if key_obj else list(range(12))
    beats_bar = beats_per_bar_from_ts(time_sig)

    ornamented = []
    orn_log = []

    for idx, note in enumerate(adapted_melody):
        offset, pitch, dur, vel = note
        bar = note_bar(offset, beats_bar)

        # ¿Aplicar ornamento en este compás?
        if ornament_bars is not None and bar not in ornament_bars:
            ornamented.append(note)
            continue

        # Probabilidad
        if random.random() > ornament_prob:
            ornamented.append(note)
            continue

        chord = chord_at_beat(chords, offset)
        if chord is None:
            ornamented.append(note)
            continue

        # Elegir tipo de ornamento al azar entre los disponibles
        chosen_type = random.choice(ornament_types)
        replacement = None

        if chosen_type == 'appoggiatura':
            replacement = insert_appoggiatura(idx, adapted_melody, chords, scale_pcs)
        elif chosen_type == 'passing':
            replacement = insert_passing_note(idx, adapted_melody, chords, scale_pcs)
        elif chosen_type == 'neighbor':
            upper = random.random() > 0.5
            replacement = insert_neighbor_note(idx, adapted_melody, chords, scale_pcs, upper)

        if replacement and len(replacement) > 1:
            ornamented.extend(replacement)
            orn_log.append({
                'idx': idx,
                'bar': bar,
                'beat': round(offset, 3),
                'type': chosen_type,
                'orig_pitch': pitch,
                'new_pitches': [r[1] for r in replacement],
            })
            if verbose:
                pnames = [PITCH_NAMES[r[1]%12] for r in replacement]
                print(f"    Ornamento [{chosen_type}]  compás {bar}  "
                      f"beat={offset:.2f}  {PITCH_NAMES[pitch%12]} → {pnames}")
        else:
            ornamented.append(note)

    # Reordenar por offset
    ornamented.sort(key=lambda n: n[0])
    return ornamented, orn_log


# ═══════════════════════════════════════════════════════════
# ESCRITURA DEL MIDI DE SALIDA
# ═══════════════════════════════════════════════════════════

def write_harmonized_midi(adapted_melody, chords, tempo_bpm, time_sig,
                          output_path, tpb=480,
                          include_harmony=True, verbose=False):
    """
    Escribe el MIDI final con:
      Track 0: header (tempo, time_sig)
      Track 1: melodía adaptada (+ ornamentos si los hay)
      Track 2: armonía original reconstruida desde chords
    """
    beats_bar, _ = time_sig
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(1, int(round(beats * tpb)))

    def notes_to_track(notes_list, ch, program, track_name):
        events = []
        for offset, pitch, dur, vel in notes_list:
            p = max(0, min(127, int(pitch)))
            v = max(1, min(127, int(vel)))
            t_on  = to_ticks(offset)
            t_off = to_ticks(offset + max(0.05, dur))
            events.append((t_on,  'on',  p, v))
            events.append((t_off, 'off', p, 0))

        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))
        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=track_name, time=0))
        trk.append(mido.Message('program_change', channel=ch, program=program, time=0))
        prev_t = 0
        for abs_t, kind, p, v in events:
            dt = max(0, abs_t - prev_t)
            msg_type = 'note_on' if kind == 'on' else 'note_off'
            trk.append(mido.Message(msg_type, channel=ch, note=p, velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    # Track de cabecera
    hdr = mido.MidiTrack()
    hdr.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    hdr.append(mido.MetaMessage('time_signature',
                                numerator=beats_bar, denominator=4,
                                clocks_per_click=24,
                                notated_32nd_notes_per_beat=8,
                                time=0))
    hdr.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(hdr)

    # Track melodía adaptada
    mid.tracks.append(notes_to_track(adapted_melody, ch=0, program=0,
                                     track_name='Melody_Adapted'))

    # Track armonía (reconstruida desde chords)
    if include_harmony:
        harm_notes = []
        for chord in chords:
            for p in chord['pitches']:
                harm_notes.append((chord['start'], p, chord['dur'] * 0.88, 65))
        mid.tracks.append(notes_to_track(harm_notes, ch=1, program=0,
                                         track_name='Harmony'))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    mid.save(output_path)

    if verbose:
        print(f"    MIDI guardado: {output_path}  "
              f"({len(adapted_melody)} notas melodía, "
              f"{len(chords)} acordes armonía)")


# ═══════════════════════════════════════════════════════════
# ANÁLISIS Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════

def compute_stats(original_melody, adapted_melody, log):
    """Calcula estadísticas de la adaptación."""
    moved = [e for e in log if e.get('moved')]
    avoid_before = [e for e in log if e.get('cat_orig') == 'avoid']
    avoid_after  = [e for e in log if e.get('cat_new') == 'avoid']
    chord_tones  = [e for e in log if e.get('cat_new') == 'chord_tone']
    tensions     = [e for e in log if e.get('cat_new') == 'tension']

    return {
        'total_notes': len(original_melody),
        'notes_moved': len(moved),
        'pct_moved': round(100 * len(moved) / max(len(log), 1), 1),
        'avg_semitones_moved': round(
            float(np.mean([abs(e['semitones']) for e in moved])) if moved else 0, 2
        ),
        'avoid_notes_before': len(avoid_before),
        'avoid_notes_after': len(avoid_after),
        'chord_tone_pct': round(100 * len(chord_tones) / max(len(log), 1), 1),
        'tension_pct': round(100 * len(tensions) / max(len(log), 1), 1),
    }


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def harmonize_melody(harmony_midi: str,
                     melody_midi: str,
                     adapt_mode: str = 'contour',
                     ornament_types: list = None,
                     ornament_bars: list = None,
                     ornament_prob: float = 0.4,
                     key_override: str = None,
                     tempo_override: float = None,
                     out_dir: str = None,
                     report: bool = False,
                     verbose: bool = False):
    """
    Pipeline principal de adaptación de melodía a armonía dada.

    Args:
        harmony_midi    : Ruta al MIDI con la armonía (acordes).
        melody_midi     : Ruta al MIDI con la melodía a adaptar.
        adapt_mode      : Estrategia de adaptación ('nearest','above','below','contour','scale').
        ornament_types  : Lista de ornamentos a aplicar ['appoggiatura','passing','neighbor','all'].
        ornament_bars   : Lista de compases (1-based) donde aplicar ornamentos. None = todos.
        ornament_prob   : Probabilidad de ornamentación por nota candidata (0-1).
        key_override    : Tonalidad forzada, e.g. 'G major'.
        tempo_override  : BPM forzado.
        out_dir         : Carpeta de salida.
        report          : Guardar JSON de análisis.
        verbose         : Salida detallada.
    Returns:
        dict con datos del reporte.
    """
    if ornament_types is None:
        ornament_types = []

    harmony_midi = str(harmony_midi)
    melody_midi  = str(melody_midi)
    stem = Path(melody_midi).stem

    if out_dir is None:
        out_dir = str(Path(melody_midi).parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*64}")
    print(f"  MELODY HARMONIZER  ·  {os.path.basename(melody_midi)}")
    print(f"  Armonía            ·  {os.path.basename(harmony_midi)}")
    print(f"{'═'*64}")

    # ── Cargar armonía ────────────────────────────────────────────────────────
    print("\n  [1/4] Cargando armonía…")
    try:
        chords, harm_tempo, harm_tpb, harm_ts, harm_key = load_harmony(
            harmony_midi, verbose=verbose
        )
    except Exception as e:
        print(f"[ERROR] Armonía: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    # ── Cargar melodía ────────────────────────────────────────────────────────
    print("  [2/4] Cargando melodía…")
    try:
        melody, mel_tempo, mel_tpb, mel_ts, mel_key = load_melody_midi(
            melody_midi, verbose=verbose
        )
    except Exception as e:
        print(f"[ERROR] Melodía: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    # ── Resolución de parámetros ──────────────────────────────────────────────
    tempo_bpm = tempo_override if tempo_override else harm_tempo
    time_sig  = harm_ts

    if key_override:
        key_obj = parse_key_string(key_override)
        print(f"  Tonalidad forzada   : {key_obj}")
    else:
        # Preferir la tonalidad de la armonía
        key_obj = harm_key or mel_key
        print(f"  Tonalidad detectada : {key_obj}")

    beats_bar = time_sig[0]
    total_bars = max(1, int(math.ceil(
        max((n[0] + n[2]) for n in melody) / beats_bar
    )))

    print(f"  Tempo               : {tempo_bpm} BPM")
    print(f"  Compás              : {time_sig[0]}/{time_sig[1]}")
    print(f"  Compases totales    : {total_bars}")
    print(f"  Notas melodía       : {len(melody)}")
    print(f"  Acordes detectados  : {len(chords)}")
    print(f"  Modo adaptación     : {adapt_mode}")
    if ornament_types:
        bars_str = str(ornament_bars) if ornament_bars else 'todos'
        print(f"  Ornamentos          : {ornament_types}  "
              f"compases={bars_str}  prob={ornament_prob}")

    # ── Adaptación de melodía ─────────────────────────────────────────────────
    print("\n  [3/4] Adaptando melodía…")
    try:
        adapted, adapt_log = adapt_melody(
            melody, chords, key_obj, mode=adapt_mode, time_sig=time_sig, verbose=verbose
        )
    except Exception as e:
        print(f"[ERROR] Adaptación: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    stats = compute_stats(melody, adapted, adapt_log)
    print(f"    Notas desplazadas  : {stats['notes_moved']} / {stats['total_notes']} "
          f"({stats['pct_moved']}%)")
    print(f"    Despl. medio       : {stats['avg_semitones_moved']} semitonos")
    print(f"    Avoid antes/después: {stats['avoid_notes_before']} → {stats['avoid_notes_after']}")
    print(f"    Chord tones        : {stats['chord_tone_pct']}%  "
          f"Tensiones: {stats['tension_pct']}%")

    # ── Ornamentación ─────────────────────────────────────────────────────────
    orn_log = []
    final_melody = adapted

    if ornament_types:
        print("\n  [3b] Aplicando ornamentos…")
        try:
            final_melody, orn_log = apply_ornaments(
                adapted, chords, key_obj, time_sig,
                ornament_types=ornament_types,
                ornament_bars=ornament_bars,
                ornament_prob=ornament_prob,
                verbose=verbose,
            )
            print(f"    Ornamentos insertados: {len(orn_log)}")
        except Exception as e:
            print(f"    [AVISO] Ornamentación fallida ({e}), usando melodía sin ornamentos.")
            if verbose:
                traceback.print_exc()

    # ── Escritura MIDI ────────────────────────────────────────────────────────
    print("\n  [4/4] Escribiendo MIDI…")
    out_name = f"{stem}.harmonized.mid"
    out_path = os.path.join(out_dir, out_name)

    try:
        write_harmonized_midi(
            adapted_melody=final_melody,
            chords=chords,
            tempo_bpm=tempo_bpm,
            time_sig=time_sig,
            output_path=out_path,
            tpb=harm_tpb,
            include_harmony=True,
            verbose=verbose,
        )
        print(f"    → {out_name}")
    except Exception as e:
        print(f"[ERROR] Escritura MIDI: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    # ── Reporte ───────────────────────────────────────────────────────────────
    report_data = {
        'harmony_source': harmony_midi,
        'melody_source':  melody_midi,
        'key': str(key_obj),
        'mode': key_obj.mode if key_obj else 'major',
        'tempo_bpm': tempo_bpm,
        'time_sig': list(time_sig),
        'total_bars': total_bars,
        'adapt_mode': adapt_mode,
        'stats': stats,
        'ornaments': {
            'types': ornament_types,
            'bars': ornament_bars,
            'prob': ornament_prob,
            'count': len(orn_log),
            'log': orn_log,
        },
        'adaptation_log': adapt_log,
        'output': out_path,
    }

    if report:
        report_path = os.path.join(out_dir, f"{stem}.harmonized_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Reporte: {report_path}")

    print(f"\n{'═'*64}")
    print(f"  Completado. Archivo guardado: {out_name}")
    print(f"{'═'*64}\n")

    return report_data


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='MELODY HARMONIZER — Adapta una melodía MIDI a una armonía dada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de adaptación:
  nearest   Nota admisible más cercana al original
  above     Chord tone más próximo por arriba
  below     Chord tone más próximo por abajo
  contour   Preserva el contorno melódico (subidas/bajadas) [default]
  scale     Proyecta sobre la escala del acorde vigente
  reposo    Clasifica notas de reposo (estables) y de paso: las de reposo
            se fijan al chord tone más cercano, y las de paso se recalculan
            por grados conjuntos entre las notas de reposo que las rodean

Ornamentos disponibles:
  appoggiatura  Nota disonante en tiempo fuerte que resuelve por semitono
  passing       Nota de paso entre dos notas de acorde separadas por tercera
  neighbor      Bordadura (nota auxiliar diatónica que vuelve a la original)
  all           Aplica todos los ornamentos según contexto

Ejemplos:
  python melody_harmonizer.py harmony.mid melody.mid
  python melody_harmonizer.py harmony.mid melody.mid --adapt-mode contour
  python melody_harmonizer.py harmony.mid melody.mid --ornaments passing appoggiatura --ornament-bars 3 7 11 15
  python melody_harmonizer.py harmony.mid melody.mid --ornaments all --ornament-prob 0.5
  python melody_harmonizer.py harmony.mid melody.mid --key "D minor" --verbose --report
        """
    )
    p.add_argument('harmony', help='MIDI con la armonía (acordes)')
    p.add_argument('melody',  help='MIDI con la melodía a adaptar')
    p.add_argument('--adapt-mode', default='contour',
                   choices=['nearest', 'above', 'below', 'contour', 'scale', 'reposo'],
                   help='Estrategia de adaptación melódica (default: contour)')
    p.add_argument('--ornaments', nargs='+', default=[],
                   choices=['appoggiatura', 'passing', 'neighbor', 'all'],
                   metavar='ORNAMENT',
                   help='Tipos de ornamento: appoggiatura passing neighbor all')
    p.add_argument('--ornament-bars', nargs='+', type=int, default=None,
                   metavar='N',
                   help='Compases (1-based) donde aplicar ornamentos. '
                        'Si no se indica, se aplican en todos los compases.')
    p.add_argument('--ornament-prob', type=float, default=0.4,
                   metavar='F',
                   help='Probabilidad de ornamentación por nota candidata (0-1, default: 0.4)')
    p.add_argument('--key', default=None, metavar='KEY',
                   help='Tonalidad forzada, e.g. "G major" o "D minor"')
    p.add_argument('--tempo', type=float, default=None,
                   help='BPM forzado (default: detectado del MIDI armonía)')
    p.add_argument('--out-dir', default=None,
                   help='Carpeta de salida (default: misma carpeta que el MIDI melodía)')
    p.add_argument('--report', action='store_true',
                   help='Guardar reporte JSON con el análisis completo')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    harmonize_melody(
        harmony_midi=args.harmony,
        melody_midi=args.melody,
        adapt_mode=args.adapt_mode,
        ornament_types=args.ornaments if args.ornaments else [],
        ornament_bars=args.ornament_bars,
        ornament_prob=args.ornament_prob,
        key_override=args.key,
        tempo_override=args.tempo,
        out_dir=args.out_dir,
        report=args.report,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
