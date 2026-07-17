#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          METER ADAPTER  v1.1                                 ║
║        Adaptación métrica de MIDI: reescritura rítmica entre compases        ║
║                                                                              ║
║  Implementa las guías de "Adapting Music from one Meter to Another" (Active  ║
║  Analysis): tomar una pieza y reescribirla en OTRO compás añadiendo o        ║
║  quitando pulsos, preservando en la medida de lo posible:                   ║
║    1. El material asociado a tiempos fuertes/downbeats (incl. anacrusas)    ║
║    2. La textura rítmica (sin exceder la subdivisión más rápida original)   ║
║    3. La forma melódica / contorno                                          ║
║                                                                              ║
║  ESTRATEGIA:                                                                 ║
║    Al AÑADIR pulsos (compás destino más largo):                             ║
║      · si ya hay silencio de cola en la unidad → se usa ese margen          ║
║        (equivale a "separar más los motivos", sin tocar ninguna nota)       ║
║      · si la unidad está saturada → se alarga la nota de tiempo más débil,  ║
║        o se le añade una repetición melismática (--melisma) si el hueco     ║
║        es grande, en vez de un único valor larguísimo                      ║
║    Al QUITAR pulsos (compás destino más corto):                             ║
║      · si ya hay silencio de cola suficiente → simplemente se acorta el     ║
║        contenedor, sin tocar ninguna nota                                   ║
║      · si la unidad está saturada → se eliminan notas por orden de         ║
║        prioridad: repetidas > de paso (stepwise) > tiempo débil, siempre   ║
║        preservando la primera y última nota de la unidad (anclas de        ║
║        tiempo fuerte); si aún sobra duración, se comprime proporcionalmente ║
║                                                                              ║
║  AGRUPACIÓN (--group N): permite reproducir ejemplos como "cada dos         ║
║  compases de origen se convierten en un compás de destino" (p.ej. 2×4/4 →   ║
║  1×7/4, o 2×3/4 → 1×5/4 de la charla sobre My Favorite Things / montuno).   ║
║                                                                              ║
║  ACOMPAÑAMIENTO (--accompaniment-style):                                    ║
║    mirror   (default) — cada pista se adapta con el mismo motor que la      ║
║              melodía (independientemente, según su propio contenido)       ║
║    waltz    — colapsa las pistas de acompañamiento a un patrón de           ║
║              "bajo en el tiempo 1 + un acorde por cada tiempo restante"     ║
║              (técnica del Rondo alla Turca → vals)                         ║
║    none     — descarta las pistas de acompañamiento, solo adapta la melodía║
║                                                                              ║
║  --coordinate-layers: deriva el mapa de tiempo EXACTO de la melodía y lo    ║
║  aplica igual a todas las capas 'mirror' (incl. percusión), en vez de que   ║
║  cada una decida qué añadir/quitar por su cuenta. Preserva la relación      ║
║  temporal relativa entre capas — pensado para patrones entrelazados como    ║
║  montuno/tumbao/clave/cáscara, donde importa que las capas se mantengan     ║
║  sincronizadas como conjunto.                                               ║
║                                                                              ║
║  USO:                                                                        ║
║    python meter_adapter.py tema.mid --to 5/4                                ║
║    python meter_adapter.py tema.mid --to 5/8 --from 6/8                    ║
║    python meter_adapter.py rondo.mid --to 3/4 --accompaniment-style waltz  ║
║    python meter_adapter.py montuno.mid --to 7/4 --group 2                  ║
║    python meter_adapter.py tema.mid --preset waltz                          ║
║    python meter_adapter.py tema.mid --to 4/4 --no-melisma --verbose        ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --to N/D            Compás destino (obligatorio), e.g. "5/4", "7/8"     ║
║    --from N/D          Compás origen (default: detectado del MIDI)         ║
║    --group N           Compases origen agrupados por cada compás destino    ║
║                        (default: 1)                                         ║
║    --melisma/--no-melisma   Permitir subdivisión melismática al añadir     ║
║                        pulsos grandes (default: activado)                   ║
║    --melisma-threshold B    Umbral en negras para preferir melisma sobre    ║
║                        una sola nota larga (default: 1.0)                   ║
║    --accompaniment-style {mirror,waltz,none}   (default: mirror)            ║
║    --melody-track N    Forzar qué pista es la melodía (default: autodetecta)║
║    --preset {waltz}    Atajo: --to 3/4 --accompaniment-style waltz          ║
║    --out PATH          Fichero de salida (default: <nombre>.meter_N-D.mid) ║
║    --report            Guardar reporte JSON                                 ║
║    --verbose           Informe detallado por stdout                        ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import math
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import mido


PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PRESETS = {
    'waltz': dict(to='3/4', accompaniment_style='waltz'),
}


# ═══════════════════════════════════════════════════════════
# UTILIDADES MÉTRICAS
# ═══════════════════════════════════════════════════════════

def parse_meter(s):
    if not s or '/' not in s:
        raise ValueError(f"Formato de compás inválido: '{s}' (usa N/D, e.g. 5/4)")
    n, d = s.split('/')
    return int(n), int(d)


def bar_qlen(num, den):
    """Longitud de un compás en negras (quarter-length), independiente de tempo."""
    return num * 4.0 / den


def beat_strength(offset, unit_len, tol=0.05):
    """
    Heurística de fuerza métrica DENTRO de una unidad (compás o grupo de
    compases) de longitud unit_len: 1.0 en el primer pulso (downbeat de la
    unidad), 0.6 en otros pulsos de negra, 0.25 en contratiempos. Es una
    aproximación deliberada (no re-analiza el compás notado internamente),
    suficiente para decidir qué notas son más "prescindibles".
    """
    if abs(offset % unit_len) < tol or abs((offset % unit_len) - unit_len) < tol:
        return 1.0
    if abs(offset % 1.0) < tol:
        return 0.6
    return 0.25


def window_pc_weights(notes, win_start, win_end, unit_len):
    weights = defaultdict(float)
    for (off, p, dur, vel) in notes:
        overlap = min(off + dur, win_end) - max(off, win_start)
        if overlap <= 0:
            continue
        weights[p % 12] += overlap * (1.0 + beat_strength(off, unit_len))
    return weights


def nearest_pitch_to_register(pc, center):
    pc = pc % 12
    base = center - (center % 12) + pc
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda x: abs(x - center))


# ═══════════════════════════════════════════════════════════
# CARGA DE MIDI (TODAS LAS PISTAS, NO SOLO LA MELODÍA)
# ═══════════════════════════════════════════════════════════

def load_all_layers(midi_path, verbose=False):
    """
    A diferencia de theme_transformer.py (que solo necesita LA melodía),
    aquí necesitamos TODAS las pistas con contenido para poder adaptar
    también el acompañamiento. Devuelve una lista de capas:
      {'index', 'notes', 'program', 'is_percussion', 'mean_pitch'}
    notes: list of (offset_beats, pitch, duration_beats, velocity)
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    layers = []

    for track_idx, track in enumerate(mid.tracks):
        abs_t = 0
        pending = {}
        notes = []
        program = 0
        has_perc = False
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'program_change':
                program = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
                if msg.channel == 9:
                    has_perc = True
            elif msg.type in ('note_off', 'note_on'):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    on_t, vel = pending.pop(key_)
                    notes.append((on_t / tpb, msg.note, max(0.1, (abs_t - on_t) / tpb), vel))
        if notes:
            notes.sort(key=lambda n: n[0])
            layers.append({
                'index': track_idx, 'notes': notes, 'program': program,
                'is_percussion': has_perc,
                'mean_pitch': float(np.mean([p for _, p, _, _ in notes])),
            })

    if not layers:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    tempo_bpm = round(60_000_000 / tempo_us, 2)
    if verbose:
        print(f"  {len(layers)} pista(s) con notas · tempo {tempo_bpm} BPM · "
              f"compás original {ts_num}/{ts_den}")
        for l in layers:
            tag = " [percusión]" if l['is_percussion'] else ""
            print(f"    pista {l['index']}: {len(l['notes'])} notas, "
                  f"programa GM {l['program']}, pitch medio {l['mean_pitch']:.1f}{tag}")

    return layers, tempo_bpm, (ts_num, ts_den), tpb


# ═══════════════════════════════════════════════════════════
# ADAPTACIÓN DE UNA UNIDAD (compás o grupo de compases)
# ═══════════════════════════════════════════════════════════

def adapt_unit_add(notes, source_len, target_len, allow_melisma=True, melisma_threshold=1.0,
                   return_warp=False):
    """
    Unidad más larga en el destino. Si ya hay silencio de cola suficiente,
    no se toca ninguna nota (se limita a agrandar el contenedor). Si no,
    se alarga la nota más débil métricamente (nunca la primera, que ancla
    el downbeat/anacrusa) o se le añade una repetición melismática.
    Si return_warp=True, también devuelve el mapa de tiempo (lista de
    puntos (old_t, new_t)) que resultó de esta decisión — usado por
    --coordinate-layers para aplicar EXACTAMENTE la misma deformación
    temporal a las demás pistas.
    """
    if not notes:
        warp = [(0.0, 0.0), (source_len, target_len)]
        return ([], warp) if return_warp else []
    notes = sorted(notes, key=lambda n: n[0])
    content_end = max(o + d for o, p, d, v in notes)
    slack = max(0.0, source_len - content_end)
    delta = target_len - source_len

    if delta <= slack + 1e-6:
        if return_warp:
            warp = [(0.0, 0.0), (content_end, content_end), (source_len, target_len)]
            return list(notes), warp
        return list(notes)

    extra = delta - slack
    if len(notes) > 1:
        idx = max(range(1, len(notes)), key=lambda i: (1.0 - beat_strength(notes[i][0], source_len)))
    else:
        idx = 0

    out = []
    off_idx, p_idx, dur_idx, vel_idx = notes[idx]
    insertion_point = off_idx + dur_idx
    for i, (off, p, dur, vel) in enumerate(notes):
        if i < idx:
            out.append((off, p, dur, vel))
        elif i == idx:
            if allow_melisma and extra > melisma_threshold:
                half = extra / 2.0
                out.append((off, p, dur + half, vel))
                out.append((off + dur + half, p, max(0.15, half), max(40, int(vel) - 10)))
            else:
                out.append((off, p, dur + extra, vel))
        else:
            out.append((off + extra, p, dur, vel))

    if return_warp:
        warp = [(0.0, 0.0), (insertion_point, insertion_point),
               (insertion_point, insertion_point + extra), (source_len, target_len)]
        return out, warp
    return out


def _removal_score(idx, notes, unit_len):
    """Cuanto más alto, más 'prescindible' es la nota en idx."""
    if idx == 0 or idx == len(notes) - 1:
        return -100.0  # nunca la primera ni la última: anclan tiempos fuertes
    off, p, dur, vel = notes[idx]
    score = (1.0 - beat_strength(off, unit_len)) * 2.0
    if notes[idx - 1][1] == p:
        score += 3.0  # nota repetida: candidata favorita a eliminar
    prev_p, next_p = notes[idx - 1][1], notes[idx + 1][1]
    step_prev, step_next = p - prev_p, next_p - p
    if abs(step_prev) <= 2 and abs(step_next) <= 2 and step_prev != 0:
        if (step_prev > 0) == (step_next > 0):
            score += 1.5  # nota de paso (stepwise, mismo sentido): segunda prioridad
    score += max(0.0, 0.5 - dur) * 0.5  # notas cortas, ligero extra
    return score


def adapt_unit_remove(notes, source_len, target_len, return_warp=False):
    """
    Unidad más corta en el destino. Si el margen de silencio de cola ya
    cubre la reducción, no se toca ninguna nota. Si no, se eliminan notas
    (nunca la primera ni la última) por prioridad: repetidas > de paso >
    tiempo débil, cerrando el hueco cada vez (el resto se desliza antes).
    Si aún sobra duración tras vaciar candidatos, se comprime
    proporcionalmente lo que quede. return_warp: ver adapt_unit_add.
    """
    if not notes:
        warp = [(0.0, 0.0), (source_len, target_len)]
        return ([], warp) if return_warp else []
    notes = sorted(notes, key=lambda n: n[0])
    content_end = max(o + d for o, p, d, v in notes)
    needed = content_end - target_len

    if needed <= 1e-6:
        if return_warp:
            warp = [(0.0, 0.0), (content_end, content_end), (source_len, target_len)]
            return list(notes), warp
        return list(notes)

    working = list(notes)
    removed_total = 0.0
    deleted_intervals = []  # en coordenadas ORIGINALES de la unidad
    while removed_total < needed - 1e-6 and len(working) > 2:
        idx = max(range(len(working)), key=lambda i: _removal_score(i, working, source_len))
        if _removal_score(idx, working, source_len) <= -50:
            break  # solo quedan la primera/última: no seguir
        off, p, dur, vel = working.pop(idx)
        orig_start = off + removed_total
        deleted_intervals.append((orig_start, orig_start + dur))
        for j in range(idx, len(working)):
            o2, p2, d2, v2 = working[j]
            working[j] = (o2 - dur, p2, d2, v2)
        removed_total += dur

    content_end2 = max((o + d for o, p, d, v in working), default=0.0)
    scale = 1.0
    if working and content_end2 > target_len + 1e-6:
        scale = target_len / content_end2
        working = [(o * scale, p, d * scale, v) for (o, p, d, v) in working]

    if not return_warp:
        return working

    deleted_intervals.sort(key=lambda iv: iv[0])
    cum = 0.0
    pts = [(0.0, 0.0)]
    for ds, de in deleted_intervals:
        new_ds = (ds - cum) * scale
        pts.append((ds, new_ds))
        cum += (de - ds)
        pts.append((de, new_ds))
    pts.append((source_len, target_len))
    return working, pts


def adapt_layer(notes, group, from_bar, to_bar, allow_melisma, melisma_threshold):
    """Procesa una pista completa unidad por unidad (unidad = 'group' compases
    de origen → exactamente 1 compás de destino) y concatena el resultado."""
    if not notes:
        return [], {'units': 0, 'added': 0, 'removed': 0, 'unchanged': 0}

    source_unit = group * from_bar
    target_unit = to_bar
    total_len = max(o + d for o, p, d, v in notes)
    n_units = max(1, math.ceil(total_len / source_unit))

    out = []
    cursor = 0.0
    stats = {'units': 0, 'added': 0, 'removed': 0, 'unchanged': 0}
    for u in range(n_units):
        u_start, u_end = u * source_unit, (u + 1) * source_unit
        unit_notes = [(o - u_start, p, d, v) for (o, p, d, v) in notes if u_start <= o < u_end]
        if not unit_notes:
            cursor += target_unit
            continue
        if target_unit > source_unit + 1e-9:
            new_unit = adapt_unit_add(unit_notes, source_unit, target_unit, allow_melisma, melisma_threshold)
            stats['added'] += 1
        elif target_unit < source_unit - 1e-9:
            new_unit = adapt_unit_remove(unit_notes, source_unit, target_unit)
            stats['removed'] += 1
        else:
            new_unit = unit_notes
            stats['unchanged'] += 1
        stats['units'] += 1
        for (o, p, d, v) in new_unit:
            out.append((cursor + o, p, d, v))
        cursor += target_unit
    return out, stats


# ═══════════════════════════════════════════════════════════
# COORDINACIÓN MULTICAPA (--coordinate-layers)
# ═══════════════════════════════════════════════════════════
# El ejemplo afrocubano (montuno/tumbao/clave/cáscara) exige que las capas
# mantengan su relación compuesta (p.ej. "solo hay un punto donde coinciden
# las cuatro"). adapt_layer() por defecto deja que cada pista decida de
# forma independiente qué nota añadir/cortar, lo que puede romper esa
# relación. Aquí se deriva, unidad por unidad, el mapa de tiempo EXACTO que
# resultó de las decisiones tomadas sobre la MELODÍA, y se aplica ese mismo
# mapa a todas las demás pistas — así, dos eventos que eran simultáneos en
# el original siguen siéndolo en el resultado.

def apply_time_warp(notes, warp_pts):
    """
    Aplica un mapa de tiempo por tramos (lista de (old_t,new_t) no decreciente
    en old_t, con posibles saltos: old_t repetido = inserción, new_t repetido
    = eliminación) a una lista de notas con offsets relativos a la unidad.
    """
    def warp_time(t):
        for i in range(len(warp_pts) - 1):
            o0, n0 = warp_pts[i]
            o1, n1 = warp_pts[i + 1]
            if o0 <= t <= o1:
                if o1 == o0:
                    return n1 if t > o0 else n0
                frac = (t - o0) / (o1 - o0)
                return n0 + frac * (n1 - n0)
        return warp_pts[-1][1]

    out = []
    for (off, p, dur, vel) in notes:
        new_off = warp_time(off)
        new_end = warp_time(off + dur)
        new_dur = max(0.05, new_end - new_off)
        out.append((new_off, p, new_dur, vel))
    return out


def compute_unit_warps(reference_notes, group, from_bar, to_bar, allow_melisma, melisma_threshold):
    """Corre el motor de adaptación SOLO sobre la pista de referencia
    (normalmente la melodía) para derivar, unidad por unidad, el mapa de
    tiempo resultante de sus decisiones de añadir/quitar contenido."""
    source_unit = group * from_bar
    target_unit = to_bar
    if not reference_notes:
        return []
    total_len = max(o + d for o, p, d, v in reference_notes)
    n_units = max(1, math.ceil(total_len / source_unit))

    warps = []
    for u in range(n_units):
        u_start, u_end = u * source_unit, (u + 1) * source_unit
        unit_notes = [(o - u_start, p, d, v) for (o, p, d, v) in reference_notes
                     if u_start <= o < u_end]
        if not unit_notes:
            warps.append([(0.0, 0.0), (source_unit, target_unit)])
            continue
        if target_unit > source_unit + 1e-9:
            _new, warp = adapt_unit_add(unit_notes, source_unit, target_unit,
                                        allow_melisma, melisma_threshold, return_warp=True)
        elif target_unit < source_unit - 1e-9:
            _new, warp = adapt_unit_remove(unit_notes, source_unit, target_unit, return_warp=True)
        else:
            warp = [(0.0, 0.0), (source_unit, target_unit)]
        warps.append(warp)
    return warps


def adapt_layer_with_warps(notes, group, from_bar, to_bar, warps):
    """Aplica a esta pista los MISMOS mapas de tiempo por unidad calculados
    para la pista de referencia (--coordinate-layers), en vez de tomar sus
    propias decisiones de qué añadir/quitar. Garantiza que la relación
    temporal relativa entre pistas se preserve como conjunto."""
    if not notes:
        return []
    source_unit = group * from_bar
    target_unit = to_bar
    total_len = max(o + d for o, p, d, v in notes)
    n_units = max(len(warps), math.ceil(total_len / source_unit) if source_unit > 0 else 1)

    out, cursor = [], 0.0
    for u in range(n_units):
        u_start, u_end = u * source_unit, (u + 1) * source_unit
        unit_notes = [(o - u_start, p, d, v) for (o, p, d, v) in notes if u_start <= o < u_end]
        warp = warps[u] if u < len(warps) else [(0.0, 0.0), (source_unit, target_unit)]
        if unit_notes:
            new_unit = apply_time_warp(unit_notes, warp)
            for (o, p, d, v) in new_unit:
                out.append((cursor + o, p, d, v))
        cursor += target_unit
    return out


def waltz_accompaniment(notes, group, from_bar, to_bar, to_beats_count):
    """
    Técnica del Rondo alla Turca → vals: bajo en el primer tiempo del
    compás destino + exactamente UN acorde por cada tiempo restante,
    tomando el contenido dominante de la ventana proporcional de origen
    correspondiente a cada tiempo.
    """
    if not notes:
        return []
    source_unit = group * from_bar
    target_unit = to_bar
    to_beats_count = max(1, int(round(to_beats_count)))
    beat_len = target_unit / to_beats_count
    total_len = max(o + d for o, p, d, v in notes)
    n_units = max(1, math.ceil(total_len / source_unit))

    out = []
    cursor = 0.0
    for u in range(n_units):
        u_start, u_end = u * source_unit, (u + 1) * source_unit
        unit_notes = [(o - u_start, p, d, v) for (o, p, d, v) in notes if u_start <= o < u_end]
        if not unit_notes:
            cursor += target_unit
            continue
        for beat_i in range(to_beats_count):
            win_start = beat_i * (source_unit / to_beats_count)
            win_end = (beat_i + 1) * (source_unit / to_beats_count)
            weights = window_pc_weights(unit_notes, win_start, win_end, source_unit)
            if not weights:
                continue
            slot = cursor + beat_i * beat_len
            if beat_i == 0:
                cand = [p for (o, p, d, v) in unit_notes if win_start <= o < win_end]
                pitch = min(cand) if cand else 48
                out.append((slot, pitch, beat_len * 0.95, 82))
            else:
                top_pcs = sorted(weights, key=weights.get, reverse=True)[:3]
                for pc in top_pcs:
                    pitch = nearest_pitch_to_register(pc, 55)
                    out.append((slot, pitch, beat_len * 0.9, 60))
        cursor += target_unit
    return out


# ═══════════════════════════════════════════════════════════
# ESCRITURA DE MIDI
# ═══════════════════════════════════════════════════════════

def write_meter_midi(out_layers, tempo_bpm, to_num, to_den, out_path):
    tpb = 480
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(0, int(round(beats * tpb)))

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    header = mido.MidiTrack()
    header.append(mido.MetaMessage('track_name', name='MeterAdapter', time=0))
    header.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    header.append(mido.MetaMessage('time_signature', numerator=to_num, denominator=to_den,
                                   clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    header.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(header)

    for ch, layer in enumerate(out_layers):
        events = []
        for off, p, dur, vel in layer['notes']:
            pitch = max(0, min(127, int(round(p))))
            v = max(1, min(127, int(round(vel))))
            t_on = to_ticks(off)
            t_off = to_ticks(off + max(0.05, dur))
            if t_off <= t_on:
                t_off = t_on + 1
            events.append((t_on, 'on', pitch, v))
            events.append((t_off, 'off', pitch, 0))
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

        trk = mido.MidiTrack()
        role = 'Melody' if layer['is_melody'] else 'Accompaniment'
        trk.append(mido.MetaMessage('track_name', name=f"{role} (pista {layer['index']})", time=0))
        ch_num = min(ch, 15)
        trk.append(mido.Message('program_change', channel=ch_num, program=layer['program'], time=0))
        prev_t = 0
        for abs_t, kind, p, v in events:
            dt = max(0, abs_t - prev_t)
            msg_type = 'note_on' if kind == 'on' else 'note_off'
            trk.append(mido.Message(msg_type, channel=ch_num, note=p, velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(trk)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mid.save(out_path)


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def adapt_meter(midi_path: str,
                to_meter: str,
                from_meter: str = None,
                group: int = 1,
                allow_melisma: bool = True,
                melisma_threshold: float = 1.0,
                accompaniment_style: str = 'mirror',
                coordinate_layers: bool = False,
                melody_track: int = None,
                out_path: str = None,
                report: bool = False,
                verbose: bool = False):
    layers, tempo_bpm, ts, tpb = load_all_layers(midi_path, verbose=verbose)

    from_num, from_den = parse_meter(from_meter) if from_meter else ts
    to_num, to_den = parse_meter(to_meter)
    from_bar = bar_qlen(from_num, from_den)
    to_bar = bar_qlen(to_num, to_den)

    if verbose:
        delta = to_bar - group * from_bar
        print(f"\n═══ {os.path.basename(midi_path)}: {from_num}/{from_den} → {to_num}/{to_den} "
              f"(grupo={group}) ═══")
        print(f"  Unidad origen: {group * from_bar:.2f} negras   "
              f"Unidad destino: {to_bar:.2f} negras   "
              f"({'+' if delta >= 0 else ''}{delta:.2f})")

    non_perc = [l for l in layers if not l['is_percussion']]
    pool = non_perc or layers
    if melody_track is not None:
        melody_layer = next((l for l in layers if l['index'] == melody_track), None)
        if melody_layer is None:
            raise ValueError(f"No existe la pista {melody_track} en {midi_path}")
    else:
        melody_layer = max(pool, key=lambda l: l['mean_pitch'])

    shared_warps = None
    if coordinate_layers:
        shared_warps = compute_unit_warps(melody_layer['notes'], group, from_bar, to_bar,
                                          allow_melisma, melisma_threshold)
        if verbose:
            print(f"  --coordinate-layers: {len(shared_warps)} mapa(s) de tiempo derivados de "
                  f"la pista {melody_layer['index']} (melodía), compartidos por todas las capas")

    out_layers = []
    for layer in layers:
        is_melody = layer is melody_layer
        if coordinate_layers and (is_melody or accompaniment_style == 'mirror' or layer['is_percussion']):
            new_notes = adapt_layer_with_warps(layer['notes'], group, from_bar, to_bar, shared_warps)
            method_used = 'coordinated'
        elif is_melody or accompaniment_style == 'mirror' or layer['is_percussion']:
            new_notes, _stats = adapt_layer(layer['notes'], group, from_bar, to_bar,
                                            allow_melisma, melisma_threshold)
            method_used = 'mirror'
        elif accompaniment_style == 'waltz':
            new_notes = waltz_accompaniment(layer['notes'], group, from_bar, to_bar, to_num)
            method_used = 'waltz'
        else:  # 'none'
            continue

        out_layers.append({
            'index': layer['index'], 'notes': new_notes, 'program': layer['program'],
            'is_melody': is_melody, 'method': method_used,
        })
        if verbose:
            role = 'MELODÍA' if is_melody else 'acompañamiento'
            print(f"  pista {layer['index']} ({role}, método={method_used}): "
                  f"{len(layer['notes'])} → {len(new_notes)} notas")

    if not out_layers:
        raise RuntimeError("No queda ninguna pista tras aplicar --accompaniment-style none "
                          "(¿la melodía era la única pista?)")

    stem = Path(midi_path).stem
    output_path = out_path or f"{stem}.meter_{to_num}-{to_den}.mid"
    write_meter_midi(out_layers, tempo_bpm, to_num, to_den, output_path)

    if verbose:
        print(f"  → {output_path}")

    result = {
        'input': midi_path, 'output': output_path,
        'from_meter': f"{from_num}/{from_den}", 'to_meter': f"{to_num}/{to_den}",
        'group': group, 'accompaniment_style': accompaniment_style,
        'coordinate_layers': coordinate_layers,
        'allow_melisma': allow_melisma, 'melody_track': melody_layer['index'],
        'n_layers_out': len(out_layers),
        'n_layers_total': len(layers),
    }

    if report:
        report_path = os.path.splitext(output_path)[0] + '_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  → {report_path}")

    return result


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='METER ADAPTER — Adaptación métrica (cambio de compás) de MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python meter_adapter.py tema.mid --to 5/4
  python meter_adapter.py tema.mid --to 5/8 --from 6/8
  python meter_adapter.py rondo.mid --to 3/4 --accompaniment-style waltz
  python meter_adapter.py rondo.mid --preset waltz
  python meter_adapter.py montuno.mid --to 7/4 --group 2
  python meter_adapter.py montuno.mid --to 7/8 --coordinate-layers --verbose
  python meter_adapter.py tema.mid --to 4/4 --no-melisma --verbose
        """
    )
    p.add_argument('input', nargs='?', help='Archivo(s) MIDI de entrada (admite glob)')
    p.add_argument('--to', default=None, metavar='N/D', help='Compás destino, e.g. "5/4"')
    p.add_argument('--from', dest='from_meter', default=None, metavar='N/D',
                   help='Compás origen (default: detectado del MIDI)')
    p.add_argument('--group', type=int, default=1,
                   help='Compases origen agrupados por cada compás destino (default: 1)')
    p.add_argument('--melisma', dest='melisma', action='store_true', default=True,
                   help='Permitir subdivisión melismática al añadir pulsos grandes (default)')
    p.add_argument('--no-melisma', dest='melisma', action='store_false',
                   help='Desactivar melisma: siempre alargar una única nota')
    p.add_argument('--melisma-threshold', type=float, default=1.0,
                   help='Umbral en negras para preferir melisma (default: 1.0)')
    p.add_argument('--accompaniment-style', choices=['mirror', 'waltz', 'none'], default='mirror',
                   help='Estrategia para pistas de acompañamiento (default: mirror)')
    p.add_argument('--coordinate-layers', action='store_true',
                   help="Deriva el mapa de tiempo de la melodía y lo aplica IGUAL a todas las "
                        "capas 'mirror' (incl. percusión), en vez de que cada una decida qué "
                        "añadir/quitar por su cuenta — preserva la relación temporal entre "
                        "capas (p.ej. patrones afrocubanos entrelazados). No afecta a 'waltz'")
    p.add_argument('--melody-track', type=int, default=None,
                   help='Forzar qué pista es la melodía (default: autodetecta por pitch medio)')
    p.add_argument('--preset', choices=list(PRESETS.keys()), default=None,
                   help="Atajo de configuración, e.g. 'waltz' = --to 3/4 --accompaniment-style waltz")
    p.add_argument('--out', default=None, help='Fichero de salida (default: <nombre>.meter_N-D.mid)')
    p.add_argument('--report', action='store_true', help='Guardar reporte JSON')
    p.add_argument('--verbose', action='store_true', help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    to_meter = args.to
    accompaniment_style = args.accompaniment_style
    if args.preset:
        preset = PRESETS[args.preset]
        to_meter = to_meter or preset.get('to')
        if args.accompaniment_style == 'mirror' and 'accompaniment_style' in preset:
            accompaniment_style = preset['accompaniment_style']

    if not to_meter:
        print("[ERROR] Debes especificar --to N/D (o --preset).")
        sys.exit(1)

    import glob
    midi_files = glob.glob(args.input) or [args.input]
    midi_files = [f for f in midi_files if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        print(f"[ERROR] No se encontraron archivos MIDI: {args.input}")
        sys.exit(1)

    for midi_path in midi_files:
        try:
            adapt_meter(
                midi_path=midi_path,
                to_meter=to_meter,
                from_meter=args.from_meter,
                group=args.group,
                allow_melisma=args.melisma,
                melisma_threshold=args.melisma_threshold,
                accompaniment_style=accompaniment_style,
                coordinate_layers=args.coordinate_layers,
                melody_track=args.melody_track,
                out_path=args.out if len(midi_files) == 1 else None,
                report=args.report,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"[ERROR] {midi_path}: {e}")
            if args.verbose:
                traceback.print_exc()


if __name__ == '__main__':
    main()
