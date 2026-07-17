#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        THEME TRANSFORMER  v2.0                               ║
║       Mapeo escalar (scalar mapping) y transformación temática de MIDI       ║
║                                                                              ║
║  Implementa las técnicas de "thematic transformation via scalar mapping"     ║
║  descritas en la serie de tutoriales sobre transformación de temas: toma    ║
║  una melodía MIDI y la traslada a otras escalas/modos preservando su        ║
║  identidad, con armonización y orquestación acordes a cada estilo.          ║
║                                                                              ║
║  NOVEDADES v2.0:                                                             ║
║    --sections        transformación PROGRESIVA por tramos de la pieza       ║
║                       (como la cadena de reharmonizaciones de LOTR/GoT,     ║
║                       no un estilo fijo de principio a fin)                 ║
║    chromatic_mediant  nuevo estilo: cadena de tríadas por 3ª cromática      ║
║                       sin función tonal (Zimmer/McCreary)                   ║
║    --body-tail        distingue cuerpo (identidad protegida) de cola        ║
║                       (libertad ornamental), como en HTTYD                  ║
║    --wt-variant / --octatonic-variant   elegir manualmente la variante      ║
║                       de tonos enteros / octatónica en vez de autodetectar  ║
║                                                                              ║
║  MÉTODOS DE MAPEO (--method, o definidos por cada --style):                 ║
║    degree    — cada nota se re-ancla a su grado de escala más cercano       ║
║                (independiente por nota; ideal para transposición/cambio     ║
║                de modo simple)                                              ║
║    interval  — se preservan los intervalos ENTRE notas consecutivas         ║
║                medidos en pasos de la escala destino (mejor cuando la       ║
║                escala destino tiene distinta cardinalidad que la origen)    ║
║                                                                              ║
║  ESTILOS DISPONIBLES (--style):                                             ║
║    heroic_minor          menor paralela + tríadas mayores (nota fuerte=3ª) ║
║    dorian_heroic         dórico + tríadas mayores (estilo Fellowship)      ║
║    whole_tone            tonos enteros + dominantes 7/9 sin 5ª (Debussy)   ║
║    octatonic             colección octatónica + acordes nativos (Shapiro) ║
║    chromatic_magical     cromática + tríadas menores paralelas (Hedwig)   ║
║    chromatic_counterpoint cromática + contrapunto independiente (Bartók)  ║
║    pentatonic             pentatónica mayor/menor + armonía dispersa      ║
║    double_harmonic        doble armónica / mayamalavagowla                ║
║    acoustic                modo acústico (lidio dominante)                ║
║    elvish                  5º modo de la menor melódica, reorientado      ║
║    got_fragment            fragmentación + menor armónica + desplazo. métrico║
║    custom                  escala/método libres vía --scale / --method    ║
║                                                                              ║
║  TÉCNICAS ADICIONALES:                                                      ║
║    --chord-tone-priority   fuerza notas en tiempo fuerte a un tono         ║
║                             del acorde vigente (corrección ≤3 semitonos)   ║
║    --fragment / --fragment-order   fragmentación motívica al estilo GoT   ║
║    --metric-shift          alinea la primera nota al downbeat             ║
║    --start-degree          desplaza todos los grados de escala destino    ║
║                                                                              ║
║  USO:                                                                        ║
║    python theme_transformer.py tema.mid                                     ║
║    python theme_transformer.py tema.mid --style octatonic chromatic_magical ║
║    python theme_transformer.py tema.mid --style got_fragment --verbose     ║
║    python theme_transformer.py tema.mid --style whole_tone \\               ║
║        --chord-tone-priority                                                ║
║    python theme_transformer.py tema.mid --style custom \\                   ║
║        --scale double_harmonic --method interval                            ║
║    python theme_transformer.py tema.mid --style elvish --start-degree 2    ║
║    python theme_transformer.py --catalog                                    ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --style S [S…]     Estilos a aplicar (default: heroic_minor whole_tone  ║
║                       octatonic chromatic_magical)                         ║
║    --key "D minor"    Tonalidad origen forzada (default: autodetecta)      ║
║    --tempo BPM        Tempo de salida forzado (default: del MIDI)          ║
║    --beats-per-bar N  Beats por compás (default: del MIDI)                 ║
║    --start-degree N   Desplazamiento extra de grado (default: 0)           ║
║    --chord-tone-priority   Ver arriba                                       ║
║    --fragment / --fragment-order   Ver arriba                               ║
║    --metric-shift     Ver arriba                                            ║
║    --scale / --method (solo con --style custom)                            ║
║    --out-dir DIR      Carpeta de salida (default: junto al MIDI)           ║
║    --report           Guardar reporte JSON por transformación               ║
║    --verbose          Informe detallado por stdout                          ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    tema.octatonic.mid                                                       ║
║    tema.chromatic_magical.mid                                               ║
║    tema.octatonic_report.json   (con --report)                              ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21 (opcional), numpy                              ║
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

# ── música21 (opcional, solo para detección de tonalidad más precisa) ────────
try:
    from music21 import converter, key as m21key, pitch as m21pitch
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Se usará detección de tonalidad "
          "por perfil de Krumhansl-Schmuckler simplificado.")


# ═══════════════════════════════════════════════════════════
# CONSTANTES: NOMBRES DE NOTA Y ESCALAS
# ═══════════════════════════════════════════════════════════

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

NAME_TO_PC = {
    'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
    'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}

# Cada escala: tupla de semitonos ascendentes desde la tónica (0 primero)
SCALES = {
    'major':             (0, 2, 4, 5, 7, 9, 11),
    'natural_minor':     (0, 2, 3, 5, 7, 8, 10),
    'harmonic_minor':    (0, 2, 3, 5, 7, 8, 11),
    'melodic_minor':     (0, 2, 3, 5, 7, 9, 11),
    'dorian':            (0, 2, 3, 5, 7, 9, 10),
    'phrygian':          (0, 1, 3, 5, 7, 8, 10),
    'lydian':            (0, 2, 4, 6, 7, 9, 11),
    'mixolydian':        (0, 2, 4, 5, 7, 9, 10),
    'locrian':           (0, 1, 3, 5, 6, 8, 10),
    'acoustic':          (0, 2, 4, 6, 7, 9, 10),   # 4º modo de la menor melódica (lidio dominante)
    'melodic_minor_m5':  (0, 2, 4, 5, 7, 8, 10),   # 5º modo de la menor melódica (mixolidio b6, "elfico")
    'double_harmonic':   (0, 1, 4, 5, 7, 8, 11),   # mayamalavagowla / escala bizantina
    'whole_tone':        (0, 2, 4, 6, 8, 10),
    'octatonic_wh':      (0, 2, 3, 5, 6, 8, 9, 11),  # tono-semitono
    'octatonic_hw':      (0, 1, 3, 4, 6, 7, 9, 10),  # semitono-tono
    'chromatic':         tuple(range(12)),
    'major_pentatonic':  (0, 2, 4, 7, 9),
    'minor_pentatonic':  (0, 3, 5, 7, 10),
}

# Escalas "virtuales" resueltas dinámicamente en tiempo de ejecución
_DYNAMIC_SCALES = {'octatonic_auto', 'pentatonic_auto', 'whole_tone'}


# ═══════════════════════════════════════════════════════════
# CATÁLOGO DE ESTILOS
# ═══════════════════════════════════════════════════════════
# Cada estilo define: escala destino, método de mapeo, armonizador,
# instrumentación GM (melodía / armonía / bajo), y flags opcionales.

STYLE_PRESETS = {
    'heroic_minor': dict(
        scale='natural_minor', method='degree', start_shift=0,
        harmonizer='major_third_rule',
        melody_program=60, harmony_program=48, bass_program=42,
        desc='Menor paralela + tríadas mayores (nota fuerte = 3ª del acorde)',
    ),
    'dorian_heroic': dict(
        scale='dorian', method='degree', start_shift=0,
        harmonizer='major_third_rule',
        melody_program=60, harmony_program=61, bass_program=42,
        desc='Dórico + tríadas mayores, estilo Fellowship (Shore)',
    ),
    'whole_tone': dict(
        scale='whole_tone', method='interval', start_shift=0,
        harmonizer='whole_tone',
        melody_program=0, harmony_program=89, bass_program=32,
        desc='Tonos enteros + dominantes 7ª/9ª con 5ª omitida (Debussy)',
    ),
    'octatonic': dict(
        scale='octatonic_auto', method='interval', start_shift=0,
        harmonizer='generic',
        melody_program=0, harmony_program=0, bass_program=32,
        desc='Colección octatónica + tríadas/7ª nativas (Shapiro / Severance)',
    ),
    'chromatic_magical': dict(
        scale='chromatic', method='degree', start_shift=0,
        harmonizer='parallel_minor',
        melody_program=9, harmony_program=61, bass_program=42,
        desc='Melodía intacta + tríadas menores paralelas cerradas debajo (Hedwig)',
    ),
    'chromatic_counterpoint': dict(
        scale='chromatic', method='degree', start_shift=0,
        harmonizer='counterpoint',
        melody_program=0, harmony_program=0, bass_program=32,
        desc='Melodía intacta + contrapunto cromático independiente (Bartók)',
    ),
    'pentatonic': dict(
        scale='pentatonic_auto', method='degree', start_shift=0,
        harmonizer='generic_sparse',
        melody_program=46, harmony_program=46, bass_program=42,
        desc='Pentatónica mayor/menor (según modo detectado) + armonía dispersa',
    ),
    'double_harmonic': dict(
        scale='double_harmonic', method='degree', start_shift=0,
        harmonizer='generic',
        melody_program=104, harmony_program=48, bass_program=42,
        desc='Doble armónica / mayamalavagowla (LOTR: The Stranger)',
    ),
    'acoustic': dict(
        scale='acoustic', method='degree', start_shift=0,
        harmonizer='generic',
        melody_program=0, harmony_program=48, bass_program=42,
        desc='Modo acústico / lidio dominante (LOTR: History of the Ring)',
    ),
    'elvish': dict(
        scale='melodic_minor_m5', method='degree', start_shift=0,
        reorient_first_note=True,
        harmonizer='generic',
        melody_program=73, harmony_program=48, bass_program=42,
        desc='5º modo de la menor melódica, reorientado a la 1ª nota (LOTR: Valinor)',
    ),
    'got_fragment': dict(
        scale='harmonic_minor', method='degree', start_shift=0,
        fragment=True, metric_shift=True,
        harmonizer='generic',
        melody_program=24, harmony_program=61, bass_program=42,
        desc='Fragmentación motívica + menor armónica + downbeat (Game of Thrones)',
    ),
    'chromatic_mediant': dict(
        scale='chromatic', method='degree', start_shift=0,
        harmonizer='chromatic_mediant',
        melody_program=48, harmony_program=61, bass_program=42,
        desc='Melodía intacta + cadena de tríadas por 3ª cromática sin función tonal (LOTR)',
    ),
    'custom': dict(
        scale=None, method='degree', start_shift=0,
        harmonizer='generic',
        melody_program=0, harmony_program=48, bass_program=32,
        desc='Escala y método libres, definidos vía --scale / --method',
    ),
}


# ═══════════════════════════════════════════════════════════
# MATEMÁTICA DE ESCALAS Y GRADOS
# ═══════════════════════════════════════════════════════════

def snap_pc_to_scale(pc, scale_intervals):
    """Devuelve (idx, distancia_semitonos) al grado más cercano de la escala."""
    best_idx, best_dist = 0, 99
    for i, iv in enumerate(scale_intervals):
        dist = min(abs(pc - iv), 12 - abs(pc - iv))
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx, best_dist


def pitch_to_abs_degree(pitch, root_pc, scale_intervals):
    """
    Convierte una altura MIDI en un 'grado absoluto' (entero) relativo a
    root_pc dentro de scale_intervals. Cada +cardinalidad equivale a +1 octava.
    Si la nota no pertenece exactamente a la escala, se ancla al grado más
    cercano (comportamiento de snapping usado por el método 'degree').
    """
    card = len(scale_intervals)
    semis = pitch - root_pc
    octv = semis // 12
    pc_in_oct = semis - octv * 12
    idx, _ = snap_pc_to_scale(pc_in_oct, scale_intervals)
    return octv * card + idx


def abs_degree_to_pitch(abs_degree, root_pc, scale_intervals):
    """Inversa de pitch_to_abs_degree: grado absoluto → altura MIDI."""
    card = len(scale_intervals)
    octv, idx = divmod(abs_degree, card)
    return root_pc + octv * 12 + scale_intervals[idx]


def total_snap_error(notes, root_pc, scale_intervals):
    """Suma de distancias de snapping: mide qué tan bien 'encaja' una escala."""
    total = 0
    for (_, p, _, _) in notes:
        pc = (p - root_pc) % 12
        _, d = snap_pc_to_scale(pc, scale_intervals)
        total += d
    return total


def anchor_octave(pitch, reference_pitch):
    """
    Corrige 'pitch' por octavas completas para que quede lo más cerca posible
    de 'reference_pitch'. Es necesario porque el 'grado absoluto' se computa
    con la cardinalidad de la escala ORIGEN pero se reconstruye con la
    cardinalidad de la escala DESTINO: si difieren (p.ej. 7 vs 6 u 8), el
    mismo número de grado corresponde a una octava distinta y, sin esta
    corrección, el registro de la melodía se desbocaría con cada nota.
    """
    diff = pitch - reference_pitch
    correction = round(diff / 12.0) * 12
    return pitch - correction


def nearest_pitch_to_register(pc, center):
    """Altura MIDI de clase de tono pc más cercana al registro 'center'."""
    pc = pc % 12
    base = center - (center % 12) + pc
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda x: abs(x - center))


def nearest_step_for_semitone_gap(gap_semitones, prev_new_pitch, cur_abs_degree,
                                  root_pc, scale_intervals):
    """
    Para el método 'interval': busca el número de pasos de la escala DESTINO
    (delta, puede ser negativo) que mejor reproduce, en semitonos reales, el
    salto 'gap_semitones' observado entre dos notas consecutivas del original.
    """
    card = len(scale_intervals)
    best_delta, best_err = 0, float('inf')
    search_range = card * 2 + 4
    for delta in range(-search_range, search_range + 1):
        cand_pitch = abs_degree_to_pitch(cur_abs_degree + delta, root_pc, scale_intervals)
        err = abs((cand_pitch - prev_new_pitch) - gap_semitones)
        if err < best_err or (err == best_err and abs(delta) < abs(best_delta)):
            best_err, best_delta = err, delta
    return best_delta


# ═══════════════════════════════════════════════════════════
# MÉTODOS DE MAPEO
# ═══════════════════════════════════════════════════════════

def map_degree_method(melody, src_root_pc, src_scale, tgt_root_pc, tgt_scale,
                      start_shift=0):
    """
    Método de grado de escala: cada nota se mide independientemente como un
    grado (con snapping) respecto a la escala origen, y ese mismo número de
    grado (± start_shift) se re-materializa en la escala destino.
    """
    out = []
    for (off, p, dur, vel) in melody:
        d = pitch_to_abs_degree(p, src_root_pc, src_scale) + start_shift
        newp = abs_degree_to_pitch(d, tgt_root_pc, tgt_scale)
        newp = anchor_octave(newp, p)
        out.append((off, newp, dur, vel))
    return out


def map_interval_method(melody, src_root_pc, src_scale, tgt_root_pc, tgt_scale,
                        start_shift=0, register_clamp=12):
    """
    Método de intervalos: la primera nota se ancla por grado (como en
    'degree'); cada nota siguiente se coloca buscando, dentro de la escala
    destino, el número de pasos que mejor reproduce el intervalo real (en
    semitonos) que había entre las dos notas originales consecutivas. Con
    'register_clamp' se evita que el registro se desboque cuando la
    cardinalidad de origen y destino difieren mucho.
    """
    if not melody:
        return []
    out = []
    card = len(tgt_scale)

    first_off, first_p, first_dur, first_vel = melody[0]
    cur_abs = pitch_to_abs_degree(first_p, src_root_pc, src_scale) + start_shift
    new_p = abs_degree_to_pitch(cur_abs, tgt_root_pc, tgt_scale)
    anchored = anchor_octave(new_p, first_p)
    if anchored != new_p:
        cur_abs += (anchored - new_p) // 12 * card
        new_p = anchored
    out.append((first_off, new_p, first_dur, first_vel))

    prev_src_p = first_p
    prev_new_p = new_p

    for (off, p, dur, vel) in melody[1:]:
        gap = p - prev_src_p
        delta = nearest_step_for_semitone_gap(gap, prev_new_p, cur_abs, tgt_root_pc, tgt_scale)
        cur_abs += delta
        new_p = abs_degree_to_pitch(cur_abs, tgt_root_pc, tgt_scale)

        if register_clamp:
            while new_p - prev_new_p > register_clamp:
                cur_abs -= card
                new_p -= 12
            while prev_new_p - new_p > register_clamp:
                cur_abs += card
                new_p += 12

        out.append((off, new_p, dur, vel))
        prev_src_p, prev_new_p = p, new_p

    return out


# ═══════════════════════════════════════════════════════════
# CHUNKING / JERARQUÍA MÉTRICA
# ═══════════════════════════════════════════════════════════

def beat_strength(offset, beats_per_bar, tol=0.05):
    """1.0 = downbeat, 0.6 = otro tiempo entero, 0.25 = contratiempo."""
    if abs((offset % beats_per_bar)) < tol or abs((offset % beats_per_bar) - beats_per_bar) < tol:
        return 1.0
    if abs((offset % 1.0)) < tol:
        return 0.6
    return 0.25


def window_pc_weights(notes, win_start, win_end, beats_per_bar):
    """Pesa cada clase de tono presente en [win_start, win_end) por duración
    solapada y fuerza métrica; usado para elegir raíces de acorde por compás."""
    weights = defaultdict(float)
    for (off, p, dur, vel) in notes:
        overlap = min(off + dur, win_end) - max(off, win_start)
        if overlap <= 0:
            continue
        weights[p % 12] += overlap * (1.0 + beat_strength(off, beats_per_bar))
    return weights


def melody_total_beats(notes):
    if not notes:
        return 0.0
    last = max(notes, key=lambda n: n[0] + n[2])
    return last[0] + last[2]


# ═══════════════════════════════════════════════════════════
# FRAGMENTACIÓN Y DESPLAZAMIENTO MÉTRICO
# ═══════════════════════════════════════════════════════════

def split_fragments(melody, gap_threshold=0.5):
    """Divide la melodía en motivos separados por silencios >= gap_threshold."""
    frags, cur, prev_end = [], [], None
    for n in melody:
        off, p, dur, vel = n
        if prev_end is not None and off - prev_end > gap_threshold:
            if cur:
                frags.append(cur)
                cur = []
        cur.append(n)
        prev_end = off + dur
    if cur:
        frags.append(cur)
    return frags


def reorder_fragments(frags, order_spec):
    """Reordena/repite fragmentos según una lista de índices (--fragment-order)."""
    if not order_spec:
        return frags
    idxs = [i for i in order_spec if 0 <= i < len(frags)]
    return [frags[i] for i in idxs] if idxs else frags


def flatten_fragments(frags, gap_beats=0.25):
    """Reconcatena fragmentos en el tiempo, con un pequeño respiro entre ellos."""
    out, cursor = [], 0.0
    for frag in frags:
        if not frag:
            continue
        shift = cursor - frag[0][0]
        for (off, p, dur, vel) in frag:
            out.append((off + shift, p, dur, vel))
        cursor = frag[-1][0] + frag[-1][2] + shift + gap_beats
    return out


def apply_metric_shift(melody, beats_per_bar):
    """Desplaza toda la melodía para que la primera nota caiga en el downbeat."""
    if not melody:
        return melody
    first_off = melody[0][0]
    nearest_bar_start = round(first_off / beats_per_bar) * beats_per_bar
    shift = nearest_bar_start - first_off
    return [(max(0.0, off + shift), p, dur, vel) for (off, p, dur, vel) in melody]


# ═══════════════════════════════════════════════════════════
# CORRECCIÓN DE TONO DE ACORDE (chord-tone override)
# ═══════════════════════════════════════════════════════════

def snap_strong_beats_to_chords(mapped_notes, chords, beats_per_bar, max_correction=3,
                                allowed_indices=None):
    """
    Principio de la teoría: cuando un mapeo mecánico de intervalos no aterriza
    en un tono del acorde, suele sonar mejor sacrificar el tamaño exacto del
    intervalo por caer en un chord tone — pero SOLO en tiempos fuertes, y
    solo con una corrección pequeña (<= max_correction semitonos).
    Si se pasa allowed_indices (set de índices), solo esas notas son
    candidatas a corrección — usado por --body-tail para proteger la
    identidad del 'body' sin tocar la 'tail'.
    """
    def chord_at(t):
        for (cs, cd, cp) in chords:
            if cs <= t < cs + cd:
                return cp
        return None

    out = []
    for i, (off, p, dur, vel) in enumerate(mapped_notes):
        if (allowed_indices is None or i in allowed_indices) and beat_strength(off, beats_per_bar) >= 1.0:
            cp = chord_at(off)
            if cp:
                pcs = {x % 12 for x in cp}
                if p % 12 not in pcs:
                    best, best_d = None, max_correction + 1
                    for ct_pc in pcs:
                        for oct_shift in (-12, 0, 12):
                            cand = p - (p % 12) + ct_pc + oct_shift
                            d = abs(cand - p)
                            if d < best_d:
                                best_d, best = d, cand
                    if best is not None:
                        p = best
        out.append((off, p, dur, vel))
    return out


# ═══════════════════════════════════════════════════════════
# ARMONIZADORES
# ═══════════════════════════════════════════════════════════

def choose_bar_root_idx(mapped_notes, bstart, bend, root_pc, scale, beats_per_bar):
    w = window_pc_weights(mapped_notes, bstart, bend, beats_per_bar)
    if not w:
        return 0
    best_pc = max(w, key=w.get)
    idx, _ = snap_pc_to_scale((best_pc - root_pc) % 12, scale)
    return idx


def chord_pitches_from_scale(root_idx, scale, root_pc, base_register=48, size=3):
    """Apila terceras DENTRO de la escala destino a partir de root_idx
    (armonización 'nativa' de cualquier escala, diatónica o exótica)."""
    root_pitch = abs_degree_to_pitch(root_idx, root_pc, scale)
    while root_pitch < base_register:
        root_pitch += 12
    while root_pitch - 12 >= base_register:
        root_pitch -= 12
    pitches = [root_pitch]
    idx = root_idx
    for _ in range(1, size):
        idx += 2
        p = abs_degree_to_pitch(idx, root_pc, scale)
        while p <= pitches[-1]:
            p += 12
        pitches.append(p)
    return pitches


def harmonize_generic(mapped_notes, root_pc, scale, beats_per_bar, total_beats,
                      chord_size=3, base_register=48):
    """Armonización por compás apilando terceras nativas de la escala destino."""
    chords, bar = [], 0
    while bar * beats_per_bar < total_beats:
        bstart = bar * beats_per_bar
        bend = min(bstart + beats_per_bar, total_beats)
        idx = choose_bar_root_idx(mapped_notes, bstart, bend, root_pc, scale, beats_per_bar)
        pitches = chord_pitches_from_scale(idx, scale, root_pc, base_register, chord_size)
        chords.append((bstart, bend - bstart, pitches))
        bar += 1
    return chords


def harmonize_sparse_fifths(mapped_notes, root_pc, scale, beats_per_bar, total_beats,
                            base_register=48):
    """Armonía dispersa (raíz + 5ª, sin 3ª) para texturas pentatónicas abiertas."""
    chords, bar = [], 0
    while bar * beats_per_bar < total_beats:
        bstart = bar * beats_per_bar
        bend = min(bstart + beats_per_bar, total_beats)
        idx = choose_bar_root_idx(mapped_notes, bstart, bend, root_pc, scale, beats_per_bar)
        root_pitch = abs_degree_to_pitch(idx, root_pc, scale)
        root_pitch = nearest_pitch_to_register(root_pitch % 12, base_register)
        chords.append((bstart, bend - bstart, [root_pitch, root_pitch + 7]))
        bar += 1
    return chords


def harmonize_major_third_rule(mapped_notes, beats_per_bar, total_beats, base_register=52):
    """
    'Heroic minor' / dórico heroico: SIEMPRE tríadas mayores, elegidas para
    que la nota más importante del compás (peso por tiempo fuerte) caiga
    como 3ª del acorde — la regla explícita de la teoría para este ejercicio.
    """
    chords, bar = [], 0
    while bar * beats_per_bar < total_beats:
        bstart = bar * beats_per_bar
        bend = min(bstart + beats_per_bar, total_beats)
        w = window_pc_weights(mapped_notes, bstart, bend, beats_per_bar)
        strong_pc = max(w, key=w.get) if w else 0
        root_pc = (strong_pc - 4) % 12
        root_pitch = nearest_pitch_to_register(root_pc, base_register)
        chords.append((bstart, bend - bstart, [root_pitch, root_pitch + 4, root_pitch + 7]))
        bar += 1
    return chords


def harmonize_whole_tone(mapped_notes, wt_root_pc, wt_scale, beats_per_bar, total_beats,
                         base_register=52):
    """Dominantes 7ª/9ª con la 5ª omitida, raíces tomadas de la propia
    colección de tonos enteros (técnica descrita para el ejercicio Debussy)."""
    chords, bar = [], 0
    while bar * beats_per_bar < total_beats:
        bstart = bar * beats_per_bar
        bend = min(bstart + beats_per_bar, total_beats)
        w = window_pc_weights(mapped_notes, bstart, bend, beats_per_bar)
        strong_pc = max(w, key=w.get) if w else wt_root_pc
        idx, _ = snap_pc_to_scale((strong_pc - wt_root_pc) % 12, wt_scale)
        root = abs_degree_to_pitch(idx, wt_root_pc, wt_scale)
        root = nearest_pitch_to_register(root % 12, base_register)
        pitches = [root, root + 4, root + 10]
        if bar % 2 == 0:
            pitches.append(root + 14)  # 9ª ocasional
        chords.append((bstart, bend - bstart, pitches))
        bar += 1
    return chords


def harmonize_parallel_minor(mapped_notes, voice_as='fifth'):
    """
    Tríadas menores paralelas en posición cerrada DEBAJO de cada nota de la
    melodía (técnica 'chromatic magical' / estilo Hedwig's Theme). voice_as
    controla si la melodía funciona como raíz, 3ª o 5ª del acorde paralelo.
    """
    offset_map = {'root': 0, 'third': 3, 'fifth': 7}
    off_semi = offset_map.get(voice_as, 7)
    chords = []
    for (off, p, dur, vel) in mapped_notes:
        root = p - off_semi
        triad = [root, root + 3, root + 7]
        adjusted = []
        for t in triad:
            while t > p:
                t -= 12
            adjusted.append(t)
        chords.append((off, dur, adjusted))
    return chords


def generate_counterpoint_line(mapped_notes, register_offset=-7):
    """
    Línea contrapuntística ligera e independiente (estilo Bartók/'creepy
    counterpoint'): favorece movimiento contrario respecto a la melodía,
    se mantiene en un registro relativo por debajo, y evita duplicar la
    melodía al unísono/octava. No pretende sustituir a counterpoint.py
    (contrapunto estricto de especies); es un contorno libre y evocador.
    """
    if not mapped_notes:
        return []
    out, prev_mel, prev_cp = [], None, None
    for (off, p, dur, vel) in mapped_notes:
        if prev_mel is None:
            cp = p + register_offset
        else:
            mel_dir = p - prev_mel
            if mel_dir == 0:
                step = random.choice([-2, -1, 1, 2])
            else:
                sign = -1 if mel_dir > 0 else 1
                step = sign * random.choice([1, 2, 3, 4])
            cp = prev_cp + step
        while cp > p - 3:
            cp -= 12
        while cp < p - 18:
            cp += 12
        if (cp - p) % 12 == 0:
            cp += random.choice([-2, -1, 1, 2])
        out.append((off, cp, dur, max(40, int(vel) - 15)))
        prev_mel, prev_cp = p, cp
    return out


def harmonize_chromatic_mediant(mapped_notes, beats_per_bar, total_beats, seed=42, base_register=52):
    """
    Encadenamiento de tríadas por movimiento de 3ª cromática (mayor o menor,
    ascendente o descendente), SIN función tonal, con calidad mayor/menor
    alternada libremente — el recurso de reharmonización cromático-mediántica
    (estilo Bear McCreary / Hans Zimmer) descrito para la cadena de LOTR, en
    vez de la armonía diatónica de harmonize_generic.
    """
    rnd = random.Random(seed + 999)
    chords, bar, prev_root = [], 0, None
    while bar * beats_per_bar < total_beats:
        bstart = bar * beats_per_bar
        bend = min(bstart + beats_per_bar, total_beats)
        w = window_pc_weights(mapped_notes, bstart, bend, beats_per_bar)
        strong_pc = max(w, key=w.get) if w else 0
        if prev_root is None:
            root_pc = strong_pc
        else:
            step = rnd.choice([3, 4, 8, 9])  # 3ª menor/mayor, ascendente o descendente (mod 12)
            root_pc = (prev_root + step) % 12
        quality = rnd.choice(['M', 'm'])
        root_pitch = nearest_pitch_to_register(root_pc, base_register)
        third = root_pitch + (4 if quality == 'M' else 3)
        chords.append((bstart, bend - bstart, [root_pitch, third, root_pitch + 7]))
        prev_root = root_pc
        bar += 1
    return chords


def body_tail_tags(melody):
    """
    Etiqueta cada nota como 'body' o 'tail' según su posición dentro de su
    fragmento (separado por silencios, ver split_fragments): primera mitad
    del fragmento = cuerpo (identidad protegida), segunda mitad = cola (más
    libertad ornamental). Refleja la estructura 'body/tail' descrita en el
    análisis del tema de vuelo de How to Train Your Dragon.
    """
    frags = split_fragments(melody)
    tags = ['body'] * len(melody)
    idx = 0
    for frag in frags:
        if not frag:
            continue
        start = frag[0][0]
        end = frag[-1][0] + frag[-1][2]
        midpoint = start + (end - start) / 2.0
        for _ in frag:
            tags[idx] = 'body' if melody[idx][0] < midpoint else 'tail'
            idx += 1
    return tags


def ornament_tail_notes(mapped_notes, tags, tgt_root_pc, tgt_scale, seed=42):
    """Añade, con probabilidad moderada, una breve nota vecina de adorno
    (grado superior de la escala destino) antes de notas etiquetadas 'tail',
    dándoles más libertad melódica que al 'body' (que --chord-tone-priority
    protege con --body-tail activo)."""
    rnd = random.Random(seed + 777)
    out = []
    for i, (off, p, dur, vel) in enumerate(mapped_notes):
        tag = tags[i] if i < len(tags) else 'body'
        if tag == 'tail' and dur >= 0.4 and rnd.random() < 0.5:
            deg = pitch_to_abs_degree(p, tgt_root_pc, tgt_scale)
            neighbor = abs_degree_to_pitch(deg + 1, tgt_root_pc, tgt_scale)
            grace_dur = min(0.25, dur * 0.3)
            out.append((off, neighbor, grace_dur, max(40, int(vel) - 15)))
            out.append((off + grace_dur, p, dur - grace_dur, vel))
        else:
            out.append((off, p, dur, vel))
    return out


def harmonize_theme(preset, mapped_notes, root_pc, scale, wt_root_pc, beats_per_bar,
                    total_beats, seed=42):
    random.seed(seed)
    kind = preset['harmonizer']
    chords, counterpoint_notes = None, None
    if kind == 'generic':
        chords = harmonize_generic(mapped_notes, root_pc, scale, beats_per_bar, total_beats)
    elif kind == 'generic_sparse':
        chords = harmonize_sparse_fifths(mapped_notes, root_pc, scale, beats_per_bar, total_beats)
    elif kind == 'major_third_rule':
        chords = harmonize_major_third_rule(mapped_notes, beats_per_bar, total_beats)
    elif kind == 'whole_tone':
        chords = harmonize_whole_tone(mapped_notes, wt_root_pc, scale, beats_per_bar, total_beats)
    elif kind == 'parallel_minor':
        chords = harmonize_parallel_minor(mapped_notes, voice_as='fifth')
    elif kind == 'chromatic_mediant':
        chords = harmonize_chromatic_mediant(mapped_notes, beats_per_bar, total_beats, seed)
    elif kind == 'counterpoint':
        counterpoint_notes = generate_counterpoint_line(mapped_notes)
    return chords, counterpoint_notes


# ═══════════════════════════════════════════════════════════
# CARGA DE MIDI Y DETECCIÓN DE TONALIDAD
# ═══════════════════════════════════════════════════════════

def load_melody(midi_path, verbose=False):
    """
    Carga la pista melódica principal (canal de pitch medio más alto,
    excluyendo percusión). Devuelve (melody, tempo_bpm, (ts_num, ts_den), tpb).
    melody: list of (offset_beats, pitch_midi, duration_beats, velocity)
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    notes_by_track = defaultdict(list)
    track_channels = defaultdict(set)

    for track_idx, track in enumerate(mid.tracks):
        abs_t = 0
        pending = {}
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
                track_channels[track_idx].add(msg.channel)
            elif msg.type in ('note_off', 'note_on'):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    on_t, vel = pending.pop(key_)
                    on_beat = on_t / tpb
                    dur_beat = max(0.1, (abs_t - on_t) / tpb)
                    notes_by_track[track_idx].append((on_beat, msg.note, dur_beat, vel))

    if not notes_by_track:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    def mean_pitch(ns):
        return float(np.mean([p for _, p, _, _ in ns])) if ns else 0.0

    # Excluir pistas de percusión (canal 9 / GM channel 10)
    non_perc = {t: ns for t, ns in notes_by_track.items()
               if 9 not in track_channels.get(t, set()) and ns}
    pool = non_perc or notes_by_track
    mel_track = max(pool, key=lambda t: mean_pitch(pool[t]))
    melody = sorted(notes_by_track[mel_track], key=lambda x: x[0])

    tempo_bpm = round(60_000_000 / tempo_us, 2)
    if verbose:
        print(f"  Melodía: {len(melody)} notas, pista {mel_track} "
              f"(de {len(notes_by_track)} pistas con notas)")
        print(f"  Tempo: {tempo_bpm} BPM, compás: {ts_num}/{ts_den}")

    return melody, tempo_bpm, (ts_num, ts_den), tpb


def parse_key_string(key_str):
    """Parsea 'D minor', 'F# major', 'Am', etc. → (tonic_pc, mode)."""
    if not key_str:
        return None
    s = key_str.strip()
    if s.endswith('m') and 'major' not in s.lower() and 'minor' not in s.lower():
        name, mode = s[:-1], 'minor'
    else:
        parts = s.split()
        name = parts[0]
        mode = parts[1].lower() if len(parts) > 1 else 'major'
    pc = NAME_TO_PC.get(name, NAME_TO_PC.get(name.capitalize(), 0))
    return pc, mode


_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _detect_key_fallback(notes):
    """Perfil de Krumhansl-Kessler simplificado (correlación de Pearson)."""
    if not notes:
        return 0, 'major'
    pc_counts = defaultdict(float)
    for (_, p, dur, _) in notes:
        pc_counts[p % 12] += max(dur, 0.25)
    total = sum(pc_counts.values()) or 1.0
    counts = [pc_counts[i] / total for i in range(12)]
    best_r, best_tpc, best_mode = -2.0, 0, 'major'
    for tpc in range(12):
        for profile, mode in [(_MAJOR_PROFILE, 'major'), (_MINOR_PROFILE, 'minor')]:
            rot = profile[tpc:] + profile[:tpc]
            r = float(np.corrcoef(rot, counts)[0, 1]) if np.std(counts) > 0 else 0.0
            if r > best_r:
                best_r, best_tpc, best_mode = r, tpc, mode
    return best_tpc, best_mode


def detect_key(midi_path, notes, verbose=False):
    if MUSIC21_OK:
        try:
            score = converter.parse(midi_path)
            k = score.analyze('key')
            tonic_pc = m21pitch.Pitch(k.tonic.name).pitchClass
            if verbose:
                print(f"  Tonalidad detectada (music21): {k.tonic.name} {k.mode}")
            return tonic_pc, k.mode
        except Exception:
            pass
    tonic_pc, mode = _detect_key_fallback(notes)
    if verbose:
        print(f"  Tonalidad detectada (perfil KK): {PITCH_NAMES[tonic_pc]} {mode}")
    return tonic_pc, mode


# ═══════════════════════════════════════════════════════════
# ESCRITURA DE MIDI
# ═══════════════════════════════════════════════════════════

def write_theme_midi(melody_notes, chords, counterpoint_notes, tempo_bpm, beats_per_bar,
                     output_path, style_name='', melody_program=0, harmony_program=48,
                     bass_program=32, add_bass=True):
    tpb = 480
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(0, int(round(beats * tpb)))

    def make_header():
        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=f'ThemeTransformer:{style_name}', time=0))
        trk.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        trk.append(mido.MetaMessage('time_signature', numerator=int(round(beats_per_bar)),
                                    denominator=4, clocks_per_click=24,
                                    notated_32nd_notes_per_beat=8, time=0))
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    def notes_to_track(notes_list, ch, program, name):
        events = []
        for offset, pitch, dur, vel in notes_list:
            p = max(0, min(127, int(round(pitch))))
            v = max(1, min(127, int(round(vel))))
            t_on = to_ticks(offset)
            t_off = to_ticks(offset + max(0.05, dur))
            if t_off <= t_on:
                t_off = t_on + 1
            events.append((t_on, 'on', p, v))
            events.append((t_off, 'off', p, 0))
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=name, time=0))
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
    mid.tracks.append(make_header())
    mid.tracks.append(notes_to_track(melody_notes, ch=0, program=melody_program, name='Melody'))

    if chords:
        chord_notes, bass_notes = [], []
        for (cs, cd, pitches) in chords:
            for p in pitches:
                chord_notes.append((cs, p, cd * 0.95, 58))
            if add_bass and pitches:
                root = min(pitches)
                b = root - 24
                while b < 24:
                    b += 12
                bass_notes.append((cs, b, cd * 0.9, 66))
        mid.tracks.append(notes_to_track(chord_notes, ch=1, program=harmony_program, name='Harmony'))
        if bass_notes:
            mid.tracks.append(notes_to_track(bass_notes, ch=2, program=bass_program, name='Bass'))

    if counterpoint_notes:
        mid.tracks.append(notes_to_track(counterpoint_notes, ch=3, program=42, name='Counterpoint'))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mid.save(output_path)


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def _run_style_pipeline(working_melody, preset, src_root_pc, src_scale, src_mode,
                        beats_per_bar, start_degree_extra=0, chord_tone_priority=False,
                        fragment_override=False, fragment_order=None, metric_shift_override=False,
                        body_tail=False, wt_variant='auto', oct_variant='auto',
                        pentatonic_variant='auto', seed=42, verbose=False):
    """
    Núcleo reusable: aplica UN estilo a UNA lista de notas (offsets relativos
    a 0). Usado tanto por transform_theme (todo el fichero) como por
    transform_theme_sections (un tramo del fichero por sección). Devuelve
    None si el segmento no tiene notas.
    """
    do_fragment = fragment_override or preset.get('fragment', False)
    do_metric_shift = metric_shift_override or preset.get('metric_shift', False)

    n_frags = 1
    if do_fragment:
        frags = split_fragments(working_melody)
        n_frags = len(frags)
        if fragment_order:
            frags = reorder_fragments(frags, fragment_order)
        working_melody = flatten_fragments(frags)
        if verbose:
            print(f"  Fragmentado en {n_frags} motivos"
                  + (f", reordenados como {fragment_order}" if fragment_order else ""))

    if do_metric_shift:
        working_melody = apply_metric_shift(working_melody, beats_per_bar)

    if not working_melody:
        return None

    tags = body_tail_tags(working_melody) if body_tail else None

    # ── resolver escala destino ──────────────────────────────────────────
    scale_name = preset['scale']
    reorient = preset.get('reorient_first_note', False)
    wt_root_pc = None

    if scale_name == 'octatonic_auto':
        anchor_pc = working_melody[0][1] % 12
        if oct_variant in ('wh', 'hw'):
            chosen = f'octatonic_{oct_variant}'
        else:
            err_wh = total_snap_error(working_melody, anchor_pc, SCALES['octatonic_wh'])
            err_hw = total_snap_error(working_melody, anchor_pc, SCALES['octatonic_hw'])
            chosen = 'octatonic_wh' if err_wh <= err_hw else 'octatonic_hw'
        tgt_scale = SCALES[chosen]
        tgt_root_pc = anchor_pc
        scale_name = chosen
        if verbose:
            print(f"  Octatónica elegida: {chosen} (raíz pc={anchor_pc})")
    elif scale_name == 'pentatonic_auto':
        if pentatonic_variant in ('major', 'minor'):
            chosen = f'{pentatonic_variant}_pentatonic'
        else:
            chosen = 'major_pentatonic' if src_mode.lower().startswith('maj') else 'minor_pentatonic'
        tgt_scale = SCALES[chosen]
        tgt_root_pc = src_root_pc
        scale_name = chosen
        if verbose:
            print(f"  Pentatónica elegida: {chosen}")
    elif scale_name == 'whole_tone':
        if wt_variant in (0, 1):
            wt_root_pc = (src_root_pc + wt_variant) % 12
        else:
            err0 = total_snap_error(working_melody, src_root_pc, SCALES['whole_tone'])
            err1 = total_snap_error(working_melody, (src_root_pc + 1) % 12, SCALES['whole_tone'])
            wt_root_pc = src_root_pc if err0 <= err1 else (src_root_pc + 1) % 12
        tgt_scale = SCALES['whole_tone']
        tgt_root_pc = wt_root_pc
        if verbose:
            print(f"  Colección de tonos enteros: raíz pc={wt_root_pc}")
    else:
        tgt_scale = SCALES[scale_name]
        tgt_root_pc = (working_melody[0][1] % 12) if reorient else src_root_pc

    start_shift = preset.get('start_shift', 0) + start_degree_extra
    method = preset['method']

    if scale_name == 'chromatic':
        # La escala cromática contiene todas las alturas: mapear "por grado"
        # reinterpretando el número de grado con otra cardinalidad distorsionaría
        # la clase de altura sin necesidad. La transformación real en estos
        # estilos ocurre en la armonización, no en la melodía (ver video-teoría).
        mapped = [(off, p + start_shift, dur, vel) for (off, p, dur, vel) in working_melody]
    elif method == 'degree':
        mapped = map_degree_method(working_melody, src_root_pc, src_scale,
                                   tgt_root_pc, tgt_scale, start_shift)
    else:
        mapped = map_interval_method(working_melody, src_root_pc, src_scale,
                                     tgt_root_pc, tgt_scale, start_shift)

    total_beats = melody_total_beats(mapped)
    total_beats = (math.ceil(total_beats / beats_per_bar) * beats_per_bar
                  if total_beats > 0 else beats_per_bar)

    chords, counterpoint_notes = harmonize_theme(
        preset, mapped, tgt_root_pc, tgt_scale, wt_root_pc, beats_per_bar, total_beats, seed
    )

    if chord_tone_priority and chords:
        allowed = {i for i, t in enumerate(tags) if t == 'body'} if tags else None
        mapped = snap_strong_beats_to_chords(mapped, chords, beats_per_bar, allowed_indices=allowed)

    if body_tail and tags:
        mapped = ornament_tail_notes(mapped, tags, tgt_root_pc, tgt_scale, seed)

    return dict(
        mapped=mapped, chords=chords, counterpoint_notes=counterpoint_notes,
        scale_name=scale_name, tgt_scale=tgt_scale, tgt_root_pc=tgt_root_pc,
        wt_root_pc=wt_root_pc, total_beats=total_beats, method=method,
        start_shift=start_shift, do_fragment=do_fragment, do_metric_shift=do_metric_shift,
        n_frags=n_frags,
    )


def transform_theme(midi_path: str,
                    style: str,
                    key_override: str = None,
                    tempo_override: float = None,
                    beats_per_bar_override: int = None,
                    start_degree_extra: int = 0,
                    chord_tone_priority: bool = False,
                    fragment: bool = False,
                    fragment_order: list = None,
                    metric_shift: bool = False,
                    body_tail: bool = False,
                    wt_variant: str = 'auto',
                    oct_variant: str = 'auto',
                    pentatonic_variant: str = 'auto',
                    custom_scale: str = None,
                    custom_method: str = None,
                    out_dir: str = None,
                    report: bool = False,
                    verbose: bool = False,
                    seed: int = 42):
    if style not in STYLE_PRESETS:
        raise ValueError(f"Estilo desconocido: '{style}'. Usa --catalog para ver la lista.")

    preset = dict(STYLE_PRESETS[style])
    if style == 'custom':
        if not custom_scale:
            raise ValueError("--style custom requiere --scale (ver --catalog)")
        if custom_scale not in SCALES:
            raise ValueError(f"Escala desconocida: '{custom_scale}'. Usa --catalog.")
        preset['scale'] = custom_scale
        preset['method'] = custom_method or 'degree'

    if verbose:
        print(f"\n═══ {os.path.basename(midi_path)} → estilo '{style}' ═══")

    melody, tempo_bpm, ts, tpb = load_melody(midi_path, verbose=verbose)
    beats_per_bar = beats_per_bar_override or ts[0]
    if tempo_override:
        tempo_bpm = tempo_override

    if key_override:
        src_root_pc, src_mode = parse_key_string(key_override)
    else:
        src_root_pc, src_mode = detect_key(midi_path, melody, verbose=verbose)

    src_scale = SCALES['major'] if src_mode.lower().startswith('maj') else SCALES['natural_minor']

    res = _run_style_pipeline(
        melody, preset, src_root_pc, src_scale, src_mode, beats_per_bar,
        start_degree_extra=start_degree_extra, chord_tone_priority=chord_tone_priority,
        fragment_override=fragment, fragment_order=fragment_order,
        metric_shift_override=metric_shift, body_tail=body_tail,
        wt_variant=wt_variant, oct_variant=oct_variant,
        pentatonic_variant=pentatonic_variant, seed=seed, verbose=verbose,
    )
    if res is None:
        raise RuntimeError(f"{midi_path}: la melodía quedó vacía tras el preprocesado")

    mapped, chords, counterpoint_notes = res['mapped'], res['chords'], res['counterpoint_notes']
    scale_name, tgt_scale, total_beats = res['scale_name'], res['tgt_scale'], res['total_beats']
    method, start_shift = res['method'], res['start_shift']

    stem = Path(midi_path).stem
    out_directory = out_dir or os.path.dirname(os.path.abspath(midi_path))
    output_path = os.path.join(out_directory, f"{stem}.{style}.mid")

    write_theme_midi(
        mapped, chords, counterpoint_notes, tempo_bpm, beats_per_bar, output_path,
        style_name=style,
        melody_program=preset['melody_program'],
        harmony_program=preset['harmony_program'],
        bass_program=preset['bass_program'],
    )

    if verbose:
        print(f"  Escala destino: {scale_name} ({len(tgt_scale)} notas/8va)  método: {method}")
        print(f"  → {output_path}")

    result = {
        'input': midi_path,
        'style': style,
        'output': output_path,
        'source_key': f"{PITCH_NAMES[src_root_pc]} {src_mode}",
        'target_scale': scale_name,
        'target_scale_cardinality': len(tgt_scale),
        'method': method,
        'start_degree_shift': start_shift,
        'n_notes': len(mapped),
        'n_bars': int(total_beats / beats_per_bar),
        'n_fragments': res['n_frags'],
        'fragmented': res['do_fragment'],
        'metric_shifted': res['do_metric_shift'],
        'chord_tone_priority': chord_tone_priority,
        'body_tail': body_tail,
        'has_harmony': bool(chords),
        'has_counterpoint': bool(counterpoint_notes),
    }

    if report:
        report_path = os.path.join(out_directory, f"{stem}.{style}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  → {report_path}")

    return result


# ═══════════════════════════════════════════════════════════
# MODO SECCIONES: transformación progresiva a lo largo de la pieza
# ═══════════════════════════════════════════════════════════
# Refleja cómo los videos de LOTR/GoT NO aplican un estilo fijo a toda la
# pieza: reharmonizan/cambian de escala por tramos (p.ej. menor → doble
# armónica → dórico → octatónica → 5º modo de menor melódica en LOTR; o
# menor durante casi toda la pieza con modulación al clímax en GoT).

def parse_sections_spec(spec):
    """
    '0:heroic_minor,16:whole_tone,32:got_fragment'
        → [(0,'heroic_minor',{}), (16,'whole_tone',{}), (32,'got_fragment',{})]
    Admite secciones 'custom' con escala/método propios:
    '0:heroic_minor,16:custom:double_harmonic:interval'
        → [(0,'heroic_minor',{}), (16,'custom',{'scale':'double_harmonic','method':'interval'})]
    """
    sections = []
    for part in spec.split(','):
        fields = [f.strip() for f in part.split(':')]
        if len(fields) < 2:
            raise ValueError(f"Formato de --sections inválido en '{part}' (usa compás:estilo)")
        bar, style = int(fields[0]), fields[1]
        extra = {}
        if style == 'custom':
            if len(fields) < 3:
                raise ValueError(
                    f"Una sección 'custom' requiere escala: 'compás:custom:escala[:método]' "
                    f"(en '{part}')")
            extra['scale'] = fields[2]
            extra['method'] = fields[3] if len(fields) > 3 else 'degree'
        sections.append((bar, style, extra))
    sections.sort(key=lambda x: x[0])
    if not sections or sections[0][0] != 0:
        raise ValueError("La primera sección de --sections debe empezar en el compás 0")
    return sections


def write_theme_midi_sectioned(melody_notes, chords, counterpoint_notes, tempo_bpm, beats_per_bar,
                               output_path, prog_changes_mel, prog_changes_harm, bass_program=32):
    """Como write_theme_midi, pero admite varios program_change (uno por
    sección) insertados en el punto de la pieza donde arranca cada tramo."""
    tpb = 480
    tempo_us = int(60_000_000 / max(tempo_bpm, 1))

    def to_ticks(beats):
        return max(0, int(round(beats * tpb)))

    def make_header():
        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name='ThemeTransformer:sections', time=0))
        trk.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        trk.append(mido.MetaMessage('time_signature', numerator=int(round(beats_per_bar)),
                                    denominator=4, clocks_per_click=24,
                                    notated_32nd_notes_per_beat=8, time=0))
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    def notes_to_track(notes_list, ch, prog_changes, name):
        events = []
        for offset, pitch, dur, vel in notes_list:
            p = max(0, min(127, int(round(pitch))))
            v = max(1, min(127, int(round(vel))))
            t_on = to_ticks(offset)
            t_off = to_ticks(offset + max(0.05, dur))
            if t_off <= t_on:
                t_off = t_on + 1
            events.append((t_on, 1, 'on', p, v))
            events.append((t_off, 1, 'off', p, 0))
        for (start_beat, program) in prog_changes:
            events.append((to_ticks(start_beat), 0, 'prog', program, 0))
        events.sort(key=lambda x: (x[0], x[1], 0 if x[2] == 'off' else 1))

        trk = mido.MidiTrack()
        trk.append(mido.MetaMessage('track_name', name=name, time=0))
        prev_t = 0
        for abs_t, _prio, kind, p, v in events:
            dt = max(0, abs_t - prev_t)
            if kind == 'prog':
                trk.append(mido.Message('program_change', channel=ch, program=p, time=dt))
            else:
                msg_type = 'note_on' if kind == 'on' else 'note_off'
                trk.append(mido.Message(msg_type, channel=ch, note=p, velocity=v, time=dt))
            prev_t = abs_t
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    mid.tracks.append(make_header())
    mid.tracks.append(notes_to_track(melody_notes, 0, prog_changes_mel, 'Melody'))

    if chords:
        chord_notes, bass_notes = [], []
        for (cs, cd, pitches) in chords:
            for p in pitches:
                chord_notes.append((cs, p, cd * 0.95, 58))
            if pitches:
                root = min(pitches)
                b = root - 24
                while b < 24:
                    b += 12
                bass_notes.append((cs, b, cd * 0.9, 66))
        mid.tracks.append(notes_to_track(chord_notes, 1, prog_changes_harm, 'Harmony'))
        if bass_notes:
            mid.tracks.append(notes_to_track(bass_notes, 2, [(0.0, bass_program)], 'Bass'))

    if counterpoint_notes:
        mid.tracks.append(notes_to_track(counterpoint_notes, 3, [(0.0, 42)], 'Counterpoint'))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mid.save(output_path)


def transform_theme_sections(midi_path: str,
                             sections: list,
                             transition_beats: float = 0,
                             key_override: str = None,
                             tempo_override: float = None,
                             beats_per_bar_override: int = None,
                             start_degree_extra: int = 0,
                             chord_tone_priority: bool = False,
                             fragment: bool = False,
                             fragment_order: list = None,
                             metric_shift: bool = False,
                             body_tail: bool = False,
                             wt_variant: str = 'auto',
                             oct_variant: str = 'auto',
                             pentatonic_variant: str = 'auto',
                             out_dir: str = None,
                             report: bool = False,
                             verbose: bool = False,
                             seed: int = 42):
    melody, tempo_bpm, ts, tpb = load_melody(midi_path, verbose=verbose)
    beats_per_bar = beats_per_bar_override or ts[0]
    if tempo_override:
        tempo_bpm = tempo_override

    if key_override:
        src_root_pc, src_mode = parse_key_string(key_override)
    else:
        src_root_pc, src_mode = detect_key(midi_path, melody, verbose=verbose)
    src_scale = SCALES['major'] if src_mode.lower().startswith('maj') else SCALES['natural_minor']

    total_len = melody_total_beats(melody)
    bar_starts = [bar * beats_per_bar for bar, _style, _extra in sections]
    boundaries = bar_starts + [max(total_len, bar_starts[-1] + beats_per_bar)]
    if boundaries[-1] < total_len:
        boundaries[-1] = math.ceil(total_len / beats_per_bar) * beats_per_bar

    if verbose:
        print(f"\n═══ {os.path.basename(midi_path)} → SECCIONES ═══")
        for (bar, style, extra), b0, b1 in zip(sections, boundaries[:-1], boundaries[1:]):
            tag = style if not extra else f"{style}({extra.get('scale')}/{extra.get('method')})"
            print(f"  compás {bar} ({b0:.1f}–{b1:.1f} negras): estilo '{tag}'")

    results = []
    for i, (bar, style, extra) in enumerate(sections):
        if style == 'custom':
            if extra.get('scale') not in SCALES:
                raise ValueError(f"Escala desconocida en sección 'custom': '{extra.get('scale')}'. "
                                f"Usa --catalog.")
            preset = dict(STYLE_PRESETS['custom'])
            preset['scale'] = extra['scale']
            preset['method'] = extra.get('method', 'degree')
        elif style not in STYLE_PRESETS:
            raise ValueError(f"Estilo de sección desconocido: '{style}'. Usa --catalog.")
        else:
            preset = dict(STYLE_PRESETS[style])
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_notes = [(o - seg_start, p, d, v) for (o, p, d, v) in melody if seg_start <= o < seg_end]
        if verbose:
            print(f"\n  ── sección {i + 1}/{len(sections)}: '{style}' "
                  f"({len(seg_notes)} notas) ──")
        res = None
        if seg_notes:
            res = _run_style_pipeline(
                seg_notes, preset, src_root_pc, src_scale, src_mode, beats_per_bar,
                start_degree_extra=start_degree_extra, chord_tone_priority=chord_tone_priority,
                fragment_override=fragment, fragment_order=fragment_order,
                metric_shift_override=metric_shift, body_tail=body_tail,
                wt_variant=wt_variant, oct_variant=oct_variant,
                pentatonic_variant=pentatonic_variant, seed=seed, verbose=verbose,
            )
        if res is not None:
            res['style'] = style
            res['preset'] = preset
            res['seg_start'] = seg_start
        results.append(res)

    # ── transición: pivotar el final de cada sección hacia el 1er acorde de la siguiente ──
    if transition_beats > 0:
        for i in range(len(results) - 1):
            cur, nxt = results[i], results[i + 1]
            if cur is None or nxt is None or not nxt.get('chords'):
                continue
            first_chord_pitches = nxt['chords'][0][2]
            pivot_chord = [(0.0, transition_beats, first_chord_pitches)]
            window_start = max(0.0, cur['total_beats'] - transition_beats)
            in_window = [n for n in cur['mapped'] if n[0] >= window_start]
            rest = [n for n in cur['mapped'] if n[0] < window_start]
            shifted = [(o - window_start, p, d, v) for (o, p, d, v) in in_window]
            snapped = snap_strong_beats_to_chords(shifted, pivot_chord, beats_per_bar, max_correction=3)
            back = [(o + window_start, p, d, v) for (o, p, d, v) in snapped]
            cur['mapped'] = rest + back
            if verbose:
                print(f"  Transición: últimos {transition_beats} tiempos de la sección {i + 1} "
                      f"pivotan hacia el acorde inicial de la sección {i + 2}")

    # ── concatenar con offsets absolutos ────────────────────────────────
    all_melody, all_chords, all_cp = [], [], []
    prog_changes_mel, prog_changes_harm = [], []
    bass_program = 32
    for res in results:
        if res is None:
            continue
        base = res['seg_start']
        prog_changes_mel.append((base, res['preset']['melody_program']))
        prog_changes_harm.append((base, res['preset']['harmony_program']))
        bass_program = res['preset']['bass_program']
        for (o, p, d, v) in res['mapped']:
            all_melody.append((o + base, p, d, v))
        if res['chords']:
            for (cs, cd, pitches) in res['chords']:
                all_chords.append((cs + base, cd, pitches))
        if res['counterpoint_notes']:
            for (o, p, d, v) in res['counterpoint_notes']:
                all_cp.append((o + base, p, d, v))

    if not all_melody:
        raise RuntimeError(f"{midi_path}: ninguna sección produjo notas")

    stem = Path(midi_path).stem
    out_directory = out_dir or os.path.dirname(os.path.abspath(midi_path))
    tag = "_".join(s for _, s, _ in sections)
    output_path = os.path.join(out_directory, f"{stem}.sections_{tag}.mid")

    write_theme_midi_sectioned(
        all_melody, all_chords, all_cp, tempo_bpm, beats_per_bar, output_path,
        prog_changes_mel, prog_changes_harm, bass_program=bass_program,
    )

    if verbose:
        print(f"  → {output_path}")

    result = {
        'input': midi_path, 'output': output_path,
        'sections': [{'bar': b, 'style': s, **extra} for b, s, extra in sections],
        'transition_beats': transition_beats, 'body_tail': body_tail,
        'source_key': f"{PITCH_NAMES[src_root_pc]} {src_mode}",
        'n_notes': len(all_melody),
    }
    if report:
        report_path = os.path.join(out_directory, f"{stem}.sections_{tag}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  → {report_path}")

    return result


# ═══════════════════════════════════════════════════════════
# CATÁLOGO
# ═══════════════════════════════════════════════════════════

def print_catalog():
    print("\n╔══ ESCALAS DISPONIBLES (--scale, con --style custom) ═══════════════╗")
    for name, intervals in SCALES.items():
        print(f"  {name:20s} {len(intervals):2d} notas/8va   {list(intervals)}")
    print("╚══════════════════════════════════════════════════════════════════╝")

    print("\n╔══ ESTILOS DISPONIBLES (--style) ════════════════════════════════════╗")
    for name, preset in STYLE_PRESETS.items():
        print(f"  {name:22s} [{preset['method']:8s}] {preset['desc']}")
    print("╚══════════════════════════════════════════════════════════════════╝\n")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='THEME TRANSFORMER — Mapeo escalar y transformación temática MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estilos disponibles:
  heroic_minor, dorian_heroic, whole_tone, octatonic, chromatic_magical,
  chromatic_counterpoint, pentatonic, double_harmonic, acoustic, elvish,
  got_fragment, chromatic_mediant, custom

Ejemplos:
  python theme_transformer.py tema.mid
  python theme_transformer.py tema.mid --style octatonic chromatic_magical
  python theme_transformer.py tema.mid --style got_fragment --verbose
  python theme_transformer.py tema.mid --style whole_tone --chord-tone-priority
  python theme_transformer.py tema.mid --style custom --scale double_harmonic --method interval
  python theme_transformer.py tema.mid --style elvish --start-degree 2
  python theme_transformer.py tema.mid --style octatonic --octatonic-variant hw
  python theme_transformer.py tema.mid --style whole_tone --wt-variant 1
  python theme_transformer.py tema.mid --style got_fragment --body-tail --chord-tone-priority
  python theme_transformer.py tema.mid --sections "0:heroic_minor,16:double_harmonic,32:elvish" \\
      --transition-beats 2 --verbose
  python theme_transformer.py --catalog
        """
    )
    p.add_argument('input', nargs='?', help='Archivo(s) MIDI de entrada (admite glob)')
    p.add_argument('--catalog', action='store_true',
                   help='Mostrar catálogo de escalas y estilos y salir')
    p.add_argument('--style', nargs='+',
                   default=['heroic_minor', 'whole_tone', 'octatonic', 'chromatic_magical'],
                   metavar='S',
                   help='Estilos a aplicar (default: heroic_minor whole_tone octatonic '
                        'chromatic_magical). Ignorado si se usa --sections')
    p.add_argument('--sections', default=None, metavar='"BAR:ESTILO,BAR:ESTILO,..."',
                   help='Transformación progresiva por tramos, e.g. '
                        '"0:heroic_minor,16:double_harmonic,32:elvish" (la 1ª sección debe '
                        'empezar en el compás 0). Sustituye a --style')
    p.add_argument('--transition-beats', type=float, default=0,
                   help='(con --sections) Nº de tiempos al final de cada sección que pivotan '
                        'hacia el acorde inicial de la siguiente (default: 0 = corte seco)')
    p.add_argument('--key', default=None, metavar='KEY',
                   help='Tonalidad origen forzada, e.g. "D minor". Si no se indica, se autodetecta')
    p.add_argument('--tempo', type=float, default=None,
                   help='BPM de salida forzado (default: detectado del MIDI)')
    p.add_argument('--beats-per-bar', type=int, default=None,
                   help='Beats por compás (default: detectado del MIDI)')
    p.add_argument('--start-degree', type=int, default=0,
                   help='Desplazamiento adicional de grado de escala sobre el preset (default: 0)')
    p.add_argument('--chord-tone-priority', action='store_true',
                   help='Corrige notas en tiempo fuerte hacia el tono de acorde más cercano '
                        '(<= 3 semitonos)')
    p.add_argument('--fragment', action='store_true',
                   help='Fragmenta la melodía en motivos (separados por silencios) antes de mapear')
    p.add_argument('--fragment-order', nargs='+', type=int, default=None,
                   help='Orden/repetición de fragmentos por índice, e.g. '
                        '--fragment-order 0 0 1 2 1')
    p.add_argument('--metric-shift', action='store_true',
                   help='Desplaza la melodía para que la primera nota caiga en el downbeat')
    p.add_argument('--body-tail', action='store_true',
                   help="Distingue 'cuerpo' (1ª mitad de cada motivo, protegido por "
                        "--chord-tone-priority) de 'cola' (2ª mitad, con adornos melódicos "
                        "libres) — ver análisis body/tail de How to Train Your Dragon")
    p.add_argument('--wt-variant', choices=['auto', '0', '1'], default='auto',
                   help='Forzar la colección de tonos enteros (WT0/WT1) en vez de autodetectar')
    p.add_argument('--octatonic-variant', choices=['auto', 'wh', 'hw'], default='auto',
                   help='Forzar la variante octatónica (tono-semitono/semitono-tono) en vez de '
                        'autodetectar')
    p.add_argument('--pentatonic-variant', choices=['auto', 'major', 'minor'], default='auto',
                   help='Forzar pentatónica mayor/menor en vez de deducirla del modo detectado')
    p.add_argument('--scale', default=None,
                   help='(con --style custom) escala destino, ver --catalog')
    p.add_argument('--method', default=None, choices=['degree', 'interval'],
                   help='(con --style custom) método de mapeo')
    p.add_argument('--out-dir', default=None,
                   help='Carpeta de salida (default: junto al MIDI de entrada)')
    p.add_argument('--report', action='store_true',
                   help='Guardar reporte JSON por cada transformación')
    p.add_argument('--seed', type=int, default=42,
                   help='Semilla aleatoria (default: 42)')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.catalog:
        print_catalog()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    wt_variant = int(args.wt_variant) if args.wt_variant in ('0', '1') else 'auto'
    oct_variant = args.octatonic_variant

    import glob
    midi_files = glob.glob(args.input) or [args.input]
    midi_files = [f for f in midi_files if f.endswith(('.mid', '.midi'))]

    if not midi_files:
        print(f"[ERROR] No se encontraron archivos MIDI: {args.input}")
        sys.exit(1)

    if args.sections:
        try:
            sections = parse_sections_spec(args.sections)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        for midi_path in midi_files:
            try:
                transform_theme_sections(
                    midi_path=midi_path,
                    sections=sections,
                    transition_beats=args.transition_beats,
                    key_override=args.key,
                    tempo_override=args.tempo,
                    beats_per_bar_override=args.beats_per_bar,
                    start_degree_extra=args.start_degree,
                    chord_tone_priority=args.chord_tone_priority,
                    fragment=args.fragment,
                    fragment_order=args.fragment_order,
                    metric_shift=args.metric_shift,
                    body_tail=args.body_tail,
                    wt_variant=wt_variant,
                    oct_variant=oct_variant,
                    pentatonic_variant=args.pentatonic_variant,
                    out_dir=args.out_dir,
                    report=args.report,
                    verbose=args.verbose,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[ERROR] {midi_path} / --sections: {e}")
                if args.verbose:
                    traceback.print_exc()
        return

    for midi_path in midi_files:
        for style in args.style:
            try:
                transform_theme(
                    midi_path=midi_path,
                    style=style,
                    key_override=args.key,
                    tempo_override=args.tempo,
                    beats_per_bar_override=args.beats_per_bar,
                    start_degree_extra=args.start_degree,
                    chord_tone_priority=args.chord_tone_priority,
                    fragment=args.fragment,
                    fragment_order=args.fragment_order,
                    metric_shift=args.metric_shift,
                    body_tail=args.body_tail,
                    wt_variant=wt_variant,
                    oct_variant=oct_variant,
                    pentatonic_variant=args.pentatonic_variant,
                    custom_scale=args.scale,
                    custom_method=args.method,
                    out_dir=args.out_dir,
                    report=args.report,
                    verbose=args.verbose,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[ERROR] {midi_path} / estilo '{style}': {e}")
                if args.verbose:
                    traceback.print_exc()


if __name__ == '__main__':
    main()
