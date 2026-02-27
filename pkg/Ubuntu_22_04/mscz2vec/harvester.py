#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          HARVESTER  v1.0                                    ║
║              Extracción de fragmentos musicales desde MIDIs                 ║
║                                                                              ║
║  Segmenta MIDIs en fragmentos reutilizables usando criterios musicales:     ║
║    · Detección de fronteras por curva de novedad (SSM checkerboard)         ║
║    · Detección de cadencias (auténtica, plagal, semicadencia, engañosa)     ║
║    · Repeticiones de motivos vía algoritmo Re-Pair (Sequitur simplificado)  ║
║    · Silencio / pausa larga como frontera natural                           ║
║    · Combinación ponderada de todos los criterios                           ║
║                                                                              ║
║  Sufijos de salida:                                                          ║
║    .frag_A.mid, .frag_B.mid …  — secciones formales (forma detectada)      ║
║    .motif_01.mid, .motif_02.mid — motivos repetidos (Sequitur)             ║
║    .cadence_auth.mid            — fragmento antes de cadencia auténtica     ║
║    .cadence_plag.mid            — fragmento antes de cadencia plagal        ║
║    .cadence_half.mid            — fragmento antes de semicadencia           ║
║    .cadence_dec.mid             — fragmento antes de cadencia engañosa      ║
║    .phrase_01.mid …             — frases (entre cadencias)                 ║
║    .texture_01.mid …            — segmentos por textura rítmica            ║
║                                                                              ║
║  USO:                                                                        ║
║    python harvester.py archivo.mid                                           ║
║    python harvester.py *.mid --out-dir fragmentos/                          ║
║    python harvester.py archivo.mid --mode all                               ║
║    python harvester.py archivo.mid --mode form cadence motif               ║
║    python harvester.py archivo.mid --min-bars 2 --max-bars 16              ║
║    python harvester.py archivo.mid --novelty-threshold 0.2                  ║
║    python harvester.py archivo.mid --report                                 ║
║                                                                              ║
║  MODOS (--mode):                                                             ║
║    form      — secciones formales por curva de novedad                      ║
║    cadence   — frases entre cadencias y contextos de cadencia               ║
║    motif     — motivos repetidos (Sequitur)                                 ║
║    texture   — segmentos por densidad y textura rítmica                     ║
║    all       — todos los modos (por defecto)                                ║
║                                                                              ║
║  DEPENDENCIAS: mido, music21, numpy, scipy, sklearn                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import argparse
import tempfile
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# music21 para análisis armónico y cadencias
try:
    from music21 import converter, stream, roman, note, chord
    from music21.analysis import discrete
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[AVISO] music21 no disponible. Los modos 'cadence' y 'motif' estarán limitados.")

# ═══════════════════════════════════════════════════════════
# UTILIDADES MIDI (basadas en mido, sin music21)
# ═══════════════════════════════════════════════════════════

def midi_to_absolute(mid: MidiFile):
    """
    Convierte todos los tracks a eventos con tiempo absoluto en ticks.
    Devuelve lista de (abs_tick, track_idx, msg).
    """
    events = []
    for ti, track in enumerate(mid.tracks):
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            events.append((abs_t, ti, msg))
    events.sort(key=lambda x: x[0])
    return events


def get_tempo_map(mid: MidiFile):
    """Extrae el mapa de tempo: lista de (abs_tick, tempo_us)."""
    tempo_map = [(0, 500000)]  # default 120 BPM
    abs_t = 0
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_t, msg.tempo))
    tempo_map.sort(key=lambda x: x[0])
    return tempo_map


def ticks_to_seconds(ticks, tpb, tempo_map):
    """Convierte ticks absolutos a segundos usando el mapa de tempo."""
    seconds = 0.0
    prev_tick, prev_tempo = 0, 500000
    for tm_tick, tm_tempo in tempo_map:
        if ticks <= tm_tick:
            break
        dt = min(ticks, tm_tick) - prev_tick
        seconds += dt * prev_tempo / (tpb * 1_000_000)
        prev_tick, prev_tempo = tm_tick, tm_tempo
    dt = ticks - prev_tick
    seconds += dt * prev_tempo / (tpb * 1_000_000)
    return seconds


def get_total_ticks(mid: MidiFile):
    max_t = 0
    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
        max_t = max(max_t, abs_t)
    return max_t


def tpb_to_beats(ticks, tpb):
    return ticks / tpb


def beats_to_ticks(beats, tpb):
    return int(beats * tpb)


def get_time_signature(mid: MidiFile):
    """Devuelve (numerator, denominator) del primer time signature encontrado."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return msg.numerator, msg.denominator
    return 4, 4  # default


def estimate_bar_ticks(mid: MidiFile):
    """Estima la duración de un compás en ticks."""
    num, den = get_time_signature(mid)
    beats_per_bar = num * (4 / den)
    return int(beats_per_bar * mid.ticks_per_beat)


def extract_note_events(mid: MidiFile):
    """
    Extrae todas las notas activas como lista de:
    (start_tick, end_tick, pitch, velocity, channel, track_idx)
    """
    notes = []
    active = {}  # (track, channel, pitch) -> (start_tick, velocity)
    events = midi_to_absolute(mid)

    for abs_t, ti, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (ti, msg.channel, msg.note)
            active[key] = (abs_t, msg.velocity)
        elif msg.type in ('note_off', 'note_on') and (msg.type == 'note_off' or msg.velocity == 0):
            key = (ti, msg.channel, msg.note)
            if key in active:
                start_t, vel = active.pop(key)
                notes.append((start_t, abs_t, msg.note, vel, msg.channel, ti))

    # Notas que no tienen note_off (cierre forzado al final)
    total = get_total_ticks(mid)
    for (ti, ch, pitch), (start_t, vel) in active.items():
        notes.append((start_t, total, pitch, vel, ch, ti))

    notes.sort(key=lambda x: x[0])
    return notes


def notes_in_range(notes, start_tick, end_tick):
    """Filtra notas que empiezan dentro del rango [start_tick, end_tick)."""
    return [n for n in notes if start_tick <= n[0] < end_tick]


def slice_midi(mid: MidiFile, start_tick: int, end_tick: int) -> MidiFile:
    """
    Extrae el segmento [start_tick, end_tick) del MIDI original.
    Reescribe los tiempos relativamente al nuevo inicio.
    Incluye todos los meta-mensajes relevantes.
    """
    out = MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = MidiTrack()
        abs_t = 0
        prev_abs = start_tick  # punto de referencia para tiempos relativos
        in_range_started = False

        # Primero recopilamos meta-mensajes anteriores al inicio (tempo, key, time sig)
        # que deben reproducirse al principio del fragmento
        preamble_meta = []
        events_in_range = []

        abs_t2 = 0
        for msg in track:
            abs_t2 += msg.time
            if msg.is_meta:
                if abs_t2 <= start_tick:
                    preamble_meta.append((abs_t2, msg))
                elif abs_t2 < end_tick:
                    events_in_range.append((abs_t2, msg))
            else:
                if start_tick <= abs_t2 < end_tick:
                    events_in_range.append((abs_t2, msg))
                elif abs_t2 < start_tick:
                    # Notas sostenidas que empezaron antes: las ignoramos
                    # (se podrían manejar con note_on al inicio, pero complica)
                    pass

        # Escribir preamble (con tiempo 0)
        for _, msg in preamble_meta:
            new_track.append(msg.copy(time=0))

        # Escribir eventos en rango con tiempos relativos al inicio del segmento
        prev = start_tick
        for abs_t3, msg in sorted(events_in_range, key=lambda x: x[0]):
            dt = abs_t3 - prev
            new_track.append(msg.copy(time=max(0, dt)))
            prev = abs_t3

        # Cerrar track
        if not any(m.type == 'end_of_track' for _, m in events_in_range if hasattr(m, 'type') and m.is_meta):
            new_track.append(MetaMessage('end_of_track', time=0))

        out.tracks.append(new_track)

    return out


# ═══════════════════════════════════════════════════════════
# ANÁLISIS: DESCRIPTORES POR VENTANA DE COMPÁS
# ═══════════════════════════════════════════════════════════

def descriptor_for_window(notes_in_window, bar_ticks):
    """
    Vector de características simple para una ventana de compases.
    Usado para SSM y novelty detection.
    """
    if not notes_in_window:
        return np.zeros(8)

    pitches = [n[2] for n in notes_in_window]
    velocities = [n[3] for n in notes_in_window]
    durations = [n[1] - n[0] for n in notes_in_window]
    intervals = np.diff(pitches) if len(pitches) > 1 else [0]

    density = len(notes_in_window) / max(bar_ticks, 1)
    mean_pitch = np.mean(pitches)
    pitch_std = np.std(pitches)
    mean_vel = np.mean(velocities)
    vel_var = np.var(velocities)
    mean_dur = np.mean(durations)
    prop_asc = np.sum(np.array(intervals) > 0) / max(len(intervals), 1)
    prop_jump = np.sum(np.abs(np.array(intervals)) > 5) / max(len(intervals), 1)

    return np.array([density, mean_pitch / 127, pitch_std / 64, mean_vel / 127,
                     vel_var / (127**2), mean_dur / max(bar_ticks, 1),
                     prop_asc, prop_jump])


def compute_ssm(descriptors):
    """Matriz de Auto-Similitud (coseno)."""
    X = np.array(descriptors, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    return np.dot(X_norm, X_norm.T)


def get_checkerboard_kernel(size):
    """Kernel checkerboard suavizado para novelty detection."""
    m = size // 2
    kernel = np.array([[1, -1], [-1, 1]])
    kernel = np.kron(kernel, np.ones((m, m)))
    return gaussian_filter(kernel.astype(float), sigma=size / 6)


# ═══════════════════════════════════════════════════════════
# MODO: FORM — detección de secciones formales (SSM + novelty)
# ═══════════════════════════════════════════════════════════

def detect_form_boundaries(notes, bar_ticks, total_ticks,
                            kernel_size=8, threshold=0.15):
    """
    Calcula la curva de novedad sobre ventanas de compases y devuelve
    los ticks de frontera detectados.
    """
    if bar_ticks <= 0:
        return [0, total_ticks]

    # Dividir en compases
    n_bars = max(1, int(total_ticks / bar_ticks))
    bar_starts = [i * bar_ticks for i in range(n_bars)]

    descriptors = []
    for bs in bar_starts:
        w_notes = [n for n in notes if bs <= n[0] < bs + bar_ticks]
        descriptors.append(descriptor_for_window(w_notes, bar_ticks))

    if len(descriptors) < kernel_size:
        return [0, total_ticks]

    ssm = compute_ssm(descriptors)
    kernel = get_checkerboard_kernel(min(kernel_size, len(descriptors) // 2 * 2))

    N = ssm.shape[0]
    novelty = np.zeros(N)
    m = kernel.shape[0] // 2

    for i in range(m, N - m):
        sub = ssm[i - m: i + m, i - m: i + m]
        if sub.shape == kernel.shape:
            novelty[i] = np.sum(sub * kernel)

    if np.max(np.abs(novelty)) > 0:
        novelty /= np.max(np.abs(novelty))

    peaks, props = find_peaks(novelty, height=threshold, distance=2)

    boundaries_ticks = [0] + [bar_starts[p] for p in peaks if p < len(bar_starts)] + [total_ticks]
    boundaries_ticks = sorted(set(boundaries_ticks))
    return boundaries_ticks, novelty, bar_starts


# ═══════════════════════════════════════════════════════════
# MODO: CADENCE — detección de cadencias con music21
# ═══════════════════════════════════════════════════════════

CADENCE_PATTERNS = {
    # (penúltimo_grado, último_grado) : tipo
    ('V',  'I' ): 'auth',
    ('V',  'i' ): 'auth',
    ('VII','I' ): 'auth',
    ('VII','i' ): 'auth',
    ('IV', 'I' ): 'plag',
    ('iv', 'i' ): 'plag',
    ('I',  'V' ): 'half',
    ('i',  'V' ): 'half',
    ('ii', 'V' ): 'half',
    ('IV', 'V' ): 'half',
    ('V',  'vi'): 'dec',
    ('V',  'VI'): 'dec',
}


def detect_cadences_m21(score_m21):
    """
    Detecta cadencias en una partitura music21.
    Devuelve lista de (offset_beat, tipo_str, grado_anterior, grado_final)
    """
    cadences = []
    if score_m21 is None:
        return cadences

    try:
        key = score_m21.analyze('key')
    except Exception:
        return cadences

    try:
        chords_obj = score_m21.chordify()
        chord_list = list(chords_obj.flatten().getElementsByClass('Chord'))
    except Exception:
        return cadences

    if len(chord_list) < 2:
        return cadences

    for i in range(1, len(chord_list)):
        ch_prev = chord_list[i - 1]
        ch_curr = chord_list[i]
        try:
            rn_prev = roman.romanNumeralFromChord(ch_prev, key).figure
            rn_curr = roman.romanNumeralFromChord(ch_curr, key).figure

            # Normalizar: quitar inversiones y modificaciones secundarias
            def base_figure(fig):
                return fig.split('/')[0].rstrip('0123456789').rstrip('+°')

            bp = base_figure(rn_prev)
            bc = base_figure(rn_curr)

            for (pat_p, pat_c), ctype in CADENCE_PATTERNS.items():
                if bp.upper() == pat_p.upper() and bc.upper() == pat_c.upper():
                    offset = ch_curr.offset
                    cadences.append((offset, ctype, bp, bc))
                    break
        except Exception:
            continue

    return cadences


def load_score_m21(midi_path):
    """Carga un MIDI con music21."""
    if not MUSIC21_OK:
        return None
    try:
        score = converter.parse(midi_path)
        return score
    except Exception as e:
        print(f"  [AVISO] music21 no pudo cargar {midi_path}: {e}")
        return None


def offset_to_ticks(offset_beats, tpb):
    """Convierte offset en beats (music21) a ticks."""
    return int(offset_beats * tpb)


# ═══════════════════════════════════════════════════════════
# MODO: MOTIF — detección de motivos repetidos (Re-Pair)
# ═══════════════════════════════════════════════════════════

def notes_to_interval_sequence(notes, quantize=True):
    """
    Convierte lista de notas a secuencia de intervalos melódicos.
    Incluye duración cuantizada como segundo elemento.
    Devuelve: [(interval, dur_symbol), ...] e índices originales.
    """
    if len(notes) < 2:
        return [], []

    # Ordenar por inicio
    notes = sorted(notes, key=lambda n: n[0])

    def quantize_dur(dur_ticks, tpb=480):
        q = dur_ticks / tpb  # en beats
        if q < 0.4:
            return 'S'   # corta
        elif q < 0.9:
            return 'Q'   # negra ~
        elif q < 1.6:
            return 'H'   # corchea blanca ~
        else:
            return 'L'   # larga

    seq = []
    idxs = []
    for i in range(len(notes) - 1):
        interval = notes[i + 1][2] - notes[i][2]
        dur = quantize_dur(notes[i][1] - notes[i][0])
        if quantize:
            interval = max(-12, min(12, interval))
        seq.append((interval, dur))
        idxs.append(i)

    return seq, idxs


def repair_find_motifs(seq, min_len=3, max_len=12, min_count=2):
    """
    Algoritmo Re-Pair simplificado: encuentra sub-secuencias repetidas.
    Devuelve lista de (motif_tuple, count, [start_positions]).
    """
    best = {}

    for length in range(min_len, min(max_len + 1, len(seq) - 1)):
        counts = defaultdict(list)
        for i in range(len(seq) - length + 1):
            key = tuple(seq[i: i + length])
            counts[key].append(i)

        for key, positions in counts.items():
            # Filtrar posiciones solapadas
            filtered = []
            prev = -length
            for pos in sorted(positions):
                if pos >= prev + length:
                    filtered.append(pos)
                    prev = pos
            if len(filtered) >= min_count:
                # Criterio: longitud * repeticiones
                score = length * len(filtered)
                if key not in best or score > best[key][0]:
                    best[key] = (score, len(filtered), filtered)

    # Ordenar por score descendente y eliminar solapamientos
    motifs = sorted(best.items(), key=lambda x: -x[1][0])

    result = []
    covered = set()
    for motif_key, (score, count, positions) in motifs[:20]:  # top 20
        clean_pos = []
        for pos in positions:
            span = set(range(pos, pos + len(motif_key)))
            if not span & covered:
                clean_pos.append(pos)
                covered |= span
        if len(clean_pos) >= min_count:
            result.append((motif_key, len(clean_pos), clean_pos))

    return result


def motif_positions_to_ticks(positions, notes_sorted, motif_len):
    """
    Convierte posiciones en la secuencia de intervalos a rangos de ticks.
    positions: índices en seq de intervalos → nota notes[pos] a notes[pos+motif_len]
    """
    ranges = []
    for pos in positions:
        if pos + motif_len < len(notes_sorted):
            start = notes_sorted[pos][0]
            end = notes_sorted[pos + motif_len][1]
            ranges.append((start, end))
    return ranges


# ═══════════════════════════════════════════════════════════
# MODO: TEXTURE — segmentación por textura rítmica
# ═══════════════════════════════════════════════════════════

def texture_descriptor(notes, start_tick, end_tick, tpb):
    """
    Descriptor de textura rítmica para una ventana.
    Devuelve vector [density, syncopation, polyphony, velocity_var].
    """
    window = [n for n in notes if start_tick <= n[0] < end_tick]
    dur = max(end_tick - start_tick, 1)

    if not window:
        return np.zeros(4)

    # Densidad
    density = len(window) / (dur / tpb)

    # Polifonía media (notas simultáneas)
    times = sorted(set([n[0] for n in window]))
    poly_counts = []
    for t in times:
        simultaneous = sum(1 for n in window if n[0] <= t < n[1])
        poly_counts.append(simultaneous)
    polyphony = np.mean(poly_counts)

    # Síncopa: notas que empiezan en tiempo débil
    beat_ticks = tpb
    syncopated = sum(1 for n in window if (n[0] % beat_ticks) > beat_ticks * 0.4)
    syncopation = syncopated / max(len(window), 1)

    # Variación de velocidad
    velocities = [n[3] for n in window]
    vel_var = np.std(velocities) / 127 if len(velocities) > 1 else 0

    return np.array([density / 10, polyphony / 8, syncopation, vel_var])


def detect_texture_changes(notes, bar_ticks, total_ticks, tpb,
                            window_bars=2, threshold=0.25):
    """
    Detecta cambios de textura comparando ventanas adyacentes.
    Devuelve lista de ticks donde la textura cambia significativamente.
    """
    if bar_ticks <= 0:
        return [0, total_ticks]

    n_bars = max(1, int(total_ticks / bar_ticks))
    window_size = max(1, window_bars)
    boundaries = [0]

    descriptors = []
    window_starts = []
    for i in range(0, n_bars, max(1, window_size // 2)):
        start = i * bar_ticks
        end = start + window_size * bar_ticks
        descriptors.append(texture_descriptor(notes, start, end, tpb))
        window_starts.append(start)

    for i in range(1, len(descriptors)):
        diff = np.linalg.norm(descriptors[i] - descriptors[i - 1])
        if diff > threshold:
            boundaries.append(window_starts[i])

    boundaries.append(total_ticks)
    return sorted(set(boundaries))


# ═══════════════════════════════════════════════════════════
# GUARDAR FRAGMENTO
# ═══════════════════════════════════════════════════════════

def save_fragment(mid: MidiFile, start_tick: int, end_tick: int,
                  out_path: str, min_notes=4):
    """
    Extrae y guarda un fragmento MIDI si contiene suficientes notas.
    Devuelve True si se guardó.
    """
    # Verificar que hay notas en el rango
    all_notes = extract_note_events(mid)
    notes_in = notes_in_range(all_notes, start_tick, end_tick)

    if len(notes_in) < min_notes:
        return False

    try:
        fragment = slice_midi(mid, start_tick, end_tick)
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        fragment.save(out_path)
        return True
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar {out_path}: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def harvest(midi_path: str,
            out_dir: str = None,
            modes: list = None,
            min_bars: int = 2,
            max_bars: int = 32,
            novelty_threshold: float = 0.15,
            report: bool = False):
    """
    Ejecuta el pipeline completo de extracción de fragmentos.

    Args:
        midi_path:          Ruta al MIDI de entrada.
        out_dir:            Carpeta de salida. Si None, usa la carpeta del MIDI.
        modes:              Lista de modos a ejecutar ('form','cadence','motif','texture','all').
        min_bars:           Mínimo de compases para guardar un fragmento.
        max_bars:           Máximo de compases para guardar un fragmento.
        novelty_threshold:  Umbral de detección de fronteras (0-1).
        report:             Si True, guarda un JSON con el análisis completo.
    """
    if modes is None or 'all' in modes:
        modes = ['form', 'cadence', 'motif', 'texture']

    midi_path = str(midi_path)
    stem = Path(midi_path).stem

    if out_dir is None:
        out_dir = str(Path(midi_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  HARVESTER  ·  {os.path.basename(midi_path)}")
    print(f"{'═'*60}")

    # Cargar MIDI
    try:
        mid = MidiFile(midi_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {midi_path}: {e}")
        return {}

    tpb = mid.ticks_per_beat
    total_ticks = get_total_ticks(mid)
    bar_ticks = estimate_bar_ticks(mid)
    total_bars = max(1, int(total_ticks / bar_ticks))
    all_notes = extract_note_events(mid)

    min_ticks = min_bars * bar_ticks
    max_ticks = max_bars * bar_ticks

    print(f"  ticks_per_beat : {tpb}")
    print(f"  total_ticks    : {total_ticks}")
    print(f"  bar_ticks      : {bar_ticks}")
    print(f"  total_bars     : {total_bars}")
    print(f"  total_notas    : {len(all_notes)}")
    print(f"  modos          : {modes}")
    print(f"  min/max_bars   : {min_bars}/{max_bars}")

    saved = []   # lista de dicts con info de cada fragmento guardado
    report_data = {
        'source': midi_path,
        'tpb': tpb,
        'total_bars': total_bars,
        'total_notes': len(all_notes),
        'fragments': []
    }

    # ─────────────────────────────────────────
    # MODO: FORM
    # ─────────────────────────────────────────
    if 'form' in modes:
        print(f"\n  ── MODO: FORM ──")
        try:
            result = detect_form_boundaries(
                all_notes, bar_ticks, total_ticks,
                kernel_size=min(8, total_bars // 2),
                threshold=novelty_threshold
            )
            if isinstance(result, tuple):
                boundaries, novelty_curve, bar_starts = result
            else:
                boundaries = result
                novelty_curve = []

            # Asignar letra a cada sección via clustering simple
            # Reusamos descriptores de compases para asignar etiquetas
            section_descs = []
            valid_sections = []
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                if min_ticks <= (e - s) <= max_ticks:
                    w_notes = [n for n in all_notes if s <= n[0] < e]
                    section_descs.append(descriptor_for_window(w_notes, bar_ticks))
                    valid_sections.append((s, e))

            # Clustering para asignar letras
            section_labels = []
            if len(section_descs) >= 2:
                from sklearn.cluster import AgglomerativeClustering
                try:
                    n_clust = min(len(section_descs), 8)
                    clf = AgglomerativeClustering(
                        n_clusters=n_clust,
                        metric='cosine',
                        linkage='average'
                    )
                    raw_labels = clf.fit_predict(section_descs)
                    # Normalizar a letras
                    mapping = {}
                    next_letter = ord('A')
                    for lbl in raw_labels:
                        if lbl not in mapping:
                            mapping[lbl] = chr(next_letter)
                            next_letter += 1
                    section_labels = [mapping[l] for l in raw_labels]
                except Exception:
                    section_labels = [chr(ord('A') + i) for i in range(len(valid_sections))]
            else:
                section_labels = [chr(ord('A') + i) for i in range(len(valid_sections))]

            # Guardar secciones
            letter_count = Counter()
            for (s, e), letter in zip(valid_sections, section_labels):
                letter_count[letter] += 1
                n = letter_count[letter]
                suffix = f"frag_{letter}" if n == 1 else f"frag_{letter}{n}"
                out_path = os.path.join(out_dir, f"{stem}.{suffix}.mid")
                if save_fragment(mid, s, e, out_path):
                    bars_here = int((e - s) / bar_ticks)
                    print(f"    [FORM] {suffix:15s}  compases {int(s/bar_ticks)+1:3d}-{int(e/bar_ticks):3d}  → {os.path.basename(out_path)}")
                    info = {'type': 'form', 'label': letter, 'start_bar': int(s/bar_ticks)+1,
                            'end_bar': int(e/bar_ticks), 'bars': bars_here,
                            'start_tick': s, 'end_tick': e, 'path': out_path}
                    saved.append(info)
                    report_data['fragments'].append(info)

            form_str = ''.join(section_labels)
            print(f"    Forma detectada: {form_str}")
            report_data['form_sequence'] = form_str

        except Exception as e:
            print(f"    [ERROR] Modo form: {e}")
            if report:
                traceback.print_exc()

    # ─────────────────────────────────────────
    # MODO: CADENCE
    # ─────────────────────────────────────────
    if 'cadence' in modes:
        print(f"\n  ── MODO: CADENCE ──")

        if not MUSIC21_OK:
            print("    [OMITIDO] music21 no disponible.")
        else:
            score_m21 = load_score_m21(midi_path)
            cadences = detect_cadences_m21(score_m21) if score_m21 else []

            if not cadences:
                print("    No se detectaron cadencias.")
            else:
                print(f"    {len(cadences)} cadencias detectadas.")

                # Convertir offsets a ticks
                cadence_ticks = []
                for (offset_beat, ctype, g1, g2) in cadences:
                    tick = offset_to_ticks(offset_beat, tpb)
                    cadence_ticks.append((tick, ctype, g1, g2))

                # Guardar contexto de cada cadencia (N compases antes)
                context_bars = min(4, max_bars)
                cad_type_count = Counter()
                for tick, ctype, g1, g2 in cadence_ticks:
                    context_start = max(0, tick - context_bars * bar_ticks)
                    context_end = min(total_ticks, tick + bar_ticks)
                    if (context_end - context_start) < min_ticks:
                        continue
                    cad_type_count[ctype] += 1
                    n = cad_type_count[ctype]
                    suffix = f"cadence_{ctype}_{n:02d}"
                    out_path = os.path.join(out_dir, f"{stem}.{suffix}.mid")
                    if save_fragment(mid, context_start, context_end, out_path):
                        print(f"    [CAD]  {ctype:6s} ({g1}→{g2})  tick {tick:6d}  → {os.path.basename(out_path)}")
                        info = {'type': 'cadence', 'cadence_type': ctype,
                                'degree_prev': g1, 'degree_curr': g2,
                                'cadence_tick': tick,
                                'start_tick': context_start, 'end_tick': context_end,
                                'path': out_path}
                        saved.append(info)
                        report_data['fragments'].append(info)

                # Guardar frases entre cadencias
                phrase_boundaries = sorted([0] + [t for t, *_ in cadence_ticks] + [total_ticks])
                phrase_count = 0
                for i in range(len(phrase_boundaries) - 1):
                    s = phrase_boundaries[i]
                    e = phrase_boundaries[i + 1]
                    if min_ticks <= (e - s) <= max_ticks:
                        phrase_count += 1
                        suffix = f"phrase_{phrase_count:02d}"
                        out_path = os.path.join(out_dir, f"{stem}.{suffix}.mid")
                        if save_fragment(mid, s, e, out_path):
                            bars_here = int((e - s) / bar_ticks)
                            print(f"    [PHRASE] {suffix:12s}  {bars_here} compases  → {os.path.basename(out_path)}")
                            info = {'type': 'phrase', 'phrase_num': phrase_count,
                                    'start_tick': s, 'end_tick': e,
                                    'bars': bars_here, 'path': out_path}
                            saved.append(info)
                            report_data['fragments'].append(info)

    # ─────────────────────────────────────────
    # MODO: MOTIF
    # ─────────────────────────────────────────
    if 'motif' in modes:
        print(f"\n  ── MODO: MOTIF ──")

        notes_sorted = sorted(all_notes, key=lambda n: n[0])
        seq, idxs = notes_to_interval_sequence(notes_sorted)

        if len(seq) < 6:
            print("    Secuencia demasiado corta para detectar motivos.")
        else:
            motifs = repair_find_motifs(seq, min_len=3, max_len=10, min_count=2)

            if not motifs:
                print("    No se encontraron motivos repetidos.")
            else:
                print(f"    {len(motifs)} motivos repetidos encontrados.")
                for mi, (motif_key, count, positions) in enumerate(motifs[:12]):  # top 12
                    ranges = motif_positions_to_ticks(positions, notes_sorted, len(motif_key))

                    # Guardar primera aparición como referencia
                    if ranges:
                        s, e = ranges[0]
                        # Padding de medio compás antes y después
                        s = max(0, s - bar_ticks // 2)
                        e = min(total_ticks, e + bar_ticks // 2)

                        if (e - s) < min_ticks:
                            continue

                        suffix = f"motif_{mi+1:02d}"
                        out_path = os.path.join(out_dir, f"{stem}.{suffix}.mid")
                        if save_fragment(mid, s, e, out_path):
                            intervals_str = ','.join([str(iv) for iv, _ in motif_key[:6]])
                            print(f"    [MOTIF] {suffix}  ×{count} repeticiones  intervalos=[{intervals_str}]  → {os.path.basename(out_path)}")
                            info = {'type': 'motif', 'motif_num': mi + 1,
                                    'repetitions': count,
                                    'interval_pattern': [iv for iv, _ in motif_key],
                                    'start_tick': s, 'end_tick': e,
                                    'all_positions_ticks': ranges[:5],
                                    'path': out_path}
                            saved.append(info)
                            report_data['fragments'].append(info)

                        # Guardar también cada aparición individual del motivo
                        for ri, (rs, re) in enumerate(ranges[1:], 2):
                            rs2 = max(0, rs - bar_ticks // 2)
                            re2 = min(total_ticks, re + bar_ticks // 2)
                            if (re2 - rs2) < min_ticks:
                                continue
                            suffix_occ = f"motif_{mi+1:02d}_occ{ri:02d}"
                            out_path2 = os.path.join(out_dir, f"{stem}.{suffix_occ}.mid")
                            if save_fragment(mid, rs2, re2, out_path2):
                                info2 = {'type': 'motif_occurrence',
                                         'motif_num': mi + 1, 'occurrence': ri,
                                         'start_tick': rs2, 'end_tick': re2,
                                         'path': out_path2}
                                saved.append(info2)
                                report_data['fragments'].append(info2)

    # ─────────────────────────────────────────
    # MODO: TEXTURE
    # ─────────────────────────────────────────
    if 'texture' in modes:
        print(f"\n  ── MODO: TEXTURE ──")
        try:
            tex_boundaries = detect_texture_changes(
                all_notes, bar_ticks, total_ticks, tpb,
                window_bars=2, threshold=0.25
            )
            tex_count = 0
            for i in range(len(tex_boundaries) - 1):
                s, e = tex_boundaries[i], tex_boundaries[i + 1]
                if min_ticks <= (e - s) <= max_ticks:
                    tex_count += 1
                    suffix = f"texture_{tex_count:02d}"
                    out_path = os.path.join(out_dir, f"{stem}.{suffix}.mid")
                    if save_fragment(mid, s, e, out_path):
                        bars_here = int((e - s) / bar_ticks)
                        w_notes = notes_in_range(all_notes, s, e)
                        desc = texture_descriptor(w_notes, s, e, tpb)
                        print(f"    [TEX]  {suffix:12s}  {bars_here:2d} compases  "
                              f"dens={desc[0]:.2f} poly={desc[1]:.2f} sync={desc[2]:.2f}  "
                              f"→ {os.path.basename(out_path)}")
                        info = {'type': 'texture', 'texture_num': tex_count,
                                'start_tick': s, 'end_tick': e,
                                'bars': bars_here,
                                'descriptor': desc.tolist(),
                                'path': out_path}
                        saved.append(info)
                        report_data['fragments'].append(info)

        except Exception as e:
            print(f"    [ERROR] Modo texture: {e}")
            if report:
                traceback.print_exc()

    # ─────────────────────────────────────────
    # RESUMEN
    # ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Total fragmentos guardados: {len(saved)}")
    by_type = Counter(f['type'] for f in saved)
    for t, c in by_type.items():
        print(f"    {t:20s}: {c}")

    # Guardar reporte JSON
    if report:
        report_path = os.path.join(out_dir, f"{stem}.harvest_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Reporte guardado en: {report_path}")

    print(f"{'═'*60}\n")
    return report_data


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='HARVESTER — Extracción de fragmentos musicales desde MIDIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python harvester.py bach.mid
  python harvester.py *.mid --out-dir fragmentos/
  python harvester.py beethoven.mid --mode form cadence --min-bars 4 --max-bars 16
  python harvester.py pieza.mid --mode motif --report
  python harvester.py *.mid --mode all --novelty-threshold 0.2
        """
    )
    p.add_argument('inputs', nargs='+', help='Archivos MIDI de entrada (acepta globs)')
    p.add_argument('--out-dir', default=None,
                   help='Carpeta de salida. Por defecto: misma carpeta que el MIDI.')
    p.add_argument('--mode', nargs='+',
                   choices=['form', 'cadence', 'motif', 'texture', 'all'],
                   default=['all'],
                   help='Modos de segmentación a ejecutar (default: all)')
    p.add_argument('--min-bars', type=int, default=2,
                   help='Mínimo de compases para guardar un fragmento (default: 2)')
    p.add_argument('--max-bars', type=int, default=32,
                   help='Máximo de compases para guardar un fragmento (default: 32)')
    p.add_argument('--novelty-threshold', type=float, default=0.15,
                   help='Umbral para detección de fronteras formales (0-1, default: 0.15)')
    p.add_argument('--report', action='store_true',
                   help='Guardar reporte JSON con el análisis completo')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolver globs / múltiples archivos
    import glob
    midi_files = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        if expanded:
            midi_files.extend(expanded)
        elif pattern.endswith('.mid') or pattern.endswith('.midi'):
            print(f"[AVISO] No se encontró: {pattern}")
        else:
            midi_files.append(pattern)

    midi_files = [f for f in midi_files if f.endswith(('.mid', '.midi'))]

    if not midi_files:
        print("[ERROR] No se encontraron archivos MIDI.")
        sys.exit(1)

    print(f"Procesando {len(midi_files)} archivo(s) MIDI...")

    all_results = {}
    for midi_path in midi_files:
        result = harvest(
            midi_path=midi_path,
            out_dir=args.out_dir,
            modes=args.mode,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            novelty_threshold=args.novelty_threshold,
            report=args.report
        )
        all_results[midi_path] = result

    total_fragments = sum(len(r.get('fragments', [])) for r in all_results.values())
    print(f"\n✓ Procesamiento completo. {total_fragments} fragmentos totales extraídos.")


if __name__ == '__main__':
    main()
