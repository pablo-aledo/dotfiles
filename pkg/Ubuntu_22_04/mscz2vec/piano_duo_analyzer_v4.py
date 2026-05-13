#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PIANO DUO ANALYZER  v2.0                                   ║
║         Visualizador de piano (mano izq. + mano der.) con análisis armónico  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DESCRIPCIÓN:                                                                ║
║    Toma un MIDI de dos pistas (mano izquierda y mano derecha del piano) y   ║
║    genera un informe HTML interactivo con tres filas:                        ║
║                                                                              ║
║    · Fila 1 — Piano Roll mano DERECHA                                        ║
║      Cada nota se colorea según su posición en el acorde de la mano         ║
║      izquierda en ese instante:                                               ║
║        Azul     → 1ª nota del acorde (la más grave / fundamental)            ║
║        Amarillo → 2ª nota del acorde                                          ║
║        Verde    → 3ª nota del acorde                                          ║
║        Blanco   → 4ª nota del acorde                                          ║
║        Rojo     → no pertenece al acorde                                      ║
║                                                                              ║
║    · Fila 2 — Piano Roll mano IZQUIERDA                                       ║
║      Notas del acorde coloreadas por la nota raíz detectada:                 ║
║        Do → Rojo        Re → Naranja     Mi → Amarillo                       ║
║        Fa → Verde       Sol → Azul claro La → Azul                           ║
║        Si → Violeta                                                           ║
║                                                                              ║
║    · Fila 3 — Acordes detectados (por compás)                                 ║
║      Nombre del acorde + función armónica (T/PD/D/Dsec)                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO                                                                         ║
║                                                                              ║
║    python3 piano_duo_analyzer.py archivo.mid                                 ║
║    python3 piano_duo_analyzer.py archivo.mid -o informe.html                 ║
║    python3 piano_duo_analyzer.py archivo.mid --right 0 --left 1             ║
║    python3 piano_duo_analyzer.py --test                                      ║
║                                                                              ║
║  OPCIONES                                                                    ║
║    midi             Ruta al archivo MIDI de entrada                          ║
║    -o / --output    Ruta del HTML de salida (default: <midi>_piano.html)    ║
║    --right N        Índice de pista (0-based) para mano derecha (default 0) ║
║    --left  N        Índice de pista (0-based) para mano izquierda (default 1)║
║    --arp-window B   Ventana en tiempos para agrupar arpegios (default 1.0)  ║
║                     Aumentar para arpegios lentos; 0 = sin agrupación        ║
║    --key-window N   Compases para detección local de tonalidad (default 8)  ║
║                     0 = solo tonalidad global                                 ║
║    --test           Genera un MIDI sintético de prueba y lo analiza          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS:  mido  (pip install mido)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import sys, os, json, math, argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import mido
except ImportError:
    sys.exit("ERROR: instala mido  →  pip install mido")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE COLOR
# ═══════════════════════════════════════════════════════════════════════════════

# Posición en el acorde (mano derecha)
CHORD_POSITION_COLORS = {
    0: '#3b82f6',   # 1ª nota del acorde → Azul
    1: '#fbbf24',   # 2ª nota del acorde → Amarillo
    2: '#22c55e',   # 3ª nota del acorde → Verde
    3: '#f8fafc',   # 4ª nota del acorde → Blanco
   -1: '#ef4444',   # No pertenece       → Rojo
}
CHORD_POSITION_LABELS = {
    0: '1ª (fundamental)',
    1: '2ª nota',
    2: '3ª nota',
    3: '4ª nota',
   -1: 'Fuera del acorde',
}

# Nota raíz del acorde (mano izquierda) — por pitch class
ROOT_NOTE_COLORS = {
    0:  '#ef4444',   # Do  → Rojo
    1:  '#f97316',   # Do# → Naranja (entre Do y Re)
    2:  '#f97316',   # Re  → Naranja
    3:  '#eab308',   # Re# → Amarillo (entre Re y Mi)
    4:  '#fbbf24',   # Mi  → Amarillo
    5:  '#22c55e',   # Fa  → Verde
    6:  '#16a34a',   # Fa# → Verde oscuro
    7:  '#38bdf8',   # Sol → Azul claro
    8:  '#38bdf8',   # Sol#→ Azul claro
    9:  '#3b82f6',   # La  → Azul
    10: '#7c3aed',   # La# → Violeta (entre La y Si)
    11: '#8b5cf6',   # Si  → Violeta
}

# Mapeamos exactamente a las notas "blancas" con los colores pedidos,
# accidentales reciben el color de la nota más cercana
ROOT_NOTE_COLORS_EXACT = {
    0:  '#ef4444',   # Do  → Rojo
    1:  '#c084fc',   # Do# → Púrpura (entre rojo y naranja)
    2:  '#f97316',   # Re  → Naranja
    3:  '#f5a623',   # Re# → Naranja-amarillo (entre naranja y amarillo)
    4:  '#fbbf24',   # Mi  → Amarillo
    5:  '#22c55e',   # Fa  → Verde
    6:  '#0ea5e9',   # Fa# → Azul cielo (entre verde y azul claro)
    7:  '#38bdf8',   # Sol → Azul claro
    8:  '#06b6d4',   # Sol#→ Cian (entre azul claro y azul)
    9:  '#3b82f6',   # La  → Azul
    10: '#e879f9',   # La# → Fucsia (entre azul y violeta)
    11: '#8b5cf6',   # Si  → Violeta
}

ROOT_NOTE_NAMES = ['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si']
NOTE_NAMES      = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Funciones armónicas
HARM_FUNC_COLORS = {
    'T':    '#3b82f6',
    'PD':   '#10b981',
    'D':    '#ef4444',
    'Dsec': '#f59e0b',
    'Other':'#6b7280',
    '—':    '#1e293b',
}
HARM_FUNC_LABELS = {
    'T':    'T — Tónica',
    'PD':   'PD — Predominante',
    'D':    'D — Dominante',
    'Dsec': 'Dsec — Dom. secundaria',
    'Other':'Otro',
    '—':    'Silencio',
}

CHORD_TEMPLATES = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'dom7': [0, 4, 7, 10],
    'min7': [0, 3, 7, 10],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
}

_KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
_KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

# ═══════════════════════════════════════════════════════════════════════════════
# MIDI PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tempo_map(mid) -> List[Tuple[int, int]]:
    events = [(0, 500000)]
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                events.append((tick, msg.tempo))
    events.sort()
    return events

def beats_per_bar(mid) -> float:
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return msg.numerator
    return 4.0

def extract_track(mid, track_index: int) -> Dict:
    """Extract notes from a single track by index."""
    tpb = mid.ticks_per_beat
    bpb = beats_per_bar(mid)

    if track_index >= len(mid.tracks):
        return {'notes': [], 'name': f'Track {track_index}', 'bpb': bpb, 'tpb': tpb}

    track = mid.tracks[track_index]
    name = track.name or f'Track {track_index}'
    notes_on = {}
    notes = []
    tick = 0

    for msg in track:
        tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes_on[(msg.channel, msg.note)] = (tick, msg.velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in notes_on:
                start_tick, vel = notes_on.pop(key)
                dur_ticks  = tick - start_tick
                start_beat = start_tick / tpb
                dur_beat   = dur_ticks / tpb
                start_bar  = int(start_beat / bpb)
                notes.append({
                    'pitch':      msg.note,
                    'vel':        vel,
                    'start_tick': start_tick,
                    'end_tick':   tick,
                    'start_beat': start_beat,
                    'end_beat':   start_beat + dur_beat,
                    'dur_beat':   dur_beat,
                    'start_bar':  start_bar,
                })

    notes.sort(key=lambda n: n['start_tick'])
    return {'notes': notes, 'name': name, 'bpb': bpb, 'tpb': tpb}

# ═══════════════════════════════════════════════════════════════════════════════
# CHORD DETECTION (from left hand) — arpeggio-aware + local key detection
# ═══════════════════════════════════════════════════════════════════════════════

def _ks_score(pc_hist: List[float]) -> Tuple[int, str]:
    """Krumhansl-Schmuckler: return (tonic_pc, mode) for a pitch-class histogram."""
    total = sum(pc_hist) or 1.0
    pc_norm = [v / total for v in pc_hist]
    best_score, best_tonic, best_mode = -999.0, 0, 'major'
    for tonic in range(12):
        for mode, profile in [('major', _KS_MAJOR), ('minor', _KS_MINOR)]:
            rotated = [profile[(i - tonic) % 12] for i in range(12)]
            m1 = sum(pc_norm) / 12
            m2 = sum(rotated) / 12
            num = sum((pc_norm[i]-m1)*(rotated[i]-m2) for i in range(12))
            d1  = math.sqrt(sum((pc_norm[i]-m1)**2 for i in range(12)))
            d2  = math.sqrt(sum((rotated[i]-m2)**2 for i in range(12)))
            r   = num / (d1 * d2 + 1e-9)
            if r > best_score:
                best_score, best_tonic, best_mode = r, tonic, mode
    return best_tonic, best_mode

def detect_key(all_notes: List[Dict]) -> Tuple[int, str]:
    """Global Krumhansl-Schmuckler key detection over all notes."""
    pc_hist = [0.0] * 12
    for n in all_notes:
        pc_hist[n['pitch'] % 12] += n['vel']
    return _ks_score(pc_hist)

def detect_key_per_bar(all_notes: List[Dict], n_bars: int,
                        bpb: float, window_bars: int = 8) -> List[Tuple[int, str]]:
    """
    Local key detection: slide a window of `window_bars` compases over the piece.
    Returns a list of (tonic_pc, mode) per bar.
    Bars with no notes inherit the nearest neighbour's key.
    window_bars=0 → use global key everywhere (fast path).
    """
    # Global fallback
    global_key = detect_key(all_notes)
    if window_bars <= 0 or n_bars < 2:
        return [global_key] * n_bars

    # Build per-bar pc histograms
    bar_hist = [[0.0] * 12 for _ in range(n_bars)]
    for n in all_notes:
        b = n['start_bar']
        if 0 <= b < n_bars:
            bar_hist[b][n['pitch'] % 12] += n['vel']

    keys_per_bar: List[Optional[Tuple[int, str]]] = [None] * n_bars
    half = window_bars // 2

    for b in range(n_bars):
        lo  = max(0, b - half)
        hi  = min(n_bars, b + half + 1)
        combined = [0.0] * 12
        for bi in range(lo, hi):
            for pc in range(12):
                combined[pc] += bar_hist[bi][pc]
        if sum(combined) > 0:
            keys_per_bar[b] = _ks_score(combined)

    # Fill gaps (bars with no notes) by propagating from neighbours
    # Forward pass
    last = global_key
    for b in range(n_bars):
        if keys_per_bar[b] is not None:
            last = keys_per_bar[b]
        else:
            keys_per_bar[b] = last
    # Backward pass (fills leading gaps)
    last = global_key
    for b in range(n_bars - 1, -1, -1):
        if keys_per_bar[b] is not None and sum(bar_hist[b]) > 0:
            last = keys_per_bar[b]
        elif keys_per_bar[b] is None:
            keys_per_bar[b] = last

    return keys_per_bar  # type: ignore

def _best_chord(pitches: List[int]) -> Tuple[str, int, int, str, str]:
    """
    Find the best-matching chord for a list of pitches, with inversion detection.

    Scoring: overlap - penalty (0.3 per extra pitch class outside template).
    Tiebreakers when scores are equal:
      1. Prefer root == bass_pc (root-position over inversion)
      2. Prefer fewer template notes (parsimony: maj before maj7)
    """
    if not pitches:
        return '—', -1, -1, '', ''
    pcs     = list({p % 12 for p in pitches})
    bass_pc = min(pitches) % 12

    if len(pcs) == 1:
        return NOTE_NAMES[pcs[0]], pcs[0], bass_pc, '', ''

    best_name, best_score, best_root, best_ctype, best_tmpl = '?', -1.0, pcs[0], '', [0]
    for root in range(12):
        for ctype, template in CHORD_TEMPLATES.items():
            chord_pcs = set((root + i) % 12 for i in template)
            overlap   = len(chord_pcs & set(pcs))
            penalty   = len(set(pcs) - chord_pcs) * 0.3
            score     = overlap - penalty

            if score > best_score:
                best_score = score
                best_name  = f'{NOTE_NAMES[root]}{ctype}'
                best_root  = root
                best_ctype = ctype
                best_tmpl  = template
            elif score == best_score:
                cur_root_is_bass = (best_root == bass_pc)
                new_root_is_bass = (root == bass_pc)
                if new_root_is_bass and not cur_root_is_bass:
                    best_name  = f'{NOTE_NAMES[root]}{ctype}'
                    best_root  = root
                    best_ctype = ctype
                    best_tmpl  = template
                elif new_root_is_bass == cur_root_is_bass:
                    if len(template) < len(best_tmpl):
                        best_name  = f'{NOTE_NAMES[root]}{ctype}'
                        best_root  = root
                        best_ctype = ctype
                        best_tmpl  = template

    # Detect inversion
    struct_pcs = [(best_root + iv) % 12 for iv in best_tmpl]
    inv_str = ''; slash = ''
    if bass_pc != best_root and bass_pc in struct_pcs:
        pos     = struct_pcs.index(bass_pc)
        inv_map = {1: '1ª inv', 2: '2ª inv', 3: '3ª inv'}
        inv_str = inv_map.get(pos, f'{pos}ª inv')
        slash   = f'/{NOTE_NAMES[bass_pc]}'

    return best_name + slash, best_root, bass_pc, inv_str, best_ctype

def build_chord_timeline(harmony_notes: List[Dict],
                          arp_window: float = 1.0,
                          bpb: float = 4.0,
                          bass_notes: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Build a chord event timeline.

    Groups are defined EXCLUSIVELY by onsets in `harmony_notes`.
    Notes from `bass_notes` (the other track) are added to whichever
    group covers their beat for richer chord detection, but never create
    new groups on their own.

    Sub-groups within a bar are created only when harmony_notes has
    multiple onsets AND at least one coincides with a bass_notes onset
    (real chord change). A single sustained block chord stays as one group
    even if the other hand has many melody notes in between.

    end_beat = next group start_beat (no gaps).
    Consecutive identical chords are merged.
    """
    if not harmony_notes:
        return []

    from collections import defaultdict as _dd2

    harm_by_bar: Dict[int, List[Dict]] = _dd2(list)
    for n in harmony_notes:
        harm_by_bar[n['start_bar']].append(n)

    bass_by_bar: Dict[int, List[Dict]] = _dd2(list)
    for n in (bass_notes or []):
        bass_by_bar[n['start_bar']].append(n)

    groups: List[List[Dict]] = []

    for bar in sorted(harm_by_bar.keys()):
        h_notes = sorted(harm_by_bar[bar], key=lambda n: n['start_beat'])
        b_notes = sorted(bass_by_bar.get(bar, []), key=lambda n: n['start_beat'])

        h_onsets = sorted(set(round(n['start_beat'], 3) for n in h_notes))
        b_onset_set = {round(n['start_beat'], 3) for n in b_notes}

        # Split onsets: harmony onsets that coincide with a bass onset
        split_onsets = [ho for ho in h_onsets
                        if any(abs(ho - bo) < 0.05 for bo in b_onset_set)]

        boundaries = split_onsets if len(split_onsets) > 1 else [h_onsets[0]]

        def group_idx(beat, _b=boundaries):
            return max(0, sum(1 for b in _b if b <= beat + 0.01) - 1)

        sub_groups: Dict[int, List[Dict]] = {}
        for n in h_notes:
            gi = group_idx(n['start_beat'])
            sub_groups.setdefault(gi, []).append(n)
        # b_notes drive split boundaries only — do NOT add their pitches
        # to the chord group, as they are melody notes, not harmony notes

        for gi in sorted(sub_groups.keys()):
            groups.append(sub_groups[gi])

    chord_events: List[Dict] = []
    for grp in groups:
        pitches = sorted(n['pitch'] for n in grp)
        start_b = min(n['start_beat'] for n in grp)
        end_b   = max(n['end_beat']   for n in grp)
        chord_label, root_pc, bass_pc, inv_str, ctype = _best_chord(pitches)
        chord_events.append({
            'start_beat': start_b,
            'end_beat':   end_b,
            'chord_name': chord_label,
            'root_pc':    root_pc,
            'bass_pc':    bass_pc,
            'inv_str':    inv_str,
            'ctype':      ctype,
            'pitches':    pitches,
        })

    for i in range(len(chord_events) - 1):
        chord_events[i]['end_beat'] = chord_events[i + 1]['start_beat']

    merged: List[Dict] = []
    for ev in chord_events:
        if merged and merged[-1]['chord_name'] == ev['chord_name']:
            merged[-1]['end_beat'] = ev['end_beat']
            merged[-1]['pitches']  = sorted(set(merged[-1]['pitches'] + ev['pitches']))
        else:
            merged.append(dict(ev))

    return merged

def chord_position_for_note(pitch: int, chord_event: Optional[Dict]) -> int:
    """
    Returns the position (0-3) of the note in the chord ordered by ascending
    pitch class, or -1 if the note does not belong to the chord.

    Position is defined by the ORDER of distinct pitch classes present in the
    chord (from the actual sounding pitches), sorted from lowest to highest:

      Amaj/C# sounding pitches [C#4, E4, A4, C#5]
      → distinct PCs sorted: [C#(1), E(4), A(9)]
      → C#→0 (blue), E→1 (yellow), A→2 (green)

    This matches the user's expectation: "the 3rd note of the chord from low
    to high gets green", regardless of theoretical root or inversion.
    """
    if chord_event is None:
        return -1
    pitches = chord_event.get('pitches', [])
    if not pitches:
        return -1

    # Build list of distinct pitch classes in ascending order
    # (use the lowest actual pitch for each PC to determine sort order)
    pc_to_lowest: Dict[int, int] = {}
    for p in pitches:
        pc = p % 12
        if pc not in pc_to_lowest or p < pc_to_lowest[pc]:
            pc_to_lowest[pc] = p

    sorted_pcs = [pc for pc, _ in sorted(pc_to_lowest.items(), key=lambda x: x[1])]

    note_pc = pitch % 12
    if note_pc in sorted_pcs:
        return min(sorted_pcs.index(note_pc), 3)
    return -1

def find_chord_at_beat(chord_timeline: List[Dict], beat: float) -> Optional[Dict]:
    """Binary-search-like lookup of the active chord at a given beat."""
    result = None
    for ev in chord_timeline:
        if ev['start_beat'] <= beat:
            result = ev
        else:
            break
    return result

def harmonic_function(chord_name: str, tonic: int, mode: str) -> str:
    if chord_name in ('—', '?', ''):
        return '—'
    # Extract root from chord name
    root = None
    for length in (2, 1):
        candidate = chord_name[:length]
        if candidate in NOTE_NAMES:
            root = NOTE_NAMES.index(candidate)
            break
    if root is None:
        return 'Other'

    ctype    = chord_name[2:] if len(chord_name) > 2 and chord_name[:2] in NOTE_NAMES else chord_name[1:]
    interval = (root - tonic) % 12
    is_minor = mode == 'minor'

    if interval == 0: return 'T'
    if interval == 9 and not is_minor: return 'T'
    if interval == 3 and is_minor: return 'T'
    if interval == 7: return 'D'
    if interval == 11: return 'D'
    if 'dom7' in ctype and interval not in (0, 7): return 'Dsec'
    if interval in (2, 5): return 'PD'
    if interval == 9 and is_minor: return 'PD'
    return 'Other'

def compute_chords_per_bar(left_notes: List[Dict], n_bars: int,
                            bpb: float,
                            keys_per_bar: List[Tuple[int, str]]) -> List[Dict]:
    """
    Detect the dominant chord for each bar and classify its harmonic function
    using the *local* key (keys_per_bar[b]) instead of a single global tonic.
    Within each bar all left-hand pitches are pooled (arpeggio-friendly).
    """
    bar_notes: Dict[int, List[Dict]] = defaultdict(list)
    for n in left_notes:
        bar_notes[n['start_bar']].append(n)

    result = []
    for b in range(n_bars):
        bn = bar_notes.get(b, [])
        tonic, mode = keys_per_bar[b] if b < len(keys_per_bar) else (0, 'major')

        if not bn:
            result.append({'label': '—', 'function': '—',
                           'color': HARM_FUNC_COLORS['—'], 'root_pc': -1,
                           'tonic': tonic, 'mode': mode})
            continue

        pitches    = [n['pitch'] for n in bn]
        chord_label, best_root, bass_pc, inv_str, ctype = _best_chord(pitches)
        func  = harmonic_function(chord_label, tonic, mode)
        color = HARM_FUNC_COLORS.get(func, HARM_FUNC_COLORS['Other'])
        result.append({'label': chord_label, 'function': func,
                       'color': color, 'root_pc': best_root,
                       'bass_pc': bass_pc, 'inv_str': inv_str, 'ctype': ctype,
                       'tonic': tonic, 'mode': mode})
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# HTML BUILD
# ═══════════════════════════════════════════════════════════════════════════════


def build_html(mid_path: str,
               right_track: Dict, left_track: Dict,
               chord_timeline: List[Dict],
               chords_per_bar: List[Dict],
               global_tonic: int, global_mode: str,
               keys_per_bar: List[Tuple[int, str]],
               src_per_bar: List[str],
               n_bars: int,
               tension: List[float] = None,
               density: List[float] = None,
               avg_velocity: List[float] = None,
               motifs: List[Dict] = None,
               cadences: List[Dict] = None,
               sections: List[Dict] = None,
               repeated_bars: Dict = None) -> str:

    import json as _json

    tension      = tension      or [0.0] * n_bars
    density      = density      or [0.0] * n_bars
    avg_velocity = avg_velocity or [0.5] * n_bars
    motifs       = motifs       or []
    cadences     = cadences     or []
    sections     = sections     or [{'bar': 0, 'end': n_bars, 'label': 'A', 'color': '#3b82f6'}]
    repeated_bars = repeated_bars or {}

    bpb         = right_track['bpb']
    total_beats = n_bars * bpb

    PIANO_W   = 32        # width of piano keyboard strip (px)
    CANVAS_W  = 1600      # roll canvas width
    ROW_H     = 160       # height of each piano roll
    CHORD_H   = 52        # chord row height
    LABEL_W   = 210       # left label column width

    NOTE_NAMES_ES = ['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si']

    def src_at_bar(bar):
        return src_per_bar[bar] if bar < len(src_per_bar) else 'left'

    def bx(beat):  return round(beat / total_beats * CANVAS_W, 2)
    def bw(dur):   return round(max(2.0, dur / total_beats * CANVAS_W), 2)

    # ── Pitch range per track (with padding) ──────────────────────────────
    def pitch_range(notes, pad=2):
        if not notes: return 48, 72
        pitches = [n['pitch'] for n in notes]
        return max(0, min(pitches)-pad), min(127, max(pitches)+pad)

    rh_p_min, rh_p_max = pitch_range(right_track['notes'])
    lh_p_min, lh_p_max = pitch_range(left_track['notes'])
    rh_p_range = max(1, rh_p_max - rh_p_min)
    lh_p_range = max(1, lh_p_max - lh_p_min)

    def note_y(pitch, p_min, p_range, row_h):
        note_h = max(3.0, round(row_h / p_range * 0.85, 2))
        frac   = (pitch - p_min) / p_range
        y      = round((1.0 - frac) * (row_h - note_h), 2)
        return y, note_h

    # ── Piano keyboard data (for SVG strip) ──────────────────────────────
    def piano_keys(p_min, p_max, row_h):
        """Return list of key rects for an SVG piano strip."""
        keys = []
        p_range = max(1, p_max - p_min)
        for pitch in range(p_min, p_max + 1):
            pc = pitch % 12
            is_black = pc in (1, 3, 6, 8, 10)
            y, h = note_y(pitch, p_min, p_range, row_h)
            keys.append({'pitch': pitch, 'pc': pc, 'black': is_black, 'y': y, 'h': h})
        return keys

    rh_keys = piano_keys(rh_p_min, rh_p_max, ROW_H)
    lh_keys = piano_keys(lh_p_min, lh_p_max, ROW_H)

    # ── C guide lines (y positions of every C in range) ───────────────────
    def c_lines(p_min, p_max, row_h):
        lines = []
        p_range = max(1, p_max - p_min)
        for pitch in range(p_min, p_max + 1):
            if pitch % 12 == 0:
                y, h = note_y(pitch, p_min, p_range, row_h)
                lines.append({'y': round(y + h/2, 2),
                               'label': f'C{pitch//12 - 1}'})
        return lines

    rh_c_lines = c_lines(rh_p_min, rh_p_max, ROW_H)
    lh_c_lines = c_lines(lh_p_min, lh_p_max, ROW_H)

    # ── Chord background spans (for roll background tinting) ──────────────
    chord_bg = []
    for ev in chord_timeline:
        bar = int(ev['start_beat'] / bpb)
        cpb = chords_per_bar[bar] if bar < len(chords_per_bar) else {}
        color = cpb.get('color', '#1e293b')
        chord_bg.append({'x': bx(ev['start_beat']),
                         'w': bw(ev['end_beat'] - ev['start_beat']),
                         'color': color})

    # ── Note rects ────────────────────────────────────────────────────────
    def make_rects(notes, p_min, p_range, is_right):
        rects = []
        for n in notes:
            bar  = n['start_bar']
            src  = src_at_bar(bar)
            ev   = find_chord_at_beat(chord_timeline, n['start_beat'])
            if (is_right and src == 'right') or (not is_right and src == 'left'):
                bass_pc = ev['bass_pc'] if ev and ev.get('bass_pc',-1) >= 0 else n['pitch']%12
                color   = ROOT_NOTE_COLORS_EXACT.get(bass_pc, '#94a3b8')
                role    = 'harmony'
            else:
                pos   = chord_position_for_note(n['pitch'], ev)
                color = CHORD_POSITION_COLORS.get(pos, '#ef4444')
                role  = 'melody'
            x = bx(n['start_beat'])
            w = bw(n['dur_beat'])
            y, h = note_y(n['pitch'], p_min, p_range, ROW_H)
            pc   = n['pitch'] % 12
            name = NOTE_NAMES_ES[pc]
            chord_id = id(ev) if ev else -1
            vel  = round(n.get('vel_norm', n['vel']/127.0), 3)
            rects.append({'x': x, 'y': round(y,2), 'w': w, 'h': round(h,2),
                          'c': color, 'role': role, 'name': name,
                          'pitch': n['pitch'], 'cid': chord_id, 'v': vel})
        return rects

    right_rects = make_rects(right_track['notes'], rh_p_min, rh_p_range, True)
    left_rects  = make_rects(left_track['notes'],  lh_p_min, lh_p_range, False)

    # Add chord id index to chord_timeline events (for hover linking)
    ct_indexed = []
    for ev in chord_timeline:
        bar = int(ev['start_beat'] / bpb)
        cpb = chords_per_bar[bar] if bar < len(chords_per_bar) else {}
        local_key = (f"{ROOT_NOTE_NAMES[cpb['tonic']]} {cpb['mode']}"
                     if cpb.get('tonic',-1) >= 0 else '')
        ct_indexed.append({
            'x':     bx(ev['start_beat']),
            'w':     bw(ev['end_beat'] - ev['start_beat']),
            'label': ev['chord_name'],
            'func':  cpb.get('function','—'),
            'color': cpb.get('color', HARM_FUNC_COLORS['—']),
            'inv':   ev.get('inv_str',''),
            'id':    id(ev),
        })

    # ── New analysis data for JS ──────────────────────────────────────────
    # Per-bar curves (x = bar center in canvas pixels)
    bar_w_px = CANVAS_W / max(n_bars, 1)
    tension_pts   = [{'x': round((b+0.5)*bar_w_px, 1), 'v': round(tension[b], 3)}
                     for b in range(n_bars)]
    density_pts   = [{'x': round((b+0.5)*bar_w_px, 1), 'v': round(density[b], 3)}
                     for b in range(n_bars)]
    velocity_pts  = [{'x': round((b+0.5)*bar_w_px, 1), 'v': round(avg_velocity[b], 3)}
                     for b in range(n_bars)]

    # Motif spans: each occurrence tagged with which track it belongs to
    # A motif occurrence belongs to the track that is MELODY at that beat
    motif_spans_rh = []
    motif_spans_lh = []
    for m in motifs:
        for occ in m['occurrences']:
            bar = int(occ['start_beat'] / bpb)
            src = src_per_bar[bar] if bar < len(src_per_bar) else 'left'
            span = {
                'x':     bx(occ['start_beat']),
                'w':     bw(occ['end_beat'] - occ['start_beat']),
                'label': m['label'],
                'color': m['color'],
            }
            # melody is on RH when LH is harmony (src='left'), and vice versa
            if src == 'left':
                motif_spans_rh.append(span)
            else:
                motif_spans_lh.append(span)

    # Cadence markers
    cadence_marks = [{'x': bx(c['beat']), 'type': c['type'], 'color': c['color']}
                     for c in cadences]

    # Section spans
    section_spans = [{'x':     bx(s['bar'] * bpb),
                      'w':     bw((s['end'] - s['bar']) * bpb),
                      'label': s['label'],
                      'color': s['color']}
                     for s in sections]

    # ── Source spans ──────────────────────────────────────────────────────
    src_spans = []
    i = 0
    while i < n_bars:
        s = src_per_bar[i] if i < len(src_per_bar) else 'left'
        j = i
        while j < n_bars and (src_per_bar[j] if j < len(src_per_bar) else 'left') == s:
            j += 1
        src_spans.append({'x': bx(i*bpb), 'w': bw((j-i)*bpb),
                          'label': 'MD' if s=='right' else 'MI',
                          'color': '#3b82f6' if s=='right' else '#22c55e'})
        i = j

    # ── Repeated bars spans (for ruler overlay) ───────────────────────────
    # Each entry: {x, w, color, label} — shown as a thin colored stripe
    # in the bar ruler so repeated bars are immediately visible
    repeated_bar_spans = []
    for bar_idx, info in repeated_bars.items():
        repeated_bar_spans.append({
            'x':     bx(bar_idx * bpb),
            'w':     bw(bpb),
            'color': info['color'],
            'group': info['group'],
        })

    # ── Bar ruler HTML ─────────────────────────────────────────────────────
    bar_ticks_html = ''
    tick_every = max(1, n_bars // 32)
    for bi in range(0, n_bars, tick_every):
        x_px = round(bi * CANVAS_W / n_bars, 2)
        bar_ticks_html += f'<div class="bar-tick" style="left:{x_px}px">{bi+1}</div>'

    # ── Piano SVG ─────────────────────────────────────────────────────────
    def piano_svg(keys, row_h, p_min, p_max, c_lines_data):
        p_range = max(1, p_max - p_min)
        lines = [f'<svg width="{PIANO_W}" height="{row_h}" xmlns="http://www.w3.org/2000/svg">']
        # White keys background
        lines.append(f'<rect width="{PIANO_W}" height="{row_h}" fill="#1a2744"/>')
        # C guide lines (extend into roll — drawn on canvas, but mark on piano too)
        for cl in c_lines_data:
            lines.append(f'<line x1="0" y1="{cl["y"]:.1f}" x2="{PIANO_W}" y2="{cl["y"]:.1f}" '
                         f'stroke="#3b82f6" stroke-width="0.8" opacity="0.6"/>')
            lines.append(f'<text x="2" y="{cl["y"]-1:.1f}" fill="#3b82f6" '
                         f'font-size="6" font-family="monospace">{cl["label"]}</text>')
        # White keys
        for k in keys:
            if not k['black']:
                lines.append(f'<rect x="1" y="{k["y"]:.1f}" width="{PIANO_W-2}" height="{k["h"]:.1f}" '
                             f'fill="#e8eef8" rx="1" opacity="0.85"/>')
        # Black keys
        for k in keys:
            if k['black']:
                lines.append(f'<rect x="1" y="{k["y"]:.1f}" width="{int(PIANO_W*0.6)}" height="{k["h"]:.1f}" '
                             f'fill="#0a1628" rx="1"/>')
        lines.append('</svg>')
        return ''.join(lines)

    rh_piano_svg = piano_svg(rh_keys, ROW_H, rh_p_min, rh_p_max, rh_c_lines)
    lh_piano_svg = piano_svg(lh_keys, ROW_H, lh_p_min, lh_p_max, lh_c_lines)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_html  = '<span class="leg-title">FUENTE ARMÓNICA:</span>'
    legend_html += ('<div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>MD lleva armonía</div>'
                    '<div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>MI lleva armonía</div>')
    legend_html += '<span class="leg-title" style="margin-left:16px">POSICIÓN EN ACORDE (mano melódica):</span>'
    for pos, lbl in [(0,'1ª fund.'),(1,'2ª'),(2,'3ª'),(3,'7ª/4ª'),(-1,'Fuera')]:
        col = CHORD_POSITION_COLORS[pos]
        legend_html += (f'<div class="legend-item"><div class="legend-dot" style="background:{col};'
                        f'{"border:1px solid #555" if pos==3 else ""}"></div>{lbl}</div>')
    legend_html += '<span class="leg-title" style="margin-left:16px">BAJO (mano armónica):</span>'
    for pc, name in [(0,'Do'),(2,'Re'),(4,'Mi'),(5,'Fa'),(7,'Sol'),(9,'La'),(11,'Si')]:
        col = ROOT_NOTE_COLORS_EXACT[pc]
        legend_html += f'<div class="legend-item"><div class="legend-dot" style="background:{col}"></div>{name}</div>'
    legend_html += '<span class="leg-title" style="margin-left:16px">COMPASES REPETIDOS:</span>'
    legend_html += '<div class="legend-item"><div class="legend-dot" style="background:linear-gradient(to right,#ff6b6b,#ffd93d,#6bcb77)"></div>mismo color = mismo patrón melódico</div>'
    legend_html += '<span class="leg-title" style="margin-left:16px">FUNCIÓN:</span>'
    for func, col in HARM_FUNC_COLORS.items():
        if func == '—': continue
        legend_html += (f'<div class="legend-item"><div class="legend-dot" style="background:{col}"></div>'
                        f'{HARM_FUNC_LABELS.get(func,func)}</div>')

    title = os.path.basename(mid_path)
    tonic_name = ROOT_NOTE_NAMES[global_tonic]
    key_changes = []
    prev_key = None
    for b, k in enumerate(keys_per_bar):
        lbl = f'{ROOT_NOTE_NAMES[k[0]]} {k[1]}'
        if lbl != prev_key:
            key_changes.append((b+1, lbl)); prev_key = lbl
    key_summary = (f'{tonic_name} {global_mode}' if len(key_changes) <= 1
                   else ' → '.join(f'c.{b} {l}' for b,l in key_changes[:6]))

    # ─────────────────────────────────────────────────────────────────────
    # JavaScript — all rendering + interactivity
    # ─────────────────────────────────────────────────────────────────────
    js = f"""
(function() {{{{
  var CW = {CANVAS_W}, NBARS = {n_bars}, TOTAL_BEATS = {total_beats};

  // ── Zoom / pan state ──────────────────────────────────────────────────
  var zoom = 1.0;
  var MIN_ZOOM = 1.0, MAX_ZOOM = 20.0;

  // Native scroll container — much simpler than manual panX
  var scrollWrap = document.querySelector('.scroll-wrap');

  // Resize all canvases and piano wraps to match zoom
  function applyZoom() {{
    var w = Math.round({CANVAS_W} * zoom);
    document.querySelectorAll('.zoomable-canvas').forEach(function(cv) {{
      cv.style.width = w + 'px';
    }});
    document.querySelectorAll('.piano-wrap').forEach(function(p) {{
      p.style.flexShrink = '0';
    }});
    document.querySelectorAll('.ruler-track').forEach(function(rt) {{
      rt.style.width = w + 'px';
      rt.style.minWidth = w + 'px';
    }});
    // Reposition bar ticks using same pixel formula as canvas notes
    var ticks = document.querySelectorAll('.bar-tick');
    ticks.forEach(function(t) {{
      var b = parseInt(t.textContent) - 1;
      var xPx = b * ({CANVAS_W} / {n_bars}) * zoom;
      t.style.left = xPx + 'px';
    }});
  }}

  // Hover state

  // ── Data ──────────────────────────────────────────────────────────────
  var SRC_SPANS   = {_json.dumps(src_spans)};
  var CHORD_BG    = {_json.dumps(chord_bg)};
  var CHORD_ITEMS = {_json.dumps(ct_indexed)};
  var RH_RECTS    = {_json.dumps(right_rects)};
  var LH_RECTS    = {_json.dumps(left_rects)};
  var RH_C_LINES  = {_json.dumps(rh_c_lines)};
  var LH_C_LINES  = {_json.dumps(lh_c_lines)};
  var TENSION_PTS   = {_json.dumps(tension_pts)};
  var DENSITY_PTS   = {_json.dumps(density_pts)};
  var VELOCITY_PTS  = {_json.dumps(velocity_pts)};
  var MOTIF_SPANS_RH = {_json.dumps(motif_spans_rh)};
  var MOTIF_SPANS_LH = {_json.dumps(motif_spans_lh)};
  var CADENCE_MARKS = {_json.dumps(cadence_marks)};
  var SECTION_SPANS = {_json.dumps(section_spans)};
  var REPEATED_BAR_SPANS = {_json.dumps(repeated_bar_spans)};

  // ── Drawing helpers ───────────────────────────────────────────────────
  // Canvas coords: notes are drawn at their natural pixel positions (no zoom transform)
  // The canvas is CSS-scaled via style.width; the canvas resolution stays fixed.
  // So all drawing uses original coordinates — CSS handles zoom scaling.

  function clearCanvas(ctx, cv, bg) {{
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cv.width, cv.height);
  }}

  function drawBarLines(ctx, H) {{
    var barW = {CANVAS_W} / {n_bars};
    ctx.lineWidth = 0.5;
    for (var b = 0; b <= {n_bars}; b++) {{
      var x = b * barW;
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }}
  }}

  function drawCLines(ctx, cLines, H) {{
    cLines.forEach(function(cl) {{
      ctx.strokeStyle = 'rgba(59,130,246,0.18)';
      ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(0, cl.y); ctx.lineTo({CANVAS_W}, cl.y); ctx.stroke();
      ctx.fillStyle = 'rgba(59,130,246,0.45)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(cl.label, 2, cl.y);
    }});
  }}

  function drawChordBg(ctx, H) {{
    CHORD_BG.forEach(function(cb) {{
      ctx.fillStyle = cb.color + '18';
      ctx.fillRect(cb.x, 0, cb.w, H);
    }});
  }}

  function drawNotes(ctx, rects) {{
    // Pass 1: fill using original color, opacity from velocity
    rects.forEach(function(r) {{
      var alpha = 0.35 + (r.v || 0.7) * 0.65;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = r.c;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = 'rgba(0,0,0,0.45)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      if (r.w > 14 && r.h > 6) {{
        ctx.globalAlpha = Math.min(1.0, alpha + 0.15);
        ctx.fillStyle = 'rgba(0,0,0,0.8)';
        ctx.font = 'bold ' + Math.min(r.h - 2, 10) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(r.name, r.x + r.w / 2, r.y + r.h / 2);
        ctx.globalAlpha = 1.0;
      }}
    }});
    ctx.globalAlpha = 1.0;
  }}

  // ── Render functions ──────────────────────────────────────────────────
  function renderSrc() {{
    var cv = document.getElementById('cv_src'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#060c1a');
    SRC_SPANS.forEach(function(s) {{
      ctx.fillStyle = s.color + '30'; ctx.fillRect(s.x, 0, s.w, H);
      ctx.fillStyle = s.color; ctx.fillRect(s.x, 0, s.w, 3);
      if (s.w > 24) {{
        ctx.font = 'bold 9px monospace'; ctx.fillStyle = s.color;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(s.label, s.x + Math.min(s.w / 2, 30), H / 2 - 2);
      }}
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 0.5;
      ctx.strokeRect(s.x + 0.5, 0.5, s.w - 1, H - 1);
    }});
    // Repeated bars: thin stripe at bottom of src row
    REPEATED_BAR_SPANS.forEach(function(r) {{
      ctx.fillStyle = r.color + 'cc';
      ctx.fillRect(r.x, H - 4, r.w, 4);
    }});
  }}

  function drawRepeatedBarBg(ctx, H) {{
    REPEATED_BAR_SPANS.forEach(function(r) {{
      ctx.fillStyle = r.color + '14';
      ctx.fillRect(r.x, 0, r.w, H);
      // Top border stripe (2px)
      ctx.fillStyle = r.color + '55';
      ctx.fillRect(r.x, 0, r.w, 2);
    }});
  }}

  function renderRight() {{
    var cv = document.getElementById('cv_right'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#0b1628');
    drawRepeatedBarBg(ctx, H);
    drawMotifs(ctx, H, MOTIF_SPANS_RH);
    drawChordBg(ctx, H);
    drawBarLines(ctx, H);
    drawCLines(ctx, RH_C_LINES, H);
    drawNotes(ctx, RH_RECTS);
  }}

  function renderLeft() {{
    var cv = document.getElementById('cv_left'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#0b1420');
    drawRepeatedBarBg(ctx, H);
    drawMotifs(ctx, H, MOTIF_SPANS_LH);
    drawChordBg(ctx, H);
    drawBarLines(ctx, H);
    drawCLines(ctx, LH_C_LINES, H);
    drawNotes(ctx, LH_RECTS);
  }}

  function renderChords() {{
    var cv = document.getElementById('cv_chords'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#080f1e');
    CHORD_ITEMS.forEach(function(c) {{
      ctx.fillStyle = c.color + '22';
      ctx.fillRect(c.x, 0, c.w, H);
      ctx.fillStyle = c.color;
      ctx.fillRect(c.x, 0, c.w, 3);
      if (c.label && c.label !== '\u2014' && c.w > 6) {{
        ctx.save();
        ctx.beginPath(); ctx.rect(c.x, 0, c.w, H); ctx.clip();
        ctx.fillStyle = c.color;
        ctx.font = 'bold ' + (c.w > 30 ? 11 : 8) + 'px monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        var nameY = (c.inv && c.w > 40) ? H / 2 - 4 : H / 2 + 1;
        ctx.fillText(c.label, c.x + c.w / 2, nameY);
        if (c.inv && c.w > 40) {{
          ctx.fillStyle = c.color + 'bb'; ctx.font = '7px monospace';
          ctx.textBaseline = 'top';
          ctx.fillText(c.inv, c.x + c.w / 2, nameY + 9);
        }}
        if (c.func && c.func !== '\u2014' && c.w > 28) {{
          ctx.fillStyle = c.color + 'aa'; ctx.font = '8px monospace';
          ctx.textBaseline = 'bottom';
          ctx.fillText(c.func, c.x + c.w / 2, H - 2);
        }}
        ctx.restore();
      }}
      ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 0.5;
      ctx.strokeRect(c.x + 0.5, 0.5, c.w - 1, H - 1);
    }});
  }}


  // ── Motif overlay on piano rolls ─────────────────────────────────────
  function drawMotifs(ctx, H, spans) {{
    // Draw motif spans as bottom-edge colored bracket + very faint background
    // Drawn FIRST (before notes) so notes are always visible on top
    spans.forEach(function(m) {{
      // Very faint background tint — won't obscure notes drawn later
      ctx.fillStyle = m.color + '14';
      ctx.fillRect(m.x, 0, m.w, H);
      // Colored bracket at bottom edge
      ctx.fillStyle = m.color + 'cc';
      ctx.fillRect(m.x, H - 3, m.w, 3);
      // Left and right tick marks
      ctx.fillRect(m.x, H - 10, 2, 10);
      ctx.fillRect(m.x + m.w - 2, H - 10, 2, 10);
      // Label at bottom
      if (m.w > 12) {{
        ctx.fillStyle = m.color;
        ctx.font = 'bold 8px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(m.label, m.x + m.w / 2, H - 4);
      }}
    }});
  }}

  // ── Sections row ──────────────────────────────────────────────────────
  function renderSections() {{
    var cv = document.getElementById('cv_sections'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#060c18');
    SECTION_SPANS.forEach(function(s) {{
      ctx.fillStyle = s.color + '28';
      ctx.fillRect(s.x, 0, s.w, H);
      ctx.fillStyle = s.color;
      ctx.fillRect(s.x, 0, s.w, 4);
      ctx.strokeStyle = s.color + '55';
      ctx.lineWidth = 1;
      ctx.strokeRect(s.x + 0.5, 0.5, s.w - 1, H - 1);
      if (s.w > 20) {{
        ctx.fillStyle = s.color;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(s.label, s.x + Math.min(s.w / 2, s.x + 18), H / 2);
      }}
    }});
  }}

  // ── Curves row (tension / density / velocity) ─────────────────────────
  function renderCurves() {{
    var cv = document.getElementById('cv_curves'); if (!cv) return;
    var ctx = cv.getContext('2d'), H = cv.height;
    clearCanvas(ctx, cv, '#06091a');
    var pad = 4;
    var plotH = H - pad * 2;

    function drawCurve(pts, color, label) {{
      if (!pts.length) return;
      ctx.beginPath();
      pts.forEach(function(p, i) {{
        var x = p.x, y = pad + plotH * (1 - p.v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Fill under curve
      ctx.lineTo(pts[pts.length-1].x, H);
      ctx.lineTo(pts[0].x, H);
      ctx.closePath();
      ctx.fillStyle = color + '22';
      ctx.fill();
    }}

    drawCurve(TENSION_PTS,  '#ef4444', 'T');
    drawCurve(DENSITY_PTS,  '#3b82f6', 'D');
    drawCurve(VELOCITY_PTS, '#f59e0b', 'V');

    // Cadence markers
    CADENCE_MARKS.forEach(function(c) {{
      ctx.strokeStyle = c.color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 2]);
      ctx.beginPath(); ctx.moveTo(c.x, 0); ctx.lineTo(c.x, H); ctx.stroke();
      ctx.setLineDash([]);
      if (c.x > 2) {{
        ctx.fillStyle = c.color;
        ctx.font = '7px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(c.type.charAt(0), c.x + 2, 2);
      }}
    }});

    // Legend
    var legend = [{{'c':'#ef4444','l':'Tensión'}},{{'c':'#3b82f6','l':'Densidad'}},{{'c':'#f59e0b','l':'Dinámica'}}];
    var lx = 4;
    legend.forEach(function(l) {{
      ctx.fillStyle = l.c; ctx.fillRect(lx, H-10, 8, 6);
      ctx.fillStyle = '#64748b'; ctx.font = '7px sans-serif';
      ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
      ctx.fillText(l.l, lx + 10, H - 1);
      lx += 55;
    }});
  }}

  // ── Minimap ───────────────────────────────────────────────────────────
  function renderMinimap() {{
    var cv = document.getElementById('cv_minimap'); if (!cv) return;
    var ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    clearCanvas(ctx, cv, '#060c18');

    // Draw chord background as tiny bars
    CHORD_BG.forEach(function(cb) {{
      var x = cb.x / {CANVAS_W} * W;
      var w = Math.max(1, cb.w / {CANVAS_W} * W);
      ctx.fillStyle = cb.color + '55';
      ctx.fillRect(x, 0, w, H);
    }});

    // Draw section labels
    SECTION_SPANS.forEach(function(s) {{
      var x = s.x / {CANVAS_W} * W;
      var w = Math.max(1, s.w / {CANVAS_W} * W);
      ctx.fillStyle = s.color + '40';
      ctx.fillRect(x, 0, w, H);
      ctx.strokeStyle = s.color + '88'; ctx.lineWidth = 0.5;
      ctx.strokeRect(x, 0, w, H);
    }});

    // Draw all notes as tiny dots
    function drawMinimapNotes(rects, yOff, h) {{
      rects.forEach(function(r) {{
        var x = r.x / {CANVAS_W} * W;
        var w = Math.max(0.5, r.w / {CANVAS_W} * W);
        ctx.fillStyle = r.c;
        ctx.globalAlpha = 0.7;
        ctx.fillRect(x, yOff + (1 - (r.y / {ROW_H})) * h, w, 1.5);
      }});
    }}
    drawMinimapNotes(RH_RECTS, 0, H/2 - 2);
    drawMinimapNotes(LH_RECTS, H/2 + 2, H/2 - 2);
    ctx.globalAlpha = 1.0;

    // Draw repeated bars in minimap
    REPEATED_BAR_SPANS.forEach(function(r) {{
      var x = r.x / {CANVAS_W} * W;
      var w = Math.max(1, r.w / {CANVAS_W} * W);
      ctx.fillStyle = r.color + '55';
      ctx.fillRect(x, H - 5, w, 5);
    }});
    var scrollWrap = document.querySelector('.scroll-wrap');
    if (scrollWrap) {{
      var totalW = {CANVAS_W} * zoom;
      var viewL  = scrollWrap.scrollLeft;
      var viewW  = scrollWrap.clientWidth - {LABEL_W} - {PIANO_W};
      var vx = viewL / totalW * W;
      var vw = viewW / totalW * W;
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 1;
      ctx.strokeRect(vx, 0, vw, H);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fillRect(vx, 0, vw, H);
    }}
  }}

  function renderAll() {{
    renderSections();
    renderSrc();
    renderRight();
    renderLeft();
    renderChords();
    renderCurves();
    renderMinimap();
  }}



  // ── Tooltip hover (no visual effect on notes) ────────────────────────
  var tooltip = document.getElementById('chord-tooltip');

  function findChordAtCanvasX(canvasX) {{
    var found = null;
    CHORD_ITEMS.forEach(function(c) {{
      if (canvasX >= c.x && canvasX < c.x + c.w) found = c;
    }});
    return found;
  }}

  function canvasXFromEvent(e, cv) {{
    var rect  = cv.getBoundingClientRect();
    var scaleX = cv.width / rect.width;
    return (e.clientX - rect.left) * scaleX;
  }}

  ['cv_right', 'cv_left', 'cv_chords'].forEach(function(id) {{
    var cv = document.getElementById(id); if (!cv) return;
    cv.addEventListener('mousemove', function(e) {{
      var chord = findChordAtCanvasX(canvasXFromEvent(e, cv));
      if (chord && tooltip) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 16) + 'px';
        tooltip.style.top  = (e.clientY - 12) + 'px';
        var html = '<span style="font-size:14px;font-weight:700;color:#f8fafc">' + chord.label + '</span>';
        if (chord.inv) html += ' <span style="font-size:10px;opacity:0.6">(' + chord.inv + ')</span>';
        if (chord.func && chord.func !== '\u2014')
          html += '<br><span style="color:' + chord.color + ';font-size:11px">' + chord.func + '</span>';
        if (chord.key) html += '<br><span style="font-size:9px;opacity:0.5">' + chord.key + '</span>';
        tooltip.innerHTML = html;
      }} else if (tooltip) {{
        tooltip.style.display = 'none';
      }}
    }});
    cv.addEventListener('mouseleave', function() {{
      if (tooltip) tooltip.style.display = 'none';
    }});
  }});

  // ── Double-click: reset zoom ───────────────────────────────────────────
  scrollWrap.addEventListener('dblclick', function() {{
    zoom = 1.0; applyZoom(); renderAll();
    scrollWrap.scrollLeft = 0;
  }});

  // ── Init ──────────────────────────────────────────────────────────────
  applyZoom();
  renderAll();

}}}})();
"""


    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Piano Duo — {title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #080f1c; color: #e2e8f0; font-size: 13px; user-select: none; }}
.header {{ padding: 16px 24px 10px; border-bottom: 1px solid #1a2744; }}
.header h1 {{ font-size: 16px; font-weight: 700; color: #f8fafc; }}
.header .meta {{ font-size: 11px; color: #4a6080; margin-top: 4px; }}
.header .hint {{ font-size: 10px; color: #2a4060; margin-top: 3px; }}
.scroll-wrap {{ overflow-x: auto; overflow-y: visible; position: relative; }}
.tl-outer {{ display: flex; flex-direction: column; min-width: max-content; }}
.ruler-row {{ display: flex; height: 22px; border-bottom: 1px solid #1a2744; }}
.ruler-label {{ width: {LABEL_W}px; min-width: {LABEL_W}px; flex-shrink: 0;
                background: #070e1a; border-right: 2px solid #1a2744;
                position: sticky; left: 0; z-index: 10; }}
.ruler-track {{ position: relative; background: #070e1a; min-width: {CANVAS_W}px; }}
.bar-tick {{ position: absolute; top: 0; height: 100%;
             border-left: 1px solid rgba(255,255,255,0.08);
             font-size: 9px; color: #334466; padding-left: 3px;
             line-height: 22px; pointer-events: none; }}
.tl-row {{ display: flex; border-bottom: 1px solid #1a2744; }}
.tl-label {{ width: {LABEL_W}px; min-width: {LABEL_W}px; padding: 4px 6px 4px 12px;
             display: flex; flex-direction: column; justify-content: center;
             border-right: 2px solid #1a2744; background: #070e1a; flex-shrink: 0;
             position: sticky; left: 0; z-index: 10; }}
.piano-wrap {{ width: {PIANO_W}px; min-width: {PIANO_W}px; flex-shrink: 0;
               border-right: 1px solid #1a2744;
               position: sticky; left: {LABEL_W}px; z-index: 9; background: #0a1220; }}
.tl-canvas-wrap {{ flex: 1; overflow: visible; cursor: default; min-width: {CANVAS_W}px; }}
canvas.zoomable-canvas {{ display: block; }}
.trk-name {{ font-size: 11px; font-weight: 700; color: #dde6f0; }}
.trk-sub  {{ font-size: 9px; margin-top: 2px; opacity: 0.7; line-height: 1.4; }}
.legend-row {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 10px 16px;
               border-top: 1px solid #1a2744; background: #050c18; align-items: center; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 10px; color: #64748b; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
#chord-tooltip {{
  position: fixed; pointer-events: none; display: none; z-index: 200;
  background: #0d1b2e; border: 1px solid #2a4060; border-radius: 7px;
  padding: 7px 12px; line-height: 1.6; white-space: nowrap;
  box-shadow: 0 4px 20px rgba(0,0,0,0.6); font-family: monospace;
}}
</style>
</head>
<body>
<div class="header">
  <h1>🎹 {title}</h1>
  <div class="meta">{n_bars} compases · {bpb:.0f}/4 · Tonalidad: {key_summary}
    · RH: {right_track['name']} · LH: {left_track['name']}</div>
  <div class="hint">🖱 Arrastrar: desplazar horizontalmente · Doble clic: reset</div>
</div>

<div id="chord-tooltip"></div>

<div class="scroll-wrap">
<div class="tl-outer">

  <div class="ruler-row">
    <div class="ruler-label"></div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;flex-shrink:0;background:#070e1a"></div>
    <div class="ruler-track">{bar_ticks_html}</div>
  </div>

  <!-- SECTIONS ROW -->
  <div class="tl-row" style="height:24px">
    <div class="tl-label" style="border-left:3px solid #8b5cf6">
      <div class="trk-name" style="font-size:9px;color:#8b5cf6">Secciones</div>
    </div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;background:#060c18;border-right:1px solid #1a2744;flex-shrink:0;position:sticky;left:{LABEL_W}px;z-index:9"></div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_sections" width="{CANVAS_W}" height="24" class="zoomable-canvas"></canvas>
    </div>
  </div>

  <!-- SOURCE ROW -->
  <div class="tl-row" style="height:18px">
    <div class="tl-label" style="border-left:3px solid #475569">
      <div class="trk-name" style="font-size:8px;color:#475569">Fuente armonía</div>
    </div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;background:#060c1a;border-right:1px solid #1a2744;flex-shrink:0"></div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_src" width="{CANVAS_W}" height="18" class="zoomable-canvas"></canvas>
    </div>
  </div>

  <!-- RIGHT HAND -->
  <div class="tl-row" style="height:{ROW_H}px">
    <div class="tl-label" style="border-left:3px solid #3b82f6">
      <div class="trk-name">🎹 Mano Derecha</div>
      <div class="trk-sub" style="color:#3b82f6">{right_track['name']}</div>
    </div>
    <div class="piano-wrap">{rh_piano_svg}</div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_right" width="{CANVAS_W}" height="{ROW_H}" class="zoomable-canvas"></canvas>
    </div>
  </div>

  <!-- LEFT HAND -->
  <div class="tl-row" style="height:{ROW_H}px">
    <div class="tl-label" style="border-left:3px solid #22c55e">
      <div class="trk-name">🎹 Mano Izquierda</div>
      <div class="trk-sub" style="color:#22c55e">{left_track['name']}</div>
    </div>
    <div class="piano-wrap">{lh_piano_svg}</div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_left" width="{CANVAS_W}" height="{ROW_H}" class="zoomable-canvas"></canvas>
    </div>
  </div>

  <!-- CHORDS -->
  <div class="tl-row" style="height:{CHORD_H}px">
    <div class="tl-label" style="border-left:3px solid #f59e0b">
      <div class="trk-name">Acordes</div>
      <div class="trk-sub" style="color:#f59e0b">función armónica</div>
    </div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;background:#080f1e;border-right:1px solid #1a2744;flex-shrink:0"></div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_chords" width="{CANVAS_W}" height="{CHORD_H}" class="zoomable-canvas"></canvas>
    </div>
  </div>


  <!-- CURVES ROW: tension / density / velocity / cadences -->
  <div class="tl-row" style="height:52px">
    <div class="tl-label" style="border-left:3px solid #ef4444">
      <div class="trk-name" style="font-size:10px">Tensión · Densidad</div>
      <div class="trk-sub" style="color:#ef4444">dinámica y cadencias</div>
    </div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;background:#06091a;border-right:1px solid #1a2744;flex-shrink:0;position:sticky;left:{LABEL_W}px;z-index:9"></div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_curves" width="{CANVAS_W}" height="52" class="zoomable-canvas"></canvas>
    </div>
  </div>

  <!-- MINIMAP ROW -->
  <div class="tl-row" style="height:36px">
    <div class="tl-label" style="border-left:3px solid #475569">
      <div class="trk-name" style="font-size:9px;color:#475569">Vista general</div>
    </div>
    <div style="width:{PIANO_W}px;min-width:{PIANO_W}px;background:#060c18;border-right:1px solid #1a2744;flex-shrink:0;position:sticky;left:{LABEL_W}px;z-index:9"></div>
    <div class="tl-canvas-wrap" style="min-width:0;flex:1;overflow:hidden">
      <canvas id="cv_minimap" width="800" height="36" style="display:block;width:100%;height:36px"></canvas>
    </div>
  </div>

</div>
</div>

<div class="legend-row">{legend_html}</div>

<script>
{js}
</script>
</body>
</html>"""

    return html


def _harmonic_density(notes: List[Dict], bar: int, bpb: float) -> float:
    """
    Score for how 'harmonically rich' a track is in a given bar.
    Combines: number of distinct pitch classes, average simultaneous notes,
    and total note count.  Higher = more likely to be the harmony source.
    """
    bn = [n for n in notes if n['start_bar'] == bar]
    if not bn:
        return 0.0
    pcs     = len({n['pitch'] % 12 for n in bn})
    n_notes = len(bn)
    onsets  = sorted(set(round(n['start_beat'], 3) for n in bn))
    poly_vals = []
    for o in onsets:
        sim = sum(1 for n in bn if n['start_beat'] <= o < n['end_beat'])
        poly_vals.append(sim)
    avg_poly = sum(poly_vals) / len(poly_vals) if poly_vals else 1.0
    return pcs * 1.0 + avg_poly * 2.0 + n_notes * 0.3

def _parse_harmony_source(spec: Optional[str], n_bars: int) -> Optional[List[str]]:
    """
    Parse --harmony-source spec into a per-bar list of 'right'/'left'/'auto'.
    Spec format (comma-separated ranges):
      'left'                      → all bars = left
      'right'                     → all bars = right
      '1-4:right,5-12:left'       → bars 1-4 from right, 5-12 from left, rest auto
      '1-4:right'                 → bars 1-4 from right, rest auto
    Bar numbers are 1-based.
    """
    result = ['auto'] * n_bars
    if spec is None:
        return result
    spec = spec.strip().lower()
    if spec in ('left', 'right', 'auto'):
        return [spec] * n_bars
    for part in spec.split(','):
        part = part.strip()
        if ':' not in part:
            continue
        rng, src = part.rsplit(':', 1)
        src = src.strip()
        rng = rng.strip()
        if '-' in rng:
            lo, hi = rng.split('-', 1)
            lo, hi = int(lo.strip()) - 1, int(hi.strip()) - 1
        else:
            lo = hi = int(rng) - 1
        for b in range(max(0, lo), min(n_bars, hi + 1)):
            result[b] = src
    return result

def harmony_source_per_bar(right_notes: List[Dict], left_notes: List[Dict],
                            n_bars: int, bpb: float,
                            spec: Optional[str] = None) -> List[str]:
    """
    Return per-bar harmony source: 'right' or 'left'.

    Key improvement: bars where one or both tracks have no notes are marked
    as 'inherit' and filled by propagation from the nearest non-empty bar.
    This handles pieces where the harmony track only plays on some bars
    (e.g. sustained chords with rests in between).
    """
    directives = _parse_harmony_source(spec, n_bars)

    # First pass: compute raw decision only for bars with actual notes
    raw = []
    for b in range(n_bars):
        if directives[b] in ('right', 'left'):
            raw.append(directives[b])
        else:
            dr = _harmonic_density(right_notes, b, bpb)
            dl = _harmonic_density(left_notes,  b, bpb)
            if dr == 0 and dl == 0:
                raw.append('inherit')          # empty bar — propagate later
            elif dl == 0:
                raw.append('right')            # only RH has notes
            elif dr == 0:
                raw.append('left')             # only LH has notes
            else:
                raw.append('right' if dr >= dl else 'left')

    # Second pass: fill 'inherit' bars by propagating from neighbours
    # Forward pass
    last = 'left'
    for b in range(n_bars):
        if raw[b] != 'inherit':
            last = raw[b]
        else:
            raw[b] = last
    # Backward pass (fills leading inherit bars)
    last = raw[-1] if raw else 'left'
    for b in range(n_bars - 1, -1, -1):
        if raw[b] == 'inherit':
            raw[b] = last
        else:
            last = raw[b]

    # Third pass: smoothing — require ≥2 consecutive bars to switch source
    smoothed = list(raw)
    for b in range(1, n_bars - 1):
        if directives[b] in ('right', 'left'):
            continue
        if smoothed[b] != smoothed[b - 1]:
            if b + 1 < n_bars and smoothed[b + 1] == smoothed[b]:
                pass   # genuine switch confirmed by next bar
            else:
                smoothed[b] = smoothed[b - 1]

    return smoothed


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS AVANZADO — motivos, cadencias, tensión, densidad, velocity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tension_curve(chords_per_bar: List[Dict], n_bars: int) -> List[float]:
    """Harmonic tension per bar: T=0.0, PD=0.4, Dsec=0.7, D=1.0, Other=0.5"""
    tension_map = {'T': 0.0, 'PD': 0.4, 'Dsec': 0.7, 'D': 1.0, 'Other': 0.5, '—': 0.0}
    return [tension_map.get(chords_per_bar[b]['function'] if b < len(chords_per_bar) else '—', 0.3)
            for b in range(n_bars)]

def compute_density_per_bar(all_notes: List[Dict], n_bars: int, bpb: float) -> List[float]:
    """Notes per beat per bar, normalised to [0,1]."""
    counts = [0.0] * n_bars
    for n in all_notes:
        b = n['start_bar']
        if 0 <= b < n_bars:
            counts[b] += 1
    max_c = max(counts) or 1.0
    return [c / max_c for c in counts]

def compute_avg_velocity_per_bar(all_notes: List[Dict], n_bars: int) -> List[float]:
    """Average velocity per bar, normalised to [0,1]."""
    sums   = [0.0] * n_bars
    counts = [0]   * n_bars
    for n in all_notes:
        b = n['start_bar']
        if 0 <= b < n_bars:
            sums[b]   += n['vel']
            counts[b] += 1
    return [(sums[b] / counts[b] / 127.0) if counts[b] else 0.0 for b in range(n_bars)]

def detect_motifs(melody_notes: List[Dict], min_len: int = 3,
                  max_len: int = 8, min_reps: int = 2) -> List[Dict]:
    """
    Detect repeated melodic phrases using Re-Pair grammar compression.

    Unlike the sliding-window approach, Re-Pair builds a non-overlapping
    hierarchical grammar over the full interval sequence. Each rule that
    appears ≥ min_reps times in the root sequence becomes a motif.

    Steps:
      1. Collapse simultaneous notes into chord events (lowest pitch).
      2. Compute quantised intervals between consecutive events.
      3. Run Re-Pair; collect rules that repeat ≥ min_reps times in root.
      4. Filter dominated sub-rules (prefer longer phrases).
      5. Map each rule occurrence back to beat positions.

    Returns list of {label, color, occurrences, intervals} dicts,
    guaranteed non-overlapping by construction.
    """
    if not melody_notes:
        return []

    # ── 1. Collapse simultaneous notes → take HIGHEST pitch (melody voice)
    # The melody track may have two simultaneous notes in some bars (e.g.
    # Canon c.9-12 has two-voice chords). We always take the highest note
    # as the melodic representative, ignoring inner voices.
    notes_sorted = sorted(melody_notes, key=lambda n: (n['start_beat'], -n['pitch']))
    events: List[Dict] = []
    i = 0
    while i < len(notes_sorted):
        group = [notes_sorted[i]]
        j = i + 1
        while j < len(notes_sorted) and abs(notes_sorted[j]['start_beat'] - notes_sorted[i]['start_beat']) < 0.05:
            group.append(notes_sorted[j]); j += 1
        # Take the highest pitch as the melody representative
        rep = max(group, key=lambda n: n['pitch'])
        events.append({
            'start_beat': rep['start_beat'],
            'end_beat':   rep['end_beat'],
            'pitch':      rep['pitch'],
            'start_bar':  rep['start_bar'],
            'all_pitches': [rep['pitch']],   # only the melody note
        })
        i = j

    if len(events) < min_len * min_reps:
        return []

    # ── 2. Interval sequence ──────────────────────────────────────────────
    # Quantise to ±12 semitones; offset +12 so all values ≥ 0.
    # IMPORTANT: skip intervals where there is a large temporal gap between
    # consecutive events (> 2 beats) — these are cross-hand or cross-phrase
    # leaps that are not melodic intervals and would corrupt the grammar.
    MAX_GAP = 2.0   # beats; larger = cross-hand leap, not melodic interval
    symbols: List[int] = []
    beat_at: List[float] = []
    end_at:  List[float] = []
    for k in range(len(events)-1):
        gap = events[k+1]['start_beat'] - events[k]['end_beat']
        if gap > MAX_GAP:
            continue   # skip cross-hand / cross-phrase jumps
        iv = events[k+1]['pitch'] - events[k]['pitch']
        symbols.append(max(-12, min(12, iv)) + 12)
        beat_at.append(events[k]['start_beat'])
        end_at.append(events[k+1]['end_beat'])

    if len(symbols) < min_len:
        return []

    # ── 3. Re-Pair grammar ────────────────────────────────────────────────
    rules = repair_grammar(symbols)
    root  = rules.get('root', [])

    # Expand a rule fully to a list of terminal interval symbols
    _expand_cache: Dict[str, List[int]] = {}
    def expand(sym):
        if sym in _expand_cache: return _expand_cache[sym]
        if sym not in rules or sym == 'root':
            result = [sym] if isinstance(sym, int) else []
        else:
            result = []
            for s in rules[sym]: result.extend(expand(s))
        _expand_cache[sym] = result
        return result

    # Count how many times each rule symbol appears in root
    root_counts: Dict = defaultdict(int)
    for sym in root:
        root_counts[sym] += 1

    # ── 4. Collect candidate rules ────────────────────────────────────────
    candidates = []
    for sym, count in root_counts.items():
        if not isinstance(sym, str): continue          # skip terminal ints
        if sym == 'root': continue
        if count < min_reps: continue
        expanded = expand(sym)
        if len(expanded) < min_len: continue
        if len(expanded) > max_len * 3: continue       # skip huge rules
        intervals = [s - 12 for s in expanded if isinstance(s, int)]
        candidates.append((sym, count, intervals))

    if not candidates:
        return []

    # ── 5. Filter dominated sub-rules ─────────────────────────────────────
    # A rule is dominated if its expansion is a contiguous sub-sequence of
    # a longer rule that also appears in root
    def is_subseq(short, long):
        ls, ll = len(short), len(long)
        for start in range(ll - ls + 1):
            if long[start:start+ls] == short:
                return True
        return False

    candidates.sort(key=lambda x: (-len(x[2]), -x[1]))
    kept = []
    for sym, count, ivs in candidates:
        dominated = any(is_subseq(ivs, k_ivs) and ivs != k_ivs
                        for _, _, k_ivs in kept)
        if not dominated:
            kept.append((sym, count, ivs))

    # ── 6. Map occurrences back to beat positions ─────────────────────────
    # Walk the root sequence and record where each kept rule appears
    motif_colors = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff922b',
                    '#cc5de8','#20c997','#f06595','#74c0fc','#a9e34b']

    # Build set of kept rule names for fast lookup
    kept_syms = {sym for sym, _, _ in kept}
    sym_to_ivs = {sym: ivs for sym, _, ivs in kept}

    motifs: List[Dict] = []
    for label_idx, (sym, count, ivs) in enumerate(kept):
        occurrences = []
        pos = 0
        for root_sym in root:
            sym_len = len(expand(root_sym)) if isinstance(root_sym, str) else 1
            if root_sym == sym:
                # This occurrence spans [pos, pos+sym_len) in the symbol array
                if pos < len(beat_at) and pos + sym_len - 1 < len(end_at):
                    start_b = beat_at[pos]
                    end_b   = end_at[pos + sym_len - 1]
                    # Collect all original pitches in this span
                    pitches = []
                    for ev in events:
                        if start_b - 0.05 <= ev['start_beat'] <= end_b + 0.05:
                            pitches.extend(ev['all_pitches'])
                    occurrences.append({
                        'start_beat': start_b,
                        'end_beat':   end_b,
                        'pitches':    pitches,
                    })
            pos += sym_len

        if len(occurrences) >= min_reps:
            motifs.append({
                'label':       chr(65 + label_idx),
                'color':       motif_colors[label_idx % len(motif_colors)],
                'occurrences': occurrences,
                'intervals':   ivs,
            })

    # Re-label sequentially
    for i, m in enumerate(motifs):
        m['label'] = chr(65 + i)

    return motifs


def detect_cadences(chord_timeline: List[Dict], tonic: int) -> List[Dict]:
    """
    Detect cadence points in the chord timeline.
    Returns list of {beat, type, color} where type is one of:
    'Perfecta' (V→I), 'Plagal' (IV→I), 'Rota' (V→vi), 'Semicadencia' (→V)
    """
    cadences = []
    for i in range(1, len(chord_timeline)):
        prev = chord_timeline[i-1]
        curr = chord_timeline[i]
        pr, cr = prev['root_pc'], curr['root_pc']
        pt, ct = prev['ctype'],  curr['ctype']
        beat = curr['start_beat']

        is_dom  = lambda pc, ct: (pc - tonic) % 12 == 7
        is_ton  = lambda pc, ct: (pc - tonic) % 12 == 0
        is_sub  = lambda pc, ct: (pc - tonic) % 12 == 5
        is_vi   = lambda pc, ct: (pc - tonic) % 12 == 9

        if is_dom(pr, pt) and is_ton(cr, ct):
            cadences.append({'beat': beat, 'type': 'Perfecta', 'color': '#22c55e'})
        elif is_sub(pr, pt) and is_ton(cr, ct):
            cadences.append({'beat': beat, 'type': 'Plagal', 'color': '#3b82f6'})
        elif is_dom(pr, pt) and is_vi(cr, ct):
            cadences.append({'beat': beat, 'type': 'Rota', 'color': '#f59e0b'})
        elif is_ton(pr, pt) and is_dom(cr, ct):
            cadences.append({'beat': beat, 'type': 'Semicad.', 'color': '#ef4444'})

    return cadences

# ═══════════════════════════════════════════════════════════════════════════════
# RE-PAIR GRAMMAR  (adaptado de mscz2vec.py / SequiturWithCustomRules)
# Detecta estructura jerárquica de la pieza comprimiendo pares repetidos
# de alturas MIDI en reglas, sin depender de music21.
# ═══════════════════════════════════════════════════════════════════════════════

def repair_grammar(symbols: List[int]) -> Dict:
    """
    Re-Pair algorithm: compresses a sequence of MIDI pitches by repeatedly
    replacing the most frequent adjacent pair with a new rule symbol.

    Returns dict: {rule_name: [symbol, symbol], ...}  plus  'root' key
    with the compressed root sequence.
    """
    if not symbols:
        return {'root': []}

    current = list(symbols)
    rules: Dict[str, list] = {}
    next_id = [1]

    def new_rule(a, b):
        name = f'R{next_id[0]}'; next_id[0] += 1
        rules[name] = [a, b]
        return name

    for _ in range(200):          # cap iterations
        pairs: Dict[tuple, int] = {}
        for i in range(len(current) - 1):
            p = (current[i], current[i+1])
            pairs[p] = pairs.get(p, 0) + 1
        if not pairs:
            break
        best_pair, count = max(pairs.items(), key=lambda x: x[1])
        if count < 2:
            break
        rule = new_rule(*best_pair)
        new_seq, i = [], 0
        while i < len(current):
            if i < len(current)-1 and (current[i], current[i+1]) == best_pair:
                new_seq.append(rule); i += 2
            else:
                new_seq.append(current[i]); i += 1
        current = new_seq

    rules['root'] = current
    return rules


def repair_sections(melody_notes: List[Dict], n_bars: int,
                    bpb: float) -> List[Dict]:
    """
    Use Re-Pair to detect structural sections.

    Steps:
    1. Build a per-bar pitch sequence (median pitch per bar, or 0 for empty).
    2. Run Re-Pair on the sequence.
    3. Find the top-level rules (those that appear in the root sequence ≥2×)
       and map them back to bar ranges.
    4. Return list of {bar, end, label, color, rule} section dicts.

    This produces sections that reflect actual melodic repetition, not density.
    """
    if not melody_notes or n_bars < 2:
        return [{'bar': 0, 'end': n_bars, 'label': 'A', 'color': '#3b82f6', 'rule': None}]

    # ── Build bar-level pitch fingerprint ────────────────────────────────
    # Each bar becomes its median MIDI pitch (normalised relative to bar 0 for
    # transposition-invariance) quantised to semitone.
    bar_pitches: Dict[int, List[int]] = defaultdict(list)
    for n in melody_notes:
        bar_pitches[n['start_bar']].append(n['pitch'])

    # Represent each bar as a single integer (median pitch).
    # Empty bars get -1 (treated as silence).
    bar_symbols: List[int] = []
    for b in range(n_bars):
        pitches = bar_pitches.get(b, [])
        if pitches:
            bar_symbols.append(int(sorted(pitches)[len(pitches)//2]))
        else:
            bar_symbols.append(-1)

    # ── Run Re-Pair ───────────────────────────────────────────────────────
    rules = repair_grammar(bar_symbols)
    root  = rules.get('root', bar_symbols)

    # ── Map rule occurrences back to bars ─────────────────────────────────
    # Expand each rule symbol in root back to a span of bars.
    def rule_len(sym, rules, memo={}):
        if sym in memo: return memo[sym]
        if sym not in rules or sym == 'root':
            memo[sym] = 1; return 1
        l = sum(rule_len(s, rules, memo) for s in rules[sym])
        memo[sym] = l; return l

    # Build (bar_start, bar_end, symbol) for every element in root
    spans = []
    bar = 0
    for sym in root:
        length = rule_len(sym, rules)
        spans.append((bar, bar + length, sym))
        bar += length

    # Count which rule symbols appear ≥2× in root
    sym_counts: Dict = defaultdict(int)
    for _, _, sym in spans:
        sym_counts[sym] += 1

    # Repeated symbols (rules that appear multiple times) → structural sections
    # Unique/single symbols → merge with neighbours
    MIN_SECTION = max(4, n_bars // 8)

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    colors = ['#3b82f6','#22c55e','#f59e0b','#ef4444','#8b5cf6',
              '#06b6d4','#ec4899','#84cc16','#f97316','#a855f7']

    # Assign a stable label to each unique rule symbol (by first appearance)
    sym_label: Dict = {}
    label_idx = 0
    sections = []

    for start, end, sym in spans:
        if sym not in sym_label:
            sym_label[sym] = label_idx % len(labels)
            label_idx += 1
        li   = sym_label[sym]
        sections.append({
            'bar':   start,
            'end':   min(end, n_bars),
            'label': labels[li],
            'color': colors[li % len(colors)],
            'rule':  sym,
        })

    # Merge sections that are too short into their neighbours
    merged = []
    for s in sections:
        if merged and (s['end'] - s['bar']) < MIN_SECTION:
            # extend previous section
            merged[-1]['end'] = s['end']
        elif merged and merged[-1]['label'] == s['label']:
            merged[-1]['end'] = s['end']
        else:
            merged.append(dict(s))

    # Final merge: collapse consecutive same-label sections
    final = []
    for s in merged:
        if final and final[-1]['label'] == s['label']:
            final[-1]['end'] = s['end']
        else:
            final.append(s)

    return final if final else [{'bar': 0, 'end': n_bars,
                                  'label': 'A', 'color': '#3b82f6', 'rule': None}]


# ═══════════════════════════════════════════════════════════════════════════════
# REPEATED MEASURES  (adaptado de mscz2vec.py / identify_repeated_measures)
# Detecta compases cuyo patrón melódico relativo se repite en otro lugar
# ═══════════════════════════════════════════════════════════════════════════════

def detect_repeated_bars(melody_notes: List[Dict], n_bars: int,
                          bpb: float) -> Dict[int, Dict]:
    """
    Find bars whose melodic contour (relative pitch sequence) repeats
    elsewhere in the piece.

    Normalisation: intervals between consecutive notes within the bar
    (transposition-invariant, like mscz2vec's normalize_pitches_by_first_note).

    Returns dict: bar_index → {group, color, bars_in_group}
    Only bars that share a pattern with at least one other bar are included.
    """
    from collections import defaultdict

    group_colors = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff922b',
                    '#cc5de8','#20c997','#f06595','#74c0fc','#a9e34b',
                    '#fb923c','#34d399','#60a5fa','#f472b6','#a3e635']

    # Build per-bar note sequence sorted by onset
    bar_notes: Dict[int, List[Dict]] = defaultdict(list)
    for n in melody_notes:
        bar_notes[n['start_bar']].append(n)

    # Fingerprint each bar: tuple of quantised intervals (semitones, capped ±12)
    # Require at least 4 notes (3 intervals) for a meaningful fingerprint —
    # fewer notes produce trivial single-interval patterns that match too broadly.
    MIN_NOTES = 4
    fingerprints: Dict[int, tuple] = {}
    for b in range(n_bars):
        notes = sorted(bar_notes.get(b, []), key=lambda n: n['start_beat'])
        if len(notes) < MIN_NOTES:
            fingerprints[b] = ()
            continue
        ivs = tuple(max(-12, min(12, notes[i+1]['pitch'] - notes[i]['pitch']))
                    for i in range(len(notes)-1))
        fingerprints[b] = ivs

    # Group bars by fingerprint
    fp_groups: Dict[tuple, List[int]] = defaultdict(list)
    for b, fp in fingerprints.items():
        if fp:   # skip empty bars
            fp_groups[fp].append(b)

    # Keep only groups with ≥2 members (actual repetitions)
    result: Dict[int, Dict] = {}
    group_idx = 0
    for fp, bars in sorted(fp_groups.items(), key=lambda x: -len(x[1])):
        if len(bars) < 2:
            continue
        color = group_colors[group_idx % len(group_colors)]
        for b in bars:
            result[b] = {
                'group': group_idx,
                'color': color,
                'bars_in_group': bars,
                'fingerprint': fp,
            }
        group_idx += 1
        if group_idx >= len(group_colors):
            break   # limit to palette size

    return result


def detect_sections(chords_per_bar: List[Dict], density: List[float],
                    n_bars: int, bpb: float,
                    motifs: List[Dict] = None,
                    repair: List[Dict] = None) -> List[Dict]:
    """
    Section detection. Priority:
    1. Re-Pair grammar sections — melodic structure from note repetition
    2. Motif recurrence boundaries — fallback
    3. Equal division — last resort
    """
    if n_bars < 4:
        return [{'bar': 0, 'end': n_bars, 'label': 'A', 'color': '#3b82f6'}]

    MIN_SECTION = max(4, n_bars // 8)
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    colors = ['#3b82f6','#22c55e','#f59e0b','#ef4444','#8b5cf6',
              '#06b6d4','#ec4899','#84cc16']

    if repair and len(repair) >= 2:
        return repair

    if motifs:
        boundaries = sorted(set(
            int(occ['start_beat'] / bpb)
            for m in motifs for occ in m['occurrences']
        ) | {0})
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] >= MIN_SECTION:
                merged.append(b)
        if len(merged) >= 2:
            return [{'bar': merged[i],
                     'end': merged[i+1] if i+1 < len(merged) else n_bars,
                     'label': labels[i % len(labels)],
                     'color': colors[i % len(colors)]}
                    for i in range(len(merged))]

    n_divs = max(2, min(4, n_bars // max(MIN_SECTION, 1)))
    step = n_bars // n_divs
    return [{'bar': i*step,
             'end': (i+1)*step if i < n_divs-1 else n_bars,
             'label': labels[i % len(labels)],
             'color': colors[i % len(colors)]}
            for i in range(n_divs)]

def analyze(mid_path: str, right_idx: int, left_idx: int,
            arp_window: float = 1.0, key_window_bars: int = 8,
            harmony_source_spec: Optional[str] = None) -> str:
    print(f"  Cargando MIDI: {mid_path}")
    mid = mido.MidiFile(mid_path)
    print(f"  Pistas en el archivo: {len(mid.tracks)}")
    for i, t in enumerate(mid.tracks):
        print(f"    [{i}] '{t.name}'  ({sum(1 for m in t if m.type == 'note_on')} note_on)")

    right_track = extract_track(mid, right_idx)
    left_track  = extract_track(mid, left_idx)

    print(f"  Mano derecha [{right_idx}]: {right_track['name']} — {len(right_track['notes'])} notas")
    print(f"  Mano izquierda [{left_idx}]: {left_track['name']} — {len(left_track['notes'])} notas")

    if not right_track['notes'] and not left_track['notes']:
        return "<html><body>No se encontraron notas en las pistas seleccionadas.</body></html>"

    bpb       = right_track['bpb']
    all_notes = right_track['notes'] + left_track['notes']
    if not all_notes:
        return "<html><body>Sin notas.</body></html>"

    max_bar = max(n['start_bar'] for n in all_notes)
    n_bars  = max_bar + 2

    # ── Detección de tonalidad ─────────────────────────────────────────────
    global_tonic, global_mode = detect_key(all_notes)
    print(f"  Tonalidad global: {ROOT_NOTE_NAMES[global_tonic]} {global_mode}")

    keys_per_bar = detect_key_per_bar(all_notes, n_bars, bpb, key_window_bars)
    changes = []
    prev = None
    for b, k in enumerate(keys_per_bar):
        lbl = f"{ROOT_NOTE_NAMES[k[0]]} {k[1]}"
        if lbl != prev:
            changes.append((b + 1, lbl))
            prev = lbl
    if len(changes) > 1:
        print(f"  Modulaciones detectadas ({len(changes)} regiones):")
        for bar, lbl in changes:
            print(f"    compás {bar}: {lbl}")
    else:
        print(f"  Sin modulaciones detectadas (ventana={key_window_bars} compases)")

    # ── Fuente armónica dinámica por compás ───────────────────────────────
    src_per_bar = harmony_source_per_bar(
        right_track['notes'], left_track['notes'],
        n_bars, bpb, harmony_source_spec)

    # Report switches
    src_changes = []
    prev_src = None
    for b, s in enumerate(src_per_bar):
        if s != prev_src:
            src_changes.append((b + 1, s))
            prev_src = s
    print(f"  Fuente armónica por sección:")
    for bar, src in src_changes:
        label = 'Mano Derecha' if src == 'right' else 'Mano Izquierda'
        print(f"    compás {bar}: {label}")

    # Build chord timeline from harmony-source notes only.
    # The other track's notes are passed as bass_notes — they enrich chord
    # detection and drive sub-bar splits, but never create groups on their own.
    rh_harmony_bars = {b for b, s in enumerate(src_per_bar) if s == 'right'}
    lh_harmony_bars = {b for b, s in enumerate(src_per_bar) if s == 'left'}
    harmony_notes = (
        [n for n in right_track['notes'] if n['start_bar'] in rh_harmony_bars] +
        [n for n in left_track['notes']  if n['start_bar'] in lh_harmony_bars]
    )
    other_notes = (
        [n for n in left_track['notes']  if n['start_bar'] in rh_harmony_bars] +
        [n for n in right_track['notes'] if n['start_bar'] in lh_harmony_bars]
    )
    print(f"  Construyendo timeline de acordes (bar-aligned, arp_window={arp_window})...")
    chord_timeline = build_chord_timeline(harmony_notes, arp_window=arp_window, bpb=bpb,
                                          bass_notes=other_notes)
    print(f"  {len(chord_timeline)} eventos de acorde detectados")

    # Acordes por compas (fila 3)
    chords_per_bar = compute_chords_per_bar(all_notes, n_bars, bpb, keys_per_bar)

    # ── Advanced analysis ─────────────────────────────────────────────────
    print("  Analizando tensión, densidad, motivos, cadencias...")
    tension      = compute_tension_curve(chords_per_bar, n_bars)
    density      = compute_density_per_bar(all_notes, n_bars, bpb)
    avg_velocity = compute_avg_velocity_per_bar(all_notes, n_bars)

    # Motifs from all melody notes across the whole piece
    melody_notes = (
        [n for n in right_track['notes'] if src_per_bar[n['start_bar']] == 'left']  +
        [n for n in left_track['notes']  if src_per_bar[n['start_bar']] == 'right']
    )
    motifs    = detect_motifs(melody_notes, min_len=3, max_len=8)
    cadences  = detect_cadences(chord_timeline, global_tonic)

    # ── Re-Pair structural analysis ───────────────────────────────────────
    print("  Analizando estructura con Re-Pair...")
    repair   = repair_sections(melody_notes, n_bars, bpb)
    sections = detect_sections(chords_per_bar, density, n_bars, bpb,
                                motifs=motifs, repair=repair)

    # ── Repeated bars ─────────────────────────────────────────────────────
    repeated_bars = detect_repeated_bars(melody_notes, n_bars, bpb)

    print(f"  {len(motifs)} motivos, {len(cadences)} cadencias, "
          f"{len(sections)} secciones, {len(repeated_bars)} compases repetidos")

    # Per-note velocity for color modulation
    for n in right_track['notes'] + left_track['notes']:
        n['vel_norm'] = round(n['vel'] / 127.0, 3)

    print("  Generando HTML...")
    html = build_html(mid_path, right_track, left_track,
                      chord_timeline, chords_per_bar,
                      global_tonic, global_mode, keys_per_bar,
                      src_per_bar, n_bars,
                      tension=tension, density=density,
                      avg_velocity=avg_velocity, motifs=motifs,
                      cadences=cadences, sections=sections,
                      repeated_bars=repeated_bars)
    return html

# ═══════════════════════════════════════════════════════════════════════════════
# MIDI DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_midi(path: str):
    """
    MIDI de prueba con:
      · Sección A (c.1-8):   Do mayor, mano izquierda con arpegios
      · Sección B (c.9-16):  La menor, acordes bloque + arpegios mixtos
      · Sección A'(c.17-24): Do mayor, vuelta con inversiones

    Pista 0: Mano derecha (melodía)
    Pista 1: Mano izquierda (arpegios / acordes)
    """
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    TPB = 480

    def beats(b): return int(b * TPB)

    def make_track(name, channel, note_list, add_meta=False, bpm=96):
        t = mido.MidiTrack()
        t.name = name
        t.append(mido.MetaMessage('track_name', name=name, time=0))
        if add_meta:
            t.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
            t.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                       clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        events = []
        for pitch, vel, sb, db in note_list:
            st = beats(sb); et = beats(sb + db)
            events.append((st, 'on',  pitch, vel, channel))
            events.append((et, 'off', pitch, 0,   channel))
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        prev_tick = 0
        for ev in events:
            dt = ev[0] - prev_tick; prev_tick = ev[0]
            if ev[1] == 'on':
                t.append(mido.Message('note_on',  channel=ev[4], note=ev[2], velocity=ev[3], time=dt))
            else:
                t.append(mido.Message('note_off', channel=ev[4], note=ev[2], velocity=0, time=dt))
        return t

    left = []
    right = []

    # ── SECCIÓN A: Do mayor, arpegios ascendentes (8 compases) ───────────
    # Progresión: C - F - G - C  ×2
    # Arpegios en corcheas: c,e,g,c' por tiempo
    arp_A = {
        'C': [48, 52, 55, 60],   # C E G C
        'F': [53, 57, 60, 65],   # F A C F
        'G': [55, 59, 62, 67],   # G B D G
        'Am': [45, 48, 52, 57],  # A C E A
    }
    prog_A = ['C','F','G','C',  'C','Am','F','G']
    mel_A  = {
        'C':  [72, 71, 69, 67],
        'F':  [65, 67, 69, 67],
        'G':  [67, 69, 71, 72],
        'Am': [69, 67, 65, 64],
    }
    for ci, chord_name in enumerate(prog_A):
        base = ci * 4
        arp  = arp_A[chord_name]
        # Arpegio en 4 tiempos (una nota por tiempo, duración 0.9)
        for qi, p in enumerate(arp):
            left.append((p, 68, base + qi, 0.9))
        # Mano derecha: melodía en negras
        for qi, p in enumerate(mel_A[chord_name]):
            right.append((p, 78, base + qi, 0.88))
        # Nota cromática de adorno fuera del acorde en algunos compases
        if ci % 3 == 0:
            right.append((73, 55, base + 0.5, 0.3))  # Db — fuera de C maj

    # ── SECCIÓN B: La menor, acordes bloque + mano derecha melódica (c.9-16) ─
    prog_B = ['Am','Dm','E','Am',  'Am','F','G','Am']
    chords_B = {
        'Am': [45, 48, 52],
        'Dm': [50, 53, 57],
        'E':  [52, 56, 59],
        'F':  [53, 57, 60],
        'G':  [55, 59, 62],
    }
    mel_B = {
        'Am': [69, 67, 65, 64],
        'Dm': [62, 60, 62, 65],
        'E':  [64, 66, 68, 69],
        'F':  [65, 67, 69, 67],
        'G':  [67, 65, 64, 62],
    }
    for ci, chord_name in enumerate(prog_B):
        base = 32 + ci * 4
        plist = chords_B[chord_name]
        # Mezcla: primer tiempo bloque, resto arpegios
        left.append((plist[0], 72, base,     3.8))  # bajo sostenido
        for qi, p in enumerate(plist[1:], 1):
            left.append((p, 65, base + qi, 0.85))   # resto en arpegio
        for qi, p in enumerate(mel_B[chord_name]):
            right.append((p, 80, base + qi, 0.88))

    # ── SECCIÓN A': Do mayor, acordes en inversión + arpegios lentos (c.17-24) ─
    # Primera inversión: bajo en la 3ª del acorde
    # Segunda inversión: bajo en la 5ª del acorde
    prog_Ap = [
        ('C',  [64, 67, 72]),   # C/E  — 1ª inv: bajo en Mi
        ('G',  [62, 67, 71]),   # G/D  — 2ª inv: bajo en Re
        ('Am', [60, 64, 69]),   # Am/C — 1ª inv: bajo en Do
        ('F',  [60, 65, 69]),   # F/C  — 2ª inv: bajo en Do
        ('C',  [64, 67, 72]),   # C/E  — 1ª inv
        ('F',  [57, 60, 65]),   # Fmaj — posición raíz
        ('G',  [55, 59, 62]),   # Gmaj — posición raíz
        ('C',  [60, 64, 67]),   # C    — posición raíz (cadencia final)
    ]
    mel_Ap_map = {
        'C':  [72, 74, 76, 77],
        'G':  [74, 72, 71, 69],
        'Am': [69, 71, 72, 71],
        'F':  [69, 67, 65, 64],
    }
    for ci, (chord_name, chord_pitches) in enumerate(prog_Ap):
        base = 64 + ci * 4
        for qi, p in enumerate(chord_pitches):
            left.append((p, 65, base + qi, 0.85))
        for qi, p in enumerate(mel_Ap_map[chord_name]):
            right.append((p, 75, base + qi, 0.88))

    mid.tracks.append(make_track('Mano Derecha',   0, right, add_meta=True, bpm=96))
    mid.tracks.append(make_track('Mano Izquierda', 1, left,  add_meta=False))

    mid.save(path)
    print(f"✓ MIDI de prueba v2 guardado: {path}")
    print(f"  A (Do mayor, arpegios) | B (La menor) | A' (Do mayor, inv.)")
    print(f"  Pista 0: Mano Derecha  |  Pista 1: Mano Izquierda")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Piano Duo Analyzer v2 — visualizador HTML de piano a dos manos')
    parser.add_argument('midi', nargs='?', help='Archivo MIDI de entrada')
    parser.add_argument('-o', '--output', default=None,
                        help='Salida HTML (default: <midi>_piano.html)')
    parser.add_argument('--right', type=int, default=0,
                        help='Índice de pista para mano derecha (default: 0)')
    parser.add_argument('--left',  type=int, default=1,
                        help='Índice de pista para mano izquierda (default: 1)')
    parser.add_argument('--arp-window', type=float, default=1.0, dest='arp_window',
                        help='Ventana en tiempos para agrupar arpegios (default: 1.0; 0=desactivado)')
    parser.add_argument('--key-window', type=int, default=8, dest='key_window',
                        help='Compases para detección local de tonalidad (default: 8; 0=solo global)')
    parser.add_argument('--harmony-source', default=None, dest='harmony_source',
                        help=('Fuente armónica por sección. Ejemplos: '
                              '"left", "right", "1-4:right,5-12:left". '
                              'Sin especificar → detección automática.'))
    parser.add_argument('--test', action='store_true',
                        help='Genera MIDI sintético de prueba y lo analiza')
    args = parser.parse_args()

    if args.test or args.midi is None:
        test_path = '/home/claude/test_piano_duo_v2.mid'
        generate_test_midi(test_path)
        mid_path = test_path
    else:
        mid_path = args.midi

    out_path = args.output or (os.path.splitext(mid_path)[0] + '_piano.html')

    print(f"\n{'═'*58}")
    print(f"  PIANO DUO ANALYZER  v2.0")
    print(f"  arp_window={args.arp_window} beats · key_window={args.key_window} bars")
    print(f"{'═'*58}")

    html = analyze(mid_path, args.right, args.left,
                   arp_window=args.arp_window,
                   key_window_bars=args.key_window,
                   harmony_source_spec=args.harmony_source)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  ✓ Informe HTML generado: {out_path}")
    print(f"{'═'*58}\n")


if __name__ == '__main__':
    main()
