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
    1:  '#f08030',   # Do#
    2:  '#f97316',   # Re  → Naranja
    3:  '#f0b020',   # Re#
    4:  '#fbbf24',   # Mi  → Amarillo
    5:  '#22c55e',   # Fa  → Verde
    6:  '#1aaa40',   # Fa#
    7:  '#38bdf8',   # Sol → Azul claro
    8:  '#2080d8',   # Sol#
    9:  '#3b82f6',   # La  → Azul
    10: '#6040b8',   # La#
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
        for n in b_notes:
            gi = group_idx(n['start_beat'])
            sub_groups.setdefault(gi, []).append(n)

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
    Returns the *structural* position of the note in the chord (0-3), or -1.

    Uses template interval order from the root — NOT the voicing order of the
    sounding pitches — so inversions always produce the same colours:

      Cmaj root pos  (C3-E3-G3):   C→0  E→1  G→2
      Cmaj 1st inv   (E3-G3-C4):   C→0  E→1  G→2   ← same colours
      Cmaj 2nd inv   (G2-C3-E3):   C→0  E→1  G→2   ← same colours

    Template order for common types:
      maj/min/dim/aug/sus: root(0), 3rd/2nd/4th(1), 5th(2)
      *7 chords:           root(0), 3rd(1), 5th(2), 7th(3)
    """
    if chord_event is None:
        return -1
    root_pc = chord_event.get('root_pc', -1)
    ctype   = chord_event.get('ctype', '')
    if root_pc < 0:
        return -1

    template = CHORD_TEMPLATES.get(ctype)
    if not template:
        return 0 if pitch % 12 == root_pc else -1

    # Structural PCs in template order (not sorted)
    struct_pcs = [(root_pc + iv) % 12 for iv in template]
    note_pc    = pitch % 12
    if note_pc in struct_pcs:
        return min(struct_pcs.index(note_pc), 3)
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
               n_bars: int) -> str:

    bpb       = right_track['bpb']
    all_beats = ([n['end_beat'] for n in right_track['notes']] +
                 [n['end_beat'] for n in left_track['notes']])
    # Use n_bars * bpb as total_beats so bar boundaries land on exact pixel columns.
    # max(end_beat) can be fractionally short due to note-off quantization.
    total_beats = n_bars * bpb

    CANVAS_W    = 1600
    ROW_H_ROLL  = 110
    ROW_H_CHORD = 44
    LABEL_W     = 200

    # Build a beat→source lookup for fast per-note decisions
    # src_per_bar is indexed by bar number
    def src_at_bar(bar: int) -> str:
        return src_per_bar[bar] if bar < len(src_per_bar) else 'left'

    # ── Right hand note rects ──────────────────────────────────────────────
    # When RH IS the harmony source → color by root (same as LH coloring)
    # When RH is NOT the harmony source → color by structural position in chord
    right_notes = right_track['notes']
    rh_pitches  = [n['pitch'] for n in right_notes] or [60]
    rh_p_min    = max(0,   min(rh_pitches) - 2)
    rh_p_max    = min(127, max(rh_pitches) + 2)
    rh_p_range  = max(1, rh_p_max - rh_p_min)

    right_rects = []
    for n in right_notes:
        bar      = n['start_bar']
        src      = src_at_bar(bar)
        chord_ev = find_chord_at_beat(chord_timeline, n['start_beat'])
        if src == 'right':
            # This track IS the harmony source → color by bass note (lowest pitch of chord)
            bass_pc = chord_ev['bass_pc'] if chord_ev and chord_ev.get('bass_pc', -1) >= 0 else n['pitch'] % 12
            color   = ROOT_NOTE_COLORS_EXACT.get(bass_pc, '#94a3b8')
            role    = 'harmony'
        else:
            # This track is the melody → color by position in chord
            pos   = chord_position_for_note(n['pitch'], chord_ev)
            color = CHORD_POSITION_COLORS.get(pos, '#ef4444')
            role  = 'melody'
        x      = round(n['start_beat'] / total_beats * CANVAS_W, 2)
        w      = round(max(2.0, n['dur_beat'] / total_beats * CANVAS_W), 2)
        frac   = (n['pitch'] - rh_p_min) / rh_p_range
        note_h = max(2.0, round(ROW_H_ROLL / rh_p_range * 0.9, 2))
        y      = round((1.0 - frac) * (ROW_H_ROLL - note_h), 2)
        right_rects.append({'x': x, 'y': y, 'w': w, 'h': note_h, 'c': color, 'role': role})

    # ── Left hand note rects ───────────────────────────────────────────────
    # When LH IS the harmony source → color by root
    # When LH is NOT the harmony source → color by structural position in chord
    left_notes = left_track['notes']
    lh_pitches = [n['pitch'] for n in left_notes] or [48]
    lh_p_min   = max(0,   min(lh_pitches) - 2)
    lh_p_max   = min(127, max(lh_pitches) + 2)
    lh_p_range = max(1, lh_p_max - lh_p_min)

    left_rects = []
    for n in left_notes:
        bar      = n['start_bar']
        src      = src_at_bar(bar)
        chord_ev = find_chord_at_beat(chord_timeline, n['start_beat'])
        if src == 'left':
            # LH is harmony source → color by bass note (lowest pitch of chord)
            bass_pc = chord_ev['bass_pc'] if chord_ev and chord_ev.get('bass_pc', -1) >= 0 else n['pitch'] % 12
            color   = ROOT_NOTE_COLORS_EXACT.get(bass_pc, '#94a3b8')
            role    = 'harmony'
        else:
            # LH is melody/bass → position color
            pos   = chord_position_for_note(n['pitch'], chord_ev)
            color = CHORD_POSITION_COLORS.get(pos, '#ef4444')
            role  = 'melody'
        x      = round(n['start_beat'] / total_beats * CANVAS_W, 2)
        w      = round(max(2.0, n['dur_beat'] / total_beats * CANVAS_W), 2)
        frac   = (n['pitch'] - lh_p_min) / lh_p_range
        note_h = max(2.0, round(ROW_H_ROLL / lh_p_range * 0.9, 2))
        y      = round((1.0 - frac) * (ROW_H_ROLL - note_h), 2)
        left_rects.append({'x': x, 'y': y, 'w': w, 'h': note_h, 'c': color, 'role': role})

    # Unified beat→pixel converter (used everywhere)
    def bx(beat: float) -> float:
        return round(beat / total_beats * CANVAS_W, 2)
    def bw(dur: float) -> float:
        return round(max(2.0, dur / total_beats * CANVAS_W), 2)

    # ── Chord bar items — positioned by chord_timeline start_beat ──────────
    # Use chord_timeline directly (beat-accurate) instead of per-bar buckets
    chord_items = []
    for ev in chord_timeline:
        # Find the bar's harmonic function from chords_per_bar
        bar = int(ev['start_beat'] / bpb)
        cpb = chords_per_bar[bar] if bar < len(chords_per_bar) else {}
        local_key_label = (f"{ROOT_NOTE_NAMES[cpb['tonic']]} {cpb['mode']}"
                           if cpb.get('tonic', -1) >= 0 else '')
        func  = cpb.get('function', '—')
        color = cpb.get('color', HARM_FUNC_COLORS['—'])
        x = bx(ev['start_beat'])
        w = bw(ev['end_beat'] - ev['start_beat'])
        chord_items.append({
            'x':     x,
            'w':     w,
            'label': ev['chord_name'],
            'func':  func,
            'color': color,
            'key':   local_key_label,
            'inv':   ev.get('inv_str', ''),
        })

    # ── Harmony-source spans ───────────────────────────────────────────────
    src_spans = []
    i = 0
    while i < n_bars:
        s = src_per_bar[i] if i < len(src_per_bar) else 'left'
        j = i
        while j < n_bars and (src_per_bar[j] if j < len(src_per_bar) else 'left') == s:
            j += 1
        x = bx(i * bpb)
        w = bw((j - i) * bpb)
        src_spans.append({
            'x':  x,
            'w':  w,
            'src': s,
            'label': 'MD' if s == 'right' else 'MI',
            'color': '#3b82f6' if s == 'right' else '#22c55e',
        })
        i = j

    # ── Bar ruler — beat-based x positions ────────────────────────────────
    bar_ticks_html = ''
    tick_every = max(1, n_bars // 32)
    for bi in range(0, n_bars, tick_every):
        x_pct = (bi * bpb) / total_beats * 100.0
        bar_ticks_html += (
            f'<div class="bar-tick" style="left:{x_pct:.4f}%">{bi+1}</div>'
        )

    # ── Legend HTML ─────────────────────────────────────────────────────────
    legend_html = ''
    legend_html += '<span class="leg-title">FUENTE ARMÓNICA:</span>'
    legend_html += ('<div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>MD lleva armonía</div>'
                    '<div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>MI lleva armonía</div>')
    legend_html += '<span class="leg-title" style="margin-left:16px">POSICIÓN EN ACORDE (mano melódica):</span>'
    pos_labels = [(0,'1ª (fund.)'), (1,'2ª'), (2,'3ª'), (3,'4ª'), (-1,'Fuera')]
    for pos, lbl in pos_labels:
        col = CHORD_POSITION_COLORS[pos]
        legend_html += (f'<div class="legend-item">'
                        f'<div class="legend-dot" style="background:{col};'
                        f'{"border:1px solid #555" if pos==3 else ""}"></div>{lbl}</div>')
    legend_html += '<span class="leg-title" style="margin-left:16px">NOTA MÁS GRAVE (mano armónica):</span>'
    root_samples = [(0,'Do'),(2,'Re'),(4,'Mi'),(5,'Fa'),(7,'Sol'),(9,'La'),(11,'Si')]
    for pc, name in root_samples:
        col = ROOT_NOTE_COLORS_EXACT[pc]
        legend_html += (f'<div class="legend-item">'
                        f'<div class="legend-dot" style="background:{col}"></div>{name}</div>')
    legend_html += '<span class="leg-title" style="margin-left:16px">FUNCIÓN ARMÓNICA:</span>'
    for func, col in HARM_FUNC_COLORS.items():
        if func == '—': continue
        legend_html += (f'<div class="legend-item">'
                        f'<div class="legend-dot" style="background:{col}"></div>'
                        f'{HARM_FUNC_LABELS.get(func, func)}</div>')

    title = os.path.basename(mid_path)
    tonic_name = ROOT_NOTE_NAMES[global_tonic]

    # Build compact key-change summary for the header
    key_changes = []
    prev_key = None
    for b, k in enumerate(keys_per_bar):
        label = f'{ROOT_NOTE_NAMES[k[0]]} {k[1]}'
        if label != prev_key:
            key_changes.append((b + 1, label))
            prev_key = label
    if len(key_changes) <= 1:
        key_summary = f'{tonic_name} {global_mode}'
    else:
        key_summary = ' → '.join(f'c.{b} {l}' for b, l in key_changes[:6])
        if len(key_changes) > 6:
            key_summary += ' …'

    js = f"""
(function() {{
  var CW    = {CANVAS_W};
  var NBARS = {n_bars};

  function drawGrid(ctx, h) {{
    var bw = CW / NBARS;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    for (var b = 0; b <= NBARS; b++) {{
      ctx.beginPath(); ctx.moveTo(b*bw, 0); ctx.lineTo(b*bw, h); ctx.stroke();
    }}
  }}

  var SRC_SPANS = {json.dumps(src_spans)};

  // Helper: draw faint source-region background tint on a canvas
  function drawSrcBg(ctx, H) {{
    SRC_SPANS.forEach(function(s) {{
      ctx.fillStyle = s.color + '18';
      ctx.fillRect(s.x, 0, s.w, H);
    }});
  }}

  // ── SOURCE INDICATOR ROW ───────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_src');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#060c1a';
    ctx.fillRect(0, 0, CW, H);
    SRC_SPANS.forEach(function(s) {{
      ctx.fillStyle = s.color + '30';
      ctx.fillRect(s.x, 0, s.w, H);
      ctx.fillStyle = s.color;
      ctx.fillRect(s.x, 0, s.w, 3);
      if (s.w > 20) {{
        ctx.font = 'bold 9px monospace';
        ctx.fillStyle = s.color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(s.label, s.x + Math.min(s.w / 2, 30), H / 2);
      }}
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(s.x + 0.5, 0.5, s.w - 1, H - 1);
    }});
  }})();

  // ── RIGHT HAND ─────────────────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_right');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#0b1628';
    ctx.fillRect(0, 0, CW, H);
    drawSrcBg(ctx, H);
    drawGrid(ctx, H);
    var rects = {json.dumps(right_rects)};
    rects.forEach(function(r) {{
      ctx.fillStyle = r.c;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.strokeStyle = 'rgba(0,0,0,0.4)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }});
  }})();

  // ── LEFT HAND ──────────────────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_left');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#0b1420';
    ctx.fillRect(0, 0, CW, H);
    drawSrcBg(ctx, H);
    drawGrid(ctx, H);
    var rects = {json.dumps(left_rects)};
    rects.forEach(function(r) {{
      ctx.fillStyle = r.c;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.strokeStyle = 'rgba(0,0,0,0.4)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }});
  }})();

  // ── CHORDS ─────────────────────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_chords');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#080f1e';
    ctx.fillRect(0, 0, CW, H);
    var items = {json.dumps(chord_items)};
    items.forEach(function(c) {{
      ctx.fillStyle = c.color + '28';
      ctx.fillRect(c.x, 0, c.w, H);
      ctx.fillStyle = c.color;
      ctx.fillRect(c.x, 0, c.w, 3);
      if (c.label && c.label !== '\u2014') {{
        // If there's an inversion label, shift chord name up slightly
        var hasInv = c.inv && c.inv.length > 0 && c.w > 36;
        var nameY  = hasInv ? H / 2 - 3 : H / 2 + 1;
        ctx.fillStyle = c.color;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(c.label, c.x + c.w / 2, nameY);
        if (hasInv) {{
          ctx.fillStyle = c.color + 'bb';
          ctx.font = '7px monospace';
          ctx.textBaseline = 'top';
          ctx.fillText(c.inv, c.x + c.w / 2, nameY + 8);
        }}
      }}
      if (c.func && c.func !== '\u2014' && c.w > 28) {{
        ctx.fillStyle = c.color + 'aa';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(c.func, c.x + c.w / 2, H - 2);
      }}
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c.x + 0.5, 0.5, c.w - 1, H - 1);
    }});
  }})();

}})();
"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Piano Duo — {title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
       background: #080f1c; color: #e2e8f0; font-size: 13px; }}
.header {{ padding: 18px 24px 12px; border-bottom: 1px solid #1a2744; }}
.header h1 {{ font-size: 17px; font-weight: 700; color: #f8fafc; letter-spacing: 0.01em; }}
.header .meta {{ font-size: 11px; color: #4a6080; margin-top: 5px; }}
.scroll-wrap {{ overflow-x: auto; }}
.tl-outer {{ display: inline-flex; flex-direction: column; min-width: 100%; }}
.ruler-row {{ display: flex; height: 22px; border-bottom: 1px solid #1a2744; }}
.ruler-label {{ width: {LABEL_W}px; min-width: {LABEL_W}px; flex-shrink: 0;
                background: #070e1a; border-right: 2px solid #1a2744; }}
.ruler-track {{ flex: 1; position: relative; background: #070e1a;
                min-width: {CANVAS_W}px; }}
.bar-tick {{ position: absolute; top: 0; height: 100%;
             border-left: 1px solid rgba(255,255,255,0.08);
             font-size: 9px; color: #334466; padding-left: 3px;
             line-height: 22px; user-select: none; }}
.tl-row {{ display: flex; border-bottom: 1px solid #1a2744; }}
.tl-label {{ width: {LABEL_W}px; min-width: {LABEL_W}px; padding: 5px 8px 5px 14px;
             display: flex; flex-direction: column; justify-content: center;
             border-right: 2px solid #1a2744; background: #070e1a;
             flex-shrink: 0; }}
.trk-name {{ font-size: 12px; font-weight: 700; color: #dde6f0; }}
.trk-sub  {{ font-size: 9px; margin-top: 3px; opacity: 0.75; line-height: 1.4; }}
.tl-canvas-wrap {{ flex: 1; overflow: hidden; }}
canvas {{ display: block; }}
.legend-row {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 18px;
               border-top: 1px solid #1a2744; background: #050c18; align-items: center; }}
.legend-item {{ display: flex; align-items: center; gap: 5px;
                font-size: 10px; color: #64748b; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
.leg-title {{ font-size: 10px; color: #475569; font-weight: 700; margin-right: 2px; }}
.divider {{ height: 3px; background: linear-gradient(to right, #1a2744, #0d1b35, #1a2744); }}
</style>
</head>
<body>
<div class="header">
  <h1>🎹 {title}</h1>
  <div class="meta">
    {n_bars} compases · {bpb:.0f}/4 · Tonalidad: {key_summary}
    · Mano derecha: {right_track['name']} · Mano izquierda: {left_track['name']}
  </div>
</div>

<div class="scroll-wrap">
<div class="tl-outer">

  <div class="ruler-row">
    <div class="ruler-label"></div>
    <div class="ruler-track">{bar_ticks_html}</div>
  </div>

  <!-- SOURCE INDICATOR ROW -->
  <div class="tl-row" style="height:20px">
    <div class="tl-label" style="border-left:3px solid #475569">
      <div class="trk-name" style="font-size:9px;color:#475569">Fuente armonía</div>
    </div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_src" width="{CANVAS_W}" height="20"></canvas>
    </div>
  </div>

  <!-- ROW 1: RIGHT HAND -->
  <div class="tl-row" style="height:{ROW_H_ROLL}px">
    <div class="tl-label" style="border-left:3px solid #3b82f6">
      <div class="trk-name">🎹 Mano Derecha</div>
      <div class="trk-sub" style="color:#3b82f6">
        {right_track['name']}<br>
        color → posición en acorde izq.
      </div>
    </div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_right" width="{CANVAS_W}" height="{ROW_H_ROLL}"></canvas>
    </div>
  </div>

  <!-- ROW 2: LEFT HAND -->
  <div class="tl-row" style="height:{ROW_H_ROLL}px">
    <div class="tl-label" style="border-left:3px solid #22c55e">
      <div class="trk-name">🎹 Mano Izquierda</div>
      <div class="trk-sub" style="color:#22c55e">
        {left_track['name']}<br>
        color → nota raíz del acorde
      </div>
    </div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_left" width="{CANVAS_W}" height="{ROW_H_ROLL}"></canvas>
    </div>
  </div>

  <!-- ROW 3: CHORDS -->
  <div class="tl-row" style="height:{ROW_H_CHORD}px">
    <div class="tl-label" style="border-left:3px solid #f59e0b">
      <div class="trk-name">Acordes</div>
      <div class="trk-sub" style="color:#f59e0b">función armónica</div>
    </div>
    <div class="tl-canvas-wrap">
      <canvas id="cv_chords" width="{CANVAS_W}" height="{ROW_H_CHORD}"></canvas>
    </div>
  </div>

</div>
</div>

<div class="legend-row">
  {legend_html}
</div>

<script>
{js}
</script>
</body>
</html>"""

    return html

# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

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
    If spec is given, manual ranges override; 'auto' bars are decided by
    harmonic density comparison.
    Smoothing: a source switch is only accepted if it persists for ≥2 bars
    (avoids flickering on transitional bars).
    """
    directives = _parse_harmony_source(spec, n_bars)
    raw = []
    for b in range(n_bars):
        if directives[b] in ('right', 'left'):
            raw.append(directives[b])
        else:
            dr = _harmonic_density(right_notes, b, bpb)
            dl = _harmonic_density(left_notes,  b, bpb)
            raw.append('right' if dr >= dl else 'left')

    # ── Smoothing: require ≥2 consecutive bars to switch source ──────────
    smoothed = list(raw)
    for b in range(1, n_bars - 1):
        if directives[b] in ('right', 'left'):
            continue   # manual override — never smooth
        if smoothed[b] != smoothed[b - 1]:
            # Check if next bar agrees with the switch
            if b + 1 < n_bars and smoothed[b + 1] == smoothed[b]:
                pass   # genuine switch confirmed by next bar
            else:
                smoothed[b] = smoothed[b - 1]   # revert isolated flip

    return smoothed

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
    print("  Generando HTML...")
    html = build_html(mid_path, right_track, left_track,
                      chord_timeline, chords_per_bar,
                      global_tonic, global_mode, keys_per_bar,
                      src_per_bar, n_bars)
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
