#!/usr/bin/env python3
"""
midi_pianoroll_analyzer.py
══════════════════════════
Analizador multi-pista MIDI con visualización HTML interactiva.

Algoritmos incluidos (sin dependencias externas salvo mido + numpy):
  • Re-Pair / Sequitur simplificado — segmentación por frases en cada pista
  • Segmentación global por secciones (SSM sobre ventanas de compás)
  • Detección de acordes (armonía)
  • Métricas por compás: tensión, densidad, velocidad media, contorno melódico

Uso:
    python3 midi_pianoroll_analyzer.py <archivo.mid> [config.yaml] [-o salida.html]

Configuración YAML (opcional):
    tracks:
      0: {role: melody,       name: "Violín"}
      1: {role: counter,      name: "Viola"}
      2: {role: harmony,      name: "Piano der."}
      3: {role: bass,         name: "Contrabajo"}
      4: {role: accompaniment,name: "Piano izq."}
    algorithms:
      sequitur: true
      global_segmentation: true
      chords: true
      metrics: true
    output: report.html
"""

from __future__ import annotations
import sys, os, json, math, argparse, colorsys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import mido
except ImportError:
    sys.exit("ERROR: instala mido  →  pip install mido")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

# ═══════════════════════════════════════════════════════════════════════════════
# MIDI PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_midi(path: str):
    return mido.MidiFile(path)

def ticks_to_beats(ticks: int, tpb: int) -> float:
    return ticks / tpb

def extract_tempo_map(mid) -> List[Tuple[int, int]]:
    """Returns [(tick, tempo_us), ...] sorted by tick."""
    events = [(0, 500000)]
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                events.append((tick, msg.tempo))
    events.sort()
    return events

def tick_to_seconds(tick: int, tempo_map: List[Tuple[int, int]], tpb: int) -> float:
    t_sec = 0.0
    prev_tick, prev_tempo = 0, 500000
    for ev_tick, ev_tempo in tempo_map:
        if ev_tick >= tick:
            break
        t_sec += (ev_tick - prev_tick) / tpb * prev_tempo / 1e6
        prev_tick, prev_tempo = ev_tick, ev_tempo
    t_sec += (tick - prev_tick) / tpb * prev_tempo / 1e6
    return t_sec

def beats_per_bar(mid) -> float:
    """Returns beats per bar from first time signature, default 4."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return msg.numerator
    return 4.0

def extract_tracks(mid) -> List[Dict]:
    """
    Returns list of track dicts:
      {name, channel, notes: [{pitch, vel, start_tick, end_tick, start_beat, dur_beat}]}
    """
    tpb = mid.ticks_per_beat
    tempo_map = extract_tempo_map(mid)
    bpb = beats_per_bar(mid)
    tracks = []

    for ti, track in enumerate(mid.tracks):
        name = track.name or f"Track {ti}"
        notes_on = {}  # (ch, pitch) -> (tick, vel)
        notes = []
        tick = 0
        channel_set = set()

        for msg in track:
            tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes_on[(msg.channel, msg.note)] = (tick, msg.velocity)
                channel_set.add(msg.channel)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in notes_on:
                    start_tick, vel = notes_on.pop(key)
                    dur_ticks = tick - start_tick
                    start_beat = ticks_to_beats(start_tick, tpb)
                    dur_beat   = ticks_to_beats(dur_ticks, tpb)
                    start_bar  = int(start_beat / bpb)
                    notes.append({
                        'pitch': msg.note,
                        'vel':   vel,
                        'start_tick': start_tick,
                        'end_tick':   tick,
                        'start_beat': start_beat,
                        'end_beat':   start_beat + dur_beat,
                        'dur_beat':   dur_beat,
                        'start_bar':  start_bar,
                        'start_sec':  tick_to_seconds(start_tick, tempo_map, tpb),
                    })

        if notes:
            notes.sort(key=lambda n: n['start_tick'])
            tracks.append({
                'index':    ti,
                'name':     name,
                'channel':  min(channel_set) if channel_set else 0,
                'notes':    notes,
                'tpb':      tpb,
                'bpb':      bpb,
                'role':     None,
            })

    return tracks

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG / ROLE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

ROLES = ['melody', 'counter', 'harmony', 'bass', 'accompaniment']
ROLE_LABELS = {
    'melody':        '♪ Melodía',
    'counter':       '♫ Contra-melodía',
    'harmony':       '♬ Armonía',
    'bass':          '♩ Bajo',
    'accompaniment': '⁝ Acompañamiento',
}
ROLE_COLORS = {
    'melody':        '#6366f1',
    'counter':       '#10b981',
    'harmony':       '#f59e0b',
    'bass':          '#ef4444',
    'accompaniment': '#8b5cf6',
    None:            '#64748b',
}

def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        text = f.read()
    if ext in ('.yaml', '.yml'):
        # Minimal YAML parser — supports simple key: value and nested with 2-space indent
        return _parse_simple_yaml(text)
    return json.loads(text)

def _parse_simple_yaml(text: str) -> dict:
    """
    Minimal YAML parser supporting:
      - Nested dicts via indentation
      - Scalars: str, int, float, bool, null
      - Inline comments (# ...)
      - Compact inline dicts: {key: val, key2: val2}
      - Inline lists: [a, b, c]
      - Block lists: items starting with '- '
    """
    import re

    def parse_scalar(s):
        s = s.strip()
        if not s or s.lower() in ('null', '~'):
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s.strip('"').strip("'")

    def parse_inline_list(s):
        s = s.strip()
        if not (s.startswith('[') and s.endswith(']')):
            return None
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(x) for x in re.split(r',\s*', inner)]

    def parse_inline_dict(s):
        s = s.strip()
        if not (s.startswith('{') and s.endswith('}')):
            return None
        inner = s[1:-1].strip()
        if not inner:
            return {}
        result = {}
        for part in re.split(r',\s*(?=[^{}]*$)', inner):
            if ':' in part:
                k, _, v = part.partition(':')
                k = k.strip().strip('"').strip("'")
                result[k] = parse_scalar(v.strip())
        return result

    def strip_comment(s):
        in_q = None
        for i, c in enumerate(s):
            if c in ('"', "'") and in_q is None:
                in_q = c
            elif c == in_q:
                in_q = None
            elif c == '#' and in_q is None:
                return s[:i].rstrip()
        return s.rstrip()

    # Tokenize: (indent, content)
    lines = []
    for raw in text.splitlines():
        content = raw.rstrip()
        stripped = content.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        indent  = len(content) - len(stripped)
        stripped = strip_comment(stripped)
        if stripped:
            lines.append((indent, stripped))

    # Stack entries: (indent, container)
    # container is either a dict or a list
    root  = {}
    stack = [(-1, root)]   # (indent_of_key, container)

    i = 0
    while i < len(lines):
        indent, content = lines[i]

        # Pop stack to correct parent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        # ── List item: starts with '- ' ────────────────────────────────
        if content.startswith('- '):
            rest = content[2:].strip()
            # Ensure parent is a list; if not, create one at the right key
            if not isinstance(parent, list):
                # This shouldn't happen in well-formed YAML, skip
                i += 1
                continue

            if rest.startswith('{'):
                parsed = parse_inline_dict(rest)
                parent.append(parsed if parsed is not None else rest)
            elif ':' in rest:
                # Inline dict item without braces: key: val
                new_dict = {}
                k, _, v = rest.partition(':')
                k = k.strip(); v = v.strip()
                if v:
                    new_dict[k] = parse_scalar(v)
                    # Consume following lines at deeper indent as more keys
                    while i + 1 < len(lines):
                        ni, nc = lines[i + 1]
                        if ni > indent and ':' in nc and not nc.startswith('- '):
                            i += 1
                            k2, _, v2 = nc.partition(':')
                            new_dict[k2.strip()] = parse_scalar(v2.strip())
                        else:
                            break
                    parent.append(new_dict)
                else:
                    new_dict[k] = {}
                    parent.append(new_dict)
                    stack.append((indent, new_dict[k]))
            else:
                parent.append(parse_scalar(rest))

        # ── Dict entry: key: value ──────────────────────────────────────
        elif ':' in content:
            key, _, rest = content.partition(':')
            key  = key.strip().strip('"').strip("'")
            rest = rest.strip()

            # Numeric key
            try:
                key = int(key)
            except (ValueError, TypeError):
                pass

            if rest == '':
                # Look ahead: is next line a list item?
                if (i + 1 < len(lines) and
                        lines[i + 1][0] > indent and
                        lines[i + 1][1].startswith('- ')):
                    new_list = []
                    parent[key] = new_list
                    stack.append((indent, new_list))
                else:
                    new_dict = {}
                    parent[key] = new_dict
                    stack.append((indent, new_dict))
            elif rest.startswith('['):
                parsed = parse_inline_list(rest)
                parent[key] = parsed if parsed is not None else rest
            elif rest.startswith('{'):
                parsed = parse_inline_dict(rest)
                parent[key] = parsed if parsed is not None else rest
            else:
                parent[key] = parse_scalar(rest)

        i += 1

    return root

def apply_config(tracks: List[Dict], cfg: dict) -> List[Dict]:
    """Apply role/name config to tracks. Auto-assign roles if not configured."""
    track_cfg = cfg.get('tracks', {})

    for t in tracks:
        tc = track_cfg.get(t['index'], track_cfg.get(str(t['index']), {}))
        if isinstance(tc, dict):
            if 'role' in tc:
                t['role'] = tc['role']
            if 'name' in tc:
                t['name'] = tc['name']

    # Auto-assign roles if none configured
    assigned_roles = {t['role'] for t in tracks if t['role']}
    if not assigned_roles and tracks:
        _auto_assign_roles(tracks)

    return tracks

def _auto_assign_roles(tracks: List[Dict]):
    """Heuristic role assignment based on pitch range and density."""
    if not tracks:
        return

    scored = []
    for t in tracks:
        if not t['notes']:
            continue
        pitches = [n['pitch'] for n in t['notes']]
        avg_p  = sum(pitches) / len(pitches)
        n_bars = max(n['start_bar'] for n in t['notes']) + 1 if t['notes'] else 1
        density = len(t['notes']) / max(n_bars, 1)
        scored.append((t['index'], avg_p, density, len(t['notes'])))

    scored.sort(key=lambda x: -x[1])  # highest pitch first

    role_order = ROLES[:len(scored)]
    idx_map = {s[0]: i for i, s in enumerate(scored)}

    for t in tracks:
        i = idx_map.get(t['index'])
        if i is not None and i < len(role_order):
            t['role'] = role_order[i]

# ═══════════════════════════════════════════════════════════════════════════════
# SEQUITUR (Re-Pair variant) — assigns phrase IDs to notes
# ═══════════════════════════════════════════════════════════════════════════════

def repairify(symbols: List, custom_rules: Optional[List[Dict]] = None) -> Tuple[Dict, List]:
    """
    Re-Pair compression with optional custom rules pre-seeded.
    custom_rules: list of {'name': str, 'pitches': [int, ...]}
    Returns (rules, compressed_sequence).
    """
    rules: Dict = {}
    seq   = list(symbols)

    # ── 1. Pre-seed custom rules ──────────────────────────────────────────
    if custom_rules:
        # Sort longest first to avoid partial overlaps
        for cr in sorted(custom_rules, key=lambda r: -len(r['pitches'])):
            pattern = cr['pitches']
            name    = cr['name']
            if len(pattern) < 2:
                continue
            rules[name] = pattern
            # Replace occurrences in sequence
            new_seq = []
            i = 0
            while i < len(seq):
                if seq[i:i+len(pattern)] == pattern:
                    new_seq.append(name)
                    i += len(pattern)
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq

    # ── 2. Standard Re-Pair ───────────────────────────────────────────────
    rule_id = 0
    while True:
        pairs = Counter()
        for i in range(len(seq) - 1):
            pairs[(seq[i], seq[i+1])] += 1
        if not pairs:
            break
        best, cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        rule_id += 1
        rname = f'R{rule_id}'
        rules[rname] = list(best)
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq)-1 and (seq[i], seq[i+1]) == best:
                new_seq.append(rname)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq

    return rules, seq

def expand_rule(sym, rules: Dict) -> List:
    if sym not in rules:
        return [sym]
    result = []
    for s in rules[sym]:
        result.extend(expand_rule(s, rules))
    return result

def simplify_segments(segments: List[Dict], target: int) -> List[Dict]:
    """
    Merge the shortest segment with its smaller neighbor iteratively
    until len(segments) == target. Preserves note_indices.
    Each segment: {'rule': str, 'note_indices': [int,...], 'len': int}
    Adapted from mscz2vec.simplify_segments_preserving_indices().
    """
    import copy
    segs = copy.deepcopy(segments)
    while len(segs) > target:
        # Find shortest segment
        min_len = min(s['len'] for s in segs)
        min_idx = next(i for i, s in enumerate(segs) if s['len'] == min_len)

        if min_idx == 0:
            a, b = 0, 1
        elif min_idx == len(segs) - 1:
            a, b = len(segs) - 2, len(segs) - 1
        else:
            # Merge with the shorter neighbor
            prev_len = segs[min_idx - 1]['len']
            next_len = segs[min_idx + 1]['len']
            if prev_len <= next_len:
                a, b = min_idx - 1, min_idx
            else:
                a, b = min_idx, min_idx + 1

        merged = {
            'rule':         f"({segs[a]['rule']}+{segs[b]['rule']})",
            'note_indices': segs[a]['note_indices'] + segs[b]['note_indices'],
            'len':          segs[a]['len'] + segs[b]['len'],
        }
        segs[a] = merged
        del segs[b]
    return segs

def assign_phrase_ids(notes: List[Dict], seq_cfg: Dict) -> List[int]:
    """
    Run Re-Pair/Sequitur on the note sequence of one track.

    seq_cfg keys (all optional):
      n_phrases    int   — collapse output to exactly N phrases via simplify
      custom_rules list  — list of {'name': str, 'pitches': [int,...]}
                           or {'name': str, 'bars': [int,...]}   (bar numbers, 0-indexed)
      token        str   — 'pitch' (default) | 'pitch_class' | 'pitch_dur'
      min_count    int   — min frequency for Re-Pair rule (default 2, not exposed yet)

    Returns list of phrase IDs (int), one per note.
    Notes with identical onset always share the same phrase ID.
    """
    if not notes:
        return []

    # ── Token selection ───────────────────────────────────────────────────
    token_mode = seq_cfg.get('token', 'pitch')
    if token_mode == 'pitch_class':
        symbols = [n['pitch'] % 12 for n in notes]
    elif token_mode == 'pitch_dur':
        symbols = [(n['pitch'], round(n['dur_beat'] * 4) / 4) for n in notes]
    else:
        symbols = [n['pitch'] for n in notes]

    # ── Resolve bar-based custom rules to pitch lists ─────────────────────
    raw_custom = seq_cfg.get('custom_rules', [])
    resolved_custom: List[Dict] = []
    for cr in raw_custom:
        name = cr.get('name', 'custom')
        if 'pitches' in cr:
            resolved_custom.append({'name': name, 'pitches': list(cr['pitches'])})
        elif 'bars' in cr:
            # Collect pitches of notes whose start_bar is in the given list
            bars_raw = cr['bars']
            if isinstance(bars_raw, str):
                import re
                bars_raw = [int(x) for x in re.findall(r'\d+', bars_raw)]
            bar_set = set(bars_raw)
            pitches_in_bars = [n['pitch'] for n in notes if n['start_bar'] in bar_set]
            if pitches_in_bars:
                resolved_custom.append({'name': name, 'pitches': pitches_in_bars})

    # ── Run Re-Pair ───────────────────────────────────────────────────────
    rules, compressed = repairify(symbols, resolved_custom or None)

    # ── Build initial segments list ───────────────────────────────────────
    # Each token in compressed → one segment with its expanded note indices
    segments: List[Dict] = []
    note_cursor = 0
    phrase_map: Dict[str, int] = {}

    for token in compressed:
        expanded = expand_rule(token, rules)
        n_notes  = len(expanded)
        indices  = list(range(note_cursor, min(note_cursor + n_notes, len(notes))))
        note_cursor += n_notes

        if isinstance(token, str) and token in rules:
            rule_name = token
        else:
            rule_name = f'_t{note_cursor}'   # terminal

        segments.append({
            'rule':         rule_name,
            'note_indices': indices,
            'len':          len(indices),
        })

    # ── Simplify to target n_phrases ──────────────────────────────────────
    n_phrases = seq_cfg.get('n_phrases', None)
    if n_phrases is not None:
        target = max(1, int(n_phrases))
        if len(segments) > target:
            segments = simplify_segments(segments, target)

    # ── Canonicalize rules by their full expansion ────────────────────────
    # Two rules with the same expanded symbol sequence are the same motif
    # (Re-Pair often builds R11=motif×2, R12=motif×4 as distinct rules).
    # We map each rule to a canonical key = tuple(expansion), then assign
    # a single phrase ID per unique canonical form.
    def canonical(token) -> tuple:
        return tuple(expand_rule(token, rules))

    canon_to_pid: Dict[tuple, int] = {}
    pid_counter   = 1
    terminal_ctr  = 10000

    # Pre-compute canonical for every unique rule appearing in segments
    for seg in segments:
        rule = seg['rule']
        if rule.startswith('_t'):
            continue
        key = canonical(rule)
        # Find if this expansion is a prefix-repetition of a shorter pattern
        # e.g. [A,B,A,B,A,B] -> base [A,B]  so all repetition counts map to same ID
        def minimal_period(seq):
            n = len(seq)
            for p in range(1, n // 2 + 1):
                if n % p == 0 and seq == seq[:p] * (n // p):
                    return seq[:p]
            return seq
        key = minimal_period(list(key))
        key = tuple(key)
        if key not in canon_to_pid:
            canon_to_pid[key] = pid_counter
            pid_counter += 1

    # ── Assign phrase IDs from segments ───────────────────────────────────
    phrase_ids = [0] * len(notes)

    for seg in segments:
        rule = seg['rule']
        if rule.startswith('_t'):
            pid = terminal_ctr
            terminal_ctr += 1
        else:
            key = tuple(minimal_period(list(canonical(rule))))
            pid = canon_to_pid.get(key, terminal_ctr)

        for ni in seg['note_indices']:
            if ni < len(notes):
                phrase_ids[ni] = pid

    # ── Unify simultaneous notes (chords) → same phrase ID ────────────────
    onset_to_pid: Dict[float, int] = {}
    for ni, n in enumerate(notes):
        onset = round(n['start_beat'], 3)
        if onset not in onset_to_pid:
            onset_to_pid[onset] = phrase_ids[ni]
        else:
            phrase_ids[ni] = onset_to_pid[onset]

    return phrase_ids

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SECTION SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_bar_vectors(tracks, n_bars):
    """
    Build a rich multi-group feature vector per bar for SSM segmentation.

    30 features (each min-max normalized across all bars after computation):
      [0:12]  Pitch-class histogram (normalized by note count in bar)
      [12]    Note density  (notes per beat, capped at 1)
      [13]    Avg velocity  (0-1)
      [14]    Velocity variance (normalized)
      [15]    Avg MIDI pitch (0-127 -> 0-1)
      [16]    Pitch span / 48
      [17]    Pitch centroid octave / 9
      [18]    Avg note duration / 4 beats
      [19]    Duration variance (normalized)
      [20]    Rhythmic density: unique onsets per beat
      [21]    IOI variance
      [22]    Avg polyphony / 6
      [23]    Texture entropy / 3
      [24]    Melodic contour: ascending fraction
      [25]    Interval dissonance (mean Plomp-Levelt)
      [26]    Bass register fraction (pitch < 48)
      [27]    Treble register fraction (pitch >= 72)
      [28]    Unique pitch classes / 12
      [29]    Melodic leap ratio (interval > 2 semitones)
    """
    import math
    from collections import defaultdict, Counter

    bpb = tracks[0]['bpb'] if tracks else 4.0

    bar_notes = defaultdict(list)
    for t in tracks:
        for n in t['notes']:
            bar_notes[n['start_bar']].append(n)

    TENSION = {0:0.0,1:0.9,2:0.5,3:0.3,4:0.2,5:0.2,6:1.0,7:0.1,8:0.5,9:0.3,10:0.7,11:0.8}
    raw = []

    for bi in range(n_bars):
        bn  = bar_notes.get(bi, [])
        vec = [0.0] * 30

        if not bn:
            raw.append(vec)
            continue

        pitches   = [n['pitch']     for n in bn]
        vels      = [n['vel']       for n in bn]
        durs      = [n['dur_beat']  for n in bn]
        bar_start = bi * bpb

        # 0-11: pitch-class histogram
        pc = [0.0]*12
        for p in pitches: pc[p%12] += 1
        tot = sum(pc) or 1
        for i in range(12): vec[i] = pc[i]/tot

        # 12: density
        vec[12] = min(1.0, len(bn)/(bpb*4))

        # 13-14: velocity
        av = sum(vels)/len(vels)
        vec[13] = av/127.0
        vec[14] = min(1.0, sum((v-av)**2 for v in vels)/len(vels)/(30.0**2))

        # 15-17: pitch statistics
        ap = sum(pitches)/len(pitches)
        vec[15] = ap/127.0
        vec[16] = min(1.0,(max(pitches)-min(pitches))/48.0)
        vec[17] = (ap//12)/9.0

        # 18-19: duration
        ad = sum(durs)/len(durs)
        vec[18] = min(1.0, ad/4.0)
        vec[19] = min(1.0, sum((d-ad)**2 for d in durs)/len(durs)/(2.0**2))

        # 20-21: rhythm/IOI
        onsets = sorted(set(round(n['start_beat']-bar_start,3) for n in bn))
        vec[20] = min(1.0, len(onsets)/(bpb*2))
        iois = [onsets[i+1]-onsets[i] for i in range(len(onsets)-1)]
        if iois:
            ai = sum(iois)/len(iois)
            vec[21] = min(1.0, sum((x-ai)**2 for x in iois)/len(iois)/(1.0**2))

        # 22-23: polyphony / texture entropy
        vc = []
        for o in onsets:
            ao = bar_start+o
            vc.append(sum(1 for n in bn
                          if n['start_beat']<=ao<n['start_beat']+max(n['dur_beat'],0.01)))
        if vc:
            avg_poly = sum(vc)/len(vc)
            vec[22] = min(1.0, avg_poly/6.0)
            ch = Counter(vc); tv = len(vc)
            ent = -sum((c/tv)*math.log2(c/tv) for c in ch.values() if c>0)
            vec[23] = min(1.0, ent/3.0)

        # 24: contour
        sbn = sorted(bn, key=lambda n:(n['start_beat'],n['pitch']))
        mp  = [n['pitch'] for n in sbn]
        if len(mp)>1:
            diffs = [mp[i+1]-mp[i] for i in range(len(mp)-1)]
            asc  = sum(1 for d in diffs if d>0)
            tot2 = sum(1 for d in diffs if d!=0) or 1
            vec[24] = asc/tot2

        # 25: harmonic tension
        pcs2 = list({p%12 for p in pitches})
        tv2  = []
        for ii in range(len(pcs2)):
            for jj in range(ii+1,len(pcs2)):
                tv2.append(TENSION.get((pcs2[jj]-pcs2[ii])%12,0.5))
        if tv2: vec[25] = sum(tv2)/len(tv2)

        # 26-27: register
        vec[26] = sum(1 for p in pitches if p<48)/len(pitches)
        vec[27] = sum(1 for p in pitches if p>=72)/len(pitches)

        # 28: unique PC
        vec[28] = len(set(p%12 for p in pitches))/12.0

        # 29: leap ratio
        if len(mp)>1:
            leaps = sum(1 for d in diffs if abs(d)>2)
            vec[29] = leaps/len(diffs)

        raw.append(vec)

    # min-max normalize each feature across all bars
    n_feats = 30
    for fi in range(n_feats):
        col = [raw[bi][fi] for bi in range(n_bars)]
        lo,hi = min(col),max(col)
        sp = hi-lo
        if sp>1e-6:
            for bi in range(n_bars):
                raw[bi][fi] = (raw[bi][fi]-lo)/sp

    return raw



def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-9)

def euclidean_sim(a: List[float], b: List[float]) -> float:
    """Convert L2 distance to a [0,1] similarity via exp(-dist)."""
    dist = math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    # scale: distance of sqrt(n_feats) ~ 0 similarity; 0 ~ 1.0
    return math.exp(-dist / max(1.0, math.sqrt(len(a)) * 0.3))

def combined_sim(a: List[float], b: List[float],
                 cosine_w: float = 0.4, euclid_w: float = 0.6) -> float:
    """Weighted combination of cosine and Euclidean similarity.
    Euclidean is more sensitive to magnitude differences (register, velocity),
    cosine captures shape similarity (pitch-class distribution patterns).
    """
    return cosine_w * cosine_sim(a, b) + euclid_w * euclidean_sim(a, b)

def ssm_matrix(vecs: List[List[float]]) -> List[List[float]]:
    n = len(vecs)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = combined_sim(vecs[i], vecs[j])
    return mat

def novelty_curve(ssm: List[List[float]], kernel_size: int = 4) -> List[float]:
    """Foote novelty curve from SSM (checkerboard kernel)."""
    n = len(ssm)
    k = kernel_size
    novelty = [0.0] * n
    for i in range(k, n - k):
        score = 0.0
        for di in range(k):
            for dj in range(k):
                # upper-left vs lower-right (same block) → positive
                a = ssm[i - di - 1][i - dj - 1]
                b = ssm[i + di][i + dj]
                # upper-right vs lower-left (cross block) → negative
                c = ssm[i - di - 1][i + dj]
                d = ssm[i + di][i - dj - 1]
                score += (a + b) - (c + d)
        novelty[i] = score  # allow negative, we pick positive peaks
    # normalize to [0,1]
    mx = max(novelty) if max(novelty) > 0 else 1.0
    return [max(0.0, v / mx) for v in novelty]

def pick_peaks(curve: List[float], min_gap: int = 4, threshold: float = 0.4) -> List[int]:
    """Pick local maxima above threshold with a minimum gap."""
    peaks = []
    n = len(curve)
    for i in range(1, n - 1):
        if curve[i] >= threshold and curve[i] > curve[i-1] and curve[i] > curve[i+1]:
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    return peaks

def assign_section_labels(boundaries: List[int], n_bars: int,
                           vecs: List[List[float]]) -> List[str]:
    """
    Assign section labels A, B, C… by clustering detected segments.
    Uses k-means-style cluster assignment: each segment is compared to all
    existing cluster centroids and joins the nearest one if similarity
    exceeds an adaptive threshold; otherwise starts a new cluster.
    Cluster centroids are updated as a running mean.
    """
    # Build segment centroids
    segments = []
    prev = 0
    for b in boundaries + [n_bars]:
        if b > prev:
            bar_vecs  = vecs[prev:b]
            n_feats   = len(bar_vecs[0])
            centroid  = [sum(x[i] for x in bar_vecs) / len(bar_vecs)
                         for i in range(n_feats)]
            segments.append({'start': prev, 'end': b, 'cent': centroid,
                              'size': b - prev})
        prev = b

    n_segs = len(segments)
    if n_segs == 0:
        return ['A'] * n_bars
    if n_segs == 1:
        return ['A'] * n_bars

    # Compute all pairwise similarities between segment centroids
    all_sims = []
    for i in range(n_segs):
        for j in range(i+1, n_segs):
            all_sims.append(combined_sim(segments[i]['cent'], segments[j]['cent']))
    all_sims.sort()

    # Adaptive threshold: find the largest gap in the similarity distribution.
    # Segments above the gap threshold are considered "same section".
    # Fallback: use the median as threshold.
    sim_threshold = 0.70   # conservative default
    if len(all_sims) >= 4:
        # Find the largest gap between consecutive sorted similarities
        gaps = [(all_sims[i+1] - all_sims[i], i) for i in range(len(all_sims)-1)]
        gap_val, gap_idx = max(gaps, key=lambda x: x[0])
        if gap_val > 0.05:  # meaningful gap exists
            sim_threshold = all_sims[gap_idx] + gap_val * 0.5
        else:
            # No clear gap: use a percentile-based threshold
            pct_idx = int(len(all_sims) * 0.35)
            sim_threshold = all_sims[pct_idx]
    elif len(all_sims) >= 2:
        sim_threshold = (all_sims[0] + all_sims[-1]) / 2.0

    sim_threshold = max(0.50, min(0.92, sim_threshold))

    # Cluster centroids accumulator: list of {label, cent, count}
    clusters: List[Dict] = []
    labels   = ['?'] * n_segs
    letter_i = 0

    for si, seg in enumerate(segments):
        cent = seg['cent']
        best_sim = sim_threshold
        best_ci  = -1
        for ci, cl in enumerate(clusters):
            s = combined_sim(cent, cl['cent'])
            if s > best_sim:
                best_sim = s
                best_ci  = ci

        if best_ci >= 0:
            # Join existing cluster, update centroid (online mean)
            cl    = clusters[best_ci]
            count = cl['count']
            cl['cent'] = [(cl['cent'][i]*count + cent[i]) / (count+1)
                          for i in range(len(cent))]
            cl['count'] += 1
            labels[si] = cl['label']
        else:
            lbl = chr(ord('A') + letter_i)
            letter_i += 1
            clusters.append({'label': lbl, 'cent': list(cent), 'count': 1})
            labels[si] = lbl

    # Expand labels to per-bar
    bar_labels = ['A'] * n_bars
    for si, seg in enumerate(segments):
        for b in range(seg['start'], seg['end']):
            if b < n_bars:
                bar_labels[b] = labels[si]
    return bar_labels

def compute_global_sections(tracks: List[Dict], n_bars: int, seg_cfg: Dict = None) -> Dict:
    if seg_cfg is None:
        seg_cfg = {}
    if n_bars < 2:
        return {'bar_labels': ['A'] * n_bars, 'boundaries': [], 'n_bars': n_bars}

    vecs = build_bar_vectors(tracks, n_bars)

    # Optional per-group feature weighting.
    # Groups: pitch_class(0-11) density(12) velocity(13-14) pitch_stats(15-17)
    #         duration(18-19) rhythm(20-21) texture(22-23) contour(24)
    #         harmony(25) register(26-27) complexity(28-29)
    group_slices = {
        'pitch_class': range(0, 12), 'density':     range(12, 13),
        'velocity':    range(13, 15), 'pitch_stats': range(15, 18),
        'duration':    range(18, 20), 'rhythm':      range(20, 22),
        'texture':     range(22, 24), 'contour':     range(24, 25),
        'harmony':     range(25, 26), 'register':    range(26, 28),
        'complexity':  range(28, 30),
    }
    for group, w in (seg_cfg.get('feature_weights') or {}).items():
        sl = group_slices.get(group)
        if sl:
            for bi in range(n_bars):
                for fi in sl:
                    vecs[bi][fi] *= float(w)

    ssm = ssm_matrix(vecs)
    k         = int(seg_cfg.get('kernel',   min(4, max(2, n_bars // 8))))
    nov       = novelty_curve(ssm, kernel_size=k)
    mean_nov  = sum(nov) / len(nov) if nov else 0
    threshold = float(seg_cfg.get('threshold', max(0.35, mean_nov + 0.15)))
    min_gap   = int(seg_cfg.get('min_gap',   max(2, n_bars // 10)))
    boundaries = pick_peaks(nov, min_gap=min_gap, threshold=threshold)
    bar_labels = assign_section_labels(boundaries, n_bars, vecs)
    return {
        'bar_labels': bar_labels,
        'boundaries': boundaries,
        'n_bars':     n_bars,
        'novelty':    nov,
        'ssm':        ssm,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CHORD DETECTION + HARMONIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'dom7': [0, 4, 7, 10],
    'min7': [0, 3, 7, 10],
}
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Harmonic function colors (dark-theme friendly)
HARM_FUNC_COLORS = {
    'T':    '#3b82f6',   # tónica       — azul
    'PD':   '#10b981',   # predominante — verde
    'D':    '#ef4444',   # dominante    — rojo
    'Dsec': '#f59e0b',   # dom. secundaria — ámbar
    'Other':'#6b7280',   # otro         — gris
    '—':    '#1e293b',   # silencio     — casi negro
}
HARM_FUNC_LABELS = {
    'T':    'T — Tónica',
    'PD':   'PD — Predominante',
    'D':    'D — Dominante',
    'Dsec': 'Dsec — Dom. secundaria',
    'Other':'Otro',
    '—':    'Silencio',
}

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
_KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

def detect_key(tracks: List[Dict]) -> Tuple[int, str]:
    """
    Krumhansl-Schmuckler key detection over all notes.
    Returns (tonic_pc, mode) where mode is 'major' or 'minor'.
    """
    pc_hist = [0.0] * 12
    for t in tracks:
        for n in t['notes']:
            pc_hist[n['pitch'] % 12] += n['vel']
    total = sum(pc_hist) or 1.0
    pc_norm = [v / total for v in pc_hist]

    best_score = -999.0
    best_tonic = 0
    best_mode  = 'major'

    for tonic in range(12):
        for mode, profile in [('major', _KS_MAJOR), ('minor', _KS_MINOR)]:
            rotated = [profile[(i - tonic) % 12] for i in range(12)]
            # Pearson correlation
            m1 = sum(pc_norm) / 12
            m2 = sum(rotated) / 12
            num = sum((pc_norm[i]-m1)*(rotated[i]-m2) for i in range(12))
            d1  = math.sqrt(sum((pc_norm[i]-m1)**2 for i in range(12)))
            d2  = math.sqrt(sum((rotated[i]-m2)**2 for i in range(12)))
            r   = num / (d1 * d2 + 1e-9)
            if r > best_score:
                best_score = r
                best_tonic = tonic
                best_mode  = mode

    return best_tonic, best_mode

def chord_root_from_name(chord_name: str) -> Optional[int]:
    """Extract root pitch class from chord name like 'Cmaj', 'F#min7'."""
    if chord_name in ('—', '?'):
        return None
    for length in (2, 1):
        candidate = chord_name[:length]
        if candidate in NOTE_NAMES:
            return NOTE_NAMES.index(candidate)
    return None

def chord_type_from_name(chord_name: str) -> str:
    """Extract chord type suffix from chord name."""
    for length in (2, 1):
        candidate = chord_name[:length]
        if candidate in NOTE_NAMES:
            return chord_name[length:]
    return ''

def harmonic_function(chord_name: str, tonic: int, mode: str) -> str:
    """
    Classify a chord into T / PD / D / Dsec / Other.
    Logic adapted from html_report.py _chord_to_function().
    """
    if chord_name in ('—', '?', ''):
        return '—'
    root = chord_root_from_name(chord_name)
    if root is None:
        return 'Other'
    ctype    = chord_type_from_name(chord_name)
    interval = (root - tonic) % 12
    is_minor = mode == 'minor'

    # Tonic function
    if interval == 0:
        return 'T'
    if interval == 9 and not is_minor:   # vi in major
        return 'T'
    if interval == 3 and is_minor:       # III in minor
        return 'T'

    # Dominant function
    if interval == 7:                    # V
        return 'D'
    if interval == 11:                   # viidim
        return 'D'

    # Secondary dominant: dom7 chord whose root is not the tonic or V
    if 'dom7' in ctype and interval not in (0, 7):
        return 'Dsec'

    # Predominant function
    if interval in (2, 5):              # ii, IV
        return 'PD'
    if interval == 9 and is_minor:      # VI in minor
        return 'PD'

    return 'Other'

def detect_chord_for_bar(notes_in_bar: List[Dict]) -> str:
    if not notes_in_bar:
        return '—'
    pcs = list({n['pitch'] % 12 for n in notes_in_bar})
    if len(pcs) < 2:
        return NOTE_NAMES[pcs[0]]

    best_name  = '?'
    best_score = -1

    for root in range(12):
        for ctype, template in CHORD_TEMPLATES.items():
            chord_pcs = set((root + i) % 12 for i in template)
            pcs_set   = set(pcs)
            overlap   = len(chord_pcs & pcs_set)
            penalty   = len(pcs_set - chord_pcs) * 0.3
            score     = overlap - penalty
            if score > best_score:
                best_score = score
                best_name  = f'{NOTE_NAMES[root]}{ctype}'

    return best_name

def compute_chords_per_bar(harmony_track: Dict, n_bars: int,
                            tonic: int, mode: str) -> List[Dict]:
    """
    Returns list of dicts per bar:
      {label, function, color}
    """
    bar_notes: Dict[int, List[Dict]] = defaultdict(list)
    for n in harmony_track['notes']:
        bar_notes[n['start_bar']].append(n)

    result = []
    for b in range(n_bars):
        label = detect_chord_for_bar(bar_notes.get(b, []))
        func  = harmonic_function(label, tonic, mode)
        color = HARM_FUNC_COLORS.get(func, HARM_FUNC_COLORS['Other'])
        result.append({'label': label, 'function': func, 'color': color})
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# PER-BAR METRICS
# ═══════════════════════════════════════════════════════════════════════════════

INTERVAL_TENSION = {0:0.0,1:0.9,2:0.5,3:0.3,4:0.2,5:0.2,6:1.0,7:0.1,8:0.5,9:0.3,10:0.7,11:0.8}

def compute_bar_metrics(tracks: List[Dict], n_bars: int) -> Dict[str, List[float]]:
    """Returns dict of metric_name -> [value_per_bar]."""
    tension    = [0.0] * n_bars
    density    = [0.0] * n_bars
    avg_vel    = [0.0] * n_bars
    contour    = [0.0] * n_bars  # avg pitch change direction

    bar_notes_all: Dict[int, List[Dict]] = defaultdict(list)
    for t in tracks:
        for n in t['notes']:
            bar_notes_all[n['start_bar']].append(n)

    for b in range(n_bars):
        bn = bar_notes_all.get(b, [])
        if not bn:
            continue
        density[b] = min(1.0, len(bn) / 16.0)
        avg_vel[b] = sum(n['vel'] for n in bn) / len(bn) / 127.0

        # harmonic tension: mean interval tension of simultaneous pitches
        pitches = sorted({n['pitch'] % 12 for n in bn})
        t_vals = []
        for i in range(len(pitches)):
            for j in range(i+1, len(pitches)):
                iv = (pitches[j] - pitches[i]) % 12
                t_vals.append(INTERVAL_TENSION.get(iv, 0.5))
        tension[b] = sum(t_vals) / len(t_vals) if t_vals else 0.0

        # contour: average pitch difference between consecutive notes
        sorted_bn = sorted(bn, key=lambda n: n['start_tick'])
        diffs = [sorted_bn[i+1]['pitch'] - sorted_bn[i]['pitch']
                 for i in range(len(sorted_bn)-1)]
        if diffs:
            avg_d = sum(diffs) / len(diffs)
            contour[b] = max(-1.0, min(1.0, avg_d / 12.0))

    return {
        'tension': tension,
        'density': density,
        'velocity': avg_vel,
        'contour': contour,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTES
# ═══════════════════════════════════════════════════════════════════════════════

def phrase_palette(n_phrases: int, base_hue: float = 0.6) -> List[str]:
    """Generate n visually distinct colors for phrases."""
    colors = []
    for i in range(n_phrases):
        hue = (base_hue + i * 0.618033988) % 1.0  # golden ratio
        sat = 0.65 + (i % 3) * 0.1
        val = 0.75 + (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors

SECTION_COLORS = {
    'A': '#3b82f6', 'B': '#ef4444', 'C': '#10b981', 'D': '#f59e0b',
    'E': '#8b5cf6', 'F': '#ec4899', 'G': '#06b6d4', 'H': '#84cc16',
    '?': '#94a3b8',
}

# ═══════════════════════════════════════════════════════════════════════════════
# HTML RENDER
# ═══════════════════════════════════════════════════════════════════════════════



METRIC_COLORS = {
    'tension':  '#ef4444',
    'density':  '#3b82f6',
    'velocity': '#10b981',
    'contour':  '#f59e0b',
}

def build_html(mid_path: str, tracks: List[Dict], phrase_ids_per_track: List[List[int]],
               sections: Dict, chords_per_bar: Optional[List[str]],
               metrics: Dict, cfg: dict) -> str:

    n_bars     = sections['n_bars']
    bar_labels = sections['bar_labels']
    bpb        = tracks[0]['bpb'] if tracks else 4.0

    all_beats   = [n['end_beat'] for t in tracks for n in t['notes']]
    total_beats = max(all_beats) if all_beats else (n_bars * bpb)

    CANVAS_W      = 1600   # logical canvas width in pixels
    ROW_H_TRACK   = 88
    ROW_H_CHORD   = 40
    ROW_H_SECTION = 32
    ROW_H_METRICS = 110
    LABEL_W       = 190

    # ── Build per-track note arrays ────────────────────────────────────────
    all_track_notes = []   # list of lists of note dicts (plain Python)
    for ti, t in enumerate(tracks):
        pids    = phrase_ids_per_track[ti] if ti < len(phrase_ids_per_track) else []
        notes   = t['notes']
        pitches = [n['pitch'] for n in notes]
        if not pitches:
            all_track_notes.append([])
            continue
        p_min   = max(0,   min(pitches) - 2)
        p_max   = min(127, max(pitches) + 2)
        p_range = max(1, p_max - p_min)

        unique_pids = sorted(set(pids)) if pids else [0]
        pid_colors  = {}
        palette     = phrase_palette(len(unique_pids))
        for i, pid in enumerate(unique_pids):
            pid_colors[pid] = palette[i % len(palette)]

        rects = []
        for ni, n in enumerate(notes):
            pid    = pids[ni] if ni < len(pids) else 0
            col    = pid_colors.get(pid, '#94a3b8')
            x      = round(n['start_beat'] / total_beats * CANVAS_W, 2)
            w      = round(max(2.0, n['dur_beat'] / total_beats * CANVAS_W), 2)
            # y: higher pitch → smaller y (top of canvas)
            frac   = (n['pitch'] - p_min) / p_range   # 0=lowest, 1=highest
            note_h = max(2.0, round(ROW_H_TRACK / p_range * 0.9, 2))
            y      = round((1.0 - frac) * (ROW_H_TRACK - note_h), 2)
            rects.append({'x': x, 'y': y, 'w': w, 'h': note_h, 'c': col})
        all_track_notes.append(rects)

    # ── Chord data (one item per bar) ──────────────────────────────────────
    chord_items = []
    if chords_per_bar:
        bar_w = CANVAS_W / max(n_bars, 1)
        for bi, chord in enumerate(chords_per_bar):
            chord_items.append({
                'x':     round(bi * bar_w, 2),
                'w':     round(bar_w, 2),
                'label': chord['label'],
                'func':  chord['function'],
                'color': chord['color'],
                'even':  bi % 2 == 0,
            })

    # ── Section spans ──────────────────────────────────────────────────────
    section_spans = []
    if bar_labels:
        bar_w = CANVAS_W / max(n_bars, 1)
        i = 0
        while i < len(bar_labels):
            label = bar_labels[i]
            j = i
            while j < len(bar_labels) and bar_labels[j] == label:
                j += 1
            section_spans.append({
                'x':     round(i * bar_w, 2),
                'w':     round((j - i) * bar_w, 2),
                'label': label,
                'color': SECTION_COLORS.get(label, '#94a3b8'),
            })
            i = j

    # ── Metrics series ─────────────────────────────────────────────────────
    metrics_series = {}
    for name, vals in metrics.items():
        metrics_series[name] = [round(v, 4) for v in vals]

    # ── Track metadata ─────────────────────────────────────────────────────
    track_meta = []
    for t in tracks:
        track_meta.append({
            'name':       t['name'],
            'role_label': ROLE_LABELS.get(t['role'], '') if t['role'] else '',
            'role_color': ROLE_COLORS.get(t['role'], '#64748b'),
        })

    # ── Bar ruler ticks ────────────────────────────────────────────────────
    bar_ticks_html = ''
    tick_every = max(1, n_bars // 32)
    bar_w_pct  = 100.0 / max(n_bars, 1)
    for bi in range(0, n_bars, tick_every):
        x_pct = bi * bar_w_pct
        bar_ticks_html += (
            f'<div class="bar-tick" style="left:{x_pct:.3f}%">'
            f'{bi+1}</div>'
        )

    # ── Build HTML rows (canvas elements only — JS does the drawing) ────────
    rows_html = ''
    for ti, tm in enumerate(track_meta):
        rows_html += f"""
<div class="tl-row" style="height:{ROW_H_TRACK}px">
  <div class="tl-label" style="border-left:3px solid {tm['role_color']}">
    <div class="trk-name">{tm['name']}</div>
    <div class="trk-role" style="color:{tm['role_color']}">{tm['role_label']}</div>
  </div>
  <div class="tl-canvas-wrap">
    <canvas id="cv_track_{ti}" width="{CANVAS_W}" height="{ROW_H_TRACK}"></canvas>
  </div>
</div>"""

    if chord_items:
        rows_html += f"""
<div class="tl-row" style="height:{ROW_H_CHORD}px">
  <div class="tl-label" style="border-left:3px solid #f59e0b">
    <div class="trk-name">Acordes</div>
    <div class="trk-role" style="color:#f59e0b">♬ Armonía detectada</div>
  </div>
  <div class="tl-canvas-wrap">
    <canvas id="cv_chords" width="{CANVAS_W}" height="{ROW_H_CHORD}"></canvas>
  </div>
</div>"""

    rows_html += f"""
<div class="tl-row" style="height:{ROW_H_SECTION}px">
  <div class="tl-label" style="border-left:3px solid #6366f1">
    <div class="trk-name">Sección</div>
    <div class="trk-role" style="color:#6366f1">Forma global</div>
  </div>
  <div class="tl-canvas-wrap">
    <canvas id="cv_sections" width="{CANVAS_W}" height="{ROW_H_SECTION}"></canvas>
  </div>
</div>
<div class="tl-row" style="height:{ROW_H_METRICS}px">
  <div class="tl-label" style="border-left:3px solid #94a3b8">
    <div class="trk-name">Métricas</div>
    <div class="trk-role" style="color:#94a3b8">por compás</div>
  </div>
  <div class="tl-canvas-wrap">
    <canvas id="cv_metrics" width="{CANVAS_W}" height="{ROW_H_METRICS}"></canvas>
  </div>
</div>"""

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_html = '<span style="font-size:11px;color:#475569;font-weight:600;margin-right:6px">MÉTRICAS:</span>'
    for key, col in METRIC_COLORS.items():
        legend_html += (f'<div class="legend-item">'
                        f'<div class="legend-dot" style="background:{col}"></div>{key}</div>')
    legend_html += '<span style="font-size:11px;color:#475569;font-weight:600;margin:0 6px 0 16px">FUNCIÓN ARMÓNICA:</span>'
    for func, col in HARM_FUNC_COLORS.items():
        if func == '—':
            continue
        label = HARM_FUNC_LABELS.get(func, func)
        legend_html += (f'<div class="legend-item">'
                        f'<div class="legend-dot" style="background:{col}"></div>{label}</div>')
    legend_html += '<span style="font-size:11px;color:#475569;font-weight:600;margin:0 6px 0 16px">SECCIONES:</span>'
    seen_sections = {s['label'] for s in section_spans}
    for lbl, col in SECTION_COLORS.items():
        if lbl in seen_sections:
            legend_html += (f'<div class="legend-item">'
                            f'<div class="legend-dot" style="background:{col}"></div>{lbl}</div>')

    # ── Inline JS — all data as plain JS literals, NO JSON.parse ───────────
    title = os.path.basename(mid_path)

    js = f"""
(function() {{
  // ── constants ──────────────────────────────────────────────────────────
  var CW    = {CANVAS_W};
  var NBARS = {n_bars};

  // ── helper: draw vertical bar grid ────────────────────────────────────
  function drawGrid(ctx, h) {{
    var bw = CW / NBARS;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    for (var b = 0; b <= NBARS; b++) {{
      ctx.beginPath(); ctx.moveTo(b*bw, 0); ctx.lineTo(b*bw, h); ctx.stroke();
    }}
  }}

  // ── PIANO ROLLS ────────────────────────────────────────────────────────
  var allNotes = {json.dumps(all_track_notes)};
  allNotes.forEach(function(rects, ti) {{
    var cv = document.getElementById('cv_track_' + ti);
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#0b1628';
    ctx.fillRect(0, 0, CW, H);
    drawGrid(ctx, H);
    rects.forEach(function(r) {{
      ctx.fillStyle = r.c;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }});
  }});

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
      // background tinted by harmonic function color
      ctx.fillStyle = c.color + '28';
      ctx.fillRect(c.x, 0, c.w, H);
      // top bar stripe in function color
      ctx.fillStyle = c.color;
      ctx.fillRect(c.x, 0, c.w, 3);
      // chord label
      if (c.label && c.label !== '\u2014') {{
        ctx.fillStyle = c.color;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(c.label, c.x + c.w / 2, H / 2 + 1);
      }}
      // function badge (small, bottom)
      if (c.func && c.func !== '\u2014' && c.w > 28) {{
        ctx.fillStyle = c.color + 'aa';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(c.func, c.x + c.w / 2, H - 2);
      }}
      // separator
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c.x + 0.5, 0.5, c.w - 1, H - 1);
    }});
  }})();

  // ── SECTIONS ───────────────────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_sections');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#060d1c';
    ctx.fillRect(0, 0, CW, H);
    var spans = {json.dumps(section_spans)};
    spans.forEach(function(s) {{
      // filled band
      ctx.fillStyle = s.color + '40';
      ctx.fillRect(s.x, 0, s.w, H);
      // left border
      ctx.fillStyle = s.color;
      ctx.fillRect(s.x, 0, 2, H);
      // label
      ctx.fillStyle = s.color;
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (s.w > 20) {{
        ctx.fillText(s.label, s.x + Math.min(s.w / 2, 40), H / 2);
      }}
    }});
  }})();

  // ── METRICS ────────────────────────────────────────────────────────────
  (function() {{
    var cv = document.getElementById('cv_metrics');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var H = cv.height;
    ctx.fillStyle = '#060c1a';
    ctx.fillRect(0, 0, CW, H);
    drawGrid(ctx, H);

    var series  = {json.dumps(metrics_series)};
    var colors  = {json.dumps(METRIC_COLORS)};
    var PAD     = 6;

    Object.keys(series).forEach(function(name) {{
      var vals = series[name];
      var col  = colors[name] || '#94a3b8';
      var n    = vals.length;
      var bw   = CW / Math.max(n, 1);

      // find min/max for this series to normalize independently
      var vmin = Infinity, vmax = -Infinity;
      vals.forEach(function(v) {{ if (v < vmin) vmin = v; if (v > vmax) vmax = v; }});
      var vrange = vmax - vmin || 1;

      function vy(v) {{
        return H - PAD - ((v - vmin) / vrange) * (H - PAD * 2);
      }}

      // area fill
      ctx.beginPath();
      ctx.moveTo(0, H);
      vals.forEach(function(v, bi) {{
        var x = bi * bw + bw / 2;
        ctx.lineTo(x, vy(v));
      }});
      ctx.lineTo((n - 1) * bw + bw / 2, H);
      ctx.closePath();
      ctx.fillStyle = col + '20';
      ctx.fill();

      // line
      ctx.beginPath();
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      vals.forEach(function(v, bi) {{
        var x = bi * bw + bw / 2;
        if (bi === 0) ctx.moveTo(x, vy(v));
        else          ctx.lineTo(x, vy(v));
      }});
      ctx.stroke();
    }});
  }})();

}})();
"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Piano Roll — {title}</title>
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
.trk-name {{ font-size: 11px; font-weight: 700; color: #dde6f0;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.trk-role {{ font-size: 9px; margin-top: 2px; opacity: 0.8; }}
.tl-canvas-wrap {{ flex: 1; overflow: hidden; }}
canvas {{ display: block; }}
.legend-row {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 10px 18px;
               border-top: 1px solid #1a2744; background: #050c18; align-items: center; }}
.legend-item {{ display: flex; align-items: center; gap: 5px;
                font-size: 10px; color: #64748b; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>🎵 {title}</h1>
  <div class="meta">{n_bars} compases · {bpb:.0f}/4 · {len(tracks)} pistas
    · Re-Pair/Sequitur + SSM segmentación global</div>
</div>

<div class="scroll-wrap">
<div class="tl-outer">

  <div class="ruler-row">
    <div class="ruler-label"></div>
    <div class="ruler-track">
      {bar_ticks_html}
    </div>
  </div>

  {rows_html}

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
# SYNTHETIC MIDI GENERATOR (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_midi(path: str):
    """
    Generate a multi-track MIDI for testing.
    Structure: A (bars 1-8, C major) | B (bars 9-16, F major, higher register) |
               C (bars 17-20, sparse/transition) | A' (bars 21-28, C major reprise)
    5 tracks: Melody, Counter-melody, Harmony, Bass, Accompaniment
    """
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    TPB = 480

    def make_track(name: str, channel: int, notes_seq, add_meta=False, bpm=120) -> mido.MidiTrack:
        t = mido.MidiTrack()
        t.name = name
        t.append(mido.MetaMessage('track_name', name=name, time=0))
        if add_meta:
            t.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
            t.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                      clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        events = []
        for pitch, vel, start_beat, dur_beat in notes_seq:
            s = int(start_beat * TPB)
            e = int((start_beat + dur_beat) * TPB)
            events.append((s, 'on',  pitch, vel,   channel))
            events.append((e, 'off', pitch, 0,     channel))
        events.sort(key=lambda ev: (ev[0], 0 if ev[1]=='off' else 1))
        prev = 0
        for tick, kind, note, vel, ch in events:
            delta = tick - prev; prev = tick
            if kind == 'on':
                t.append(mido.Message('note_on',  note=note, velocity=vel,  channel=ch, time=delta))
            else:
                t.append(mido.Message('note_off', note=note, velocity=0,    channel=ch, time=delta))
        t.append(mido.MetaMessage('end_of_track', time=0))
        return t

    def chord_block(root, ivs, start, dur, vel):
        return [(root+i, vel, start, dur) for i in ivs]

    # ── Section A: bars 0-7 (beats 0-31), C major, moderate ──────────────
    # Melody: stepwise C major melody with 2-bar motif repeated
    mel_A = []
    motif1 = [(72,88,0,.5),(74,85,.5,.5),(76,90,1,.5),(77,85,1.5,.5),
               (79,92,2,1),(77,82,3,.5),(76,82,3.5,.5)]  # 4 beats
    motif2 = [(72,88,0,.5),(74,85,.5,.5),(76,90,1,1),(74,82,2,1),(72,80,3,1)]
    for rep in range(2):  # 2×4 bars = 8 bars
        for off in range(2):
            base = rep * 16 + off * 8
            for p,v,t,d in (motif1 if off==0 else motif2):
                mel_A.append((p,v,t+base,d))
    # Add an extra melodic phrase to cap section A
    mel_A += [(76,90,24,1),(77,85,25,1),(79,88,26,1),(81,92,27,1),(79,85,28,2),(77,80,30,2)]

    # ── Section B: bars 8-15 (beats 32-63), F major, higher ──────────────
    mel_B = []
    motif_B1 = [(84,95,0,.5),(83,90,.5,.5),(81,92,1,1),(79,88,2,1),(77,82,3,1)]
    motif_B2 = [(77,85,0,.5),(79,88,.5,.5),(81,90,1,.5),(83,92,1.5,.5),
                 (84,95,2,1.5),(83,82,3.5,.5)]
    for rep in range(2):
        for off in range(2):
            base = 32 + rep*16 + off*8
            for p,v,t,d in (motif_B1 if off==0 else motif_B2):
                mel_B.append((p,v,t+base,d))

    # ── Section C: bars 16-19 (beats 64-79), sparse/transition ───────────
    mel_C = [
        (72,70,64,2),(76,68,66,2),(79,72,68,2),(76,65,70,2),
        (74,70,72,1.5),(72,65,73.5,.5),(71,60,74,4),
    ]

    # ── Section A': bars 20-27 (beats 80-111), reprise of A ──────────────
    mel_A2 = [(p,v,t+80,d) for p,v,t,d in
              [(72,88,0,.5),(74,85,.5,.5),(76,90,1,.5),(77,85,1.5,.5),
               (79,92,2,1),(77,82,3,.5),(76,82,3.5,.5),
               (72,88,8,.5),(74,85,8.5,.5),(76,90,9,1),(74,82,10,1),(72,80,11,1),
               (76,88,16,1),(77,85,17,1),(79,90,18,1),(81,95,19,1),(79,88,20,2),(72,75,22,2),
               (72,92,24,1),(74,90,25,1),(76,92,26,1),(77,95,27,1),(79,88,28,4)]]

    melody_notes = mel_A + mel_B + mel_C + mel_A2

    # ── Counter-melody ────────────────────────────────────────────────────
    counter = []
    # A: slower moving, lower
    for bi in range(8):
        base = bi * 4
        p = [65,67,69,67, 65,67,69,68][bi]
        counter += [(p,70,base,2),(p+2,68,base+2,2)]
    # B: syncopated, higher
    for bi in range(8):
        base = 32 + bi * 4
        p = [77,79,81,79, 77,79,81,79][bi]
        counter += [(p,75,base,1.5),(p,70,base+2,1.5),(p-2,65,base+3.5,.5)]
    # C: long sustained notes
    for bi in range(4):
        counter += [(60+bi*2, 58, 64+bi*4, 4)]
    # A': like A but an octave up
    for bi in range(8):
        base = 80 + bi * 4
        p = [65,67,69,67, 65,67,69,68][bi]
        counter += [(p+12,72,base,2),(p+14,68,base+2,2)]

    # ── Harmony chords ─────────────────────────────────────────────────────
    harmony = []
    # A: C-Am-F-G progression (2 beats each)
    prog_A = [(60,[0,4,7],70),(57,[0,3,7],65),(65,[0,4,7],65),(67,[0,4,7],68)]
    for rep in range(4):
        for ci,(root,ivs,vel) in enumerate(prog_A):
            harmony += chord_block(root,ivs, rep*16 + ci*4, 3.8, vel)
    # B: F-Dm-Bb-C (F major feel)
    prog_B = [(65,[0,4,7],72),(62,[0,3,7],68),(58,[0,4,7],65),(60,[0,4,7],70)]
    for rep in range(4):
        for ci,(root,ivs,vel) in enumerate(prog_B):
            harmony += chord_block(root,ivs, 32 + rep*16 + ci*4, 3.8, vel)
    # C: sparse, dim/sus chords
    prog_C = [(60,[0,3,6],55),(65,[0,5,7],52),(67,[0,4,7],55),(62,[0,3,7],50)]
    for ci,(root,ivs,vel) in enumerate(prog_C):
        harmony += chord_block(root,ivs, 64 + ci*4, 3.8, vel)
    # A': same as A
    for rep in range(4):
        for ci,(root,ivs,vel) in enumerate(prog_A):
            harmony += chord_block(root,ivs, 80 + rep*16 + ci*4, 3.8, vel)

    # ── Bass ──────────────────────────────────────────────────────────────
    bass_A_roots = [48,45,53,55]  # C-Am-F-G (low)
    bass_B_roots = [53,50,46,48]  # F-Dm-Bb-C
    bass_C_roots = [48,53,55,50]
    bass = []
    for rep in range(4):
        for ci,root in enumerate(bass_A_roots):
            bass += [(root,85,rep*16+ci*4,.9),(root,70,rep*16+ci*4+2,.9)]
    for rep in range(4):
        for ci,root in enumerate(bass_B_roots):
            bass += [(root,85,32+rep*16+ci*4,.9),(root,70,32+rep*16+ci*4+2,.9)]
    for ci,root in enumerate(bass_C_roots):
        bass += [(root,65,64+ci*4,3.8)]
    for rep in range(4):
        for ci,root in enumerate(bass_A_roots):
            bass += [(root,85,80+rep*16+ci*4,.9),(root,70,80+rep*16+ci*4+2,.9)]

    # ── Accompaniment: 8th-note arpeggios ────────────────────────────────
    acc = []
    # A and A': C major arpeggio pattern
    arp_A  = [48, 52, 55, 60, 55, 52]
    arp_B  = [53, 57, 60, 65, 60, 57]  # F major
    arp_C  = [48, 51, 55, 60, 55, 51]  # C dim
    for beat in range(32):  # A: 32 beats
        for si, p in enumerate(arp_A):
            acc.append((p, 52, beat + si*(2/3), 0.55))
    for beat in range(32):  # B
        p_arr = arp_B
        for si, p in enumerate(p_arr):
            acc.append((p, 55, 32 + beat + si*(2/3), 0.55))
    for beat in range(16):  # C: sparser
        if beat % 4 == 0:
            for si, p in enumerate(arp_C[:3]):
                acc.append((p, 45, 64 + beat + si, 0.9))
    for beat in range(32):  # A'
        for si, p in enumerate(arp_A):
            acc.append((p, 54, 80 + beat + si*(2/3), 0.55))

    mid.tracks.append(make_track("Violín",     0, melody_notes, add_meta=True, bpm=108))
    mid.tracks.append(make_track("Viola",      1, counter,      add_meta=False))
    mid.tracks.append(make_track("Piano der.", 2, harmony,      add_meta=False))
    mid.tracks.append(make_track("Contrabajo", 3, bass,         add_meta=False))
    mid.tracks.append(make_track("Piano izq.", 4, acc,          add_meta=False))

    mid.save(path)
    print(f"✓ MIDI sintético guardado: {path}  (estructura A-B-C-A', 5 pistas)")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(mid_path: str, cfg: dict) -> str:
    """Full analysis pipeline. Returns HTML string."""
    print(f"  Cargando MIDI: {mid_path}")
    mid    = load_midi(mid_path)
    tracks = extract_tracks(mid)
    print(f"  Pistas encontradas: {len(tracks)} con notas")

    tracks = apply_config(tracks, cfg)

    if not tracks:
        return "<html><body>No se encontraron pistas con notas.</body></html>"

    # n_bars
    all_bars = [n['start_bar'] for t in tracks for n in t['notes']]
    n_bars   = (max(all_bars) + 2) if all_bars else 8

    # ── Sequitur per track ────────────────────────────────────────────────
    print("  Ejecutando Sequitur por pista...")
    seq_global_cfg = cfg.get('algorithms', {}).get('sequitur', {})
    track_cfgs     = cfg.get('tracks', {})
    phrase_ids_per_track = []
    for t in tracks:
        # Per-track sequitur config overrides global
        tc      = track_cfgs.get(t['index'], track_cfgs.get(str(t['index']), {}))
        seq_cfg = {**seq_global_cfg, **tc.get('sequitur', {})} if isinstance(tc, dict) else seq_global_cfg
        pids    = assign_phrase_ids(t['notes'], seq_cfg)
        phrase_ids_per_track.append(pids)
        n_phrases = len(set(pids))
        target    = seq_cfg.get('n_phrases', '—')
        print(f"    [{t['name']}] {len(t['notes'])} notas → {n_phrases} frases"
              + (f" (objetivo {target})" if target != '—' else ""))

    # ── Global segmentation ────────────────────────────────────────────────
    print("  Ejecutando segmentación global...")
    seg_cfg  = cfg.get('algorithms', {}).get('segmentation', {})
    sections = compute_global_sections(tracks, n_bars, seg_cfg)
    unique_secs = len(set(sections['bar_labels']))
    print(f"    {n_bars} compases → {unique_secs} secciones: {' '.join(sorted(set(sections['bar_labels'])))}")

    # ── Key detection ──────────────────────────────────────────────────────
    tonic, mode = detect_key(tracks)
    print(f"  Tonalidad detectada: {NOTE_NAMES[tonic]} {mode}")

    # ── Chord detection ────────────────────────────────────────────────────
    chords_per_bar = None
    harmony_track  = next((t for t in tracks if t['role'] == 'harmony'), None)
    if harmony_track:
        print("  Detectando acordes y funciones armónicas...")
        chords_per_bar = compute_chords_per_bar(harmony_track, n_bars, tonic, mode)

    # ── Metrics ────────────────────────────────────────────────────────────
    print("  Calculando métricas por compás...")
    metrics = compute_bar_metrics(tracks, n_bars)

    # ── Render HTML ─────────────────────────────────────────────────────────
    print("  Generando HTML...")
    html = build_html(mid_path, tracks, phrase_ids_per_track,
                      sections, chords_per_bar, metrics, cfg)
    return html


def main():
    parser = argparse.ArgumentParser(description='MIDI multi-track analyzer + HTML visualizer')
    parser.add_argument('midi', nargs='?', help='Archivo MIDI de entrada')
    parser.add_argument('config', nargs='?', help='Config YAML/JSON (opcional)')
    parser.add_argument('-o', '--output', default=None, help='Salida HTML (default: <midi>.html)')
    parser.add_argument('--test', action='store_true', help='Generar MIDI sintético y analizar')
    args = parser.parse_args()

    if args.test or args.midi is None:
        test_midi = '/home/claude/test_multitrack.mid'
        generate_test_midi(test_midi)
        mid_path = test_midi
    else:
        mid_path = args.midi

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    out_path = args.output or (os.path.splitext(mid_path)[0] + '_report.html')

    print(f"\n{'═'*50}")
    print(f"  MIDI Piano Roll Analyzer")
    print(f"{'═'*50}")
    html = analyze(mid_path, cfg)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  ✓ Informe HTML generado: {out_path}")
    print(f"{'═'*50}\n")


if __name__ == '__main__':
    main()

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH TEST MIDI GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_track(name, channel, notes_seq, add_meta=False, bpm=120, tpb=480):
    import mido
    t = mido.MidiTrack()
    t.name = name
    t.append(mido.MetaMessage('track_name', name=name, time=0))
    if add_meta:
        t.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        t.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                  clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    events = []
    for pitch, vel, start_beat, dur_beat in notes_seq:
        s = int(start_beat * tpb)
        e = int((start_beat + dur_beat) * tpb)
        events.append((s, 'on',  pitch, vel, channel))
        events.append((e, 'off', pitch, 0,   channel))
    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 'off' else 1))
    prev = 0
    for tick, kind, note, vel, ch in events:
        delta = tick - prev; prev = tick
        if kind == 'on':
            t.append(mido.Message('note_on',  note=note, velocity=vel, channel=ch, time=delta))
        else:
            t.append(mido.Message('note_off', note=note, velocity=0,   channel=ch, time=delta))
    t.append(mido.MetaMessage('end_of_track', time=0))
    return t

def _chord_block(root, ivs, start, dur, vel):
    return [(root + i, vel, start, dur) for i in ivs]

def gen_midi_01_rondo(path):
    """Forma Rondo A-B-A-C-A (D minor, allegro)"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    mel, ctr, har, bas, acc = [], [], [], [], []
    # A theme (bars 0-7): D minor motif
    A = [(62,90,0,.5),(65,85,.5,.5),(69,88,1,1),(67,82,2,.5),(65,80,2.5,.5),(62,85,3,1),
         (60,80,4,.5),(62,82,.5+4,.5),(65,85,5,1),(67,88,6,.5),(69,90,6.5,.5),(70,85,7,1)]
    A2 = [(p,v,t+8,d) for p,v,t,d in A]
    # B theme (bars 8-15): F major
    B = [(65,88,16,.5),(67,85,16.5,.5),(69,90,17,1),(72,92,18,1),(70,85,19,1),
         (69,88,20,.5),(67,82,20.5,.5),(65,85,21,1),(64,80,22,1),(65,82,23,1)]
    B2 = [(p,v,t+8,d) for p,v,t,d in B]
    # A again (bars 16-23)
    A3 = [(p,v,t+32,d) for p,v,t,d in A+A2]
    # C theme (bars 24-31): Bb major, lyrical
    C = [(70,75,48,2),(69,72,50,2),(67,75,52,2),(65,70,54,2),
         (67,75,56,2),(69,78,58,2),(70,80,60,2),(72,82,62,2)]
    # Final A (bars 32-39)
    A4 = [(p,v,t+64,d) for p,v,t,d in A+A2]
    mel = A+A2+B+B2+A3+C+A4
    # Harmony: Am-F-C-G in A sections, F-C-Dm-Bb in B, Bb-F-Gm-Eb in C
    def harm_block(prog, start, reps):
        h = []
        for r in range(reps):
            for ci,(root,ivs,vel) in enumerate(prog):
                h += _chord_block(root,ivs,start+r*16+ci*4,3.8,vel)
        return h
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 0, 2)
    har += harm_block([(53,[0,4,7],65),(60,[0,4,7],62),(62,[0,3,7],65),(58,[0,4,7],68)], 32, 2)
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 64, 2)
    har += harm_block([(58,[0,4,7],62),(53,[0,4,7],60),(55,[0,3,7],62),(51,[0,4,7],58)], 96, 2)
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 128, 2)
    # Bass, counter, acc (generic)
    total_beats = 160
    for b in range(40):
        bas += [(45,85,b*4,.9),(45,70,b*4+2,.9)]
        ctr += [(57,68,b*4,2),(60,65,b*4+2,2)]
        for i,p in enumerate([48,52,55,52]):
            acc.append((p,50,b*4+i,0.9))
    mid.tracks += [_make_track("Melodía",1,mel,True,132),_make_track("Contramelodía",2,ctr),
                   _make_track("Armonía",3,har),_make_track("Bajo",4,bas),_make_track("Acomp.",5,acc)]
    mid.save(path)

def gen_midi_02_blues(path):
    """Blues 12-compases en A, forma AAB, 3 choruses"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # 12-bar blues progression: A7-D7-A7-A7-D7-D7-A7-A7-E7-D7-A7-E7
    blues_prog = [
        (57,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),
        (62,[0,4,7,10],70),(62,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),
        (64,[0,4,7,10],72),(62,[0,4,7,10],70),(57,[0,4,7,10],70),(64,[0,4,7,10],72),
    ]
    har = []
    for chorus in range(3):
        for bi,(root,ivs,vel) in enumerate(blues_prog):
            har += _chord_block(root,ivs,chorus*48+bi*4,3.8,vel)
    # Melody: pentatonic licks
    pentatonic = [57,60,62,64,67,69]
    mel = []
    for chorus in range(3):
        for bar in range(12):
            base = chorus*48+bar*4
            lick = [pentatonic[(bar*3+i)%6] for i in range(4)]
            for i,p in enumerate(lick):
                mel.append((p,85+chorus*3,base+i,0.9))
    bas, ctr, acc = [], [], []
    for b in range(36):
        bas += [(45,88,b*4,.5),(45,75,b*4+2,.5)]
        ctr += [(57,65,b*4,4)]
        for i,p in enumerate([45,52,57,52]):
            acc.append((p,52,b*4+i*0.5,0.45))
    mid.tracks += [_make_track("Guitar",1,mel,True,96),_make_track("Keys",2,ctr),
                   _make_track("Piano",3,har),_make_track("Bass",4,bas),_make_track("Rhythm",5,acc)]
    mid.save(path)

def gen_midi_03_waltz(path):
    """Vals en 3/4, La bemol mayor, forma A-B-A'-Coda"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    mid2 = mido.MidiFile(ticks_per_beat=480, type=1)
    # Waltz: 3 beats/bar
    TPB = 480
    def waltz_track(name,ch,notes,meta=False,bpm=160):
        t = mido.MidiTrack()
        t.name=name
        t.append(mido.MetaMessage('track_name',name=name,time=0))
        if meta:
            t.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bpm),time=0))
            t.append(mido.MetaMessage('time_signature',numerator=3,denominator=4,
                                      clocks_per_click=24,notated_32nd_notes_per_beat=8,time=0))
        events=[]
        for pitch,vel,start,dur in notes:
            s=int(start*TPB); e=int((start+dur)*TPB)
            events+=[(s,'on',pitch,vel,ch),(e,'off',pitch,0,ch)]
        events.sort(key=lambda x:(x[0],0 if x[1]=='off' else 1))
        prev=0
        for tick,kind,note,vel,c in events:
            d=tick-prev;prev=tick
            if kind=='on': t.append(mido.Message('note_on',note=note,velocity=vel,channel=c,time=d))
            else:          t.append(mido.Message('note_off',note=note,velocity=0,channel=c,time=d))
        t.append(mido.MetaMessage('end_of_track',time=0))
        return t
    # A: Ab major theme (bar=3 beats)
    mel=[]
    A_mel=[(68,88,0,1.5),(70,82,1.5,1.5),(72,85,3,3),(70,80,6,1.5),(68,82,7.5,1.5),(65,85,9,3),
           (63,82,12,1.5),(65,80,13.5,1.5),(68,85,15,3),(70,88,18,3),(68,90,21,3),(65,85,24,3)]
    mel+=A_mel
    # B: Eb major, higher
    B_mel=[(75,90,27,1.5),(77,85,28.5,1.5),(79,88,30,3),(77,82,33,1.5),(75,80,34.5,1.5),(72,85,36,3),
           (70,82,39,1.5),(72,80,40.5,1.5),(75,85,42,3),(77,88,45,3),(75,90,48,3),(72,85,51,3)]
    mel+=B_mel
    # A' reprise
    mel+=[(p,v,t+54,d) for p,v,t,d in A_mel]
    # Coda
    mel+=[(68,95,81,3),(65,90,84,3),(63,88,87,3),(60,85,90,6)]
    har,bas,ctr,acc=[],[],[],[]
    for b in range(32):
        base=b*3
        har+=_chord_block(56,[0,4,7],base,2.8,65)
        bas+=[(44,85,base,1),(44,65,base+1,1),(44,65,base+2,1)]
        ctr+=[(60,68,base,3)]
        acc+=[(56,50,base,1),(60,50,base+1,1),(63,50,base+2,1)]
    mid.tracks+=[waltz_track("Violín",1,mel,True,160),waltz_track("Viola",2,ctr),
                 waltz_track("Piano",3,har),waltz_track("Cello",4,bas),waltz_track("Arpa",5,acc)]
    mid.save(path)

def gen_midi_04_fugue(path):
    """Fuga a 4 voces en Re menor, sujeto + respuesta + episodios"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    subject = [(62,85,0,1),(65,80,.5,.5),(69,82,1,.5),(67,78,1.5,.5),(65,82,2,1),(62,78,3,1)]
    answer  = [(69,85,0,1),(72,80,.5,.5),(76,82,1,.5),(74,78,1.5,.5),(72,82,2,1),(69,78,3,1)]
    def voice(subj,offset,transp,start_bar):
        return [(p+transp,v,t+start_bar*4,d) for p,v,t,d in subj[offset:]+subj[:offset]]
    mel  = subject + [(p,v,t+16,d) for p,v,t,d in subject]  # S enters bar 0,4
    mel += [(p,v,t+32,d) for p,v,t,d in subject]             # episode bar 8
    mel += [(p,v,t+48,d) for p,v,t,d in answer]              # stretto bar 12
    ctr  = [(p,v,t+8,d) for p,v,t,d in answer]               # A enters bar 2
    ctr += [(p,v,t+24,d) for p,v,t,d in subject]
    ctr += [(p,v,t+40,d) for p,v,t,d in answer]
    har  = [(p-12,v,t+12,d) for p,v,t,d in subject]          # T enters bar 3
    har += [(p-12,v,t+28,d) for p,v,t,d in answer]
    har += [(p-12,v,t+44,d) for p,v,t,d in subject]
    bas  = [(p-24,v,t+16,d) for p,v,t,d in answer]           # B enters bar 4
    bas += [(p-24,v,t+32,d) for p,v,t,d in subject]
    bas += [(p-24,v,t+48,d) for p,v,t,d in answer]
    # Fill with sustained harmony
    har_chords=[]
    for b in range(16):
        har_chords+=_chord_block(57,[0,3,7],b*4,3.8,60)
    acc=[]
    for b in range(16):
        for i,p in enumerate([45,50,53,57]):
            acc.append((p,45,b*4+i,0.9))
    mid.tracks+=[_make_track("Soprano",1,mel,True,80),_make_track("Alto",2,ctr),
                 _make_track("Tenor",3,har+har_chords),_make_track("Bajo",4,bas),
                 _make_track("Continuo",5,acc)]
    mid.save(path)

def gen_midi_05_tango(path):
    """Tango en La menor, forma A-B-A con habanera y bandoneón"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Habanera rhythm: 1 . . 1 1 . 1 1 (dotted)
    def habanera(root, start, bars=4):
        n=[]
        for b in range(bars):
            base=start+b*4
            n+=[(root,85,base,1.5),(root,80,base+1.5,.5),(root,78,base+2,1),(root,75,base+3,1)]
        return n
    # Melody A: Am pentatonic with chromatic passing tones
    A_mel=[(69,90,0,1),(68,85,1,1),(67,88,2,.5),(65,82,2.5,.5),(64,85,3,1),
           (62,88,4,1),(64,85,5,1),(65,82,6,1),(67,88,7,1),
           (69,90,8,1),(71,92,9,1),(72,88,10,2),(69,85,12,4),
           (67,88,16,1),(68,85,17,1),(69,88,18,.5),(71,85,18.5,.5),(72,90,19,1),
           (74,92,20,1),(72,88,21,1),(71,85,22,1),(69,82,23,1),
           (67,85,24,1),(65,80,25,1),(64,82,26,1),(62,85,27,1),(57,90,28,4)]
    # B: D minor, more lyrical
    B_mel=[(65,85,32,2),(67,82,34,2),(69,85,36,1),(67,80,37,1),(65,82,38,2),
           (64,80,40,2),(62,82,42,2),(60,85,44,4),
           (62,88,48,1),(64,85,49,1),(65,88,50,1),(67,85,51,1),(69,90,52,4),
           (67,85,56,1),(65,80,57,1),(64,82,58,1),(62,80,59,1),(57,88,60,4)]
    # A reprise
    A2 = [(p,v,t+64,d) for p,v,t,d in A_mel]
    mel = A_mel+B_mel+A2
    har,bas,ctr,acc=[],[],[],[]
    for b in range(24):
        root=[57,57,57,57,62,62,57,57,64,62,57,64][b%12]
        ivs=[0,3,7] if b%4<2 else [0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,68)
        bas+=habanera(root-12,b*4,1)
        ctr+=[(root+7,65,b*4,2),(root+5,62,b*4+2,2)]
        acc+=[(root,48,b*4+i,0.9) for i in range(4)]
    mid.tracks+=[_make_track("Bandoneón",1,mel,True,92),_make_track("Violín",2,ctr),
                 _make_track("Piano",3,har),_make_track("Contrabajo",4,bas),
                 _make_track("Guitarra",5,acc)]
    mid.save(path)

def gen_midi_06_modal(path):
    """Modal jazz en Dorian (D Dorian), forma AABA 32 bars × 3 choruses"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    dorian = [62,64,65,67,69,71,72,74]
    def dorian_line(start, offset=0, vel_base=82):
        notes=[]
        for i in range(8):
            p=dorian[(i+offset)%8]
            notes.append((p,vel_base+i%3*3,start+i*0.5,0.45))
        return notes
    mel=[]
    for chorus in range(3):
        base=chorus*128
        # AA: D dorian vamp (8 bars each)
        for rep in range(2):
            for bar in range(8):
                mel+=dorian_line(base+rep*32+bar*4, bar%4, 80+chorus*4)
        # B: G dorian (8 bars)
        g_dorian=[55,57,58,60,62,64,65,67]
        for bar in range(8):
            p=g_dorian[bar%8]
            mel+=[(p,85,base+64+bar*4+i*0.5,0.45) for i in range(8)]
        # A: D dorian (8 bars)
        for bar in range(8):
            mel+=dorian_line(base+96+bar*4, bar%6, 82)
    # Harmony: Dm7 - Gm7 alternating
    har=[]
    for b in range(48):
        root=62 if b%8<4 else 55
        har+=_chord_block(root,[0,3,7,10],b*4,3.8,62)
    bas,ctr,acc=[],[],[]
    for b in range(48):
        bas+=[(50,88,b*4,.9),(50,72,b*4+2,.9)]
        ctr+=[(57,65,b*4,4)]
        for i,p in enumerate([50,57,62,57]):
            acc.append((p,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Sax",1,mel,True,120),_make_track("Trompeta",2,ctr),
                 _make_track("Piano",3,har),_make_track("Contrabajo",4,bas),
                 _make_track("Batería",5,acc)]
    mid.save(path)

def gen_midi_07_minimalist(path):
    """Minimalismo (estilo Reich), Do mayor, adición de fases"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Short motif repeated with phase shift
    motif=[60,64,67,72,71,67,64,60]
    mel,ctr,har,bas,acc=[],[],[],[],[]
    total_bars=48
    for rep in range(total_bars*4):   # 1 rep = 1 beat
        for i,p in enumerate(motif):
            beat=rep+i*0.125
            mel.append((p,72+(rep%8),beat,0.12))
    # Phase voice: same motif shifted by 1/8 beat, gradually drifts
    for rep in range(total_bars*4):
        shift=rep*(1/64)
        for i,p in enumerate(motif):
            beat=rep+i*0.125+shift
            if beat < total_bars*4+4:
                ctr.append((p-12,65,beat,0.12))
    # Sustained harmony pads
    prog=[(60,[0,4,7],58),(67,[0,4,7],55),(65,[0,4,7],55),(62,[0,3,7],52)]
    for b in range(total_bars):
        root,ivs,vel=prog[b%4]
        har+=_chord_block(root,ivs,b*4,3.8,vel)
        bas+=[(root-12,80,b*4,2),(root-12,72,b*4+2,2)]
        for i,p in enumerate([48,52,55,60]):
            acc.append((p,45,b*4+i,0.9))
    mid.tracks+=[_make_track("Voz1",1,mel,True,132),_make_track("Voz2",2,ctr),
                 _make_track("Pad",3,har),_make_track("Bajo",4,bas),
                 _make_track("Pulse",5,acc)]
    mid.save(path)

def gen_midi_08_flamenco(path):
    """Flamenco (Phrygian, Mi), forma A-B-A con falseta y compás por 12"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Phrygian E: E F G A B C D E
    phryg=[64,65,67,69,71,72,74,76]
    # Falseta A (12-beat compás × 4)
    def falseta(start):
        n=[]
        pat=[64,65,64,62,64,65,67,65,64,62,60,62]
        for i,p in enumerate(pat):
            n.append((p,88-i%3*4,start+i,0.9))
        return n
    mel=[]
    for rep in range(4):
        mel+=falseta(rep*12)
    # B section: more lyrical, higher
    B_mel=[(76,90,48,1.5),(74,85,49.5,1.5),(72,88,51,1.5),(71,82,52.5,1.5),
           (69,85,54,3),(71,88,57,3),(72,90,60,6),
           (74,88,66,1.5),(72,85,67.5,1.5),(71,88,69,1.5),(69,82,70.5,1.5),
           (67,85,72,3),(69,88,75,3),(71,90,78,6)]
    mel+=B_mel
    # Reprise A
    mel+=[(p,v,t+84,d) for p,v,t,d in [(p,v,t,d) for p,v,t,d in [item for item in [falseta(0)+falseta(12)][0]]]]
    # Harmony: Am - G - F - E (Andalusian cadence)
    har=[]
    for b in range(28):
        root=[57,55,53,52][b%4]
        ivs=[0,3,7] if b%4<3 else [0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,68)
    bas,ctr,acc=[],[],[]
    for b in range(28):
        rt2 = [57,55,53,52][b%4]
        bas+=[(rt2-12,85,b*4,1)]
        ctr+=[(rt2+7,65,b*4,2),(rt2+5,62,b*4+2,2)]
        for i in range(4):
            acc.append((rt2,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Guitarra",1,mel,True,88),_make_track("Cante",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Bajo",4,bas),
                 _make_track("Palmas",5,acc)]
    mid.save(path)

def gen_midi_09_march(path):
    """Marcha en Sol mayor, forma intro-A-B-Trio-A-B-Coda"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Intro: fanfare (4 bars)
    intro=[(67,95,0,1),(71,90,1,1),(74,92,2,1),(79,95,3,1),
           (79,90,4,.5),(77,85,4.5,.5),(79,90,5,1),(77,85,6,1),(74,88,7,1)]
    # A theme (8 bars): martial melody
    A_mel=[(67,88,8,1),(69,82,9,1),(71,85,10,1),(72,88,11,1),(74,90,12,2),(72,85,14,2),
           (71,88,16,1),(69,82,17,1),(67,85,18,1),(65,82,19,1),(64,85,20,2),(67,88,22,2),
           (69,85,24,1),(71,82,25,1),(72,85,26,1),(74,88,27,1),(76,90,28,2),(74,85,30,2),
           (72,88,32,.5),(71,85,32.5,.5),(69,82,33,1),(67,85,34,1),(65,82,35,1),(67,90,36,4)]
    # B theme (8 bars): contrasting
    B_mel=[(74,85,40,2),(72,80,42,2),(71,82,44,2),(69,80,46,2),
           (67,85,48,2),(69,82,50,2),(71,85,52,2),(74,88,54,2),
           (76,90,56,1),(74,85,57,1),(72,82,58,1),(71,80,59,1),(67,85,60,4),
           (69,88,64,.5),(71,85,64.5,.5),(72,82,65,1),(74,85,66,1),(76,88,67,1),(79,92,68,4)]
    # Trio (8 bars): in C major, softer
    trio_mel=[(72,78,72,2),(74,75,74,2),(76,78,76,2),(77,75,78,2),
              (79,80,80,2),(77,75,82,2),(76,78,84,2),(74,72,86,2),
              (72,75,88,2),(71,72,90,2),(69,75,92,2),(67,72,94,2),
              (69,75,96,2),(71,78,98,2),(72,80,100,2),(74,82,102,2),
              (76,85,104,4),(74,80,108,4),(72,78,112,8)]
    mel=intro+A_mel+B_mel+trio_mel
    # Harmony, bass, counter, acc
    har,bas,ctr,acc=[],[],[],[]
    for b in range(30):
        root=[55,55,60,55, 55,62,55,62][b%8]
        ivs=[0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,65)
        bas+=[(root-12,90,b*4,1),(root-12,72,b*4+2,1)]
        ctr+=[(root+4,68,b*4,2),(root+7,65,b*4+2,2)]
        for i in range(4):
            acc.append((root,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Trompeta",1,mel,True,120),_make_track("Trombón",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Tuba",4,bas),
                 _make_track("Tambor",5,acc)]
    mid.save(path)

def gen_midi_10_sonata(path):
    """Forma sonata: Expo(P-T-S-K) - Desarrollo - Recapitulación, Do mayor"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Primary theme P (C major, bars 0-7)
    P=[(60,90,0,.5),(64,85,.5,.5),(67,88,1,.5),(72,90,1.5,.5),(74,88,2,1),(72,85,3,1),
       (71,88,4,.5),(69,82,4.5,.5),(67,85,5,1),(65,80,6,1),(64,85,7,1),
       (60,88,8,.5),(62,82,.5+8,.5),(64,85,9,.5),(65,82,9.5,.5),(67,85,10,2),(60,80,12,4)]
    # Transition T (bars 8-11): modulating to G
    T=[(62,85,16,1),(64,82,17,1),(65,85,18,1),(67,88,19,1),(69,90,20,1),(71,85,21,1),(72,82,22,2)]
    # Secondary theme S (G major, bars 12-19): lyrical
    S=[(67,80,24,2),(69,78,26,2),(71,80,28,2),(72,78,30,2),
       (74,82,32,2),(72,78,34,2),(71,80,36,2),(69,78,38,2),
       (67,82,40,4),(71,78,44,4),(74,80,48,4),(72,75,52,4)]
    # Closing K (bars 20-23): cadential
    K=[(74,85,56,1),(72,80,57,1),(71,82,58,1),(69,80,59,1),(67,88,60,4),(55,92,64,8)]
    # Development (bars 24-35): fragmentation
    dev=[]
    frag=P[:4]
    for step in range(12):
        transp=[0,2,4,5,7,9,10,12,7,5,2,0][step]
        dev+=[(p+transp,85,t+72+step*4,d) for p,v,t,d in frag]
    # Recapitulation (bars 36-51): P+T+S in C
    recap_P=[(p,v,t+120,d) for p,v,t,d in P]
    recap_S=[(p-7,v,t+144,d) for p,v,t,d in S]  # S transposed back to C
    recap_K=[(p-7,v,t+168,d) for p,v,t,d in K]
    mel=P+T+S+K+dev+recap_P+recap_S+recap_K
    har,bas,ctr,acc=[],[],[],[]
    for b in range(46):
        root=[60,60,67,60, 60,62,60,67][b%8]
        ivs=[0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,65)
        bas+=[(root-12,85,b*4,1.5),(root-12,70,b*4+2,1.5)]
        ctr+=[(root+4,65,b*4,2),(root+7,62,b*4+2,2)]
        for i in range(4):
            acc.append((root,48,b*4+i,0.9))
    mid.tracks+=[_make_track("Piano",1,mel,True,116),_make_track("Violín",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Cello",4,bas),
                 _make_track("Continuo",5,acc)]
    mid.save(path)


BATCH_GENERATORS = [
    ("01_rondo_Dm",      gen_midi_01_rondo,      "Rondo A-B-A-C-A, Re menor"),
    ("02_blues_Am",      gen_midi_02_blues,       "Blues 12 compases, 3 choruses"),
    ("03_waltz_Ab",      gen_midi_03_waltz,       "Vals 3/4, La bemol mayor"),
    ("04_fugue_Dm",      gen_midi_04_fugue,       "Fuga a 4 voces, Re menor"),
    ("05_tango_Am",      gen_midi_05_tango,       "Tango, La menor"),
    ("06_modal_Dorian",  gen_midi_06_modal,       "Modal jazz, D Dorian, AABA × 3"),
    ("07_minimalist_C",  gen_midi_07_minimalist,  "Minimalismo en fase, Do mayor"),
    ("08_flamenco_E",    gen_midi_08_flamenco,    "Flamenco Frigio, Mi"),
    ("09_march_G",       gen_midi_09_march,       "Marcha, Sol mayor"),
    ("10_sonata_C",      gen_midi_10_sonata,      "Forma sonata, Do mayor"),
]


def generate_batch(out_dir: str, cfg: dict = None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or {}
    for slug, gen_fn, desc in BATCH_GENERATORS:
        mid_path  = os.path.join(out_dir, f"{slug}.mid")
        html_path = os.path.join(out_dir, f"{slug}_report.html")
        print(f"\n  ── {slug}: {desc}")
        try:
            gen_fn(mid_path)
            html = analyze(mid_path, cfg)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"     ✓ {html_path}")
        except Exception as e:
            import traceback
            print(f"     ✗ ERROR: {e}")
            traceback.print_exc()
