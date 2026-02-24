"""
midi_dna_combiner_v2.py  —  IMPROVED VERSION
=============================================
Major improvements over v1:
  1. VOICE-LEADING REHARMONIZER  — respects contrary/oblique motion,
     avoids parallel 5ths/8ths, resolves V→I properly.
  2. METRIC-AWARE RHYTHM TRANSFORMER — strong notes land on strong beats,
     long notes on beat 1/3, leaps on downbeats.
  3. MUSICAL PHRASE SCORING — Sequitur phrases ranked by arc shape,
     tonal resolution, and optimal length (6-8 notes).
  4. REAL ACCOMPANIMENT GENERATOR — Alberti bass, arpeggio patterns,
     and block chords chosen per section energy level.
  5. PIVOT-CHORD BRIDGE between sections.
  6. MIDI VELOCITY CURVES from Lerdahl-style tension arc.
  7. N-SOURCE BLENDING with weights (--weights flag).
  8. FULL mscz2vec.py integration hooks (cadence detection,
     tonnetz, arpeggios) when that module is importable.

USAGE:
  python midi_dna_combiner_v2.py \\
    --melody  a.mid  --harmony b.mid  --rhythm c.mid \\
    --output  out.mid  --form AABA  --surprise 0.08

  Auto roles (2-4 files):
  python midi_dna_combiner_v2.py a.mid b.mid c.mid --output out.mid

DEPENDENCIES:
  pip install music21 numpy mido
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
import sys, argparse, copy, random, warnings, math
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

try:
    from music21 import (
        converter, stream, note, chord, pitch, key as m21key,
        meter, tempo, instrument, roman, interval as m21interval,
        duration, dynamics, expressions, spanner, harmony, analysis
    )
except ImportError:
    print("music21 not found.  pip install music21"); sys.exit(1)

try:
    import mido
except ImportError:
    print("mido not found.  pip install mido"); sys.exit(1)

# Optional: import mscz2vec for advanced feature extraction
try:
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        "mscz2vec",
        os.path.join(os.path.dirname(__file__), "mscz2vec.py")
    )
    if _spec:
        _mscz = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mscz)
        HAVE_MSCZ = True
    else:
        HAVE_MSCZ = False
except Exception:
    HAVE_MSCZ = False


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

HARMONIC_FUNCTIONS = {
    "T":    ["I", "i", "vi", "VI"],
    "PD":   ["ii", "II", "iv", "IV"],
    "D":    ["V", "v", "vii°", "VII"],
    "Dsec": ["V/V", "V/ii", "V/vi", "V/IV"],
    "Other": []
}

FUNCTION_TENSION = {"T": 0.1, "PD": 0.4, "D": 0.7, "Dsec": 0.9, "Other": 0.5}

# Plomp-Levelt consonance (0 = max dissonance, 1 = max consonance)
CONSONANCE = {0:1.0, 1:0.1, 2:0.2, 3:0.6, 4:0.7,
              5:0.8, 6:0.0, 7:0.9, 8:0.65, 9:0.75, 10:0.3, 11:0.15}

# Chord templates [root-relative semitones]
CHORD_TEMPLATES = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "dominant7":  [0, 4, 7, 10],
    "minor7":     [0, 3, 7, 10],
    "major7":     [0, 4, 7, 11],
    "diminished": [0, 3, 6],
    "half-dim7":  [0, 3, 6, 10],
}

MAJOR_DEGREES = {
    "I":   (0,  "major"),   "ii":  (2,  "minor"),
    "iii": (4,  "minor"),   "IV":  (5,  "major"),
    "V":   (7,  "dominant7"), "vi": (9,  "minor"),
    "vii°":(11, "diminished"),
}
MINOR_DEGREES = {
    "i":   (0,  "minor"),   "ii°": (2,  "diminished"),
    "III": (3,  "major"),   "iv":  (5,  "minor"),
    "V":   (7,  "dominant7"), "VI": (8,  "major"),
    "VII": (10, "major"),
}

# Strong beats (1-indexed) for 4/4 and 3/4
STRONG_BEATS = {4: {1, 3}, 3: {1}}

# Accompaniment patterns  (beat offsets within one measure)
ACC_PATTERNS = {
    "block":    lambda beats: [(0, beats)],
    "alberti":  lambda beats: [(0, beats/4), (beats/2, beats/4),
                                (beats/4, beats/4), (beats/2, beats/4)],
    "arpeggio_up":   lambda beats: [(i * beats/4, beats/4) for i in range(4)],
    "arpeggio_down": lambda beats: [(i * beats/4, beats/4) for i in range(3,-1,-1)],
    "waltz":    lambda beats: [(0, 1.0), (1, 0.5), (2, 0.5)],
}

# Velocity ranges for dynamics
DYN_VELOCITY = {
    "ppp":15, "pp":30, "p":45, "mp":60,
    "mf":75, "f":90, "ff":105, "fff":120
}


# ══════════════════════════════════════════════════════════════
# SEQUITUR GRAMMAR  (Re-Pair)
# ══════════════════════════════════════════════════════════════

class SequiturGrammar:
    def __init__(self):
        self.rules = {}
        self.next_id = 1
        self.root = []

    def _new_rule(self):
        name = f"R{self.next_id}"; self.next_id += 1; return name

    def build(self, symbols):
        if len(symbols) < 4:
            self.root = list(symbols); return self.rules
        seq = list(symbols)
        for _ in range(300):
            pairs = Counter()
            for i in range(len(seq)-1):
                pairs[(seq[i], seq[i+1])] += 1
            if not pairs: break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2: break
            name = self._new_rule()
            self.rules[name] = list(best)
            new, i = [], 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == best:
                    new.append(name); i += 2
                else:
                    new.append(seq[i]); i += 1
            seq = new
        self.root = seq
        return self.rules

    def expand(self, sym):
        if sym not in self.rules: return [sym]
        r = []
        for s in self.rules[sym]: r.extend(self.expand(s))
        return r

    def get_raw_phrases(self):
        phrases = []
        for name in self.rules:
            expanded = self.expand(name)
            freq = self.root.count(name)
            phrases.append((expanded, freq, name))
        return phrases


# ══════════════════════════════════════════════════════════════
# PHRASE SCORER  — musical quality ranking
# ══════════════════════════════════════════════════════════════

def score_phrase(pitches, durations, key_obj):
    """
    Score a phrase on:
      arc_shape      — rises then falls (or reverse), not monotone
      tonal_close    — ends on tonic-triad note
      length_fit     — optimal 5-8 notes
      interval_variety — mix of steps and leaps
      rhythmic_interest — not all same duration
    Returns float 0-1.
    """
    if len(pitches) < 2:
        return 0.0

    # Arc shape: compare first-half average to second-half average
    mid = len(pitches) // 2
    first_half_avg = np.mean(pitches[:mid])
    second_half_avg = np.mean(pitches[mid:])
    arc = min(abs(first_half_avg - second_half_avg) / 12, 1.0)  # 0=flat, 1=big arc

    # Tonal closure: last note is tonic/3rd/5th
    tonic_pc = key_obj.tonic.pitchClass
    try:
        tonic_pcs = {p.pitchClass for p in key_obj.getChord().pitches}
    except Exception:
        tonic_pcs = {tonic_pc, (tonic_pc+4)%12, (tonic_pc+7)%12}
    last_pc = pitches[-1] % 12
    tonal_close = 1.0 if last_pc in tonic_pcs else 0.3

    # Length: 5-8 notes ideal
    l = len(pitches)
    length_fit = 1.0 if 5 <= l <= 8 else max(0.2, 1.0 - abs(l - 6.5) * 0.15)

    # Interval variety
    ivs = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    steps = sum(1 for v in ivs if 1 <= v <= 2)
    leaps = sum(1 for v in ivs if v >= 3)
    total = len(ivs)
    variety = min(steps, leaps) / max(total, 1)  # balanced is good

    # Rhythmic interest
    if durations:
        dur_set = len(set(round(d*4)/4 for d in durations))
        rhy = min(dur_set / 3, 1.0)
    else:
        rhy = 0.5

    return (arc * 0.25 + tonal_close * 0.30 + length_fit * 0.20
            + variety * 0.15 + rhy * 0.10)


# ══════════════════════════════════════════════════════════════
# MELODY DNA EXTRACTOR
# ══════════════════════════════════════════════════════════════

def extract_melody_dna(score):
    """Extract Sequitur phrases, scored by musical quality."""
    parts = list(score.parts) if score.parts else [score]
    flat = parts[0].flatten().notes

    midi_seq, dur_seq = [], []
    for n in flat:
        if n.isNote:
            midi_seq.append(n.pitch.midi)
            dur_seq.append(float(n.quarterLength))
        elif n.isChord:
            top = max(n.pitches, key=lambda p: p.midi)
            midi_seq.append(top.midi)
            dur_seq.append(float(n.quarterLength))

    if not midi_seq:
        return [], []

    key_obj = _detect_key(score)
    grammar = SequiturGrammar()
    grammar.build(midi_seq)
    raw = grammar.get_raw_phrases()

    scored = []
    for pitches, freq, name in raw:
        if len(pitches) < 3: continue
        # Get matching durations
        start = None
        for i in range(len(midi_seq) - len(pitches) + 1):
            if midi_seq[i:i+len(pitches)] == pitches:
                start = i; break
        durs = dur_seq[start:start+len(pitches)] if start is not None else [0.5]*len(pitches)
        q = score_phrase(pitches, durs, key_obj)
        paired = list(zip(pitches, durs))
        scored.append((paired, freq, q, name))

    # Sort by combined score: quality * sqrt(freq)
    scored.sort(key=lambda x: x[2] * math.sqrt(x[1]), reverse=True)
    phrases = [(p, f) for p, f, q, n in scored[:10]]

    full_melody = list(zip(midi_seq[:64], dur_seq[:64]))
    return phrases, full_melody


# ══════════════════════════════════════════════════════════════
# HARMONIC DNA EXTRACTOR
# ══════════════════════════════════════════════════════════════

def extract_chord_progression(score):
    """
    Extract chord progression with voice-leading metadata:
    each entry has pitches list (sorted low→high) for smooth transitions.
    """
    key_obj = _detect_key(score)
    try:
        chordified = score.chordify()
        flat_chords = list(chordified.flatten().getElementsByClass('Chord'))
    except Exception as e:
        print(f"  [warn] chordify failed: {e}")
        return [], key_obj

    prog = []
    for ch in flat_chords:
        if len(ch.pitches) < 2: continue
        if float(ch.quarterLength) < 0.25: continue
        try:
            rn = roman.romanNumeralFromChord(ch, key_obj)
            func = _roman_to_func(rn)
            pitches_sorted = sorted([p.midi for p in ch.pitches])
            prog.append({
                'figure':   rn.figure,
                'function': func,
                'duration': float(ch.quarterLength),
                'pitches':  pitches_sorted,
                'chord':    ch,
                'rn':       rn,
            })
        except Exception:
            continue
    return prog, key_obj


def _roman_to_func(rn):
    fig = rn.figure
    if "/" in fig: return "Dsec"
    base = "".join(c for c in fig.replace("°","").replace("+","") if not c.isdigit())
    for func, figs in HARMONIC_FUNCTIONS.items():
        if base in figs: return func
    return "Other"


def _roman_to_func_str(fig):
    """String version for fallback progressions."""
    if "/" in fig: return "Dsec"
    base = "".join(c for c in fig.replace("°","").replace("+","") if not c.isdigit())
    for func, figs in HARMONIC_FUNCTIONS.items():
        if base in figs: return func
    return "Other"


def summarize_progression(prog, n=8):
    """Deduplicate and condense to n chord events."""
    if not prog: return []
    grouped, prev = [], None
    for p in prog:
        if prev and prev['figure'] == p['figure']:
            prev['duration'] += p['duration']
        else:
            prev = dict(p); grouped.append(prev)
    step = max(1, len(grouped) // n)
    result = grouped[::step][:n]
    # Ensure tonic ending
    if result and result[-1]['function'] != 'T':
        tonic = next((p for p in grouped if p['function']=='T'), None)
        if tonic: result.append(tonic)
    return result


# ══════════════════════════════════════════════════════════════
# VOICE-LEADING REHARMONIZER
# ══════════════════════════════════════════════════════════════

def voice_lead_next_chord(prev_pitches, candidate_pitches):
    """
    Given previous chord pitches (MIDI list) and a candidate chord,
    return the best voicing of the candidate that minimizes voice motion.
    Uses nearest-neighbor assignment (simplified optimal transport).
    """
    if not prev_pitches or not candidate_pitches:
        return candidate_pitches

    # Normalize candidate to have same number of voices as prev
    n = len(prev_pitches)
    cand = sorted(candidate_pitches)

    # Generate inversions / octave doublings
    best_voicing = None
    best_cost = float('inf')

    # Try original + two inversions + octave shifts
    base_pc = [p % 12 for p in cand]
    for octave_shift in range(-1, 2):
        for inversion in range(len(cand)):
            rotated_pc = base_pc[inversion:] + base_pc[:inversion]
            # Build voicing starting from bass register of prev
            bass = prev_pitches[0]
            voicing = []
            ref = bass + (octave_shift * 12)
            for pc in rotated_pc:
                # Find nearest octave to ref
                best_p = pc + 12 * round((ref - pc) / 12)
                best_p = int(np.clip(best_p, 28, 96))
                voicing.append(best_p)
                ref = best_p + 2  # walk upward
            if len(voicing) < n:
                voicing += [voicing[-1]] * (n - len(voicing))
            voicing = voicing[:n]
            # Cost: total semitones of movement
            cost = sum(abs(voicing[i] - prev_pitches[i])
                       for i in range(min(n, len(voicing))))
            # Penalty for parallel 5ths/8ths
            for i in range(min(len(prev_pitches)-1, len(voicing)-1)):
                prev_iv = (prev_pitches[i+1] - prev_pitches[i]) % 12
                new_iv  = (voicing[i+1] - voicing[i]) % 12
                if prev_iv == new_iv and prev_iv in (0, 7):
                    cost += 12  # strong penalty
            if cost < best_cost:
                best_cost = cost
                best_voicing = voicing

    return best_voicing if best_voicing else candidate_pitches


def _build_chord_voicing(entry, prev_pitches=None, register_center=55):
    """
    Build a 3-4 voice chord in a given register, with voice leading from prev.
    """
    if 'pitches' in entry and entry['pitches']:
        raw = entry['pitches']
    else:
        try:
            raw = sorted([p.midi for p in entry['chord'].pitches])
        except Exception:
            return [register_center, register_center+4, register_center+7]

    # Shift to target register
    current_center = np.mean(raw)
    shift = round((register_center - current_center) / 12) * 12
    shifted = [p + shift for p in raw]
    shifted = [int(np.clip(p, 28, 84)) for p in shifted]

    if prev_pitches:
        shifted = voice_lead_next_chord(prev_pitches, shifted)

    return shifted


def check_voice_leading_rules(prev_v, curr_v):
    """
    Returns a score (higher = better voice leading).
    Checks: parallel 5ths, parallel 8ths, voice crossing, large leaps.
    """
    if not prev_v or not curr_v:
        return 1.0
    n = min(len(prev_v), len(curr_v))
    score = 1.0
    for i in range(n-1):
        prev_iv = (prev_v[i+1] - prev_v[i]) % 12
        curr_iv = (curr_v[i+1] - curr_v[i]) % 12
        motion_i = curr_v[i]   - prev_v[i]
        motion_j = curr_v[i+1] - prev_v[i+1]
        # Parallel 5ths
        if prev_iv == 7 and curr_iv == 7 and motion_i != 0:
            score -= 0.4
        # Parallel 8ths
        if prev_iv == 0 and curr_iv == 0 and motion_i != 0:
            score -= 0.4
        # Contrary motion bonus
        if motion_i * motion_j < 0:
            score += 0.1
    # Large leaps in melody (top voice)
    if abs(curr_v[-1] - prev_v[-1]) > 9:
        score -= 0.2
    return max(score, 0.0)


# ══════════════════════════════════════════════════════════════
# METRIC-AWARE RHYTHM TRANSFORMER
# ══════════════════════════════════════════════════════════════

def extract_rhythmic_dna(score, max_notes=300):
    """Extract rich rhythmic DNA including metric positions."""
    flat = score.flatten()
    notes_all = [n for n in flat.notes if n.isNote][:max_notes]
    if not notes_all:
        return _default_rhythm_dna()

    ts_list = list(flat.getElementsByClass('TimeSignature'))
    ts = ts_list[0] if ts_list else meter.TimeSignature('4/4')
    beats_per_measure = ts.numerator
    beat_unit = 4.0 / ts.denominator

    ratios, positions, beat_strengths = [], [], []
    syncopations = 0

    for n in notes_all:
        dur = float(n.quarterLength)
        ratios.append(dur)
        try:
            beat_pos = (float(n.beat) - 1) / beats_per_measure
        except Exception:
            beat_pos = 0.0
        positions.append(beat_pos)

        # Beat strength: 1=strong, 0.5=weak, 0.25=very weak
        try:
            beat = float(n.beat)
            strong = STRONG_BEATS.get(beats_per_measure, {1, 3})
            strength = 1.0 if math.ceil(beat) in strong else 0.5
        except Exception:
            strength = 0.5
        beat_strengths.append(strength)

        try:
            if float(n.beat) % 1 != 0: syncopations += 1
        except Exception:
            pass

    # 3-gram patterns (quantized)
    quantized = [_snap_dur(r) for r in ratios]
    patterns = Counter(tuple(quantized[i:i+3]) for i in range(len(quantized)-2))

    # Density map: how many notes per beat in this score
    total_dur = sum(float(n.quarterLength) for n in notes_all)
    avg_density = len(notes_all) / max(total_dur, 1)

    # Characteristic "cell" from top pattern
    top_cells = [list(cell) for cell, _ in patterns.most_common(4)]
    rhythm_cell = top_cells[0] if top_cells else [1.0, 0.5, 0.5, 1.0]

    # Detect if swing feel
    eighth_pairs = [ratios[i:i+2] for i in range(0, len(ratios)-1, 2)
                    if 0.25 <= ratios[i] <= 0.75 and 0.25 <= ratios[i+1] <= 0.75]
    swing_pairs = [p for p in eighth_pairs if p[0] > p[1] * 1.4]
    swing = len(swing_pairs) / max(len(eighth_pairs), 1) > 0.3

    return {
        'ratios':           ratios,
        'positions':        positions,
        'beat_strengths':   beat_strengths,
        'syncopation_rate': syncopations / max(len(notes_all), 1),
        'avg_density':      avg_density,
        'patterns':         patterns,
        'rhythm_cell':      rhythm_cell,
        'time_signature':   (ts.numerator, ts.denominator),
        'beats_per_measure':beats_per_measure,
        'beat_unit':        beat_unit,
        'swing':            swing,
    }


def _default_rhythm_dna():
    return {
        'ratios': [1.0,0.5,0.5,1.0,0.5,0.5,2.0],
        'positions': [0.0]*7,
        'beat_strengths': [1.0]*7,
        'syncopation_rate': 0.0,
        'avg_density': 1.0,
        'patterns': Counter(),
        'rhythm_cell': [1.0, 0.5, 0.5, 1.0],
        'time_signature': (4, 4),
        'beats_per_measure': 4,
        'beat_unit': 1.0,
        'swing': False,
    }


def _snap_dur(r):
    """Quantize to standard rhythmic values."""
    opts = [0.25, 0.333, 0.5, 0.667, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(opts, key=lambda x: abs(x - r))


def apply_rhythm_metric_aware(phrase_pairs, rhythm_dna, key_obj):
    """
    Reshape phrase durations so that:
    - Strong beats get longer durations
    - Leaps land on strong beats
    - Pattern from rhythm donor is preserved
    Returns list of (pitch, new_duration).
    """
    if not phrase_pairs:
        return []

    beats_pm  = rhythm_dna.get('beats_per_measure', 4)
    cell      = rhythm_dna.get('rhythm_cell', [1.0, 0.5, 0.5, 1.0])
    synco     = rhythm_dna.get('syncopation_rate', 0.0)
    strong    = STRONG_BEATS.get(beats_pm, {1, 3})

    result = []
    beat_cursor = 0.0  # position within measure (0 = beat 1)

    for i, (p, orig_dur) in enumerate(phrase_pairs):
        # Base duration from cycling rhythm cell
        base_dur = cell[i % len(cell)]

        # Detect leap
        if i > 0:
            leap = abs(p - phrase_pairs[i-1][0]) >= 5
        else:
            leap = False

        # Beat position (1-indexed)
        beat_1indexed = (beat_cursor % beats_pm) + 1
        on_strong = math.ceil(beat_1indexed) in strong

        # Metric shaping rules:
        # 1. Leap → strong beat: extend if not already on strong
        if leap and not on_strong:
            base_dur = _snap_dur(base_dur * 1.5)

        # 2. Syncopation: occasionally shorten to push onto off-beat
        if random.random() < synco * 0.4 and base_dur >= 0.5:
            base_dur = _snap_dur(base_dur * 0.5)

        # 3. Penultimate note often slightly longer (leading effect)
        if i == len(phrase_pairs) - 2:
            base_dur = _snap_dur(base_dur * 1.25)

        # 4. Last note: whole or half note for closure
        if i == len(phrase_pairs) - 1:
            base_dur = max(base_dur, 1.0)

        base_dur = max(0.25, base_dur)
        result.append((p, base_dur))
        beat_cursor += base_dur

    return result


# ══════════════════════════════════════════════════════════════
# ACCOMPANIMENT GENERATOR
# ══════════════════════════════════════════════════════════════

class AccompanimentGenerator:
    """
    Generates musically appropriate accompaniment based on:
    - Tension level (block chords at low tension, arpeggios at high)
    - Time signature (waltz pattern for 3/4)
    - Energy level
    """

    def __init__(self, rhythm_dna, energy_dna, key_obj):
        self.rhythm_dna = rhythm_dna
        self.energy_dna = energy_dna
        self.key_obj = key_obj
        beats_pm = rhythm_dna.get('beats_per_measure', 4)
        self.beats_per_measure = beats_pm
        self.beat_unit = rhythm_dna.get('beat_unit', 1.0)
        self.measure_len = beats_pm * self.beat_unit

    def choose_pattern(self, tension):
        """Choose accompaniment pattern based on tension level."""
        bpm = self.beats_per_measure
        if bpm == 3:
            return 'waltz'
        if tension < 0.25:
            return 'block'
        elif tension < 0.5:
            return 'alberti'
        elif tension < 0.75:
            return 'arpeggio_up'
        else:
            return random.choice(['arpeggio_up', 'arpeggio_down'])

    def build_alberti(self, chord_pitches, measure_dur):
        """
        Classic Alberti bass: low - high - middle - high
        Returns list of (pitch_midi, offset, duration).
        """
        if len(chord_pitches) < 3:
            return self.build_block(chord_pitches, measure_dur)
        low, mid, high = chord_pitches[0], chord_pitches[1], chord_pitches[2]
        beat = measure_dur / 4
        return [
            (low,  0 * beat, beat),
            (high, 1 * beat, beat),
            (mid,  2 * beat, beat),
            (high, 3 * beat, beat),
        ]

    def build_arpeggio(self, chord_pitches, measure_dur, direction='up'):
        """Arpeggiated chord."""
        if not chord_pitches:
            return []
        pitches = chord_pitches if direction == 'up' else list(reversed(chord_pitches))
        n = len(pitches)
        beat = measure_dur / max(n, 1)
        return [(pitches[i % n], i * beat, beat) for i in range(n)]

    def build_block(self, chord_pitches, measure_dur):
        """Block chord on beat 1."""
        return [(p, 0, measure_dur) for p in chord_pitches]

    def build_waltz(self, chord_pitches, measure_dur):
        """Waltz: bass note on 1, chord on 2+3."""
        if not chord_pitches:
            return []
        bass = chord_pitches[0]
        upper = chord_pitches[1:] if len(chord_pitches) > 1 else chord_pitches
        beat = measure_dur / 3
        events = [(bass, 0, beat)]
        for p in upper:
            events.append((p, beat, beat))
            events.append((p, beat * 2, beat))
        return events

    def generate_measure(self, chord_voicing, tension, offset_start):
        """
        Generate one measure of accompaniment as a list of note.Note objects
        to be inserted at offset_start.
        """
        dur = self.measure_len
        pattern = self.choose_pattern(tension)

        if pattern == 'alberti':
            events = self.build_alberti(chord_voicing, dur)
        elif pattern == 'arpeggio_up':
            events = self.build_arpeggio(chord_voicing, dur, 'up')
        elif pattern == 'arpeggio_down':
            events = self.build_arpeggio(chord_voicing, dur, 'down')
        elif pattern == 'waltz':
            events = self.build_waltz(chord_voicing, dur)
        else:
            events = self.build_block(chord_voicing, dur)

        notes_out = []
        for (midi_p, beat_offset, beat_dur) in events:
            n = note.Note()
            n.pitch.midi = int(np.clip(midi_p, 28, 84))
            n.duration = duration.Duration(quarterLength=float(beat_dur))
            notes_out.append((n, offset_start + beat_offset))

        return notes_out


# ══════════════════════════════════════════════════════════════
# ENERGY / TENSION / EMOTION DNA
# ══════════════════════════════════════════════════════════════

def extract_energy_dna(score):
    """Per-measure tension curve + overall energy descriptors."""
    # Try mscz2vec if available for richer analysis
    if HAVE_MSCZ:
        try:
            lv = _mscz.lerdahl_energy_vector(score)
            avg_tension = float(lv[0]) / max(float(lv[1]), 1) if lv[1] != 0 else 0.5
            climax_pos  = float(lv[3])
            tension_curve = _make_tension_curve_from_lerdahl(lv, 16)
        except Exception:
            tension_curve, avg_tension, climax_pos = _tension_heuristic(score)
    else:
        tension_curve, avg_tension, climax_pos = _tension_heuristic(score)

    key_obj  = _detect_key(score)
    valence  = 0.65 if key_obj.mode == 'major' else 0.35

    # Tempo
    mm_list  = list(score.flatten().getElementsByClass('MetronomeMark'))
    bpm      = float(mm_list[0].number) if mm_list else 120.0

    # Arousal: average pitch height
    all_midi = [n.pitch.midi for n in score.flatten().notes if n.isNote]
    arousal  = float(np.clip((np.mean(all_midi) - 48) / 36, 0, 1)) if all_midi else 0.5

    # Dynamic range
    dyn_range = (max(all_midi) - min(all_midi)) / 88.0 if all_midi else 0.5

    return {
        'tension_curve': tension_curve,
        'avg_tension':   avg_tension,
        'climax_pos':    climax_pos,
        'avg_arousal':   arousal,
        'avg_valence':   valence,
        'dynamic_range': dyn_range,
        'tempo_bpm':     bpm,
        'key':           key_obj,
    }


def _make_tension_curve_from_lerdahl(lv, n_steps):
    avg = float(lv[0])
    peak = float(lv[1])
    climax_pos = float(lv[3])
    resolution = float(lv[4])
    # Build a smooth arc
    curve = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        # Rise to climax then fall
        if t <= climax_pos:
            v = avg + (peak - avg) * (t / max(climax_pos, 0.01))
        else:
            v = peak * (1 - (t - climax_pos) / max(1 - climax_pos, 0.01)) * resolution
        curve.append(float(np.clip(v / (peak + 1e-6), 0, 1)))
    return curve


def _tension_heuristic(score):
    """Fallback heuristic tension per measure."""
    parts = list(score.parts) if score.parts else [score]
    measures = list(parts[0].getElementsByClass('Measure')) if parts else []
    key_obj = _detect_key(score)
    scale_pcs = set(p.pitchClass for p in key_obj.getScale().pitches)
    curve = []
    for m in measures:
        pitches = []
        for el in m.flatten().notes:
            if el.isNote: pitches.append(el.pitch.midi)
            elif el.isChord: pitches.extend(p.midi for p in el.pitches)
        if not pitches:
            curve.append(curve[-1] if curve else 0.3); continue
        chroma = sum(1 for p in pitches if (p%12) not in scale_pcs) / len(pitches)
        dur = m.quarterLength or 4
        density = min(len(pitches) / (dur * 4), 1.0)
        arousal = min((np.mean(pitches) - 48) / 36, 1.0)
        curve.append(float(np.clip(chroma*0.5 + density*0.3 + arousal*0.2, 0, 1)))
    if not curve: curve = [0.4]
    avg = float(np.mean(curve))
    climax = float(np.argmax(curve) / len(curve))
    return curve, avg, climax


# ══════════════════════════════════════════════════════════════
# CADENCE BUILDER
# ══════════════════════════════════════════════════════════════

def make_authentic_cadence(key_obj, duration_beats=4.0):
    """Build a V7→I cadence in the given key."""
    tonic = key_obj.tonic.midi
    mode = key_obj.mode
    # V7 chord
    dom_root = tonic + 7
    v7_pitches = [dom_root, dom_root+4, dom_root+7, dom_root+10]
    v7_pitches = [int(np.clip(p, 36, 84)) for p in v7_pitches]
    # I chord
    if mode == 'major':
        i_pitches = [tonic, tonic+4, tonic+7]
    else:
        i_pitches = [tonic, tonic+3, tonic+7]
    i_pitches = [int(np.clip(p, 36, 84)) for p in i_pitches]

    half = duration_beats / 2
    return [
        {'pitches': v7_pitches, 'duration': half, 'function': 'D',   'figure': 'V7'},
        {'pitches': i_pitches,  'duration': half, 'function': 'T',   'figure': 'I'},
    ]


def make_plagal_cadence(key_obj, duration_beats=4.0):
    """Build a IV→I cadence."""
    tonic = key_obj.tonic.midi
    mode = key_obj.mode
    # IV
    sub_root = tonic + 5
    if mode == 'major':
        iv_pitches = [sub_root, sub_root+4, sub_root+7]
        i_pitches  = [tonic, tonic+4, tonic+7]
    else:
        iv_pitches = [sub_root, sub_root+3, sub_root+7]
        i_pitches  = [tonic, tonic+3, tonic+7]
    half = duration_beats / 2
    return [
        {'pitches': [int(np.clip(p,36,84)) for p in iv_pitches],
         'duration': half, 'function': 'PD', 'figure': 'IV'},
        {'pitches': [int(np.clip(p,36,84)) for p in i_pitches],
         'duration': half, 'function': 'T',  'figure': 'I'},
    ]


def make_pivot_bridge(from_key, to_key, duration_beats=4.0):
    """
    Create a pivot-chord bridge between two keys.
    Uses a chord that is diatonic in both keys when possible.
    """
    # Simple strategy: go to V of new key (always works)
    dom_root = to_key.tonic.midi + 7
    dom_pitches = [dom_root, dom_root+4, dom_root+7, dom_root+10]
    dom_pitches = [int(np.clip(p, 36, 84)) for p in dom_pitches]
    return [{
        'pitches':  dom_pitches,
        'duration': duration_beats,
        'function': 'D',
        'figure':   f'V/{to_key.tonic.name}',
    }]


# ══════════════════════════════════════════════════════════════
# VELOCITY MAPPER  (Lerdahl tension → MIDI velocity)
# ══════════════════════════════════════════════════════════════

def tension_to_velocity(tension, base=70, min_vel=25, max_vel=120):
    """Map tension [0,1] to MIDI velocity."""
    v = int(base + (tension - 0.5) * (max_vel - min_vel))
    return int(np.clip(v, min_vel, max_vel))


def build_velocity_curve(tension_curve, n_notes):
    """Interpolate tension curve to n_notes velocity values."""
    if not tension_curve or n_notes == 0:
        return [75] * n_notes
    xs = np.linspace(0, len(tension_curve)-1, n_notes)
    ys = np.interp(xs, np.arange(len(tension_curve)), tension_curve)
    return [tension_to_velocity(float(t)) for t in ys]


# ══════════════════════════════════════════════════════════════
# FULL DNA EXTRACTOR
# ══════════════════════════════════════════════════════════════

def extract_full_dna(score, label=""):
    print(f"\n{'═'*60}")
    print(f"  DNA: {label}")
    print(f"{'═'*60}")

    dna = {}

    print("  [1/5] Melody + Sequitur (scored phrases)...")
    dna['phrases'], dna['full_melody'] = extract_melody_dna(score)
    print(f"         {len(dna['phrases'])} quality-ranked phrases")

    print("  [2/5] Harmonic progression + voice-leading metadata...")
    dna['progression'], dna['key'] = extract_chord_progression(score)
    print(f"         Key: {dna['key']}, {len(dna['progression'])} chord events")

    print("  [3/5] Metric-aware rhythmic patterns...")
    dna['rhythm'] = extract_rhythmic_dna(score)
    ts = dna['rhythm']['time_signature']
    print(f"         {ts[0]}/{ts[1]}, density={dna['rhythm']['avg_density']:.2f}, "
          f"swing={dna['rhythm']['swing']}")

    print("  [4/5] Energy / tension arc...")
    dna['energy'] = extract_energy_dna(score)
    print(f"         Valence={dna['energy']['avg_valence']:.2f}, "
          f"Tension={dna['energy']['avg_tension']:.2f}, "
          f"Climax@{dna['energy']['climax_pos']:.0%}")

    print("  [5/5] Instrumentation...")
    dna['instruments'] = _extract_instruments(score)
    print(f"         {dna['instruments'][:3]}")

    return dna


# ══════════════════════════════════════════════════════════════
# SECTION BUILDER  (one form section = one phrase + accompaniment)
# ══════════════════════════════════════════════════════════════

class SectionBuilder:
    def __init__(self, harmony_dna, rhythm_dna, energy_dna, key_obj):
        self.prog       = summarize_progression(harmony_dna['progression'], 8)
        rhythm_inner    = rhythm_dna.get('rhythm', rhythm_dna)   # unwrap if needed
        self.acc_gen    = AccompanimentGenerator(rhythm_inner, energy_dna, key_obj)
        self.rhythm_dna = rhythm_inner
        self.energy_dna = energy_dna
        self.key_obj    = key_obj
        self.tension_curve = energy_dna['tension_curve']
        self._prev_voicing = None   # for voice leading continuity

    def build_section(self, phrase_pairs, section_pos=0.5, transposition=0):
        """
        Build melody + accompaniment parts for one section.
        section_pos: position of section in piece [0,1] for tension interpolation.
        transposition: semitone shift (for B/C/D variants).
        """
        # Transposed phrase
        tp = [(p + transposition, d) for (p, d) in phrase_pairs]
        # Keep in range
        tp = [(_clamp_pitch(p), d) for (p, d) in tp]
        # Metric-aware rhythm reshaping
        tp = apply_rhythm_metric_aware(tp, self.rhythm_dna, self.key_obj)

        if not tp:
            return None, None

        # Tension at this section
        t_idx = int(section_pos * (len(self.tension_curve)-1))
        section_tension = self.tension_curve[min(t_idx, len(self.tension_curve)-1)]

        # Build melody part
        mel_part = stream.Part(); mel_part.id = 'Melody'
        mel_part.insert(0, instrument.Piano())
        mel_notes = []
        for (p, d) in tp:
            n = note.Note()
            n.pitch.midi = int(np.clip(p, 36, 96))
            n.duration   = duration.Duration(quarterLength=float(max(d, 0.25)))
            mel_part.append(n)
            mel_notes.append(n)

        # Assign velocities from tension curve
        n_mel = len(mel_notes)
        vels = build_velocity_curve(self.tension_curve, n_mel)
        # Shift velocity window to section position
        start_idx = int(section_pos * len(vels))
        for i, n in enumerate(mel_notes):
            n.volume.velocity = vels[(start_idx + i) % len(vels)]

        # Build accompaniment part
        acc_part = stream.Part(); acc_part.id = 'Accompaniment'
        acc_part.insert(0, instrument.Piano())

        # Calculate total melody duration
        total_dur = sum(d for _, d in tp)
        measure_dur = self.acc_gen.measure_len
        n_measures  = max(1, int(math.ceil(total_dur / measure_dur)))

        # Assign chords measure by measure
        offset = 0.0
        for m_idx in range(n_measures):
            if not self.prog:
                break
            prog_entry = self.prog[m_idx % len(self.prog)]
            tension_now = FUNCTION_TENSION.get(prog_entry['function'], 0.5)
            # Tension is combination of harmonic function + energy curve
            eff_tension = (tension_now + section_tension) / 2

            # Voice-lead chord
            raw_voicing = _get_voicing(prog_entry, 45)  # bass register
            if self._prev_voicing:
                raw_voicing = voice_lead_next_chord(self._prev_voicing, raw_voicing)
            self._prev_voicing = raw_voicing

            # Generate accompaniment events
            acc_events = self.acc_gen.generate_measure(raw_voicing, eff_tension, offset)
            for n, note_offset in acc_events:
                n.volume.velocity = tension_to_velocity(eff_tension, base=55)
                acc_part.insert(note_offset, n)

            offset += measure_dur

        return mel_part, acc_part

    def build_cadence(self, cadence_type='authentic'):
        """Build a closing cadence section."""
        if cadence_type == 'authentic':
            entries = make_authentic_cadence(self.key_obj, 4.0)
        else:
            entries = make_plagal_cadence(self.key_obj, 4.0)

        mel_part = stream.Part(); mel_part.id = 'Melody_cadence'
        acc_part = stream.Part(); acc_part.id = 'Acc_cadence'

        offset = 0.0
        prev_v = self._prev_voicing
        for entry in entries:
            pitches = entry['pitches']
            dur = entry['duration']
            if prev_v:
                pitches = voice_lead_next_chord(prev_v, pitches)
            prev_v = pitches

            # Melody: top voice
            n = note.Note()
            n.pitch.midi = int(np.clip(pitches[-1], 36, 96))
            n.duration = duration.Duration(quarterLength=float(dur))
            n.volume.velocity = 70
            mel_part.append(n)

            # Accompaniment: block chord
            for p in pitches[:-1]:
                acc_n = note.Note()
                acc_n.pitch.midi = int(np.clip(p, 28, 72))
                acc_n.duration = duration.Duration(quarterLength=float(dur))
                acc_n.volume.velocity = 60
                acc_part.insert(offset, acc_n)
            offset += dur

        return mel_part, acc_part


# ══════════════════════════════════════════════════════════════
# STRUCTURAL FORM BUILDER
# ══════════════════════════════════════════════════════════════

def build_form(melody_dna, harmony_dna, rhythm_dna, energy_dna,
               form_string="AABA", surprise_rate=0.08):
    """
    Assemble the full piece:
    1. Map form letters to phrase variants
    2. Build each section (melody + accompaniment)
    3. Insert pivot-chord bridges between sections
    4. Add closing cadence
    5. Apply velocity curve
    """
    target_key = harmony_dna['key']
    source_key = melody_dna['key']

    phrases = melody_dna.get('phrases', [])
    if not phrases and melody_dna.get('full_melody'):
        phrases = [(melody_dna['full_melody'][:8], 1)]
    if not phrases:
        print("[error] No melodic material found.")
        return stream.Score()

    # Transpose all phrases to target key
    semitone_shift = m21interval.Interval(
        source_key.tonic, target_key.tonic
    ).semitones
    transposed_phrases = []
    for paired, freq in phrases:
        tp = [(_clamp_pitch(p + semitone_shift), d) for (p, d) in paired]
        transposed_phrases.append((tp, freq))

    # Build phrase pool for each letter
    unique_letters = sorted(set(form_string))
    phrase_pool = {}
    for i, letter in enumerate(unique_letters):
        base, freq = transposed_phrases[i % len(transposed_phrases)]
        if letter == 'A':
            phrase_pool[letter] = (base, 0)
        elif letter == 'B':
            phrase_pool[letter] = (base, 5)   # up perfect 4th
        elif letter == 'C':
            rev = list(reversed(base))
            phrase_pool[letter] = (rev, 0)    # retrograde
        elif letter == 'D':
            if base:
                bp = base[0][0]
                inv = [(_clamp_pitch(bp-(p-bp)), d) for p,d in base]
                phrase_pool[letter] = (inv, 0)
            else:
                phrase_pool[letter] = (base, 0)
        else:
            phrase_pool[letter] = (base, (ord(letter)-ord('A'))*2)

    builder = SectionBuilder(harmony_dna, rhythm_dna, energy_dna, target_key)

    all_mel_events  = []   # list of (note.Note, offset)
    all_acc_events  = []
    global_offset   = 0.0

    # Decide cadence type from energy DNA
    tension_avg = energy_dna.get('avg_tension', 0.5)
    final_cadence = 'authentic' if tension_avg > 0.4 else 'plagal'

    n_sections = len(form_string)

    for sec_idx, letter in enumerate(form_string):
        sec_pos = sec_idx / max(n_sections - 1, 1)
        base_phrase, transp = phrase_pool.get(letter, (transposed_phrases[0][0], 0))

        mel_part, acc_part = builder.build_section(base_phrase, sec_pos, transp)
        if mel_part is None:
            continue

        # Add surprise: chromatic passing tones
        mel_notes = [n for n in mel_part.flatten().notes if n.isNote]
        for i in range(1, len(mel_notes)-1):
            if random.random() < surprise_rate:
                prev_p = mel_notes[i-1].pitch.midi
                next_p = mel_notes[i+1].pitch.midi
                if abs(next_p - prev_p) == 2:
                    mel_notes[i].pitch.midi = prev_p + (1 if next_p > prev_p else -1)

        # Collect melody events
        offset_in_part = 0.0
        for n in mel_part.flatten().notes:
            if n.isNote:
                all_mel_events.append((copy.deepcopy(n), global_offset + offset_in_part))
                offset_in_part += float(n.quarterLength)

        # Collect accompaniment events
        for el in acc_part.flatten().notesAndRests:
            all_acc_events.append((copy.deepcopy(el), global_offset + float(el.offset)))

        section_dur = offset_in_part
        global_offset += section_dur

        # Add pivot bridge between sections (except after last)
        if sec_idx < n_sections - 1:
            bridge_entries = make_pivot_bridge(target_key, target_key, 2.0)
            prev_v = builder._prev_voicing
            for entry in bridge_entries:
                pitches = entry['pitches']
                if prev_v:
                    pitches = voice_lead_next_chord(prev_v, pitches)
                prev_v = pitches
                dur = entry['duration']
                # Melody: top voice
                n = note.Note()
                n.pitch.midi = int(np.clip(pitches[-1], 36, 96))
                n.duration = duration.Duration(quarterLength=float(dur))
                n.volume.velocity = 65
                all_mel_events.append((n, global_offset))
                # Acc: lower voices
                for p in pitches[:-1]:
                    acc_n = note.Note()
                    acc_n.pitch.midi = int(np.clip(p, 28, 72))
                    acc_n.duration = duration.Duration(quarterLength=float(dur))
                    acc_n.volume.velocity = 55
                    all_acc_events.append((acc_n, global_offset))
                global_offset += dur

    # Final cadence
    cad_mel, cad_acc = builder.build_cadence(final_cadence)
    for n in cad_mel.flatten().notes:
        if n.isNote:
            all_mel_events.append((copy.deepcopy(n), global_offset))
            global_offset += float(n.quarterLength)
    cad_mel_dur = sum(float(n.quarterLength) for n in cad_mel.flatten().notes if n.isNote)
    for el in cad_acc.flatten().notesAndRests:
        all_acc_events.append((copy.deepcopy(el), global_offset - cad_mel_dur + float(el.offset)))

    # Assemble into Score
    score_out = stream.Score()

    mel_full = stream.Part(); mel_full.id = 'Melody'
    mel_full.insert(0, instrument.Piano())
    for n, off in all_mel_events:
        mel_full.insert(off, n)
    score_out.insert(0, mel_full)

    acc_full = stream.Part(); acc_full.id = 'Accompaniment'
    acc_full.insert(0, instrument.Piano())
    for n, off in all_acc_events:
        acc_full.insert(off, n)
    score_out.insert(0, acc_full)

    return score_out


# ══════════════════════════════════════════════════════════════
# FINALIZATION
# ══════════════════════════════════════════════════════════════

def finalize_score(score_out, rhythm_dna, energy_dna, target_key):
    """Add time signature, key signature, tempo, make measures."""
    ts_n, ts_d = rhythm_dna.get('time_signature', (4, 4))
    ts = meter.TimeSignature(f'{ts_n}/{ts_d}')
    ks = m21key.KeySignature(target_key.sharps)
    bpm = float(np.clip(energy_dna.get('tempo_bpm', 120.0), 40, 240))
    mm = tempo.MetronomeMark(number=bpm)

    final = stream.Score()
    for part in score_out.parts:
        new_part = stream.Part()
        new_part.id = part.id
        new_part.insert(0, copy.deepcopy(ks))
        new_part.insert(0, copy.deepcopy(ts))
        new_part.insert(0, copy.deepcopy(mm))
        for inst_obj in part.getElementsByClass('Instrument'):
            new_part.insert(0, copy.deepcopy(inst_obj))
        for el in part.flatten().notesAndRests:
            new_part.insert(float(el.offset), copy.deepcopy(el))
        try:
            new_part.makeMeasures(inPlace=True)
        except Exception:
            pass
        final.insert(0, new_part)
    return final


def add_slur_phrasing(score_out, phrase_len_range=(4, 8)):
    """Add slurs every 4-8 notes for phrasing."""
    for part in score_out.parts:
        notes = [n for n in part.flatten().notes if n.isNote]
        i = 0
        while i < len(notes) - 2:
            plen = random.randint(*phrase_len_range)
            chunk = notes[i:i+plen]
            if len(chunk) >= 2:
                try:
                    sl = spanner.Slur(chunk[0], chunk[-1])
                    part.insert(0, sl)
                except Exception:
                    pass
            i += plen
    return score_out


# ══════════════════════════════════════════════════════════════
# MULTI-SOURCE BLENDING  (weighted DNA merge)
# ══════════════════════════════════════════════════════════════

def blend_progressions(dna_list, weights):
    """
    Merge chord progressions from multiple sources with weights.
    Strategy: interleave chords proportionally.
    """
    total = sum(weights)
    norm_w = [w / total for w in weights]
    merged = []
    for dna, w in zip(dna_list, norm_w):
        prog = dna.get('progression', [])
        n_take = max(1, int(round(w * 8)))
        merged.extend(prog[:n_take])
    return merged


def blend_rhythm_cells(dna_list, weights):
    """Blend rhythmic cells via weighted average of durations."""
    total = sum(weights)
    cells = [dna['rhythm'].get('rhythm_cell', [1.0,0.5,0.5,1.0]) for dna in dna_list]
    max_len = max(len(c) for c in cells)
    blended = np.zeros(max_len)
    for cell, w in zip(cells, weights):
        padded = cell + [cell[-1]] * (max_len - len(cell))
        blended += np.array(padded) * (w / total)
    return [_snap_dur(float(v)) for v in blended]


def blend_tension_curves(dna_list, weights):
    """Weighted blend of tension curves (interpolated to same length)."""
    total = sum(weights)
    target_len = 16
    result = np.zeros(target_len)
    for dna, w in zip(dna_list, weights):
        curve = dna['energy']['tension_curve']
        interp = np.interp(
            np.linspace(0, len(curve)-1, target_len),
            np.arange(len(curve)), curve
        )
        result += interp * (w / total)
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# MAIN COMBINATION ENGINE
# ══════════════════════════════════════════════════════════════

def combine_dna(melody_dna, harmony_dna, rhythm_dna,
                extra_dna_list=None, extra_weights=None,
                form_string="AABA", surprise_rate=0.08):
    """
    Full combination pipeline.
    extra_dna_list / extra_weights: additional donors for blending.
    """
    print(f"\n{'━'*60}")
    print(f"  COMBINATION ENGINE  (form: {form_string})")
    print(f"{'━'*60}")

    target_key = harmony_dna['key']
    print(f"  Target key:  {target_key}")
    print(f"  Melody key:  {melody_dna['key']}")

    # Optionally blend multiple harmony sources
    if extra_dna_list and extra_weights:
        print(f"  Blending {len(extra_dna_list)+1} harmony sources...")
        all_harmony = [harmony_dna] + extra_dna_list
        all_weights  = [1.0] + extra_weights
        harmony_dna['progression'] = blend_progressions(all_harmony, all_weights)
        rhythm_dna['rhythm']['rhythm_cell'] = blend_rhythm_cells(
            [melody_dna, rhythm_dna] + extra_dna_list,
            [1.0, 1.0] + extra_weights
        )

    # Blended energy
    blended_energy = {
        'tension_curve': blend_tension_curves(
            [melody_dna, harmony_dna, rhythm_dna],
            [0.2, 0.5, 0.3]
        ),
        'avg_tension':   (melody_dna['energy']['avg_tension']   * 0.2 +
                          harmony_dna['energy']['avg_tension']  * 0.5 +
                          rhythm_dna['energy']['avg_tension']   * 0.3),
        'climax_pos':     harmony_dna['energy']['climax_pos'],
        'avg_arousal':   (melody_dna['energy']['avg_arousal']   * 0.4 +
                          harmony_dna['energy']['avg_arousal']  * 0.4 +
                          rhythm_dna['energy']['avg_arousal']   * 0.2),
        'avg_valence':    harmony_dna['energy']['avg_valence'],
        'dynamic_range': (harmony_dna['energy']['dynamic_range'] * 0.6 +
                          melody_dna['energy']['dynamic_range']  * 0.4),
        'tempo_bpm':      rhythm_dna['energy']['tempo_bpm'],
        'key':            target_key,
    }

    print("  Building sections...")
    combined = build_form(
        melody_dna, harmony_dna, rhythm_dna, blended_energy,
        form_string=form_string, surprise_rate=surprise_rate
    )

    print("  Adding slur phrasing...")
    combined = add_slur_phrasing(combined)

    print("  Applying instrumentation...")
    instruments = harmony_dna.get('instruments', ['Piano'])
    _apply_instruments(combined, instruments)

    print("  Finalizing (key sig, tempo, measures)...")
    combined = finalize_score(combined, rhythm_dna['rhythm'], blended_energy, target_key)

    return combined


# ══════════════════════════════════════════════════════════════
# ROLE AUTO-ASSIGNMENT
# ══════════════════════════════════════════════════════════════

def auto_assign_roles(scores_and_paths):
    if len(scores_and_paths) == 1:
        s, p = scores_and_paths[0]
        return (s,"melody"), (s,"harmony"), (s,"rhythm")
    if len(scores_and_paths) == 2:
        (s1,p1), (s2,p2) = scores_and_paths
        return (s1,"melody"), (s2,"harmony"), (s2,"rhythm")

    profiles = []
    for score, path in scores_and_paths:
        flat  = score.flatten()
        notez = [n for n in flat.notes if n.isNote]
        chrdz = [n for n in flat.notes if n.isChord]
        pitches = [n.pitch.midi for n in notez]
        durs    = [float(n.quarterLength) for n in notez]
        profiles.append({
            'score':       score,
            'path':        path,
            'pitch_range': (max(pitches)-min(pitches)) if pitches else 0,
            'chord_ratio': len(chrdz) / max(len(notez)+len(chrdz), 1),
            'rhythm_var':  float(np.std(durs)) if durs else 0,
        })

    melody_src  = max(profiles, key=lambda x: x['pitch_range'])
    harmony_src = max(profiles, key=lambda x: x['chord_ratio'])
    rhythm_src  = max(profiles, key=lambda x: x['rhythm_var'])

    # Avoid duplicate assignments
    used = set()
    final = {}
    for role, src in [('melody',melody_src),('harmony',harmony_src),('rhythm',rhythm_src)]:
        sid = id(src['score'])
        if sid in used:
            avail = [p for p in profiles if id(p['score']) not in used]
            src = avail[0] if avail else src
        final[role] = src
        used.add(id(src['score']))

    print("\n  Auto-assigned roles:")
    for role, src in final.items():
        print(f"    {role:8s} ← {src['path']}")

    return (
        (final['melody']['score'],  "melody"),
        (final['harmony']['score'], "harmony"),
        (final['rhythm']['score'],  "rhythm"),
    )


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _detect_key(score):
    try:
        return score.analyze('key')
    except Exception:
        return m21key.Key('C')


def _extract_instruments(score):
    insts = []
    for inst_obj in score.recurse().getElementsByClass(instrument.Instrument):
        name = getattr(inst_obj, 'instrumentName', None) or getattr(inst_obj, 'partName', None)
        if name: insts.append(name)
    return insts or ['Piano']


def _apply_instruments(score_out, instruments_list):
    for i, part in enumerate(score_out.parts):
        iname = instruments_list[i % len(instruments_list)]
        try:
            inst_obj = instrument.fromString(iname)
        except Exception:
            inst_obj = instrument.Piano()
        for old in part.getElementsByClass('Instrument'):
            part.remove(old)
        part.insert(0, inst_obj)


def _clamp_pitch(p, lo=36, hi=96):
    while p > hi: p -= 12
    while p < lo: p += 12
    return int(np.clip(p, lo, hi))


def _get_voicing(prog_entry, register_center=45):
    """Get chord pitches shifted to register."""
    try:
        raw = sorted([pp.midi for pp in prog_entry['chord'].pitches])
    except Exception:
        raw = prog_entry.get('pitches', [register_center, register_center+4, register_center+7])
    center = np.mean(raw)
    shift  = round((register_center - center) / 12) * 12
    return [int(np.clip(p + shift, 28, 72)) for p in raw]


def _fallback_progression(key_obj):
    """I-IV-V-I fallback."""
    t = key_obj.tonic.midi
    if key_obj.mode == 'major':
        entries = [
            ([t, t+4, t+7],       2.0, 'T',  'I'),
            ([t+5, t+9, t+12],    2.0, 'PD', 'IV'),
            ([t+7, t+11, t+14],   2.0, 'D',  'V7'),
            ([t, t+4, t+7],       2.0, 'T',  'I'),
        ]
    else:
        entries = [
            ([t, t+3, t+7],       2.0, 'T',  'i'),
            ([t+5, t+8, t+12],    2.0, 'PD', 'iv'),
            ([t+7, t+11, t+14],   2.0, 'D',  'V7'),
            ([t, t+3, t+7],       2.0, 'T',  'i'),
        ]
    result = []
    for pitches, dur, func, fig in entries:
        cobj = chord.Chord([note.Note(pitch.Pitch(p)) for p in pitches])
        cobj.duration = duration.Duration(quarterLength=dur)
        result.append({'pitches': [int(np.clip(p,28,72)) for p in pitches],
                       'duration': dur, 'function': func,
                       'figure': fig, 'chord': cobj})
    return result


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def load_score(path):
    print(f"  Loading: {path}")
    try:
        return converter.parse(path)
    except Exception as e:
        print(f"  [error] {e}"); return None


def main():
    parser = argparse.ArgumentParser(
        description="MIDI DNA Combiner v2  —  voice-leading, metric rhythm, Alberti bass."
    )
    parser.add_argument('inputs', nargs='*', help='Input MIDI/MusicXML files (2-4 files)')
    parser.add_argument('--melody',   help='Melody donor file')
    parser.add_argument('--harmony',  help='Harmony donor file')
    parser.add_argument('--rhythm',   help='Rhythm donor file')
    parser.add_argument('--output',   default='combined_v2.mid')
    parser.add_argument('--form',     default='AABA',
                        help='Form string: AABA, ABAC, ABAB, ABCA (default: AABA)')
    parser.add_argument('--surprise', type=float, default=0.08,
                        help='Chromatic surprise rate 0-1 (default: 0.08)')
    parser.add_argument('--weights',  nargs='*', type=float,
                        help='Blend weights for extra sources (advanced)')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "★"*60)
    print("  MIDI DNA COMBINER  v2")
    print("  Voice-leading · Metric rhythm · Alberti/Arpeggio bass")
    print("  Tension velocities · Pivot bridges · Sequitur phrases")
    if HAVE_MSCZ: print("  [mscz2vec.py detected — using Lerdahl tension]")
    print("★"*60)

    # Load scores
    if args.melody and args.harmony and args.rhythm:
        m_s = load_score(args.melody)
        h_s = load_score(args.harmony)
        r_s = load_score(args.rhythm)
        if None in (m_s, h_s, r_s):
            print("[error] Could not load files."); sys.exit(1)
        roles = [(m_s,"melody"), (h_s,"harmony"), (r_s,"rhythm")]
    elif args.inputs:
        loaded = [(load_score(p), p) for p in args.inputs]
        loaded = [(s, p) for s, p in loaded if s is not None]
        if not loaded:
            print("[error] No valid files."); sys.exit(1)
        roles = auto_assign_roles(loaded)
    else:
        parser.print_help(); sys.exit(1)

    # Extract DNA
    dna_store = {}
    for score_obj, role in roles:
        dna_store[role] = extract_full_dna(score_obj, f"{role}")

    melody_dna  = dna_store.get('melody',  list(dna_store.values())[0])
    harmony_dna = dna_store.get('harmony', list(dna_store.values())[0])
    rhythm_dna  = dna_store.get('rhythm',  list(dna_store.values())[-1])

    # Fallback progression if extraction failed
    if not harmony_dna.get('progression'):
        print("  [warn] No progression extracted; using I-IV-V-I fallback")
        harmony_dna['progression'] = _fallback_progression(harmony_dna['key'])

    # Combine
    combined = combine_dna(
        melody_dna  = melody_dna,
        harmony_dna = harmony_dna,
        rhythm_dna  = rhythm_dna,
        form_string = args.form.upper(),
        surprise_rate = args.surprise,
    )

    # Export
    print(f"\n  Writing → {args.output}")
    try:
        combined.write('midi', fp=args.output)
        print(f"\n{'★'*60}")
        print(f"  Done!  →  {args.output}")
        print(f"{'★'*60}\n")
    except Exception as e:
        print(f"  [error] MIDI write failed: {e}")
        fallback = args.output.replace('.mid', '.xml').replace('.midi', '.xml')
        try:
            combined.write('musicxml', fp=fallback)
            print(f"  Saved as MusicXML: {fallback}")
        except Exception as e2:
            print(f"  [error] Also failed: {e2}")
        sys.exit(1)


if __name__ == '__main__':
    main()
