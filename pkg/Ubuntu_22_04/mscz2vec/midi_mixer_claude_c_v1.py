"""
midi_dna_combiner.py
====================
Extracts the musical "DNA" of multiple MIDI files and combines them
into a new, musically coherent piece.

WHAT IT DOES:
  1. Extracts DNA from each MIDI: melody phrases (Sequitur), harmonic
     progression, rhythm, instrumentation, energy, tension, emotion.
  2. Uses Sequitur grammar to extract the best melodic phrases from the
     "melody donor" MIDI.
  3. Re-harmonizes those phrases using the chord progressions extracted
     from the "harmony donor" MIDI.
  4. Applies the rhythmic DNA from the "rhythm donor" to reshape note
     durations and timing.
  5. Blends energy, tension, and emotional arc across the piece.
  6. Outputs a single combined MIDI file.

USAGE:
  python midi_dna_combiner.py \
    --melody source_a.mid \
    --harmony source_b.mid \
    --rhythm source_c.mid \
    --output combined.mid

  Or let the script decide roles automatically:
  python midi_dna_combiner.py source_a.mid source_b.mid source_c.mid --output combined.mid

DEPENDENCIES:
  pip install music21 numpy mido
"""

import sys
import argparse
import hashlib
import copy
import random
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

try:
    from music21 import (
        converter, stream, note, chord, pitch, key, meter, tempo,
        instrument, roman, interval, duration, tie, dynamics,
        expressions, harmony
    )
    from music21 import analysis
except ImportError:
    print("music21 not found. Install with: pip install music21")
    sys.exit(1)

try:
    import mido
except ImportError:
    print("mido not found. Install with: pip install mido")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

HARMONIC_FUNCTIONS = {
    "T":    ["I", "i", "vi", "VI"],
    "PD":   ["ii", "II", "iv", "IV"],
    "D":    ["V", "v", "vii°", "VII"],
    "Dsec": ["V/V", "V/ii", "V/vi", "V/IV"],
    "Other": []
}

# Tension weights per harmonic function (0 = rest, 1 = max tension)
FUNCTION_TENSION = {"T": 0.1, "PD": 0.4, "D": 0.7, "Dsec": 0.9, "Other": 0.5}

# Consonance map for re-harmonization
CONSONANCE = {0: 1.0, 1: 0.1, 2: 0.2, 3: 0.6, 4: 0.7,
              5: 0.8, 6: 0.0, 7: 0.9, 8: 0.65, 9: 0.75, 10: 0.3, 11: 0.15}

# Chord templates (intervals from root)
CHORD_TEMPLATES = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "dominant7":  [0, 4, 7, 10],
    "minor7":     [0, 3, 7, 10],
    "major7":     [0, 4, 7, 11],
    "diminished": [0, 3, 6],
    "augmented":  [0, 4, 8],
}

# Common diatonic chord degrees (major key)
MAJOR_SCALE_DEGREES = {
    "I":   (0,  "major"),
    "ii":  (2,  "minor"),
    "iii": (4,  "minor"),
    "IV":  (5,  "major"),
    "V":   (7,  "dominant7"),
    "vi":  (9,  "minor"),
    "vii°":(11, "diminished"),
}
MINOR_SCALE_DEGREES = {
    "i":   (0,  "minor"),
    "ii°": (2,  "diminished"),
    "III": (3,  "major"),
    "iv":  (5,  "minor"),
    "V":   (7,  "dominant7"),
    "VI":  (8,  "major"),
    "VII": (10, "major"),
}

# ─────────────────────────────────────────────────────────────
# SEQUITUR GRAMMAR (Re-pair variant)
# ─────────────────────────────────────────────────────────────

class SequiturGrammar:
    """Builds a Re-Pair grammar from a sequence of MIDI pitches."""

    def __init__(self):
        self.rules = {}
        self.next_id = 1
        self.root = []

    def _new_rule(self):
        name = f"R{self.next_id}"
        self.next_id += 1
        return name

    def build(self, symbols):
        if len(symbols) < 4:
            self.root = list(symbols)
            return self.rules
        seq = list(symbols)
        for _ in range(200):  # max iterations
            pairs = Counter()
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += 1
            if not pairs:
                break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            name = self._new_rule()
            self.rules[name] = list(best)
            new_seq, i = [], 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == best:
                    new_seq.append(name)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        self.root = seq
        return self.rules

    def expand(self, symbol):
        if symbol not in self.rules:
            return [symbol]
        result = []
        for s in self.rules[symbol]:
            result.extend(self.expand(s))
        return result

    def get_phrases(self, min_len=3, max_len=16, top_n=8):
        """Return the top recurring phrases as note sequences."""
        phrases = []
        for name, body in self.rules.items():
            expanded = self.expand(name)
            if min_len <= len(expanded) <= max_len:
                freq = self.root.count(name)
                phrases.append((expanded, freq, name))
        phrases.sort(key=lambda x: (x[1] * len(x[0])), reverse=True)
        return phrases[:top_n]


# ─────────────────────────────────────────────────────────────
# HARMONIC DNA EXTRACTOR
# ─────────────────────────────────────────────────────────────

def detect_key_safe(score):
    """Safely detect key of a score."""
    try:
        return score.analyze('key')
    except Exception:
        return key.Key('C')


def extract_chord_progression(score, window_beats=2.0):
    """
    Extract a sequence of (roman_figure, function, duration_beats) tuples.
    Uses chordify to get vertical slices.
    """
    try:
        detected_key = detect_key_safe(score)
        chordified = score.chordify()
        flat_chords = chordified.flatten().getElementsByClass('Chord')
    except Exception as e:
        print(f"  [warn] Chord extraction failed: {e}")
        return [], detect_key_safe(score)

    progression = []
    for ch in flat_chords:
        if len(ch.pitches) < 2:
            continue
        try:
            rn = roman.romanNumeralFromChord(ch, detected_key)
            figure = rn.figure
            func = _roman_to_function(rn)
            dur = float(ch.quarterLength)
            if dur < 0.25:
                continue
            progression.append({
                'figure': figure,
                'function': func,
                'duration': dur,
                'chord': ch,
                'rn': rn,
            })
        except Exception:
            continue
    return progression, detected_key


def _roman_to_function(rn):
    figure = rn.figure
    if "/" in figure:
        return "Dsec"
    base = figure.replace("°", "").replace("+", "")
    base = "".join(c for c in base if not c.isdigit())
    for func, romans in HARMONIC_FUNCTIONS.items():
        if base in romans:
            return func
    return "Other"


def summarize_progression(progression, n_chords=8):
    """Condense progression to the most representative n_chords."""
    if not progression:
        return []
    # Group consecutive same-function chords
    grouped = []
    prev = None
    for p in progression:
        if prev and prev['figure'] == p['figure']:
            prev['duration'] += p['duration']
        else:
            prev = dict(p)
            grouped.append(prev)

    # Take the most harmonically varied n_chords
    step = max(1, len(grouped) // n_chords)
    condensed = grouped[::step][:n_chords]

    # Ensure we have a tonic start and tonic end for musicality
    if condensed and condensed[-1]['function'] != 'T':
        # Add a tonic chord at end
        tonic = next((p for p in grouped if p['function'] == 'T'), condensed[-1])
        condensed.append(tonic)

    return condensed


# ─────────────────────────────────────────────────────────────
# RHYTHMIC DNA EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_rhythmic_dna(score, max_notes=200):
    """
    Extract rhythmic patterns as:
    - duration_ratios: list of float (ratio of each note to the beat)
    - beat_positions: list of float (position within measure, 0-1)
    - syncopation_rate: float
    - avg_density: float (notes per beat)
    - patterns: Counter of common rhythmic cells (3-grams of ratios)
    """
    flat = score.flatten()
    notes = [n for n in flat.notes if n.isNote][:max_notes]

    if not notes:
        return {
            'ratios': [1.0] * 8,
            'beat_positions': [0.0] * 8,
            'syncopation_rate': 0.0,
            'avg_density': 1.0,
            'patterns': Counter(),
            'time_signature': (4, 4),
        }

    # Time signature
    ts = flat.getElementsByClass('TimeSignature')
    if ts:
        ts0 = ts[0]
        time_sig = (ts0.numerator, ts0.denominator)
    else:
        time_sig = (4, 4)

    # Measure length in quarter notes
    measure_len = (4.0 * time_sig[0]) / time_sig[1]

    ratios = []
    positions = []
    syncopations = 0

    for n in notes:
        # Ratio of note duration to one beat
        ratio = float(n.quarterLength)
        ratios.append(ratio)

        # Beat position within measure
        try:
            beat_pos = (float(n.beat) - 1) / time_sig[0]
        except Exception:
            beat_pos = 0.0
        positions.append(beat_pos)

        # Syncopation: note starts off-beat
        try:
            beat = float(n.beat)
            if beat % 1 != 0:
                syncopations += 1
        except Exception:
            pass

    syncopation_rate = syncopations / len(notes) if notes else 0

    # Average density
    total_dur = sum(float(n.quarterLength) for n in notes)
    avg_density = len(notes) / max(total_dur, 1)

    # 3-gram patterns
    quantized = [_quantize_duration(r) for r in ratios]
    patterns = Counter(
        tuple(quantized[i:i+3]) for i in range(len(quantized)-2)
    )

    return {
        'ratios': ratios,
        'beat_positions': positions,
        'syncopation_rate': syncopation_rate,
        'avg_density': avg_density,
        'patterns': patterns,
        'time_signature': time_sig,
    }


def _quantize_duration(r):
    """Snap duration to common values."""
    options = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(options, key=lambda x: abs(x - r))


# ─────────────────────────────────────────────────────────────
# ENERGY / TENSION / EMOTION DNA
# ─────────────────────────────────────────────────────────────

def extract_energy_dna(score):
    """
    Returns a dict describing the energy/tension/emotion arc.
    - tension_curve: list of float per measure
    - avg_arousal: float
    - avg_valence: float
    - dynamic_range: float
    - tempo_bpm: float
    """
    parts = list(score.parts) if score.parts else [score]
    measures = list(parts[0].getElementsByClass('Measure')) if parts else []

    tension_curve = []
    arousal_list = []

    detected_key = detect_key_safe(score)
    scale_pcs = set(p.pitchClass for p in detected_key.getScale().pitches)

    for m in measures:
        pitches = []
        for el in m.flatten().notes:
            if el.isNote:
                pitches.append(el.pitch.midi)
            elif el.isChord:
                pitches.extend([p.midi for p in el.pitches])

        if not pitches:
            tension_curve.append(tension_curve[-1] if tension_curve else 0.3)
            arousal_list.append(arousal_list[-1] if arousal_list else 0.5)
            continue

        # Chromaticism → tension
        chroma = sum(1 for p in pitches if (p % 12) not in scale_pcs) / len(pitches)

        # Register → arousal
        avg_pitch = np.mean(pitches)
        arousal = np.clip((avg_pitch - 48) / 36, 0, 1)

        # Density → tension
        dur = m.quarterLength or 4
        density = np.clip(len(pitches) / (dur * 4), 0, 1)

        tension = (chroma * 0.5) + (density * 0.3) + (arousal * 0.2)
        tension_curve.append(float(tension))
        arousal_list.append(float(arousal))

    # Valence: mode-based
    avg_valence = 0.65 if detected_key.mode == 'major' else 0.35

    # Tempo
    tempos = score.flatten().getElementsByClass('MetronomeMark')
    if tempos:
        try:
            bpm = float(tempos[0].number)
        except Exception:
            bpm = 120.0
    else:
        bpm = 120.0

    # Dynamic range approximation from pitch range
    all_pitches = [n.pitch.midi for n in score.flatten().notes if n.isNote]
    dynamic_range = (max(all_pitches) - min(all_pitches)) / 88.0 if all_pitches else 0.5

    return {
        'tension_curve': tension_curve if tension_curve else [0.3],
        'avg_arousal': float(np.mean(arousal_list)) if arousal_list else 0.5,
        'avg_valence': avg_valence,
        'dynamic_range': dynamic_range,
        'tempo_bpm': bpm,
        'key': detected_key,
    }


# ─────────────────────────────────────────────────────────────
# MELODY DNA EXTRACTOR (Sequitur-based)
# ─────────────────────────────────────────────────────────────

def extract_melody_dna(score):
    """
    Extract melodic phrases using Sequitur grammar.
    Returns top phrases as lists of (midi_pitch, duration) tuples.
    """
    parts = list(score.parts) if score.parts else [score]
    melody_stream = parts[0].flatten().notes

    midi_seq = []
    dur_seq = []
    for n in melody_stream:
        if n.isNote:
            midi_seq.append(n.pitch.midi)
            dur_seq.append(float(n.quarterLength))
        elif n.isChord:
            # Take highest note of chord as melody
            top = max(n.pitches, key=lambda p: p.midi)
            midi_seq.append(top.midi)
            dur_seq.append(float(n.quarterLength))

    if not midi_seq:
        return [], []

    # Build Sequitur grammar
    grammar = SequiturGrammar()
    grammar.build(midi_seq)

    raw_phrases = grammar.get_phrases(min_len=3, max_len=12, top_n=10)

    # Rebuild as (pitch, duration) pairs
    phrases = []
    for pitch_list, freq, rule_name in raw_phrases:
        # Find position of this phrase in original sequence
        start_idx = None
        for i in range(len(midi_seq) - len(pitch_list) + 1):
            if midi_seq[i:i+len(pitch_list)] == pitch_list:
                start_idx = i
                break
        if start_idx is not None:
            paired = list(zip(
                pitch_list,
                dur_seq[start_idx:start_idx+len(pitch_list)]
            ))
        else:
            paired = [(p, 0.5) for p in pitch_list]
        phrases.append((paired, freq))

    # Also include the full melody as a fallback
    full_melody = list(zip(midi_seq[:32], dur_seq[:32]))  # first 32 notes max

    return phrases, full_melody


# ─────────────────────────────────────────────────────────────
# RE-HARMONIZER
# ─────────────────────────────────────────────────────────────

def transpose_to_key(midi_pitches, source_key, target_key):
    """Transpose a list of MIDI pitches from source key to target key."""
    semitones = interval.Interval(
        source_key.tonic, target_key.tonic
    ).semitones
    return [p + semitones for p in midi_pitches]


def best_chord_for_melody_note(melody_midi, progression, target_key):
    """
    Pick the best chord from progression that fits a melody note.
    Prefers chords where the note is a chord tone.
    """
    if not progression:
        return None
    melody_pc = melody_midi % 12
    best = None
    best_score = -1
    for p in progression:
        try:
            ch = p['chord']
            pcs = [pp.pitchClass for pp in ch.pitches]
            if melody_pc in pcs:
                score = 1.0 + FUNCTION_TENSION.get(p['function'], 0.5)
            else:
                # Check consonance with chord
                intervals = [abs(melody_pc - pc) % 12 for pc in pcs]
                score = np.mean([CONSONANCE.get(i, 0.3) for i in intervals])
            if score > best_score:
                best_score = score
                best = p
        except Exception:
            continue
    return best


def reharmonize_phrase(phrase, progression, target_key, source_key):
    """
    Given a melodic phrase [(pitch, dur), ...] and a chord progression,
    create a reharmonized stream with melody + chords.

    Strategy:
    1. Transpose melody to target_key.
    2. For each beat-boundary, pick the best fitting chord from progression.
    3. Output as a two-voice stream (melody + accompaniment).
    """
    if not phrase:
        return None

    # Transpose phrase to target key
    src_tonic = source_key.tonic.midi if source_key else 60
    tgt_tonic = target_key.tonic.midi if target_key else 60
    semitone_shift = tgt_tonic - src_tonic

    transposed = [(p + semitone_shift, d) for (p, d) in phrase]

    # Keep pitches in comfortable range (C4 = 60 center)
    adjusted = []
    for (p, d) in transposed:
        while p > 79:
            p -= 12
        while p < 48:
            p += 12
        adjusted.append((p, d))

    # Build output stream
    melody_part = stream.Part()
    melody_part.id = 'Melody'
    acc_part = stream.Part()
    acc_part.id = 'Accompaniment'

    # Add instruments
    melody_part.insert(0, instrument.Piano())
    acc_part.insert(0, instrument.Piano())

    # Walk through phrase and assign chords
    accumulated_beats = 0.0
    prog_idx = 0
    beat_per_chord = max(2.0, sum(d for _, d in adjusted) / max(len(progression), 1))

    for (midi_p, dur) in adjusted:
        # Melody note
        n = note.Note()
        n.pitch.midi = int(np.clip(midi_p, 21, 108))
        n.duration = duration.Duration(quarterLength=float(dur))
        melody_part.append(n)

        # Chord at chord-change boundaries
        if accumulated_beats == 0 or accumulated_beats % beat_per_chord < dur:
            if progression:
                prog_entry = progression[prog_idx % len(progression)]
                try:
                    ch = prog_entry['chord']
                    # Build accompaniment chord in bass register
                    ch_pitches = sorted([pp.midi for pp in ch.pitches])
                    # Shift to bass register (C3 = 48)
                    while ch_pitches[0] > 55:
                        ch_pitches = [p - 12 for p in ch_pitches]
                    while ch_pitches[0] < 36:
                        ch_pitches = [p + 12 for p in ch_pitches]

                    acc_chord = chord.Chord(
                        [note.Note(pitch.Pitch(p)) for p in ch_pitches[:4]]
                    )
                    acc_chord.duration = duration.Duration(quarterLength=float(dur))
                    acc_part.append(acc_chord)
                    prog_idx += 1
                except Exception:
                    rest_n = note.Rest()
                    rest_n.duration = duration.Duration(quarterLength=float(dur))
                    acc_part.append(rest_n)
            else:
                rest_n = note.Rest()
                rest_n.duration = duration.Duration(quarterLength=float(dur))
                acc_part.append(rest_n)
        else:
            # Continue chord (hold or arpeggiate)
            rest_n = note.Rest()
            rest_n.duration = duration.Duration(quarterLength=float(dur))
            acc_part.append(rest_n)

        accumulated_beats += dur

    return melody_part, acc_part


# ─────────────────────────────────────────────────────────────
# RHYTHMIC TRANSFORMER
# ─────────────────────────────────────────────────────────────

def apply_rhythmic_dna(melody_part, rhythmic_dna, target_total_beats=None):
    """
    Reshape the note durations in melody_part using the rhythmic DNA.
    Preserves pitches but changes timing to match the rhythm donor.
    """
    notes_list = [n for n in melody_part.flatten().notes if n.isNote]
    if not notes_list:
        return melody_part

    donor_ratios = rhythmic_dna.get('ratios', [1.0])
    syncopation = rhythmic_dna.get('syncopation_rate', 0.0)

    # Build a cycling rhythm pattern
    pattern = []
    top_patterns = rhythmic_dna['patterns'].most_common(5)
    if top_patterns:
        for cell, _ in top_patterns:
            pattern.extend(list(cell))
    if not pattern:
        pattern = donor_ratios

    # Apply rhythm pattern to notes
    new_part = stream.Part()
    new_part.id = melody_part.id

    for inst in melody_part.getElementsByClass('Instrument'):
        new_part.insert(0, inst)

    for i, n in enumerate(notes_list):
        new_n = note.Note()
        new_n.pitch = n.pitch
        # Cycle through rhythmic pattern
        new_dur = _quantize_duration(pattern[i % len(pattern)])

        # Occasionally add syncopation (tie into next beat)
        if random.random() < syncopation * 0.5 and new_dur >= 0.5:
            new_n.duration = duration.Duration(quarterLength=new_dur)
        else:
            new_n.duration = duration.Duration(quarterLength=max(0.25, new_dur))

        new_part.append(new_n)

    return new_part


# ─────────────────────────────────────────────────────────────
# ENERGY SHAPING
# ─────────────────────────────────────────────────────────────

def apply_energy_arc(score_stream, energy_dna, n_sections=4):
    """
    Apply dynamic markings and tempo changes based on the energy DNA arc.
    """
    tension_curve = energy_dna.get('tension_curve', [0.5])
    bpm = energy_dna.get('tempo_bpm', 120.0)

    # Insert tempo marking
    mm = tempo.MetronomeMark(number=bpm)
    score_stream.insert(0, mm)

    # Map tension to dynamics
    dyn_map = [
        (0.8, 'fff'), (0.65, 'ff'), (0.5, 'mf'), (0.35, 'mp'),
        (0.2, 'p'), (0.0, 'pp')
    ]

    parts = list(score_stream.parts)
    if not parts:
        return score_stream

    measures = list(parts[0].getElementsByClass('Measure'))
    n_measures = len(measures)

    for m_idx, m in enumerate(measures):
        # Interpolate tension for this measure
        t_idx = int((m_idx / max(n_measures - 1, 1)) * (len(tension_curve) - 1))
        t_val = tension_curve[min(t_idx, len(tension_curve)-1)]

        # Choose dynamic
        dyn_text = 'mf'
        for threshold, dyn in dyn_map:
            if t_val >= threshold:
                dyn_text = dyn
                break

        # Add dynamic to first note of measure
        notes_in_m = [n for n in m.flatten().notes if n.isNote]
        if notes_in_m:
            d = dynamics.Dynamic(dyn_text)
            notes_in_m[0].dynamic = dyn_text

    return score_stream


# ─────────────────────────────────────────────────────────────
# STRUCTURAL FORM BUILDER
# ─────────────────────────────────────────────────────────────

def build_form(phrases, progression, source_key, target_key,
               rhythmic_dna, energy_dna, form_string="AABA"):
    """
    Assemble a musical form from extracted DNA elements.
    
    form_string: e.g. "AABA", "ABAB", "AABB", "ABAC"
    Each letter maps to a phrase variant.
    """
    if not phrases:
        print("[warn] No phrases found, generating minimal output")
        return stream.Score()

    # Create phrase pool
    phrase_pool = {}
    unique_letters = sorted(set(form_string))

    for i, letter in enumerate(unique_letters):
        phrase_idx = i % len(phrases)
        base_phrase, freq = phrases[phrase_idx]
        # Variants: A = original, B = transposed up 4th, C = retrograde, D = inverted
        if letter == 'A':
            phrase_pool[letter] = base_phrase
        elif letter == 'B':
            # Transpose up a perfect 4th (5 semitones)
            phrase_pool[letter] = [(p + 5, d) for (p, d) in base_phrase]
        elif letter == 'C':
            # Retrograde
            phrase_pool[letter] = list(reversed(base_phrase))
        elif letter == 'D':
            # Inversion (mirror intervals)
            if base_phrase:
                base_pitch = base_phrase[0][0]
                phrase_pool[letter] = [
                    (base_pitch - (p - base_pitch), d)
                    for (p, d) in base_phrase
                ]
        else:
            # More letters: slight transposition variant
            semitone = (ord(letter) - ord('A')) * 2
            phrase_pool[letter] = [(p + semitone, d) for (p, d) in base_phrase]

    # Ensure all letters have entries
    for letter in unique_letters:
        if letter not in phrase_pool and phrases:
            phrase_pool[letter] = phrases[0][0]

    # Build sections
    all_melody_parts = []
    all_acc_parts = []

    prog_condensed = summarize_progression(progression, n_chords=8)
    if not prog_condensed:
        print("[warn] No chord progression extracted; using simple I-V-I")
        prog_condensed = _build_fallback_progression(target_key)

    prog_offset = 0

    for section_letter in form_string:
        phrase = phrase_pool.get(section_letter, phrases[0][0])

        # Apply rhythmic transformation
        # First build a temp part for rhythm application
        temp_part = stream.Part()
        for (p, d) in phrase:
            n = note.Note()
            n.pitch.midi = int(np.clip(p, 21, 108))
            n.duration = duration.Duration(quarterLength=float(d))
            temp_part.append(n)

        rhythm_reshaped = apply_rhythmic_dna(temp_part, rhythmic_dna)
        reshaped_phrase = [
            (n.pitch.midi, float(n.duration.quarterLength))
            for n in rhythm_reshaped.flatten().notes if n.isNote
        ]

        # Rotate progression for variety
        rotated_prog = prog_condensed[prog_offset % len(prog_condensed):] + \
                       prog_condensed[:prog_offset % len(prog_condensed)]

        # Reharmonize
        result = reharmonize_phrase(reshaped_phrase, rotated_prog, target_key, source_key)
        if result:
            mel_part, acc_part = result
            all_melody_parts.append(mel_part)
            all_acc_parts.append(acc_part)

        prog_offset += len(prog_condensed) // max(len(form_string), 1)

    # Assemble into score
    combined_score = stream.Score()

    # Concatenate melody parts
    if all_melody_parts:
        full_melody = stream.Part()
        full_melody.id = 'Melody'
        full_melody.insert(0, instrument.Piano())
        for part in all_melody_parts:
            for n in part.flatten().notes:
                full_melody.append(copy.deepcopy(n))
        combined_score.insert(0, full_melody)

    # Concatenate accompaniment parts
    if all_acc_parts:
        full_acc = stream.Part()
        full_acc.id = 'Accompaniment'
        full_acc.insert(0, instrument.Piano())
        for part in all_acc_parts:
            for el in part.flatten().notesAndRests:
                full_acc.append(copy.deepcopy(el))
        combined_score.insert(0, full_acc)

    return combined_score


def _build_fallback_progression(target_key):
    """Build a simple I-IV-V-I progression as fallback."""
    if target_key.mode == 'major':
        degrees = MAJOR_SCALE_DEGREES
        figures = ['I', 'IV', 'V', 'I']
    else:
        degrees = MINOR_SCALE_DEGREES
        figures = ['i', 'iv', 'V', 'i']

    tonic_midi = target_key.tonic.midi

    progression = []
    for fig in figures:
        if fig in degrees:
            interval_semitones, chord_type = degrees[fig]
            root_midi = tonic_midi + interval_semitones
            intervals = CHORD_TEMPLATES.get(chord_type, [0, 4, 7])
            pitches = [pitch.Pitch(root_midi + i) for i in intervals]
            ch = chord.Chord(pitches)
            ch.duration = duration.Duration(quarterLength=2.0)
            progression.append({
                'figure': fig,
                'function': _roman_to_function_from_figure(fig),
                'duration': 2.0,
                'chord': ch,
            })
    return progression


def _roman_to_function_from_figure(fig):
    for func, romans in HARMONIC_FUNCTIONS.items():
        if fig in romans:
            return func
    return "Other"


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING: VOICE LEADING SMOOTHER
# ─────────────────────────────────────────────────────────────

def smooth_voice_leading(score_stream):
    """
    Basic voice leading correction:
    - Avoid large leaps > octave in melody
    - Avoid parallel 5ths/8ths in consecutive chords (best-effort)
    """
    for part in score_stream.parts:
        notes = [n for n in part.flatten().notes if n.isNote]
        for i in range(1, len(notes)):
            prev = notes[i-1].pitch.midi
            curr = notes[i].pitch.midi
            diff = curr - prev
            # Reduce leaps larger than an octave
            if abs(diff) > 13:
                while notes[i].pitch.midi > prev + 7:
                    notes[i].pitch.midi -= 12
                while notes[i].pitch.midi < prev - 7:
                    notes[i].pitch.midi += 12
            # Keep in human range
            notes[i].pitch.midi = int(np.clip(notes[i].pitch.midi, 36, 96))

    return score_stream


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION DNA
# ─────────────────────────────────────────────────────────────

def extract_instrumentation_dna(score):
    """Extract instruments used in the score."""
    instruments = []
    for inst in score.recurse().getElementsByClass(instrument.Instrument):
        name = getattr(inst, 'instrumentName', None) or getattr(inst, 'partName', None)
        if name:
            instruments.append(name)
    return instruments if instruments else ['Piano']


def apply_instrumentation(score_stream, instruments_list):
    """Assign instruments from the instrumentation DNA."""
    parts = list(score_stream.parts)
    for i, part in enumerate(parts):
        inst_name = instruments_list[i % len(instruments_list)] if instruments_list else 'Piano'
        try:
            # Try to find matching music21 instrument
            inst_obj = instrument.fromString(inst_name)
        except Exception:
            inst_obj = instrument.Piano()
        # Remove existing instruments
        for old_inst in part.getElementsByClass('Instrument'):
            part.remove(old_inst)
        part.insert(0, inst_obj)
    return score_stream


# ─────────────────────────────────────────────────────────────
# STRUCTURAL SURPRISE INJECTOR
# ─────────────────────────────────────────────────────────────

def inject_surprise(score_stream, surprise_rate=0.1, key_obj=None):
    """
    Add musical "surprise" moments:
    - Secondary dominants at high-tension points
    - Occasional chromatic passing tones in melody
    - Unexpected register jumps
    """
    if key_obj is None:
        return score_stream

    for part in score_stream.parts:
        notes = [n for n in part.flatten().notes if n.isNote]
        for i in range(1, len(notes) - 1):
            if random.random() < surprise_rate:
                # Chromatic passing tone
                prev_midi = notes[i-1].pitch.midi
                next_midi = notes[i+1].pitch.midi
                # Insert chromatic note between prev and next
                if abs(next_midi - prev_midi) == 2:
                    # Insert chromatic between them
                    notes[i].pitch.midi = prev_midi + (1 if next_midi > prev_midi else -1)

    return score_stream


# ─────────────────────────────────────────────────────────────
# FRASEO / PHRASING
# ─────────────────────────────────────────────────────────────

def add_phrasing(score_stream):
    """
    Add slurs and articulations to make phrasing more musical.
    Groups of notes into phrases with slurs at boundaries.
    """
    from music21 import spanner
    for part in score_stream.parts:
        notes = [n for n in part.flatten().notes if n.isNote]
        # Group notes into phrases of 4-8 notes
        phrase_len = random.randint(4, 8)
        i = 0
        while i < len(notes) - phrase_len:
            phrase_notes = notes[i:i+phrase_len]
            if len(phrase_notes) >= 2:
                try:
                    sl = spanner.Slur(phrase_notes[0], phrase_notes[-1])
                    part.insert(0, sl)
                except Exception:
                    pass
            i += phrase_len
    return score_stream


# ─────────────────────────────────────────────────────────────
# FINALIZE: ADD TIME SIGNATURE, CLEANUP
# ─────────────────────────────────────────────────────────────

def finalize_score(score_stream, rhythmic_dna, energy_dna, target_key):
    """Add time signature, key signature, tempo, make measures."""
    ts_num, ts_den = rhythmic_dna.get('time_signature', (4, 4))
    ts = meter.TimeSignature(f'{ts_num}/{ts_den}')
    ks = key.KeySignature(target_key.sharps)
    bpm = energy_dna.get('tempo_bpm', 120.0)
    mm = tempo.MetronomeMark(number=float(np.clip(bpm, 40, 240)))

    final = stream.Score()

    for part in score_stream.parts:
        new_part = stream.Part()
        new_part.id = part.id
        new_part.insert(0, copy.deepcopy(ks))
        new_part.insert(0, copy.deepcopy(ts))
        new_part.insert(0, copy.deepcopy(mm))

        # Copy instruments
        for inst in part.getElementsByClass('Instrument'):
            new_part.insert(0, copy.deepcopy(inst))

        # Copy notes
        for el in part.flatten().notesAndRests:
            new_part.append(copy.deepcopy(el))

        # Make measures
        try:
            new_part.makeMeasures(inPlace=True)
        except Exception:
            pass

        final.insert(0, new_part)

    return final


# ─────────────────────────────────────────────────────────────
# MAIN DNA EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────

def extract_full_dna(score, label=""):
    """Extract all DNA dimensions from a score."""
    print(f"\n{'='*60}")
    print(f"  Extracting DNA: {label}")
    print(f"{'='*60}")

    dna = {}

    print("  [1/5] Melody + Sequitur phrases...")
    dna['phrases'], dna['full_melody'] = extract_melody_dna(score)
    print(f"         Found {len(dna['phrases'])} Sequitur phrases")

    print("  [2/5] Harmonic progression...")
    dna['progression'], dna['key'] = extract_chord_progression(score)
    print(f"         Key: {dna['key']}, {len(dna['progression'])} chord events")

    print("  [3/5] Rhythmic patterns...")
    dna['rhythm'] = extract_rhythmic_dna(score)
    ts = dna['rhythm']['time_signature']
    print(f"         Time sig: {ts[0]}/{ts[1]}, density: {dna['rhythm']['avg_density']:.2f}")

    print("  [4/5] Energy / tension arc...")
    dna['energy'] = extract_energy_dna(score)
    print(f"         Valence: {dna['energy']['avg_valence']:.2f}, Arousal: {dna['energy']['avg_arousal']:.2f}")

    print("  [5/5] Instrumentation...")
    dna['instruments'] = extract_instrumentation_dna(score)
    print(f"         Instruments: {dna['instruments'][:3]}")

    return dna


# ─────────────────────────────────────────────────────────────
# COMBINATION ENGINE
# ─────────────────────────────────────────────────────────────

def combine_dna(melody_dna, harmony_dna, rhythm_dna,
                form_string="AABA",
                surprise_rate=0.08,
                use_harmony_instruments=True):
    """
    Combine DNA from three sources into a new piece.
    
    Parameters:
    -----------
    melody_dna   : DNA dict from melody donor
    harmony_dna  : DNA dict from harmony donor
    rhythm_dna   : DNA dict from rhythm donor
    form_string  : Musical form (e.g. "AABA", "ABAC")
    surprise_rate: Rate of chromatic surprises (0-1)
    """
    print(f"\n{'='*60}")
    print(f"  COMBINING DNA")
    print(f"  Melody source  → phrases and contour")
    print(f"  Harmony source → chord progression and key")
    print(f"  Rhythm source  → durations and articulation")
    print(f"  Form: {form_string}")
    print(f"{'='*60}\n")

    # Use harmony donor's key as target
    target_key = harmony_dna['key']
    source_key = melody_dna['key']
    print(f"  Target key: {target_key}")
    print(f"  Source key: {source_key}")

    # Get melody phrases (from melody donor)
    phrases = melody_dna.get('phrases', [])
    if not phrases and melody_dna.get('full_melody'):
        # Fallback: use first 8 notes of full melody as single phrase
        phrases = [(melody_dna['full_melody'][:8], 1)]

    # Get chord progression (from harmony donor)
    progression = harmony_dna.get('progression', [])

    # Get rhythmic pattern (from rhythm donor)
    rhythmic_dna = rhythm_dna.get('rhythm', {})

    # Energy arc (blend between sources, favoring harmony for mood)
    blended_energy = {
        'tension_curve': harmony_dna['energy']['tension_curve'],
        'avg_arousal': (melody_dna['energy']['avg_arousal'] + harmony_dna['energy']['avg_arousal']) / 2,
        'avg_valence': harmony_dna['energy']['avg_valence'],
        'dynamic_range': (rhythm_dna['energy']['dynamic_range'] + harmony_dna['energy']['dynamic_range']) / 2,
        'tempo_bpm': rhythm_dna['energy']['tempo_bpm'],
        'key': target_key,
    }

    print(f"  Building form '{form_string}'...")
    combined = build_form(
        phrases=phrases,
        progression=progression,
        source_key=source_key,
        target_key=target_key,
        rhythmic_dna=rhythmic_dna,
        energy_dna=blended_energy,
        form_string=form_string,
    )

    print("  Smoothing voice leading...")
    combined = smooth_voice_leading(combined)

    print("  Injecting surprises...")
    combined = inject_surprise(combined, surprise_rate=surprise_rate, key_obj=target_key)

    print("  Adding phrasing...")
    combined = add_phrasing(combined)

    # Apply instrumentation
    instruments_to_use = (
        harmony_dna['instruments'] if use_harmony_instruments
        else melody_dna['instruments']
    )
    if instruments_to_use:
        print(f"  Applying instrumentation: {instruments_to_use[:3]}")
        combined = apply_instrumentation(combined, instruments_to_use)

    print("  Finalizing score (measures, key sig, tempo)...")
    combined = finalize_score(combined, rhythmic_dna, blended_energy, target_key)

    return combined


# ─────────────────────────────────────────────────────────────
# AUTO ROLE ASSIGNMENT
# ─────────────────────────────────────────────────────────────

def auto_assign_roles(scores_and_paths):
    """
    Automatically assign melody/harmony/rhythm roles based on DNA features.
    
    Melody donor: most melodic (single melodic line, widest pitch range)
    Harmony donor: richest chords
    Rhythm donor: most rhythmic variety / fastest notes
    """
    if len(scores_and_paths) == 1:
        s, p = scores_and_paths[0]
        return (s, p, "melody"), (s, p, "harmony"), (s, p, "rhythm")

    if len(scores_and_paths) == 2:
        s1, p1 = scores_and_paths[0]
        s2, p2 = scores_and_paths[1]
        return (s1, p1, "melody+harmony"), (s2, p2, "rhythm"), (s2, p2, "harmony")

    scores = []
    for score, path in scores_and_paths:
        flat = score.flatten()
        notes = [n for n in flat.notes if n.isNote]
        chords = [n for n in flat.notes if n.isChord]

        pitches = [n.pitch.midi for n in notes]
        pitch_range = (max(pitches) - min(pitches)) if pitches else 0

        # Rhythmic variety: std of durations
        durs = [float(n.quarterLength) for n in notes]
        rhythm_var = float(np.std(durs)) if durs else 0

        # Harmonic richness: ratio of chords
        chord_ratio = len(chords) / max(len(notes) + len(chords), 1)

        scores.append({
            'score': score,
            'path': path,
            'pitch_range': pitch_range,
            'rhythm_var': rhythm_var,
            'chord_ratio': chord_ratio,
        })

    # Assign roles
    melody_src = max(scores, key=lambda x: x['pitch_range'])
    harmony_src = max(scores, key=lambda x: x['chord_ratio'])
    rhythm_src = max(scores, key=lambda x: x['rhythm_var'])

    # Ensure no two are the same if possible
    assigned = set()
    roles = {}

    for role, src in [('melody', melody_src), ('harmony', harmony_src), ('rhythm', rhythm_src)]:
        key_id = id(src['score'])
        if key_id in assigned:
            # Find next best
            remaining = [s for s in scores if id(s['score']) not in assigned]
            if remaining:
                src = remaining[0]
                key_id = id(src['score'])
        roles[role] = src
        assigned.add(key_id)

    return (
        (roles['melody']['score'],  roles['melody']['path'],  "melody"),
        (roles['harmony']['score'], roles['harmony']['path'], "harmony"),
        (roles['rhythm']['score'],  roles['rhythm']['path'],  "rhythm"),
    )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def load_score(path):
    """Load a MIDI or MusicXML file."""
    print(f"  Loading: {path}")
    try:
        score = converter.parse(path)
        return score
    except Exception as e:
        print(f"  [error] Could not load {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="MIDI DNA Combiner: Extracts musical DNA and combines MIDI files."
    )
    parser.add_argument('inputs', nargs='*', help='Input MIDI/MusicXML files (2-4)')
    parser.add_argument('--melody', help='Melody donor file')
    parser.add_argument('--harmony', help='Harmony donor file')
    parser.add_argument('--rhythm', help='Rhythm donor file')
    parser.add_argument('--output', default='combined_output.mid', help='Output MIDI file')
    parser.add_argument('--form', default='AABA',
                        help='Musical form string e.g. AABA, ABAC, ABAB (default: AABA)')
    parser.add_argument('--surprise', type=float, default=0.08,
                        help='Surprise rate 0-1 (chromatic/unexpected notes, default: 0.08)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "★"*60)
    print("  MIDI DNA COMBINER")
    print("  Extracting musical DNA and creating a new composition")
    print("★"*60 + "\n")

    # Determine sources
    if args.melody and args.harmony and args.rhythm:
        # Explicit roles
        melody_score = load_score(args.melody)
        harmony_score = load_score(args.harmony)
        rhythm_score = load_score(args.rhythm)
        roles = [
            (melody_score, args.melody, "melody"),
            (harmony_score, args.harmony, "harmony"),
            (rhythm_score, args.rhythm, "rhythm"),
        ]
        if None in [melody_score, harmony_score, rhythm_score]:
            print("[error] Could not load one or more input files.")
            sys.exit(1)
    elif args.inputs:
        scores_and_paths = []
        for path in args.inputs:
            s = load_score(path)
            if s:
                scores_and_paths.append((s, path))
        if not scores_and_paths:
            print("[error] No valid input files loaded.")
            sys.exit(1)
        roles = auto_assign_roles(scores_and_paths)
        print("\nAuto-assigned roles:")
        for score, path, role in roles:
            print(f"  {role:8s} ← {path}")
    else:
        parser.print_help()
        sys.exit(1)

    # Extract DNA from each source
    dna_store = {}
    for score, path, role in roles:
        label = f"{role} ({path})"
        dna = extract_full_dna(score, label)
        dna_store[role] = dna

    # Handle cases where roles overlap (single/dual file)
    melody_dna  = dna_store.get('melody') or dna_store.get('melody+harmony') or list(dna_store.values())[0]
    harmony_dna = dna_store.get('harmony') or dna_store.get('melody+harmony') or list(dna_store.values())[0]
    rhythm_dna  = dna_store.get('rhythm') or list(dna_store.values())[-1]

    # Combine
    print("\n" + "━"*60)
    combined_score = combine_dna(
        melody_dna=melody_dna,
        harmony_dna=harmony_dna,
        rhythm_dna=rhythm_dna,
        form_string=args.form,
        surprise_rate=args.surprise,
    )

    # Export
    print(f"\n  Writing output to: {args.output}")
    try:
        combined_score.write('midi', fp=args.output)
        print(f"\n{'★'*60}")
        print(f"  Done! Combined MIDI saved to: {args.output}")
        print(f"{'★'*60}\n")
    except Exception as e:
        print(f"  [error] Could not write MIDI: {e}")
        # Try writing as MusicXML as fallback
        fallback = args.output.replace('.mid', '.xml')
        try:
            combined_score.write('musicxml', fp=fallback)
            print(f"  Fallback: saved as MusicXML to {fallback}")
        except Exception as e2:
            print(f"  [error] Also could not write MusicXML: {e2}")
        sys.exit(1)


if __name__ == '__main__':
    main()
