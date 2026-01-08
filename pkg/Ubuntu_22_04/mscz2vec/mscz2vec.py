# =========================
# DEBUG GLOBAL
# =========================

DEBUG = True

def debug(*args):
    if DEBUG:
        print("[DEBUG]", *args)


# =========================
# IMPORTS
# =========================

from music21 import *
from music21 import instrument, roman, analysis, note, stream, meter
import numpy as np
from collections import Counter
import re
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import copy
import hashlib

# =========================
# CONSTANTES
# =========================

INSTRUMENT_FAMILIES = {
    "piano": ["piano"],
    "strings": ["violin", "viola", "cello", "contrabass", "double bass", "string"],
    "woodwinds": ["flute", "piccolo", "oboe", "clarinet", "bassoon", "recorder"],
    "brass": ["trumpet", "horn", "trombone", "tuba"],
    "guitar": ["guitar", "electric guitar", "acoustic guitar"],
    "voice": ["voice", "vocal", "soprano", "alto", "tenor", "bass", "choir"],
    "percussion": ["percussion", "drum", "timpani", "cymbal", "snare"]
}

HARMONIC_FUNCTIONS = {
    "T": ["I", "i", "vi", "VI"],
    "PD": ["ii", "II", "iv", "IV"],
    "D": ["V", "v", "vii°", "VII"],
    "Dsec": ["V/V", "V/ii", "V/vi", "V/IV"],
    "Other": []
}

# =========================
# MELODÍA
# =========================

def melodic_features(score, n_intervals=12):
    notes = score.flatten().notes
    debug("Melodic: nº notas =", len(notes))

    pitches = [n.pitch.midi for n in notes if n.isNote]
    if len(pitches) < 2:
        debug("Melodic: muy pocas notas")
        return np.zeros(n_intervals + 4)

    intervals = np.diff(pitches)
    debug("Melodic: intervalos calculados: ", intervals)

    interval_hist, _ = np.histogram(
        intervals,
        bins=n_intervals,
        range=(-12, 12),
        density=True
    )

    features = np.concatenate([
        interval_hist,
        [
            np.mean(pitches),
            np.std(pitches),
            max(pitches) - min(pitches),
            np.mean(np.sign(intervals))
        ]
    ])

    return features


# =========================
# ARMONÍA
# =========================

def roman_to_function(rn):
    figure = rn.figure

    if "/" in figure:
        return "Dsec"

    base = figure.replace("°", "").replace("+", "")
    base = "".join(c for c in base if not c.isdigit())

    for func, romans in HARMONIC_FUNCTIONS.items():
        if base in romans:
            return func

    return "Other"

def extract_harmonic_chords(stream):
    debug("Harmony: extrayendo acordes armónicos")
    debug(f"Harmony: tipo de stream = {type(stream)}")

    notes_by_offset = defaultdict(list)

    # ── Recolectar notas y acordes explícitos
    for el in stream.recurse():
        # Nota individual
        if isinstance(el, note.Note):
            abs_offset = float(el.getOffsetInHierarchy(stream))
            notes_by_offset[abs_offset].append(el)

            debug(
                f"Harmony: NOTE  {el.pitch.nameWithOctave} "
                f"@ {abs_offset:.3f}"
            )

        # Acorde explícito (MusicXML / MuseScore)
        elif isinstance(el, chord.Chord):
            abs_offset = float(el.getOffsetInHierarchy(stream))

            debug(
                f"Harmony: CHORD {[p.nameWithOctave for p in el.pitches]} "
                f"@ {abs_offset:.3f}"
            )

            for p in el.pitches:
                notes_by_offset[abs_offset].append(note.Note(p))

    debug(f"Harmony: offsets únicos = {len(notes_by_offset)}")

    # ── Construir acordes armónicos
    chords_list = []

    for offset in sorted(notes_by_offset):
        group = notes_by_offset[offset]

        debug(f"\nHarmony: offset {offset:7.3f} → {len(group)} nota(s)")

        for n in group:
            staff = n.getContextByClass('Staff')
            part = n.getContextByClass('Part')

            debug(
                f"  - {n.pitch.nameWithOctave:>4} "
                f"| dur={float(n.quarterLength):4.2f} "
                f"| part={part.id if part else '?'} "
                f"| staff={staff.id if staff else '?'}"
            )

        if len(group) >= 2:
            ch = chord.Chord(group)
            chords_list.append(ch)

            debug(
                "Harmony: acorde detectado → "
                f"{[p.nameWithOctave for p in ch.pitches]}"
            )

    debug("Harmony: nº total de acordes =", len(chords_list))
    return chords_list

def harmonic_features(score):
    try:
        key = score.analyze('key')
        debug("Harmony: tonalidad =", key)
    except:
        debug("Harmony: no se pudo detectar tonalidad")
        return np.zeros(len(HARMONIC_FUNCTIONS))

    chords = extract_harmonic_chords(score)

    debug("Harmony: nº acordes =", len(chords))

    if not chords:
        return np.zeros(len(HARMONIC_FUNCTIONS))

    function_counts = {f: 0 for f in HARMONIC_FUNCTIONS}

    for chord_obj in chords:
        try:
            # Roman numeral
            rn = roman.romanNumeralFromChord(chord_obj, key)
            func = roman_to_function(rn)
            function_counts[func] += 1

            # Notas del acorde
            notes_in_chord = [p.nameWithOctave for p in chord_obj.pitches]

            # Debug
            debug(
                f"Harmony DEBUG: notas = {notes_in_chord} | figure = {rn.figure} | función = {func}"
            )

        except Exception as e:
            debug("Harmony DEBUG: error con acorde", chord_obj, e)
            function_counts["Other"] += 1

    total = sum(function_counts.values())
    vector = np.array(
        [function_counts[f] / total for f in HARMONIC_FUNCTIONS]
    )

    debug("Harmony: vector =", vector)
    return vector

def harmonic_transition_features(score):
    try:
        key = score.analyze('key')
        debug("Harmony transitions: tonalidad =", key)
    except:
        debug("Harmony transitions: sin tonalidad")
        return np.zeros(25)

    # chords = score.chordify().flatten().getElementsByClass('Chord')
    chords = extract_harmonic_chords(score)

    funcs = []

    for chord in chords:
        try:
            rn = roman.romanNumeralFromChord(chord, key)
            funcs.append(roman_to_function(rn))
        except:
            continue

    debug("Harmony transitions: funciones =", funcs)

    labels = ["T", "PD", "D", "Dsec", "Other"]
    matrix = np.zeros((len(labels), len(labels)))
    idx = {l: i for i, l in enumerate(labels)}

    for i in range(len(funcs) - 1):
        matrix[idx[funcs[i]], idx[funcs[i+1]]] += 1

    matrix /= np.sum(matrix) if np.sum(matrix) > 0 else 1
    return matrix.flatten()


# =========================
# RITMO
# =========================

def rhythmic_sequence(score):
    sequence = []
    for n in score.flatten().notes:
        m = n.getContextByClass('Measure')
        if m is None:
            continue
        measure_len = m.barDuration.quarterLength
        sequence.append(n.quarterLength / measure_len)
    debug("Rhythm: secuencia =", sequence[:10], "...")
    return sequence

def rhythmic_ngrams(sequence, n=3):
    return Counter(tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1))

def rhythmic_fft(sequence):
    signal = np.array(sequence)
    if len(signal) == 0:
        return np.zeros(16)
    mag = np.abs(np.fft.fft(signal))
    return mag[:16] / (np.sum(mag[:16]) + 1e-9)


def rhythmic_features(score, ngram_n=3, fft_len=16, hist_bins=12):
    sequence = rhythmic_sequence(score)
    debug("Rhythm: longitud secuencia =", len(sequence))

    if len(sequence) == 0:
        debug("Rhythm: secuencia vacía")
        return np.zeros(hist_bins + fft_len + ngram_n)

    hist, _ = np.histogram(sequence, bins=hist_bins, range=(0, 2))
    hist = hist / (np.sum(hist) + 1e-9)

    fft_vec = rhythmic_fft(sequence)

    ngrams = rhythmic_ngrams(sequence, n=ngram_n)
    ngram_vec = np.zeros(ngram_n)
    for i, k in enumerate(ngrams):
        ngram_vec[i % ngram_n] += ngrams[k]

    ngram_vec /= np.linalg.norm(ngram_vec) + 1e-9

    return np.concatenate([hist, fft_vec, ngram_vec])


# =========================
# INSTRUMENTACIÓN
# =========================

def normalize_name(name):
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\b[i,v,x]+\b", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


def instrumental_features(score):
    vector = {fam: 0.0 for fam in INSTRUMENT_FAMILIES}
    instruments = score.recurse().getElementsByClass(instrument.Instrument)
    debug("Instrumentation: nº instrumentos =", len(instruments))

    for inst in instruments:
        debug("Instrumento:", inst.instrumentName, inst.partName)
        names = [n for n in [inst.instrumentName, inst.partName] if n]

        matched = False
        for name in names:
            norm = normalize_name(name)
            for family, keywords in INSTRUMENT_FAMILIES.items():
                if any(k in norm for k in keywords):
                    vector[family] = 1.0
                    matched = True

        if not matched:
            if isinstance(inst, instrument.StringInstrument):
                vector["strings"] = 1.0
            elif isinstance(inst, instrument.WoodwindInstrument):
                vector["woodwinds"] = 1.0
            elif isinstance(inst, instrument.BrassInstrument):
                vector["brass"] = 1.0
            elif isinstance(inst, instrument.Percussion):
                vector["percussion"] = 1.0
            elif isinstance(inst, instrument.KeyboardInstrument):
                vector["piano"] = 1.0
            elif isinstance(inst, instrument.Vocalist):
                vector["voice"] = 1.0

    return np.array(list(vector.values()))


# =========================
# MOTIVOS
# =========================

def motif_is_submotif(small, large):
    """Devuelve True si small está contenido en large (subsecuencia contigua)."""
    if len(small) > len(large):
        return False

    for i in range(len(large) - len(small) + 1):
        if large[i:i+len(small)] == small:
            return True
    return False

def extract_melody(score):
    parts = getattr(score, 'parts', [score])
    melody_part = parts[0].flatten().notes if parts else score.flatten().notes
    seq = [(n.pitch.midi, n.pitch.nameWithOctave, n.duration.quarterLength)
           for n in melody_part if n.isNote]
    debug("Motifs: longitud melodía =", len(seq))
    if len(seq) > 0:
        debug("Motifs: primeras 5 notas =", seq[:5])
    return seq

def melodic_intervals(sequence):
    intervals = [sequence[i+1][0] - sequence[i][0] for i in range(len(sequence) - 1)]
    debug("Motifs: primeros 10 intervalos =", intervals[:10])
    return intervals

def rhythmic_ratios(sequence):
    ratios = []
    for i in range(len(sequence) - 1):
        if sequence[i][2] > 0:
            r = sequence[i+1][2] / sequence[i][2]
        else:
            r = 1.0
        ratios.append(r)
    debug("Motifs: primeros 10 ratios rítmicos =", ratios[:10])
    return ratios

def quantize_interval(i):
    return max(-12, min(12, i))

def quantize_ratio(r):
    if r < 0.75:
        return "shorter"
    elif r > 1.33:
        return "longer"
    else:
        return "same"

def filter_maximal_motifs(motifs, top_n):
    """
    motifs: lista de (motif, notes, count)
    Devuelve solo los motifs más grandes, eliminando submotifs.
    """

    # Ordenar: primero longitud DESC, luego frecuencia DESC
    motifs_sorted = sorted(
        motifs,
        key=lambda x: (len(x[0]), x[2]),
        reverse=True
    )

    selected = []

    for motif, notes, count in motifs_sorted:
        is_sub = False
        for sel_motif, _, _ in selected:
            if motif_is_submotif(motif, sel_motif):
                is_sub = True
                break

        if not is_sub:
            selected.append((motif, notes, count))
            debug(f"Motif aceptado (len={len(motif)}, count={count})")

        if len(selected) >= top_n:
            break

    return selected

def extract_top_motifs(sequence, min_length=3, max_length=6, top_n=10):
    intervals = melodic_intervals(sequence)
    rhythms = rhythmic_ratios(sequence)

    motif_counter = Counter()
    motif_notes_dict = {}

    # 1. Extraer TODOS los motifs
    for length in range(min_length, max_length + 1):
        for i in range(len(intervals) - length + 1):
            motif = tuple(
                (quantize_interval(intervals[i+j]),
                 quantize_ratio(rhythms[i+j]))
                for j in range(length)
            )
            notes = sequence[i:i+length+1]  # +1 nota
            motif_counter[motif] += 1
            if motif not in motif_notes_dict:
                motif_notes_dict[motif] = notes

    all_motifs = [
        (m, motif_notes_dict[m], count)
        for m, count in motif_counter.items()
    ]

    debug(f"Motifs totales antes de filtrar = {len(all_motifs)}")

    # 2. Filtrar solo motifs máximos
    filtered = filter_maximal_motifs(all_motifs, top_n)

    debug(f"Motifs finales (sin submotifs) = {len(filtered)}")
    for i, (m, notes, c) in enumerate(filtered):
        note_info = [(n[1], n[2]) for n in notes]
        debug(f"Motif {i+1}: len={len(m)} | reps={c} | notas={note_info}")

    return filtered

def motif_vector(score, dim=128, min_length=3, max_length=6, top_n=10):
    """
    Genera vector con los `top_n` motifs más frecuentes.
    """
    seq = extract_melody(score)
    seq_len = len(seq)

    if seq_len < min_length + 1:
        debug(f"Motifs: secuencia demasiado corta (len={seq_len}, min_length={min_length})")
        return np.zeros(dim, dtype=float)

    motifs = extract_top_motifs(seq, min_length, max_length, top_n)
    if not motifs:
        debug("Motifs: no se generaron motifs")
        return np.zeros(dim, dtype=float)

    vec = np.zeros(dim, dtype=float)

    for m_idx, (m, notes, count) in enumerate(motifs):
        h = int(hashlib.md5(str(m).encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        vec[idx] += count

        note_info = [(p[1], p[2]) for p in notes]
        debug(f"Motif {m_idx+1}: {m} | Notas = {note_info} | Repeticiones = {count} → índice vector {idx}")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        debug("Motifs: vector todo ceros, normalización omitida")

    debug(f"Motifs: vector final generado, norma={norm:.3f}, dimensión={dim}")
    return vec

# =========================
# FORMA
# =========================

def segment_by_measures(score, window_size=4):
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))
    debug("Form: nº compases =", len(measures))
    segments = []

    for i in range(0, len(measures), window_size):
        segments.append(measures[i:i+window_size])

    debug("Form: nº segmentos =", len(segments))
    return segments

def segment_descriptor(segment):
    seg_stream = stream.Stream(segment)
    flat = seg_stream.flatten()
    v = np.concatenate([melodic_features(flat), harmonic_features(flat)])
    return v / (np.linalg.norm(v) + 1e-6)

def cluster_segments(descriptors, similarity_threshold=0.85):
    debug("Clustering: nº descriptores =", len(descriptors))
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    return clustering.fit_predict(descriptors)


def normalize_form(labels):
    mapping = {}
    next_label = "A"
    form = []

    for l in labels:
        if l not in mapping:
            mapping[l] = next_label
            next_label = chr(ord(next_label) + 1)
        form.append(mapping[l])

    debug("Form normalizada:", form)
    return form

def form_statistics(form):
    unique_sections = len(set(form))
    length = len(form)

    repetitions = sum(
        form[i] == form[i+1] for i in range(len(form)-1)
    )

    return np.array([
        unique_sections,
        repetitions / max(1, length),
        length
    ])

def form_ngrams(form, n=3, dim=32):
    ngrams = [
        tuple(form[i:i+n])
        for i in range(len(form) - n + 1)
    ]

    vector = np.zeros(dim)
    for ng in ngrams:
        idx = hash(ng) % dim
        vector[idx] += 1

    return vector / (np.linalg.norm(vector) + 1e-6)

def form_structure_vector(score,
                          window_size=4,
                          similarity_threshold=0.85):
    segments = segment_by_measures(score, window_size)

    if len(segments) < 2:
        return np.zeros(64)

    descriptors = [segment_descriptor(seg) for seg in segments]
    labels = cluster_segments(descriptors, similarity_threshold)
    form = normalize_form(labels)

    v_stats = form_statistics(form)
    v_ngrams = form_ngrams(form, n=3, dim=32)

    return np.concatenate([v_stats, v_ngrams])

# =========================
# EJECUCIÓN
# =========================

debug("Cargando score...")
score = converter.parse('./dreams_red.musicxml')
debug("Score cargado correctamente")

# print(rhythmic_features(score))
# print(melodic_features(score))
# print(harmonic_features(score))
# print(harmonic_transition_features(score))
# print(instrumental_features(score))
print(motif_vector(score))
# print(form_structure_vector(score))
