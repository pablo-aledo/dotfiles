from music21 import *
import numpy as np
from collections import Counter
from music21.instrument import Instrument
import re
import unicodedata
from music21 import roman, analysis
from music21 import note, stream
from music21 import meter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from music21 import stream


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

def melodic_features(score, n_intervals=12):
    notes = score.flatten().notes

    pitches = [n.pitch.midi for n in notes if n.isNote]
    if len(pitches) < 2:
        return np.zeros(n_intervals + 4)

    intervals = np.diff(pitches)

    # Histograma de intervalos (−12 a +12)
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
            max(pitches) - min(pitches),   # rango
            np.mean(np.sign(intervals))    # dirección media
        ]
    ])

    return features

def harmonic_features(score):
    chords = score.chordify().flatten().getElementsByClass('Chord')

    if not chords:
        return np.zeros(12)

    pitch_classes = []
    for chord in chords:
        pitch_classes.extend([p.pitchClass for p in chord.pitches])

    counter = Counter(pitch_classes)

    # Pitch Class Profile (12 dimensiones)
    pcp = np.array([counter.get(i, 0) for i in range(12)], dtype=float)
    pcp /= np.sum(pcp) if np.sum(pcp) > 0 else 1

    return pcp

def roman_to_function(rn):
    figure = rn.figure

    # Dominantes secundarios
    if "/" in figure:
        return "Dsec"

    base = figure.replace("°", "").replace("+", "")
    base = "".join(c for c in base if not c.isdigit())

    for func, romans in HARMONIC_FUNCTIONS.items():
        if base in romans:
            return func

    return "Other"

def harmonic_features(score):
    """
    Extrae un vector de funciones armónicas normalizado
    """
    # 1. Estimar tonalidad global
    try:
        key = score.analyze('key')
    except:
        return np.zeros(len(HARMONIC_FUNCTIONS))

    chords = score.chordify().flatten().getElementsByClass('Chord')

    if not chords:
        return np.zeros(len(HARMONIC_FUNCTIONS))

    function_counts = {f: 0 for f in HARMONIC_FUNCTIONS}

    for chord in chords:
        try:
            rn = roman.romanNumeralFromChord(chord, key)
            func = roman_to_function(rn)
            function_counts[func] += 1
        except:
            function_counts["Other"] += 1

    # Normalizar
    total = sum(function_counts.values())
    vector = np.array(
        [function_counts[f] / total for f in HARMONIC_FUNCTIONS]
    )

    return vector

def harmonic_transition_features(score):
    try:
        key = score.analyze('key')
    except:
        return np.zeros(16)

    chords = score.chordify().flatten().getElementsByClass('Chord')
    funcs = []

    for chord in chords:
        try:
            rn = roman.romanNumeralFromChord(chord, key)
            funcs.append(roman_to_function(rn))
        except:
            continue

    transitions = [(funcs[i], funcs[i+1]) for i in range(len(funcs)-1)]
    labels = ["T", "PD", "D", "Dsec", "Other"]

    matrix = np.zeros((len(labels), len(labels)))
    idx = {l: i for i, l in enumerate(labels)}

    for a, b in transitions:
        matrix[idx[a], idx[b]] += 1

    matrix /= np.sum(matrix) if np.sum(matrix) > 0 else 1
    return matrix.flatten()

def rhythmic_features(score, n_bins=6):
    notes = score.flatten().notesAndRests
    durations = [n.duration.quarterLength for n in notes]

    if not durations:
        return np.zeros(n_bins + 2)

    hist, _ = np.histogram(
        durations,
        bins=n_bins,
        range=(0, 4),
        density=True
    )

    features = np.concatenate([
        hist,
        [
            np.mean(durations),
            np.std(durations)
        ]
    ])

    return features

def normalize_name(name):
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\(.*?\)", "", name)     # eliminar paréntesis
    name = re.sub(r"\b[i,v,x]+\b", "", name)  # eliminar numerales romanos
    name = re.sub(r"[^a-z\s]", "", name)
    name = name.strip()
    return name

def instrumental_features(score):
    """
    Devuelve un vector binario por familias instrumentales
    """
    vector = {fam: 0.0 for fam in INSTRUMENT_FAMILIES}

    instruments = score.recurse().getElementsByClass(instrument.Instrument)

    for inst in instruments:
        # 1. Nombre explícito
        names = []
        if inst.instrumentName:
            names.append(inst.instrumentName)
        if inst.partName:
            names.append(inst.partName)

        matched = False
        for name in names:
            norm = normalize_name(name)

            for family, keywords in INSTRUMENT_FAMILIES.items():
                if any(k in norm for k in keywords):
                    vector[family] = 1.0
                    matched = True

        # 2. Fallback por clase music21
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

def extract_melody(score):
    """
    Devuelve una secuencia ordenada de notas (pitch midi, duración)
    """
    # Usamos la parte superior como aproximación
    parts = score.parts
    if parts:
        melody = parts[0].flatten().notes
    else:
        melody = score.flatten().notes

    sequence = [
        (n.pitch.midi, n.duration.quarterLength)
        for n in melody
        if n.isNote
    ]
    return sequence

def melodic_intervals(sequence):
    return [
        sequence[i+1][0] - sequence[i][0]
        for i in range(len(sequence) - 1)
    ]

def rhythmic_ratios(sequence):
    return [
        sequence[i+1][1] / sequence[i][1]
        if sequence[i][1] > 0 else 1
        for i in range(len(sequence) - 1)
    ]

def quantize_interval(i):
    return max(-12, min(12, i))  # limitar saltos extremos

def quantize_ratio(r):
    if r < 0.75:
        return "shorter"
    elif r > 1.33:
        return "longer"
    else:
        return "same"

def extract_motifs(sequence, n=3):
    intervals = melodic_intervals(sequence)
    rhythms = rhythmic_ratios(sequence)

    motifs = []
    for i in range(len(intervals) - n + 1):
        motif = tuple(
            (quantize_interval(intervals[i+j]),
             quantize_ratio(rhythms[i+j]))
            for j in range(n)
        )
        motifs.append(motif)

    return motifs

def motif_hash(motif, dim=128):
    return hash(motif) % dim

def motif_vector(score, dim=128, n=3):
    sequence = extract_melody(score)
    if len(sequence) < n + 1:
        return np.zeros(dim)

    motifs = extract_motifs(sequence, n=n)

    vector = np.zeros(dim)
    for m in motifs:
        idx = motif_hash(m, dim)
        vector[idx] += 1

    # normalizar
    vector /= np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else 1
    return vector

def segment_by_measures(score, window_size=4):
    """
    Divide la partitura en segmentos de N compases
    """
    measures = list(score.parts[0].getElementsByClass('Measure'))
    segments = []

    for i in range(0, len(measures), window_size):
        seg = measures[i:i + window_size]
        if seg:
            segments.append(seg)

    return segments

def segment_descriptor(segment):
    """
    Convierte una lista de medidas en un stream y devuelve descriptor
    """
    # si segment es lista de measures, lo convertimos en Stream
    seg_stream = stream.Stream(segment)
    flat = seg_stream.flatten()

    v_mel = melodic_features(flat)
    v_har = harmonic_features(flat)

    v = np.concatenate([v_mel, v_har])
    return v / (np.linalg.norm(v) + 1e-6)

def segment_similarity_matrix(descriptors):
    return cosine_similarity(descriptors)

def cluster_segments(descriptors, similarity_threshold=0.85):
    n = len(descriptors)
    if n <= 1:
        return [0]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',    # <- versión antigua usa affinity
        linkage='average',    # promedio de similitud
        distance_threshold=1 - similarity_threshold
    )

    labels = clustering.fit_predict(descriptors)
    return labels

def normalize_form(labels):
    mapping = {}
    next_label = "A"
    form = []

    for l in labels:
        if l not in mapping:
            mapping[l] = next_label
            next_label = chr(ord(next_label) + 1)
        form.append(mapping[l])

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

# Cargar el archivo MSCZ
score = converter.parse('./output.musicxml')

print(form_structure_vector(score))
