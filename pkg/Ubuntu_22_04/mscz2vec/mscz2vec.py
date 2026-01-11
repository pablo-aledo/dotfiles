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
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

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

def chord_short_name(ch):
    try:
        root = ch.root().name
    except:
        return "?"

    qual = ""

    # Triadas
    if ch.seventh is None:
        if ch.quality == "major":
            qual = "maj"
        elif ch.quality == "minor":
            qual = "min"
        elif ch.quality == "diminished":
            qual = "dim"
        elif ch.quality == "augmented":
            qual = "aug"
        else:
            qual = "?"

    # Séptimas
    else:
        if ch.seventh == "dominant":
            qual = "7"
        elif ch.seventh == "major":
            qual = "maj7"
        elif ch.seventh == "minor":
            qual = "min7"
        elif ch.seventh == "half-diminished":
            qual = "hdim7"
        elif ch.seventh == "diminished":
            qual = "dim7"
        else:
            qual = "7?"

    return f"{root}:{qual}"

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
            short_name = chord_short_name(chord_obj)
            debug(
                f"Harmony DEBUG: acorde = {short_name} | "
                f"notas = {notes_in_chord} | "
                f"figure = {rn.figure} | función = {func}"
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

def filter_maximal_motifs_with_positions(motifs, top_n):
    """
    Filtra motivos eliminando submotifs, preservando información de posiciones.
    motifs: lista de (motif, notes, count, positions)
    """
    # Ordenar: primero longitud DESC, luego frecuencia DESC
    motifs_sorted = sorted(
        motifs,
        key=lambda x: (len(x[0]), x[2]),
        reverse=True
    )

    selected = []

    for motif, notes, count, positions in motifs_sorted:
        is_sub = False
        for sel_motif, _, _, _ in selected:
            if motif_is_submotif(motif, sel_motif):
                is_sub = True
                break

        if not is_sub:
            selected.append((motif, notes, count, positions))
            debug(f"Motif aceptado (len={len(motif)}, count={count}, posiciones={len(positions)})")

        if len(selected) >= top_n:
            break

    return selected

def extract_top_motifs(sequence, min_length=3, max_length=6, top_n=10):
    """
    Extrae los motivos más frecuentes de una secuencia melódica,
    incluyendo información de en qué compases aparecen.
    """
    intervals = melodic_intervals(sequence)
    rhythms = rhythmic_ratios(sequence)

    motif_counter = Counter()
    motif_notes_dict = {}
    motif_positions = defaultdict(list)  # Nueva: guardar posiciones

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

            # Guardar la posición (índice de inicio en la secuencia)
            motif_positions[motif].append(i)

    all_motifs = [
        (m, motif_notes_dict[m], count, motif_positions[m])
        for m, count in motif_counter.items()
    ]

    debug(f"Motifs totales antes de filtrar = {len(all_motifs)}")

    # 2. Filtrar solo motifs máximos (adaptado para incluir posiciones)
    filtered = filter_maximal_motifs_with_positions(all_motifs, top_n)

    debug(f"Motifs finales (sin submotifs) = {len(filtered)}")
    for i, (m, notes, c, positions) in enumerate(filtered):
        note_info = [(n[1], n[2]) for n in notes]
        debug(f"Motif {i+1}: len={len(m)} | reps={c} | notas={note_info} | posiciones={positions[:5]}...")

    return filtered

def extract_melody_with_measures(score):
    """
    Extrae la melodía con información de compases.
    Returns: lista de tuplas (midi, name, duration, measure_number)
    """
    parts = getattr(score, 'parts', [score])
    melody_part = parts[0].flatten().notes if parts else score.flatten().notes

    seq = []
    for n in melody_part:
        if n.isNote:
            measure = n.getContextByClass('Measure')
            measure_num = measure.measureNumber if measure else 0
            seq.append((n.pitch.midi, n.pitch.nameWithOctave, n.duration.quarterLength, measure_num))

    debug("Melody with measures: longitud =", len(seq))
    if len(seq) > 0:
        debug("Melody with measures: primeras 5 notas =", seq[:5])

    return seq

def motif_vector(score, dim=128, min_length=3, max_length=6, top_n=10):
    """
    Genera vector con los `top_n` motifs más frecuentes.
    Ahora muestra información de compases donde aparecen.
    """
    seq = extract_melody_with_measures(score)
    seq_len = len(seq)

    if seq_len < min_length + 1:
        debug(f"Motifs: secuencia demasiado corta (len={seq_len}, min_length={min_length})")
        return np.zeros(dim, dtype=float)

    # Convertir a formato compatible con extract_top_motifs (sin measure_num para cálculo)
    seq_for_motifs = [(n[0], n[1], n[2]) for n in seq]

    motifs = extract_top_motifs(seq_for_motifs, min_length, max_length, top_n)
    if not motifs:
        debug("Motifs: no se generaron motifs")
        return np.zeros(dim, dtype=float)

    vec = np.zeros(dim, dtype=float)

    print("\n" + "="*50)
    print("MOTIVOS MELÓDICOS DETECTADOS")
    print("="*50)

    for m_idx, (m, notes, count, positions) in enumerate(motifs):
        h = int(hashlib.md5(str(m).encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        vec[idx] += count

        note_info = [(p[1], p[2]) for p in notes]

        # Obtener compases donde aparece este motif
        measures_where_appears = []
        for pos in positions[:5]:  # Mostramos solo las primeras 5 apariciones
            if pos < len(seq):
                measures_where_appears.append(seq[pos][3])  # measure_num

        print(f"\nMotivo {m_idx+1}:")
        print(f"  ├─ Notas: {note_info}")
        print(f"  ├─ Repeticiones: {count}")
        print(f"  ├─ Aparece en compases: {measures_where_appears}" + ("..." if len(positions) > 5 else ""))
        print(f"  └─ Índice en vector: {idx}")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        debug("Motifs: vector todo ceros, normalización omitida")

    print("="*50)
    debug(f"Motifs: vector final generado, norma={norm:.3f}, dimensión={dim}")
    return vec


# =========================
# MOTIVOS2
# =========================

def normalize_pitches_by_first_note(melody):
    """
    Normaliza las notas de un compás de acuerdo con la primera nota.
    Devuelve una lista de notas normalizadas, donde la primera nota tiene altura 0.
    """
    if len(melody) == 0:
        return []

    # Obtenemos la altura MIDI de la primera nota
    first_pitch = melody[0].pitch.midi

    # Normalizamos las notas restando la altura de la primera nota
    normalized_melody = [n.pitch.midi - first_pitch for n in melody]

    return normalized_melody

def extract_melody_by_measure(score):
    """
    Extrae la melodía dividida por compases, devolviendo una lista de secuencias de notas por compás.
    Normaliza las alturas para que todas las notas estén relativas a la primera nota del compás.
    """
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))
    melody_per_measure = []

    debug("Motifs by measure: nº compases =", len(measures))

    for measure in measures:
        melody_part = measure.flatten().notes
        seq = [n for n in melody_part if n.isNote]  # Solo notas
        normalized_seq = normalize_pitches_by_first_note(seq)  # Normalizamos las alturas
        melody_per_measure.append(normalized_seq)

    return melody_per_measure

def identify_repeated_measures(melody_per_measure):
    """
    Identifica compases repetidos y cuenta cuántas veces se repite cada compás.
    También registra en qué compás(es) se encuentra cada patrón.
    """
    measure_counter = Counter()
    measure_locations = defaultdict(list)  # Diccionario para almacenar los índices de compás

    for i, measure in enumerate(melody_per_measure):
        # Convertimos cada compás a una tupla para que sea hashable
        measure_counter[tuple(measure)] += 1
        measure_locations[tuple(measure)].append(i)  # Guardamos el índice del compás

    debug(f"Motifs by measure: nº compases únicos = {len(measure_counter)}")
    return measure_counter, measure_locations

def compass_motif_vector(score, dim=128, top_n=10):
    """
    Genera un vector con los `top_n` compases más repetidos, y muestra en qué compás(es)
    se encuentran esos patrones, usando patrones invariables a la altura de las notas.
    """
    # Paso 1: Extraer la melodía por compases
    melody_per_measure = extract_melody_by_measure(score)

    # Paso 2: Identificar compases repetidos
    repeated_measures, measure_locations = identify_repeated_measures(melody_per_measure)

    if not repeated_measures:
        debug("Motifs by measure: no se generaron compases repetidos")
        return np.zeros(dim, dtype=float)

    # Paso 3: Seleccionar los compases más repetidos (top_n)
    most_common_measures = repeated_measures.most_common(top_n)

    vec = np.zeros(dim, dtype=float)

    for idx, (measure, count) in enumerate(most_common_measures):
        h = int(hashlib.md5(str(measure).encode('utf-8')).hexdigest(), 16)
        measure_idx = h % dim
        vec[measure_idx] += count

        # Localizamos en qué compás(es) encontramos este patrón
        compasses = measure_locations[measure]

        debug(f"Motif {idx+1}: compás normalizado = {measure} | Repeticiones = {count} → índice vector {measure_idx} | "
              f"en compases: {compasses}")

    # Normalizamos el vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        debug("Motifs by measure: vector todo ceros, normalización omitida")

    debug(f"Motifs by measure: vector final generado, norma={norm:.3f}, dimensión={dim}")
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

    # Extraer características
    v_mel = melodic_features(flat)
    v_har = harmonic_features(flat)
    v = np.concatenate([v_mel, v_har])

    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # Retorna vector de ceros tal cual

    return v / norm

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

def form_structure_vector(score, window_size=4, similarity_threshold=0.85):
    segments = segment_by_measures(score, window_size)
    if len(segments) < 2:
        return np.zeros(64)

    descriptors = [segment_descriptor(seg) for seg in segments]

    # Identificar cuáles son vectores de ceros (segmentos vacíos/silencios)
    is_zero = np.array([np.all(d == 0) for d in descriptors])

    if np.all(is_zero):
        debug("Form: Todos los segmentos están vacíos.")
        return np.zeros(64)

    # Solo hacemos clustering de los segmentos NO vacíos
    non_zero_indices = np.where(~is_zero)[0]
    non_zero_descriptors = [descriptors[i] for i in non_zero_indices]

    labels = np.full(len(descriptors), -1) # -1 será la etiqueta para "vacío"

    if len(non_zero_descriptors) > 1:
        # Realizar clustering solo con los datos válidos
        cluster_labels = cluster_segments(non_zero_descriptors, similarity_threshold)
        labels[non_zero_indices] = cluster_labels
    elif len(non_zero_descriptors) == 1:
        labels[non_zero_indices] = 0

    # Normalizar etiquetas: Los vacíos se llamarán "Z" (o cualquier letra)
    # y el resto A, B, C...
    form = []
    mapping = {-1: "Z"} # Z para silencios/vacíos
    next_label = "A"

    for l in labels:
        if l not in mapping:
            mapping[l] = next_label
            next_label = chr(ord(next_label) + 1)
        form.append(mapping[l])

    debug("Form detectada:", form)

    v_stats = form_statistics(form)
    v_ngrams = form_ngrams(form, n=3, dim=32)

    return np.concatenate([v_stats, v_ngrams])

# =========================
# SEQUITUR
# =========================

class Rule:
    def __init__(self, name, symbols):
        self.name = name
        self.symbols = symbols  # lista de notas o subreglas

    def __repr__(self):
        return f"{self.name} -> {self.symbols}"

class SequiturGrammar:
    def __init__(self):
        self.rules = {}
        self.next_rule_id = 1
        self.root_sequence = []

    def _get_new_rule_name(self):
        name = f"R{self.next_rule_id}"
        self.next_rule_id += 1
        return name

    def build(self, sequence):
        """
        Implementación estilo Re-Pair (más estable para música que Sequitur).
        Prioriza el reemplazo del par MÁS FRECUENTE globalmente.
        """
        if not sequence: return {}

        # Copia de trabajo
        current_seq = list(sequence)

        while True:
            # 1. Contar pares adyacentes
            pairs = Counter()
            for i in range(len(current_seq) - 1):
                pair = (current_seq[i], current_seq[i+1])
                pairs[pair] += 1

            # 2. Condición de parada: No hay pares repetidos
            if not pairs: break

            # Encontramos el par más frecuente
            most_common_pair, count = pairs.most_common(1)[0]

            if count < 2:
                break # Si el más común aparece solo una vez, terminamos

            # 3. Crear nueva regla
            rule_name = self._get_new_rule_name()
            self.rules[rule_name] = list(most_common_pair)

            # 4. Reemplazar en la secuencia (sin solapamiento)
            new_seq = []
            i = 0
            while i < len(current_seq):
                if i < len(current_seq) - 1 and (current_seq[i], current_seq[i+1]) == most_common_pair:
                    new_seq.append(rule_name)
                    i += 2
                else:
                    new_seq.append(current_seq[i])
                    i += 1

            current_seq = new_seq

        self.root_sequence = current_seq
        return self.rules

    def expand_rule(self, symbol):
        """
        Despliega recursivamente una regla para ver las notas reales
        que contiene, no solo las sub-reglas.
        Ejemplo: Convierte 'R5' -> [60, 62, 64, 60]
        """
        if symbol not in self.rules:
            return [symbol] # Es una nota terminal (número)

        expansion = []
        for s in self.rules[symbol]:
            expansion.extend(self.expand_rule(s))
        return expansion

    def get_musical_hierarchy(self):
        """
        Devuelve una estructura legible para humanos.
        Muestra qué frases musicales reales representa cada regla.
        """
        hierarchy = {}
        # Ordenamos por ID para entender la construcción de abajo hacia arriba
        sorted_keys = sorted(self.rules.keys(), key=lambda x: int(x[1:]))

        for rule_name in sorted_keys:
            # Expandimos completamente para ver las notas
            flat_notes = self.expand_rule(rule_name)

            # Obtenemos la definición inmediata (quizás contenga otras reglas)
            structure = self.rules[rule_name]

            hierarchy[rule_name] = {
                "structure": structure,     # Cómo se construye (ej: [R1, 67])
                "phrase": flat_notes,       # Qué suena realmente (ej: [60, 62, 67])
                "length": len(flat_notes)
            }

        return hierarchy, self.root_sequence






def expand_rule_with_indices(symbol, grammar, current_pos=0):
    """
    Expande una regla y devuelve tanto las notas como los índices originales.
    """
    if symbol not in grammar.rules:
        # Es una nota terminal (número MIDI)
        return [symbol], [current_pos]

    notes = []
    indices = []
    pos = current_pos

    for s in grammar.rules[symbol]:
        sub_notes, sub_indices = expand_rule_with_indices(s, grammar, pos)
        notes.extend(sub_notes)
        indices.extend(sub_indices)
        pos += len(sub_notes)

    return notes, indices

def simplify_segments_preserving_indices(segments_list, target_length):
    """
    Simplifica la lista de segmentos fusionando los más cortos
    hasta alcanzar target_length, preservando los índices de notas.
    """
    if not segments_list:
        return []

    current_segments = copy.deepcopy(segments_list)

    while len(current_segments) > target_length:
        min_len = float('inf')
        min_idx = -1

        for i, seg in enumerate(current_segments):
            if seg['len'] < min_len:
                min_len = seg['len']
                min_idx = i

        if min_idx == 0:
            merge_idx = 1
            target_idx = 0
        elif min_idx == len(current_segments) - 1:
            merge_idx = min_idx
            target_idx = min_idx - 1
        else:
            prev_len = current_segments[min_idx-1]['len']
            next_len = current_segments[min_idx+1]['len']
            if prev_len < next_len:
                merge_idx = min_idx
                target_idx = min_idx - 1
            else:
                merge_idx = min_idx + 1
                target_idx = min_idx

        seg_a = current_segments[target_idx]
        seg_b = current_segments[merge_idx]

        new_seg = {
            'symbol': f"({seg_a['symbol']}+{seg_b['symbol']})",
            'notes': seg_a['notes'] + seg_b['notes'],
            'note_indices': seg_a['note_indices'] + seg_b['note_indices'],
            'len': seg_a['len'] + seg_b['len']
        }

        current_segments[target_idx] = new_seg
        del current_segments[merge_idx]

    return current_segments


def melody_to_sequitur_absolute_pitch_symbols(score):
    """
    Convierte la melodía en una secuencia de símbolos
    usando SOLO la altura absoluta (MIDI) de cada nota.
    """
    # 1. Usamos tu función extract_melody ya definida arriba
    seq = extract_melody(score)

    if len(seq) < 2:
        debug("Sequitur abs-pitch: melodía demasiado corta")
        return []

    # 2. Extraemos el valor MIDI (n[0]) de cada tupla (pitch, name, duration)
    symbols = [n[0] for n in seq]

    debug(
        "Sequitur abs-pitch: símbolos =",
        symbols[:12],
        "..." if len(symbols) > 12 else ""
    )

    return symbols

def melody_to_sequitur_with_measures(score):
    """
    Convierte la melodía en símbolos (altura MIDI) manteniendo
    un registro de a qué compás pertenece cada nota.

    Returns:
        symbols: lista de alturas MIDI
        measure_map: lista paralela indicando el número de compás (1-indexed)
    """
    parts = getattr(score, 'parts', [score])
    melody_part = parts[0].flatten().notes

    symbols = []
    measure_map = []

    for n in melody_part:
        if n.isNote:
            # Obtener el compás al que pertenece esta nota
            measure = n.getContextByClass('Measure')
            if measure:
                measure_number = measure.measureNumber
            else:
                measure_number = 0  # Fallback si no se encuentra

            symbols.append(n.pitch.midi)
            measure_map.append(measure_number)

    debug(f"Sequitur tracking: {len(symbols)} notas en {max(measure_map) if measure_map else 0} compases")

    return symbols, measure_map

def sequitur_absolute_pitch_semantic_vector(score, dim=128):
    """
    Genera un vector semántico y un reporte jerárquico detallado
    usando la clase SequiturGrammar.
    """
    # 1. Obtener símbolos (alturas MIDI)
    symbols = melody_to_sequitur_absolute_pitch_symbols(score)

    if len(symbols) < 4:
        debug("Sequitur: secuencia demasiado corta")
        return np.zeros(dim)

    # 2. Construir Gramática con SequiturGrammar
    grammar = SequiturGrammar()
    rules = grammar.build(symbols)
    hierarchy, root_seq = grammar.get_musical_hierarchy()

    # 3. Crear Vector y Reporte Educativo
    vec = np.zeros(dim)

    print("\n" + "="*50)
    print("ANÁLISIS JERÁRQUICO DE LA MELODÍA (SEQUITUR)")
    print("="*50)
    print(f"Estructura Maestra (Root): {root_seq}\n")
    print("Motivos y Frases Identificadas:")

    # Ordenamos las reglas para ver cómo se construyen de menor a mayor
    for rule_name, data in hierarchy.items():
        # Cálculo del vector (seguimos usando hashing para la representación numérica)
        # Usamos la frase expandida para que el hash sea único según el contenido real
        h = int(hashlib.md5(str(data['phrase']).encode()).hexdigest(), 16)
        idx = h % dim

        # El peso es la longitud de la frase por el número de veces que aparece
        # (Esto da más importancia a motivos largos y frecuentes)
        weight = data['length']
        vec[idx] += weight

        # --- Reporte para el usuario ---
        # Traducimos las notas MIDI a nombres de notas para que sea legible
        notas_legibles = [pitch.Pitch(midi).nameWithOctave for midi in data['phrase']]

        print(f"\n▶ {rule_name}:")
        print(f"  └─ Contenido: {data['structure']}")
        print(f"  └─ Notas Reales: {notas_legibles}")
        print(f"  └─ Longitud: {data['length']} notas")
        print(f"  └─ Impacto en Vector (Índice {idx}): +{weight}")

    # Normalización del vector para comparaciones futuras
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    print("\n" + "="*50)
    debug(f"Sequitur: Vector final generado (norma={norm:.3f})")

    return vec

# =========================
# NOVELTY
# =========================

def get_checkerboard_kernel(size):
    """Crea un kernel de tablero de ajedrez suavizado."""
    m = size // 2
    kernel = np.array([[1, -1], [-1, 1]])
    kernel = np.kron(kernel, np.ones((m, m)))
    return gaussian_filter(kernel, sigma=size/6)

def compute_ssm(descriptors):
    """Matriz de Auto-Similitud usando similitud de coseno."""
    X = np.array(descriptors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Evitar división por cero
    X_norm = np.divide(X, norms, out=np.zeros_like(X), where=norms!=0)
    return np.dot(X_norm, X_norm.T)

def novelty_structure_vector(score, kernel_size=8, threshold=0.15, dim=64):
    """
    Genera un vector de estructura basado en segmentación dinámica
    por curva de novedad (Novelty Detection).
    """
    debug("Novelty: Iniciando segmentación dinámica...")

    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))

    if len(measures) < kernel_size:
        debug("Novelty: Score demasiado corto para el kernel actual.")
        return np.zeros(dim)

    # 1. Obtener descriptores compás a compás
    # Reutilizamos tu segment_descriptor pasando una lista con un solo compás
    measure_descriptors = []
    for m in measures:
        d = segment_descriptor([m])
        measure_descriptors.append(d)

    # 2. Calcular SSM y Curva de Novedad
    ssm = compute_ssm(measure_descriptors)
    kernel = get_checkerboard_kernel(kernel_size)

    N = ssm.shape[0]
    novelty_curve = np.zeros(N)
    m = kernel_size // 2

    for i in range(m, N - m):
        sub_matrix = ssm[i - m : i + m, i - m : i + m]
        novelty_curve[i] = np.sum(sub_matrix * kernel)

    if np.max(novelty_curve) > 0:
        novelty_curve /= np.max(novelty_curve)

    # 3. Detectar Fronteras (Picos)
    peaks, _ = find_peaks(novelty_curve, height=threshold, distance=2)
    boundaries = [0] + list(peaks) + [len(measures)]

    # 4. Crear Segmentos Dinámicos
    dynamic_segments = []
    for i in range(len(boundaries) - 1):
        seg = measures[boundaries[i] : boundaries[i+1]]
        dynamic_segments.append(seg)

    debug(f"Novelty: Detectadas {len(dynamic_segments)} secciones dinámicas.")

    # 5. Clustering y Vectorización (reutilizando tu lógica de forma)
    if len(dynamic_segments) < 2:
        return np.zeros(dim)

    # Obtenemos descriptores de las nuevas secciones dinámicas
    descriptors = [segment_descriptor(seg) for seg in dynamic_segments]

    # Clustering (reutilizando tu función cluster_segments)
    try:
        labels = cluster_segments(descriptors, similarity_threshold=0.85)
        form = normalize_form(labels)

        v_stats = form_statistics(form)
        # Añadimos una estadística extra: varianza de la longitud de los segmentos
        seg_lengths = [len(s) for s in dynamic_segments]
        v_extra = np.array([np.std(seg_lengths) / (np.mean(seg_lengths) + 1e-6)])

        v_ngrams = form_ngrams(form, n=2, dim=dim - len(v_stats) - len(v_extra))

        return np.concatenate([v_stats, v_extra, v_ngrams])
    except Exception as e:
        debug(f"Novelty Error: {e}")
        return np.zeros(dim)


# =========================
# N-GRAMAS MELÓDICOS
# =========================

def melodic_ngram_vector(score, n=3, dim=128):
    """
    Vector de n-gramas melódicos basado en intervalos cuantizados.
    Invariante a transposición.
    """
    seq = extract_melody(score)
    if len(seq) < n + 1:
        debug("Melodic n-grams: melodía demasiado corta")
        return np.zeros(dim)

    # Intervalos melódicos
    intervals = [
        quantize_interval(seq[i+1][0] - seq[i][0])
        for i in range(len(seq) - 1)
    ]

    ngrams = Counter(
        tuple(intervals[i:i+n])
        for i in range(len(intervals) - n + 1)
    )

    vec = np.zeros(dim)

    for ng, count in ngrams.items():
        h = int(hashlib.md5(str(ng).encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += count

    # Normalización
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    debug(
        f"Melodic n-grams: n={n}, únicos={len(ngrams)}, norma={norm:.3f}"
    )

    return vec

# =========================
# N-GRAMAS RÍTMICOS
# =========================

def quantize_duration_ratio(r):
    if r < 0.75:
        return "shorter"
    elif r > 1.33:
        return "longer"
    else:
        return "same"


def rhythmic_ngram_vector(score, n=3, dim=128):
    """
    Vector de n-gramas rítmicos basado en ratios de duración.
    Invariante al tempo.
    """
    seq = extract_melody(score)
    if len(seq) < n + 1:
        debug("Rhythmic n-grams: secuencia demasiado corta")
        return np.zeros(dim)

    durations = [n[2] for n in seq]

    ratios = []
    for i in range(len(durations) - 1):
        if durations[i] > 0:
            ratios.append(
                quantize_duration_ratio(durations[i+1] / durations[i])
            )
        else:
            ratios.append("same")

    ngrams = Counter(
        tuple(ratios[i:i+n])
        for i in range(len(ratios) - n + 1)
    )

    vec = np.zeros(dim)

    for ng, count in ngrams.items():
        h = int(hashlib.md5(str(ng).encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += count

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    debug(
        f"Rhythmic n-grams: n={n}, únicos={len(ngrams)}, norma={norm:.3f}"
    )

    return vec

# =========================
# ANÁLISIS DE FORMA AVANZADO (SEQUITUR + CLUSTERING)
# =========================

def compute_segment_ngram_vector(notes, n=3, bins=12):
    """
    Calcula un vector de n-gramas de intervalos para una lista de notas MIDI.
    Usado para comparar el contenido de las secciones de Sequitur.
    """
    # El tamaño total esperado es bins (histograma) + bins (n-gramas)
    total_dim = bins * 2

    if len(notes) < 2:
        # CORRECCIÓN: Devolver ceros del tamaño TOTAL, no solo 'bins'
        return np.zeros(total_dim)

    # 1. Calcular intervalos
    intervals = np.diff(notes)

    # 2. Histograma de intervalos (perfil armónico/melódico simple)
    hist, _ = np.histogram(intervals, bins=bins, range=(-12, 12), density=True)

    # 3. N-gramas de intervalos (perfil secuencial)
    # Cuantizamos intervalos para hacer los n-gramas manejables
    quantized = [max(-12, min(12, i)) for i in intervals]
    ngrams = [tuple(quantized[i:i+n]) for i in range(len(quantized)-n+1)]

    # Hashing trick para vectorizar n-gramas en espacio fijo
    ngram_vec = np.zeros(bins)
    for ng in ngrams:
        h = int(hashlib.md5(str(ng).encode()).hexdigest(), 16)
        ngram_vec[h % bins] += 1

    # Normalizar
    norm_hist = hist / (np.sum(hist) + 1e-9)
    norm_ngram = ngram_vec / (np.linalg.norm(ngram_vec) + 1e-9)

    return np.concatenate([norm_hist, norm_ngram])

def merge_similar_clusters(labels, vectors, similarity_threshold):
    """
    Fusiona clusters cuyos centroides tengan una similitud coseno > threshold.
    """
    unique_labels = sorted(list(set(labels)))
    centroids = {}

    # 1. Calcular centroides
    for l in unique_labels:
        idxs = np.where(labels == l)[0]
        cluster_vecs = vectors[idxs]
        centroid = np.mean(cluster_vecs, axis=0)
        centroids[l] = centroid / (np.linalg.norm(centroid) + 1e-9)

    # 2. Matriz de similitud entre clusters
    mapping = {l: l for l in unique_labels}
    merged_any = False

    # Comparamos todos contra todos
    for l1 in unique_labels:
        for l2 in unique_labels:
            if l1 >= l2: continue # Evitar duplicados y auto-comparación

            # Si ya fueron mapeados al mismo, saltar
            if mapping[l1] == mapping[l2]: continue

            sim = np.dot(centroids[l1], centroids[l2])

            if sim > similarity_threshold:
                debug(f"Form Advanced: Fusionando Cluster {l2} -> {l1} (Sim: {sim:.3f})")
                # Mapear l2 al ID de l1
                old_target = mapping[l2]
                new_target = mapping[l1]

                # Actualizar todos los que apuntaban a old_target
                for k, v in mapping.items():
                    if v == old_target:
                        mapping[k] = new_target
                merged_any = True

    # 3. Reescribir etiquetas
    new_labels = np.array([mapping[l] for l in labels])
    return new_labels


def simplify_sequence(sequence, grammar, target_length):
    """
    Simplifica la secuencia raíz fusionando los segmentos más cortos
    hasta alcanzar target_length.
    """
    # Convertimos la secuencia en una lista de objetos con metadatos
    segments = []
    for s in sequence:
        notes = grammar.expand_rule(s)
        segments.append({
            'symbol': s,
            'notes': notes,
            'len': len(notes)
        })

    current_segments = copy.deepcopy(segments)

    # Bucle de fusión
    while len(current_segments) > target_length:
        min_len = float('inf')
        min_idx = -1

        for i, seg in enumerate(current_segments):
            if seg['len'] < min_len:
                min_len = seg['len']
                min_idx = i

        if min_idx == 0:
            merge_idx = 1
            target_idx = 0
        elif min_idx == len(current_segments) - 1:
            merge_idx = min_idx
            target_idx = min_idx - 1
        else:
            prev_len = current_segments[min_idx-1]['len']
            next_len = current_segments[min_idx+1]['len']
            if prev_len < next_len:
                merge_idx = min_idx
                target_idx = min_idx - 1
            else:
                merge_idx = min_idx + 1
                target_idx = min_idx

        seg_a = current_segments[target_idx]
        seg_b = current_segments[merge_idx]

        new_seg = {
            'symbol': f"({seg_a['symbol']}+{seg_b['symbol']})",
            'notes': seg_a['notes'] + seg_b['notes'],
            'len': seg_a['len'] + seg_b['len']
        }

        current_segments[target_idx] = new_seg
        del current_segments[merge_idx]

    return current_segments

def merge_similar_clusters(labels, vectors, similarity_threshold):
    """
    Fusiona clusters cuyos centroides tengan una similitud coseno > threshold.
    """
    unique_labels = sorted(list(set(labels)))
    centroids = {}

    for l in unique_labels:
        idxs = np.where(labels == l)[0]
        cluster_vecs = vectors[idxs]
        centroid = np.mean(cluster_vecs, axis=0)
        centroids[l] = centroid / (np.linalg.norm(centroid) + 1e-9)

    mapping = {l: l for l in unique_labels}

    for l1 in unique_labels:
        for l2 in unique_labels:
            if l1 >= l2: continue
            if mapping[l1] == mapping[l2]: continue

            sim = np.dot(centroids[l1], centroids[l2])
            if sim > similarity_threshold:
                old_target = mapping[l2]
                new_target = mapping[l1]
                for k, v in mapping.items():
                    if v == old_target:
                        mapping[k] = new_target

    new_labels = np.array([mapping[l] for l in labels])
    return new_labels


def advanced_sequitur_form(score, max_parts=4, n_clusters=4, similarity_threshold=0.85):
    """
    Algoritmo Híbrido con tracking de compases:
    1. Sequitur (Gramática) -> Estructura base
    2. Simplificación -> Reducción a 'max_parts' segmentos
    3. Feature Extraction -> Vectores de cada segmento
    4. Clustering -> Agrupación inicial
    5. Merging -> Combinación por similitud
    6. Form String -> AABA...
    7. Detección de compases por parte
    """
    print("\n" + "="*50)
    print("ANÁLISIS DE FORMA AVANZADO (SEQUITUR + CLUSTERING)")
    print("="*50)

    # 1. Obtener símbolos y mapeo de compases
    symbols, measure_map = melody_to_sequitur_with_measures(score)

    if not symbols:
        debug("No se pudieron obtener símbolos")
        return "N/A"

    # 2. Crear gramática
    grammar = SequiturGrammar()
    grammar.build(symbols)
    root_seq = grammar.root_sequence

    debug(f"Root sequence: {root_seq}")

    # 3. Expandir root_seq con tracking de índices
    if len(root_seq) == 1 and str(root_seq[0]).startswith('R'):
        debug("Raíz unitaria detectada, expandiendo un nivel...")
        root_seq = grammar.rules[root_seq[0]]

    if not root_seq:
        debug("Root sequence vacía después de expansión")
        return "N/A"

    segments = []
    current_index = 0

    for symbol in root_seq:
        notes, indices = expand_rule_with_indices(symbol, grammar, current_index)

        segments.append({
            'symbol': symbol,
            'notes': notes,
            'note_indices': indices,
            'len': len(notes)
        })

        current_index += len(notes)

    debug(f"Segmentos creados: {len(segments)}")

    if not segments:
        debug("No se crearon segmentos")
        return "N/A"

    # 4. Simplificación
    segments = simplify_segments_preserving_indices(segments, max_parts)

    debug(f"Segmentos después de simplificación: {segments}")

    if segments is None:
        debug("ERROR: simplify_segments_preserving_indices devolvió None")
        return "N/A"

    if not segments:
        debug("Lista de segmentos vacía después de simplificación")
        return "N/A"

    # 5. Extracción de características
    vectors = []
    for seg in segments:
        v = compute_segment_ngram_vector(seg['notes'], n=3, bins=12)
        if np.all(v == 0):
            v = v + 1e-9
        vectors.append(v)

    X = np.array(vectors)

    # 6. Clustering
    k = min(n_clusters, len(segments))
    if k < 2:
        final_form = ["A"] * len(segments)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=k,
            affinity='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(X)
        debug(f"Etiquetas clustering inicial: {labels}")

        # 5. Combinar Clusters similares
        refined_labels = merge_similar_clusters(labels, X, similarity_threshold)
        debug(f"Etiquetas refinadas: {refined_labels}")

        # 6. Generar cadena de forma (A, B, C...)
        # Asignamos letras basándonos en el orden de aparición
        label_map = {}
        next_char = ord('A')
        final_form = []

        for l in refined_labels:
            if l not in label_map:
                label_map[l] = chr(next_char)
                next_char += 1
            final_form.append(label_map[l])

    form_string = "".join(final_form)

    # 7. MOSTRAR RESULTADOS CON COMPASES
    print(f"\n📊 Estructura Resultante: {form_string}")
    print("\n🎵 Detalle de Secciones:\n")

    for i, char in enumerate(final_form):
        seg = segments[i]

        # Obtener rango de compases usando measure_map
        if seg['note_indices'] and measure_map:
            measures = [measure_map[idx] for idx in seg['note_indices'] if idx < len(measure_map)]
            if measures:
                min_measure = min(measures)
                max_measure = max(measures)
                measure_range = f"Compases {min_measure}-{max_measure}"
            else:
                measure_range = "Compases: N/A"
        else:
            measure_range = "Compases: N/A"

        # Snippet de notas
        snippet = seg['notes'][:5]
        snippet_str = ", ".join([str(pitch.Pitch(midi).nameWithOctave) for midi in snippet])

        print(f"  Parte {i+1} ({char}):")
        print(f"    ├─ {measure_range}")
        print(f"    ├─ {len(seg['notes'])} notas")
        print(f"    └─ Inicio: [{snippet_str}...]")
        print()

    print("="*50)
    return form_string

def form_string_to_vector(form_string, dim=32):
    """
    Convierte una cadena de forma (AABA) en un vector numérico.
    Captura: complejidad, repeticiones y patrones secuenciales.
    """
    if not form_string or form_string == "N/A":
        return np.zeros(dim)

    # 1. Estadísticas básicas
    n_total = len(form_string)
    n_unique = len(set(form_string))

    # Ratio de redundancia (0 = todo nuevo, 1 = todo repetido)
    redundancy = 1.0 - (n_unique / n_total) if n_total > 0 else 0

    # 2. Distribución de secciones (Histograma)
    counts = Counter(form_string)
    dist = np.array([counts[char] for char in sorted(counts.keys())])
    dist = dist / np.sum(dist)
    # Padding de la distribución para que tenga tamaño 8 (máx secciones probables)
    dist_padded = np.zeros(8)
    dist_padded[:min(len(dist), 8)] = dist[:8]

    # 3. N-gramas de la forma (transiciones estructurales)
    # Ejemplo: AABA -> (A,A), (A,B), (B,A)
    ngram_dim = dim - 10 # Resto del espacio para n-gramas
    v_ngrams = np.zeros(ngram_dim)

    if n_total > 1:
        pairs = [form_string[i:i+2] for i in range(n_total - 1)]
        for p in pairs:
            h = int(hashlib.md5(p.encode()).hexdigest(), 16)
            v_ngrams[h % ngram_dim] += 1
        # Normalizar n-gramas
        v_ngrams = v_ngrams / (np.linalg.norm(v_ngrams) + 1e-9)

    # 4. Concatenación final
    # [n_total, n_unique, redundancy, dist_0...7, ngrams...]
    stats = np.array([n_total, n_unique, redundancy])
    vector = np.concatenate([stats, dist_padded, v_ngrams])

    # Asegurar tamaño exacto
    if len(vector) > dim:
        vector = vector[:dim]
    else:
        vector = np.pad(vector, (0, dim - len(vector)))

    return vector

# =========================
# EJECUCIÓN
# =========================

debug("Cargando score...")
score = converter.parse('./coming_fix.musicxml')
debug("Score cargado correctamente")


# print(rhythmic_features(score))
# print(melodic_features(score))
# print(harmonic_features(score))
# print(harmonic_transition_features(score))
# print(instrumental_features(score))
# print(motif_vector(score))
# print(compass_motif_vector(score))
# print(form_structure_vector(score))
# print(sequitur_absolute_pitch_semantic_vector(score))
# print(novelty_structure_vector(score))
# print(melodic_ngram_vector(score))
# print(rhythmic_ngram_vector(score))
print(advanced_sequitur_form(score))
# print(form_string_to_vector(advanced_sequitur_form(score)))
