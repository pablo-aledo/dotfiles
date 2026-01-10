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
        self.next_rule_id = 1 # R0 suele ser la raíz

    def _new_rule_name(self):
        name = f"R{self.next_rule_id}"
        self.next_rule_id += 1
        return name

    def build(self, sequence):
        """
        Construye una gramática simplificada detectando pares repetidos.
        """
        if not sequence:
            return {}

        symbols = list(sequence)
        changed = True

        while changed:
            changed = False
            counts = {}
            # Contar pares
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                counts[pair] = counts.get(pair, 0) + 1

            # Si un par se repite, creamos regla
            for pair, count in counts.items():
                if count > 1:
                    rule_name = self._new_rule_name()
                    self.rules[rule_name] = list(pair)

                    # Sustituir en la secuencia actual
                    new_symbols = []
                    i = 0
                    while i < len(symbols):
                        if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                            new_symbols.append(rule_name)
                            i += 2
                            changed = True
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                    symbols = new_symbols
                    break

        self.rules["R0"] = symbols
        return self.rules

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

def sequitur_absolute_pitch_semantic_vector(score, dim=128):
    """
    Genera un vector basado en la frecuencia y longitud de las reglas 
    encontradas por Sequitur usando alturas MIDI absolutas.
    """
    # 1. Obtener símbolos (alturas MIDI)
    symbols = melody_to_sequitur_absolute_pitch_symbols(score)

    if len(symbols) < 4:
        debug("Sequitur abs-pitch: secuencia demasiado corta")
        return np.zeros(dim)

    # 2. Construir Gramática
    grammar = SequiturGrammar()
    rules = grammar.build(symbols)

    # 3. Crear Vector
    vec = np.zeros(dim)
    debug("\n=== SEQUITUR ABS-PITCH: GENERANDO VECTOR ===")

    for name, symbols_in_rule in rules.items():
        # if name == "R0": continue # Ignorar la regla raíz para el vector de rasgos

        # Peso basado en longitud de la regla
        h = int(hashlib.md5(str(symbols_in_rule).encode()).hexdigest(), 16)
        idx = h % dim
        weight = len(symbols_in_rule)
        vec[idx] += weight

        debug(f"Regla {name}: {symbols_in_rule} | Peso: {weight} | Índice: {idx}")

    # Normalización
    norm = np.linalg.norm(vec)
    if norm > 0: vec /= norm

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
# EJECUCIÓN
# =========================

debug("Cargando score...")
score = converter.parse('./dreams2.musicxml')
debug("Score cargado correctamente")

# print(rhythmic_features(score))
# print(melodic_features(score))
# print(harmonic_features(score))
# print(harmonic_transition_features(score))
# print(instrumental_features(score))
# print(motif_vector(score))
# print(compass_motif_vector(score))
# print(form_structure_vector(score))
# print(sequitur_semantic_vector(score))
# print(sequitur_pitch_semantic_vector(score))
# print(sequitur_absolute_pitch_semantic_vector(score))
print(novelty_structure_vector(score))

