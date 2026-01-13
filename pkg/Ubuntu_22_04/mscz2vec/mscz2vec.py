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
    debug("Harmony: Extracting chords")
    debug(f"Harmony: stream type = {type(stream)}")

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

    debug(f"Harmony: unique offsets = {len(notes_by_offset)}")

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

            # *** AÑADIR: Obtener el compás de la primera nota del grupo ***
            first_note = group[0]
            measure = first_note.getContextByClass('Measure')
            if measure:
                # Guardar el número de compás como atributo personalizado
                ch._stored_measure_number = measure.measureNumber
            else:
                ch._stored_measure_number = None

            chords_list.append(ch)

            debug(
                "Harmony: chord detected: "
                f"{[p.nameWithOctave for p in ch.pitches]}"
            )

    debug("Harmony: # total chords =", len(chords_list))
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
        debug("Harmony: key =", key)
    except:
        debug("Harmony: could not detect key")
        return np.zeros(len(HARMONIC_FUNCTIONS))

    chords = extract_harmonic_chords(score)

    debug("Harmony: # chords =", len(chords))

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

            # *** MODIFICADO: Obtener el compás del atributo personalizado ***
            measure_num = getattr(chord_obj, '_stored_measure_number', '?')
            if measure_num is None:
                measure_num = '?'

            # Debug con información de compás
            short_name = chord_short_name(chord_obj)
            debug(
                f"Harmony DEBUG: chord = {short_name} | "
                f"notes = {notes_in_chord} | "
                f"figure = {rn.figure} | function = {func} | "
                f"measure = {measure_num}"
            )

        except Exception as e:
            debug("Harmony DEBUG: chord error", chord_obj, e)
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
        debug("Harmony transitions: key =", key)
    except:
        debug("Harmony transitions: no key")
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

    debug("Harmony transitions: functions =", funcs)

    labels = ["T", "PD", "D", "Dsec", "Other"]
    matrix = np.zeros((len(labels), len(labels)))
    idx = {l: i for i, l in enumerate(labels)}

    for i in range(len(funcs) - 1):
        matrix[idx[funcs[i]], idx[funcs[i+1]]] += 1

    matrix /= np.sum(matrix) if np.sum(matrix) > 0 else 1
    return matrix.flatten()

# =========================
# ACORDES ARPEGIADOS
# =========================

def is_bass_clef_staff(element):
    """
    Determina si un elemento pertenece a un pentagrama en clave de fa.
    """
    # Buscar el contexto de Staff
    staff = element.getContextByClass('Staff')
    if staff is None:
        # Si no hay Staff, buscar en el Part
        part = element.getContextByClass('Part')
        if part is None:
            return False
        # Buscar clef en el part
        clefs = part.flatten().getElementsByClass('Clef')
    else:
        # Buscar clef en el staff
        clefs = staff.flatten().getElementsByClass('Clef')

    if not clefs:
        return False

    # Obtener la clave más reciente antes o en la posición del elemento
    element_offset = element.getOffsetInHierarchy(element.getContextByClass('Score'))
    relevant_clef = None

    for clef in clefs:
        clef_offset = clef.getOffsetInHierarchy(clef.getContextByClass('Score'))
        if clef_offset <= element_offset:
            relevant_clef = clef
        else:
            break

    if relevant_clef is None:
        return False

    # Verificar si es clave de fa
    # Bass clef, F clef, Subbass clef
    from music21 import clef as clef_module

    is_bass = isinstance(relevant_clef, (
        clef_module.BassClef,
        clef_module.Bass8vbClef,
        clef_module.Bass8vaClef,
        clef_module.FClef
    ))

    return is_bass


def extract_bass_elements(part):
    """
    Extrae elementos (notas y acordes) del pentagrama en clave de fa.
    Retorna una lista de tuplas (element, offset, pitches)
    donde pitches es una lista de objetos Pitch.
    """
    elements = []

    for el in part.flatten().notesAndRests:
        # Saltar silencios
        if el.isRest:
            continue

        # Solo elementos en clave de fa
        if not is_bass_clef_staff(el):
            continue

        offset = el.offset

        if el.isNote:
            # Nota individual
            elements.append({
                'element': el,
                'offset': offset,
                'pitches': [el.pitch],
                'type': 'note'
            })
        elif el.isChord:
            # Acorde
            elements.append({
                'element': el,
                'offset': offset,
                'pitches': list(el.pitches),
                'type': 'chord'
            })

    return elements


def detect_alberti_bass(elements, start_idx, max_notes=8):
    """
    Detecta patrones de bajo de Alberti.
    Patrón típico: bajo - agudo - medio - agudo (se repite)
    Ejemplo: Do - Sol - Mi - Sol - Do - Sol - Mi - Sol

    Características del bajo Alberti:
    1. Mínimo 4 notas (un ciclo completo)
    2. Las notas forman un acorde cuando se consideran todas juntas
    3. Patrón repetitivo con movimiento característico
    4. La nota más grave suele aparecer en posiciones 1, 5, 9... (cada 4 notas)
    """
    if start_idx >= len(elements):
        return None

    # Recolectar notas consecutivas individuales
    notes = []
    i = start_idx

    while i < len(elements) and len(notes) < max_notes:
        if elements[i]['type'] != 'note':
            break
        notes.append(elements[i])
        i += 1

    if len(notes) < 4:
        return None

    # Extraer las alturas MIDI
    pitches = [n['pitches'][0].midi for n in notes]

    # Verificar patrón de bajo Alberti
    # Características:
    # 1. La primera nota suele ser la más grave
    # 2. Hay un patrón de saltos (no es cromático ni escalar puro)
    # 3. Se repite cada 3-4 notas

    # Detectar si hay un patrón repetitivo cada 3 o 4 notas
    for pattern_length in [3, 4]:
        if len(notes) < pattern_length * 2:
            continue

        # Extraer el patrón inicial (en términos de intervalos relativos)
        pattern = []
        base_pitch = pitches[0]
        for j in range(pattern_length):
            pattern.append(pitches[j] - base_pitch)

        # Verificar si el patrón se repite
        matches = 0
        for start in range(pattern_length, len(pitches) - pattern_length + 1, pattern_length):
            chunk_base = pitches[start]
            chunk_pattern = [pitches[start + j] - chunk_base for j in range(pattern_length)]

            if chunk_pattern == pattern:
                matches += 1

        # Si encontramos al menos una repetición completa
        if matches >= 1:
            # Verificar que las notas forman un acorde
            unique_pitches = list(set([p % 12 for p in pitches]))

            if len(unique_pitches) >= 3:
                # Calcular cuántas notas forman el patrón completo
                total_pattern_notes = pattern_length * (matches + 1)

                return {
                    'elements': notes[:total_pattern_notes],
                    'all_pitches': [n['pitches'][0] for n in notes[:total_pattern_notes]],
                    'end_index': start_idx + total_pattern_notes,
                    'pattern_type': 'alberti',
                    'pattern_length': pattern_length,
                    'repetitions': matches + 1
                }

    # Verificar patrón de bajo Alberti menos estricto:
    # Movimiento bajo-alto-medio-alto (sin necesidad de repetición exacta)
    if len(notes) >= 4:
        # Verificar que hay saltos característicos (no movimiento escalar)
        intervals = [abs(pitches[j+1] - pitches[j]) for j in range(len(pitches)-1)]
        avg_interval = np.mean(intervals)

        # Si hay saltos promedio de 3+ semitonos (terceras o mayores)
        if avg_interval >= 3:
            # Verificar que las notas forman un acorde
            unique_pitches = list(set([p % 12 for p in pitches[:4]]))

            if len(unique_pitches) >= 3:
                # Tomar hasta 8 notas o hasta que cambie el patrón
                alberti_notes = notes[:min(8, len(notes))]

                return {
                    'elements': alberti_notes,
                    'all_pitches': [n['pitches'][0] for n in alberti_notes],
                    'end_index': start_idx + len(alberti_notes),
                    'pattern_type': 'alberti',
                    'pattern_length': 4,
                    'repetitions': len(alberti_notes) // 4
                }

    return None


def detect_bass_chord_pattern(elements, start_idx, time_window):
    """
    Detecta el patrón: nota/acorde + acorde formando un acorde completo.
    Por ejemplo: Do (negra) + (Mi+Sol) (negra)
    """
    if start_idx >= len(elements) - 1:
        return None

    first_element = elements[start_idx]
    first_offset = first_element['offset']

    # Buscar el siguiente elemento que sea un acorde
    for i in range(start_idx + 1, len(elements)):
        next_element = elements[i]

        # Si está fuera de la ventana temporal, terminar
        if next_element['offset'] - first_offset > time_window:
            break

        # Si el siguiente elemento es un acorde (2+ notas)
        if next_element['type'] == 'chord' and len(next_element['pitches']) >= 2:
            # Combinar las notas del primer elemento con el acorde
            all_pitches = first_element['pitches'] + next_element['pitches']
            unique_pitches = list(set([p.midi % 12 for p in all_pitches]))

            # Verificar que tenemos al menos 3 notas únicas
            if len(unique_pitches) >= 3:
                return {
                    'elements': [first_element, next_element],
                    'all_pitches': all_pitches,
                    'end_index': i + 1,
                    'pattern_type': 'bass+chord'
                }

    return None


def detect_arpeggiated_chords(score, time_window=2.0, min_notes=3):
    """
    Detecta acordes arpegiados analizando notas sucesivas en una ventana temporal.
    SOLO analiza pentagramas en clave de fa.

    Detecta tres patrones:
    1. Arpegios tradicionales (notas consecutivas ascendentes/descendentes)
    2. Bajo + acorde (nota/acorde + acorde simultáneo)
       Ejemplo: Do (negra) + (Mi+Sol) (negra)
    3. Bajo de Alberti (patrón bajo-alto-medio-alto repetitivo)
       Ejemplo: Do - Sol - Mi - Sol - Do - Sol - Mi - Sol

    Args:
        score: partitura music21
        time_window: ventana temporal en quarterLength para considerar un arpegio
        min_notes: número mínimo de notas diferentes para formar un acorde

    Returns:
        lista de dict con información de los acordes detectados
    """
    debug("Arpeggios: Detecting arpeggiated chords (bass clef only)...")

    parts = getattr(score, 'parts', [score])
    arpeggiated_chords = []

    for part_idx, part in enumerate(parts):
        # Extraer elementos (notas y acordes) en clave de fa
        elements = extract_bass_elements(part)

        debug(f"Arpeggios: Part {part_idx} has {len(elements)} elements in bass clef")

        if len(elements) == 0:
            debug(f"Arpeggios: Part {part_idx} has no bass clef elements, skipping...")
            continue

        i = 0
        while i < len(elements):
            current_element = elements[i]
            current_offset = current_element['offset']

            # PRIMERO: Intentar detectar patrón de bajo Alberti
            alberti_pattern = detect_alberti_bass(elements, i)

            if alberti_pattern:
                all_pitches = alberti_pattern['all_pitches']
                pattern_type = 'ALBERTI'

                try:
                    chord_pitches = sorted(list(set(all_pitches)), key=lambda p: p.midi)
                    ch = chord.Chord(chord_pitches)

                    # Obtener compás del primer elemento
                    measure = alberti_pattern['elements'][0]['element'].getContextByClass('Measure')
                    measure_num = measure.measureNumber if measure else '?'

                    # Información del patrón
                    note_names = [p.nameWithOctave for p in all_pitches]
                    pattern_info = f"{pattern_type} (length={alberti_pattern['pattern_length']}, reps={alberti_pattern['repetitions']})"

                    arpeggiated_chords.append({
                        'chord': ch,
                        'measure': measure_num,
                        'notes': note_names,
                        'offset': current_offset,
                        'direction': pattern_type,
                        'span': alberti_pattern['elements'][-1]['offset'] - current_offset,
                        'part': part_idx,
                        'pattern_info': pattern_info
                    })

                    debug(f"Arpeggios: Found {pattern_info} at measure {measure_num}: {note_names}")

                    # Saltar los elementos que ya procesamos
                    i = alberti_pattern['end_index']
                    continue

                except Exception as e:
                    debug(f"Arpeggios: Error creating chord from Alberti pattern: {e}")

            # SEGUNDO: Intentar detectar patrón bajo+acorde
            bass_chord_pattern = detect_bass_chord_pattern(elements, i, time_window)

            if bass_chord_pattern:
                all_pitches = bass_chord_pattern['all_pitches']
                pattern_type = 'BASS+CHORD'

                try:
                    chord_pitches = sorted(list(set(all_pitches)), key=lambda p: p.midi)
                    ch = chord.Chord(chord_pitches)

                    # Obtener compás del primer elemento
                    measure = bass_chord_pattern['elements'][0]['element'].getContextByClass('Measure')
                    measure_num = measure.measureNumber if measure else '?'

                    # Información del patrón
                    note_names = [p.nameWithOctave for p in all_pitches]

                    arpeggiated_chords.append({
                        'chord': ch,
                        'measure': measure_num,
                        'notes': note_names,
                        'offset': current_offset,
                        'direction': pattern_type,
                        'span': bass_chord_pattern['elements'][-1]['offset'] - current_offset,
                        'part': part_idx,
                        'pattern_info': pattern_type
                    })

                    debug(f"Arpeggios: Found {pattern_type} at measure {measure_num}: {note_names}")

                    # Saltar los elementos que ya procesamos
                    i = bass_chord_pattern['end_index']
                    continue

                except Exception as e:
                    debug(f"Arpeggios: Error creating chord from bass+chord pattern: {e}")

            # TERCERO: Si no es bajo+acorde ni Alberti, intentar detectar arpegio tradicional
            # Solo con notas individuales consecutivas
            if current_element['type'] == 'note':
                # Recolectar notas individuales consecutivas dentro de la ventana temporal
                window_notes = [current_element['element']]
                j = i + 1

                while j < len(elements):
                    next_element = elements[j]

                    # Solo considerar notas individuales para arpegios
                    if next_element['type'] != 'note':
                        break

                    if next_element['offset'] - current_offset <= time_window:
                        window_notes.append(next_element['element'])
                        j += 1
                    else:
                        break

                # Verificar si las notas forman un acorde válido
                if len(window_notes) >= min_notes:
                    # Obtener pitches únicos
                    unique_pitches = list(set([n.pitch.midi % 12 for n in window_notes]))

                    # Si hay al menos min_notes pitches únicos (mod 12)
                    if len(unique_pitches) >= min_notes:
                        # Verificar que las notas son principalmente ascendentes o descendentes
                        pitches_sequence = [n.pitch.midi for n in window_notes]

                        # Calcular dirección predominante
                        directions = [pitches_sequence[k+1] - pitches_sequence[k]
                                    for k in range(len(pitches_sequence)-1)]

                        ascending = sum(1 for d in directions if d > 0)
                        descending = sum(1 for d in directions if d < 0)

                        # Si hay una dirección predominante (al menos 60% en una dirección)
                        total_moves = ascending + descending
                        if total_moves > 0:
                            predominance = max(ascending, descending) / total_moves

                            if predominance >= 0.6:
                                # Crear acorde con todas las notas únicas
                                try:
                                    chord_pitches = sorted(list(set([n.pitch for n in window_notes])),
                                                         key=lambda p: p.midi)
                                    ch = chord.Chord(chord_pitches)

                                    # Obtener compás
                                    measure = current_element['element'].getContextByClass('Measure')
                                    measure_num = measure.measureNumber if measure else '?'

                                    # Información del arpegio
                                    note_names = [n.pitch.nameWithOctave for n in window_notes]
                                    direction = "ASC" if ascending > descending else "DESC"

                                    arpeggiated_chords.append({
                                        'chord': ch,
                                        'measure': measure_num,
                                        'notes': note_names,
                                        'offset': current_offset,
                                        'direction': direction,
                                        'span': window_notes[-1].offset - current_offset,
                                        'part': part_idx,
                                        'pattern_info': direction
                                    })

                                    debug(f"Arpeggios: Found {direction} arpegio at measure {measure_num}: {note_names}")

                                    # Saltar las notas que ya procesamos
                                    i = j
                                    continue

                                except Exception as e:
                                    debug(f"Arpeggios: Error creating chord: {e}")

            i += 1

    debug(f"Arpeggios: Total arpeggiated chords found = {len(arpeggiated_chords)}")
    return arpeggiated_chords


def arpeggiated_chord_features(score, time_window=2.0, min_notes=3):
    """
    Genera un vector de características basado en acordes arpegiados.
    SOLO analiza pentagramas en clave de fa.

    Returns:
        vector numpy con:
        - Funciones armónicas de arpegios (5 dims)
        - Estadísticas de arpegios (6 dims) - añadidas para bajo+acorde y Alberti
        - Distribución direccional (4 dims) - añadida para Alberti
        - Patrones de intervalos en arpegios (16 dims)
    """
    try:
        key = score.analyze('key')
        debug("Arpeggios: key =", key)
    except:
        debug("Arpeggios: could not detect key")
        key = None

    arpeggios = detect_arpeggiated_chords(score, time_window, min_notes)

    if not arpeggios:
        debug("Arpeggios: No arpeggiated chords found in bass clef")
        return np.zeros(31)

    # ── 1. FUNCIONES ARMÓNICAS (5 dims)
    function_counts = {f: 0 for f in HARMONIC_FUNCTIONS}

    print("\n" + "="*50)
    print("ARPEGGIATED CHORDS ANALYSIS (BASS CLEF ONLY)")
    print("="*50)

    for arp_info in arpeggios:
        chord_obj = arp_info['chord']
        measure_num = arp_info['measure']
        notes = arp_info['notes']
        direction = arp_info['direction']
        pattern_info = arp_info.get('pattern_info', direction)

        if key:
            try:
                rn = roman.romanNumeralFromChord(chord_obj, key)
                func = roman_to_function(rn)
                function_counts[func] += 1

                short_name = chord_short_name(chord_obj)

                print(f"\nChord found:")
                print(f"  ├─ Measure: {measure_num}")
                print(f"  ├─ Pattern: {pattern_info}")
                print(f"  ├─ Notes: {notes}")
                print(f"  ├─ Chord: {short_name}")
                print(f"  ├─ Figure: {rn.figure}")
                print(f"  └─ Function: {func}")

            except Exception as e:
                debug(f"Arpeggios: Error analyzing chord: {e}")
                function_counts["Other"] += 1
        else:
            function_counts["Other"] += 1

    total_funcs = sum(function_counts.values())
    harmonic_vector = np.array(
        [function_counts[f] / total_funcs if total_funcs > 0 else 0
         for f in HARMONIC_FUNCTIONS]
    )

    # ── 2. ESTADÍSTICAS DE ARPEGIOS (6 dims)
    spans = [arp['span'] for arp in arpeggios]
    note_counts = [len(arp['notes']) for arp in arpeggios]

    # Contar patrones por tipo
    bass_chord_count = sum(1 for arp in arpeggios if arp['direction'] == 'BASS+CHORD')
    alberti_count = sum(1 for arp in arpeggios if arp['direction'] == 'ALBERTI')
    bass_chord_ratio = bass_chord_count / len(arpeggios) if arpeggios else 0
    alberti_ratio = alberti_count / len(arpeggios) if arpeggios else 0

    stats_vector = np.array([
        len(arpeggios),  # número total de arpegios
        np.mean(spans) if spans else 0,  # duración promedio
        np.mean(note_counts) if note_counts else 0,  # notas promedio por arpegio
        np.std(note_counts) if note_counts else 0,  # variabilidad
        bass_chord_ratio,  # proporción de patrones bajo+acorde
        alberti_ratio  # proporción de patrones Alberti
    ])

    # ── 3. DISTRIBUCIÓN DIRECCIONAL (4 dims)
    ascending_count = sum(1 for arp in arpeggios if arp['direction'] == 'ASC')
    descending_count = sum(1 for arp in arpeggios if arp['direction'] == 'DESC')

    direction_vector = np.array([
        ascending_count / len(arpeggios) if arpeggios else 0,
        descending_count / len(arpeggios) if arpeggios else 0,
        bass_chord_count / len(arpeggios) if arpeggios else 0,
        alberti_count / len(arpeggios) if arpeggios else 0
    ])

    # ── 4. PATRONES DE INTERVALOS EN ARPEGIOS (16 dims)
    interval_hist = np.zeros(16)

    for arp_info in arpeggios:
        chord_obj = arp_info['chord']
        pitches = sorted([p.midi for p in chord_obj.pitches])

        # Calcular intervalos entre notas del acorde
        for i in range(len(pitches) - 1):
            interval = pitches[i+1] - pitches[i]
            # Mapear intervalos a bins (0-15 semitonos)
            if 0 <= interval < 16:
                interval_hist[interval] += 1

    # Normalizar
    if np.sum(interval_hist) > 0:
        interval_hist /= np.sum(interval_hist)

    # ── CONCATENAR TODO
    final_vector = np.concatenate([
        harmonic_vector,      # 5 dims
        stats_vector,         # 6 dims
        direction_vector,     # 4 dims
        interval_hist         # 16 dims
    ])

    print("\n" + "="*50)
    print(f"Total found chords: {len(arpeggios)}")
    print(f"  ├─ Ascending: {ascending_count}")
    print(f"  ├─ Descending: {descending_count}")
    print(f"  ├─ Bass + Chord: {bass_chord_count}")
    print(f"  └─ Alberti: {alberti_count}")
    print("="*50)

    debug(f"Arpeggios: Final vector shape = {final_vector.shape}")
    return final_vector

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
    debug("Rhythm: sequence =", sequence[:10], "...")
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
    debug("Rhythm: sequence length =", len(sequence))

    if len(sequence) == 0:
        debug("Rhythm: empty sequence")
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
    debug("Instrumentation: # instruments =", len(instruments))

    for inst in instruments:
        debug("Instrument:", inst.instrumentName, inst.partName)
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
    debug("Motifs: melody length =", len(seq))
    if len(seq) > 0:
        debug("Motifs: First 5 notes =", seq[:5])
    return seq

def melodic_intervals(sequence):
    intervals = [sequence[i+1][0] - sequence[i][0] for i in range(len(sequence) - 1)]
    debug("Motifs: First 10 intervals =", intervals[:10])
    return intervals

def rhythmic_ratios(sequence):
    ratios = []
    for i in range(len(sequence) - 1):
        if sequence[i][2] > 0:
            r = sequence[i+1][2] / sequence[i][2]
        else:
            r = 1.0
        ratios.append(r)
    debug("Motifs: first 10 rhythmic ratios =", ratios[:10])
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
            debug(f"Motif accepted (len={len(motif)}, count={count}, posiciones={len(positions)})")

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
    motif_positions = defaultdict(list)

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

    debug(f"Total motifs before filter = {len(all_motifs)}")

    # 2. Filtrar solo motifs máximos (adaptado para incluir posiciones)
    filtered = filter_maximal_motifs_with_positions(all_motifs, top_n)

    debug(f"Final motifs = {len(filtered)}")
    for i, (m, notes, c, positions) in enumerate(filtered):
        note_info = [(n[1], n[2]) for n in notes]
        debug(f"Motif {i+1}: len={len(m)} | reps={c} | notes={note_info} | positions={positions[:5]}...")

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
        debug("Melody with measures: First 5 notes =", seq[:5])

    return seq

def motif_vector(score, dim=128, min_length=3, max_length=6, top_n=10):
    """
    Genera vector con los `top_n` motifs más frecuentes.
    Ahora muestra información de compases donde aparecen.
    """
    seq = extract_melody_with_measures(score)
    seq_len = len(seq)

    if seq_len < min_length + 1:
        debug(f"Motifs: sequence too short (len={seq_len}, min_length={min_length})")
        return np.zeros(dim, dtype=float)

    # Convertir a formato compatible con extract_top_motifs (sin measure_num para cálculo)
    seq_for_motifs = [(n[0], n[1], n[2]) for n in seq]

    motifs = extract_top_motifs(seq_for_motifs, min_length, max_length, top_n)
    if not motifs:
        debug("Motifs: no motifs")
        return np.zeros(dim, dtype=float)

    vec = np.zeros(dim, dtype=float)

    print("\n" + "="*50)
    print("Melodic motifs detected")
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

        print(f"\nMotif {m_idx+1}:")
        print(f"  ├─ Notes: {note_info}")
        print(f"  ├─ Repetitions: {count}")
        print(f"  ├─ In measures: {measures_where_appears}" + ("..." if len(positions) > 5 else ""))
        print(f"  └─ Index: {idx}")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        debug("Motifs: Empty vector")

    print("="*50)
    debug(f"Motifs: Final vector, norm={norm:.3f}, dimension={dim}")
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

    debug("Motifs by measure: # measures =", len(measures))

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
    measure_locations = defaultdict(list)

    for i, measure in enumerate(melody_per_measure):
        # Convertimos cada compás a una tupla para que sea hashable
        measure_counter[tuple(measure)] += 1
        measure_locations[tuple(measure)].append(i)

    debug(f"Motifs by measure: # unique measures = {len(measure_counter)}")
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
        debug("Motifs by measure: no repeated measures")
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

        debug(f"Motif {idx+1}: Measure = {measure} | Repetitions = {count} → index {measure_idx} | "
              f"in measures: {compasses}")

    # Normalizamos el vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        debug("Motifs by measure: empty vector")

    debug(f"Motifs by measure: Final vector, norm={norm:.3f}, dimension={dim}")
    return vec


# =========================
# FORMA
# =========================

def segment_by_measures(score, window_size=4):
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))
    debug("Form: # measures =", len(measures))
    segments = []

    for i in range(0, len(measures), window_size):
        segments.append(measures[i:i+window_size])

    debug("Form: # segments =", len(segments))
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
        return v

    return v / norm

def cluster_segments(descriptors, similarity_threshold=0.85):
    debug("Clustering: # descriptors =", len(descriptors))
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

    debug("Normalized form:", form)
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
        debug("Form: All the segments are empty.")
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

    debug("Form:", form)

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

class CustomRule:
    """
    Representa una regla personalizada definida por el usuario.
    Puede ser:
    - Un compás específico
    - Un conjunto de notas (alturas MIDI)
    - Una regla recursiva que referencia otras reglas
    """
    def __init__(self, name, rule_type, content):
        """
        Args:
            name: nombre de la regla (ej: "intro", "estribillo")
            rule_type: "measure", "notes", o "recursive"
            content:
                - Si type="measure": número de compás (int)
                - Si type="notes": lista de alturas MIDI [60, 62, 64]
                - Si type="recursive": lista de nombres de reglas ["intro", "intro"]
        """
        self.name = name
        self.rule_type = rule_type
        self.content = content

    def __repr__(self):
        return f"CustomRule({self.name}, {self.rule_type}, {self.content})"


class SequiturWithCustomRules:
    """
    Versión extendida de Sequitur que acepta reglas personalizadas iniciales.
    """
    def __init__(self, custom_rules=None):
        """
        Args:
            custom_rules: lista de objetos CustomRule
        """
        self.rules = {}
        self.next_rule_id = 1
        self.root_sequence = []
        self.custom_rules = custom_rules or []
        self.custom_rule_map = {}  # nombre -> símbolos expandidos

    def _get_new_rule_name(self):
        name = f"R{self.next_rule_id}"
        self.next_rule_id += 1
        return name

    def _expand_custom_rule(self, custom_rule, symbols, measure_map=None):
        """
        Convierte una CustomRule en una secuencia de símbolos (notas MIDI).
        """
        if custom_rule.rule_type == "notes":
            # Directamente una lista de notas MIDI
            return custom_rule.content

        elif custom_rule.rule_type == "measure":
            # Extraer notas de un compás específico
            if measure_map is None:
                debug(f"Error: measure_map needed for measure rule '{custom_rule.name}'")
                return []

            measure_num = custom_rule.content
            # Encontrar índices de notas en ese compás
            note_indices = [i for i, m in enumerate(measure_map) if m == measure_num]
            return [symbols[i] for i in note_indices if i < len(symbols)]

        elif custom_rule.rule_type == "recursive":
            # Expandir recursivamente otras reglas
            expanded = []
            for ref_name in custom_rule.content:
                if ref_name in self.custom_rule_map:
                    expanded.extend(self.custom_rule_map[ref_name])
                else:
                    debug(f"Warning: rule '{ref_name}' not found, ignoring")
            return expanded

        return []

    def _initialize_custom_rules(self, symbols, measure_map=None):
        """
        Procesa las reglas personalizadas y las convierte en símbolos.
        Las almacena en self.rules y self.custom_rule_map.
        """
        print("\n" + "="*50)
        print("Initializing custom rules")
        print("="*50)

        for custom_rule in self.custom_rules:
            expanded = self._expand_custom_rule(custom_rule, symbols, measure_map)

            if expanded:
                self.custom_rule_map[custom_rule.name] = expanded
                self.rules[custom_rule.name] = expanded

                # Mostrar información
                notes_str = ", ".join([str(pitch.Pitch(midi).nameWithOctave) for midi in expanded[:8]])
                if len(expanded) > 8:
                    notes_str += "..."

                print(f"\n Rule '{custom_rule.name}':")
                print(f"  ├─ Type: {custom_rule.rule_type}")
                print(f"  ├─ Content: {custom_rule.content}")
                print(f"  ├─ Notes: [{notes_str}]")
                print(f"  └─ Length: {len(expanded)} notes")

        print("="*50 + "\n")

    def _replace_custom_rules_in_sequence(self, symbols):
        """
        Busca y reemplaza ocurrencias de reglas personalizadas en la secuencia.
        Retorna una nueva secuencia con los reemplazos aplicados.
        """
        if not self.custom_rule_map:
            return symbols

        current_seq = list(symbols)
        replacements_made = 0

        # Ordenar reglas por longitud (más largas primero para evitar solapamientos)
        sorted_rules = sorted(
            self.custom_rule_map.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for rule_name, pattern in sorted_rules:
            pattern_len = len(pattern)
            if pattern_len == 0:
                continue

            # Buscar y reemplazar el patrón
            i = 0
            new_seq = []

            while i < len(current_seq):
                # Verificar si hay coincidencia
                if i + pattern_len <= len(current_seq):
                    if current_seq[i:i+pattern_len] == pattern:
                        new_seq.append(rule_name)
                        i += pattern_len
                        replacements_made += 1
                        continue

                new_seq.append(current_seq[i])
                i += 1

            current_seq = new_seq

        debug(f"Replacements of custom rules: {replacements_made}")
        return current_seq

    def build(self, symbols, measure_map=None):
        """
        Construye la gramática Sequitur con soporte para reglas personalizadas.

        Args:
            symbols: lista de alturas MIDI
            measure_map: mapeo de índice de nota -> número de compás (opcional)
        """
        if not symbols:
            return {}

        # 1. Inicializar reglas personalizadas
        self._initialize_custom_rules(symbols, measure_map)

        # 2. Reemplazar ocurrencias de reglas personalizadas en la secuencia
        current_seq = self._replace_custom_rules_in_sequence(symbols)

        debug(f"Sequence after applying custom rules: {current_seq[:20]}...")

        # 3. Aplicar algoritmo Re-Pair (igual que antes)
        while True:
            # Contar pares adyacentes
            pairs = Counter()
            for i in range(len(current_seq) - 1):
                pair = (current_seq[i], current_seq[i+1])
                pairs[pair] += 1

            if not pairs:
                break

            most_common_pair, count = pairs.most_common(1)[0]

            if count < 2:
                break

            # Crear nueva regla
            rule_name = self._get_new_rule_name()
            self.rules[rule_name] = list(most_common_pair)

            # Reemplazar en la secuencia
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
        Despliega recursivamente una regla.
        """
        if symbol not in self.rules:
            return [symbol]

        expansion = []
        for s in self.rules[symbol]:
            expansion.extend(self.expand_rule(s))
        return expansion

    def get_musical_hierarchy(self):
        """
        Devuelve jerarquía de reglas (tanto automáticas como personalizadas).
        """
        hierarchy = {}
        all_rules = list(self.rules.keys())

        # Separar reglas personalizadas de automáticas
        custom_rule_names = set(self.custom_rule_map.keys())
        auto_rules = [r for r in all_rules if r not in custom_rule_names]

        # Ordenar reglas automáticas por ID
        sorted_auto = sorted([r for r in auto_rules if r.startswith('R')],
                           key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

        # Primero mostrar reglas personalizadas, luego automáticas
        sorted_keys = list(custom_rule_names) + sorted_auto

        for rule_name in sorted_keys:
            if rule_name not in self.rules:
                continue

            flat_notes = self.expand_rule(rule_name)
            structure = self.rules[rule_name]

            hierarchy[rule_name] = {
                "structure": structure,
                "phrase": flat_notes,
                "length": len(flat_notes),
                "is_custom": rule_name in custom_rule_names
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
        debug("Sequitur abs-pitch: melody too short")
        return []

    # 2. Extraemos el valor MIDI (n[0]) de cada tupla (pitch, name, duration)
    symbols = [n[0] for n in seq]

    debug(
        "Sequitur abs-pitch: symbols =",
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

    debug(f"Sequitur tracking: {len(symbols)} notes in {max(measure_map) if measure_map else 0} measures")

    return symbols, measure_map

def sequitur_absolute_pitch_semantic_vector(score, dim=128):
    """
    Genera un vector semántico y un reporte jerárquico detallado
    usando la clase SequiturGrammar.
    """
    # 1. Obtener símbolos (alturas MIDI)
    symbols = melody_to_sequitur_absolute_pitch_symbols(score)

    if len(symbols) < 4:
        debug("Sequitur: sequence too short")
        return np.zeros(dim)

    # 2. Construir Gramática con SequiturGrammar
    grammar = SequiturWithCustomRules()
    rules = grammar.build(symbols)
    hierarchy, root_seq = grammar.get_musical_hierarchy()

    # 3. Crear Vector y Reporte Educativo
    vec = np.zeros(dim)

    print("\n" + "="*50)
    print("MELODY ANALYSIS (SEQUITUR)")
    print("="*50)
    print(f"Root sequence: {root_seq}\n")

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

        print(f"\n {rule_name}:")
        print(f"  └─ Content: {data['structure']}")
        print(f"  └─ Notes: {notas_legibles}")
        print(f"  └─ Length: {data['length']} notas")
        print(f"  └─ Weight (Index {idx}): +{weight}")

    # Normalización del vector para comparaciones futuras
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    print("\n" + "="*50)
    debug(f"Sequitur: Final vector (norma={norm:.3f})")

    return vec


def sequitur_absolute_pitch_semantic_vector_with_custom_rules(score, custom_rules=None, dim=128):
    """
    Genera un vector semántico usando Sequitur con reglas personalizadas.

    Args:
        score: partitura music21
        custom_rules: lista de objetos CustomRule (opcional)
        dim: dimensión del vector resultante

    Ejemplo de uso:
        custom_rules = [
            CustomRule("intro", "measure", 1),  # Usar compás 1 como regla "intro"
            CustomRule("motivo_a", "notes", [60, 62, 64, 65]),  # Do-Re-Mi-Fa
            CustomRule("estribillo", "recursive", ["intro", "intro"])  # intro dos veces
        ]
        vec = sequitur_absolute_pitch_semantic_vector_with_custom_rules(score, custom_rules)
    """
    # 1. Obtener símbolos y mapeo de compases
    symbols, measure_map = melody_to_sequitur_with_measures(score)

    if len(symbols) < 4:
        debug("Sequitur: sequence too short")
        return np.zeros(dim)

    # 2. Construir Gramática con reglas personalizadas
    grammar = SequiturWithCustomRules(custom_rules=custom_rules)
    rules = grammar.build(symbols, measure_map)
    hierarchy, root_seq = grammar.get_musical_hierarchy()

    # 3. Crear Vector y Reporte Educativo
    vec = np.zeros(dim)

    print("\n" + "="*50)
    print("MELODY ANALYSIS (SEQUITUR + CUSTOM RULES)")
    print("="*50)
    print(f"Root sequence: {root_seq}\n")

    if custom_rules:
        print("Custom rules:")
        for cr in custom_rules:
            print(f"  - {cr.name} ({cr.rule_type})")
        print()

    print("Motifs and phrases:")

    # Función auxiliar para encontrar compases de una frase
    def get_measure_range_from_notes(note_indices):
        if not note_indices or not measure_map:
            return "N/A"
        measures = [measure_map[i] for i in note_indices if i < len(measure_map)]
        if not measures:
            return "N/A"
        return f"{min(measures)}-{max(measures)}"

    # Rastrear índices para cada regla
    def get_note_indices_for_rule(rule_name, start_idx=0):
        """Devuelve los índices de notas que componen una regla."""
        if rule_name not in grammar.rules:
            return [start_idx]

        indices = []
        current = start_idx
        for symbol in grammar.rules[rule_name]:
            sub_indices = get_note_indices_for_rule(symbol, current)
            indices.extend(sub_indices)
            current += len(sub_indices)
        return indices

    # Separar reglas personalizadas de automáticas
    custom_rule_names = [cr.name for cr in (custom_rules or [])]

    # Mostrar primero reglas personalizadas
    if custom_rule_names:
        print("\n--- CUSTOM RULES ---")

    for rule_name, data in hierarchy.items():
        if not data.get('is_custom', False):
            continue

        # Cálculo del vector
        h = int(hashlib.md5(str(data['phrase']).encode()).hexdigest(), 16)
        idx = h % dim
        weight = data['length']
        vec[idx] += weight

        # Traducimos las notas MIDI a nombres de notas
        notas_legibles = [pitch.Pitch(midi).nameWithOctave for midi in data['phrase']]

        # Obtener índices de notas para esta regla
        note_indices = get_note_indices_for_rule(rule_name)
        measure_range = get_measure_range_from_notes(note_indices)

        print(f"\n {rule_name} (CUSTOM):")
        print(f"  ├─ Content: {data['structure']}")
        print(f"  ├─ Notes: {notas_legibles}")
        print(f"  ├─ Length: {data['length']} notes")
        print(f"  ├─ Measure: {measure_range}")
        print(f"  └─ Weight (Index {idx}): +{weight}")

    # Mostrar reglas automáticas
    auto_rules_exist = any(not data.get('is_custom', False) for data in hierarchy.values())
    if auto_rules_exist:
        print("\n--- AUTOMATIC RULES (SEQUITUR) ---")

    for rule_name, data in hierarchy.items():
        if data.get('is_custom', False):
            continue

        # Cálculo del vector
        h = int(hashlib.md5(str(data['phrase']).encode()).hexdigest(), 16)
        idx = h % dim
        weight = data['length']
        vec[idx] += weight

        # Traducimos las notas MIDI a nombres de notas
        notas_legibles = [pitch.Pitch(midi).nameWithOctave for midi in data['phrase']]

        # Obtener índices de notas para esta regla
        note_indices = get_note_indices_for_rule(rule_name)
        measure_range = get_measure_range_from_notes(note_indices)

        print(f"\n {rule_name}:")
        print(f"  ├─ Content: {data['structure']}")
        print(f"  ├─ Notes: {notas_legibles}")
        print(f"  ├─ Length: {data['length']} notes")
        print(f"  ├─ Measures: {measure_range}")
        print(f"  └─ Weight (Index {idx}): +{weight}")

    # Normalización del vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    print("\n" + "="*50)
    debug(f"Sequitur: Final vector (norm={norm:.3f})")

    return vec
    """
    Genera un vector semántico y un reporte jerárquico detallado
    usando la clase SequiturGrammar, con información de compases.
    """
    # 1. Obtener símbolos y mapeo de compases
    symbols, measure_map = melody_to_sequitur_with_measures(score)

    if len(symbols) < 4:
        debug("Sequitur: Sequence too short")
        return np.zeros(dim)

    # 2. Construir Gramática con SequiturGrammar
    grammar = SequiturWithCustomRules()
    rules = grammar.build(symbols)
    hierarchy, root_seq = grammar.get_musical_hierarchy()

    # 3. Crear Vector y Reporte Educativo
    vec = np.zeros(dim)

    print("\n" + "="*50)
    print("MELODY ANALYSIS (SEQUITUR)")
    print("="*50)
    print(f"Root sequence: {root_seq}\n")
    print("Motifs and Phrases:")

    # Función auxiliar para encontrar compases de una frase
    def get_measure_range_from_notes(note_indices):
        if not note_indices or not measure_map:
            return "N/A"
        measures = [measure_map[i] for i in note_indices if i < len(measure_map)]
        if not measures:
            return "N/A"
        return f"{min(measures)}-{max(measures)}"

    # Rastrear índices para cada regla
    def get_note_indices_for_rule(rule_name, start_idx=0):
        """Devuelve los índices de notas que componen una regla."""
        if rule_name not in grammar.rules:
            return [start_idx]

        indices = []
        current = start_idx
        for symbol in grammar.rules[rule_name]:
            sub_indices = get_note_indices_for_rule(symbol, current)
            indices.extend(sub_indices)
            current += len(sub_indices)
        return indices

    # Ordenamos las reglas para ver cómo se construyen de menor a mayor
    for rule_name, data in hierarchy.items():
        # Cálculo del vector
        h = int(hashlib.md5(str(data['phrase']).encode()).hexdigest(), 16)
        idx = h % dim
        weight = data['length']
        vec[idx] += weight

        # Traducimos las notas MIDI a nombres de notas
        notas_legibles = [pitch.Pitch(midi).nameWithOctave for midi in data['phrase']]

        # Obtener índices de notas para esta regla
        note_indices = get_note_indices_for_rule(rule_name)
        measure_range = get_measure_range_from_notes(note_indices)

        print(f"\n {rule_name}:")
        print(f"  ├─ Content: {data['structure']}")
        print(f"  ├─ Notes: {notas_legibles}")
        print(f"  ├─ Length: {data['length']} notes")
        print(f"  ├─ Measures: {measure_range}")
        print(f"  └─ Weight (Index {idx}): +{weight}")

    # Normalización del vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    print("\n" + "="*50)
    debug(f"Sequitur: Final vector (norma={norm:.3f})")

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
    debug("Novelty: Dynamic segmentation...")

    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))

    if len(measures) < kernel_size:
        debug("Novelty: Score too short for current kernel.")
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

    debug(f"Novelty: {len(dynamic_segments)} dynamic sections detected.")

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
        debug("Melodic n-grams: Melody too short")
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
        f"Melodic n-grams: n={n}, unique={len(ngrams)}, norm={norm:.3f}"
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
        debug("Rhythmic n-grams: Sequence too short")
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
        f"Rhythmic n-grams: n={n}, unique={len(ngrams)}, norm={norm:.3f}"
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
                debug(f"Form Advanced: Cluster Fusion {l2} -> {l1} (Sim: {sim:.3f})")
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
    print("ADVANCED FORM ANALYSIS (SEQUITUR + CLUSTERING)")
    print("="*50)

    # 1. Obtener símbolos y mapeo de compases
    symbols, measure_map = melody_to_sequitur_with_measures(score)

    if not symbols:
        debug("Could not obtain symbols")
        return "N/A"

    # 2. Crear gramática
    grammar = SequiturWithCustomRules()
    grammar.build(symbols)
    root_seq = grammar.root_sequence

    debug(f"Root sequence: {root_seq}")

    # 3. Expandir root_seq con tracking de índices
    if len(root_seq) == 1 and str(root_seq[0]).startswith('R'):
        debug("Unitary root. Expand one level...")
        root_seq = grammar.rules[root_seq[0]]

    if not root_seq:
        debug("Empty root after expansion")
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

    debug(f"Segment: {len(segments)}")

    if not segments:
        debug("No segmentos")
        return "N/A"

    # 4. Simplificación
    segments = simplify_segments_preserving_indices(segments, max_parts)

    debug(f"Segments after simplification: {segments}")

    if segments is None:
        debug("ERROR: simplify_segments_preserving_indices is None")
        return "N/A"

    if not segments:
        debug("Empty segments list after simplification")
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
        debug(f"Initial clustering labels: {labels}")

        # 5. Combinar Clusters similares
        refined_labels = merge_similar_clusters(labels, X, similarity_threshold)
        debug(f"Refined labels: {refined_labels}")

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
    print(f"\nResulting form: {form_string}")
    print("\nSections:\n")

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

        print(f"  Part {i+1} ({char}):")
        print(f"    ├─ {measure_range}")
        print(f"    ├─ {len(seg['notes'])} notas")
        print(f"    └─ Start: [{snippet_str}...]")
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

debug("Loading score...")
score = converter.parse('./coming_fix.musicxml')
debug("Score loaded")

custom_rules = [
    CustomRule("intro", "measure", 1),  # Usar compás 1 como regla "intro"
    # CustomRule("motivo_a", "notes", [60, 62, 64, 65]),  # Do-Re-Mi-Fa
    # CustomRule("estribillo", "recursive", ["intro", "intro"])  # intro dos veces
]

# print(rhythmic_features(score))
# print(melodic_features(score))
# print(harmonic_features(score))
print(arpeggiated_chord_features(score))
# print(harmonic_transition_features(score))
# print(instrumental_features(score))
# print(motif_vector(score))
# print(compass_motif_vector(score))
# print(form_structure_vector(score))
# print(sequitur_absolute_pitch_semantic_vector(score))
# print(novelty_structure_vector(score))
# print(melodic_ngram_vector(score))
# print(rhythmic_ngram_vector(score))
# print(advanced_sequitur_form(score))
# print(form_string_to_vector(advanced_sequitur_form(score)))
