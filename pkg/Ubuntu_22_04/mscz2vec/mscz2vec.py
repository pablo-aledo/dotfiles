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

def extract_harmonic_chords(stream_input):
    debug("Harmony: Extracting chords")
    debug(f"Harmony: stream type = {type(stream_input)}")

    notes_by_offset = defaultdict(list)

    # ── Recolectar notas y acordes explícitos
    for el in stream_input.recurse():
        # Nota individual
        if isinstance(el, note.Note):
            abs_offset = float(el.getOffsetInHierarchy(stream_input))
            notes_by_offset[abs_offset].append(el)

            debug(
                f"Harmony: NOTE  {el.pitch.nameWithOctave} "
                f"@ {abs_offset:.3f}"
            )

        # Acorde explícito (MusicXML / MuseScore)
        elif isinstance(el, chord.Chord):
            abs_offset = float(el.getOffsetInHierarchy(stream_input))

            debug(
                f"Harmony: CHORD {[p.nameWithOctave for p in el.pitches]} "
                f"@ {abs_offset:.3f}"
            )

            for p in el.pitches:
                notes_by_offset[abs_offset].append(note.Note(p))

    debug(f"Harmony: unique offsets = {len(notes_by_offset)}")

    # ── Construir acordes armónicos
    chords_list = []

    # Pre-construir índice de compases para búsqueda rápida
    measures_index = []
    try:
        # Intentar obtener compases de diferentes maneras
        measures = None

        # Método 1: Desde parts si existen
        if hasattr(stream_input, 'parts') and stream_input.parts:
            first_part = stream_input.parts[0]
            measures = list(first_part.getElementsByClass('Measure'))
            debug(f"Harmony: Got {len(measures)} measures from first part")

        # Método 2: Directamente del stream
        if not measures:
            measures = list(stream_input.getElementsByClass('Measure'))
            debug(f"Harmony: Got {len(measures)} measures from stream directly")

        # Método 3: Flatten y buscar
        if not measures:
            measures = list(stream_input.flatten().getElementsByClass('Measure'))
            debug(f"Harmony: Got {len(measures)} measures from flattened stream")

        for measure in measures:
            measure_info = {
                'number': measure.measureNumber,
                'start': float(measure.offset),
                'end': float(measure.offset + measure.quarterLength),
                'length': float(measure.quarterLength)
            }
            measures_index.append(measure_info)
            debug(f"Harmony: Measure {measure_info['number']}: offset {measure_info['start']:.3f} to {measure_info['end']:.3f} (length: {measure_info['length']:.3f})")

        debug(f"Harmony: Built index with {len(measures_index)} measures")
    except Exception as e:
        debug(f"Harmony: Error building measure index: {e}")
        import traceback
        debug(traceback.format_exc())

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

            # *** MEJORADO: Múltiples métodos para obtener el compás ***
            measure_num = None

            # Método 1: Desde la primera nota del grupo
            if group:
                first_note = group[0]
                try:
                    measure = first_note.getContextByClass('Measure')
                    if measure and hasattr(measure, 'measureNumber'):
                        measure_num = measure.measureNumber
                        debug(f"Harmony: Measure from note context: {measure_num}")
                except Exception as e:
                    debug(f"Harmony: Error getting measure from note context: {e}")

            # Método 2: Buscar en el índice de compases por offset
            if measure_num is None and measures_index:
                debug(f"Harmony: Searching for offset {offset:.3f} in measure index...")
                for m_info in measures_index:
                    debug(f"  Checking measure {m_info['number']}: {m_info['start']:.3f} <= {offset:.3f} < {m_info['end']:.3f}?")
                    # Usar <= para el inicio y < para el final
                    if m_info['start'] <= offset < m_info['end']:
                        measure_num = m_info['number']
                        debug(f"Harmony: Measure from offset index: {measure_num}")
                        break

                if measure_num is None:
                    debug(f"Harmony: No match found in index for offset {offset:.3f}")

            # Método 3: Buscar directamente en el stream usando getElementAtOrBefore
            if measure_num is None:
                try:
                    # Primero intentar con parts
                    search_stream = stream_input
                    if hasattr(stream_input, 'parts') and stream_input.parts:
                        search_stream = stream_input.parts[0]

                    element = search_stream.flatten().getElementAtOrBefore(offset, classList=['Measure'])
                    if element and hasattr(element, 'measureNumber'):
                        measure_num = element.measureNumber
                        debug(f"Harmony: Measure from getElementAtOrBefore: {measure_num}")
                except Exception as e:
                    debug(f"Harmony: Error with getElementAtOrBefore: {e}")

            # Método 4: Último recurso - buscar el compás que contiene este offset
            if measure_num is None and measures_index:
                debug(f"Harmony: Last resort - finding measure containing offset {offset:.3f}")
                for m_info in measures_index:
                    # Más permisivo: incluir el extremo final
                    if m_info['start'] <= offset <= m_info['end']:
                        measure_num = m_info['number']
                        debug(f"Harmony: Measure from inclusive search: {measure_num}")
                        break

            # Método 5: Si el offset está justo en el límite, tomar el siguiente compás
            if measure_num is None and measures_index:
                tolerance = 0.01  # Tolerancia para errores de punto flotante
                for m_info in measures_index:
                    if abs(offset - m_info['start']) < tolerance:
                        measure_num = m_info['number']
                        debug(f"Harmony: Measure from boundary match (start): {measure_num}")
                        break
                    if abs(offset - m_info['end']) < tolerance:
                        # Offset al final de un compás - tomar el siguiente
                        next_measure = next((m for m in measures_index if m['number'] > m_info['number']), None)
                        if next_measure:
                            measure_num = next_measure['number']
                            debug(f"Harmony: Measure from boundary match (end, taking next): {measure_num}")
                        else:
                            measure_num = m_info['number']
                            debug(f"Harmony: Measure from boundary match (end, no next): {measure_num}")
                        break

            # Almacenar en el acorde
            ch._stored_measure_number = measure_num
            ch._stored_offset = offset

            chords_list.append(ch)

            if measure_num is None:
                debug(
                    f"Harmony: WARNING - chord detected WITHOUT measure: "
                    f"{[p.nameWithOctave for p in ch.pitches]} "
                    f"at offset {offset:.3f}"
                )
            else:
                debug(
                    f"Harmony: chord detected: "
                    f"{[p.nameWithOctave for p in ch.pitches]} "
                    f"at measure {measure_num} (offset: {offset:.3f})"
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

def extract_all_elements(part):
    """Extrae elementos de TODOS los pentagramas, no solo clave de fa"""
    elements = []
    for el in part.flatten().notesAndRests:
        if el.isRest:
            continue

        offset = el.offset
        if el.isNote:
            elements.append({
                'element': el,
                'offset': offset,
                'pitches': [el.pitch],
                'type': 'note'
            })
        elif el.isChord:
            elements.append({
                'element': el,
                'offset': offset,
                'pitches': list(el.pitches),
                'type': 'chord'
            })
    return elements


def detect_broken_chord_with_rests(elements, start_idx, time_window, min_notes=3):
    """
    Detecta acordes rotos que incluyen silencios intercalados.
    Ejemplo: Do - silencio - Mi - silencio - Sol

    Este patrón es común en:
    - Música barroca (especialmente en bajo continuo)
    - Música romántica (arpegios expresivos)
    - Música contemporánea (texturas espaciadas)
    """
    if start_idx >= len(elements):
        return None

    first_element = elements[start_idx]

    # Solo comenzar con notas individuales
    if first_element['type'] != 'note':
        return None

    first_offset = first_element['offset']

    # Recolectar notas dentro de la ventana, ignorando gaps temporales (silencios implícitos)
    collected_notes = [first_element]
    i = start_idx + 1
    rest_count = 0
    last_offset = first_offset

    while i < len(elements) and len(collected_notes) < 8:  # Máximo 8 notas
        next_element = elements[i]

        # Si estamos fuera de la ventana temporal, terminar
        if next_element['offset'] - first_offset > time_window:
            break

        # Solo considerar notas individuales
        if next_element['type'] == 'note':
            # Verificar si hay un "gap" (silencio implícito)
            gap = next_element['offset'] - last_offset

            # Si hay un gap significativo, contar como silencio
            if gap > 0.25:  # Más de una semicorchea de gap
                rest_count += 1

            collected_notes.append(next_element)
            last_offset = next_element['offset']

        i += 1

    # Necesitamos al menos min_notes y al menos un silencio para que sea "broken"
    if len(collected_notes) < min_notes or rest_count == 0:
        return None

    # Extraer pitches
    all_pitches = [n['pitches'][0] for n in collected_notes]
    unique_pitches = list(set([p.midi % 12 for p in all_pitches]))

    # Verificar que forman un acorde (al menos min_notes pitches únicos)
    if len(unique_pitches) < min_notes:
        return None

    # Verificar que hay cierta direccionalidad o patrón
    pitches_midi = [p.midi for p in all_pitches]

    # Calcular varianza de alturas (acordes rotos típicamente tienen cierto rango)
    pitch_range = max(pitches_midi) - min(pitches_midi)

    # Si el rango es muy pequeño (< 4 semitonos), probablemente no es un arpegio
    if pitch_range < 4:
        return None

    return {
        'elements': collected_notes,
        'all_pitches': all_pitches,
        'end_index': i,
        'pattern_type': 'broken',
        'rest_count': rest_count
    }


def detect_arpeggiated_chords(score, time_window=2.0, min_notes=3, analyze_all_staves=True):
    """
    Versión mejorada que puede analizar todos los pentagramas o solo clave de fa.
    Detecta múltiples patrones de arpegio incluyendo acordes rotos con silencios.

    Args:
        score: partitura music21
        time_window: ventana temporal en quarterLength para considerar un arpegio
        min_notes: número mínimo de notas diferentes para formar un acorde
        analyze_all_staves: True para analizar todos los pentagramas, False solo clave de fa

    Returns:
        lista de dict con información de los acordes detectados
    """
    debug(f"Arpeggios: Detecting arpeggiated chords (all staves: {analyze_all_staves})...")

    parts = getattr(score, 'parts', [score])
    arpeggiated_chords = []

    for part_idx, part in enumerate(parts):
        # Extraer elementos según configuración
        if analyze_all_staves:
            elements = extract_all_elements(part)
            debug(f"Arpeggios: Part {part_idx} has {len(elements)} elements (all staves)")
        else:
            elements = extract_bass_elements(part)
            debug(f"Arpeggios: Part {part_idx} has {len(elements)} elements (bass clef only)")

        if len(elements) == 0:
            debug(f"Arpeggios: Part {part_idx} has no elements, skipping...")
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

            # TERCERO: Detectar acordes rotos con silencios intercalados
            broken_chord_pattern = detect_broken_chord_with_rests(elements, i, time_window, min_notes)

            if broken_chord_pattern:
                all_pitches = broken_chord_pattern['all_pitches']
                pattern_type = 'BROKEN'

                try:
                    chord_pitches = sorted(list(set(all_pitches)), key=lambda p: p.midi)
                    ch = chord.Chord(chord_pitches)

                    # Obtener compás del primer elemento
                    measure = broken_chord_pattern['elements'][0]['element'].getContextByClass('Measure')
                    measure_num = measure.measureNumber if measure else '?'

                    # Información del patrón
                    note_names = [p.nameWithOctave for p in all_pitches]

                    arpeggiated_chords.append({
                        'chord': ch,
                        'measure': measure_num,
                        'notes': note_names,
                        'offset': current_offset,
                        'direction': pattern_type,
                        'span': broken_chord_pattern['elements'][-1]['offset'] - current_offset,
                        'part': part_idx,
                        'pattern_info': f"{pattern_type} (with {broken_chord_pattern['rest_count']} rests)"
                    })

                    debug(f"Arpeggios: Found {pattern_type} at measure {measure_num}: {note_names}")

                    # Saltar los elementos que ya procesamos
                    i = broken_chord_pattern['end_index']
                    continue

                except Exception as e:
                    debug(f"Arpeggios: Error creating chord from broken chord pattern: {e}")

            # CUARTO: Si no es bajo+acorde, Alberti ni broken chord, intentar detectar arpegio tradicional
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

    arpeggios = detect_arpeggiated_chords(score, time_window, min_notes, False)

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
# NUEVAS FUNCIONES DE ANÁLISIS AVANZADO
# =========================

def get_measure_from_chord_or_offset(chord_obj, score, offset=None):
    """
    Función universal para obtener el número de compás de un acorde.
    Intenta múltiples métodos en orden de preferencia.

    Args:
        chord_obj: objeto Chord de music21
        score: partitura completa (para búsqueda por offset)
        offset: offset opcional si se conoce

    Returns:
        int o None: número de compás
    """
    # Método 1: Atributo almacenado
    if hasattr(chord_obj, '_stored_measure_number') and chord_obj._stored_measure_number is not None:
        return chord_obj._stored_measure_number

    # Método 2: Contexto directo
    try:
        measure = chord_obj.getContextByClass('Measure')
        if measure and hasattr(measure, 'measureNumber'):
            return measure.measureNumber
    except:
        pass

    # Método 3: Usar offset almacenado o proporcionado
    if offset is None and hasattr(chord_obj, '_stored_offset'):
        offset = chord_obj._stored_offset

    if offset is not None and score is not None:
        try:
            # Buscar por offset en el score
            element = score.flatten().getElementAtOrBefore(offset, classList=['Measure'])
            if element and hasattr(element, 'measureNumber'):
                return element.measureNumber
        except:
            pass

        # Último recurso: iterar por todos los compases
        try:
            for measure in score.flatten().getElementsByClass('Measure'):
                measure_start = measure.offset
                measure_end = measure_start + measure.quarterLength
                if measure_start <= offset < measure_end:
                    return measure.measureNumber
        except:
            pass

    # Método 4: Si el acorde tiene pitches, intentar desde las notas originales
    try:
        if hasattr(chord_obj, 'pitches') and len(chord_obj.pitches) > 0:
            # Buscar en el score una nota con el mismo pitch en un offset similar
            if offset is not None and score is not None:
                for n in score.flatten().notes:
                    if n.isNote and abs(n.offset - offset) < 0.01:
                        if n.pitch.midi == chord_obj.pitches[0].midi:
                            measure = n.getContextByClass('Measure')
                            if measure and hasattr(measure, 'measureNumber'):
                                return measure.measureNumber
    except:
        pass

    return None

def detect_cadences(score):
    """
    Detecta cadencias musicales y retorna un vector de características.
    Returns: vector [authentic_perfect, authentic_imperfect, plagal, half,
                     deceptive, special, density]
    """
    try:
        key = score.analyze('key')
    except:
        debug("Cadences: Could not detect key")
        return np.zeros(7)

    chords = extract_harmonic_chords(score)

    if len(chords) < 2:
        return np.zeros(7)

    cadence_types = {
        'authentic_perfect': 0,
        'authentic_imperfect': 0,
        'plagal': 0,
        'half': 0,
        'deceptive': 0,
        'phrygian': 0,
        'landini': 0
    }

    print("\n" + "="*50)
    print("CADENCE ANALYSIS")
    print("="*50)

    for i in range(len(chords) - 1):
        try:
            rn1 = roman.romanNumeralFromChord(chords[i], key)
            rn2 = roman.romanNumeralFromChord(chords[i+1], key)

            # Usar la función auxiliar
            measure_num = get_measure_from_chord_or_offset(chords[i+1], score)
            measure_str = str(measure_num) if measure_num is not None else '?'

            # Información adicional para debugging cuando measure_num es None
            if measure_num is None:
                offset_info = f"(offset: {chords[i+1]._stored_offset:.3f})" if hasattr(chords[i+1], '_stored_offset') else ""
                debug(f"Cadences: Could not determine measure for chord {[p.nameWithOctave for p in chords[i+1].pitches]} {offset_info}")

            # Cadencia Auténtica Perfecta: V (fundamental) -> I (fundamental)
            if (rn1.figure == "V" and rn2.figure in ["I", "i"] and
                chords[i].inversion() == 0 and chords[i+1].inversion() == 0):
                cadence_types['authentic_perfect'] += 1
                print(f"  Perfect Authentic at measure {measure_str}: {rn1.figure} → {rn2.figure}")

            # Cadencia Auténtica Imperfecta: V -> I (con inversiones)
            elif (rn1.figure.startswith("V") and rn2.figure.startswith("I") and
                  (chords[i].inversion() != 0 or chords[i+1].inversion() != 0)):
                cadence_types['authentic_imperfect'] += 1
                inv1 = chords[i].inversion()
                inv2 = chords[i+1].inversion()
                print(f"  Imperfect Authentic at measure {measure_str}: {rn1.figure} (inv:{inv1}) → {rn2.figure} (inv:{inv2})")

            # Cadencia Plagal: IV -> I
            elif rn1.figure.startswith("IV") and rn2.figure.startswith("I"):
                cadence_types['plagal'] += 1
                print(f"  Plagal at measure {measure_str}: {rn1.figure} → {rn2.figure}")

            # Semicadencia: X -> V
            elif rn2.figure.startswith("V") and not rn1.figure.startswith("V"):
                cadence_types['half'] += 1
                inv = chords[i+1].inversion()
                inv_str = f" (inv:{inv})" if inv > 0 else ""
                print(f"  Half at measure {measure_str}: {rn1.figure} → {rn2.figure}{inv_str}")

            # Cadencia Rota: V -> vi
            elif rn1.figure.startswith("V") and rn2.figure.startswith("vi"):
                cadence_types['deceptive'] += 1
                print(f"  Deceptive at measure {measure_str}: {rn1.figure} → {rn2.figure}")

            # Cadencia Frigia: iv6 -> V
            elif (key.mode == 'minor' and rn1.figure == "iv" and
                  chords[i].inversion() == 1 and rn2.figure.startswith("V")):
                cadence_types['phrygian'] += 1
                print(f"  Phrygian at measure {measure_str}: {rn1.figure}6 → {rn2.figure}")

        except Exception as e:
            debug(f"Cadences: Error analyzing chord pair: {e}")
            continue

    # Detectar cadencias de tres acordes
    for i in range(len(chords) - 2):
        try:
            rn1 = roman.romanNumeralFromChord(chords[i], key)
            rn2 = roman.romanNumeralFromChord(chords[i+1], key)
            rn3 = roman.romanNumeralFromChord(chords[i+2], key)

            if (rn1.figure.startswith("vi") and rn2.figure.startswith("V") and
                rn3.figure.startswith("I")):
                cadence_types['landini'] += 1
                measure_num = get_measure_from_chord_or_offset(chords[i+2], score)
                measure_str = str(measure_num) if measure_num is not None else '?'
                print(f"  Landini (vi-V-I) at measure {measure_str}")
        except:
            continue

    total_cadences = sum(cadence_types.values())
    density = total_cadences / len(chords) if len(chords) > 0 else 0

    print(f"\nTotal cadences: {total_cadences}")
    print("="*50)

    vector = np.array([
        cadence_types['authentic_perfect'],
        cadence_types['authentic_imperfect'],
        cadence_types['plagal'],
        cadence_types['half'],
        cadence_types['deceptive'],
        cadence_types['phrygian'] + cadence_types['landini'],
        density
    ])

    if total_cadences > 0:
        vector[:6] = vector[:6] / total_cadences

    return vector

def analyze_voice_leading(score, max_parts=4):
    """
    Analiza el movimiento entre voces (paralelo, contrario, oblicuo, similar).
    Returns: vector [parallel, contrary, oblique, similar] promediado entre todos los pares de voces
    """
    parts = getattr(score, 'parts', [score])

    if len(parts) < 2:
        debug("Voice leading: Less than 2 parts")
        return np.zeros(4)

    all_movements = []

    # Analizar cada par de voces
    num_parts = min(len(parts), max_parts)

    print("\n" + "="*50)
    print("VOICE LEADING ANALYSIS")
    print("="*50)

    for i in range(num_parts - 1):
        for j in range(i + 1, num_parts):
            voice1 = parts[i].flatten().notes
            voice2 = parts[j].flatten().notes

            movements = {'parallel': 0, 'contrary': 0, 'oblique': 0, 'similar': 0}

            # Alinear por offset
            v1_dict = {n.offset: n.pitch.midi for n in voice1 if n.isNote}
            v2_dict = {n.offset: n.pitch.midi for n in voice2 if n.isNote}

            common_offsets = sorted(set(v1_dict.keys()) & set(v2_dict.keys()))

            if len(common_offsets) < 2:
                continue

            for k in range(len(common_offsets) - 1):
                off1 = common_offsets[k]
                off2 = common_offsets[k+1]

                interval_v1 = v1_dict[off2] - v1_dict[off1]
                interval_v2 = v2_dict[off2] - v2_dict[off1]

                # Clasificar movimiento
                if interval_v1 == 0 and interval_v2 == 0:
                    continue
                elif interval_v1 == 0 or interval_v2 == 0:
                    movements['oblique'] += 1
                elif (interval_v1 > 0 and interval_v2 > 0) or (interval_v1 < 0 and interval_v2 < 0):
                    if abs(interval_v1) == abs(interval_v2):
                        movements['parallel'] += 1
                    else:
                        movements['similar'] += 1
                else:
                    movements['contrary'] += 1

            total = sum(movements.values())
            if total > 0:
                normalized = [movements[k]/total for k in ['parallel', 'contrary', 'oblique', 'similar']]
                all_movements.append(normalized)

                print(f"\nVoices {i+1} ↔ {j+1}:")
                print(f"  Parallel:  {movements['parallel']:3d} ({normalized[0]:.2%})")
                print(f"  Contrary:  {movements['contrary']:3d} ({normalized[1]:.2%})")
                print(f"  Oblique:   {movements['oblique']:3d} ({normalized[2]:.2%})")
                print(f"  Similar:   {movements['similar']:3d} ({normalized[3]:.2%})")

    print("="*50)

    if not all_movements:
        return np.zeros(4)

    # Promediar movimientos de todos los pares de voces
    return np.mean(all_movements, axis=0)


def harmonic_rhythm_analysis(score):
    """
    Analiza la frecuencia de cambios armónicos (ritmo armónico).
    Returns: vector [avg_change, std_change, harmonic_density]
    """
    chords = extract_harmonic_chords(score)

    if len(chords) < 2:
        debug("Harmonic rhythm: Too few chords")
        return np.zeros(3)

    total_length = score.quarterLength

    # Estimar offsets de acordes (basado en primera nota)
    chord_offsets = []
    for ch in chords:
        # Intentar obtener offset de la primera nota del acorde
        offset = 0
        for p in ch.pitches:
            if hasattr(p, 'offset'):
                offset = p.offset
                break
        chord_offsets.append(offset)

    # Calcular distancias entre cambios de acorde
    chord_changes = []
    for i in range(1, len(chord_offsets)):
        change = chord_offsets[i] - chord_offsets[i-1]
        if change > 0:
            chord_changes.append(change)

    if not chord_changes:
        return np.zeros(3)

    print("\n" + "="*50)
    print("HARMONIC RHYTHM ANALYSIS")
    print("="*50)
    print(f"Total chords: {len(chords)}")
    print(f"Average change interval: {np.mean(chord_changes):.2f} quarters")
    print(f"Std deviation: {np.std(chord_changes):.2f}")
    print(f"Harmonic density: {len(chords) / total_length:.3f} chords/quarter")
    print("="*50)

    return np.array([
        np.mean(chord_changes),           # Cambio promedio
        np.std(chord_changes),            # Variabilidad
        len(chords) / total_length        # Densidad armónica
    ])


def texture_analysis(score):
    """
    Clasifica la textura musical y retorna un vector one-hot.
    Returns: vector [monophonic, homophonic, polyphonic]
    """
    parts = getattr(score, 'parts', [score])

    print("\n" + "="*50)
    print("TEXTURE ANALYSIS")
    print("="*50)

    if len(parts) == 1:
        # Verificar si hay acordes (homofonía) o solo melodía
        chords = score.flatten().getElementsByClass('Chord')
        notes = score.flatten().notes

        chord_ratio = len(chords) / len(notes) if len(notes) > 0 else 0

        if chord_ratio > 0.3:
            texture = "homophonic"
            vector = np.array([0, 1, 0])
        else:
            texture = "monophonic"
            vector = np.array([1, 0, 0])
    else:
        # Analizar independencia rítmica entre voces
        rhythmic_independence = 0

        for i in range(len(parts) - 1):
            p1_offsets = set(n.offset for n in parts[i].flatten().notes if n.isNote)
            p2_offsets = set(n.offset for n in parts[i+1].flatten().notes if n.isNote)

            if len(p1_offsets) == 0 or len(p2_offsets) == 0:
                continue

            overlap = len(p1_offsets & p2_offsets)
            total = len(p1_offsets | p2_offsets)

            if total > 0:
                rhythmic_independence += 1 - (overlap / total)

        avg_independence = rhythmic_independence / (len(parts) - 1) if len(parts) > 1 else 0

        if avg_independence > 0.6:
            texture = "polyphonic"
            vector = np.array([0, 0, 1])
        else:
            texture = "homophonic"
            vector = np.array([0, 1, 0])

    print(f"Detected texture: {texture.upper()}")
    print(f"Number of parts: {len(parts)}")
    print("="*50)

    return vector


def detect_modulations(score, window_size=4, min_confidence=0.7):
    """
    Versión mejorada con detección de tonalidades pivote y análisis de confianza
    """
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))

    if len(measures) < window_size * 2:
        debug("Modulations: Score too short")
        return np.zeros(5)  # Aumentado

    print("\n" + "="*50)
    print("MODULATION ANALYSIS (ENHANCED)")
    print("="*50)

    modulations = []
    current_key = None
    keys_detected = []

    # Analizar cada ventana
    key_confidence = []

    for i in range(0, len(measures), window_size // 2):  # Overlap para mejor detección
        segment = measures[i:min(i+window_size, len(measures))]
        seg_stream = stream.Stream(segment)

        try:
            # Usar múltiples métodos de análisis
            detected_key = seg_stream.analyze('key')

            # Calcular confianza basada en correlación con perfil de Krumhansl
            confidence = detected_key.correlationCoefficient if hasattr(detected_key, 'correlationCoefficient') else 0.5

            keys_detected.append((detected_key, confidence, i))

            if confidence < min_confidence:
                debug(f"Low confidence ({confidence:.2f}) at measure {i+1}, skipping")
                continue

            key_confidence.append(confidence)

            if current_key is None:
                current_key = detected_key
                print(f"Initial key: {current_key} (confidence: {confidence:.2f})")

            elif (detected_key.tonic.name != current_key.tonic.name or
                  detected_key.mode != current_key.mode):

                # Calcular distancia en círculo de quintas
                distance = interval.Interval(current_key.tonic, detected_key.tonic).semitones % 12

                # Determinar tipo de modulación
                if distance == 0:
                    mod_type = "parallel"  # C major -> C minor
                elif distance in [5, 7]:  # Quinta arriba/abajo
                    mod_type = "dominant/subdominant"
                elif distance in [2, 10]:  # Segunda
                    mod_type = "sequential"
                elif distance in [3, 9]:  # Relativa
                    mod_type = "relative"
                else:
                    mod_type = "chromatic"

                modulations.append({
                    'from': str(current_key),
                    'to': str(detected_key),
                    'measure': i + 1,
                    'distance': distance,
                    'type': mod_type,
                    'confidence': confidence
                })

                print(f"  {mod_type.upper()} modulation at measure {i+1}:")
                print(f"    {current_key} → {detected_key} (distance: {distance} semitones, conf: {confidence:.2f})")
                current_key = detected_key

        except Exception as e:
            debug(f"Modulations: Error analyzing segment: {e}")
            continue

    num_modulations = len(modulations)
    modulation_density = num_modulations / len(measures) if len(measures) > 0 else 0
    avg_distance = np.mean([m['distance'] for m in modulations]) if modulations else 0
    avg_confidence = np.mean(key_confidence) if key_confidence else 0

    # Contar modulaciones por tipo
    chromatic_count = sum(1 for m in modulations if m['type'] == 'chromatic')
    chromatic_ratio = chromatic_count / num_modulations if num_modulations > 0 else 0

    print(f"\nTotal modulations: {num_modulations}")
    print(f"Modulation density: {modulation_density:.3f} per measure")
    print(f"Average key confidence: {avg_confidence:.2f}")
    print(f"Chromatic modulations: {chromatic_ratio:.1%}")
    print("="*50)

    return np.array([
        num_modulations / 10.0,  # Normalizado
        modulation_density,
        avg_distance / 12.0,  # Normalizado
        avg_confidence,
        chromatic_ratio
    ])


def inversion_analysis(score):
    """
    Analiza el uso de inversiones en los acordes.
    Returns: vector [root_pos_ratio, first_inv_ratio, second_inv_ratio, third_inv_ratio]
    """
    try:
        key = score.analyze('key')
    except:
        debug("Inversions: Could not detect key")
        return np.zeros(4)

    chords = extract_harmonic_chords(score)

    if not chords:
        return np.zeros(4)

    inversions = {0: 0, 1: 0, 2: 0, 3: 0}

    print("\n" + "="*50)
    print("INVERSION ANALYSIS")
    print("="*50)

    for chord_obj in chords:
        try:
            inv = chord_obj.inversion()
            inversions[inv] += 1
        except:
            inversions[0] += 1  # Default a posición fundamental

    total = sum(inversions.values())

    print(f"Root position:    {inversions[0]:3d} ({inversions[0]/total:.1%})")
    print(f"First inversion:  {inversions[1]:3d} ({inversions[1]/total:.1%})")
    print(f"Second inversion: {inversions[2]:3d} ({inversions[2]/total:.1%})")
    print(f"Third inversion:  {inversions[3]:3d} ({inversions[3]/total:.1%})")
    print("="*50)

    vector = np.array([inversions[i] / total for i in range(4)])
    return vector


def advanced_harmonic_features_vector(score):
    """
    Genera un vector completo con todas las características armónicas avanzadas.

    Estructura del vector (total: 22 dimensiones):
    - Cadencias (5): [authentic, plagal, half, deceptive, density]
    - Voice Leading (4): [parallel, contrary, oblique, similar]
    - Ritmo Armónico (3): [avg_change, std_change, density]
    - Textura (3): [monophonic, homophonic, polyphonic]
    - Modulaciones (3): [num_modulations, density, avg_distance]
    - Inversiones (4): [root, first_inv, second_inv, third_inv]
    """
    print("\n" + "="*60)
    print(" ADVANCED HARMONIC ANALYSIS - COMPLETE VECTOR")
    print("="*60)

    v_cadences = detect_cadences(score)
    v_voice_leading = analyze_voice_leading(score)
    v_harmonic_rhythm = harmonic_rhythm_analysis(score)
    v_texture = texture_analysis(score)
    v_modulations = detect_modulations(score)
    v_inversions = inversion_analysis(score)

    final_vector = np.concatenate([
        v_cadences,          # 5 dims
        v_voice_leading,     # 4 dims
        v_harmonic_rhythm,   # 3 dims
        v_texture,           # 3 dims
        v_modulations,       # 3 dims
        v_inversions         # 4 dims
    ])

    print("\n" + "="*60)
    print("VECTOR SUMMARY")
    print("="*60)
    print(f"Total dimensions: {len(final_vector)}")
    print(f"Cadences:        dims 0-4   → {v_cadences}")
    print(f"Voice Leading:   dims 5-8   → {v_voice_leading}")
    print(f"Harmonic Rhythm: dims 9-11  → {v_harmonic_rhythm}")
    print(f"Texture:         dims 12-14 → {v_texture}")
    print(f"Modulations:     dims 15-17 → {v_modulations}")
    print(f"Inversions:      dims 18-21 → {v_inversions}")
    print("="*60 + "\n")

    return final_vector

def dissonance_analysis(score):
    """
    Versión mejorada con modelo de consonancia/disonancia más preciso
    """
    chords = extract_harmonic_chords(score)

    if not chords:
        return np.zeros(5)  # Aumentado

    print("\n" + "="*50)
    print("DISSONANCE ANALYSIS (ENHANCED)")
    print("="*50)

    # Tabla de consonancia de Helmholtz (0 = muy disonante, 1 = muy consonante)
    consonance_map = {
        0: 1.0,   # Unísono
        1: 0.1,   # 2m - muy disonante
        2: 0.2,   # 2M - disonante
        3: 0.6,   # 3m - consonante imperfecta
        4: 0.7,   # 3M - consonante imperfecta
        5: 0.8,   # 4J - consonante perfecta
        6: 0.0,   # 4A/5d - tritono, máxima disonancia
        7: 0.9,   # 5J - consonante perfecta
        8: 0.65,  # 6m - consonante imperfecta
        9: 0.75,  # 6M - consonante imperfecta
        10: 0.3,  # 7m - disonante
        11: 0.15  # 7M - muy disonante
    }

    dissonance_levels = []
    dissonant_count = 0
    roughness_scores = []

    for chord_obj in chords:
        try:
            pitches = sorted([p.midi for p in chord_obj.pitches])

            if len(pitches) < 2:
                dissonance_levels.append(0)
                roughness_scores.append(0)
                continue

            # Calcular todos los intervalos
            intervals = []
            consonances = []

            for i in range(len(pitches)):
                for j in range(i+1, len(pitches)):
                    interval = (pitches[j] - pitches[i]) % 12
                    intervals.append(interval)
                    consonances.append(consonance_map.get(interval, 0.5))

            # Nivel de disonancia = 1 - consonancia promedio
            avg_consonance = np.mean(consonances)
            dissonance_level = 1 - avg_consonance

            # Roughness de Sethares: más disonante cuando hay intervalos pequeños
            roughness = sum(1 / (abs(pitches[j] - pitches[i]) + 1)
                          for i in range(len(pitches))
                          for j in range(i+1, len(pitches)))
            roughness_normalized = roughness / len(intervals) if intervals else 0

            dissonance_levels.append(dissonance_level)
            roughness_scores.append(roughness_normalized)

            if dissonance_level > 0.5:
                dissonant_count += 1

        except Exception as e:
            debug(f"Dissonance: Error analyzing chord: {e}")
            dissonance_levels.append(0)
            roughness_scores.append(0)

    dissonance_ratio = dissonant_count / len(chords) if chords else 0
    avg_dissonance = np.mean(dissonance_levels) if dissonance_levels else 0
    tension_variance = np.var(dissonance_levels) if dissonance_levels else 0
    avg_roughness = np.mean(roughness_scores) if roughness_scores else 0

    # Tendencia de tensión (¿aumenta o disminuye a lo largo de la pieza?)
    if len(dissonance_levels) > 1:
        tension_trend = np.polyfit(range(len(dissonance_levels)), dissonance_levels, 1)[0]
    else:
        tension_trend = 0

    print(f"Dissonant chords: {dissonant_count}/{len(chords)} ({dissonance_ratio:.1%})")
    print(f"Average dissonance level: {avg_dissonance:.3f}")
    print(f"Tension variance: {tension_variance:.3f}")
    print(f"Average roughness: {avg_roughness:.3f}")
    print(f"Tension trend: {'increasing' if tension_trend > 0 else 'decreasing'} ({abs(tension_trend):.4f})")
    print("="*50)

    return np.array([
        dissonance_ratio,
        avg_dissonance,
        tension_variance,
        avg_roughness,
        tension_trend
    ])


def melodic_contour_analysis(score):
    """
    Analiza el contorno melódico (shape, dirección, saltos).
    Returns: vector [ascending_ratio, descending_ratio, static_ratio,
                     leap_ratio, step_ratio, avg_leap_size, contour_complexity]
    """
    seq = extract_melody(score)

    if len(seq) < 2:
        return np.zeros(7)

    print("\n" + "="*50)
    print("MELODIC CONTOUR ANALYSIS")
    print("="*50)

    pitches = [n[0] for n in seq]
    intervals = np.diff(pitches)

    # Dirección
    ascending = sum(1 for i in intervals if i > 0)
    descending = sum(1 for i in intervals if i < 0)
    static = sum(1 for i in intervals if i == 0)

    total_moves = len(intervals)

    # Saltos vs pasos
    leaps = sum(1 for i in intervals if abs(i) > 2)
    steps = sum(1 for i in intervals if 0 < abs(i) <= 2)

    # Tamaño promedio de saltos
    leap_sizes = [abs(i) for i in intervals if abs(i) > 2]
    avg_leap_size = np.mean(leap_sizes) if leap_sizes else 0

    # Complejidad del contorno (cambios de dirección)
    direction_changes = 0
    for i in range(len(intervals) - 1):
        if (intervals[i] > 0 and intervals[i+1] < 0) or \
           (intervals[i] < 0 and intervals[i+1] > 0):
            direction_changes += 1

    contour_complexity = direction_changes / (len(intervals) - 1) if len(intervals) > 1 else 0

    print(f"Direction:")
    print(f"  Ascending:  {ascending}/{total_moves} ({ascending/total_moves:.1%})")
    print(f"  Descending: {descending}/{total_moves} ({descending/total_moves:.1%})")
    print(f"  Static:     {static}/{total_moves} ({static/total_moves:.1%})")
    print(f"\nMovement:")
    print(f"  Leaps: {leaps}/{total_moves} ({leaps/total_moves:.1%})")
    print(f"  Steps: {steps}/{total_moves} ({steps/total_moves:.1%})")
    print(f"  Average leap size: {avg_leap_size:.2f} semitones")
    print(f"\nContour complexity: {contour_complexity:.3f}")
    print("="*50)

    return np.array([
        ascending / total_moves,
        descending / total_moves,
        static / total_moves,
        leaps / total_moves,
        steps / total_moves,
        avg_leap_size / 12.0,  # Normalizado
        contour_complexity
    ])


def register_analysis(score):
    """
    Analiza el uso del registro (tessitura) con detección de cruces de voces.
    Returns: vector [avg_pitch, pitch_range, low_register_ratio,
                     mid_register_ratio, high_register_ratio, register_changes,
                     voice_crossing_rate]
    """
    from music21 import pitch as pitch_module  # Importar explícitamente para evitar conflictos

    notes = score.flatten().notes

    if not notes:
        return np.zeros(7)

    print("\n" + "="*50)
    print("REGISTER ANALYSIS (ENHANCED)")
    print("="*50)

    pitches = [n.pitch.midi for n in notes if n.isNote]

    if not pitches:
        return np.zeros(7)

    avg_pitch = np.mean(pitches)
    pitch_range = max(pitches) - min(pitches)

    # Dividir en registros (bajo < 60, medio 60-72, alto > 72)
    low = sum(1 for p in pitches if p < 60)
    mid = sum(1 for p in pitches if 60 <= p <= 72)
    high = sum(1 for p in pitches if p > 72)

    total = len(pitches)

    # Cambios de registro
    register_changes = 0
    prev_register = None

    for p in pitches:
        if p < 60:
            curr_register = 'low'
        elif p <= 72:
            curr_register = 'mid'
        else:
            curr_register = 'high'

        if prev_register and prev_register != curr_register:
            register_changes += 1

        prev_register = curr_register

    register_change_rate = register_changes / total if total > 0 else 0

    # NUEVO: Detectar cruces de voces
    parts = getattr(score, 'parts', [score])
    voice_crossings = 0
    total_simultaneities = 0

    if len(parts) >= 2:
        print("\nVoice crossing analysis:")

        for i in range(len(parts) - 1):
            upper_part = parts[i]
            lower_part = parts[i + 1]

            # Extraer notas con sus offsets
            upper_notes = [(n.offset, n.pitch.midi) for n in upper_part.flatten().notes if n.isNote]
            lower_notes = [(n.offset, n.pitch.midi) for n in lower_part.flatten().notes if n.isNote]

            if not upper_notes or not lower_notes:
                continue

            # Crear diccionarios offset -> pitch para búsqueda rápida
            upper_dict = {}
            lower_dict = {}

            for offset, pitch_midi in upper_notes:
                if offset not in upper_dict:
                    upper_dict[offset] = []
                upper_dict[offset].append(pitch_midi)

            for offset, pitch_midi in lower_notes:
                if offset not in lower_dict:
                    lower_dict[offset] = []
                lower_dict[offset].append(pitch_midi)

            # Buscar cruces en momentos simultáneos
            all_offsets = set(upper_dict.keys()) | set(lower_dict.keys())

            for offset in sorted(all_offsets):
                # Buscar notas cercanas temporalmente (dentro de un margen)
                tolerance = 0.1  # quarterLength tolerance

                upper_pitches_at_offset = []
                lower_pitches_at_offset = []

                # Buscar en ventana temporal
                for off, pitch_midi in upper_notes:
                    if abs(off - offset) <= tolerance:
                        upper_pitches_at_offset.append(pitch_midi)

                for off, pitch_midi in lower_notes:
                    if abs(off - offset) <= tolerance:
                        lower_pitches_at_offset.append(pitch_midi)

                if upper_pitches_at_offset and lower_pitches_at_offset:
                    total_simultaneities += 1

                    # Verificar si hay cruce (voz inferior más alta que superior)
                    min_upper = min(upper_pitches_at_offset)
                    max_lower = max(lower_pitches_at_offset)

                    if max_lower > min_upper:
                        voice_crossings += 1

                        # Debug: mostrar algunos cruces (máximo 5)
                        if voice_crossings <= 5:
                            measure_num = '?'
                            try:
                                # Intentar obtener número de compás
                                for n in upper_part.flatten().notes:
                                    if abs(n.offset - offset) <= tolerance:
                                        m = n.getContextByClass('Measure')
                                        if m:
                                            measure_num = m.measureNumber
                                        break
                            except:
                                pass

                            upper_note_name = pitch_module.Pitch(min_upper).nameWithOctave
                            lower_note_name = pitch_module.Pitch(max_lower).nameWithOctave

                            print(f"  Crossing {voice_crossings}: Parts {i+1}/{i+2} at measure {measure_num}")
                            print(f"    Upper voice: {upper_note_name} < Lower voice: {lower_note_name}")

        if voice_crossings > 5:
            print(f"  ... and {voice_crossings - 5} more crossings")

    crossing_rate = voice_crossings / total_simultaneities if total_simultaneities > 0 else 0

    print(f"\nRegister statistics:")
    print(f"  Average pitch: {avg_pitch:.1f} MIDI ({pitch_module.Pitch(int(avg_pitch)).nameWithOctave})")
    print(f"  Pitch range: {pitch_range} semitones")
    print(f"\nRegister distribution:")
    print(f"  Low (< C4):    {low}/{total} ({low/total:.1%})")
    print(f"  Mid (C4-C5):   {mid}/{total} ({mid/total:.1%})")
    print(f"  High (> C5):   {high}/{total} ({high/total:.1%})")
    print(f"\nRegister changes: {register_changes} ({register_change_rate:.3f} per note)")

    if len(parts) >= 2:
        print(f"\nVoice crossings: {voice_crossings}/{total_simultaneities} simultaneities ({crossing_rate:.1%})")
    else:
        print(f"\nVoice crossings: N/A (single part)")

    print("="*50)

    return np.array([
        avg_pitch / 127.0,  # Normalizado
        pitch_range / 88.0,  # Normalizado (88 teclas piano)
        low / total,
        mid / total,
        high / total,
        register_change_rate,
        crossing_rate
    ])


def rhythmic_density_analysis(score):
    """
    Analiza la densidad rítmica y variedad.
    Returns: vector [note_density, rest_ratio, syncopation_level,
                     rhythmic_variety, tuplet_ratio]
    """
    notes_and_rests = score.flatten().notesAndRests

    if not notes_and_rests:
        return np.zeros(5)

    print("\n" + "="*50)
    print("RHYTHMIC DENSITY ANALYSIS")
    print("="*50)

    total_length = score.quarterLength
    notes = [n for n in notes_and_rests if n.isNote]
    rests = [r for r in notes_and_rests if r.isRest]

    # Densidad de notas
    note_density = len(notes) / total_length if total_length > 0 else 0
    rest_ratio = sum(r.quarterLength for r in rests) / total_length if total_length > 0 else 0

    # Variedad rítmica (entropía de duraciones)
    durations = [n.quarterLength for n in notes]
    duration_counts = Counter(durations)
    duration_probs = np.array(list(duration_counts.values())) / len(durations)
    rhythmic_variety = -np.sum(duration_probs * np.log2(duration_probs + 1e-9))

    # Síncopa (notas que comienzan en tiempos débiles)
    syncopations = 0
    for n in notes:
        beat_position = n.beat
        if beat_position and beat_position % 1 != 0:  # No está en tiempo fuerte
            syncopations += 1

    syncopation_level = syncopations / len(notes) if notes else 0

    # Tresillo y tuplets
    tuplets = sum(1 for n in notes if hasattr(n.duration, 'tuplets') and n.duration.tuplets)
    tuplet_ratio = tuplets / len(notes) if notes else 0

    print(f"Note density: {note_density:.2f} notes/quarter")
    print(f"Rest ratio: {rest_ratio:.1%}")
    print(f"Rhythmic variety (entropy): {rhythmic_variety:.3f}")
    print(f"Syncopation level: {syncopation_level:.1%}")
    print(f"Tuplet usage: {tuplet_ratio:.1%}")
    print("="*50)

    return np.array([
        note_density / 10.0,  # Normalizado (max ~10 notas/quarter)
        rest_ratio,
        syncopation_level,
        rhythmic_variety / 4.0,  # Normalizado (max entropy ~4)
        tuplet_ratio
    ])


def articulation_dynamics_analysis(score):
    """
    Analiza articulaciones y dinámicas.
    Returns: vector [staccato_ratio, legato_ratio, accent_ratio,
                     dynamic_range, dynamic_changes]
    """
    notes = score.flatten().notes

    if not notes:
        return np.zeros(5)

    print("\n" + "="*50)
    print("ARTICULATION & DYNAMICS ANALYSIS")
    print("="*50)

    staccato_count = 0
    legato_count = 0
    accent_count = 0

    for n in notes:
        if n.isNote:
            # Articulaciones
            if hasattr(n, 'articulations'):
                for art in n.articulations:
                    art_name = art.__class__.__name__.lower()
                    if 'staccato' in art_name:
                        staccato_count += 1
                    elif 'accent' in art_name:
                        accent_count += 1

            # Legato (buscar slurs)
            if n.getSpannerSites():
                for spanner in n.getSpannerSites():
                    if 'Slur' in spanner.__class__.__name__:
                        legato_count += 1
                        break

    total_notes = len([n for n in notes if n.isNote])

    staccato_ratio = staccato_count / total_notes if total_notes > 0 else 0
    legato_ratio = legato_count / total_notes if total_notes > 0 else 0
    accent_ratio = accent_count / total_notes if total_notes > 0 else 0

    # Dinámicas
    dynamics = score.flatten().getElementsByClass('Dynamic')
    dynamic_values = []

    dynamic_map = {
        'ppp': 1, 'pp': 2, 'p': 3, 'mp': 4, 'mf': 5,
        'f': 6, 'ff': 7, 'fff': 8
    }

    for dyn in dynamics:
        if hasattr(dyn, 'value') and dyn.value in dynamic_map:
            dynamic_values.append(dynamic_map[dyn.value])

    if dynamic_values:
        dynamic_range = (max(dynamic_values) - min(dynamic_values)) / 7.0
        dynamic_changes = len(dynamic_values) / total_length if total_length > 0 else 0
    else:
        dynamic_range = 0
        dynamic_changes = 0

    print(f"Articulations:")
    print(f"  Staccato: {staccato_ratio:.1%}")
    print(f"  Legato:   {legato_ratio:.1%}")
    print(f"  Accents:  {accent_ratio:.1%}")
    print(f"\nDynamics:")
    print(f"  Range: {dynamic_range:.2f}")
    print(f"  Changes: {dynamic_changes:.3f} per quarter")
    print("="*50)

    return np.array([
        staccato_ratio,
        legato_ratio,
        accent_ratio,
        dynamic_range,
        dynamic_changes
    ])


def metric_complexity_analysis(score):
    """
    Analiza la complejidad métrica (cambios de compás, polimetría).
    Returns: vector [time_sig_changes, compound_meter_ratio, asymmetric_meter_ratio,
                     metric_consistency]
    """
    time_sigs = score.flatten().getElementsByClass('TimeSignature')

    print("\n" + "="*50)
    print("METRIC COMPLEXITY ANALYSIS")
    print("="*50)

    if not time_sigs:
        print("No time signatures found")
        print("="*50)
        return np.zeros(4)

    time_sig_changes = len(time_sigs) - 1

    # Clasificar compases
    compound_count = 0
    asymmetric_count = 0

    for ts in time_sigs:
        numerator = ts.numerator
        denominator = ts.denominator

        # Compás compuesto (numerador divisible por 3)
        if numerator % 3 == 0 and numerator > 3:
            compound_count += 1

        # Compás asimétrico (5, 7, 11, 13, etc.)
        if numerator in [5, 7, 11, 13]:
            asymmetric_count += 1

    total_ts = len(time_sigs)
    compound_ratio = compound_count / total_ts if total_ts > 0 else 0
    asymmetric_ratio = asymmetric_count / total_ts if total_ts > 0 else 0

    # Consistencia métrica (inversa de cambios)
    measures = len(list(score.flatten().getElementsByClass('Measure')))
    metric_consistency = 1 - (time_sig_changes / measures) if measures > 0 else 1

    print(f"Time signature changes: {time_sig_changes}")
    print(f"Compound meters: {compound_ratio:.1%}")
    print(f"Asymmetric meters: {asymmetric_ratio:.1%}")
    print(f"Metric consistency: {metric_consistency:.3f}")
    print("="*50)

    return np.array([
        time_sig_changes / 10.0,  # Normalizado
        compound_ratio,
        asymmetric_ratio,
        metric_consistency
    ])


def ornament_analysis(score):
    """
    Analiza ornamentación (trinos, mordentes, apoyaturas, etc.).
    Returns: vector [trill_ratio, turn_ratio, grace_note_ratio, ornament_density]
    """
    notes = score.flatten().notes

    if not notes:
        return np.zeros(4)

    print("\n" + "="*50)
    print("ORNAMENT ANALYSIS")
    print("="*50)

    trills = 0
    turns = 0
    grace_notes = 0

    for n in notes:
        if n.isNote:
            # Expresiones (trinos, mordentes)
            if hasattr(n, 'expressions'):
                for expr in n.expressions:
                    expr_name = expr.__class__.__name__.lower()
                    if 'trill' in expr_name:
                        trills += 1
                    elif 'turn' in expr_name or 'mordent' in expr_name:
                        turns += 1

            # Notas de adorno
            if hasattr(n.duration, 'isGrace') and n.duration.isGrace:
                grace_notes += 1

    total_notes = len([n for n in notes if n.isNote])

    trill_ratio = trills / total_notes if total_notes > 0 else 0
    turn_ratio = turns / total_notes if total_notes > 0 else 0
    grace_note_ratio = grace_notes / total_notes if total_notes > 0 else 0
    ornament_density = (trills + turns + grace_notes) / total_notes if total_notes > 0 else 0

    print(f"Trills: {trill_ratio:.1%}")
    print(f"Turns/Mordents: {turn_ratio:.1%}")
    print(f"Grace notes: {grace_note_ratio:.1%}")
    print(f"Total ornament density: {ornament_density:.1%}")
    print("="*50)

    return np.array([
        trill_ratio,
        turn_ratio,
        grace_note_ratio,
        ornament_density
    ])


def tempo_character_analysis(score):
    """
    Analiza tempo y carácter expresivo.
    Returns: vector [avg_tempo, tempo_changes, tempo_variance, expressive_marks]
    """
    tempos = score.flatten().getElementsByClass('MetronomeMark')
    tempo_texts = score.flatten().getElementsByClass('TempoText')

    print("\n" + "="*50)
    print("TEMPO & CHARACTER ANALYSIS")
    print("="*50)

    if tempos:
        tempo_values = [t.number for t in tempos if hasattr(t, 'number')]

        if tempo_values:
            avg_tempo = np.mean(tempo_values)
            tempo_changes = len(tempo_values) - 1
            tempo_variance = np.var(tempo_values)
        else:
            avg_tempo = 120  # Default
            tempo_changes = 0
            tempo_variance = 0
    else:
        avg_tempo = 120
        tempo_changes = 0
        tempo_variance = 0

    expressive_marks = len(tempo_texts)

    print(f"Average tempo: {avg_tempo:.1f} BPM")
    print(f"Tempo changes: {tempo_changes}")
    print(f"Tempo variance: {tempo_variance:.2f}")
    print(f"Expressive marks: {expressive_marks}")
    print("="*50)

    return np.array([
        avg_tempo / 200.0,  # Normalizado (max ~200 BPM)
        tempo_changes / 10.0,  # Normalizado
        tempo_variance / 1000.0,  # Normalizado
        expressive_marks / 20.0  # Normalizado
    ])


def ultra_advanced_features_vector(score):
    """
    Genera un vector ULTRA completo con TODAS las características avanzadas.

    Estructura del vector (total: 63 dimensiones):

    HARMONIC FEATURES (22):
    - Cadencias (5): [authentic, plagal, half, deceptive, density]
    - Voice Leading (4): [parallel, contrary, oblique, similar]
    - Ritmo Armónico (3): [avg_change, std_change, density]
    - Textura (3): [monophonic, homophonic, polyphonic]
    - Modulaciones (3): [num_modulations, density, avg_distance]
    - Inversiones (4): [root, first_inv, second_inv, third_inv]

    MELODIC FEATURES (7):
    - Contorno Melódico (7): [asc_ratio, desc_ratio, static, leap_ratio,
                               step_ratio, avg_leap, complexity]

    REGISTER FEATURES (6):
    - Registro (6): [avg_pitch, range, low_ratio, mid_ratio, high_ratio, changes]

    RHYTHMIC FEATURES (5):
    - Densidad Rítmica (5): [note_density, rest_ratio, syncopation,
                              variety, tuplet_ratio]

    TENSION FEATURES (3):
    - Disonancia (3): [dissonance_ratio, avg_level, variance]

    PERFORMANCE FEATURES (13):
    - Articulación/Dinámicas (5): [staccato, legato, accent, dyn_range, dyn_changes]
    - Métrica (4): [time_sig_changes, compound, asymmetric, consistency]
    - Ornamentación (4): [trill, turn, grace_note, density]

    CHARACTER FEATURES (4):
    - Tempo/Carácter (4): [avg_tempo, changes, variance, expressive_marks]
    """
    print("\n" + "="*70)
    print(" ULTRA ADVANCED MUSIC ANALYSIS - COMPLETE FEATURE VECTOR")
    print("="*70)

    # HARMONIC FEATURES (22)
    v_cadences = detect_cadences(score)
    v_voice_leading = analyze_voice_leading(score)
    v_harmonic_rhythm = harmonic_rhythm_analysis(score)
    v_texture = texture_analysis(score)
    v_modulations = detect_modulations(score)
    v_inversions = inversion_analysis(score)

    # MELODIC FEATURES (7)
    v_contour = melodic_contour_analysis(score)

    # REGISTER FEATURES (6)
    v_register = register_analysis(score)

    # RHYTHMIC FEATURES (5)
    v_rhythmic_density = rhythmic_density_analysis(score)

    # TENSION FEATURES (3)
    v_dissonance = dissonance_analysis(score)

    # PERFORMANCE FEATURES (13)
    v_articulation = articulation_dynamics_analysis(score)
    v_metric = metric_complexity_analysis(score)
    v_ornament = ornament_analysis(score)

    # CHARACTER FEATURES (4)
    v_tempo = tempo_character_analysis(score)

    final_vector = np.concatenate([
        v_cadences,          # 5 dims
        v_voice_leading,     # 4 dims
        v_harmonic_rhythm,   # 3 dims
        v_texture,           # 3 dims
        v_modulations,       # 3 dims
        v_inversions,        # 4 dims
        v_contour,           # 7 dims
        v_register,          # 6 dims
        v_rhythmic_density,  # 5 dims
        v_dissonance,        # 3 dims
        v_articulation,      # 5 dims
        v_metric,            # 4 dims
        v_ornament,          # 4 dims
        v_tempo              # 4 dims
    ])

    print("\n" + "="*70)
    print("COMPLETE VECTOR SUMMARY")
    print("="*70)
    print(f"Total dimensions: {len(final_vector)}")
    print(f"\nHARMONIC FEATURES (22 dims):")
    print(f"  Cadences:        dims 0-4")
    print(f"  Voice Leading:   dims 5-8")
    print(f"  Harmonic Rhythm: dims 9-11")
    print(f"  Texture:         dims 12-14")
    print(f"  Modulations:     dims 15-17")
    print(f"  Inversions:      dims 18-21")
    print(f"\nMELODIC FEATURES (7 dims):")
    print(f"  Contour:         dims 22-28")
    print(f"\nREGISTER FEATURES (6 dims):")
    print(f"  Register:        dims 29-34")
    print(f"\nRHYTHMIC FEATURES (5 dims):")
    print(f"  Density:         dims 35-39")
    print(f"\nTENSION FEATURES (3 dims):")
    print(f"  Dissonance:      dims 40-42")
    print(f"\nPERFORMANCE FEATURES (13 dims):")
    print(f"  Articulation:    dims 43-47")
    print(f"  Metric:          dims 48-51")
    print(f"  Ornaments:       dims 52-55")
    print(f"\nCHARACTER FEATURES (4 dims):")
    print(f"  Tempo:           dims 56-59")
    print("="*70 + "\n")

    return final_vector

# =========================
# ENTROPÍA Y SORPRESA (INFORMATION DYNAMICS)
# =========================

def calculate_information_dynamics_with_measures(sequence_objs, n_gram=3, label="Feature"):
    """
    Versión mejorada que rastrea el número de compás para el debug.
    sequence_objs: lista de tuplas (valor_analizar, objeto_music21_original)
    """
    # Extraemos solo los valores para el modelo (ej. los intervalos o ritmos)
    values = [item[0] for item in sequence_objs]

    if len(values) < n_gram + 1:
        return {'mean_entropy': 0, 'max_surprise': 0, 'variance': 0, 'trend': 0, 'surprise_profile': []}

    # 1. Entrenamiento del modelo (Markov)
    model = defaultdict(Counter)
    context_counts = defaultdict(int)

    for i in range(len(values) - 1):
        start_idx = max(0, i - (n_gram - 1))
        context = tuple(values[start_idx : i + 1])
        next_val = values[i + 1]
        model[context][next_val] += 1
        context_counts[context] += 1

    # 2. Análisis con Debug de Compases
    surprise_profile = []
    print(f"\n--- {label.upper()} SURPRISE ANALYSIS (Measure-Aware) ---")

    for i in range(len(values) - 1):
        start_idx = max(0, i - (n_gram - 1))
        context = tuple(values[start_idx : i + 1])
        actual_val = values[i + 1]

        # Recuperamos el objeto music21 para saber el compás
        # i + 1 porque estamos analizando la sorpresa de la nota "siguiente"
        original_note = sequence_objs[i + 1][1]
        m_num = original_note.measureNumber
        beat = original_note.beat

        prob = model[context][actual_val] / context_counts[context] if context_counts[context] > 0 else 0.0001
        ic = -np.log2(prob)
        surprise_profile.append(ic)

        # DEBUG: Mostramos si es una sorpresa relevante (> 2.5 bits) o el inicio
        if ic > 2.5 or i < 3:
            status = " [!] HIGH" if ic > 3.5 else " [?] MID " if ic > 2.5 else " [.] LOW "
            ctx_str = f"{context}"[-15:]
            print(f"  Compás {m_num:<3} (Beat {beat:.2f}) | {status} | Ctx: {ctx_str} -> Next: {actual_val:<3} | IC: {ic:.2f} bits")

    # (Cálculos de media, varianza y tendencia iguales al anterior...)
    surprise_array = np.array(surprise_profile)
    return {
        'mean_entropy': np.mean(surprise_array),
        'max_surprise': np.max(surprise_array),
        'variance': np.var(surprise_array),
        'trend': np.polyfit(np.arange(len(surprise_array)), surprise_array, 1)[0] * 100 if len(surprise_array) > 1 else 0,
        'surprise_profile': surprise_array
    }


def get_pitch_from_element(n):
    """
    Extrae el valor MIDI de un objeto music21 (Note o Chord).
    Si es un Chord, devuelve la nota más aguda (soprano).
    """
    if n.isNote:
        return n.pitch.midi
    elif n.isChord:
        # Ordenamos de grave a agudo y tomamos el último (el más alto)
        return n.sortAscending().pitches[-1].midi
    return None

def entropy_surprise_vector(score, n_gram=3):
    """
    Genera un vector de 10 dimensiones basado en la sorpresa (Information Content)
    y la entropía melódica/rítmica, con debug detallado por compases.
    """
    # 1. Preparación de datos: Aplanamos y filtramos solo notas/acordes
    all_elements = list(score.flatten().notes)
    if len(all_elements) < n_gram + 1:
        debug("Entropy: Partitura demasiado corta para el análisis.")
        return np.zeros(10)

    # Creamos secuencias de (valor, objeto_original) para el análisis
    melodic_data = []
    rhythmic_data = []

    for i in range(len(all_elements) - 1):
        n1 = all_elements[i]
        n2 = all_elements[i+1]

        # --- Datos Melódicos (Intervalos) ---
        p1 = get_pitch_from_element(n1)
        p2 = get_pitch_from_element(n2)
        if p1 is not None and p2 is not None:
            intervalo = p2 - p1
            # Normalizamos el intervalo para que el modelo no se disperse demasiado
            # (Ej: saltos de más de una octava se agrupan en +/- 12)
            val_mel = max(-12, min(12, intervalo))
            melodic_data.append((val_mel, n2))

        # --- Datos Rítmicos (Relaciones de duración) ---
        d1 = n1.duration.quarterLength
        d2 = n2.duration.quarterLength
        if d1 > 0:
            ratio = d2 / d1
            if ratio > 1.1:   cat_rhy = "Longer"
            elif ratio < 0.9: cat_rhy = "Shorter"
            else:             cat_rhy = "Equal"
            rhythmic_data.append((cat_rhy, n2))

    # 2. Llamada al motor de cálculo (usando la función que definimos previamente)
    # Nota: Asegúrate de tener 'calculate_information_dynamics_with_measures' definida
    mel_dyn = calculate_information_dynamics_with_measures(melodic_data, n_gram, "Melodic")
    rhy_dyn = calculate_information_dynamics_with_measures(rhythmic_data, n_gram, "Rhythmic")

    # 3. Cálculo de Information Flow (bits por tiempo de negra)
    total_duration = score.quarterLength
    mel_flow = np.sum(mel_dyn['surprise_profile']) / total_duration if total_duration > 0 else 0
    rhy_flow = np.sum(rhy_dyn['surprise_profile']) / total_duration if total_duration > 0 else 0

    # 4. Construcción del vector final (10 dimensiones)
    final_vector = np.array([
        mel_dyn['mean_entropy'],     # [0] Promedio de sorpresa melódica
        mel_dyn['max_surprise'],     # [1] Pico de sorpresa (shock melódico)
        mel_dyn['variance'],         # [2] Variabilidad de la complejidad
        mel_dyn['trend'],            # [3] Tendencia (¿se vuelve más complejo?)
        mel_flow,                    # [4] Densidad de info melódica por negra
        rhy_dyn['mean_entropy'],     # [5] Promedio de sorpresa rítmica
        rhy_dyn['max_surprise'],     # [6] Pico de sorpresa rítmica
        rhy_dyn['variance'],         # [7] Consistencia rítmica
        rhy_dyn['trend'],            # [8] Tendencia rítmica
        rhy_flow                     # [9] Densidad de info rítmica
    ])

    debug(f"Entropy Vector calculado. Mel_Mean_IC: {mel_dyn['mean_entropy']:.2f}")

    return final_vector



# =========================
# LERDAHL
# =========================


def calculate_lerdahl_tension(score):
    """
    Calcula la curva de tensión basada en la jerarquía tonal de Lerdahl y Jackendoff.
    Corregido para usar .isConsonant() y .pitchFromDegree()
    """
    try:
        main_key = score.analyze('key')
    except:
        main_key = key.Key('C')

    # Obtenemos los nombres de las notas de la tríada de tónica (máxima estabilidad)
    tonica = main_key.pitchFromDegree(1).name
    mediante = main_key.pitchFromDegree(3).name
    dominante = main_key.pitchFromDegree(5).name

    # Mapa de estabilidad de Lerdahl
    stability_map = {
        tonica: 0,
        dominante: 1,
        mediante: 2,
    }

    elements = list(score.flatten().notes)
    tension_profile = []

    debug(f"Analizando Tensión Lerdahl | Tonalidad detectada: {main_key.name}")

    for n in elements:
        # --- 1. Tensión Melódica (Pitch Hierarchy) ---
        p_val = get_pitch_from_element(n)
        if p_val is None: continue

        p_obj = pitch.Pitch(p_val)
        p_name = p_obj.name

        # Estabilidad jerárquica: si no es de la tríada, base 4 de tensión
        m_tension = stability_map.get(p_name, 4)

        # Penalización por nota cromática (fuera de la escala diatónica)
        scale_names = [p.name for p in main_key.getScale().pitches]
        if p_name not in scale_names:
            m_tension += 3

        # --- 2. Tensión Armónica (Disonancia y Masa) ---
        h_tension = 0
        if n.isChord:
            # CORRECCIÓN: Usamos 'not isConsonant()' para detectar disonancia
            if not n.isConsonant():
                h_tension += 4
            # Masa sonora: mayor densidad de notas = más energía
            h_tension += (len(n.pitches) * 0.5)

        # --- 3. Tensión Rítmica (Densidad temporal) ---
        # A menor duración, más 'tensión cinética'
        r_tension = 1.0 / (n.duration.quarterLength + 0.1)

        total_tension = m_tension + h_tension + r_tension

        tension_profile.append({
            'tension': total_tension,
            'm': n.measureNumber,
            'b': n.beat,
            'obj': n
        })

    return tension_profile, main_key

def lerdahl_energy_vector(score):
    """
    Genera un vector semántico de 8 dimensiones y vuelca por consola
    la energía media calculada para cada compás.
    """
    profile, m_key = calculate_lerdahl_tension(score)
    if not profile:
        return np.zeros(8)

    # 1. Agrupar tensiones por número de compás
    measures_data = defaultdict(list)
    for entry in profile:
        measures_data[entry['m']].append(entry['tension'])

    # 2. Debug línea a línea por compás
    print(f"\n--- DEBUG DE ENERGÍA POR COMPÁS (Tonalidad: {m_key.name}) ---")

    sorted_measures = sorted(measures_data.keys())
    compas_energies = []

    for m_num in sorted_measures:
        avg_m_energy = np.mean(measures_data[m_num])
        compas_energies.append(avg_m_energy)

        # Formatear visualmente el nivel de energía
        indicator = "!" if avg_m_energy > 10 else ">" if avg_m_energy > 7 else "-"
        print(f"[DEBUG] Compás {m_num:<3} | Energía Media: {avg_m_energy:6.2f} | {indicator*int(min(avg_m_energy, 20))}")

    # 3. Preparación del vector semántico (usando los valores suavizados)
    t_values = np.array([x['tension'] for x in profile])
    smooth_t = gaussian_filter(t_values, sigma=2)

    avg_e = np.mean(smooth_t)
    max_e = np.max(smooth_t)
    climax_pos = np.argmax(smooth_t) / len(smooth_t)

    # Identificar en qué compás cae el clímax para el reporte final
    climax_measure = profile[np.argmax(smooth_t)]['m']

    # Resolución: ratio de los últimos 5 eventos respecto a la media
    resolution = np.mean(smooth_t[-5:]) / avg_e if avg_e > 0 else 1

    print(f"--- RESUMEN ESTRUCTURAL ---")
    print(f"Clímax localizado en Compás: {climax_measure} (Posición relativa: {climax_pos:.2%})")
    print(f"Resolución final: {resolution:.2f} (Valores < 1 indican relajación)")

    # Vector de salida (8 dimensiones)
    return np.array([
        avg_e,                                      # [0] Energía media total
        max_e,                                      # [1] Energía máxima (Pico)
        np.std(smooth_t),                           # [2] Contraste/Variabilidad
        climax_pos,                                 # [3] Ubicación del clímax (0-1)
        resolution,                                 # [4] Índice de resolución
        np.polyfit(np.arange(len(smooth_t)), smooth_t, 1)[0] * 100, # [5] Pendiente (Tendencia)
        len(find_peaks(smooth_t, height=avg_e)[0]) / len(smooth_t), # [6] Densidad de picos
        np.sum(t_values > 8) / len(t_values)        # [7] % de tiempo en tensión crítica
    ])


# =========================
# TONNETZ (NEO-RIEMANNIAN)
# =========================

def get_tonnetz_coordinates(relative_pitch_class):
    """
    Mapea una clase de altura (0-11) relativa a la tónica a coordenadas (x, y)
    en el Tonnetz, usando una base triangular estándar.

    Base:
    - Eje X (Quinta Justa): (1, 0)
    - Eje Diagonal (Tercera Mayor): (0.5, 0.866)

    Esta tabla busca la representación más 'compacta' (cercana al centro)
    para cada intervalo cromático.
    """
    # Constante para la altura del triángulo equilátero (sqrt(3)/2)
    h = 0.8660254

    # Mapa de coordenadas optimizado para minimizar distancia al centro (0,0)
    # Formato: interval_class: (x, y)
    coords = {
        0:  (0.0, 0.0),   # Unísono (Centro)
        1:  (-2.5, h),    # 2m (ej. C->Db: C-Ab-Db = 4 quintas atrás, 1 tercera arriba... simplificado)
        2:  (2.0, 0.0),   # 2M (2 quintas: C->G->D)
        3:  (-1.5, h),    # 3m (C -> Eb: 3 quintas atrás + 1 tercera arriba)
        4:  (0.5, h),     # 3M (1 tercera arriba: C->E)
        5:  (-1.0, 0.0),  # 4J (1 quinta atrás: C->F)
        6:  (2.5, h),     # Tritono (Complejo, lejos)
        7:  (1.0, 0.0),   # 5J (1 quinta adelante: C->G)
        8:  (-0.5, h),    # 6m (C->Ab: 4 quintas atrás + 1 tercera) -> Simplificado
        9:  (1.5, h),     # 6M (C->A: 1 quinta + 1 tercera)
        10: (-2.0, 0.0),  # 7m (2 quintas atrás: C->Bb)
        11: (3.5, h)      # 7M (1 tercera + 1 quinta... o simplemente C->B es 1 quinta+1tercera)
        # Nota: Estas son aproximaciones euclidianas para visualización de "excursión"
    }

    # Fallback para casos raros, retornamos el centro
    return coords.get(relative_pitch_class, (0.0, 0.0))

def calculate_tonnetz_trajectory(score):
    """
    Calcula la trayectoria del centroide armónico compás a compás.
    """
    try:
        main_key = score.analyze('key')
        tonic_midi = main_key.tonic.pitchClass
    except:
        tonic_midi = 0 # Default C
        debug("Tonnetz: No key found, assuming C")

    # Obtener compases
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))

    trajectory = []

    print(f"\n--- TONNETZ TRAJECTORY (Key: {main_key}) ---")

    for measure in measures:
        # Obtener todas las notas del compás
        pitches = [n.pitch.pitchClass for n in measure.flatten().notes if n.isNote]

        # Si es un acorde, añadir sus notas
        for chord in measure.flatten().getElementsByClass('Chord'):
             pitches.extend([p.pitchClass for p in chord.pitches])

        if not pitches:
            # Si el compás está vacío (silencio), mantener la posición anterior o 0
            last_dist = trajectory[-1]['dist'] if trajectory else 0
            trajectory.append({'m': measure.measureNumber, 'dist': last_dist, 'x':0, 'y':0})
            continue

        # Calcular centroide del compás
        sum_x, sum_y = 0.0, 0.0

        for p in pitches:
            # 1. Normalizar relativo a la tónica (0-11)
            rel_p = (p - tonic_midi) % 12

            # 2. Obtener coordenadas
            x, y = get_tonnetz_coordinates(rel_p)
            sum_x += x
            sum_y += y

        # Promedio (Centroide)
        avg_x = sum_x / len(pitches)
        avg_y = sum_y / len(pitches)

        # Distancia Euclidiana desde el origen (0,0)
        dist = np.sqrt(avg_x**2 + avg_y**2)

        trajectory.append({
            'm': measure.measureNumber,
            'dist': dist,
            'x': avg_x,
            'y': avg_y
        })

        # Visualización ASCII simple para debug
        bar = "=" * int(dist * 4)
        print(f"Measure {measure.measureNumber:<3} | Dist: {dist:.2f} | {bar}")

    return trajectory

def tonnetz_distance_vector(score):
    """
    Genera vector de estadísticas de excursión armónica (6 dimensiones).
    """
    traj = calculate_tonnetz_trajectory(score)

    if not traj:
        return np.zeros(6)

    distances = np.array([t['dist'] for t in traj])

    # Suavizar curva para tendencias
    if len(distances) > 4:
        smooth_dist = gaussian_filter(distances, sigma=1)
    else:
        smooth_dist = distances

    avg_dist = np.mean(distances)
    max_dist = np.max(distances)

    # Distancia final (Resolución): Promedio de los últimos 3 compases
    final_dist = np.mean(distances[-3:]) if len(distances) >= 3 else distances[-1]

    # "Viaje Total": Suma de las diferencias (cuánto se movió el centroide paso a paso)
    coords = np.array([(t['x'], t['y']) for t in traj])
    steps = np.diff(coords, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    total_travel = np.sum(step_lengths)

    # Normalización aproximada para el viaje total basada en la longitud
    travel_density = total_travel / len(traj)

    print(f"\n--- TONNETZ SUMMARY ---")
    print(f"Max Excursion: {max_dist:.2f} units")
    print(f"Final Distance: {final_dist:.2f} (Close to 0 = Perfect Return)")
    print(f"Harmonic Travel Density: {travel_density:.2f}")

    return np.array([
        avg_dist,        # [0] Distancia media a la tónica
        max_dist,        # [1] Excursión máxima
        np.std(distances), # [2] Variabilidad del viaje
        final_dist,      # [3] Resolución (¿Acaba en casa?)
        travel_density,  # [4] Movimiento armónico acumulado
        np.argmax(smooth_dist) / len(smooth_dist) # [5] Posición relativa del punto más lejano
    ])


# =========================
# SENSORY ROUGHNESS (PSYCHOACOUSTIC)
# =========================

def calculate_roughness_curve(score):
    """
    Calcula la curva de disonancia sensorial (rugosidad) basada en la
    interferencia de intervalos.
    
    A diferencia de la disonancia teórica, esto mide la 'aspereza' física
    del sonido sumando la disonancia de todos los pares de notas simultáneos.
    """
    # Mapa de pesos de disonancia simplificado (basado en Plomp-Levelt/Sethares)
    # 0 = Unísono/Octava, 1.0 = Máxima disonancia (segunda menor)
    dissonance_weights = {
        0: 0.0,   # P1 (Unísono)
        1: 1.0,   # m2 (Disonancia máxima)
        2: 0.8,   # M2 (Disonante)
        3: 0.2,   # m3 (Consonante suave)
        4: 0.1,   # M3 (Consonante)
        5: 0.05,  # P4 (Casi perfecta)
        6: 0.9,   # Tritono (Muy tenso, 'diabolus in musica')
        7: 0.02,  # P5 (Perfecta)
        8: 0.25,  # m6
        9: 0.15,  # M6
        10: 0.7,  # m7 (Disonante bluesy)
        11: 0.95, # M7 (Muy disonante punzante)
    }

    # Obtener acordes o 'slices' verticales por compás
    # Para simplificar y mantener alineación con otros análisis,
    # colapsamos todo el contenido del compás en un 'super-acorde' representativo
    # o analizamos acorde por acorde y promediamos por compás.
    
    parts = getattr(score, 'parts', [score])
    measures = list(parts[0].getElementsByClass('Measure'))
    
    roughness_profile = []
    
    print(f"\n--- SENSORY ROUGHNESS PROFILE ---")

    for measure in measures:
        # Extraer todas las notas que suenan en el compás
        # (Nota: Una implementación más precisa haría esto por 'beat', 
        # aquí lo hacemos por compás para visión macro)
        simultaneous_pitches = set()
        
        # Aplanar y obtener notas
        notes = measure.flatten().notes
        for n in notes:
            if n.isNote:
                simultaneous_pitches.add(n.pitch.midi)
            elif n.isChord:
                for p in n.pitches:
                    simultaneous_pitches.add(p.midi)
        
        pitches = sorted(list(simultaneous_pitches))
        
        # Si hay menos de 2 notas, no hay rugosidad interválica
        if len(pitches) < 2:
            roughness_profile.append(0.0)
            continue
            
        # Calcular interferencia de todos los pares únicos
        total_roughness = 0.0
        pairs_count = 0
        
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                p1 = pitches[i]
                p2 = pitches[j]
                
                # Intervalo módulo 12 (clase de intervalo)
                interval = abs(p2 - p1) % 12
                
                # Penalización adicional si están muy cerca en registro grave
                # (El oído distingue peor las frecuencias graves, creando más 'barro')
                low_register_penalty = 1.0
                if p1 < 48 or p2 < 48: # Debajo de C3
                    low_register_penalty = 1.2
                
                weight = dissonance_weights.get(interval, 0.5)
                total_roughness += weight * low_register_penalty
                pairs_count += 1
        
        # Normalizar por densidad: 
        # Queremos saber la "calidad" de la rugosidad, no solo que suba porque hay 100 notas.
        # Dividimos por la raíz cuadrada de pares para equilibrar texturas densas vs ligeras.
        if pairs_count > 0:
            avg_roughness = total_roughness / np.sqrt(pairs_count)
        else:
            avg_roughness = 0.0
            
        roughness_profile.append(avg_roughness)
        
        # Visualización ASCII
        intensity = int(avg_roughness * 5)
        bar = "#" * intensity
        print(f"Measure {measure.measureNumber:<3} | Roughness: {avg_roughness:5.2f} | {bar}")

    return roughness_profile

def psychoacoustic_vector(score):
    """
    Genera vector estadístico de la experiencia sensorial (Rugosidad).
    """
    roughness = calculate_roughness_curve(score)
    
    if not roughness:
        return np.zeros(6)
        
    r_arr = np.array(roughness)
    
    # Suavizado para tendencias
    if len(r_arr) > 4:
        smooth_r = gaussian_filter(r_arr, sigma=1)
    else:
        smooth_r = r_arr
        
    mean_r = np.mean(r_arr)
    max_r = np.max(r_arr)
    
    # Contraste: Diferencia entre momentos más suaves y más ásperos
    contrast = max_r - np.min(r_arr)
    
    # "Grit": Porcentaje de tiempo que la obra pasa en alta rugosidad (> 0.6)
    grit = np.sum(r_arr > 0.6) / len(r_arr)
    
    # Tendencia de relajación: ¿Se vuelve más suave al final?
    # Comparamos el primer tercio con el último tercio
    third = len(r_arr) // 3
    if third > 0:
        start_avg = np.mean(r_arr[:third])
        end_avg = np.mean(r_arr[-third:])
        relaxation = start_avg - end_avg # Positivo = Relajación, Negativo = Tensión creciente
    else:
        relaxation = 0

    print(f"\n--- PSYCHOACOUSTIC SUMMARY ---")
    print(f"Average Roughness: {mean_r:.2f}")
    print(f"Peak Roughness (Grit): {max_r:.2f}")
    print(f"Relaxation Trend: {relaxation:+.2f}")

    return np.array([
        mean_r,      # [0] Nivel medio de conflicto sonoro
        max_r,       # [1] Momento más áspero
        np.std(r_arr), # [2] Variedad tímbrica
        contrast,    # [3] Rango dinámico de la disonancia
        grit,        # [4] Densidad de momentos tensos
        relaxation   # [5] Dirección de la obra (hacia el caos o la calma)
    ])

# =========================
# AROUSAL VALENCE
# =========================


def get_emotional_label(arousal, valence):
    """
    Convierte las coordenadas del modelo circunflejo a una etiqueta semántica.
    """
    if valence >= 0:
        return "Excitación/Felicidad" if arousal >= 0 else "Calma/Serenidad"
    else:
        return "Tensión/Ira" if arousal >= 0 else "Tristeza/Depresión"

def extract_arousal_valence(stream_fragment, global_mode_val=0):
    """
    Calcula las coordenadas de Arousal y Valence para un fragmento musical (o compás).
    """
    notes = stream_fragment.flatten().notes
    if not notes:
        return -1.0, -1.0  # Silencio o ausencia de datos

    # --- CÁLCULO DE AROUSAL (Energía) ---
    # Densidad rítmica: Notas por pulso (quarterLength)
    duration = stream_fragment.duration.quarterLength
    density = len(notes) / (duration if duration > 0 else 1)

    # Registro: Altura media (MIDI pitch)
    pitches = [n.pitch.ps for n in notes if n.isNote]
    avg_pitch = np.mean(pitches) if pitches else 60

    # Normalización (Heurística: densidad máx 6, registro C2-C6)
    norm_density = np.clip(density / 6, 0, 1)
    norm_pitch = np.clip((avg_pitch - 36) / 48, 0, 1)
    arousal = (norm_density * 0.7 + norm_pitch * 0.3) * 2 - 1

    # --- CÁLCULO DE VALENCE (Positividad) ---
    # 1. Modo (Mayor/Menor)
    try:
        local_key = stream_fragment.analyze('key')
        mode_score = 0.5 if local_key.mode == 'major' else -0.5
    except:
        mode_score = global_mode_val # Usar el global si el fragmento es ambiguo

    # 2. Consonancia (Acorde resultante del fragmento)
    try:
        m_chord = stream_fragment.chordify().flatten().notes.first()
        consonance_val = 0.3 if (m_chord and m_chord.isConsonant()) else -0.3
    except:
        consonance_val = 0

    valence = np.clip(mode_score + consonance_val, -1, 1)

    return round(arousal, 3), round(valence, 3)

def emotional_evolution_analysis(score):
    """
    Analiza la partitura compás por compás, genera logs y devuelve la serie temporal.
    """
    debug("Iniciando análisis perceptual de evolución emocional...")

    # Análisis global previo para consistencia tonal
    try:
        g_key = score.analyze('key')
        g_mode = 0.5 if g_key.mode == 'major' else -0.5
        debug(f"Tonalidad global detectada: {g_key}")
    except:
        g_mode = 0
        debug("No se pudo detectar tonalidad global. Usando valor neutro.")

    measure_stats = []
    # Usamos la primera parte como referencia para los compases
    main_part = score.parts[0]

    for m in main_part.getElementsByClass(stream.Measure):
        a, v = extract_arousal_valence(m, global_mode_val=g_mode)
        label = get_emotional_label(a, v)

        # Log por compás
        debug(f"Compás {m.measureNumber:3} | Arousal: {a:6.2f} | Valence: {v:6.2f} | Estado: {label}")

        measure_stats.append({'a': a, 'v': v})

    return measure_stats

def semantic_emotion_vector(score):
    """
    Genera un vector semántico de 6 dimensiones que resume la narrativa emocional.
    """
    stats = emotional_evolution_analysis(score)
    if not stats:
        return np.zeros(6)

    a_series = [s['a'] for s in stats]
    v_series = [s['v'] for s in stats]

    # Helper para promedios por tramos (Inicio, Medio, Fin)
    def get_segment_avg(series, segment):
        l = len(series)
        if l < 3: return series[0]
        step = l // 3
        if segment == 0: return np.mean(series[:step])
        if segment == 1: return np.mean(series[step:2*step])
        return np.mean(series[2*step:])

    # Construcción del vector
    vector = np.array([
        np.mean(a_series),                    # [0] Energía media
        np.mean(v_series),                    # [1] Positividad media
        np.std(a_series),                     # [2] Volatilidad (Contraste)
        get_segment_avg(v_series, 2) - get_segment_avg(v_series, 0), # [3] Evolución humor (Final - Inicio)
        get_segment_avg(a_series, 2) - get_segment_avg(a_series, 0), # [4] Evolución tensión (Final - Inicio)
        np.corrcoef(a_series, v_series)[0,1] if len(a_series) > 1 else 0 # [5] Coherencia A/V
    ])

    debug(f"Vector Semántico Emocional: {vector}")
    return vector


# =========================
# EJECUCIÓN
# =========================

debug("Loading score...")
score = converter.parse('./dreams2.musicxml')
debug("Score loaded")

custom_rules = [
    CustomRule("intro", "measure", 1),  # Usar compás 1 como regla "intro"
    # CustomRule("motivo_a", "notes", [60, 62, 64, 65]),  # Do-Re-Mi-Fa
    # CustomRule("estribillo", "recursive", ["intro", "intro"])  # intro dos veces
]

# print(rhythmic_features(score))
# print(melodic_features(score))
# print(harmonic_features(score))
# print(arpeggiated_chord_features(score))
# print(harmonic_transition_features(score))
# print(advanced_harmonic_features_vector(score))
# print(ultra_advanced_features_vector(score))
# print(instrumental_features(score))
# print(motif_vector(score))
# print(compass_motif_vector(score))
# print(form_structure_vector(score))
# print(sequitur_absolute_pitch_semantic_vector(score))
# sequitur_absolute_pitch_semantic_vector_with_custom_rules(score, custom_rules)
# print(novelty_structure_vector(score))
# print(melodic_ngram_vector(score))
# print(rhythmic_ngram_vector(score))
# print(advanced_sequitur_form(score))
# print(form_string_to_vector(advanced_sequitur_form(score)))
# entropy_surprise_vector(score)
# lerdahl_energy_vector(score)
# tonnetz_distance_vector(score)
# psychoacoustic_vector(score)
print(semantic_emotion_vector(score))
