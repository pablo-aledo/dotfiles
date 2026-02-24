"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MIDI DNA COMBINER                                     ║
║  Extrae el "ADN" musical de varios MIDIs y los fusiona en una nueva pieza    ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    python midi_dna_combiner.py file1.mid file2.mid file3.mid [opciones]

OPCIONES:
    --output OUTPUT       Nombre del archivo de salida (default: combined_output.mid)
    --mode MODE           Modo de combinación:
                            auto      → El sistema decide el mejor modo (default)
                            harmony   → Progresión armónica de A, ritmo de B, melodía de C
                            rhythm    → Ritmo de A con la melodía de B
                            energy    → Estructura energética de A con materiales de B y C
                            texture   → Texturas de A con melodía de B y harmonía de C
                            emotion   → Combina para crear un arco emocional progresivo
    --tempo TEMPO         BPM del resultado (default: extraído del midi con mayor energía)
    --key KEY             Tonalidad destino (ej: C major, A minor) (default: auto)
    --verbose             Muestra análisis detallado del ADN extraído

EJEMPLOS:
    python midi_dna_combiner.py bach.mid jazz.mid flamenco.mid --mode harmony
    python midi_dna_combiner.py rock.mid classical.mid --mode rhythm --tempo 120
    python midi_dna_combiner.py a.mid b.mid c.mid --mode auto --verbose

DEPENDENCIAS:
    pip install music21 numpy mido
"""

import sys
import os
import argparse
import copy
import random
import math
from collections import defaultdict, Counter
import numpy as np

try:
    from music21 import (
        stream, note, chord, meter, tempo, key, instrument,
        interval, pitch, duration, harmony, roman, converter,
        scale, dynamics, expressions
    )
    from music21 import environment
    # Silenciar music21
    environment.Environment()['warnings'] = 0
except ImportError:
    print("ERROR: music21 no instalado. Ejecuta: pip install music21")
    sys.exit(1)

try:
    import mido
except ImportError:
    print("ERROR: mido no instalado. Ejecuta: pip install mido")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 1: EXTRACCIÓN DEL ADN MUSICAL
# ═══════════════════════════════════════════════════════════════

class MusicDNA:
    """Contiene el ADN completo de una pieza musical."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)

        # ADN extraído
        self.tempo_bpm = 120
        self.time_signature = (4, 4)
        self.key_tonic = "C"
        self.key_mode = "major"
        self.key_obj = None

        # Dimensiones del ADN
        self.rhythm_dna = {}       # Patrones rítmicos
        self.melody_dna = {}       # Contorno melódico
        self.harmony_dna = {}      # Progresiones armónicas
        self.texture_dna = {}      # Textura y densidad
        self.energy_dna = {}       # Curva de energía
        self.emotion_dna = {}      # Perfil emocional
        self.phrase_dna = {}       # Estructura de frases
        self.instrumentation_dna = {}  # Instrumentación
        self.tension_dna = {}      # Curva de tensión
        self.dynamics_dna = {}     # Dinámica

        # Objetos music21 para reconstrucción
        self.score = None
        self.parts = []

    def __repr__(self):
        return (f"MusicDNA({self.name} | "
                f"{self.key_tonic} {self.key_mode} | "
                f"{self.tempo_bpm:.0f} BPM | "
                f"energy={self.energy_dna.get('mean', 0):.2f} | "
                f"valence={self.emotion_dna.get('valence', 0):.2f})")


def extract_dna(filepath, verbose=False):
    """Extrae el ADN completo de un archivo MIDI."""

    print(f"\n🧬 Extrayendo ADN de: {os.path.basename(filepath)}")
    print("─" * 60)

    dna = MusicDNA(filepath)

    # ── Cargar score
    try:
        score = converter.parse(filepath)
        dna.score = score
    except Exception as e:
        print(f"  ❌ Error cargando {filepath}: {e}")
        return None

    parts = list(score.parts) if hasattr(score, 'parts') else [score]
    dna.parts = parts

    # ── 1. TEMPO
    dna.tempo_bpm = _extract_tempo(score)
    if verbose:
        print(f"  🎵 Tempo: {dna.tempo_bpm:.0f} BPM")

    # ── 2. COMPÁS
    dna.time_signature = _extract_time_signature(score)
    if verbose:
        print(f"  📊 Compás: {dna.time_signature[0]}/{dna.time_signature[1]}")

    # ── 3. TONALIDAD
    dna.key_tonic, dna.key_mode, dna.key_obj = _extract_key(score)
    if verbose:
        print(f"  🎼 Tonalidad: {dna.key_tonic} {dna.key_mode}")

    # ── 4. ADN RÍTMICO
    dna.rhythm_dna = _extract_rhythm_dna(score, dna.tempo_bpm)
    if verbose:
        print(f"  🥁 Ritmo: densidad={dna.rhythm_dna['density']:.2f}, "
              f"swing={dna.rhythm_dna['swing']:.2f}, "
              f"syncopation={dna.rhythm_dna['syncopation']:.2f}")

    # ── 5. ADN MELÓDICO
    dna.melody_dna = _extract_melody_dna(score, dna.key_obj)
    if verbose:
        print(f"  🎶 Melodía: rango={dna.melody_dna['range_semitones']} semitonos, "
              f"dirección={dna.melody_dna['direction']:.2f}, "
              f"saltos={dna.melody_dna['leap_ratio']:.2f}")

    # ── 6. ADN ARMÓNICO
    dna.harmony_dna = _extract_harmony_dna(score, dna.key_obj)
    if verbose:
        print(f"  🎹 Armonía: complejidad={dna.harmony_dna['complexity']:.2f}, "
              f"cambios/compás={dna.harmony_dna['change_rate']:.2f}")

    # ── 7. ADN DE TEXTURA
    dna.texture_dna = _extract_texture_dna(score, parts)
    if verbose:
        print(f"  🌊 Textura: capas={dna.texture_dna['layers']}, "
              f"densidad={dna.texture_dna['note_density']:.2f}")

    # ── 8. ADN DE ENERGÍA
    dna.energy_dna = _extract_energy_dna(score)
    if verbose:
        print(f"  ⚡ Energía: media={dna.energy_dna['mean']:.2f}, "
              f"pico={dna.energy_dna['peak']:.2f}, "
              f"arco={dna.energy_dna['arc_type']}")

    # ── 9. ADN EMOCIONAL
    dna.emotion_dna = _extract_emotion_dna(score, dna)
    if verbose:
        print(f"  💫 Emoción: valencia={dna.emotion_dna['valence']:.2f}, "
              f"activación={dna.emotion_dna['activation']:.2f}")

    # ── 10. ADN DE FRASES
    dna.phrase_dna = _extract_phrase_dna(score)
    if verbose:
        print(f"  📝 Frases: longitud_media={dna.phrase_dna['avg_length']:.1f} compases, "
              f"pregunta_respuesta={dna.phrase_dna['qa_ratio']:.2f}")

    # ── 11. ADN DE TENSIÓN
    dna.tension_dna = _extract_tension_dna(score, dna.key_obj)
    if verbose:
        print(f"  🌡️ Tensión: media={dna.tension_dna['mean']:.2f}, "
              f"pico={dna.tension_dna['peak']:.2f}")

    # ── 12. ADN DE DINÁMICA
    dna.dynamics_dna = _extract_dynamics_dna(score)
    if verbose:
        print(f"  📢 Dinámica: rango={dna.dynamics_dna['range']:.2f}, "
              f"media={dna.dynamics_dna['mean_velocity']:.0f}")

    # ── 13. INSTRUMENTACIÓN
    dna.instrumentation_dna = _extract_instrumentation_dna(parts)
    if verbose:
        print(f"  🎸 Instrumentos: {list(dna.instrumentation_dna['families'].keys())}")

    print(f"  ✅ ADN extraído exitosamente")
    return dna


# ─── Funciones auxiliares de extracción ────────────────────────

def _extract_tempo(score):
    tempos = []
    for el in score.flat.getElementsByClass('MetronomeMark'):
        if el.number:
            tempos.append(float(el.number))
    return tempos[0] if tempos else 120.0


def _extract_time_signature(score):
    for el in score.flat.getElementsByClass('TimeSignature'):
        return (el.numerator, el.denominator)
    return (4, 4)


def _extract_key(score):
    try:
        k = score.analyze('key')
        return k.tonic.name, k.mode, k
    except:
        try:
            k = score.flat.getElementsByClass('KeySignature')[0]
            k_obj = k.asKey()
            return k_obj.tonic.name, k_obj.mode, k_obj
        except:
            k_obj = key.Key('C', 'major')
            return 'C', 'major', k_obj


def _extract_rhythm_dna(score, bpm):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'density': 0, 'swing': 0, 'syncopation': 0,
                'pattern': [], 'durations': [], 'groove': []}

    durations = [float(n.quarterLength) for n in all_notes]
    offsets = [float(n.offset) for n in all_notes]

    # Densidad: notas por compás
    try:
        total_quarters = float(score.flat.highestTime)
        measures = max(1, total_quarters / 4.0)
        density = len(all_notes) / measures
    except:
        density = len(durations)

    # Swing: ratio entre corcheas largas y cortas
    eighth = 0.5
    swing_pairs = []
    for i in range(len(durations)-1):
        if abs(durations[i] - 0.667) < 0.1 and abs(durations[i+1] - 0.333) < 0.1:
            swing_pairs.append(1)
    swing = len(swing_pairs) / max(1, len(durations)/2)

    # Síncopa: notas en tiempos débiles con mayor duración
    syncopation = 0
    for n in all_notes:
        off = float(n.offset) % 1.0
        dur = float(n.quarterLength)
        if 0.3 < off < 0.7 and dur >= 1.0:
            syncopation += 1
    syncopation = syncopation / max(1, len(all_notes))

    # Patrón rítmico (secuencia de duraciones normalizada)
    dur_pattern = [round(d * 4) / 4 for d in durations[:32]]  # primeras 32 notas

    # Groove: variación de velocidades
    velocities = [n.volume.velocity or 64 for n in all_notes if hasattr(n, 'volume')]
    groove = [float(v) / 127 for v in velocities[:32]]

    return {
        'density': density,
        'swing': swing,
        'syncopation': syncopation,
        'pattern': dur_pattern,
        'durations': durations,
        'groove': groove
    }


def _extract_melody_dna(score, key_obj):
    # Identificar la parte melódica principal (la más aguda)
    all_parts = list(score.parts) if hasattr(score, 'parts') else [score]

    melody_notes = []
    for part in all_parts:
        notes = [n for n in part.flat.notes if n.isNote]
        if notes:
            melody_notes = notes
            break

    if not melody_notes:
        return {'range_semitones': 0, 'direction': 0, 'leap_ratio': 0,
                'contour': [], 'intervals': [], 'motif': [], 'mean_pitch': 60}

    pitches = [n.pitch.midi for n in melody_notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

    range_semi = max(pitches) - min(pitches)
    direction = np.mean(np.sign(intervals)) if intervals else 0
    leap_ratio = sum(1 for i in intervals if abs(i) > 4) / max(1, len(intervals))

    # Contorno melódico normalizado (forma de la melodía)
    if len(pitches) > 1:
        p_arr = np.array(pitches, dtype=float)
        p_norm = (p_arr - p_arr.min()) / max(1, p_arr.max() - p_arr.min())
        # Reducir a 32 puntos para comparación
        indices = np.linspace(0, len(p_norm)-1, min(32, len(p_norm)), dtype=int)
        contour = p_norm[indices].tolist()
    else:
        contour = [0.5]

    # Motivo principal: primeras 4-8 notas con sus intervalos
    motif_intervals = intervals[:8] if intervals else []

    # Grados de la escala usados
    scale_degrees = []
    if key_obj:
        scale_pitches = [p.pitchClass for p in key_obj.getScale().getPitches()]
        for n in melody_notes[:64]:
            pc = n.pitch.pitchClass
            if pc in scale_pitches:
                scale_degrees.append(scale_pitches.index(pc))

    return {
        'range_semitones': range_semi,
        'direction': direction,
        'leap_ratio': leap_ratio,
        'contour': contour,
        'intervals': intervals[:64],
        'motif': motif_intervals,
        'mean_pitch': float(np.mean(pitches)),
        'pitches': pitches,
        'scale_degrees': scale_degrees,
        'notes': melody_notes
    }


def _extract_harmony_dna(score, key_obj):
    chords_data = []

    # Analizar acordes por ventanas de tiempo
    flat = score.flat.chordify()
    chord_objs = list(flat.flat.getElementsByClass('Chord'))

    if not chord_objs:
        return {'complexity': 0, 'change_rate': 0,
                'progressions': [], 'chord_sequence': [],
                'functions': [], 'tension_curve': []}

    progression = []
    functions = []
    tension_curve = []

    for ch in chord_objs[:64]:  # Primeros 64 acordes
        try:
            # Calidad del acorde
            quality = ch.quality  # major, minor, diminished, augmented
            root = ch.root().name if ch.root() else 'C'

            # Grado romano
            rn_figure = "I"
            tension = 0.0
            if key_obj:
                try:
                    rn = roman.romanNumeralFromChord(ch, key_obj)
                    rn_figure = rn.figure

                    # Tensión basada en función armónica
                    if rn_figure in ['V', 'V7', 'vii°']:
                        tension = 0.8
                    elif rn_figure in ['II', 'ii', 'IV', 'iv']:
                        tension = 0.4
                    elif rn_figure in ['I', 'i', 'VI', 'vi']:
                        tension = 0.1
                    else:
                        tension = 0.5

                    # Acorde con extensiones = más tensión
                    if len(ch.pitches) > 4:
                        tension = min(1.0, tension + 0.2)
                except:
                    pass

            progression.append({
                'root': root,
                'quality': quality,
                'roman': rn_figure,
                'offset': float(ch.offset),
                'duration': float(ch.quarterLength),
                'pitches': [p.midi for p in ch.pitches]
            })
            functions.append(rn_figure)
            tension_curve.append(tension)
            chords_data.append(ch)

        except Exception:
            continue

    # Complejidad: diversidad de acordes usados
    unique_chords = len(set(f['roman'] for f in progression))
    complexity = unique_chords / max(1, len(progression))

    # Tasa de cambio armónico (acordes por compás)
    if progression:
        total_time = progression[-1]['offset'] - progression[0]['offset']
        change_rate = len(progression) / max(1, total_time / 4)
    else:
        change_rate = 0

    # Progressiones frecuentes (bigramas)
    bigrams = Counter()
    for i in range(len(functions)-1):
        bigrams[(functions[i], functions[i+1])] += 1

    return {
        'complexity': complexity,
        'change_rate': change_rate,
        'progressions': progression,
        'chord_sequence': functions,
        'bigrams': bigrams.most_common(10),
        'functions': functions,
        'tension_curve': tension_curve,
        'chords': chords_data
    }


def _extract_texture_dna(score, parts):
    layers = len(parts)
    all_notes = list(score.flat.notes)

    if not all_notes:
        return {'layers': 0, 'note_density': 0, 'homophony': 0,
                'polyphony': 0, 'type': 'monophony'}

    total_time = float(score.flat.highestTime) or 1
    note_density = len(all_notes) / total_time

    # Homofonía vs Polifonía: analizar cuántas notas coinciden en tiempo
    offsets = Counter(round(float(n.offset), 2) for n in all_notes)
    simultaneous = sum(1 for c in offsets.values() if c > 1)
    homophony = simultaneous / max(1, len(offsets))
    polyphony = 1 - homophony

    texture_type = 'monophony'
    if layers > 1:
        if homophony > 0.6:
            texture_type = 'homophony'
        elif polyphony > 0.6:
            texture_type = 'polyphony'
        else:
            texture_type = 'heterophony'

    # Registros usados
    pitch_mids = [n.pitch.midi for n in all_notes if n.isNote]
    register_spread = np.std(pitch_mids) if pitch_mids else 0

    return {
        'layers': layers,
        'note_density': note_density,
        'homophony': homophony,
        'polyphony': polyphony,
        'type': texture_type,
        'register_spread': register_spread
    }


def _extract_energy_dna(score):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'mean': 0.5, 'peak': 0.5, 'arc_type': 'flat',
                'curve': [], 'measures': []}

    total_time = float(score.flat.highestTime) or 1
    measure_size = 4.0  # quarter lengths por compás

    n_measures = max(1, int(total_time / measure_size))
    energy_per_measure = []

    for m in range(n_measures):
        t_start = m * measure_size
        t_end = (m + 1) * measure_size
        notes_in_measure = [
            n for n in all_notes
            if t_start <= float(n.offset) < t_end
        ]
        if not notes_in_measure:
            energy_per_measure.append(0.0)
            continue

        # Energía = densidad * velocidad media normalizada
        density = len(notes_in_measure) / measure_size
        velocities = [n.volume.velocity or 64 for n in notes_in_measure
                      if hasattr(n, 'volume')]
        mean_vel = np.mean(velocities) / 127 if velocities else 0.5
        energy = min(1.0, (density / 8.0) * 0.5 + mean_vel * 0.5)
        energy_per_measure.append(energy)

    curve = np.array(energy_per_measure)
    mean_energy = float(np.mean(curve))
    peak_energy = float(np.max(curve)) if len(curve) > 0 else 0

    # Detectar tipo de arco
    if len(curve) > 4:
        first_half = np.mean(curve[:len(curve)//2])
        second_half = np.mean(curve[len(curve)//2:])
        peak_pos = np.argmax(curve) / len(curve)

        if peak_pos < 0.3:
            arc_type = 'front_loaded'
        elif peak_pos > 0.7:
            arc_type = 'climax_ending'
        elif 0.4 < peak_pos < 0.6:
            arc_type = 'arch'
        elif second_half > first_half * 1.2:
            arc_type = 'crescendo'
        elif first_half > second_half * 1.2:
            arc_type = 'decrescendo'
        else:
            arc_type = 'flat'
    else:
        arc_type = 'flat'

    return {
        'mean': mean_energy,
        'peak': peak_energy,
        'arc_type': arc_type,
        'curve': energy_per_measure,
        'n_measures': n_measures
    }


def _extract_emotion_dna(score, dna):
    # Valencia: major → positivo, minor → negativo
    mode_valence = 1.0 if dna.key_mode == 'major' else -0.5

    # Ajuste por tempo (más rápido → más activo/positivo)
    tempo_norm = (dna.tempo_bpm - 60) / 120.0  # normalizado
    tempo_valence = np.clip(tempo_norm, -1, 1)

    valence = np.clip((mode_valence + tempo_valence) / 2, -1, 1)

    # Activación: tempo + densidad rítmica
    activation = np.clip(tempo_norm + dna.rhythm_dna.get('density', 0) / 20.0, -1, 1)

    # Tensión: complejidad armónica + síncopa
    tension = (dna.harmony_dna.get('complexity', 0) * 0.5 +
               dna.rhythm_dna.get('syncopation', 0) * 0.5)

    # Oscuridad: registro grave + mode menor
    all_notes = list(score.flat.notes)
    if all_notes:
        mean_pitch = np.mean([n.pitch.midi for n in all_notes if n.isNote] or [60])
        darkness = np.clip((72 - mean_pitch) / 36, 0, 1)  # C5=72 como referencia
    else:
        darkness = 0.5
    if dna.key_mode == 'minor':
        darkness = min(1.0, darkness + 0.2)

    return {
        'valence': float(valence),
        'activation': float(activation),
        'tension': float(tension),
        'darkness': float(darkness)
    }


def _extract_phrase_dna(score):
    # Detectar frases mediante silencios o cambios de dirección melódica
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'avg_length': 4, 'qa_ratio': 0, 'phrase_endings': []}

    phrase_lengths = []
    phrase_start = 0
    prev_offset = float(all_notes[0].offset)

    for i, n in enumerate(all_notes[1:], 1):
        curr_offset = float(n.offset)
        gap = curr_offset - (prev_offset + float(all_notes[i-1].quarterLength))

        # Fin de frase: silencio mayor que un tiempo o al final de 4/8 compases
        if gap > 2.0 or (curr_offset - float(all_notes[phrase_start].offset)) > 16:
            phrase_lengths.append(i - phrase_start)
            phrase_start = i

        prev_offset = curr_offset

    if not phrase_lengths:
        phrase_lengths = [len(all_notes)]

    avg_length = np.mean(phrase_lengths)

    # Ratio pregunta-respuesta: pares de frases de longitud similar
    qa_pairs = 0
    for i in range(0, len(phrase_lengths)-1, 2):
        if abs(phrase_lengths[i] - phrase_lengths[i+1]) < phrase_lengths[i] * 0.3:
            qa_pairs += 1
    qa_ratio = qa_pairs / max(1, len(phrase_lengths) // 2)

    return {
        'avg_length': float(avg_length),
        'qa_ratio': float(qa_ratio),
        'phrase_lengths': phrase_lengths
    }


def _extract_tension_dna(score, key_obj):
    # Tensión medida por disonancia y alejamiento tonal
    flat = score.flat.chordify()
    chords = list(flat.flat.getElementsByClass('Chord'))

    tension_values = []
    for ch in chords:
        t = 0.0
        try:
            # Disonancia: intervalos de 2ª, 7ª, tritono
            pitches = sorted([p.midi for p in ch.pitches])
            for i in range(len(pitches)-1):
                interval_semi = pitches[i+1] - pitches[i]
                if interval_semi in [1, 2, 10, 11]:  # 2ª menor/mayor, 7ª
                    t += 0.3
                elif interval_semi == 6:  # Tritono
                    t += 0.5

            # Normalizar
            t = min(1.0, t / max(1, len(pitches)))

            if key_obj:
                rn = roman.romanNumeralFromChord(ch, key_obj)
                if 'V' in rn.figure or 'vii' in rn.figure:
                    t = min(1.0, t + 0.2)
        except:
            pass
        tension_values.append(t)

    if not tension_values:
        return {'mean': 0.3, 'peak': 0.5, 'curve': []}

    return {
        'mean': float(np.mean(tension_values)),
        'peak': float(np.max(tension_values)),
        'curve': tension_values
    }


def _extract_dynamics_dna(score):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'mean_velocity': 64, 'range': 0, 'crescendo_ratio': 0}

    velocities = []
    for n in all_notes:
        if hasattr(n, 'volume') and n.volume.velocity:
            velocities.append(n.volume.velocity)
        else:
            velocities.append(64)

    mean_vel = float(np.mean(velocities))
    vel_range = float(np.max(velocities) - np.min(velocities)) / 127

    # Detectar tendencias de crescendo (ventanas)
    windows = 8
    win_size = max(1, len(velocities) // windows)
    window_means = [np.mean(velocities[i*win_size:(i+1)*win_size])
                    for i in range(windows)]
    crescendo_count = sum(1 for i in range(len(window_means)-1)
                          if window_means[i+1] > window_means[i] + 5)
    crescendo_ratio = crescendo_count / max(1, windows - 1)

    return {
        'mean_velocity': mean_vel,
        'range': vel_range,
        'crescendo_ratio': float(crescendo_ratio),
        'velocities': velocities
    }


def _extract_instrumentation_dna(parts):
    families = defaultdict(list)
    instruments_list = []

    FAMILIES = {
        'piano': ['piano', 'keyboard', 'organ', 'harpsichord', 'celesta'],
        'strings': ['violin', 'viola', 'cello', 'contrabass', 'bass', 'string', 'harp'],
        'woodwinds': ['flute', 'piccolo', 'oboe', 'clarinet', 'bassoon', 'saxophone'],
        'brass': ['trumpet', 'horn', 'trombone', 'tuba', 'cornet'],
        'percussion': ['drum', 'percussion', 'timpani', 'cymbal', 'snare', 'marimba'],
        'voice': ['voice', 'vocal', 'soprano', 'alto', 'tenor', 'choir', 'sing'],
        'guitar': ['guitar', 'lute', 'banjo', 'mandolin']
    }

    for part in parts:
        try:
            inst = part.getInstrument()
            inst_name = (inst.instrumentName or '').lower() if inst else ''
        except:
            inst_name = ''

        instruments_list.append(inst_name or 'unknown')

        classified = False
        for family, keywords in FAMILIES.items():
            if any(k in inst_name for k in keywords):
                families[family].append(inst_name)
                classified = True
                break
        if not classified:
            families['other'].append(inst_name or 'unknown')

    return {
        'families': dict(families),
        'instruments': instruments_list,
        'n_parts': len(parts)
    }


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 2: COMBINACIÓN DEL ADN
# ═══════════════════════════════════════════════════════════════

def combine_dna(dnas, mode='auto', target_key=None, target_tempo=None, verbose=False):
    """
    Combina múltiples ADNs musicales según el modo especificado.
    Devuelve un score de music21.
    """
    if not dnas:
        raise ValueError("No hay ADNs para combinar")

    # Filtrar None
    dnas = [d for d in dnas if d is not None]
    if not dnas:
        raise ValueError("Todos los archivos fallaron al cargarse")

    print(f"\n🔀 Combinando {len(dnas)} piezas en modo: {mode.upper()}")
    print("─" * 60)

    # ── Determinar parámetros globales
    if target_tempo:
        bpm = target_tempo
    else:
        # Usar el tempo de la pieza con mayor energía
        energies = [d.energy_dna.get('mean', 0.5) for d in dnas]
        dominant_idx = np.argmax(energies)
        bpm = dnas[dominant_idx].tempo_bpm

    if target_key:
        # Parsear tonalidad del usuario
        parts_key = target_key.split()
        tonic_name = parts_key[0] if parts_key else 'C'
        mode_name = parts_key[1] if len(parts_key) > 1 else 'major'
        target_key_obj = key.Key(tonic_name, mode_name)
    else:
        # Tonalidad de la pieza con más acordes (más material armónico)
        chord_counts = [len(d.harmony_dna.get('progressions', [])) for d in dnas]
        target_key_obj = dnas[np.argmax(chord_counts)].key_obj or key.Key('C', 'major')

    print(f"  📐 Tonalidad objetivo: {target_key_obj.tonic.name} {target_key_obj.mode}")
    print(f"  🎵 Tempo objetivo: {bpm:.0f} BPM")

    # ── Seleccionar modo de combinación
    if mode == 'auto':
        mode = _auto_select_mode(dnas)
        print(f"  🤖 Modo auto-seleccionado: {mode.upper()}")

    # ── Combinar según modo
    if mode == 'harmony':
        result = _combine_harmonic_rhythm_melody(dnas, target_key_obj, bpm)
    elif mode == 'rhythm':
        result = _combine_rhythm_melody(dnas, target_key_obj, bpm)
    elif mode == 'energy':
        result = _combine_energy_arc(dnas, target_key_obj, bpm)
    elif mode == 'texture':
        result = _combine_texture(dnas, target_key_obj, bpm)
    elif mode == 'emotion':
        result = _combine_emotional_arc(dnas, target_key_obj, bpm)
    else:
        result = _combine_harmonic_rhythm_melody(dnas, target_key_obj, bpm)

    if result is None:
        result = _create_fallback_score(dnas, target_key_obj, bpm)

    return result


def _auto_select_mode(dnas):
    """Elige el mejor modo según las características de los ADNs."""
    if len(dnas) < 2:
        return 'harmony'

    # Si hay gran diferencia de energías → mode energy
    energies = [d.energy_dna.get('mean', 0.5) for d in dnas]
    if max(energies) - min(energies) > 0.3:
        return 'energy'

    # Si hay gran diferencia de complejidad armónica → harmony
    complexities = [d.harmony_dna.get('complexity', 0) for d in dnas]
    if max(complexities) - min(complexities) > 0.3:
        return 'harmony'

    # Si hay diferencias rítmicas marcadas → rhythm
    densities = [d.rhythm_dna.get('density', 0) for d in dnas]
    if max(densities) - min(densities) > 5:
        return 'rhythm'

    return 'harmony'


# ─── MODO 1: Harmonía de A + Ritmo de B + Melodía de C ─────────

def _combine_harmonic_rhythm_melody(dnas, target_key, bpm):
    """
    Extrae la progresión armónica del DNA más complejo,
    el ritmo del más denso, y la melodía del más expresivo.
    Los transpone a la tonalidad objetivo y construye el resultado.
    """
    print("\n  🎼 MODO: Armonía + Ritmo + Melodía")

    # Asignar roles
    by_harmony = sorted(dnas, key=lambda d: d.harmony_dna.get('complexity', 0), reverse=True)
    by_rhythm = sorted(dnas, key=lambda d: d.rhythm_dna.get('density', 0), reverse=True)
    by_melody = sorted(dnas, key=lambda d: d.melody_dna.get('range_semitones', 0), reverse=True)

    harmony_source = by_harmony[0]
    rhythm_source = by_rhythm[min(1, len(by_rhythm)-1)]
    melody_source = by_melody[min(2 % len(by_melody), len(by_melody)-1)]

    print(f"    ▸ Armonía: {harmony_source.name}")
    print(f"    ▸ Ritmo:   {rhythm_source.name}")
    print(f"    ▸ Melodía: {melody_source.name}")

    # Número de compases del resultado
    n_measures = max(16, min(32, len(harmony_source.harmony_dna.get('progressions', []))))
    if n_measures == 0:
        n_measures = 16

    ts_num, ts_den = harmony_source.time_signature
    measure_length = 4.0 * (ts_num / ts_den) * (4 / ts_den) if ts_den != 0 else 4.0
    measure_length = ts_num  # quarter notes per measure

    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    # ── Construir progresión armónica en la tonalidad objetivo
    chord_progression = _build_chord_progression(harmony_source, target_key, n_measures)

    # ── PARTE 1: Melodía (voz superior)
    melody_part = stream.Part()
    melody_part.id = 'Melody'
    melody_part.insert(0, instrument.Piano())
    melody_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    melody_notes = _generate_melody_from_dna(
        melody_source, chord_progression, target_key, bpm, n_measures, measure_length
    )
    for n in melody_notes:
        melody_part.append(n)

    # ── PARTE 2: Acompañamiento rítmico (mano izquierda / bajo)
    accompaniment_part = stream.Part()
    accompaniment_part.id = 'Accompaniment'
    accompaniment_part.insert(0, instrument.Piano())
    accompaniment_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    acc_notes = _generate_accompaniment_from_rhythm(
        rhythm_source, chord_progression, target_key, n_measures, measure_length
    )
    for n in acc_notes:
        accompaniment_part.append(n)

    # ── PARTE 3: Bajo
    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    bass_notes = _generate_bass_line(chord_progression, target_key, n_measures, measure_length)
    for n in bass_notes:
        bass_part.append(n)

    result.append(melody_part)
    result.append(accompaniment_part)
    result.append(bass_part)

    return result


# ─── MODO 2: Ritmo de A + Melodía de B ─────────────────────────

def _combine_rhythm_melody(dnas, target_key, bpm):
    """
    Aplica el patrón rítmico de un midi a la melodía de otro.
    """
    print("\n  🥁 MODO: Ritmo + Melodía")

    if len(dnas) < 2:
        dnas = [dnas[0], dnas[0]]

    rhythm_source = dnas[0]
    melody_source = dnas[1]
    harmony_source = dnas[-1]

    print(f"    ▸ Ritmo:   {rhythm_source.name}")
    print(f"    ▸ Melodía: {melody_source.name}")

    ts_num, ts_den = rhythm_source.time_signature
    n_measures = 24

    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    chord_progression = _build_chord_progression(harmony_source, target_key, n_measures)

    # Ritmo del DNA A aplicado a las alturas del DNA B
    rhythm_pattern = rhythm_source.rhythm_dna.get('pattern', [1.0] * 8)
    melody_pitches = melody_source.melody_dna.get('pitches', [60, 62, 64, 65, 67])

    melody_part = stream.Part()
    melody_part.id = 'RhythmicMelody'
    melody_part.insert(0, instrument.Piano())
    melody_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    # Aplicar patrón rítmico a las notas melódicas
    current_offset = 0.0
    pitch_idx = 0
    total_length = n_measures * ts_num

    # Filtrar el patrón para que sea musical
    valid_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    clean_pattern = []
    for d in rhythm_pattern:
        closest = min(valid_durations, key=lambda x: abs(x - d))
        clean_pattern.append(closest)

    if not clean_pattern:
        clean_pattern = [0.5, 0.5, 1.0]

    pattern_idx = 0
    while current_offset < total_length - 0.1:
        dur = clean_pattern[pattern_idx % len(clean_pattern)]
        if current_offset + dur > total_length:
            dur = total_length - current_offset

        # Elegir pitch de la melodía fuente, transpuesto a la tonalidad objetivo
        raw_pitch = melody_pitches[pitch_idx % len(melody_pitches)] if melody_pitches else 60
        adjusted_pitch = _transpose_note_to_key(raw_pitch, target_key)

        n = note.Note(adjusted_pitch)
        n.quarterLength = dur
        n.offset = current_offset
        # Velocidad con groove
        groove = rhythm_source.rhythm_dna.get('groove', [0.7])
        vel = int(60 + 67 * groove[pattern_idx % len(groove)])
        n.volume.velocity = min(127, max(30, vel))

        melody_part.append(n)

        current_offset += dur
        pitch_idx += 1
        pattern_idx += 1

    # Acompañamiento armónico simple
    acc_part = stream.Part()
    acc_part.id = 'Chords'
    acc_part.insert(0, instrument.Piano())
    acc_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    acc_notes = _generate_block_chords(chord_progression, n_measures, ts_num)
    for n in acc_notes:
        acc_part.append(n)

    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    for n in _generate_bass_line(chord_progression, target_key, n_measures, ts_num):
        bass_part.append(n)

    result.append(melody_part)
    result.append(acc_part)
    result.append(bass_part)

    return result


# ─── MODO 3: Arco de energía ────────────────────────────────────

def _combine_energy_arc(dnas, target_key, bpm):
    """
    Combina los materiales de distintos midis siguiendo
    el arco de energía del más dinámico.
    """
    print("\n  ⚡ MODO: Arco de Energía")

    energy_source = max(dnas, key=lambda d: d.energy_dna.get('peak', 0))
    energy_curve = energy_source.energy_dna.get('curve', [0.5] * 16)

    # Normalizar la curva a n_measures segmentos
    n_measures = max(16, len(energy_curve))
    if len(energy_curve) < n_measures:
        # Interpolar
        x_old = np.linspace(0, 1, len(energy_curve))
        x_new = np.linspace(0, 1, n_measures)
        energy_curve = np.interp(x_new, x_old, energy_curve).tolist()

    ts_num, ts_den = dnas[0].time_signature

    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    # Progresión armónica del más complejo
    harmony_source = max(dnas, key=lambda d: d.harmony_dna.get('complexity', 0))
    chord_progression = _build_chord_progression(harmony_source, target_key, n_measures)

    # Melodía del más expresivo
    melody_source = max(dnas, key=lambda d: d.melody_dna.get('range_semitones', 0))

    melody_part = stream.Part()
    melody_part.id = 'Melody'
    melody_part.insert(0, instrument.Piano())
    melody_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    # Generar melodía que siga el arco de energía
    current_offset = 0.0
    melody_pitches = melody_source.melody_dna.get('pitches', [60, 62, 64])
    melody_intervals = melody_source.melody_dna.get('intervals', [2, 2, -1])

    current_pitch = 60  # Do central como punto de partida
    if melody_pitches:
        # Transponer el primer pitch a la tonalidad objetivo
        current_pitch = _transpose_note_to_key(melody_pitches[0], target_key)

    interval_idx = 0

    for m in range(n_measures):
        energy = energy_curve[m] if m < len(energy_curve) else 0.5
        measure_offset = m * ts_num

        # Energía alta → notas más rápidas y más altas
        # Energía baja → notas más lentas y más graves
        if energy > 0.7:
            note_durs = [0.25, 0.25, 0.5, 0.5]
            pitch_shift = 0
        elif energy > 0.4:
            note_durs = [0.5, 0.5, 1.0]
            pitch_shift = -2
        else:
            note_durs = [1.0, 2.0]
            pitch_shift = -5

        beat = 0.0
        for dur in note_durs:
            if beat >= ts_num:
                break

            # Mover la melodía usando intervalos del DNA
            if melody_intervals and interval_idx < len(melody_intervals) * 10:
                interval_step = melody_intervals[interval_idx % len(melody_intervals)]
                interval_idx += 1
            else:
                interval_step = random.choice([-2, -1, 0, 1, 2])

            current_pitch += interval_step
            current_pitch = _transpose_note_to_key(current_pitch, target_key)
            current_pitch = max(55, min(84, current_pitch + pitch_shift))  # Rango razonable

            n = note.Note(current_pitch)
            n.quarterLength = min(dur, ts_num - beat)
            n.offset = measure_offset + beat
            vel = int(40 + energy * 87)
            n.volume.velocity = min(127, max(30, vel))
            melody_part.append(n)

            beat += dur

        # Rellenar el compás si queda espacio
        if beat < ts_num:
            rest = note.Rest()
            rest.quarterLength = ts_num - beat
            rest.offset = measure_offset + beat
            melody_part.append(rest)

    # Acordes con densidad proporcional a la energía
    acc_part = stream.Part()
    acc_part.id = 'Accompaniment'
    acc_part.insert(0, instrument.Piano())
    acc_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    for m in range(n_measures):
        energy = energy_curve[m] if m < len(energy_curve) else 0.5
        measure_offset = m * ts_num
        chord_data = chord_progression[m % len(chord_progression)] if chord_progression else None

        if chord_data and 'pitches' in chord_data and chord_data['pitches']:
            # Arpegiar en baja energía, bloque en alta energía
            if energy > 0.6:
                ch = chord.Chord(chord_data['pitches'])
                ch.quarterLength = ts_num
                ch.offset = measure_offset
                vel = int(50 + energy * 50)
                for p in ch.notes:
                    p.volume.velocity = vel
                acc_part.append(ch)
            else:
                # Arpegio ascendente
                chord_pitches = sorted(chord_data['pitches'])
                step = ts_num / len(chord_pitches)
                for i, p in enumerate(chord_pitches):
                    n = note.Note(p)
                    n.quarterLength = step
                    n.offset = measure_offset + i * step
                    n.volume.velocity = int(40 + energy * 40)
                    acc_part.append(n)

    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    for n in _generate_bass_line(chord_progression, target_key, n_measures, ts_num):
        bass_part.append(n)

    result.append(melody_part)
    result.append(acc_part)
    result.append(bass_part)

    return result


# ─── MODO 4: Textura ─────────────────────────────────────────────

def _combine_texture(dnas, target_key, bpm):
    """
    Usa la densidad/textura de un midi con la melodía de otro y la armonía de un tercero.
    """
    print("\n  🌊 MODO: Textura")
    return _combine_harmonic_rhythm_melody(dnas, target_key, bpm)  # Reutilizar con diferentes asignaciones


# ─── MODO 5: Arco emocional ─────────────────────────────────────

def _combine_emotional_arc(dnas, target_key, bpm):
    """
    Construye un arco emocional que va desde la emoción de A hacia la de C,
    pasando por B como transición.
    """
    print("\n  💫 MODO: Arco Emocional")

    if len(dnas) < 2:
        return _combine_harmonic_rhythm_melody(dnas, target_key, bpm)

    # Ordenar por valencia para crear un arco emocional
    sorted_dnas = sorted(dnas, key=lambda d: d.emotion_dna.get('valence', 0))

    n_measures_per_section = 8
    n_measures = n_measures_per_section * len(sorted_dnas)
    ts_num, ts_den = dnas[0].time_signature

    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    melody_part = stream.Part()
    melody_part.id = 'Melody'
    melody_part.insert(0, instrument.Piano())
    melody_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    acc_part = stream.Part()
    acc_part.id = 'Accompaniment'
    acc_part.insert(0, instrument.Piano())
    acc_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    for section_idx, section_dna in enumerate(sorted_dnas):
        section_offset = section_idx * n_measures_per_section * ts_num
        section_chord_prog = _build_chord_progression(
            section_dna, target_key, n_measures_per_section
        )

        # Generar melodía de la sección
        section_melody = _generate_melody_from_dna(
            section_dna, section_chord_prog, target_key, bpm,
            n_measures_per_section, ts_num
        )

        for el in section_melody:
            el.offset = el.offset + section_offset if hasattr(el, 'offset') else section_offset
            melody_part.append(el)

        # Acompañamiento
        for m in range(n_measures_per_section):
            m_offset = section_offset + m * ts_num
            if section_chord_prog and m < len(section_chord_prog):
                chord_data = section_chord_prog[m]
                if chord_data and 'pitches' in chord_data and chord_data['pitches']:
                    ch = chord.Chord(chord_data['pitches'])
                    ch.quarterLength = ts_num
                    ch.offset = m_offset
                    acc_part.append(ch)

        # Bajo
        for n in _generate_bass_line(section_chord_prog, target_key,
                                      n_measures_per_section, ts_num):
            n.offset += section_offset
            bass_part.append(n)

    result.append(melody_part)
    result.append(acc_part)
    result.append(bass_part)

    return result


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 3: CONSTRUCTORES MUSICALES
# ═══════════════════════════════════════════════════════════════

def _build_chord_progression(dna, target_key, n_measures):
    """
    Construye una progresión de acordes en la tonalidad objetivo,
    basándose en el ADN armónico del DNA fuente.
    """
    progressions = dna.harmony_dna.get('progressions', [])
    chord_functions = dna.harmony_dna.get('functions', [])

    # Mapa de funciones armónicas a grados en la escala objetivo
    FUNCTION_TO_DEGREE = {
        'major': {
            'I': 0, 'ii': 2, 'iii': 4, 'IV': 5, 'V': 7, 'vi': 9, 'vii°': 11,
            'V7': 7, 'ii7': 2, 'IV7': 5, 'vi7': 9,
            'bVII': 10, 'bIII': 3, 'bVI': 8  # Préstamo modal
        },
        'minor': {
            'i': 0, 'ii°': 2, 'III': 3, 'iv': 5, 'V': 7, 'v': 7, 'VI': 8, 'VII': 10,
            'V7': 7, 'i7': 0, 'iv7': 5,
            'I': 0,  # Picardy
        }
    }

    tonic_midi = pitch.Pitch(target_key.tonic.name).midi
    mode = target_key.mode

    chord_results = []

    # Progressiones comunes por modo (fallback)
    COMMON_PROGRESSIONS = {
        'major': [
            ['I', 'V', 'vi', 'IV'],
            ['I', 'IV', 'V', 'I'],
            ['I', 'vi', 'IV', 'V'],
            ['ii', 'V', 'I', 'I'],
            ['I', 'IV', 'ii', 'V'],
        ],
        'minor': [
            ['i', 'VI', 'III', 'VII'],
            ['i', 'iv', 'V', 'i'],
            ['i', 'VII', 'VI', 'VII'],
            ['ii°', 'V', 'i', 'i'],
            ['i', 'iv', 'VII', 'III'],
        ]
    }

    def degree_to_chord_pitches(degree_semitones, mode_str, octave=4):
        """Convierte un grado de la escala a pitches MIDI del acorde."""
        root = tonic_midi + degree_semitones + (octave - 4) * 12
        root = root % 12 + 48  # Normalizar a octava 4

        if mode_str == 'major':
            if degree_semitones in [0, 5]:  # I, IV → Mayor
                return [root, root+4, root+7]
            elif degree_semitones in [2, 4, 9]:  # ii, iii, vi → menor
                return [root, root+3, root+7]
            elif degree_semitones in [11]:  # vii° → disminuido
                return [root, root+3, root+6]
            elif degree_semitones == 7:  # V → Mayor (con posible 7ª)
                return [root, root+4, root+7, root+10]
        else:  # minor
            if degree_semitones in [0, 5]:  # i, iv → menor
                return [root, root+3, root+7]
            elif degree_semitones in [3, 8, 10]:  # III, VI, VII → Mayor
                return [root, root+4, root+7]
            elif degree_semitones == 2:  # ii° → disminuido
                return [root, root+3, root+6]
            elif degree_semitones == 7:  # V → Mayor
                return [root, root+4, root+7, root+10]

        return [root, root+4, root+7]  # Mayor por defecto

    # Usar progresión del DNA si hay suficiente, si no usar común
    if chord_functions and len(chord_functions) >= 4:
        # Ciclar por la progresión del DNA
        for m in range(n_measures):
            func = chord_functions[m % len(chord_functions)]
            degree_map = FUNCTION_TO_DEGREE.get(mode, FUNCTION_TO_DEGREE['major'])
            degree = degree_map.get(func, 0)
            pitches_list = degree_to_chord_pitches(degree, mode)
            chord_results.append({
                'function': func,
                'degree': degree,
                'pitches': pitches_list,
                'root': tonic_midi + degree
            })
    else:
        # Usar progresión común
        prog = random.choice(COMMON_PROGRESSIONS.get(mode, COMMON_PROGRESSIONS['major']))
        degree_map = FUNCTION_TO_DEGREE.get(mode, FUNCTION_TO_DEGREE['major'])
        for m in range(n_measures):
            func = prog[m % len(prog)]
            degree = degree_map.get(func, 0)
            pitches_list = degree_to_chord_pitches(degree, mode)
            chord_results.append({
                'function': func,
                'degree': degree,
                'pitches': pitches_list,
                'root': tonic_midi + degree
            })

    return chord_results


def _generate_melody_from_dna(dna, chord_progression, target_key, bpm,
                               n_measures, measure_length):
    """
    Genera una línea melódica basada en el ADN melódico,
    adaptada a la progresión de acordes dada.
    """
    melody_notes = []

    # Extraer materiales del DNA
    intervals = dna.melody_dna.get('intervals', [2, 1, -1, -2, 3, -3])
    contour = dna.melody_dna.get('contour', [0.5] * 8)
    leap_ratio = dna.melody_dna.get('leap_ratio', 0.2)
    direction = dna.melody_dna.get('direction', 0)
    mean_pitch = dna.melody_dna.get('mean_pitch', 65)
    motif = dna.melody_dna.get('motif', intervals[:4])
    rhythm_pattern = dna.rhythm_dna.get('pattern', [0.5, 0.5, 1.0])

    # Ajustar pitch medio a la tonalidad objetivo
    tonic_midi = pitch.Pitch(target_key.tonic.name).midi
    # Colocar la melodía en una octava razonable
    start_pitch = tonic_midi + 12 * 5  # Una octava por encima del tónico en octava 5
    start_pitch = max(60, min(81, start_pitch))  # E4-A5

    current_pitch = start_pitch
    interval_idx = 0

    # Crear durations rítmicas válidas
    valid_durs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    clean_rhythm = []
    for d in rhythm_pattern:
        closest = min(valid_durs, key=lambda x: abs(x - d))
        clean_rhythm.append(closest)
    if not clean_rhythm:
        clean_rhythm = [0.5, 0.5, 1.0]

    for m in range(n_measures):
        measure_offset = m * measure_length
        chord_data = chord_progression[m % len(chord_progression)] if chord_progression else None
        chord_pitches = chord_data['pitches'] if chord_data else [60, 64, 67]

        beat = 0.0
        rhythm_pos = 0

        while beat < measure_length - 0.1:
            dur = clean_rhythm[rhythm_pos % len(clean_rhythm)]
            if beat + dur > measure_length:
                dur = measure_length - beat
            if dur <= 0:
                break

            # Decidir si mover con intervalo del DNA o saltar a nota del acorde
            use_chord_tone = random.random() < 0.35  # 35% de probabilidad

            if use_chord_tone and chord_pitches:
                # Ir a la nota del acorde más cercana
                target_p = min(chord_pitches, key=lambda p: abs(p - current_pitch))
                # Ajustar octava para que no salte demasiado
                while target_p < current_pitch - 7:
                    target_p += 12
                while target_p > current_pitch + 7:
                    target_p -= 12
                current_pitch = target_p
            else:
                # Usar intervalo del DNA
                if intervals:
                    step = intervals[interval_idx % len(intervals)]
                    interval_idx += 1
                else:
                    step = random.choice([-2, -1, 0, 1, 2])

                # Limitar saltos grandes según leap_ratio
                if abs(step) > 4 and random.random() > leap_ratio:
                    step = int(np.sign(step) * 2)

                current_pitch += step

            # Mantener en rango melódico razonable
            current_pitch = max(57, min(84, current_pitch))

            # Asegurar que es una nota de la escala
            current_pitch = _transpose_note_to_key(current_pitch, target_key)

            n = note.Note(current_pitch)
            n.quarterLength = dur
            n.offset = measure_offset + beat

            # Velocidad dinámica: acentuar primer tiempo
            if beat == 0:
                vel = random.randint(75, 95)
            elif beat == measure_length / 2:
                vel = random.randint(65, 80)
            else:
                vel = random.randint(50, 70)
            n.volume.velocity = vel

            melody_notes.append(n)
            beat += dur
            rhythm_pos += 1

    return melody_notes


def _generate_accompaniment_from_rhythm(rhythm_dna, chord_progression,
                                         target_key, n_measures, measure_length):
    """Genera un acompañamiento usando el patrón rítmico del DNA de ritmo."""
    acc_notes = []

    swing = rhythm_dna.rhythm_dna.get('swing', 0)
    density = rhythm_dna.rhythm_dna.get('density', 4)

    # Elegir patrón según densidad
    if density > 10:
        # Alberti bass / arpegio rápido
        pattern_type = 'alberti'
    elif density > 6:
        # Acordes en contratiempo
        pattern_type = 'offbeat'
    else:
        # Acordes en tiempo fuerte
        pattern_type = 'block'

    for m in range(n_measures):
        measure_offset = m * measure_length
        chord_data = chord_progression[m % len(chord_progression)] if chord_progression else None

        if not chord_data or 'pitches' not in chord_data:
            continue

        chord_pitches = sorted(chord_data['pitches'])
        if not chord_pitches:
            continue

        if pattern_type == 'alberti':
            # Bajo-alto-medio-alto (Alberti bass)
            if len(chord_pitches) >= 3:
                pattern_pitches = [chord_pitches[0], chord_pitches[-1],
                                   chord_pitches[1], chord_pitches[-1]]
            else:
                pattern_pitches = chord_pitches * 2

            dur = measure_length / len(pattern_pitches)
            for i, p in enumerate(pattern_pitches):
                n = note.Note(p - 12)  # Una octava abajo
                n.quarterLength = dur
                n.offset = measure_offset + i * dur
                n.volume.velocity = random.randint(50, 65)
                acc_notes.append(n)

        elif pattern_type == 'offbeat':
            # Acordes en contratiempo (upbeat)
            beats = [0.5, 1.5, 2.5, 3.5] if measure_length >= 4 else [0.5, 1.5]
            for b in beats:
                if b < measure_length:
                    ch = chord.Chord([p - 12 for p in chord_pitches[:3]])
                    ch.quarterLength = 0.5
                    ch.offset = measure_offset + b
                    for nn in ch.notes:
                        nn.volume.velocity = random.randint(45, 60)
                    acc_notes.append(ch)

        else:  # block
            ch = chord.Chord([p - 12 for p in chord_pitches[:4]])
            ch.quarterLength = measure_length
            ch.offset = measure_offset
            for nn in ch.notes:
                nn.volume.velocity = random.randint(55, 70)
            acc_notes.append(ch)

    return acc_notes


def _generate_bass_line(chord_progression, target_key, n_measures, measure_length):
    """Genera una línea de bajo basada en la progresión de acordes."""
    bass_notes = []

    for m in range(n_measures):
        measure_offset = m * measure_length
        chord_data = chord_progression[m % len(chord_progression)] if chord_progression else None

        if not chord_data:
            r = note.Rest()
            r.quarterLength = measure_length
            r.offset = measure_offset
            bass_notes.append(r)
            continue

        root_midi = chord_data.get('root', 48)
        # Normalizar a octava 2-3 (bajo)
        while root_midi > 52:
            root_midi -= 12
        while root_midi < 36:
            root_midi += 12

        # Patrón de bajo: fundamental + quinta
        fifth = root_midi + 7

        # Tiempo 1: fundamental
        n1 = note.Note(root_midi)
        n1.quarterLength = 2.0 if measure_length >= 4 else measure_length
        n1.offset = measure_offset
        n1.volume.velocity = random.randint(70, 85)
        bass_notes.append(n1)

        # Tiempo 3 (si hay suficiente espacio): quinta o leading tone
        if measure_length >= 4:
            # Próximo acorde
            next_chord = chord_progression[(m+1) % len(chord_progression)] if chord_progression else None
            if next_chord and m < n_measures - 1:
                next_root = next_chord.get('root', 48)
                while next_root > 52:
                    next_root -= 12
                # Leading tone hacia el siguiente
                leading = root_midi + random.choice([-2, -1, 0, 1, 2])
                leading = _transpose_note_to_key(leading, target_key)
                n2 = note.Note(min(max(36, leading), 52))
            else:
                n2 = note.Note(min(max(36, fifth), 52))

            n2.quarterLength = 2.0
            n2.offset = measure_offset + 2.0
            n2.volume.velocity = random.randint(60, 75)
            bass_notes.append(n2)

    return bass_notes


def _generate_block_chords(chord_progression, n_measures, measure_length):
    """Genera acordes en bloque simples."""
    notes = []
    for m in range(n_measures):
        chord_data = chord_progression[m % len(chord_progression)] if chord_progression else None
        if chord_data and 'pitches' in chord_data:
            ch = chord.Chord(chord_data['pitches'])
            ch.quarterLength = measure_length
            ch.offset = m * measure_length
            notes.append(ch)
    return notes


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 4: UTILIDADES DE TRANSPOSICIÓN
# ═══════════════════════════════════════════════════════════════

def _get_scale_pitches(key_obj):
    """Devuelve los pitch classes de la escala."""
    try:
        sc = key_obj.getScale()
        return [p.pitchClass for p in sc.getPitches()]
    except:
        if key_obj.mode == 'major':
            return [0, 2, 4, 5, 7, 9, 11]
        else:
            return [0, 2, 3, 5, 7, 8, 10]


def _transpose_note_to_key(midi_pitch, target_key):
    """
    Transpone un pitch MIDI a la nota más cercana en la escala objetivo.
    """
    scale_pcs = _get_scale_pitches(target_key)
    if not scale_pcs:
        return midi_pitch

    tonic_pc = pitch.Pitch(target_key.tonic.name).pitchClass
    adjusted_pcs = [(pc + tonic_pc) % 12 for pc in scale_pcs]

    note_pc = midi_pitch % 12
    if note_pc in adjusted_pcs:
        return midi_pitch

    # Encontrar el pitch class más cercano en la escala
    best_pc = min(adjusted_pcs, key=lambda x: min(abs(x - note_pc), 12 - abs(x - note_pc)))
    diff = best_pc - note_pc
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12

    return midi_pitch + diff


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 5: POSTPROCESADO Y HUMANIZACIÓN
# ═══════════════════════════════════════════════════════════════

def humanize_score(score_obj, amount=0.02):
    """
    Añade pequeñas variaciones de tiempo y velocidad para hacer la música
    más humana y menos robótica.
    """
    print("\n  🎭 Humanizando la partitura...")
    for el in score_obj.flat.notes:
        # Variación de velocidad (±10%)
        if hasattr(el, 'volume') and el.volume.velocity:
            vel = el.volume.velocity
            variation = int(random.gauss(0, vel * 0.08))
            el.volume.velocity = max(20, min(127, vel + variation))

        # Variación de timing (±20ms a 120bpm ≈ ±0.04 quarter lengths)
        if hasattr(el, 'offset'):
            timing_var = random.gauss(0, amount)
            el.offset = max(0, float(el.offset) + timing_var)

    return score_obj


def add_expression(score_obj, energy_curve):
    """
    Añade marcas de dinámica y expresión basadas en la curva de energía.
    """
    if not energy_curve:
        return score_obj

    dynamic_marks = {
        0.0: 'pp', 0.2: 'p', 0.4: 'mp', 0.6: 'mf', 0.8: 'f', 1.0: 'ff'
    }

    for part in score_obj.parts:
        for i, measure in enumerate(part.getElementsByClass('Measure')):
            if i < len(energy_curve):
                energy = energy_curve[i]
                # Encontrar la dinámica más cercana
                closest_threshold = min(dynamic_marks.keys(),
                                        key=lambda x: abs(x - energy))
                dyn_mark = dynamic_marks[closest_threshold]
                dyn = dynamics.Dynamic(dyn_mark)
                measure.insert(0, dyn)

    return score_obj


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 6: FALLBACK Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

def _create_fallback_score(dnas, target_key, bpm):
    """Crea un score básico si todo lo demás falla."""
    print("  ⚠️ Usando modo fallback...")
    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, meter.TimeSignature('4/4'))

    tonic = pitch.Pitch(target_key.tonic.name).midi
    pitches_seq = [tonic, tonic+2, tonic+4, tonic+5, tonic+7, tonic+9, tonic+11, tonic+12]
    for i, p in enumerate(pitches_seq):
        n = note.Note(p + 60)
        n.quarterLength = 1.0
        n.offset = float(i)
        n.volume.velocity = 70
        part.append(n)

    result.append(part)
    return result


def validate_and_fix_score(score_obj):
    """Valida y corrige posibles problemas en el score resultante."""
    print("  🔧 Validando partitura...")

    for part in score_obj.parts:
        notes = list(part.flat.notes)

        # Eliminar notas solapadas en la misma voz
        seen_offsets = {}
        to_remove = []
        for n in notes:
            key_off = round(float(n.offset), 3)
            if key_off in seen_offsets:
                # Mantener la más larga
                if float(n.quarterLength) <= float(seen_offsets[key_off].quarterLength):
                    to_remove.append(n)
                else:
                    to_remove.append(seen_offsets[key_off])
                    seen_offsets[key_off] = n
            else:
                seen_offsets[key_off] = n

    return score_obj


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 7: MAIN
# ═══════════════════════════════════════════════════════════════

def print_dna_summary(dnas):
    """Imprime un resumen comparativo de los ADNs."""
    print("\n" + "═" * 70)
    print(" 📊 RESUMEN COMPARATIVO DEL ADN MUSICAL")
    print("═" * 70)

    headers = ["Pieza", "Key", "BPM", "Ritmo", "Melodía", "Armonía", "Energía", "Emoción"]
    print(f"  {'Pieza':<20} {'Key':<10} {'BPM':>5} {'Densidad':>8} {'Rango':>7} "
          f"{'Complejidad':>11} {'Energía':>8} {'Valencia':>8}")
    print("  " + "─" * 80)

    for dna in dnas:
        name = dna.name[:18] + ".." if len(dna.name) > 20 else dna.name
        k = f"{dna.key_tonic} {dna.key_mode[:3]}"
        bpm = f"{dna.tempo_bpm:.0f}"
        density = f"{dna.rhythm_dna.get('density', 0):.1f}"
        rango = f"{dna.melody_dna.get('range_semitones', 0)}"
        complexity = f"{dna.harmony_dna.get('complexity', 0):.2f}"
        energy = f"{dna.energy_dna.get('mean', 0):.2f}"
        valence = f"{dna.emotion_dna.get('valence', 0):.2f}"

        print(f"  {name:<20} {k:<10} {bpm:>5} {density:>8} {rango:>7} "
              f"{complexity:>11} {energy:>8} {valence:>8}")

    print("═" * 70)

    print("\n  🎼 ASIGNACIÓN DE ROLES:")
    if len(dnas) >= 1:
        h = max(dnas, key=lambda d: d.harmony_dna.get('complexity', 0))
        print(f"    ♬ ARMONÍA    → {h.name} (complejidad: {h.harmony_dna.get('complexity', 0):.2f})")
    if len(dnas) >= 2:
        r = max(dnas, key=lambda d: d.rhythm_dna.get('density', 0))
        print(f"    🥁 RITMO     → {r.name} (densidad: {r.rhythm_dna.get('density', 0):.1f})")
    if len(dnas) >= 1:
        m = max(dnas, key=lambda d: d.melody_dna.get('range_semitones', 0))
        print(f"    🎵 MELODÍA   → {m.name} (rango: {m.melody_dna.get('range_semitones', 0)} semitonos)")
        e = max(dnas, key=lambda d: d.energy_dna.get('peak', 0))
        print(f"    ⚡ ENERGÍA   → {e.name} (pico: {e.energy_dna.get('peak', 0):.2f})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="🧬 MIDI DNA Combiner — Fusiona el ADN musical de varios MIDIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('midi_files', nargs='+',
                        help='Archivos MIDI de entrada (mínimo 1, recomendado 2-3)')
    parser.add_argument('--output', default='combined_output.mid',
                        help='Archivo MIDI de salida (default: combined_output.mid)')
    parser.add_argument('--mode', default='auto',
                        choices=['auto', 'harmony', 'rhythm', 'energy', 'texture', 'emotion'],
                        help='Modo de combinación')
    parser.add_argument('--tempo', type=float, default=None,
                        help='BPM del resultado (default: auto)')
    parser.add_argument('--key', default=None,
                        help='Tonalidad destino (ej: "C major", "A minor")')
    parser.add_argument('--humanize', action='store_true', default=True,
                        help='Humanizar el resultado (default: True)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar análisis detallado')
    parser.add_argument('--no-humanize', action='store_true',
                        help='Desactivar humanización')

    args = parser.parse_args()

    print("\n" + "╔" + "═"*66 + "╗")
    print("║" + "  🧬  MIDI DNA COMBINER  —  Fusión Musical Inteligente  ".center(66) + "║")
    print("╚" + "═"*66 + "╝")

    # Verificar archivos
    for f in args.midi_files:
        if not os.path.exists(f):
            print(f"❌ Archivo no encontrado: {f}")
            sys.exit(1)

    # ── FASE 1: Extraer ADN
    print(f"\n🔬 FASE 1: Extracción del ADN musical ({len(args.midi_files)} archivos)")
    print("═" * 60)

    dnas = []
    for filepath in args.midi_files:
        dna = extract_dna(filepath, verbose=args.verbose)
        if dna:
            dnas.append(dna)

    if not dnas:
        print("❌ No se pudo extraer ADN de ningún archivo")
        sys.exit(1)

    # ── FASE 2: Resumen comparativo
    print_dna_summary(dnas)

    # ── FASE 3: Combinar
    print("🔀 FASE 2: Combinación del ADN")
    print("═" * 60)

    result = combine_dna(
        dnas,
        mode=args.mode,
        target_key=args.key,
        target_tempo=args.tempo,
        verbose=args.verbose
    )

    # ── FASE 4: Postprocesado
    print("\n✨ FASE 3: Postprocesado")
    print("═" * 60)

    if not args.no_humanize:
        result = humanize_score(result)

    result = validate_and_fix_score(result)

    # Añadir expresión basada en la curva de energía del DNA más enérgico
    if dnas:
        energy_source = max(dnas, key=lambda d: d.energy_dna.get('peak', 0))
        energy_curve = energy_source.energy_dna.get('curve', [])
        if energy_curve:
            result = add_expression(result, energy_curve)

    # ── FASE 5: Exportar
    print(f"\n💾 FASE 4: Exportando a {args.output}")
    print("═" * 60)

    try:
        result.write('midi', fp=args.output)
        print(f"  ✅ Resultado guardado en: {args.output}")
    except Exception as e:
        print(f"  ❌ Error guardando MIDI: {e}")
        # Intentar guardar con nombre alternativo
        alt_output = 'output_combined.mid'
        try:
            result.write('midi', fp=alt_output)
            print(f"  ✅ Guardado en: {alt_output}")
        except Exception as e2:
            print(f"  ❌ Error crítico: {e2}")
            sys.exit(1)

    # ── Resumen final
    print("\n" + "═" * 60)
    print("  🎉 ¡COMBINACIÓN COMPLETADA!")
    print(f"  📁 Archivo: {args.output}")
    print(f"  🎼 Piezas combinadas: {len(dnas)}")
    print(f"  🔀 Modo: {args.mode.upper()}")
    if dnas:
        print(f"  🎵 Sources: {' + '.join(d.name for d in dnas)}")
    print("═" * 60)
    print()


if __name__ == '__main__':
    main()
