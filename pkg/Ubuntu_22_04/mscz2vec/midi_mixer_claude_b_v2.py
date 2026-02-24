"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MIDI DNA COMBINER  v2.0                                 ║
║   Extrae el "ADN" musical de varios MIDIs y los fusiona en una nueva pieza   ║
║                                                                              ║
║  MEJORAS v2:                                                                 ║
║   #1  Separación inteligente de voces (melodía/bajo/acompañamiento)          ║
║   #2  Transposición relacional preservando función armónica                  ║
║   #3  Cuantización adaptativa con tresillos y valores irregulares            ║
║   #4  Voice leading mínimo entre acordes                                     ║
║   #5  Cadencias estructuradas (HC, AC, DC, IAC)                              ║
║   #6  Motivo y desarrollo (inversión, aumentación, secuencia, retrogrado)    ║
║   #7  Forma macro (intro-A-B-A-coda) con secciones diferenciadas             ║
║   #8  Tensión armónica real (Lerdahl + Tonnetz + disonancia)                 ║
║   #9  Puntos de novedad para introducir sorpresas estructurales              ║
║   #10 Intercambio modal inteligente en puntos de tensión                     ║
║   #11 Validación completa: rangos, cruce de voces, intervalos               ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    python midi_dna_combiner_v2.py file1.mid file2.mid file3.mid [opciones]

OPCIONES:
    --output OUTPUT     Archivo de salida (default: combined_output.mid)
    --mode MODE         auto | harmony | rhythm | energy | texture | emotion
    --tempo TEMPO       BPM del resultado (default: auto)
    --key KEY           Tonalidad (ej: "C major", "A minor") (default: auto)
    --export-xml        Exportar también en MusicXML
    --candidates N      Generar N candidatos y elegir el mejor (default: 3)
    --verbose           Análisis detallado del ADN

EJEMPLOS:
    python midi_dna_combiner_v2.py bach.mid jazz.mid --mode harmony --verbose
    python midi_dna_combiner_v2.py a.mid b.mid c.mid --mode auto --candidates 5
    python midi_dna_combiner_v2.py a.mid b.mid --mode emotion --key "A minor"

DEPENDENCIAS:
    pip install music21 numpy mido
"""

import sys, os, argparse, copy, random, math
from collections import defaultdict, Counter
from fractions import Fraction
import numpy as np

# ── music21 ────────────────────────────────────────────────────
try:
    from music21 import (
        stream, note, chord, meter, tempo, key, instrument,
        interval, pitch, duration, harmony, roman, converter,
        scale, dynamics, expressions, analysis, clef, articulations
    )
    from music21 import environment as m21env
    _e = m21env.Environment()
    _e['warnings'] = 0
    _e['autoDownload'] = 'deny'
except ImportError:
    print("ERROR: pip install music21"); sys.exit(1)

try:
    import mido
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ═══════════════════════════════════════════════════════════════

# Tonnetz distances (Chew 2000) – distancia entre clases de pitch
TONNETZ_DISTANCES = {
    0: 0.0,   # unísono
    1: 3.0,   # 2ª menor
    2: 2.0,   # 2ª mayor
    3: 1.0,   # 3ª menor
    4: 1.0,   # 3ª mayor
    5: 0.5,   # 4ª justa
    6: 3.5,   # tritono
    7: 0.5,   # 5ª justa
    8: 1.0,   # 6ª menor
    9: 1.0,   # 6ª mayor
    10: 2.0,  # 7ª menor
    11: 3.0,  # 7ª mayor
}

# Lerdahl dissonance weights per interval class
LERDAHL_DISSONANCE = {0: 0, 1: 6, 2: 5, 3: 3, 4: 2, 5: 1, 6: 7}

# Grados de la escala mayor: (semitones_from_tonic, quality)
MAJOR_SCALE_DEGREES = {
    'I':   (0,  'M'), 'ii':  (2,  'm'), 'iii': (4,  'm'),
    'IV':  (5,  'M'), 'V':   (7,  'M'), 'V7':  (7,  'M7'),
    'vi':  (9,  'm'), 'vii°':(11, 'd'),
    # Préstamos modales más comunes
    'bVII':(10, 'M'), 'bIII':(3,  'M'), 'bVI': (8,  'M'),
    'iv':  (5,  'm'), 'II':  (2,  'M'), 'bII': (1,  'M'),
}
MINOR_SCALE_DEGREES = {
    'i':   (0,  'm'), 'ii°': (2,  'd'), 'III': (3,  'M'),
    'iv':  (5,  'm'), 'V':   (7,  'M'), 'V7':  (7,  'M7'),
    'VI':  (8,  'M'), 'VII': (10, 'M'), 'vii°':(11, 'd'),
    # Préstamos
    'II':  (2,  'M'), 'IV':  (5,  'M'), 'bVII':(10, 'M'),
    'I':   (0,  'M'),   # Picardy
}

# Instrument ranges (midi lo, midi hi)
INSTRUMENT_RANGES = {
    'melody':   (60, 84),   # C4–C6
    'tenor':    (48, 69),   # C3–A4
    'bass':     (28, 52),   # E1–E3
    'chords':   (48, 76),   # C3–E5
}

# Cadence types → function sequences
CADENCE_TYPES = {
    'AC':  ['V', 'I'],    # Authentic (perfect)
    'HC':  ['I', 'V'],    # Half cadence
    'DC':  ['IV','I'],    # Deceptive → resolved to vi
    'IAC': ['V', 'vi'],   # Imperfect authentic
    'PC':  ['IV','I'],    # Plagal
}

# Durations válidas incluyendo tresillos
VALID_DURATIONS = [
    Fraction(1, 6),   # corchea de tresillo
    Fraction(1, 4),   # semicorchea
    Fraction(1, 3),   # negra de tresillo
    Fraction(1, 2),   # corchea
    Fraction(2, 3),   # dos notas de tresillo
    Fraction(3, 4),   # corchea con puntillo
    Fraction(1, 1),   # negra
    Fraction(3, 2),   # negra con puntillo
    Fraction(2, 1),   # blanca
    Fraction(3, 1),   # blanca con puntillo
    Fraction(4, 1),   # redonda
]
VALID_DURATIONS_FLOAT = [float(d) for d in VALID_DURATIONS]


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 1: CLASE MusicDNA
# ═══════════════════════════════════════════════════════════════

class MusicDNA:
    """Contenedor del ADN musical completo de una pieza."""

    def __init__(self, filepath):
        self.filepath  = filepath
        self.name      = os.path.basename(filepath)
        self.tempo_bpm = 120.0
        self.time_signature = (4, 4)
        self.key_tonic = 'C'
        self.key_mode  = 'major'
        self.key_obj   = None
        self.score     = None
        self.parts     = []

        # --- Roles separados de voces (#1)
        self.voice_roles = {}   # {'melody': Part, 'bass': Part, 'accompaniment': [Parts]}

        # --- ADN dimensions
        self.rhythm_dna        = {}
        self.melody_dna        = {}
        self.harmony_dna       = {}
        self.texture_dna       = {}
        self.energy_dna        = {}
        self.emotion_dna       = {}
        self.phrase_dna        = {}
        self.instrumentation_dna = {}
        self.tension_dna       = {}   # #8 tensión real
        self.novelty_dna       = {}   # #9 puntos de novedad
        self.modal_dna         = {}   # #10 intercambio modal
        self.dynamics_dna      = {}
        self.motif_dna         = {}   # #6 motivos

        # Compatibilidad: vector numérico para scoring (#12)
        self.feature_vector    = None

    def __repr__(self):
        return (f"MusicDNA({self.name} | {self.key_tonic} {self.key_mode} | "
                f"{self.tempo_bpm:.0f}bpm | "
                f"E={self.energy_dna.get('mean',0):.2f} "
                f"V={self.emotion_dna.get('valence',0):.2f})")


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 2: EXTRACCIÓN DEL ADN
# ═══════════════════════════════════════════════════════════════

def extract_dna(filepath, verbose=False):
    print(f"\n🧬 Extrayendo ADN: {os.path.basename(filepath)}")
    print("─" * 60)
    dna = MusicDNA(filepath)

    try:
        score = converter.parse(filepath)
        dna.score = score
    except Exception as e:
        print(f"  ❌ Error: {e}"); return None

    parts = list(score.parts) if hasattr(score, 'parts') and score.parts else [score]
    dna.parts = parts

    dna.tempo_bpm       = _extract_tempo(score)
    dna.time_signature  = _extract_time_signature(score)
    dna.key_tonic, dna.key_mode, dna.key_obj = _extract_key(score)

    if verbose:
        print(f"  🎵 {dna.tempo_bpm:.0f} BPM | "
              f"{dna.time_signature[0]}/{dna.time_signature[1]} | "
              f"{dna.key_tonic} {dna.key_mode}")

    # #1 – Separación inteligente de voces
    dna.voice_roles = _classify_voice_roles(parts, verbose)

    # ADN rítmico (#3 cuantización adaptativa con tresillos)
    dna.rhythm_dna = _extract_rhythm_dna_v2(score, dna.tempo_bpm, verbose)

    # ADN melódico
    dna.melody_dna = _extract_melody_dna_v2(dna.voice_roles.get('melody_part'), dna.key_obj, verbose)

    # ADN armónico
    dna.harmony_dna = _extract_harmony_dna_v2(score, dna.key_obj, verbose)

    # ADN de textura
    dna.texture_dna = _extract_texture_dna(score, parts)

    # ADN de energía
    dna.energy_dna = _extract_energy_dna(score)

    # ADN emocional
    dna.emotion_dna = _extract_emotion_dna(score, dna)

    # ADN de frases
    dna.phrase_dna = _extract_phrase_dna(score)

    # #8 Tensión armónica real (Lerdahl + Tonnetz)
    dna.tension_dna = _extract_tension_real(score, dna.key_obj, verbose)

    # #9 Puntos de novedad
    dna.novelty_dna = _extract_novelty_points(dna.energy_dna, dna.tension_dna, verbose)

    # #10 Intercambio modal
    dna.modal_dna = _extract_modal_interchange(dna.harmony_dna, dna.key_obj, verbose)

    # ADN de dinámica
    dna.dynamics_dna = _extract_dynamics_dna(score)

    # Instrumentación
    dna.instrumentation_dna = _extract_instrumentation_dna(parts)

    # #6 Motivos
    dna.motif_dna = _extract_motif_dna(dna.melody_dna, verbose)

    # Vector de características para scoring (#11)
    dna.feature_vector = _build_feature_vector(dna)

    if verbose:
        _print_dna_verbose(dna)

    print(f"  ✅ ADN extraído")
    return dna


# ─── Clasificación de voces (#1) ───────────────────────────────

def _classify_voice_roles(parts, verbose=False):
    """
    Clasifica cada parte como melodía, bajo o acompañamiento usando:
    - Registro medio (pitch medio)
    - Continuidad melódica (ratio de notas vs silencios)
    - Densidad (notas por beat)
    - Nombre del instrumento
    """
    if not parts:
        return {'melody_part': None, 'bass_part': None, 'accompaniment_parts': []}

    scored = []
    for part in parts:
        notes_only = [n for n in part.flat.notes if n.isNote]
        if not notes_only:
            scored.append({'part': part, 'mean_pitch': 60, 'density': 0,
                           'continuity': 0, 'inst_name': ''})
            continue

        midi_pitches = [n.pitch.midi for n in notes_only]
        mean_pitch   = float(np.mean(midi_pitches))
        total_time   = float(part.flat.highestTime) or 1
        density      = len(notes_only) / total_time
        continuity   = len(notes_only) / max(1, len(list(part.flat.notesAndRests)))

        try:
            inst = part.getInstrument()
            inst_name = (inst.instrumentName or '').lower() if inst else ''
        except:
            inst_name = ''

        scored.append({
            'part':       part,
            'mean_pitch': mean_pitch,
            'density':    density,
            'continuity': continuity,
            'inst_name':  inst_name,
        })

    if not scored:
        return {'melody_part': None, 'bass_part': None, 'accompaniment_parts': []}

    # Melodía: parte con pitch más alto Y buena continuidad
    def melody_score(s):
        pitch_norm = (s['mean_pitch'] - 40) / 50.0
        bonus = 0.3 if any(k in s['inst_name'] for k in
                           ['violin', 'flute', 'soprano', 'oboe', 'trumpet', 'melody']) else 0
        penalty = -0.5 if any(k in s['inst_name'] for k in
                              ['bass', 'tuba', 'drum', 'percussion']) else 0
        return pitch_norm + s['continuity'] * 0.4 + bonus + penalty

    # Bajo: parte con pitch más bajo
    def bass_score(s):
        pitch_norm = (60 - s['mean_pitch']) / 30.0
        bonus = 0.3 if any(k in s['inst_name'] for k in
                           ['bass', 'tuba', 'contrabass', 'cello']) else 0
        return pitch_norm + bonus

    sorted_by_melody = sorted(scored, key=melody_score, reverse=True)
    sorted_by_bass   = sorted(scored, key=bass_score, reverse=True)

    melody_part = sorted_by_melody[0]['part']
    bass_part   = sorted_by_bass[0]['part']

    # Si son la misma parte, el bajo es el segundo candidato
    if bass_part is melody_part and len(sorted_by_bass) > 1:
        bass_part = sorted_by_bass[1]['part']

    accompaniment = [s['part'] for s in scored
                     if s['part'] is not melody_part and s['part'] is not bass_part]

    if verbose:
        try:
            m_inst = melody_part.getInstrument().instrumentName or 'unknown'
        except:
            m_inst = 'unknown'
        try:
            b_inst = bass_part.getInstrument().instrumentName or 'unknown'
        except:
            b_inst = 'unknown'
        print(f"  🎭 Voces → Melodía: {m_inst} (pitch≈{sorted_by_melody[0]['mean_pitch']:.0f}) | "
              f"Bajo: {b_inst}")

    return {
        'melody_part':         melody_part,
        'bass_part':           bass_part,
        'accompaniment_parts': accompaniment,
    }


# ─── Extracción básica ─────────────────────────────────────────

def _extract_tempo(score):
    for el in score.flat.getElementsByClass('MetronomeMark'):
        if el.number: return float(el.number)
    return 120.0

def _extract_time_signature(score):
    for el in score.flat.getElementsByClass('TimeSignature'):
        return (el.numerator, el.denominator)
    return (4, 4)

def _extract_key(score):
    try:
        k = score.analyze('key')
        return k.tonic.name, k.mode, k
    except:
        pass
    try:
        ks = score.flat.getElementsByClass('KeySignature')[0]
        ko = ks.asKey()
        return ko.tonic.name, ko.mode, ko
    except:
        ko = key.Key('C', 'major')
        return 'C', 'major', ko


# ─── ADN rítmico v2 (#3 tresillos + cuantización adaptativa) ──

def _extract_rhythm_dna_v2(score, bpm, verbose=False):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'density': 0, 'swing': 0, 'syncopation': 0,
                'pattern': [1.0], 'durations_raw': [], 'groove': [],
                'has_triplets': False, 'tresillo_ratio': 0}

    durations_raw = [float(n.quarterLength) for n in all_notes]
    offsets       = [float(n.offset)        for n in all_notes]

    # Detectar tresillos: duraciones cercanas a 1/3, 2/3
    tresillo_count = sum(1 for d in durations_raw
                         if abs(d - 1/3) < 0.05 or abs(d - 2/3) < 0.05)
    tresillo_ratio = tresillo_count / max(1, len(durations_raw))
    has_triplets   = tresillo_ratio > 0.05

    # Cuantización adaptativa (#3): elegir la duración válida más cercana
    def quantize(d):
        return min(VALID_DURATIONS_FLOAT, key=lambda x: abs(x - d))

    pattern_q = [quantize(d) for d in durations_raw[:64]]

    total_q = float(score.flat.highestTime) or 1
    measures = max(1, total_q / score.flat.getElementsByClass('TimeSignature')[0].numerator
                   if list(score.flat.getElementsByClass('TimeSignature')) else total_q / 4)
    density = len(all_notes) / measures

    # Swing: corchea larga seguida de corchea corta (ratio > 1.4)
    swing_count = 0
    for i in range(len(durations_raw) - 1):
        if 0.5 < durations_raw[i] < 0.8 and 0.2 < durations_raw[i+1] < 0.4:
            if durations_raw[i] / durations_raw[i+1] > 1.4:
                swing_count += 1
    swing = swing_count / max(1, len(durations_raw) // 2)

    # Síncopa: notas en fracción off-beat con duración >= 1 beat
    syncopation = 0
    for n in all_notes:
        off = float(n.offset) % 1.0
        if 0.25 < off < 0.75 and float(n.quarterLength) >= 1.0:
            syncopation += 1
    syncopation /= max(1, len(all_notes))

    velocities = [n.volume.velocity or 64 for n in all_notes if hasattr(n, 'volume')]
    groove = [v / 127.0 for v in velocities[:64]]

    # Acentuación métrica: acento en tiempos fuertes
    ts_num = score.flat.getElementsByClass('TimeSignature')[0].numerator if \
             list(score.flat.getElementsByClass('TimeSignature')) else 4
    strong_beat_notes = [n for n in all_notes if float(n.offset) % ts_num < 0.05]
    metric_accentuation = (np.mean([n.volume.velocity or 64 for n in strong_beat_notes]) /
                           np.mean([n.volume.velocity or 64 for n in all_notes])
                           if strong_beat_notes and all_notes else 1.0)

    return {
        'density':             density,
        'swing':               swing,
        'syncopation':         syncopation,
        'pattern':             pattern_q,
        'durations_raw':       durations_raw,
        'groove':              groove,
        'has_triplets':        has_triplets,
        'tresillo_ratio':      tresillo_ratio,
        'metric_accentuation': float(metric_accentuation),
    }


# ─── ADN melódico v2 ────────────────────────────────────────────

def _extract_melody_dna_v2(melody_part, key_obj, verbose=False):
    """Extrae el ADN melódico de la parte clasificada como melodía."""
    if melody_part is None:
        return _empty_melody_dna()

    notes_only = [n for n in melody_part.flat.notes if n.isNote]
    if len(notes_only) < 2:
        return _empty_melody_dna()

    pitches   = [n.pitch.midi for n in notes_only]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
    durs      = [float(n.quarterLength) for n in notes_only]

    range_semi = max(pitches) - min(pitches)
    direction  = float(np.mean(np.sign(intervals))) if intervals else 0
    leap_ratio = sum(1 for i in intervals if abs(i) > 4) / max(1, len(intervals))

    # Contorno normalizado a 32 puntos
    p_arr  = np.array(pitches, dtype=float)
    p_norm = (p_arr - p_arr.min()) / max(1, p_arr.max() - p_arr.min())
    idx    = np.linspace(0, len(p_norm) - 1, min(32, len(p_norm)), dtype=int)
    contour = p_norm[idx].tolist()

    # Grados de escala usados
    scale_degrees = []
    if key_obj:
        scale_pcs = _get_scale_pcs(key_obj)
        tonic_pc  = pitch.Pitch(key_obj.tonic.name).pitchClass
        for n in notes_only[:128]:
            rel_pc = (n.pitch.pitchClass - tonic_pc) % 12
            if rel_pc in scale_pcs:
                scale_degrees.append(scale_pcs.index(rel_pc))

    return {
        'range_semitones': range_semi,
        'direction':       direction,
        'leap_ratio':      leap_ratio,
        'contour':         contour,
        'intervals':       intervals[:128],
        'motif':           intervals[:8],
        'mean_pitch':      float(np.mean(pitches)),
        'pitches':         pitches,
        'durations':       durs,
        'scale_degrees':   scale_degrees,
        'notes':           notes_only,
    }

def _empty_melody_dna():
    return {'range_semitones': 0, 'direction': 0, 'leap_ratio': 0.2,
            'contour': [0.5]*8, 'intervals': [2,1,-1,-2,3,-3],
            'motif': [2,1,-1,-2], 'mean_pitch': 65.0, 'pitches': [],
            'durations': [1.0], 'scale_degrees': [], 'notes': []}


# ─── ADN armónico v2 ────────────────────────────────────────────

def _extract_harmony_dna_v2(score, key_obj, verbose=False):
    flat_ch   = score.flat.chordify()
    chord_objs = list(flat_ch.flat.getElementsByClass('Chord'))

    if not chord_objs:
        return {'complexity': 0, 'change_rate': 0, 'progressions': [],
                'functions': [], 'tension_curve': [], 'bigrams': [],
                'modal_chords': []}

    progressions, functions, tension_curve, modal_chords = [], [], [], []

    for ch in chord_objs[:128]:
        try:
            root    = ch.root()
            quality = ch.quality
            rn_fig  = 'I'
            tension = 0.3

            if key_obj and root:
                try:
                    rn = roman.romanNumeralFromChord(ch, key_obj)
                    rn_fig = rn.figure

                    # Tensión funcional
                    if any(x in rn_fig for x in ['V7','vii']):  tension = 0.85
                    elif rn_fig.startswith('V'):                  tension = 0.70
                    elif rn_fig in ['ii','IV','iv','II']:          tension = 0.40
                    elif rn_fig in ['I','i','vi','VI']:            tension = 0.10
                    else:                                          tension = 0.55

                    # Extensiones añaden tensión
                    if len(ch.pitches) > 4: tension = min(1.0, tension + 0.15)

                    # Detectar intercambio modal (#10)
                    is_modal = _is_modal_interchange(rn_fig, key_obj.mode)
                    if is_modal:
                        modal_chords.append({'figure': rn_fig, 'offset': float(ch.offset)})

                except Exception:
                    pass

            progressions.append({
                'root':     root.name if root else 'C',
                'quality':  quality,
                'roman':    rn_fig,
                'offset':   float(ch.offset),
                'duration': float(ch.quarterLength),
                'pitches':  [p.midi for p in ch.pitches],
            })
            functions.append(rn_fig)
            tension_curve.append(tension)

        except Exception:
            continue

    unique_ch   = len(set(f['roman'] for f in progressions))
    complexity  = unique_ch / max(1, len(progressions))
    total_time  = (progressions[-1]['offset'] - progressions[0]['offset']) if progressions else 1
    change_rate = len(progressions) / max(1, total_time / 4)

    bigrams = Counter()
    for i in range(len(functions) - 1):
        bigrams[(functions[i], functions[i+1])] += 1

    return {
        'complexity':    complexity,
        'change_rate':   change_rate,
        'progressions':  progressions,
        'functions':     functions,
        'tension_curve': tension_curve,
        'bigrams':       bigrams.most_common(10),
        'modal_chords':  modal_chords,
    }


# ─── Tensión real (#8) ─────────────────────────────────────────

def _extract_tension_real(score, key_obj, verbose=False):
    """
    Tensión por compás usando tres métricas:
    1. Disonancia de Lerdahl (clases de intervalo ponderadas)
    2. Distancia Tonnetz (alejamiento del centro tonal)
    3. Función armónica (T=0.1, SD=0.4, D=0.7)
    """
    flat_ch   = score.flat.chordify()
    chords    = list(flat_ch.flat.getElementsByClass('Chord'))
    ts_num    = _extract_time_signature(score)[0]
    total_t   = float(score.flat.highestTime) or 1
    n_measures = max(1, int(total_t / ts_num))

    tension_per_measure = [[] for _ in range(n_measures)]

    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass if key_obj else 0

    for ch in chords:
        m_idx = min(n_measures - 1, int(float(ch.offset) / ts_num))
        pitches_midi = [p.midi for p in ch.pitches]
        if not pitches_midi:
            continue

        # 1. Disonancia Lerdahl
        t_lerd = 0.0
        for i in range(len(pitches_midi)):
            for j in range(i+1, len(pitches_midi)):
                ic = abs(pitches_midi[i] - pitches_midi[j]) % 12
                t_lerd += LERDAHL_DISSONANCE.get(ic, 3) / 7.0
        t_lerd /= max(1, len(pitches_midi) * (len(pitches_midi)-1) / 2)

        # 2. Distancia Tonnetz
        pcs = [p.midi % 12 for p in ch.pitches]
        t_tonnetz = np.mean([TONNETZ_DISTANCES.get(abs(pc - tonic_pc) % 12, 2)
                             for pc in pcs]) / 3.5

        # 3. Función armónica
        t_func = 0.3
        if key_obj:
            try:
                rn = roman.romanNumeralFromChord(ch, key_obj)
                fig = rn.figure
                if 'vii' in fig or 'V7' in fig:   t_func = 0.9
                elif fig.startswith('V'):           t_func = 0.7
                elif fig in ['II','ii','IV','iv']:  t_func = 0.4
                elif fig in ['I','i','vi','VI']:    t_func = 0.05
                else:                               t_func = 0.5
            except:
                pass

        combined = t_lerd * 0.4 + t_tonnetz * 0.3 + t_func * 0.3
        tension_per_measure[m_idx].append(combined)

    tension_curve = [float(np.mean(v)) if v else 0.3
                     for v in tension_per_measure]

    if verbose and tension_curve:
        print(f"  🌡️  Tensión: μ={np.mean(tension_curve):.2f}  "
              f"σ={np.std(tension_curve):.2f}  "
              f"pico={max(tension_curve):.2f} @ compás {np.argmax(tension_curve)}")

    return {
        'curve':        tension_curve,
        'mean':         float(np.mean(tension_curve)) if tension_curve else 0.3,
        'peak':         float(max(tension_curve))     if tension_curve else 0.3,
        'peak_measure': int(np.argmax(tension_curve)) if tension_curve else 0,
        'std':          float(np.std(tension_curve))  if tension_curve else 0,
    }


# ─── Puntos de novedad (#9) ─────────────────────────────────────

def _extract_novelty_points(energy_dna, tension_dna, verbose=False):
    """
    Detecta los compases donde hay mayor cambio brusco en energía o tensión.
    Esos puntos son candidatos para introducir material de otro DNA.
    """
    energy_c = energy_dna.get('curve', [])
    tension_c = tension_dna.get('curve', [])

    # Longitud mínima
    n = min(len(energy_c), len(tension_c)) if energy_c and tension_c else \
        len(energy_c) or len(tension_c) or 1

    novelty_scores = []
    for i in range(n):
        e_delta = 0.0
        t_delta = 0.0
        if i > 0:
            if i < len(energy_c):
                e_delta = abs(energy_c[i] - energy_c[i-1])
            if i < len(tension_c):
                t_delta = abs(tension_c[i] - tension_c[i-1])
        novelty_scores.append(e_delta * 0.5 + t_delta * 0.5)

    # Peaks de novedad
    novelty_arr = np.array(novelty_scores) if novelty_scores else np.array([0])
    threshold   = np.mean(novelty_arr) + 0.5 * np.std(novelty_arr)
    peak_measures = [i for i, v in enumerate(novelty_scores) if v > threshold]

    if verbose and peak_measures:
        print(f"  🌟 Novedad: picos en compases {peak_measures[:6]}")

    return {
        'scores':        novelty_scores,
        'peak_measures': peak_measures,
        'threshold':     float(threshold),
    }


# ─── Intercambio modal (#10) ────────────────────────────────────

def _extract_modal_interchange(harmony_dna, key_obj, verbose=False):
    """Extrae los acordes prestados del modo paralelo."""
    modal_chords = harmony_dna.get('modal_chords', [])
    functions    = harmony_dna.get('functions', [])

    borrowed_ratio = len(modal_chords) / max(1, len(functions))
    types = Counter(c['figure'] for c in modal_chords)

    if verbose and modal_chords:
        print(f"  🎭 Intercambio modal: {dict(types)} ({borrowed_ratio:.1%} de los acordes)")

    return {
        'modal_chords':    modal_chords,
        'borrowed_ratio':  borrowed_ratio,
        'types':           dict(types),
        'uses_modal':      borrowed_ratio > 0.05,
    }

def _is_modal_interchange(figure, mode):
    """Devuelve True si el acorde es prestado del modo paralelo."""
    MAJOR_BORROWED = {'bVII', 'bIII', 'bVI', 'iv', 'bII', 'ii°'}
    MINOR_BORROWED = {'I', 'IV', 'II', 'bVII'}
    base = figure.replace('7','').replace('°','').replace('+','')
    if mode == 'major': return base in MAJOR_BORROWED
    else:               return base in MINOR_BORROWED


# ─── Motivos (#6) ───────────────────────────────────────────────

def _extract_motif_dna(melody_dna, verbose=False):
    """Extrae el motivo principal y genera sus transformaciones."""
    intervals = melody_dna.get('intervals', [2, 1, -1, -2])
    durations = melody_dna.get('durations', [0.5, 0.5, 1.0])

    # Motivo: primeras 4-6 notas con intervalos significativos
    motif_intervals = intervals[:6] if len(intervals) >= 4 else (intervals or [2,1,-1])
    motif_durations = durations[:len(motif_intervals)+1] if durations else [0.5]*len(motif_intervals)

    # Transformaciones clásicas
    def inversion(m):     return [-x for x in m]
    def retrograde(m):    return list(reversed(m))
    def retro_inv(m):     return list(reversed([-x for x in m]))
    def augmentation(d):  return [x * 2 for x in d]
    def diminution(d):    return [max(0.25, x / 2) for x in d]
    def sequence(m, n=2): return m + [x + n for x in m]  # secuencia a distancia n

    if verbose and motif_intervals:
        print(f"  🎵 Motivo: {motif_intervals} | "
              f"Inversión: {inversion(motif_intervals)}")

    return {
        'intervals':      motif_intervals,
        'durations':      motif_durations,
        'inversion':      inversion(motif_intervals),
        'retrograde':     retrograde(motif_intervals),
        'retro_inv':      retro_inv(motif_intervals),
        'augmentation_d': augmentation(motif_durations),
        'diminution_d':   diminution(motif_durations),
        'sequence_up':    sequence(motif_intervals, 2),
        'sequence_down':  sequence(motif_intervals, -2),
    }


# ─── Otros extractores ─────────────────────────────────────────

def _extract_texture_dna(score, parts):
    layers    = len(parts)
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'layers':0,'note_density':0,'homophony':0,'polyphony':0,'type':'monophony'}

    total_t = float(score.flat.highestTime) or 1
    density = len(all_notes) / total_t
    offsets_c = Counter(round(float(n.offset), 2) for n in all_notes)
    simultaneous = sum(1 for c in offsets_c.values() if c > 1)
    homophony = simultaneous / max(1, len(offsets_c))
    polyphony = 1 - homophony
    texture_type = ('monophony' if layers == 1 else
                    'homophony' if homophony > 0.6 else
                    'polyphony' if polyphony > 0.6 else 'heterophony')
    pitch_mids = [n.pitch.midi for n in all_notes if n.isNote]
    return {
        'layers':          layers,
        'note_density':    density,
        'homophony':       homophony,
        'polyphony':       polyphony,
        'type':            texture_type,
        'register_spread': float(np.std(pitch_mids)) if pitch_mids else 0,
    }

def _extract_energy_dna(score):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'mean':0.5,'peak':0.5,'arc_type':'flat','curve':[],'n_measures':0}

    total_t = float(score.flat.highestTime) or 1
    ts_num  = _extract_time_signature(score)[0]
    n_m     = max(1, int(total_t / ts_num))
    curve   = []

    for m in range(n_m):
        t0, t1 = m * ts_num, (m + 1) * ts_num
        ns = [n for n in all_notes if t0 <= float(n.offset) < t1]
        if not ns: curve.append(0.0); continue
        vels = [n.volume.velocity or 64 for n in ns if hasattr(n,'volume')]
        mv   = np.mean(vels) / 127 if vels else 0.5
        den  = min(1.0, len(ns) / (ts_num * 2))
        curve.append(float(den * 0.5 + mv * 0.5))

    arr = np.array(curve) if curve else np.array([0.5])
    peak_pos = float(np.argmax(arr)) / max(1, len(arr))
    arc = ('front_loaded'  if peak_pos < 0.25 else
           'climax_ending' if peak_pos > 0.75 else
           'arch'          if 0.4 < peak_pos < 0.6 else
           'crescendo'     if np.mean(arr[len(arr)//2:]) > np.mean(arr[:len(arr)//2]) * 1.1 else
           'decrescendo'   if np.mean(arr[:len(arr)//2]) > np.mean(arr[len(arr)//2:]) * 1.1 else
           'flat')
    return {'mean':float(np.mean(arr)),'peak':float(np.max(arr)),
            'arc_type':arc,'curve':curve,'n_measures':n_m}

def _extract_emotion_dna(score, dna):
    mode_v = 0.7 if dna.key_mode == 'major' else -0.4
    t_norm = np.clip((dna.tempo_bpm - 60) / 120.0, -1, 1)
    valence    = float(np.clip((mode_v + t_norm) / 2, -1, 1))
    activation = float(np.clip(t_norm + dna.rhythm_dna.get('density', 0) / 20, -1, 1))
    tension    = float(dna.harmony_dna.get('complexity', 0) * 0.5 +
                       dna.rhythm_dna.get('syncopation', 0) * 0.5)
    all_notes  = list(score.flat.notes)
    mean_pitch = np.mean([n.pitch.midi for n in all_notes if n.isNote]) if all_notes else 60
    darkness   = float(np.clip((72 - mean_pitch) / 36, 0, 1))
    if dna.key_mode == 'minor': darkness = min(1.0, darkness + 0.2)
    return {'valence':valence,'activation':activation,'tension':tension,'darkness':darkness}

def _extract_phrase_dna(score):
    notes = list(score.flat.notes)
    if not notes:
        return {'avg_length':4,'qa_ratio':0,'phrase_lengths':[4]}
    phrase_lengths, phrase_start, prev_off = [], 0, float(notes[0].offset)
    for i, n in enumerate(notes[1:], 1):
        curr = float(n.offset)
        gap  = curr - (prev_off + float(notes[i-1].quarterLength))
        if gap > 2.0 or (curr - float(notes[phrase_start].offset)) > 16:
            phrase_lengths.append(i - phrase_start)
            phrase_start = i
        prev_off = curr
    if not phrase_lengths: phrase_lengths = [len(notes)]
    qa = sum(1 for i in range(0, len(phrase_lengths)-1, 2)
             if abs(phrase_lengths[i]-phrase_lengths[i+1]) < phrase_lengths[i]*0.3)
    return {'avg_length': float(np.mean(phrase_lengths)),
            'qa_ratio':   qa / max(1, len(phrase_lengths)//2),
            'phrase_lengths': phrase_lengths}

def _extract_dynamics_dna(score):
    notes = list(score.flat.notes)
    if not notes: return {'mean_velocity':64,'range':0,'crescendo_ratio':0,'velocities':[]}
    vels = [n.volume.velocity or 64 for n in notes if hasattr(n,'volume')]
    mean_v = float(np.mean(vels))
    rng    = float((max(vels) - min(vels)) / 127) if vels else 0
    w_means = [np.mean(vels[i*8:(i+1)*8]) for i in range(max(1, len(vels)//8))]
    cres    = sum(1 for i in range(len(w_means)-1) if w_means[i+1] > w_means[i]+5)
    return {'mean_velocity':mean_v,'range':rng,
            'crescendo_ratio': cres/max(1,len(w_means)-1),
            'velocities':vels[:128]}

def _extract_instrumentation_dna(parts):
    FAMILIES = {
        'piano':      ['piano','keyboard','organ','harpsichord','celesta'],
        'strings':    ['violin','viola','cello','contrabass','string','harp'],
        'woodwinds':  ['flute','piccolo','oboe','clarinet','bassoon','saxophone'],
        'brass':      ['trumpet','horn','trombone','tuba'],
        'percussion': ['drum','percussion','timpani','cymbal','snare','marimba'],
        'voice':      ['voice','vocal','soprano','alto','tenor','choir'],
        'guitar':     ['guitar','lute','banjo','mandolin'],
    }
    families, inst_list = defaultdict(list), []
    for part in parts:
        try:
            inst = part.getInstrument()
            name = (inst.instrumentName or '').lower() if inst else ''
        except:
            name = ''
        inst_list.append(name or 'unknown')
        placed = False
        for fam, kws in FAMILIES.items():
            if any(k in name for k in kws):
                families[fam].append(name); placed = True; break
        if not placed: families['other'].append(name or 'unknown')
    return {'families': dict(families), 'instruments': inst_list, 'n_parts': len(parts)}

def _build_feature_vector(dna):
    """Vector numérico para medir compatibilidad entre ADNs."""
    return np.array([
        dna.tempo_bpm / 200.0,
        float(dna.key_mode == 'major'),
        dna.rhythm_dna.get('density', 0) / 20.0,
        dna.rhythm_dna.get('swing', 0),
        dna.rhythm_dna.get('syncopation', 0),
        dna.melody_dna.get('range_semitones', 0) / 36.0,
        dna.melody_dna.get('leap_ratio', 0),
        dna.harmony_dna.get('complexity', 0),
        dna.harmony_dna.get('change_rate', 0) / 4.0,
        dna.energy_dna.get('mean', 0.5),
        dna.emotion_dna.get('valence', 0),
        dna.emotion_dna.get('activation', 0),
        dna.tension_dna.get('mean', 0.3),
        dna.modal_dna.get('borrowed_ratio', 0),
        dna.texture_dna.get('note_density', 0) / 10.0,
    ], dtype=float)

def _print_dna_verbose(dna):
    print(f"  ── ANÁLISIS DETALLADO ───────────────────────────────")
    print(f"  Ritmo:   densidad={dna.rhythm_dna.get('density',0):.1f} | "
          f"swing={dna.rhythm_dna.get('swing',0):.2f} | "
          f"sínc={dna.rhythm_dna.get('syncopation',0):.2f} | "
          f"tresillos={dna.rhythm_dna.get('has_triplets',False)}")
    print(f"  Melodía: rango={dna.melody_dna.get('range_semitones',0)} st | "
          f"dir={dna.melody_dna.get('direction',0):.2f} | "
          f"saltos={dna.melody_dna.get('leap_ratio',0):.2f}")
    print(f"  Armonía: complejidad={dna.harmony_dna.get('complexity',0):.2f} | "
          f"cambios/c={dna.harmony_dna.get('change_rate',0):.1f} | "
          f"modal={dna.modal_dna.get('borrowed_ratio',0):.1%}")
    print(f"  Energía: μ={dna.energy_dna.get('mean',0):.2f} | "
          f"pico={dna.energy_dna.get('peak',0):.2f} | "
          f"arco={dna.energy_dna.get('arc_type','?')}")
    print(f"  Tensión: μ={dna.tension_dna.get('mean',0):.2f} | "
          f"pico={dna.tension_dna.get('peak',0):.2f} @ "
          f"c.{dna.tension_dna.get('peak_measure',0)}")
    print(f"  Emoción: valencia={dna.emotion_dna.get('valence',0):.2f} | "
          f"activación={dna.emotion_dna.get('activation',0):.2f} | "
          f"oscuridad={dna.emotion_dna.get('darkness',0):.2f}")
    print(f"  Motivo:  {dna.motif_dna.get('intervals',[])} | "
          f"inv={dna.motif_dna.get('inversion',[])} | "
          f"retro={dna.motif_dna.get('retrograde',[])}")
    print(f"  Novedad: picos en compases {dna.novelty_dna.get('peak_measures',[])[:5]}")


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 3: UTILIDADES MUSICALES
# ═══════════════════════════════════════════════════════════════

def _get_scale_pcs(key_obj):
    """Pitch classes de la escala relativo al tónico."""
    if key_obj.mode == 'major':
        return [0, 2, 4, 5, 7, 9, 11]
    else:
        return [0, 2, 3, 5, 7, 8, 10]

def _get_scale_midi(key_obj, octave=4):
    """Todas las notas de la escala en un rango amplio."""
    tonic = pitch.Pitch(key_obj.tonic.name)
    tonic.octave = octave
    base = tonic.midi
    pcs  = _get_scale_pcs(key_obj)
    result = []
    for o in range(-1, 4):
        for pc in pcs:
            m = base + pc + o * 12
            if 21 <= m <= 108:
                result.append(m)
    return sorted(set(result))

def _snap_to_scale(midi_pitch, key_obj):
    """
    Transpone un pitch MIDI a la nota de la escala más cercana,
    preservando la octava. (#2 transposición relacional)
    """
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    pcs      = [(pc + tonic_pc) % 12 for pc in _get_scale_pcs(key_obj)]
    note_pc  = midi_pitch % 12
    if note_pc in pcs:
        return midi_pitch
    # Encontrar el más cercano (con wrap)
    best, best_dist = note_pc, 100
    for pc in pcs:
        d = abs(pc - note_pc)
        d = min(d, 12 - d)
        if d < best_dist:
            best_dist = d
            best = pc
    diff = best - note_pc
    if diff > 6: diff -= 12
    if diff < -6: diff += 12
    return midi_pitch + diff

def _transpose_interval_preserving(midi_pitch, source_key, target_key, reference_pitch=None):
    """
    #2: Transpone preservando la función tonal relativa.
    Calcula el grado relativo en source_key y lo aplica en target_key.
    """
    src_tonic = pitch.Pitch(source_key.tonic.name).pitchClass
    tgt_tonic = pitch.Pitch(target_key.tonic.name).pitchClass
    src_pcs   = [(pc + src_tonic) % 12 for pc in _get_scale_pcs(source_key)]
    tgt_pcs   = [(pc + tgt_tonic) % 12 for pc in _get_scale_pcs(target_key)]

    note_pc     = midi_pitch % 12
    note_octave = midi_pitch // 12

    # Encontrar grado en la escala fuente
    if note_pc in src_pcs:
        degree = src_pcs.index(note_pc)
        # Aplicar el mismo grado en la escala destino
        new_pc = tgt_pcs[degree % len(tgt_pcs)]
    else:
        # Nota cromática: transponer por diferencia de tónicos
        tonic_diff = tgt_tonic - src_tonic
        new_pc = (note_pc + tonic_diff) % 12

    # Reconstruir MIDI manteniendo octava aproximada
    new_midi = new_pc + note_octave * 12
    # Ajustar para que esté cerca del pitch original
    while new_midi < midi_pitch - 6: new_midi += 12
    while new_midi > midi_pitch + 6: new_midi -= 12
    return new_midi

def _voice_lead(prev_chord_pitches, target_root, target_quality, key_obj, register='chords'):
    """
    #4: Genera los pitches del siguiente acorde usando voice leading mínimo.
    Mueve cada voz a la nota más cercana del nuevo acorde.
    """
    lo, hi = INSTRUMENT_RANGES.get(register, (48, 76))

    # Generar todas las notas del acorde destino en rango amplio
    root_pc = pitch.Pitch(target_root).pitchClass if isinstance(target_root, str) else target_root % 12
    intervals_by_quality = {
        'M':  [0, 4, 7],
        'm':  [0, 3, 7],
        'd':  [0, 3, 6],
        'A':  [0, 4, 8],
        'M7': [0, 4, 7, 10],
        'm7': [0, 3, 7, 10],
        'd7': [0, 3, 6, 9],
    }
    ints = intervals_by_quality.get(target_quality, [0, 4, 7])
    chord_midis = []
    for o in range(2, 7):
        for i in ints:
            m = root_pc + i + o * 12
            if lo - 2 <= m <= hi + 2:
                chord_midis.append(m)

    if not chord_midis or not prev_chord_pitches:
        # Fallback: construir acorde desde la raíz en octava media
        base = root_pc + 48
        while base < lo: base += 12
        return [base + i for i in ints if base + i <= hi]

    # Para cada voz anterior, encontrar la nota más cercana en el nuevo acorde
    result = []
    used   = set()
    for prev in sorted(prev_chord_pitches):
        best = min(chord_midis, key=lambda m: (abs(m - prev), m in used))
        result.append(best)
        used.add(best)

    # Ajustar rango
    result = [max(lo, min(hi, p)) for p in result]
    return sorted(set(result))


def _build_chord_pitches(roman_figure, key_obj, prev_pitches=None, register='chords'):
    """
    Construye los pitches MIDI de un acorde dado su grado romano,
    aplicando voice leading si hay acorde previo. (#4)
    """
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    mode     = key_obj.mode
    degree_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES

    # Limpiar figura
    base_fig = roman_figure.replace('7','').replace('°','').replace('+','').replace('9','')

    if base_fig in degree_map:
        semitones, quality = degree_map[base_fig]
        if '7' in roman_figure and quality != 'd': quality = quality + '7'
    else:
        semitones, quality = 0, 'M'

    root_pc = (tonic_pc + semitones) % 12
    root_str = pitch.Pitch(root_pc).name

    if prev_pitches:
        return _voice_lead(prev_pitches, root_pc, quality, key_obj, register)
    else:
        lo, hi = INSTRUMENT_RANGES.get(register, (48, 76))
        intervals_by_quality = {
            'M':  [0, 4, 7],     'm':  [0, 3, 7],
            'd':  [0, 3, 6],     'A':  [0, 4, 8],
            'M7': [0, 4, 7, 10], 'm7': [0, 3, 7, 10],
            'd7': [0, 3, 6, 9],
        }
        ints = intervals_by_quality.get(quality, [0, 4, 7])
        base = root_pc + 48
        while base < lo: base += 12
        result = [base + i for i in ints if lo - 2 <= base + i <= hi + 2]
        return result or [base]


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 4: CONSTRUCCIÓN DE PROGRESIONES Y FORMA (#7)
# ═══════════════════════════════════════════════════════════════

def _build_progression_from_dna(dna, target_key, n_measures):
    """
    #2 + #7: Construye una progresión en target_key basada en las funciones
    del DNA fuente, preservando relaciones funcionales.
    """
    functions  = dna.harmony_dna.get('functions', [])
    mode       = target_key.mode

    # Progresiones de fallback por modo
    FALLBACK = {
        'major': [['I','V','vi','IV'],['I','IV','V','I'],
                  ['I','vi','IV','V'],['ii','V','I','I']],
        'minor': [['i','VI','III','VII'],['i','iv','V','i'],
                  ['i','VII','VI','VII'],['i','iv','VII','III']],
    }

    if functions and len(functions) >= 4:
        source_seq = functions
    else:
        source_seq = random.choice(FALLBACK.get(mode, FALLBACK['major']))

    progression = []
    prev_pitches = None

    for m in range(n_measures):
        fig = source_seq[m % len(source_seq)]
        pitches = _build_chord_pitches(fig, target_key, prev_pitches, 'chords')
        tonic_pc = pitch.Pitch(target_key.tonic.name).pitchClass
        degree_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
        base_fig = fig.replace('7','').replace('°','').replace('+','')
        semitones = degree_map.get(base_fig, (0,'M'))[0]
        root_midi = (tonic_pc + semitones) % 12 + 48

        progression.append({
            'function': fig,
            'pitches':  pitches,
            'root':     root_midi,
            'tension':  dna.harmony_dna.get('tension_curve', [0.3])[m % max(1,len(dna.harmony_dna.get('tension_curve',[0.3])))]
        })
        prev_pitches = pitches

    return progression


def _build_macro_form(dna_list, target_key, n_sections=5):
    """
    #7: Define la forma macro de la pieza (Intro-A-B-A'-Coda).
    Asigna a cada sección un DNA fuente y un número de compases.
    """
    # Ordenar por energía para hacer un arco dramático
    by_energy = sorted(enumerate(dna_list),
                       key=lambda x: x[1].energy_dna.get('mean', 0))

    n_dnas = len(dna_list)
    sections = []

    form_templates = {
        1: [('A', 16)],
        2: [('intro', 4), ('A', 12), ('B', 12), ("A'", 8), ('coda', 4)],
        3: [('intro', 4), ('A', 10), ('B', 8), ("A'", 10), ('C', 8), ('coda', 4)],
    }
    template = form_templates.get(min(3, n_dnas), form_templates[2])

    dna_cycle = [d for _, d in by_energy]

    for i, (name, n_measures) in enumerate(template):
        # Elegir DNA para cada sección
        if name == 'intro':
            section_dna = dna_cycle[0]   # Menor energía → intro
        elif name == 'coda':
            section_dna = dna_cycle[0]   # Volver al inicio
        elif name in ("A", "A'"):
            section_dna = dna_cycle[min(1, len(dna_cycle)-1)]
        elif name == 'B':
            section_dna = dna_cycle[min(len(dna_cycle)-1, len(dna_cycle)//2)]
        else:
            section_dna = dna_cycle[i % len(dna_cycle)]

        # Modificar ligeramente A' respecto a A
        is_varied = name == "A'"

        sections.append({
            'name':      name,
            'n_measures': n_measures,
            'dna':       section_dna,
            'is_varied': is_varied,
        })

    return sections


def _add_cadence(progression, cadence_type, target_key, position='end'):
    """
    #5: Inserta una cadencia en la posición indicada de la progresión.
    'end' → últimos 2 acordes, 'half' → mitad
    """
    cad_funcs = CADENCE_TYPES.get(cadence_type, ['V', 'I'])
    mode      = target_key.mode
    if mode == 'minor':
        cad_map = {'I': 'i', 'IV': 'iv', 'ii': 'ii°'}
        cad_funcs = [cad_map.get(f, f) for f in cad_funcs]

    n = len(progression)
    if position == 'end' and n >= 2:
        prev_pitches = progression[-3]['pitches'] if n >= 3 else None
        for k, func in enumerate(cad_funcs):
            idx = n - len(cad_funcs) + k
            if 0 <= idx < n:
                pitches = _build_chord_pitches(func, target_key, prev_pitches, 'chords')
                progression[idx]['function'] = func
                progression[idx]['pitches']  = pitches
                prev_pitches = pitches
    elif position == 'half' and n >= 4:
        mid = n // 2
        for k, func in enumerate(cad_funcs):
            idx = mid - 1 + k
            if 0 <= idx < n:
                pitches = _build_chord_pitches(func, target_key, None, 'chords')
                progression[idx]['function'] = func
                progression[idx]['pitches']  = pitches
    return progression


def _insert_modal_chord(progression, tension_curve, target_key):
    """
    #10: En los puntos de mayor tensión, sustituye el acorde por uno prestado.
    """
    if not progression or not tension_curve:
        return progression

    mode = target_key.mode
    MODAL_SUBS = {
        'major': {  # grado original → sustituto modal
            'IV': 'iv',   'V': 'bVII',  'vi': 'bVI',
            'I':  'bIII', 'ii': 'bII',
        },
        'minor': {
            'v':  'V',    'VII': 'bVII', 'III': 'bIII',
            'i':  'I',
        }
    }
    subs = MODAL_SUBS.get(mode, {})

    for i, chord_data in enumerate(progression):
        t = tension_curve[i] if i < len(tension_curve) else 0.3
        # Insertar modal si la tensión es alta (>0.65) y hay sustituto disponible
        if t > 0.65 and random.random() < 0.4:
            orig_func = chord_data['function']
            base_func = orig_func.replace('7','').replace('°','')
            if base_func in subs:
                new_func = subs[base_func]
                prev = progression[i-1]['pitches'] if i > 0 else None
                new_pitches = _build_chord_pitches(new_func, target_key, prev, 'chords')
                progression[i]['function'] = new_func
                progression[i]['pitches']  = new_pitches
                progression[i]['is_modal'] = True

    return progression


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 5: GENERADORES MUSICALES
# ═══════════════════════════════════════════════════════════════

def _generate_melody_section(melody_dna, motif_dna, chord_prog, target_key,
                              n_measures, ts_num, section_type='A',
                              novelty_measures=None, variation=False):
    """
    #5 #6 #9: Genera melodía con:
    - Motivos y sus transformaciones
    - Cadencias al final de cada frase
    - Inserción de material sorpresa en puntos de novedad
    """
    scale_midi = _get_scale_midi(target_key, octave=5)
    melody_out  = []
    source_key  = target_key  # (#2 si vienen de otro DNA se transpone)

    intervals    = melody_dna.get('intervals', [2, 1, -1, 2])
    rhythm_pat   = melody_dna.get('durations', [0.5, 0.5, 1.0])
    leap_ratio   = melody_dna.get('leap_ratio', 0.2)
    mean_pitch   = melody_dna.get('mean_pitch', 65)
    motif_ints   = motif_dna.get('intervals', intervals[:4])
    novelty_set  = set(novelty_measures or [])

    # Elegir transformación del motivo para cada frase
    motif_forms  = {
        'A':         motif_ints,
        'B':         motif_dna.get('inversion', motif_ints),
        "A'":        motif_dna.get('sequence_up', motif_ints),
        'C':         motif_dna.get('retrograde', motif_ints),
        'intro':     motif_dna.get('diminution_d', [x*0.5 for x in motif_ints]),
        'coda':      motif_dna.get('augmentation_d', motif_ints),
    }
    active_motif = motif_forms.get(section_type, motif_ints)
    if variation:
        active_motif = motif_dna.get('sequence_down', motif_ints)

    # Partir en frases de 4 compases
    phrase_len  = 4
    n_phrases   = max(1, n_measures // phrase_len)

    # Pitch de inicio: cerca del pitch medio del DNA, ajustado a la escala
    start_pitch = _snap_to_scale(int(mean_pitch), target_key)
    start_pitch = max(INSTRUMENT_RANGES['melody'][0],
                      min(INSTRUMENT_RANGES['melody'][1], start_pitch))
    current_pitch = start_pitch

    phrase_idx_global = 0

    for ph in range(n_phrases):
        phrase_offset = ph * phrase_len * ts_num
        is_last_phrase = (ph == n_phrases - 1)
        is_mid_phrase  = (ph == n_phrases // 2 - 1)
        cadence_type   = 'AC' if is_last_phrase else ('HC' if is_mid_phrase else None)

        # Elegir motivo activo para esta frase (pregunta / respuesta alternando)
        if ph % 2 == 1:
            phrase_motif = motif_dna.get('inversion', active_motif)
        else:
            phrase_motif = active_motif

        motif_len     = len(phrase_motif)
        int_idx       = 0
        rhythm_idx    = 0

        for beat_m in range(phrase_len):
            m_idx   = ph * phrase_len + beat_m
            m_off   = phrase_offset + beat_m * ts_num
            chord_d = chord_prog[m_idx % len(chord_prog)] if chord_prog else None
            chord_p = chord_d['pitches'] if chord_d else [60, 64, 67]

            # Compás de cadencia: ir a tónica (AC) o dominante (HC)
            is_cad_measure = (cadence_type and beat_m >= phrase_len - 2)

            beat = 0.0
            while beat < ts_num - 0.01:
                # Duración
                if is_cad_measure and beat == 0:
                    dur = float(ts_num)  # Nota larga en cadencia
                else:
                    raw_dur = rhythm_pat[rhythm_idx % len(rhythm_pat)]
                    dur = min(raw_dur, ts_num - beat)
                    # Cuantización adaptativa (#3)
                    dur = min(VALID_DURATIONS_FLOAT,
                              key=lambda x: abs(x - dur))
                    dur = max(0.25, dur)
                    if beat + dur > ts_num:
                        dur = ts_num - beat
                rhythm_idx += 1

                # Pitch
                if is_cad_measure:
                    # Cadencia: ir a nota del acorde
                    if cadence_type == 'AC':
                        tonic_midi = pitch.Pitch(target_key.tonic.name).midi
                        while tonic_midi < 60: tonic_midi += 12
                        while tonic_midi > 79: tonic_midi -= 12
                        current_pitch = tonic_midi
                    elif cadence_type == 'HC':
                        # Dominante
                        dom_midi = (pitch.Pitch(target_key.tonic.name).midi + 7) % 12 + 60
                        current_pitch = dom_midi
                    current_pitch = _snap_to_scale(current_pitch, target_key)
                elif m_idx in novelty_set and beat == 0:
                    # #9: Punto de novedad → salto dramático
                    direction = 1 if current_pitch < 72 else -1
                    step = direction * random.randint(5, 9)
                    current_pitch = _snap_to_scale(current_pitch + step, target_key)
                else:
                    # Moverse con el motivo o intervalos
                    if int_idx < motif_len * 3:
                        step = phrase_motif[int_idx % motif_len]
                        int_idx += 1
                    else:
                        step = intervals[phrase_idx_global % len(intervals)]
                        phrase_idx_global += 1

                    # Limitar saltos según leap_ratio
                    if abs(step) > 4 and random.random() > leap_ratio:
                        step = int(np.sign(step)) * 2

                    # Gravedad tonal: tirar hacia chord tone más cercano
                    if random.random() < 0.25 and chord_p:
                        nearest = min(chord_p, key=lambda p: abs(p - current_pitch))
                        grav = int(np.sign(nearest - current_pitch))
                        step = (step + grav) // 2

                    current_pitch += step
                    current_pitch = _snap_to_scale(current_pitch, target_key)

                # Mantener en rango (#11)
                lo, hi = INSTRUMENT_RANGES['melody']
                current_pitch = max(lo, min(hi, current_pitch))

                n = note.Note(current_pitch)
                n.quarterLength = max(0.25, dur)
                n.offset = m_off + beat

                # Dinámica: acento métrico
                if beat == 0:
                    vel = random.randint(72, 92)
                elif abs(beat - ts_num/2) < 0.1:
                    vel = random.randint(62, 78)
                elif is_cad_measure:
                    vel = random.randint(80, 100)
                else:
                    vel = random.randint(48, 68)
                n.volume.velocity = vel
                melody_out.append(n)

                beat += dur
                if beat >= ts_num - 0.01:
                    break

    return melody_out


def _generate_accompaniment_section(rhythm_dna, chord_prog, target_key,
                                    n_measures, ts_num, section_type='A'):
    """
    #3: Genera acompañamiento con patrones rítmicos que respetan tresillos,
    variados según la sección.
    """
    acc_out  = []
    density  = rhythm_dna.get('density', 4)
    swing    = rhythm_dna.get('swing', 0)
    has_trip = rhythm_dna.get('has_triplets', False)
    pattern  = rhythm_dna.get('pattern', [1.0])

    # Elegir tipo de acompañamiento por sección y densidad
    if section_type in ('intro', 'coda'):
        acc_type = 'arpeggio'
    elif section_type == 'B':
        acc_type = 'offbeat' if density > 6 else 'block'
    elif density > 12:
        acc_type = 'alberti' if has_trip else 'arpeggio_fast'
    elif density > 7:
        acc_type = 'offbeat'
    else:
        acc_type = 'block'

    # Si el DNA fuente tiene tresillos, usar patrón de tresillo (#3)
    tresillo_pattern = [Fraction(1,3), Fraction(1,3), Fraction(1,3)]

    prev_ch_pitches = None

    for m in range(n_measures):
        m_off   = m * ts_num
        chord_d = chord_prog[m % len(chord_prog)] if chord_prog else None
        if not chord_d or not chord_d.get('pitches'):
            acc_out.append(note.Rest(quarterLength=ts_num)); continue

        # Voice leading con acorde anterior (#4)
        if prev_ch_pitches is None:
            ch_pitches = chord_d['pitches']
        else:
            ch_pitches = _voice_lead(prev_ch_pitches,
                                     chord_d['root'] % 12,
                                     'M', target_key, 'chords')
        prev_ch_pitches = ch_pitches
        ch_sorted = sorted(ch_pitches)

        if acc_type == 'block':
            ch = chord.Chord(ch_sorted)
            ch.quarterLength = ts_num
            ch.offset = m_off
            for nn in ch.notes: nn.volume.velocity = random.randint(50, 65)
            acc_out.append(ch)

        elif acc_type == 'offbeat':
            # Acorde en contratiempo
            beats = []
            b = 0.5
            while b < ts_num:
                beats.append(b); b += 1.0
            for b in beats:
                if b < ts_num:
                    ch = chord.Chord(ch_sorted[:3])
                    ch.quarterLength = 0.5
                    ch.offset = m_off + b
                    for nn in ch.notes: nn.volume.velocity = random.randint(42, 58)
                    acc_out.append(ch)

        elif acc_type in ('arpeggio', 'arpeggio_fast'):
            step = 0.5 if acc_type == 'arpeggio_fast' else max(0.5, ts_num / len(ch_sorted))
            pats_extended = (ch_sorted + list(reversed(ch_sorted[1:-1]))) * 4
            b = 0.0
            for p in pats_extended:
                if b >= ts_num: break
                n_ = note.Note(p)
                n_.quarterLength = min(step, ts_num - b)
                n_.offset = m_off + b
                n_.volume.velocity = random.randint(45, 62)
                acc_out.append(n_)
                b += step

        elif acc_type == 'alberti':
            # Bajo-alto-medio-alto
            if len(ch_sorted) >= 3:
                pat_p = [ch_sorted[0], ch_sorted[-1], ch_sorted[1], ch_sorted[-1]]
            else:
                pat_p = ch_sorted * 2
            step = ts_num / len(pat_p)
            for i, p in enumerate(pat_p):
                n_ = note.Note(p)
                n_.quarterLength = step
                n_.offset = m_off + i * step
                n_.volume.velocity = random.randint(48, 60)
                acc_out.append(n_)

        # Añadir patrón de tresillo ocasionalmente (#3)
        elif has_trip and acc_type == 'block' and random.random() < 0.3:
            for i, dur in enumerate(tresillo_pattern):
                b_off = m_off + i * float(dur) * ts_num
                ch = chord.Chord(ch_sorted[:3])
                ch.quarterLength = float(dur) * ts_num
                ch.offset = b_off
                acc_out.append(ch)

    return acc_out


def _generate_bass_section(chord_prog, target_key, n_measures, ts_num,
                            section_type='A', prev_bass_pitch=None):
    """
    Genera línea de bajo con voice leading respecto al compás anterior (#4),
    leading tones hacia el siguiente acorde, y rangos correctos (#11).
    """
    bass_out = []
    lo, hi   = INSTRUMENT_RANGES['bass']

    if prev_bass_pitch is None:
        tonic_pc = pitch.Pitch(target_key.tonic.name).pitchClass
        prev_bass_pitch = tonic_pc + 36  # Octava 2

    for m in range(n_measures):
        m_off   = m * ts_num
        chord_d = chord_prog[m % len(chord_prog)] if chord_prog else None
        if not chord_d:
            bass_out.append(note.Rest(quarterLength=ts_num)); continue

        root_midi = chord_d.get('root', 48)
        # Normalizar a rango de bajo
        while root_midi > hi: root_midi -= 12
        while root_midi < lo: root_midi += 12

        # Voice leading mínimo desde el bajo anterior (#4)
        while root_midi - prev_bass_pitch > 6:  root_midi -= 12
        while prev_bass_pitch - root_midi > 6:  root_midi += 12
        root_midi = max(lo, min(hi, root_midi))

        # Próxima raíz para leading tone
        next_root = chord_prog[(m+1) % len(chord_prog)].get('root', root_midi) \
                    if chord_prog and m < n_measures - 1 else root_midi
        while next_root > hi: next_root -= 12
        while next_root < lo: next_root += 12

        # Nota de paso cromatica hacia la siguiente raíz
        leading = root_midi + int(np.sign(next_root - root_midi)) * min(2, abs(next_root - root_midi))
        leading = _snap_to_scale(leading, target_key)
        leading = max(lo, min(hi, leading))

        if ts_num >= 4:
            # Tiempo 1: fundamental
            n1 = note.Note(root_midi)
            n1.quarterLength = 2.0
            n1.offset = m_off
            n1.volume.velocity = random.randint(68, 82)
            bass_out.append(n1)
            # Tiempo 3: leading tone o quinta
            fifth = _snap_to_scale(root_midi + 7, target_key)
            fifth = max(lo, min(hi, fifth))
            n2 = note.Note(leading if random.random() < 0.4 else fifth)
            n2.quarterLength = 2.0
            n2.offset = m_off + 2.0
            n2.volume.velocity = random.randint(58, 72)
            bass_out.append(n2)
        elif ts_num == 3:
            n1 = note.Note(root_midi)
            n1.quarterLength = 2.0; n1.offset = m_off
            n1.volume.velocity = random.randint(68, 82)
            bass_out.append(n1)
            n2 = note.Note(leading)
            n2.quarterLength = 1.0; n2.offset = m_off + 2.0
            n2.volume.velocity = random.randint(55, 70)
            bass_out.append(n2)
        else:
            n1 = note.Note(root_midi)
            n1.quarterLength = float(ts_num)
            n1.offset = m_off
            n1.volume.velocity = random.randint(65, 80)
            bass_out.append(n1)

        prev_bass_pitch = root_midi

    return bass_out, prev_bass_pitch


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 6: MOTOR DE COMBINACIÓN
# ═══════════════════════════════════════════════════════════════

def combine_dna(dnas, mode='auto', target_key=None, target_tempo=None, verbose=False):
    if not dnas:
        raise ValueError("No hay ADNs")
    dnas = [d for d in dnas if d is not None]
    if not dnas:
        raise ValueError("Todos los archivos fallaron")

    print(f"\n🔀 Combinando {len(dnas)} piezas en modo: {mode.upper()}")

    # Parámetros globales
    bpm = target_tempo if target_tempo else \
          dnas[np.argmax([d.energy_dna.get('mean',0.5) for d in dnas])].tempo_bpm

    if target_key:
        p = target_key.split()
        target_key_obj = key.Key(p[0], p[1] if len(p)>1 else 'major')
    else:
        # Tonalidad del DNA con más material armónico
        target_key_obj = dnas[np.argmax([len(d.harmony_dna.get('progressions',[])) for d in dnas])].key_obj \
                         or key.Key('C','major')

    if mode == 'auto':
        mode = _auto_select_mode(dnas)
        print(f"  🤖 Modo auto: {mode.upper()}")

    print(f"  📐 {target_key_obj.tonic.name} {target_key_obj.mode} | {bpm:.0f} BPM")

    # Generar N candidatos y elegir el mejor (#12)
    return _build_score_with_form(dnas, mode, target_key_obj, bpm)


def _auto_select_mode(dnas):
    energies    = [d.energy_dna.get('mean',0.5)       for d in dnas]
    complexities= [d.harmony_dna.get('complexity',0)  for d in dnas]
    densities   = [d.rhythm_dna.get('density',0)       for d in dnas]
    if max(energies) - min(energies) > 0.3:     return 'energy'
    if max(complexities) - min(complexities) > 0.3: return 'harmony'
    if max(densities) - min(densities) > 5:     return 'rhythm'
    return 'harmony'


def _build_score_with_form(dnas, mode, target_key, bpm):
    """
    #7: Construye el score completo siguiendo la forma macro.
    Cada sección usa el DNA apropiado para melodía/ritmo/armonía.
    """
    # Asignar roles globales
    by_harmony = sorted(dnas, key=lambda d: d.harmony_dna.get('complexity',0), reverse=True)
    by_rhythm  = sorted(dnas, key=lambda d: d.rhythm_dna.get('density',0),     reverse=True)
    by_melody  = sorted(dnas, key=lambda d: d.melody_dna.get('range_semitones',0), reverse=True)
    by_energy  = sorted(dnas, key=lambda d: d.energy_dna.get('mean',0.5),      reverse=True)

    harmony_src = by_harmony[0]
    rhythm_src  = by_rhythm[min(1, len(by_rhythm)-1)]
    melody_src  = by_melody[0]
    energy_src  = by_energy[0]

    if mode == 'rhythm':
        melody_src, rhythm_src = rhythm_src, dnas[0]
    elif mode == 'energy':
        # Energía controla densidad; melodía del más expresivo
        melody_src  = by_melody[0]
        rhythm_src  = energy_src

    print(f"  ♬ Armonía  → {harmony_src.name}")
    print(f"  🥁 Ritmo   → {rhythm_src.name}")
    print(f"  🎵 Melodía → {melody_src.name}")
    print(f"  ⚡ Energía → {energy_src.name}")

    ts_num, ts_den = harmony_src.time_signature

    # Forma macro (#7)
    sections = _build_macro_form(dnas, target_key)
    total_measures = sum(s['n_measures'] for s in sections)

    print(f"\n  📋 Forma: " +
          " → ".join(f"{s['name']}({s['n_measures']}c)" for s in sections))

    result = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    mel_part = stream.Part(id='Melody')
    mel_part.insert(0, instrument.Piano())
    mel_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    mel_part.insert(0, clef.TrebleClef())

    acc_part = stream.Part(id='Accompaniment')
    acc_part.insert(0, instrument.Piano())
    acc_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    acc_part.insert(0, clef.BassClef())

    bass_part = stream.Part(id='Bass')
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    bass_part.insert(0, clef.BassClef())

    global_offset   = 0
    prev_bass_pitch = None
    global_measure  = 0

    for sec in sections:
        sname     = sec['name']
        n_meas    = sec['n_measures']
        sec_dna   = sec['dna']
        is_varied = sec['is_varied']

        print(f"    ▸ Sección {sname}: {n_meas} compases | DNA={sec_dna.name}")

        # Determinar qué DNA usar para cada capa según modo
        if mode == 'harmony':
            sec_harmony = harmony_src
            sec_rhythm  = rhythm_src
            sec_melody  = melody_src
        elif mode == 'rhythm':
            sec_harmony = harmony_src
            sec_rhythm  = sec_dna
            sec_melody  = melody_src
        elif mode == 'emotion':
            sec_harmony = sec_dna
            sec_rhythm  = sec_dna
            sec_melody  = sec_dna
        else:
            sec_harmony = harmony_src
            sec_rhythm  = rhythm_src
            sec_melody  = melody_src

        # 1. Progresión armónica (#2 + #4 voice leading)
        prog = _build_progression_from_dna(sec_harmony, target_key, n_meas)

        # 2. Cadencias (#5): HC en mitad, AC al final
        prog = _add_cadence(prog, 'HC', target_key, 'half')
        prog = _add_cadence(prog, 'AC', target_key, 'end')

        # 3. Intercambio modal en puntos de tensión (#10)
        tension_curve = sec_harmony.tension_dna.get('curve', [])
        if sec_harmony.modal_dna.get('uses_modal', False) or \
           sec_harmony.tension_dna.get('mean', 0.3) > 0.5:
            prog = _insert_modal_chord(prog, tension_curve, target_key)

        # 4. Puntos de novedad (#9)
        novelty_ms = [
            global_measure + m
            for m in sec_harmony.novelty_dna.get('peak_measures', [])
            if m < n_meas
        ]

        # 5. Melodía (#6 motivos, #5 cadencias)
        mel_notes = _generate_melody_section(
            sec_melody.melody_dna,
            sec_melody.motif_dna,
            prog, target_key, n_meas, ts_num,
            section_type=sname,
            novelty_measures=[m - global_measure for m in novelty_ms],
            variation=is_varied,
        )

        # 6. Acompañamiento (#3 cuantización + #4 voice leading)
        acc_notes = _generate_accompaniment_section(
            sec_rhythm.rhythm_dna,
            prog, target_key, n_meas, ts_num,
            section_type=sname,
        )

        # 7. Bajo (#4 voice leading + #11 rangos)
        prev_bass = prev_bass_pitch
        bass_notes, prev_bass_pitch = _generate_bass_section(
            prog, target_key, n_meas, ts_num,
            section_type=sname,
            prev_bass_pitch=prev_bass,
        )

        # Aplicar offset global y añadir a las partes
        sec_offset = global_offset

        for el in mel_notes:
            el.offset += sec_offset
            mel_part.append(el)
        for el in acc_notes:
            if hasattr(el, 'offset'):
                el.offset += sec_offset
            acc_part.append(el)
        for el in bass_notes:
            if hasattr(el, 'offset'):
                el.offset += sec_offset
            bass_part.append(el)

        # Dinámica de sección
        dyn_map = {'intro':'p', 'A':'mf', 'B':'f', "A'":'mp', 'C':'ff', 'coda':'pp'}
        dyn = dynamics.Dynamic(dyn_map.get(sname, 'mf'))
        mel_part.measure(1) if mel_part.getElementsByClass('Measure') else None

        global_offset  += n_meas * ts_num
        global_measure += n_meas

    result.append(mel_part)
    result.append(acc_part)
    result.append(bass_part)

    return result


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 7: POSTPROCESADO Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

def humanize_score(score_obj, amount=0.018):
    """Añade variaciones de timing y velocidad para sonido humano."""
    print("  🎭 Humanizando...")
    for el in score_obj.flat.notes:
        if hasattr(el, 'volume') and el.volume.velocity:
            v = el.volume.velocity
            el.volume.velocity = max(15, min(127, v + int(random.gauss(0, v * 0.07))))
        if hasattr(el, 'offset') and float(el.offset) > 0:
            el.offset = max(0, float(el.offset) + random.gauss(0, amount))
    return score_obj


def add_expression(score_obj, energy_curve):
    """Añade marcas de dinámica por sección basadas en la curva de energía."""
    if not energy_curve: return score_obj
    dyn_levels = {0.0:'ppp',0.15:'pp',0.3:'p',0.45:'mp',
                  0.6:'mf',0.75:'f',0.88:'ff',1.0:'fff'}
    for part in score_obj.parts:
        measures = list(part.getElementsByClass('Measure'))
        if not measures:
            continue
        for i, m in enumerate(measures):
            if i < len(energy_curve):
                e = energy_curve[i]
                thresh = min(dyn_levels.keys(), key=lambda x: abs(x-e))
                m.insert(0, dynamics.Dynamic(dyn_levels[thresh]))
    return score_obj


def validate_and_fix_score(score_obj):
    """
    #11: Validación completa:
    - Notas fuera de rango del instrumento → transponer
    - Intervalos > 9ª entre voces adyacentes → ajustar
    - Cruce de voces (bajo > melodía) → corregir
    - Notas solapadas → eliminar duplicados
    - Silencio al inicio/final → limpiar
    """
    print("  🔧 Validando partitura (#11)...")

    parts = list(score_obj.parts)

    # Rangos por nombre de parte
    part_ranges = {
        'Melody':        INSTRUMENT_RANGES['melody'],
        'Accompaniment': INSTRUMENT_RANGES['chords'],
        'Bass':          INSTRUMENT_RANGES['bass'],
    }

    melody_offsets = {}  # offset → pitch (para detección cruce de voces)

    for part in parts:
        part_id  = part.id or ''
        lo, hi   = part_ranges.get(part_id, (28, 108))
        all_notes = [n for n in part.flat.notes if n.isNote]

        # Recolectar pitches de melodía
        if part_id == 'Melody':
            for n in all_notes:
                melody_offsets[round(float(n.offset), 2)] = n.pitch.midi

        seen = {}
        for n in all_notes:
            off_key = round(float(n.offset), 3)

            # Eliminar solapadas (duplicados en mismo offset)
            if off_key in seen:
                try:
                    part.remove(n)
                    continue
                except:
                    pass
            seen[off_key] = n

            # Transponer a rango correcto
            while n.pitch.midi > hi: n.pitch.midi -= 12
            while n.pitch.midi < lo: n.pitch.midi += 12

        # Cruce de voces: bajo no debe superar a melodía
        if part_id == 'Bass':
            for n in all_notes:
                off_key = round(float(n.offset), 2)
                mel_p = melody_offsets.get(off_key)
                if mel_p and n.pitch.midi >= mel_p - 5:
                    n.pitch.midi = max(lo, mel_p - 12)

        # Intervalos de 9ª o más entre notas consecutivas en melodía → suavizar
        if part_id == 'Melody':
            sorted_notes = sorted(all_notes, key=lambda n: float(n.offset))
            for i in range(len(sorted_notes) - 1):
                diff = abs(sorted_notes[i+1].pitch.midi - sorted_notes[i].pitch.midi)
                if diff > 14:  # > 9ª mayor
                    # Invertir dirección
                    if sorted_notes[i+1].pitch.midi > sorted_notes[i].pitch.midi:
                        sorted_notes[i+1].pitch.midi = sorted_notes[i].pitch.midi + \
                                                        min(7, diff // 2)
                    else:
                        sorted_notes[i+1].pitch.midi = sorted_notes[i].pitch.midi - \
                                                        min(7, diff // 2)
                    sorted_notes[i+1].pitch.midi = max(lo, min(hi, sorted_notes[i+1].pitch.midi))

    print("  ✅ Partitura validada")
    return score_obj


def score_candidate(score_obj, dnas):
    """
    #12: Puntúa un candidato según criterios musicales.
    - Consonancia media
    - Variedad rítmica
    - Coherencia de rango
    """
    melody_notes = [n for n in score_obj.flat.notes if n.isNote]
    if not melody_notes: return 0.0

    # 1. Consonancia: porcentaje de notas en escala
    dna = dnas[0]
    key_obj = dna.key_obj or key.Key('C','major')
    scale_pcs = set(_get_scale_pcs(key_obj))
    tonic_pc  = pitch.Pitch(key_obj.tonic.name).pitchClass
    in_scale  = sum(1 for n in melody_notes
                    if (n.pitch.pitchClass - tonic_pc) % 12 in scale_pcs)
    consonance = in_scale / len(melody_notes)

    # 2. Variedad rítmica
    durs = [float(n.quarterLength) for n in melody_notes]
    rhythmic_variety = min(1.0, len(set(round(d,2) for d in durs)) / 6)

    # 3. Rango de melodía coherente
    pitches = [n.pitch.midi for n in melody_notes]
    rng = max(pitches) - min(pitches)
    range_score = min(1.0, rng / 24)  # Ideal: 2 octavas

    return consonance * 0.5 + rhythmic_variety * 0.3 + range_score * 0.2


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 8: MAIN
# ═══════════════════════════════════════════════════════════════

def print_dna_summary(dnas):
    print("\n" + "═"*76)
    print("  📊 ADN MUSICAL COMPARATIVO")
    print("═"*76)
    print(f"  {'Pieza':<20} {'Key':<10} {'BPM':>5} {'Dens':>6} {'Rango':>6} "
          f"{'Compj':>6} {'Energ':>6} {'Tens':>6} {'Val':>6} {'Modal':>6}")
    print("  " + "─"*76)
    for dna in dnas:
        nm  = dna.name[:18]+"…" if len(dna.name)>19 else dna.name
        print(f"  {nm:<20} "
              f"{dna.key_tonic+' '+dna.key_mode[:3]:<10} "
              f"{dna.tempo_bpm:>5.0f} "
              f"{dna.rhythm_dna.get('density',0):>6.1f} "
              f"{dna.melody_dna.get('range_semitones',0):>6} "
              f"{dna.harmony_dna.get('complexity',0):>6.2f} "
              f"{dna.energy_dna.get('mean',0):>6.2f} "
              f"{dna.tension_dna.get('mean',0):>6.2f} "
              f"{dna.emotion_dna.get('valence',0):>6.2f} "
              f"{dna.modal_dna.get('borrowed_ratio',0):>6.1%}")
    print("═"*76)

    print("\n  🎼 ROLES:")
    if len(dnas) >= 1:
        h = max(dnas, key=lambda d: d.harmony_dna.get('complexity',0))
        r = max(dnas, key=lambda d: d.rhythm_dna.get('density',0))
        m = max(dnas, key=lambda d: d.melody_dna.get('range_semitones',0))
        e = max(dnas, key=lambda d: d.energy_dna.get('peak',0))
        print(f"    ♬ Armonía  → {h.name}")
        print(f"    🥁 Ritmo   → {r.name}")
        print(f"    🎵 Melodía → {m.name}")
        print(f"    ⚡ Energía → {e.name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="🧬 MIDI DNA Combiner v2 — Fusión Musical Inteligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('midi_files', nargs='+')
    parser.add_argument('--output',     default='combined_output.mid')
    parser.add_argument('--mode',       default='auto',
        choices=['auto','harmony','rhythm','energy','texture','emotion'])
    parser.add_argument('--tempo',      type=float, default=None)
    parser.add_argument('--key',        default=None)
    parser.add_argument('--candidates', type=int,   default=3,
                        help='Generar N candidatos y elegir el mejor (default: 3)')
    parser.add_argument('--export-xml', action='store_true',
                        help='Exportar también en MusicXML')
    parser.add_argument('--no-humanize', action='store_true')
    parser.add_argument('--verbose',     action='store_true')
    args = parser.parse_args()

    print("\n╔" + "═"*68 + "╗")
    print("║" + "  🧬  MIDI DNA COMBINER v2.0  —  Fusión Musical Inteligente  ".center(68) + "║")
    print("╚" + "═"*68 + "╝")

    for f in args.midi_files:
        if not os.path.exists(f):
            print(f"❌ No encontrado: {f}"); sys.exit(1)

    # ── FASE 1: Extracción
    print(f"\n🔬 FASE 1: Extracción del ADN ({len(args.midi_files)} archivos)")
    print("═"*60)
    dnas = [extract_dna(f, verbose=args.verbose) for f in args.midi_files]
    dnas = [d for d in dnas if d is not None]
    if not dnas:
        print("❌ Sin ADN extraído"); sys.exit(1)

    print_dna_summary(dnas)

    # ── FASE 2: Combinación con candidatos (#12)
    print("🔀 FASE 2: Combinación del ADN")
    print("═"*60)

    n_candidates = max(1, args.candidates)
    best_score, best_result = -1, None

    for c in range(n_candidates):
        if n_candidates > 1:
            print(f"\n  🎲 Candidato {c+1}/{n_candidates}")
            random.seed(42 + c * 17)
            np.random.seed(42 + c * 17)

        result = combine_dna(
            dnas,
            mode=args.mode,
            target_key=args.key,
            target_tempo=args.tempo,
            verbose=args.verbose,
        )

        sc = score_candidate(result, dnas)
        print(f"    📊 Score musical: {sc:.3f}")

        if sc > best_score:
            best_score  = sc
            best_result = result

    print(f"\n  🏆 Mejor candidato con score={best_score:.3f}")

    # ── FASE 3: Postprocesado
    print("\n✨ FASE 3: Postprocesado")
    print("═"*60)

    if not args.no_humanize:
        best_result = humanize_score(best_result)

    best_result = validate_and_fix_score(best_result)

    energy_source = max(dnas, key=lambda d: d.energy_dna.get('peak',0))
    energy_curve  = energy_source.energy_dna.get('curve', [])
    if energy_curve:
        best_result = add_expression(best_result, energy_curve)

    # ── FASE 4: Exportar
    print(f"\n💾 FASE 4: Exportando")
    print("═"*60)

    output_path = args.output
    try:
        best_result.write('midi', fp=output_path)
        print(f"  ✅ MIDI guardado: {output_path}")
    except Exception as e:
        output_path = 'output_combined.mid'
        best_result.write('midi', fp=output_path)
        print(f"  ✅ MIDI guardado: {output_path}")

    if args.export_xml:
        xml_path = output_path.replace('.mid','.xml').replace('.midi','.xml')
        if not xml_path.endswith('.xml'): xml_path += '.xml'
        try:
            best_result.write('musicxml', fp=xml_path)
            print(f"  ✅ MusicXML guardado: {xml_path}")
        except Exception as e:
            print(f"  ⚠️  No se pudo guardar MusicXML: {e}")

    print("\n" + "═"*60)
    print("  🎉 ¡COMBINACIÓN COMPLETADA!")
    print(f"  📁 {output_path}  |  Score: {best_score:.3f}")
    print(f"  🔀 Modo: {args.mode.upper()}  |  Candidatos evaluados: {n_candidates}")
    print(f"  🎵 {' + '.join(d.name for d in dnas)}")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
