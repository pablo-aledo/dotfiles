"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MIDI DNA COMBINER  v3.0                                 ║
║   Extrae el "ADN" musical de varios MIDIs y los fusiona en una nueva pieza   ║
║                                                                              ║
║  NUEVAS MEJORAS v3 (calidad musical):                                        ║
║   #A  Mosaic splicing: usa fragmentos reales del MIDI fuente                 ║
║   #B  Markov de 2º orden para melodía (intervalos × duraciones)             ║
║   #C  Ornamentación idiomática (mordentes, apoyaturas, notas de paso)       ║
║   #D  Articulación y fraseo (slurs, staccato, acentos métricos)             ║
║   #E  Contrapunto de 2 voces en acompañamiento                              ║
║   #F  Detección y réplica del patrón de acompañamiento real                 ║
║   #G  Modulación temporal en sección B (relativo / dominante)               ║
║   #H  Groove real: timing map extraído del MIDI fuente                      ║
║                                                                              ║
║  Mantiene todas las mejoras de v2 (#1-#11)                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    python midi_dna_combiner_v3.py file1.mid file2.mid [file3.mid] [opciones]

OPCIONES:
    --output OUTPUT      Archivo de salida (default: combined_output.mid)
    --mode MODE          auto | harmony | rhythm | energy | emotion | mosaic
    --tempo TEMPO        BPM del resultado (default: auto)
    --key KEY            Tonalidad (ej: "C major", "A minor") (default: auto)
    --export-xml         Exportar también en MusicXML
    --candidates N       Generar N candidatos y elegir el mejor (default: 3)
    --no-humanize        Desactivar humanización
    --verbose            Análisis detallado del ADN

MODOS:
    auto      El sistema elige el mejor modo según las características
    harmony   Armonía de A + ritmo de B + melodía de C (Markov)
    rhythm    Patrón rítmico real de A aplicado a melodía Markov de B
    energy    Arco de energía de A controla densidad de materiales de B/C
    emotion   Arco emocional progresivo entre los DNAs
    mosaic    Collage de fragmentos reales transpuestos (más fiel al original)

EJEMPLOS:
    python midi_dna_combiner_v3.py bach.mid jazz.mid --mode mosaic --verbose
    python midi_dna_combiner_v3.py a.mid b.mid c.mid --mode harmony --candidates 5
    python midi_dna_combiner_v3.py a.mid b.mid --mode emotion --key "A minor"
    python midi_dna_combiner_v3.py a.mid b.mid --mode rhythm --export-xml

DEPENDENCIAS:
    pip install music21 numpy mido
"""

import sys, os, argparse, copy, random, math
from collections import defaultdict, Counter
from fractions import Fraction
import numpy as np

try:
    from music21 import (
        stream, note, chord, meter, tempo, key, instrument,
        interval as m21interval, pitch, duration, harmony,
        roman, converter, scale, dynamics, expressions,
        analysis, clef, articulations, spanner, tie
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

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ═══════════════════════════════════════════════════════════════

TONNETZ_DISTANCES = {0:0.0,1:3.0,2:2.0,3:1.0,4:1.0,5:0.5,6:3.5,7:0.5,8:1.0,9:1.0,10:2.0,11:3.0}
LERDAHL_DISSONANCE = {0:0,1:6,2:5,3:3,4:2,5:1,6:7}

MAJOR_SCALE_DEGREES = {
    'I':(0,'M'),'ii':(2,'m'),'iii':(4,'m'),'IV':(5,'M'),'V':(7,'M'),'V7':(7,'M7'),
    'vi':(9,'m'),'vii°':(11,'d'),
    'bVII':(10,'M'),'bIII':(3,'M'),'bVI':(8,'M'),'iv':(5,'m'),'II':(2,'M'),'bII':(1,'M'),
}
MINOR_SCALE_DEGREES = {
    'i':(0,'m'),'ii°':(2,'d'),'III':(3,'M'),'iv':(5,'m'),'V':(7,'M'),'V7':(7,'M7'),
    'VI':(8,'M'),'VII':(10,'M'),'vii°':(11,'d'),
    'II':(2,'M'),'IV':(5,'M'),'bVII':(10,'M'),'I':(0,'M'),
}

INSTRUMENT_RANGES = {
    'melody':  (60,84),
    'tenor':   (48,69),
    'bass':    (28,52),
    'chords':  (48,76),
}

CADENCE_TYPES = {
    'AC': ['V','I'], 'HC': ['I','V'], 'DC': ['V','vi'], 'IAC': ['V','I'], 'PC': ['IV','I'],
}

VALID_DURATIONS_FLOAT = [
    1/6, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0
]

# Estilos para ornamentación idiomática (#C)
ORNAMENTATION_STYLES = {
    'baroque':    {'mordent':0.25, 'trill':0.10, 'passing':0.35, 'neighbor':0.20, 'appoggiatura':0.10},
    'classical':  {'mordent':0.10, 'trill':0.08, 'passing':0.30, 'neighbor':0.25, 'appoggiatura':0.15},
    'romantic':   {'mordent':0.05, 'trill':0.05, 'passing':0.25, 'neighbor':0.30, 'appoggiatura':0.25},
    'jazz':       {'mordent':0.02, 'trill':0.02, 'passing':0.40, 'neighbor':0.30, 'appoggiatura':0.05},
    'generic':    {'mordent':0.08, 'trill':0.05, 'passing':0.30, 'neighbor':0.25, 'appoggiatura':0.12},
}

# ═══════════════════════════════════════════════════════════════
#  CLASE MusicFragment (#A mosaic splicing)
# ═══════════════════════════════════════════════════════════════

class MusicFragment:
    """
    Un fragmento real de 1-2 compases extraído del MIDI fuente,
    anotado con sus características para el mosaic splicing.
    """
    def __init__(self, notes, offset, duration_ql, measure_idx,
                 tension=0.3, energy=0.5, function='I',
                 key_obj=None, source_name=''):
        self.notes        = notes          # lista de note.Note con offsets relativos
        self.offset       = offset         # offset absoluto en el original
        self.duration_ql  = duration_ql    # duración total en quarter lengths
        self.measure_idx  = measure_idx
        self.tension      = tension
        self.energy       = energy
        self.function     = function       # función armónica dominante
        self.key_obj      = key_obj
        self.source_name  = source_name

        # Características del fragmento
        self.pitch_mean   = np.mean([n.pitch.midi for n in notes if n.isNote]) if notes else 60
        self.pitch_range  = (max([n.pitch.midi for n in notes if n.isNote], default=60) -
                             min([n.pitch.midi for n in notes if n.isNote], default=60))
        self.n_notes      = len([n for n in notes if n.isNote])
        self.intervals    = []
        pitches = [n.pitch.midi for n in notes if n.isNote]
        if len(pitches) > 1:
            self.intervals = [pitches[i+1]-pitches[i] for i in range(len(pitches)-1)]

    def transpose_to(self, target_key):
        """
        #A + #B: Transpone el fragmento a la tonalidad objetivo
        preservando las relaciones interválicas.
        """
        if not self.key_obj or not target_key:
            return self._clone_notes()

        src_tonic = pitch.Pitch(self.key_obj.tonic.name).pitchClass
        tgt_tonic = pitch.Pitch(target_key.tonic.name).pitchClass
        semitone_shift = (tgt_tonic - src_tonic)

        # Ajustar para mínimo movimiento (máximo ±6 semitonos)
        if semitone_shift > 6:  semitone_shift -= 12
        if semitone_shift < -6: semitone_shift += 12

        transposed = []
        for n in self.notes:
            if n.isNote:
                new_n = copy.deepcopy(n)
                new_midi = n.pitch.midi + semitone_shift
                # Snap a la escala destino
                new_midi = _snap_to_scale(new_midi, target_key)
                new_n.pitch.midi = max(21, min(108, new_midi))
                transposed.append(new_n)
            else:
                transposed.append(copy.deepcopy(n))
        return transposed

    def _clone_notes(self):
        return [copy.deepcopy(n) for n in self.notes]


# ═══════════════════════════════════════════════════════════════
#  CLASE MarkovMelody (#B)
# ═══════════════════════════════════════════════════════════════

class MarkovMelody:
    """
    Cadena de Markov de 2º orden sobre (intervalo, duración_cuantizada).
    Captura el estilo idiomático del MIDI fuente.
    """
    def __init__(self, order=2):
        self.order      = order
        self.transitions = defaultdict(Counter)  # estado_n → Counter de estado_n+1
        self.start_states = Counter()

    def train(self, melody_dna):
        """Entrena sobre los intervalos y duraciones del DNA melódico."""
        intervals = melody_dna.get('intervals', [])
        durations = melody_dna.get('durations', [])
        if not intervals:
            return

        # Cuantizar duraciones a valores discretos
        def q_dur(d):
            buckets = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            return min(buckets, key=lambda x: abs(x - d))

        # Crear estados: (intervalo_cuantizado, duración_cuantizada)
        def q_int(i):
            # Redondear intervalos a semitonos enteros, clampeados a ±12
            return max(-12, min(12, round(i)))

        n = min(len(intervals), len(durations) - 1)
        if n < self.order + 1:
            # Dataset pequeño: reducir orden
            self.order = max(1, n - 1)

        states = [(q_int(intervals[i]), q_dur(durations[i])) for i in range(n)]

        if len(states) < 2:
            return

        # Registrar estados iniciales
        for i in range(min(4, len(states))):
            self.start_states[states[i]] += 1

        # Entrenar transiciones de orden k
        for i in range(len(states) - self.order):
            context = tuple(states[i:i + self.order])
            next_st = states[i + self.order]
            self.transitions[context][next_st] += 1

        # Fallback: también entrenar orden 1
        if self.order > 1:
            for i in range(len(states) - 1):
                ctx1 = (states[i],)
                self.transitions[ctx1][states[i+1]] += 1

    def generate(self, n_steps, start_pitch, key_obj, seed=None):
        """
        Genera una secuencia de (pitch_midi, duration) usando la cadena.
        Garantiza que todas las notas estén en la escala.
        """
        if seed is not None:
            random.seed(seed)

        if not self.transitions:
            # Sin datos: generar escala simple
            sc = _get_scale_midi(key_obj, octave=5)
            result = []
            p = start_pitch
            for _ in range(n_steps):
                p = _snap_to_scale(p + random.choice([-2,-1,0,1,2]), key_obj)
                p = max(60, min(84, p))
                result.append((p, random.choice([0.5, 0.5, 1.0])))
            return result

        # Elegir estado inicial
        if self.start_states:
            starts = list(self.start_states.keys())
            weights = list(self.start_states.values())
            total = sum(weights)
            probs = [w/total for w in weights]
            idx = np.random.choice(len(starts), p=probs)
            history = [starts[idx]]
        else:
            history = [random.choice(list(
                set(k[0] for k in self.transitions.keys() if k)
            ))]

        result = [(start_pitch, history[0][1])]
        current_pitch = start_pitch

        lo, hi = INSTRUMENT_RANGES['melody']

        for step in range(1, n_steps):
            # Buscar contexto de mayor orden disponible
            next_state = None
            for ord_try in range(min(self.order, len(history)), 0, -1):
                ctx = tuple(history[-ord_try:])
                if ctx in self.transitions and self.transitions[ctx]:
                    candidates = list(self.transitions[ctx].keys())
                    weights = list(self.transitions[ctx].values())
                    total = sum(weights)
                    probs = [w/total for w in weights]
                    idx = np.random.choice(len(candidates), p=probs)
                    next_state = candidates[idx]
                    break

            if next_state is None:
                # Sin transición: movimiento pequeño aleatorio
                next_state = (random.choice([-2,-1,0,1,2]), random.choice([0.5,1.0]))

            interval_step, dur = next_state

            # Gravedad tonal: 30% de probabilidad de corregir hacia escala
            new_pitch = current_pitch + interval_step
            if random.random() < 0.3:
                new_pitch = _snap_to_scale(new_pitch, key_obj)

            # Mantener en rango vocal/melódico
            if new_pitch > hi:
                new_pitch -= 12
            if new_pitch < lo:
                new_pitch += 12
            new_pitch = max(lo, min(hi, new_pitch))
            new_pitch = _snap_to_scale(new_pitch, key_obj)

            result.append((new_pitch, dur))
            current_pitch = new_pitch
            history.append(next_state)
            if len(history) > self.order + 1:
                history.pop(0)

        return result


# ═══════════════════════════════════════════════════════════════
#  CLASE GrooveMap (#H)
# ═══════════════════════════════════════════════════════════════

class GrooveMap:
    """
    Extrae el timing map real del MIDI fuente:
    para cada posición métrica (beat subdivisions),
    guarda la desviación media respecto al grid perfecto.
    """
    def __init__(self, resolution=8):
        self.resolution = resolution  # subdivisiones por beat
        self.timing_offsets  = defaultdict(list)  # posición → [desviaciones]
        self.velocity_map    = defaultdict(list)  # posición → [velocidades]
        self.trained         = False

    def train(self, score, ts_num=4):
        """Extrae el groove del score."""
        all_notes = [n for n in score.flat.notes if n.isNote]
        if len(all_notes) < 8:
            return

        beat_dur = 1.0  # quarter note = 1 beat
        grid_step = beat_dur / self.resolution

        for n in all_notes:
            off = float(n.offset)
            beat_pos = off % ts_num
            grid_pos = round(beat_pos / grid_step)
            grid_ideal = grid_pos * grid_step
            deviation = beat_pos - grid_ideal
            # Solo guardar desviaciones pequeñas (no errores de cuantización)
            if abs(deviation) < 0.15:
                key_pos = grid_pos % (ts_num * self.resolution)
                self.timing_offsets[key_pos].append(deviation)
                vel = n.volume.velocity if hasattr(n,'volume') and n.volume.velocity else 64
                self.velocity_map[key_pos].append(vel)

        self.trained = len(self.timing_offsets) > 0

    def get_offset(self, beat_position, ts_num=4):
        """Devuelve la desviación de timing para una posición métrica."""
        if not self.trained:
            return 0.0
        beat_dur = 1.0
        grid_step = beat_dur / self.resolution
        grid_pos = round((beat_position % ts_num) / grid_step)
        key_pos = grid_pos % (ts_num * self.resolution)
        offsets = self.timing_offsets.get(key_pos, [0.0])
        return float(np.mean(offsets)) if offsets else 0.0

    def get_velocity(self, beat_position, base_vel=70, ts_num=4):
        """Devuelve la velocidad típica para una posición métrica."""
        if not self.trained:
            return base_vel
        beat_dur = 1.0
        grid_step = beat_dur / self.resolution
        grid_pos = round((beat_position % ts_num) / grid_step)
        key_pos = grid_pos % (ts_num * self.resolution)
        vels = self.velocity_map.get(key_pos, [base_vel])
        return int(np.clip(np.mean(vels), 20, 127)) if vels else base_vel


# ═══════════════════════════════════════════════════════════════
#  CLASE AccompanimentPattern (#F)
# ═══════════════════════════════════════════════════════════════

class AccompanimentPattern:
    """
    Patrón de acompañamiento real extraído del MIDI fuente.
    Guarda la figura rítmica y la distribución de voces.
    """
    def __init__(self):
        self.beat_pattern   = []   # lista de (beat_offset, voice_role, dur)
                                   # voice_role: 'bass','mid','top'
        self.measure_length = 4.0
        self.style          = 'block'
        self.trained        = False

    def train(self, acc_parts, ts_num=4):
        """Extrae el patrón de la(s) parte(s) de acompañamiento."""
        if not acc_parts:
            return

        # Recoger notas de la primera ventana de 2 compases
        window = ts_num * 2
        all_notes = []
        for part in acc_parts[:2]:
            for n in part.flat.notes:
                if float(n.offset) < window:
                    all_notes.append(n)

        if len(all_notes) < 3:
            return

        self.measure_length = float(ts_num)
        pitches_at_beat = defaultdict(list)
        for n in all_notes:
            beat = round(float(n.offset) % ts_num, 3)
            pitches_at_beat[beat].append({
                'midi': n.pitch.midi if n.isNote else 60,
                'dur':  float(n.quarterLength),
                'vel':  n.volume.velocity if hasattr(n,'volume') and n.volume.velocity else 64,
            })

        # Clasificar cada evento como bajo/medio/agudo
        self.beat_pattern = []
        for beat in sorted(pitches_at_beat.keys()):
            events = sorted(pitches_at_beat[beat], key=lambda x: x['midi'])
            for i, ev in enumerate(events):
                if i == 0:
                    role = 'bass'
                elif i == len(events) - 1:
                    role = 'top'
                else:
                    role = 'mid'
                self.beat_pattern.append({
                    'beat': beat,
                    'role': role,
                    'dur':  ev['dur'],
                    'vel_ratio': ev['vel'] / 80.0,
                })

        # Detectar estilo
        beats = [p['beat'] for p in self.beat_pattern]
        on_beat   = sum(1 for b in beats if b % 1.0 < 0.05)
        off_beat  = sum(1 for b in beats if 0.2 < b % 1.0 < 0.8)
        total     = max(1, len(beats))

        if off_beat / total > 0.5:
            self.style = 'syncopated'
        elif on_beat / total > 0.7:
            n_unique = len(set(round(b,1) for b in beats))
            self.style = 'alberti' if n_unique >= 3 else 'block'
        else:
            self.style = 'arpeggio'

        self.trained = True

    def render(self, chord_pitches, measure_offset, ts_num, groove_map=None):
        """
        Renderiza el patrón sobre los pitches de un acorde dado.
        Aplica el groove map si está disponible.
        """
        result = []
        if not self.trained or not chord_pitches:
            # Fallback: acorde en bloque
            ch = chord.Chord(sorted(chord_pitches)[:4])
            ch.quarterLength = ts_num
            ch.offset = measure_offset
            return [ch]

        ch_sorted = sorted(chord_pitches)
        lo_bass, hi_bass = INSTRUMENT_RANGES['bass']
        # Separar bajo y acordes
        bass_midi = ch_sorted[0]
        while bass_midi > hi_bass: bass_midi -= 12
        while bass_midi < lo_bass: bass_midi += 12
        chord_upper = [p for p in ch_sorted if p > bass_midi + 2]
        if not chord_upper:
            chord_upper = ch_sorted[1:] if len(ch_sorted) > 1 else ch_sorted

        for pat in self.beat_pattern:
            beat = pat['beat']
            if beat >= ts_num:
                continue
            role    = pat['role']
            dur     = min(pat['dur'], ts_num - beat)
            vel_r   = pat['vel_ratio']

            # Timing groove (#H)
            groove_dev = groove_map.get_offset(beat, ts_num) if groove_map else 0.0
            actual_beat = max(0, beat + groove_dev)

            if role == 'bass':
                midi = bass_midi
                base_vel = 75
            elif role == 'top' and chord_upper:
                midi = chord_upper[-1]
                base_vel = 60
            else:
                # mid: nota del medio del acorde
                midi = chord_upper[len(chord_upper)//2] if chord_upper else ch_sorted[0]
                base_vel = 55

            vel = int(np.clip(base_vel * vel_r, 20, 110))
            if groove_map:
                vel = groove_map.get_velocity(beat, vel, ts_num)

            n = note.Note(midi)
            n.quarterLength = max(0.25, dur)
            n.offset = measure_offset + actual_beat
            n.volume.velocity = vel
            result.append(n)

        return result if result else [chord.Chord(chord_pitches, quarterLength=ts_num,
                                                   offset=measure_offset)]

# ═══════════════════════════════════════════════════════════════
#  UTILIDADES MUSICALES (heredadas y mejoradas de v2)
# ═══════════════════════════════════════════════════════════════

def _get_scale_pcs(key_obj):
    if key_obj.mode == 'major': return [0,2,4,5,7,9,11]
    else:                       return [0,2,3,5,7,8,10]

def _get_scale_midi(key_obj, octave=4):
    tonic = pitch.Pitch(key_obj.tonic.name)
    tonic.octave = octave
    base = tonic.midi
    pcs  = _get_scale_pcs(key_obj)
    result = []
    for o in range(-2, 5):
        for pc in pcs:
            m = base + pc + o*12
            if 21 <= m <= 108: result.append(m)
    return sorted(set(result))

def _snap_to_scale(midi_pitch, key_obj):
    """Transpone al grado de escala más cercano."""
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    pcs      = [(pc + tonic_pc) % 12 for pc in _get_scale_pcs(key_obj)]
    note_pc  = midi_pitch % 12
    if note_pc in pcs: return midi_pitch
    best, best_d = note_pc, 100
    for pc in pcs:
        d = abs(pc - note_pc); d = min(d, 12-d)
        if d < best_d: best_d = d; best = pc
    diff = best - note_pc
    if diff > 6: diff -= 12
    if diff < -6: diff += 12
    return midi_pitch + diff

def _get_relative_key(key_obj):
    """Devuelve la tonalidad relativa (mayor→relativo menor, menor→relativo mayor)."""
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    if key_obj.mode == 'major':
        rel_pc = (tonic_pc + 9) % 12
        rel_tonic = pitch.Pitch(rel_pc).name
        return key.Key(rel_tonic, 'minor')
    else:
        rel_pc = (tonic_pc + 3) % 12
        rel_tonic = pitch.Pitch(rel_pc).name
        return key.Key(rel_tonic, 'major')

def _get_dominant_key(key_obj):
    """Devuelve la tonalidad de la dominante (+5 semitonos)."""
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    dom_pc   = (tonic_pc + 7) % 12
    dom_tonic = pitch.Pitch(dom_pc).name
    return key.Key(dom_tonic, key_obj.mode)

def _build_chord_pitches(roman_figure, key_obj, prev_pitches=None, register='chords'):
    """Construye pitches de un acorde con voice leading opcional."""
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    mode     = key_obj.mode
    degree_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
    base_fig = roman_figure.replace('7','').replace('°','').replace('+','').replace('9','')
    if base_fig in degree_map:
        semitones, quality = degree_map[base_fig]
        if '7' in roman_figure and quality not in ('d','M7','m7'): quality += '7'
    else:
        semitones, quality = 0, 'M'

    root_pc = (tonic_pc + semitones) % 12
    lo, hi  = INSTRUMENT_RANGES.get(register, (48,76))
    iq = {'M':[0,4,7],'m':[0,3,7],'d':[0,3,6],'A':[0,4,8],
          'M7':[0,4,7,10],'m7':[0,3,7,10],'d7':[0,3,6,9]}
    ints = iq.get(quality, [0,4,7])

    if prev_pitches:
        return _voice_lead(prev_pitches, root_pc, quality, key_obj, register)

    base = root_pc + 48
    while base < lo: base += 12
    result = [base + i for i in ints if lo-2 <= base+i <= hi+2]
    return result or [base]

def _voice_lead(prev_chord_pitches, root_pc, quality, key_obj, register='chords'):
    lo, hi = INSTRUMENT_RANGES.get(register, (48,76))
    iq = {'M':[0,4,7],'m':[0,3,7],'d':[0,3,6],'A':[0,4,8],
          'M7':[0,4,7,10],'m7':[0,3,7,10],'d7':[0,3,6,9]}
    ints = iq.get(quality, [0,4,7])
    chord_midis = []
    for o in range(2,7):
        for i in ints:
            m = root_pc + i + o*12
            if lo-2 <= m <= hi+2: chord_midis.append(m)
    if not chord_midis:
        base = root_pc + 48
        while base < lo: base += 12
        return [base + i for i in ints if base+i <= hi]
    result, used = [], set()
    for prev in sorted(prev_chord_pitches):
        best = min(chord_midis, key=lambda m: (abs(m-prev), m in used))
        result.append(best); used.add(best)
    return sorted(set(max(lo, min(hi, p)) for p in result))

def _detect_style(dna):
    """Detecta el estilo musical aproximado del DNA para ornamentación."""
    tempo = dna.tempo_bpm
    complexity = dna.harmony_dna.get('complexity', 0)
    swing = dna.rhythm_dna.get('swing', 0)
    syncopation = dna.rhythm_dna.get('syncopation', 0)

    if swing > 0.15 or syncopation > 0.2:
        return 'jazz'
    if tempo < 100 and complexity > 0.5:
        return 'romantic'
    if 60 <= tempo <= 140 and complexity < 0.4:
        return 'classical'
    if tempo > 100 and complexity > 0.3:
        return 'baroque'
    return 'generic'


# ═══════════════════════════════════════════════════════════════
#  CLASE MusicDNA (v3 extendida)
# ═══════════════════════════════════════════════════════════════

class MusicDNA:
    def __init__(self, filepath):
        self.filepath   = filepath
        self.name       = os.path.basename(filepath)
        self.tempo_bpm  = 120.0
        self.time_signature = (4,4)
        self.key_tonic  = 'C'
        self.key_mode   = 'major'
        self.key_obj    = None
        self.score      = None
        self.parts      = []
        self.voice_roles = {}

        # ADN dimensions (v2)
        self.rhythm_dna  = {}
        self.melody_dna  = {}
        self.harmony_dna = {}
        self.texture_dna = {}
        self.energy_dna  = {}
        self.emotion_dna = {}
        self.phrase_dna  = {}
        self.instrumentation_dna = {}
        self.tension_dna = {}
        self.novelty_dna = {}
        self.modal_dna   = {}
        self.dynamics_dna= {}
        self.motif_dna   = {}

        # NUEVO v3
        self.fragments   = []          # #A lista de MusicFragment
        self.markov      = MarkovMelody(order=2)   # #B
        self.groove_map  = GrooveMap()             # #H
        self.acc_pattern = AccompanimentPattern()  # #F
        self.style       = 'generic'              # para ornamentación #C
        self.feature_vector = None

    def __repr__(self):
        return (f"MusicDNA({self.name} | {self.key_tonic} {self.key_mode} | "
                f"{self.tempo_bpm:.0f}bpm | style={self.style})")


# ═══════════════════════════════════════════════════════════════
#  EXTRACCIÓN DEL ADN v3
# ═══════════════════════════════════════════════════════════════

def extract_dna(filepath, verbose=False):
    print(f"\n🧬 Extrayendo ADN: {os.path.basename(filepath)}")
    print("─"*60)
    dna = MusicDNA(filepath)

    try:
        score = converter.parse(filepath)
        dna.score = score
    except Exception as e:
        print(f"  ❌ Error: {e}"); return None

    parts = list(score.parts) if hasattr(score,'parts') and score.parts else [score]
    dna.parts = parts

    dna.tempo_bpm      = _extract_tempo(score)
    dna.time_signature = _extract_time_signature(score)
    dna.key_tonic, dna.key_mode, dna.key_obj = _extract_key(score)
    ts_num = dna.time_signature[0]

    if verbose:
        print(f"  🎵 {dna.tempo_bpm:.0f} BPM | "
              f"{ts_num}/{dna.time_signature[1]} | "
              f"{dna.key_tonic} {dna.key_mode}")

    # v2 extractions
    dna.voice_roles      = _classify_voice_roles(parts, verbose)
    dna.rhythm_dna       = _extract_rhythm_dna(score, dna.tempo_bpm)
    dna.melody_dna       = _extract_melody_dna(dna.voice_roles.get('melody_part'), dna.key_obj)
    dna.harmony_dna      = _extract_harmony_dna(score, dna.key_obj)
    dna.texture_dna      = _extract_texture_dna(score, parts)
    dna.energy_dna       = _extract_energy_dna(score, ts_num)
    dna.emotion_dna      = _extract_emotion_dna(score, dna)
    dna.phrase_dna       = _extract_phrase_dna(score)
    dna.tension_dna      = _extract_tension_real(score, dna.key_obj, ts_num)
    dna.novelty_dna      = _extract_novelty_points(dna.energy_dna, dna.tension_dna)
    dna.modal_dna        = _extract_modal_interchange(dna.harmony_dna, dna.key_obj)
    dna.dynamics_dna     = _extract_dynamics_dna(score)
    dna.instrumentation_dna = _extract_instrumentation_dna(parts)
    dna.motif_dna        = _extract_motif_dna(dna.melody_dna)

    # ── NUEVO v3 ──────────────────────────────────────────────

    # #A Mosaic fragments
    dna.fragments = _extract_fragments(
        dna.voice_roles.get('melody_part') or score,
        dna.harmony_dna, dna.tension_dna, dna.energy_dna,
        dna.key_obj, ts_num, dna.name
    )
    if verbose:
        print(f"  🧩 Fragmentos: {len(dna.fragments)} extraídos")

    # #B Markov
    dna.markov.train(dna.melody_dna)
    if verbose:
        n_trans = sum(len(v) for v in dna.markov.transitions.values())
        print(f"  🎲 Markov: {n_trans} transiciones (orden {dna.markov.order})")

    # #H Groove map
    dna.groove_map.train(score, ts_num)
    if verbose and dna.groove_map.trained:
        print(f"  🥁 Groove: {len(dna.groove_map.timing_offsets)} posiciones mapeadas")

    # #F Patrón de acompañamiento real
    acc_parts = dna.voice_roles.get('accompaniment_parts', [])
    if not acc_parts and len(parts) > 1:
        acc_parts = parts[1:]
    dna.acc_pattern.train(acc_parts, ts_num)
    if verbose:
        print(f"  🎹 Patrón acompañamiento: {dna.acc_pattern.style}")

    # Estilo para ornamentación (#C)
    dna.style = _detect_style(dna)
    if verbose:
        print(f"  🎨 Estilo detectado: {dna.style}")

    dna.feature_vector = _build_feature_vector(dna)
    print(f"  ✅ ADN extraído")
    return dna


# ─── Extractores básicos ───────────────────────────────────────

def _extract_tempo(score):
    for el in score.flat.getElementsByClass('MetronomeMark'):
        if el.number: return float(el.number)
    return 120.0

def _extract_time_signature(score):
    for el in score.flat.getElementsByClass('TimeSignature'):
        return (el.numerator, el.denominator)
    return (4,4)

def _extract_key(score):
    try:
        k = score.analyze('key'); return k.tonic.name, k.mode, k
    except: pass
    try:
        ks = score.flat.getElementsByClass('KeySignature')[0]
        ko = ks.asKey(); return ko.tonic.name, ko.mode, ko
    except:
        ko = key.Key('C','major'); return 'C','major',ko

def _classify_voice_roles(parts, verbose=False):
    if not parts:
        return {'melody_part':None,'bass_part':None,'accompaniment_parts':[]}
    scored = []
    for part in parts:
        ns = [n for n in part.flat.notes if n.isNote]
        if not ns:
            scored.append({'part':part,'mean_pitch':60,'density':0,'continuity':0,'inst_name':''})
            continue
        pitches = [n.pitch.midi for n in ns]
        mean_p  = float(np.mean(pitches))
        total_t = float(part.flat.highestTime) or 1
        density = len(ns) / total_t
        continuity = len(ns) / max(1, len(list(part.flat.notesAndRests)))
        try:
            inst = part.getInstrument()
            inst_name = (inst.instrumentName or '').lower() if inst else ''
        except: inst_name = ''
        scored.append({'part':part,'mean_pitch':mean_p,'density':density,
                       'continuity':continuity,'inst_name':inst_name})
    if not scored:
        return {'melody_part':None,'bass_part':None,'accompaniment_parts':[]}

    def mel_score(s):
        pn = (s['mean_pitch']-40)/50
        b  = 0.3 if any(k in s['inst_name'] for k in ['violin','flute','soprano','oboe','trumpet']) else 0
        p  = -0.5 if any(k in s['inst_name'] for k in ['bass','tuba','drum','percussion']) else 0
        return pn + s['continuity']*0.4 + b + p

    def bass_score(s):
        pn = (60-s['mean_pitch'])/30
        b  = 0.3 if any(k in s['inst_name'] for k in ['bass','tuba','contrabass','cello']) else 0
        return pn + b

    sm = sorted(scored, key=mel_score,  reverse=True)
    sb = sorted(scored, key=bass_score, reverse=True)
    melody_part = sm[0]['part']
    bass_part   = sb[0]['part']
    if bass_part is melody_part and len(sb) > 1:
        bass_part = sb[1]['part']
    acc = [s['part'] for s in scored if s['part'] is not melody_part and s['part'] is not bass_part]
    if verbose:
        try: mi = melody_part.getInstrument().instrumentName or '?'
        except: mi = '?'
        print(f"  🎭 Melodía:{mi}(pitch≈{sm[0]['mean_pitch']:.0f})")
    return {'melody_part':melody_part,'bass_part':bass_part,'accompaniment_parts':acc}

def _extract_rhythm_dna(score, bpm):
    all_notes = list(score.flat.notes)
    if not all_notes:
        return {'density':0,'swing':0,'syncopation':0,'pattern':[1.0],
                'durations_raw':[],'groove':[],'has_triplets':False,'tresillo_ratio':0}
    durs = [float(n.quarterLength) for n in all_notes]
    tresillo_c = sum(1 for d in durs if abs(d-1/3)<0.05 or abs(d-2/3)<0.05)
    tresillo_ratio = tresillo_c / max(1, len(durs))
    def qd(d): return min(VALID_DURATIONS_FLOAT, key=lambda x: abs(x-d))
    pattern_q = [qd(d) for d in durs[:64]]
    ts_num = _extract_time_signature(score)[0]
    total_t = float(score.flat.highestTime) or 1
    measures = max(1, total_t / ts_num)
    density  = len(all_notes) / measures
    swing_c  = sum(1 for i in range(len(durs)-1)
                   if 0.5<durs[i]<0.8 and 0.2<durs[i+1]<0.4
                   and durs[i]/durs[i+1]>1.4)
    swing    = swing_c / max(1, len(durs)//2)
    synco_c  = sum(1 for n in all_notes
                   if 0.25 < float(n.offset)%1.0 < 0.75 and float(n.quarterLength)>=1.0)
    syncopation = synco_c / max(1, len(all_notes))
    vels = [n.volume.velocity or 64 for n in all_notes if hasattr(n,'volume')]
    groove = [v/127.0 for v in vels[:64]]
    return {'density':density,'swing':swing,'syncopation':syncopation,
            'pattern':pattern_q,'durations_raw':durs,'groove':groove,
            'has_triplets':tresillo_ratio>0.05,'tresillo_ratio':tresillo_ratio}

def _extract_melody_dna(melody_part, key_obj):
    if melody_part is None:
        return {'range_semitones':0,'direction':0,'leap_ratio':0.2,
                'contour':[0.5]*8,'intervals':[2,1,-1,-2,3,-3],'motif':[2,1,-1,-2],
                'mean_pitch':65.0,'pitches':[],'durations':[1.0],'scale_degrees':[],'notes':[]}
    ns = [n for n in melody_part.flat.notes if n.isNote]
    if len(ns) < 2:
        return {'range_semitones':0,'direction':0,'leap_ratio':0.2,
                'contour':[0.5]*8,'intervals':[2,1,-1,-2],'motif':[2,1,-1],
                'mean_pitch':65.0,'pitches':[],'durations':[1.0],'scale_degrees':[],'notes':[]}
    pitches   = [n.pitch.midi for n in ns]
    intervals = [pitches[i+1]-pitches[i] for i in range(len(pitches)-1)]
    durs      = [float(n.quarterLength) for n in ns]
    rng       = max(pitches)-min(pitches)
    direction = float(np.mean(np.sign(intervals))) if intervals else 0
    leap_ratio= sum(1 for i in intervals if abs(i)>4)/max(1,len(intervals))
    pa = np.array(pitches,dtype=float)
    pn = (pa-pa.min())/max(1,pa.max()-pa.min())
    idx= np.linspace(0,len(pn)-1,min(32,len(pn)),dtype=int)
    contour = pn[idx].tolist()
    scale_degrees = []
    if key_obj:
        pcs = _get_scale_pcs(key_obj)
        tc  = pitch.Pitch(key_obj.tonic.name).pitchClass
        for n in ns[:128]:
            rpc = (n.pitch.pitchClass - tc) % 12
            if rpc in pcs: scale_degrees.append(pcs.index(rpc))
    return {'range_semitones':rng,'direction':direction,'leap_ratio':leap_ratio,
            'contour':contour,'intervals':intervals[:128],'motif':intervals[:8],
            'mean_pitch':float(np.mean(pitches)),'pitches':pitches,'durations':durs,
            'scale_degrees':scale_degrees,'notes':ns}

def _extract_harmony_dna(score, key_obj):
    flat_ch   = score.flat.chordify()
    chord_objs = list(flat_ch.flat.getElementsByClass('Chord'))
    if not chord_objs:
        return {'complexity':0,'change_rate':0,'progressions':[],'functions':[],
                'tension_curve':[],'bigrams':[],'modal_chords':[]}
    progressions,functions,tension_curve,modal_chords = [],[],[],[]
    for ch in chord_objs[:128]:
        try:
            root    = ch.root()
            quality = ch.quality
            rn_fig  = 'I'; tension = 0.3
            if key_obj and root:
                try:
                    rn = roman.romanNumeralFromChord(ch, key_obj)
                    rn_fig = rn.figure
                    if any(x in rn_fig for x in ['V7','vii']): tension = 0.85
                    elif rn_fig.startswith('V'):                tension = 0.70
                    elif rn_fig in ['ii','IV','iv','II']:       tension = 0.40
                    elif rn_fig in ['I','i','vi','VI']:         tension = 0.10
                    else:                                       tension = 0.55
                    if len(ch.pitches)>4: tension = min(1.0, tension+0.15)
                    if _is_modal_interchange(rn_fig, key_obj.mode):
                        modal_chords.append({'figure':rn_fig,'offset':float(ch.offset)})
                except: pass
            progressions.append({'root':root.name if root else 'C','quality':quality,
                                  'roman':rn_fig,'offset':float(ch.offset),
                                  'duration':float(ch.quarterLength),
                                  'pitches':[p.midi for p in ch.pitches]})
            functions.append(rn_fig); tension_curve.append(tension)
        except: continue
    unique_ch  = len(set(f['roman'] for f in progressions))
    complexity = unique_ch / max(1, len(progressions))
    total_t    = (progressions[-1]['offset']-progressions[0]['offset']) if progressions else 1
    change_rate= len(progressions)/max(1,total_t/4)
    bigrams    = Counter()
    for i in range(len(functions)-1): bigrams[(functions[i],functions[i+1])] += 1
    return {'complexity':complexity,'change_rate':change_rate,'progressions':progressions,
            'functions':functions,'tension_curve':tension_curve,
            'bigrams':bigrams.most_common(10),'modal_chords':modal_chords}

def _is_modal_interchange(figure, mode):
    MB = {'bVII','bIII','bVI','iv','bII','ii°'}
    mB = {'I','IV','II','bVII'}
    base = figure.replace('7','').replace('°','').replace('+','')
    return base in (MB if mode=='major' else mB)

def _extract_texture_dna(score, parts):
    layers = len(parts)
    ns = list(score.flat.notes)
    if not ns: return {'layers':0,'note_density':0,'homophony':0,'polyphony':0,'type':'monophony'}
    total_t = float(score.flat.highestTime) or 1
    density = len(ns)/total_t
    oc = Counter(round(float(n.offset),2) for n in ns)
    simult = sum(1 for c in oc.values() if c>1)
    homo = simult/max(1,len(oc)); poly = 1-homo
    ttype = ('monophony' if layers==1 else
             'homophony' if homo>0.6 else
             'polyphony' if poly>0.6 else 'heterophony')
    pm = [n.pitch.midi for n in ns if n.isNote]
    return {'layers':layers,'note_density':density,'homophony':homo,'polyphony':poly,
            'type':ttype,'register_spread':float(np.std(pm)) if pm else 0}

def _extract_energy_dna(score, ts_num=4):
    ns = list(score.flat.notes)
    if not ns: return {'mean':0.5,'peak':0.5,'arc_type':'flat','curve':[],'n_measures':0}
    total_t = float(score.flat.highestTime) or 1
    n_m     = max(1, int(total_t/ts_num))
    curve   = []
    for m in range(n_m):
        t0,t1 = m*ts_num,(m+1)*ts_num
        mn    = [n for n in ns if t0<=float(n.offset)<t1]
        if not mn: curve.append(0.0); continue
        vels  = [n.volume.velocity or 64 for n in mn if hasattr(n,'volume')]
        mv    = np.mean(vels)/127 if vels else 0.5
        den   = min(1.0, len(mn)/(ts_num*2))
        curve.append(float(den*0.5+mv*0.5))
    arr = np.array(curve) if curve else np.array([0.5])
    pp  = float(np.argmax(arr))/max(1,len(arr))
    arc = ('front_loaded' if pp<0.25 else 'climax_ending' if pp>0.75 else
           'arch' if 0.4<pp<0.6 else
           'crescendo' if np.mean(arr[len(arr)//2:])>np.mean(arr[:len(arr)//2])*1.1 else
           'decrescendo' if np.mean(arr[:len(arr)//2])>np.mean(arr[len(arr)//2:])*1.1 else 'flat')
    return {'mean':float(np.mean(arr)),'peak':float(np.max(arr)),
            'arc_type':arc,'curve':curve,'n_measures':n_m}

def _extract_emotion_dna(score, dna):
    mv     = 0.7 if dna.key_mode=='major' else -0.4
    tn     = np.clip((dna.tempo_bpm-60)/120,-1,1)
    valence    = float(np.clip((mv+tn)/2,-1,1))
    activation = float(np.clip(tn + dna.rhythm_dna.get('density',0)/20,-1,1))
    tension    = float(dna.harmony_dna.get('complexity',0)*0.5 +
                       dna.rhythm_dna.get('syncopation',0)*0.5)
    ns    = list(score.flat.notes)
    mp    = np.mean([n.pitch.midi for n in ns if n.isNote]) if ns else 60
    dark  = float(np.clip((72-mp)/36,0,1))
    if dna.key_mode=='minor': dark = min(1.0,dark+0.2)
    return {'valence':valence,'activation':activation,'tension':tension,'darkness':dark}

def _extract_phrase_dna(score):
    ns = list(score.flat.notes)
    if not ns: return {'avg_length':4,'qa_ratio':0,'phrase_lengths':[4]}
    phrase_lengths,phrase_start,prev_off = [],[],float(ns[0].offset)
    phrase_start = 0
    for i,n in enumerate(ns[1:],1):
        curr = float(n.offset)
        gap  = curr-(prev_off+float(ns[i-1].quarterLength))
        if gap>2.0 or (curr-float(ns[phrase_start].offset))>16:
            phrase_lengths.append(i-phrase_start); phrase_start = i
        prev_off = curr
    if not phrase_lengths: phrase_lengths = [len(ns)]
    qa = sum(1 for i in range(0,len(phrase_lengths)-1,2)
             if abs(phrase_lengths[i]-phrase_lengths[i+1])<phrase_lengths[i]*0.3)
    return {'avg_length':float(np.mean(phrase_lengths)),
            'qa_ratio':qa/max(1,len(phrase_lengths)//2),'phrase_lengths':phrase_lengths}

def _extract_tension_real(score, key_obj, ts_num=4):
    flat_ch = score.flat.chordify()
    chords  = list(flat_ch.flat.getElementsByClass('Chord'))
    total_t = float(score.flat.highestTime) or 1
    n_m     = max(1, int(total_t/ts_num))
    tpm     = [[] for _ in range(n_m)]
    tc_pc   = pitch.Pitch(key_obj.tonic.name).pitchClass if key_obj else 0
    for ch in chords:
        m_idx = min(n_m-1, int(float(ch.offset)/ts_num))
        pm    = [p.midi for p in ch.pitches]
        if not pm: continue
        tl = 0.0
        for i in range(len(pm)):
            for j in range(i+1,len(pm)):
                ic = abs(pm[i]-pm[j])%12
                tl += LERDAHL_DISSONANCE.get(ic,3)/7.0
        tl /= max(1, len(pm)*(len(pm)-1)/2)
        pcs   = [p.midi%12 for p in ch.pitches]
        tt    = np.mean([TONNETZ_DISTANCES.get(abs(pc-tc_pc)%12,2) for pc in pcs])/3.5
        tf    = 0.3
        if key_obj:
            try:
                rn  = roman.romanNumeralFromChord(ch,key_obj)
                fig = rn.figure
                if 'vii' in fig or 'V7' in fig: tf = 0.9
                elif fig.startswith('V'):        tf = 0.7
                elif fig in ['II','ii','IV','iv']: tf = 0.4
                elif fig in ['I','i','vi','VI']:  tf = 0.05
                else:                             tf = 0.5
            except: pass
        tpm[m_idx].append(tl*0.4+tt*0.3+tf*0.3)
    tc = [float(np.mean(v)) if v else 0.3 for v in tpm]
    return {'curve':tc,'mean':float(np.mean(tc)) if tc else 0.3,
            'peak':float(max(tc)) if tc else 0.3,
            'peak_measure':int(np.argmax(tc)) if tc else 0,
            'std':float(np.std(tc)) if tc else 0}

def _extract_novelty_points(energy_dna, tension_dna):
    ec = energy_dna.get('curve',[])
    tc = tension_dna.get('curve',[])
    n  = min(len(ec),len(tc)) if ec and tc else len(ec) or len(tc) or 1
    scores = []
    for i in range(n):
        ed = abs(ec[i]-ec[i-1]) if i>0 and i<len(ec) else 0
        td = abs(tc[i]-tc[i-1]) if i>0 and i<len(tc) else 0
        scores.append(ed*0.5+td*0.5)
    arr = np.array(scores) if scores else np.array([0])
    thr = float(np.mean(arr)+0.5*np.std(arr))
    return {'scores':scores,'peak_measures':[i for i,v in enumerate(scores) if v>thr],
            'threshold':thr}

def _extract_modal_interchange(harmony_dna, key_obj):
    mc  = harmony_dna.get('modal_chords',[])
    fn  = harmony_dna.get('functions',[])
    br  = len(mc)/max(1,len(fn))
    types = Counter(c['figure'] for c in mc)
    return {'modal_chords':mc,'borrowed_ratio':br,'types':dict(types),'uses_modal':br>0.05}

def _extract_dynamics_dna(score):
    ns = list(score.flat.notes)
    if not ns: return {'mean_velocity':64,'range':0,'crescendo_ratio':0,'velocities':[]}
    vels = [n.volume.velocity or 64 for n in ns if hasattr(n,'volume')]
    return {'mean_velocity':float(np.mean(vels)),'range':float((max(vels)-min(vels))/127),
            'crescendo_ratio':0,'velocities':vels[:128]}

def _extract_instrumentation_dna(parts):
    FAMILIES = {'piano':['piano','keyboard','organ'],'strings':['violin','viola','cello','contrabass','string','harp'],
                'woodwinds':['flute','oboe','clarinet','bassoon','saxophone'],'brass':['trumpet','horn','trombone','tuba'],
                'percussion':['drum','percussion','timpani'],'voice':['voice','soprano','alto','tenor','choir'],
                'guitar':['guitar','lute']}
    families,ilist = defaultdict(list),[]
    for part in parts:
        try: name=(part.getInstrument().instrumentName or '').lower()
        except: name=''
        ilist.append(name or 'unknown')
        placed=False
        for fam,kws in FAMILIES.items():
            if any(k in name for k in kws): families[fam].append(name); placed=True; break
        if not placed: families['other'].append(name or 'unknown')
    return {'families':dict(families),'instruments':ilist,'n_parts':len(parts)}

def _extract_motif_dna(melody_dna):
    intervals = melody_dna.get('intervals',[2,1,-1,-2])
    durations = melody_dna.get('durations',[0.5,0.5,1.0])
    mi = intervals[:6] if len(intervals)>=4 else (intervals or [2,1,-1])
    md = durations[:len(mi)+1] if durations else [0.5]*len(mi)
    return {'intervals':mi,'durations':md,
            'inversion':  [-x for x in mi],
            'retrograde': list(reversed(mi)),
            'retro_inv':  list(reversed([-x for x in mi])),
            'sequence_up':   mi+[x+2 for x in mi],
            'sequence_down': mi+[x-2 for x in mi],
            'augmentation_d':[x*2 for x in md],
            'diminution_d':  [max(0.25,x/2) for x in md]}

def _build_feature_vector(dna):
    return np.array([
        dna.tempo_bpm/200,float(dna.key_mode=='major'),
        dna.rhythm_dna.get('density',0)/20,dna.rhythm_dna.get('swing',0),
        dna.rhythm_dna.get('syncopation',0),dna.melody_dna.get('range_semitones',0)/36,
        dna.melody_dna.get('leap_ratio',0),dna.harmony_dna.get('complexity',0),
        dna.harmony_dna.get('change_rate',0)/4,dna.energy_dna.get('mean',0.5),
        dna.emotion_dna.get('valence',0),dna.emotion_dna.get('activation',0),
        dna.tension_dna.get('mean',0.3),dna.modal_dna.get('borrowed_ratio',0),
        dna.texture_dna.get('note_density',0)/10,
    ],dtype=float)


# ─── Extracción de fragmentos (#A) ────────────────────────────

def _extract_fragments(source_part, harmony_dna, tension_dna, energy_dna,
                       key_obj, ts_num, source_name, fragment_len=2):
    """
    #A: Divide la parte melódica en fragmentos de fragment_len compases,
    anotando cada uno con tensión, energía y función armónica dominante.
    """
    if source_part is None:
        return []

    ns = [n for n in source_part.flat.notes if n.isNote]
    if not ns:
        return []

    total_t    = float(source_part.flat.highestTime) or 1
    n_measures = max(1, int(total_t / ts_num))
    functions  = harmony_dna.get('functions', [])
    t_curve    = tension_dna.get('curve', [])
    e_curve    = energy_dna.get('curve', [])

    fragments = []

    for m in range(0, n_measures - fragment_len + 1, fragment_len):
        t_start = m * ts_num
        t_end   = (m + fragment_len) * ts_num

        # Notas del fragmento con offsets relativos
        frag_notes = []
        for n in ns:
            noff = float(n.offset)
            if t_start <= noff < t_end:
                nc = copy.deepcopy(n)
                nc.offset = noff - t_start  # offset relativo
                frag_notes.append(nc)

        if len(frag_notes) < 2:
            continue

        # Anotar
        tension = np.mean(t_curve[m:m+fragment_len]) if t_curve else 0.3
        energy  = np.mean(e_curve[m:m+fragment_len]) if e_curve else 0.5
        func    = functions[m % len(functions)] if functions else 'I'

        frag = MusicFragment(
            notes       = frag_notes,
            offset      = t_start,
            duration_ql = t_end - t_start,
            measure_idx = m,
            tension     = float(tension),
            energy      = float(energy),
            function    = func,
            key_obj     = key_obj,
            source_name = source_name,
        )
        fragments.append(frag)

    return fragments

# ═══════════════════════════════════════════════════════════════
#  GENERADORES MUSICALES v3
# ═══════════════════════════════════════════════════════════════

# ─── #A Mosaic splicing ────────────────────────────────────────

def _select_fragment(fragments, target_tension, target_energy, target_function,
                     prev_pitch=None, avoid_repeat=None):
    """
    Selecciona el fragmento más adecuado del pool según:
    - Similitud de tensión/energía
    - Compatibilidad de función armónica
    - Continuidad de pitch (sin saltos grandes)
    - Evitar repetición inmediata
    """
    if not fragments:
        return None

    FUNC_COMPAT = {
        'I': ['I','iii','vi','IV'], 'i': ['i','III','VI','iv'],
        'V': ['V','V7','vii°'],     'IV': ['IV','ii','vi'],
        'iv': ['iv','ii°','VI'],    'vi': ['vi','IV','ii','I'],
    }
    compatible = FUNC_COMPAT.get(target_function, ['I','V','IV','vi'])

    scored = []
    for frag in fragments:
        if avoid_repeat and frag is avoid_repeat:
            continue
        # Similitud de tensión y energía
        score = 1.0 - (abs(frag.tension - target_tension) +
                       abs(frag.energy  - target_energy)) / 2.0
        # Bonus función armónica compatible
        if frag.function in compatible:
            score += 0.3
        # Continuidad de pitch
        if prev_pitch is not None and frag.notes:
            pitch_jump = abs(frag.pitch_mean - prev_pitch)
            score -= pitch_jump / 24.0  # Penalizar saltos > 2 octavas
        scored.append((score, frag))

    if not scored:
        return random.choice(fragments)

    # Selección probabilística ponderada por score
    scored.sort(key=lambda x: -x[0])
    top = scored[:max(3, len(scored)//3)]
    scores   = [max(0.01, s) for s,_ in top]
    total    = sum(scores)
    probs    = [s/total for s in scores]
    idx = np.random.choice(len(top), p=probs)
    return top[idx][1]


def _build_mosaic_melody(dna_sources, chord_prog, target_key, n_measures, ts_num,
                          tension_curve=None, energy_curve=None, prev_pitch=None):
    """
    #A: Construye la melodía ensamblando fragmentos reales transpuestos.
    """
    # Pool de fragmentos de todos los DNA fuente
    all_fragments = []
    for dna in dna_sources:
        all_fragments.extend(dna.fragments)

    if not all_fragments:
        return []  # Fallback al generador Markov

    result_notes = []
    current_offset = 0.0
    last_frag = None
    last_pitch = prev_pitch or 65

    # Dividir en ventanas de fragment_len compases
    frag_len = 2  # compases por fragmento
    n_windows = max(1, n_measures // frag_len)

    for w in range(n_windows):
        m_idx = w * frag_len
        frag_offset = m_idx * ts_num

        t_val = (tension_curve[min(m_idx, len(tension_curve)-1)]
                 if tension_curve else 0.4)
        e_val = (energy_curve[min(m_idx, len(energy_curve)-1)]
                 if energy_curve else 0.5)
        func  = chord_prog[m_idx % len(chord_prog)]['function'] if chord_prog else 'I'

        frag = _select_fragment(all_fragments, t_val, e_val, func,
                                 prev_pitch=last_pitch, avoid_repeat=last_frag)

        if frag is None:
            continue

        # Transponer a la tonalidad objetivo (#A)
        transposed_notes = frag.transpose_to(target_key)

        # Ajustar offsets al fragmento actual
        for n in transposed_notes:
            if n.isNote:
                n.offset = frag_offset + float(n.offset)
                # Ajustar pitch a rango melódico
                lo, hi = INSTRUMENT_RANGES['melody']
                while n.pitch.midi > hi: n.pitch.midi -= 12
                while n.pitch.midi < lo: n.pitch.midi += 12
                result_notes.append(n)

        if transposed_notes:
            pitches = [n.pitch.midi for n in transposed_notes if n.isNote]
            if pitches:
                last_pitch = pitches[-1]
        last_frag = frag

    return result_notes


# ─── #B Markov melody generator ───────────────────────────────

def _build_markov_melody(markov_models, chord_prog, target_key, n_measures, ts_num,
                          section_type='A', motif_dna=None, novelty_measures=None,
                          variation=False, seed=None):
    """
    #B: Genera melodía usando el modelo de Markov entrenado,
    con: motivos (#6), cadencias (#5), puntos de novedad (#9).
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # Elegir el modelo Markov con más transiciones
    best_markov = max(markov_models, key=lambda m: sum(len(v) for v in m.transitions.values()),
                      default=markov_models[0] if markov_models else MarkovMelody())

    # Generar secuencia base con Markov
    n_total_steps = n_measures * int(ts_num / 0.5) + 16  # generoso, se truncará
    tonic_midi = pitch.Pitch(target_key.tonic.name).midi + 60
    tonic_midi = max(INSTRUMENT_RANGES['melody'][0],
                     min(INSTRUMENT_RANGES['melody'][1], tonic_midi))

    markov_seq = best_markov.generate(n_total_steps, tonic_midi, target_key, seed=seed)
    # markov_seq: lista de (pitch_midi, duration)

    melody_out = []
    phrase_len = 4  # compases
    n_phrases  = max(1, n_measures // phrase_len)
    novelty_set = set(novelty_measures or [])

    motif_ints = []
    if motif_dna:
        motif_forms = {
            'A':    motif_dna.get('intervals', []),
            'B':    motif_dna.get('inversion', []),
            "A'":   motif_dna.get('sequence_up', []),
            'C':    motif_dna.get('retrograde', []),
            'intro':motif_dna.get('diminution_d', []),
            'coda': motif_dna.get('augmentation_d', []),
        }
        motif_ints = motif_forms.get(section_type, motif_dna.get('intervals', []))
        if variation:
            motif_ints = motif_dna.get('sequence_down', motif_ints)

    seq_idx = 0

    for ph in range(n_phrases):
        phrase_off   = ph * phrase_len * ts_num
        is_last      = (ph == n_phrases - 1)
        is_mid       = (ph == n_phrases // 2 - 1)
        cadence_type = 'AC' if is_last else ('HC' if is_mid else None)
        is_answer    = (ph % 2 == 1)  # frase respuesta

        for m in range(phrase_len):
            m_global = ph * phrase_len + m
            if m_global >= n_measures:
                break
            m_off   = phrase_off + m * ts_num
            chord_d = chord_prog[m_global % len(chord_prog)] if chord_prog else None
            chord_p = chord_d['pitches'] if chord_d else [60,64,67]

            is_cad_m = cadence_type and (m >= phrase_len - 2)

            beat = 0.0
            while beat < ts_num - 0.01:
                # Duración desde Markov
                if seq_idx < len(markov_seq):
                    pitch_midi, dur = markov_seq[seq_idx]
                    seq_idx += 1
                else:
                    pitch_midi = tonic_midi
                    dur = 1.0

                dur = min(VALID_DURATIONS_FLOAT, key=lambda x: abs(x-dur))
                dur = max(0.25, min(dur, ts_num - beat))

                # Pitch: cadencia, novedad o Markov
                if is_cad_m and beat == 0:
                    # #5 Cadencia: notar tónico o dominante
                    if cadence_type == 'AC':
                        p = _snap_to_scale(
                            pitch.Pitch(target_key.tonic.name).midi + 60, target_key)
                    else:
                        p = _snap_to_scale(
                            pitch.Pitch(target_key.tonic.name).midi + 67, target_key)
                    p = max(INSTRUMENT_RANGES['melody'][0],
                            min(INSTRUMENT_RANGES['melody'][1], p))
                    dur = float(ts_num)  # nota larga en cadencia
                elif m_global in novelty_set and beat == 0:
                    # #9 Punto de novedad: salto dramático
                    direction = 1 if pitch_midi < 72 else -1
                    p = _snap_to_scale(pitch_midi + direction*random.randint(5,9), target_key)
                    p = max(INSTRUMENT_RANGES['melody'][0],
                            min(INSTRUMENT_RANGES['melody'][1], p))
                elif motif_ints and m == 0 and beat == 0 and not is_answer:
                    # #6 Inicio de frase: presentar motivo
                    p = pitch_midi
                    # Primer interval del motivo
                    if motif_ints:
                        p = _snap_to_scale(p + motif_ints[0], target_key)
                    p = max(INSTRUMENT_RANGES['melody'][0],
                            min(INSTRUMENT_RANGES['melody'][1], p))
                else:
                    p = _snap_to_scale(pitch_midi, target_key)
                    p = max(INSTRUMENT_RANGES['melody'][0],
                            min(INSTRUMENT_RANGES['melody'][1], p))
                    # Gravedad hacia chord tone (30%)
                    if chord_p and random.random() < 0.30:
                        nearest = min(chord_p, key=lambda cp: abs(cp-p))
                        p = _snap_to_scale(nearest, target_key)

                n = note.Note(p)
                n.quarterLength = dur
                n.offset = m_off + beat

                # Velocidad con acento métrico
                if beat < 0.05:
                    vel = random.randint(72,92)
                elif abs(beat - ts_num/2) < 0.1:
                    vel = random.randint(62,80)
                else:
                    vel = random.randint(48,68)
                n.volume.velocity = vel

                melody_out.append(n)
                beat += dur
                if is_cad_m and beat == 0:
                    break  # nota larga: salir del bucle

    return melody_out


# ─── #C Ornamentación idiomática ──────────────────────────────

def _add_ornamentation(melody_notes, key_obj, style='generic', density=0.3):
    """
    #C: Añade ornamentación idiomática a las notas de la melodía:
    - Nota de paso: entre dos notas separadas por 3ª o más
    - Nota vecina: decoración de una nota larga
    - Apoyatura: nota de adorno antes de una nota fuerte
    - Mordente: alternancia rápida
    """
    if not melody_notes or not key_obj:
        return melody_notes

    style_probs = ORNAMENTATION_STYLES.get(style, ORNAMENTATION_STYLES['generic'])
    scale_m     = _get_scale_midi(key_obj, octave=4)
    scale_set   = set(scale_m)

    result = []
    i = 0

    while i < len(melody_notes):
        n = melody_notes[i]

        if not n.isNote or float(n.quarterLength) < 0.5:
            result.append(n); i += 1; continue

        beat_pos = float(n.offset) % 4.0
        is_strong_beat = beat_pos < 0.05 or abs(beat_pos - 2.0) < 0.05

        applied = False

        # Apoyatura (en tiempo fuerte, si hay nota siguiente)
        if (is_strong_beat and not applied
                and random.random() < style_probs['appoggiatura']
                and float(n.quarterLength) >= 1.0
                and i + 1 < len(melody_notes)):
            # Nota superior o inferior por semitono
            app_pitch = n.pitch.midi + random.choice([1,-1,2,-2])
            app_pitch = max(21, min(108, app_pitch))
            app_dur   = min(0.25, float(n.quarterLength)/3)
            main_dur  = float(n.quarterLength) - app_dur

            app = note.Note(app_pitch)
            app.quarterLength = app_dur
            app.offset = n.offset
            app.volume.velocity = int(n.volume.velocity * 0.9) if hasattr(n,'volume') and n.volume.velocity else 60

            main = copy.deepcopy(n)
            main.quarterLength = main_dur
            main.offset = float(n.offset) + app_dur

            result.append(app)
            result.append(main)
            applied = True

        # Nota de paso (entre dos notas con intervalo ≥ 3ª)
        elif (not applied
              and random.random() < style_probs['passing']
              and i + 1 < len(melody_notes)
              and melody_notes[i+1].isNote
              and float(n.quarterLength) >= 0.5):
            next_n   = melody_notes[i+1]
            interval = next_n.pitch.midi - n.pitch.midi
            if abs(interval) >= 3 and float(n.quarterLength) >= 1.0:
                # Insertar nota de paso entre las dos
                pass_pitch = n.pitch.midi + (1 if interval > 0 else -1)
                pass_pitch = _snap_to_scale(pass_pitch, key_obj)
                half_dur   = float(n.quarterLength) / 2

                first = copy.deepcopy(n)
                first.quarterLength = half_dur

                pn = note.Note(pass_pitch)
                pn.quarterLength = half_dur
                pn.offset = float(n.offset) + half_dur
                pn.volume.velocity = int((n.volume.velocity or 64) * 0.75)

                result.append(first)
                result.append(pn)
                applied = True

        # Nota vecina (en nota larga, añadir vecina superior breve)
        elif (not applied
              and random.random() < style_probs['neighbor']
              and float(n.quarterLength) >= 1.5):
            neigh_pitch = _snap_to_scale(n.pitch.midi + 2, key_obj)
            neigh_dur   = 0.25
            main_dur    = float(n.quarterLength) - neigh_dur * 2

            if main_dur >= 0.5:
                first = copy.deepcopy(n)
                first.quarterLength = main_dur / 2

                nb = note.Note(neigh_pitch)
                nb.quarterLength = neigh_dur
                nb.offset = float(n.offset) + main_dur/2
                nb.volume.velocity = int((n.volume.velocity or 64) * 0.7)

                last_note = copy.deepcopy(n)
                last_note.quarterLength = main_dur/2 + neigh_dur
                last_note.offset = float(n.offset) + main_dur/2 + neigh_dur

                result.append(first)
                result.append(nb)
                result.append(last_note)
                applied = True

        if not applied:
            result.append(n)

        i += 1

    return result


# ─── #D Articulación y fraseo ─────────────────────────────────

def _add_articulation(melody_notes, style='generic', energy_curve=None, ts_num=4):
    """
    #D: Añade marcas de articulación:
    - Staccato en notas cortas y en estilos barroco/clásico
    - Tenuto en notas largas y tiempos fuertes
    - Acento en anacrusa y tiempos débiles sincopados
    - Slur en frases legato (romántico/jazz)
    """
    if not melody_notes:
        return melody_notes

    slur_style = style in ('romantic', 'jazz')
    phrase_groups = []
    current_phrase = []

    for i, n in enumerate(melody_notes):
        if not n.isNote:
            if current_phrase:
                phrase_groups.append(current_phrase)
                current_phrase = []
            continue

        dur  = float(n.quarterLength)
        beat = float(n.offset) % ts_num

        # Staccato: notas muy cortas en tiempo débil
        if dur <= 0.25 and beat > 0.1 and style in ('baroque','classical'):
            n.articulations.append(articulations.Staccato())

        # Tenuto: nota larga en tiempo fuerte
        elif dur >= 2.0 and beat < 0.1:
            n.articulations.append(articulations.Tenuto())

        # Acento: nota on-beat fuerte
        elif beat < 0.05 and i > 0:
            if energy_curve:
                m_idx = int(float(n.offset) / ts_num)
                energy = energy_curve[min(m_idx, len(energy_curve)-1)]
                if energy > 0.65:
                    n.articulations.append(articulations.Accent())

        current_phrase.append(n)

        # Detectar fin de frase (silencio o gap grande)
        if i + 1 < len(melody_notes) and melody_notes[i+1].isNote:
            gap = float(melody_notes[i+1].offset) - (float(n.offset) + dur)
            if gap > 1.5:
                phrase_groups.append(current_phrase)
                current_phrase = []

    if current_phrase:
        phrase_groups.append(current_phrase)

    # Anacrusas: la nota antes de un tiempo fuerte = acento leve
    all_notes_flat = [n for group in phrase_groups for n in group]
    for i in range(len(all_notes_flat) - 1):
        n    = all_notes_flat[i]
        next_n = all_notes_flat[i+1]
        beat_n = float(next_n.offset) % ts_num
        if beat_n < 0.05 and float(n.offset) % ts_num > (ts_num - 0.75):
            # Nota de anacrusa: ligeramente acentuada
            if n.volume.velocity:
                n.volume.velocity = min(127, n.volume.velocity + 8)

    return all_notes_flat


# ─── #E Contrapunto simple ────────────────────────────────────

def _generate_counterpoint(melody_notes, chord_prog, target_key, n_measures, ts_num):
    """
    #E: Genera una segunda voz en movimiento contrario a la melodía.
    Reglas básicas de contrapunto de primera especie:
    - Movimiento contrario preferido (60%), oblicuo (25%), directo (15%)
    - Evitar quintas/octavas paralelas
    - Preferir 3ª y 6ª como intervalos de armonización
    - Mantenerse en rango de tenor
    """
    if not melody_notes:
        return []

    cp_out = []
    lo, hi = INSTRUMENT_RANGES['tenor']

    # Notas de la melodía por compás
    mel_by_measure = defaultdict(list)
    for n in melody_notes:
        if n.isNote:
            m_idx = int(float(n.offset) / ts_num)
            mel_by_measure[m_idx].append(n)

    GOOD_INTERVALS = {3, 4, 8, 9}    # 3ª menor/mayor, 6ª menor/mayor
    AVOID_PARALLEL = {7, 12}          # 5ª justa, octava

    prev_cp_pitch    = None
    prev_mel_pitch   = None
    prev_cp_interval = None

    for m in range(n_measures):
        chord_d  = chord_prog[m % len(chord_prog)] if chord_prog else None
        chord_p  = chord_d['pitches'] if chord_d else [60,64,67]
        mel_m    = mel_by_measure.get(m, [])

        if not mel_m:
            r = note.Rest(); r.quarterLength = ts_num; r.offset = m * ts_num
            cp_out.append(r); continue

        # Una nota de contrapunto por tiempo fuerte del compás
        # (primera especie simplificada: 1 nota cp por beat)
        beats_in_measure = [0.0, float(ts_num)/2] if ts_num >= 2 else [0.0]

        for beat in beats_in_measure:
            mel_at_beat = [n for n in mel_m if abs(float(n.offset) - (m*ts_num + beat)) < 0.5]
            if not mel_at_beat:
                continue
            mel_n = mel_at_beat[0]
            mel_p = mel_n.pitch.midi

            # Elegir pitch de contrapunto
            best_cp = None
            best_score = -999

            for cp_cand in [p for p in chord_p if lo <= p <= hi]:
                interval_to_mel = abs(mel_p - cp_cand) % 12
                score = 0

                # Intervalos buenos
                if interval_to_mel in GOOD_INTERVALS:
                    score += 3
                elif interval_to_mel in {0, 12}:
                    score += 1  # unísono/octava: aceptable pero no preferido
                elif interval_to_mel in {7}:
                    score += 0  # quinta: neutra
                else:
                    score -= 1

                # Evitar paralelas de 5ª/8ª con el acorde anterior
                if prev_cp_pitch and prev_mel_pitch:
                    prev_int_mel = abs(prev_mel_pitch - prev_cp_pitch) % 12
                    curr_int_mel = interval_to_mel
                    if prev_int_mel in AVOID_PARALLEL and curr_int_mel == prev_int_mel:
                        score -= 4  # paralelas prohibidas

                # Preferir movimiento contrario a la melodía
                if prev_cp_pitch and prev_mel_pitch:
                    mel_dir = np.sign(mel_p - prev_mel_pitch)
                    cp_dir  = np.sign(cp_cand - prev_cp_pitch)
                    if mel_dir != 0 and cp_dir == -mel_dir:
                        score += 2  # movimiento contrario
                    elif cp_dir == 0:
                        score += 1  # oblicuo

                # Penalizar saltos grandes
                if prev_cp_pitch:
                    jump = abs(cp_cand - prev_cp_pitch)
                    if jump > 7: score -= 2
                    elif jump > 4: score -= 1

                if score > best_score:
                    best_score = score
                    best_cp = cp_cand

            if best_cp is None:
                # Fallback: tercera inferior a la melodía
                best_cp = _snap_to_scale(mel_p - 4, target_key)
                best_cp = max(lo, min(hi, best_cp))

            dur = float(ts_num) / len(beats_in_measure)
            n_cp = note.Note(best_cp)
            n_cp.quarterLength = dur
            n_cp.offset = m * ts_num + beat
            n_cp.volume.velocity = random.randint(45, 62)
            cp_out.append(n_cp)

            prev_cp_pitch    = best_cp
            prev_mel_pitch   = mel_p
            prev_cp_interval = abs(mel_p - best_cp) % 12

    return cp_out


# ─── #G Modulación temporal ───────────────────────────────────

def _build_modulated_progression(base_key, target_key, n_measures, source_functions):
    """
    #G: En la sección B, modula brevemente a la tonalidad relativa o dominante.
    Estructura: 2 compases en tónica → pivot → n-4 compases en nueva tonalidad
                → 2 compases de retorno
    """
    mode       = base_key.mode
    new_key    = _get_relative_key(base_key) if mode == 'major' else _get_dominant_key(base_key)
    progression = []
    prev_pitches = None

    # Fase 1: 2 compases en tonalidad original
    FALLBACK_MAJOR = ['I','IV','V','I']
    FALLBACK_MINOR = ['i','iv','V','i']
    base_fns = source_functions if len(source_functions) >= 4 else \
               (FALLBACK_MAJOR if mode == 'major' else FALLBACK_MINOR)

    for m in range(min(2, n_measures)):
        fig = base_fns[m % len(base_fns)]
        pitches = _build_chord_pitches(fig, base_key, prev_pitches, 'chords')
        tonic_pc = pitch.Pitch(base_key.tonic.name).pitchClass
        deg_map  = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
        base_fig = fig.replace('7','').replace('°','').replace('+','')
        semi     = deg_map.get(base_fig, (0,'M'))[0]
        progression.append({'function':fig,'pitches':pitches,
                            'root':(tonic_pc+semi)%12+48,'key':base_key,'tension':0.3})
        prev_pitches = pitches

    # Fase 2: n-4 compases en nueva tonalidad
    new_mode = new_key.mode
    new_fns  = FALLBACK_MAJOR if new_mode == 'major' else FALLBACK_MINOR
    for m in range(2, max(2, n_measures - 2)):
        fig = new_fns[(m-2) % len(new_fns)]
        pitches = _build_chord_pitches(fig, new_key, prev_pitches, 'chords')
        tonic_pc = pitch.Pitch(new_key.tonic.name).pitchClass
        deg_map  = MAJOR_SCALE_DEGREES if new_mode == 'major' else MINOR_SCALE_DEGREES
        base_fig = fig.replace('7','').replace('°','').replace('+','')
        semi     = deg_map.get(base_fig, (0,'M'))[0]
        progression.append({'function':fig,'pitches':pitches,
                            'root':(tonic_pc+semi)%12+48,'key':new_key,'tension':0.5})
        prev_pitches = pitches

    # Fase 3: 2 compases de retorno a la tónica original
    RETURN_MAJOR = ['V','I']
    RETURN_MINOR = ['V','i']
    ret_fns = RETURN_MAJOR if mode == 'major' else RETURN_MINOR
    for m in range(max(0, n_measures - 2), n_measures):
        fig = ret_fns[(m - (n_measures-2)) % len(ret_fns)]
        pitches = _build_chord_pitches(fig, base_key, prev_pitches, 'chords')
        tonic_pc = pitch.Pitch(base_key.tonic.name).pitchClass
        deg_map  = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
        base_fig = fig.replace('7','').replace('°','').replace('+','')
        semi     = deg_map.get(base_fig, (0,'M'))[0]
        progression.append({'function':fig,'pitches':pitches,
                            'root':(tonic_pc+semi)%12+48,'key':base_key,'tension':0.7})
        prev_pitches = pitches

    return progression, new_key


# ─── Bajo con groove (#H) ─────────────────────────────────────

def _generate_bass_with_groove(chord_prog, target_key, n_measures, ts_num,
                                groove_map=None, section_type='A',
                                prev_bass_pitch=None):
    """
    Genera línea de bajo aplicando el groove map del DNA fuente (#H).
    """
    bass_out = []
    lo, hi   = INSTRUMENT_RANGES['bass']

    if prev_bass_pitch is None:
        prev_bass_pitch = pitch.Pitch(target_key.tonic.name).pitchClass + 36

    for m in range(n_measures):
        m_off   = m * ts_num
        chord_d = chord_prog[m % len(chord_prog)] if chord_prog else None
        if not chord_d:
            bass_out.append(note.Rest(quarterLength=ts_num)); continue

        root = chord_d.get('root', 48)
        while root > hi: root -= 12
        while root < lo: root += 12
        # Voice leading mínimo
        while root - prev_bass_pitch > 6:  root -= 12
        while prev_bass_pitch - root > 6:  root += 12
        root = max(lo, min(hi, root))

        # Próxima raíz para leading tone
        next_root = chord_prog[(m+1) % len(chord_prog)].get('root', root) \
                    if chord_prog and m < n_measures-1 else root
        while next_root > hi: next_root -= 12
        while next_root < lo: next_root += 12
        leading = _snap_to_scale(
            root + int(np.sign(next_root-root)) * min(2, abs(next_root-root)), target_key)
        leading = max(lo, min(hi, leading))

        if ts_num >= 4:
            beats = [(0.0, root, 2.0), (2.0, leading, 2.0)]
        elif ts_num == 3:
            beats = [(0.0, root, 2.0), (2.0, leading, 1.0)]
        else:
            beats = [(0.0, root, float(ts_num))]

        for beat_off, midi, dur in beats:
            # #H Aplicar groove timing
            groove_dev = groove_map.get_offset(beat_off, ts_num) if groove_map else 0.0
            actual_off = max(0, m_off + beat_off + groove_dev)

            base_vel = 75 if beat_off < 0.05 else 62
            vel = groove_map.get_velocity(beat_off, base_vel, ts_num) if groove_map else base_vel

            n = note.Note(midi)
            n.quarterLength = dur
            n.offset = actual_off
            n.volume.velocity = int(np.clip(vel, 20, 110))
            bass_out.append(n)

        prev_bass_pitch = root

    return bass_out, prev_bass_pitch


# ═══════════════════════════════════════════════════════════════
#  CONSTRUCTOR DE FORMA MACRO (v3)
# ═══════════════════════════════════════════════════════════════

def _build_progression_from_dna(dna, target_key, n_measures, modulate=False):
    """Construye una progresión en target_key con voice leading."""
    functions = dna.harmony_dna.get('functions', [])
    mode      = target_key.mode
    FALLBACK  = {
        'major': [['I','V','vi','IV'],['I','IV','V','I'],['I','vi','IV','V'],['ii','V','I','I']],
        'minor': [['i','VI','III','VII'],['i','iv','V','i'],['i','VII','VI','VII'],['i','iv','VII','III']],
    }
    if functions and len(functions) >= 4:
        source_seq = functions
    else:
        source_seq = random.choice(FALLBACK.get(mode, FALLBACK['major']))

    progression  = []
    prev_pitches = None

    for m in range(n_measures):
        fig = source_seq[m % len(source_seq)]
        p   = _build_chord_pitches(fig, target_key, prev_pitches, 'chords')
        tc  = pitch.Pitch(target_key.tonic.name).pitchClass
        deg = MAJOR_SCALE_DEGREES if mode=='major' else MINOR_SCALE_DEGREES
        bf  = fig.replace('7','').replace('°','').replace('+','')
        semi= deg.get(bf,(0,'M'))[0]
        t_v = dna.harmony_dna.get('tension_curve',[0.3])
        t   = t_v[m % len(t_v)] if t_v else 0.3
        progression.append({'function':fig,'pitches':p,'root':(tc+semi)%12+48,
                             'tension':t,'key':target_key})
        prev_pitches = p

    return progression

def _add_cadence(prog, cadence_type, target_key, position='end'):
    cad = CADENCE_TYPES.get(cadence_type, ['V','I'])
    mode = target_key.mode
    if mode == 'minor':
        cad = [{'I':'i','IV':'iv','ii':'ii°'}.get(f,f) for f in cad]
    n = len(prog)
    if position == 'end' and n >= 2:
        prev = prog[-3]['pitches'] if n >= 3 else None
        for k,func in enumerate(cad):
            idx = n - len(cad) + k
            if 0 <= idx < n:
                p = _build_chord_pitches(func, target_key, prev, 'chords')
                prog[idx]['function'] = func; prog[idx]['pitches'] = p; prev = p
    elif position == 'half' and n >= 4:
        mid = n//2
        for k,func in enumerate(cad):
            idx = mid-1+k
            if 0 <= idx < n:
                p = _build_chord_pitches(func, target_key, None, 'chords')
                prog[idx]['function'] = func; prog[idx]['pitches'] = p
    return prog

def _insert_modal_chord(prog, tension_curve, target_key):
    mode = target_key.mode
    SUBS = {
        'major': {'IV':'iv','V':'bVII','vi':'bVI','I':'bIII','ii':'bII'},
        'minor': {'v':'V','VII':'bVII','III':'bIII','i':'I'},
    }
    subs = SUBS.get(mode, {})
    for i,cd in enumerate(prog):
        t = tension_curve[i] if i < len(tension_curve) else 0.3
        if t > 0.65 and random.random() < 0.4:
            bf = cd['function'].replace('7','').replace('°','')
            if bf in subs:
                nf = subs[bf]
                prev = prog[i-1]['pitches'] if i > 0 else None
                np_ = _build_chord_pitches(nf, target_key, prev, 'chords')
                prog[i]['function'] = nf; prog[i]['pitches'] = np_
                prog[i]['is_modal'] = True
    return prog

def _build_macro_form(dna_list, target_key):
    by_e = sorted(enumerate(dna_list), key=lambda x: x[1].energy_dna.get('mean',0))
    templates = {
        1: [('intro',4),('A',12),("A'",8),('coda',4)],
        2: [('intro',4),('A',10),('B',8),("A'",8),('coda',4)],
        3: [('intro',4),('A',8),('B',8),("A'",8),('C',6),('coda',4)],
    }
    tpl = templates.get(min(3,len(dna_list)), templates[2])
    dc  = [d for _,d in by_e]
    sections = []
    for i,(name,nm) in enumerate(tpl):
        if name == 'intro':   sd = dc[0]
        elif name == 'coda':  sd = dc[0]
        elif name in ('A',"A'"): sd = dc[min(1,len(dc)-1)]
        elif name == 'B':     sd = dc[min(len(dc)-1, len(dc)//2)]
        else:                 sd = dc[i % len(dc)]
        sections.append({'name':name,'n_measures':nm,'dna':sd,'is_varied':name=="A'"})
    return sections

# ═══════════════════════════════════════════════════════════════
#  MOTOR DE COMBINACIÓN v3
# ═══════════════════════════════════════════════════════════════

def combine_dna(dnas, mode='auto', target_key=None, target_tempo=None, verbose=False):
    dnas = [d for d in dnas if d is not None]
    if not dnas: raise ValueError("Sin ADN válido")

    print(f"\n🔀 Combinando {len(dnas)} piezas | modo: {mode.upper()}")

    bpm = target_tempo if target_tempo else \
          dnas[np.argmax([d.energy_dna.get('mean',0.5) for d in dnas])].tempo_bpm

    if target_key:
        p = target_key.split()
        tko = key.Key(p[0], p[1] if len(p)>1 else 'major')
    else:
        tko = dnas[np.argmax([len(d.harmony_dna.get('progressions',[])) for d in dnas])].key_obj \
              or key.Key('C','major')

    if mode == 'auto':
        mode = _auto_select_mode(dnas)
        print(f"  🤖 Modo auto → {mode.upper()}")

    print(f"  📐 {tko.tonic.name} {tko.mode} | {bpm:.0f} BPM")
    return _build_score_v3(dnas, mode, tko, bpm)


def _auto_select_mode(dnas):
    e = [d.energy_dna.get('mean',0.5) for d in dnas]
    c = [d.harmony_dna.get('complexity',0) for d in dnas]
    r = [d.rhythm_dna.get('density',0) for d in dnas]
    # Si todos tienen fragmentos → mosaic
    if all(len(d.fragments) > 2 for d in dnas): return 'mosaic'
    if max(e)-min(e) > 0.3: return 'energy'
    if max(c)-min(c) > 0.3: return 'harmony'
    if max(r)-min(r) > 5:   return 'rhythm'
    return 'harmony'


def _build_score_v3(dnas, mode, target_key, bpm):
    """Motor principal v3: construye el score completo con forma macro."""

    by_harmony = sorted(dnas, key=lambda d: d.harmony_dna.get('complexity',0), reverse=True)
    by_rhythm  = sorted(dnas, key=lambda d: d.rhythm_dna.get('density',0), reverse=True)
    by_melody  = sorted(dnas, key=lambda d: d.melody_dna.get('range_semitones',0), reverse=True)
    by_energy  = sorted(dnas, key=lambda d: d.energy_dna.get('mean',0.5), reverse=True)
    by_frags   = sorted(dnas, key=lambda d: len(d.fragments), reverse=True)

    harmony_src = by_harmony[0]
    rhythm_src  = by_rhythm[min(1,len(by_rhythm)-1)]
    melody_src  = by_melody[0]
    energy_src  = by_energy[0]
    frag_src    = by_frags[0]

    if mode == 'rhythm':
        melody_src = by_melody[0]
        rhythm_src = dnas[0]
    elif mode == 'energy':
        rhythm_src = energy_src
    elif mode == 'mosaic':
        frag_src = by_frags[0]

    print(f"  ♬ Armonía  → {harmony_src.name}")
    print(f"  🥁 Ritmo   → {rhythm_src.name} (groove:{rhythm_src.groove_map.trained})")
    print(f"  🎵 Melodía → {melody_src.name} (markov:{sum(len(v) for v in melody_src.markov.transitions.values())} trans.)")
    print(f"  🧩 Fragmentos → {frag_src.name} ({len(frag_src.fragments)} frags)")

    ts_num, ts_den = harmony_src.time_signature
    sections = _build_macro_form(dnas, target_key)

    print(f"\n  📋 Forma: " + " → ".join(f"{s['name']}({s['n_measures']}c)" for s in sections))

    result   = stream.Score()
    result.insert(0, tempo.MetronomeMark(number=bpm))
    result.insert(0, target_key)

    mel_part = stream.Part(id='Melody')
    mel_part.insert(0, instrument.Piano())
    mel_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    mel_part.insert(0, clef.TrebleClef())

    cp_part  = stream.Part(id='Counterpoint')
    cp_part.insert(0, instrument.Piano())
    cp_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    cp_part.insert(0, clef.TrebleClef())

    acc_part = stream.Part(id='Accompaniment')
    acc_part.insert(0, instrument.Piano())
    acc_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    acc_part.insert(0, clef.BassClef())

    bass_part = stream.Part(id='Bass')
    bass_part.insert(0, instrument.AcousticBass())
    bass_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    bass_part.insert(0, clef.BassClef())

    global_offset    = 0.0
    prev_bass_pitch  = None
    global_measure   = 0
    all_melody_notes = []   # para ornamentación y articulación posterior

    for sec in sections:
        sname    = sec['name']
        n_meas   = sec['n_measures']
        sec_dna  = sec['dna']
        is_var   = sec['is_varied']
        sec_off  = global_offset

        print(f"    ▸ {sname}  {n_meas}c | dna={sec_dna.name}")

        # Asignar fuentes por modo y sección
        if mode == 'emotion':
            sh = sr = sm = sec_dna
        else:
            sh = harmony_src
            sr = rhythm_src
            sm = melody_src

        # ── 1. Progresión armónica
        is_B_section = (sname == 'B')
        if is_B_section:
            # #G Modulación en sección B
            prog, mod_key = _build_modulated_progression(
                target_key, None, n_meas, sh.harmony_dna.get('functions',[]))
            print(f"      🎼 Módula a: {mod_key.tonic.name} {mod_key.mode}")
        else:
            prog     = _build_progression_from_dna(sh, target_key, n_meas)
            mod_key  = target_key

        # Cadencias (#5)
        prog = _add_cadence(prog, 'HC', target_key, 'half')
        prog = _add_cadence(prog, 'AC', target_key, 'end')

        # Intercambio modal (#10)
        if sh.modal_dna.get('uses_modal',False) or sh.tension_dna.get('mean',0.3) > 0.5:
            prog = _insert_modal_chord(prog, sh.tension_dna.get('curve',[]), target_key)

        # Curvas de tensión/energía de la sección
        t_curve = sh.tension_dna.get('curve', [])
        e_curve = sh.energy_dna.get('curve', [])

        # Puntos de novedad
        novelty_ms = [m for m in sh.novelty_dna.get('peak_measures',[]) if m < n_meas]

        # ── 2. Melodía: mosaic o Markov según modo
        if mode == 'mosaic' and frag_src.fragments:
            mel_notes = _build_mosaic_melody(
                dnas if mode == 'mosaic' else [sm],
                prog, target_key, n_meas, ts_num,
                tension_curve=t_curve, energy_curve=e_curve,
            )
            if not mel_notes:
                mel_notes = _build_markov_melody(
                    [sm.markov], prog, target_key, n_meas, ts_num,
                    section_type=sname, motif_dna=sm.motif_dna,
                    novelty_measures=novelty_ms, variation=is_var,
                )
        else:
            mel_notes = _build_markov_melody(
                [sm.markov], prog, target_key, n_meas, ts_num,
                section_type=sname, motif_dna=sm.motif_dna,
                novelty_measures=novelty_ms, variation=is_var,
            )

        # ── 3. Contrapunto (#E)
        cp_notes = _generate_counterpoint(mel_notes, prog, target_key, n_meas, ts_num)

        # ── 4. Acompañamiento: patrón real (#F) con groove (#H)
        acc_notes = []
        if sr.acc_pattern.trained:
            for m in range(n_meas):
                cd = prog[m % len(prog)]
                events = sr.acc_pattern.render(
                    cd['pitches'], sec_off + m*ts_num, ts_num,
                    groove_map=sr.groove_map if sr.groove_map.trained else None
                )
                acc_notes.extend(events)
        else:
            acc_notes = _fallback_accompaniment(sr.rhythm_dna, prog, target_key,
                                                 n_meas, ts_num, sname,
                                                 sr.groove_map if sr.groove_map.trained else None)

        # ── 5. Bajo con groove (#H)
        bass_notes, prev_bass_pitch = _generate_bass_with_groove(
            prog, target_key, n_meas, ts_num,
            groove_map=sr.groove_map if sr.groove_map.trained else None,
            section_type=sname, prev_bass_pitch=prev_bass_pitch,
        )

        # ── Aplicar offsets globales y acumular
        for el in mel_notes:
            el.offset = float(el.offset) + sec_off
        for el in cp_notes:
            if hasattr(el,'offset'): el.offset = float(el.offset) + sec_off
        for el in acc_notes:
            if hasattr(el,'offset'): el.offset = float(el.offset) + sec_off
        for el in bass_notes:
            if hasattr(el,'offset'): el.offset = float(el.offset) + sec_off

        all_melody_notes.extend(mel_notes)

        for el in mel_notes:  mel_part.append(el)
        for el in cp_notes:   cp_part.append(el)
        for el in acc_notes:  acc_part.append(el)
        for el in bass_notes: bass_part.append(el)

        global_offset  += n_meas * ts_num
        global_measure += n_meas

    # ── Post-proceso global: ornamentación (#C) y articulación (#D)
    print(f"\n  🎨 Añadiendo ornamentación ({melody_src.style})...")
    all_melody_notes = _add_ornamentation(
        sorted(all_melody_notes, key=lambda n: float(n.offset)),
        target_key, style=melody_src.style, density=0.25
    )
    print(f"  ✍️  Añadiendo articulación...")
    all_melody_notes = _add_articulation(
        all_melody_notes, style=melody_src.style,
        energy_curve=energy_src.energy_dna.get('curve',[]), ts_num=ts_num
    )

    # Reconstruir la parte de melodía con las notas ornamentadas
    mel_part = stream.Part(id='Melody')
    mel_part.insert(0, instrument.Piano())
    mel_part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
    mel_part.insert(0, clef.TrebleClef())
    for n in sorted(all_melody_notes, key=lambda x: float(x.offset)):
        mel_part.append(n)

    result.append(mel_part)
    result.append(cp_part)
    result.append(acc_part)
    result.append(bass_part)

    return result


def _fallback_accompaniment(rhythm_dna, chord_prog, target_key, n_measures, ts_num,
                              section_type, groove_map=None):
    """Acompañamiento de fallback cuando no hay patrón real."""
    acc_out = []
    density  = rhythm_dna.get('density', 4)
    has_trip = rhythm_dna.get('has_triplets', False)

    if section_type in ('intro','coda'): acc_type = 'arpeggio'
    elif section_type == 'B':            acc_type = 'offbeat'
    elif density > 10:                   acc_type = 'alberti'
    elif density > 6:                    acc_type = 'offbeat'
    else:                                acc_type = 'block'

    prev_ch = None
    for m in range(n_measures):
        m_off  = m * ts_num
        cd     = chord_prog[m % len(chord_prog)] if chord_prog else None
        if not cd or not cd.get('pitches'): continue
        cp = _voice_lead(prev_ch, cd['root']%12, 'M', target_key, 'chords') \
             if prev_ch else cd['pitches']
        prev_ch = cp
        cs = sorted(cp)

        if acc_type == 'block':
            ch = chord.Chord(cs[:4])
            ch.quarterLength = ts_num; ch.offset = m_off
            for nn in ch.notes:
                nn.volume.velocity = groove_map.get_velocity(0, 58, ts_num) if groove_map else 58
            acc_out.append(ch)

        elif acc_type == 'offbeat':
            beats = [0.5 + b for b in range(int(ts_num))]
            for b in beats:
                if b < ts_num:
                    dev = groove_map.get_offset(b, ts_num) if groove_map else 0
                    ch  = chord.Chord(cs[:3])
                    ch.quarterLength = 0.5
                    ch.offset = m_off + b + dev
                    for nn in ch.notes:
                        nn.volume.velocity = groove_map.get_velocity(b, 52, ts_num) if groove_map else 52
                    acc_out.append(ch)

        elif acc_type == 'alberti':
            if len(cs) >= 3:
                pat = [cs[0],cs[-1],cs[1],cs[-1]]
            else:
                pat = cs * 2
            step = ts_num / len(pat)
            for i,p in enumerate(pat):
                dev = groove_map.get_offset(i*step, ts_num) if groove_map else 0
                n   = note.Note(p)
                n.quarterLength = step; n.offset = m_off + i*step + dev
                n.volume.velocity = groove_map.get_velocity(i*step, 52, ts_num) if groove_map else 52
                acc_out.append(n)

        elif acc_type == 'arpeggio':
            step = max(0.5, ts_num/max(1,len(cs)))
            for i,p in enumerate(cs):
                if i*step >= ts_num: break
                n = note.Note(p)
                n.quarterLength = min(step, ts_num - i*step)
                n.offset = m_off + i*step
                n.volume.velocity = 50
                acc_out.append(n)

    return acc_out


# ═══════════════════════════════════════════════════════════════
#  POSTPROCESADO Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

def humanize_score(score_obj, groove_maps=None, ts_num=4, amount=0.015):
    """
    #H: Humaniza usando el groove map si está disponible,
    o variación gaussiana en caso contrario.
    """
    print("  🎭 Humanizando con groove...")
    for el in score_obj.flat.notes:
        if hasattr(el,'volume') and el.volume.velocity:
            v = el.volume.velocity
            el.volume.velocity = max(15, min(127, v + int(random.gauss(0, v*0.06))))

        if hasattr(el,'offset'):
            off = float(el.offset)
            if groove_maps:
                best_gm = max(groove_maps, key=lambda g: g.trained)
                if best_gm.trained:
                    beat_pos = off % ts_num
                    dev = best_gm.get_offset(beat_pos, ts_num) * 0.5  # 50% del groove
                    el.offset = max(0, off + dev)
                    continue
            el.offset = max(0, off + random.gauss(0, amount))
    return score_obj


def add_expression(score_obj, energy_curve):
    if not energy_curve: return score_obj
    DYN = {0.0:'ppp',0.12:'pp',0.25:'p',0.4:'mp',0.55:'mf',0.7:'f',0.85:'ff',1.0:'fff'}
    for part in score_obj.parts:
        measures = list(part.getElementsByClass('Measure'))
        if not measures: continue
        for i,m in enumerate(measures):
            if i < len(energy_curve):
                e = energy_curve[i]
                t = min(DYN.keys(), key=lambda x: abs(x-e))
                m.insert(0, dynamics.Dynamic(DYN[t]))
    return score_obj


def validate_and_fix_score(score_obj):
    """#11: Validación completa de rangos, cruce de voces e intervalos."""
    print("  🔧 Validando (#11)...")
    RANGES = {'Melody':INSTRUMENT_RANGES['melody'],'Counterpoint':INSTRUMENT_RANGES['tenor'],
              'Accompaniment':INSTRUMENT_RANGES['chords'],'Bass':INSTRUMENT_RANGES['bass']}
    mel_pitches = {}
    for part in score_obj.parts:
        pid = part.id or ''
        lo,hi = RANGES.get(pid,(28,108))
        ns = [n for n in part.flat.notes if n.isNote]
        if pid == 'Melody':
            for n in ns: mel_pitches[round(float(n.offset),2)] = n.pitch.midi
        seen = {}
        for n in ns:
            ok = round(float(n.offset),3)
            if ok in seen:
                try: part.remove(n)
                except: pass
                continue
            seen[ok] = n
            while n.pitch.midi > hi: n.pitch.midi -= 12
            while n.pitch.midi < lo: n.pitch.midi += 12
        if pid == 'Bass':
            for n in ns:
                ok = round(float(n.offset),2)
                mp = mel_pitches.get(ok)
                if mp and n.pitch.midi >= mp - 5:
                    n.pitch.midi = max(lo, mp - 12)
        if pid == 'Melody':
            sn = sorted(ns, key=lambda n: float(n.offset))
            for i in range(len(sn)-1):
                diff = abs(sn[i+1].pitch.midi - sn[i].pitch.midi)
                if diff > 14:
                    if sn[i+1].pitch.midi > sn[i].pitch.midi:
                        sn[i+1].pitch.midi = sn[i].pitch.midi + min(7, diff//2)
                    else:
                        sn[i+1].pitch.midi = sn[i].pitch.midi - min(7, diff//2)
                    sn[i+1].pitch.midi = max(lo, min(hi, sn[i+1].pitch.midi))
    print("  ✅ Partitura validada")
    return score_obj


def score_candidate(score_obj, dnas):
    """Puntúa un candidato por consonancia, variedad rítmica y rango."""
    ns = [n for n in score_obj.flat.notes if n.isNote]
    if not ns: return 0.0
    dna = dnas[0]
    ko  = dna.key_obj or key.Key('C','major')
    pcs = set(_get_scale_pcs(ko))
    tc  = pitch.Pitch(ko.tonic.name).pitchClass
    in_scale = sum(1 for n in ns if (n.pitch.pitchClass-tc)%12 in pcs)
    consonance = in_scale / len(ns)
    durs = [float(n.quarterLength) for n in ns]
    variety = min(1.0, len(set(round(d,2) for d in durs)) / 6)
    pitches = [n.pitch.midi for n in ns]
    rng_score = min(1.0, (max(pitches)-min(pitches)) / 24)
    # Bonus si tiene contrapunto activo
    cp_ns = [n for n in score_obj.parts if n.id == 'Counterpoint']
    cp_bonus = 0.05 if cp_ns else 0
    return consonance*0.45 + variety*0.3 + rng_score*0.2 + cp_bonus


# ═══════════════════════════════════════════════════════════════
#  RESUMEN Y MAIN
# ═══════════════════════════════════════════════════════════════

def print_dna_summary(dnas):
    print("\n" + "═"*80)
    print("  📊 ADN MUSICAL COMPARATIVO  v3")
    print("═"*80)
    print(f"  {'Pieza':<20} {'Key':<10} {'BPM':>5} {'Dens':>6} {'Rango':>6} "
          f"{'Harm':>5} {'E':>5} {'T':>5} {'Val':>5} {'Frags':>6} {'Style':<10}")
    print("  " + "─"*80)
    for d in dnas:
        nm = d.name[:18]+"…" if len(d.name)>19 else d.name
        print(f"  {nm:<20} "
              f"{d.key_tonic+' '+d.key_mode[:3]:<10} "
              f"{d.tempo_bpm:>5.0f} "
              f"{d.rhythm_dna.get('density',0):>6.1f} "
              f"{d.melody_dna.get('range_semitones',0):>6} "
              f"{d.harmony_dna.get('complexity',0):>5.2f} "
              f"{d.energy_dna.get('mean',0):>5.2f} "
              f"{d.tension_dna.get('mean',0):>5.2f} "
              f"{d.emotion_dna.get('valence',0):>5.2f} "
              f"{len(d.fragments):>6} "
              f"{d.style:<10}")
    print("═"*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="🧬 MIDI DNA Combiner v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('midi_files', nargs='+')
    parser.add_argument('--output',      default='combined_output.mid')
    parser.add_argument('--mode',        default='auto',
        choices=['auto','harmony','rhythm','energy','emotion','mosaic'])
    parser.add_argument('--tempo',       type=float, default=None)
    parser.add_argument('--key',         default=None)
    parser.add_argument('--candidates',  type=int,   default=3)
    parser.add_argument('--export-xml',  action='store_true')
    parser.add_argument('--no-humanize', action='store_true')
    parser.add_argument('--verbose',     action='store_true')
    args = parser.parse_args()

    print("\n╔" + "═"*72 + "╗")
    print("║" + "  🧬  MIDI DNA COMBINER v3.0  —  Fusión Musical de Alta Calidad  ".center(72) + "║")
    print("╚" + "═"*72 + "╝")

    for f in args.midi_files:
        if not os.path.exists(f):
            print(f"❌ No encontrado: {f}"); sys.exit(1)

    # FASE 1: Extracción
    print(f"\n🔬 FASE 1: Extracción del ADN ({len(args.midi_files)} archivos)")
    print("═"*60)
    dnas = [extract_dna(f, verbose=args.verbose) for f in args.midi_files]
    dnas = [d for d in dnas if d is not None]
    if not dnas:
        print("❌ Sin ADN"); sys.exit(1)
    print_dna_summary(dnas)

    # FASE 2: Combinación con candidatos
    print("🔀 FASE 2: Combinación del ADN")
    print("═"*60)

    n_cands = max(1, args.candidates)
    best_score, best_result = -1, None

    for c in range(n_cands):
        if n_cands > 1:
            print(f"\n  🎲 Candidato {c+1}/{n_cands}")
        random.seed(42 + c*17); np.random.seed(42 + c*17)

        result = combine_dna(
            dnas, mode=args.mode,
            target_key=args.key, target_tempo=args.tempo, verbose=args.verbose,
        )
        sc = score_candidate(result, dnas)
        print(f"    📊 Score: {sc:.3f}")
        if sc > best_score:
            best_score = sc; best_result = result

    print(f"\n  🏆 Mejor candidato: score={best_score:.3f}")

    # FASE 3: Postprocesado
    print("\n✨ FASE 3: Postprocesado")
    print("═"*60)

    if not args.no_humanize:
        groove_maps = [d.groove_map for d in dnas if d.groove_map.trained]
        ts_num = dnas[0].time_signature[0]
        best_result = humanize_score(best_result, groove_maps or None, ts_num)

    best_result = validate_and_fix_score(best_result)

    energy_src  = max(dnas, key=lambda d: d.energy_dna.get('peak',0))
    energy_curve= energy_src.energy_dna.get('curve',[])
    if energy_curve:
        best_result = add_expression(best_result, energy_curve)

    # FASE 4: Exportar
    print(f"\n💾 FASE 4: Exportando")
    print("═"*60)

    out = args.output
    try:
        best_result.write('midi', fp=out)
        print(f"  ✅ MIDI: {out}")
    except Exception as e:
        out = 'output_combined.mid'
        best_result.write('midi', fp=out)
        print(f"  ✅ MIDI: {out}")

    if args.export_xml:
        xml = out.replace('.mid','.xml').replace('.midi','.xml')
        if not xml.endswith('.xml'): xml += '.xml'
        try:
            best_result.write('musicxml', fp=xml)
            print(f"  ✅ MusicXML: {xml}")
        except Exception as e:
            print(f"  ⚠️  MusicXML: {e}")

    print("\n" + "═"*60)
    print("  🎉 COMPLETADO")
    print(f"  📁 {out}  |  Score: {best_score:.3f}")
    print(f"  🔀 Modo: {args.mode.upper()}  |  Candidatos: {n_cands}")
    print(f"  🎵 {' + '.join(d.name for d in dnas)}")
    print("  🆕 v3: Mosaic·Markov·Ornamentación·Articulación·Contrapunto·")
    print("         PatrónReal·Modulación·Groove")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
