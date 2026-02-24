"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               MIDI DNA UNIFIED MIXER  v1.1                                  ║
║   Fusión de lo mejor de tres generaciones de mezcladores MIDI               ║
║                                                                              ║
║  CARACTERÍSTICAS:                                                            ║
║  [A] Extracción rítmica avanzada: histograma 16 subdivisiones, síncopa,     ║
║      accent weights, subdivisión primaria                                    ║
║  [B] Curvas emocionales completas: Lerdahl, Tonnetz, Valencia, Arousal,     ║
║      Actividad, Estabilidad, Entropía, clasificación de arco                ║
║  [C] Estructura formal automática: segmentación A/B/C, frases, cadencias   ║
║  [D] Markov de 2º orden (intervalo × duración) para melodía estilística    ║
║  [E] Mosaic splicing: fragmentos reales transpuestos                        ║
║  [F] Ornamentación idiomática: mordentes, apoyaturas, notas de paso        ║
║      → Aplicada también en contrapunto y voces dobles                       ║
║  [G] Contrapunto multi-especie: 1ª/2ª/3ª especie según densidad emocional  ║
║      Resolución automática de disonancias por grado conjunto                ║
║  [H] Groove map: timing humanizado extraído del MIDI fuente                ║
║  [I] Patrón de acompañamiento real detectado del MIDI fuente               ║
║  [J] Sequitur grammar: frases repetidas clasificadas por calidad musical    ║
║  [K] Voice-leading riguroso: sin paralelas, movimiento mínimo              ║
║  [L] Bajo Alberti/arpegio/bloque elegido por energía y tensión             ║
║  [M] Puentes pivot-chord entre secciones                                    ║
║  [N] Candidatos múltiples con scoring de calidad musical extendido:         ║
║      consonancia + variedad + rango + continuidad + arco dinámico          ║
║  [O] EmotionalController: parámetros compás-a-compás desde curvas          ║
║  [P] FormGenerator: estructura formal heredada y escalada                   ║
║  [Q] Modulación real por sección: relativo en B, dominante en C            ║
║  [R] Export MusicXML opcional                                               ║
║  [S] Silencios estructurales de frase en cadencias                          ║
║      CC#64 sustain pedal + CC#11 expression curve en todas las pistas       ║
║      Acordes de 7ª y 9ª automáticos según harmony_complexity               ║
║  [T] Walking bass melódico: notas de paso, 5ª/7ª, aproximación cromática   ║
║  [U] Percusión generada desde rhythm_grid: kick, snare, hi-hat, crash      ║
║  [V] Export split-tracks: una pista por fichero para DAW                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    python midi_dna_unified.py midi1.mid midi2.mid [midi3.mid ...] [opciones]

    También con roles explícitos:
    python midi_dna_unified.py --melody a.mid --harmony b.mid --rhythm c.mid

OPCIONES PRINCIPALES:
    --mode          auto | rhythm_melody | harmony_melody | full_blend |
                    custom | mosaic | energy | emotion
    --emotion_src N Índice del MIDI que dona el arco emocional (default: 0)
    --form_src N    Índice del MIDI que dona la estructura formal (default: 0)
    --sources       Para modo custom: rhythm=N,melody=N,harmony=N
    --key           Tonalidad destino ("C major", "A minor", …)
    --bars N        Número de compases a generar (default: 16)
    --tempo BPM     BPM (default: detectado del primer MIDI)
    --form FORM     Forma musical: AABA, ABAC, ABAB… (default: auto)
    --surprise R    Tasa de cromatismo sorpresa 0-1 (default: 0.08)
    --rhythm_strength F  Fuerza del groove 0-2 (default: 1.0)
    --candidates N  Generar N candidatos y elegir el mejor (default: 1)
    --no-humanize   Desactivar humanización de groove
    --no-percussion Desactivar pista de percusión automática
    --split-tracks  Exportar cada pista como fichero MIDI independiente
    --export-xml    Exportar también en MusicXML
    --verbose       Informe detallado del ADN
    --output FILE   Archivo de salida (default: output_unified.mid)
    --seed N        Semilla aleatoria (default: 42)

EJEMPLOS:

  ── Uso básico ──────────────────────────────────────────────────────────────
    # Mezcla automática de dos MIDIs, resultado en archivo por defecto
    python midi_dna_unified.py bach.mid jazz.mid

    # Modo mosaico con informe detallado del ADN
    python midi_dna_unified.py bach.mid jazz.mid --mode mosaic --verbose

    # Mezcla de tres fuentes, 32 compases, armonía+melodía
    python midi_dna_unified.py a.mid b.mid c.mid --mode harmony_melody --bars 32

    # Arco emocional de b.mid, forma AABA, tonalidad destino La menor
    python midi_dna_unified.py a.mid b.mid --mode emotion \
        --key "A minor" --form AABA --emotion_src 1

    # Fuentes explícitas para modo custom
    python midi_dna_unified.py a.mid b.mid --mode custom \
        --sources rhythm=0,melody=1,harmony=0

    # Roles explícitos por fichero
    python midi_dna_unified.py \
        --melody a.mid --harmony b.mid --rhythm c.mid --form ABAB

  ── Tempo y semilla ─────────────────────────────────────────────────────────
    # Fijar tempo a 120 BPM y semilla reproducible
    python midi_dna_unified.py a.mid b.mid --tempo 120 --seed 7

    # Resultado determinista para comparación de modos
    python midi_dna_unified.py a.mid b.mid --mode full_blend \
        --bars 24 --tempo 96 --seed 0 --output mezcla_96bpm.mid

  ── Estilo de acompañamiento ────────────────────────────────────────────────
    # Forzar bajo Alberti en todos los compases
    python midi_dna_unified.py a.mid b.mid --acc-style alberti

    # Arpegio ascendente, sin humanización de groove
    python midi_dna_unified.py a.mid b.mid --acc-style arpeggio --no-humanize

    # Vals (3/4), acompañamiento en bloque con arch emocional
    python midi_dna_unified.py valse.mid otra.mid \
        --acc-style waltz --mode emotion --bars 32

  ── Voces extra ─────────────────────────────────────────────────────────────
    # Cuarteto de cuerda completo
    python midi_dna_unified.py a.mid b.mid \
        --voices "violin,viola:inner,cello:inner,contrabass:bass_double"

    # Orquestación clásica de cámara
    python midi_dna_unified.py a.mid b.mid \
        --voices "flute:melody_double,oboe:counterpoint,horn:pedal,bassoon:bass_double"

    # Voces con vel_scale personalizado (0.0–1.0)
    python midi_dna_unified.py a.mid b.mid \
        --voices "violin:counterpoint:0.9,pad:pedal:0.5"

    # Acc-style + voces extra combinados
    python midi_dna_unified.py a.mid b.mid \
        --acc-style alberti --voices "flute,horn:pedal"

    # Ostinato de vibraphone + doblaje de bajo
    python midi_dna_unified.py a.mid b.mid \
        --voices "vibraphone:ostinato:0.7,contrabass:bass_double:0.85"

  ── Candidatos y calidad ────────────────────────────────────────────────────
    # Generar 5 candidatos y conservar el de mejor scoring
    python midi_dna_unified.py a.mid b.mid --candidates 5

    # Candidatos + cromtismo reducido + groove fuerte
    python midi_dna_unified.py a.mid b.mid \
        --candidates 3 --surprise 0.02 --rhythm_strength 1.5

    # Exploración creativa: mucho cromatismo, varios candidatos
    python midi_dna_unified.py a.mid b.mid \
        --candidates 8 --surprise 0.25 --mode mosaic --seed 99

  ── Percusión ───────────────────────────────────────────────────────────────
    # Percusión activada por defecto; desactivar si no se necesita
    python midi_dna_unified.py a.mid b.mid --no-percussion

    # Percusión + walking bass (jazz, acordes de 1 compás)
    python midi_dna_unified.py jazz_a.mid jazz_b.mid \
        --mode full_blend --acc-style arpeggio --bars 32

  ── Pistas separadas (split) ────────────────────────────────────────────────
    # Exportar cada pista como fichero independiente para DAW
    python midi_dna_unified.py a.mid b.mid --split-tracks \
        --output sesion.mid

    # Pipeline completo con split y XML
    python midi_dna_unified.py a.mid b.mid --split-tracks --export-xml \
        --output obra.mid

  ── Pipeline completo v1.1 ──────────────────────────────────────────────────
    # Máxima calidad: 5 candidatos, walking bass, percusión, acordes 7ª/9ª,
    # modulación real en sección B, ornamentación en todas las voces
    python midi_dna_unified.py bach.mid jazz.mid salsa.mid \
        --mode full_blend --bars 32 --tempo 112 --key "D major" \
        --form AABA --candidates 5 --rhythm_strength 1.2 --surprise 0.05 \
        --acc-style arpeggio \
        --voices "violin:melody_double:0.8,cello:inner:0.75,horn:pedal:0.6" \
        --export-xml --split-tracks --verbose --seed 42 --output obra_maestra.mid

DEPENDENCIAS:
    pip install music21 mido numpy scipy scikit-learn
"""

import sys
import os
import argparse
import random
import copy
import math
from collections import defaultdict, Counter
from fractions import Fraction

import numpy as np

# ── scipy (opcional) ──────────────────────────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    def gaussian_filter(arr, sigma=1):
        return arr
    def find_peaks(arr, **kw):
        return [], {}

# ── scikit-learn (opcional) ───────────────────────────────────────────────────
try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── music21 ───────────────────────────────────────────────────────────────────
try:
    from music21 import (
        converter, stream, note, chord, meter, tempo, key as m21key,
        instrument, harmony, roman, pitch, interval as m21interval,
        duration, analysis, expressions, articulations, spanner, clef,
        dynamics, scale, environment
    )
    environment.UserSettings()['warnings'] = 0
    environment.UserSettings()['autoDownload'] = 'deny'
except ImportError:
    print("ERROR: pip install music21")
    sys.exit(1)

# ── mido ──────────────────────────────────────────────────────────────────────
try:
    import mido
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

HARMONIC_FUNCTIONS = {
    "T":    ["I", "i", "vi", "VI"],
    "PD":   ["ii", "II", "iv", "IV"],
    "D":    ["V", "v", "vii°", "VII"],
    "Dsec": ["V/V", "V/ii", "V/vi", "V/IV"],
    "Other": []
}
FUNC_TENSION = {"T": 0.1, "PD": 0.4, "D": 0.7, "Dsec": 0.9, "Other": 0.5}

DISSONANCE_WEIGHTS = {
    0: 0.0, 1: 1.0, 2: 0.8, 3: 0.2, 4: 0.1,
    5: 0.05, 6: 0.9, 7: 0.02, 8: 0.25, 9: 0.15,
    10: 0.7, 11: 0.95
}
CONSONANCE = {0:1.0, 1:0.1, 2:0.2, 3:0.6, 4:0.7,
              5:0.8, 6:0.0, 7:0.9, 8:0.65, 9:0.75, 10:0.3, 11:0.15}

TONNETZ = {0:(0,0), 1:(-2.5,.866), 2:(2,0), 3:(-1.5,.866),
           4:(.5,.866), 5:(-1,0), 6:(2.5,.866), 7:(1,0),
           8:(-.5,.866), 9:(1.5,.866), 10:(-2,0), 11:(3.5,.866)}

MAJOR_SCALE_DEGREES = {
    'I':(0,'M'), 'ii':(2,'m'), 'iii':(4,'m'), 'IV':(5,'M'),
    'V':(7,'M'), 'V7':(7,'M7'), 'vi':(9,'m'), 'vii°':(11,'d'),
    'bVII':(10,'M'), 'bIII':(3,'M'), 'bVI':(8,'M'),
    'iv':(5,'m'), 'II':(2,'M'), 'bII':(1,'M'),
}
MINOR_SCALE_DEGREES = {
    'i':(0,'m'), 'ii°':(2,'d'), 'III':(3,'M'), 'iv':(5,'m'),
    'V':(7,'M'), 'V7':(7,'M7'), 'VI':(8,'M'), 'VII':(10,'M'),
    'vii°':(11,'d'), 'II':(2,'M'), 'IV':(5,'M'), 'bVII':(10,'M'), 'I':(0,'M'),
}

INSTRUMENT_RANGES = {
    'melody':  (60, 84),
    'tenor':   (48, 69),
    'bass':    (28, 52),
    'chords':  (48, 76),
}

CADENCE_TYPES = {
    'AC': ['V','I'], 'HC': ['I','V'], 'DC': ['V','vi'],
    'IAC': ['V','I'], 'PC': ['IV','I'],
}

VALID_DURATIONS_FLOAT = [
    1/6, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0
]

STRONG_BEATS = {4: {1, 3}, 3: {1}}

ORNAMENTATION_STYLES = {
    'baroque':   {'mordent':0.25, 'trill':0.10, 'passing':0.35, 'neighbor':0.20, 'appoggiatura':0.10},
    'classical': {'mordent':0.10, 'trill':0.08, 'passing':0.30, 'neighbor':0.25, 'appoggiatura':0.15},
    'romantic':  {'mordent':0.05, 'trill':0.05, 'passing':0.25, 'neighbor':0.30, 'appoggiatura':0.25},
    'jazz':      {'mordent':0.02, 'trill':0.02, 'passing':0.40, 'neighbor':0.30, 'appoggiatura':0.05},
    'generic':   {'mordent':0.08, 'trill':0.05, 'passing':0.30, 'neighbor':0.25, 'appoggiatura':0.12},
}

ACC_PATTERNS = {
    "block":         lambda beats: [(0, beats)],
    "alberti":       lambda beats: [(0, beats/4), (beats/2, beats/4),
                                    (beats/4, beats/4), (beats/2, beats/4)],
    "arpeggio_up":   lambda beats: [(i * beats/4, beats/4) for i in range(4)],
    "arpeggio_down": lambda beats: [(i * beats/4, beats/4) for i in range(3,-1,-1)],
    "waltz":         lambda beats: [(0, 1.0), (1, 0.5), (2, 0.5)],
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def _snap_to_scale(midi_pitch, key_obj):
    """Ajusta la nota al grado de escala más cercano."""
    try:
        tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
        pcs = _get_scale_pcs(key_obj)
        scale_pcs = [(pc + tonic_pc) % 12 for pc in pcs]
        note_pc = midi_pitch % 12
        if note_pc in scale_pcs:
            return midi_pitch
        best, best_d = note_pc, 100
        for pc in scale_pcs:
            d = abs(pc - note_pc)
            d = min(d, 12 - d)
            if d < best_d:
                best_d, best = d, pc
        diff = best - note_pc
        if diff > 6:  diff -= 12
        if diff < -6: diff += 12
        return midi_pitch + diff
    except Exception:
        return midi_pitch

def _get_scale_pcs(key_obj):
    return [0,2,4,5,7,9,11] if key_obj.mode == 'major' else [0,2,3,5,7,8,10]

def _get_scale_midi(key_obj, octave=4):
    tonic = pitch.Pitch(key_obj.tonic.name)
    tonic.octave = octave
    base = tonic.midi
    pcs = _get_scale_pcs(key_obj)
    result = []
    for o in range(-2, 5):
        for pc in pcs:
            m = base + pc + o * 12
            if 21 <= m <= 108:
                result.append(m)
    return sorted(set(result))

def _clamp_pitch(p, lo=36, hi=96):
    while p > hi: p -= 12
    while p < lo: p += 12
    return int(np.clip(p, lo, hi))

def _snap_dur(r):
    opts = [0.25, 0.333, 0.5, 0.667, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(opts, key=lambda x: abs(x - r))

def _quarter_to_ticks(quarters, ticks_per_beat=480):
    return max(1, int(round(quarters * ticks_per_beat)))

def _roman_to_func_str(fig):
    if "/" in fig: return "Dsec"
    base = "".join(c for c in fig.replace("°","").replace("+","") if not c.isdigit())
    for func, figs in HARMONIC_FUNCTIONS.items():
        if base in figs: return func
    return "Other"

def _get_relative_key(key_obj):
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    if key_obj.mode == 'major':
        rel_pc = (tonic_pc + 9) % 12
        return m21key.Key(pitch.Pitch(rel_pc).name, 'minor')
    else:
        rel_pc = (tonic_pc + 3) % 12
        return m21key.Key(pitch.Pitch(rel_pc).name, 'major')

def _get_dominant_key(key_obj):
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    dom_pc = (tonic_pc + 7) % 12
    return m21key.Key(pitch.Pitch(dom_pc).name, key_obj.mode)

def _detect_style(tempo_bpm, swing, syncopation, complexity):
    if swing > 0.15 or syncopation > 0.2:
        return 'jazz'
    if tempo_bpm < 100 and complexity > 0.5:
        return 'romantic'
    if 60 <= tempo_bpm <= 140 and complexity < 0.4:
        return 'classical'
    if tempo_bpm > 100 and complexity > 0.3:
        return 'baroque'
    return 'generic'

def _build_chord_pitches_from_roman(roman_figure, key_obj, prev_pitches=None,
                                     register='chords', harmony_complexity=0.3):
    """Construye pitches de un acorde con voice leading opcional y extensiones [S]."""
    tonic_pc = pitch.Pitch(key_obj.tonic.name).pitchClass
    mode = key_obj.mode
    degree_map = MAJOR_SCALE_DEGREES if mode == 'major' else MINOR_SCALE_DEGREES
    base_fig = roman_figure.replace('7','').replace('°','').replace('+','').replace('9','')
    if base_fig in degree_map:
        semitones, quality = degree_map[base_fig]
        if '7' in roman_figure and quality not in ('d','M7','m7'):
            quality += '7'
    else:
        semitones, quality = 0, 'M'

    # Extensiones: añadir 7ª automáticamente si harmony_complexity > 0.5
    # y añadir 9ª si > 0.75 (sólo en acordes de tónica y subdominante)
    if harmony_complexity > 0.5 and '7' not in quality and quality in ('M', 'm'):
        quality += '7'

    root_pc = (tonic_pc + semitones) % 12
    lo, hi = INSTRUMENT_RANGES.get(register, (48, 76))
    iq = {'M':[0,4,7], 'm':[0,3,7], 'd':[0,3,6], 'A':[0,4,8],
          'M7':[0,4,7,10], 'm7':[0,3,7,10], 'd7':[0,3,6,9],
          'Mm7':[0,4,7,10], 'MM7':[0,4,7,11]}
    ints = iq.get(quality, [0,4,7])

    # Añadir 9ª si complexity muy alta
    if harmony_complexity > 0.75 and register == 'chords' and len(ints) >= 3:
        ninth = 14  # 9ª mayor
        ints = list(ints) + [ninth % 12 + 12]  # como extensión en octava superior

    if prev_pitches:
        return _voice_lead(prev_pitches, root_pc, quality, lo, hi)

    base = root_pc + 48
    while base < lo: base += 12
    result = [base + i for i in ints if lo - 2 <= base + i <= hi + 2]
    return result or [base]

def _voice_lead(prev_chord_pitches, root_pc, quality, lo, hi):
    iq = {'M':[0,4,7], 'm':[0,3,7], 'd':[0,3,6], 'A':[0,4,8],
          'M7':[0,4,7,10], 'm7':[0,3,7,10], 'd7':[0,3,6,9]}
    ints = iq.get(quality, [0,4,7])
    chord_midis = []
    for o in range(2, 7):
        for i in ints:
            m = root_pc + i + o * 12
            if lo - 2 <= m <= hi + 2:
                chord_midis.append(m)
    if not chord_midis:
        base = root_pc + 48
        while base < lo: base += 12
        return [base + i for i in ints if base + i <= hi]
    result, used = [], set()
    for prev in sorted(prev_chord_pitches):
        best = min(chord_midis, key=lambda m: (abs(m - prev), m in used))
        result.append(best)
        used.add(best)
    return sorted(set(max(lo, min(hi, p)) for p in result))

def voice_lead_next_chord(prev_pitches, candidate_pitches):
    """Voice-leading riguroso con penalización de paralelas. [K]"""
    if not prev_pitches or not candidate_pitches:
        return candidate_pitches
    n = len(prev_pitches)
    cand = sorted(candidate_pitches)
    base_pc = [p % 12 for p in cand]
    best_voicing, best_cost = None, float('inf')
    for octave_shift in range(-1, 2):
        for inversion in range(len(cand)):
            rotated_pc = base_pc[inversion:] + base_pc[:inversion]
            bass = prev_pitches[0]
            voicing = []
            ref = bass + (octave_shift * 12)
            for pc in rotated_pc:
                best_p = pc + 12 * round((ref - pc) / 12)
                best_p = int(np.clip(best_p, 28, 96))
                voicing.append(best_p)
                ref = best_p + 2
            if len(voicing) < n:
                voicing += [voicing[-1]] * (n - len(voicing))
            voicing = voicing[:n]
            cost = sum(abs(voicing[i] - prev_pitches[i]) for i in range(min(n, len(voicing))))
            for i in range(min(len(prev_pitches)-1, len(voicing)-1)):
                prev_iv = (prev_pitches[i+1] - prev_pitches[i]) % 12
                new_iv  = (voicing[i+1] - voicing[i]) % 12
                if prev_iv == new_iv and prev_iv in (0, 7):
                    cost += 12
            if cost < best_cost:
                best_cost, best_voicing = cost, voicing
    return best_voicing if best_voicing else candidate_pitches


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUITUR GRAMMAR  [J]
# ══════════════════════════════════════════════════════════════════════════════

class SequiturGrammar:
    """Re-Pair grammar para detectar frases repetidas en la melodía."""
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
        for _ in range(300):
            pairs = Counter()
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += 1
            if not pairs: break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2: break
            name = self._new_rule()
            self.rules[name] = list(best)
            new, i = [], 0
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == best:
                    new.append(name); i += 2
                else:
                    new.append(seq[i]); i += 1
            seq = new
        self.root = seq
        return self.rules

    def expand(self, sym):
        if sym not in self.rules: return [sym]
        r = []
        for s in self.rules[sym]: r.extend(self.expand(s))
        return r

    def get_raw_phrases(self):
        phrases = []
        for name in self.rules:
            expanded = self.expand(name)
            freq = self.root.count(name)
            phrases.append((expanded, freq, name))
        return phrases


def score_phrase(pitches, durations, key_obj):
    """Puntúa una frase melódica por calidad musical. [J]"""
    if len(pitches) < 2:
        return 0.0
    mid = len(pitches) // 2
    arc = min(abs(np.mean(pitches[:mid]) - np.mean(pitches[mid:])) / 12, 1.0)
    tonic_pc = key_obj.tonic.pitchClass
    try:
        tonic_pcs = {p.pitchClass for p in key_obj.getChord().pitches}
    except Exception:
        tonic_pcs = {tonic_pc, (tonic_pc+4)%12, (tonic_pc+7)%12}
    tonal_close = 1.0 if pitches[-1] % 12 in tonic_pcs else 0.3
    l = len(pitches)
    length_fit = 1.0 if 5 <= l <= 8 else max(0.2, 1.0 - abs(l - 6.5) * 0.15)
    ivs = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    steps = sum(1 for v in ivs if 1 <= v <= 2)
    leaps = sum(1 for v in ivs if v >= 3)
    total = len(ivs)
    variety = min(steps, leaps) / max(total, 1)
    if durations:
        dur_set = len(set(round(d*4)/4 for d in durations))
        rhy = min(dur_set / 3, 1.0)
    else:
        rhy = 0.5
    return (arc * 0.25 + tonal_close * 0.30 + length_fit * 0.20
            + variety * 0.15 + rhy * 0.10)


# ══════════════════════════════════════════════════════════════════════════════
#  MARKOV MELODY  [D]
# ══════════════════════════════════════════════════════════════════════════════

class MarkovMelody:
    """Cadena de Markov de 2º orden sobre (intervalo, duración_cuantizada)."""
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.start_states = Counter()

    def train(self, intervals, durations):
        if not intervals:
            return
        def q_dur(d):
            buckets = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            return min(buckets, key=lambda x: abs(x - d))
        def q_int(i):
            return max(-12, min(12, round(i)))
        n = min(len(intervals), len(durations) - 1)
        if n < self.order + 1:
            self.order = max(1, n - 1)
        states = [(q_int(intervals[i]), q_dur(durations[i])) for i in range(n)]
        if len(states) < 2:
            return
        for i in range(min(4, len(states))):
            self.start_states[states[i]] += 1
        for i in range(len(states) - self.order):
            context = tuple(states[i:i + self.order])
            self.transitions[context][states[i + self.order]] += 1
        if self.order > 1:
            for i in range(len(states) - 1):
                ctx1 = (states[i],)
                self.transitions[ctx1][states[i+1]] += 1

    def generate(self, n_steps, start_pitch, key_obj, seed=None):
        if seed is not None:
            random.seed(seed)
        if not self.transitions:
            sc = _get_scale_midi(key_obj, octave=5)
            result = []
            p = start_pitch
            for _ in range(n_steps):
                p = _snap_to_scale(p + random.choice([-2,-1,0,1,2]), key_obj)
                p = max(60, min(84, p))
                result.append((p, random.choice([0.5, 0.5, 1.0])))
            return result
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
                next_state = (random.choice([-2,-1,0,1,2]), random.choice([0.5,1.0]))
            interval_step, dur = next_state
            new_pitch = current_pitch + interval_step
            if random.random() < 0.3:
                new_pitch = _snap_to_scale(new_pitch, key_obj)
            if new_pitch > hi: new_pitch -= 12
            if new_pitch < lo: new_pitch += 12
            new_pitch = max(lo, min(hi, new_pitch))
            new_pitch = _snap_to_scale(new_pitch, key_obj)
            result.append((new_pitch, dur))
            current_pitch = new_pitch
            history.append(next_state)
            if len(history) > self.order + 1:
                history.pop(0)
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  GROOVE MAP  [H]
# ══════════════════════════════════════════════════════════════════════════════

class GrooveMap:
    """Extrae el timing map real del MIDI fuente."""
    def __init__(self, resolution=8):
        self.resolution = resolution
        self.timing_offsets = defaultdict(list)
        self.velocity_map = defaultdict(list)
        self.trained = False

    def train(self, score, ts_num=4):
        all_notes = [n for n in score.flat.notes if n.isNote]
        if len(all_notes) < 8:
            return
        grid_step = 1.0 / self.resolution
        for n in all_notes:
            off = float(n.offset)
            beat_pos = off % ts_num
            grid_pos = round(beat_pos / grid_step)
            grid_ideal = grid_pos * grid_step
            deviation = beat_pos - grid_ideal
            if abs(deviation) < 0.15:
                key_pos = grid_pos % (ts_num * self.resolution)
                self.timing_offsets[key_pos].append(deviation)
                vel = n.volume.velocity if hasattr(n,'volume') and n.volume.velocity else 64
                self.velocity_map[key_pos].append(vel)
        self.trained = len(self.timing_offsets) > 0

    def get_offset(self, beat_position, ts_num=4):
        if not self.trained: return 0.0
        grid_step = 1.0 / self.resolution
        grid_pos = round((beat_position % ts_num) / grid_step)
        key_pos = grid_pos % (ts_num * self.resolution)
        offsets = self.timing_offsets.get(key_pos, [0.0])
        return float(np.mean(offsets)) if offsets else 0.0

    def get_velocity(self, beat_position, base_vel=70, ts_num=4):
        if not self.trained: return base_vel
        grid_step = 1.0 / self.resolution
        grid_pos = round((beat_position % ts_num) / grid_step)
        key_pos = grid_pos % (ts_num * self.resolution)
        vels = self.velocity_map.get(key_pos, [base_vel])
        return int(np.clip(np.mean(vels), 20, 127)) if vels else base_vel


# ══════════════════════════════════════════════════════════════════════════════
#  MUSIC FRAGMENT  [E]
# ══════════════════════════════════════════════════════════════════════════════

class MusicFragment:
    """Fragmento real de 1-2 compases extraído del MIDI fuente."""
    def __init__(self, notes, offset, duration_ql, measure_idx,
                 tension=0.3, energy=0.5, function='I', key_obj=None, source_name=''):
        self.notes = notes
        self.offset = offset
        self.duration_ql = duration_ql
        self.measure_idx = measure_idx
        self.tension = tension
        self.energy = energy
        self.function = function
        self.key_obj = key_obj
        self.source_name = source_name
        self.pitch_mean = np.mean([n.pitch.midi for n in notes if n.isNote]) if notes else 60
        self.pitch_range = (max([n.pitch.midi for n in notes if n.isNote], default=60) -
                            min([n.pitch.midi for n in notes if n.isNote], default=60))
        pitches = [n.pitch.midi for n in notes if n.isNote]
        self.intervals = [pitches[i+1]-pitches[i] for i in range(len(pitches)-1)] if len(pitches) > 1 else []

    def transpose_to(self, target_key):
        if not self.key_obj or not target_key:
            return [copy.deepcopy(n) for n in self.notes]
        src_tonic = pitch.Pitch(self.key_obj.tonic.name).pitchClass
        tgt_tonic = pitch.Pitch(target_key.tonic.name).pitchClass
        semitone_shift = (tgt_tonic - src_tonic)
        if semitone_shift > 6:  semitone_shift -= 12
        if semitone_shift < -6: semitone_shift += 12
        transposed = []
        for n in self.notes:
            if n.isNote:
                new_n = copy.deepcopy(n)
                new_midi = n.pitch.midi + semitone_shift
                new_midi = _snap_to_scale(new_midi, target_key)
                new_n.pitch.midi = max(21, min(108, new_midi))
                transposed.append(new_n)
            else:
                transposed.append(copy.deepcopy(n))
        return transposed


# ══════════════════════════════════════════════════════════════════════════════
#  CLASE UnifiedDNA  —  extracción completa
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedDNA:
    """
    ADN musical unificado: extrae todas las características de los tres scripts.
    """
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.score = None

        # Básico
        self.key_obj = m21key.Key('C', 'major')
        self.tempo_bpm = 120.0
        self.time_sig = (4, 4)

        # Rítmico [A]
        self.rhythm_pattern = []           # compases con (offset, dur, accent, sincopa)
        self.rhythm_grid = np.zeros(16)    # histograma 16 subdivisiones
        self.rhythm_accent_grid = np.zeros(16)
        self.primary_subdivision = 0.25
        self.syncopation_ratio = 0.0
        self.rhythm_cell = [1.0, 0.5, 0.5, 1.0]
        self.swing = False

        # Melódico
        self.pitch_sequence = []
        self.pitch_contour = []
        self.pitch_register = 60
        self.motif_intervals = []
        self.full_melody = []           # [(midi, dur), …]
        self.sequitur_phrases = []      # [([(p,d),…], freq), …]  [J]
        self.intervals = []
        self.durations = []

        # Armónico
        self.harmony_prog = []          # [(fig, dur), …]  de script A
        self.harmony_functions = []     # [fig, …] lista plana
        self.harmony_complexity = 0.0

        # Curvas emocionales [B]
        self.tension_curve = []
        self.tonnetz_curve = []
        self.roughness_curve = []
        self.activity_curve = []
        self.valence_curve = []
        self.arousal_curve = []
        self.stability_curve = []
        self.entropy_melodic = 1.0
        self.entropy_rhythmic = 1.0
        self.climax_position = 0.75
        self.resolution_index = 0.5
        self.emotional_arc_label = "neutral"

        # Energía
        self.energy_mean = 0.5
        self.energy_curve = []

        # Forma [C]
        self.form_string = "AABA"
        self.section_map = []
        self.phrase_lengths = [4, 4, 4, 4]
        self.cadence_positions = []
        self.n_unique_sections = 2

        # Dinámica
        self.dynamics_mean = 72
        self.dynamics_std = 12

        # Estilo
        self.style = 'generic'

        # v3 extras [D][E][H][I]
        self.markov = MarkovMelody(order=2)
        self.groove_map = GrooveMap()
        self.fragments = []

    # ──────────────────────────────────────────────────────────────────────────
    #  EXTRACCIÓN PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, verbose=False):
        try:
            sc = converter.parse(self.path)
        except Exception as e:
            print(f"  ERROR cargando {self.path}: {e}")
            return False
        self.score = sc

        # Básico
        self.key_obj = self._detect_key(sc)
        self.tempo_bpm = self._detect_tempo(sc)
        self.time_sig = self._detect_timesig(sc)
        ts_num = self.time_sig[0]

        if verbose:
            print(f"    {self.key_obj.tonic.name} {self.key_obj.mode} | "
                  f"{self.tempo_bpm:.0f} BPM | {ts_num}/{self.time_sig[1]}")

        # Extracciones
        self._extract_melody(sc, verbose)
        self._extract_rhythm(sc, verbose)
        self._extract_harmony(sc, verbose)
        self._extract_sequitur(verbose)     # [J]
        self._extract_emotional_curves(sc, verbose)  # [B]
        self._extract_form(sc, verbose)     # [C]
        self._extract_dynamics(sc)
        self._extract_fragments(ts_num, verbose)  # [E]
        self._train_markov()                # [D]
        self.groove_map.train(sc, ts_num)   # [H]
        self.style = _detect_style(
            self.tempo_bpm, self.swing,
            self.syncopation_ratio,
            self.harmony_complexity
        )
        return True

    # ── Detectores básicos ────────────────────────────────────────────────────

    def _detect_key(self, sc):
        try:
            return sc.analyze('key')
        except Exception:
            return m21key.Key('C', 'major')

    def _detect_tempo(self, sc):
        for el in sc.flatten():
            if isinstance(el, tempo.MetronomeMark) and el.number:
                return float(el.number)
        return 120.0

    def _detect_timesig(self, sc):
        ts_list = []
        for el in sc.flatten():
            if isinstance(el, meter.TimeSignature):
                ts_list.append((el.numerator, el.denominator))
        return Counter(ts_list).most_common(1)[0][0] if ts_list else (4, 4)

    def _get_melody_part(self, sc):
        parts = sc.parts
        if not parts:
            return sc.flatten()
        best, best_sc = None, -1
        for p in parts:
            ns = [n for n in p.flatten().notes if isinstance(n, note.Note)]
            if not ns: continue
            s = np.mean([n.pitch.midi for n in ns]) * 0.5 + len(ns) * 0.01
            if s > best_sc:
                best_sc, best = s, p
        return best if best else parts[0]

    def _get_measures(self, sc):
        parts = getattr(sc, 'parts', [])
        if parts:
            ms = list(parts[0].getElementsByClass('Measure'))
            if ms: return ms
        return list(sc.flatten().getElementsByClass('Measure'))

    # ── Melodía ───────────────────────────────────────────────────────────────

    def _extract_melody(self, sc, verbose):
        part = self._get_melody_part(sc)
        mel = []
        for el in part.flatten().notes:
            if isinstance(el, note.Note) and hasattr(el, 'pitch'):
                mel.append((float(el.offset), el.pitch.midi, float(el.quarterLength)))
            elif isinstance(el, chord.Chord) and el.pitches:
                try:
                    top = max(el.pitches, key=lambda p: p.midi)
                    mel.append((float(el.offset), top.midi, float(el.quarterLength)))
                except Exception:
                    pass
        mel.sort(key=lambda x: x[0])
        self.pitch_sequence = [m[1] for m in mel]
        self.pitch_register = int(np.mean(self.pitch_sequence)) if self.pitch_sequence else 60
        if len(self.pitch_sequence) >= 2:
            self.pitch_contour = list(np.diff(self.pitch_sequence))
        self.motif_intervals = self.pitch_contour[:8] if len(self.pitch_contour) >= 8 \
            else (self.pitch_contour * max(1, 8 // max(len(self.pitch_contour), 1) + 1))[:8]
        self.full_melody = [(m[1], m[2]) for m in mel[:64]]
        self.intervals = [self.pitch_sequence[i+1]-self.pitch_sequence[i]
                          for i in range(len(self.pitch_sequence)-1)]
        self.durations = [m[2] for m in mel]
        if verbose:
            print(f"    Melodía: {len(self.pitch_sequence)} notas")

    # ── Ritmo [A] ─────────────────────────────────────────────────────────────

    def _extract_rhythm(self, sc, verbose):
        part = self._get_melody_part(sc)
        bpb = self.time_sig[0]

        def accent_weight(offset_in_bar, bpb):
            beat_pos = offset_in_bar % bpb
            sub = offset_in_bar - int(offset_in_bar)
            if beat_pos < 0.05: return 2.0
            strong_subs = {4: [2.0], 3: [], 2: [], 6: [3.0]}
            for sb in strong_subs.get(bpb, []):
                if abs(beat_pos - sb) < 0.05: return 1.5
            if sub < 0.05: return 1.0
            if abs(sub - 0.5) < 0.05 or abs(sub - 0.75) < 0.05: return 0.7
            return 0.6

        def is_syncopated(offset_in_bar, dur, bpb):
            sub = offset_in_bar - int(offset_in_bar)
            return sub > 0.1 and dur >= 0.5

        measures = list(part.getElementsByClass('Measure'))
        all_events = []
        if not measures:
            flat_notes = list(part.flatten().notes)
            for el in flat_notes:
                o, dur = float(el.offset), float(el.quarterLength)
                vel = (el.volume.velocity or 64) if isinstance(el, note.Note) else 64
                bs = int(o / bpb) * bpb
                all_events.append((o, o - bs, dur, vel))
        else:
            for m in measures:
                bs = float(m.offset)
                for el in m.flatten().notes:
                    if isinstance(el, (note.Note, chord.Chord)):
                        o, dur = float(el.offset), float(el.quarterLength)
                        vel = 64
                        if isinstance(el, note.Note) and el.volume.velocity:
                            vel = el.volume.velocity
                        elif isinstance(el, chord.Chord):
                            vels = [n2.volume.velocity for n2 in el.notes if n2.volume.velocity]
                            vel = int(np.mean(vels)) if vels else 64
                        all_events.append((o + bs, o, dur, vel))

        if not all_events:
            self.rhythm_pattern = [[(0.0, float(bpb), 2.0, False)]]
            return

        bar_dict = defaultdict(list)
        for o_abs, o_in_bar, dur, vel in all_events:
            bar_idx = int(o_abs / bpb)
            aw = accent_weight(o_in_bar, bpb)
            syn = is_syncopated(o_in_bar, dur, bpb)
            vel_factor = vel / 80.0
            aw_scaled = np.clip(aw * vel_factor, 0.4, 3.0)
            bar_dict[bar_idx].append((
                round(o_in_bar, 4), round(dur, 4),
                round(float(aw_scaled), 3), bool(syn)
            ))

        n_bars_src = max(bar_dict.keys()) + 1 if bar_dict else 1
        for bi in range(n_bars_src):
            bar = sorted(bar_dict.get(bi, []), key=lambda x: x[0])
            if not bar: bar = [(0.0, float(bpb), 2.0, False)]
            self.rhythm_pattern.append(bar)
        if len(self.rhythm_pattern) < 4:
            self.rhythm_pattern = self.rhythm_pattern * 8

        # Histograma de 16 subdivisiones
        GRID = 16
        grid_hits = np.zeros(GRID)
        grid_vel_sum = np.zeros(GRID)
        grid_vel_cnt = np.zeros(GRID)
        for o_abs, o_in_bar, dur, vel in all_events:
            pos_norm = (o_in_bar % bpb) / bpb
            bin_idx = int(pos_norm * GRID) % GRID
            grid_hits[bin_idx] += 1
            grid_vel_sum[bin_idx] += vel
            grid_vel_cnt[bin_idx] += 1
        total_hits = grid_hits.sum()
        self.rhythm_grid = grid_hits / total_hits if total_hits > 0 else np.ones(GRID) / GRID
        with np.errstate(divide='ignore', invalid='ignore'):
            self.rhythm_accent_grid = np.where(
                grid_vel_cnt > 0, grid_vel_sum / grid_vel_cnt / 127.0, 0.0)

        # Subdivisión primaria
        all_durs = [e[2] for e in all_events]
        if all_durs:
            dur_counts = Counter([round(d * 4) / 4 for d in all_durs])
            self.primary_subdivision = dur_counts.most_common(1)[0][0]

        # Síncopa
        syn_count = sum(1 for e in all_events if is_syncopated(e[1], e[2], bpb))
        self.syncopation_ratio = syn_count / max(len(all_events), 1)

        # Swing
        durs_all = [e[2] for e in all_events]
        eighth_pairs = [durs_all[i:i+2] for i in range(0, len(durs_all)-1, 2)
                        if 0.25 <= durs_all[i] <= 0.75 and 0.25 <= durs_all[i+1] <= 0.75]
        swing_pairs = [p for p in eighth_pairs if p[0] > p[1] * 1.4]
        self.swing = len(swing_pairs) / max(len(eighth_pairs), 1) > 0.3

        # Rhythm cell
        quantized = [_snap_dur(r) for r in durs_all[:64]]
        patterns = Counter(tuple(quantized[i:i+3]) for i in range(len(quantized)-2))
        top_cells = [list(cell) for cell, _ in patterns.most_common(4)]
        self.rhythm_cell = top_cells[0] if top_cells else [1.0, 0.5, 0.5, 1.0]

        if verbose:
            print(f"    Ritmo: subdiv={self.primary_subdivision}♩ | "
                  f"síncopa={self.syncopation_ratio:.0%} | swing={self.swing}")

    # ── Armonía ───────────────────────────────────────────────────────────────

    def _extract_harmony(self, sc, verbose):
        bpb = self.time_sig[0]
        k = self.key_obj
        all_notes = defaultdict(list)
        for el in sc.flatten().notes:
            if isinstance(el, note.Note) and hasattr(el, 'pitch'):
                slot = round(float(el.offset))
                all_notes[slot].append(el.pitch)
            elif isinstance(el, chord.Chord) and el.pitches:
                slot = round(float(el.offset))
                for p in el.pitches:
                    if hasattr(p, 'midi'):
                        all_notes[slot].append(p)
        if not all_notes:
            self.harmony_prog = [('I', 2.0), ('IV', 2.0), ('V', 2.0), ('I', 2.0)] * 4
            return
        raw = []
        for slot in sorted(all_notes.keys()):
            ps = all_notes[slot]
            if len(ps) >= 2:
                try:
                    ch = chord.Chord(ps)
                    rn = roman.romanNumeralFromChord(ch, k)
                    raw.append((slot, rn.figure))
                except Exception:
                    pass
        if not raw:
            self.harmony_prog = [('I', bpb)] * 8
            return
        compacted = []
        prev, start = raw[0][1], raw[0][0]
        for slot, fig in raw[1:]:
            if fig != prev:
                compacted.append((prev, max(1.0, float(slot - start))))
                prev, start = fig, slot
        compacted.append((prev, bpb))
        self.harmony_prog = compacted
        self.harmony_functions = [fig for fig, _ in compacted]
        # Complejidad: proporción de acordes no-diatónicos
        non_diatonic = sum(1 for f in self.harmony_functions
                           if _roman_to_func_str(f) in ('Dsec', 'Other'))
        self.harmony_complexity = non_diatonic / max(len(self.harmony_functions), 1)
        if verbose:
            print(f"    Armonía: {len(compacted)} acordes | complejidad={self.harmony_complexity:.2f}")

    # ── Sequitur [J] ──────────────────────────────────────────────────────────

    def _extract_sequitur(self, verbose):
        if not self.pitch_sequence:
            return
        key_obj = self.key_obj
        grammar = SequiturGrammar()
        grammar.build(self.pitch_sequence)
        raw = grammar.get_raw_phrases()
        midi_seq = self.pitch_sequence
        dur_seq = self.durations
        scored = []
        for pitches, freq, name in raw:
            if len(pitches) < 3: continue
            start = None
            for i in range(len(midi_seq) - len(pitches) + 1):
                if midi_seq[i:i+len(pitches)] == pitches:
                    start = i; break
            durs = dur_seq[start:start+len(pitches)] if start is not None else [0.5]*len(pitches)
            q = score_phrase(pitches, durs, key_obj)
            scored.append((list(zip(pitches, durs)), freq, q))
        scored.sort(key=lambda x: x[2] * math.sqrt(x[1]), reverse=True)
        self.sequitur_phrases = [(p, f) for p, f, q in scored[:10]]
        if verbose:
            print(f"    Sequitur: {len(self.sequitur_phrases)} frases extraídas")

    # ── Curvas emocionales [B] ────────────────────────────────────────────────

    def _extract_emotional_curves(self, sc, verbose):
        measures = self._get_measures(sc)
        if not measures:
            self.tension_curve = [0.5]
            self.arousal_curve = [0.0]
            self.valence_curve = [0.0]
            self.activity_curve = [0.5]
            self.stability_curve = [0.7]
            self.tonnetz_curve = [0.3]
            self.roughness_curve = [0.3]
            self.energy_curve = [0.5]
            return

        k = self.key_obj
        tonic_pc = k.tonic.pitchClass
        scale_pcs = set(p.pitchClass for p in k.getScale().pitches)
        try:
            tonica = k.pitchFromDegree(1).name
            medinat = k.pitchFromDegree(3).name
            dominant = k.pitchFromDegree(5).name
        except Exception:
            tonica = medinat = dominant = "C"
        stability_map = {tonica: 0, dominant: 1, medinat: 2}

        tension_raw, tonnetz_raw, roughness_raw = [], [], []
        activity_raw, valence_raw, arousal_raw, stability_raw, energy_raw = [], [], [], [], []
        global_mode_val = 0.5 if k.mode == 'major' else -0.5

        for m in measures:
            ns = list(m.flatten().notes)
            if not ns:
                for lst in [tension_raw, tonnetz_raw, roughness_raw, activity_raw,
                            valence_raw, arousal_raw, stability_raw, energy_raw]:
                    lst.append(lst[-1] if lst else 0.4)
                continue

            # Tensión Lerdahl
            m_t, h_t, r_t = [], [], []
            for el in ns:
                if isinstance(el, note.Note):
                    pn = el.pitch.name
                    mt = stability_map.get(pn, 4)
                    if pn not in [p.name for p in k.getScale().pitches]: mt += 3
                    m_t.append(mt)
                    r_t.append(1.0 / (el.quarterLength + 0.1))
                elif isinstance(el, chord.Chord):
                    ht = 0
                    if not el.isConsonant(): ht += 4
                    ht += len(el.pitches) * 0.5
                    h_t.append(ht)
                    r_t.append(1.0 / (el.quarterLength + 0.1))
            total = (np.mean(m_t) if m_t else 3.0) + \
                    (np.mean(h_t) if h_t else 0.0) + \
                    (np.mean(r_t) if r_t else 1.0)
            tension_raw.append(total)

            # Tonnetz
            pcs = []
            for el in ns:
                if isinstance(el, note.Note): pcs.append(el.pitch.pitchClass)
                elif isinstance(el, chord.Chord): pcs.extend(p.pitchClass for p in el.pitches)
            if pcs:
                coords = [TONNETZ.get((p - tonic_pc) % 12, (0,0)) for p in pcs]
                cx = np.mean([c[0] for c in coords])
                cy = np.mean([c[1] for c in coords])
                tonnetz_raw.append(float(np.sqrt(cx**2 + cy**2)))
            else:
                tonnetz_raw.append(tonnetz_raw[-1] if tonnetz_raw else 0.0)

            # Rugosidad (se excluyen PercussionChord y elementos sin .pitch estándar)
            all_midi = sorted(set(
                p.midi for el in ns
                for p in (el.pitches if isinstance(el, chord.Chord) else
                          ([el.pitch] if isinstance(el, note.Note) else []))
                if hasattr(p, 'midi')
            ))
            if len(all_midi) >= 2:
                rough = 0.0; n_pairs = 0
                for i in range(len(all_midi)):
                    for j in range(i+1, len(all_midi)):
                        ic = abs(all_midi[j] - all_midi[i]) % 12
                        w = DISSONANCE_WEIGHTS.get(ic, 0.5)
                        rough += w * (1.2 if all_midi[i] < 48 else 1.0)
                        n_pairs += 1
                roughness_raw.append(rough / n_pairs)
            else:
                roughness_raw.append(0.0)

            # Actividad
            dur_m = m.quarterLength or 4
            note_ns = [el for el in ns if isinstance(el, note.Note)]
            density = min(len(note_ns) / max(dur_m * 2, 1), 1.0)
            ps = [n.pitch.midi for n in note_ns]
            rng = min((max(ps) - min(ps)) / 24.0, 1.0) if ps else 0.0
            if len(ps) >= 2:
                ivs = np.abs(np.diff(ps))
                motion = min(np.mean(ivs) / 4.0, 1.0)
                leaps = sum(1 for i in ivs if i > 2) / len(ivs)
            else:
                motion = leaps = 0.0
            activity_raw.append(density * 0.35 + rng * 0.25 + motion * 0.25 + leaps * 0.15)

            # Arousal / Valencia
            all_ps = []
            for el in ns:
                if isinstance(el, note.Note): all_ps.append(el.pitch.midi)
                elif isinstance(el, chord.Chord): all_ps.append(el.sortAscending().pitches[-1].midi)
            avg_p = np.mean(all_ps) if all_ps else 60
            dens_a = len(ns) / dur_m
            ar = (min(dens_a / 8, 1.0) * 0.6 + (avg_p - 36) / 48 * 0.4) * 2 - 1
            arousal_raw.append(float(np.clip(ar, -1, 1)))

            all_pcs = []
            for el in ns:
                if isinstance(el, note.Note): all_pcs.append(el.pitch.pitchClass)
                elif isinstance(el, chord.Chord): all_pcs.extend(p.pitchClass for p in el.pitches)
            dia = sum(1 for p in all_pcs if p in scale_pcs) / len(all_pcs) if all_pcs else 0.5
            unique_pcs = sorted(set(all_pcs))
            if len(unique_pcs) >= 2:
                ivs2 = [(unique_pcs[i+1] - unique_pcs[i]) % 12 for i in range(len(unique_pcs)-1)]
                cons = sum(1 for iv in ivs2 if iv in {0,3,4,5,7,8,9}) / len(ivs2)
            else:
                cons = 0.5
            val = (global_mode_val + (cons - 0.5)) * dia
            valence_raw.append(float(np.clip(val, -1, 1)))

            # Estabilidad
            stable_notes = sum(1 for el in ns if isinstance(el, note.Note)
                               and el.pitch.name in stability_map)
            stability_raw.append(stable_notes / max(len(ns), 1))

            # Energía
            vels = [el.volume.velocity for el in ns
                    if isinstance(el, note.Note) and el.volume.velocity]
            vel_mean = np.mean(vels) / 127 if vels else 0.5
            energy_raw.append(float(np.clip(density * 0.4 + vel_mean * 0.4 + rng * 0.2, 0, 1)))

        def smooth_normalize(lst, norm=True):
            a = np.array(lst, dtype=float)
            if SCIPY_OK and len(a) > 3:
                a = gaussian_filter(a, sigma=1.5)
            if norm and a.max() > a.min():
                a = (a - a.min()) / (a.max() - a.min())
            return a.tolist()

        tension_arr = np.array(tension_raw, dtype=float)
        if tension_arr.max() > tension_arr.min():
            tension_arr = (tension_arr - tension_arr.min()) / (tension_arr.max() - tension_arr.min())
        if SCIPY_OK and len(tension_arr) > 3:
            tension_arr = gaussian_filter(tension_arr, sigma=1.5)
        self.tension_curve = tension_arr.tolist()
        self.climax_position = float(np.argmax(tension_arr) / max(len(tension_arr)-1, 1))
        self.resolution_index = float(np.mean(tension_arr[-max(3, len(tension_arr)//8):]))

        self.tonnetz_curve = smooth_normalize(tonnetz_raw)
        self.roughness_curve = smooth_normalize(roughness_raw)
        self.activity_curve = smooth_normalize(activity_raw, norm=False)
        self.valence_curve = smooth_normalize(valence_raw, norm=False)
        self.arousal_curve = smooth_normalize(arousal_raw, norm=False)
        self.stability_curve = smooth_normalize(stability_raw)
        self.energy_curve = smooth_normalize(energy_raw)
        self.energy_mean = float(np.mean(energy_raw)) if energy_raw else 0.5

        self._classify_emotional_arc()
        self._extract_entropy()

        if verbose:
            print(f"    Arco: {self.emotional_arc_label} | clímax@{self.climax_position:.0%}")

    def _classify_emotional_arc(self):
        if not self.tension_curve or not self.arousal_curve:
            self.emotional_arc_label = "neutral"; return
        t = np.array(self.tension_curve)
        a = np.array(self.arousal_curve)
        n = len(t)
        t_ini = np.mean(t[:max(1, n//4)])
        t_med = np.mean(t[n//4:3*n//4])
        t_fin = np.mean(t[3*n//4:])
        a_ini = np.mean(a[:max(1, n//4)])
        a_fin = np.mean(a[3*n//4:])
        if t_ini < t_med and t_fin < t_med:          self.emotional_arc_label = "arch"
        elif t_fin > t_ini * 1.3:                    self.emotional_arc_label = "crescendo"
        elif t_ini > t_fin * 1.3:                    self.emotional_arc_label = "decrescendo"
        elif a_ini < a_fin:                           self.emotional_arc_label = "awakening"
        elif a_ini > a_fin:                           self.emotional_arc_label = "lullaby"
        elif np.std(t) < 0.12:                        self.emotional_arc_label = "plateau"
        elif self.climax_position > 0.65:             self.emotional_arc_label = "late_climax"
        else:                                         self.emotional_arc_label = "neutral"

    def _extract_entropy(self):
        def ic_series(seq):
            cnt = Counter(seq)
            total = sum(cnt.values())
            if total == 0: return 0.0
            probs = [v/total for v in cnt.values()]
            return -sum(p * math.log2(p+1e-12) for p in probs)
        if len(self.pitch_sequence) >= 3:
            ivs = [max(-12, min(12, int(self.pitch_sequence[i+1]-self.pitch_sequence[i])))
                   for i in range(len(self.pitch_sequence)-1)]
            self.entropy_melodic = float(ic_series(ivs))
        if len(self.durations) >= 3:
            rhy = []
            for i in range(len(self.durations)-1):
                r = self.durations[i+1] / (self.durations[i]+1e-9)
                rhy.append("L" if r>1.1 else "S" if r<0.9 else "E")
            self.entropy_rhythmic = float(ic_series(rhy))

    # ── Forma [C] ─────────────────────────────────────────────────────────────

    def _extract_form(self, sc, verbose):
        measures = self._get_measures(sc)
        if not measures:
            self.form_string = "A"; self.section_map = [("A",1,1)]
            self.phrase_lengths = [4]; return
        n_bars = len(measures)
        window = max(2, min(4, n_bars // 4))
        descriptors = []
        bar_windows = []
        i = 0
        while i < n_bars:
            seg_measures = measures[i:i+window]
            seg_stream = stream.Stream()
            for m in seg_measures: seg_stream.append(copy.deepcopy(m))
            flat = seg_stream.flatten()
            ns = [el for el in flat.notes if isinstance(el, note.Note)]
            pitches = [n.pitch.midi for n in ns]
            if len(pitches) >= 2:
                ivs = list(np.diff(pitches))
                hist, _ = np.histogram(ivs, bins=12, range=(-12,12))
                mel_vec = hist / (hist.sum()+1e-9)
            else:
                mel_vec = np.zeros(12)
            extra = np.array([np.mean(pitches)/127, (max(pitches)-min(pitches))/24]) \
                    if pitches else np.zeros(2)
            desc = np.concatenate([mel_vec, extra])
            norm = np.linalg.norm(desc)
            descriptors.append(desc / norm if norm > 0 else desc)
            bar_windows.append((i+1, min(i+window, n_bars)))
            i += window
        n_segs = len(descriptors)
        if n_segs < 2:
            labels = [0] * n_segs
        elif SKLEARN_OK:
            try:
                n_clust = max(2, min(n_segs, max(2, n_segs//3)))
                clust = AgglomerativeClustering(n_clusters=n_clust, metric='cosine', linkage='average')
                labels = list(clust.fit_predict(np.array(descriptors)))
            except Exception:
                labels = list(range(n_segs))
        else:
            labels = [0]
            for j in range(1, n_segs):
                d, ld = descriptors[j], descriptors[j-1]
                cos = float(np.dot(d,ld)/(np.linalg.norm(d)*np.linalg.norm(ld)+1e-9))
                labels.append(labels[-1] if cos > 0.92 else max(labels)+1)
        lmap = {}; next_ch = ord('A')
        letter_labels = []
        for lbl in labels:
            if lbl not in lmap: lmap[lbl] = chr(next_ch); next_ch += 1
            letter_labels.append(lmap[lbl])
        self.form_string = "".join(letter_labels)
        self.n_unique_sections = len(set(letter_labels))
        self.section_map = [(letter_labels[j], bar_windows[j][0], bar_windows[j][1])
                            for j in range(len(letter_labels))]
        self.phrase_lengths = [w[1]-w[0]+1 for w in bar_windows]
        self.cadence_positions = [w[1] for w in bar_windows]
        if verbose:
            print(f"    Forma: {self.form_string} | {self.n_unique_sections} secciones")

    # ── Dinámica ──────────────────────────────────────────────────────────────

    def _extract_dynamics(self, sc):
        vels = []
        for el in sc.flatten().notes:
            if isinstance(el, note.Note) and el.volume.velocity:
                vels.append(el.volume.velocity)
            elif isinstance(el, chord.Chord):
                for n2 in el.notes:
                    if n2.volume.velocity: vels.append(n2.volume.velocity)
        if vels:
            self.dynamics_mean = int(np.mean(vels))
            self.dynamics_std = int(np.std(vels))

    # ── Fragmentos [E] ────────────────────────────────────────────────────────

    def _extract_fragments(self, ts_num, verbose):
        sc = self.score
        if sc is None: return
        part = self._get_melody_part(sc)
        ns = [n for n in part.flat.notes if n.isNote]
        if not ns: return
        total_t = float(part.flat.highestTime) or 1
        n_measures = max(1, int(total_t / ts_num))
        fragment_len = 2
        t_curve = self.tension_curve
        e_curve = self.energy_curve
        functions = self.harmony_functions
        for m in range(0, n_measures - fragment_len + 1, fragment_len):
            t_start = m * ts_num
            t_end = (m + fragment_len) * ts_num
            frag_notes = []
            for n in ns:
                noff = float(n.offset)
                if t_start <= noff < t_end:
                    nc = copy.deepcopy(n)
                    nc.offset = noff - t_start
                    frag_notes.append(nc)
            if len(frag_notes) < 2: continue
            tension = np.mean(t_curve[m:m+fragment_len]) if t_curve else 0.3
            energy = np.mean(e_curve[m:m+fragment_len]) if e_curve else 0.5
            func = functions[m % len(functions)] if functions else 'I'
            self.fragments.append(MusicFragment(
                notes=frag_notes, offset=t_start,
                duration_ql=t_end-t_start, measure_idx=m,
                tension=float(tension), energy=float(energy),
                function=func, key_obj=self.key_obj, source_name=self.name
            ))
        if verbose:
            print(f"    Fragmentos: {len(self.fragments)} extraídos")

    # ── Markov [D] ────────────────────────────────────────────────────────────

    def _train_markov(self):
        self.markov.train(self.intervals, self.durations)


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTIONAL CONTROLLER  [O]
# ══════════════════════════════════════════════════════════════════════════════

class EmotionalController:
    """Convierte curvas emocionales en parámetros musicales compás a compás."""
    def __init__(self, tension_curve, arousal_curve, valence_curve,
                 stability_curve, activity_curve, emotional_arc_label):
        self.tension   = self._norm(tension_curve)
        self.arousal   = self._norm_pm(arousal_curve)
        self.valence   = self._norm_pm(valence_curve)
        self.stability = self._norm(stability_curve)
        self.activity  = self._norm(activity_curve)
        self.arc       = emotional_arc_label

    @staticmethod
    def _norm(curve):
        a = np.array(curve, dtype=float)
        if a.max() > a.min():
            return ((a - a.min()) / (a.max() - a.min())).tolist()
        return [0.5] * len(a)

    @staticmethod
    def _norm_pm(curve):
        a = np.array(curve, dtype=float)
        mx = max(abs(a.max()), abs(a.min()))
        return (a / mx).tolist() if mx > 0 else list(curve)

    def get_bar_params(self, bar_idx, total_bars):
        def sample(curve, idx, total):
            if not curve: return 0.5
            frac = idx / max(total-1, 1)
            pos = frac * (len(curve)-1)
            lo = int(pos); hi = min(lo+1, len(curve)-1)
            t = pos - lo
            return curve[lo]*(1-t) + curve[hi]*t
        tension   = sample(self.tension,   bar_idx, total_bars)
        arousal   = sample(self.arousal,   bar_idx, total_bars)
        stability = sample(self.stability, bar_idx, total_bars)
        activity  = sample(self.activity,  bar_idx, total_bars)
        velocity  = int(np.clip(60 + arousal*25 + tension*15, 35, 110))
        register_offset = int(arousal * 7)
        density_mult = float(np.clip(0.6 + activity*0.8 + tension*0.4, 0.4, 2.2))
        if tension > 0.7:      acc_style = 'block'
        elif arousal > 0.3:    acc_style = 'arpeggio'
        else:                  acc_style = 'alberti'
        return {
            'velocity': velocity, 'register_offset': register_offset,
            'density_mult': density_mult, 'acc_style': acc_style,
            'harmony_tension': float(tension), 'stability': stability,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FORM GENERATOR  [P]
# ══════════════════════════════════════════════════════════════════════════════

class FormGenerator:
    """Hereda la estructura formal y la escala al nº de compases de salida."""
    def __init__(self, form_string, section_map, phrase_lengths,
                 cadence_positions, n_bars_out):
        self.form_string = form_string or "AABA"
        self.section_map = section_map or []
        self.phrase_lengths_src = phrase_lengths or [4]
        self.cadence_positions = cadence_positions or []
        self.n_bars = n_bars_out
        self._build_output_map()

    def _build_output_map(self):
        form = self.form_string
        n = self.n_bars
        bar_labels_src = []
        for i, letter in enumerate(form):
            phrase_len = self.phrase_lengths_src[i % len(self.phrase_lengths_src)]
            bar_labels_src.extend([letter] * phrase_len)
        scale = n / max(len(bar_labels_src), 1)
        self.bar_section = [bar_labels_src[min(int(bar/scale), len(bar_labels_src)-1)]
                            for bar in range(n)]
        self.cadence_bars_out = set()
        prev = None
        for bi, lbl in enumerate(self.bar_section):
            if lbl != prev and prev is not None:
                self.cadence_bars_out.add(bi)
            prev = lbl
        self.cadence_bars_out.add(n-1)

    def is_cadence_bar(self, bar_idx):
        return bar_idx in self.cadence_bars_out

    def section_of(self, bar_idx):
        if 0 <= bar_idx < len(self.bar_section):
            return self.bar_section[bar_idx]
        return 'A'

    def section_contour_modifier(self, bar_idx):
        sec = self.section_of(bar_idx)
        return {
            'A': {'invert': False, 'transpose': 0,  'compress': 1.0},
            'B': {'invert': True,  'transpose': 5,  'compress': 0.7},
            'C': {'invert': False, 'transpose': -3, 'compress': 1.3},
            'D': {'invert': True,  'transpose': 2,  'compress': 0.8},
        }.get(sec, {'invert': False, 'transpose': 0, 'compress': 1.0})


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE MELODÍA
# ══════════════════════════════════════════════════════════════════════════════

def _harmony_timeline(harmony_prog, total_beats):
    """Construye timeline de armonía: [(beat_start, beat_end, figure)]."""
    h_timeline = []
    beat = 0.0
    for fig, dur in harmony_prog:
        h_timeline.append((beat, beat+dur, fig))
        beat += dur
    total_h = beat if beat > 0 else total_beats
    return h_timeline, total_h

def _chord_at(beat, h_timeline, total_h):
    bm = beat % total_h
    for s, e, fig in h_timeline:
        if s <= bm < e: return fig
    return 'I'

def _chord_tones_at(beat, h_timeline, total_h, key_obj, octave=5):
    fig = _chord_at(beat, h_timeline, total_h)
    return _build_chord_pitches_from_roman(fig, key_obj, register='melody')


def generate_melody(
    harmony_prog, key_obj, rhythm_pattern,
    pitch_contour, pitch_register, motif_intervals,
    n_bars, emotional_ctrl, form_gen,
    beats_per_bar=4, rhythm_strength=1.0,
    markov_model=None, sequitur_phrases=None,
    melody_mode='contour',   # 'contour' | 'markov' | 'sequitur'
    surprise_rate=0.08,
):
    """
    Genera melodía integrando:
    - Modo 'contour' (script A): contorno + ritmo + arco emocional + forma
    - Modo 'markov'  (script B): Cadena de Markov de 2º orden
    - Modo 'sequitur' (script C): Frases Sequitur rescored
    """
    total_beats = n_bars * beats_per_bar
    h_timeline, total_h = _harmony_timeline(harmony_prog, total_beats)

    if melody_mode == 'markov' and markov_model and \
            sum(len(v) for v in markov_model.transitions.values()) > 0:
        return _generate_melody_markov(
            markov_model, harmony_prog, key_obj, n_bars, beats_per_bar,
            emotional_ctrl, form_gen, rhythm_strength, h_timeline, total_h)

    if melody_mode == 'sequitur' and sequitur_phrases:
        return _generate_melody_sequitur(
            sequitur_phrases, harmony_prog, key_obj, rhythm_pattern,
            n_bars, beats_per_bar, emotional_ctrl, form_gen,
            rhythm_strength, surprise_rate, h_timeline, total_h)

    return _generate_melody_contour(
        harmony_prog, key_obj, rhythm_pattern, pitch_contour, pitch_register,
        motif_intervals, n_bars, beats_per_bar, emotional_ctrl, form_gen,
        rhythm_strength, h_timeline, total_h)


def _generate_melody_contour(
    harmony_prog, key_obj, rhythm_pattern, pitch_contour, pitch_register,
    motif_intervals, n_bars, beats_per_bar, emotional_ctrl, form_gen,
    rhythm_strength, h_timeline, total_h
):
    """Melodía basada en contorno, con control emocional y formal. [O][P]"""
    result = []
    contour = list(pitch_contour) if pitch_contour else [2,-1,3,-2,1,-3]
    if not contour: contour = [2,-1,3,-2]
    contour_idx = 0
    current_pitch = max(52, min(79, pitch_register))
    current_pitch = _snap_to_scale(current_pitch, key_obj)

    for bar_idx in range(n_bars):
        global_beat = bar_idx * beats_per_bar
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        fm = form_gen.section_contour_modifier(bar_idx)

        raw_bar = rhythm_pattern[bar_idx % len(rhythm_pattern)]
        if not raw_bar:
            raw_bar = [(i*beats_per_bar/4, beats_per_bar/4, 1.0, False) for i in range(4)]
        bar_rhythm = []
        for item in raw_bar:
            if len(item) == 2: bar_rhythm.append((item[0], item[1], 1.0, False))
            elif len(item) >= 4: bar_rhythm.append((item[0], item[1], item[2], item[3]))
            else: bar_rhythm.append((item[0], item[1], 1.0, False))

        density_mult = ep['density_mult']
        if density_mult < 0.7 and len(bar_rhythm) > 2:
            bar_rhythm_sorted = sorted(bar_rhythm, key=lambda x: -x[2])
            n_keep = max(1, int(len(bar_rhythm) * density_mult))
            kept = set(id(x) for x in bar_rhythm_sorted[:n_keep])
            bar_rhythm = sorted([x for x in bar_rhythm if id(x) in kept], key=lambda x: x[0])

        target_register = max(48, min(84, current_pitch + ep['register_offset']))
        is_cadence = form_gen.is_cadence_bar(bar_idx)

        for note_idx, (local_offset, note_dur, accent_w, is_syn) in enumerate(bar_rhythm):
            beat = global_beat + local_offset
            step = contour[contour_idx % len(contour)]; contour_idx += 1
            if fm['invert']: step = -step
            step = int(step * fm['compress'])
            candidate = current_pitch + step + fm['transpose']
            if candidate > target_register + 12: candidate = current_pitch - abs(step)
            elif candidate < target_register - 12: candidate = current_pitch + abs(step)
            while candidate > 84: candidate -= 12
            while candidate < 45: candidate += 12
            candidate = _snap_to_scale(candidate, key_obj)
            is_strong = accent_w >= 1.4
            ct = _chord_tones_at(beat, h_timeline, total_h, key_obj)
            if is_strong and ct:
                nearest_ct = min(ct, key=lambda p: abs(p-candidate))
                threshold = 5 if accent_w >= 2.0 else 4
                if abs(nearest_ct - candidate) <= threshold:
                    candidate = nearest_ct
            if is_cadence and note_idx == len(bar_rhythm)-1:
                fig = _chord_at(beat, h_timeline, total_h)
                if fig.startswith('I') or fig.startswith('V'):
                    ct_f = _chord_tones_at(beat, h_timeline, total_h, key_obj)
                    if ct_f: candidate = min(ct_f, key=lambda p: abs(p-candidate))
            if contour_idx % 4 == 0:
                drift = int((target_register - current_pitch) * 0.15)
                candidate = _snap_to_scale(candidate + drift, key_obj)
                candidate = max(45, min(84, candidate))
            # Cromatismo sorpresa
            if random.random() < 0.04:
                candidate = _snap_to_scale(candidate + random.choice([-1,1]), key_obj)
            # Velocidad
            accent_scaled = 1.0 + (accent_w - 1.0) * rhythm_strength
            vel_accent = int((accent_scaled - 1.0) * 20)
            vel_syn = 5 if is_syn else 0
            jitter = random.randint(-3 if is_strong else -6, 3 if is_strong else 6)
            vel = max(35, min(110, ep['velocity'] + vel_accent + vel_syn + jitter))
            # Duración
            eff_aw = 1.0 + (accent_w - 1.0) * rhythm_strength
            dur_ratio = float(np.clip(0.55 + eff_aw * 0.20, 0.50, 0.97))
            actual_dur = note_dur * dur_ratio
            # Silencio estructural de frase [S]: en compás de cadencia,
            # la última nota se acorta para crear un respiro antes de la siguiente frase.
            if is_cadence and note_idx == len(bar_rhythm) - 1:
                actual_dur = max(0.1, actual_dur * 0.55)
            result.append((beat, candidate, max(0.1, actual_dur), vel))
            current_pitch = candidate
    return result


def _generate_melody_markov(
    markov_model, harmony_prog, key_obj, n_bars, beats_per_bar,
    emotional_ctrl, form_gen, rhythm_strength, h_timeline, total_h
):
    """Melodía Markov de 2º orden con control emocional. [D][O]"""
    result = []
    tonic_midi = pitch.Pitch(key_obj.tonic.name).midi + 60
    lo, hi = INSTRUMENT_RANGES['melody']
    tonic_midi = max(lo, min(hi, tonic_midi))
    ts_num = beats_per_bar
    n_steps = n_bars * max(4, int(ts_num / 0.5)) + 32
    markov_seq = markov_model.generate(n_steps, tonic_midi, key_obj)
    seq_idx = 0

    for bar_idx in range(n_bars):
        global_beat = bar_idx * beats_per_bar
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        fm = form_gen.section_contour_modifier(bar_idx)
        is_cadence = form_gen.is_cadence_bar(bar_idx)
        beat = 0.0
        while beat < beats_per_bar - 0.01:
            if seq_idx < len(markov_seq):
                p_midi, dur = markov_seq[seq_idx]; seq_idx += 1
            else:
                p_midi, dur = tonic_midi, 1.0
            dur = min(VALID_DURATIONS_FLOAT, key=lambda x: abs(x-dur))
            dur = max(0.25, min(dur, beats_per_bar - beat))
            # Transformación formal
            if fm['invert']:
                p_midi = tonic_midi - (p_midi - tonic_midi)
            p_midi = _snap_to_scale(p_midi + fm['transpose'], key_obj)
            p_midi = max(lo, min(hi, p_midi))
            # Cadencia
            if is_cadence and beat >= beats_per_bar - dur:
                ct = _chord_tones_at(global_beat+beat, h_timeline, total_h, key_obj)
                if ct: p_midi = min(ct, key=lambda p: abs(p-p_midi))
                dur = beats_per_bar - beat
            # Velocidad
            is_downbeat = beat < 0.05
            vel_base = ep['velocity']
            if is_downbeat:   vel = random.randint(vel_base-5, vel_base+10)
            else:             vel = random.randint(vel_base-15, vel_base)
            vel = max(35, min(110, vel))
            # Silencio estructural: en cadencia acortar la última nota de la frase
            if is_cadence and beat + dur >= beats_per_bar - 0.05:
                dur = max(0.1, dur * 0.55)
            result.append((global_beat + beat, p_midi, dur * 0.9, vel))
            beat += dur
    return result


def _generate_melody_sequitur(
    sequitur_phrases, harmony_prog, key_obj, rhythm_pattern,
    n_bars, beats_per_bar, emotional_ctrl, form_gen,
    rhythm_strength, surprise_rate, h_timeline, total_h
):
    """Melodía basada en frases Sequitur con métrica adaptada. [J]"""
    if not sequitur_phrases:
        return []
    result = []
    all_dur = 0.0
    target_dur = n_bars * beats_per_bar
    phrase_idx = 0
    bar_idx = 0

    while all_dur < target_dur - 0.1:
        ep = emotional_ctrl.get_bar_params(bar_idx % n_bars, n_bars)
        fm = form_gen.section_contour_modifier(bar_idx % n_bars)
        base_phrase, freq = sequitur_phrases[phrase_idx % len(sequitur_phrases)]
        phrase_idx += 1
        # Transportar según sección
        phrase = [(_clamp_pitch(p + fm['transpose']), d) for p, d in base_phrase]
        # Adaptar ritmo
        beats_pm = beats_per_bar
        cell = rhythm_pattern[bar_idx % len(rhythm_pattern)] if rhythm_pattern else []
        rhy_cell = [item[1] for item in cell] if cell else [1.0, 0.5, 0.5, 1.0]
        adapted = []
        for i, (p, orig_d) in enumerate(phrase):
            base_d = rhy_cell[i % len(rhy_cell)]
            if i == len(phrase)-1: base_d = max(base_d, 1.0)
            adapted.append((p, max(0.25, base_d)))
        # Sorpresa cromática
        for i in range(1, len(adapted)-1):
            if random.random() < surprise_rate:
                pv, nv = adapted[i-1][0], adapted[i+1][0]
                if abs(nv - pv) == 2:
                    new_p = pv + (1 if nv > pv else -1)
                    adapted[i] = (new_p, adapted[i][1])
        for p, d in adapted:
            if all_dur >= target_dur: break
            vel_base = ep['velocity']
            vel = max(35, min(110, vel_base + random.randint(-8, 8)))
            result.append((all_dur, p, min(d*0.9, target_dur-all_dur), vel))
            all_dur += d
        bar_idx += 1
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE MELODÍA MOSAIC  [E]
# ══════════════════════════════════════════════════════════════════════════════

def generate_melody_mosaic(dnas, harmony_prog, target_key, n_bars, ts_num,
                            emotional_ctrl, form_gen):
    """Ensambla fragmentos reales transpuestos. [E]"""
    all_fragments = [f for d in dnas for f in d.fragments]
    if not all_fragments:
        return []

    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * ts_num)
    t_curve = dnas[0].tension_curve
    e_curve = dnas[0].energy_curve
    result_notes = []
    current_offset = 0.0
    last_frag = None
    last_pitch = 65
    frag_len = 2
    n_windows = max(1, n_bars // frag_len)

    FUNC_COMPAT = {
        'I':['I','iii','vi','IV'], 'i':['i','III','VI','iv'],
        'V':['V','V7','vii°'], 'IV':['IV','ii','vi'],
    }

    for w in range(n_windows):
        m_idx = w * frag_len
        frag_offset = m_idx * ts_num
        t_val = t_curve[min(m_idx, len(t_curve)-1)] if t_curve else 0.4
        e_val = e_curve[min(m_idx, len(e_curve)-1)] if e_curve else 0.5
        func = _chord_at(frag_offset, h_timeline, total_h)
        compatible = FUNC_COMPAT.get(func, list(FUNC_COMPAT['I']))

        scored_frags = []
        for frag in all_fragments:
            if frag is last_frag: continue
            sc = 1.0 - (abs(frag.tension-t_val) + abs(frag.energy-e_val)) / 2.0
            if frag.function in compatible: sc += 0.3
            if last_pitch: sc -= abs(frag.pitch_mean - last_pitch) / 24.0
            scored_frags.append((sc, frag))
        if not scored_frags: continue
        scored_frags.sort(key=lambda x: -x[0])
        top = scored_frags[:max(3, len(scored_frags)//3)]
        scores = [max(0.01, s) for s, _ in top]
        total = sum(scores)
        probs = [s/total for s in scores]
        chosen = top[np.random.choice(len(top), p=probs)][1]

        transposed = chosen.transpose_to(target_key)
        lo, hi = INSTRUMENT_RANGES['melody']
        for n in transposed:
            if n.isNote:
                n.offset = frag_offset + float(n.offset)
                while n.pitch.midi > hi: n.pitch.midi -= 12
                while n.pitch.midi < lo: n.pitch.midi += 12
                result_notes.append(n)
        pitches_t = [n.pitch.midi for n in transposed if n.isNote]
        if pitches_t: last_pitch = pitches_t[-1]
        last_frag = chosen

    # Convertir a lista de eventos (offset, pitch, dur, vel)
    result = []
    for n in result_notes:
        result.append((float(n.offset), n.pitch.midi,
                       float(n.quarterLength) * 0.9,
                       n.volume.velocity or 75))
    return sorted(result, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE ACOMPAÑAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def generate_accompaniment(
    harmony_prog, key_obj, n_bars, emotional_ctrl, form_gen,
    beats_per_bar=4, acc_pattern=None, groove_map=None,
    force_style=None, harmony_complexity=0.3
):
    """
    Acompañamiento con:
    - Estilo variable por tensión (block/arpeggio/alberti/waltz) [L]
    - Voice leading entre acordes [K]
    - Dinámica desde curva emocional [O]
    - force_style: fuerza un estilo fijo ('alberti','arpeggio','block','waltz')
    """
    result = []
    total_beats = n_bars * beats_per_bar
    h_exp = []
    bt = 0.0
    while bt < total_beats:
        for fig, dur in harmony_prog:
            h_exp.append((bt, min(dur, total_beats-bt), fig))
            bt += dur
            if bt >= total_beats: break

    prev_pitches = None
    for chord_start, chord_dur, fig in h_exp:
        bar_idx = int(chord_start / beats_per_bar)
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        tension = ep['harmony_tension']

        # Prioridad: force_style > 3/4 waltz > emocional
        if force_style:
            acc_style = force_style
        elif beats_per_bar == 3:
            acc_style = 'waltz'
        else:
            acc_style = ep['acc_style']

        pitches = _build_chord_pitches_from_roman(fig, key_obj, prev_pitches, 'chords',
                                                   harmony_complexity=harmony_complexity)
        if not pitches: pitches = [48, 52, 55]
        prev_pitches = pitches
        vel_base = max(35, ep['velocity'] - 15)
        bpb = beats_per_bar

        if acc_style == 'block' or chord_dur <= 1.0:
            for p in pitches:
                groove_dev = groove_map.get_offset(chord_start, bpb) if groove_map else 0.0
                result.append((max(0, chord_start+groove_dev), p,
                               min(chord_dur, bpb)*0.9, vel_base))

        elif acc_style == 'alberti':
            # Rellenar a mínimo 3 notas si el acorde es incompleto
            if len(pitches) < 3:
                pitches = pitches + [pitches[0] + 12]
            pat = [pitches[0], pitches[-1], pitches[1], pitches[-1]]
            sub_dur = chord_dur / len(pat)
            t = chord_start
            for i, p in enumerate(pat):
                gd = groove_map.get_offset(t % bpb, bpb) if groove_map else 0.0
                v = groove_map.get_velocity(t % bpb, vel_base + (4 if i==0 else 0), bpb) \
                    if groove_map else vel_base + (4 if i==0 else 0)
                result.append((max(0, t+gd), p, sub_dur*0.88, int(v)))
                t += sub_dur

        elif acc_style == 'arpeggio':
            n_reps = max(1, int(chord_dur))
            sub_dur = chord_dur / (len(pitches) * n_reps)
            t = chord_start
            for _ in range(n_reps):
                for p in sorted(pitches):
                    gd = groove_map.get_offset(t % bpb, bpb) if groove_map else 0.0
                    v = groove_map.get_velocity(t % bpb, vel_base+random.randint(0,8), bpb) \
                        if groove_map else vel_base + random.randint(0,8)
                    result.append((max(0, t+gd), p, sub_dur*0.85, int(v)))
                    t += sub_dur

        elif acc_style == 'waltz':
            bass = pitches[0]
            upper = pitches[1:] if len(pitches) > 1 else pitches
            beat = chord_dur / 3
            gd = groove_map.get_offset(chord_start % bpb, bpb) if groove_map else 0.0
            result.append((max(0, chord_start+gd), bass, beat*0.9, vel_base+5))
            for p in upper:
                for b in [1, 2]:
                    gd2 = groove_map.get_offset((chord_start+b*beat) % bpb, bpb) if groove_map else 0.0
                    result.append((max(0, chord_start+b*beat+gd2), p, beat*0.85, vel_base-5))

    return result


def generate_bass(harmony_prog, key_obj, n_bars, beats_per_bar=4, groove_map=None):
    """
    Línea de bajo melódica con:
    - Notas de paso cromáticas entre raíces (leading-tone approach)
    - Walking bass cuando la energía es alta (ritmo en negras)
    - 5ª y 7ª del acorde en tiempos débiles según tensión
    - Voice leading de raíz a raíz con movimiento mínimo [K]
    """
    result = []
    lo, hi = INSTRUMENT_RANGES['bass']
    total_beats = n_bars * beats_per_bar
    bt = 0.0
    h_exp = []
    while bt < total_beats:
        for fig, dur in harmony_prog:
            h_exp.append((bt, min(dur, total_beats-bt), fig))
            bt += dur
            if bt >= total_beats: break

    def _bass_tones(fig, key_obj, lo, hi):
        """Devuelve (root, fifth, seventh) en rango de bajo."""
        pitches = _build_chord_pitches_from_roman(fig, key_obj, register='bass')
        root = pitches[0] if pitches else 36
        while root > hi: root -= 12
        while root < lo: root += 12

        # 5ª
        fifth = root + 7
        if fifth > hi: fifth -= 12
        # 7ª (si existe en el acorde)
        seventh = root + 10 if '7' in fig else root + 9
        if seventh > hi: seventh -= 12
        return root, max(lo, min(hi, fifth)), max(lo, min(hi, seventh))

    # Detectar si el estilo pide walking (jazz/swing) o estilo clásico
    # Lo inferimos de la complejidad armónica promedio
    avg_dur = np.mean([d for _, d in harmony_prog]) if harmony_prog else beats_per_bar
    walking = avg_dur <= beats_per_bar  # acordes de 1 compás o menos → walking posible

    prev_root = None
    for chord_idx, (chord_start, chord_dur, fig) in enumerate(h_exp):
        root, fifth, seventh = _bass_tones(fig, key_obj, lo, hi)

        # Voice-leading: preferir movimiento mínimo desde raíz anterior
        if prev_root is not None:
            while root - prev_root > 6:  root -= 12
            while prev_root - root > 6:  root += 12
        root = max(lo, min(hi, root))
        fifth = max(lo, min(hi, root + 7 if root + 7 <= hi else root - 5))
        seventh = max(lo, min(hi, root + 10 if root + 10 <= hi else root - 2))

        bpb = beats_per_bar
        gd = groove_map.get_offset(0.0, bpb) if groove_map else 0.0
        base_vel = groove_map.get_velocity(0.0, 80, bpb) if groove_map else 80

        if walking and chord_dur >= bpb and bpb >= 4:
            # Walking bass: negra en cada tiempo del compás
            # tiempo 1: raíz, tiempo 2: quinta o tercera, tiempo 3: 7ª o paso,
            # tiempo 4: nota de aproximación cromática a la siguiente raíz
            next_root = root  # por defecto vuelve a sí mismo
            if chord_idx + 1 < len(h_exp):
                next_fig = h_exp[chord_idx + 1][2]
                nps = _build_chord_pitches_from_roman(next_fig, key_obj, register='bass')
                next_root = nps[0] if nps else root
                while next_root > hi: next_root -= 12
                while next_root < lo: next_root += 12

            approach = next_root - 1 if next_root > root else next_root + 1
            approach = max(lo, min(hi, approach))

            walk = [root, fifth, seventh, approach]
            n_steps = min(bpb, len(walk))
            sub_dur = chord_dur / n_steps
            for i in range(n_steps):
                t = chord_start + i * sub_dur
                gd_w = groove_map.get_offset(t % bpb, bpb) if groove_map else 0.0
                v_w = groove_map.get_velocity(t % bpb,
                      base_vel - (0 if i == 0 else 8 if i < n_steps-1 else 5), bpb) \
                      if groove_map else (base_vel if i == 0 else base_vel - 8)
                result.append((max(0, t + gd_w), walk[i], sub_dur * 0.85,
                               int(np.clip(v_w, 20, 110))))
        else:
            # Bajo clásico: tiempo 1 = raíz, tiempo 3 = quinta (o 7ª si tensión)
            if bpb >= 4:
                beats_in = [(0.0, 2.0, root, base_vel),
                            (2.0, 2.0, fifth, base_vel - 12)]
            elif bpb == 3:
                beats_in = [(0.0, 2.0, root, base_vel),
                            (2.0, 1.0, fifth, base_vel - 10)]
            else:
                beats_in = [(0.0, float(bpb), root, base_vel)]

            for beat_off, dur_b, pitch_b, vel_b in beats_in:
                gd_b = groove_map.get_offset(beat_off, bpb) if groove_map else 0.0
                actual_vel = groove_map.get_velocity(beat_off, vel_b, bpb) \
                             if groove_map else vel_b
                result.append((max(0, chord_start + beat_off + gd_b),
                               pitch_b, dur_b * 0.88,
                               int(np.clip(actual_vel, 20, 110))))

        prev_root = root
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRAPUNTO  [G]
# ══════════════════════════════════════════════════════════════════════════════

def generate_counterpoint(melody_notes_list, harmony_prog, key_obj, n_bars, beats_per_bar=4,
                           emotional_ctrl=None):
    """
    Genera 2ª voz en movimiento contrario a la melodía.
    - Primera especie: nota contra nota en tiempos fuertes
    - Segunda especie: dos notas de cp por nota de melodía en pasajes tranquilos
    - Tercera especie: notas de paso en semicorcheas cuando densidad es alta
    - Resolución de disonancias por grado [G]
    """
    if not melody_notes_list:
        return []

    lo, hi = INSTRUMENT_RANGES['tenor']
    mel_by_measure = defaultdict(list)
    for offset, midi, dur, vel in melody_notes_list:
        m_idx = int(offset / beats_per_bar)
        mel_by_measure[m_idx].append((offset, midi, dur))

    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * beats_per_bar)
    GOOD_INTERVALS = {3, 4, 8, 9}
    AVOID_PARALLEL = {7, 12}
    DISSONANT      = {1, 2, 6, 10, 11}

    result = []
    prev_cp, prev_mel, prev_cp_int = None, None, None

    for m in range(n_bars):
        m_off = m * beats_per_bar

        # Densidad del compás desde EmotionalController
        density = 1.0
        if emotional_ctrl:
            ep = emotional_ctrl.get_bar_params(m, n_bars)
            density = ep.get('density_mult', 1.0)

        chord_p = _build_chord_pitches_from_roman(
            _chord_at(m_off, h_timeline, total_h), key_obj, register='chords')
        mel_m = mel_by_measure.get(m, [])
        if not mel_m:
            continue

        # Decidir especie: densidad alta → más notas
        if density >= 1.5:
            # Tercera especie: notas de cp cada 0.5 beats
            beats_in = [i * 0.5 for i in range(int(beats_per_bar / 0.5))]
            cp_dur   = 0.5
        elif density >= 0.9:
            # Segunda especie: cada beat
            beats_in = list(range(int(beats_per_bar)))
            cp_dur   = 1.0
        else:
            # Primera especie: dos notas por compás
            beats_in = [0.0, float(beats_per_bar) / 2]
            cp_dur   = float(beats_per_bar) / 2

        for beat in beats_in:
            beat_abs = m_off + beat
            # Nota de melodía más cercana al beat
            mel_at = sorted(mel_m, key=lambda x: abs(x[0] - beat_abs))
            mel_p = mel_at[0][1] if mel_at else (prev_mel or 65)

            candidates = [p for p in chord_p if lo <= p <= hi]
            if not candidates:
                c = _snap_to_scale(mel_p - 4, key_obj)
                while c > hi: c -= 12
                while c < lo: c += 12
                candidates = [max(lo, min(hi, c))]

            # Añadir notas de escala como candidatos para notas de paso
            scale_midi = _get_scale_midi(key_obj, octave=4)
            extra = [p for p in scale_midi if lo <= p <= hi]
            candidates = sorted(set(candidates + extra))

            best_cp, best_score = None, -999
            for cp_cand in candidates:
                int_to_mel = abs(mel_p - cp_cand) % 12
                score = 0
                if int_to_mel in GOOD_INTERVALS: score += 3
                elif int_to_mel in {0, 12}: score += 1
                elif int_to_mel in DISSONANT: score -= 2
                if prev_cp and prev_mel:
                    prev_iv = abs(prev_mel - prev_cp) % 12
                    if prev_iv in AVOID_PARALLEL and int_to_mel == prev_iv:
                        score -= 4
                    mel_dir = np.sign(mel_p - prev_mel)
                    cp_dir  = np.sign(cp_cand - prev_cp)
                    if mel_dir != 0 and cp_dir == -mel_dir: score += 2
                    elif cp_dir == 0: score += 1
                if prev_cp:
                    jump = abs(cp_cand - prev_cp)
                    if jump > 7: score -= 2
                    elif jump <= 2: score += 1  # preferir grado conjunto
                if score > best_score:
                    best_score, best_cp = score, cp_cand

            if best_cp is None:
                best_cp = max(lo, min(hi, _snap_to_scale(mel_p - 4, key_obj)))

            # Resolución de disonancia: si el intervalo es disonante, resolver por grado
            if best_cp is not None:
                curr_int = abs(mel_p - best_cp) % 12
                if curr_int in DISSONANT and prev_cp is not None:
                    # Resolver al intervalo consonante más cercano
                    for step in [1, -1, 2, -2]:
                        resolved = best_cp + step
                        if lo <= resolved <= hi:
                            new_int = abs(mel_p - resolved) % 12
                            if new_int in GOOD_INTERVALS:
                                best_cp = resolved
                                break

            vel = int(np.clip(random.randint(42, 62), 20, 90))
            result.append((max(0, beat_abs), best_cp, cp_dur * 0.87, vel))
            prev_cp, prev_mel, prev_cp_int = best_cp, mel_p, abs(mel_p - best_cp) % 12

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ORNAMENTACIÓN  [F]
# ══════════════════════════════════════════════════════════════════════════════

def add_ornamentation(melody_notes_list, key_obj, style='generic'):
    """
    Añade ornamentación idiomática: notas de paso, vecinas, apoyaturas. [F]
    Entrada/salida: lista de (offset, midi, dur, vel)
    """
    if not melody_notes_list or len(melody_notes_list) < 2:
        return melody_notes_list
    style_probs = ORNAMENTATION_STYLES.get(style, ORNAMENTATION_STYLES['generic'])
    result = []
    i = 0
    notes = sorted(melody_notes_list, key=lambda x: x[0])
    while i < len(notes):
        offset, midi, dur, vel = notes[i]
        applied = False
        # Nota de paso entre dos notas separadas por 3ª
        if (not applied and i < len(notes)-1
                and random.random() < style_probs['passing']
                and dur >= 0.5):
            next_midi = notes[i+1][1]
            diff = next_midi - midi
            if abs(diff) in (3, 4):
                pass_midi = _snap_to_scale(midi + int(np.sign(diff)), key_obj)
                half = dur / 2
                result.append((offset, midi, half*0.9, vel))
                result.append((offset+half, pass_midi, half*0.85, int(vel*0.85)))
                applied = True

        # Apoyatura superior
        if (not applied and i > 0
                and random.random() < style_probs['appoggiatura']
                and dur >= 0.5):
            app_pitch = _snap_to_scale(midi + 2, key_obj)
            app_dur = 0.25
            main_dur = dur - app_dur
            if main_dur >= 0.25:
                result.append((offset, app_pitch, app_dur*0.85, int(vel*0.9)))
                result.append((offset+app_dur, midi, main_dur*0.9, vel))
                applied = True

        # Nota vecina en nota larga
        if (not applied and random.random() < style_probs['neighbor'] and dur >= 1.5):
            neigh = _snap_to_scale(midi + 2, key_obj)
            nd = 0.25
            main_d = dur - nd * 2
            if main_d >= 0.5:
                result.append((offset, midi, main_d/2*0.9, vel))
                result.append((offset+main_d/2, neigh, nd*0.85, int(vel*0.7)))
                result.append((offset+main_d/2+nd, midi, (main_d/2+nd)*0.9, vel))
                applied = True

        if not applied:
            result.append((offset, midi, dur, vel))
        i += 1
    return sorted(result, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════════════════
#  HUMANIZACIÓN  [H]
# ══════════════════════════════════════════════════════════════════════════════

def humanize(notes_list, groove_map=None, ts_num=4, micro_jitter=True):
    """
    Aplica humanización: groove timing + micro-variación de velocidad. [H]
    Entrada/salida: lista de (offset, midi, dur, vel)
    """
    result = []
    for offset, midi, dur, vel in notes_list:
        new_offset = offset
        new_vel = vel
        if groove_map and groove_map.trained:
            beat_pos = offset % ts_num
            groove_dev = groove_map.get_offset(beat_pos, ts_num)
            new_offset = max(0, offset + groove_dev)
            new_vel = groove_map.get_velocity(beat_pos, vel, ts_num)
        if micro_jitter:
            new_offset += random.uniform(-0.01, 0.01)
            new_offset = max(0, new_offset)
            new_vel = int(np.clip(new_vel + random.randint(-4, 4), 20, 127))
        result.append((new_offset, midi, dur, new_vel))
    return result


def generate_percussion(rhythm_grid, rhythm_accent_grid, n_bars, beats_per_bar=4,
                         groove_map=None, style='generic'):
    """
    Genera una pista de percusión simple desde el rhythm_grid (histograma 16 subds.). [A]
    Instrumentos GM en canal 9:
      36 = Kick (bombo), 38 = Snare, 42 = Hi-hat closed, 46 = Hi-hat open, 49 = Crash
    """
    KICK   = 36
    SNARE  = 38
    HIHAT  = 42
    HIHAT_O= 46
    CRASH  = 49

    result = []
    n_divs = len(rhythm_grid)  # normalmente 16
    if n_divs == 0:
        return result

    # Normalizar grid
    grid = np.array(rhythm_grid, dtype=float)
    if grid.max() > 0:
        grid = grid / grid.max()
    accent = np.array(rhythm_accent_grid, dtype=float)
    if accent.max() > 0:
        accent = accent / accent.max()

    sub_dur = beats_per_bar / (n_divs / (beats_per_bar / beats_per_bar))
    # Duración de cada subdivisión en quarter notes
    sub_ql = beats_per_bar / n_divs * (16 / n_divs) if n_divs != 16 else beats_per_bar / 4

    for bar_idx in range(n_bars):
        bar_start = bar_idx * beats_per_bar
        is_first_bar = (bar_idx == 0)

        for div in range(n_divs):
            beat_pos = div * beats_per_bar / n_divs
            t = bar_start + beat_pos
            g_val = float(grid[div])
            a_val = float(accent[div])
            gd = groove_map.get_offset(beat_pos % beats_per_bar, beats_per_bar) \
                 if groove_map and groove_map.trained else 0.0

            # --- Crash en el primer tiempo del primer compás ---
            if is_first_bar and div == 0:
                result.append((max(0, t + gd), CRASH, sub_ql * 2, 90))

            # --- Bombo: tiempo 1 y tiempo 3 (divs 0 y 8 en 16 subdivs) ---
            if div % (n_divs // beats_per_bar) == 0:
                beat_num = div // (n_divs // beats_per_bar)
                is_strong = (beat_num % 2 == 0)  # tiempos 1 y 3
                if is_strong:
                    v_kick = int(np.clip(85 + a_val * 25, 70, 115))
                    result.append((max(0, t + gd), KICK, sub_ql * 0.8, v_kick))

            # --- Caja: tiempos 2 y 4 (divs 4 y 12 en 16 subdivs) ---
            if n_divs >= 8:
                beat_num = div // (n_divs // beats_per_bar)
                if beat_num % 2 == 1 and div % (n_divs // beats_per_bar) == 0:
                    v_snare = int(np.clip(75 + a_val * 20, 60, 105))
                    result.append((max(0, t + gd), SNARE, sub_ql * 0.75, v_snare))

            # --- Hi-hat: cuando el rhythm_grid indica actividad ---
            if g_val > 0.25:
                # Hi-hat abierto en tiempos fuertes marcados, cerrado el resto
                is_accent_div = a_val > 0.6
                hh = HIHAT_O if is_accent_div else HIHAT
                v_hh = int(np.clip(50 + g_val * 35 + a_val * 20, 35, 90))
                result.append((max(0, t + gd), hh, sub_ql * 0.6, v_hh))

    return sorted(result, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MIDI FINAL  (mido directo)
# ══════════════════════════════════════════════════════════════════════════════

def build_midi(melody_notes, acc_notes, bass_notes, cp_notes,
               target_key, tempo_bpm, time_sig, n_bars,
               form_gen=None, output_path='output_unified.mid',
               extra_voices=None, percussion_notes=None,
               split_tracks=False):
    """
    Escribe el MIDI con mido.
    Pistas fijas: Melody, Counterpoint, Accompaniment, Bass.
    Pistas extra: una por cada voz en extra_voices (lista de dicts con
                  'notes', 'name', 'program', 'channel').
    """
    TICKS = 480
    bpb, bu = time_sig
    us_per_beat = int(60_000_000 / max(tempo_bpm, 1))

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS)

    section_events = []
    if form_gen:
        prev_sec = None
        for bi in range(n_bars):
            sec = form_gen.section_of(bi)
            if sec != prev_sec:
                tick = _quarter_to_ticks(bi * bpb, TICKS)
                section_events.append((tick, f'[{sec}]'))
                prev_sec = sec

    def notes_to_track(notes_list, ch_num, track_name, program=0,
                       add_sustain=False, expression_curve=None):
        trk = mido.MidiTrack()
        trk.name = track_name
        trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        trk.append(mido.MetaMessage('time_signature',
                                    numerator=bpb, denominator=bu,
                                    clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                    time=0))
        trk.append(mido.Message('program_change', channel=ch_num,
                                program=program, time=0))
        events = []
        for tick_abs, label in section_events:
            events.append((tick_abs, 'marker', label, 0))
        for offset, mp, dur, vel in notes_list:
            mp  = max(0, min(127, int(mp)))
            vel = max(1, min(127, int(vel)))
            dur = max(0.05, float(dur))
            t_on  = _quarter_to_ticks(float(offset), TICKS)
            t_off = _quarter_to_ticks(float(offset)+dur, TICKS)
            events.append((t_on,  'note_on',  mp, vel))
            events.append((t_off, 'note_off', mp, 0))

        # CC#11 Expression: curva dinámica por compás [S]
        if expression_curve:
            total_ticks = _quarter_to_ticks(n_bars * bpb, TICKS)
            n_points = max(len(expression_curve), 1)
            for i, val in enumerate(expression_curve):
                tick = int(i * total_ticks / n_points)
                cc_val = int(np.clip(val * 127, 20, 127))
                events.append((tick, 'cc', 11, cc_val))

        # CC#64 Sustain pedal: ON al inicio de cada compás, OFF en el beat 3 [S]
        if add_sustain:
            total_beats_n = n_bars * bpb
            beat_step = bpb  # un compás
            b = 0.0
            while b < total_beats_n:
                t_on_s  = _quarter_to_ticks(b, TICKS)
                t_off_s = _quarter_to_ticks(b + beat_step * 0.6, TICKS)
                events.append((t_on_s,  'cc', 64, 127))
                events.append((t_off_s, 'cc', 64, 0))
                b += beat_step

        events.sort(key=lambda e: (e[0], 0 if e[1]=='note_off' else 1))
        prev_tick = 0
        for ev in events:
            delta = max(0, ev[0] - prev_tick)
            prev_tick = ev[0]
            if ev[1] == 'marker':
                trk.append(mido.MetaMessage('marker', text=ev[2], time=delta))
            elif ev[1] == 'note_on':
                trk.append(mido.Message('note_on', channel=ch_num,
                                        note=ev[2], velocity=ev[3], time=delta))
            elif ev[1] == 'note_off':
                trk.append(mido.Message('note_off', channel=ch_num,
                                        note=ev[2], velocity=0, time=delta))
            elif ev[1] == 'cc':
                trk.append(mido.Message('control_change', channel=ch_num,
                                        control=ev[2], value=ev[3], time=delta))
        trk.append(mido.MetaMessage('end_of_track', time=0))
        return trk

    # Construir curva de expresión desde el form_gen si está disponible
    expr_curve = None
    if form_gen:
        expr_curve = []
        for bi in range(n_bars):
            sec = form_gen.section_of(bi)
            # Sección B un poco más suave, A y resto normal
            base = 0.75 if sec == 'B' else 0.88
            expr_curve.append(base)

    mid.tracks.append(notes_to_track(melody_notes, 0, 'Melody',       program=0,
                                     expression_curve=expr_curve))
    mid.tracks.append(notes_to_track(cp_notes,     1, 'Counterpoint', program=0,
                                     expression_curve=expr_curve))
    mid.tracks.append(notes_to_track(acc_notes,    2, 'Accompaniment',program=0,
                                     add_sustain=True, expression_curve=expr_curve))
    mid.tracks.append(notes_to_track(bass_notes,   3, 'Bass',         program=32,
                                     expression_curve=expr_curve))

    # Pistas de voces adicionales (canales 4+, hasta 15 para evitar canal 9=percusión)
    if extra_voices:
        for i, ev in enumerate(extra_voices):
            ch = min(4 + i, 15)
            if ch == 9: ch = 10  # saltar canal de percusión
            mid.tracks.append(notes_to_track(
                ev['notes'], ch, ev['name'], program=ev['program']))
            print(f"  → Voz '{ev['name']}' ({len(ev['notes'])} notas, "
                  f"prog={ev['program']}, ch={ch})")

    # Pista de percusión en canal 9 [A]
    if percussion_notes:
        perc_trk = mido.MidiTrack()
        perc_trk.name = 'Percussion'
        perc_trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        perc_trk.append(mido.MetaMessage('time_signature',
                                          numerator=bpb, denominator=bu,
                                          clocks_per_click=24,
                                          notated_32nd_notes_per_beat=8, time=0))
        p_events = []
        for offset, mp, dur, vel in percussion_notes:
            mp  = max(0, min(127, int(mp)))
            vel = max(1, min(127, int(vel)))
            dur = max(0.05, float(dur))
            t_on  = _quarter_to_ticks(float(offset), TICKS)
            t_off = _quarter_to_ticks(float(offset) + dur, TICKS)
            p_events.append((t_on,  'on',  mp, vel))
            p_events.append((t_off, 'off', mp, 0))
        p_events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        prev_t = 0
        for ev in p_events:
            delta = max(0, ev[0] - prev_t)
            prev_t = ev[0]
            msg_type = 'note_on' if ev[1] == 'on' else 'note_off'
            perc_trk.append(mido.Message(msg_type, channel=9,
                                         note=ev[2], velocity=ev[3], time=delta))
        perc_trk.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(perc_trk)
        print(f"  → Percusión ({len(percussion_notes)} golpes, ch=9)")

    mid.save(output_path)
    print(f"  → MIDI guardado: {output_path}")

    # Exportar pistas separadas [split_tracks]
    if split_tracks:
        base = output_path.rsplit('.', 1)[0]
        track_names = [
            ('Melody', melody_notes, 0, 0),
            ('Counterpoint', cp_notes, 1, 0),
            ('Accompaniment', acc_notes, 2, 0),
            ('Bass', bass_notes, 3, 32),
        ]
        if extra_voices:
            for i, ev in enumerate(extra_voices):
                ch = min(4 + i, 15)
                if ch == 9: ch = 10
                track_names.append((ev['name'], ev['notes'], ch, ev['program']))
        for tname, tnotes, tch, tprog in track_names:
            if not tnotes:
                continue
            mid_s = mido.MidiFile(type=0, ticks_per_beat=TICKS)
            mid_s.tracks.append(notes_to_track(tnotes, tch, tname, tprog,
                                               expression_curve=expr_curve))
            sname = f"{base}_{tname.lower()}.mid"
            mid_s.save(sname)
            print(f"  → Split: {sname}")



# ══════════════════════════════════════════════════════════════════════════════
#  VOCES ADICIONALES — presets de instrumento y generadores por rol
# ══════════════════════════════════════════════════════════════════════════════

# Presets: nombre → {program MIDI, rango (lo,hi), rol por defecto, vel_scale}
VOICE_PRESETS = {
    # Cuerdas
    'violin':      {'program': 40, 'range': (64, 88), 'role': 'counterpoint', 'vel': 0.85},
    'viola':       {'program': 41, 'range': (55, 76), 'role': 'inner',        'vel': 0.80},
    'cello':       {'program': 42, 'range': (36, 60), 'role': 'inner',        'vel': 0.80},
    'contrabass':  {'program': 43, 'range': (28, 50), 'role': 'bass_double',  'vel': 0.90},
    'harp':        {'program': 46, 'range': (48, 84), 'role': 'arpeggio',     'vel': 0.75},
    'strings':     {'program': 48, 'range': (48, 76), 'role': 'inner',        'vel': 0.70},
    # Maderas
    'flute':       {'program': 73, 'range': (72, 96), 'role': 'melody_double','vel': 0.80},
    'piccolo':     {'program': 72, 'range': (79,103), 'role': 'melody_double','vel': 0.75},
    'oboe':        {'program': 68, 'range': (60, 84), 'role': 'counterpoint', 'vel': 0.80},
    'clarinet':    {'program': 71, 'range': (52, 84), 'role': 'inner',        'vel': 0.80},
    'bassoon':     {'program': 70, 'range': (34, 60), 'role': 'bass_double',  'vel': 0.85},
    'saxophone':   {'program': 65, 'range': (49, 80), 'role': 'counterpoint', 'vel': 0.85},
    # Metales
    'trumpet':     {'program': 56, 'range': (55, 82), 'role': 'melody_double','vel': 0.90},
    'horn':        {'program': 60, 'range': (40, 72), 'role': 'pedal',        'vel': 0.80},
    'trombone':    {'program': 57, 'range': (40, 67), 'role': 'inner',        'vel': 0.85},
    'tuba':        {'program': 58, 'range': (28, 52), 'role': 'bass_double',  'vel': 0.90},
    # Teclados/Pads
    'organ':       {'program': 19, 'range': (36, 84), 'role': 'inner',        'vel': 0.70},
    'pad':         {'program': 88, 'range': (48, 76), 'role': 'pedal',        'vel': 0.60},
    'choir':       {'program': 52, 'range': (52, 79), 'role': 'inner',        'vel': 0.65},
    'vibraphone':  {'program': 11, 'range': (53, 89), 'role': 'ostinato',     'vel': 0.75},
    'marimba':     {'program': 12, 'range': (48, 84), 'role': 'ostinato',     'vel': 0.75},
    # Guitarras
    'guitar':      {'program': 24, 'range': (40, 76), 'role': 'arpeggio',     'vel': 0.80},
    'guitar_mute': {'program': 28, 'range': (40, 76), 'role': 'ostinato',     'vel': 0.85},
}

VOICE_ROLES = ['counterpoint', 'inner', 'melody_double', 'pedal',
               'bass_double', 'ostinato', 'arpeggio']


def parse_voices_arg(voices_str):
    """
    Parsea --voices "violin,viola:inner,horn:pedal:0.7"
    Cada token: nombre[:rol[:vel_override]]
    Devuelve lista de dicts con preset resuelto.
    """
    if not voices_str:
        return []
    voices = []
    for token in voices_str.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        inst_name = parts[0].strip().lower()
        if inst_name not in VOICE_PRESETS:
            print(f"  ⚠ Instrumento desconocido: '{inst_name}' — ignorado.")
            print(f"    Disponibles: {', '.join(sorted(VOICE_PRESETS))}")
            continue
        preset = dict(VOICE_PRESETS[inst_name])
        preset['name'] = inst_name
        if len(parts) >= 2 and parts[1].strip() in VOICE_ROLES:
            preset['role'] = parts[1].strip()
        if len(parts) >= 3:
            try:
                preset['vel'] = float(parts[2])
            except ValueError:
                pass
        voices.append(preset)
    return voices


# ── Rol: inner ────────────────────────────────────────────────────────────────

def generate_voice_inner(harmony_prog, key_obj, n_bars, emotional_ctrl,
                          beats_per_bar, lo, hi, vel_scale, groove_map=None):
    """
    Voz interior: nota media del acorde, notas largas (una por acorde).
    Usa voice-leading mínimo entre acordes. [K]
    """
    result = []
    total_beats = n_bars * beats_per_bar
    h_exp = []
    bt = 0.0
    while bt < total_beats:
        for fig, dur in harmony_prog:
            h_exp.append((bt, min(dur, total_beats - bt), fig))
            bt += dur
            if bt >= total_beats:
                break
    prev_p = None
    for chord_start, chord_dur, fig in h_exp:
        bar_idx = int(chord_start / beats_per_bar)
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        pitches = _build_chord_pitches_from_roman(fig, key_obj, register='chords')
        # Filtrar al rango del instrumento
        in_range = [p for p in pitches if lo <= p <= hi]
        if not in_range:
            # Transponer la más cercana al rango
            p_base = pitches[len(pitches) // 2] if pitches else (lo + hi) // 2
            while p_base > hi: p_base -= 12
            while p_base < lo: p_base += 12
            in_range = [max(lo, min(hi, p_base))]
        # Voice leading: elegir la más cercana a la nota anterior
        if prev_p is not None:
            p = min(in_range, key=lambda x: abs(x - prev_p))
        else:
            p = in_range[len(in_range) // 2]
        vel_base = max(25, int(ep['velocity'] * vel_scale * 0.85))
        gd = groove_map.get_offset(chord_start % beats_per_bar, beats_per_bar)              if groove_map and groove_map.trained else 0.0
        result.append((max(0, chord_start + gd), p,
                       chord_dur * 0.88, vel_base))
        prev_p = p
    return result


# ── Rol: counterpoint ────────────────────────────────────────────────────────

def generate_voice_counterpoint(melody_notes, harmony_prog, key_obj,
                                 n_bars, beats_per_bar, lo, hi,
                                 vel_scale, groove_map=None):
    """
    Segunda voz en movimiento contrario a la melodía, en rango del instrumento.
    Reutiliza la lógica de generate_counterpoint ajustando el rango. [G]
    """
    if not melody_notes:
        return []
    GOOD_INTERVALS = {3, 4, 8, 9}
    AVOID_PARALLEL = {7, 12}

    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * beats_per_bar)
    mel_by_measure = defaultdict(list)
    for offset, midi, dur, vel in melody_notes:
        m_idx = int(offset / beats_per_bar)
        mel_by_measure[m_idx].append((offset, midi))

    result = []
    prev_cp, prev_mel = None, None

    for m in range(n_bars):
        m_off = m * beats_per_bar
        chord_p = _build_chord_pitches_from_roman(
            _chord_at(m_off, h_timeline, total_h), key_obj, register='chords')
        mel_m = mel_by_measure.get(m, [])
        if not mel_m:
            continue
        beats_in = [0.0, float(beats_per_bar) / 2] if beats_per_bar >= 2 else [0.0]
        for beat in beats_in:
            mel_at = [(o, p) for o, p in mel_m
                      if abs(o - (m * beats_per_bar + beat)) < 0.5]
            if not mel_at:
                continue
            mel_p = mel_at[0][1]
            # Candidatos en rango del instrumento
            candidates = [p for p in chord_p if lo <= p <= hi]
            if not candidates:
                c = _snap_to_scale(mel_p - 5, key_obj)
                while c > hi: c -= 12
                while c < lo: c += 12
                candidates = [max(lo, min(hi, c))]
            best_cp, best_score = None, -999
            for cp_c in candidates:
                iv = abs(mel_p - cp_c) % 12
                score = 3 if iv in GOOD_INTERVALS else (1 if iv in {0,12} else -1)
                if prev_cp and prev_mel:
                    prev_iv = abs(prev_mel - prev_cp) % 12
                    if prev_iv in AVOID_PARALLEL and iv == prev_iv:
                        score -= 4
                    mel_dir = np.sign(mel_p - prev_mel)
                    cp_dir  = np.sign(cp_c - prev_cp)
                    if mel_dir != 0 and cp_dir == -mel_dir: score += 2
                    elif cp_dir == 0: score += 1
                if prev_cp and abs(cp_c - prev_cp) > 7: score -= 2
                if score > best_score:
                    best_score, best_cp = score, cp_c
            if best_cp is None:
                best_cp = candidates[0]
            dur = float(beats_per_bar) / len(beats_in)
            gd = groove_map.get_offset(beat, beats_per_bar)                  if groove_map and groove_map.trained else 0.0
            vel = max(25, int(random.randint(45, 65) * vel_scale))
            result.append((max(0, m_off + beat + gd), best_cp, dur * 0.87, vel))
            prev_cp, prev_mel = best_cp, mel_p
    return result


# ── Rol: melody_double ────────────────────────────────────────────────────────

def generate_voice_melody_double(melody_notes, key_obj, lo, hi,
                                  vel_scale, interval_semitones=4):
    """
    Dobla la melodía a un intervalo (3ª mayor=4, 6ª menor=8, octava=12…).
    El intervalo se calcula automáticamente según el rango del instrumento:
    si el instrumento es agudo (lo>64) dobla a la octava superior,
    si es grave dobla a la tercera inferior.
    """
    result = []
    mel_mean = np.mean([m for _, m, _, _ in melody_notes]) if melody_notes else 65
    # Decidir dirección: si rango del instrumento está por encima de la melodía → subir
    center = (lo + hi) / 2
    if center > mel_mean + 6:
        iv = 12   # octava superior
    elif center < mel_mean - 6:
        iv = -7   # quinta inferior
    else:
        iv = interval_semitones  # 3ª mayor por defecto

    for offset, midi, dur, vel in melody_notes:
        doubled = _snap_to_scale(midi + iv, key_obj)
        while doubled > hi: doubled -= 12
        while doubled < lo: doubled += 12
        doubled = max(lo, min(hi, doubled))
        new_vel = max(20, int(vel * vel_scale * 0.80))
        result.append((offset, doubled, dur * 0.92, new_vel))
    return result


# ── Rol: pedal ────────────────────────────────────────────────────────────────

def generate_voice_pedal(harmony_prog, key_obj, n_bars, emotional_ctrl,
                          beats_per_bar, lo, hi, vel_scale):
    """
    Nota de pedal: alterna tónica y dominante según la función armónica.
    Notas largas (un valor por compás). Ideal para trompa, pad, órgano.
    """
    result = []
    tonic_pc  = pitch.Pitch(key_obj.tonic.name).pitchClass
    dom_pc    = (tonic_pc + 7) % 12

    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * beats_per_bar)

    for bar_idx in range(n_bars):
        beat = bar_idx * beats_per_bar
        ep   = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        fig  = _chord_at(beat, h_timeline, total_h)
        func = _roman_to_func_str(fig)

        # Elegir tónica o dominante según función
        base_pc = dom_pc if func in ('D', 'Dsec') else tonic_pc
        p = base_pc + 48
        while p > hi: p -= 12
        while p < lo: p += 12
        p = max(lo, min(hi, p))

        # Tensión alta → más forte
        vel = max(25, int((ep['velocity'] - 20) * vel_scale * 0.75))
        result.append((float(beat), p, float(beats_per_bar) * 0.92, vel))
    return result


# ── Rol: bass_double ─────────────────────────────────────────────────────────

def generate_voice_bass_double(bass_notes, key_obj, lo, hi, vel_scale):
    """
    Dobla la línea de bajo a la octava inferior (o superior si el instrumento
    está por encima). Ideal para contrabajo, fagot, tuba.
    """
    result = []
    bass_mean = np.mean([m for _, m, _, _ in bass_notes]) if bass_notes else 40
    center = (lo + hi) / 2
    iv = -12 if center < bass_mean else 12

    for offset, midi, dur, vel in bass_notes:
        doubled = midi + iv
        while doubled > hi: doubled -= 12
        while doubled < lo: doubled += 12
        doubled = max(lo, min(hi, doubled))
        new_vel = max(20, int(vel * vel_scale))
        result.append((offset, doubled, dur, new_vel))
    return result


# ── Rol: ostinato ─────────────────────────────────────────────────────────────

def generate_voice_ostinato(harmony_prog, key_obj, n_bars, emotional_ctrl,
                              beats_per_bar, lo, hi, vel_scale,
                              rhythm_cell=None, groove_map=None):
    """
    Célula rítmica repetida en notas del acorde.
    Ideal para vibráfono, marimba, guitarra muted.
    """
    result = []
    cell = rhythm_cell or [0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * beats_per_bar)

    for bar_idx in range(n_bars):
        bar_start = bar_idx * beats_per_bar
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        fig = _chord_at(float(bar_start), h_timeline, total_h)
        pitches = _build_chord_pitches_from_roman(fig, key_obj, register='chords')
        in_range = [p for p in pitches if lo <= p <= hi]
        if not in_range:
            p_base = pitches[0] if pitches else (lo + hi) // 2
            while p_base > hi: p_base -= 12
            while p_base < lo: p_base += 12
            in_range = [max(lo, min(hi, p_base))]

        beat_cursor = 0.0
        cell_idx = 0
        while beat_cursor < beats_per_bar - 0.01:
            dur = cell[cell_idx % len(cell)]
            dur = min(dur, beats_per_bar - beat_cursor)
            p   = in_range[cell_idx % len(in_range)]
            gd  = groove_map.get_offset(beat_cursor, beats_per_bar)                   if groove_map and groove_map.trained else 0.0
            vel = max(20, int(ep['velocity'] * vel_scale * 0.75
                              + random.randint(-5, 5)))
            result.append((max(0, bar_start + beat_cursor + gd),
                           p, dur * 0.80, vel))
            beat_cursor += dur
            cell_idx += 1
    return result


# ── Rol: arpeggio ─────────────────────────────────────────────────────────────

def generate_voice_arpeggio(harmony_prog, key_obj, n_bars, emotional_ctrl,
                              beats_per_bar, lo, hi, vel_scale, groove_map=None):
    """
    Arpegio ascendente/descendente alternado por compás.
    Ideal para arpa, guitarra.
    """
    result = []
    h_timeline, total_h = _harmony_timeline(harmony_prog, n_bars * beats_per_bar)
    direction = 1  # alterna por compás

    for bar_idx in range(n_bars):
        bar_start = bar_idx * beats_per_bar
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        fig = _chord_at(float(bar_start), h_timeline, total_h)
        pitches_raw = _build_chord_pitches_from_roman(fig, key_obj, register='chords')
        # Expandir al rango del instrumento (añadir octavas)
        expanded = []
        for p in pitches_raw:
            for shift in [-12, 0, 12]:
                np2 = p + shift
                if lo <= np2 <= hi:
                    expanded.append(np2)
        expanded = sorted(set(expanded)) if expanded else [max(lo, min(hi, pitches_raw[0] if pitches_raw else 60))]

        if direction < 0:
            expanded = list(reversed(expanded))
        direction *= -1

        n_notes = len(expanded)
        sub_dur = beats_per_bar / max(n_notes, 1)
        for i, p in enumerate(expanded):
            gd = groove_map.get_offset(i * sub_dur, beats_per_bar)                  if groove_map and groove_map.trained else 0.0
            # Primer nota más fuerte
            v_mult = 1.0 if i == 0 else (0.85 if i % 2 == 0 else 0.70)
            vel = max(20, int(ep['velocity'] * vel_scale * v_mult))
            result.append((max(0, bar_start + i * sub_dur + gd),
                           p, sub_dur * 0.82, vel))
    return result


# ── Dispatcher principal de voces ─────────────────────────────────────────────

def generate_extra_voice(voice_preset, melody_notes, bass_notes,
                          harmony_prog, key_obj, n_bars, emotional_ctrl,
                          beats_per_bar, rhythm_cell=None, groove_map=None,
                          humanize_flag=True):
    """
    Genera una voz adicional completa según su rol y preset.
    Devuelve lista de (offset, midi, dur, vel).
    """
    lo, hi      = voice_preset['range']
    vel_scale   = voice_preset.get('vel', 0.80)
    role        = voice_preset['role']
    gm = groove_map if humanize_flag else None

    if role == 'inner':
        notes = generate_voice_inner(
            harmony_prog, key_obj, n_bars, emotional_ctrl,
            beats_per_bar, lo, hi, vel_scale, gm)

    elif role == 'counterpoint':
        notes = generate_voice_counterpoint(
            melody_notes, harmony_prog, key_obj,
            n_bars, beats_per_bar, lo, hi, vel_scale, gm)

    elif role == 'melody_double':
        notes = generate_voice_melody_double(
            melody_notes, key_obj, lo, hi, vel_scale)

    elif role == 'pedal':
        notes = generate_voice_pedal(
            harmony_prog, key_obj, n_bars, emotional_ctrl,
            beats_per_bar, lo, hi, vel_scale)

    elif role == 'bass_double':
        notes = generate_voice_bass_double(
            bass_notes, key_obj, lo, hi, vel_scale)

    elif role == 'ostinato':
        notes = generate_voice_ostinato(
            harmony_prog, key_obj, n_bars, emotional_ctrl,
            beats_per_bar, lo, hi, vel_scale, rhythm_cell, gm)

    elif role == 'arpeggio':
        notes = generate_voice_arpeggio(
            harmony_prog, key_obj, n_bars, emotional_ctrl,
            beats_per_bar, lo, hi, vel_scale, gm)

    else:
        notes = []

    # Micro-humanización final
    if humanize_flag and notes:
        notes = [(max(0, o + random.uniform(-0.008, 0.008)),
                  m, d, int(np.clip(v + random.randint(-3, 3), 20, 127)))
                 for o, m, d, v in notes]

    # Ornamentación para voces melódicas [F]
    if role in ('melody_double', 'counterpoint') and notes:
        notes = add_ornamentation(notes, key_obj, 'classical')

    return sorted(notes, key=lambda x: x[0])

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING DE CANDIDATOS  [N]
# ══════════════════════════════════════════════════════════════════════════════

def score_candidate(melody_notes, acc_notes, key_obj):
    """
    Puntúa un candidato por:
    - Consonancia tonal (notas en escala)
    - Variedad rítmica
    - Rango melódico
    - Continuidad: penaliza saltos excesivos consecutivos [N]
    - Arco dinámico: penaliza melodías planas sin clímax
    - Presencia de contrapunto
    """
    if not melody_notes:
        return 0.0

    pcs = set(_get_scale_pcs(key_obj))
    tc  = pitch.Pitch(key_obj.tonic.name).pitchClass
    scale_pcs_abs = {(pc + tc) % 12 for pc in pcs}

    pitches = [mp for _, mp, _, _ in melody_notes]
    durs    = [d  for _, _,  d, _ in melody_notes]
    vels    = [v  for _, _, _,  v in melody_notes]

    # 1. Consonancia tonal
    in_scale   = sum(1 for mp in pitches if mp % 12 in scale_pcs_abs)
    consonance = in_scale / max(len(pitches), 1)

    # 2. Variedad rítmica
    variety = min(1.0, len(set(round(d, 2) for d in durs)) / 6)

    # 3. Rango melódico
    rng_score = min(1.0, (max(pitches) - min(pitches)) / 24) if pitches else 0.0

    # 4. Continuidad: penalizar saltos consecutivos de >7 semitonos
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    big_leaps  = sum(1 for iv in intervals if iv > 7)
    continuity = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 5. Arco dinámico: debe haber variación de velocidad (clímax)
    if len(vels) >= 4:
        vel_std   = float(np.std(vels))
        arc_score = min(1.0, vel_std / 20.0)
    else:
        arc_score = 0.5

    # 6. Bonus por contrapunto
    cp_bonus = 0.05 if len(acc_notes) > 0 else 0.0

    return (consonance  * 0.30 +
            variety     * 0.20 +
            rng_score   * 0.15 +
            continuity  * 0.20 +
            arc_score   * 0.10 +
            cp_bonus)


# ══════════════════════════════════════════════════════════════════════════════
#  MODOS DE MEZCLA
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_controllers(dnas, emotion_src_idx, form_src_idx, n_bars, form_override=None):
    ed = dnas[min(emotion_src_idx, len(dnas)-1)]
    fd = dnas[min(form_src_idx,   len(dnas)-1)]
    ec = EmotionalController(
        tension_curve   = ed.tension_curve   or [0.5],
        arousal_curve   = ed.arousal_curve   or [0.0],
        valence_curve   = ed.valence_curve   or [0.0],
        stability_curve = ed.stability_curve or [0.7],
        activity_curve  = ed.activity_curve  or [0.5],
        emotional_arc_label = ed.emotional_arc_label
    )
    form_str = form_override or fd.form_string
    fg = FormGenerator(
        form_string      = form_str,
        section_map      = fd.section_map,
        phrase_lengths   = fd.phrase_lengths,
        cadence_positions= fd.cadence_positions,
        n_bars_out       = n_bars
    )
    return ec, fg


def _generate_melody_with_modulation(
    h_prog, target_key, r_pat, contour, reg, motif,
    n_bars, ec, fg, bpb, rhythm_strength, markov,
    seq_phrases, melody_mode, surprise_rate
):
    """
    Genera la melodía sección a sección. En sección B modula al relativo o dominante. [Q]
    Une los fragmentos ajustando los offsets.
    """
    # Recopilar compases por sección
    sections = {}
    for bi in range(n_bars):
        sec = fg.section_of(bi)
        sections.setdefault(sec, []).append(bi)

    all_notes = []
    for sec_label in dict.fromkeys(fg.bar_section):  # orden de aparición
        bars = sections.get(sec_label, [])
        if not bars:
            continue
        n_sec = len(bars)
        bar_offset = bars[0]  # compás inicial de la sección

        # Elegir tonalidad de la sección [Q]
        if sec_label == 'B':
            sec_key = _get_relative_key(target_key)
        elif sec_label == 'C':
            sec_key = _get_dominant_key(target_key)
        else:
            sec_key = target_key

        # Crear un FormGenerator parcial para esta sección
        fg_sec = FormGenerator(
            form_string=sec_label,
            section_map=[sec_label] * n_sec,
            phrase_lengths=[n_sec],
            cadence_positions=[n_sec - 1],
            n_bars_out=n_sec
        )

        frag = generate_melody(
            h_prog, sec_key, r_pat, contour, reg, motif,
            n_sec, ec, fg_sec, bpb, rhythm_strength, markov,
            seq_phrases, melody_mode, surprise_rate
        )
        # Reubicar al offset absoluto de la sección y ajustar al rango de target_key
        for offset, midi, dur, vel in frag:
            new_midi = _snap_to_scale(midi, target_key)
            all_notes.append((offset + bar_offset * bpb, new_midi, dur, vel))

    return sorted(all_notes, key=lambda x: x[0])


def _run_generation(
    harmony_src_dna, melody_src_dna, rhythm_src_dna,
    target_key, n_bars, tempo_bpm, ec, fg, time_sig,
    rhythm_strength=1.0, melody_mode='contour',
    surprise_rate=0.08, humanize_groove=True,
    dnas_all=None, force_acc_style=None,
):
    bpb = time_sig[0]
    h_prog = harmony_src_dna.harmony_prog
    r_pat  = rhythm_src_dna.rhythm_pattern
    contour = melody_src_dna.pitch_contour
    reg = melody_src_dna.pitch_register
    motif = melody_src_dna.motif_intervals
    markov = melody_src_dna.markov
    seq_phrases = melody_src_dna.sequitur_phrases
    groove = rhythm_src_dna.groove_map if humanize_groove else None
    style = melody_src_dna.style

    # Melodía
    if melody_mode == 'mosaic' and dnas_all:
        mel = generate_melody_mosaic(dnas_all, h_prog, target_key, n_bars, bpb, ec, fg)
        if not mel:
            mel = generate_melody(h_prog, target_key, r_pat, contour, reg, motif,
                                  n_bars, ec, fg, bpb, rhythm_strength, markov,
                                  seq_phrases, 'markov', surprise_rate)
    else:
        # Generación sección a sección con modulación real en sección B [Q]
        mel = _generate_melody_with_modulation(
            h_prog, target_key, r_pat, contour, reg, motif,
            n_bars, ec, fg, bpb, rhythm_strength, markov,
            seq_phrases, melody_mode, surprise_rate
        )

    # Ornamentación melodía [F]
    mel = add_ornamentation(mel, target_key, style)

    # Humanización [H]
    if humanize_groove and groove and groove.trained:
        mel = humanize(mel, groove, bpb)

    # Acompañamiento [L]
    acc = generate_accompaniment(h_prog, target_key, n_bars, ec, fg, bpb,
                                 groove_map=groove, force_style=force_acc_style,
                                 harmony_complexity=harmony_src_dna.harmony_complexity)

    # Bajo
    bass = generate_bass(h_prog, target_key, n_bars, bpb, groove_map=groove)

    # Contrapunto [G]
    cp = generate_counterpoint(mel, h_prog, target_key, n_bars, bpb,
                               emotional_ctrl=ec)

    # Ornamentación contrapunto [F] — más sutil que la melodía
    cp_style = style if style in ORNAMENTATION_STYLES else 'classical'
    cp = add_ornamentation(cp, target_key, cp_style)

    return mel, acc, bass, cp, h_prog, rhythm_src_dna.rhythm_cell


def _auto_select_mode(dnas):
    e = [d.energy_mean for d in dnas]
    c = [d.harmony_complexity for d in dnas]
    r = [len(d.rhythm_pattern) for d in dnas]
    if all(len(d.fragments) > 2 for d in dnas): return 'mosaic'
    if max(e) - min(e) > 0.3: return 'energy'
    if max(c) - min(c) > 0.3: return 'harmony_melody'
    if max(r) - min(r) > 5:   return 'rhythm_melody'
    return 'harmony_melody'


def _transpose_sequence(pitch_seq, from_key, to_key):
    if not from_key or not to_key: return pitch_seq
    try:
        st = (to_key.tonic.midi % 12) - (from_key.tonic.midi % 12)
        if st > 6:  st -= 12
        if st < -6: st += 12
        return [p + st for p in pitch_seq]
    except Exception:
        return pitch_seq


def run_mixing(dnas, target_key, n_bars, tempo_bpm, time_sig, mode,
               emotion_src_idx, form_src_idx, sources,
               rhythm_strength, surprise_rate, humanize_groove,
               form_override=None, n_candidates=1, verbose=False,
               force_acc_style=None, voice_presets=None):
    """Motor principal de mezcla. Genera candidatos y elige el mejor. [N]"""

    ec, fg = _prepare_controllers(dnas, emotion_src_idx, form_src_idx, n_bars, form_override)

    # Seleccionar fuentes por modo
    by_harmony = sorted(dnas, key=lambda d: d.harmony_complexity, reverse=True)
    by_melody  = sorted(dnas, key=lambda d: len(d.pitch_sequence), reverse=True)
    by_rhythm  = sorted(dnas, key=lambda d: d.syncopation_ratio, reverse=True)
    by_energy  = sorted(dnas, key=lambda d: d.energy_mean, reverse=True)

    if mode == 'auto':
        mode = _auto_select_mode(dnas)
        print(f"  [auto] → modo {mode.upper()}")

    melody_mode = 'contour'

    if mode == 'rhythm_melody':
        h_src = dnas[2] if len(dnas) > 2 else dnas[0]
        m_src = dnas[1] if len(dnas) > 1 else dnas[0]
        r_src = dnas[0]
    elif mode == 'harmony_melody':
        h_src = dnas[0]
        m_src = dnas[1] if len(dnas) > 1 else dnas[0]
        r_src = dnas[2] if len(dnas) > 2 else dnas[0]
    elif mode == 'full_blend':
        h_src = by_harmony[0]
        m_src = by_melody[0]
        r_src = by_rhythm[0]
        # Contorno promediado
        contours = [d.pitch_contour for d in dnas if d.pitch_contour]
        if contours:
            max_len = max(len(c) for c in contours)
            m_src.pitch_contour = [int(np.mean([c[i%len(c)] for c in contours]))
                                   for i in range(max_len)]
    elif mode == 'custom':
        h_src = dnas[min(sources.get('harmony', 0), len(dnas)-1)]
        m_src = dnas[min(sources.get('melody',  1 if len(dnas)>1 else 0), len(dnas)-1)]
        r_src = dnas[min(sources.get('rhythm',  0), len(dnas)-1)]
    elif mode == 'mosaic':
        h_src = by_harmony[0]
        m_src = by_melody[0]
        r_src = by_rhythm[0]
        melody_mode = 'mosaic'
    elif mode == 'energy':
        h_src = by_harmony[0]
        m_src = by_energy[0]
        r_src = by_energy[0]
    elif mode == 'emotion':
        h_src = m_src = r_src = dnas[emotion_src_idx]
    else:
        h_src = m_src = r_src = dnas[0]

    # Decidir motor melódico
    has_markov = sum(len(v) for v in m_src.markov.transitions.values()) > 0
    has_sequitur = len(m_src.sequitur_phrases) > 0
    if melody_mode == 'contour':
        if has_markov:   melody_mode = 'markov'
        elif has_sequitur: melody_mode = 'sequitur'

    if verbose:
        print(f"  ♬ Armonía  → {h_src.name}")
        print(f"  🥁 Ritmo   → {r_src.name}")
        print(f"  🎵 Melodía → {m_src.name} (motor: {melody_mode})")

    # Candidatos [N]
    best_score, best_result, best_h_prog, best_rhythm_cell = -1, None, None, None
    for c in range(max(1, n_candidates)):
        random.seed(42 + c * 17)
        np.random.seed(42 + c * 17)
        if n_candidates > 1:
            print(f"  🎲 Candidato {c+1}/{n_candidates}")
        mel, acc, bass, cp, h_prog, rhythm_cell = _run_generation(
            h_src, m_src, r_src,
            target_key, n_bars, tempo_bpm, ec, fg, time_sig,
            rhythm_strength, melody_mode, surprise_rate, humanize_groove,
            dnas_all=dnas, force_acc_style=force_acc_style
        )
        sc = score_candidate(mel, acc, target_key)
        print(f"    Score: {sc:.3f}")
        if sc > best_score:
            best_score, best_result = sc, (mel, acc, bass, cp)
            best_h_prog, best_rhythm_cell = h_prog, rhythm_cell

    if n_candidates > 1:
        print(f"  🏆 Mejor candidato: score={best_score:.3f}")

    mel, acc, bass, cp = best_result
    groove = r_src.groove_map if r_src.groove_map.trained else None

    # ── Percusión desde rhythm_grid [A] ───────────────────────────────────────
    perc_notes = generate_percussion(
        r_src.rhythm_grid,
        r_src.rhythm_accent_grid,
        n_bars, time_sig[0],
        groove_map=groove,
        style=r_src.style
    )
    print(f"  🥁 Percusión: {len(perc_notes)} golpes generados")

    # ── Voces adicionales ─────────────────────────────────────────────────────
    extra_voices_out = []
    if voice_presets:
        print(f"\n  🎻 Generando {len(voice_presets)} voz/voces adicional(es)…")
        for vp in voice_presets:
            inst_name = vp['name']
            print(f"    ▸ {inst_name} (rol: {vp['role']}, "
                  f"rango: {vp['range'][0]}-{vp['range'][1]})")
            notes = generate_extra_voice(
                vp, mel, bass, best_h_prog,
                target_key, n_bars, ec,
                time_sig[0],
                rhythm_cell=best_rhythm_cell,
                groove_map=groove,
                humanize_flag=humanize_groove,
            )
            extra_voices_out.append({
                'notes':   notes,
                'name':    inst_name,
                'program': vp['program'],
            })
            print(f"      → {len(notes)} notas")

    return (mel, acc, bass, cp), ec, fg, extra_voices_out, perc_notes


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-ASIGNACIÓN DE ROLES  (script C)
# ══════════════════════════════════════════════════════════════════════════════

def auto_assign_roles(dnas):
    """Asigna roles melody/harmony/rhythm según características."""
    if len(dnas) == 1:
        return 0, 0, 0
    profiles = []
    for i, d in enumerate(dnas):
        profiles.append({
            'idx': i,
            'pitch_range': max(d.pitch_sequence)-min(d.pitch_sequence) if d.pitch_sequence else 0,
            'chord_ratio': d.harmony_complexity,
            'rhythm_var': d.syncopation_ratio,
        })
    melody_src  = max(profiles, key=lambda x: x['pitch_range'])['idx']
    harmony_src = max(profiles, key=lambda x: x['chord_ratio'])['idx']
    rhythm_src  = max(profiles, key=lambda x: x['rhythm_var'])['idx']
    return melody_src, harmony_src, rhythm_src


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME ASCII
# ══════════════════════════════════════════════════════════════════════════════

def print_dna_report(dnas, emotion_src_idx, form_src_idx):
    print("\n" + "═"*60)
    print("  INFORME DE ADN MUSICAL UNIFICADO")
    print("═"*60)
    for i, d in enumerate(dnas):
        markers = []
        if i == emotion_src_idx: markers.append("EMOCIÓN")
        if i == form_src_idx:    markers.append("FORMA")
        tag = f" [{', '.join(markers)}]" if markers else ""
        print(f"\n  #{i+1}{tag}: {d.name}")
        print(f"    Tonalidad : {d.key_obj.tonic.name} {d.key_obj.mode} | {d.tempo_bpm:.0f} BPM")
        print(f"    Forma     : {d.form_string}  ({d.n_unique_sections} secciones)")
        print(f"    Arco emoc.: {d.emotional_arc_label}")
        print(f"    Clímax    : {d.climax_position:.0%}")
        print(f"    Entropía  : mel={d.entropy_melodic:.2f}b  rit={d.entropy_rhythmic:.2f}b")
        print(f"    Estilo    : {d.style} | swing={d.swing}")
        print(f"    Síncopa   : {d.syncopation_ratio:.0%} | complejidad_arm={d.harmony_complexity:.2f}")
        markov_trans = sum(len(v) for v in d.markov.transitions.values())
        print(f"    Markov    : {markov_trans} transiciones | Frags: {len(d.fragments)}")
        tc = d.tension_curve
        if tc:
            width = 32
            buckets = [tc[int(j * len(tc) / width)] for j in range(width)]
            bar_str = "".join(
                "█" if v > 0.75 else "▓" if v > 0.5 else "░" if v > 0.25 else " "
                for v in buckets
            )
            print(f"    Tensión   : |{bar_str}|")
        ac = d.arousal_curve
        if ac:
            width = 32
            buckets_a = [(ac[int(j * len(ac) / width)] + 1) / 2 for j in range(width)]
            bar_str_a = "".join(
                "█" if v > 0.75 else "▓" if v > 0.5 else "░" if v > 0.25 else " "
                for v in buckets_a
            )
            print(f"    Arousal   : |{bar_str_a}|")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_sources(s):
    result = {}
    for item in s.split(','):
        if '=' in item:
            k, v = item.strip().split('=')
            result[k.strip()] = int(v.strip())
    return result

def parse_key_arg(s):
    try:
        parts = s.strip().split()
        return m21key.Key(parts[0], parts[1]) if len(parts) == 2 else m21key.Key(parts[0])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='MIDI DNA UNIFIED MIXER — Fusión de tres generaciones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Ficheros de entrada
    parser.add_argument('inputs', nargs='*', help='Ficheros MIDI de entrada')
    parser.add_argument('--melody',  help='Fichero MIDI que dona la melodía')
    parser.add_argument('--harmony', help='Fichero MIDI que dona la armonía')
    parser.add_argument('--rhythm',  help='Fichero MIDI que dona el ritmo')
    # Modo
    parser.add_argument('--mode', default='auto',
        choices=['auto','rhythm_melody','harmony_melody','full_blend',
                 'custom','mosaic','energy','emotion'])
    parser.add_argument('--emotion_src', type=int, default=0)
    parser.add_argument('--form_src',    type=int, default=0)
    parser.add_argument('--sources',     default='rhythm=0,melody=1,harmony=0')
    # Parámetros musicales
    parser.add_argument('--key',    default=None)
    parser.add_argument('--bars',   type=int,   default=16)
    parser.add_argument('--tempo',  type=float, default=None)
    parser.add_argument('--form',   default=None, help='Forma musical: AABA, ABAB, …')
    parser.add_argument('--surprise', type=float, default=0.08)
    parser.add_argument('--rhythm_strength', type=float, default=1.0)
    # Generación
    parser.add_argument('--acc-style', default=None,
        choices=['alberti', 'arpeggio', 'block', 'waltz'],
        help='Forzar estilo de acompañamiento en todos los compases')
    parser.add_argument('--voices', default=None,
        help='Voces extra: "violin,viola:inner,horn:pedal:0.7"  '
             'Formato: instrumento[:rol[:vel_scale]]  '
             f'Instrumentos: {", ".join(sorted(["violin","viola","cello","contrabass","harp","strings","flute","piccolo","oboe","clarinet","bassoon","saxophone","trumpet","horn","trombone","tuba","organ","pad","choir","vibraphone","marimba","guitar","guitar_mute"]))}  '
             f'Roles: counterpoint, inner, melody_double, pedal, bass_double, ostinato, arpeggio')
    parser.add_argument('--candidates', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    # Opciones
    parser.add_argument('--no-humanize', action='store_true')
    parser.add_argument('--export-xml',  action='store_true')
    parser.add_argument('--split-tracks',action='store_true',
        help='Exportar cada pista como fichero MIDI independiente')
    parser.add_argument('--no-percussion', action='store_true',
        help='Desactivar pista de percusión generada automáticamente')
    parser.add_argument('--verbose',     action='store_true')
    parser.add_argument('--output', default='output_unified.mid')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("═"*65)
    print("  MIDI DNA UNIFIED MIXER  v1.1")
    print("  Markov · Mosaic · Contrapunto · Groove · Forma · Emoción")
    print("  Walking Bass · Percusión · Modulación · Silencios · CC#64/11")
    print("═"*65)

    # ── Cargar ficheros ────────────────────────────────────────────────────────
    midi_paths = []
    if args.melody and args.harmony and args.rhythm:
        midi_paths = [args.melody, args.harmony, args.rhythm]
    elif args.inputs:
        midi_paths = [p for p in args.inputs if os.path.exists(p)]
    else:
        parser.print_help(); sys.exit(1)

    for p in midi_paths:
        if not os.path.exists(p):
            print(f"ERROR: no encontrado: {p}"); sys.exit(1)

    # ── Extraer ADN ───────────────────────────────────────────────────────────
    print(f"\n[1/4] Extrayendo ADN de {len(midi_paths)} MIDI(s)…")
    dnas = []
    for path in midi_paths:
        print(f"\n  ▶ {os.path.basename(path)}")
        dna = UnifiedDNA(path)
        if dna.extract(verbose=args.verbose):
            dnas.append(dna)
    if not dnas:
        print("ERROR: no se pudo extraer ADN."); sys.exit(1)

    # Auto-asignación si se usan roles explícitos
    if args.melody and args.harmony and args.rhythm:
        print(f"\n  Roles explícitos: melody={args.melody}, harmony={args.harmony}, rhythm={args.rhythm}")

    # ── Informe ───────────────────────────────────────────────────────────────
    esi = min(args.emotion_src, len(dnas)-1)
    fsi = min(args.form_src,   len(dnas)-1)
    print_dna_report(dnas, esi, fsi)

    # ── Parámetros globales ───────────────────────────────────────────────────
    target_key = parse_key_arg(args.key) if args.key else dnas[0].key_obj
    tempo_bpm  = args.tempo or dnas[0].tempo_bpm
    time_sig   = dnas[0].time_sig
    n_bars     = args.bars
    sources    = parse_sources(args.sources)

    # Parsear voces adicionales
    voice_presets = parse_voices_arg(args.voices) if args.voices else []

    print(f"\n[2/4] Configuración:")
    print(f"    Tonalidad : {target_key.tonic.name} {target_key.mode}")
    print(f"    Tempo     : {tempo_bpm:.0f} BPM | Compás: {time_sig[0]}/{time_sig[1]}")
    print(f"    Compases  : {n_bars} | Modo: {args.mode.upper()}")
    print(f"    Sorpresa  : {args.surprise:.2f} | Groove: {not args.no_humanize}")
    print(f"    Candidatos: {args.candidates}")
    if voice_presets:
        names = ", ".join(f"{vp['name']}({vp['role']})" for vp in voice_presets)
        print(f"    Voces ext.: {names}")

    # ── Mezcla ────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Generando…")
    (mel, acc, bass, cp), ec, fg, extra_voices, perc_notes = run_mixing(
        dnas, target_key, n_bars, tempo_bpm, time_sig,
        mode             = args.mode,
        emotion_src_idx  = esi,
        form_src_idx     = fsi,
        sources          = sources,
        rhythm_strength  = args.rhythm_strength,
        surprise_rate    = args.surprise,
        humanize_groove  = not args.no_humanize,
        form_override    = args.form,
        n_candidates     = args.candidates,
        verbose          = args.verbose,
        force_acc_style  = args.acc_style,
        voice_presets    = voice_presets,
    )

    print(f"    → Melodía         : {len(mel)} notas")
    print(f"    → Contrapunto     : {len(cp)} notas")
    print(f"    → Acompañamiento  : {len(acc)} eventos")
    print(f"    → Bajo            : {len(bass)} notas")
    if not args.no_percussion:
        print(f"    → Percusión       : {len(perc_notes)} golpes")
    for ev in extra_voices:
        print(f"    → {ev['name']:16s}: {len(ev['notes'])} notas")

    # ── Exportar MIDI ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Exportando…")
    build_midi(mel, acc, bass, cp, target_key, tempo_bpm, time_sig, n_bars,
               form_gen=fg, output_path=args.output,
               extra_voices=extra_voices if extra_voices else None,
               percussion_notes=perc_notes if not args.no_percussion else None,
               split_tracks=args.split_tracks)

    # ── Export XML opcional [R] ───────────────────────────────────────────────
    if args.export_xml:
        xml_path = args.output.replace('.mid', '.xml').replace('.midi', '.xml')
        if not xml_path.endswith('.xml'): xml_path += '.xml'
        try:
            sc_out = stream.Score()
            sc_out.insert(0, tempo.MetronomeMark(number=tempo_bpm))
            sc_out.insert(0, target_key)

            def notes_to_part(notes_list, part_name, program=0):
                p = stream.Part(id=part_name)
                p.insert(0, instrument.Piano() if program == 0 else instrument.AcousticBass())
                p.insert(0, meter.TimeSignature(f'{time_sig[0]}/{time_sig[1]}'))
                for offset, mp, dur, vel in sorted(notes_list, key=lambda x: x[0]):
                    n = note.Note(mp)
                    n.quarterLength = max(0.1, dur)
                    n.volume.velocity = int(np.clip(vel, 1, 127))
                    p.insert(float(offset), n)
                return p

            sc_out.insert(0, notes_to_part(mel,  'Melody'))
            sc_out.insert(0, notes_to_part(cp,   'Counterpoint'))
            sc_out.insert(0, notes_to_part(acc,  'Accompaniment'))
            sc_out.insert(0, notes_to_part(bass, 'Bass', 32))
            for ev in extra_voices:
                sc_out.insert(0, notes_to_part(ev['notes'], ev['name'], ev['program']))
            sc_out.write('musicxml', fp=xml_path)
            print(f"  → MusicXML guardado: {xml_path}")
        except Exception as e:
            print(f"  ⚠ MusicXML falló: {e}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  RESUMEN FINAL")
    print("═"*65)
    print(f"  Tonalidad : {target_key.tonic.name} {target_key.mode}")
    print(f"  Tempo     : {tempo_bpm:.0f} BPM | Compás: {time_sig[0]}/{time_sig[1]}")
    print(f"  Compases  : {n_bars}")
    print(f"  Forma     : {fg.form_string}")
    print(f"  Arco      : {dnas[esi].emotional_arc_label}")
    print(f"  Fuentes   : {' + '.join(d.name for d in dnas)}")
    if extra_voices:
        voice_summary = ', '.join(f"{ev['name']}({VOICE_PRESETS[ev['name']]['role']})"
                                   for ev in extra_voices)
        print(f"  Voces ext.: {voice_summary}")
    n_perc = 0 if args.no_percussion else 1
    print(f"  Pistas    : {4 + len(extra_voices) + n_perc}")
    print(f"  Fichero   : {args.output}")
    print("═"*65)


if __name__ == '__main__':
    main()
