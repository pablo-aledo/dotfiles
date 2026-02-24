"""
midi_dna_mixer.py  v2.0
=======================
Extrae el "ADN musical completo" de varios MIDIs y los combina en uno nuevo.

NOVEDADES v2.0 respecto a v1:
  ① ADN EXTENDIDO      Tensión (Lerdahl), Tonnetz, disonancia sensorial,
                        entropía/sorpresa, cadencias, textura, contorno avanzado.
  ② ESTRUCTURA FORMAL  Segmentación automática en secciones (A, B, C…),
                        detección de frases, puntos de cadencia y repeticiones.
                        La pieza generada hereda la forma de la fuente.
  ③ ARCO EMOCIONAL     Curvas compás-a-compás de Valencia, Arousal, Tensión
                        armónica y Actividad melódica. El generador modula
                        dinámica, densidad, registro y armonía para reproducir
                        el arco emocional elegido.

USO:
    python midi_dna_mixer.py midi1.mid midi2.mid [midi3.mid ...] [opciones]

OPCIONES PRINCIPALES:
    --mode          rhythm_melody | harmony_melody | full_blend | custom
    --emotion_src   Índice del MIDI que dona el arco emocional (default: 0)
    --form_src      Índice del MIDI que dona la estructura formal (default: 0)
    --sources       Para modo custom: rhythm=N,melody=N,harmony=N
    --key           Tonalidad destino ("C major", "A minor", …)
    --bars          Número de compases a generar (default: 16)
    --tempo         BPM (default: detectado del primer MIDI)
    --output        Fichero de salida (default: output_dna_mix.mid)

EJEMPLO:
    python midi_dna_mixer.py bach.mid jazz.mid bossa.mid \\
        --mode harmony_melody --emotion_src 0 --form_src 1 --bars 32

DEPENDENCIAS:
    pip install music21 mido numpy scipy scikit-learn
"""

import sys
import os
import argparse
import random
import copy
import hashlib
from collections import defaultdict, Counter

import numpy as np

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

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from music21 import (
        converter, stream, note, chord, meter, tempo, key as m21key,
        instrument, harmony, roman, pitch, interval, scale,
        duration, analysis, environment
    )
    environment.UserSettings()['warnings'] = 0
except ImportError:
    print("ERROR: pip install music21")
    sys.exit(1)

try:
    import mido
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  CONSTANTES
# ════════════════════════════════════════════════════════════════

HARMONIC_FUNCTIONS = {
    "T":    ["I", "i", "vi", "VI"],
    "PD":   ["ii", "II", "iv", "IV"],
    "D":    ["V", "v", "vii°", "VII"],
    "Dsec": ["V/V", "V/ii", "V/vi", "V/IV"],
    "Other": []
}

FUNC_TENSION = {"T": 0.1, "PD": 0.4, "D": 0.7, "Dsec": 0.9, "Other": 0.5}

DISSONANCE_WEIGHTS = {
    0: 0.0,  1: 1.0,  2: 0.8,  3: 0.2,  4: 0.1,
    5: 0.05, 6: 0.9,  7: 0.02, 8: 0.25, 9: 0.15,
    10: 0.7, 11: 0.95
}


# ════════════════════════════════════════════════════════════════
#  UTILIDADES GENERALES
# ════════════════════════════════════════════════════════════════

def load_score(path):
    try:
        return converter.parse(path)
    except Exception as e:
        print(f"  ERROR cargando {path}: {e}")
        return None

def detect_key(score):
    try:
        return score.analyze('key')
    except:
        return m21key.Key('C', 'major')

def detect_tempo_bpm(score):
    for el in score.flatten():
        if isinstance(el, tempo.MetronomeMark) and el.number:
            return float(el.number)
    return 120.0

def detect_time_signature(score):
    ts_list = []
    for el in score.flatten():
        if isinstance(el, meter.TimeSignature):
            ts_list.append((el.numerator, el.denominator))
    return Counter(ts_list).most_common(1)[0][0] if ts_list else (4, 4)

def snap_to_scale(midi_note, key_obj):
    try:
        sc = key_obj.getScale()
        pcs = [p.midi % 12 for p in sc.getPitches()]
        octave = midi_note // 12
        note_pc = midi_note % 12
        if note_pc in pcs:
            return midi_note
        nearest = min(pcs, key=lambda p: abs(p - note_pc))
        return octave * 12 + nearest
    except:
        return midi_note

def roman_to_chord_pitches(roman_figure, key_obj, octave=4):
    try:
        rn = roman.RomanNumeral(roman_figure, key_obj)
        base_midi = key_obj.tonic.midi + (octave - 4) * 12
        pitches = []
        for p in rn.pitches:
            mv = p.midi
            while mv < base_midi - 6:  mv += 12
            while mv > base_midi + 18: mv -= 12
            pitches.append(mv)
        return sorted(pitches)
    except:
        try:
            t = key_obj.tonic.midi + (octave - 4) * 12
            return [t, t + (4 if key_obj.mode == 'major' else 3), t + 7]
        except:
            return [60, 64, 67]

def roman_to_function(fig):
    if "/" in fig:
        return "Dsec"
    base = fig.replace("°","").replace("+","")
    base = "".join(c for c in base if not c.isdigit())
    for func, romans in HARMONIC_FUNCTIONS.items():
        if base in romans:
            return func
    return "Other"

def transpose_to_key(pitch_seq, from_key, to_key):
    if not from_key or not to_key:
        return pitch_seq
    try:
        st = (to_key.tonic.midi % 12) - (from_key.tonic.midi % 12)
        if st > 6:  st -= 12
        if st < -6: st += 12
        return [p + st for p in pitch_seq]
    except:
        return pitch_seq


# ════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ADN EXTENDIDO
# ════════════════════════════════════════════════════════════════

class MusicalDNA:
    """
    ADN musical completo extraído de un MIDI:

    BÁSICO
      key_obj, tempo_bpm, time_sig
      rhythm_pattern, pitch_sequence, pitch_contour, pitch_register
      harmony_prog, motif_intervals, dynamics_mean, dynamics_std

    EXTENDIDO (nuevo en v2)
      tension_curve        → Lerdahl: tensión normalizada compás a compás [0-1]
      tonnetz_curve        → Excursión armónica compás a compás (distancia tonal)
      roughness_curve      → Disonancia sensorial compás a compás [0-1]
      activity_curve       → Actividad melódica compás a compás [0-1]
      valence_curve        → Valencia emocional compás a compás [-1,1]
      arousal_curve        → Arousal compás a compás [-1,1]
      stability_curve      → Estabilidad tonal compás a compás [0-1]
      entropy_melodic      → Sorpresa melódica media (bits)
      entropy_rhythmic     → Sorpresa rítmica media (bits)
      climax_position      → Posición relativa del clímax [0-1]
      resolution_index     → Índice de resolución final [0-1]
      emotional_arc_label  → Clasificación del arco ("crescendo","decrescendo"…)

    ESTRUCTURA FORMAL (nuevo en v2)
      form_string          → Secuencia de secciones ("AABACBA")
      section_map          → [(section_label, bar_start, bar_end), …]
      phrase_lengths       → Lista de duraciones de frase (en compases)
      cadence_positions    → Lista de compases con cadencia
      n_unique_sections    → Nº de secciones distintas
    """

    def __init__(self, path):
        self.path = path
        self.score = None
        # básico
        self.key_obj = m21key.Key('C', 'major')
        self.tempo_bpm = 120.0
        self.time_sig = (4, 4)
        self.rhythm_pattern = []
        self.pitch_sequence = []
        self.pitch_contour = []
        self.pitch_register = 60
        self.harmony_prog = []
        self.motif_intervals = []
        self.dynamics_mean = 72
        self.dynamics_std = 12
        # extendido
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
        # estructura formal
        self.form_string = "AABA"
        self.section_map = []
        self.phrase_lengths = [4, 4, 4, 4]
        self.cadence_positions = []
        self.n_unique_sections = 2

    # ─────────────────────────────────────────────────────────────
    #  EXTRACCIÓN PRINCIPAL
    # ─────────────────────────────────────────────────────────────

    def extract(self):
        sc = load_score(self.path)
        if sc is None:
            return False
        self.score = sc

        print(f"    → Tonalidad…")
        self.key_obj = detect_key(sc)
        self.tempo_bpm = detect_tempo_bpm(sc)
        self.time_sig = detect_time_signature(sc)
        print(f"       {self.key_obj.tonic.name} {self.key_obj.mode} | "
              f"{self.tempo_bpm:.0f} BPM | {self.time_sig[0]}/{self.time_sig[1]}")

        print(f"    → Melodía, ritmo, armonía…")
        self._extract_melody()
        self._extract_rhythm()
        self._extract_harmony()
        self._extract_motif()
        self._extract_dynamics()

        print(f"    → Curvas emocionales (Lerdahl, Tonnetz, Valencia/Arousal)…")
        self._extract_tension_curve()
        self._extract_tonnetz_curve()
        self._extract_roughness_curve()
        self._extract_activity_curve()
        self._extract_valence_arousal_curves()
        self._extract_stability_curve()
        self._extract_entropy()
        self._classify_emotional_arc()

        print(f"    → Estructura formal…")
        self._extract_form_structure()

        print(f"    ✓ ADN extraído. Notas={len(self.pitch_sequence)}, "
              f"Acordes={len(self.harmony_prog)}, "
              f"Forma={self.form_string}, "
              f"Arco={self.emotional_arc_label}")
        return True

    # ─────────────────────────────────────────────────────────────
    #  MELODÍA / RITMO / ARMONÍA (igual que v1, compactado)
    # ─────────────────────────────────────────────────────────────

    def _get_main_melody_part(self):
        parts = self.score.parts
        if not parts:
            return self.score.flatten()
        best, best_sc = None, -1
        for p in parts:
            ns = [n for n in p.flatten().notes if isinstance(n, note.Note)]
            if not ns:
                continue
            sc = np.mean([n.pitch.midi for n in ns]) * 0.5 + len(ns) * 0.01
            if sc > best_sc:
                best_sc, best = sc, p
        return best if best else parts[0]

    def _extract_melody(self):
        part = self._get_main_melody_part()
        mel = []
        for el in part.flatten().notes:
            if isinstance(el, note.Note):
                mel.append((float(el.offset), el.pitch.midi, float(el.quarterLength)))
            elif isinstance(el, chord.Chord):
                top = max(el.pitches, key=lambda p: p.midi)
                mel.append((float(el.offset), top.midi, float(el.quarterLength)))
        mel.sort(key=lambda x: x[0])
        self.pitch_sequence = [m[1] for m in mel]
        self.pitch_register = int(np.mean(self.pitch_sequence)) if self.pitch_sequence else 60
        if len(self.pitch_sequence) >= 2:
            self.pitch_contour = list(np.diff(self.pitch_sequence))

    def _extract_rhythm(self):
        part = self._get_main_melody_part()
        bpb = self.time_sig[0]
        measures = list(part.getElementsByClass('Measure'))
        if not measures:
            all_notes = [(float(el.offset), float(el.quarterLength))
                         for el in part.flatten().notes]
            if not all_notes:
                self.rhythm_pattern = [[(0.0, 1.0)] * bpb]
                return
            max_off = max(n[0] for n in all_notes)
            for bs in range(0, int(max_off) + 1, bpb):
                bar = [(o - bs, d) for o, d in all_notes
                       if bs <= o < bs + bpb]
                self.rhythm_pattern.append(bar)
        else:
            for m in measures:
                bar = []
                for el in m.notes:
                    bar.append((float(el.offset), float(el.quarterLength)))
                self.rhythm_pattern.append(bar)
        if len(self.rhythm_pattern) < 4:
            self.rhythm_pattern = self.rhythm_pattern * 8

    def _extract_harmony(self):
        sc = self.score
        bpb = self.time_sig[0]
        k = self.key_obj
        all_notes = defaultdict(list)
        for el in sc.flatten().notes:
            if isinstance(el, note.Note):
                slot = round(float(el.offset))
                all_notes[slot].append(el.pitch)
            elif isinstance(el, chord.Chord):
                slot = round(float(el.offset))
                for p in el.pitches:
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
                except:
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

    def _extract_motif(self):
        c = self.pitch_contour
        self.motif_intervals = c[:8] if len(c) >= 8 else (c * (8 // max(len(c), 1) + 1))[:8]

    def _extract_dynamics(self):
        vels = []
        for el in self.score.flatten().notes:
            if isinstance(el, note.Note) and el.volume.velocity:
                vels.append(el.volume.velocity)
            elif isinstance(el, chord.Chord):
                for n2 in el.notes:
                    if n2.volume.velocity:
                        vels.append(n2.volume.velocity)
        if vels:
            self.dynamics_mean = int(np.mean(vels))
            self.dynamics_std  = int(np.std(vels))

    # ─────────────────────────────────────────────────────────────
    #  CURVAS EMOCIONALES (nuevo v2)
    # ─────────────────────────────────────────────────────────────

    def _get_measures(self):
        """Obtiene lista de Measure de la primera parte."""
        parts = getattr(self.score, 'parts', [])
        if parts:
            ms = list(parts[0].getElementsByClass('Measure'))
            if ms:
                return ms
        return list(self.score.flatten().getElementsByClass('Measure'))

    def _extract_tension_curve(self):
        """
        Tensión de Lerdahl por compás: combinación de tensión melódica
        (jerarquía tonal), armónica (disonancia) y rítmica (densidad).
        Resultado normalizado a [0-1].
        """
        measures = self._get_measures()
        if not measures:
            self.tension_curve = [0.5]
            return
        k = self.key_obj
        try:
            tonica  = k.pitchFromDegree(1).name
            medinat = k.pitchFromDegree(3).name
            dominant= k.pitchFromDegree(5).name
        except:
            tonica = medinat = dominant = "C"
        stability_map = {tonica: 0, dominant: 1, medinat: 2}
        scale_names = [p.name for p in k.getScale().pitches]

        curve = []
        for m in measures:
            ns = [el for el in m.flatten().notes]
            if not ns:
                curve.append(curve[-1] if curve else 0.3)
                continue
            m_tensions, h_tensions, r_tensions = [], [], []
            for el in ns:
                if isinstance(el, note.Note):
                    pn = el.pitch.name
                    mt = stability_map.get(pn, 4)
                    if pn not in scale_names:
                        mt += 3
                    m_tensions.append(mt)
                    r_tensions.append(1.0 / (el.quarterLength + 0.1))
                elif isinstance(el, chord.Chord):
                    ht = 0
                    if not el.isConsonant():
                        ht += 4
                    ht += len(el.pitches) * 0.5
                    h_tensions.append(ht)
                    r_tensions.append(1.0 / (el.quarterLength + 0.1))

            total = (np.mean(m_tensions) if m_tensions else 3.0) + \
                    (np.mean(h_tensions) if h_tensions else 0.0) + \
                    (np.mean(r_tensions) if r_tensions else 1.0)
            curve.append(total)

        arr = np.array(curve, dtype=float)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.ones_like(arr) * 0.5
        if SCIPY_OK and len(arr) > 3:
            arr = gaussian_filter(arr, sigma=1.5)
        self.tension_curve = arr.tolist()

        # Posición del clímax y resolución
        self.climax_position = float(np.argmax(arr) / max(len(arr) - 1, 1))
        self.resolution_index = float(np.mean(arr[-max(3, len(arr)//8):]))

    def _extract_tonnetz_curve(self):
        """
        Excursión armónica por compás usando coordenadas Tonnetz.
        Distancia euclidiana al centro tonal.
        """
        measures = self._get_measures()
        if not measures:
            self.tonnetz_curve = [0.3]
            return
        try:
            tonic_pc = self.key_obj.tonic.pitchClass
        except:
            tonic_pc = 0

        TONNETZ = {0:(0,0), 1:(-2.5,.866), 2:(2,0), 3:(-1.5,.866),
                   4:(.5,.866), 5:(-1,0), 6:(2.5,.866), 7:(1,0),
                   8:(-.5,.866), 9:(1.5,.866), 10:(-2,0), 11:(3.5,.866)}

        curve = []
        for m in measures:
            pcs = [el.pitch.pitchClass for el in m.flatten().notes
                   if isinstance(el, note.Note)]
            for ch in m.flatten().notes:
                if isinstance(ch, chord.Chord):
                    pcs.extend(p.pitchClass for p in ch.pitches)
            if not pcs:
                curve.append(curve[-1] if curve else 0.0)
                continue
            coords = [TONNETZ.get((p - tonic_pc) % 12, (0,0)) for p in pcs]
            cx = np.mean([c[0] for c in coords])
            cy = np.mean([c[1] for c in coords])
            curve.append(float(np.sqrt(cx**2 + cy**2)))

        arr = np.array(curve, dtype=float)
        if arr.max() > 0:
            arr /= arr.max()
        if SCIPY_OK and len(arr) > 3:
            arr = gaussian_filter(arr, sigma=1.2)
        self.tonnetz_curve = arr.tolist()

    def _extract_roughness_curve(self):
        """
        Disonancia sensorial (rugosidad psicoacústica) compás a compás.
        Suma de pesos de disonancia por pares de notas simultáneas.
        """
        measures = self._get_measures()
        if not measures:
            self.roughness_curve = [0.3]
            return
        curve = []
        for m in measures:
            pcs = set()
            for el in m.flatten().notes:
                if isinstance(el, note.Note):
                    pcs.add(el.pitch.midi)
                elif isinstance(el, chord.Chord):
                    pcs.update(p.midi for p in el.pitches)
            ps = sorted(pcs)
            if len(ps) < 2:
                curve.append(0.0)
                continue
            rough = 0.0
            n_pairs = 0
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    ic = abs(ps[j] - ps[i]) % 12
                    w = DISSONANCE_WEIGHTS.get(ic, 0.5)
                    penalty = 1.2 if ps[i] < 48 else 1.0
                    rough += w * penalty
                    n_pairs += 1
            curve.append(rough / n_pairs if n_pairs else 0.0)

        arr = np.array(curve, dtype=float)
        if arr.max() > 0:
            arr /= arr.max()
        if SCIPY_OK and len(arr) > 3:
            arr = gaussian_filter(arr, sigma=1.0)
        self.roughness_curve = arr.tolist()

    def _extract_activity_curve(self):
        """
        Actividad melódica compás a compás:
        combina densidad, rango, movimiento y saltos.
        """
        measures = self._get_measures()
        if not measures:
            self.activity_curve = [0.5]
            return
        curve = []
        for m in measures:
            ns = [el for el in m.flatten().notes if isinstance(el, note.Note)]
            if not ns:
                curve.append(0.0)
                continue
            dur = m.quarterLength or 4
            density = min(len(ns) / max(dur * 2, 1), 1.0)
            ps = [n.pitch.midi for n in ns]
            rng = min((max(ps) - min(ps)) / 24.0, 1.0)
            if len(ps) >= 2:
                ivs = np.abs(np.diff(ps))
                motion = min(np.mean(ivs) / 4.0, 1.0)
                leaps  = sum(1 for i in ivs if i > 2) / len(ivs)
            else:
                motion = 0.0
                leaps  = 0.0
            activity = density * 0.35 + rng * 0.25 + motion * 0.25 + leaps * 0.15
            curve.append(float(activity))

        arr = np.array(curve, dtype=float)
        if SCIPY_OK and len(arr) > 3:
            arr = gaussian_filter(arr, sigma=1.5)
        self.activity_curve = arr.tolist()

    def _extract_valence_arousal_curves(self):
        """
        Modelo circunflejo de Russell por compás:
        Valence [-1,1]: modo + consonancia
        Arousal [-1,1]: densidad + registro
        """
        measures = self._get_measures()
        if not measures:
            self.valence_curve = [0.0]
            self.arousal_curve = [0.0]
            return
        k = self.key_obj
        scale_pcs = set(p.pitchClass for p in k.getScale().pitches)
        global_mode_val = 0.5 if k.mode == 'major' else -0.5

        valences, arousals = [], []
        for m in measures:
            ns = list(m.flatten().notes)
            if not ns:
                valences.append(0.0)
                arousals.append(-0.5)
                continue

            # Arousal
            dur = m.quarterLength or 4
            density = len(ns) / dur
            ps_midi = []
            for el in ns:
                if isinstance(el, note.Note):
                    ps_midi.append(el.pitch.midi)
                elif isinstance(el, chord.Chord):
                    ps_midi.append(el.sortAscending().pitches[-1].midi)
            avg_pitch = np.mean(ps_midi) if ps_midi else 60
            ar = (min(density / 8, 1.0) * 0.6 + (avg_pitch - 36) / 48 * 0.4) * 2 - 1
            arousals.append(float(np.clip(ar, -1, 1)))

            # Valence: diatonismo + modo
            all_ps = []
            for el in ns:
                if isinstance(el, note.Note):
                    all_ps.append(el.pitch.pitchClass)
                elif isinstance(el, chord.Chord):
                    all_ps.extend(p.pitchClass for p in el.pitches)
            if all_ps:
                dia = sum(1 for p in all_ps if p in scale_pcs) / len(all_ps)
            else:
                dia = 0.5
            # Consonancia de intervalos únicos
            unique_ps = sorted(set(all_ps))
            if len(unique_ps) >= 2:
                ivs = [(unique_ps[i+1] - unique_ps[i]) % 12 for i in range(len(unique_ps)-1)]
                cons_ivs = {0, 3, 4, 5, 7, 8, 9}
                cons = sum(1 for iv in ivs if iv in cons_ivs) / len(ivs)
            else:
                cons = 0.5
            val = (global_mode_val + (cons - 0.5)) * dia
            valences.append(float(np.clip(val, -1, 1)))

        def smooth(lst):
            a = np.array(lst, dtype=float)
            if SCIPY_OK and len(a) > 3:
                return gaussian_filter(a, sigma=2.0).tolist()
            return lst

        self.valence_curve = smooth(valences)
        self.arousal_curve = smooth(arousals)

    def _extract_stability_curve(self):
        """
        Estabilidad tonal por compás [0-1]:
        diatonismo + proximidad a tónica + claridad funcional.
        """
        measures = self._get_measures()
        if not measures:
            self.stability_curve = [0.7]
            return
        k = self.key_obj
        scale_pcs = set(p.pitchClass for p in k.getScale().pitches)
        tonic_pc  = k.tonic.pitchClass
        func_map  = {"T": 1.0, "PD": 0.6, "D": 0.4, "Dsec": 0.2, "Other": 0.3}

        curve = []
        for m in measures:
            all_ps = []
            for el in m.flatten().notes:
                if isinstance(el, note.Note):
                    all_ps.append(el.pitch)
                elif isinstance(el, chord.Chord):
                    all_ps.extend(el.pitches)
            if not all_ps:
                curve.append(1.0)
                continue

            dia = sum(1 for p in all_ps if p.pitchClass in scale_pcs) / len(all_ps)

            dists = [min(abs(p.pitchClass - tonic_pc), 12 - abs(p.pitchClass - tonic_pc))
                     for p in all_ps]
            prox = 1.0 - (np.mean(dists) / 6.0)

            try:
                pcs = list(set(p.pitchClass for p in all_ps))
                if len(pcs) >= 3:
                    tc = chord.Chord([pitch.Pitch(pc) for pc in pcs])
                    rn = roman.romanNumeralFromChord(tc, k)
                    fc = func_map.get(roman_to_function(rn.figure), 0.5)
                else:
                    fc = 0.5
            except:
                fc = 0.5

            stab = dia * 0.40 + prox * 0.35 + fc * 0.25
            curve.append(float(np.clip(stab, 0, 1)))

        arr = np.array(curve, dtype=float)
        if SCIPY_OK and len(arr) > 3:
            arr = gaussian_filter(arr, sigma=1.5)
        self.stability_curve = arr.tolist()

    def _extract_entropy(self):
        """
        Entropía melódica y rítmica usando modelo de Markov de n-grama.
        Sorpresa (information content) media.
        """
        all_els = [el for el in self.score.flatten().notes
                   if isinstance(el, (note.Note, chord.Chord))]
        if len(all_els) < 4:
            return

        def ic_series(values, n=3):
            model = defaultdict(Counter)
            ctx_cnt = defaultdict(int)
            for i in range(len(values) - 1):
                ctx = tuple(values[max(0, i - n + 1): i + 1])
                model[ctx][values[i + 1]] += 1
                ctx_cnt[ctx] += 1
            ics = []
            for i in range(len(values) - 1):
                ctx = tuple(values[max(0, i - n + 1): i + 1])
                nxt = values[i + 1]
                prob = model[ctx][nxt] / ctx_cnt[ctx] if ctx_cnt[ctx] > 0 else 0.0001
                ics.append(-np.log2(prob))
            return np.mean(ics) if ics else 1.0

        # Intervalos melódicos
        pitches = []
        for el in all_els:
            if isinstance(el, note.Note):
                pitches.append(el.pitch.midi)
            elif isinstance(el, chord.Chord):
                try:
                    pitches.append(el.sortAscending().pitches[-1].midi)
                except:
                    pass
        if len(pitches) >= 3:
            ivs = [max(-12, min(12, int(pitches[i+1] - pitches[i])))
                   for i in range(len(pitches) - 1)]
            self.entropy_melodic = float(ic_series(ivs))

        # Relaciones rítmicas
        durs = [el.quarterLength for el in all_els if hasattr(el, 'quarterLength')]
        if len(durs) >= 3:
            rhy = []
            for i in range(len(durs) - 1):
                r = durs[i + 1] / (durs[i] + 1e-9)
                rhy.append("L" if r > 1.1 else "S" if r < 0.9 else "E")
            self.entropy_rhythmic = float(ic_series(rhy))

    def _classify_emotional_arc(self):
        """
        Clasifica el arco emocional en una de 8 categorías según las curvas.
        """
        if not self.tension_curve or not self.arousal_curve:
            self.emotional_arc_label = "neutral"
            return

        t = np.array(self.tension_curve)
        a = np.array(self.arousal_curve)
        n = len(t)

        t_inicio = np.mean(t[:max(1, n // 4)])
        t_medio  = np.mean(t[n // 4: 3 * n // 4])
        t_fin    = np.mean(t[3 * n // 4:])

        a_inicio = np.mean(a[:max(1, n // 4)])
        a_fin    = np.mean(a[3 * n // 4:])

        climax_early = self.climax_position < 0.4
        climax_late  = self.climax_position > 0.65

        if t_inicio < t_medio and t_fin < t_medio:
            arc = "arch"           # Tensión crece y cae: forma arco
        elif t_fin > t_inicio * 1.3:
            arc = "crescendo"      # Tensión creciente
        elif t_inicio > t_fin * 1.3:
            arc = "decrescendo"    # Tensión decreciente
        elif a_inicio < a_fin:
            arc = "awakening"      # Energía que despierta
        elif a_inicio > a_fin:
            arc = "lullaby"        # Energía que adormece
        elif np.std(t) < 0.12:
            arc = "plateau"        # Tensión estable
        elif climax_late:
            arc = "late_climax"    # Clímax tardío (sonata, rondó)
        else:
            arc = "neutral"

        self.emotional_arc_label = arc

    # ─────────────────────────────────────────────────────────────
    #  ESTRUCTURA FORMAL (nuevo v2)
    # ─────────────────────────────────────────────────────────────

    def _extract_form_structure(self):
        """
        Segmenta la pieza en secciones (A, B, C…) comparando
        descriptores melódico-armónicos de ventanas de compases.
        Detecta también la longitud de frase y posiciones de cadencia.
        """
        measures = self._get_measures()
        if not measures:
            self.form_string = "A"
            self.section_map = [("A", 1, 1)]
            self.phrase_lengths = [4]
            return

        n_bars = len(measures)
        window = max(2, min(4, n_bars // 4))

        # Descriptores por ventana
        descriptors = []
        bar_windows = []
        i = 0
        while i < n_bars:
            seg_measures = measures[i: i + window]
            seg_stream = stream.Stream()
            for m in seg_measures:
                seg_stream.append(copy.deepcopy(m))
            flat = seg_stream.flatten()

            # Vector descriptor: intervalos melódicos + notas de la escala
            ns = [el for el in flat.notes if isinstance(el, note.Note)]
            pitches = [n.pitch.midi for n in ns]
            if len(pitches) >= 2:
                ivs = list(np.diff(pitches))
                hist, _ = np.histogram(ivs, bins=12, range=(-12, 12))
                mel_vec = hist / (hist.sum() + 1e-9)
            else:
                mel_vec = np.zeros(12)

            # Altura media + rango
            if pitches:
                extra = np.array([np.mean(pitches) / 127, (max(pitches) - min(pitches)) / 24])
            else:
                extra = np.zeros(2)

            desc = np.concatenate([mel_vec, extra])
            norm = np.linalg.norm(desc)
            desc = desc / norm if norm > 0 else desc

            descriptors.append(desc)
            bar_windows.append((i + 1, min(i + window, n_bars)))
            i += window

        # Clustering: agrupa ventanas similares en secciones
        n_segs = len(descriptors)
        if n_segs < 2:
            labels = [0] * n_segs
        elif SKLEARN_OK:
            try:
                n_clust = max(2, min(n_segs, max(2, n_segs // 3)))
                clust = AgglomerativeClustering(
                    n_clusters=n_clust, metric='cosine', linkage='average'
                )
                labels = list(clust.fit_predict(np.array(descriptors)))
            except Exception:
                labels = list(range(n_segs))
        else:
            # Fallback sin sklearn: umbral de similitud coseno
            labels = [0]
            for j in range(1, n_segs):
                d = descriptors[j]
                # Comparar con el último grupo
                last_d = descriptors[j - 1]
                cos = float(np.dot(d, last_d) / (np.linalg.norm(d) * np.linalg.norm(last_d) + 1e-9))
                if cos > 0.92:
                    labels.append(labels[-1])
                else:
                    labels.append(max(labels) + 1)

        # Mapear labels a letras A, B, C...
        lmap = {}
        next_ch = ord('A')
        letter_labels = []
        for lbl in labels:
            if lbl not in lmap:
                lmap[lbl] = chr(next_ch)
                next_ch += 1
            letter_labels.append(lmap[lbl])

        self.form_string = "".join(letter_labels)
        self.n_unique_sections = len(set(letter_labels))

        # Mapa de secciones: (letra, bar_inicio, bar_fin)
        self.section_map = [
            (letter_labels[j], bar_windows[j][0], bar_windows[j][1])
            for j in range(len(letter_labels))
        ]

        # Longitudes de frase: tamaño de cada ventana en compases
        self.phrase_lengths = [w[1] - w[0] + 1 for w in bar_windows]

        # Posiciones de cadencia: último compás de cada sección
        self.cadence_positions = [w[1] for w in bar_windows]

        print(f"       Forma: {self.form_string} | "
              f"{self.n_unique_sections} secciones distintas | "
              f"Frases: {self.phrase_lengths[:4]}…")


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN: MÓDULO DE CONTROL EMOCIONAL
# ════════════════════════════════════════════════════════════════

class EmotionalController:
    """
    Convierte curvas emocionales en parámetros musicales compás a compás.

    Parámetros de salida por compás:
      velocity_target   → velocidad MIDI [40-110]
      register_offset   → desplazamiento de registro en semitonos [-12, 12]
      density_mult      → multiplicador de densidad rítmica [0.4, 2.0]
      acc_style         → estilo de acompañamiento ('block','arpeggio','alberti')
      harmony_tension   → preferencia de acordes: 0=tónica, 1=dominante
    """

    def __init__(self, tension_curve, arousal_curve, valence_curve,
                 stability_curve, activity_curve, emotional_arc_label):
        self.tension   = self._normalize(tension_curve)
        self.arousal   = self._normalize_pm(arousal_curve)
        self.valence   = self._normalize_pm(valence_curve)
        self.stability = self._normalize(stability_curve)
        self.activity  = self._normalize(activity_curve)
        self.arc       = emotional_arc_label

    @staticmethod
    def _normalize(curve):
        a = np.array(curve, dtype=float)
        if a.max() > a.min():
            return ((a - a.min()) / (a.max() - a.min())).tolist()
        return [0.5] * len(a)

    @staticmethod
    def _normalize_pm(curve):
        """Mantiene en [-1,1] pero normaliza."""
        a = np.array(curve, dtype=float)
        mx = max(abs(a.max()), abs(a.min()))
        if mx > 0:
            return (a / mx).tolist()
        return curve

    def get_bar_params(self, bar_idx, total_bars):
        """Devuelve los parámetros para el compás bar_idx."""
        def sample(curve, idx, total):
            if not curve:
                return 0.5
            frac = idx / max(total - 1, 1)
            pos  = frac * (len(curve) - 1)
            lo   = int(pos)
            hi   = min(lo + 1, len(curve) - 1)
            t    = pos - lo
            return curve[lo] * (1 - t) + curve[hi] * t

        tension   = sample(self.tension,   bar_idx, total_bars)
        arousal   = sample(self.arousal,   bar_idx, total_bars)
        valence   = sample(self.valence,   bar_idx, total_bars)
        stability = sample(self.stability, bar_idx, total_bars)
        activity  = sample(self.activity,  bar_idx, total_bars)

        # Velocidad: crece con arousal y tension
        velocity = int(np.clip(60 + arousal * 25 + tension * 15, 35, 110))

        # Registro: arousal alto → más agudo
        register_offset = int(arousal * 7)

        # Densidad rítmica: actividad y tension la elevan
        density_mult = np.clip(0.6 + activity * 0.8 + tension * 0.4, 0.4, 2.2)

        # Estilo de acompañamiento
        if tension > 0.7:
            acc_style = 'block'
        elif arousal > 0.3 and activity > 0.5:
            acc_style = 'arpeggio'
        else:
            acc_style = 'alberti'

        # Preferencia armónica: tensión alta → acordes de función D/PD
        harmony_tension = float(tension)

        return {
            'velocity':        velocity,
            'register_offset': register_offset,
            'density_mult':    float(density_mult),
            'acc_style':       acc_style,
            'harmony_tension': harmony_tension,
            'stability':       stability,
        }


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN: MÓDULO DE ESTRUCTURA FORMAL
# ════════════════════════════════════════════════════════════════

class FormGenerator:
    """
    Usa la estructura formal de la pieza fuente para:
    1. Dividir los N compases de salida en secciones (A, B, C…)
    2. Asignar diferente material melódico/armónico a cada sección
    3. Generar repeticiones y contrastes según la forma original
    """

    def __init__(self, form_string, section_map, phrase_lengths,
                 cadence_positions, n_bars_out):
        self.form_string = form_string or "AABA"
        self.section_map = section_map or []
        self.phrase_lengths_src = phrase_lengths or [4]
        self.cadence_positions = cadence_positions or []
        self.n_bars = n_bars_out

        # Escalar la forma original al número de compases de salida
        self._build_output_map()

    def _build_output_map(self):
        """
        Genera self.bar_section[bar_idx] → letra de sección (A, B, C…)
        escalando la forma original a n_bars compases.
        """
        form = self.form_string
        n = self.n_bars
        self.bar_section = []

        # Escalar: repetir la forma hasta llenar n_bars
        # (o truncar si es más larga)
        expanded = []
        src_len  = len(self.phrase_lengths_src)
        # Calcular cuántos compases por sección en la fuente
        src_total = sum(self.phrase_lengths_src)
        if src_total == 0:
            src_total = len(form) * 4

        # Expandir frases a compases
        bar_labels_src = []
        for i, letter in enumerate(form):
            phrase_len = self.phrase_lengths_src[i % len(self.phrase_lengths_src)]
            bar_labels_src.extend([letter] * phrase_len)

        # Escalar al número de compases deseado
        scale = n / max(len(bar_labels_src), 1)
        for bar in range(n):
            src_idx = min(int(bar / scale), len(bar_labels_src) - 1)
            self.bar_section.append(bar_labels_src[src_idx])

        # Detectar posiciones de cambio de sección → son posibles cadencias
        self.cadence_bars_out = set()
        prev = None
        for bi, lbl in enumerate(self.bar_section):
            if lbl != prev and prev is not None:
                self.cadence_bars_out.add(bi)
            prev = lbl
        # Último compás siempre tiene cadencia
        self.cadence_bars_out.add(n - 1)

    def is_cadence_bar(self, bar_idx):
        return bar_idx in self.cadence_bars_out

    def section_of(self, bar_idx):
        if 0 <= bar_idx < len(self.bar_section):
            return self.bar_section[bar_idx]
        return 'A'

    def section_contour_modifier(self, bar_idx):
        """
        Devuelve un modificador de contorno según la sección.
        A: material original, B: invertido, C: transposición, etc.
        """
        sec = self.section_of(bar_idx)
        return {
            'A': {'invert': False, 'transpose': 0,  'compress': 1.0},
            'B': {'invert': True,  'transpose': 5,  'compress': 0.7},
            'C': {'invert': False, 'transpose': -3, 'compress': 1.3},
            'D': {'invert': True,  'transpose': 2,  'compress': 0.8},
            'Z': {'invert': False, 'transpose': 0,  'compress': 0.5},
        }.get(sec, {'invert': False, 'transpose': 0, 'compress': 1.0})


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN DE MELODÍA (v2: con control emocional y formal)
# ════════════════════════════════════════════════════════════════

def generate_melody_v2(
    harmony_prog, key_obj, rhythm_pattern,
    pitch_contour, pitch_register,
    n_bars, motif_intervals,
    emotional_ctrl,  # EmotionalController
    form_gen,        # FormGenerator
    beats_per_bar=4
):
    """
    Genera melodía integrando:
    - Contorno melódico de la fuente (con transformaciones por sección)
    - Ritmo de la fuente
    - Armonía de la fuente (con notas del acorde en tiempos fuertes)
    - Parámetros emocionales compás a compás (registro, densidad, velocidad)
    - Estructura formal (transformaciones A/B/C, cadencias)
    """
    result = []  # [(offset_global, midi_pitch, duration_ql, velocity)]

    # Construir timeline de armonía
    h_timeline = []
    beat = 0.0
    for fig, dur in harmony_prog:
        h_timeline.append((beat, beat + dur, fig))
        beat += dur
    total_h = beat if beat > 0 else n_bars * beats_per_bar

    def chord_at(beat):
        bm = beat % total_h
        for s, e, fig in h_timeline:
            if s <= bm < e:
                return fig
        return 'I'

    def chord_tones_at(beat, octave=5):
        return roman_to_chord_pitches(chord_at(beat), key_obj, octave)

    # Contorno extendido
    contour = list(pitch_contour) if pitch_contour else [2, -1, 3, -2, 1, -3]
    if not contour:
        contour = [2, -1, 3, -2]
    contour_idx = 0

    current_pitch = max(52, min(79, pitch_register))
    current_pitch = snap_to_scale(current_pitch, key_obj)

    for bar_idx in range(n_bars):
        global_beat = bar_idx * beats_per_bar

        # Parámetros emocionales de este compás
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)

        # Modificador formal
        fm = form_gen.section_contour_modifier(bar_idx)

        # Patrón rítmico (con densidad escalada por emoción)
        bar_rhythm = rhythm_pattern[bar_idx % len(rhythm_pattern)]
        if not bar_rhythm:
            bar_rhythm = [(i * beats_per_bar / 4, beats_per_bar / 4)
                          for i in range(4)]

        # Filtrar/duplicar notas según density_mult
        density_mult = ep['density_mult']
        if density_mult < 0.7 and len(bar_rhythm) > 2:
            # Menos notas: quitar algunas al azar
            n_keep = max(1, int(len(bar_rhythm) * density_mult))
            bar_rhythm = sorted(random.sample(bar_rhythm, n_keep), key=lambda x: x[0])

        # Registro objetivo para este compás
        target_register = current_pitch + ep['register_offset']
        target_register = max(48, min(84, target_register))

        # ¿Es compás de cadencia?
        is_cadence = form_gen.is_cadence_bar(bar_idx)
        if is_cadence:
            # En cadencia, la nota final debe ser del acorde
            pass

        for note_idx, (local_offset, note_dur) in enumerate(bar_rhythm):
            beat = global_beat + local_offset

            # Contorno con transformaciones formales
            step = contour[contour_idx % len(contour)]
            contour_idx += 1

            # Aplicar transformaciones de sección
            if fm['invert']:
                step = -step
            step = int(step * fm['compress'])

            candidate = current_pitch + step + fm['transpose']

            # Corrección de registro
            if candidate > target_register + 12:
                candidate = current_pitch - abs(step)
            elif candidate < target_register - 12:
                candidate = current_pitch + abs(step)

            # Mantener en rango vocal cómodo
            while candidate > 84: candidate -= 12
            while candidate < 45: candidate += 12

            # Snap a escala
            candidate = snap_to_scale(candidate, key_obj)

            # En tiempos fuertes, preferir notas del acorde
            beat_in_bar = local_offset % beats_per_bar
            is_strong = beat_in_bar in (0.0, 2.0) or (beats_per_bar == 3 and beat_in_bar == 0)
            ct = chord_tones_at(beat)
            if is_strong and ct:
                nearest_ct = min(ct, key=lambda p: abs(p - candidate))
                if abs(nearest_ct - candidate) <= 4:
                    candidate = nearest_ct

            # En último tiempo de compás de cadencia → nota de tónica o quinta
            if is_cadence and note_idx == len(bar_rhythm) - 1:
                chord_fig = chord_at(beat)
                if chord_fig.startswith('I') or chord_fig.startswith('V'):
                    ct_final = chord_tones_at(beat)
                    if ct_final:
                        # Acercarse a la nota del acorde más próxima
                        candidate = min(ct_final, key=lambda p: abs(p - candidate))

            # Drifting hacia registro objetivo cada 4 notas
            if contour_idx % 4 == 0:
                drift = int((target_register - current_pitch) * 0.15)
                candidate = snap_to_scale(candidate + drift, key_obj)
                candidate = max(45, min(84, candidate))

            # Velocidad con micro-variación expresiva
            vel = ep['velocity'] + random.randint(-5, 5)
            # Acentuar tiempos fuertes
            if is_strong:
                vel = min(110, vel + 8)
            vel = max(35, min(110, vel))

            result.append((beat, candidate, note_dur * 0.88, vel))
            current_pitch = candidate

    return result


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN DE ACOMPAÑAMIENTO (v2: emocional)
# ════════════════════════════════════════════════════════════════

def generate_accompaniment_v2(
    harmony_prog, key_obj, n_bars,
    emotional_ctrl, form_gen,
    beats_per_bar=4
):
    """
    Acompañamiento armónico con:
    - Estilo variable por compás (block/arpeggio/alberti) según emoción
    - Dinámica modulada por la curva de arousal
    - Octava baja en secciones de reposo, media en tensión
    """
    result = []
    total_beats = n_bars * beats_per_bar

    # Expandir progresión a los compases necesarios
    h_exp = []
    total_prog = sum(d for _, d in harmony_prog) if harmony_prog else 1
    bt = 0.0
    while bt < total_beats:
        for fig, dur in harmony_prog:
            h_exp.append((bt, min(dur, total_beats - bt), fig))
            bt += dur
            if bt >= total_beats:
                break

    for chord_start, chord_dur, fig in h_exp:
        bar_idx = int(chord_start / beats_per_bar)
        ep = emotional_ctrl.get_bar_params(bar_idx, n_bars)
        acc_style = ep['acc_style']

        # Octava según estabilidad
        octave = 3 if ep['stability'] > 0.6 else 3
        pitches = roman_to_chord_pitches(fig, key_obj, octave)
        if not pitches:
            pitches = [48, 52, 55]

        vel_base = max(35, ep['velocity'] - 15)

        if acc_style == 'block':
            for p in pitches:
                result.append((chord_start, p, min(chord_dur, beats_per_bar), vel_base))

        elif acc_style == 'arpeggio':
            n_reps = max(1, int(chord_dur))
            sub_dur = chord_dur / (len(pitches) * n_reps)
            t = chord_start
            for _ in range(n_reps):
                for p in sorted(pitches):
                    result.append((t, p, sub_dur * 0.85, vel_base + random.randint(0, 8)))
                    t += sub_dur

        elif acc_style == 'alberti':
            if len(pitches) >= 3:
                pat = [pitches[0], pitches[-1], pitches[1], pitches[-1]]
            else:
                pat = pitches * 2
            sub_dur = chord_dur / len(pat)
            t = chord_start
            for i, p in enumerate(pat):
                v = vel_base + (4 if i == 0 else 0)  # Acentuar bajo
                result.append((t, p, sub_dur * 0.9, v))
                t += sub_dur

    return result


# ════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL SCORE FINAL
# ════════════════════════════════════════════════════════════════

def build_score(melody_notes, acc_notes, target_key,
                tempo_bpm, time_sig, n_bars, form_gen=None):
    sc = stream.Score()
    bpb, bu = time_sig

    def make_part(name, notes_list, is_melody=True):
        part = stream.Part()
        part.id = name
        part.insert(0, instrument.Piano())
        if is_melody:
            part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
        part.insert(0, meter.TimeSignature(f'{bpb}/{bu}'))
        part.insert(0, target_key)

        for bar_idx in range(n_bars):
            bs = bar_idx * bpb
            be = bs + bpb
            m = stream.Measure(number=bar_idx + 1)

            # Añadir anotación de sección si cambia
            if form_gen:
                sec = form_gen.section_of(bar_idx)
                if bar_idx == 0 or form_gen.section_of(bar_idx - 1) != sec:
                    from music21 import expressions
                    txt = expressions.TextExpression(f'[{sec}]')
                    txt.style.fontStyle = 'italic'
                    m.insert(0, txt)

            bar_notes = [(o, p2, d, v) for o, p2, d, v in notes_list
                         if bs <= o < be]

            if is_melody:
                if not bar_notes:
                    m.append(note.Rest(quarterLength=bpb))
                else:
                    for offset, mp, dur, vel in bar_notes:
                        lo = offset - bs
                        n2 = note.Note()
                        n2.pitch.midi = max(21, min(108, int(mp)))
                        n2.quarterLength = max(0.25, dur)
                        n2.volume.velocity = int(vel)
                        m.insert(lo, n2)
                    m.makeRests(fillGaps=True, inPlace=True)
            else:
                if not bar_notes:
                    m.append(note.Rest(quarterLength=bpb))
                else:
                    by_off = defaultdict(list)
                    for offset, mp, dur, vel in bar_notes:
                        lo = round((offset - bs) * 4) / 4
                        if lo < bpb:
                            by_off[lo].append((mp, dur, vel))

                    for lo in sorted(by_off.keys()):
                        pts = by_off[lo]
                        if len(pts) == 1:
                            mp, dur, vel = pts[0]
                            n2 = note.Note()
                            n2.pitch.midi = max(21, min(108, int(mp)))
                            n2.quarterLength = max(0.25, dur)
                            n2.volume.velocity = int(vel)
                            m.insert(lo, n2)
                        else:
                            min_dur = min(d for _, d, _ in pts)
                            avg_vel = int(np.mean([v for _, _, v in pts]))
                            ch_ps = [pitch.Pitch(max(21, min(108, int(mp)))) for mp, _, _ in pts]
                            ch = chord.Chord(ch_ps)
                            ch.quarterLength = max(0.25, min_dur)
                            ch.volume.velocity = avg_vel
                            m.insert(lo, ch)

                    m.makeRests(fillGaps=True, inPlace=True)

            part.append(m)
        return part

    sc.append(make_part('Melody',        melody_notes, is_melody=True))
    sc.append(make_part('Accompaniment', acc_notes,    is_melody=False))
    return sc


# ════════════════════════════════════════════════════════════════
#  MODOS DE MEZCLA (v2)
# ════════════════════════════════════════════════════════════════

def _prepare_controllers(dnas, emotion_src_idx, form_src_idx, n_bars):
    """Crea EmotionalController y FormGenerator desde los DNAs fuente."""
    ed = dnas[min(emotion_src_idx, len(dnas) - 1)]
    fd = dnas[min(form_src_idx,   len(dnas) - 1)]

    ec = EmotionalController(
        tension_curve   = ed.tension_curve   or [0.5],
        arousal_curve   = ed.arousal_curve   or [0.0],
        valence_curve   = ed.valence_curve   or [0.0],
        stability_curve = ed.stability_curve or [0.7],
        activity_curve  = ed.activity_curve  or [0.5],
        emotional_arc_label = ed.emotional_arc_label
    )

    fg = FormGenerator(
        form_string      = fd.form_string,
        section_map      = fd.section_map,
        phrase_lengths   = fd.phrase_lengths,
        cadence_positions= fd.cadence_positions,
        n_bars_out       = n_bars
    )
    return ec, fg


def _run_generation(harmony_prog, key_obj, rhythm_pattern,
                    pitch_contour, pitch_register, motif_intervals,
                    n_bars, tempo_bpm, ec, fg, time_sig):
    bpb = time_sig[0]
    mel = generate_melody_v2(
        harmony_prog   = harmony_prog,
        key_obj        = key_obj,
        rhythm_pattern = rhythm_pattern,
        pitch_contour  = pitch_contour,
        pitch_register = pitch_register,
        n_bars         = n_bars,
        motif_intervals= motif_intervals,
        emotional_ctrl = ec,
        form_gen       = fg,
        beats_per_bar  = bpb
    )
    acc = generate_accompaniment_v2(
        harmony_prog = harmony_prog,
        key_obj      = key_obj,
        n_bars       = n_bars,
        emotional_ctrl = ec,
        form_gen       = fg,
        beats_per_bar  = bpb
    )
    return mel, acc


def mix_rhythm_melody(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig):
    print("\n[Modo] rhythm_melody: ritmo de #1, melodía de #2, armonía de #1-3")
    d_rhythm   = dnas[0]
    d_melody   = dnas[1] if len(dnas) > 1 else dnas[0]
    d_harmony  = dnas[2] if len(dnas) > 2 else dnas[0]

    seq = transpose_to_key(d_melody.pitch_sequence, d_melody.key_obj, target_key)
    reg = max(52, min(79, int(np.mean(seq)))) if seq else 64

    return _run_generation(
        harmony_prog   = d_harmony.harmony_prog,
        key_obj        = target_key,
        rhythm_pattern = d_rhythm.rhythm_pattern,
        pitch_contour  = d_melody.pitch_contour,
        pitch_register = reg,
        motif_intervals= d_melody.motif_intervals,
        n_bars=n_bars, tempo_bpm=tempo_bpm, ec=ec, fg=fg, time_sig=time_sig
    )


def mix_harmony_melody(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig):
    print("\n[Modo] harmony_melody: armonía de #1, melodía de #2, ritmo de #3")
    d_harmony = dnas[0]
    d_melody  = dnas[1] if len(dnas) > 1 else dnas[0]
    d_rhythm  = dnas[2] if len(dnas) > 2 else dnas[0]

    seq = transpose_to_key(d_melody.pitch_sequence, d_melody.key_obj, target_key)
    reg = max(52, min(79, int(np.mean(seq)))) if seq else 64

    return _run_generation(
        harmony_prog   = d_harmony.harmony_prog,
        key_obj        = target_key,
        rhythm_pattern = d_rhythm.rhythm_pattern,
        pitch_contour  = d_melody.pitch_contour,
        pitch_register = reg,
        motif_intervals= d_melody.motif_intervals,
        n_bars=n_bars, tempo_bpm=tempo_bpm, ec=ec, fg=fg, time_sig=time_sig
    )


def mix_full_blend(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig):
    print("\n[Modo] full_blend: mezcla ponderada de todos los MIDIs")

    # Contorno: promedio
    contours = [d.pitch_contour for d in dnas if d.pitch_contour]
    max_len = max(len(c) for c in contours) if contours else 8
    blended = [int(np.mean([c[i % len(c)] for c in contours]))
               for i in range(max_len)]

    reg = int(np.mean([d.pitch_register for d in dnas]))
    reg = max(52, min(79, reg))

    def rhythm_complexity(d):
        return np.mean([len(b) for b in d.rhythm_pattern]) if d.rhythm_pattern else 0

    d_rhythm  = max(dnas, key=rhythm_complexity)
    d_harmony = max(dnas, key=lambda d: len(d.harmony_prog))

    return _run_generation(
        harmony_prog   = d_harmony.harmony_prog,
        key_obj        = target_key,
        rhythm_pattern = d_rhythm.rhythm_pattern,
        pitch_contour  = blended,
        pitch_register = reg,
        motif_intervals= blended[:8],
        n_bars=n_bars, tempo_bpm=tempo_bpm, ec=ec, fg=fg, time_sig=time_sig
    )


def mix_custom(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig, sources):
    print(f"\n[Modo] custom: fuentes={sources}")

    def get_d(k):
        return dnas[min(sources.get(k, 0), len(dnas) - 1)]

    d_h = get_d('harmony')
    d_m = get_d('melody')
    d_r = get_d('rhythm')

    seq = transpose_to_key(d_m.pitch_sequence, d_m.key_obj, target_key)
    reg = max(52, min(79, int(np.mean(seq)))) if seq else 64

    return _run_generation(
        harmony_prog   = d_h.harmony_prog,
        key_obj        = target_key,
        rhythm_pattern = d_r.rhythm_pattern,
        pitch_contour  = d_m.pitch_contour,
        pitch_register = reg,
        motif_intervals= d_m.motif_intervals,
        n_bars=n_bars, tempo_bpm=tempo_bpm, ec=ec, fg=fg, time_sig=time_sig
    )


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

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
    except:
        return None

def print_arc_report(dnas, emotion_src_idx, form_src_idx):
    ed = dnas[emotion_src_idx]
    fd = dnas[form_src_idx]
    print("\n" + "─" * 55)
    print("  INFORME DE ADN MUSICAL")
    print("─" * 55)
    for i, d in enumerate(dnas):
        markers = []
        if i == emotion_src_idx: markers.append("EMOCIÓN")
        if i == form_src_idx:    markers.append("FORMA")
        tag = f" [{', '.join(markers)}]" if markers else ""
        print(f"\n  #{i+1}{tag}: {os.path.basename(d.path)}")
        print(f"    Tonalidad : {d.key_obj.tonic.name} {d.key_obj.mode} | {d.tempo_bpm:.0f} BPM")
        print(f"    Forma     : {d.form_string}  ({d.n_unique_sections} secciones)")
        print(f"    Arco emoc.: {d.emotional_arc_label}")
        print(f"    Clímax    : {d.climax_position:.0%} de la pieza")
        print(f"    Resolución: {d.resolution_index:.2f}  "
              f"(< 0.4 = resuelve bien)")
        print(f"    Entropía  : mel={d.entropy_melodic:.2f}b  "
              f"rit={d.entropy_rhythmic:.2f}b")
        # Mini gráfica de tensión
        tc = d.tension_curve
        if tc:
            width = 30
            buckets = [tc[int(j * len(tc) / width)] for j in range(width)]
            bar_str = "".join(
                "█" if v > 0.75 else "▓" if v > 0.5 else "░" if v > 0.25 else " "
                for v in buckets
            )
            print(f"    Tensión   : |{bar_str}|")
        # Mini gráfica de arousal
        ac = d.arousal_curve
        if ac:
            width = 30
            buckets_a = [(ac[int(j * len(ac) / width)] + 1) / 2 for j in range(width)]
            bar_str_a = "".join(
                "█" if v > 0.75 else "▓" if v > 0.5 else "░" if v > 0.25 else " "
                for v in buckets_a
            )
            print(f"    Arousal   : |{bar_str_a}|")

def main():
    parser = argparse.ArgumentParser(
        description='MIDI DNA Mixer v2 – ADN extendido, forma y arco emocional'
    )
    parser.add_argument('midis', nargs='+', help='MIDIs de entrada')
    parser.add_argument('--mode', default='rhythm_melody',
                        choices=['rhythm_melody','harmony_melody','full_blend','custom'])
    parser.add_argument('--emotion_src', type=int, default=0,
                        help='Índice del MIDI que dona el arco emocional (default: 0)')
    parser.add_argument('--form_src', type=int, default=0,
                        help='Índice del MIDI que dona la estructura formal (default: 0)')
    parser.add_argument('--sources', default='rhythm=0,melody=1,harmony=0',
                        help='Para modo custom: rhythm=N,melody=N,harmony=N')
    parser.add_argument('--key',    default=None)
    parser.add_argument('--bars',   type=int,   default=16)
    parser.add_argument('--tempo',  type=float, default=None)
    parser.add_argument('--output', default='output_dna_mix.mid')
    args = parser.parse_args()

    print("═" * 60)
    print("  MIDI DNA MIXER  v2.0")
    print("  ADN extendido · Estructura formal · Arco emocional")
    print("═" * 60)

    # Validar ficheros
    midi_files = [p for p in args.midis if os.path.exists(p)]
    if not midi_files:
        print("ERROR: ningún fichero MIDI encontrado.")
        sys.exit(1)

    # ── Extraer ADN
    print(f"\n[1/4] Extrayendo ADN de {len(midi_files)} MIDI(s)…")
    dnas = []
    for path in midi_files:
        print(f"\n  ▶ {os.path.basename(path)}")
        dna = MusicalDNA(path)
        if dna.extract():
            dnas.append(dna)
    if not dnas:
        print("ERROR: no se pudo extraer ADN.")
        sys.exit(1)

    # ── Informe
    esi = min(args.emotion_src, len(dnas) - 1)
    fsi = min(args.form_src,    len(dnas) - 1)
    print_arc_report(dnas, esi, fsi)

    # ── Tonalidad y tempo
    target_key = parse_key_arg(args.key) if args.key else dnas[0].key_obj
    tempo_bpm  = args.tempo or dnas[0].tempo_bpm
    time_sig   = dnas[0].time_sig
    n_bars     = args.bars

    print(f"\n[2/4] Preparando controladores emocional y formal…")
    ec, fg = _prepare_controllers(dnas, esi, fsi, n_bars)
    print(f"    Arco emocional heredado : {dnas[esi].emotional_arc_label}")
    print(f"    Forma heredada          : {dnas[fsi].form_string}")
    print(f"    Secciones en {n_bars} compases: {fg.form_string}")

    print(f"\n[3/4] Generando ({args.mode}, {n_bars} compases, "
          f"{target_key.tonic.name} {target_key.mode}, {tempo_bpm:.0f} BPM)…")

    if args.mode == 'rhythm_melody':
        mel, acc = mix_rhythm_melody(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig)
    elif args.mode == 'harmony_melody':
        mel, acc = mix_harmony_melody(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig)
    elif args.mode == 'full_blend':
        mel, acc = mix_full_blend(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig)
    elif args.mode == 'custom':
        sources = parse_sources(args.sources)
        mel, acc = mix_custom(dnas, target_key, n_bars, tempo_bpm, ec, fg, time_sig, sources)

    print(f"    → Melodía:         {len(mel)} notas")
    print(f"    → Acompañamiento:  {len(acc)} eventos")

    print(f"\n[4/4] Construyendo MIDI…")
    sc = build_score(mel, acc, target_key, tempo_bpm, time_sig, n_bars, fg)
    sc.write('midi', fp=args.output)
    print(f"\n✓ Generado: {args.output}")

    # ── Resumen final
    print("\n" + "═" * 60)
    print("  RESUMEN FINAL")
    print("═" * 60)
    print(f"  Tonalidad : {target_key.tonic.name} {target_key.mode}")
    print(f"  Tempo     : {tempo_bpm:.0f} BPM | Compás: {time_sig[0]}/{time_sig[1]}")
    print(f"  Compases  : {n_bars}")
    print(f"  Forma     : {fg.form_string}")
    print(f"  Arco      : {dnas[esi].emotional_arc_label}")
    print(f"  Fichero   : {args.output}")
    print("═" * 60)


if __name__ == '__main__':
    main()
