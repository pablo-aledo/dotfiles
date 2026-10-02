#!/usr/bin/env python3
"""
fingering_v4.py — Generador automático de digitación pianística a partir de MIDI.

Novedad de v4 (modo opcional --chords):
  - Calcula los compases, detecta el acorde de cada compás (plantillas de
    clases de altura + nota del bajo) y asigna a cada nota su función (R,3,5,7…).
  - Agrupa los compases de cada mano en "clases de voicing" (mismo acorde y
    mismas notas / misma función transportada / misma forma melódica) y busca
    una digitación de consenso por clase.
  - Esa digitación de consenso entra en el optimizador como una penalización
    blanda controlada por --chord-weight (λ):
        λ = 0    → solución agnóstica (idéntica a v3)
        λ ~ 1    → una nota que rompe la plantilla cuesta como una nota media
        λ → ∞    → uniformidad forzada entre compases de la misma clase
  - Sin --chords el comportamiento es exactamente el de v3.

Mejoras de v3 respecto a v2:
  - Memoria de postura activada (relocation_alpha=0.3)
  - Separación inteligente de manos (un solo track) minimizando cruces
  - Tempo variable: cálculo de compás correcto con múltiples cambios de tempo
  - Optimización de acordes con módulo específico (span real de mano)
  - Puntuación de confianza por nota (marca notas con digitación alternativa válida con '(alt)')
  - Salida estructurada como lista de dicts (API usable por otros módulos)
  - Separación de voces dentro de un track (notas largas vs corcheas)
  - Digitación de voz sostenida independiente de la melodía
  - Exportación a MusicXML con marcas de digitación estándar (opcional)

Uso:
    python fingering_v4.py archivo.mid --chords --chord-weight 1.0 --chord-report
    python fingering_v4.py archivo.mid
    python fingering_v2.py archivo.mid --hand-size L
    python fingering_v2.py archivo.mid --measures 8
    python fingering_v2.py archivo.mid --right-only
    python fingering_v2.py archivo.mid --left-only
    python fingering_v2.py archivo.mid --xml salida.xml
    python fingering_v2.py archivo.mid --json salida.json
    python fingering_v3.py archivo.mid --confidence      # muestra puntuación de confianza
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Sequence

try:
    import pretty_midi
except ImportError:
    print("Falta pretty_midi. Instala con: pip install pretty_midi")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class INote:
    """Representación interna de una nota para el optimizador."""
    pitch: int    = 0
    octave: int   = 0
    name: str     = ""
    x: float      = 0.0      # posición física en el teclado (cm)
    time: float   = 0.0      # inicio en segundos
    duration: float = 0.0
    isBlack: bool = False
    isChord: bool = False
    chordID: int  = 0
    chordnr: int  = 0
    NinChord: int = 0
    noteID: int   = 0
    measure: int  = 0
    beat: float   = 0.0      # beat dentro del compás (0-based)
    hand: str     = ""        # "right" | "left"
    fingering: int = 0
    confidence: float = 1.0  # 0..1, baja cuando el coste óptimo ≈ subóptimo
    cost: float   = 0.0
    voice: str    = "melody"  # "melody" | "sustained"
    # ── v4: noción de acorde ────────────────────────────────────────────
    chord: str      = ""      # etiqueta del acorde del compás ("Fm", "Bbsus2"…)
    chord_role: str = ""      # función de la nota en ese acorde ("R","3","5"…)
    ref_finger: int = 0       # dedo de la plantilla de clase (0 = sin plantilla)
    ref_weight: float = 0.0   # peso del nivel de la plantilla (0..1)
    ref_tier: str   = ""      # "exact" | "roles" | "shape"


@dataclass
class FingeringResult:
    """Resultado estructurado exportable como JSON o MusicXML."""
    midi_file: str
    bpm: float
    time_signature: str
    measures: int
    notes: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Geometría del teclado
# ─────────────────────────────────────────────────────────────────────────────

_MIDI_NOTE_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_KEY_CM_PER_OCT   = 16.5
_KEY_CM_PER_WHITE = _KEY_CM_PER_OCT / 7.0
_SEMITONE_OFFSET  = [0.5,1.0,1.5,2.0,2.5,3.5,4.0,4.5,5.0,5.5,6.0,6.5]

# Distancias físicas entre teclas adyacentes (cm), usadas en optimización de acordes
_WHITE_WIDTH  = _KEY_CM_PER_WHITE          # ≈2.36 cm
_BLACK_WIDTH  = _WHITE_WIDTH * 0.6         # ≈1.41 cm

# Máximas distancias cómodas entre pares de dedos (mano M, sin escalar)
_MAX_FINGER_SPAN: dict[tuple[int,int], float] = {
    (1,2): 12.0, (1,3): 14.0, (1,4): 16.0, (1,5): 19.0,
    (2,3):  6.0, (2,4):  8.0, (2,5): 11.0,
    (3,4):  5.0, (3,5):  8.0,
    (4,5):  5.0,
}


def keypos_midi(pitch: int) -> float:
    octave = pitch // 12
    semi   = pitch % 12
    return _KEY_CM_PER_OCT * octave + _SEMITONE_OFFSET[semi] * _KEY_CM_PER_WHITE


def pitch_name(pitch: int) -> str:
    octave = (pitch // 12) - 1
    return f"{_MIDI_NOTE_NAMES[pitch % 12]}{octave}"


# ─────────────────────────────────────────────────────────────────────────────
# Tempo variable: tiempo → compás/beat
# ─────────────────────────────────────────────────────────────────────────────

class TempoMap:
    """Convierte tiempo en segundos a (compás, beat) respetando cambios de tempo."""

    def __init__(self, pm: pretty_midi.PrettyMIDI, beats_per_measure: int = 4) -> None:
        times, tempos = pm.get_tempo_changes()
        self.segments: list[tuple[float, float, float]] = []  # (t_start, t_end, bpm)
        self.bpm0 = float(tempos[0]) if len(tempos) > 0 else 120.0
        self.beats_per_measure = beats_per_measure

        for i, (t, bpm) in enumerate(zip(times, tempos)):
            t_end = times[i+1] if i+1 < len(times) else float("inf")
            self.segments.append((float(t), t_end, float(bpm)))

        if not self.segments:
            self.segments = [(0.0, float("inf"), 120.0)]

    def spb_at(self, t: float) -> float:
        """Segundos por beat en el instante t."""
        for t0, t1, bpm in self.segments:
            if t0 <= t < t1:
                return 60.0 / bpm
        return 60.0 / self.bpm0

    def beats_elapsed(self, t: float) -> float:
        """Número de beats transcurridos desde t=0 hasta t."""
        beats = 0.0
        prev_t = 0.0
        for t0, t1, bpm in self.segments:
            seg_start = max(t0, prev_t)
            seg_end   = min(t1, t)
            if seg_end <= seg_start:
                continue
            beats += (seg_end - seg_start) / (60.0 / bpm)
            if t <= t1:
                break
            prev_t = t1
        return beats

    def measure_and_beat(self, t: float) -> tuple[int, float]:
        """(compás 1-based, beat 0-based dentro del compás)."""
        total_beats = self.beats_elapsed(t)
        # `beats_elapsed` acumula tiempo en segundos convertido a beats, lo
        # que a veces produce un valor ligerísimamente por debajo del límite
        # exacto de compás (p. ej. 15.999999998 en vez de 16.0) por errores
        # de redondeo de punto flotante. Sin corrección, eso hacía que una
        # nota que en realidad empieza en el compás n se asociara al
        # compás n-1. Con un epsilon muy pequeño "empujamos" ese caso al
        # lado correcto del límite sin afectar a compases intermedios.
        eps = 1e-6
        measure      = int((total_beats + eps) / self.beats_per_measure) + 1
        beat_in_meas = (total_beats + eps) % self.beats_per_measure
        return measure, beat_in_meas


# ─────────────────────────────────────────────────────────────────────────────
# Separación inteligente de manos (track único)
# ─────────────────────────────────────────────────────────────────────────────

def _split_single_track_smart(notes: list) -> tuple[list, list]:
    """
    Separa notas de un track único en mano derecha e izquierda.

    Algoritmo:
      1. Ventana deslizante de 1 segundo: calcula el pitch mediano local.
      2. Asigna cada nota a la mano cuya mediana local sea más cercana.
      3. Post-proceso: ajusta notas aisladas que forman cruces obvios.

    Mejor que la mediana global porque adapta el punto de corte cuando
    la melodía sube o baja a lo largo de la pieza.
    """
    if not notes:
        return [], []

    notes = sorted(notes, key=lambda n: n.start)
    window_s = 1.0
    assignments: list[str] = []

    for i, n in enumerate(notes):
        t0 = n.start - window_s / 2
        t1 = n.start + window_s / 2
        local = [m.pitch for m in notes if t0 <= m.start <= t1]
        local_median = sorted(local)[len(local) // 2]
        assignments.append("right" if n.pitch >= local_median else "left")

    # Post-proceso: nota aislada rodeada del otro lado → reasignar
    for i in range(1, len(assignments) - 1):
        if assignments[i-1] == assignments[i+1] != assignments[i]:
            assignments[i] = assignments[i-1]

    rh = [n for n, a in zip(notes, assignments) if a == "right"]
    lh = [n for n, a in zip(notes, assignments) if a == "left"]
    return rh, lh



# ─────────────────────────────────────────────────────────────────────────────
# Separación de voces dentro de un track (melodía vs notas sostenidas)
# ─────────────────────────────────────────────────────────────────────────────

def _separate_voices(pm_notes: list, spb: float,
                     sustained_threshold_beats: float = 1.0
                     ) -> tuple[list, list]:
    """
    Separa un track con polifonía interna en dos voces:
      - melody:    notas cortas (< threshold beats) → digitación secuencial normal
      - sustained: notas largas (≥ threshold beats) → digitación independiente

    Algoritmo:
      1. Clasifica cada nota por duración relativa al beat.
      2. Para cada nota sostenida, busca si hay una nota de melodía simultánea
         que empiece en el mismo instante. Si la hay, la sostenida es voz tenor/bajo.
      3. Las notas sostenidas sin melodía simultánea se tratan como melodía
         (pueden ser notas largas aisladas, no voces independientes).

    Returns (melody_notes, sustained_notes).
    """
    if not pm_notes:
        return pm_notes, []

    pm_notes = sorted(pm_notes, key=lambda n: n.start)
    melody, sustained = [], []

    for n in pm_notes:
        dur_beats = (n.end - n.start) / spb
        if dur_beats >= sustained_threshold_beats:
            # Verificar si hay notas cortas simultáneas (mismo onset ±10ms)
            has_simultaneous_melody = any(
                abs(m.start - n.start) < 0.010
                and (m.end - m.start) / spb < sustained_threshold_beats
                for m in pm_notes
                if m is not n
            )
            if has_simultaneous_melody:
                sustained.append(n)
            else:
                melody.append(n)
        else:
            melody.append(n)

    return melody, sustained

# ─────────────────────────────────────────────────────────────────────────────
# Lectura de MIDI → secuencias de INote
# ─────────────────────────────────────────────────────────────────────────────

_CHORD_STAGGER_S = 0.02   # stagger más pequeño en v2 (menos artificial)


def _build_note_seq(pm_notes: list, tmap: TempoMap, hand: str) -> list[INote]:
    """Convierte una lista de pretty_midi.Note en INote[], agrupando acordes."""
    pm_notes = sorted(pm_notes, key=lambda n: n.start)
    noteseq: list[INote] = []
    chord_id = note_id = 0
    i = 0

    while i < len(pm_notes):
        onset = pm_notes[i].start
        j = i + 1
        # Agrupa notas con el mismo onset (tolerancia 5ms)
        while j < len(pm_notes) and abs(pm_notes[j].start - onset) < 0.005:
            j += 1

        group = [n for n in pm_notes[i:j] if (n.end - n.start) > 0]
        if not group:
            i = j
            continue

        measure, beat = tmap.measure_and_beat(onset)

        if len(group) == 1:
            n = group[0]
            an = INote()
            an.noteID   = note_id; note_id += 1
            an.pitch    = n.pitch
            an.octave   = n.pitch // 12
            an.name     = pitch_name(n.pitch)
            an.x        = keypos_midi(n.pitch)
            an.time     = n.start
            an.duration = n.end - n.start
            an.measure  = measure
            an.beat     = round(beat, 3)
            an.hand     = hand
            an.isBlack  = (n.pitch % 12) in {1,3,6,8,10}
            an.voice    = "melody"
            noteseq.append(an)
        else:
            # Ordenar el grupo por pitch (grave→agudo) para consistencia
            group_sorted = sorted(group, key=lambda n: n.pitch)
            for k, cn in enumerate(group_sorted):
                an = INote()
                an.chordID  = chord_id
                an.noteID   = note_id; note_id += 1
                an.isChord  = True
                an.pitch    = cn.pitch
                an.chordnr  = k
                an.NinChord = len(group_sorted)
                an.octave   = cn.pitch // 12
                an.name     = pitch_name(cn.pitch)
                an.x        = keypos_midi(cn.pitch)
                an.time     = onset + _CHORD_STAGGER_S * k
                an.duration = (cn.end - cn.start)
                an.measure  = measure
                an.beat     = round(beat, 3)
                an.hand     = hand
                an.isBlack  = (cn.pitch % 12) in {1,3,6,8,10}
                an.voice    = "melody"
                noteseq.append(an)
            chord_id += 1
        i = j

    return noteseq



def _build_sustained_seq(pm_notes: list, tmap: TempoMap, hand: str) -> list[INote]:
    """Construye secuencia de INote para notas sostenidas (voz independiente)."""
    noteseq = []
    for note_id, n in enumerate(sorted(pm_notes, key=lambda n: n.start)):
        measure, beat = tmap.measure_and_beat(n.start)
        an = INote()
        an.noteID   = note_id
        an.pitch    = n.pitch
        an.octave   = n.pitch // 12
        an.name     = pitch_name(n.pitch)
        an.x        = keypos_midi(n.pitch)
        an.time     = n.start
        an.duration = n.end - n.start
        an.measure  = measure
        an.beat     = round(beat, 3)
        an.hand     = hand
        an.isBlack  = (n.pitch % 12) in {1,3,6,8,10}
        an.voice    = "sustained"
        noteseq.append(an)
    return noteseq


def load_midi(path: str, right_track: int = 0, left_track: int = 1,
              auto_split: bool = False,
              voice_split: bool = True,
              sustained_threshold: float = 1.0,
              ) -> tuple[pretty_midi.PrettyMIDI,
                         list[INote], list[INote],
                         list[INote], list[INote],
                         TempoMap]:
    """
    Carga un fichero MIDI y devuelve
    (pm, rh_melody, lh_melody, rh_sustained, lh_sustained, tmap).

    voice_split=True (defecto): separa notas largas en voz sostenida independiente.
    sustained_threshold: duración mínima en beats para considerar nota sostenida (defecto: 1 beat).
    """
    pm = pretty_midi.PrettyMIDI(path)
    instruments = [ins for ins in pm.instruments if not ins.is_drum]

    # Firma de tiempo
    ts = pm.time_signature_changes
    bpm_num = ts[0].numerator if ts else 4
    tmap = TempoMap(pm, beats_per_measure=bpm_num)
    spb  = 60.0 / tmap.bpm0

    if auto_split or len(instruments) == 1:
        if instruments:
            all_notes = sorted(instruments[0].notes, key=lambda n: n.start)
            rh_raw, lh_raw = _split_single_track_smart(all_notes)
        else:
            rh_raw = lh_raw = []
    else:
        rh_raw = list(instruments[right_track].notes) if right_track < len(instruments) else []
        lh_raw = list(instruments[left_track].notes)  if left_track  < len(instruments) else []

    # Separación de voces
    if voice_split:
        rh_mel_raw, rh_sus_raw = _separate_voices(rh_raw, spb, sustained_threshold)
        lh_mel_raw, lh_sus_raw = _separate_voices(lh_raw, spb, sustained_threshold)
    else:
        rh_mel_raw, rh_sus_raw = rh_raw, []
        lh_mel_raw, lh_sus_raw = lh_raw, []

    rh_seq = _build_note_seq(rh_mel_raw, tmap, "right") if rh_mel_raw else []
    lh_seq = _build_note_seq(lh_mel_raw, tmap, "left")  if lh_mel_raw else []
    rh_sus = _build_sustained_seq(rh_sus_raw, tmap, "right") if rh_sus_raw else []
    lh_sus = _build_sustained_seq(lh_sus_raw, tmap, "left")  if lh_sus_raw else []

    return pm, rh_seq, lh_seq, rh_sus, lh_sus, tmap


# ─────────────────────────────────────────────────────────────────────────────
# Optimización específica de acordes
# ─────────────────────────────────────────────────────────────────────────────

def _fingering_cost_chord(fingers: list[int], pitches: list[int],
                           hf: float, side: str) -> float:
    """
    Coste intrínseco de una asignación de dedos a un acorde.

    Penaliza:
      - Spans que superan el máximo cómodo para ese par de dedos
      - Uso del pulgar en teclas negras interiores del acorde
      - Orden de dedos que no respeta la dirección del acorde
    """
    cost = 0.0
    xs = [keypos_midi(p) for p in pitches]
    n = len(fingers)

    for i in range(n):
        for j in range(i+1, n):
            fa, fb = fingers[i], fingers[j]
            xa, xb = xs[i], xs[j]
            span = abs(xb - xa)
            pair = (min(fa,fb), max(fa,fb))
            max_ok = _MAX_FINGER_SPAN.get(pair, 8.0) * hf
            if span > max_ok:
                cost += (span - max_ok) ** 2 * 10.0

            expected_dir = 1 if side == "right" else -1
            actual_dir   = 1 if fb > fa else -1
            pitch_dir    = 1 if xb > xa else -1
            if actual_dir * expected_dir != pitch_dir:
                cost += 5.0

    for i, (f, p) in enumerate(zip(fingers, pitches)):
        if f == 1 and (p % 12) in {1,3,6,8,10}:
            if side == "right" and i > 0:
                cost += 3.0
            elif side == "left" and i < n-1:
                cost += 3.0

    return cost


def _transition_cost_chord(fingers_a: list[int], pitches_a: list[int],
                            fingers_b: list[int], pitches_b: list[int],
                            hf: float) -> float:
    """
    Coste de transición entre dos acordes consecutivos.

    Modela el movimiento físico de cada dedo desde su posición actual
    hasta la posición requerida en el acorde siguiente. Un dedo que
    toca la misma tecla o se mueve poco tiene coste bajo; un salto
    grande o un cambio de dedo para la misma tecla tiene coste alto.
    """
    if not fingers_b or not pitches_b:
        return 0.0

    xs_a = {f: keypos_midi(p) for f, p in zip(fingers_a, pitches_a)}
    xs_b = {f: keypos_midi(p) for f, p in zip(fingers_b, pitches_b)}

    cost = 0.0
    for fb, xb in xs_b.items():
        if fb in xs_a:
            # Mismo dedo: coste proporcional al desplazamiento
            cost += abs(xb - xs_a[fb]) * 0.5
        else:
            # Dedo nuevo: viene desde su posición de reposo aproximada
            # (penalización fija moderada)
            cost += 2.0

    # Penalizar cambio de pulgar entre acordes próximos en el teclado
    # (el pulgar tiene que cruzar si los acordes están muy juntos)
    if 1 in xs_a and 1 in xs_b:
        thumb_jump = abs(xs_b[1] - xs_a[1])
        if thumb_jump > 8.0 * hf:
            cost += (thumb_jump - 8.0 * hf) * 1.5

    return cost


_CHORD_REF_SCALE = 5.0   # coste (unidades de _fingering_cost_chord) de λ=1 por nota que rompe la plantilla


def optimize_chord(notes: list[INote], hf: float, side: str,
                   next_notes: list[INote] | None = None,
                   start_finger: int = 0,
                   lam: float = 0.0) -> list[int]:
    """
    Asigna dedos óptimos a un grupo de notas de acorde.

    Considera tanto el coste intrínseco del acorde como el coste de
    transición hacia el acorde siguiente (si se proporciona).

    v4: si lam > 0 y las notas traen ref_finger (plantilla de clase), cada nota
    cuyo dedo difiere de la plantilla suma lam * _CHORD_REF_SCALE * ref_weight.
    """
    from itertools import combinations

    n = len(notes)
    if n == 0:
        return []
    if n > 5:
        return list(range(1, 6))[:n]

    pitches = [note.pitch for note in notes]

    # Preparar acorde siguiente si existe
    next_pitches  = [nn.pitch for nn in next_notes] if next_notes else []

    best_fingers = list(range(1, n+1))
    best_cost    = float("inf")

    for finger_combo in combinations(range(1, 6), n):
        if side == "right":
            fingers = sorted(finger_combo)
        else:
            fingers = sorted(finger_combo, reverse=True)

        if start_finger and start_finger not in fingers:
            continue

        c = _fingering_cost_chord(fingers, pitches, hf, side)
        if lam > 0:
            c += lam * _CHORD_REF_SCALE * sum(
                nt.ref_weight for nt, f in zip(notes, fingers)
                if nt.ref_finger and f != nt.ref_finger)

        # Añadir coste de transición si hay acorde siguiente
        if next_pitches:
            # Evaluar el mejor acorde siguiente dado este acorde actual
            best_next_cost = float("inf")
            for nc in combinations(range(1, 6), len(next_pitches)):
                nf = sorted(nc) if side == "right" else sorted(nc, reverse=True)
                tc = _transition_cost_chord(fingers, pitches, nf, next_pitches, hf)
                nc_cost = _fingering_cost_chord(nf, next_pitches, hf, side)
                best_next_cost = min(best_next_cost, tc + nc_cost * 0.3)
            c += best_next_cost * 0.4   # peso de la transición vs. coste intrínseco

        if c < best_cost:
            best_cost    = c
            best_fingers = fingers

    return best_fingers



def optimize_sustained(sus_seq: list[INote], mel_seq: list[INote],
                       hf: float, side: str) -> None:
    """
    Asigna digitación a las notas de la voz sostenida.

    Principio: la nota sostenida necesita un dedo que:
      1. Pueda alcanzar el pitch cómodamente.
      2. No colisione con los dedos que la melodía tiene asignados en ese intervalo.
      3. Respete el orden anatómico: dedo mayor → tecla más aguda (MD) / grave (MI).

    Si no existe ningún dedo libre que cumpla el orden anatómico, la nota se
    marca como imposible (fingering=0, confidence=0) y se advierte al usuario.
    """
    if not sus_seq:
        return

    for sn in sus_seq:
        t0, t1 = sn.time, sn.time + sn.duration

        # Dedos ya asignados a la melodía en este intervalo, con sus pitches
        mel_finger_pitch: list[tuple[int,int]] = [
            (mn.fingering, mn.pitch) for mn in mel_seq
            if mn.fingering > 0
            and mn.time < t1 and (mn.time + mn.duration) > t0
        ]
        occupied = {fp[0] for fp in mel_finger_pitch}

        simultaneous_pitches = [p for _, p in mel_finger_pitch]

        # Candidatos libres en orden de preferencia según posición relativa
        if side == "right":
            if not simultaneous_pitches or sn.pitch < min(simultaneous_pitches):
                preference = [1, 2, 3, 4, 5]   # grave → pulgar primero
            else:
                preference = [2, 1, 3, 4, 5]
        else:
            if not simultaneous_pitches or sn.pitch < min(simultaneous_pitches):
                preference = [5, 4, 3, 2, 1]   # grave en MI → meñique primero
            else:
                preference = [4, 5, 3, 2, 1]

        # Filtrar candidatos que respeten el orden anatómico:
        # en MD: dedo sostenida < dedo melodía si pitch sostenida < pitch melodía
        def anatomically_valid(f: int) -> bool:
            for fm, pm_pitch in mel_finger_pitch:
                if side == "right":
                    # MD: dedo mayor debe tocar nota más aguda
                    if sn.pitch < pm_pitch and f > fm:
                        return False   # sostenida grave pero dedo más alto: imposible
                    if sn.pitch > pm_pitch and f < fm:
                        return False
                else:
                    # MI: espejo
                    if sn.pitch < pm_pitch and f < fm:
                        return False
                    if sn.pitch > pm_pitch and f > fm:
                        return False
            return True

        chosen = None
        for f in preference:
            if f not in occupied and anatomically_valid(f):
                chosen = f
                break

        if chosen is None:
            # Ningún dedo válido: situación anatómicamente imposible
            sn.fingering  = 0
            sn.confidence = 0.0
        else:
            sn.fingering  = chosen
            sn.confidence = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Motor de optimización principal
# ─────────────────────────────────────────────────────────────────────────────

class Hand:
    """Optimizador de digitación para una mano sobre una secuencia de notas."""

    _SIZE_FACTORS = {
        "XXS": 0.33, "XS": 0.46, "S": 0.64,
        "M":   0.82, "L":  1.0,  "XL": 1.1, "XXL": 1.2,
    }

    def __init__(self, noteseq: list[INote], side: str = "right",
                 size: str = "M") -> None:
        self.LR      = side
        self.noteseq = list(noteseq)
        self.fingers = (1, 2, 3, 4, 5)

        self.frest   = [None, -7.0, -2.8,  0.0,  2.8,  5.6]
        self.weights = [None,  1.1,  1.0,  1.1,  0.9,  0.8]
        self.bfactor = [None,  0.3,  1.0,  1.1,  0.8,  0.7]

        self.hf = self._SIZE_FACTORS.get(size, self._SIZE_FACTORS["M"])
        for i in range(1, 6):
            if self.frest[i] is not None:
                self.frest[i] *= self.hf  # type: ignore[operator]

        self.depth     = 9
        self.autodepth = True

        # ── Memoria de postura (ACTIVADA en v2) ──────────────────────────────
        self.finger_positions     = list(self.frest)
        self._has_position_state  = False
        self.preserve_posture_mem = True      # ← activada
        self.relocation_alpha     = 0.3       # 0=siempre reposo, 1=nunca se mueve
        self.max_span_cm          = 21.0 * self.hf
        self.max_follow_lag_cm    =  2.5 * self.hf
        self.min_finger_gap_cm    =  0.15 * self.hf

        # Umbral de confianza: si el 2º mejor coste está dentro de este
        # porcentaje del mejor, la nota se marca como ambigua
        self.confidence_threshold = 0.05  # gap del 5% ya es significativo en arpeggios regulares

        self.fingerseq: list[list] = []

        # ── v4: penalización por desviarse de la plantilla de acorde ─────────
        self.chord_lambda = 0.0   # λ (0 = agnóstico, idéntico a v3)
        self.cost_scale   = 1.0   # coste típico de una nota (mediana del pase agnóstico)
        self._last_pure   = 0.0   # coste de velocidad del mejor candidato, sin penalización

    # ── geometría ─────────────────────────────────────────────────────────────

    def _relaxed_targets(self, fi: int, note_x: float) -> dict[int, float]:
        ifx = self.frest[fi]
        if ifx is None:
            return {}
        return {j: (self.frest[j] - ifx) + note_x  # type: ignore[operator]
                for j in range(1, 6) if self.frest[j] is not None}

    def _apply_position_constraints(self, fp: list, fi: int,
                                     note_x: float, targets: dict) -> None:
        for j in range(1, 6):
            if j == fi:
                continue
            pos = fp[j]; tgt = targets.get(j)
            if pos is None or tgt is None:
                continue
            lag = pos - tgt
            if   lag >  self.max_follow_lag_cm: fp[j] = tgt + self.max_follow_lag_cm
            elif lag < -self.max_follow_lag_cm: fp[j] = tgt - self.max_follow_lag_cm

        for j in range(2, 6):
            a, b = fp[j-1], fp[j]
            if a is not None and b is not None and b < a + self.min_finger_gap_cm:
                fp[j] = a + self.min_finger_gap_cm

        if fp[1] is not None and fp[5] is not None:
            span = fp[5] - fp[1]
            if span > self.max_span_cm:
                limit = self.max_span_cm / 2.0
                for j in range(1, 6):
                    if j == fi or fp[j] is None:
                        continue
                    off = fp[j] - note_x
                    fp[j] = note_x + max(-limit, min(limit, off))

        fp[fi] = note_x

    def set_fingers_positions(self, fings: Sequence[int], notes: Sequence[INote],
                               i: int, *, fp: list | None = None,
                               force_relaxed: bool = False) -> None:
        if fp is None:
            fp = self.finger_positions
            force_relaxed = not self._has_position_state

        fi     = fings[i]
        note_x = notes[i].x
        targets = self._relaxed_targets(fi, note_x)
        if not targets:
            return

        if force_relaxed or not self.preserve_posture_mem:
            for j in range(1, 6):
                fp[j] = targets.get(j)
            fp[fi] = note_x
            if fp is self.finger_positions:
                self._has_position_state = True
            return

        for j in range(1, 6):
            tgt  = targets.get(j)
            if tgt is None:
                fp[j] = None; continue
            if j == fi:
                fp[j] = note_x; continue
            prev = fp[j]
            fp[j] = tgt if prev is None else (
                self.relocation_alpha * prev + (1.0 - self.relocation_alpha) * tgt
            )

        self._apply_position_constraints(fp, fi, note_x, targets)
        if fp is self.finger_positions:
            self._has_position_state = True

    # ── función de coste ──────────────────────────────────────────────────────

    def ave_velocity(self, fingering: Sequence[int], notes: Sequence[INote]) -> float:
        """Coste promedio de velocidad de dedo para una digitación candidata."""
        fp = list(self.finger_positions)
        self.set_fingers_positions(fingering, notes, 0, fp=fp, force_relaxed=False)
        vmean = 0.0
        for i in range(1, self.depth):
            na, nb = notes[i-1], notes[i]
            fb  = fingering[i]
            pos = fp[fb]
            if pos is None:
                continue
            dx = abs(nb.x - pos)
            dt = abs(nb.time - na.time) + 0.1
            v  = dx / dt
            w  = self.weights[fb] or 1.0
            bf = (self.bfactor[fb] or 1.0) if nb.isBlack else 1.0
            vmean += v / (w * bf)
            self.set_fingers_positions(fingering, notes, i, fp=fp, force_relaxed=False)
        return vmean / max(1, self.depth - 1)

    # ── reglas de poda ────────────────────────────────────────────────────────

    def skip(self, fa: int, fb: int, na: INote, nb: INote) -> bool:
        xba = nb.x - na.x

        if not na.isChord and not nb.isChord:
            if fa == fb and xba and na.duration < 4:
                return True
            if fa > 1:
                if fb > 1 and (fb - fa) * xba < 0:
                    return True
                if fb == 1 and nb.isBlack and xba > 0:
                    return True
            elif na.isBlack and xba < 0 and fb > 1 and na.duration < 2:
                return True

        elif na.isChord and nb.isChord and na.chordID == nb.chordID:
            axba = abs(xba) * self.hf / 0.8
            if fa == fb:
                return True
            if fa < fb and self.LR == "left":
                return True
            if fa > fb and self.LR == "right":
                return True
            pair = (min(fa,fb), max(fa,fb))
            thresh = _MAX_FINGER_SPAN.get(pair)
            if thresh and axba > thresh * self.hf:
                return True

        return False

    # ── optimización por ventana deslizante con confianza ────────────────────

    def optimize_seq(self, nseq: Sequence[INote],
                     istart: int) -> tuple[list[int], float, float]:
        """
        Mejor digitación para una ventana de hasta 9 notas.

        Devuelve (fingering, best_cost, confidence).
        confidence ∈ [0,1]: baja cuando hay varios candidatos casi igual de buenos.
        """
        if self.autodepth:
            if nseq[0].isChord:
                self.depth = max(3, nseq[0].NinChord - nseq[0].chordnr + 1)
            else:
                t0 = nseq[0].time
                for i in range(4, 10):
                    self.depth = i
                    if nseq[i-1].time - t0 > 3.5:
                        break

        depth    = self.depth
        u_start  = list(self.fingers) if istart == 0 else [istart]
        best     = [0] * 9
        minv     = 1e10
        second_v = 1e10   # segundo mejor coste, para calcular confianza
        cand     = [0] * 9
        best_pure = 0.0

        # v4: penalización blanda por romper la plantilla de clase
        lam_scale = self.chord_lambda * self.cost_scale
        use_ref   = lam_scale > 0 and any(nt.ref_finger for nt in nseq[:depth])

        def bt(level: int, pen: float) -> None:
            nonlocal best, minv, second_v, best_pure
            if level == depth:
                pure = self.ave_velocity(cand, nseq)
                v = pure + pen
                if v < minv:
                    second_v  = minv
                    best[:]   = cand[:]
                    minv      = v
                    best_pure = pure
                elif v < second_v:
                    second_v = v
                return
            choices = u_start if level == 0 else self.fingers
            for f in choices:
                if level > 0 and self.skip(cand[level-1], f,
                                            nseq[level-1], nseq[level]):
                    continue
                cand[level] = f
                p = pen
                if use_ref:
                    nt = nseq[level]
                    if nt.ref_finger and f != nt.ref_finger:
                        p += lam_scale * nt.ref_weight
                bt(level + 1, p)

        bt(0, 0.0)
        self._last_pure = best_pure

        # Confianza: 1.0 si el gap entre 1º y 2º es grande; 0.0 si son iguales
        if second_v >= 1e9 or minv < 1e-9:
            confidence = 1.0
        else:
            gap = (second_v - minv) / (minv + 1e-9)
            confidence = min(1.0, gap / self.confidence_threshold)

        return best, minv, confidence

    # ── generación completa ───────────────────────────────────────────────────

    def generate(self) -> None:
        """Asigna fingering y confidence a cada nota en self.noteseq."""
        init_autodepth = self.autodepth
        init_depth     = self.depth
        original_x: list[float] | None = None

        if self.LR == "left":
            original_x = [n.x for n in self.noteseq]
            for n in self.noteseq:
                n.x = -n.x

        # Separar melodía y acordes
        chord_groups: dict[int, list[INote]] = defaultdict(list)
        melody_notes: list[INote] = []
        for n in self.noteseq:
            if n.isChord:
                chord_groups[n.chordID].append(n)
            else:
                melody_notes.append(n)

        # ── 1. Optimizar acordes con módulo específico ────────────────────────
        # Construir lista ordenada de IDs de acorde para acceder al siguiente
        chord_ids_ordered = sorted(chord_groups.keys())
        for idx, chord_id in enumerate(chord_ids_ordered):
            cnotes = chord_groups[chord_id]
            cnotes_sorted = sorted(cnotes, key=lambda n: n.pitch)

            # Acorde siguiente (para coste de transición)
            next_cnotes: list[INote] | None = None
            if idx + 1 < len(chord_ids_ordered):
                next_id = chord_ids_ordered[idx + 1]
                next_cnotes = sorted(chord_groups[next_id], key=lambda n: n.pitch)

            fingers = optimize_chord(cnotes_sorted, self.hf, self.LR,
                                     next_notes=next_cnotes,
                                     lam=self.chord_lambda)
            for note, f in zip(cnotes_sorted, fingers):
                note.fingering   = f
                note.confidence  = 0.9

        # ── 2. Optimizar melodía con ventana deslizante ───────────────────────
        self.fingerseq = []
        self.finger_positions    = list(self.frest)
        self._has_position_state = False

        out: list[int]    = []
        vel: float        = 0.0
        conf: float       = 1.0
        start_finger      = 0
        n_total           = len(melody_notes)

        try:
            for i in range(n_total):
                an = melody_notes[i]

                # En la cola reducimos el depth al número real de notas restantes
                # para no rellenar la ventana con duplicados que distorsionan el coste.
                remaining = n_total - i
                if remaining < 9:
                    if self.autodepth:
                        self.autodepth = False
                    self.depth = max(2, remaining)

                window = list(melody_notes[i : i + 9])
                if window and len(window) < 9:
                    window += [window[-1]] * (9 - len(window))
                if not window:
                    break

                out, vel, conf = self.optimize_seq(window, start_finger)
                best_finger    = out[0]
                start_finger   = out[1] if len(out) > 1 else out[0]

                an.fingering  = best_finger
                an.confidence = round(conf, 3)
                an.cost       = round(self._last_pure, 4)
                self.set_fingers_positions(out, window, 0)
                self.fingerseq.append(list(self.finger_positions))

        finally:
            self.autodepth = init_autodepth
            self.depth     = init_depth
            if original_x is not None:
                for n, x in zip(self.noteseq, original_x):
                    n.x = x

        # Reparar ceros residuales
        last = 1
        for n in self.noteseq:
            if n.fingering == 0:
                n.fingering = last
            else:
                last = n.fingering


# ─────────────────────────────────────────────────────────────────────────────
# v4 — Noción de acorde: compases → acorde → clases de voicing → plantilla
# ─────────────────────────────────────────────────────────────────────────────
#
# Flujo del modo --chords:
#   0. Pase agnóstico (= v3) para ambas manos. Da la digitación base y la escala
#      de coste típica de cada mano.
#   1. Por compás se calcula el acorde (plantillas de clases de altura, con
#      bonificación para la nota del bajo) y la función de cada nota (R, 3, 5…).
#   2. Por mano y compás se construye un "voicing" y se agrupa en clases:
#         exact : mismas notas, mismo orden               (peso 1.0)
#         roles : mismo tipo de acorde y mismas funciones (peso 0.7)
#                 en el mismo orden/registro, transportado
#         shape : misma forma (contorno y agrupación de   (peso 0.4)
#                 onsets) y tamaño de intervalo similar;
#                 las escaladas estrictas se excluyen (--chord-runs las incluye)
#      Cada voicing usa la clase más específica que tenga ≥2 miembros.
#   3. Para cada clase se elige una digitación de consenso (voto por posición,
#      desempate por comodidad geométrica).
#   4. Se vuelve a optimizar con penalización λ·escala·peso por cada nota que
#      rompe la plantilla. Se itera hasta que la digitación deje de cambiar.

_PC_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_PC_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6}     # F Bb Eb Ab Db Gb
_FLAT_MINOR_TONICS = {2, 7, 0, 5, 10, 3}     # Dm Gm Cm Fm Bbm Ebm

_CHORD_TEMPLATES: list[tuple[str, tuple[int, ...]]] = [
    ("",     (0, 4, 7)),
    ("m",    (0, 3, 7)),
    ("dim",  (0, 3, 6)),
    ("aug",  (0, 4, 8)),
    ("sus4", (0, 5, 7)),
    ("sus2", (0, 2, 7)),
    ("7",    (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)),
    ("m7",   (0, 3, 7, 10)),
    ("m7b5", (0, 3, 6, 10)),
    ("dim7", (0, 3, 6, 9)),
    ("6",    (0, 4, 7, 9)),
    ("m6",   (0, 3, 7, 9)),
    ("5",    (0, 7)),
]
_ROLE_NAMES = {0: "R", 1: "b9", 2: "9", 3: "b3", 4: "3", 5: "4", 6: "b5",
               7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}

_TIER_ORDER  = ("exact", "roles", "shape")
_TIER_WEIGHT = {"exact": 1.0, "roles": 0.7, "shape": 0.4}


def uses_flats(pm: pretty_midi.PrettyMIDI) -> bool:
    """True si la armadura del MIDI es de bemoles (para nombrar los acordes)."""
    ks = pm.key_signature_changes
    if not ks:
        return False
    k = ks[0].key_number
    return (k in _FLAT_MAJOR_TONICS) if k < 12 else ((k - 12) in _FLAT_MINOR_TONICS)


@dataclass
class ChordInfo:
    measure: int
    root_pc: int = -1
    quality: str = ""
    bass_pc: int = -1
    label:   str = "?"
    margin:  float = 0.0     # ventaja de puntuación sobre la 2ª mejor alternativa


def detect_chord(notes: list[INote], spb: float, flats: bool) -> ChordInfo:
    """
    Acorde de un compás a partir de TODAS las notas que suenan en él.
    Puntuación: peso de las clases de altura dentro de la plantilla menos el
    peso de las que quedan fuera, penalizando notas de la plantilla ausentes y
    bonificando que la raíz coincida con el bajo (nota más grave del primer tiempo).
    """
    if not notes:
        return ChordInfo(measure=0)
    names = _PC_FLAT if flats else _PC_SHARP
    measure = notes[0].measure

    pool = [n for n in notes if n.beat < 1.0] or notes
    bass_pc = min(pool, key=lambda n: n.pitch).pitch % 12

    pcp = [0.0] * 12
    for n in notes:
        pcp[n.pitch % 12] += 0.35 + min(n.duration / spb, 2.0)
    pcp[bass_pc] += 1.0
    total = sum(pcp)

    scored: list[tuple[float, int, str]] = []
    for root in range(12):
        for q, tpl in _CHORD_TEMPLATES:
            tones = {(root + i) % 12 for i in tpl}
            inw = sum(pcp[t] for t in tones)
            outw = total - inw
            missing = sum(1 for t in tones if pcp[t] == 0)
            sc = inw - outw - 0.35 * total * missing / len(tpl)
            if root == bass_pc:
                sc += 0.3 * total
            if len(tpl) == 3:
                sc += 0.02 * total
            scored.append((sc, root, q))
    scored.sort(key=lambda t: -t[0])
    best, second = scored[0], scored[1]
    root, q = best[1], best[2]
    label = f"{names[root]}{q}" + (f"/{names[bass_pc]}" if bass_pc != root else "")
    return ChordInfo(measure=measure, root_pc=root, quality=q, bass_pc=bass_pc,
                     label=label, margin=round(best[0] - second[0], 3))


def analyze_chords(all_notes: list[INote], tmap: TempoMap,
                   flats: bool) -> dict[int, ChordInfo]:
    """Acorde por compás; anota en cada nota su acorde y su función."""
    spb = 60.0 / tmap.bpm0
    by_m: dict[int, list[INote]] = defaultdict(list)
    for n in all_notes:
        by_m[n.measure].append(n)
    chords: dict[int, ChordInfo] = {}
    for m in sorted(by_m):
        ci = detect_chord(by_m[m], spb, flats)
        chords[m] = ci
        for n in by_m[m]:
            n.chord = ci.label
            n.chord_role = _ROLE_NAMES[(n.pitch - ci.root_pc) % 12] if ci.root_pc >= 0 else ""
    return chords


@dataclass
class Voicing:
    """Las notas que una mano toca en un compás, más sus claves de clase."""
    side: str
    measure: int
    notes: list[INote]
    chord: ChordInfo
    keys: dict[str, tuple] = field(default_factory=dict)

    @property
    def pitches(self) -> list[int]:
        return [n.pitch for n in self.notes]

    @property
    def fingers(self) -> tuple[int, ...]:
        return tuple(n.fingering for n in self.notes)


def _span_bucket(semitones: int) -> int:
    return 0 if semitones <= 4 else 1 if semitones <= 8 else 2 if semitones <= 12 else 3


def _is_run(contour: tuple[int, ...], grouping: tuple[int, ...]) -> bool:
    """Escalada/descenso estricto de ≥3 notas sucesivas (gesto melódico, no voicing)."""
    if len(contour) < 3 or len(set(grouping)) != len(grouping):
        return False
    up   = all(b > a for a, b in zip(contour, contour[1:]))
    down = all(b < a for a, b in zip(contour, contour[1:]))
    return up or down


def build_voicings(seq: list[INote], side: str,
                   chords: dict[int, ChordInfo],
                   shape_runs: bool = False) -> list[Voicing]:
    by_m: dict[int, list[INote]] = defaultdict(list)
    for n in seq:
        by_m[n.measure].append(n)
    out: list[Voicing] = []
    for m in sorted(by_m):
        ns = sorted(by_m[m], key=lambda n: (n.time, n.pitch))
        pitches = [n.pitch for n in ns]
        uniq = sorted(set(pitches))
        contour  = tuple(uniq.index(p) for p in pitches)
        beats    = sorted(set(n.beat for n in ns))
        grouping = tuple(beats.index(n.beat) for n in ns)
        span     = max(pitches) - min(pitches)
        ci = chords.get(m, ChordInfo(measure=m))
        keys: dict[str, tuple] = {"exact": ("exact", tuple(pitches), grouping)}
        if ci.root_pc >= 0:
            keys["roles"] = ("roles", ci.quality, tuple(n.chord_role for n in ns),
                             contour, grouping)
        # Las escaladas se comparan por acorde/notas (exact, roles), pero no por
        # mera forma: dos escaladas de 3 notas no son evidencia de "mismo acorde".
        if shape_runs or not _is_run(contour, grouping):
            keys["shape"] = ("shape", contour, grouping, _span_bucket(span))
        out.append(Voicing(side=side, measure=m, notes=ns, chord=ci, keys=keys))
    return out


def _template_comfort(fingers: Sequence[int], pitches: Sequence[int],
                      hf: float, side: str) -> float:
    """Coste geométrico de aplicar una plantilla de dedos a un voicing concreto."""
    pts: dict[int, set[int]] = defaultdict(set)
    for f, p in zip(fingers, pitches):
        pts[f].add(p)
    fs = sorted(pts)
    exp = 1 if side == "right" else -1
    cost = 0.0
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            fa, fb = fs[i], fs[j]
            max_ok = _MAX_FINGER_SPAN.get((fa, fb), 8.0) * hf
            for pa in pts[fa]:
                for pb in pts[fb]:
                    span = abs(keypos_midi(pb) - keypos_midi(pa))
                    if span > max_ok:
                        cost += (span - max_ok) ** 2 * 10.0
                    if pb != pa and (1 if pb > pa else -1) != exp:
                        cost += 5.0          # dedo mayor en tecla más grave (MD)
    return cost


def consensus_template(members: list[Voicing], hf: float, side: str) -> list[int]:
    """
    Digitación de consenso de una clase de voicings.

    Candidatos: todas las combinaciones de los dedos empatados en cada posición
    (voto por posición). Ranking:
      1. menor desacuerdo total (distancia de Hamming a las digitaciones actuales),
      2. menor incomodidad geométrica sumada sobre los miembros (spans y orden
         anatómico de los dedos; descarta mezclas como 3-1-3),
      3. preferir una tupla que algún miembro ya use (la haya encontrado el optimizador),
      4. orden lexicográfico (determinista).
    """
    from itertools import product

    k = len(members[0].notes)
    observed = {v.fingers for v in members}
    tied_sets: list[list[int]] = []
    for pos in range(k):
        votes: dict[int, int] = defaultdict(int)
        for v in members:
            votes[v.notes[pos].fingering] += 1
        top = max(votes.values())
        tied_sets.append(sorted(f for f, c in votes.items() if c == top))

    cands = set(product(*tied_sets)) | observed
    ranked: list[tuple] = []
    for t in sorted(cands):
        ham = sum(sum(a != b for a, b in zip(t, v.fingers)) for v in members)
        comfort = sum(_template_comfort(t, v.pitches, hf, side) for v in members)
        ranked.append((ham, round(comfort, 6), 0 if t in observed else 1, t))
    ranked.sort()
    return list(ranked[0][3])


def voicing_classes(voicings: list[Voicing]) -> dict[tuple, list[Voicing]]:
    classes: dict[tuple, list[Voicing]] = defaultdict(list)
    for v in voicings:
        if len(v.notes) < 2:
            continue
        for key in v.keys.values():
            classes[key].append(v)
    return classes


def assign_templates(voicings: list[Voicing], hf: float, side: str) -> None:
    """Escribe ref_finger/ref_weight/ref_tier en las notas según la clase más específica."""
    classes = voicing_classes(voicings)
    cache: dict[tuple, list[int]] = {}
    for v in voicings:
        for n in v.notes:
            n.ref_finger, n.ref_weight, n.ref_tier = 0, 0.0, ""
        if len(v.notes) < 2:
            continue
        for tier in _TIER_ORDER:
            key = v.keys.get(tier)
            if key is None or len(classes[key]) < 2:
                continue
            if key not in cache:
                cache[key] = consensus_template(classes[key], hf, side)
            for n, f in zip(v.notes, cache[key]):
                n.ref_finger, n.ref_weight, n.ref_tier = f, _TIER_WEIGHT[tier], tier
            break


@dataclass
class ChordAnalysis:
    chords: dict[int, ChordInfo]
    voicings: dict[str, list[Voicing]]
    iterations: int = 0
    converged: bool = True


def generate_chord_mode(rh_seq: list[INote], lh_seq: list[INote],
                        context_notes: list[INote], tmap: TempoMap,
                        size: str, lam: float, flats: bool,
                        max_iter: int = 3, log=print,
                        shape_runs: bool = False) -> ChordAnalysis:
    """
    Digitación con noción de acorde. lam=0 devuelve exactamente la solución de v3.

    context_notes: TODAS las notas de la pieza (ambas manos, melodía y sostenidas)
    para el análisis armónico, aunque solo se digite una mano (--left-only).
    """
    hf = Hand._SIZE_FACTORS.get(size, 0.82)
    seqs = {"right": rh_seq, "left": lh_seq}
    seqs = {s: q for s, q in seqs.items() if q}

    # 0. pase agnóstico
    scale: dict[str, float] = {}
    for side, seq in seqs.items():
        Hand(seq, side, size).generate()
        costs = sorted(n.cost for n in seq if not n.isChord and n.cost > 0)
        scale[side] = costs[len(costs) // 2] if costs else 1.0

    # 1-2. acordes y clases
    chords = analyze_chords(context_notes, tmap, flats)
    voicings = {side: build_voicings(seq, side, chords, shape_runs)
                for side, seq in seqs.items()}
    result = ChordAnalysis(chords=chords, voicings=voicings)
    if lam <= 0:
        return result

    # 3-4. plantillas + reoptimización hasta punto fijo
    prev = None
    for it in range(1, max_iter + 1):
        for side, seq in seqs.items():
            assign_templates(voicings[side], hf, side)
            h = Hand(seq, side, size)
            h.chord_lambda, h.cost_scale = lam, scale[side]
            h.generate()
        snap = tuple(n.fingering for seq in seqs.values() for n in seq)
        result.iterations = it
        if snap == prev:
            result.converged = True
            break
        prev = snap
        result.converged = False
        log(f"  iteración {it}: digitación actualizada")
    return result


# ── informes ────────────────────────────────────────────────────────────────

def _fmt_fingers(fs: Sequence[int]) -> str:
    return "-".join(str(f) for f in fs)


def print_chord_report(analysis: ChordAnalysis, hand_size: str = "M") -> None:
    side_label = {"right": "MD", "left": "MI"}
    print("Acorde por compás (todas las notas del compás, ambas manos):")
    for m, ci in sorted(analysis.chords.items()):
        print(f"  c.{m:<3} {ci.label:<10} margen={ci.margin:.2f}")
    print()
    for side, vs in analysis.voicings.items():
        print(f"Voicings repetidos — {side_label[side]}:")
        classes = voicing_classes(vs)
        shown = 0
        for tier in _TIER_ORDER:
            for key, members in sorted(classes.items(), key=lambda kv: kv[1][0].measure):
                if key[0] != tier or len(members) < 2:
                    continue
                # no repetir clases menos específicas ya cubiertas por una más específica
                if tier != "exact" and all(
                        len(classes[v.keys[t]]) >= 2
                        for v in members for t in _TIER_ORDER[:_TIER_ORDER.index(tier)]
                        if t in v.keys):
                    continue
                fps = {v.fingers for v in members}
                mark = "✓ coinciden" if len(fps) == 1 else "✗ difieren"
                notes = " ".join(pitch_name(p) for p in members[0].pitches) \
                    if tier == "exact" else "(transportado)"
                ms = ", ".join(str(v.measure) for v in members)
                fs = " | ".join(_fmt_fingers(f) for f in
                                [v.fingers for v in members])
                print(f"  [{tier:<5}] c.{ms:<14} {notes:<24} {fs}  {mark}")
                shown += 1
        if not shown:
            print("  (ninguno)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Salida estructurada
# ─────────────────────────────────────────────────────────────────────────────

_FINGER_NAMES = {1:"pulgar", 2:"índice", 3:"corazón", 4:"anular", 5:"meñique"}


def build_result(midi_path: str, tmap: TempoMap,
                 rh_seq: list[INote], lh_seq: list[INote],
                 rh_sus: list[INote] | None = None,
                 lh_sus: list[INote] | None = None) -> FingeringResult:
    """Construye el resultado estructurado como FingeringResult."""
    all_notes = rh_seq + lh_seq + (rh_sus or []) + (lh_sus or [])
    all_measures = set(n.measure for n in all_notes)

    notes_out = []
    for n in sorted(all_notes, key=lambda n: (n.measure, n.beat, n.hand, n.pitch)):
        notes_out.append({
            "measure":    n.measure,
            "beat":       n.beat,
            "hand":       n.hand,
            "note":       n.name,
            "pitch":      n.pitch,
            "fingering":  n.fingering,
            "finger_name": _FINGER_NAMES.get(n.fingering, "?"),
            "confidence": n.confidence,
            "is_chord":   n.isChord,
            "chord_id":   n.chordID if n.isChord else None,
            "duration_s": round(n.duration, 3),
            "voice":      n.voice,
            "chord":      n.chord,
            "chord_role": n.chord_role,
        })

    return FingeringResult(
        midi_file=midi_path,
        bpm=round(tmap.bpm0, 1),
        time_signature=f"{tmap.beats_per_measure}/4",
        measures=max(all_measures) if all_measures else 0,
        notes=notes_out,
    )


def print_fingering(result: FingeringResult,
                    right_only: bool = False,
                    left_only:  bool = False,
                    show_confidence: bool = False,
                    show_chords: bool = False) -> None:
    """Imprime el resultado agrupado por compás."""

    # Umbral adaptativo: percentil 25 de las confianzas de melodía.
    # Así solo el cuartil inferior (realmente ambiguo) recibe ?.
    melody_confs = [n["confidence"] for n in result.notes
                    if not n["is_chord"] and n["confidence"] < 0.89]
    if melody_confs:
        sorted_c = sorted(melody_confs)
        p25 = sorted_c[len(sorted_c) // 4]
        ambig_threshold = max(0.04, min(p25, 0.30))
    else:
        ambig_threshold = 0.15

    by_measure: dict[int, dict[str, list]] = defaultdict(lambda: {"right":[], "left":[]})
    for n in result.notes:
        if right_only and n["hand"] == "left":
            continue
        if left_only  and n["hand"] == "right":
            continue
        by_measure[n["measure"]][n["hand"]].append(n)

    for m in sorted(by_measure):
        chord_lbl = ""
        if show_chords:
            for hk in ("right", "left"):
                for nn in by_measure[m][hk]:
                    if nn.get("chord"):
                        chord_lbl = nn["chord"]
                        break
                if chord_lbl:
                    break
        print(f"Compás {m}" + (f"  [{chord_lbl}]" if chord_lbl else "") + ":")
        hands = []
        if not left_only:  hands.append(("right", "Mano derecha"))
        if not right_only: hands.append(("left",  "Mano izquierda"))

        for hand_key, hand_label in hands:
            notes = by_measure[m][hand_key]
            print(f"  {hand_label}:")
            if not notes:
                print("    (sin notas)")
                continue
            for n in sorted(notes, key=lambda x: (x["beat"], x["pitch"])):
                f    = n["fingering"]
                name = _FINGER_NAMES.get(f, "?")
                conf = n["confidence"]
                ambig = " (alt)" if (not n["is_chord"] and conf < ambig_threshold) else ""
                chord = " [acorde]" if n["is_chord"] else ""
                sust  = " [sostenida]" if n.get("voice") == "sustained" else ""
                imposible = " ⚠ IMPOSIBLE (demasiadas voces)" if (n.get("voice") == "sustained" and f == 0) else ""
                conf_str = f"  conf={conf:.2f}" if show_confidence else ""
                fname = _FINGER_NAMES.get(f, "?") if f > 0 else "—"
                fnum  = str(f) if f > 0 else "?"
                print(f"    {n['note']:<5} — {fnum} ({fname}){ambig}{chord}{sust}{imposible}{conf_str}")
        print()


def export_json(result: FingeringResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "midi_file":      result.midi_file,
            "bpm":            result.bpm,
            "time_signature": result.time_signature,
            "measures":       result.measures,
            "notes":          result.notes,
        }, f, ensure_ascii=False, indent=2)
    print(f"JSON exportado → {path}")


def export_musicxml(result: FingeringResult, rh_seq: list[INote],
                    lh_seq: list[INote], path: str) -> None:
    """
    Exporta a MusicXML con marcas de digitación estándar (<fingering>).
    Genera un fichero mínimo válido que MuseScore puede abrir.
    """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"')
    lines.append('  "http://www.musicxml.org/dtds/partwise.dtd">')
    lines.append('<score-partwise version="3.1">')
    lines.append('  <part-list>')
    lines.append('    <score-part id="P1"><part-name>Piano RH</part-name></score-part>')
    lines.append('    <score-part id="P2"><part-name>Piano LH</part-name></score-part>')
    lines.append('  </part-list>')

    def notes_to_part(seq: list[INote], part_id: str, clef: str) -> list[str]:
        out = [f'  <part id="{part_id}">']
        by_m: dict[int, list[INote]] = defaultdict(list)
        for n in seq:
            by_m[n.measure].append(n)

        # Calcular divisions (quarter note = 1 beat; corcheas = 2 divisions)
        divisions = 8
        bpm = result.bpm
        quarter_s = 60.0 / bpm

        for m in sorted(by_m):
            out.append(f'    <measure number="{m}">')
            if m == 1:
                out.append(f'      <attributes>')
                out.append(f'        <divisions>{divisions}</divisions>')
                num, den = result.time_signature.split("/")
                out.append(f'        <time><beats>{num}</beats><beat-type>{den}</beat-type></time>')
                out.append(f'        <clef><sign>{clef}</sign></clef>')
                out.append(f'      </attributes>')

            for n in sorted(by_m[m], key=lambda x: x.time):
                # Duración en divisions (aproximada a la corchea más cercana)
                dur_beats = n.duration / quarter_s
                dur_div   = max(1, round(dur_beats * divisions))

                step = n.name[:-1].replace("#","").replace("b","")
                alter_str = ""
                if "#" in n.name[:-1]:
                    alter_str = "<alter>1</alter>"
                elif "b" in n.name[:-1]:
                    alter_str = "<alter>-1</alter>"
                octave_xml = n.name[-1]

                chord_tag = "<chord/>" if n.isChord and n.chordnr > 0 else ""

                out.append(f'      <note>')
                if chord_tag:
                    out.append(f'        {chord_tag}')
                out.append(f'        <pitch>')
                out.append(f'          <step>{step[0]}</step>')
                if alter_str:
                    out.append(f'          {alter_str}')
                out.append(f'          <octave>{octave_xml}</octave>')
                out.append(f'        </pitch>')
                out.append(f'        <duration>{dur_div}</duration>')
                out.append(f'        <type>eighth</type>')
                if n.fingering:
                    out.append(f'        <notations>')
                    out.append(f'          <technical>')
                    out.append(f'            <fingering>{n.fingering}</fingering>')
                    out.append(f'          </technical>')
                    out.append(f'        </notations>')
                out.append(f'      </note>')

            out.append(f'    </measure>')
        out.append(f'  </part>')
        return out

    lines += notes_to_part(rh_seq, "P1", "G")
    lines += notes_to_part(lh_seq, "P2", "F")
    lines.append("</score-partwise>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"MusicXML exportado → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera digitación pianística a partir de un fichero MIDI (v3)."
    )
    parser.add_argument("midi", help="Ruta al fichero MIDI")
    parser.add_argument(
        "--hand-size", choices=["XXS","XS","S","M","L","XL","XXL"],
        default="M", metavar="SIZE",
        help="Tamaño de mano: XXS XS S M L XL XXL  (defecto: M ≈ 17 cm)"
    )
    parser.add_argument("--right-track", type=int, default=0)
    parser.add_argument("--left-track",  type=int, default=1)
    parser.add_argument(
        "--auto-split", action="store_true",
        help="Separar manos automáticamente aunque haya 2 tracks (usa ventana deslizante)"
    )
    parser.add_argument("--measures", type=int, default=0,
                        help="Número máximo de compases (0 = todos)")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--left-only",  action="store_true")
    parser.add_argument("--confidence", action="store_true",
                        help="Mostrar puntuación de confianza por nota")
    parser.add_argument("--chords", action="store_true",
                        help="Modo v4: noción de acorde (acorde por compás + consistencia entre compases)")
    parser.add_argument("--chord-weight", type=float, default=1.0, metavar="W",
                        help="λ: 0 = agnóstico (v3); ~1 equilibrado; ≥5 uniformidad casi forzada (defecto 1.0)")
    parser.add_argument("--chord-runs", action="store_true",
                        help="Con --chords: comparar también escaladas estrictas por su forma (nivel shape)")
    parser.add_argument("--chord-report", action="store_true",
                        help="Con --chords: acorde por compás y voicings repetidos con sus digitaciones")
    parser.add_argument("--json", metavar="FICHERO",
                        help="Exportar resultado a JSON")
    parser.add_argument("--xml", metavar="FICHERO",
                        help="Exportar resultado a MusicXML")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    try:
        pm, rh_seq, lh_seq, rh_sus, lh_sus, tmap = load_midi(
            args.midi,
            right_track=args.right_track,
            left_track=args.left_track,
            auto_split=args.auto_split,
        )
    except Exception as e:
        print(f"Error al leer el MIDI: {e}", file=sys.stderr)
        sys.exit(1)

    instruments = [i for i in pm.instruments if not i.is_drum]
    print(f"MIDI cargado: {args.midi}")
    print(f"  Tempo: {tmap.bpm0:.1f} bpm  |  Compás: {tmap.beats_per_measure}/4")
    print(f"  Tracks (no-drum): {len(instruments)}")
    for idx, inst in enumerate(instruments):
        print(f"    [{idx}] {inst.name or '(sin nombre)'} — {len(inst.notes)} notas")
    print()

    # v4: el análisis armónico usa siempre ambas manos, aunque se digite solo una
    ctx_notes = rh_seq + lh_seq + rh_sus + lh_sus

    if args.left_only:  rh_seq = []; rh_sus = []
    if args.right_only: lh_seq = []; lh_sus = []

    if args.measures > 0:
        rh_seq = [n for n in rh_seq if n.measure <= args.measures]
        lh_seq = [n for n in lh_seq if n.measure <= args.measures]
        rh_sus = [n for n in rh_sus if n.measure <= args.measures]
        lh_sus = [n for n in lh_sus if n.measure <= args.measures]

    print(f"Notas a procesar → MD: {len(rh_seq)} melodía + {len(rh_sus)} sostenidas, "
          f"MI: {len(lh_seq)} melodía + {len(lh_sus)} sostenidas")
    print()

    analysis = None
    if args.chords:
        print(f"Calculando digitación con noción de acorde (λ={args.chord_weight})…")
        analysis = generate_chord_mode(
            rh_seq, lh_seq, ctx_notes, tmap,
            size=args.hand_size, lam=args.chord_weight, flats=uses_flats(pm),
            shape_runs=args.chord_runs)
    else:
        if rh_seq:
            print("Calculando digitación mano derecha (melodía)…")
            Hand(rh_seq, side="right", size=args.hand_size).generate()

        if lh_seq:
            print("Calculando digitación mano izquierda (melodía)…")
            Hand(lh_seq, side="left",  size=args.hand_size).generate()

    if rh_sus:
        print("Calculando digitación mano derecha (voz sostenida)…")
        optimize_sustained(rh_sus, rh_seq, Hand._SIZE_FACTORS.get(args.hand_size, 0.82), "right")

    if lh_sus:
        print("Calculando digitación mano izquierda (voz sostenida)…")
        optimize_sustained(lh_sus, lh_seq, Hand._SIZE_FACTORS.get(args.hand_size, 0.82), "left")

    result = build_result(args.midi, tmap, rh_seq, lh_seq, rh_sus, lh_sus)

    print()
    print("─" * 60)
    print()

    print_fingering(result,
                    right_only=args.right_only,
                    left_only=args.left_only,
                    show_confidence=args.confidence,
                    show_chords=args.chords)

    if analysis is not None and args.chord_report:
        print("─" * 60)
        print()
        print_chord_report(analysis, args.hand_size)

    if args.json:
        export_json(result, args.json)

    if args.xml:
        export_musicxml(result, rh_seq + rh_sus, lh_seq + lh_sus, args.xml)


if __name__ == "__main__":
    main()
