#!/usr/bin/env python3
"""
fingering_v4.py — Generador automático de digitación pianística a partir de MIDI.

Novedades respecto a v3:
  - NUEVO modo acordes (opcional, --chord-mode): calcula el acorde (armonía) de cada
    compás y mantiene la digitación consistente entre acordes de la misma calidad
    (p. ej. el mismo arpegio sobre C, F y G usa los mismos dedos cuando es cómodo).
    El peso --chord-weight (λ) pondera entre la solución agnóstica (λ=0, = v3) y
    la uniformidad total de patrones (λ=1).
  - Corrección: los compases se calculan con la firma de tiempo completa (num/den).
    En v3 un 6/8 se interpretaba como 6 negras por compás.
  - Claves de consistencia por FORMA (--chord-keys shape|harmony|both): gestos
    idénticos (misma posición en el compás e intervalos con las notas vecinas) reciben
    los mismos dedos aunque la armonía detectada difiera. Defecto: both.
  - Prior de bajo en la mano izquierda (--bass-prior B): favorece el dedo 5 en la nota
    más grave de cada compás cuando cae en el tiempo 1. Independiente de --chord-mode.
  - Enlace de gestos repetidos (--tie-gestures [--tie-tol T]): los compases de una mano
    con exactamente el mismo gesto (mismas posiciones e intervalos, en cualquier altura)
    reciben una digitación común, elegida por menor coste conjunto entre las candidatas,
    siempre que ningún compás empeore más de T (relativo; defecto 0.5).
  - Escaladas (≥3 notas sucesivas estrictamente ascendentes o descendentes): no se
    comparan por su forma entre alturas distintas, solo con las mismas notas exactas
    (una escala en otra posición cambia con las teclas negras). --chord-runs lo desactiva.
  - Informe (--chord-report): acorde por compás con su margen (marca los empates) y tabla
    de gestos repetidos por mano con las digitaciones y si coinciden. Por sí solo no
    cambia ningún dedo respecto a v3.
  - Coste ergonómico sin penalizaciones (cost_pure): permite comparar, compás a compás,
    cuánto cuesta la digitación final frente a la agnóstica y avisa (⚠) cuando un compás
    cambiado cuesta más de --cost-alert (relativo; defecto 0.5 = +50%) y, además, al menos
    un 25% del coste típico de ventana. Mide el coste según el modelo de v3: no detecta
    digitaciones que ese modelo considera baratas aunque sean incómodas (p. ej. 5-4-4).
  - El informe y el JSON muestran el bajo del compás con barra (p. ej. Fm/Ab).
  - Sin --chord-mode, --bass-prior, --tie-gestures ni --chord-report el comportamiento
    es el de v3.

Rendimiento: optimize_seq usa ramificación y poda con coste incremental (cotas exactas,
  todos los términos del coste son >= 0). Resultado idéntico a la enumeración exhaustiva,
  órdenes de magnitud más rápido.

Mejoras heredadas de v3:
  - Memoria de postura activada (relocation_alpha=0.3)
  - Separación inteligente de manos (un solo track) minimizando cruces
  - Tempo variable: cálculo de compás correcto con múltiples cambios de tempo
  - Optimización de acordes en bloque con módulo específico (span real de mano)
  - Puntuación de confianza por nota (marca notas con digitación alternativa válida con '(alt)')
  - Salida estructurada como lista de dicts (API usable por otros módulos)
  - Separación de voces dentro de un track (notas largas vs corcheas)
  - Digitación de voz sostenida independiente de la melodía
  - Exportación a MusicXML con marcas de digitación estándar (opcional)

Uso:
    python fingering_v4.py archivo.mid
    python fingering_v4.py archivo.mid --chord-mode                      # λ=0.5
    python fingering_v4.py archivo.mid --chord-mode --chord-weight 0.8   # más uniforme
    python fingering_v4.py archivo.mid --chord-mode --chord-weight 0     # solo análisis armónico
    python fingering_v4.py archivo.mid --chord-mode --chord-keys shape   # solo claves de forma
    python fingering_v4.py archivo.mid --bass-prior 0.3                  # bajo de MI → dedo 5
    python fingering_v4.py archivo.mid --tie-gestures --tie-tol 0.5      # gestos repetidos → mismos dedos
    python fingering_v4.py archivo.mid --chord-report                    # solo diagnóstico (dedos = v3)
    python fingering_v4.py archivo.mid --tie-gestures --cost-alert 0.3   # avisa si un compás cuesta +30%
    python fingering_v4.py archivo.mid --hand-size L
    python fingering_v4.py archivo.mid --measures 8
    python fingering_v4.py archivo.mid --right-only
    python fingering_v4.py archivo.mid --left-only
    python fingering_v4.py archivo.mid --xml salida.xml
    python fingering_v4.py archivo.mid --json salida.json
    python fingering_v4.py archivo.mid --confidence      # muestra puntuación de confianza
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
    # ── modo acordes (v4) ────────────────────────────────────────────────────
    finger_agnostic: int = 0       # dedo de la pasada 0 (agnóstica, = v3)
    chord_name: str    = ""        # armonía del compás, p. ej. "G7"
    chord_quality: str = ""        # "maj", "min", "7"…
    chord_role: str    = ""        # "R", "3", "5", "b7"… o "x" (no es nota del acorde)
    slot: int          = 0         # orden de la nota suelta dentro de su compás
    ctx: tuple         = ()        # (rol de la nota previa, intervalo hasta ella)
    voicing: tuple     = ()        # roles del bloque (solo acordes en bloque)
    cons_pen: list     = field(default_factory=list)  # penalización por dedo 1..5
    shape: tuple       = ()        # (beat/12, intervalo previo|None, intervalo siguiente|None)
    shape_c: tuple     = ()        # versión gruesa de shape
    bshape: tuple      = ()        # intervalos sobre la nota grave (solo bloques)
    mshape: tuple      = ()        # forma del compás completo (posiciones + intervalos)
    mindex: int        = 0         # índice de la nota dentro de su compás (melodía)
    is_bass: bool      = False     # bajo del compás (MI, prior de bajo)
    prior_pen: list    = field(default_factory=list)  # penalización del prior por dedo 1..5
    tie_pen: list      = field(default_factory=list)  # dedo obligatorio (enlace de gestos)
    tied: bool         = False     # la nota pertenece a un gesto enlazado
    in_run: bool       = False     # el compás es una escalada estricta (≥3 notas)
    cost_pure: float   = 0.0       # coste de ventana SIN penalizaciones (solución final)
    cost_pure_agn: float = 0.0     # ídem en la pasada agnóstica (pasada 0)


@dataclass
class FingeringResult:
    """Resultado estructurado exportable como JSON o MusicXML."""
    midi_file: str
    bpm: float
    time_signature: str
    measures: int
    notes: list[dict] = field(default_factory=list)
    chords: list[dict] = field(default_factory=list)   # armonía por compás (modo acordes)


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

    def __init__(self, pm: pretty_midi.PrettyMIDI, beats_per_measure: float = 4.0,
                 ts: tuple[int, int] = (4, 4)) -> None:
        times, tempos = pm.get_tempo_changes()
        self.segments: list[tuple[float, float, float]] = []  # (t_start, t_end, bpm)
        self.bpm0 = float(tempos[0]) if len(tempos) > 0 else 120.0
        self.beats_per_measure = beats_per_measure   # en NEGRAS (6/8 → 3.0)
        self.ts_num, self.ts_den = ts

        for i, (t, bpm) in enumerate(zip(times, tempos)):
            t_end = times[i+1] if i+1 < len(times) else float("inf")
            self.segments.append((float(t), t_end, float(bpm)))

        if not self.segments:
            self.segments = [(0.0, float("inf"), 120.0)]

    @property
    def time_signature(self) -> str:
        return f"{self.ts_num}/{self.ts_den}"

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
    ts_num = ts[0].numerator   if ts else 4
    ts_den = ts[0].denominator if ts else 4
    # Los "beats" del TempoMap son negras (pretty_midi da BPM = negras/min):
    # un compás de num/den dura num·4/den negras (6/8 → 3, 3/2 → 6, 4/4 → 4).
    tmap = TempoMap(pm, beats_per_measure=ts_num * 4.0 / ts_den, ts=(ts_num, ts_den))
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
# Modo acordes (v4): armonía por compás + consistencia de digitación
# ─────────────────────────────────────────────────────────────────────────────
#
# Terminología (ojo, en este fichero conviven dos significados):
#   - "acorde en bloque": notas simultáneas (INote.isChord) — ya existía en v3.
#   - "acorde" en el modo acordes: la ARMONÍA del compás (Cmaj, G7, Am…),
#     deducida de todas las notas del compás, en ambas manos.
#
# Planteamiento:
#   1. Se calcula el acorde de cada compás (plantillas de clases de altura).
#   2. Cada nota recibe un ROL dentro de ese acorde (R, 3, 5, b7… o "x" si no
#      es nota del acorde) y un contexto de patrón (nota previa, posición en
#      el compás). Es lo que permite comparar C-E-G sobre C con F-A-C sobre F.
#   3. Pasada 0: digitación agnóstica (idéntica a v3).
#   4. Pasadas de refinamiento: para cada patrón se toma el dedo "consenso" y
#      se penaliza desviarse de él. El coste final es
#            (1-λ)·coste_ergonómico + λ·v_ref·penalización_de_consistencia
#      con λ = --chord-weight. λ=0 → v3; λ=1 → solo consistencia.
#
#   Claves de patrón (--chord-keys): "shape" (forma: posición en el compás + intervalos
#   a las notas vecinas; no depende de la armonía), "harmony" (calidad + rol en el
#   acorde) o "both" (defecto; la forma pesa 2-4 veces más que la armonía, y dentro de
#   la forma pesa más la clave más específica: el gesto completo del compás).
#
#   Escaladas: un compás cuyas notas suben (o bajan) estrictamente, ≥3 notas, no usa las
#   claves locales ("s", "sc") y su gesto completo ("sm") incluye las notas exactas: solo
#   se compara con compases con las mismas notas. --chord-runs lo desactiva.
#
#   Prior de bajo (--bass-prior B, solo mano izquierda): suma a la ventana
#   B·v_ref·penalización(dedo) en la nota más grave de cada compás cuando cae en el
#   tiempo 1 (5→0, 4→0.35, 3→0.65, 2→0.9, 1→1). Independiente de --chord-mode.
#
#   Enlace de gestos (--tie-gestures): tras las pasadas anteriores, los compases con
#   gesto idéntico (clave `mshape`) forman un grupo. Las candidatas son las digitaciones
#   que ya aparecen en el grupo (pasada agnóstica y refinada). Cada candidata se evalúa
#   en cada compás forzándola (coste de ventana desde el estado de mano guardado) y se
#   elige la que admiten más compases (los que la empeoran más de `tie_tol`, en relativo,
#   quedan excluidos del grupo) y, a igualdad, la de menor coste total.
#   El patrón elegido se fija con penalización enorme y se repite la pasada para que las
#   notas vecinas se adapten; si algún compás no puede seguirlo (p. ej. por la transición
#   con el compás vecino, que también está fijado) se prueba la siguiente candidata o se
#   desenlaza el grupo.

_CHORD_ROOT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# (calidad, sufijo, intervalos desde la fundamental, prior)
_CHORD_TEMPLATES: list[tuple[str, str, tuple[int, ...], float]] = [
    ("maj",  "",     (0, 4, 7),         0.00),
    ("min",  "m",    (0, 3, 7),         0.00),
    ("dim",  "dim",  (0, 3, 6),        -0.06),
    ("aug",  "aug",  (0, 4, 8),        -0.08),
    ("7",    "7",    (0, 4, 7, 10),    -0.02),
    ("maj7", "maj7", (0, 4, 7, 11),    -0.02),
    ("min7", "m7",   (0, 3, 7, 10),    -0.02),
    ("m7b5", "m7b5", (0, 3, 6, 10),    -0.06),
    ("dim7", "dim7", (0, 3, 6, 9),     -0.10),
]

_ROLE_BY_INTERVAL = {0: "R", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5",
                     8: "#5", 9: "bb7", 10: "b7", 11: "7"}


@dataclass
class ChordInfo:
    """Acorde (armonía) detectado para un compás."""
    measure: int
    root: int                      # clase de altura 0-11
    quality: str
    name: str
    intervals: tuple[int, ...]
    score: float
    margin: float                  # ventaja sobre la 2ª mejor hipótesis
    bass: int = -1                 # clase de altura de la nota más grave del compás

    def label(self) -> str:
        """Nombre con bajo si no es la fundamental (p. ej. "Fm/Ab")."""
        if self.bass < 0 or self.bass == self.root:
            return self.name
        return f"{self.name}/{_CHORD_ROOT_NAMES[self.bass]}"


def _score_chord(profile: list[float], root: int, intervals: tuple[int, ...],
                 prior: float, bass_pc: int) -> float:
    tones   = {(root + i) % 12 for i in intervals}
    inside  = sum(profile[t] for t in tones)
    missing = sum(1 for t in tones if profile[t] < 0.04)   # nota del acorde ausente
    score   = inside - 0.9 * (1.0 - inside) - 0.12 * missing + prior
    if bass_pc == root:
        score += 0.10                                       # el bajo suele ser la fundamental
    return score


def analyze_chords(seqs: Sequence[Sequence[INote]],
                   tmap: TempoMap) -> dict[int, ChordInfo]:
    """
    Devuelve {compás: ChordInfo}. Usa TODAS las notas (ambas manos, melodía y
    sostenidas). Cada nota pesa por su duración dentro del compás y por su
    posición métrica (tiempo fuerte > tiempo débil).
    """
    by_measure: dict[int, list[INote]] = defaultdict(list)
    for seq in seqs:
        for n in seq:
            by_measure[n.measure].append(n)

    bpm = tmap.beats_per_measure
    out: dict[int, ChordInfo] = {}

    for m, notes in sorted(by_measure.items()):
        profile = [0.0] * 12
        for n in notes:
            dur_b = min(n.duration / tmap.spb_at(n.time), bpm - n.beat)
            dur_b = max(dur_b, 0.1)
            if n.beat < 0.05:
                strong = 1.5
            elif abs(n.beat - round(n.beat)) < 0.05:
                strong = 1.2
            else:
                strong = 1.0
            profile[n.pitch % 12] += dur_b * strong

        total = sum(profile)
        if total <= 0:
            continue
        profile = [p / total for p in profile]
        bass_pc = min(notes, key=lambda n: n.pitch).pitch % 12

        scored: list[tuple[float, int, tuple[str, str, tuple[int, ...], float]]] = []
        for root in range(12):
            for tpl in _CHORD_TEMPLATES:
                scored.append((_score_chord(profile, root, tpl[2], tpl[3], bass_pc), root, tpl))
        scored.sort(key=lambda s: -s[0])

        best, root, (quality, suffix, intervals, _prior) = scored[0]
        margin = best - scored[1][0]
        out[m] = ChordInfo(measure=m, root=root, quality=quality,
                           name=_CHORD_ROOT_NAMES[root] + suffix,
                           intervals=intervals, score=round(best, 3),
                           margin=round(margin, 3), bass=bass_pc)
    return out


def annotate_chords(seq: list[INote], chords: dict[int, ChordInfo]) -> None:
    """
    Rellena en cada nota: chord_name, chord_quality, chord_role y, según el caso,
      - notas sueltas: slot (orden dentro del compás) y ctx (rol de la nota previa
        + intervalo hasta ella) → firma de patrón independiente del acorde concreto.
      - notas de acorde en bloque: voicing (roles de todas las notas del bloque).
    `seq` debe estar en orden temporal (así la construye _build_note_seq).
    """
    slot_counter: dict[int, int] = defaultdict(int)
    prev: INote | None = None
    groups: dict[int, list[INote]] = defaultdict(list)

    for n in seq:
        ci = chords.get(n.measure)
        if ci is None:
            continue
        n.chord_name    = ci.name
        n.chord_quality = ci.quality
        interval        = (n.pitch - ci.root) % 12
        n.chord_role    = _ROLE_BY_INTERVAL[interval] if interval in ci.intervals else "x"

        if n.isChord:
            groups[n.chordID].append(n)
            prev = None                       # un bloque rompe el contexto melódico
        else:
            n.slot = slot_counter[n.measure]
            slot_counter[n.measure] += 1
            if prev is not None and prev.measure == n.measure:
                n.ctx = (prev.chord_role, max(-12, min(12, n.pitch - prev.pitch)))
            else:
                n.ctx = ("|", 0)              # inicio de compás / tras un bloque
            prev = n

    for g in groups.values():
        g.sort(key=lambda x: x.chordnr)
        voicing = tuple(x.chord_role for x in g)
        for x in g:
            x.voicing = voicing


def _interval_class(d: int) -> int:
    """
    Clase gruesa de intervalo, con signo: 0 repetida, ±1 paso (≤2 semitonos),
    ±2 salto corto (3-4), ±3 salto (5-7), ±4 salto amplio (≥8).
    """
    a = abs(d)
    c = 0 if a == 0 else 1 if a <= 2 else 2 if a <= 4 else 3 if a <= 7 else 4
    return c if d >= 0 else -c


def _is_run(pitches: Sequence[int]) -> bool:
    """Escalada: ≥3 notas sucesivas estrictamente ascendentes o estrictamente descendentes."""
    if len(pitches) < 3:
        return False
    up   = all(b > a for a, b in zip(pitches, pitches[1:]))
    down = all(b < a for a, b in zip(pitches, pitches[1:]))
    return up or down


def annotate_shapes(seq: list[INote], shape_runs: bool = False) -> None:
    """
    Rellena la "forma" de cada nota, independiente de la armonía detectada:
      - notas sueltas: mshape  = forma del compás completo en esa mano: para cada nota
                                 (posición en 1/12 de tiempo, intervalo a la siguiente);
                       mindex  = índice de la nota en el compás  [gesto completo]
                       shape   = (posición en el compás en 1/12 de tiempo,
                                  intervalo desde la nota anterior del compás | None,
                                  intervalo hasta la siguiente del compás | None)
                       shape_c = (¿tiempo entero?, clase del intervalo previo,
                                  clase del intervalo siguiente)  [versión gruesa]
      - notas de acorde en bloque: bshape = intervalos de todas las notas del
        bloque sobre su nota más grave.
    El contexto se corta en las barras de compás (None), de modo que un mismo gesto
    en compases distintos tiene la misma forma aunque venga de notas distintas.
    Escaladas (≥3 notas estrictamente ascendentes/descendentes): salvo `shape_runs`,
    no llevan shape/shape_c y su mshape incluye las notas exactas, así que solo se
    comparan con compases con las mismas notas.
    `seq` debe estar en orden temporal.
    """
    by_measure: dict[int, list[INote]] = defaultdict(list)
    blocks: dict[int, list[INote]] = defaultdict(list)
    for n in seq:
        if n.isChord:
            blocks[n.chordID].append(n)
        else:
            by_measure[n.measure].append(n)

    for notes in by_measure.values():
        msh = tuple((int(round(m.beat * 12)),
                     (notes[j + 1].pitch - m.pitch) if j + 1 < len(notes) else None)
                    for j, m in enumerate(notes))
        run = (not shape_runs) and _is_run([m.pitch for m in notes])
        if run:
            msh = msh + (("notas", tuple(m.pitch for m in notes)),)
        for i, n in enumerate(notes):
            n.mshape, n.mindex = msh, i
            n.in_run = run
            if run:
                n.shape, n.shape_c = (), ()
                continue
            pi = (n.pitch - notes[i - 1].pitch) if i > 0 else None
            ni = (notes[i + 1].pitch - n.pitch) if i + 1 < len(notes) else None
            b12 = int(round(n.beat * 12))
            n.shape   = (b12, pi, ni)
            n.shape_c = (b12 % 12 == 0,
                         _interval_class(pi) if pi is not None else None,
                         _interval_class(ni) if ni is not None else None)

    for g in blocks.values():
        low = min(x.pitch for x in g)
        bshape = tuple(sorted(x.pitch - low for x in g))
        for x in g:
            x.bshape = bshape


class ChordConsistency:
    """
    Modelo de consistencia: tabla patrón → {dedo: votos}.

    Dos familias de claves (se eligen con `mode`; peso de cada clave entre paréntesis):

      Forma — no dependen de la armonía: gestos idénticos → mismos dedos.
      Más específica = más peso (la más específica evita coincidencias espurias):
        ("sm", mshape, índice)   gesto completo del compás (posiciones + todos los
                                 intervalos) + índice de la nota                (1.0)
        ("s",  shape)            posición en el compás + intervalos a los dos
                                 vecinos inmediatos                              (0.5)
        ("sc", shape_c)          ídem en versión gruesa (clases de intervalo)   (0.25)
        ("bs", bshape, índice)   nota de un bloque + intervalos sobre su grave   (1.0)

      Armonía — dependen del acorde detectado por compás
        ("m1", calidad, rol, ctx)   nota suelta + su contexto melódico inmediato
        ("m2", calidad, rol, slot)  nota suelta + su orden dentro del compás
        ("b",  voicing, índice)     nota de un bloque + disposición del bloque
                                    (1.0 en modo "harmony"; 0.25 en modo "both")

    La penalización de una nota para el dedo f es 1 - p(f), con p(f) la fracción de
    votos (de OTRAS notas con la misma clave) que usaron f, suavizada con `prior`.
    Las filas de varias claves se promedian ponderadas (ver `prepare` para la
    atenuación cuando falta la clave específica) y se resta el mínimo de la fila:
    el dedo mayoritario cuesta 0.
    """

    MODES = ("harmony", "shape", "both")

    def __init__(self, weight: float, mode: str = "both", min_obs: float = 1.0,
                 prior: float = 1.0, chord_scale: float = 10.0) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode debe ser uno de {self.MODES}")
        self.weight      = weight
        self.mode        = mode
        self.min_obs     = min_obs       # votos ajenos mínimos para que una clave cuente
        self.prior       = prior         # suavizado: pocas observaciones → tirón débil
        self.chord_scale = chord_scale   # escala de la penalización en optimize_chord
        self.v_ref       = 1.0           # coste ergonómico típico (se fija tras la pasada 0)
        self.active      = False
        self.table: dict[tuple, dict[int, float]] = {}

    @staticmethod
    def keys(n: INote, mode: str = "both") -> list[tuple[tuple, float]]:
        """Lista de (clave, peso) de una nota según el modo."""
        out: list[tuple[tuple, float]] = []
        if mode in ("shape", "both"):
            if n.isChord:
                if n.bshape:
                    out.append((("bs", n.bshape, n.chordnr), 1.0))
            elif n.mshape:
                out.append((("sm", n.mshape, n.mindex), 1.0))
                if n.shape:                       # las escaladas no tienen claves locales
                    out.append((("s", n.shape), 0.5))
                    out.append((("sc", n.shape_c), 0.25))
        if mode in ("harmony", "both") and n.chord_name:
            w = 1.0 if mode == "harmony" else 0.25
            if n.isChord:
                if n.voicing:
                    out.append((("b", n.voicing, n.chordnr), w))
            elif n.chord_role != "x":    # nota de paso/adorno: sin restricción armónica
                out.append((("m1", n.chord_quality, n.chord_role, n.ctx), w))
                out.append((("m2", n.chord_quality, n.chord_role, n.slot), w))
        return out

    @staticmethod
    def _vote(n: INote) -> float:
        """Las notas con digitación más segura votan más fuerte."""
        return 0.25 + 0.75 * max(0.0, min(1.0, n.confidence))

    def rebuild(self, notes: Sequence[INote]) -> None:
        self.table = defaultdict(lambda: defaultdict(float))
        for n in notes:
            if n.fingering <= 0:
                continue
            w = self._vote(n)
            for k, _kw in self.keys(n, self.mode):
                self.table[k][n.fingering] += w

    def prepare(self, notes: Sequence[INote]) -> None:
        """Calcula n.cons_pen = [_, pen(dedo1) … pen(dedo5)] para cada nota."""
        for n in notes:
            row = [0.0] * 6
            w_self = self._vote(n) if n.fingering > 0 else 0.0
            rows: list[tuple[float, list[float]]] = []
            keys = self.keys(n, self.mode)
            for k, kw in keys:
                cnt = self.table.get(k)
                if not cnt:
                    continue
                total = sum(cnt.values()) - w_self           # leave-one-out
                if total < self.min_obs:
                    continue
                r = [0.0] * 6
                for f in range(1, 6):
                    c = cnt.get(f, 0.0) - (w_self if f == n.fingering else 0.0)
                    r[f] = 1.0 - max(0.0, c) / (total + self.prior)
                rows.append((kw, r))
            if rows:
                # "harmony": se promedia solo entre claves con datos (como la v4 original).
                # "shape"/"both": se divide entre el peso de TODAS las claves de la nota;
                # si falta la clave específica (el gesto completo), el tirón de las claves
                # locales se atenúa en vez de aplicarse a pleno (evita coincidencias espurias).
                denom = (sum(kw for kw, _ in rows) if self.mode == "harmony"
                         else sum(kw for _, kw in keys))
                for f in range(1, 6):
                    row[f] = sum(kw * r[f] for kw, r in rows) / denom
                lo = min(row[1:])
                for f in range(1, 6):
                    row[f] -= lo
            n.cons_pen = row
        self.active = True

    @classmethod
    def agreement(cls, notes: Sequence[INote], mode: str = "both",
                  min_others: int = 2) -> tuple[float, int]:
        """
        Métrica de diagnóstico: fracción de pares (nota, clave) cuyo dedo coincide
        con el mayoritario entre las OTRAS notas de esa clave (≥ `min_others`). Devuelve
        (fracción, nº de pares evaluados). Con mode="shape" mide si los gestos
        idénticos reciben los mismos dedos.
        """
        table: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for n in notes:
            if n.fingering > 0:
                for k, _kw in cls.keys(n, mode):
                    table[k][n.fingering] += 1
        agree = tot = 0
        for n in notes:
            if n.fingering <= 0:
                continue
            for k, _kw in cls.keys(n, mode):
                others = dict(table[k])
                others[n.fingering] -= 1
                if sum(others.values()) < min_others:
                    continue
                tot += 1
                if others.get(n.fingering, 0) == max(others.values()):
                    agree += 1
        return (agree / tot if tot else 1.0), tot


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


def optimize_chord(notes: list[INote], hf: float, side: str,
                   next_notes: list[INote] | None = None,
                   start_finger: int = 0,
                   cons: "ChordConsistency | None" = None) -> list[int]:
    """
    Asigna dedos óptimos a un grupo de notas de acorde.

    Considera tanto el coste intrínseco del acorde como el coste de
    transición hacia el acorde siguiente (si se proporciona). Con `cons` activo
    mezcla además la penalización de consistencia de patrón (peso λ).
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
    use_cons     = (cons is not None and cons.active
                    and all(len(nt.cons_pen) == 6 for nt in notes))

    for finger_combo in combinations(range(1, 6), n):
        if side == "right":
            fingers = sorted(finger_combo)
        else:
            fingers = sorted(finger_combo, reverse=True)

        if start_finger and start_finger not in fingers:
            continue

        c = _fingering_cost_chord(fingers, pitches, hf, side)

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

        if use_cons:
            pen = sum(notes[k].cons_pen[fingers[k]] for k in range(n)) / n
            c = (1.0 - cons.weight) * c + cons.weight * cons.chord_scale * pen

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
                 size: str = "M", chord_weight: float = 0.0,
                 chord_iters: int = 2, chord_keys: str = "both",
                 bass_prior: float = 0.0, tie_tol: float | None = None) -> None:
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

        # Modo acordes: λ=0 → sin modelo (v3 exacto)
        self.chord_iters = max(1, chord_iters)
        self.cons: ChordConsistency | None = (
            ChordConsistency(chord_weight, mode=chord_keys) if chord_weight > 0 else None)

        # Prior de bajo (solo mano izquierda): 0 → desactivado
        self.bass_prior = bass_prior if side == "left" else 0.0
        self._prior_active = False

        # Enlace de gestos repetidos: None → desactivado
        self.tie_tol = tie_tol
        self._tie_active = False
        self._tie_big = 1e3
        self._record_snaps = False
        self._snaps: dict[int, tuple] = {}
        self.tie_report: list[dict] = []

        # Coste ergonómico de la mejor ventana SIN penalizaciones
        self._pure_last = 0.0
        self._last_pure = 0.0

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
        vmean /= max(1, self.depth - 1)
        self._pure_last = vmean            # coste ergonómico puro de este candidato

        cons = self.cons
        if cons is not None and cons.active:
            pen = 0.0
            for i in range(self.depth):
                row = notes[i].cons_pen
                if row:
                    pen += row[fingering[i]]
            pen /= self.depth
            base = (1.0 - cons.weight) * vmean + cons.weight * cons.v_ref * pen
        else:
            base = vmean

        if self._prior_active:
            extra = 0.0
            for i in range(self.depth):
                row = notes[i].prior_pen
                # las notas de relleno de la cola (duplicados) no cuentan dos veces
                if row and not (i > 0 and notes[i] is notes[i - 1]):
                    extra += row[fingering[i]]
            base += extra

        if self._tie_active:
            for i in range(self.depth):
                row = notes[i].tie_pen
                if row and not (i > 0 and notes[i] is notes[i - 1]):
                    base += row[fingering[i]]
        return base

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
        pure_best = 0.0   # coste ergonómico (sin penalizaciones) del mejor candidato

        # ── Ramificación y poda con coste incremental ────────────────────────
        # Todos los términos del coste son >= 0, así que la suma parcial de un
        # prefijo es una cota inferior del coste final de cualquier extensión.
        # Un prefijo con cota > segundo mejor no puede cambiar ni el mejor ni
        # el segundo mejor → se descarta (resultado idéntico al backtracking
        # exhaustivo, que evaluaba todo en las hojas).
        cons     = self.cons
        use_cons = cons is not None and cons.active
        w_c      = cons.weight if use_cons else 0.0
        vref_c   = cons.v_ref if use_cons else 0.0
        use_pri  = self._prior_active
        use_tie  = self._tie_active
        dm1      = max(1, depth - 1)
        can_bound = (0.0 <= w_c <= 1.0) and self.bass_prior >= 0.0
        a_v  = (1.0 - w_c) / dm1 if use_cons else 1.0 / dm1
        a_p  = (w_c * vref_c / depth) if use_cons else 0.0
        weights, bfactor = self.weights, self.bfactor
        set_fp   = self.set_fingers_positions
        skip     = self.skip
        SLACK_R, SLACK_A = 1e-9, 1e-12

        def leaf_cost(vsum: float) -> tuple[float, float]:
            """Coste exacto (misma fórmula y orden de sumas que ave_velocity)."""
            vmean = vsum / dm1
            base = vmean
            if use_cons:
                pen = 0.0
                for i in range(depth):
                    row = nseq[i].cons_pen
                    if row:
                        pen += row[cand[i]]
                pen /= depth
                base = (1.0 - w_c) * vmean + w_c * vref_c * pen
            if use_pri:
                extra = 0.0
                for i in range(depth):
                    row = nseq[i].prior_pen
                    if row and not (i > 0 and nseq[i] is nseq[i - 1]):
                        extra += row[cand[i]]
                base += extra
            if use_tie:
                for i in range(depth):
                    row = nseq[i].tie_pen
                    if row and not (i > 0 and nseq[i] is nseq[i - 1]):
                        base += row[cand[i]]
            return base, vmean

        def bt(level: int, fp: list, vsum: float, pen: float,
               extra: float) -> None:
            nonlocal best, minv, second_v, pure_best
            nb = nseq[level]
            choices = u_start if level == 0 else self.fingers
            na = nseq[level - 1] if level else None
            for f in choices:
                if level > 0 and skip(cand[level-1], f, na, nb):
                    continue
                cand[level] = f
                if level == 0:
                    vs = 0.0
                    upd = True
                else:
                    pos = fp[f]
                    if pos is None:
                        vs, upd = vsum, False       # (igual que ave_velocity: no actualiza)
                    else:
                        dx = abs(nb.x - pos)
                        dt = abs(nb.time - na.time) + 0.1
                        w  = weights[f] or 1.0
                        bf = (bfactor[f] or 1.0) if nb.isBlack else 1.0
                        vs = vsum + (dx / dt) / (w * bf)
                        upd = True
                # penalizaciones acumuladas (cotas; el valor exacto se recalcula en la hoja)
                p2, e2 = pen, extra
                if can_bound:
                    if use_cons:
                        row = nb.cons_pen
                        if row:
                            p2 += row[f]
                    if use_pri or use_tie:
                        if not (level > 0 and nb is na):
                            if use_pri:
                                row = nb.prior_pen
                                if row:
                                    e2 += row[f]
                            if use_tie:
                                row = nb.tie_pen
                                if row:
                                    e2 += row[f]
                    lb = a_v * vs + a_p * p2 + e2
                    if lb > second_v * (1.0 + SLACK_R) + SLACK_A:
                        continue
                if level + 1 == depth:
                    v, pure = leaf_cost(vs)
                    if v < minv:
                        second_v = minv
                        best[:]  = cand[:]
                        minv     = v
                        pure_best = pure
                    elif v < second_v:
                        second_v = v
                else:
                    # la postura de la mano solo hace falta si se sigue descendiendo
                    if upd:
                        fp2 = list(fp)
                        set_fp(cand, nseq, level, fp=fp2, force_relaxed=False)
                    else:
                        fp2 = fp
                    bt(level + 1, fp2, vs, p2, e2)

        bt(0, list(self.finger_positions), 0.0, 0.0, 0.0)
        self._last_pure = pure_best

        # Confianza: 1.0 si el gap entre 1º y 2º es grande; 0.0 si son iguales
        if second_v >= 1e9 or minv < 1e-9:
            confidence = 1.0
        else:
            gap = (second_v - minv) / (minv + 1e-9)
            confidence = min(1.0, gap / self.confidence_threshold)

        return best, minv, confidence

    # ── generación completa ───────────────────────────────────────────────────

    # ── pasadas de optimización ───────────────────────────────────────────────

    def _chords_pass(self, chord_groups: dict[int, list[INote]]) -> None:
        """Digitación de los acordes en bloque (notas simultáneas)."""
        chord_ids_ordered = sorted(chord_groups.keys())
        for idx, chord_id in enumerate(chord_ids_ordered):
            cnotes_sorted = sorted(chord_groups[chord_id], key=lambda n: n.pitch)

            next_cnotes: list[INote] | None = None
            if idx + 1 < len(chord_ids_ordered):
                next_id = chord_ids_ordered[idx + 1]
                next_cnotes = sorted(chord_groups[next_id], key=lambda n: n.pitch)

            fingers = optimize_chord(cnotes_sorted, self.hf, self.LR,
                                     next_notes=next_cnotes, cons=self.cons)
            for note, f in zip(cnotes_sorted, fingers):
                note.fingering  = f
                note.confidence = 0.9

    def _melody_pass(self, melody_notes: list[INote]) -> None:
        """Digitación de la melodía con ventana deslizante (lógica de v3)."""
        init_autodepth = self.autodepth
        init_depth     = self.depth

        self.fingerseq = []
        self.finger_positions    = list(self.frest)
        self._has_position_state = False

        start_finger = 0
        n_total      = len(melody_notes)

        try:
            for i in range(n_total):
                an = melody_notes[i]
                if self._record_snaps:
                    self._snaps[i] = (list(self.finger_positions), self._has_position_state,
                                      start_finger, self.autodepth, self.depth)

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
                an.cost       = round(vel, 4)
                an.cost_pure  = round(self._last_pure, 4)
                self.set_fingers_positions(out, window, 0)
                self.fingerseq.append(list(self.finger_positions))
        finally:
            self.autodepth = init_autodepth
            self.depth     = init_depth

    def _repair_zeros(self) -> None:
        """Reparar ceros residuales (todas las rutas podadas)."""
        last = 1
        for n in self.noteseq:
            if n.fingering == 0:
                n.fingering = last
            else:
                last = n.fingering

    # ── enlace de gestos repetidos ────────────────────────────────────────────

    def _tie_row(self, finger: int) -> list[float]:
        """Fila "dedo obligatorio": 0 para `finger`, _tie_big para los demás."""
        return [0.0] + [0.0 if f == finger else self._tie_big for f in range(1, 6)]

    def _eval_pattern(self, melody_notes: list[INote], i0: int,
                      pattern: tuple[int, ...]) -> tuple[float, bool]:
        """
        Coste acumulado (suma de costes de ventana) de forzar `pattern` en
        melody_notes[i0 : i0+len(pattern)], partiendo del estado de mano guardado antes
        de la nota i0. No modifica la solución. Devuelve (coste, factible).
        """
        snap = self._snaps.get(i0)
        if snap is None:
            return float("inf"), False

        saved_state  = (self.finger_positions, self._has_position_state,
                        self.autodepth, self.depth)
        saved_rows   = [melody_notes[i0 + j].tie_pen for j in range(len(pattern))]
        saved_tie    = self._tie_active
        cons         = self.cons
        saved_cons   = cons.active if cons is not None else False

        self.finger_positions    = list(snap[0])
        self._has_position_state = snap[1]
        start_finger             = snap[2]
        self.autodepth, self.depth = snap[3], snap[4]
        for j, f in enumerate(pattern):
            melody_notes[i0 + j].tie_pen = self._tie_row(f)
        self._tie_active = True
        if cons is not None:
            cons.active = False          # coste ergonómico (+prior), sin tirón del consenso

        n_total = len(melody_notes)
        total, ok = 0.0, True
        try:
            for j in range(len(pattern)):
                i = i0 + j
                remaining = n_total - i
                if remaining < 9:
                    if self.autodepth:
                        self.autodepth = False
                    self.depth = max(2, remaining)
                window = list(melody_notes[i : i + 9])
                if window and len(window) < 9:
                    window += [window[-1]] * (9 - len(window))
                # 1ª nota: el dedo lo impone el patrón candidato (no el plan de la ventana previa)
                out, vel, _conf = self.optimize_seq(window, pattern[0] if j == 0 else start_finger)
                if vel >= 0.5 * self._tie_big or out[0] != pattern[j]:
                    ok = False
                    break
                total += vel
                start_finger = out[1] if len(out) > 1 else out[0]
                self.set_fingers_positions(out, window, 0)
        finally:
            for j, row in enumerate(saved_rows):
                melody_notes[i0 + j].tie_pen = row
            (self.finger_positions, self._has_position_state,
             self.autodepth, self.depth) = saved_state
            self._tie_active = saved_tie
            if cons is not None:
                cons.active = saved_cons
        return total, ok

    @staticmethod
    def _find_gestures(melody_notes: list[INote]) -> list[list[tuple[int, int]]]:
        """
        Grupos de compases con el mismo gesto (misma `mshape`, ≥2 notas, ≥2 compases).
        Cada grupo es una lista de (índice de la 1ª nota en melody_notes, nº de notas).
        """
        groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        n, i = len(melody_notes), 0
        while i < n:
            j = i
            while j + 1 < n and melody_notes[j + 1].measure == melody_notes[i].measure:
                j += 1
            L = j - i + 1
            if L >= 2 and melody_notes[i].mshape:
                groups[melody_notes[i].mshape].append((i, L))
            i = j + 1
        return [g for g in groups.values() if len(g) >= 2]

    def _tie_gestures(self, melody_notes: list[INote], v_ref: float) -> None:
        """Enlaza la digitación de los gestos repetidos (ver comentario de cabecera)."""
        self.tie_report = []
        self._tie_big = 1e3 * max(v_ref, 1e-6)

        # Estado de la mano antes de cada nota, tomado de la solución actual
        self._record_snaps = True
        self._snaps = {}
        try:
            self._melody_pass(melody_notes)
        finally:
            self._record_snaps = False

        states: list[dict] = []
        for inst in self._find_gestures(melody_notes):
            L = inst[0][1]
            now = [tuple(melody_notes[i0 + j].fingering for j in range(L)) for i0, _ in inst]
            agn = [tuple(melody_notes[i0 + j].finger_agnostic for j in range(L)) for i0, _ in inst]
            counts: dict[tuple[int, ...], int] = {}
            for p in now + agn:
                if 0 not in p:
                    counts[p] = counts.get(p, 0) + 1
            cands = sorted(counts, key=lambda p: -counts[p])[:4]

            cache: dict[tuple, tuple[float, bool]] = {}

            def cost(p: tuple[int, ...], k: int, _c=cache, _i=inst) -> tuple[float, bool]:
                if (p, k) not in _c:
                    _c[(p, k)] = self._eval_pattern(melody_notes, _i[k][0], p)
                return _c[(p, k)]

            own = [cost(now[k], k) for k in range(len(inst))]

            def rel(c: tuple[int, ...], k: int, _own=own) -> float:
                """Aumento relativo de coste de c respecto al patrón propio (inf si infactible)."""
                t, ok = cost(c, k)
                if not ok:
                    return float("inf")
                o = _own[k][0] if _own[k][1] else t
                return (t - o) / max(o, 1e-9)

            def valid(c: tuple[int, ...], k: int, _own=own, _L=L) -> bool:
                t, ok = cost(c, k)
                if not ok:
                    return False
                o = _own[k][0] if _own[k][1] else t
                return t - o <= self.tie_tol * o + 0.02 * v_ref * _L

            # Para cada candidata, subconjunto de compases donde es válida. Se prefiere la que
            # cubre más compases (el resto queda excluido), luego menor coste conjunto.
            options: list[tuple[int, float, int, tuple[int, ...], list[int]]] = []
            for c in cands:
                S = [k for k in range(len(inst)) if valid(c, k)]
                if len(S) >= 2:
                    options.append((len(S), sum(cost(c, k)[0] for k in S),
                                    sum(1 for k in S if now[k] != c), c, S))
            options.sort(key=lambda r: (-r[0], r[1], r[2]))

            entry = {"hand": self.LR,
                     "measures": [melody_notes[i0].measure for i0, _ in inst],
                     "before": ["".join(map(str, p)) for p in now],
                     "pattern": None, "tied": False, "reason": ""}
            if options:
                states.append({"inst": inst, "options": options,
                               "idx": 0, "entry": entry, "now": now})
            else:
                if cands:
                    best = min(max(rel(c, k) for k in range(len(inst))) for c in cands)
                    entry["reason"] = ("ningún candidato es factible en todos los compases"
                                       if best == float("inf") else
                                       f"el mejor patrón común empeora un compás un "
                                       f"{100 * best:.0f}% (tolerancia {100 * self.tie_tol:.0f}%)")
                else:
                    entry["reason"] = "sin patrones candidatos"
            self.tie_report.append(entry)

        if not states:
            return

        conf0: dict[int, float] = {}
        for s in states:
            for i0, L in s["inst"]:
                for j in range(L):
                    conf0[id(melody_notes[i0 + j])] = melody_notes[i0 + j].confidence

        def apply_ties(live: list[dict]) -> None:
            for s in states:
                for i0, L in s["inst"]:
                    for j in range(L):
                        n = melody_notes[i0 + j]
                        n.tie_pen, n.tied = [], False
            for s in live:
                _sz, _tot, _nd, pat, S = s["options"][s["idx"]]
                for k in S:
                    i0, L = s["inst"][k]
                    for j in range(L):
                        n = melody_notes[i0 + j]
                        n.tie_pen = self._tie_row(pat[j])
                        n.tied    = True

        def follows(s: dict) -> bool:
            _sz, _tot, _nd, pat, S = s["options"][s["idx"]]
            return all(melody_notes[s["inst"][k][0] + j].fingering == pat[j]
                       for k in S for j in range(s["inst"][k][1]))

        # Pasadas de verificación: si un grupo no puede seguir su patrón (p. ej. porque
        # la transición con el compás vecino, también fijado, es imposible), se prueba
        # la siguiente candidata; agotadas, el grupo queda sin enlazar.
        for _round in range(8):
            live = [s for s in states if s["idx"] < len(s["options"])]
            apply_ties(live)
            self._tie_active = bool(live)
            self._melody_pass(melody_notes)
            self._repair_zeros()
            failing = [s for s in live if not follows(s)]
            if not failing:
                break
            for s in failing:
                s["idx"] += 1

        for s in states:
            e, inst = s["entry"], s["inst"]
            if s["idx"] < len(s["options"]):
                _sz, _tot, _nd, pat, S = s["options"][s["idx"]]
                e["tied"]    = True
                e["pattern"] = "".join(map(str, pat))
                e["reason"]  = ("ya coincidían" if all(s["now"][k] == pat for k in S)
                                else "enlazado")
                if len(S) < len(inst):
                    out = [melody_notes[inst[k][0]].measure for k in range(len(inst))
                           if k not in S]
                    e["reason"] += "; excluye " + ",".join(f"c{m}" for m in out)
                if s["idx"] > 0:
                    e["reason"] += f"; {s['idx']} candidata(s) descartada(s) por no encajar con compases vecinos"
            else:
                e["reason"] = ("los patrones comunes no encajan con los compases vecinos "
                               "(transición imposible)")
            for i0, L in inst:
                for j in range(L):
                    n = melody_notes[i0 + j]
                    if n.tied:
                        n.confidence = conf0[id(n)]    # la confianza "forzada" no es informativa

    # penalización relativa por dedo (índices 1..5) para el bajo de la mano izquierda
    _BASS_ROW = [0.0, 1.0, 0.9, 0.65, 0.35, 0.0]

    def _mark_bass(self, melody_notes: list[INote]) -> None:
        """Bajo = nota más grave de la melodía del compás, si cae en el tiempo 1."""
        by_m: dict[int, list[INote]] = defaultdict(list)
        for n in melody_notes:
            by_m[n.measure].append(n)
        for notes in by_m.values():
            low = min(n.pitch for n in notes)
            for n in notes:
                n.is_bass = (n.pitch == low and n.beat < 0.05)

    def _prepare_prior(self, melody_notes: list[INote], v_ref: float) -> None:
        """Calcula prior_pen en las notas de bajo (escala: fracción del coste típico)."""
        self._mark_bass(melody_notes)
        for n in melody_notes:
            if n.is_bass:
                n.prior_pen = [0.0] + [self.bass_prior * v_ref * self._BASS_ROW[f]
                                       for f in range(1, 6)]
            else:
                n.prior_pen = []
        self._prior_active = True

    def generate(self) -> None:
        """
        Asigna fingering y confidence a cada nota en self.noteseq.

        Pasada 0: agnóstica a la armonía (idéntica a v3). Se guarda en
        INote.finger_agnostic.
        Si hay modelo de consistencia (chord_weight > 0): hasta `chord_iters`
        pasadas de refinamiento, reconstruyendo cada vez la tabla de dedos
        consenso por patrón a partir del resultado anterior.
        """
        original_x: list[float] | None = None

        if self.LR == "left":
            original_x = [n.x for n in self.noteseq]
            for n in self.noteseq:
                n.x = -n.x

        # Separar melodía y acordes en bloque
        chord_groups: dict[int, list[INote]] = defaultdict(list)
        melody_notes: list[INote] = []
        for n in self.noteseq:
            if n.isChord:
                chord_groups[n.chordID].append(n)
            else:
                melody_notes.append(n)

        try:
            # ── Pasada 0: agnóstica ───────────────────────────────────────────
            self._chords_pass(chord_groups)
            self._melody_pass(melody_notes)
            self._repair_zeros()
            for n in self.noteseq:
                n.finger_agnostic = n.fingering
                n.cost_pure_agn   = n.cost_pure

            # ── Refinamiento por consistencia y/o prior de bajo ───────────────
            costs = [n.cost for n in melody_notes if n.cost > 0]
            v_ref = (sum(costs) / len(costs)) if costs else 1.0
            use_cons  = self.cons is not None and self.cons.weight > 0
            use_prior = self.bass_prior > 0
            if use_cons or use_prior:
                if use_cons:
                    self.cons.v_ref = v_ref
                if use_prior:
                    self._prepare_prior(melody_notes, v_ref)

                for _ in range(self.chord_iters):
                    before = [n.fingering for n in self.noteseq]
                    if use_cons:
                        self.cons.rebuild(self.noteseq)
                        self.cons.prepare(self.noteseq)
                    self._chords_pass(chord_groups)
                    self._melody_pass(melody_notes)
                    self._repair_zeros()
                    # sin consenso que reconstruir, una pasada basta (el prior es fijo)
                    if not use_cons or [n.fingering for n in self.noteseq] == before:
                        break                     # convergido

            # ── Enlace de gestos repetidos ─────────────────────────────────────
            if self.tie_tol is not None:
                self._tie_gestures(melody_notes, v_ref)
        finally:
            if original_x is not None:
                for n, x in zip(self.noteseq, original_x):
                    n.x = x


# ─────────────────────────────────────────────────────────────────────────────
# Salida estructurada
# ─────────────────────────────────────────────────────────────────────────────

_FINGER_NAMES = {1:"pulgar", 2:"índice", 3:"corazón", 4:"anular", 5:"meñique"}


def build_result(midi_path: str, tmap: TempoMap,
                 rh_seq: list[INote], lh_seq: list[INote],
                 rh_sus: list[INote] | None = None,
                 lh_sus: list[INote] | None = None,
                 chords: dict[int, ChordInfo] | None = None) -> FingeringResult:
    """Construye el resultado estructurado como FingeringResult."""
    all_notes = rh_seq + lh_seq + (rh_sus or []) + (lh_sus or [])
    all_measures = set(n.measure for n in all_notes)

    notes_out = []
    for n in sorted(all_notes, key=lambda n: (n.measure, n.beat, n.hand, n.pitch)):
        entry = {
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
        }
        if n.chord_name:
            entry["chord"]              = n.chord_name
            entry["chord_role"]         = n.chord_role
            entry["fingering_agnostic"] = n.finger_agnostic
        if n.is_bass:
            entry["bass"] = True
            entry["fingering_agnostic"] = n.finger_agnostic
        if n.tied:
            entry["tied"] = True
            entry["fingering_agnostic"] = n.finger_agnostic
        if (n.chord_name or n.is_bass or n.tied) and not n.isChord:
            entry["cost_pure"]          = n.cost_pure
            entry["cost_pure_agnostic"] = n.cost_pure_agn
        notes_out.append(entry)

    return FingeringResult(
        midi_file=midi_path,
        bpm=round(tmap.bpm0, 1),
        time_signature=tmap.time_signature,
        measures=max(all_measures) if all_measures else 0,
        notes=notes_out,
        chords=[{"measure": m, "chord": c.name, "label": c.label(),
                 "root": _CHORD_ROOT_NAMES[c.root], "quality": c.quality,
                 "bass": _CHORD_ROOT_NAMES[c.bass] if c.bass >= 0 else None,
                 "margin": c.margin}
                for m, c in sorted((chords or {}).items())],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Informe (--chord-report)
# ─────────────────────────────────────────────────────────────────────────────

def _gesture_groups(seq: Sequence[INote],
                    labels: dict[int, str] | None = None) -> list[dict]:
    """
    Grupos de compases con el mismo gesto en una mano (≥2 compases):
      - melodía: mismo `mshape` (posiciones + intervalos; escaladas solo con las
        mismas notas) → clave ("m", mshape)
      - bloques: misma secuencia de (posición, disposición) de acordes en bloque
        → clave ("b", firma)
    Cada grupo: {"kind", "exact", "items": [(compás, acorde, notas|None, dedos)]}.
    """
    lab = labels or {}
    melody: dict[int, list[INote]] = defaultdict(list)
    blocks: dict[int, dict[int, list[INote]]] = defaultdict(lambda: defaultdict(list))
    for n in seq:
        if n.isChord:
            blocks[n.measure][n.chordID].append(n)
        else:
            melody[n.measure].append(n)

    raw: dict[tuple, list[tuple]] = defaultdict(list)
    for m, ns in melody.items():
        if len(ns) >= 2 and ns[0].mshape:
            fing = "-".join(str(n.fingering) for n in ns)
            raw[("m", ns[0].mshape)].append(
                (m, lab.get(m, ns[0].chord_name), tuple(n.pitch for n in ns),
                 " ".join(n.name for n in ns), fing, all(n.tied for n in ns)))
    for m, chs in blocks.items():
        evs = sorted(chs.values(), key=lambda g: min(x.time for x in g))
        if all(g[0].bshape for g in evs):
            sig = tuple((int(round(min(x.beat for x in g) * 12)), g[0].bshape) for g in evs)
            fing = " ".join("-".join(str(x.fingering) for x in sorted(g, key=lambda x: x.pitch))
                            for g in evs)
            pitches = tuple(tuple(sorted(x.pitch for x in g)) for g in evs)
            names = " | ".join(" ".join(x.name for x in sorted(g, key=lambda x: x.pitch))
                               for g in evs)
            raw[("b", sig)].append((m, lab.get(m, evs[0][0].chord_name), pitches, names, fing,
                                    all(x.tied for g in evs for x in g)))

    out = []
    for (kind, _sig), items in raw.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda it: it[0])
        out.append({"kind": kind, "exact": len({it[2] for it in items}) == 1, "items": items})
    out.sort(key=lambda g: g["items"][0][0])
    return out


def _cost_by_measure(seq: Sequence[INote], alert: float):
    """
    Coste ergonómico de ventana (sin penalizaciones) por compás: media de la solución
    final frente a la agnóstica. rel = (final-agnóstica)/max(agnóstica, 0.1·v_ref); el
    suelo evita porcentajes enormes cuando el coste agnóstico es casi nulo.
    Devuelve ({compás: datos}, v_ref, variación relativa total).
    """
    notes = [n for n in seq if not n.isChord]
    if not notes:
        return {}, 1.0, 0.0
    v_ref = (sum(n.cost_pure_agn for n in notes) / len(notes)) or 1.0
    by: dict[int, list[INote]] = defaultdict(list)
    for n in notes:
        by[n.measure].append(n)
    rows: dict[int, dict] = {}
    for m, ns in by.items():
        agn = sum(n.cost_pure_agn for n in ns) / len(ns)
        fin = sum(n.cost_pure for n in ns) / len(ns)
        changed = sum(1 for n in ns if n.finger_agnostic and n.fingering != n.finger_agnostic)
        rel = (fin - agn) / max(agn, 0.1 * v_ref)
        # aviso: aumento relativo ≥ alert Y absoluto ≥ 25% del coste típico de ventana
        # (si no, un compás con coste casi nulo se marcaría por diferencias irrelevantes)
        rows[m] = {"changed": changed, "rel": rel,
                   "flag": changed > 0 and rel >= alert and (fin - agn) >= 0.25 * v_ref,
                   "f_agn": "-".join(str(n.finger_agnostic) for n in ns),
                   "f_fin": "-".join(str(n.fingering) for n in ns)}
    tot_agn = sum(n.cost_pure_agn for n in notes)
    tot_fin = sum(n.cost_pure for n in notes)
    return rows, v_ref, (tot_fin - tot_agn) / max(tot_agn, 1e-9)


def _print_cost_section(label: str, seq: Sequence[INote], alert: float) -> None:
    """Compases cuya digitación difiere de la agnóstica, con su coste relativo."""
    rows, _v, rel_tot = _cost_by_measure(seq, alert)
    changed = {m: r for m, r in rows.items() if r["changed"]}
    if not changed:
        return
    print(f"Coste ergonómico frente a la digitación agnóstica — {label} "
          f"(coste de ventana sin penalizaciones; total {100 * rel_tot:+.0f}%; ⚠ = más de "
          f"+{100 * alert:.0f}%):")
    for m in sorted(changed):
        r = changed[m]
        print(f"  {'⚠' if r['flag'] else ' '} c.{m:<3} {r['f_agn']} → {r['f_fin']}   "
              f"{100 * r['rel']:+.0f}%")
    print()


def print_chord_report(chords: dict[int, ChordInfo],
                       seqs: Sequence[tuple[str, Sequence[INote]]],
                       alert: float = 0.5) -> None:
    """Acorde por compás (con margen) y tabla de gestos repetidos por mano."""
    measures = {n.measure for _lab, seq in seqs for n in seq}
    labels = {m: ci.label() for m, ci in chords.items()}
    print("Acorde por compás (todas las notas del compás, ambas manos):")
    for m, ci in sorted(chords.items()):
        if m not in measures:
            continue
        flag = "   ⚠ empate: el nombre depende del orden de las plantillas" if ci.margin < 0.005 else ""
        print(f"  c.{m:<3} {ci.label():<10} margen={ci.margin:.2f}{flag}")
    print()

    for label, seq in seqs:
        if not seq:
            continue
        print(f"Gestos repetidos — {label} (mismas posiciones e intervalos; "
              f"las escaladas solo si tienen las mismas notas):")
        groups = _gesture_groups(seq, labels)
        if not groups:
            print("  (ninguno)")
        for g in groups:
            items = g["items"]
            ms    = ",".join(str(it[0]) for it in items)
            chs   = ",".join((it[1] or "?") for it in items)
            tipo  = ("bloques, " if g["kind"] == "b" else "") + \
                    ("mismas notas" if g["exact"] else "transportado")
            same  = len({it[4] for it in items}) == 1
            all_t = all(it[5] for it in items)
            verdict = "✓ coinciden" if same else "✗ difieren"
            if all_t:
                verdict += " (enlazados)"
            print(f"  c.{ms}  [{tipo}]  acordes: {chs}")
            if g["exact"]:
                print(f"      notas: {items[0][3]}")
            for it in items:
                extra = "" if g["exact"] else f"   ({it[3]})"
                print(f"      c.{it[0]:<3} {it[4]}{extra}")
            print(f"      → {verdict}")
        print()
        _print_cost_section(label, seq, alert)


def print_fingering(result: FingeringResult,
                    right_only: bool = False,
                    left_only:  bool = False,
                    show_confidence: bool = False) -> None:
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

    chord_by_m = {c["measure"]: c["chord"] for c in result.chords}

    by_measure: dict[int, dict[str, list]] = defaultdict(lambda: {"right":[], "left":[]})
    for n in result.notes:
        if right_only and n["hand"] == "left":
            continue
        if left_only  and n["hand"] == "right":
            continue
        by_measure[n["measure"]][n["hand"]].append(n)

    for m in sorted(by_measure):
        hdr = f"Compás {m}"
        if m in chord_by_m:
            hdr += f"  [{chord_by_m[m]}]"
        print(hdr + ":")
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
                role = "  {" + n["chord_role"] + "}" if n.get("chord_role") else ""
                fa   = n.get("fingering_agnostic", 0)
                agn  = f"  (agn:{fa})" if (fa and fa != f) else ""
                fname = _FINGER_NAMES.get(f, "?") if f > 0 else "—"
                fnum  = str(f) if f > 0 else "?"
                print(f"    {n['note']:<5} — {fnum} ({fname}){ambig}{chord}{sust}{imposible}{role}{agn}{conf_str}")
        print()


def export_json(result: FingeringResult, path: str) -> None:
    data = {
        "midi_file":      result.midi_file,
        "bpm":            result.bpm,
        "time_signature": result.time_signature,
        "measures":       result.measures,
        "notes":          result.notes,
    }
    if result.chords:
        data["chords"] = result.chords
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
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
        description="Genera digitación pianística a partir de un fichero MIDI (v4)."
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
    parser.add_argument("--chord-mode", action="store_true",
                        help="Modo acordes: calcula el acorde de cada compás y mantiene "
                             "la digitación consistente entre acordes")
    parser.add_argument("--chord-weight", type=float, default=0.5, metavar="W",
                        help="Peso λ∈[0,1] de la consistencia frente al coste ergonómico "
                             "(0 = agnóstico como v3, 1 = solo consistencia; defecto 0.5)")
    parser.add_argument("--chord-iters", type=int, default=2, metavar="N",
                        help="Pasadas máximas de refinamiento por consistencia (defecto 2)")
    parser.add_argument("--chord-keys", choices=["harmony", "shape", "both"], default="both",
                        help="Claves de consistencia: harmony (acorde+rol), shape (forma del "
                             "gesto, independiente de la armonía) o both (defecto)")
    parser.add_argument("--bass-prior", type=float, default=0.0, metavar="B",
                        help="Prior de bajo (solo mano izquierda): penaliza no usar el 5 en la "
                             "nota más grave de cada compás (si cae en el tiempo 1). B es una "
                             "fracción del coste típico de ventana; 0 = desactivado (defecto), "
                             "0.3 = suave, ≥1 = casi obligatorio")
    parser.add_argument("--chord-runs", action="store_true",
                        help="Compara también las escaladas (≥3 notas estrictamente "
                             "ascendentes/descendentes) por su forma entre alturas distintas "
                             "(por defecto solo se comparan con las mismas notas)")
    parser.add_argument("--chord-report", action="store_true",
                        help="Informe: acorde por compás con margen y gestos repetidos por "
                             "mano con sus digitaciones. Por sí solo no cambia ningún dedo")
    parser.add_argument("--cost-alert", type=float, default=0.5, metavar="X",
                        help="Umbral relativo (0.5 = +50%%) a partir del cual se avisa de que "
                             "un compás cambiado cuesta más que con la digitación agnóstica "
                             "(coste ergonómico sin penalizaciones; además debe superar el 25%% "
                             "del coste típico de ventana)")
    parser.add_argument("--tie-gestures", action="store_true",
                        help="Enlaza la digitación de los compases con exactamente el mismo "
                             "gesto en una mano (patrón común de menor coste conjunto)")
    parser.add_argument("--tie-tol", type=float, default=0.5, metavar="T",
                        help="Con --tie-gestures: máximo aumento relativo de coste permitido "
                             "en cualquier compás del grupo (defecto 0.5 = 50%%; un valor "
                             "enorme fuerza el enlace siempre)")
    parser.add_argument("--json", metavar="FICHERO",
                        help="Exportar resultado a JSON")
    parser.add_argument("--xml", metavar="FICHERO",
                        help="Exportar resultado a MusicXML")
    args = parser.parse_args()
    if not 0.0 <= args.chord_weight <= 1.0:
        parser.error("--chord-weight debe estar entre 0 y 1")
    if args.chord_iters < 1:
        parser.error("--chord-iters debe ser ≥ 1")
    if args.bass_prior < 0:
        parser.error("--bass-prior debe ser ≥ 0")
    if args.tie_tol < 0:
        parser.error("--tie-tol debe ser ≥ 0")
    if args.cost_alert < 0:
        parser.error("--cost-alert debe ser ≥ 0")

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
    print(f"  Tempo: {tmap.bpm0:.1f} bpm  |  Compás: {tmap.time_signature}")
    print(f"  Tracks (no-drum): {len(instruments)}")
    for idx, inst in enumerate(instruments):
        print(f"    [{idx}] {inst.name or '(sin nombre)'} — {len(inst.notes)} notas")
    print()

    # Análisis armónico con TODAS las notas (antes de filtrar por mano o compás)
    chords: dict[int, ChordInfo] = {}
    if args.chord_mode or args.chord_report:
        all_seqs = (rh_seq, lh_seq, rh_sus, lh_sus)
        chords = analyze_chords(all_seqs, tmap)
        for seq in all_seqs:
            annotate_chords(seq, chords)
        for seq in (rh_seq, lh_seq):
            annotate_shapes(seq, shape_runs=args.chord_runs)
        if args.chord_mode:
            print(f"Modo acordes: {len(chords)} compases analizados  |  "
                  f"peso λ={args.chord_weight}  |  claves={args.chord_keys}  |  "
                  f"pasadas máx={args.chord_iters}")
            print()
    elif args.tie_gestures:
        for seq in (rh_seq, lh_seq):
            annotate_shapes(seq, shape_runs=args.chord_runs)

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

    cw = args.chord_weight if args.chord_mode else 0.0
    tie_tol = args.tie_tol if args.tie_gestures else None
    tie_reports: list[tuple[str, list[dict]]] = []

    if rh_seq:
        print("Calculando digitación mano derecha (melodía)…")
        _h = Hand(rh_seq, side="right", size=args.hand_size,
                  chord_weight=cw, chord_iters=args.chord_iters,
                  chord_keys=args.chord_keys, tie_tol=tie_tol)
        _h.generate()
        tie_reports.append(("MD", _h.tie_report))

    if lh_seq:
        print("Calculando digitación mano izquierda (melodía)…")
        _h = Hand(lh_seq, side="left",  size=args.hand_size,
                  chord_weight=cw, chord_iters=args.chord_iters,
                  chord_keys=args.chord_keys, bass_prior=args.bass_prior,
                  tie_tol=tie_tol)
        _h.generate()
        tie_reports.append(("MI", _h.tie_report))

    if rh_sus:
        print("Calculando digitación mano derecha (voz sostenida)…")
        optimize_sustained(rh_sus, rh_seq, Hand._SIZE_FACTORS.get(args.hand_size, 0.82), "right")

    if lh_sus:
        print("Calculando digitación mano izquierda (voz sostenida)…")
        optimize_sustained(lh_sus, lh_seq, Hand._SIZE_FACTORS.get(args.hand_size, 0.82), "left")

    result = build_result(args.midi, tmap, rh_seq, lh_seq, rh_sus, lh_sus, chords)

    if args.chord_mode:
        print()
        for label, seq in (("MD", rh_seq), ("MI", lh_seq)):
            if not seq:
                continue
            g_ag, g_n = ChordConsistency.agreement(seq, "shape", min_others=1)
            h_ag, h_n = ChordConsistency.agreement(seq, "harmony")
            changed = sum(1 for n in seq if n.finger_agnostic and n.fingering != n.finger_agnostic)
            print(f"  {label}: gestos idénticos con el mismo dedo {g_ag:.0%} ({g_n} comp.)  |  "
                  f"patrones armónicos {h_ag:.0%} ({h_n} comp.)  |  "
                  f"dedos distintos a la agnóstica: {changed}/{len(seq)}")

    if args.bass_prior > 0 and lh_seq:
        bass = [n for n in lh_seq if n.is_bass]
        if bass:
            if not args.chord_mode:
                print()
            n5  = sum(1 for n in bass if n.fingering == 5)
            n5a = sum(1 for n in bass if n.finger_agnostic == 5)
            print(f"  MI: prior de bajo B={args.bass_prior}  |  bajos marcados: {len(bass)}  |  "
                  f"con dedo 5: {n5} (agnóstica: {n5a})")

    if args.tie_gestures:
        print()
        for label, rep_ in tie_reports:
            n_tied = sum(1 for r in rep_ if r["tied"])
            print(f"  {label}: gestos repetidos: {len(rep_)}  |  enlazados: {n_tied}  "
                  f"(tolerancia {100 * args.tie_tol:.0f}%)")
            for r in rep_:
                ms = ",".join(f"c{m}" for m in r["measures"])
                if r["tied"]:
                    print(f"      {ms}: {r['pattern']}  ({r['reason']}; antes: {' '.join(r['before'])})")
                else:
                    print(f"      {ms}: no enlazado — {r['reason']}  (antes: {' '.join(r['before'])})")

    if args.chord_mode or args.bass_prior > 0 or args.tie_gestures:
        lines = []
        for label, seq in (("MD", rh_seq), ("MI", lh_seq)):
            rows, _v, rel_tot = _cost_by_measure(seq, args.cost_alert)
            n_ch = sum(1 for r in rows.values() if r["changed"])
            if not n_ch:
                continue
            lines.append(f"  {label}: coste ergonómico frente a la agnóstica {100 * rel_tot:+.0f}%  "
                         f"({n_ch} compases con dedos distintos)")
            flagged = sorted(m for m, r in rows.items() if r["flag"])
            if flagged:
                det = ", ".join(f"c{m} ({100 * rows[m]['rel']:+.0f}%: "
                                f"{rows[m]['f_agn']} → {rows[m]['f_fin']})" for m in flagged)
                lines.append(f"  ⚠ {label}: compases que cuestan más de +{100 * args.cost_alert:.0f}%: {det}")
        if lines:
            print()
            for ln in lines:
                print(ln)

    print()
    print("─" * 60)
    print()

    print_fingering(result,
                    right_only=args.right_only,
                    left_only=args.left_only,
                    show_confidence=args.confidence)

    if args.chord_report:
        print()
        print("─" * 60)
        print()
        print_chord_report(chords, [("MD", rh_seq), ("MI", lh_seq)], args.cost_alert)

    if args.json:
        export_json(result, args.json)

    if args.xml:
        export_musicxml(result, rh_seq + rh_sus, lh_seq + lh_sus, args.xml)


if __name__ == "__main__":
    main()
