#!/usr/bin/env python3
"""
fingering.py — Generador automático de digitación pianística a partir de MIDI.

Algoritmo: búsqueda combinatoria con ventana deslizante de 9 notas.
Basado en pianoplayer (Marco Musy, MIT License).

Uso:
    python fingering.py archivo.mid
    python fingering.py archivo.mid --hand-size L
    python fingering.py archivo.mid --measures 8
    python fingering.py archivo.mid --right-only
    python fingering.py archivo.mid --left-only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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
    pitch: int = 0
    octave: int = 0
    name: str = ""
    x: float = 0.0           # posición física en el teclado (cm)
    time: float = 0.0         # inicio en segundos
    duration: float = 0.0
    isBlack: bool = False
    isChord: bool = False
    chordID: int = 0
    chordnr: int = 0
    NinChord: int = 0
    noteID: int = 0
    measure: int = 0
    fingering: int = 0
    cost: float = 0.0


# Notas MIDI → nombre legible
_MIDI_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Posición física de cada semitono en una octava de 16.5 cm (7 teclas blancas)
_KEY_CM_PER_OCTAVE = 16.5
_KEY_CM_PER_WHITE  = _KEY_CM_PER_OCTAVE / 7.0

# Offset dentro de la octava para cada semitono MIDI (0=C … 11=B)
_SEMITONE_OFFSET = [
    0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0,
    4.5, 5.0, 5.5, 6.0, 6.5,
]


def keypos_midi(pitch: int) -> float:
    """Posición horizontal (cm) de la tecla a partir del MIDI pitch."""
    octave = pitch // 12
    semi   = pitch % 12
    return _KEY_CM_PER_OCTAVE * octave + _SEMITONE_OFFSET[semi] * _KEY_CM_PER_WHITE


def pitch_name(pitch: int) -> str:
    """Nombre legible de la nota, p.ej. C#4."""
    octave = (pitch // 12) - 1   # convención MIDI: C4 = 60
    name   = _MIDI_NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de MIDI → secuencias de INote
# ─────────────────────────────────────────────────────────────────────────────

_CHORD_STAGGER_S = 0.05   # separación temporal artificial para acordes

def _build_note_seq(pm_notes: list, seconds_per_beat: float) -> list[INote]:
    """Convierte una lista de pretty_midi.Note en INote[], agrupando acordes."""
    pm_notes = sorted(pm_notes, key=lambda n: n.start)
    noteseq: list[INote] = []
    chord_id = note_id = 0
    i = 0

    while i < len(pm_notes):
        onset = pm_notes[i].start
        j = i + 1
        while j < len(pm_notes) and abs(pm_notes[j].start - onset) < 1e-4:
            j += 1

        group = [n for n in pm_notes[i:j] if (n.end - n.start) > 0]
        if not group:
            i = j
            continue

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
            an.measure  = int(n.start / (seconds_per_beat * 4)) + 1
            pc = n.pitch % 12
            an.isBlack  = pc in {1, 3, 6, 8, 10}
            noteseq.append(an)
        else:
            for k, cn in enumerate(group):
                an = INote()
                an.chordID  = chord_id
                an.noteID   = note_id; note_id += 1
                an.isChord  = True
                an.pitch    = cn.pitch
                an.chordnr  = k
                an.NinChord = len(group)
                an.octave   = cn.pitch // 12
                an.name     = pitch_name(cn.pitch)
                an.x        = keypos_midi(cn.pitch)
                an.time     = onset - _CHORD_STAGGER_S * (len(group) - k - 1)
                an.duration = (cn.end - cn.start) + _CHORD_STAGGER_S * (len(group) - 1)
                an.measure  = int(onset / (seconds_per_beat * 4)) + 1
                pc = cn.pitch % 12
                an.isBlack  = pc in {1, 3, 6, 8, 10}
                noteseq.append(an)
            chord_id += 1
        i = j

    return noteseq


def split_hands(pm: pretty_midi.PrettyMIDI, right_track: int = 0, left_track: int = 1
                ) -> tuple[list[INote], list[INote]]:
    """
    Separa el MIDI en mano derecha e izquierda.

    Estrategia (en orden de prioridad):
      1. Si hay ≥2 instruments, usa right_track y left_track como índices.
      2. Si hay 1 instrument, parte por la mediana de pitch: notas altas → RH, bajas → LH.
    """
    tempo_times, tempo_values = pm.get_tempo_changes()
    bpm = tempo_values[0] if len(tempo_values) > 0 else 120.0
    spb = 60.0 / bpm  # segundos por beat

    instruments = [i for i in pm.instruments if not i.is_drum]

    if len(instruments) >= 2:
        rh_notes = instruments[right_track].notes if right_track < len(instruments) else []
        lh_notes = instruments[left_track].notes  if left_track  < len(instruments) else []
    elif len(instruments) == 1:
        all_notes = sorted(instruments[0].notes, key=lambda n: n.start)
        if all_notes:
            pitches  = [n.pitch for n in all_notes]
            median_p = sorted(pitches)[len(pitches) // 2]
            rh_notes = [n for n in all_notes if n.pitch >= median_p]
            lh_notes = [n for n in all_notes if n.pitch <  median_p]
        else:
            rh_notes = lh_notes = []
    else:
        rh_notes = lh_notes = []

    rh_seq = _build_note_seq(list(rh_notes), spb) if rh_notes else []
    lh_seq = _build_note_seq(list(lh_notes), spb) if lh_notes else []
    return rh_seq, lh_seq


# ─────────────────────────────────────────────────────────────────────────────
# Motor de optimización de digitación (adaptado de pianoplayer/hand.py)
# ─────────────────────────────────────────────────────────────────────────────

class Hand:
    """Optimizador de digitación para una mano sobre una secuencia de notas."""

    _SIZE_FACTORS = {
        "XXS": 0.33, "XS": 0.46, "S": 0.64,
        "M":   0.82, "L":  1.0,  "XL": 1.1, "XXL": 1.2,
    }

    def __init__(self, noteseq: list[INote], side: str = "right", size: str = "M") -> None:
        self.LR      = side
        self.noteseq = list(noteseq)
        self.fingers = (1, 2, 3, 4, 5)

        # Posición de reposo de cada dedo (cm). Índice 0 es dummy.
        self.frest   = [None, -7.0, -2.8, 0.0, 2.8, 5.6]
        self.weights = [None,  1.1,  1.0,  1.1, 0.9, 0.8]   # fuerza relativa
        self.bfactor = [None,  0.3,  1.0,  1.1, 0.8, 0.7]   # penalización tecla negra

        self.hf = self._SIZE_FACTORS.get(size, self._SIZE_FACTORS["M"])
        for i in range(1, 6):
            if self.frest[i] is not None:
                self.frest[i] *= self.hf  # type: ignore[operator]

        self.depth     = 9
        self.autodepth = True

        # Estado de postura (memoria de posición entre notas)
        self.finger_positions      = list(self.frest)
        self._has_position_state   = False
        self.preserve_posture_mem  = True
        self.relocation_alpha      = 0.3
        self.max_span_cm           = 21.0 * self.hf
        self.max_follow_lag_cm     = 2.5  * self.hf
        self.min_finger_gap_cm     = 0.15 * self.hf

        self.fingerseq: list[list] = []

    # ── geometría de la mano ──────────────────────────────────────────────

    def _relaxed_targets(self, fi: int, note_x: float) -> dict[int, float]:
        ifx = self.frest[fi]
        if ifx is None:
            return {}
        return {j: (self.frest[j] - ifx) + note_x  # type: ignore[operator]
                for j in range(1, 6) if self.frest[j] is not None}

    def _apply_position_constraints(self, fp: list, fi: int, note_x: float,
                                     targets: dict) -> None:
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
            tgt = targets.get(j)
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

    # ── función de coste ──────────────────────────────────────────────────

    def ave_velocity(self, fingering: Sequence[int], notes: Sequence[INote]) -> float:
        """Coste promedio de velocidad de dedo para una digitación candidata."""
        fp = list(self.finger_positions)
        self.set_fingers_positions(fingering, notes, 0, fp=fp, force_relaxed=False)
        vmean = 0.0
        for i in range(1, self.depth):
            na, nb = notes[i-1], notes[i]
            fb = fingering[i]
            fpos = fp[fb]
            if fpos is None:
                continue
            dx = abs(nb.x - fpos)
            dt = abs(nb.time - na.time) + 0.1
            v  = dx / dt
            w  = self.weights[fb] or 1.0
            bf = self.bfactor[fb] or 1.0 if nb.isBlack else 1.0
            vmean += v / (w * bf)
            self.set_fingers_positions(fingering, notes, i, fp=fp, force_relaxed=False)
        return vmean / max(1, self.depth - 1)

    # ── reglas de poda ────────────────────────────────────────────────────

    def skip(self, fa: int, fb: int, na: INote, nb: INote) -> bool:
        """True si la transición fa→fb entre na→nb es inválida/improbable."""
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
            pair = (min(fa, fb), max(fa, fb))
            thresh = {(3,4):5,(4,5):5,(2,3):6,(2,4):7,(3,5):8,
                      (2,5):11,(1,2):12,(1,3):14,(1,4):16}.get(pair)
            if thresh is not None and axba > thresh:
                return True

        return False

    # ── optimización por ventana deslizante ───────────────────────────────

    def optimize_seq(self, nseq: Sequence[INote], istart: int) -> tuple[list[int], float]:
        """Mejor digitación para una ventana de hasta 9 notas (búsqueda exhaustiva podada)."""
        if self.autodepth:
            if nseq[0].isChord:
                self.depth = max(3, nseq[0].NinChord - nseq[0].chordnr + 1)
            else:
                t0 = nseq[0].time
                for i in range(4, 10):
                    self.depth = i
                    if nseq[i-1].time - t0 > 3.5:
                        break

        depth = self.depth
        u_start = list(self.fingers) if istart == 0 else [istart]
        best  = [0] * 9
        minv  = 1e10
        cand  = [0] * 9

        def bt(level: int) -> None:
            nonlocal best, minv
            if level == depth:
                v = self.ave_velocity(cand, nseq)
                if v < minv:
                    best[:] = cand[:]
                    minv = v
                return
            choices = u_start if level == 0 else self.fingers
            for f in choices:
                if level > 0 and self.skip(cand[level-1], f, nseq[level-1], nseq[level]):
                    continue
                cand[level] = f
                bt(level + 1)

        bt(0)
        return best, minv

    # ── generación completa ───────────────────────────────────────────────

    def generate(self) -> None:
        """Asigna `fingering` a cada nota en self.noteseq."""
        init_autodepth = self.autodepth
        init_depth     = self.depth
        original_x: list[float] | None = None

        # Mano izquierda: espejamos el teclado para reutilizar la misma lógica
        if self.LR == "left":
            original_x = [n.x for n in self.noteseq]
            for n in self.noteseq:
                n.x = -n.x

        self.fingerseq = []
        self.finger_positions    = list(self.frest)
        self._has_position_state = False

        out: list[int] = []
        start_finger = 0
        n_total = len(self.noteseq)
        last_valid_finger = 1

        try:
            for i in range(n_total):
                an = self.noteseq[i]

                if i > n_total - 11:
                    if self.autodepth:
                        self.autodepth = False
                        self.depth = 9

                window = list(self.noteseq[i : i + 9])
                if window and len(window) < 9:
                    window += [window[-1]] * (9 - len(window))
                if not window:
                    break

                if i > n_total - 10 and out and len(out) > 1:
                    best_finger = out.pop(1)
                    out[0] = best_finger
                    start_finger = out[1] if len(out) > 1 else best_finger
                else:
                    out, vel = self.optimize_seq(window, start_finger)
                    best_finger  = out[0]
                    start_finger = out[1] if len(out) > 1 else out[0]

                # Fallback: si la cola dejó un dedo=0, re-optimizar
                if best_finger == 0:
                    out, vel = self.optimize_seq(window, start_finger)
                    best_finger  = out[0]
                    start_finger = out[1] if len(out) > 1 else out[0]

                an.fingering = best_finger
                if best_finger:
                    last_valid_finger = best_finger
                self.set_fingers_positions(out, window, 0)
                self.fingerseq.append(list(self.finger_positions))
                an.cost = vel
        finally:
            self.autodepth = init_autodepth
            self.depth      = init_depth
            if original_x is not None:
                for n, x in zip(self.noteseq, original_x):
                    n.x = x

        # Reparar cualquier nota que haya quedado sin dedo (fingering==0)
        last = 1
        for n in self.noteseq:
            if n.fingering == 0:
                n.fingering = last
            else:
                last = n.fingering


# ─────────────────────────────────────────────────────────────────────────────
# Formateo de salida
# ─────────────────────────────────────────────────────────────────────────────

_FINGER_NAMES = {1: "pulgar", 2: "índice", 3: "corazón", 4: "anular", 5: "meñique"}


def _format_finger(f: int) -> str:
    name = _FINGER_NAMES.get(f, "?")
    return f"{f} ({name})"


def print_fingering(rh_seq: list[INote], lh_seq: list[INote],
                    right_only: bool = False, left_only: bool = False) -> None:
    """Imprime el fingering agrupado por compás."""

    # Recopilar todos los compases presentes
    all_measures: set[int] = set()
    if not left_only:
        all_measures.update(n.measure for n in rh_seq)
    if not right_only:
        all_measures.update(n.measure for n in lh_seq)

    if not all_measures:
        print("(sin notas)")
        return

    # Índices por compás
    rh_by_measure: dict[int, list[INote]] = {}
    lh_by_measure: dict[int, list[INote]] = {}
    for n in rh_seq:
        rh_by_measure.setdefault(n.measure, []).append(n)
    for n in lh_seq:
        lh_by_measure.setdefault(n.measure, []).append(n)

    for m in sorted(all_measures):
        print(f"Compás {m}:")

        if not left_only:
            rh_notes = rh_by_measure.get(m, [])
            print("  Mano derecha:")
            if rh_notes:
                for n in rh_notes:
                    chord_tag = " [acorde]" if n.isChord else ""
                    print(f"    {n.name:<5} — {_format_finger(n.fingering)}{chord_tag}")
            else:
                print("    (sin notas)")

        if not right_only:
            lh_notes = lh_by_measure.get(m, [])
            print("  Mano izquierda:")
            if lh_notes:
                for n in lh_notes:
                    chord_tag = " [acorde]" if n.isChord else ""
                    print(f"    {n.name:<5} — {_format_finger(n.fingering)}{chord_tag}")
            else:
                print("    (sin notas)")

        print()


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera digitación pianística a partir de un fichero MIDI."
    )
    parser.add_argument("midi", help="Ruta al fichero MIDI")
    parser.add_argument(
        "--hand-size", choices=["XXS","XS","S","M","L","XL","XXL"],
        default="M", metavar="SIZE",
        help="Tamaño de mano: XXS XS S M L XL XXL  (defecto: M ≈ 17 cm pulgar-meñique)"
    )
    parser.add_argument(
        "--right-track", type=int, default=0,
        help="Índice del track/instrumento para mano derecha (defecto: 0)"
    )
    parser.add_argument(
        "--left-track", type=int, default=1,
        help="Índice del track/instrumento para mano izquierda (defecto: 1)"
    )
    parser.add_argument(
        "--measures", type=int, default=0,
        help="Número máximo de compases a procesar (0 = todos)"
    )
    parser.add_argument("--right-only", action="store_true", help="Solo mano derecha")
    parser.add_argument("--left-only",  action="store_true", help="Solo mano izquierda")
    args = parser.parse_args()

    # Carga del MIDI
    try:
        pm = pretty_midi.PrettyMIDI(args.midi)
    except Exception as e:
        print(f"Error al leer el MIDI: {e}", file=sys.stderr)
        sys.exit(1)

    instruments = [i for i in pm.instruments if not i.is_drum]
    print(f"MIDI cargado: {args.midi}")
    print(f"  Tracks/instrumentos (no-drum): {len(instruments)}")
    for idx, inst in enumerate(instruments):
        print(f"    [{idx}] {inst.name or '(sin nombre)'} — {len(inst.notes)} notas")
    print()

    rh_seq, lh_seq = split_hands(pm, args.right_track, args.left_track)

    if args.left_only:
        rh_seq = []
    if args.right_only:
        lh_seq = []

    # Recortar por compases si se pide
    if args.measures > 0:
        rh_seq = [n for n in rh_seq if n.measure <= args.measures]
        lh_seq = [n for n in lh_seq if n.measure <= args.measures]

    print(f"Notas a procesar → MD: {len(rh_seq)}, MI: {len(lh_seq)}")
    print()

    # Optimización
    if rh_seq:
        rh_hand = Hand(rh_seq, side="right", size=args.hand_size)
        print("Calculando digitación mano derecha…")
        rh_hand.generate()

    if lh_seq:
        lh_hand = Hand(lh_seq, side="left", size=args.hand_size)
        print("Calculando digitación mano izquierda…")
        lh_hand.generate()

    print()
    print("─" * 60)
    print()

    print_fingering(
        rh_seq, lh_seq,
        right_only=args.right_only,
        left_only=args.left_only,
    )


if __name__ == "__main__":
    main()
