#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  variations.py — Variaciones pianísticas sobre el tema eom.dev               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Compositor en vivo de fichero único (estilo mutopia): no escribe una        ║
║  partitura fija, sino que compone de 8 compases en 8 compases mientras       ║
║  suena, y manda cada nota al piano como bytes MIDI crudos.                   ║
║                                                                              ║
║  TEMA      mi menor, 112 bpm, Em9 – Cmaj7 – Am9 – B7sus                      ║
║            hook e-o-m  =  E, B, G                                            ║
║                                                                              ║
║  MELODÍA   hook original · ornamentado · invertido · secuenciado ↑/↓ ·       ║
║            descolocado · aumentado · disminuido · desarrollo libre ·         ║
║            canon · terceras · octavas · mano izquierda                       ║
║  ARMONÍA   original · dominante real · andaluza · relativo mayor ·           ║
║            cadencia ii–V · ascenso VI–VII–V · Fa napolitano · un             ║
║            acorde por compás, dos progresiones por sección · pivote          ║
║  TEXTURA   coral · arpegios rodados · Alberti · stride · octavas ·           ║
║            arpegios en semicorcheas · ambient                                ║
║  FORMA     intro → tema → desarrollo → build → breakdown → desarrollo →      ║
║            clímax → settle  (8 compases cada sección, tempo 96–116 bpm)      ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    play    interpreta en vivo por un puerto MIDI de salida (p. ej. FP-30)    ║
║    render  escribe la misma interpretación en un fichero .mid                ║
║    plan    compone sin sonar y muestra la forma (tabla o --json)             ║
║    ports   lista los puertos MIDI de salida disponibles                      ║
║                                                                              ║
║  USO                                                                         ║
║    python variations.py ports                                                ║
║    python variations.py play --seed 11                                       ║
║    python variations.py play --port "FP-30" --humanize 8                     ║
║    python variations.py render salida.mid --seed 11 --cycles 2               ║
║    python variations.py plan --seed 11                                       ║
║                                                                              ║
║  DEPENDENCIAS   mido (+ python-rtmidi para play); el resto es stdlib         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__version__ = "0.1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════════════════════════════


class C:
    on = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    @classmethod
    def paint(cls, code: str, s) -> str:
        return f"\033[{code}m{s}\033[0m" if cls.on else str(s)


def bold(s): return C.paint("1", s)
def dim(s): return C.paint("2", s)
def red(s): return C.paint("31", s)
def green(s): return C.paint("32", s)
def yellow(s): return C.paint("33", s)
def blue(s): return C.paint("34", s)
def magenta(s): return C.paint("35", s)
def cyan(s): return C.paint("36", s)


class VariationsError(Exception):
    """Error esperado (entrada, puerto MIDI, dependencia): se muestra sin traceback."""


# ══════════════════════════════════════════════════════════════════════════════
#  TEORÍA
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR = [0, 2, 3, 5, 7, 8, 10]          # escala menor natural, grados 0..6
BEATS_PER_BAR = 4
BARS = 8                                # compases por sección
SEC_BEATS = BARS * BEATS_PER_BAR        # 32
TPQ = 480                               # ticks por negra en el .mid

# calidad -> (intervalos, sufijo)
QUAL: Dict[str, Tuple[List[int], str]] = {
    "m9": ([0, 3, 7, 10, 14], "m9"),
    "m7": ([0, 3, 7, 10], "m7"),
    "min": ([0, 3, 7], "m"),
    "maj": ([0, 4, 7], ""),
    "maj7": ([0, 4, 7, 11], "maj7"),
    "7": ([0, 4, 7, 10], "7"),
    "7sus4": ([0, 5, 7, 10], "7sus4"),
    "7b9": ([0, 4, 7, 10, 13], "7b9"),
    "m7b5": ([0, 3, 6, 10], "m7b5"),
}


@dataclass
class Chord:
    root: int
    qual: str

    @property
    def ivs(self) -> List[int]:
        return QUAL[self.qual][0]

    @property
    def pcs(self) -> set:
        return {(self.root + i) % 12 for i in self.ivs}

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.root] + QUAL[self.qual][1]


# armonías: progresiones de 4 compases, un acorde por compás.
# (semitonos sobre la tónica, calidad). Una sección son DOS progresiones (iguales o no).
HARMONIES: Dict[str, List[Tuple[int, str]]] = {
    "original":        [(0, "m9"), (8, "maj7"), (5, "m9"), (7, "7sus4")],   # Em9 Cmaj7 Am9 B7sus
    "true-dominant":   [(0, "m9"), (8, "maj7"), (5, "m9"), (7, "7")],       # B7 real
    "andalusian":      [(0, "min"), (10, "maj"), (8, "maj"), (7, "7")],     # Em D C B7
    "relative-major":  [(3, "maj7"), (10, "maj"), (0, "m7"), (8, "maj7")],  # Gmaj7 D Em7 Cmaj7 (I V vi IV)
    "cadence-ii-v":    [(5, "m9"), (6, "m7b5"), (7, "7"), (0, "m9")],       # Am9 F#m7b5 B7 Em9
    "submediant-lift": [(8, "maj7"), (10, "maj"), (7, "7"), (0, "m9")],     # Cmaj7 D B7 Em9
    "neapolitan":      [(0, "m9"), (8, "maj7"), (1, "maj7"), (7, "7")],     # Am9 Fmaj7 Bbmaj7 E7 (en la)
    "coda-end":        [(0, "m9"), (1, "maj7"), (7, "7"), (0, "m9")],       # Em9 Fmaj7 B7 Em9
}

# tema: una frase de 4 compases (un acorde por compás) que se enuncia dos veces por sección.
# (grado de escala, pulso dentro del compás, duración)
THEME_BARS: List[List[Tuple[int, float, float]]] = [
    [(0, 0, 0.95), (4, 2, 0.95), (2, 3, 0.95)],                         # Em9    E . B G
    [(0, 0, 0.95), (4, 2, 0.95), (5, 3, 0.45), (4, 3.5, 0.45)],         # Cmaj7  E . B C B
    [(0, 0, 0.95), (4, 2, 0.95), (2, 3, 0.95)],                         # Am9    E . B G
    [(7, 0, 0.95), (7, 2, 0.95), (4, 3, 0.95)],                         # B7sus  E' . E' B
]

# plantillas rítmicas de 4 pulsos (un compás) para el desarrollo libre
RHYTHMS = [
    [1, 1, 1, 1],
    [1, 2, 1],
    [2, 1, 1],
    [1, 1, 0.5, 0.5, 1],
    [1.5, 0.5, 1, 1],
    [1, 1, 2],
]


def step_to_semis(step: int) -> int:
    octv, deg = divmod(step, 7)
    return 12 * octv + MINOR[deg]


def build_harmony(name: str, tonic: int) -> List[Chord]:
    return [Chord((tonic + off) % 12, q) for off, q in HARMONIES[name]]


def stack(pcs: Sequence[int], near: int) -> List[int]:
    """Coloca las clases de altura en orden ascendente a partir de `near`."""
    out, lo = [], near
    for pc in pcs:
        p = lo + ((pc - lo) % 12)
        out.append(p)
        lo = p + 1
    return out


def vc_open(ch: Chord, near: int) -> List[int]:
    """Voicing abierto: fundamental, 5ª, 7ª, 3ª, 9ª (ascendente)."""
    iv, r = ch.ivs, ch.root
    order = [iv[0], iv[2]]
    if len(iv) > 3:
        order.append(iv[3])
    order.append(iv[1])
    if len(iv) > 4:
        order.append(iv[4])
    if len(iv) == 3:
        order.append(iv[0])
    return stack([(r + i) % 12 for i in order], near)


def vc_close(ch: Chord, near: int) -> List[int]:
    """Estructura superior cerrada (3ª, 5ª, 7ª)."""
    iv, r = ch.ivs, ch.root
    order = [iv[1], iv[2], iv[3] if len(iv) > 3 else iv[0]]
    return stack([(r + i) % 12 for i in order], near)


def vc_tri(ch: Chord, near: int) -> List[int]:
    iv, r = ch.ivs, ch.root
    return stack([(r + iv[0]) % 12, (r + iv[1]) % 12, (r + iv[2]) % 12], near)


def vc_low(ch: Chord, near: int) -> List[int]:
    """Voicing grave para arpegios: fundamental, 5ª, extensión (9ª/7ª), 3ª — p. ej. E1 B1 F#2 G2."""
    iv, r = ch.ivs, ch.root
    ext = iv[4] if len(iv) > 4 else (iv[3] if len(iv) > 3 else iv[0])
    return stack([(r + iv[0]) % 12, (r + iv[2]) % 12, (r + ext) % 12, (r + iv[1]) % 12], near)


def clamp_pitch(p: int) -> int:
    while p > 100:
        p -= 12
    while p < 28:
        p += 12
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  EVENTOS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Note:
    beat: float            # pulso dentro de la sección (0..32)
    dur: float
    pitch: int
    vel: int               # offset de velocidad (se resuelve en finalize)
    hand: str = "R"
    role: str = "acc"      # 'mel' | 'acc'


@dataclass
class MN:
    """Nota de melodía abstracta: grado de escala relativo a la tónica."""
    step: int
    beat: float
    dur: float
    semis: int = 0
    hand: str = "R"
    v: int = 0


@dataclass
class Section:
    index: int
    cycle: int
    kind: str
    tonic: int
    energy: float
    melody: str
    harmony: str
    texture: str
    chords: List[Chord]
    notes: List[Note]
    pedal: List[Tuple[float, int]]
    bpms: List[float]
    final: bool = False
    is_coda: bool = False

    def bpm_at(self, beat: float) -> float:
        return self.bpms[max(0, min(len(self.bpms) - 1, int(beat // BEATS_PER_BAR)))]

    def sec_at(self, beat: float) -> float:
        if beat < 0:
            return beat * 60.0 / self.bpms[0]
        t, b = 0.0, beat
        for bpm in self.bpms:
            if b >= BEATS_PER_BAR:
                t += BEATS_PER_BAR * 60.0 / bpm
                b -= BEATS_PER_BAR
            else:
                return t + b * 60.0 / bpm
        return t + b * 60.0 / self.bpms[-1]

    @property
    def duration(self) -> float:
        return self.sec_at(SEC_BEATS)

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} menor"

    def to_dict(self) -> dict:
        return {
            "index": self.index + 1,
            "cycle": "coda" if self.is_coda else self.cycle + 1,
            "kind": self.kind,
            "key": self.key_name,
            "energy": self.energy,
            "bpm": [round(self.bpms[0], 1), round(self.bpms[-1], 1)],
            "chords": [c.name for c in self.chords],
            "melody": self.melody,
            "harmony": self.harmony,
            "texture": self.texture,
            "notes": len(self.notes),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MELODÍA — variantes del hook
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Ctx:
    rng: random.Random
    tonic: int
    base: int              # altura de la tónica en el registro de la melodía
    chords: List[Chord]
    energy: float
    near: int = 28         # registro del acompañamiento (E1 = 28)
    high: bool = False     # acompañamiento en la mano derecha (melodía en la izquierda)


def theme() -> List[MN]:
    return [MN(s, rep * 16 + b * 4 + t, d) for rep in range(2)
            for b, bar in enumerate(THEME_BARS) for (s, t, d) in bar]


def fit(notes: List[MN], target: float = 3.0) -> List[MN]:
    """Reubica la melodía por octavas para que su media quede en el registro habitual."""
    if not notes:
        return notes
    mean = sum(n.step for n in notes) / len(notes)
    shift = 0
    while mean + shift < target - 4.5:
        shift += 7
    while mean + shift > target + 4.5:
        shift -= 7
    return [replace(n, step=n.step + shift) for n in notes] if shift else notes


def clip(notes: List[MN]) -> List[MN]:
    out = []
    for n in notes:
        if n.beat >= SEC_BEATS:
            continue
        out.append(replace(n, dur=min(n.dur, SEC_BEATS - n.beat)))
    return out


def ornament(notes: List[MN], rng: random.Random) -> List[MN]:
    out: List[MN] = []
    for n in notes:
        if n.dur >= 0.9 and rng.random() < 0.35:      # grupeto
            out += [MN(n.step + 1, n.beat, 0.25, n.semis, n.hand, n.v - 4),
                    MN(n.step, n.beat + 0.25, 0.25, n.semis, n.hand, n.v - 4),
                    MN(n.step - 1, n.beat + 0.5, 0.25, n.semis, n.hand, n.v - 4),
                    replace(n, beat=n.beat + 0.75, dur=n.dur - 0.75)]
        elif n.dur >= 0.9 and rng.random() < 0.4:       # apoyatura
            out += [MN(n.step + 1, n.beat, 0.25, n.semis, n.hand, n.v - 4),
                    replace(n, beat=n.beat + 0.25, dur=n.dur - 0.25)]
        else:
            out.append(n)
    return out


def nearest_ct_step(step: int, chord: Chord, tonic: int) -> int:
    best, bd = step, 99
    for s in range(step - 3, step + 4):
        if (tonic + step_to_semis(s)) % 12 in chord.pcs and abs(s - step) < bd:
            best, bd = s, abs(s - step)
    return best


def v_original(ctx: Ctx):
    return "original", [theme()]


def v_ornamented(ctx: Ctx):
    return "ornamented", [ornament(theme(), ctx.rng)]


def v_inverted(ctx: Ctx):
    return "inverted", [fit([replace(n, step=-n.step) for n in theme()])]


def v_sequence(ctx: Ctx):
    d = ctx.rng.choice([1, 2, -1, -2])
    notes = [replace(n, step=n.step + int(n.beat // 8) * d) for n in theme()]
    return ("sequence-up" if d > 0 else "sequence-down"), [fit(notes)]


def v_offbeat(ctx: Ctx):
    return "off-beat", [clip([replace(n, beat=n.beat + 0.5) for n in theme()])]


def v_augmented(ctx: Ctx):
    return "augmented", [clip([replace(n, beat=n.beat * 2, dur=n.dur * 2) for n in theme()])]


def v_diminished(ctx: Ctx):
    half = [replace(n, beat=n.beat * 0.5, dur=n.dur * 0.5) for n in theme()]
    sh = ctx.rng.choice([0, 0, 1, -1])
    second = [replace(n, beat=n.beat + 16, step=n.step + sh) for n in half]
    return "diminished", [half + second]


def v_free(ctx: Ctx):
    rng, out = ctx.rng, []
    for bar in range(8):
        chord = ctx.chords[bar]
        T = rng.choice(RHYTHMS)
        start = nearest_ct_step(rng.choice([0, 0, 2, 4, -1, 3]), chord, ctx.tonic)
        a = rng.choice([4, 3, 2, -4, -3, 5])
        b = rng.choice([-2, -1, 1, 2, -3])
        steps = [start, start + a, start + a + b]
        cur = steps[-1]
        while len(steps) < len(T) - 1:
            cur += rng.choice([-2, -1, -1, 1, 1, 2])
            steps.append(cur)
        steps = steps[: len(T) - 1]
        steps.append(nearest_ct_step(steps[-1] + rng.choice([-1, 0, 1]), chord, ctx.tonic))
        t = 0.0
        for st, d in zip(steps, T):
            out.append(MN(max(-4, min(9, st)), bar * 4 + t, d * 0.95))
            t += d
    return "free", [fit(out)]


def v_canon(ctx: Ctx):
    lead = ornament(theme(), ctx.rng) if ctx.rng.random() < 0.5 else theme()
    follow = clip([replace(n, step=n.step - 7, beat=n.beat + 4, hand="L", v=n.v - 10) for n in lead])
    return "canon", [lead, follow]


def v_thirds(ctx: Ctx):
    base = theme()
    return "thirds", [base, [replace(n, step=n.step - 2, v=n.v - 10) for n in base]]


def v_octaves(ctx: Ctx):
    base = theme()
    return "octaves", [base, [replace(n, semis=-12, v=n.v - 8) for n in base]]


def v_lefthand(ctx: Ctx):
    return "left-hand", [[replace(n, semis=-12, hand="L") for n in theme()]]


VARIANTS: Dict[str, Callable] = {
    "original": v_original, "ornamented": v_ornamented, "inverted": v_inverted,
    "sequence": v_sequence, "off-beat": v_offbeat, "augmented": v_augmented,
    "diminished": v_diminished, "free": v_free, "canon": v_canon,
    "thirds": v_thirds, "octaves": v_octaves, "left-hand": v_lefthand,
}


def harmonic_adjust(p: int, chord: Chord, m: MN) -> int:
    pc = p % 12
    if chord.qual in ("7", "7b9"):                    # sensible: sube el grado que choca con la 3ª
        third = (chord.root + 4) % 12
        if pc == (third - 1) % 12:
            p += 1
            pc = (pc + 1) % 12
    if m.dur >= 0.9 and m.beat % 1 == 0 and pc not in chord.pcs:   # evita roces en notas largas
        for d in (-1, 1):
            if (pc + d) % 12 in chord.pcs:
                p += d
                break
    return p


def resolve(voices: List[List[MN]], ctx: Ctx) -> List[Note]:
    out: List[Note] = []
    for voice in voices:
        for m in voice:
            p = ctx.base + step_to_semis(m.step) + m.semis
            chord = ctx.chords[min(7, int(m.beat // 4))]
            p = harmonic_adjust(p, chord, m)
            out.append(Note(m.beat, m.dur, clamp_pitch(p), m.v, m.hand, "mel"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  TEXTURAS
# ══════════════════════════════════════════════════════════════════════════════


def _hand(ctx: Ctx) -> str:
    return "R" if ctx.high else "L"


LOW_PATTERN = [0, 1, 3, 2, 3, 1, 3, 1]      # E1 B1 G2 F#2 G2 B1 G2 B1 sobre vc_low


def tex_chorale(ctx: Ctx) -> List[Note]:
    out, h = [], _hand(ctx)
    for b, ch in enumerate(ctx.chords):
        t0 = b * 4
        if ctx.high:
            tones = vc_open(ch, ctx.near)
        else:
            out.append(Note(t0, 3.9, vc_low(ch, ctx.near)[0], 4, h))
            tones = vc_close(ch, 52)
        for k in range(2):
            for p in tones:
                out.append(Note(t0 + 2 * k, 1.9, p, 0 if k == 0 else -5, h))
    return out


def _arp(ctx: Ctx, per_bar: int, dur: float) -> List[Note]:
    out, h, step = [], _hand(ctx), BEATS_PER_BAR / per_bar
    for b, ch in enumerate(ctx.chords):
        lv = vc_low(ch, ctx.near)
        for k in range(per_bar):
            strong = k % (per_bar // 2) == 0
            off = 5 if strong else (-3 if k % 2 else 0)
            out.append(Note(b * 4 + k * step, dur, lv[LOW_PATTERN[k % 8]], off, h))
        if not ctx.high:                                   # nota interna sostenida en la derecha
            out.append(Note(b * 4, 3.9, vc_close(ch, 55)[-1], -10, "R"))
    return out


def tex_rolling(ctx: Ctx) -> List[Note]:
    return _arp(ctx, 8, 0.7)


def tex_arp16(ctx: Ctx) -> List[Note]:
    return _arp(ctx, 16, 0.34)


def tex_alberti(ctx: Ctx) -> List[Note]:
    out, h = [], _hand(ctx)
    for b, ch in enumerate(ctx.chords):
        v = vc_tri(ch, 60 if ctx.high else 50)
        if not ctx.high:
            out.append(Note(b * 4, 3.9, stack([ch.root], 28)[0], 4, "L"))      # bajo sostenido
        for k in range(8):
            out.append(Note(b * 4 + k * 0.5, 0.5, v[[0, 2, 1, 2][k % 4]], 3 if k % 4 == 0 else -2, h))
    return out


def tex_stride(ctx: Ctx) -> List[Note]:
    out = []
    for b, ch in enumerate(ctx.chords):
        t = b * 4
        root = stack([ch.root], 33)[0]
        fifth = root + ch.ivs[2]
        if fifth > 52:
            fifth -= 12
        out.append(Note(t, 0.95, root, 4, "L"))
        out.append(Note(t + 2, 0.95, fifth, 0, "L"))
        for beat in (1, 3):
            for p in vc_close(ch, 55):
                out.append(Note(t + beat, 0.9, p, -8, "L"))
    return out


def tex_driving(ctx: Ctx) -> List[Note]:
    out = []
    for b, ch in enumerate(ctx.chords):
        root = stack([ch.root], 33)[0]
        for k in range(8):
            off = 6 if k % 4 == 0 else (2 if k % 2 == 0 else -4)
            t = b * 4 + k * 0.5
            out.append(Note(t, 0.45, root, off, "L"))
            out.append(Note(t, 0.45, root + 12, off - 2, "L"))
    return out


def tex_ambient(ctx: Ctx) -> List[Note]:
    out, h = [], _hand(ctx)
    for b, ch in enumerate(ctx.chords):
        t0, iv, r = b * 4, ch.ivs, ch.root
        out.append(Note(t0, 3.9, stack([r], ctx.near)[0], 8, h))
        ext = iv[4] if len(iv) > 4 else (iv[3] if len(iv) > 3 else iv[0])
        for p in stack([(r + ext) % 12, (r + iv[1]) % 12, (r + iv[2]) % 12], 62):
            out.append(Note(t0, 3.9, p, -6, "R"))
        for pos in sorted(ctx.rng.sample([0.5, 1.25, 2, 2.5, 3, 3.5], 3)):
            pc = ctx.rng.choice(sorted(ch.pcs))
            p = 79 + ((pc - 79) % 12) + 12 * ctx.rng.choice([0, 1, 1])
            out.append(Note(t0 + pos, 2.0, p, 0, "R"))
    return out


# nombre -> (función, re-pedal cada N pulsos; 0 = sin pedal). El original cambia pedal cada compás.
TEXTURES: Dict[str, Tuple[Callable, int]] = {
    "chorale": (tex_chorale, 4), "rolling": (tex_rolling, 4), "arp16": (tex_arp16, 4),
    "alberti": (tex_alberti, 4), "stride": (tex_stride, 4), "driving": (tex_driving, 4),
    "ambient": (tex_ambient, 4),
}
HIGH_OK = {"rolling", "arp16", "alberti", "chorale", "ambient"}

# ══════════════════════════════════════════════════════════════════════════════
#  FORMA
# ══════════════════════════════════════════════════════════════════════════════

FORM: List[Tuple[str, float]] = [
    ("intro", 0.30), ("theme", 0.80), ("development", 0.65), ("build", 0.85),
    ("breakdown", 0.10), ("development", 0.70), ("climax", 1.00), ("settle", 0.40),
]
CODA_ENERGY = 0.50
DYN = {  # crescendo/decrescendo dentro de la sección (multiplicador inicial, final)
    "intro": (0.70, 0.95), "theme": (1.0, 1.0), "development": (1.0, 1.05),
    "build": (0.90, 1.15), "breakdown": (0.95, 0.85), "climax": (1.0, 1.10),
    "settle": (1.0, 0.85), "coda": (1.0, 0.75),
}
RULES: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "intro": dict(
        melody=[("augmented", 3), ("original", 2), ("off-beat", 1), ("ornamented", 1)],
        harmony=[("original", 4)],
        texture=[("ambient", 3), ("chorale", 2), ("rolling", 1)]),
    "theme": dict(
        melody=[("original", 6), ("ornamented", 2), ("octaves", 1)],
        harmony=[("original", 5), ("true-dominant", 1)],
        texture=[("rolling", 3), ("alberti", 2), ("stride", 1), ("chorale", 1)]),
    "development": dict(
        melody=[("ornamented", 2), ("inverted", 2), ("sequence", 3), ("off-beat", 2), ("canon", 2),
                ("free", 3), ("thirds", 2), ("left-hand", 2)],
        harmony=[("true-dominant", 2), ("relative-major", 2), ("andalusian", 2), ("neapolitan", 1),
                 ("cadence-ii-v", 2), ("submediant-lift", 2), ("original", 1)],
        texture=[("rolling", 2), ("alberti", 2), ("stride", 2), ("chorale", 1), ("arp16", 1),
                 ("driving", 1)]),
    "build": dict(
        melody=[("diminished", 2), ("sequence", 3), ("free", 2), ("thirds", 2), ("original", 1)],
        harmony=[("submediant-lift", 3), ("cadence-ii-v", 2), ("true-dominant", 2), ("andalusian", 1)],
        texture=[("rolling", 2), ("arp16", 3), ("driving", 2), ("alberti", 1)]),
    "breakdown": dict(
        melody=[("augmented", 2), ("original", 2), ("left-hand", 3), ("off-beat", 1)],
        harmony=[("andalusian", 2), ("original", 2), ("neapolitan", 1), ("relative-major", 1)],
        texture=[("ambient", 4), ("chorale", 2)]),
    "climax": dict(
        melody=[("octaves", 4), ("thirds", 1), ("diminished", 2), ("ornamented", 1), ("original", 1)],
        harmony=[("true-dominant", 3), ("andalusian", 2), ("neapolitan", 2), ("original", 1)],
        texture=[("driving", 4), ("arp16", 3), ("stride", 1)]),
    "settle": dict(
        melody=[("original", 3), ("augmented", 1), ("left-hand", 2), ("ornamented", 1)],
        harmony=[("original", 3), ("relative-major", 1), ("cadence-ii-v", 2)],
        texture=[("chorale", 3), ("rolling", 2), ("ambient", 1)]),
    "coda": dict(
        melody=[("original", 1)],
        harmony=[("original", 1)],
        texture=[("chorale", 2), ("rolling", 1)]),
}
TONIC_E = 4


def tempo_for(energy: float) -> float:
    return 96.0 + 20.0 * energy      # 96..116 bpm


class Composer:
    """Genera las secciones una a una (8 compases cada una), de forma perezosa."""

    def __init__(self, seed: int, cycles: int = 2, coda: bool = True, tempo_scale: float = 1.0):
        self.seed = seed
        self.cycles = cycles
        self.coda = coda
        self.scale = tempo_scale
        self.rng = random.Random(seed)
        self.prev: Dict[str, Optional[str]] = {"melody": None, "harmony": None, "texture": None}
        self.bpm = 100.0 * tempo_scale

    def _pick(self, options: List[Tuple[str, int]], avoid: Optional[str] = None) -> str:
        opts = [(n, w) for n, w in options if n != avoid] or options
        total = sum(w for _, w in opts)
        r = self.rng.random() * total
        for n, w in opts:
            r -= w
            if r <= 0:
                return n
        return opts[-1][0]

    def plan(self) -> List[Tuple[int, str, float, int, int, bool]]:
        """(ciclo, tipo, energía, tónica, tónica siguiente, es_coda)"""
        out = []
        keys = [(TONIC_E + 5 * c) % 12 for c in range(self.cycles)]
        for c, t in enumerate(keys):
            nxt = keys[c + 1] if c + 1 < self.cycles else TONIC_E
            for kind, e in FORM:
                out.append((c, kind, e, t, nxt, False))
        if self.coda:
            out.append((self.cycles, "coda", CODA_ENERGY, TONIC_E, TONIC_E, True))
        return out

    def sections(self) -> Iterator[Section]:
        plan = self.plan()
        for idx, (cycle, kind, energy, tonic, nxt, is_coda) in enumerate(plan):
            yield self._compose(idx, cycle, kind, energy, tonic, nxt, is_coda, idx == len(plan) - 1)

    # ──────────────────────────────────────────────────────────────────────
    def _compose(self, idx, cycle, kind, energy, tonic, nxt, is_coda, final) -> Section:
        rng, rules = self.rng, RULES[kind]
        if cycle == 0 and kind == "theme":
            mel_key = "original"
        else:
            mel_key = self._pick(rules["melody"], self.prev["melody"])
        if is_coda:
            ha, hb = "original", "coda-end"
        else:
            ha = self._pick(rules["harmony"], self.prev["harmony"])
            hb = ha if rng.random() < 0.5 else self._pick(rules["harmony"], ha)
        harm_key = ha if ha == hb else f"{ha}+{hb}"
        chords = build_harmony(ha, tonic) + build_harmony(hb, tonic)
        if kind == "settle":                                   # pivote hacia la siguiente tonalidad
            chords[7] = Chord((nxt + 7) % 12, "7b9")

        base = tonic + 12 * round((64 - tonic) / 12)           # hook en E4 (64)
        ctx = Ctx(rng, tonic, base, chords, energy)
        ctx.high = mel_key == "left-hand"
        if ctx.high:
            ctx.near = 62

        topts = rules["texture"]
        if mel_key == "canon":
            topts = [("ambient", 1)]
        elif ctx.high:
            topts = [(n, w) for n, w in topts if n in HIGH_OK] or [("ambient", 1)]
        tex_key = self._pick(topts, self.prev["texture"])

        label, voices = VARIANTS[mel_key](ctx)
        notes = resolve(voices, ctx) + TEXTURES[tex_key][0](ctx)
        notes = self._finalize(notes, kind, 0.55 if (kind == "intro" and cycle > 0) else energy)

        pe = TEXTURES[tex_key][1]
        pedal: List[Tuple[float, int]] = []
        if pe:
            for k in range(0, SEC_BEATS, pe):
                pedal += [(float(k), 0), (k + 0.08, 127)]
        else:
            pedal.append((0.0, 0))

        target = tempo_for(energy) * self.scale
        bpms = [self.bpm + (target - self.bpm) * (i + 1) / BARS for i in range(BARS)]
        self.bpm = target
        if is_coda:                                            # ritardando final
            bpms[-2] *= 0.92
            bpms[-1] *= 0.82

        self.prev = {"melody": mel_key, "harmony": harm_key, "texture": tex_key}
        return Section(idx, cycle, kind, tonic, energy, label, harm_key, tex_key, chords, notes,
                       pedal, bpms, final, is_coda)

    @staticmethod
    def _finalize(notes: List[Note], kind: str, energy: float) -> List[Note]:
        # 1) quita duplicados y recorta solapes del mismo pitch
        notes = [n for n in notes if n.beat < SEC_BEATS and n.dur > 0]
        notes.sort(key=lambda n: (n.pitch, n.beat, 0 if n.role == "mel" else 1))
        kept: List[Note] = []
        last: Dict[int, Note] = {}
        for n in notes:
            n.dur = min(n.dur, SEC_BEATS - n.beat - 0.02)
            if n.dur < 0.05:
                continue
            p = last.get(n.pitch)
            if p is not None:
                if n.beat - p.beat < 0.06:
                    continue
                if n.beat < p.beat + p.dur:
                    p.dur = max(0.05, n.beat - p.beat - 0.03)
            last[n.pitch] = n
            kept.append(n)
        # 2) velocidades: base por energía + regulador de la sección
        ve = energy if kind == "intro" else max(energy, 0.55)  # solo la intro baja a pianissimo
        acc_base = 32 + 32 * ve                              # intro ~pp, clímax ~mf
        gap = 38 * min(1.0, max(0.0, (ve - 0.3) / 0.5))      # la melodía destaca ~35 desde el tema
        mel_base = acc_base + gap
        a, b = DYN[kind]
        for n in kept:
            mult = a + (b - a) * n.beat / SEC_BEATS
            n.vel = int(max(24, min(108, ((mel_base if n.role == "mel" else acc_base) + n.vel) * mult)))
        kept.sort(key=lambda n: (n.beat, n.pitch))
        return kept


# ══════════════════════════════════════════════════════════════════════════════
#  EVENTOS TEMPORIZADOS (bytes MIDI crudos)
# ══════════════════════════════════════════════════════════════════════════════

Raw = Tuple[int, ...]


def timed_events(sec: Section, hum_ms: float, rng: random.Random, ch: int) -> List[Tuple[float, int, Raw]]:
    """[(pulso, orden, bytes)] — orden: 0=note-off, 1=pedal, 2=note-on."""
    ev: List[Tuple[float, int, Raw]] = []
    for n in sec.notes:
        on, vel = n.beat, n.vel
        if hum_ms > 0:
            k = 1.0 / 1000.0 * sec.bpm_at(n.beat) / 60.0              # ms -> pulsos
            on += rng.gauss(0, hum_ms) * k - (hum_ms * 0.6 * k if n.role == "mel" else 0.0)
            vel = int(max(24, min(110, vel + rng.gauss(0, 2.5))))
        off = max(n.beat + n.dur, on + 0.02)
        ev.append((on, 2, (0x90 | ch, n.pitch, vel)))
        ev.append((off, 0, (0x80 | ch, n.pitch, 0)))
    for beat, val in sec.pedal:
        ev.append((beat, 1, (0xB0 | ch, 64, val)))
    return ev


def chords_str(sec: Section) -> str:
    names = [c.name for c in sec.chords]
    return " ".join(names[:4]) + " | " + " ".join(names[4:])


def describe(sec: Section) -> str:
    cyc = "coda" if sec.is_coda else f"c{sec.cycle + 1}"
    chords = chords_str(sec)
    bpm = f"{sec.bpms[0]:.0f}→{sec.bpms[-1]:.0f}"
    return (f"{sec.index + 1:>2} {cyc:<4} {sec.kind:<11} {sec.key_name:<8} {bpm:>8} bpm  "
            f"{chords:<46} {sec.melody:<14} {sec.texture:<8} {sec.harmony}")


def colored(sec: Section) -> str:
    cyc = "coda" if sec.is_coda else f"c{sec.cycle + 1}"
    chords = chords_str(sec)
    bpm = f"{sec.bpms[0]:.0f}→{sec.bpms[-1]:.0f}"
    return (f"{dim(f'{sec.index + 1:>2}')} {dim(f'{cyc:<4}')} {bold(f'{sec.kind:<11}')} "
            f"{cyan(f'{sec.key_name:<8}')} {yellow(f'{bpm:>8} bpm')}  {chords:<46} "
            f"{green(f'{sec.melody:<14}')} {magenta(f'{sec.texture:<8}')} {blue(sec.harmony)}")


HEADER = (f"{dim(' #')} {dim('cic ')} {dim('sección    ')} {dim('tonalidad')} {dim('   tempo    ')}  "
          f"{dim('acordes'):<55} {dim('melodía'):<23} {dim('textura'):<17} {dim('armonía')}")

# ══════════════════════════════════════════════════════════════════════════════
#  MIDI (mido, import perezoso)
# ══════════════════════════════════════════════════════════════════════════════


def _mido():
    try:
        import mido  # noqa: WPS433
    except ImportError:
        raise VariationsError("falta 'mido' (pip install mido python-rtmidi)")
    return mido


def list_ports(mido) -> List[str]:
    try:
        return list(mido.get_output_names())
    except Exception as e:  # backend rtmidi ausente o sin permisos
        raise VariationsError(f"backend MIDI no disponible: {e} (pip install python-rtmidi)")


def open_port(mido, wanted: Optional[str]):
    names = list_ports(mido)
    if not names:
        raise VariationsError("no hay puertos MIDI de salida; conecta el piano o usa 'render'")
    if wanted:
        hits = [n for n in names if wanted.lower() in n.lower()]
        if not hits:
            raise VariationsError(f"puerto '{wanted}' no encontrado. Disponibles: {', '.join(names)}")
        name = hits[0]
    else:
        fp = [n for n in names if "fp-30" in n.lower()]
        if fp:
            name = fp[0]
        elif len(names) == 1:
            name = names[0]
        else:
            raise VariationsError("varios puertos; elige uno con --port. Disponibles: " + ", ".join(names))
    try:
        return mido.open_output(name), name
    except Exception as e:
        raise VariationsError(f"no se pudo abrir '{name}': {e}")


def panic(port, mido, ch: int) -> None:
    """Silencia todo: pedal arriba, all-notes-off y note-off explícitos."""
    try:
        for raw in [(0xB0 | ch, 64, 0), (0xB0 | ch, 123, 0), (0xB0 | ch, 120, 0)]:
            port.send(mido.Message.from_bytes(list(raw)))
        for p in range(21, 109):
            port.send(mido.Message.from_bytes([0x80 | ch, p, 0]))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════


def _composer(args) -> Composer:
    if not 1 <= args.cycles <= 12:
        raise VariationsError("--cycles debe estar entre 1 y 12")
    if not 0.25 <= args.tempo_scale <= 3.0:
        raise VariationsError("--tempo-scale debe estar entre 0.25 y 3.0")
    if not 1 <= args.channel <= 16:
        raise VariationsError("--channel debe estar entre 1 y 16")
    if args.humanize < 0:
        raise VariationsError("--humanize no puede ser negativo")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 100000)
    args.seed = seed
    return Composer(seed, args.cycles, not args.no_coda, args.tempo_scale)


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def cmd_ports(args) -> int:
    names = list_ports(_mido())
    if not names:
        print(yellow("no hay puertos MIDI de salida"))
        return 0
    for n in names:
        print(f"  {cyan('•')} {n}")
    return 0


def cmd_plan(args) -> int:
    comp = _composer(args)
    secs = list(comp.sections())
    if args.json:
        print(json.dumps({"seed": comp.seed, "sections": [s.to_dict() for s in secs]},
                         ensure_ascii=False, indent=2))
        return 0
    print(bold(f"variations · semilla {comp.seed} · {args.cycles} ciclo(s)"
               f"{'' if args.no_coda else ' + coda'}"))
    print(HEADER)
    for s in secs:
        print(colored(s))
    total = sum(s.duration for s in secs)
    print(dim(f"{len(secs)} secciones · {len(secs) * BARS} compases · {_fmt_time(total)} · "
              f"{sum(len(s.notes) for s in secs)} notas"))
    return 0


def cmd_render(args) -> int:
    mido = _mido()
    comp = _composer(args)
    hrng = random.Random(comp.seed ^ 0xA5A5)
    ch = args.channel - 1
    mf = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    meta_ev: List[Tuple[int, int, object]] = []
    note_ev: List[Tuple[int, int, Raw]] = []
    off_beats, total_sec, n_notes = 0, 0.0, 0
    for sec in comp.sections():
        if not args.quiet:
            print(colored(sec))
        mark = (f"{'Coda' if sec.is_coda else f'Ciclo {sec.cycle + 1}'} - {sec.kind} "
                f"({sec.key_name}) - {sec.melody} / {sec.texture} / {sec.harmony}")
        mark = mark.encode("latin-1", "replace").decode("latin-1")   # mido escribe meta en latin-1
        meta_ev.append((off_beats * TPQ, 0, mido.MetaMessage("marker", text=mark)))
        for i, bpm in enumerate(sec.bpms):
            meta_ev.append(((off_beats + i * BEATS_PER_BAR) * TPQ, 1,
                            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
        for beat, order, raw in timed_events(sec, args.humanize, hrng, ch):
            note_ev.append((max(0, round((off_beats + beat) * TPQ)), order, raw))
        off_beats += SEC_BEATS
        total_sec += sec.duration
        n_notes += len(sec.notes)
    last = max((t for t, _, _ in note_ev), default=0)
    note_ev.append((last + 2 * TPQ, 1, (0xB0 | ch, 64, 0)))

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="variations - mapa de tempo", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    prev = 0
    for t, _, msg in sorted(meta_ev, key=lambda e: (e[0], e[1])):
        msg.time = t - prev
        prev = t
        meta.append(msg)
    trk = mido.MidiTrack()
    trk.append(mido.MetaMessage("track_name", name="Piano", time=0))
    if args.program is not None:
        trk.append(mido.Message("program_change", channel=ch, program=args.program, time=0))
    prev = 0
    for t, _, raw in sorted(note_ev, key=lambda e: (e[0], e[1])):
        msg = mido.Message.from_bytes(list(raw))
        msg.time = t - prev
        prev = t
        trk.append(msg)
    mf.tracks.extend([meta, trk])
    try:
        mf.save(args.output)
    except OSError as e:
        raise VariationsError(f"no se pudo escribir '{args.output}': {e}")
    print(green(f"✓ {args.output}") + dim(f"  semilla {comp.seed} · {off_beats // SEC_BEATS * BARS} compases · "
                                           f"{_fmt_time(total_sec)} · {n_notes} notas"))
    return 0


class _NullPort:
    def __init__(self):
        self.count = 0

    def send(self, msg):
        self.count += 1

    def close(self):
        pass


def cmd_play(args) -> int:
    mido = _mido()
    comp = _composer(args)
    ch = args.channel - 1
    if args.speed <= 0:
        raise VariationsError("--speed debe ser positivo")
    if args.null:
        port, pname = _NullPort(), "(sin puerto)"
    else:
        port, pname = open_port(mido, args.port)
    print(bold(f"variations · semilla {comp.seed} · {args.cycles} ciclo(s)"
               f"{'' if args.no_coda else ' + coda'} → {cyan(pname)}"))
    if not args.quiet:
        print(HEADER)
    if args.program is not None:
        port.send(mido.Message("program_change", channel=ch, program=args.program))

    hrng = random.Random(comp.seed ^ 0xA5A5)
    gen = comp.sections()
    heap: List[Tuple[float, int, int, object]] = []
    counter = [0]
    st = {"end": 0.0, "last_start": 0.0, "done": False}
    t0 = time.perf_counter() + 1.0                       # compás de cortesía

    def push(t, order, payload):
        heapq.heappush(heap, (t, order, counter[0], payload))
        counter[0] += 1

    def feed():
        try:
            sec = next(gen)
        except StopIteration:
            st["done"] = True
            return
        start = st["end"]
        for beat, order, raw in timed_events(sec, args.humanize, hrng, ch):
            push(t0 + (start + sec.sec_at(beat)) / args.speed, order, raw)
        if not args.quiet:
            push(t0 + start / args.speed, 3, lambda s=sec: print(colored(s), flush=True))
        st["last_start"] = start
        st["end"] = start + sec.duration
        if sec.final:
            push(t0 + (st["end"] + 2.5) / args.speed, 4, lambda: panic(port, mido, ch))
            st["done"] = True

    try:
        feed()
        while True:
            now = time.perf_counter()
            if not st["done"] and now >= t0 + st["last_start"] / args.speed:
                feed()                                   # compone los siguientes 8 compases
            if not heap:
                break
            t = heap[0][0]
            if t <= now:
                payload = heapq.heappop(heap)[3]
                if callable(payload):
                    payload()
                else:
                    port.send(mido.Message.from_bytes(list(payload)))
                continue
            dt = t - now
            if dt > 0.002:
                time.sleep(min(dt - 0.001, 0.005))
    except KeyboardInterrupt:
        print(yellow("\ninterrumpido"))
        return 130
    finally:
        panic(port, mido, ch)
        port.close()
    if args.null:
        print(dim(f"{port.count} mensajes MIDI (simulado)"))
    print(green("fin"))
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def _common(p, render: bool = False) -> None:
    p.add_argument("--seed", type=int, default=None, help="semilla (por defecto aleatoria; se imprime)")
    p.add_argument("--cycles", type=int, default=2, help="ciclos completos de 64 compases (def. 2: mi, la)")
    p.add_argument("--no-coda", action="store_true", help="sin coda de vuelta a mi menor")
    p.add_argument("--tempo-scale", type=float, default=1.0, help="factor sobre el tempo (0.25–3.0)")
    p.add_argument("--channel", type=int, default=1, help="canal MIDI 1–16 (def. 1)")
    p.add_argument("--program", type=int, default=None, help="program change opcional (0–127)")
    p.add_argument("--humanize", type=float, default=0.0 if render else 8.0,
                   help=f"jitter de ataque en ms, melodía adelantada (def. {0 if render else 8})")
    p.add_argument("--quiet", action="store_true", help="no imprime la forma")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="variations.py",
                                 description="Variaciones pianísticas sobre el tema eom.dev (compone en vivo).")
    ap.add_argument("--version", action="version", version=f"variations.py {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="interpreta en vivo por un puerto MIDI")
    _common(p)
    p.add_argument("--port", default=None, help="puerto de salida (subcadena; def. busca 'FP-30')")
    p.add_argument("--null", action="store_true", help="ensayo sin puerto (para probar la temporización)")
    p.add_argument("--speed", type=float, default=1.0, help="factor de velocidad del reloj (solo pruebas)")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("render", help="escribe la interpretación en un .mid")
    p.add_argument("output", help="fichero .mid de salida")
    _common(p, render=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("plan", help="compone sin sonar y muestra la forma")
    _common(p)
    p.add_argument("--json", action="store_true", help="salida JSON")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("ports", help="lista los puertos MIDI de salida")
    p.set_defaults(func=cmd_ports)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(yellow("\ninterrumpido"))
        return 130
    except BrokenPipeError:
        return 0
    except VariationsError as e:
        print(red(f"error: {e}"), file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — un solo mensaje; traceback solo con VARIATIONS_DEBUG
        print(red(f"error inesperado: {e}"), file=sys.stderr)
        if os.environ.get("VARIATIONS_DEBUG"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
