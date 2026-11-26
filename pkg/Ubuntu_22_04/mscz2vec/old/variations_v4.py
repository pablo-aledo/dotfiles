#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  variations_v4.py — Variaciones pianísticas sobre el tema eom.dev            ║
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
║    audit   compone sin sonar y mide melodía, cadencias y tocabilidad         ║
║                                                                              ║
║  USO                                                                         ║
║    python variations_v4.py ports                                             ║
║    python variations_v4.py play --seed 11                                    ║
║    python variations_v4.py play --port "FP-30" --humanize 8                  ║
║    python variations_v4.py render salida.mid --seed 11 --cycles 2            ║
║    python variations_v4.py plan --seed 11                                    ║
║                                                                              ║
║  DEPENDENCIAS   mido (+ python-rtmidi para play); el resto es stdlib         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import collections
import heapq
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__version__ = "0.4.0"

# v3 — cambios respecto a v2 (todos con interruptor en la CLI):
#   1. Voicings de mínimo movimiento (vc_smooth) encadenados entre acordes y secciones.
#   2. Fraseo: rubato por frase en un mapa de tempo de 2 pulsos, dinámica con arco de frase
#      + jerarquía métrica (performer.py), melodía adelantada --lead ms.
#   3. Dispositivos de expectativa en el compás 8: cadencia rota V→bVI, retardo 4–3 y
#      apoyatura larga (expectation_engine.py).
#   4. Ritmo: células con síncopa/tresillos/silencios (rhythm_table.py), hook en tresillo 3+3+2
#      y hemiolia, motivos (motif_table.py) y periodo A A' B A'' en la melodía libre, y ritmo
#      armónico no uniforme (ii–V y dominantes secundarios dentro del compás).
#   + bajo con notas de aproximación a la fundamental siguiente.

# v4 — cambios respecto a v3 (ideas tomadas del conjunto mutopia; cada una con interruptor en la CLI):
#   1. Crítico de melodía (melody_critic.py + melody_selector.py): la melodía libre se genera en
#      --critic N candidatas y suena la mejor según ámbito, grados/saltos, saltos compensados,
#      clímax único y bien situado, cambios de dirección y notas repetidas; además se compensan
#      los saltos grandes (_compensate).
#   2. Cadencias con papel propio (cadence_designer.py + phrase_builder.py): el compás 4 es
#      semicadencia (a veces frigia iv6–V) y el 8 puede cerrar en auténtica, plagal o tercera
#      picarda; la melodía resuelve a la fundamental (resolve_to_cadence). --cadences
#   3. Ritmo (rhythm_table.py, drummer.py): células nuevas de la tabla, textura «comping» con
#      patrones de rhythm_table (bolero, bossa, habanera, son montuno…), polirritmia 3:2 y swing
#      por función (melodía 100 %, bajo 60 %, acompañamiento 80 %). --swing
#   4. Bajo como voz propia (bass_line_composer.py): raíz de mínimo salto (best_root), bajo clásico
#      con quinta/7ª/3ª según tensión y notas de paso, tumbao, pedal de tónica.
#   5. Recursos expresivos (expressive_injector.py): anticipación, escapada, nota de paso
#      cromática, mordente, sforzando en las llegadas y silencio expresivo (caesura) antes del
#      último pulso. --expressive, --silence
#   6. Tocabilidad por mano (playability_auditor.py, ampliada): ≤ 14 semitonos y ≤ 5 notas por
#      mano en cada ataque; primero se reasignan manos (inaudible), luego octava o se omite.
#      --no-playable lo desactiva. Subcomando `audit` con métricas por sección.
#   7. Conducción de voces (harmony_checker.py): vc_smooth penaliza 5ª/8ª paralelas.

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
SEG_BEATS = 2                           # resolución del mapa de tempo (rubato): 2 pulsos
NSEG = SEC_BEATS // SEG_BEATS           # 16 segmentos de tempo por sección
PHRASE_BEATS = 16                       # una frase = 4 compases
MAX_SPAN = 14                           # extensión máxima de una mano en un ataque (semitonos)
MAX_HAND_NOTES = 5                      # una nota por dedo

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
    "cadence-ii-v":    [(5, "m9"), (2, "m7b5"), (7, "7"), (0, "m9")],       # Am9 F#m7b5 B7 Em9
    "submediant-lift": [(8, "maj7"), (10, "maj"), (7, "7"), (0, "m9")],     # Cmaj7 D B7 Em9
    "neapolitan":      [(0, "m9"), (8, "maj7"), (1, "maj7"), (7, "7")],     # Em9 Cmaj7 Fmaj7 B7  (bII mayor; en la menor: Bbmaj7)
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
# células de 4 pulsos (un compás) para el desarrollo libre: (duraciones, tensión rítmica 1–3).
# Duración negativa = silencio. Las síncopas vienen de rhythm_table.py (tresillo cubano,
# cinquillo, habanera/charleston). La tensión se casa con la energía de la sección.
RHYTHMS: List[Tuple[List[float], float]] = [
    ([1, 1, 1, 1], 1.0), ([1, 2, 1], 1.0), ([2, 1, 1], 1.0), ([1, 1, 2], 1.0),
    ([1, 1, 0.5, 0.5, 1], 1.4), ([1.5, 0.5, 1, 1], 1.8),
    ([1.5, 1.5, 1], 2.0),                      # tresillo cubano 3+3+2
    ([0.75, 0.25, 1, 2], 2.0),                 # puntillo corto
    ([1, 0.5, 1, 0.5, 1], 2.2),                # cinquillo
    ([1.5, 0.5, 1.5, 0.5], 2.2),               # habanera / charleston
    ([0.5, 1, 1, 1, 0.5], 2.4),                # anticipa el compás siguiente
    ([-0.5, 0.5, 1, 1, 1], 2.4),               # anacrusa tras silencio de corchea
    ([1, -0.5, 0.5, 1.5, 0.5], 2.8),           # silencio sincopado
    ([1 / 3, 1 / 3, 1 / 3, 1, 2], 2.8),        # tresillo de corcheas
    ([1, 1, 1 / 3, 1 / 3, 1 / 3, 1], 3.0),
]

# contornos motívicos (grados relativos al primero), al estilo de motif_table.py
MOTIFS: List[List[int]] = [
    [0, 2, 4, 7], [0, -2, -4], [0, -1], [0, 4, 3, 2], [0, 2, 1, 3], [0, 1, 2, 3],
    [0, -1, -2, -3], [0, -1, -2, -4], [0, 1, 0, -1], [0, 2, 0], [0, -3, -2],
]


# células de rhythm_table.py (rejilla de corcheas, 'x' = ataque) que v3 no tenía.
# (nombre, patrón, tensión 1–10 de la tabla original)
TABLE_CELLS: List[Tuple[str, str, int]] = [
    ("milonga", "x.x.xx..", 5), ("son-montuno", "x.x..x.x", 5), ("cha-cha-cha", "x.x.x.xx", 4),
    ("bachata", "x.xx.x.x", 4), ("bolero", "x..x.x..", 2), ("rocksteady", ".x.x.x.x", 3),
    ("klezmer", "xx.x.xx.", 6), ("baiao", "x..x.x.x", 5), ("danzon", "x.xx.x.x", 3),
]


def cell_from_pattern(pat: str) -> List[float]:
    """Patrón de corcheas -> duraciones de un compás de 4 pulsos (silencio inicial = negativo)."""
    on = [i for i, c in enumerate(pat) if c == "x"]
    cell: List[float] = [-0.5 * on[0]] if on[0] > 0 else []
    for k, i in enumerate(on):
        end = on[k + 1] if k + 1 < len(on) else len(pat)
        cell.append(0.5 * (end - i))
    return cell


# la tensión 1–10 de la tabla se lleva a la escala 1–3 que usa pick_rhythm
RHYTHMS += [(cell_from_pattern(p), round(1.0 + (t - 1) * 2.0 / 9.0, 2)) for _, p, t in TABLE_CELLS]

# acompañamiento rítmico («comping»): patrones de rhythm_table.py con uso «ritmo_armonico»
# (nombre, patrón en corcheas; 16 caracteres = 2 compases, tensión 1–10, latino)
COMP_PATTERNS: List[Tuple[str, str, int, bool]] = [
    ("pulso", "x...x...", 1, False), ("bolero", "x..x.x..", 2, False),
    ("bossa-nova", "x..x..x...x.x...", 3, False), ("habanera", "x..xx.x.", 3, True),
    ("charleston", "x..x....", 3, False), ("rocksteady", ".x.x.x.x", 3, False),
    ("cha-cha-cha", "x.x.x.xx", 4, True), ("bachata", "x.xx.x.x", 4, False),
    ("tresillo", "x..x..x.", 4, True), ("milonga", "x.x.xx..", 5, True),
    ("son-montuno", "x.x..x.x", 5, True), ("klezmer", "xx.x.xx.", 6, False),
]


def pick_comp(rng: random.Random, energy: float) -> Tuple[str, str, int, bool]:
    target = 1.0 + 8.0 * energy
    ws = [math.exp(-abs(t - target) / 2.0) for _, _, t, _ in COMP_PATTERNS]
    return rng.choices(COMP_PATTERNS, weights=ws)[0]


def pick_rhythm(rng: random.Random, energy: float) -> List[float]:
    target = 1.0 + 2.0 * energy
    ws = [math.exp(-abs(t - target)) for _, t in RHYTHMS]
    return rng.choices([c for c, _ in RHYTHMS], weights=ws)[0]


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


def _parallel_penalty(prev: Sequence[int], cur: Sequence[int], w: float = 8.0) -> float:
    """5ªs/8ªs paralelas entre dos acordes (harmony_checker.check_pair): voces i,j que se mueven en
    la misma dirección conservando el mismo intervalo perfecto."""
    pen = 0.0
    for i in range(len(cur)):
        for j in range(i + 1, len(cur)):
            ic1, ic2 = (prev[j] - prev[i]) % 12, (cur[j] - cur[i]) % 12
            di, dj = cur[i] - prev[i], cur[j] - prev[j]
            if di and dj and di * dj > 0 and ic1 == ic2 and ic2 in (0, 7):
                pen += w
    return pen


def vc_smooth(ch: Chord, prev: Optional[List[int]], lo: int, hi: int, kind: str = "shell") -> List[int]:
    """Voicing de 3 voces dentro de [lo, hi] que minimiza el movimiento respecto a `prev`.

    kind="shell": 3ª, 5ª y 7ª (en tríadas 3ª, 5ª y fundamental) · kind="tri": fundamental, 3ª, 5ª.
    El coste es el movimiento mínimo entre voces (probando las 6 asignaciones), más un castigo
    por saltos en la voz superior y una deriva suave hacia el centro del registro.
    """
    iv = ch.ivs
    idx = [0, 1, 2] if kind == "tri" else ([1, 2, 3] if len(iv) > 3 else [1, 2, 0])
    pcs = [(ch.root + iv[i]) % 12 for i in idx]
    pools = [[p for p in range(lo, hi + 1) if p % 12 == pc] for pc in pcs]
    cands = [sorted(c) for c in itertools.product(*pools) if max(c) - min(c) <= 12]
    mid = (lo + hi) / 2

    def cost(c: List[int]) -> float:
        drift = abs(sum(c) / 3 - mid) * 0.15
        if prev is None:
            return drift
        move = min(sum(abs(a - b) for a, b in zip(perm, prev)) + _parallel_penalty(prev, perm)
                   for perm in itertools.permutations(c))
        return move + 0.5 * abs(c[-1] - prev[-1]) + drift

    return min(cands, key=cost)


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
    ct: bool = False       # fuerza ajuste a nota del acorde aunque sea corta


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
    splits: Dict[int, Tuple[float, Chord]] = field(default_factory=dict)   # compás -> (pulso, 2º acorde)
    tempo_from: float = 0.0     # tempo nominal (sin rubato) al empezar / acabar la sección
    tempo_to: float = 0.0
    device: str = ""            # cadencia del compás 8: 'rota' | '4-3' | 'PAC' | 'plagal' | 'picardy' | ''
    cad4: str = ""              # cadencia del compás 4: 'HC' | 'HC-frigia' | ''
    swing: float = 0.0          # 0 = recto; 0.5–1 = intensidad del swing
    silence: bool = False       # caesura en el último pulso
    fixes: Dict[str, int] = field(default_factory=dict)   # reasignaciones / octavas / omitidas (tocabilidad)

    def bpm_at(self, beat: float) -> float:
        return self.bpms[max(0, min(len(self.bpms) - 1, int(beat // SEG_BEATS)))]

    def sec_at(self, beat: float) -> float:
        if beat < 0:
            return beat * 60.0 / self.bpms[0]
        t, b = 0.0, beat
        for bpm in self.bpms:
            if b >= SEG_BEATS:
                t += SEG_BEATS * 60.0 / bpm
                b -= SEG_BEATS
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
            "bpm": [round(self.tempo_from, 1), round(self.tempo_to, 1)],
            "chords": _bar_names(self),
            "device": self.device,
            "cad4": self.cad4,
            "swing": self.swing,
            "silence": self.silence,
            "fixes": self.fixes,
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
    splits: Dict[int, Tuple[float, Chord]] = field(default_factory=dict)   # cambio de acorde dentro del compás
    sel: Dict[int, int] = field(default_factory=dict)                      # compás -> mitad activa (0/1)
    vclose: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    vtri: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    critic: int = 1                    # candidatas de la melodía libre
    comp: str = ""                     # patrón de la textura «comping»
    bass: str = "root"                 # estilo de bajo: root | classical | tumbao | pedal
    bass_prev: Optional[int] = None    # última raíz del bajo (continuidad entre secciones)
    bass_last: Optional[int] = None    # la deja bass_line() para la sección siguiente


def chord_at(ctx: "Ctx", beat: float) -> Chord:
    """Acorde vigente en un pulso de la sección (respeta los cambios dentro del compás)."""
    b = min(7, max(0, int(beat // 4)))
    sp = ctx.splits.get(b)
    if sp and beat - 4 * b >= sp[0] - 1e-9:
        return sp[1]
    return ctx.chords[b]


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
        elif n.dur >= 0.9 and rng.random() < 0.25:      # mordente (expressive_injector.inject_floreo)
            out += [MN(n.step, n.beat, 0.125, n.semis, n.hand, n.v - 2),
                    MN(n.step - 1, n.beat + 0.125, 0.125, n.semis, n.hand, n.v - 6),
                    replace(n, beat=n.beat + 0.25, dur=n.dur - 0.25)]
        else:
            out.append(n)
    return out


def appoggiatura(voice: List[MN], rng: random.Random, prob: float) -> List[MN]:
    """Apoyatura larga acentuada en las cabezas de compás 4 y 8 (cierre de frase), que resuelve."""
    out: List[MN] = []
    for n in voice:
        if n.dur >= 0.9 and n.beat % 4 == 0 and int(n.beat // 4) % 4 == 3 and rng.random() < prob:
            out += [MN(n.step + 1, n.beat, 0.5, n.semis, n.hand, n.v + 6),
                    replace(n, beat=n.beat + 0.5, dur=n.dur - 0.5, v=n.v - 3, ct=True)]
        else:
            out.append(n)
    return out


def colour(voice: List[MN], rng: random.Random, prob: float) -> List[MN]:
    """Escapada y nota de paso cromática (expressive_injector.inject_escapada / _nota_de_paso_cromatica).

    escapada: tras una nota larga se sube un grado y se abandona por salto descendente.
    cromática: entre dos notas a distancia de tono se intercala el semitono intermedio."""
    out: List[MN] = []
    for k, n in enumerate(voice):
        nxt = voice[k + 1] if k + 1 < len(voice) else None
        if nxt is not None and rng.random() < prob:
            g = nxt.beat - n.beat
            d = (step_to_semis(nxt.step) + nxt.semis) - (step_to_semis(n.step) + n.semis)
            if g >= 1.5 and d <= -5:
                out += [replace(n, dur=g - 0.55),
                        MN(n.step + 1, nxt.beat - 0.5, 0.45, n.semis, n.hand, n.v - 5)]
                continue
            if g >= 1.0 and abs(d) == 2:
                out += [replace(n, dur=g - 0.28),
                        MN(n.step, nxt.beat - 0.25, 0.22, n.semis + (1 if d > 0 else -1), n.hand, n.v - 6)]
                continue
        out.append(n)
    return out


def cadence_melody(voice: List[MN], cad4: str, cad8: str, tonic: int, target: int) -> List[MN]:
    """La melodía llega a la cadencia como en phrase_builder.resolve_to_cadence: en cierres (PAC,
    picarda, plagal) la última nota larga del compás 8 cae en la fundamental del acorde final; en
    la semicadencia del compás 4 la última nota se ajusta al acorde (ct)."""
    voice = list(voice)
    if not voice:
        return voice
    if cad8 in ("PAC", "picardy", "plagal"):
        i = max((k for k, n in enumerate(voice) if n.beat >= 28), key=lambda k: voice[k].beat, default=None)
        if i is not None and voice[i].dur >= 0.9:
            n = voice[i]
            want = (target - tonic) % 12
            best = min((s for s in range(n.step - 4, n.step + 5)
                        if step_to_semis(s) % 12 == want), key=lambda s: abs(s - n.step), default=None)
            if best is not None:
                voice[i] = replace(n, step=best, ct=True)
    if cad4:
        idx = [k for k, n in enumerate(voice) if 12 <= n.beat < 16]
        if idx:
            k = max(idx, key=lambda k: voice[k].beat)
            voice[k] = replace(voice[k], ct=True)
    return voice


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


def _free_bar(rng: random.Random, c0: Chord, c1: Chord, tonic: int, energy: float):
    """Un compás: (células rítmicas, grados). Arranca en nota del acorde c0 y cierra en nota de c1."""
    T = pick_rhythm(rng, energy)
    n = sum(1 for d in T if d > 0)
    start = nearest_ct_step(rng.choice([0, 0, 2, 4, -1, 3]), c0, tonic)
    if rng.random() < 0.5:                                   # motivo con contorno
        steps = [start + x for x in rng.choice(MOTIFS)][:n]
        cur = steps[-1]
        while len(steps) < n:
            cur += rng.choice([-1, -1, 1, 1, 2, -2])
            steps.append(cur)
    else:                                                    # paseo con salto inicial
        a = rng.choice([4, 3, 2, -4, -3, 5])
        b = rng.choice([-2, -1, 1, 2, -3])
        steps = [start, start + a, start + a + b]
        cur = steps[-1]
        while len(steps) < n - 1:
            cur += rng.choice([-2, -1, -1, 1, 1, 2])
            steps.append(cur)
        steps = steps[: n - 1] + [0]
    steps[-1] = nearest_ct_step(steps[-2 if len(steps) > 1 else -1] + rng.choice([-1, 0, 1]), c1, tonic)
    return T, steps


def melody_metrics(p: Sequence[int]) -> Dict[str, float]:
    """Métricas de melody_critic.critique_melody sobre una lista de alturas (semitonos)."""
    iv = [b - a for a, b in zip(p, p[1:])]
    if not iv:
        return {}
    n = len(p)
    unresolved = 0
    for i in range(len(iv) - 1):
        if abs(iv[i]) >= 5 and not (iv[i] * iv[i + 1] < 0 and abs(iv[i + 1]) <= 2):
            unresolved += 1
    nz = [x for x in iv if x != 0]
    dirs = [1 if x > 0 else -1 for x in nz]
    changes = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    top = max(p)
    return {"n": n, "range": top - min(p),
            "step_ratio": sum(1 for x in iv if abs(x) <= 2) / len(iv),
            "unresolved": unresolved,
            "peak_n": sum(1 for x in p if x == top), "peak_pos": p.index(top) / max(1, n - 1),
            "dir_ratio": changes / max(1, len(dirs) - 1),
            "repeat_ratio": sum(1 for x in iv if x == 0) / len(iv)}


def score_melody(p: Sequence[int]) -> float:
    """Puntuación 0–1 (aprox.) con los umbrales de melody_critic: ámbito cantábile 7–17 st, 55–85 % de
    grados, saltos compensados, clímax único hacia el 55–80 %, contorno que alterna, pocas repeticiones."""
    if len(p) < 6:
        return 0.5
    m = melody_metrics(p)
    s = 1.0
    s -= 0.9 * max(0.0, 0.55 - m["step_ratio"]) + 0.5 * max(0.0, m["step_ratio"] - 0.85)
    s -= 0.08 * m["unresolved"]
    s -= 0.04 * max(0, m["range"] - 17) + 0.04 * max(0, 7 - m["range"])
    s -= 0.10 * max(0, m["peak_n"] - 2)
    if not 0.45 <= m["peak_pos"] <= 0.85:
        s -= 0.25
    s -= max(0.0, m["dir_ratio"] - 0.6) + max(0.0, 0.2 - m["dir_ratio"])
    s -= max(0.0, m["repeat_ratio"] - 0.3)
    return s


def _compensate(out: List[MN]) -> List[MN]:
    """Tras un salto ≥ 5 st la nota siguiente se mueve un grado en sentido contrario (melody_critic:
    «salto grande sin compensar»). No toca la primera ni la última nota de cada compás (anclas)."""
    first: Dict[int, int] = {}
    last: Dict[int, int] = {}
    for i, n in enumerate(out):
        b = int(n.beat // 4)
        first.setdefault(b, i)
        last[b] = i
    anchors = set(first.values()) | set(last.values())
    for i in range(len(out) - 2):
        a, b, c = out[i], out[i + 1], out[i + 2]
        d1 = step_to_semis(b.step) - step_to_semis(a.step)
        if abs(d1) >= 5 and (i + 2) not in anchors:
            d2 = step_to_semis(c.step) - step_to_semis(b.step)
            if d1 * d2 >= 0 or abs(d2) > 2:
                out[i + 2] = replace(c, step=b.step - (1 if d1 > 0 else -1))
    return out


def v_free(ctx: Ctx):
    """Desarrollo libre: genera ctx.critic candidatas y toca la que mejor puntúa (melody_selector)."""
    best, best_s = None, -1e9
    for _ in range(max(1, ctx.critic)):
        out = _compensate(_free_once(ctx))
        pitches = [step_to_semis(n.step) + n.semis for n in out]
        sc = score_melody(pitches)
        strong = [n for n in out if n.beat % 4 == 0 or n.dur >= 0.9]
        if strong:      # las notas fuertes deben ser del acorde vigente
            ok = sum(1 for n in strong if (ctx.tonic + step_to_semis(n.step) + n.semis) % 12
                     in chord_at(ctx, n.beat).pcs) / len(strong)
            sc += 0.4 * ok
        if sc > best_s:
            best, best_s = out, sc
    return "free", [fit(best)]


def _free_once(ctx: Ctx) -> List[MN]:
    """Una candidata de desarrollo libre con periodo: compases 5–7 repiten 1–3 (secuenciados) el 60 %."""
    rng, out = ctx.rng, []
    parallel = rng.random() < 0.6
    shift = rng.choice([0, 1, -1, 2])
    bars: Dict[int, Tuple[List[float], List[int]]] = {}
    for bar in range(8):
        c0, c1 = ctx.chords[bar], chord_at(ctx, bar * 4 + 3.5)
        if parallel and bar in (4, 5, 6):
            T, prev = bars[bar - 4]
            steps = [s + shift for s in prev]
            steps[-1] = nearest_ct_step(steps[-1], c1, ctx.tonic)
        else:
            T, steps = _free_bar(rng, c0, c1, ctx.tonic, ctx.energy)
        bars[bar] = (T, steps)
        t, k = 0.0, 0
        for d in T:
            if d < 0:
                t += -d
                continue
            out.append(MN(max(-4, min(9, steps[k])), bar * 4 + round(t, 4), d * 0.95))
            t += d
            k += 1
    return out


def _by_bar(notes: List[MN]) -> Dict[int, List[MN]]:
    d: Dict[int, List[MN]] = {}
    for n in notes:
        d.setdefault(int(n.beat // 4), []).append(n)
    return d


def v_tresillo(ctx: Ctx):
    """El hook sobre la célula 3+3+2 del tresillo cubano (la 4ª nota, si existe, en el 4º pulso y medio)."""
    out: List[MN] = []
    for bar, ns in sorted(_by_bar(theme()).items()):
        offs = [0.0, 1.5, 3.0, 3.5][: len(ns)]
        for i, (n, o) in enumerate(zip(ns, offs)):
            nxt = offs[i + 1] if i + 1 < len(offs) else 4.0
            out.append(replace(n, beat=bar * 4 + o, dur=(nxt - o) * 0.95, v=n.v + (4 if o == 1.5 else 0)))
    return "tresillo", [out]


def v_hemiola(ctx: Ctx):
    """Las notas de cada compás repartidas por igual: 3 contra 4 (hemiolia) en los compases de tres notas."""
    out: List[MN] = []
    for bar, ns in sorted(_by_bar(theme()).items()):
        w = 4.0 / len(ns)
        for i, n in enumerate(ns):
            out.append(replace(n, beat=bar * 4 + i * w, dur=w * 0.95))
    return "hemiola", [out]


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
    "tresillo": v_tresillo, "hemiola": v_hemiola,
}


def harmonic_adjust(p: int, chord: Chord, m: MN) -> int:
    pc = p % 12
    if chord.qual in ("7", "7b9"):                    # sensible: sube el grado que choca con la 3ª
        third = (chord.root + 4) % 12
        if pc == (third - 1) % 12:
            p += 1
            pc = (pc + 1) % 12
    if (m.dur >= 0.9 or m.ct) and pc not in chord.pcs:             # evita roces en notas largas
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
            chord = chord_at(ctx, m.beat)
            p = harmonic_adjust(p, chord, m)
            out.append(Note(m.beat, m.dur, clamp_pitch(p), m.v, m.hand, "mel"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BAJO (bass_line_composer.py)
# ══════════════════════════════════════════════════════════════════════════════

BASS_LO, BASS_HI = 28, 52
BASS_STYLES = ("root", "classical", "tumbao", "pedal")
# tumbao (LatinBassGenerator.TUMBAO_PATTERN_44): (pulso, tono, duración, offset de velocidad);
# tono 0 = raíz, 1 = quinta, 3 = aproximación cromática a la raíz siguiente
TUMBAO = [(0.5, 0, 0.5, 2), (1.0, 0, 0.5, 0), (2.5, 1, 0.5, -3), (3.0, 0, 0.5, 0), (3.5, 3, 0.5, -5)]


def _broot(pc: int, prev: Optional[int], lo: int = BASS_LO, hi: int = BASS_HI) -> int:
    """best_root: la octava de la fundamental que menos se mueve desde la nota anterior."""
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc % 12]
    ref = prev if prev is not None else (lo + hi) // 2
    return min(cands, key=lambda p: (abs(p - ref), p))


def bass_line(ctx: "Ctx", allow: Sequence[str] = BASS_STYLES) -> List[Note]:
    """Línea de bajo de 8 compases en la mano izquierda con el estilo ctx.bass (si está en `allow`)."""
    style = ctx.bass if ctx.bass in allow else "root"
    minor = {(ctx.tonic + i) % 12 for i in MINOR}
    out: List[Note] = []
    prev, roots = ctx.bass_prev, []
    for ch in ctx.chords:
        prev = _broot(ch.root, prev)
        roots.append(prev)
    pedal_p = _broot(ctx.tonic, ctx.bass_prev)
    for b, ch in enumerate(ctx.chords):
        t, r = 4 * b, roots[b]
        nr = roots[b + 1] if b < 7 else None
        iv = ch.ivs
        fifth = r + iv[2] - (12 if r + iv[2] > BASS_HI else 0)
        third = r + iv[1] - (12 if r + iv[1] > BASS_HI else 0)
        sev = r + (iv[3] if len(iv) > 3 else iv[0]) - (12 if r + (iv[3] if len(iv) > 3 else iv[0]) > BASS_HI else 0)
        if style == "pedal":
            if b % 2 == 0:
                out.append(Note(t, 7.8, pedal_p, 4, "L"))
        elif style == "classical":
            weak = sev if (ctx.energy > 0.6 and len(iv) > 3) else (third if ctx.energy > 0.3 else fifth)
            out += [Note(t, 2.0 * 0.9, r, 4, "L"), Note(t + 2, 0.9, fifth, -6, "L")]
            pas: List[int] = []
            if nr is not None and ctx.energy > 0.4 and abs(nr - r) > 3:     # notas de paso escalares
                span = range(r + 1, nr) if nr > r else range(r - 1, nr, -1)
                pas = [p for p in span if p % 12 in minor][:2]
            if pas:
                sub = 1.0 / (len(pas) + 1)
                out.append(Note(t + 3, sub * 0.9, weak, -11, "L"))
                for j, p in enumerate(pas):
                    out.append(Note(t + 3 + (j + 1) * sub, sub * 0.85, p, -12, "L"))
            else:
                out.append(Note(t + 3, 0.9, weak, -11, "L"))
        elif style == "tumbao":
            tgt = nr if nr is not None else r
            appr = tgt + (1 if tgt < r else -1)
            tones = {0: r, 1: fifth, 3: appr}
            for off, tone, dur, vel in TUMBAO:
                out.append(Note(t + off, dur * 0.85, tones[tone], vel, "L"))
        else:
            out.append(Note(t, 3.9, r, 4, "L"))
    ctx.bass_last = pedal_p if style == "pedal" else roots[-1]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  TEXTURAS
# ══════════════════════════════════════════════════════════════════════════════


def _hand(ctx: Ctx) -> str:
    return "R" if ctx.high else "L"


def vcl(ctx: Ctx, b: int, ch: Chord) -> List[int]:
    """Estructura 3ª-5ª-7ª de mínimo movimiento para el compás b (mitad activa según ctx.sel)."""
    return ctx.vclose.get((b, ctx.sel.get(b, 0))) or vc_close(ch, 52)


def vtr(ctx: Ctx, b: int, ch: Chord) -> List[int]:
    return ctx.vtri.get((b, ctx.sel.get(b, 0))) or vc_tri(ch, 60 if ctx.high else 50)


def approach(ctx: Ctx, b: int, frm: int, low: int) -> Optional[int]:
    """Nota de aproximación (cromática o por tono) a la fundamental del compás siguiente."""
    if ctx.high or b >= 7 or ctx.energy < 0.3 or ctx.rng.random() > 0.55:
        return None
    tgt = stack([ctx.chords[b + 1].root], low)[0]
    return tgt + ctx.rng.choices([-1, 1, -2, 2], weights=[4, 3, 2, 1])[0]


def render_texture(fn: Callable, ctx: Ctx) -> List[Note]:
    """Genera la textura; en los compases con cambio interno de acorde vuelve a generar el compás con
    el 2º acorde y empalma las dos mitades (las notas largas se re-atacan en el cambio)."""
    notes = fn(ctx)
    for b in sorted(ctx.splits):
        s, X = ctx.splits[b]
        chs = list(ctx.chords)
        chs[b] = X
        n2 = fn(replace(ctx, chords=chs, sel={**ctx.sel, b: 1}, splits={}))
        lo, hi, cut_ = 4 * b, 4 * b + 4, 4 * b + s
        kept: List[Note] = []
        for n in notes:
            if lo <= n.beat < hi:
                if n.beat >= cut_:
                    continue
                if n.beat + n.dur > cut_:
                    n = replace(n, dur=max(0.05, cut_ - n.beat - 0.03))
            kept.append(n)
        for n in n2:
            if not lo <= n.beat < hi:
                continue
            if n.beat >= cut_:
                kept.append(n)
            else:
                rem = min(n.beat + n.dur, hi) - cut_
                if rem >= 0.8:                                   # nota sostenida: se re-ataca
                    kept.append(replace(n, beat=cut_, dur=rem - 0.03))
        notes = kept
    return notes


LOW_PATTERN = [0, 1, 3, 2, 3, 1, 3, 1]      # E1 B1 G2 F#2 G2 B1 G2 B1 sobre vc_low


def tex_chorale(ctx: Ctx) -> List[Note]:
    out, h = [], _hand(ctx)
    if not ctx.high:
        out += bass_line(ctx, ("root", "classical", "pedal"))
    for b, ch in enumerate(ctx.chords):
        t0 = b * 4
        tones = vcl(ctx, b, ch)
        for k in range(2):
            for p in tones:
                out.append(Note(t0 + 2 * k, 1.9, p, 0 if k == 0 else -5, h))
    return out


def _arp(ctx: Ctx, per_bar: int, dur: float) -> List[Note]:
    out, h, step = [], _hand(ctx), BEATS_PER_BAR / per_bar
    for b, ch in enumerate(ctx.chords):
        lv = vc_low(ch, ctx.near)
        ap = approach(ctx, b, lv[LOW_PATTERN[(per_bar - 2) % 8]], 28)   # aproximación al bajo siguiente
        for k in range(per_bar):
            strong = k % (per_bar // 2) == 0
            off = 5 if strong else (-3 if k % 2 else 0)
            pitch = ap if (ap is not None and k == per_bar - 1) else lv[LOW_PATTERN[k % 8]]
            out.append(Note(b * 4 + k * step, dur, pitch, off, h))
        if not ctx.high:                                   # nota interna sostenida en la derecha
            out.append(Note(b * 4, 3.9, vcl(ctx, b, ch)[-1], -10, "R"))
    return out


def tex_rolling(ctx: Ctx) -> List[Note]:
    return _arp(ctx, 8, 0.7)


def tex_arp16(ctx: Ctx) -> List[Note]:
    return _arp(ctx, 16, 0.34)


def tex_alberti(ctx: Ctx) -> List[Note]:
    out, h = [], _hand(ctx)
    if not ctx.high:
        out += bass_line(ctx, ("root", "classical", "tumbao"))
    for b, ch in enumerate(ctx.chords):
        v = vtr(ctx, b, ch)
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
        ap = approach(ctx, b, fifth, 33)
        if ap is not None:                                  # paso al bajo del compás siguiente
            out.append(Note(t + 3, 0.9, ap, -2, "L"))
        for beat in (1, 3):
            for p in vcl(ctx, b, ch):
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
    if not ctx.high:
        out += bass_line(ctx, ("root", "pedal"))
    for b, ch in enumerate(ctx.chords):
        t0, iv, r = b * 4, ch.ivs, ch.root
        if ctx.high:
            out.append(Note(t0, 3.9, stack([r], ctx.near)[0], 8, h))
        ext = iv[4] if len(iv) > 4 else (iv[3] if len(iv) > 3 else iv[0])
        for p in stack([(r + ext) % 12, (r + iv[1]) % 12, (r + iv[2]) % 12], 62):
            out.append(Note(t0, 3.9, p, -6, "R"))
        for pos in sorted(ctx.rng.sample([0.5, 1.25, 2, 2.5, 3, 3.5], 3)):
            pc = ctx.rng.choice(sorted(ch.pcs))
            p = 79 + ((pc - 79) % 12) + 12 * ctx.rng.choice([0, 1, 1])
            out.append(Note(t0 + pos, 2.0, p, 0, "R"))
    return out


def tex_comping(ctx: Ctx) -> List[Note]:
    """Acordes sobre un patrón de rhythm_table.py (ctx.comp, corcheas; 16 caracteres = 2 compases)."""
    out, h = [], _hand(ctx)
    pat = ctx.comp or "x..x..x."
    nb = max(1, len(pat) // 8)
    if not ctx.high:
        out += bass_line(ctx, BASS_STYLES)
    for b, ch in enumerate(ctx.chords):
        seg = pat[(b % nb) * 8:(b % nb) * 8 + 8]
        on = [i for i, c in enumerate(seg) if c == "x"]
        tones = vcl(ctx, b, ch)
        for k, i in enumerate(on):
            end = on[k + 1] if k + 1 < len(on) else 8
            dur = min(0.5 * (end - i), 1.5) * 0.9
            vel = 4 if i == 0 else (0 if i % 2 == 0 else -4)
            for p in tones:
                out.append(Note(b * 4 + 0.5 * i, dur, p, vel, h))
    return out


def tex_poly32(ctx: Ctx) -> List[Note]:
    """Polirritmia 3:2 (expressive_injector.inject_polirritmia): arpegio de tresillos de negra
    (3 notas cada 2 pulsos) contra la melodía binaria."""
    out, h = [], _hand(ctx)
    order = [0, 1, 3, 2, 3, 1]
    for b, ch in enumerate(ctx.chords):
        lv = vc_low(ch, ctx.near)
        for half in range(2):
            for k in range(3):
                pitch = lv[order[half * 3 + k]]
                out.append(Note(b * 4 + 2 * half + k * (2.0 / 3.0), 0.6, pitch, 4 if k == 0 else -3, h))
        if not ctx.high:
            out.append(Note(b * 4, 3.9, vcl(ctx, b, ch)[-1], -10, "R"))
    return out


# nombre -> (función, re-pedal cada N pulsos; 0 = sin pedal). El original cambia pedal cada compás.
TEXTURES: Dict[str, Tuple[Callable, int]] = {
    "chorale": (tex_chorale, 4), "rolling": (tex_rolling, 4), "arp16": (tex_arp16, 4),
    "alberti": (tex_alberti, 4), "stride": (tex_stride, 4), "driving": (tex_driving, 4),
    "ambient": (tex_ambient, 4), "comping": (tex_comping, 4), "poly32": (tex_poly32, 4),
}
HIGH_OK = {"rolling", "arp16", "alberti", "chorale", "ambient", "comping", "poly32"}
SWING_OK = {"rolling", "alberti", "driving", "stride", "chorale", "comping"}   # rejilla de corcheas
NO_SWING = {"tresillo", "hemiola", "diminished", "canon"}

# ══════════════════════════════════════════════════════════════════════════════
#  EXPRESIVIDAD Y TOCABILIDAD (posproceso por sección)
# ══════════════════════════════════════════════════════════════════════════════


def anticipate(notes: List[Note], ctx: Ctx, prob: float, rng: random.Random) -> int:
    """Anticipación (expressive_injector.inject_anticipacion): en un cambio de acorde, la nota de
    melodía que llega con el acorde nuevo —y no pertenece al anterior— se adelanta media negra."""
    mel = sorted((n for n in notes if n.role == "mel"), key=lambda n: n.beat)
    times = {4.0 * b for b in range(1, 8)} | {4.0 * b + s for b, (s, _) in ctx.splits.items()}
    done = 0
    for t in sorted(times):
        if rng.random() >= prob:
            continue
        before, after = chord_at(ctx, t - 0.01), chord_at(ctx, t)
        if before.pcs == after.pcs:
            continue
        cur = [n for n in mel if abs(n.beat - t) < 1e-6]
        prev = [n for n in mel if n.beat < t - 1e-6]
        if len(cur) != 1 or not prev:
            continue
        c, p = cur[0], prev[-1]
        if (c.pitch % 12 in after.pcs and c.pitch % 12 not in before.pcs and p.dur >= 1.0
                and p.beat + p.dur >= t - 0.2 and c.dur >= 0.4):
            c.beat -= 0.5
            c.dur += 0.5
            c.vel += 3
            p.dur = max(0.2, c.beat - p.beat - 0.03)
            done += 1
    return done


def apply_swing(notes: List[Note], amount: float, ratio: float = 2.0 / 3.0) -> None:
    """Swing de corcheas (bass_line_composer.swing_offset) con el reparto por función de
    drummer.SWING_FACTOR_BY_FUNC: melodía/subdivisión 100 %, bajo 60 %, acompañamiento 80 %."""
    base = (ratio - 0.5) * amount
    for n in notes:
        if abs(n.beat % 1.0 - 0.5) < 1e-6:
            w = 1.0 if n.role == "mel" else (0.6 if n.pitch < 48 else 0.8)
            d = base * w
            n.beat += d
            if n.dur <= 0.55:
                n.dur = max(0.1, n.dur - d)


def cut_for_silence(notes: List[Note], at: float = 31.0) -> None:
    """Caesura: nada suena desde `at` hasta el final de la sección (silencio expresivo)."""
    kept = []
    for n in notes:
        if n.beat >= at - 1e-6:
            continue
        if n.beat + n.dur > at - 0.04:
            n.dur = at - 0.04 - n.beat
            if n.dur < 0.05:
                continue
        kept.append(n)
    notes[:] = kept


def _hand_ok(ns: Sequence[Note]) -> bool:
    if not ns:
        return True
    ps = [n.pitch for n in ns]
    return len(ns) <= MAX_HAND_NOTES and max(ps) - min(ps) <= MAX_SPAN


def _best_cut(g: List[Note]) -> Optional[Tuple[int, int]]:
    """Reparto de un grupo (ordenado por altura) en mano izquierda (graves) y derecha (agudos) que
    cumple la extensión y el nº de dedos con el menor número de cambios de etiqueta."""
    best = None
    for c in range(len(g) + 1):
        ls, rs = g[:c], g[c:]
        if _hand_ok(ls) and _hand_ok(rs):
            ch = sum(1 for n in ls if n.hand != "L") + sum(1 for n in rs if n.hand != "R")
            if best is None or ch < best[0]:
                best = (ch, c)
    return best


def playability_issues(notes: Sequence[Note]) -> Tuple[int, int]:
    """(ataques con extensión > MAX_SPAN en una mano, ataques con más de MAX_HAND_NOTES en una mano)."""
    groups: Dict[Tuple[float, str], List[int]] = collections.defaultdict(list)
    for n in notes:
        groups[(round(n.beat, 3), n.hand)].append(n.pitch)
    span = sum(1 for ps in groups.values() if max(ps) - min(ps) > MAX_SPAN)
    many = sum(1 for ps in groups.values() if len(ps) > MAX_HAND_NOTES)
    return span, many


def fix_playability(notes: List[Note]) -> Tuple[List[Note], Dict[str, int]]:
    """Hace tocables los ataques simultáneos (≤ MAX_SPAN semitonos y ≤ MAX_HAND_NOTES por mano).

    1) reasigna manos por altura (no cambia el sonido); 2) si no basta, mueve una octava hacia la
    melodía la nota de acompañamiento más alejada; 3) como último recurso la omite. La melodía
    nunca se toca. (playability_auditor.py no comprueba la extensión por mano: esto la añade.)"""
    stats = {"relabeled": 0, "shifted": 0, "dropped": 0}
    groups: Dict[float, List[Note]] = collections.defaultdict(list)
    for n in notes:
        groups[round(n.beat, 3)].append(n)
    dropped: set = set()
    for g in groups.values():
        if len(g) < 2 or (_hand_ok([n for n in g if n.hand == "L"]) and _hand_ok([n for n in g if n.hand == "R"])):
            continue
        g = sorted(g, key=lambda n: n.pitch)
        for _ in range(10):
            cut = _best_cut(g)
            if cut:
                for i, n in enumerate(g):
                    want = "L" if i < cut[1] else "R"
                    if n.hand != want:
                        n.hand = want
                        stats["relabeled"] += 1
                break
            movable = [n for n in g if n.role != "mel"]
            if not movable:
                break
            mel = [n.pitch for n in g if n.role == "mel"] or [n.pitch for n in g]
            anchor = sum(mel) / len(mel)
            n = max(movable, key=lambda x: abs(x.pitch - anchor))
            newp = n.pitch + (-12 if n.pitch > anchor else 12)
            if 28 <= newp <= 100 and all(newp != m.pitch for m in g if m is not n):
                n.pitch = newp
                stats["shifted"] += 1
            else:
                dropped.add(id(n))
                g = [m for m in g if m is not n]
                stats["dropped"] += 1
            g.sort(key=lambda x: x.pitch)
    return [n for n in notes if id(n) not in dropped], stats


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
        texture=[("rolling", 3), ("alberti", 2), ("stride", 1), ("chorale", 1), ("comping", 1)]),
    "development": dict(
        melody=[("ornamented", 2), ("inverted", 2), ("sequence", 3), ("off-beat", 2), ("canon", 2),
                ("free", 3), ("thirds", 2), ("left-hand", 2), ("tresillo", 2), ("hemiola", 1)],
        harmony=[("true-dominant", 2), ("relative-major", 2), ("andalusian", 2), ("neapolitan", 1),
                 ("cadence-ii-v", 2), ("submediant-lift", 2), ("original", 1)],
        texture=[("rolling", 2), ("alberti", 2), ("stride", 2), ("chorale", 1), ("arp16", 1),
                 ("driving", 1), ("comping", 3), ("poly32", 1)]),
    "build": dict(
        melody=[("diminished", 2), ("sequence", 3), ("free", 2), ("thirds", 2), ("original", 1),
                ("tresillo", 2), ("hemiola", 1)],
        harmony=[("submediant-lift", 3), ("cadence-ii-v", 2), ("true-dominant", 2), ("andalusian", 1)],
        texture=[("rolling", 2), ("arp16", 3), ("driving", 2), ("alberti", 1), ("comping", 2),
                 ("poly32", 1)]),
    "breakdown": dict(
        melody=[("augmented", 2), ("original", 2), ("left-hand", 3), ("off-beat", 1)],
        harmony=[("andalusian", 2), ("original", 2), ("neapolitan", 1), ("relative-major", 1)],
        texture=[("ambient", 4), ("chorale", 2)]),
    "climax": dict(
        melody=[("octaves", 4), ("thirds", 1), ("diminished", 2), ("ornamented", 1), ("original", 1),
                ("tresillo", 1), ("hemiola", 1)],
        harmony=[("true-dominant", 3), ("andalusian", 2), ("neapolitan", 2), ("original", 1)],
        texture=[("driving", 4), ("arp16", 3), ("stride", 1), ("comping", 1)]),
    "settle": dict(
        melody=[("original", 3), ("augmented", 1), ("left-hand", 2), ("ornamented", 1)],
        harmony=[("original", 3), ("relative-major", 1), ("cadence-ii-v", 2)],
        texture=[("chorale", 3), ("rolling", 2), ("ambient", 1), ("comping", 1), ("poly32", 1)]),
    "coda": dict(
        melody=[("original", 1)],
        harmony=[("original", 1)],
        texture=[("chorale", 2), ("rolling", 1)]),
}
TONIC_E = 4


def tempo_for(energy: float) -> float:
    return 96.0 + 20.0 * energy      # 96..116 bpm


SPLIT_PROB = {"intro": 0.0, "theme": 0.10, "development": 0.35, "build": 0.40,
              "breakdown": 0.0, "climax": 0.30, "settle": 0.15, "coda": 0.0}
DEVICE_PROB = {   # (cadencia rota V→bVI, retardo 4–3 sobre la dominante) según el tipo de sección
    "intro": (0.0, 0.25), "theme": (0.0, 0.20), "development": (0.20, 0.30), "build": (0.35, 0.25),
    "breakdown": (0.10, 0.20), "climax": (0.15, 0.30), "settle": (0.0, 0.0), "coda": (0.0, 0.0)}
DOMINANTS = ("7", "7sus4", "7b9")
# cierre del compás 8 cuando termina en dominante y no hay dispositivo: (auténtica, plagal, picarda)
# (fuerza de cada cadencia: cadence_designer.CADENCE_TYPES — PAC 1.0, picarda 0.8, plagal 0.6)
CAD8_PROB = {"theme": (0.15, 0.05, 0.0), "development": (0.12, 0.06, 0.02), "breakdown": (0.15, 0.15, 0.10),
             "climax": (0.30, 0.0, 0.15), "settle": (0.45, 0.0, 0.10), "intro": (0.0, 0.0, 0.0),
             "build": (0.0, 0.0, 0.0)}
PHRYG_PROB = {"intro": 0.05, "theme": 0.10, "development": 0.18, "build": 0.15, "breakdown": 0.10,
              "climax": 0.10, "settle": 0.08}                 # semicadencia frigia iv6–V en el compás 4
SWING_PROB = {"theme": 0.15, "development": 0.35, "build": 0.15, "breakdown": 0.30, "climax": 0.10,
              "settle": 0.20}
SIL_PROB = {"intro": 0.10, "development": 0.12, "build": 0.40, "breakdown": 0.25}
BASS_WEIGHTS = {"intro": [("pedal", 3), ("root", 3), ("classical", 1)],
                "theme": [("root", 3), ("classical", 2)],
                "development": [("classical", 3), ("root", 2), ("tumbao", 1)],
                "build": [("classical", 2), ("root", 2), ("pedal", 1)],
                "breakdown": [("pedal", 3), ("root", 2), ("classical", 1)],
                "climax": [("root", 3), ("classical", 1)],
                "settle": [("classical", 2), ("root", 1)]}
# factor de tempo por segmento de 2 pulsos dentro de una frase de 16: arranque contenido,
# aceleración en el cuerpo, ritardando en el cierre (reglas de performer.build_warp)
RUBATO = [0.985, 1.0, 1.01, 1.012, 1.01, 1.0, 0.97, 0.93]


def _wpick(rng: random.Random, options: List[Tuple[str, int]]) -> str:
    return rng.choices([n for n, _ in options], weights=[w for _, w in options])[0]


class Composer:
    """Genera las secciones una a una (8 compases cada una), de forma perezosa."""

    def __init__(self, seed: int, cycles: int = 2, coda: bool = True, tempo_scale: float = 1.0,
                 rubato: float = 0.6, phrasing: float = 0.8, hrhythm: float = 1.0, devices: float = 1.0,
                 swing: float = 1.0, critic: int = 6, cadences: float = 1.0, expressive: float = 1.0,
                 silence: float = 1.0, playable: bool = True):
        self.seed = seed
        self.cycles = cycles
        self.coda = coda
        self.scale = tempo_scale
        self.rubato = rubato
        self.phrasing = phrasing
        self.hrhythm = hrhythm
        self.devices = devices
        self.swing, self.critic, self.cadences = swing, critic, cadences
        self.expressive, self.silence, self.playable = expressive, silence, playable
        self.rng = random.Random(seed)
        self.rng2 = random.Random(seed ^ 0x5A5A5A)      # decisiones v4 (no altera el flujo del RNG principal)
        self.bprev: Optional[int] = None
        self.prev: Dict[str, Optional[str]] = {"melody": None, "harmony": None, "texture": None}
        self.bpm = 100.0 * tempo_scale
        self.vprev: Dict[str, Optional[List[int]]] = {"close": None, "tri": None}

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

    # ── ritmo armónico: ii–V y dominantes secundarios dentro del compás ───────
    def _plan_splits(self, kind: str, chords: List[Chord]) -> Dict[int, Tuple[float, Chord]]:
        p = SPLIT_PROB.get(kind, 0.0) * self.hrhythm
        out: Dict[int, Tuple[float, Chord]] = {}
        if p <= 0:
            return out
        for b in range(7):                                   # el compás 8 se reserva a los dispositivos
            if b - 1 in out or self.rng.random() >= p:
                continue
            cur, nxt = chords[b], chords[b + 1]
            if nxt.qual in DOMINANTS:                        # ii° antes de la dominante
                x, s = Chord((nxt.root + 7) % 12, "m7b5"), 2.0
            else:                                            # V7 del acorde siguiente
                x, s = Chord((nxt.root + 7) % 12, "7"), (3.0 if self.rng.random() < 0.3 else 2.0)
            if x.root == cur.root or (x.root == nxt.root and x.qual == nxt.qual):
                continue
            out[b] = (s, x)
        return out

    # ── dispositivos de expectativa sobre la dominante final ──────────────────
    def _plan_device(self, kind: str, chords: List[Chord], tonic: int) -> Optional[str]:
        pd, ps = DEVICE_PROB.get(kind, (0.0, 0.0))
        c = chords[7]
        if self.devices <= 0 or c.qual not in ("7", "7sus4") or c.root != (tonic + 7) % 12:
            return None
        r = self.rng.random()
        if r < pd * self.devices:
            return "rota"
        if r < (pd + ps) * self.devices:
            return "4-3"
        return None

    def _plan_cadences(self, kind: str, chords: List[Chord], tonic: int, nxt: int,
                       splits: Dict[int, Tuple[float, Chord]], device: Optional[str],
                       is_coda: bool, locked: bool) -> Tuple[str, str]:
        """Papel de cada cadencia (cadence_designer): compás 4 = semicadencia (a veces frigia iv6–V);
        compás 8 = cierre auténtico / plagal / picardo si el compás termina en dominante y no hay
        dispositivo de expectativa. Modifica `chords` y `splits`; devuelve (cad4, cad8)."""
        r2, c = self.rng2, self.cadences
        if is_coda:
            if c > 0 and r2.random() < 0.5 * c:
                chords[7] = Chord(tonic, "maj")                # tercera picarda en el último acorde
                return "", "picardy"
            return "", ""
        if locked or c <= 0:
            return "", (device or "")
        V = (tonic + 7) % 12
        cad4 = ""
        if chords[3].root == V and chords[3].qual in DOMINANTS and 3 not in splits:
            if r2.random() < PHRYG_PROB.get(kind, 0.0) * c:
                dom = chords[3]
                chords[3] = Chord((tonic + 5) % 12, "m7")      # iv … V
                splits[3] = (2.0, dom)
                cad4 = "HC-frigia"
            else:
                cad4 = "HC"
        cad8 = device or ""
        if not device:
            tgt = nxt if kind == "settle" else tonic
            Vt = (tgt + 7) % 12
            if chords[7].root == Vt and chords[7].qual in DOMINANTS and 7 not in splits:
                pac, pla, pic = CAD8_PROB.get(kind, (0.0, 0.0, 0.0))
                x = r2.random()
                if x < pac * c:
                    chords[7] = Chord(Vt, "7")
                    splits[7] = (2.0, Chord(tgt, "m9"))
                    cad8 = "PAC"
                elif x < (pac + pla) * c and tgt == tonic:
                    chords[7] = Chord((tonic + 5) % 12, "m7")
                    splits[7] = (2.0, Chord(tonic, "m9"))
                    cad8 = "plagal"
                elif x < (pac + pla + pic) * c:
                    chords[7] = Chord(Vt, "7")
                    splits[7] = (2.0, Chord(tgt, "maj"))
                    cad8 = "picardy"
        return cad4, cad8

    def _pick_bass(self, kind: str, mel_key: str, tex_key: str, locked: bool, latin: bool) -> str:
        """Estilo de bajo: tumbao con melodías/patrones sincopados, si no según el tipo de sección."""
        if locked:
            return "root"
        r2 = self.rng2
        if mel_key in ("tresillo", "hemiola") and r2.random() < 0.7:
            return "tumbao"
        if tex_key == "comping" and latin and r2.random() < 0.6:
            return "tumbao"
        return _wpick(r2, BASS_WEIGHTS.get(kind, [("root", 1)]))

    def _plan_voicings(self, ctx: Ctx) -> None:
        """Encadena voicings de mínimo movimiento a lo largo de toda la sección (y de la anterior)."""
        wc, wt = ((62, 80), (60, 76)) if ctx.high else ((46, 64), (46, 62))
        pc_, pt_ = self.vprev["close"], self.vprev["tri"]
        for b, c in enumerate(ctx.chords):
            segs = [(0, c)] + ([(1, ctx.splits[b][1])] if b in ctx.splits else [])
            for h, ch in segs:
                pc_ = vc_smooth(ch, pc_, *wc, "shell")
                pt_ = vc_smooth(ch, pt_, *wt, "tri")
                ctx.vclose[(b, h)], ctx.vtri[(b, h)] = pc_, pt_
        self.vprev = {"close": pc_, "tri": pt_}

    def _tempo_curve(self, kind: str, energy: float, start: float, target: float, is_coda: bool) -> List[float]:
        k = self.rubato * (1.2 - 0.4 * energy)               # más rubato cuanto más calma
        endk = 0.3 if kind == "build" else (1.2 if kind in ("settle", "breakdown") else 1.0)
        out = []
        for i in range(NSEG):
            x = (i + 1) / NSEG
            base = start + (target - start) * x * x * (3 - 2 * x)          # rampa en S
            j, f = i % 8, RUBATO[i % 8]
            if j >= 6 and i >= 8:                                           # cierre de sección
                f = 1 + (f - 1) * endk
            out.append(base * (1 + (f - 1) * k))
        if is_coda:                                                         # ritardando final
            for idx, fac in zip(range(NSEG - 4, NSEG), (0.95, 0.90, 0.84, 0.76)):
                out[idx] *= fac
        return out

    # ──────────────────────────────────────────────────────────────────────
    def _compose(self, idx, cycle, kind, energy, tonic, nxt, is_coda, final) -> Section:
        rng, rules = self.rng, RULES[kind]
        first_theme = cycle == 0 and kind == "theme"
        locked = first_theme or is_coda                        # el tema y la coda se quedan como están
        if first_theme:
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

        splits: Dict[int, Tuple[float, Chord]] = {}
        device = None
        if not locked:
            splits = self._plan_splits(kind, chords)
            device = self._plan_device(kind, chords, tonic)
            if device == "rota":                               # V → bVI: se niega la resolución
                r = chords[7].root
                chords[7] = Chord(r, "7")
                splits[7] = (2.0, Chord((tonic + 8) % 12, "maj7"))
            elif device == "4-3":                              # sus4 → 3ª sobre la dominante
                r = chords[7].root
                chords[7] = Chord(r, "7sus4")
                splits[7] = (2.0, Chord(r, "7"))

        cad4, cad8 = self._plan_cadences(kind, chords, tonic, nxt, splits, device, is_coda, locked)

        base = tonic + 12 * round((64 - tonic) / 12)           # hook en E4 (64)
        ctx = Ctx(rng, tonic, base, chords, energy, splits=splits,
                  critic=(1 if locked else self.critic), bass_prev=self.bprev)
        r2 = self.rng2
        ctx.high = mel_key == "left-hand"
        if ctx.high:
            ctx.near = 62
        self._plan_voicings(ctx)

        topts = rules["texture"]
        if mel_key == "canon":
            topts = [("ambient", 1)]
        elif ctx.high:
            topts = [(n, w) for n, w in topts if n in HIGH_OK] or [("ambient", 1)]
        tex_key = self._pick(topts, self.prev["texture"])
        latin = False
        if tex_key == "comping":
            _, ctx.comp, _, latin = pick_comp(r2, energy)
        ctx.bass = self._pick_bass(kind, mel_key, tex_key, locked, latin)

        label, voices = VARIANTS[mel_key](ctx)
        if not locked and self.devices > 0:
            voices = [appoggiatura(voices[0], rng, 0.4 * self.devices)] + voices[1:]
        if not locked and self.expressive > 0:
            voices = [colour(voices[0], r2, 0.30 * self.expressive)] + voices[1:]
        if not locked:
            voices = [cadence_melody(voices[0], cad4, cad8, tonic,
                                     nxt if (kind == "settle" and cad8 in ("PAC", "picardy")) else tonic)] + voices[1:]
        notes = resolve(voices, ctx) + render_texture(TEXTURES[tex_key][0], ctx)
        if ctx.bass_last is not None:
            self.bprev = ctx.bass_last
        if not locked and self.expressive > 0:
            anticipate(notes, ctx, 0.35 * self.expressive * (0.4 + energy), r2)
        swing = 0.0
        if (not locked and self.swing > 0 and tex_key in SWING_OK and mel_key not in NO_SWING
                and r2.random() < SWING_PROB.get(kind, 0.0) * self.swing):
            swing = r2.choice([0.5, 0.75, 1.0])
            apply_swing(notes, swing)
        silence = False
        if (not locked and not final and not cad8 and self.silence > 0
                and r2.random() < SIL_PROB.get(kind, 0.0) * self.silence):
            silence = True
            cut_for_silence(notes)
        fstats: Dict[str, int] = {}
        notes = self._finalize(notes, kind, 0.55 if (kind == "intro" and cycle > 0) else energy,
                               self.phrasing, self.playable, fstats)
        sf = ([0.0] if kind == "climax" else []) + ([30.0] if cad8 in ("rota", "PAC", "picardy") else [])
        if sf and self.expressive > 0:            # sforzando en las llegadas (expressive_injector)
            for n in notes:
                if n.role == "mel" and any(abs(n.beat - b_) < 0.06 for b_ in sf):
                    n.vel = min(110, n.vel + int(round(12 * self.expressive)))

        pe = TEXTURES[tex_key][1]
        pedal: List[Tuple[float, int]] = []
        if pe:
            for k in range(0, SEC_BEATS, pe):
                pedal += [(float(k), 0), (k + 0.08, 127)]
            for b, (s_, _) in splits.items():                  # re-pedal en el cambio de acorde
                pedal += [(4 * b + s_ - 0.03, 0), (4 * b + s_ + 0.05, 127)]
            pedal.sort()
        else:
            pedal.append((0.0, 0))
        if silence:                                            # caesura: el pedal sube antes del último pulso
            pedal = sorted([(b_, v_) for b_, v_ in pedal if b_ < 30.9] + [(30.95, 0)])

        target = tempo_for(energy) * self.scale
        start_bpm = self.bpm
        bpms = self._tempo_curve(kind, energy, start_bpm, target, is_coda)
        self.bpm = target

        self.prev = {"melody": mel_key, "harmony": harm_key, "texture": tex_key}
        hlabel = harm_key + "".join(f"·{x}" for x in (cad4, cad8) if x) + ("·sil" if silence else "")
        return Section(idx, cycle, kind, tonic, energy, label, hlabel,
                       tex_key + ("~" if swing else ""), chords, notes,
                       pedal, bpms, final, is_coda, splits, start_bpm,
                       bpms[-1] if is_coda else target, cad8, cad4, swing, silence, fstats)

    @staticmethod
    def _finalize(notes: List[Note], kind: str, energy: float, phrasing: float = 0.0,
                  playable: bool = True, stats: Optional[Dict[str, int]] = None) -> List[Note]:
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
        # 1b) tocabilidad: extensión y nº de dedos por mano en cada ataque
        if playable:
            kept, st = fix_playability(kept)
            if stats is not None:
                stats.update(st)
        # 2) fraseo: arco de frase hacia el pico melódico + jerarquía métrica (performer.shape_velocities)
        if phrasing > 0:
            ks = phrasing * (0.5 + 0.5 * min(1.0, energy + 0.2))
            for p0 in range(0, SEC_BEATS, PHRASE_BEATS):
                p1 = p0 + PHRASE_BEATS
                mel = [n for n in kept if n.role == "mel" and p0 <= n.beat < p1]
                pk = max(mel, key=lambda n: (n.pitch, -n.beat)).beat if mel else p0 + 0.6 * PHRASE_BEATS
                pk = min(max(pk, p0 + 4), p1 - 4)
                for n in kept:
                    if not p0 <= n.beat < p1:
                        continue
                    shape = (n.beat - p0) / (pk - p0) if n.beat <= pk else 1 - (n.beat - pk) / (p1 - pk)
                    w = 1.0 if n.role == "mel" else 0.5
                    dv = (shape - 0.5) * 18 * ks * w
                    pos = n.beat % 4
                    if abs(pos - round(pos)) < 1e-6:
                        dv += {0: 5, 2: 2}.get(int(round(pos)) % 4, 0) * ks * (1.0 if n.role == "mel" else 0.6)
                    else:
                        dv -= 3 * ks * w
                    n.vel += int(round(dv))
        # 3) velocidades: base por energía + regulador de la sección
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


def timed_events(sec: Section, hum_ms: float, rng: random.Random, ch: int,
                 lead_ms: float = 0.0) -> List[Tuple[float, int, Raw]]:
    """[(pulso, orden, bytes)] — orden: 0=note-off, 1=pedal, 2=note-on."""
    ev: List[Tuple[float, int, Raw]] = []
    for n in sec.notes:
        on, vel = n.beat, n.vel
        k = 1.0 / 1000.0 * sec.bpm_at(n.beat) / 60.0                  # ms -> pulsos
        if hum_ms > 0:
            on += rng.gauss(0, hum_ms) * k
            vel = int(max(24, min(110, vel + rng.gauss(0, 2.5))))
        if lead_ms and n.role == "mel":                               # la melodía se adelanta al acompañamiento
            on -= lead_ms * k
        off = max(n.beat + n.dur, on + 0.02)
        ev.append((on, 2, (0x90 | ch, n.pitch, vel)))
        ev.append((off, 0, (0x80 | ch, n.pitch, 0)))
    for beat, val in sec.pedal:
        ev.append((beat, 1, (0xB0 | ch, 64, val)))
    return ev


def _bar_names(sec: Section) -> List[str]:
    """Un nombre por compás; si el acorde cambia dentro del compás: 'Am9·F#m7b5'."""
    return [c.name + (f"·{sec.splits[b][1].name}" if b in sec.splits else "")
            for b, c in enumerate(sec.chords)]


def chords_str(sec: Section) -> str:
    names = _bar_names(sec)
    return " ".join(names[:4]) + " | " + " ".join(names[4:])


def describe(sec: Section) -> str:
    cyc = "coda" if sec.is_coda else f"c{sec.cycle + 1}"
    chords = chords_str(sec)
    bpm = f"{sec.tempo_from:.0f}→{sec.tempo_to:.0f}"
    return (f"{sec.index + 1:>2} {cyc:<4} {sec.kind:<11} {sec.key_name:<8} {bpm:>8} bpm  "
            f"{chords:<60} {sec.melody:<14} {sec.texture:<8} {sec.harmony}")


def colored(sec: Section) -> str:
    cyc = "coda" if sec.is_coda else f"c{sec.cycle + 1}"
    chords = chords_str(sec)
    bpm = f"{sec.tempo_from:.0f}→{sec.tempo_to:.0f}"
    return (f"{dim(f'{sec.index + 1:>2}')} {dim(f'{cyc:<4}')} {bold(f'{sec.kind:<11}')} "
            f"{cyan(f'{sec.key_name:<8}')} {yellow(f'{bpm:>8} bpm')}  {chords:<60} "
            f"{green(f'{sec.melody:<14}')} {magenta(f'{sec.texture:<8}')} {blue(sec.harmony)}")


HEADER = (f"{dim(' #')} {dim('cic ')} {dim('sección    ')} {dim('tonalidad')} {dim('   tempo    ')}  "
          f"{dim('acordes'):<69} {dim('melodía'):<23} {dim('textura'):<17} {dim('armonía')}")

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
    if not 0 <= args.rubato <= 2 or not 0 <= args.phrasing <= 2:
        raise VariationsError("--rubato y --phrasing deben estar entre 0 y 2")
    if not 0 <= args.hrhythm <= 1 or not 0 <= args.devices <= 1:
        raise VariationsError("--hrhythm y --devices deben estar entre 0 y 1")
    if not 0 <= args.lead <= 60:
        raise VariationsError("--lead debe estar entre 0 y 60 ms")
    if not all(0 <= x <= 1 for x in (args.swing, args.cadences, args.expressive, args.silence)):
        raise VariationsError("--swing, --cadences, --expressive y --silence deben estar entre 0 y 1")
    if not 1 <= args.critic <= 32:
        raise VariationsError("--critic debe estar entre 1 y 32")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 100000)
    args.seed = seed
    return Composer(seed, args.cycles, not args.no_coda, args.tempo_scale,
                    args.rubato, args.phrasing, args.hrhythm, args.devices,
                    args.swing, args.critic, args.cadences, args.expressive, args.silence,
                    not args.no_playable)


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


def top_line(notes: Sequence[Note]) -> List[int]:
    """Línea superior de la melodía: la nota más aguda de cada ataque de melodía, en orden."""
    by: Dict[float, int] = {}
    for n in notes:
        if n.role == "mel":
            k = round(n.beat, 3)
            by[k] = max(by.get(k, 0), n.pitch)
    return [by[k] for k in sorted(by)]


def cmd_audit(args) -> int:
    comp = _composer(args)
    secs = list(comp.sections())
    rows, tot_span, tot_many = [], 0, 0
    for s in secs:
        span, many = playability_issues(s.notes)
        tot_span += span
        tot_many += many
        m = melody_metrics(top_line(s.notes))
        rows.append({"index": s.index + 1, "kind": s.kind, "melody": s.melody, "texture": s.texture,
                     "cad4": s.cad4, "cad8": s.device, "swing": s.swing, "silence": s.silence,
                     "span_violations": span, "many_notes": many, "fixes": s.fixes,
                     "score": round(score_melody(top_line(s.notes)), 2) if m else None,
                     "step_ratio": round(m["step_ratio"], 2) if m else None,
                     "unresolved": m.get("unresolved") if m else None,
                     "range": m.get("range") if m else None})
    cads = collections.Counter(r["cad8"] or "-" for r in rows)
    cads4 = collections.Counter(r["cad4"] or "-" for r in rows)
    summary = {"seed": comp.seed, "sections": len(secs), "span_violations": tot_span, "many_notes": tot_many,
               "cadence8": dict(cads), "cadence4": dict(cads4),
               "swing_sections": sum(1 for r in rows if r["swing"]),
               "silences": sum(1 for r in rows if r["silence"]),
               "fixes": {k: sum(r["fixes"].get(k, 0) for r in rows) for k in ("relabeled", "shifted", "dropped")}}
    if args.json:
        print(json.dumps({"summary": summary, "sections": rows}, ensure_ascii=False, indent=2))
        return 0
    print(bold(f"variations audit · semilla {comp.seed} · {len(secs)} secciones"))
    print(dim(f"{'#':>2} {'sección':<11} {'melodía':<14} {'textura':<9} {'cad4':<9} {'cad8':<8} "
              f"{'sw':>4} {'sil':>3} {'ext':>3} {'ded':>3} {'pts':>5} {'grad':>5} {'sin comp':>8} {'ámb':>3}"))
    for r in rows:
        print(f"{r['index']:>2} {r['kind']:<11} {r['melody']:<14} {r['texture']:<9} {r['cad4'] or '-':<9} "
              f"{r['cad8'] or '-':<8} {r['swing'] or '':>4} {'sí' if r['silence'] else '':>3} "
              f"{r['span_violations']:>3} {r['many_notes']:>3} "
              f"{r['score'] if r['score'] is not None else '':>5} "
              f"{r['step_ratio'] if r['step_ratio'] is not None else '':>5} "
              f"{r['unresolved'] if r['unresolved'] is not None else '':>8} "
              f"{r['range'] if r['range'] is not None else '':>3}")
    print(dim("ext = ataques con extensión de mano > %d st · ded = ataques con > %d notas en una mano · "
              "grad = proporción de movimiento por grados · sin comp = saltos grandes sin compensar"
              % (MAX_SPAN, MAX_HAND_NOTES)))
    print(f"cadencias compás 8: {dict(cads)} · compás 4: {dict(cads4)}")
    print(f"swing en {summary['swing_sections']} secciones · caesuras: {summary['silences']} · "
          f"tocabilidad: {tot_span} ataques con extensión excesiva, {tot_many} con demasiados dedos · "
          f"ajustes {summary['fixes']}")
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
            meta_ev.append(((off_beats + i * SEG_BEATS) * TPQ, 1,
                            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
        for beat, order, raw in timed_events(sec, args.humanize, hrng, ch, args.lead):
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
        for beat, order, raw in timed_events(sec, args.humanize, hrng, ch, args.lead):
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
    p.add_argument("--rubato", type=float, default=0.6,
                   help="intensidad del rubato por frase, 0 = tempo rígido (def. 0.6; máx. 2)")
    p.add_argument("--phrasing", type=float, default=0.8,
                   help="intensidad del fraseo dinámico (arco de frase + métrica), 0 = plano (def. 0.8; máx. 2)")
    p.add_argument("--hrhythm", type=float, default=1.0,
                   help="probabilidad relativa de cambios de acorde dentro del compás, 0–1 (def. 1)")
    p.add_argument("--devices", type=float, default=1.0,
                   help="probabilidad relativa de cadencia rota / retardo 4–3 / apoyatura, 0–1 (def. 1)")
    p.add_argument("--lead", type=float, default=12.0,
                   help="adelanto de la melodía sobre el acompañamiento, en ms (def. 12)")
    p.add_argument("--swing", type=float, default=1.0,
                   help="probabilidad relativa de swing de corcheas en algunas secciones, 0–1 (def. 1; 0 = siempre recto)")
    p.add_argument("--critic", type=int, default=6,
                   help="candidatas de la melodía libre entre las que elige el crítico, 1–32 (def. 6; 1 = sin crítico)")
    p.add_argument("--cadences", type=float, default=1.0,
                   help="probabilidad relativa de semicadencia frigia / cierre auténtico, plagal o picardo, 0–1 (def. 1)")
    p.add_argument("--expressive", type=float, default=1.0,
                   help="anticipación, escapada, cromatismos, sforzando, 0–1 (def. 1)")
    p.add_argument("--silence", type=float, default=1.0,
                   help="probabilidad relativa de caesura en el último pulso de una sección, 0–1 (def. 1)")
    p.add_argument("--no-playable", action="store_true",
                   help="no ajusta la extensión de mano ni el nº de dedos (comportamiento de v3)")
    p.add_argument("--quiet", action="store_true", help="no imprime la forma")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="variations_v4.py",
                                 description="Variaciones pianísticas sobre el tema eom.dev (compone en vivo).")
    ap.add_argument("--version", action="version", version=f"variations_v4.py {__version__}")
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

    p = sub.add_parser("audit", help="compone sin sonar y mide melodía, cadencias y tocabilidad por sección")
    _common(p)
    p.add_argument("--json", action="store_true", help="salida JSON")
    p.set_defaults(func=cmd_audit)

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
