#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         UNIFIER  v2.0                                        ║
║      Fusión coherente de secciones-ancla MIDI en una obra unificada         ║
║                                                                              ║
║  v2.0 = merge de las dos implementaciones v1:                                ║
║    · de la v1 "interna": ancla explícita, tempo global con rampas,          ║
║      groove del ancla (16 subdivisiones), protección del canal 9,           ║
║      tonalidad única con relativas, salida tipo 1 por roles, arco áureo     ║
║    · de la v1 "externa": extracción de motivo por n-gramas con saliencia,   ║
║      transformaciones del motivo (inversión/retrógrado/aumentación),        ║
║      puntos estructurales de siembra, nudge armónico con conducción de      ║
║      voces, puentes con acorde pivote + V7→I, dosificación --glue,          ║
║      score de coherencia antes→después, humanización y ritardando final     ║
║                                                                              ║
║  DIMENSIONES DE UNIFICACIÓN (dosificables con --glue / --glue-preset):      ║
║  [M] motif    — motivo ancla sembrado transformado en puntos estructurales  ║
║                 de las demás secciones (pista propia "motivo")              ║
║  [H] harmony  — compases con acorde no diatónico se re-armonizan a un       ║
║                 acorde diatónico compatible con la melodía, con conducción  ║
║                 de voces (tonos comunes entre compases)                     ║
║  [R] rhythm   — los ataques se imantan a las casillas fuertes del groove    ║
║                 del ANCLA (no a una rejilla genérica)                       ║
║  [D] dynamics — nivelación entre secciones + arco global con clímax único  ║
║  [K] tonal    — siempre activo: tonalidad global única; secciones en modo   ║
║                 opuesto van a la RELATIVA (misma armadura, mismo color)     ║
║  [T] tempo    — siempre activo: tempo diana + --tempo-blend + rampas        ║
║  [I] instr    — siempre activo: un instrumento GM por rol en toda la obra   ║
║                 (paleta del ancla; --no-instr fuerza paleta de cuerdas)     ║
║                                                                              ║
║  PUENTES (v2): último acorde de A → acorde PIVOTE común → V7 de la          ║
║  tonalidad global → primer acorde de B, con conducción de voces, bajo por   ║
║  raíces y declaración del motivo ancla encima, en crescendo.                ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] ANÁLISIS   — tonalidad, tempo, tensión, roles, acordes por compás      ║
║  [2] PLAN TONAL — tonalidad global + transposición mínima (relativas)       ║
║  [3] PLAN TEMPO — tempo diana + rampas en puentes                           ║
║  [4] MOTIVO     — n-gramas de intervalos repetidos con saliencia            ║
║                   (reps × longitud × distintividad) o --motif-seed          ║
║  [5] UNIFICAR   — niveles + nudge armónico + siembra + imantación rítmica   ║
║  [6] PUENTES    — pivote + V7 + motivo + crossfade                          ║
║  [7] ENSAMBLAR  — pistas por rol (melodía/motivo/contrapunto/armonía/bajo)  ║
║  [8] ARCO       — clímax único (áureo) + humanización + ritardando final    ║
║  [9] SCORE      — coherencia antes→después + informe + escritura            ║
║                                                                              ║
║  USO:                                                                        ║
║    python unifier2.py intro.mid nudo.mid climax.mid final.mid               ║
║    python unifier2.py a.mid b.mid c.mid --key Dm --anchor 1                 ║
║    python unifier2.py a.mid b.mid --glue motif=0.8,harmony=0.7,rhythm=0.3   ║
║    python unifier2.py a.mid b.mid c.mid --glue-preset fuerte                ║
║    python unifier2.py a.mid b.mid --motif-seed mi_motivo.mid                ║
║    python unifier2.py a.mid b.mid c.mid --report            # solo plan     ║
║                                                                              ║
║  OPCIONES NUEVAS EN v2 (el resto, como en v1):                              ║
║    --glue SPEC        motif=,harmony=,rhythm=,dynamics=  (0–1 cada eje)     ║
║    --glue-preset P    sutil | medio | fuerte  (default: medio)              ║
║    --motif-seed FILE  MIDI externo con el motivo semilla (anula el ancla)   ║
║    --no-harmony       Atajo: harmony=0                                       ║
║    --no-humanize      Sin micro-timing ni variación de velocity              ║
║    --no-rit           Sin ritardando final                                   ║
║    --score            Imprime el score de coherencia antes→después          ║
║                                                                              ║
║  AUTOCONTENIDO: solo mido y numpy. Funciones copiadas/adaptadas de           ║
║  mosaic_composer, stitcher, leitmotif_tracker, climax_engineer y de la      ║
║  implementación externa v1.1 (voice leading, pivotes, n-gramas, score).     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np

try:
    from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
except ImportError:
    sys.exit("[unifier] Falta mido:  pip install mido")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES Y UTILIDADES DE CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

TPB = 480                     # ticks per beat internos (todo se normaliza aquí)
_SPARK = "▁▂▃▄▅▆▇█"

_COLORS = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
           "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
           "magenta": "\033[35m", "red": "\033[31m"}


def _c(k):
    """Color ANSI si stdout es un TTY (mismo criterio que el resto del pipeline)."""
    return _COLORS.get(k, "") if sys.stdout.isatty() else ""


def sparkline(v: np.ndarray) -> str:
    # ── copiada de climax_engineer.py ──
    v = np.asarray(v, dtype=float)
    if len(v) == 0 or v.max() <= v.min():
        return _SPARK[0] * max(1, len(v))
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  TEORÍA: ESCALAS, TONALIDADES, DISTANCIAS   (copiado de mosaic_composer /
#  stitcher, con pequeñas adaptaciones)
# ══════════════════════════════════════════════════════════════════════════════

SCALE_INTERVALS = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

NAME_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
            "Bb": 10, "B": 11}


def key_to_midi_root(key_str: str) -> int:
    # ── copiada de mosaic_composer.py ──
    parts = key_str.strip().split()
    return NAME_MAP.get(parts[0], 0)


def parse_key_arg(s: str) -> str:
    """'Dm' → 'D minor', 'Eb' → 'Eb major', 'F# minor' → tal cual."""
    s = s.strip()
    if " " in s:
        return s
    if s.endswith("m") and s[:-1] in NAME_MAP:
        return f"{s[:-1]} minor"
    return f"{s} major"


def _get_scale(key_str: str) -> list:
    # ── copiada de mosaic_composer.py ──
    parts = key_str.strip().lower().split()
    root = key_to_midi_root(key_str)
    mode = "major"
    for m in SCALE_INTERVALS:
        if m in " ".join(parts):
            mode = m
            break
    if "minor" in parts and "harmonic" not in parts:
        mode = "minor"
    intervals = SCALE_INTERVALS[mode]
    pitches = []
    for octave in range(-1, 9):
        for iv in intervals:
            p = root + iv + octave * 12
            if 21 <= p <= 108:
                pitches.append(p)
    return sorted(set(pitches))


def _snap_to_scale(pitch: int, scale: list) -> int:
    # ── copiada de mosaic_composer.py ──
    if not scale:
        return int(np.clip(pitch, 21, 108))
    return min(scale, key=lambda p: (abs(p - pitch), p))


def _fifth_distance(pc_a, pc_b):
    # ── copiada de stitcher.py ──
    diff = (pc_b - pc_a) % 12
    fifth_steps = (diff * 7) % 12
    return min(fifth_steps, 12 - fifth_steps)


def detect_key_notes(all_pitches: list) -> str:
    """Krumhansl sobre una lista de pitches (adaptado de
    mosaic_composer.detect_key_simple para trabajar con notas ya extraídas)."""
    pitch_counts = np.zeros(12)
    for p in all_pitches:
        pitch_counts[p % 12] += 1
    if pitch_counts.sum() == 0:
        return "C major"
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_key, best_score, best_mode = 0, -np.inf, "major"
    for root in range(12):
        rolled = np.roll(pitch_counts, -root)
        rolled = rolled / (rolled.sum() + 1e-9)
        maj = np.corrcoef(rolled, major_profile / major_profile.sum())[0, 1]
        mnr = np.corrcoef(rolled, minor_profile / minor_profile.sum())[0, 1]
        if maj > best_score:
            best_score, best_key, best_mode = maj, root, "major"
        if mnr > best_score:
            best_score, best_key, best_mode = mnr, root, "minor"
    return f"{NOTE_NAMES[best_key]} {best_mode}"


# ══════════════════════════════════════════════════════════════════════════════
#  HUELLAS DE MOTIVO   (copiado de leitmotif_tracker.py)
# ══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    "major": {0, 4, 7},  "minor": {0, 3, 7},  "dom7": {0, 4, 7, 10},
    "maj7":  {0, 4, 7, 11}, "min7": {0, 3, 7, 10}, "dim": {0, 3, 6},
    "sus4":  {0, 5, 7},  "power": {0, 7},
}


def identify_chord(pcs: set):
    # ── copiada de unifier_external.py ──
    if not pcs:
        return 0, "none"
    best = None
    for r in range(12):
        rel = {(p - r) % 12 for p in pcs}
        for q, t in CHORD_TEMPLATES.items():
            score = len(rel & t) - 0.5 * len(rel - t) - 0.3 * len(t - rel)
            if best is None or score > best[0]:
                best = (score, r, q)
    return best[1], best[2]


def triad_pcs(root_pc: int, quality: str) -> set:
    # ── copiada de unifier_external.py ──
    r = root_pc % 12
    if quality in ("minor", "min7"):
        return {r, (r + 3) % 12, (r + 7) % 12}
    if quality == "dim":
        return {r, (r + 3) % 12, (r + 6) % 12}
    if quality == "sus4":
        return {r, (r + 5) % 12, (r + 7) % 12}
    if quality == "dom7":
        return {r, (r + 4) % 12, (r + 7) % 12, (r + 10) % 12}
    if quality == "maj7":
        return {r, (r + 4) % 12, (r + 7) % 12, (r + 11) % 12}
    if quality == "power":
        return {r, (r + 7) % 12}
    return {r, (r + 4) % 12, (r + 7) % 12}


def scale_pcs_set(key_str: str) -> set:
    """Pitch classes de la escala de una tonalidad (major/minor)."""
    root = key_to_midi_root(key_str)
    ivs = SCALE_INTERVALS["minor" if "minor" in key_str else "major"]
    return {(root + i) % 12 for i in ivs}


def voice_lead_chord(target_pcs: set, prev_voices: list, center: int = 60,
                     n_voices: int = 3) -> list:
    # ── copiada de unifier_external.py ──
    if not target_pcs:
        return list(prev_voices) if prev_voices else []
    nv = n_voices if not prev_voices else max(n_voices, len(prev_voices))
    out, used = [], set()
    if prev_voices:
        for v in prev_voices:                       # 1) tonos comunes se quedan
            if (v % 12) in target_pcs and v not in used:
                out.append(v)
                used.add(v)
        while len(out) < nv:                        # 2) resto: mínimo movimiento
            best = None
            for p in range(center - 14, center + 14):
                if (p % 12) in target_pcs and p not in used and 0 <= p <= 127:
                    if best is None or abs(p - center) < best[0]:
                        best = (abs(p - center), p)
            if best is None:
                break
            out.append(best[1])
            used.add(best[1])
    else:
        offsets = [0, 4, -3, 7, -7, 3]
        for i in range(nv):
            target = center + (offsets[i] if i < len(offsets)
                               else (i - nv // 2) * 4)
            best = None
            for p in range(target - 7, target + 8):
                if (p % 12) in target_pcs and p not in used and 0 <= p <= 127:
                    if best is None or abs(p - target) < best[0]:
                        best = (abs(p - target), p)
            if best:
                out.append(best[1])
                used.add(best[1])
    return sorted(out)


def motif_transform(intervals: list, durs: list, kind: str):
    # ── copiada de unifier_external.py ──
    if not intervals:
        return [], []
    if kind == "retrograde":
        return list(reversed(intervals)), list(reversed(durs))
    if kind == "inversion":
        return [-x for x in intervals], list(durs)
    if kind == "augmentation":
        return list(intervals), [d * 2 for d in durs]
    return list(intervals), list(durs)


# ══════════════════════════════════════════════════════════════════════════════
#  [1] ANÁLISIS DE FRAGMENTOS
# ══════════════════════════════════════════════════════════════════════════════
#  Representación de nota:  [t_on, note, vel, dur]   (ticks normalizados a 480)
#  Un fragmento agrupa streams por (track, channel) y les asigna rol.

class Fragment:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.streams = {}        # (trk, ch) -> {"notes": [...], "program": int|None}
        self.roles = {}          # (trk, ch) -> "melodia"|"bajo"|"armonia"|"contrapunto"|"percusion"
        self.key = "C major"
        self.tempo_bpm = 120.0
        self.bars = 0
        self.bar_ticks = TPB * 4
        self.tension = 0.5
        self.transpose = 0       # decidido en el plan tonal
        self.vel_mean = 64.0
        self.motif = None        # solo el ancla: [(pitch, onset_rel, dur, vel_rel)]

    # ---- carga -------------------------------------------------------------

    def load(self):
        mid = MidiFile(self.path)
        src_tpb = mid.ticks_per_beat or 480
        k = TPB / src_tpb                       # factor de normalización de ticks
        ts_num, ts_den = 4, 4
        max_tick = 0

        for ti, track in enumerate(mid.tracks):
            t = 0
            pending = {}                        # (ch,note) -> (t_on, vel)
            for msg in track:
                t += msg.time
                if msg.type == "set_tempo" and self.tempo_bpm == 120.0:
                    self.tempo_bpm = round(60_000_000 / msg.tempo, 1)
                elif msg.type == "time_signature":
                    ts_num, ts_den = msg.numerator, msg.denominator
                elif msg.type == "program_change":
                    st = self.streams.setdefault(
                        (ti, msg.channel), {"notes": [], "program": None})
                    if st["program"] is None:
                        st["program"] = msg.program
                elif msg.type == "note_on" and msg.velocity > 0:
                    pending[(msg.channel, msg.note)] = (t, msg.velocity)
                elif msg.type == "note_off" or (msg.type == "note_on"
                                                and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in pending:
                        t_on, vel = pending.pop(key)
                        st = self.streams.setdefault(
                            (ti, msg.channel), {"notes": [], "program": None})
                        st["notes"].append([int(round(t_on * k)), msg.note, vel,
                                            max(1, int(round((t - t_on) * k)))])
                        max_tick = max(max_tick, int(round(t * k)))

        self.streams = {sid: st for sid, st in self.streams.items()
                        if st["notes"]}
        if not self.streams:
            sys.exit(f"[unifier] {self.path}: no contiene notas.")
        for st in self.streams.values():
            st["notes"].sort()

        self.bar_ticks = int(TPB * ts_num * (4 / ts_den))
        self.bars = max(1, math.ceil(max_tick / self.bar_ticks))
        self._analyze()
        return self

    # ---- análisis ------------------------------------------------------------

    def _pitched_streams(self):
        return {sid: st for sid, st in self.streams.items() if sid[1] != 9}

    def _analyze(self):
        pitched = self._pitched_streams()
        all_pitches = [n[1] for st in pitched.values() for n in st["notes"]]
        all_vels = [n[2] for st in self.streams.values() for n in st["notes"]]
        self.key = detect_key_notes(all_pitches) if all_pitches else "C major"
        self.vel_mean = float(np.mean(all_vels)) if all_vels else 64.0
        self.tension = self._estimate_tension()
        self._detect_roles()
        self.chords = self._chords_per_bar()

    def _chords_per_bar(self):
        """(root_pc, quality) por compás, pcs ponderados por duración
        (adaptado de unifier_external.chords_per_bar)."""
        bars = [set() for _ in range(self.bars)]
        weights = [{} for _ in range(self.bars)]
        for sid, st in self.streams.items():
            if sid[1] == 9:
                continue
            for n in st["notes"]:
                b = min(n[0] // self.bar_ticks, self.bars - 1)
                pc = n[1] % 12
                weights[b][pc] = weights[b].get(pc, 0) + n[3]
        out = []
        for b in range(self.bars):
            if not weights[b]:
                out.append((0, "none"))
                continue
            top = sorted(weights[b], key=weights[b].get, reverse=True)[:4]
            out.append(identify_chord(set(top)))
        return out

    def _estimate_tension(self) -> float:
        # ── adaptada de mosaic_composer.estimate_tension ──
        events = sorted((n[0], n[1]) for st in self.streams.values()
                        for n in st["notes"])
        if not events:
            return 0.5
        total_time = events[-1][0] + 1
        all_pitches = [p for _, p in events]
        chromatic_ratio = len(set(p % 12 for p in all_pitches)) / 12.0
        density = min(1.0, len(events) / max(1, total_time / TPB) / 8)
        return float(np.clip(0.4 * chromatic_ratio + 0.6 * density, 0, 1))

    def _detect_roles(self):
        # ── adaptada de orchestral_colorist.detect_roles a streams (trk,ch) ──
        stats = {}
        for sid, st in self.streams.items():
            if sid[1] == 9:
                self.roles[sid] = "percusion"
                continue
            ns = st["notes"]
            mean_p = sum(n[1] for n in ns) / len(ns)
            total = sum(n[3] for n in ns)
            span = (max(n[0] + n[3] for n in ns) - min(n[0] for n in ns)) or 1
            stats[sid] = (mean_p, total / span)      # (registro medio, polifonía)
        rest = sorted(stats)
        if len(rest) == 1:                            # una sola voz → melodía
            self.roles[rest[0]] = "melodia"
            return
        if rest:
            bass = min(rest, key=lambda s: stats[s][0])
            self.roles[bass] = "bajo"
            rest.remove(bass)
        if rest:
            mel = max(rest, key=lambda s: stats[s][0])
            self.roles[mel] = "melodia"
            rest.remove(mel)
        if rest:
            harm = max(rest, key=lambda s: stats[s][1])
            self.roles[harm] = "armonia"
            rest.remove(harm)
        for s in rest:
            self.roles[s] = "contrapunto"

    # ---- accesos -------------------------------------------------------------

    def melody_notes(self):
        for sid, role in self.roles.items():
            if role == "melodia":
                return self.streams[sid]["notes"]
        pitched = self._pitched_streams()
        if pitched:
            return next(iter(pitched.values()))["notes"]
        return []

    def role_program(self, role):
        for sid, r in self.roles.items():
            if r == role:
                return self.streams[sid].get("program")
        return None

    def apply_transpose(self):
        if self.transpose == 0:
            return
        for sid, st in self.streams.items():
            if sid[1] == 9:
                continue
            for n in st["notes"]:
                n[1] = int(np.clip(n[1] + self.transpose, 0, 127))


# ══════════════════════════════════════════════════════════════════════════════
#  [2] PLAN TONAL — transposición mínima a la tonalidad global
# ══════════════════════════════════════════════════════════════════════════════

def _minimal_shift(from_pc: int, to_pc: int) -> int:
    """Desplazamiento en semitonos con valor absoluto mínimo (rango −6..+6)."""
    d = (to_pc - from_pc) % 12
    return d - 12 if d > 6 else d


def plan_tonal(fragments, target_key: str):
    """Asigna .transpose a cada fragmento.

    Si el modo coincide: tónica del fragmento → tónica global.
    Si difiere: se lleva a la RELATIVA de la tonalidad global (misma armadura,
    escala compartida) para que conserve su color modal sin chocar.
    """
    t_root = key_to_midi_root(target_key)
    t_minor = "minor" in target_key
    decisions = []
    for f in fragments:
        f_root = key_to_midi_root(f.key)
        f_minor = "minor" in f.key
        if f_minor == t_minor:
            goal_pc = t_root
        elif f_minor:                       # fragmento menor, obra mayor → relativa menor
            goal_pc = (t_root + 9) % 12
        else:                               # fragmento mayor, obra menor → relativa mayor
            goal_pc = (t_root + 3) % 12
        f.transpose = _minimal_shift(f_root, goal_pc)
        mode = "minor" if f_minor else "major"
        decisions.append({
            "fragment": f.name, "detected_key": f.key,
            "target_pc": NOTE_NAMES[goal_pc] + (" minor" if f_minor else " major"),
            "transpose": f.transpose,
            "fifths_dist_before": _fifth_distance(f_root, t_root),
        })
        f.apply_transpose()
    return decisions


# ══════════════════════════════════════════════════════════════════════════════
#  [3] PLAN DE TEMPO
# ══════════════════════════════════════════════════════════════════════════════

def plan_tempo(fragments, target_bpm, blend: float):
    """Tempo diana global; cada fragmento suena a
    target*(1-blend) + propio*blend  (blend=0 → tempo único)."""
    if target_bpm is None:
        weighted = [(f.tempo_bpm, f.bars) for f in fragments]
        seq = sorted(t for t, b in weighted for _ in range(b))
        target_bpm = seq[len(seq) // 2] if seq else 120.0
    plan = []
    for f in fragments:
        bpm = target_bpm * (1 - blend) + f.tempo_bpm * blend
        plan.append(round(bpm, 1))
    return round(target_bpm, 1), plan


# ══════════════════════════════════════════════════════════════════════════════
#  [4] EXTRACCIÓN DEL MOTIVO ANCLA
# ══════════════════════════════════════════════════════════════════════════════

def extract_motif_ngram(frag, motif_seed_path=None, verbose=False):
    """Motivo como (intervalos, duraciones_en_beats, primer_pitch, saliencia).
    Busca la subsecuencia de intervalos exactamente repetida más saliente
    (repeticiones × longitud × distintividad) — adaptado de
    unifier_external.find_motif. Con --motif-seed usa un MIDI externo."""
    if motif_seed_path:
        seed = Fragment(motif_seed_path).load()
        mel = seed.melody_notes()
    else:
        mel = frag.melody_notes()
    if len(mel) < 4:
        return None
    pitches = [n[1] for n in mel]
    durs_b = [n[3] / TPB for n in mel]
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]

    best = None
    for L in range(3, min(8, len(intervals))):
        counts = {}
        for i in range(len(intervals) - L + 1):
            counts.setdefault(tuple(intervals[i:i + L]), []).append(i)
        for key, idxs in counts.items():
            if len(idxs) >= 2:
                nonzero = sum(1 for x in key if x != 0)
                distinct = 1.0 if nonzero >= 1 else 0.4
                sal = len(idxs) * L * distinct
                if best is None or sal > best[0]:
                    best = (sal, L, key, idxs[0])
    if best is None:                                  # sin repetición exacta
        L = min(5, len(intervals))
        best = (1.0, L, tuple(intervals[:L]), 0)
    sal, L, key, idx = best
    motif = {"intervals": list(key),
             "durs": durs_b[idx:idx + L + 1][:L],
             "first_pitch": pitches[idx],
             "salience": sal}
    if not motif["durs"]:
        motif["durs"] = [0.5] * L
    if verbose:
        src_name = os.path.basename(motif_seed_path) if motif_seed_path \
            else frag.name
        print(f"    motivo ({L} intervalos, saliencia {sal:.1f}, "
              f"de {src_name}): {motif['intervals']}")
    return motif


def realize_motif_notes(intervals, durs, start_pitch, start_tick, scale,
                        vel):
    """Intervalos+duraciones(beats) → notas [t, pitch, vel, dur] en ticks.
    Solo la nota INICIAL se ajusta a la escala global; después los intervalos
    se aplican EXACTOS para preservar la identidad del motivo (si se ajustara
    cada nota, el snap deformaría los intervalos y el hilo dejaría de ser
    reconocible — y de contar en el score)."""
    notes = []
    p = _snap_to_scale(int(np.clip(start_pitch, 21, 108)), scale)
    t = start_tick
    for i, iv in enumerate(intervals):
        d = durs[i] if i < len(durs) else (durs[-1] if durs else 0.5)
        dt = max(60, int(d * TPB))
        notes.append([int(t), int(np.clip(p, 21, 108)), vel, int(dt * 0.9)])
        p += iv
        t += dt
    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  [5a] SIEMBRA MOTÍVICA
# ══════════════════════════════════════════════════════════════════════════════

def _structural_points(frag, strength):
    """Puntos de siembra: inicio, final, límites de frase (silencios en la
    melodía > medio compás) y cada 4 compases — adaptado de
    unifier_external.find_structural_points. Devuelve ticks."""
    total = frag.bars * frag.bar_ticks
    pts = {0}
    mel = frag.melody_notes()
    for i in range(1, len(mel)):
        gap = mel[i][0] - (mel[i - 1][0] + mel[i - 1][3])
        if gap > frag.bar_ticks * 0.5:
            pts.add(mel[i][0])
    for bar in range(0, frag.bars, 4):
        pts.add(bar * frag.bar_ticks)
    pts.add(max(0, total - 2 * frag.bar_ticks))
    pts = sorted(p for p in pts if 0 <= p < total)
    n_inj = max(1, int(round(strength * frag.bars / 6)))
    if len(pts) > n_inj:
        step = len(pts) / n_inj
        pts = [pts[int(i * step)] for i in range(n_inj)]
    return pts


TRANSFORM_CYCLE = ["original", "inversion", "retrograde", "augmentation"]


def seed_motif(frag, motif, scale, strength, rng, verbose=False):
    """Siembra el motivo ancla TRANSFORMADO en puntos estructurales del
    fragmento, primer pitch alineado al acorde local y al registro melódico.
    Las notas van al rol 'motivo' (pista propia)."""
    if not motif or strength <= 0.05:
        return 0
    pts = _structural_points(frag, strength)
    mel = frag.melody_notes()
    mel_mean = int(np.mean([n[1] for n in mel])) if mel else 72
    frag.motif_notes = getattr(frag, "motif_notes", [])
    injected, bars_used = 0, []
    for i, pt in enumerate(pts):
        kind = TRANSFORM_CYCLE[i % len(TRANSFORM_CYCLE)]
        ivs, durs = motif_transform(motif["intervals"], motif["durs"], kind)
        bar = min(pt // frag.bar_ticks, frag.bars - 1)
        root, q = frag.chords[bar] if bar < len(frag.chords) else (0, "none")
        local = triad_pcs((root + frag.transpose) % 12, q) \
            if q != "none" else {p % 12 for p in scale}
        base = motif["first_pitch"]
        base += int(round((mel_mean - base) / 12.0)) * 12    # octavas al registro
        guard = 0
        while base % 12 not in local and guard < 12:         # tono del acorde
            base += 1
            guard += 1
        vel = int(np.clip(frag.vel_mean * (0.55 + 0.3 * strength), 25, 100))
        frag.motif_notes.extend(
            realize_motif_notes(ivs, durs, base, pt, scale, vel))
        injected += 1
        bars_used.append(bar + 1)
    if verbose and injected:
        print(f"    {frag.name}: motivo × {injected} "
              f"(compases {sorted(set(bars_used))})")
    return injected


# ══════════════════════════════════════════════════════════════════════════════
#  [5b] ATRACCIÓN RÍTMICA HACIA EL GROOVE DEL ANCLA
# ══════════════════════════════════════════════════════════════════════════════

def rhythm_grid(frag, subdivisions=16):
    """Histograma de ataques por subdivisión de compás, ponderado por
    velocidad (idea del rhythm_grid de midi_dna_unified, versión compacta)."""
    grid = np.zeros(subdivisions)
    sub_ticks = frag.bar_ticks / subdivisions
    for st in frag.streams.values():
        for n in st["notes"]:
            slot = int((n[0] % frag.bar_ticks) / sub_ticks) % subdivisions
            grid[slot] += n[2]
    if grid.max() > 0:
        grid /= grid.max()
    return grid


def attract_rhythm(frag, anchor_grid, strength, subdivisions=16):
    """Mueve suavemente cada ataque hacia la casilla fuerte más próxima del
    groove ancla. Movimiento acotado a media subdivisión: nunca destruye el
    ritmo original, solo lo 'imanta'."""
    if strength <= 0:
        return 0
    sub_ticks = frag.bar_ticks / subdivisions
    strong = {i for i, w in enumerate(anchor_grid) if w >= 0.35}
    if not strong:
        return 0
    moved = 0
    for sid, st in frag.streams.items():
        if frag.roles.get(sid) == "percusion":
            continue
        for n in st["notes"]:
            in_bar = n[0] % frag.bar_ticks
            slot_f = in_bar / sub_ticks
            nearest = min(strong, key=lambda s: min(abs(slot_f - s),
                                                    subdivisions - abs(slot_f - s)))
            delta_slots = nearest - slot_f
            if delta_slots > subdivisions / 2:
                delta_slots -= subdivisions
            elif delta_slots < -subdivisions / 2:
                delta_slots += subdivisions
            delta = delta_slots * sub_ticks
            if 0 < abs(delta) <= sub_ticks * 0.5:     # solo ajustes finos
                n[0] = max(0, int(n[0] + delta * strength))
                moved += 1
        st["notes"].sort()
    return moved


# ══════════════════════════════════════════════════════════════════════════════
#  [5c] NUDGE ARMÓNICO CON CONDUCCIÓN DE VOCES   (adaptado de
#  unifier_external.nudge_harmony a la representación por streams)
# ══════════════════════════════════════════════════════════════════════════════

def nudge_harmony(frag, target_key, strength, verbose=False):
    """En compases cuyo acorde (ya transportado) es no diatónico en la
    tonalidad global, re-voicea las notas del rol armonía hacia el acorde
    diatónico que mejor encaja con la melodía, manteniendo tonos comunes
    entre compases contiguos (conducción de voces)."""
    if strength < 0.5:
        return 0
    sc = scale_pcs_set(target_key)
    key_pc = key_to_midi_root(target_key)
    minor = "minor" in target_key
    harm_sids = [sid for sid, r in frag.roles.items() if r == "armonia"]
    if not harm_sids:
        return 0
    mel = frag.melody_notes()
    changed = 0
    prev_voices = []
    for b in range(frag.bars):
        t0, t1 = b * frag.bar_ticks, (b + 1) * frag.bar_ticks
        root, q = frag.chords[b] if b < len(frag.chords) else (0, "none")
        if q == "none":
            continue
        root_t = (root + frag.transpose) % 12
        if root_t in sc:                                  # ya diatónico
            harm_now = [n for sid in harm_sids
                        for n in frag.streams[sid]["notes"]
                        if t0 <= n[0] < t1]
            if harm_now:
                prev_voices = sorted(n[1] for n in harm_now)[:4]
            continue
        mel_pcs = {n[1] % 12 for n in mel if t0 <= n[0] < t1}
        candidates = []
        for shift in (0, 5, 7, 9 if not minor else 8):    # I IV V vi/VI
            r = (key_pc + shift) % 12
            if r not in sc:
                continue
            third = 3 if (r + 3) % 12 in sc else 4
            triad = {r, (r + third) % 12, (r + 7) % 12}
            common_prev = sum(1 for v in prev_voices if (v % 12) in triad)
            candidates.append((len(mel_pcs & triad) * 2 + common_prev,
                               triad))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0], reverse=True)
        triad = candidates[0][1]
        for sid in harm_sids:
            bar_notes = [n for n in frag.streams[sid]["notes"]
                         if t0 <= n[0] < t1]
            if not bar_notes:
                continue
            center = int(np.mean([n[1] for n in bar_notes]))
            voices = voice_lead_chord(triad, prev_voices, center=center,
                                      n_voices=min(3, len(bar_notes)))
            for i, n in enumerate(sorted(bar_notes, key=lambda x: x[1])):
                if voices:
                    n[1] = voices[min(i, len(voices) - 1)]
                    changed += 1
            prev_voices = voices or prev_voices
    if verbose and changed:
        print(f"    {frag.name}: {changed} notas de armonía re-voiceadas")
    return changed


# ══════════════════════════════════════════════════════════════════════════════
#  [5d] NIVELES DINÁMICOS ENTRE FRAGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def unify_levels(fragments):
    """Reescala las velocidades de cada fragmento hacia la media global para
    que ningún ancla 'grite' sobre las demás (se conserva la dinámica interna)."""
    global_mean = float(np.mean([f.vel_mean for f in fragments]))
    for f in fragments:
        ratio = global_mean / max(1.0, f.vel_mean)
        ratio = float(np.clip(ratio, 0.7, 1.4))
        if abs(ratio - 1) < 0.03:
            continue
        for st in f.streams.values():
            for n in st["notes"]:
                n[2] = int(np.clip(n[2] * ratio, 12, 120))
    return global_mean


# ══════════════════════════════════════════════════════════════════════════════
#  [6] PUENTES: pedal de dominante + carrera de conexión + crossfade
# ══════════════════════════════════════════════════════════════════════════════

def build_bridge(frag_a, frag_b, bars, scale, target_key, vel_ref,
                 motif=None):
    """Puente v2 (merge): secuencia armónica con conducción de voces
      último acorde de A → acorde PIVOTE → V7 global → primer acorde de B
    + bajo por raíces + declaración del motivo ancla en crescendo.
    Devuelve dict rol → notas (ticks relativos al inicio del puente)."""
    bar_ticks = TPB * 4
    total = bars * bar_ticks
    key_pc = key_to_midi_root(target_key)
    out = {"melodia": [], "motivo": [], "bajo": [], "armonia": []}

    # --- extremos reales, ya transportados ---------------------------------
    if frag_a.chords and frag_a.chords[-1][1] != "none":
        ra, qa = frag_a.chords[-1]
        endA = triad_pcs((ra + frag_a.transpose) % 12, qa)
        endA_root = (ra + frag_a.transpose) % 12
    else:
        endA = triad_pcs(key_pc, "minor" if "minor" in target_key else "major")
        endA_root = key_pc
    if frag_b.chords and frag_b.chords[0][1] != "none":
        rb, qb = frag_b.chords[0]
        startB = triad_pcs((rb + frag_b.transpose) % 12, qb)
        startB_root = (rb + frag_b.transpose) % 12
    else:
        startB = triad_pcs(key_pc, "minor" if "minor" in target_key else "major")
        startB_root = key_pc
    # con tonalidad global única el "pivote" clásico degenera: usamos el
    # PREDOMINANTE (IV/iv) como paso intermedio endA → IV → V7 → startB
    minor = "minor" in target_key
    pivot_root = (key_pc + 5) % 12
    pivot = triad_pcs(pivot_root, "minor" if minor else "major")
    dom_pc = (key_pc + 7) % 12
    v7 = triad_pcs(dom_pc, "dom7")

    # --- secuencia (offset_ticks, pcs, bass_root) ----------------------------
    if bars <= 1:
        seq = [(0, v7, dom_pc), (bar_ticks // 2, startB, startB_root)]
    else:
        half = bars * bar_ticks // 4
        seq = [(0, endA, endA_root), (half, pivot, pivot_root),
               (2 * half, v7, dom_pc), (3 * half, startB, startB_root)]

    prev_voices = []
    for si, (off, pcs, bass_root) in enumerate(seq):
        seg = (seq[si + 1][0] - off) if si + 1 < len(seq) else (total - off)
        cres = 0.6 + 0.35 * (off / max(1, total))
        voices = voice_lead_chord(pcs, prev_voices, center=62, n_voices=3)
        prev_voices = voices
        for v in voices:
            out["armonia"].append([off, int(np.clip(v, 40, 90)),
                                   int(vel_ref * 0.55 * cres + 20),
                                   max(60, seg - 30)])
        bass = 36 + bass_root
        while bass > 52:
            bass -= 12
        out["bajo"].append([off, bass, int(vel_ref * 0.7 * cres + 15),
                            max(60, seg - 30)])

    # --- motivo ancla encima, hacia el registro de la melodía de B ----------
    if motif and motif["intervals"]:
        mel_b = frag_b.melody_notes()
        reg = int(np.mean([n[1] for n in mel_b])) if mel_b else 72
        base = motif["first_pitch"]
        base += int(round((reg - base) / 12.0)) * 12
        guard = 0
        while base % 12 not in startB and guard < 12:
            base += 1
            guard += 1
        notes = realize_motif_notes(motif["intervals"], motif["durs"],
                                    base, 0, scale, int(vel_ref * 0.8))
        span = notes[-1][0] + notes[-1][3] if notes else 1
        for k, n in enumerate(notes):                 # crescendo hacia B
            n[2] = int(np.clip(n[2] * (0.7 + 0.5 * n[0] / max(1, span)),
                               25, 115))
            if n[0] < total:
                out["motivo"].append(n)
    return out


def crossfade(frag_a, frag_b):
    """Último compás de A en decrescendo, primer compás de B con entrada suave."""
    last_start = (frag_a.bars - 1) * frag_a.bar_ticks
    for st in frag_a.streams.values():
        for n in st["notes"]:
            if n[0] >= last_start:
                pos = (n[0] - last_start) / frag_a.bar_ticks
                n[2] = int(np.clip(n[2] * (1.0 - 0.3 * pos), 12, 127))
    for st in frag_b.streams.values():
        for n in st["notes"]:
            if n[0] < frag_b.bar_ticks:
                pos = n[0] / frag_b.bar_ticks
                n[2] = int(np.clip(n[2] * (0.8 + 0.2 * pos), 12, 127))


# ══════════════════════════════════════════════════════════════════════════════
#  [7] ENSAMBLAJE: pistas por rol + instrumentación única
# ══════════════════════════════════════════════════════════════════════════════

ROLE_CHANNELS = {"melodia": 0, "motivo": 4, "contrapunto": 1,
                 "armonia": 2, "bajo": 3, "percusion": 9}
ROLE_FALLBACK_PROGRAM = {"melodia": 40, "contrapunto": 41,     # violín, viola
                         "armonia": 48, "bajo": 42}            # strings, cello


def resolve_palette(anchor, no_instr):
    """Un programa GM por rol para TODA la obra (del ancla; fallback cuerdas)."""
    palette = {}
    for role in ("melodia", "contrapunto", "armonia", "bajo"):
        prog = None if no_instr else anchor.role_program(role)
        if prog is None:
            prog = ROLE_FALLBACK_PROGRAM[role]
        palette[role] = prog
    palette["motivo"] = palette["melodia"]        # el hilo suena al timbre melódico
    return palette


def assemble(fragments, bridges, tempo_plan, target_bpm, palette,
             bridge_bars):
    """Coloca fragmentos y puentes en el tiempo y devuelve:
    role_notes (rol → notas absolutas), tempo_events [(tick, bpm)], total_bars."""
    role_notes = {r: [] for r in ROLE_CHANNELS}
    tempo_events = []
    offset = 0
    layout = []                                        # (nombre, bar_ini, bar_fin)
    bar_ticks = TPB * 4

    for i, f in enumerate(fragments):
        tempo_events.append((offset, tempo_plan[i]))
        for sid, st in f.streams.items():
            role = f.roles.get(sid, "contrapunto")
            for n in st["notes"]:
                role_notes[role].append([offset + n[0], n[1], n[2], n[3]])
        for n in getattr(f, "motif_notes", []):
            role_notes["motivo"].append([offset + n[0], n[1], n[2], n[3]])
        f_ticks = f.bars * f.bar_ticks
        layout.append((f.name, offset // bar_ticks,
                       (offset + f_ticks) // bar_ticks))
        offset += f_ticks

        if i < len(fragments) - 1 and bridge_bars > 0:
            # rampa de tempo dentro del puente hacia el tempo del siguiente
            b = bridges[i]
            steps = max(2, bridge_bars * 2)
            for s in range(steps):
                tk = offset + int(s * bridge_bars * bar_ticks / steps)
                bpm = tempo_plan[i] + (tempo_plan[i + 1] - tempo_plan[i]) \
                    * (s / (steps - 1))
                tempo_events.append((tk, bpm))
            for role, notes in b.items():
                for n in notes:
                    role_notes[role].append([offset + n[0], n[1], n[2], n[3]])
            layout.append((f"puente {i + 1}", offset // bar_ticks,
                           (offset + bridge_bars * bar_ticks) // bar_ticks))
            offset += bridge_bars * bar_ticks

    for notes in role_notes.values():
        notes.sort()
    total_bars = max(1, math.ceil(offset / bar_ticks))
    return role_notes, tempo_events, total_bars, layout


# ══════════════════════════════════════════════════════════════════════════════
#  [8] ARCO DINÁMICO GLOBAL   (curvas de climax_engineer, adaptadas)
# ══════════════════════════════════════════════════════════════════════════════

def energy_curve_roles(role_notes, bar_ticks, n_bars) -> np.ndarray:
    # ── adaptada de climax_engineer.energy_curve a listas de notas ──
    dens = np.zeros(n_bars)
    vel = np.zeros(n_bars)
    reg = np.zeros(n_bars)
    poly = np.zeros(n_bars)
    cnt = np.zeros(n_bars)
    for notes in role_notes.values():
        for n in notes:
            b = min(n[0] // bar_ticks, n_bars - 1)
            dens[b] += 1
            vel[b] += n[2]
            reg[b] += n[1]
            poly[b] += n[3]
            cnt[b] += 1
    mask = cnt > 0
    vel[mask] /= cnt[mask]
    reg[mask] /= cnt[mask]
    poly /= bar_ticks

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    e = 0.4 * norm(dens) + 0.3 * norm(vel) + 0.2 * norm(reg) + 0.1 * norm(poly)
    if len(e) >= 3:
        e = np.convolve(e, np.ones(3) / 3, mode="same")
    return e


def target_curve(n_bars: int, position: float, width: int = 4) -> np.ndarray:
    # ── copiada de climax_engineer.py ──
    peak = max(1, min(n_bars - 2, int(round(position * (n_bars - 1)))))
    t = np.zeros(n_bars)
    for b in range(n_bars):
        if b <= peak:
            t[b] = (b / peak) ** 1.6 if peak else 1.0
        else:
            t[b] = max(0.0, 1 - ((b - peak) / max(1, (n_bars - 1 - peak))) ** 0.8)
    lo = 0.18
    return lo + (1 - lo) * t


def reshape_arc(role_notes, bar_ticks, n_bars, position, intensity):
    """Reescala velocidades compás a compás hacia una curva con clímax único."""
    before = energy_curve_roles(role_notes, bar_ticks, n_bars)
    goal = target_curve(n_bars, position)
    eps = 1e-6
    ratio = (goal + eps) / (before + eps)
    ratio = np.clip(ratio ** 0.5, 1 - 0.35 * intensity, 1 + 0.45 * intensity)
    for role, notes in role_notes.items():
        for n in notes:
            b = min(n[0] // bar_ticks, n_bars - 1)
            n[2] = int(np.clip(n[2] * ratio[b], 12, 124))
    after = energy_curve_roles(role_notes, bar_ticks, n_bars)
    return before, after


# ══════════════════════════════════════════════════════════════════════════════
#  [9] ESCRITURA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def write_midi(role_notes, tempo_events, palette, out_path):
    """MIDI tipo 1: pista de tempo + una pista por rol con su instrumento."""
    mid = MidiFile(ticks_per_beat=TPB, type=1)

    meta = MidiTrack()
    mid.tracks.append(meta)
    meta.append(MetaMessage("track_name", name="unifier", time=0))
    meta.append(MetaMessage("time_signature", numerator=4, denominator=4,
                            time=0))
    prev = 0
    for tick, bpm in sorted(tempo_events):
        meta.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm),
                                time=tick - prev))
        prev = tick
    meta.append(MetaMessage("end_of_track", time=0))

    for role, ch in ROLE_CHANNELS.items():
        notes = role_notes.get(role, [])
        if not notes:
            continue
        trk = MidiTrack()
        mid.tracks.append(trk)
        trk.append(MetaMessage("track_name", name=role, time=0))
        if ch != 9:
            trk.append(Message("program_change", channel=ch,
                               program=palette[role], time=0))
        raw = []
        for (t_on, note, vel, dur) in notes:
            raw.append((t_on, 1, "note_on", note, vel))
            raw.append((t_on + dur, 0, "note_off", note, 0))
        raw.sort()
        prev = 0
        for (t_abs, _, mtype, note, vel) in raw:
            trk.append(Message(mtype, channel=ch, note=int(note),
                               velocity=int(vel), time=max(0, t_abs - prev)))
            prev = t_abs
        trk.append(MetaMessage("end_of_track", time=0))
    mid.save(out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE DE COHERENCIA antes → después   (idea de unifier_external, con las
#  métricas corregidas: sin puntos regalados por "tener puentes"; se añade
#  consistencia de tempo y correlación de groove, que eran los ejes ausentes)
# ══════════════════════════════════════════════════════════════════════════════

def count_motif(notes_lists, intervals):
    """Apariciones exactas de la secuencia de intervalos en listas de notas."""
    if not intervals:
        return 0
    L, hits = len(intervals), 0
    for notes in notes_lists:
        ps = [n[1] for n in sorted(notes)]
        ivs = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]
        for i in range(len(ivs) - L + 1):
            if ivs[i:i + L] == intervals:
                hits += 1
    return hits


def coherence_metrics(fragments, motif, target_key, tempo_plan=None):
    """Métricas 0-1 sobre el estado ACTUAL de los fragmentos."""
    key_pc = key_to_midi_root(target_key)
    # tónica: fracción de fragmentos cuya tonalidad re-detectada cae en la
    # tónica global o su relativa
    rel_pc = (key_pc + (3 if "minor" in target_key else 9)) % 12
    ok = 0
    for f in fragments:
        pitches = [n[1] for sid, st in f.streams.items() if sid[1] != 9
                   for n in st["notes"]]
        k = detect_key_notes(pitches)
        if key_to_midi_root(k) in (key_pc, rel_pc):
            ok += 1
    tonic = ok / len(fragments)
    # tempo: 1 − dispersión relativa de los BPM efectivos
    bpms = tempo_plan if tempo_plan else [f.tempo_bpm for f in fragments]
    tempo = float(np.clip(1 - np.std(bpms) / max(1e-6, np.mean(bpms)) * 2,
                          0, 1))
    # groove: correlación media de cada rejilla con la del ancla (posición 0
    # de fragments = ancla por convención de llamada)
    grids = [rhythm_grid(f) for f in fragments]
    cors = []
    for g in grids[1:]:
        if grids[0].std() > 0 and g.std() > 0:
            cors.append(float(np.corrcoef(grids[0], g)[0, 1]))
    groove = float(np.clip(np.mean(cors), 0, 1)) if cors else 1.0
    # niveles: 1 − dispersión relativa de las medias de velocity
    vms = [float(np.mean([n[2] for st in f.streams.values()
                          for n in st["notes"]])) for f in fragments]
    levels = float(np.clip(1 - np.std(vms) / max(1e-6, np.mean(vms)) * 3,
                           0, 1))
    # motivo: apariciones del hilo común (streams + siembras)
    lists = [st["notes"] for f in fragments for st in f.streams.values()]
    lists += [getattr(f, "motif_notes", []) for f in fragments]
    m_hits = count_motif(lists, motif["intervals"]) if motif else 0
    m_score = min(1.0, m_hits / max(1, 2 * len(fragments))) if motif else 0.0
    glob = 0.25 * tonic + 0.2 * tempo + 0.2 * groove + 0.15 * levels \
        + 0.2 * m_score
    return {"tonic_consistency": round(tonic, 2),
            "tempo_consistency": round(tempo, 2),
            "groove_correlation": round(groove, 2),
            "level_balance": round(levels, 2),
            "motif_recurrences": m_hits,
            "motif_score": round(m_score, 2),
            "global_coherence": round(glob, 2)}


def print_score(before, after):
    B, R, G, D = _c("bold"), _c("reset"), _c("green"), _c("dim")
    print(f"\n{B}[9] COHERENCIA antes → después{R}")
    for k in before:
        b, a = before[k], after[k]
        mark = G if (isinstance(a, (int, float)) and a >= b) else _c("yellow")
        print(f"    {k:<22} {D}{b!s:>6}{R}  →  {mark}{a!s:>6}{R}")


# ══════════════════════════════════════════════════════════════════════════════
#  PULIDO FINAL   (adaptado de unifier_external: humanize / validate / rit)
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_fix(role_notes):
    """Deduplica (tick,pitch) por rol y acota rangos."""
    removed = 0
    for role, notes in role_notes.items():
        seen, clean = set(), []
        for n in sorted(notes):
            key = (n[0], n[1])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            n[1] = int(np.clip(n[1], 0, 127))
            n[2] = int(np.clip(n[2], 1, 127))
            n[3] = max(10, n[3])
            clean.append(n)
        role_notes[role] = clean
    return removed


def humanize(role_notes, rng):
    """Micro-timing ±6 ticks y velocity ±5, excepto percusión."""
    for role, notes in role_notes.items():
        if role == "percusion":
            continue
        for n in notes:
            n[0] = max(0, n[0] + rng.randint(-6, 6))
            n[2] = int(np.clip(n[2] + rng.randint(-5, 5), 15, 124))
        notes.sort()


def add_final_rit(tempo_events, total_ticks):
    """Ritardando final: 85 % → ×0.85, 95 % → ×0.70 del último tempo."""
    if not tempo_events:
        return
    last = sorted(tempo_events)[-1][1]
    tempo_events.append((int(total_ticks * 0.85), last * 0.85))
    tempo_events.append((int(total_ticks * 0.95), last * 0.70))


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME
# ══════════════════════════════════════════════════════════════════════════════

def print_analysis(fragments, anchor_idx, target_key, target_bpm, tempo_plan,
                   tonal_decisions):
    B, D, R = _c("bold"), _c("dim"), _c("reset")
    print(f"\n{B}[1] ANÁLISIS DE FRAGMENTOS{R}")
    hdr = f"{'#':>2} {'fragmento':<22} {'tonalidad':<11} {'tempo':>6} " \
          f"{'compases':>8} {'tensión':>8} {'streams':>8}"
    print(D + hdr + R)
    for i, f in enumerate(fragments):
        mark = " ⚓" if i == anchor_idx else "  "
        print(f"{i:>2} {f.name[:22]:<22} {f.key:<11} {f.tempo_bpm:>6.1f} "
              f"{f.bars:>8} {f.tension:>8.2f} {len(f.streams):>8}{mark}")
    print(f"\n{B}[2] PLAN TONAL{R}  →  tonalidad global: "
          f"{_c('cyan')}{target_key}{R}")
    for d in tonal_decisions:
        sign = f"+{d['transpose']}" if d["transpose"] > 0 else str(d["transpose"])
        print(f"    {d['fragment'][:22]:<22} {d['detected_key']:<11} "
              f"→ {d['target_pc']:<11} ({sign} st)")
    print(f"\n{B}[3] PLAN DE TEMPO{R}  →  diana global: "
          f"{_c('cyan')}{target_bpm} BPM{R}")
    for f, bpm in zip(fragments, tempo_plan):
        print(f"    {f.name[:22]:<22} {f.tempo_bpm:>6.1f} → {bpm:>6.1f} BPM")


def print_layout(layout):
    R_ = _c("reset")
    print(f"\n{_c('bold')}[7] MAPA DE LA OBRA{R_}")
    for name, b0, b1 in layout:
        tag = _c("magenta") if name.startswith("puente") else _c("cyan")
        print(f"    {tag}{name[:24]:<24}{R_} compases {b0 + 1:>3}–{b1:>3}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

GLUE_PRESETS = {"sutil": {"motif": 0.35, "harmony": 0.0, "rhythm": 0.25,
                          "dynamics": 0.4},
                "medio": {"motif": 0.6, "harmony": 0.6, "rhythm": 0.5,
                          "dynamics": 0.6},
                "fuerte": {"motif": 0.9, "harmony": 0.85, "rhythm": 0.7,
                           "dynamics": 0.8}}


def parse_glue(spec, preset):
    # ── idea de unifier_external.parse_glue, con presets ──
    glue = dict(GLUE_PRESETS[preset])
    if spec:
        for part in spec.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            if k in glue:
                glue[k] = float(np.clip(float(v), 0, 1))
    return glue


def build_parser():
    p = argparse.ArgumentParser(
        prog="unifier",
        description="Fusiona secciones-ancla MIDI en una composición unificada "
                    "(tonalidad, tempo, motivo, ritmo, instrumentación y "
                    "dinámica comunes).")
    p.add_argument("midis", nargs="+",
                   help="Fragmentos MIDI en el orden emocional deseado")
    p.add_argument("--output", default="obra_unificada.mid")
    p.add_argument("--anchor", type=int, default=0,
                   help="Índice del fragmento ancla (default: 0)")
    p.add_argument("--key", default=None,
                   help="Tonalidad global, p.ej. C, Am, Eb, F#m")
    p.add_argument("--tempo", type=float, default=None)
    p.add_argument("--tempo-blend", type=float, default=0.25)
    p.add_argument("--bridge-bars", type=int, default=2)
    p.add_argument("--glue", default=None,
                   help="Dosificación: motif=0.7,harmony=0.6,rhythm=0.4,"
                        "dynamics=0.6 (ejes omitidos → preset)")
    p.add_argument("--glue-preset", choices=["sutil", "medio", "fuerte"],
                   default="medio")
    p.add_argument("--motif-seed", default=None,
                   help="MIDI externo con el motivo semilla (anula el ancla)")
    p.add_argument("--snap-strength", type=float, default=0.0,
                   help="0=respetar cromatismo, 1=ajustar todo a la escala")
    p.add_argument("--climax-pos", type=float, default=0.618)
    p.add_argument("--arc-intensity", type=float, default=0.6)
    p.add_argument("--no-motif", action="store_true")
    p.add_argument("--no-rhythm", action="store_true")
    p.add_argument("--no-harmony", action="store_true")
    p.add_argument("--no-humanize", action="store_true")
    p.add_argument("--no-rit", action="store_true")
    p.add_argument("--score", action="store_true",
                   help="Imprimir score de coherencia antes→después")
    p.add_argument("--no-instr", action="store_true",
                   help="Ignorar instrumentos del ancla (paleta de cuerdas)")
    p.add_argument("--no-bridges", action="store_true")
    p.add_argument("--no-arc", action="store_true")
    p.add_argument("--no-levels", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", action="store_true",
                   help="Solo análisis y plan, sin transformar ni escribir")
    p.add_argument("--json", default=None,
                   help="Exportar decisiones y mapa a JSON")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    B, R, D = _c("bold"), _c("reset"), _c("dim")

    if len(args.midis) < 2:
        sys.exit("[unifier] Se necesitan al menos 2 fragmentos MIDI.")
    if not 0 <= args.anchor < len(args.midis):
        sys.exit(f"[unifier] --anchor fuera de rango (0–{len(args.midis)-1}).")

    glue = parse_glue(args.glue, args.glue_preset)
    if args.no_motif:
        glue["motif"] = 0.0
    if args.no_rhythm:
        glue["rhythm"] = 0.0
    if args.no_harmony:
        glue["harmony"] = 0.0

    print(f"{B}══ UNIFIER v2.0 ══{R}  {len(args.midis)} fragmentos, "
          f"ancla = #{args.anchor}, glue = " +
          ", ".join(f"{k}={v}" for k, v in glue.items()))

    # [1] análisis ----------------------------------------------------------
    fragments = [Fragment(p).load() for p in args.midis]
    anchor = fragments[args.anchor]

    # [2] plan tonal ---------------------------------------------------------
    target_key = parse_key_arg(args.key) if args.key else anchor.key
    # métricas "antes": tras cargar, antes de transformar (orden ancla primero
    # para la correlación de groove)
    ordered = [anchor] + [f for i, f in enumerate(fragments)
                          if i != args.anchor]
    motif_probe = extract_motif_ngram(anchor, args.motif_seed)
    before = coherence_metrics(ordered, motif_probe, target_key)

    tonal_decisions = plan_tonal(fragments, target_key)
    scale = _get_scale(target_key)

    # [3] plan de tempo ----------------------------------------------------------
    target_bpm, tempo_plan = plan_tempo(fragments, args.tempo,
                                        float(np.clip(args.tempo_blend, 0, 1)))

    print_analysis(fragments, args.anchor, target_key, target_bpm,
                   tempo_plan, tonal_decisions)
    if args.report:
        return

    # [4] motivo ancla ------------------------------------------------------------
    motif = None
    if glue["motif"] > 0.05:
        print(f"\n{B}[4] MOTIVO ANCLA{R}")
        motif = extract_motif_ngram(anchor, args.motif_seed, verbose=True)
        if motif is None:
            print("    (material insuficiente: siembra desactivada)")

    # [5] unificación --------------------------------------------------------------
    print(f"\n{B}[5] UNIFICACIÓN{R}")
    if not args.no_levels and glue["dynamics"] > 0.05:
        gm = unify_levels(fragments)
        print(f"    niveles dinámicos igualados (media global {gm:.0f})")
    if glue["harmony"] >= 0.5:
        total_h = sum(nudge_harmony(f, target_key, glue["harmony"],
                                    verbose=True) for f in fragments)
        if total_h == 0:
            print("    armonía: todos los compases ya eran diatónicos")
    if motif:
        total_inj = sum(
            seed_motif(f, motif, scale, glue["motif"], rng, verbose=True)
            for i, f in enumerate(fragments) if i != args.anchor)
        print(f"    siembras de motivo: {total_inj}")
    if glue["rhythm"] > 0.05:
        agrid = rhythm_grid(anchor)
        moved = sum(attract_rhythm(f, agrid, glue["rhythm"])
                    for i, f in enumerate(fragments) if i != args.anchor)
        print(f"    ataques imantados al groove ancla: {moved}  "
              f"{D}{sparkline(agrid)}{R}")
    if args.snap_strength > 0:
        snapped = 0
        for f in fragments:
            for sid, st in f.streams.items():
                if f.roles.get(sid) == "percusion":
                    continue
                for n in st["notes"]:
                    tgt = _snap_to_scale(n[1], scale)
                    if tgt != n[1] and rng.random() < args.snap_strength:
                        n[1] = tgt
                        snapped += 1
        print(f"    notas ajustadas a la escala global: {snapped}")

    # [6] puentes + crossfades -------------------------------------------------------
    bridge_bars = 0 if args.no_bridges else max(0, args.bridge_bars)
    bridges = []
    if bridge_bars > 0:
        print(f"\n{B}[6] PUENTES{R}  ({bridge_bars} compases: "
              f"fin-de-A → IV → V7 → inicio-de-B en {target_key})")
        vel_ref = float(np.mean([f.vel_mean for f in fragments]))
        for i in range(len(fragments) - 1):
            crossfade(fragments[i], fragments[i + 1])
            bridges.append(build_bridge(fragments[i], fragments[i + 1],
                                        bridge_bars, scale, target_key,
                                        vel_ref, motif=motif))
            print(f"    {fragments[i].name} ⇢ {fragments[i+1].name}")

    # [7] ensamblaje ------------------------------------------------------------------
    palette = resolve_palette(anchor, args.no_instr)
    role_notes, tempo_events, total_bars, layout = assemble(
        fragments, bridges, tempo_plan, target_bpm, palette, bridge_bars)
    print_layout(layout)
    pal_txt = ", ".join(f"{r}={p}" for r, p in palette.items())
    print(f"    instrumentación única (GM): {pal_txt}")

    # [8] arco global + pulido -------------------------------------------------------
    if not args.no_arc:
        before_e, after_e = reshape_arc(role_notes, TPB * 4, total_bars,
                                        args.climax_pos,
                                        args.arc_intensity * glue["dynamics"]
                                        / 0.6)
        print(f"\n{B}[8] ARCO GLOBAL{R}  clímax en "
              f"{args.climax_pos:.0%} (compás "
              f"~{int(args.climax_pos * total_bars)})")
        print(f"    antes   {D}{sparkline(before_e)}{R}")
        print(f"    después {_c('green')}{sparkline(after_e)}{R}")
    removed = validate_and_fix(role_notes)
    if removed:
        print(f"    duplicados eliminados: {removed}")
    if not args.no_humanize:
        humanize(role_notes, rng)
    if not args.no_rit:
        add_final_rit(tempo_events, total_bars * TPB * 4)

    # [9] score + escritura --------------------------------------------------------------
    after = coherence_metrics(ordered, motif, target_key,
                              tempo_plan=tempo_plan)
    if args.score:
        print_score(before, after)

    if args.json:
        payload = {
            "target_key": target_key, "target_bpm": target_bpm,
            "anchor": fragments[args.anchor].name, "glue": glue,
            "tonal_decisions": tonal_decisions,
            "tempo_plan": dict(zip([f.name for f in fragments], tempo_plan)),
            "palette": palette,
            "motif": ({"intervals": motif["intervals"],
                       "salience": motif["salience"]} if motif else None),
            "layout": [{"section": n, "from_bar": b0 + 1, "to_bar": b1}
                       for n, b0, b1 in layout],
            "coherence_before": before, "coherence_after": after,
            "total_bars": total_bars,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\n    plan exportado → {args.json}")

    if args.dry_run:
        print(f"\n{B}[9]{R} --dry-run: no se escribe MIDI.")
        return
    out = write_midi(role_notes, tempo_events, palette, args.output)
    n_notes = sum(len(v) for v in role_notes.values())
    print(f"\n{B}[9] ESCRITO{R}  {_c('green')}{out}{R}  "
          f"({total_bars} compases, {n_notes} notas)")


if __name__ == "__main__":
    main()
