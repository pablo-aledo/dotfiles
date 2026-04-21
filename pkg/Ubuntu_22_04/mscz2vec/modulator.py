#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MODULATOR  v2.0                                      ║
║         Modulación continua y musicalmente informada de parámetros MIDI     ║
║                                                                              ║
║  MEJORAS v2.0 sobre v1.0:                                                   ║
║  [1] Análisis armónico por compás — tonalidad local, perfiles Krumhansl     ║
║  [2] Separación de voces — melodía / bajo / armonía tratadas individualmente║
║  [3] Density con grid rítmica — notas de paso en subdivisiones coherentes   ║
║  [4] Tension con voice-leading — séptimas/novenas con resolución planificada║
║  [5] Scope por voz con offset configurable (stagger entre voces)            ║
║  [6] Detección de estructura formal — respeta cadencias y cierres de frase  ║
║  [7] Smoothing entre compases — crossfade en boundaries de parámetros       ║
║  [8] Rhythm con groove map real — amplifica micro-timings del propio MIDI   ║
║  [9] Preview mode — piano roll ASCII + curva antes de generar               ║
║  [10] API pública modulate_midi() para integración con mutator y otros      ║
║                                                                              ║
║  PARÁMETROS:                                                                 ║
║    density · tension · dynamics · rhythm · register · articulation          ║
║    spread · contour · tempo · brightness                                     ║
║                                                                              ║
║  CURVAS:                                                                     ║
║    linear · exponential · logarithmic · sigmoid · arc · arc_down            ║
║    s_curve · pulse · plateau · wave · random_walk · custom                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python modulator.py obra.mid --param density --curve arc                 ║
║    python modulator.py obra.mid --params dynamics tension --curve sigmoid   ║
║    python modulator.py obra.mid --param tension --voice melody              ║
║    python modulator.py obra.mid --param all --curve s_curve --preview       ║
║    python modulator.py obra.mid --param density --voice-stagger 2           ║
║    python modulator.py obra.mid --param rhythm --use-groove-map             ║
║    python modulator.py obra.mid --param all --smooth-boundary 2             ║
║    python modulator.py obra.mid --param density --curve custom              ║
║                      --control-points "0:0.1, 4:0.9, 8:0.5, 16:0.8"       ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --param / --params    Parámetro(s) a modular (o 'all')                   ║
║    --direction           increasing | decreasing | none                      ║
║    --curve               Forma de curva (default: linear)                   ║
║    --from / --to         Rango de la curva 0-1                              ║
║    --intensity           Intensidad global 0-1 (default: 1.0)              ║
║    --scope               Compases S:E (ej: --scope 8:24)                    ║
║    --voice               all | melody | bass | harmony (default: all)       ║
║    --voice-stagger N     Offset de compases entre voces (default: 0)       ║
║    --use-groove-map      Usar groove map real del MIDI (rhythm)             ║
║    --smooth-boundary N   Beats de crossfade en boundaries (default: 0)     ║
║    --respect-cadences    No modular notas en cadencias finales de frase     ║
║    --periods N           Para curvas wave/pulse: número de períodos         ║
║    --steepness F         Para sigmoid/exponential: pendiente 1-20           ║
║    --control-points STR  Para --curve custom: "BAR:VAL, BAR:VAL, ..."      ║
║    --output FILE         MIDI de salida                                      ║
║    --seed N              Semilla aleatoria (default: 42)                    ║
║    --dry-run             Mostrar curvas sin generar MIDI                    ║
║    --preview             Piano roll ASCII + curvas antes de generar         ║
║    --plot                Visualizar curvas con matplotlib                   ║
║    --verbose             Informe detallado                                  ║
║                                                                              ║
║  COMO MÓDULO (para mutator u otros scripts):                                ║
║    from modulator import modulate_midi, generate_curve, MusicAnalysis       ║
║    result = modulate_midi("obra.mid", params=["tension","dynamics"],        ║
║                           curve="arc", intensity=0.8, voice="melody")       ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
║  OPCIONAL:    matplotlib (--plot)                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import re
import math
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

VERSION = "2.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

ALL_PARAMS = [
    "density", "tension", "dynamics", "rhythm", "register",
    "articulation", "spread", "contour", "tempo", "brightness",
]

ALL_CURVES = [
    "linear", "exponential", "logarithmic", "sigmoid",
    "arc", "arc_down", "s_curve", "pulse", "plateau", "wave",
    "random_walk", "custom",
]

# Perfiles de Krumhansl-Schmuckler (normalizados)
_KK_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KK_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

SCALE_INTERVALS = {
    "major":          [0,2,4,5,7,9,11],
    "minor":          [0,2,3,5,7,8,10],
    "dorian":         [0,2,3,5,7,9,10],
    "phrygian":       [0,1,3,5,7,8,10],
    "lydian":         [0,2,4,6,7,9,11],
    "mixolydian":     [0,2,4,5,7,9,10],
    "locrian":        [0,1,3,5,6,8,10],
    "harmonic_minor": [0,2,3,5,7,8,11],
}

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

PARAM_META = {
    "density":      {"desc": "Notas por tiempo", "increasing": "ornamentos, notas de paso", "decreasing": "skeleton, silencios"},
    "tension":      {"desc": "Disonancia armónica", "increasing": "cromatismo, séptimas", "decreasing": "resolución consonante"},
    "dynamics":     {"desc": "Envolvente de velocidades", "increasing": "crescendo", "decreasing": "decrescendo"},
    "rhythm":       {"desc": "Groove y swing", "increasing": "swing, síncopa", "decreasing": "cuantizado"},
    "register":     {"desc": "Altura media de la textura", "increasing": "hacia el agudo", "decreasing": "hacia el grave"},
    "articulation": {"desc": "Duración de notas", "increasing": "legato", "decreasing": "staccato"},
    "spread":       {"desc": "Amplitud del voicing", "increasing": "voicing abierto", "decreasing": "voicing cerrado"},
    "contour":      {"desc": "Arco melódico", "increasing": "ascendente", "decreasing": "descendente"},
    "tempo":        {"desc": "Velocidad del pulso", "increasing": "accelerando", "decreasing": "ritardando"},
    "brightness":   {"desc": "Brillo tímbrico", "increasing": "CC74 alto, vel. altas", "decreasing": "CC74 bajo, suave"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  [1] ANÁLISIS ARMÓNICO + [2] VOCES + [6] ESTRUCTURA + [8] GROOVE MAP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BarHarmony:
    tonic_pc: int = 0
    mode: str = "major"
    scale_pcs: set = field(default_factory=set)
    is_cadence: bool = False
    is_phrase_end: bool = False
    tension_level: float = 0.5
    dominant_pitch: int = 60


@dataclass
class GrooveMapData:
    timing_offsets: dict = field(default_factory=dict)
    velocity_map: dict = field(default_factory=dict)
    resolution: int = 8
    trained: bool = False

    def get_offset_ticks(self, abs_tick: int, tpb: int, ts_num: int = 4) -> int:
        if not self.trained: return 0
        grid_step = tpb // self.resolution
        if grid_step < 1: return 0
        beat_ticks = abs_tick % (tpb * ts_num)
        gpos = int(round(beat_ticks / grid_step)) % (ts_num * self.resolution)
        offset_b = self.timing_offsets.get(gpos, 0.0)
        return int(offset_b * tpb)


class MusicAnalysis:
    """Análisis musical completo: tonalidad, voces, groove, estructura formal."""

    def __init__(self, path: str, verbose: bool = False):
        import mido
        self.path = path
        self.verbose = verbose
        self.mid = mido.MidiFile(path)
        self.tpb = self.mid.ticks_per_beat

        self.tempo_us = 500_000
        self.ts_num = 4
        self.ts_den = 4
        self._parse_meta()

        self.tpbar = self.tpb * 4 * self.ts_num // self.ts_den
        self.bpm = round(60_000_000 / self.tempo_us, 1)

        self.raw_notes = self._extract_notes()
        total_ticks = max((n["end"] for n in self.raw_notes), default=self.tpbar * 16)
        for track in self.mid.tracks:
            total_ticks = max(total_ticks, sum(m.time for m in track))
        self.n_bars = max(1, int(math.ceil(total_ticks / self.tpbar)))

        # Análisis
        # Pre-compute mean_pitch needed by _split_voices
        pitches = [n["pitch"] for n in self.raw_notes]
        self.mean_pitch = float(np.mean(pitches)) if pitches else 60.0

        self.bar_harmony: list = self._analyze_harmony()   # [1]
        self.melody_set, self.bass_set, self.harmony_set = self._split_voices()  # [2]
        self.phrase_boundaries: list = self._detect_phrases()   # [6]
        self.groove_map: GrooveMapData = self._build_groove_map()  # [8]

        vels = [n["vel"] for n in self.raw_notes] or [64]
        self.mean_velocity = float(np.mean(vels))
        self.std_velocity  = float(np.std(vels))
        self.mean_dur      = float(np.mean([n["dur"] for n in self.raw_notes] or [self.tpb]))
        self.notes_per_bar = len(self.raw_notes) / max(1, self.n_bars)

        if verbose:
            print(f"  [análisis] {self.n_bars} compases | {self.bpm} BPM | "
                  f"{len(self.raw_notes)} notas | vel={self.mean_velocity:.0f} | "
                  f"pitch_medio={self.mean_pitch:.0f}")
            print(f"  [voces]    melodía={len(self.melody_set)} "
                  f"bajo={len(self.bass_set)} armonía={len(self.harmony_set)}")
            print(f"  [groove]   {'entrenado' if self.groove_map.trained else 'no disponible'} | "
                  f"frases: {self.phrase_boundaries[:8]}")

    def _parse_meta(self):
        for track in self.mid.tracks:
            for msg in track:
                if msg.is_meta:
                    if msg.type == "set_tempo":
                        self.tempo_us = msg.tempo
                    elif msg.type == "time_signature":
                        self.ts_num = msg.numerator
                        self.ts_den = msg.denominator

    def _extract_notes(self) -> list:
        notes = []
        for ti, track in enumerate(self.mid.tracks):
            at = 0
            pending = {}
            for msg in track:
                at += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    pending[msg.note] = (at, msg.velocity, getattr(msg, "channel", 0))
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in pending:
                        s, v, ch = pending.pop(msg.note)
                        notes.append({"pitch": msg.note, "vel": v, "start": s, "end": at,
                                      "dur": at - s, "bar": s // self.tpbar,
                                      "track": ti, "channel": ch})
        return sorted(notes, key=lambda n: n["start"])

    # ── [1] Krumhansl-Schmuckler por compás ───────────────────────────────────
    def _analyze_harmony(self) -> list:
        harmonies = []
        for bar in range(self.n_bars):
            bar_notes = [n for n in self.raw_notes if n["bar"] == bar]
            h = BarHarmony()
            if bar_notes:
                pcp = np.zeros(12)
                for n in bar_notes:
                    pcp[n["pitch"] % 12] += n["dur"]
                if pcp.sum() > 0:
                    pcp /= pcp.sum()
                    best_r, best_tc, best_mode = -2.0, 0, "major"
                    for tonic in range(12):
                        r_maj = float(np.corrcoef(pcp, np.roll(_KK_MAJOR / _KK_MAJOR.sum(), tonic))[0,1])
                        r_min = float(np.corrcoef(pcp, np.roll(_KK_MINOR / _KK_MINOR.sum(), tonic))[0,1])
                        if r_maj > best_r: best_r, best_tc, best_mode = r_maj, tonic, "major"
                        if r_min > best_r: best_r, best_tc, best_mode = r_min, tonic, "minor"
                    h.tonic_pc = best_tc
                    h.mode = best_mode
                    ivs = SCALE_INTERVALS.get(best_mode, SCALE_INTERVALS["major"])
                    h.scale_pcs = set((best_tc + i) % 12 for i in ivs)
                    non_d = sum(1 for n in bar_notes if (n["pitch"] % 12) not in h.scale_pcs)
                    h.tension_level = non_d / max(1, len(bar_notes))
                    h.dominant_pitch = max(bar_notes, key=lambda n: n["vel"])["pitch"]
            harmonies.append(h)
        return harmonies

    def get_bar_scale_pcs(self, bar: int) -> set:
        bar = max(0, min(bar, self.n_bars - 1))
        h = self.bar_harmony[bar]
        return h.scale_pcs if h.scale_pcs else set([0,2,4,5,7,9,11])

    def snap_to_scale(self, pitch: int, bar: int) -> int:
        pcs = self.get_bar_scale_pcs(bar)
        pc = pitch % 12
        if pc in pcs: return pitch
        best_pc, best_d = pc, 100
        for spc in pcs:
            d = min(abs(spc - pc), 12 - abs(spc - pc))
            if d < best_d: best_d, best_pc = d, spc
        diff = best_pc - pc
        if diff > 6:  diff -= 12
        if diff < -6: diff += 12
        return int(np.clip(pitch + diff, 0, 127))

    def is_diatonic(self, pitch: int, bar: int) -> bool:
        return (pitch % 12) in self.get_bar_scale_pcs(bar)

    # ── [2] separación de voces ───────────────────────────────────────────────
    def _split_voices(self):
        CLUSTER_WINDOW = max(1, self.tpb // 8)
        mel_ids, bas_ids, har_ids = set(), set(), set()
        sorted_notes = sorted(range(len(self.raw_notes)), key=lambda i: self.raw_notes[i]["start"])
        used = [False] * len(self.raw_notes)

        for i in sorted_notes:
            if used[i]: continue
            cluster = [i]
            for j in sorted_notes:
                if j != i and not used[j]:
                    if abs(self.raw_notes[j]["start"] - self.raw_notes[i]["start"]) <= CLUSTER_WINDOW:
                        cluster.append(j); used[j] = True
                    elif self.raw_notes[j]["start"] - self.raw_notes[i]["start"] > CLUSTER_WINDOW:
                        break
            used[i] = True

            pitches = [(self.raw_notes[k]["pitch"], k) for k in cluster]
            pitches.sort()
            if len(cluster) == 1:
                p, k = pitches[0]
                if p >= self.mean_pitch: mel_ids.add(k)
                else: bas_ids.add(k)
            else:
                mel_ids.add(pitches[-1][1])   # más aguda
                bas_ids.add(pitches[0][1])    # más grave
                for _, k in pitches[1:-1]:
                    har_ids.add(k)

        mel = [self.raw_notes[i] for i in mel_ids]
        bas = [self.raw_notes[i] for i in bas_ids]
        har = [self.raw_notes[i] for i in har_ids]
        return mel, bas, har

    def is_in_voice(self, note: dict, voice: str) -> bool:
        if voice == "all": return True
        if voice == "melody":  return note in self.melody_set
        if voice == "bass":    return note in self.bass_set
        if voice == "harmony": return note in self.harmony_set
        return True

    # ── [6] estructura formal ─────────────────────────────────────────────────
    def _detect_phrases(self) -> list:
        boundaries = set()
        prev_end = 0
        for n in self.raw_notes:
            if n["start"] - prev_end > self.tpb * 1.5:
                boundaries.add(max(0, self._bar(n["start"]) - 1))
            prev_end = max(prev_end, n["end"])
        for bar in range(0, self.n_bars, 4): boundaries.add(bar)
        for bar in range(0, self.n_bars, 8): boundaries.add(bar)
        for bar in boundaries:
            if 0 <= bar < len(self.bar_harmony):
                self.bar_harmony[bar].is_phrase_end = True
                self.bar_harmony[bar].is_cadence = True
        return sorted(boundaries)

    def is_cadence_bar(self, bar: int) -> bool:
        return 0 <= bar < len(self.bar_harmony) and self.bar_harmony[bar].is_cadence

    # ── [8] groove map ────────────────────────────────────────────────────────
    def _build_groove_map(self) -> GrooveMapData:
        gm = GrooveMapData(resolution=8)
        grid_step = self.tpb // gm.resolution
        if grid_step < 1: return gm
        timing_raw, vel_raw = defaultdict(list), defaultdict(list)
        for n in self.raw_notes:
            beat_ticks = n["start"] % self.tpbar
            gpos = int(round(beat_ticks / grid_step)) % (self.ts_num * gm.resolution)
            ideal = gpos * grid_step
            deviation = (beat_ticks - ideal) / self.tpb
            if abs(deviation) < 0.15:
                timing_raw[gpos].append(deviation)
                vel_raw[gpos].append(n["vel"])
        if len(timing_raw) >= 4:
            gm.timing_offsets = {k: float(np.mean(v)) for k, v in timing_raw.items()}
            gm.velocity_map   = {k: list(v) for k, v in vel_raw.items()}
            gm.trained = True
        return gm

    def _bar(self, t: int) -> int:
        return t // self.tpbar

    def bar_of(self, t: int) -> int:
        return t // self.tpbar

    def in_scope(self, bar: int, s, e) -> bool:
        lo = s if s is not None else 0
        hi = e if e is not None else self.n_bars
        return lo <= bar < hi


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE CURVAS
# ══════════════════════════════════════════════════════════════════════════════

def generate_curve(n_bars: int, curve: str, from_val: float = 0.0, to_val: float = 1.0,
                   direction: str = "increasing", steepness: float = 6.0,
                   periods: float = 2.0, seed: int = 42,
                   control_points: Optional[str] = None) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(n_bars, 1))

    if   curve == "linear":       raw = t.copy()
    elif curve == "exponential":  raw = t ** steepness
    elif curve == "logarithmic":  raw = np.sqrt(t)
    elif curve == "sigmoid":
        raw = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
        raw = (raw - raw[0]) / (raw[-1] - raw[0] + 1e-9)
    elif curve == "arc":          raw = np.sin(np.pi * t)
    elif curve == "arc_down":     raw = 1.0 - np.sin(np.pi * t)
    elif curve == "s_curve":      raw = 3*t**2 - 2*t**3
    elif curve == "pulse":        raw = 0.5*(1.0 - np.cos(2*np.pi*periods*t))
    elif curve == "plateau":
        raw = np.zeros(n_bars)
        for i, ti in enumerate(t):
            if   ti < 0.15: raw[i] = ti / 0.15
            elif ti < 0.85: raw[i] = 1.0
            else:           raw[i] = 1.0 - (ti - 0.85) / 0.15
        raw = np.clip(raw, 0, 1)
    elif curve == "wave":         raw = 0.5*(1.0 - np.cos(2*np.pi*periods*t))
    elif curve == "random_walk":
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 0.12, n_bars)
        raw = np.cumsum(steps)
        raw -= raw.min()
        span = raw.max() - raw.min()
        raw = raw / span if span > 1e-6 else np.full(n_bars, 0.5)
        w = max(2, n_bars // 8)
        padded = np.pad(raw, (w//2, w - w//2 - 1), mode="edge")
        smoothed = np.array([padded[i:i+w].mean() for i in range(n_bars)])
        raw = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)
    elif curve == "custom":       raw = _parse_control_points(control_points, n_bars)
    else:
        raise ValueError(f"Curva desconocida: '{curve}'. Opciones: {ALL_CURVES}")

    if direction == "decreasing": raw = 1.0 - raw
    out = from_val + raw * (to_val - from_val)
    return np.clip(out, 0.0, 1.0)


def _parse_control_points(spec: Optional[str], n_bars: int) -> np.ndarray:
    if not spec: return np.linspace(0.0, 1.0, n_bars)
    try:
        pairs = []
        for token in re.split(r"[,;]\s*", spec.strip()):
            token = token.strip()
            if not token: continue
            bar_s, val_s = token.split(":")
            pairs.append((int(bar_s.strip()), float(val_s.strip())))
        pairs.sort(key=lambda x: x[0])
    except Exception as e:
        raise ValueError(f"--control-points mal formado: {e}")
    bars_cp = np.array([p[0] for p in pairs], dtype=float)
    vals_cp = np.array([p[1] for p in pairs], dtype=float)
    return np.clip(np.interp(np.arange(n_bars, dtype=float), bars_cp, vals_cp), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  [7] SMOOTHING + [5] VOICE SCOPE
# ══════════════════════════════════════════════════════════════════════════════

def smooth_factor(abs_tick: int, tpbar: int, tpb: int, smooth_beats: int) -> float:
    """Factor de suavizado [0,1] en los boundaries de compás."""
    if smooth_beats <= 0: return 1.0
    pos = abs_tick % tpbar
    sm_ticks = smooth_beats * tpb
    if pos < sm_ticks:         return pos / sm_ticks
    if pos > tpbar - sm_ticks: return (tpbar - pos) / sm_ticks
    return 1.0


def get_voice_scope(scope_start, scope_end, voice: str, stagger: int, n_bars: int):
    offsets = {"melody": 0, "harmony": stagger, "bass": stagger * 2, "all": 0}
    off = offsets.get(voice, 0)
    s = (scope_start or 0) + off
    e = (scope_end   or n_bars) + off
    return max(0, s), min(n_bars, e)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE MODULACIÓN v2
# ══════════════════════════════════════════════════════════════════════════════

def _lerp(a, b, t):
    return float(a) + (float(b) - float(a)) * float(np.clip(t, 0, 1))


class ModulationEngine:

    def __init__(self, mid, A: MusicAnalysis, verbose=False,
                 smooth_beats=0, respect_cadences=False,
                 use_groove_map=False, voice="all", voice_stagger=0):
        import mido as _mido
        self.mido = _mido
        self.mid  = mid
        self.A    = A
        self.verbose = verbose
        self.smooth_beats = smooth_beats
        self.respect_cadences = respect_cadences
        self.use_groove_map = use_groove_map
        self.voice = voice
        self.voice_stagger = voice_stagger
        self.tpb   = A.tpb
        self.tpbar = A.tpbar
        self.n_bars = A.n_bars

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bar(self, t): return t // self.tpbar
    def _cv(self, curve, bar): return float(curve[min(bar, len(curve)-1)])
    def _sm(self, at):  return smooth_factor(at, self.tpbar, self.tpb, self.smooth_beats)
    def _skip(self, bar): return self.respect_cadences and self.A.is_cadence_bar(bar)
    def _in_scope(self, bar, s, e): return self.A.in_scope(bar, s, e)
    def _in_voice(self, note): return self.A.is_in_voice(note, self.voice)

    def _collect(self):
        """Devuelve buckets[ti] = list of (abs_tick, msg)."""
        buckets = [[] for _ in self.mid.tracks]
        for ti, track in enumerate(self.mid.tracks):
            at = 0
            for msg in track:
                at += msg.time
                buckets[ti].append((at, msg))
        return buckets

    def _rebuild(self, buckets):
        import mido
        for ti in range(len(self.mid.tracks)):
            evs = sorted(buckets[ti], key=lambda x: x[0])
            new_track = mido.MidiTrack()
            prev = 0
            for at, msg in evs:
                delta = max(0, at - prev)
                new_track.append(msg.copy(time=delta))
                prev = at
            self.mid.tracks[ti] = new_track

    def _apply_pitch_map(self, pitch_map: dict):
        """pitch_map: {(ti, at, note) → new_note}"""
        buckets = self._collect()
        new_buckets = [[] for _ in self.mid.tracks]
        for ti, evs in enumerate(buckets):
            for at, msg in evs:
                key = (ti, at, msg.note) if hasattr(msg, "note") else None
                if key and key in pitch_map and msg.type in ("note_on", "note_off"):
                    new_buckets[ti].append((at, msg.copy(note=pitch_map[key])))
                else:
                    new_buckets[ti].append((at, msg))
        self._rebuild(new_buckets)

    # ── [3] density ───────────────────────────────────────────────────────────

    def apply_density(self, curve, intensity, s, e):
        """
        Grid rítmica coherente: añade notas en posiciones vacías de la grid de 1/8
        usando grados diatónicos del compás; elimina notas no estructurales.
        """
        import mido
        grid_step = max(1, self.tpb // 2)
        BEAT_WEIGHTS = {0:4, 1:1, 2:2, 3:1}  # peso de posición en 4/4

        occupied = defaultdict(set)
        for n in self.A.raw_notes:
            occupied[n["bar"]].add((n["start"] % self.tpbar) // grid_step)

        removed = set()   # (track, start, pitch)
        added   = []      # (ti, on, off, pitch, vel, ch)

        for n in self.A.raw_notes:
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue
            if not self._in_voice(n): continue
            cv = self._cv(curve, bar)
            rng = np.random.default_rng(hash((n["track"], n["start"], n["pitch"])) % (2**31))

            # Decreasing: eliminar notas en posiciones débiles y cortas
            if cv < 0.45:
                gpos = (n["start"] % self.tpbar) // grid_step
                beat_w = BEAT_WEIGHTS.get(int(gpos) % 4, 1)
                p_rem = (0.45 - cv) * 2.2 * intensity * self._sm(n["start"])
                short_f = 1.5 if n["dur"] < self.tpb // 4 else 0.6
                weak_f  = 1.3 if beat_w < 2 else 0.5
                if rng.random() < p_rem * short_f * weak_f:
                    removed.add((n["track"], n["start"], n["pitch"]))
            # Increasing: notas de paso diatónicas en huecos
            elif cv > 0.55:
                p_add = (cv - 0.55) * 2.2 * intensity * self._sm(n["start"])
                n_slots = self.tpbar // grid_step
                for gp in range(n_slots):
                    if gp in occupied[bar]: continue
                    if rng.random() > p_add * 0.4: continue
                    on_t = bar * self.tpbar + gp * grid_step
                    dur  = max(self.tpb // 8, grid_step - self.tpb // 16)
                    bar_ps = [m["pitch"] for m in self.A.raw_notes if m["bar"] == bar]
                    base_p = int(np.mean(bar_ps)) if bar_ps else int(self.A.mean_pitch)
                    new_p  = self.A.snap_to_scale(base_p + rng.integers(-3, 4), bar)
                    new_p  = int(np.clip(new_p, 36, 96))
                    vel    = int(np.clip(self.A.mean_velocity * 0.75, 20, 100))
                    added.append((n["track"], on_t, on_t + dur, new_p, vel, n.get("channel", 0)))
                    occupied[bar].add(gp)

        # Reconstruir
        all_msgs = defaultdict(list)
        for ti, track in enumerate(self.mid.tracks):
            at = 0
            for msg in track:
                at += msg.time
                if msg.type not in ("note_on", "note_off"):
                    all_msgs[ti].append((at, msg))

        for n in self.A.raw_notes:
            if (n["track"], n["start"], n["pitch"]) in removed: continue
            ch = n.get("channel", 0)
            all_msgs[n["track"]].append((n["start"], mido.Message("note_on",  channel=ch, note=n["pitch"], velocity=n["vel"], time=0)))
            all_msgs[n["track"]].append((n["end"],   mido.Message("note_off", channel=ch, note=n["pitch"], velocity=0, time=0)))

        for (ti, on_t, off_t, p, v, ch) in added:
            all_msgs[ti].append((on_t,  mido.Message("note_on",  channel=ch, note=p, velocity=v, time=0)))
            all_msgs[ti].append((off_t, mido.Message("note_off", channel=ch, note=p, velocity=0, time=0)))

        new_tracks = []
        for ti in range(len(self.mid.tracks)):
            evs = sorted(all_msgs[ti], key=lambda x: x[0])
            new_track = mido.MidiTrack()
            prev = 0
            for at, msg in evs:
                new_track.append(msg.copy(time=max(0, at - prev)))
                prev = at
            new_tracks.append(new_track)
        self.mid.tracks = new_tracks

    # ── [4] tension ───────────────────────────────────────────────────────────

    def apply_tension(self, curve, intensity, s, e):
        """
        Increasing: desplaza notas diatónicas a sus vecinos cromáticos tensos.
        Decreasing: resuelve no-diatónicas al grado más cercano (voice-leading mínimo).
        Nunca toca notas en cadencias si respect_cadences.
        """
        pitch_map = {}
        for n in self.A.raw_notes:
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue
            if not self._in_voice(n): continue
            cv = self._cv(curve, bar)
            eff = intensity * self._sm(n["start"])
            rng = np.random.default_rng(hash((n["track"], n["start"], n["pitch"])) % (2**31))
            is_d = self.A.is_diatonic(n["pitch"], bar)
            new_p = n["pitch"]

            if cv > 0.55:
                p_tens = (cv - 0.55) * 2.2 * eff
                if is_d and rng.random() < p_tens * 0.35:
                    cands = [n["pitch"] + d for d in (-1, 1)
                             if 0 <= n["pitch"]+d <= 127
                             and not self.A.is_diatonic(n["pitch"]+d, bar)]
                    if cands: new_p = int(rng.choice(cands))
            elif cv < 0.45:
                p_res = (0.45 - cv) * 2.2 * eff
                if not is_d and rng.random() < p_res * 0.65:
                    new_p = self.A.snap_to_scale(n["pitch"], bar)

            if new_p != n["pitch"]:
                pitch_map[(n["track"], n["start"], n["pitch"])] = new_p

        self._apply_pitch_map(pitch_map)

    # ── dynamics ──────────────────────────────────────────────────────────────

    def apply_dynamics(self, curve, intensity, s, e):
        buckets = self._collect()
        new_b = [[] for _ in self.mid.tracks]
        for ti, evs in enumerate(buckets):
            for at, msg in evs:
                bar = self._bar(at)
                if msg.type in ("note_on","note_off") and self._in_scope(bar, s, e) and not self._skip(bar):
                    cv = self._cv(curve, bar)
                    factor = _lerp(1.0, 0.35 + 1.65 * cv, intensity * self._sm(at))
                    new_b[ti].append((at, msg.copy(velocity=int(np.clip(msg.velocity * factor, 1, 127)))))
                else:
                    new_b[ti].append((at, msg))
        self._rebuild(new_b)

    # ── register ──────────────────────────────────────────────────────────────

    def apply_register(self, curve, intensity, s, e):
        VOICE_RANGES = {"melody":(60,96), "bass":(28,60), "harmony":(48,84), "all":(28,96)}
        lo_lim, hi_lim = VOICE_RANGES.get(self.voice, (28,96))
        pitch_map = {}
        for n in self.A.raw_notes:
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue
            if not self._in_voice(n): continue
            cv = self._cv(curve, bar)
            shift = int((cv - 0.5) * 2 * 12 * intensity * self._sm(n["start"]))
            new_p = int(np.clip(n["pitch"] + shift, lo_lim, hi_lim))
            if new_p != n["pitch"]: pitch_map[(n["track"], n["start"], n["pitch"])] = new_p
        self._apply_pitch_map(pitch_map)

    # ── articulation ──────────────────────────────────────────────────────────

    def apply_articulation(self, curve, intensity, s, e):
        import mido
        dur_map = {}  # (track, start, pitch) → new_end
        for n in self.A.raw_notes:
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue
            if not self._in_voice(n): continue
            cv = self._cv(curve, bar)
            factor = _lerp(1.0, 0.25 + 0.8 * cv, intensity * self._sm(n["start"]))
            new_dur = max(1, int(n["dur"] * factor))
            dur_map[(n["track"], n["start"], n["pitch"])] = (n["end"], n["start"] + new_dur, n.get("channel",0))

        all_msgs = defaultdict(list)
        for ti, track in enumerate(self.mid.tracks):
            at = 0
            for msg in track:
                at += msg.time
                if msg.type not in ("note_on","note_off"):
                    all_msgs[ti].append((at, msg))

        for n in self.A.raw_notes:
            key = (n["track"], n["start"], n["pitch"])
            ch = n.get("channel", 0)
            if key in dur_map:
                _, new_end, ch = dur_map[key]
                all_msgs[n["track"]].append((n["start"], mido.Message("note_on",  channel=ch, note=n["pitch"], velocity=n["vel"], time=0)))
                all_msgs[n["track"]].append((new_end,    mido.Message("note_off", channel=ch, note=n["pitch"], velocity=0, time=0)))
            else:
                all_msgs[n["track"]].append((n["start"], mido.Message("note_on",  channel=ch, note=n["pitch"], velocity=n["vel"], time=0)))
                all_msgs[n["track"]].append((n["end"],   mido.Message("note_off", channel=ch, note=n["pitch"], velocity=0, time=0)))

        new_tracks = []
        for ti in range(len(self.mid.tracks)):
            evs = sorted(all_msgs[ti], key=lambda x: x[0])
            new_track = mido.MidiTrack()
            prev = 0
            for at, msg in evs:
                new_track.append(msg.copy(time=max(0, at - prev)))
                prev = at
            new_tracks.append(new_track)
        self.mid.tracks = new_tracks

    # ── tempo ─────────────────────────────────────────────────────────────────

    def apply_tempo(self, curve, intensity, s, e):
        import mido
        base_tempo = self.A.tempo_us
        tempo_ti = 0
        for ti, track in enumerate(self.mid.tracks):
            for msg in track:
                if msg.is_meta and msg.type == "set_tempo":
                    tempo_ti = ti; break

        new_tracks = list(self.mid.tracks)
        for ti, track in enumerate(self.mid.tracks):
            at = 0
            evs = []
            for msg in track:
                at += msg.time
                evs.append((at, msg))

            modified = []
            last_bar = -1
            for at, msg in evs:
                bar = self._bar(at)
                if ti == tempo_ti and bar != last_bar and self._in_scope(bar, s, e):
                    cv = self._cv(curve, bar)
                    eff_cv = _lerp(0.5, cv, self._sm(at))
                    factor = _lerp(1.0, 0.55 + 1.05 * eff_cv, intensity)
                    new_tempo = int(np.clip(base_tempo / max(factor, 0.01), 40_000, 2_000_000))
                    modified.append((bar * self.tpbar, mido.MetaMessage("set_tempo", tempo=new_tempo, time=0)))
                    last_bar = bar
                if msg.is_meta and msg.type == "set_tempo" and self._in_scope(bar, s, e) and ti == tempo_ti:
                    continue
                modified.append((at, msg))

            modified.sort(key=lambda x: x[0])
            new_track = mido.MidiTrack()
            prev = 0
            for at, msg in modified:
                new_track.append(msg.copy(time=max(0, at - prev)))
                prev = at
            new_tracks[ti] = new_track
        self.mid.tracks = new_tracks

    # ── [8] rhythm con groove map ─────────────────────────────────────────────

    def apply_rhythm(self, curve, intensity, s, e):
        buckets = self._collect()
        new_b = [[] for _ in self.mid.tracks]
        gm = self.A.groove_map

        for ti, evs in enumerate(buckets):
            for at, msg in evs:
                bar = self._bar(at)
                if msg.type in ("note_on","note_off") and self._in_scope(bar, s, e) and not self._skip(bar):
                    cv = self._cv(curve, bar)
                    eff = cv * intensity * self._sm(at)
                    new_at = at

                    if self.use_groove_map and gm.trained:
                        grid_step = self.tpb // gm.resolution
                        beat_ticks = at % self.tpbar
                        gpos = int(round(beat_ticks / grid_step)) % (self.A.ts_num * gm.resolution)
                        offset_b = gm.timing_offsets.get(gpos, 0.0)
                        # cv<0.5: atenuar el groove; cv>0.5: amplificar
                        scale = _lerp(-1.0, 2.0, cv) * intensity
                        new_at = max(0, at + int(offset_b * self.tpb * scale))
                    else:
                        # Swing matemático en off-beats de corchea
                        subdiv = max(1, self.tpb // 2)
                        pos_in_bar = at % self.tpbar
                        if (pos_in_bar // subdiv) % 2 == 1:
                            new_at = at + int(subdiv * 0.33 * eff)

                    new_b[ti].append((new_at, msg))
                else:
                    new_b[ti].append((at, msg))
        self._rebuild(new_b)

    # ── spread ────────────────────────────────────────────────────────────────

    def apply_spread(self, curve, intensity, s, e):
        CHORD_WINDOW = max(1, self.tpb // 8)
        sorted_notes = sorted(self.A.raw_notes, key=lambda n: n["start"])
        used = [False] * len(sorted_notes)
        pitch_map = {}

        for i, n in enumerate(sorted_notes):
            if used[i]: continue
            cluster = [i]
            for j in range(i+1, len(sorted_notes)):
                if abs(sorted_notes[j]["start"] - n["start"]) <= CHORD_WINDOW:
                    cluster.append(j); used[j] = True
                elif sorted_notes[j]["start"] - n["start"] > CHORD_WINDOW:
                    break
            used[i] = True
            if len(cluster) < 2: continue
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue

            pitches = [sorted_notes[k]["pitch"] for k in cluster]
            center  = float(np.mean(pitches))
            cv = self._cv(curve, bar)
            sm = self._sm(n["start"])

            for k in cluster:
                nn = sorted_notes[k]
                if not self._in_voice(nn): continue
                dev = nn["pitch"] - center
                factor = _lerp(1.0, (cv/0.5) if cv < 0.5 else (1.0 + (cv-0.5)*2.0), intensity * sm)
                new_p = int(np.clip(center + dev * factor, 0, 127))
                if new_p != nn["pitch"]:
                    pitch_map[(nn["track"], nn["start"], nn["pitch"])] = new_p

        self._apply_pitch_map(pitch_map)

    # ── contour ───────────────────────────────────────────────────────────────

    def apply_contour(self, curve, intensity, s, e):
        """Guía el arco melódico; snap a escala después del desplazamiento."""
        pitch_map = {}
        for n in self.A.raw_notes:
            bar = n["bar"]
            if not self._in_scope(bar, s, e): continue
            if self._skip(bar): continue
            # Contour aplica preferentemente a melodía
            if self.voice != "all" and not self.A.is_in_voice(n, "melody"): continue
            cv = self._cv(curve, bar)
            shift = int((cv - 0.5) * 2 * 8 * intensity * self._sm(n["start"]))
            raw_p = int(np.clip(n["pitch"] + shift, 0, 127))
            new_p = self.A.snap_to_scale(raw_p, bar)
            if new_p != n["pitch"]:
                pitch_map[(n["track"], n["start"], n["pitch"])] = new_p
        self._apply_pitch_map(pitch_map)

    # ── brightness ────────────────────────────────────────────────────────────

    def apply_brightness(self, curve, intensity, s, e):
        import mido
        buckets = self._collect()
        new_b = [[] for _ in self.mid.tracks]
        last_cc74_bar = defaultdict(lambda: -1)

        for ti, evs in enumerate(buckets):
            for at, msg in evs:
                bar = self._bar(at)
                if self._in_scope(bar, s, e):
                    cv = self._cv(curve, bar)
                    sm = self._sm(at)
                    if bar != last_cc74_bar[ti] and not (hasattr(msg, "is_meta") and msg.is_meta):
                        ch = getattr(msg, "channel", 0)
                        new_b[ti].append((bar * self.tpbar,
                            mido.Message("control_change", channel=ch, control=74,
                                         value=int(np.clip(127 * cv, 0, 127)), time=0)))
                        last_cc74_bar[ti] = bar
                    if msg.type in ("note_on","note_off"):
                        target_v = _lerp(self.A.mean_velocity, 35 + 92 * cv, intensity * sm)
                        ratio = target_v / max(self.A.mean_velocity, 1.0)
                        new_b[ti].append((at, msg.copy(velocity=int(np.clip(msg.velocity * ratio, 1, 127)))))
                        continue
                new_b[ti].append((at, msg))
        self._rebuild(new_b)


# ══════════════════════════════════════════════════════════════════════════════
#  [9] PREVIEW ASCII
# ══════════════════════════════════════════════════════════════════════════════

def show_preview(analysis: MusicAnalysis, curves: dict, scope_start, scope_end):
    n_bars = analysis.n_bars
    lo = scope_start if scope_start is not None else 0
    hi = scope_end   if scope_end   is not None else n_bars
    cols = min(hi - lo, 40)

    print("\n  ── Piano roll (scope activo) ──")
    ROWS = list(range(84, 36, -4))
    bar_pitches = defaultdict(set)
    for n in analysis.raw_notes:
        bar_pitches[n["bar"]].add(n["pitch"])

    for row_p in ROWS:
        row_range = range(row_p, row_p + 4)
        label = f"  {NOTE_NAMES[row_p % 12]:2s}{row_p//12-1} "
        cells = []
        for bar in range(lo, lo + cols):
            ps = bar_pitches.get(bar, set())
            has = any(p in ps for p in row_range)
            diat = any(analysis.is_diatonic(p, bar) for p in row_range if p in ps)
            cells.append("▓" if has and diat else ("░" if has else "·"))
        print(label + "".join(cells))

    print()
    for param, arr in curves.items():
        cells = []
        for bar in range(lo, lo + cols):
            v = float(arr[min(bar, len(arr)-1)])
            cells.append("▁▃▅▇█"[int(v * 4.99)])
        print(f"  {param:14s} {''.join(cells)}")

    phrase_row = ["·"] * cols
    for pb in analysis.phrase_boundaries:
        col = pb - lo
        if 0 <= col < cols: phrase_row[col] = "│"
    print(f"  {'cadencias':14s} {''.join(phrase_row)}")
    print(f"  {'compás':14s} " + "".join(str((lo+i) % 10) for i in range(cols)))

    # Tonalidades por compás
    key_row = []
    for bar in range(lo, lo + cols):
        h = analysis.bar_harmony[min(bar, len(analysis.bar_harmony)-1)]
        key_row.append(NOTE_NAMES[h.tonic_pc][0])
    print(f"  {'tonalidad':14s} {''.join(key_row)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  [10] API PÚBLICA + PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def modulate_midi(
    input_path: str,
    params: list = None,
    curve: str = "linear",
    direction: str = "increasing",
    from_val: float = 0.0,
    to_val: float = 1.0,
    intensity: float = 1.0,
    scope: Optional[str] = None,
    voice: str = "all",
    voice_stagger: int = 0,
    use_groove_map: bool = False,
    smooth_beats: int = 0,
    respect_cadences: bool = False,
    steepness: float = 6.0,
    periods: float = 2.0,
    seed: int = 42,
    control_points: Optional[str] = None,
    output: Optional[str] = None,
    dry_run: bool = False,
    preview: bool = False,
    plot: bool = False,
    verbose: bool = False,
) -> str:
    """
    API pública para uso como módulo desde mutator u otros scripts.
    Devuelve la ruta del archivo de salida ('' si dry_run).
    """
    import mido

    params = params or ["density"]
    if "all" in params: params = list(ALL_PARAMS)
    for p in params:
        if p not in ALL_PARAMS: raise ValueError(f"Parámetro desconocido: '{p}'")

    print(f"\n╔═══ MODULATOR v{VERSION} ═══╗")
    print(f"  Entrada:  {input_path}")

    A = MusicAnalysis(input_path, verbose=verbose)
    n_bars = A.n_bars
    scope_start, scope_end = _parse_scope(scope, n_bars)

    print(f"  Compases: {n_bars} | BPM: {A.bpm} | Parámetros: {', '.join(params)}")
    print(f"  Curva: {curve} | Dirección: {direction} | Intensidad: {intensity}")
    if scope:      print(f"  Scope: compases {scope_start}–{scope_end}")
    if voice != "all": print(f"  Voz: {voice}" + (f" (stagger {voice_stagger})" if voice_stagger else ""))
    if smooth_beats:     print(f"  Smoothing: {smooth_beats} beats")
    if respect_cadences: print(f"  Respetando cadencias")
    if use_groove_map:   print(f"  Groove map: {'activo' if A.groove_map.trained else 'no disponible'}")

    np.random.seed(seed)
    curves = {}
    for param in params:
        curves[param] = generate_curve(
            n_bars=n_bars, curve=curve, from_val=from_val, to_val=to_val,
            direction=direction, steepness=steepness, periods=periods,
            seed=seed, control_points=control_points,
        )

    print("\n  Curvas generadas:")
    for param, arr in curves.items():
        lo = scope_start or 0; hi = scope_end or n_bars
        af = arr[lo:hi]
        print(f"  {param:15s} | min={af.min():.3f}  max={af.max():.3f}  mean={af.mean():.3f}  scope=[{lo}:{hi}]")

    if preview:
        show_preview(A, curves, scope_start, scope_end)

    if dry_run:
        print("\n  [dry-run] No se ha generado MIDI.")
        if plot: _plot_curves(curves, n_bars, scope_start, scope_end, A)
        return ""

    mid = mido.MidiFile(input_path)
    engine = ModulationEngine(
        mid=mid, A=A, verbose=verbose,
        smooth_beats=smooth_beats, respect_cadences=respect_cadences,
        use_groove_map=use_groove_map, voice=voice, voice_stagger=voice_stagger,
    )

    APPLY = {
        "density":      engine.apply_density,
        "tension":      engine.apply_tension,
        "dynamics":     engine.apply_dynamics,
        "register":     engine.apply_register,
        "articulation": engine.apply_articulation,
        "tempo":        engine.apply_tempo,
        "rhythm":       engine.apply_rhythm,
        "spread":       engine.apply_spread,
        "contour":      engine.apply_contour,
        "brightness":   engine.apply_brightness,
    }

    for param in params:
        s_eff, e_eff = get_voice_scope(scope_start, scope_end, voice, voice_stagger, n_bars) \
                       if voice_stagger > 0 else (scope_start, scope_end)
        print(f"  ▶ Aplicando {param}…")
        APPLY[param](curve=curves[param], intensity=intensity, s=s_eff, e=e_eff)

    out_path = output or _auto_output_name(input_path, params, curve)
    mid.save(out_path)
    print(f"\n  ✓ Guardado en: {out_path}")

    if plot: _plot_curves(curves, n_bars, scope_start, scope_end, A)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT v2
# ══════════════════════════════════════════════════════════════════════════════

def _plot_curves(curves, n_bars, scope_start, scope_end, A=None):
    try: import matplotlib.pyplot as plt
    except ImportError: print("[WARN] matplotlib no disponible"); return

    n = len(curves)
    if n == 0: return
    rows = n + (1 if A else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 2.5*rows), sharex=True)
    if rows == 1: axes = [axes]

    colors = ["#1D9E75","#D85A30","#534AB7","#BA7517",
              "#185FA5","#993556","#0F6E56","#3B6D11","#5F5E5A","#A32D2D"]
    x = np.arange(n_bars)
    lo = scope_start or 0; hi = scope_end or n_bars

    offset = 0
    if A:
        ax0 = axes[0]
        tl = np.array([A.bar_harmony[b].tension_level for b in range(n_bars)])
        ax0.fill_between(x, tl, alpha=0.2, color="#D85A30")
        ax0.plot(x, tl, color="#D85A30", lw=1, ls="--", label="tensión local")
        for pb in A.phrase_boundaries: ax0.axvline(pb, color="#aaa", ls=":", lw=0.7)
        ax0.set_ylabel("tensión\nlocal", fontsize=9); ax0.set_ylim(-0.05,1.05)
        ax0.grid(axis="y", ls="--", alpha=0.2)
        offset = 1

    for i, (param, arr) in enumerate(curves.items()):
        ax = axes[i + offset]
        ax.plot(x, arr, color=colors[i % len(colors)], lw=2, label=param)
        if lo > 0 or hi < n_bars: ax.axvspan(lo, hi, alpha=0.06, color=colors[i % len(colors)])
        if A:
            for pb in A.phrase_boundaries: ax.axvline(pb, color="#aaa", ls=":", lw=0.7)
        ax.set_ylabel(param, fontsize=9); ax.set_ylim(-0.05,1.05)
        ax.set_yticks([0,0.25,0.5,0.75,1.0])
        ax.grid(axis="y", ls="--", alpha=0.2)

    axes[-1].set_xlabel("Compás")
    fig.suptitle("Modulator v2 — curvas + estructura formal", fontsize=12)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_scope(scope_str, n_bars):
    if not scope_str: return None, None
    try:
        parts = scope_str.split(":")
        s = int(parts[0]) if parts[0] else None
        e = int(parts[1]) if len(parts) > 1 and parts[1] else None
        if s is not None and s < 0: s = max(0, n_bars + s)
        if e is not None and e < 0: e = max(0, n_bars + e)
        return s, e
    except Exception:
        raise ValueError(f"--scope mal formado: '{scope_str}'")

def _auto_output_name(input_path, params, curve):
    stem = Path(input_path).stem
    p_str = "_".join(params[:3]) + ("_etc" if len(params) > 3 else "")
    return str(Path(input_path).parent / f"{stem}_mod_{p_str}_{curve}.mid")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(prog="modulator.py",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("input")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--param",  metavar="P")
    grp.add_argument("--params", metavar="P", nargs="+")
    p.add_argument("--curve",     default="linear", choices=ALL_CURVES)
    p.add_argument("--direction", default="increasing", choices=["increasing","decreasing","none"])
    p.add_argument("--from",  dest="from_val", type=float, default=0.0)
    p.add_argument("--to",    dest="to_val",   type=float, default=1.0)
    p.add_argument("--intensity",       type=float, default=1.0)
    p.add_argument("--steepness",       type=float, default=6.0)
    p.add_argument("--periods",         type=float, default=2.0)
    p.add_argument("--control-points",  metavar="STR")
    p.add_argument("--scope",           metavar="S:E")
    p.add_argument("--voice",           default="all", choices=["all","melody","bass","harmony"])
    p.add_argument("--voice-stagger",   type=int, default=0, metavar="N")
    p.add_argument("--use-groove-map",  action="store_true")
    p.add_argument("--smooth-boundary", type=int, default=0, metavar="N")
    p.add_argument("--respect-cadences",action="store_true")
    p.add_argument("--output",          metavar="FILE")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--preview",         action="store_true")
    p.add_argument("--plot",            action="store_true")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--list-params",     action="store_true")
    p.add_argument("--list-curves",     action="store_true")
    p.add_argument("--version",         action="version", version=f"modulator {VERSION}")
    return p


def print_params():
    print("\n  Parámetros disponibles:\n")
    for p, m in PARAM_META.items():
        print(f"  {p:15s} — {m['desc']}")
        print(f"  {'':15s}   ▲ {m['increasing']}")
        print(f"  {'':15s}   ▼ {m['decreasing']}\n")


def print_curves():
    descs = {
        "linear":"rampa simple y = t",
        "exponential":"y = t^steepness (arranque lento)",
        "logarithmic":"y = √t (arranque rápido)",
        "sigmoid":"suave en extremos, --steepness",
        "arc":"y = sin(πt) — sube y baja",
        "arc_down":"1 - sin(πt) — baja y sube",
        "s_curve":"3t²-2t³ — cubic ease in-out",
        "pulse":"N pulsos — --periods",
        "plateau":"rampa rápida, plateau, caída",
        "wave":"onda completa N períodos — --periods",
        "random_walk":"paseo aleatorio suavizado — --seed",
        "custom":"puntos de control — --control-points 'BAR:VAL,…'",
    }
    print("\n  Curvas disponibles:\n")
    for c, d in descs.items(): print(f"  {c:15s} — {d}")
    print()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_params: print_params(); return
    if args.list_curves: print_curves(); return

    if not Path(args.input).exists():
        print(f"[ERROR] No se encuentra: {args.input}", file=sys.stderr); sys.exit(1)

    try:
        import mido
    except ImportError:
        print("[ERROR] pip install mido", file=sys.stderr); sys.exit(1)

    if args.param == "all": params = list(ALL_PARAMS)
    elif args.param:        params = [args.param]
    else:                   params = args.params or ["density"]

    try:
        modulate_midi(
            input_path=args.input, params=params,
            curve=args.curve, direction=args.direction,
            from_val=args.from_val, to_val=args.to_val,
            intensity=args.intensity, scope=args.scope,
            voice=args.voice, voice_stagger=args.voice_stagger,
            use_groove_map=args.use_groove_map,
            smooth_beats=args.smooth_boundary,
            respect_cadences=args.respect_cadences,
            steepness=args.steepness, periods=args.periods,
            seed=args.seed, control_points=args.control_points,
            output=args.output, dry_run=args.dry_run,
            preview=args.preview, plot=args.plot, verbose=args.verbose,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
