#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCHENKER COMPOSER  v1.0                                    ║
║   Recomposición neuronal por niveles de reducción schenkeriana               ║
║                                                                              ║
║  Entrena una red que aprende a "elaborar" un nivel schenkeriano n hacia el   ║
║  nivel n+1 (más cercano a la superficie). El corpus de entrenamiento se      ║
║  construye reduciendo automáticamente MIDIs reales con un analizador        ║
║  schenkeriano interno (inspirado en schenkerian_reducer.py / GTTM de        ║
║  Lerdahl & Jackendoff, pero reimplementado aquí de forma autocontenida y    ║
║  simplificada — sin depender de ningún fichero externo del ecosistema).     ║
║                                                                              ║
║  CONVENCIÓN DE NIVELES (importante, es la INVERSA de un reductor clásico):  ║
║    nivel 0        = Ursatz / estructura más reducida (fondo)                ║
║    nivel D_max     = superficie original (primer plano, el MIDI tal cual)   ║
║  Es decir "nivel n → n+1" = ELABORACIÓN (añadir ornamentación), no          ║
║  reducción. El modelo aprende ese paso hacia adelante.                      ║
║                                                                              ║
║  HONESTIDAD EPISTEMOLÓGICA: igual que en schenkerian_reducer.py, el         ║
║  análisis schenkeriano automático NO es determinista ni "la verdad" — es   ║
║  una heurística de puntuación (peso métrico + armónico + duración) que      ║
║  aproxima el proceso de reducción de forma tratable computacionalmente.     ║
║  La red neuronal aprende el ESTILO de elaboración presente en el corpus     ║
║  de entrenamiento, no reglas schenkerianas universales.                     ║
║                                                                              ║
║  PIPELINE:                                                                   ║
║    [1] synth-corpus  — (opcional) genera MIDIs sintéticos para pruebas      ║
║    [2] prepare        — corpus de MIDIs → niveles de reducción → pares      ║
║                         (nivel n, nivel n+1) tokenizados → dataset .npz     ║
║    [3] train          — entrena el Transformer encoder-decoder             ║
║    [4] generate       — MIDI de un nivel arbitrario → nivel n+1 generado   ║
║    [5] inspect        — muestra la reducción schenkeriana de un MIDI       ║
║                         (equivalente ligero de schenkerian_reducer.py,      ║
║                         útil para depurar el paso [2])                     ║
║                                                                              ║
║  USO:                                                                        ║
║    python schenker_composer.py synth-corpus corpus/ --n 40                  ║
║    python schenker_composer.py prepare corpus/ --out dataset.npz            ║
║    python schenker_composer.py train dataset.npz --model-dir modelo/        ║
║    python schenker_composer.py generate entrada.mid --model-dir modelo/ \\   ║
║           --output salida.mid --level 0                                     ║
║    python schenker_composer.py inspect obra.mid                             ║
║                                                                              ║
║  COMO MÓDULO:                                                               ║
║    from schenker_composer import reduce_piece_depths, prepare_dataset       ║
║                                                                              ║
║  DEPENDENCIAS: numpy siempre; torch SOLO para train/generate (import        ║
║  perezoso, no hace falta para prepare/inspect/synth-corpus). E/S MIDI,      ║
║  teoría armónica y tokenización están reimplementadas aquí sin importar     ║
║  schenkerian_reducer.py ni ningún otro fichero del ecosistema — el         ║
║  programa es autocontenido en un solo fichero.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  [0] E/S MIDI AUTOCONTENIDA (SMF 0/1, solo stdlib)
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


@dataclass
class MidiEvent:
    abs: int
    kind: str            # note_on | note_off | meta | channel | sysex
    channel: int = 0
    data: bytes = b""
    pitch: int = 0
    vel: int = 0
    meta_type: int = -1


@dataclass
class MidiTrackData:
    name: str = ""
    events: List[MidiEvent] = field(default_factory=list)


@dataclass
class MidiData:
    fmt: int = 1
    tpb: int = 480
    tracks: List[MidiTrackData] = field(default_factory=list)
    tempo_map: List[Tuple[int, int]] = field(default_factory=list)
    timesig_map: List[Tuple[int, int, int]] = field(default_factory=list)


def _read_vlq(buf: bytes, i: int) -> Tuple[int, int]:
    val = 0
    while True:
        b = buf[i]; i += 1
        val = (val << 7) | (b & 0x7F)
        if not b & 0x80:
            return val, i


def _write_vlq(val: int) -> bytes:
    out = [val & 0x7F]
    val >>= 7
    while val:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    return bytes(reversed(out))


def read_midi(path: str) -> MidiData:
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{path}: no es un archivo MIDI (falta MThd)")
    fmt = int.from_bytes(raw[8:10], "big")
    n_tracks = int.from_bytes(raw[10:12], "big")
    division = int.from_bytes(raw[12:14], "big")
    if division & 0x8000:
        raise ValueError(f"{path}: división SMPTE no soportada")
    mid = MidiData(fmt=fmt, tpb=division)

    i = 14
    for _ in range(n_tracks):
        if raw[i:i + 4] != b"MTrk":
            raise ValueError(f"{path}: chunk de pista corrupto en offset {i}")
        length = int.from_bytes(raw[i + 4:i + 8], "big")
        chunk = raw[i + 8:i + 8 + length]
        i += 8 + length
        trk = MidiTrackData()
        t = 0
        j = 0
        status = 0
        while j < len(chunk):
            delta, j = _read_vlq(chunk, j)
            t += delta
            b0 = chunk[j]
            if b0 & 0x80:
                status = b0
                j += 1
            if status == 0xFF:
                mtype = chunk[j]; j += 1
                mlen, j = _read_vlq(chunk, j)
                mdata = chunk[j:j + mlen]; j += mlen
                if mtype == 0x03 and not trk.name:
                    trk.name = mdata.decode("latin-1", errors="replace").strip()
                elif mtype == 0x51 and mlen == 3:
                    mid.tempo_map.append((t, int.from_bytes(mdata, "big")))
                elif mtype == 0x58 and mlen >= 2:
                    mid.timesig_map.append((t, mdata[0], 2 ** mdata[1]))
            elif status in (0xF0, 0xF7):
                slen, j = _read_vlq(chunk, j)
                j += slen
            else:
                hi, ch = status & 0xF0, status & 0x0F
                if hi in (0xC0, 0xD0):
                    j += 1
                else:
                    d1, d2 = chunk[j], chunk[j + 1]; j += 2
                    if hi == 0x90 and d2 > 0:
                        trk.events.append(MidiEvent(abs=t, kind="note_on", channel=ch,
                                                    pitch=d1, vel=d2))
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        trk.events.append(MidiEvent(abs=t, kind="note_off", channel=ch,
                                                    pitch=d1, vel=d2))
        mid.tracks.append(trk)

    if not mid.tempo_map:
        mid.tempo_map = [(0, 500000)]
    mid.tempo_map.sort()
    if not mid.timesig_map:
        mid.timesig_map = [(0, 4, 4)]
    mid.timesig_map.sort()
    return mid


def write_midi(mid: MidiData, path: str):
    chunks = []
    for trk in mid.tracks:
        evs = sorted(trk.events, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))
        body = bytearray()
        last = 0
        for ev in evs:
            body += _write_vlq(max(0, ev.abs - last))
            last = ev.abs
            if ev.kind == "note_on":
                body += bytes([0x90 | (ev.channel & 0x0F), ev.pitch & 0x7F, ev.vel & 0x7F])
            elif ev.kind == "note_off":
                body += bytes([0x80 | (ev.channel & 0x0F), ev.pitch & 0x7F, ev.vel & 0x7F])
            elif ev.kind == "meta":
                body += ev.data
            else:
                body += ev.data
        body += _write_vlq(0) + bytes([0xFF, 0x2F, 0x00])
        chunks.append(b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))
    header = (b"MThd" + (6).to_bytes(4, "big") + mid.fmt.to_bytes(2, "big")
              + len(mid.tracks).to_bytes(2, "big") + mid.tpb.to_bytes(2, "big"))
    Path(path).write_bytes(header + b"".join(chunks))


def make_tempo_track(tpb: int, bpm: float = 100.0, num: int = 4, den: int = 4) -> MidiTrackData:
    usec = int(60_000_000 / bpm)
    trk = MidiTrackData(name="tempo")
    tempo_data = bytes([0xFF, 0x51, 0x03]) + usec.to_bytes(3, "big")
    denom_pow = int(math.log2(den))
    ts_data = bytes([0xFF, 0x58, 0x04, num, denom_pow, 24, 8])
    trk.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x51, data=tempo_data))
    trk.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x58, data=ts_data))
    return trk


class TimeContext:
    """Conversión ticks <-> beats/compases."""

    def __init__(self, mid: MidiData):
        self.tpb = mid.tpb
        self.timesig_map = mid.timesig_map
        self._bars = []
        bar, prev_tick = 1.0, 0
        prev_tpc = self.tpb * 4 * self.timesig_map[0][1] // self.timesig_map[0][2]
        for tick, num, den in self.timesig_map:
            bar += (tick - prev_tick) / prev_tpc
            tpc = max(1, self.tpb * 4 * num // den)
            self._bars.append((tick, bar, tpc, num))
            prev_tick, prev_tpc = tick, tpc
        if not self._bars or self._bars[0][0] > 0:
            self._bars.insert(0, (0, 1.0, prev_tpc, self.timesig_map[0][1]))

    def beat(self, tick: int) -> float:
        return tick / self.tpb

    def tick(self, beat: float) -> int:
        return int(round(beat * self.tpb))

    def numerator_at(self, tick: int) -> int:
        seg = self._bars[0]
        for s in self._bars:
            if s[0] <= tick:
                seg = s
            else:
                break
        return seg[3]


@dataclass
class Note:
    pitch: int
    start: int      # ticks
    end: int        # ticks
    vel: int = 90
    channel: int = 0

    def start_beat(self, tc: TimeContext) -> float:
        return tc.beat(self.start)

    def duration_beats(self, tc: TimeContext) -> float:
        return (self.end - self.start) / tc.tpb


def extract_notes(trk: MidiTrackData) -> List[Note]:
    out: List[Note] = []
    stack: Dict[Tuple[int, int], List[int]] = {}
    for ev in trk.events:
        if ev.kind == "note_on":
            stack.setdefault((ev.channel, ev.pitch), []).append(ev.abs)
        elif ev.kind == "note_off":
            key = (ev.channel, ev.pitch)
            if stack.get(key):
                start = stack[key].pop(0)
                if ev.abs > start:
                    out.append(Note(pitch=ev.pitch, start=start, end=ev.abs, channel=ev.channel))
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def load_notes(path: str) -> Tuple[MidiData, TimeContext, List[Note]]:
    mid = read_midi(path)
    tc = TimeContext(mid)
    notes: List[Note] = []
    for trk in mid.tracks:
        notes.extend(extract_notes(trk))
    if not notes:
        raise ValueError(f"{path}: el MIDI no contiene notas")
    notes.sort(key=lambda n: (n.start, n.pitch))
    return mid, tc, notes


# ══════════════════════════════════════════════════════════════════════════════
#  [1] TEORÍA ARMÓNICA (tonalidad + acordes por ventana, reimplementación
#      compacta del mismo tipo de idea que schenkerian_reducer.py)
# ══════════════════════════════════════════════════════════════════════════════

_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                    5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                    4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]   # natural minor, aproximación razonable
CHORD_TEMPLATES = {"": {0, 4, 7}, "m": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8}}
_MAJ_DEGREES = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
_MIN_DEGREES = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}


def detect_key(pc_hist: np.ndarray) -> Tuple[int, str]:
    best = (0, "maj", -1e9)
    total = pc_hist.sum()
    for tonic in range(12):
        maj_prof = np.roll(_KS_MAJ, tonic)
        min_prof = np.roll(_KS_MIN, tonic)
        cm = float(np.corrcoef(pc_hist, maj_prof)[0, 1]) if total else 0.0
        cn = float(np.corrcoef(pc_hist, min_prof)[0, 1]) if total else 0.0
        if np.isnan(cm):
            cm = -1e9
        if np.isnan(cn):
            cn = -1e9
        if cm > best[2]:
            best = (tonic, "maj", cm)
        if cn > best[2]:
            best = (tonic, "min", cn)
    return best[0], best[1]


def _parse_key(key: str) -> Tuple[int, str]:
    key = key.strip()
    mode = "min" if key.lower().endswith("m") and not key.lower().endswith("maj") else "maj"
    name = key[:-1] if mode == "min" else (key[:-3] if key.lower().endswith("maj") else key)
    name = name.strip()
    letter = name[0].upper()
    if letter not in _PC:
        raise ValueError(f"tonalidad inválida: {key}")
    pc = _PC[letter]
    for ch in name[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
    return pc % 12, mode


def detect_chord(pcs_weight: Dict[int, float], bass_pc: Optional[int]) -> Tuple[Optional[int], str]:
    present = {pc for pc, w in pcs_weight.items() if w > 0}
    if not present:
        return None, ""
    best = (None, "", -1e9)
    for root in range(12):
        rel = {(pc - root) % 12 for pc in present}
        for suffix, templ in CHORD_TEMPLATES.items():
            inter = len(rel & templ)
            extra = len(rel - templ)
            missing = len(templ - rel)
            score = 2.0 * inter - 1.0 * extra - 0.7 * missing
            if bass_pc is not None and (bass_pc - root) % 12 == 0:
                score += 0.6
            if score > best[2]:
                best = (root, suffix, score)
    return best[0], best[1]


def roman_of(root_pc: int, tonic_pc: int, mode: str) -> str:
    rel = (root_pc - tonic_pc) % 12
    degs = _MAJ_DEGREES if mode == "maj" else _MIN_DEGREES
    return degs.get(rel, f"?{rel}")


@dataclass
class ChordSpan:
    start_beat: float
    end_beat: float
    root_pc: Optional[int]
    suffix: str
    roman: str


def analyze_harmony(notes: List[Note], tc: TimeContext, key: Optional[str] = None,
                     window_beats: float = 1.0) -> Tuple[int, str, List[ChordSpan]]:
    if key:
        tonic_pc, mode = _parse_key(key)
    else:
        hist = np.zeros(12)
        for n in notes:
            hist[n.pitch % 12] += n.duration_beats(tc)
        tonic_pc, mode = detect_key(hist)

    last_beat = max(n.start_beat(tc) + n.duration_beats(tc) for n in notes)
    n_windows = max(1, int(math.ceil(last_beat / window_beats)))
    spans: List[ChordSpan] = []
    for w in range(n_windows):
        w0, w1 = w * window_beats, (w + 1) * window_beats
        weights: Dict[int, float] = {}
        bass_candidates = []
        for n in notes:
            s, e = n.start_beat(tc), n.start_beat(tc) + n.duration_beats(tc)
            overlap = min(e, w1) - max(s, w0)
            if overlap > 0:
                weights[n.pitch % 12] = weights.get(n.pitch % 12, 0.0) + overlap
                bass_candidates.append(n.pitch)
        bass_pc = (min(bass_candidates) % 12) if bass_candidates else None
        root, suffix = detect_chord(weights, bass_pc)
        roman = roman_of(root, tonic_pc, mode) if root is not None else "?"
        spans.append(ChordSpan(w0, w1, root, suffix, roman))
    return tonic_pc, mode, spans


def chord_at(spans: List[ChordSpan], beat: float) -> Optional[ChordSpan]:
    for sp in spans:
        if sp.start_beat <= beat < sp.end_beat:
            return sp
    return spans[-1] if spans else None


def chord_tone_pcs(chord: Optional[ChordSpan]) -> set:
    if chord is None or chord.root_pc is None:
        return set()
    return {(chord.root_pc + iv) % 12 for iv in CHORD_TEMPLATES.get(chord.suffix, {0, 4, 7})}


def scale_degree(pitch: int, tonic_pc: int, mode: str) -> Optional[int]:
    scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
    rel = (pitch - tonic_pc) % 12
    if rel in scale:
        return scale.index(rel) + 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  [2] SKYLINE (voz superior / voz inferior) Y PESO MÉTRICO
# ══════════════════════════════════════════════════════════════════════════════

def build_skyline(notes: List[Note]) -> Tuple[List[Note], List[Note]]:
    """Separa la textura en línea superior (melodía) e inferior (bajo) por
    'skyline': en cada instante, la nota más aguda que suena va a melodía y
    la más grave a bajo. Si solo hay una voz, ambas coinciden."""
    if not notes:
        return [], []
    times = sorted(set(n.start for n in notes) | set(n.end for n in notes))
    mel_segs: List[Tuple[int, int, int]] = []
    bass_segs: List[Tuple[int, int, int]] = []
    for t0, t1 in zip(times[:-1], times[1:]):
        active = [n for n in notes if n.start <= t0 < n.end]
        if not active:
            continue
        top = max(active, key=lambda n: n.pitch)
        low = min(active, key=lambda n: n.pitch)
        mel_segs.append((t0, t1, top.pitch))
        bass_segs.append((t0, t1, low.pitch))

    def _merge(segs: List[Tuple[int, int, int]], channel: int) -> List[Note]:
        out: List[Note] = []
        for t0, t1, p in segs:
            if out and out[-1].pitch == p and out[-1].end == t0:
                out[-1] = Note(pitch=p, start=out[-1].start, end=t1, channel=channel)
            else:
                out.append(Note(pitch=p, start=t0, end=t1, channel=channel))
        return out

    mel = _merge(mel_segs, 0)
    bass = _merge(bass_segs, 1)
    return mel, bass


def metric_weights(num: int, subdiv: int) -> np.ndarray:
    """Peso métrico por posición dentro del compás (subdiv subdivisiones por
    tiempo). Tiempo 1 = 1.0, resto decrece por jerarquía métrica simple."""
    n = num * subdiv
    w = np.zeros(n)
    for i in range(n):
        beat_idx = i // subdiv
        sub_idx = i % subdiv
        if beat_idx == 0 and sub_idx == 0:
            w[i] = 1.0
        elif sub_idx == 0 and beat_idx % 2 == 0:
            w[i] = 0.75
        elif sub_idx == 0:
            w[i] = 0.55
        elif sub_idx == subdiv // 2:
            w[i] = 0.35
        else:
            w[i] = 0.15
    return w


def note_metric_weight(note: Note, tc: TimeContext, num: int, subdiv: int, W: np.ndarray) -> float:
    beat = note.start_beat(tc)
    pos_in_bar = beat % num
    idx = int(round(pos_in_bar * subdiv)) % len(W)
    return float(W[idx])


# ══════════════════════════════════════════════════════════════════════════════
#  [3] CLASIFICACIÓN DE NOTAS Y PESO ESTRUCTURAL (heurística simplificada,
#      NO pretende ser un análisis schenkeriano "correcto" — ver cabecera)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "CT": 1.0, "PT": 0.4, "NT": 0.35, "APP": 0.3,
    "SUS": 0.45, "ANT": 0.3, "ESC": 0.25, "UNCLASSIFIED": 0.5,
}


def classify_note(note: Note, prev: Optional[Note], nxt: Optional[Note],
                   chord_now: Optional[ChordSpan], mweight: float) -> str:
    pcs_now = chord_tone_pcs(chord_now)
    if (note.pitch % 12) in pcs_now:
        return "CT"
    if prev is not None and nxt is not None:
        d1 = note.pitch - prev.pitch
        d2 = nxt.pitch - note.pitch
        if abs(d1) in (1, 2) and abs(d2) in (1, 2) and (d1 > 0) == (d2 > 0):
            return "PT"
        if abs(d1) in (1, 2) and abs(d2) in (1, 2) and (d1 > 0) != (d2 > 0):
            return "NT"
    if prev is not None and abs(note.pitch - prev.pitch) in (1, 2) and mweight >= 0.5:
        return "APP"
    if nxt is not None and abs(nxt.pitch - note.pitch) in (1, 2) and mweight < 0.4:
        return "ESC"
    if prev is not None and note.pitch == prev.pitch:
        return "SUS"
    return "UNCLASSIFIED"


def structural_weight(function: str, metric_weight: float, duration_beats: float,
                       weights: Optional[Dict[str, float]] = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    base = weights.get(function, 0.4)
    duration_factor = min(max(duration_beats, 0.0) / 1.0, 1.5) if duration_beats > 0 else 1.0
    duration_factor = max(duration_factor, 0.5)
    return base * (0.5 + 0.5 * metric_weight) * duration_factor


@dataclass
class ScoredNote:
    note: Note
    function: str
    weight: float
    beat: float


def score_voice(voice_notes: List[Note], tc: TimeContext, spans: List[ChordSpan],
                 num: int, subdiv: int, W: np.ndarray) -> List[ScoredNote]:
    scored: List[ScoredNote] = []
    for i, n in enumerate(voice_notes):
        beat = n.start_beat(tc)
        chord_now = chord_at(spans, beat)
        mweight = note_metric_weight(n, tc, num, subdiv, W)
        func = classify_note(n, voice_notes[i - 1] if i > 0 else None,
                              voice_notes[i + 1] if i < len(voice_notes) - 1 else None,
                              chord_now, mweight)
        w = structural_weight(func, mweight, n.duration_beats(tc))
        scored.append(ScoredNote(note=n, function=func, weight=w, beat=beat))
    return scored


# ══════════════════════════════════════════════════════════════════════════════
#  [4] ESCALERA DE REDUCCIÓN POR FUSIÓN (surface → Ursatz)
#
#  En cada paso se elimina la nota de menor peso estructural fusionándola
#  (extendiendo su duración) dentro de la nota vecina de mayor peso. Esto
#  garantiza por construcción que cada nivel más reducido "tesela" el mismo
#  intervalo temporal que el nivel anterior — propiedad que se explota luego
#  para alinear exactamente nivel n con nivel n+1 al construir el dataset.
# ══════════════════════════════════════════════════════════════════════════════

def build_reduction_ladder(scored: List[ScoredNote], tpb: int,
                            min_notes: int = 2, max_depth: int = 6) -> List[List[ScoredNote]]:
    """Devuelve niveles en orden SUPERFICIE → REDUCIDO (ladder[0] = superficie).
    El llamador debe invertir la lista si quiere el orden de profundidad
    schenkeriana (0 = Ursatz) usado en el resto del programa."""
    ladder = [list(scored)]
    cur = list(scored)
    while len(cur) > min_notes and len(ladder) - 1 < max_depth:
        idx_min = min(range(len(cur)), key=lambda i: cur[i].weight)
        if len(cur) == 1:
            break
        if idx_min == 0:
            target = 1
        elif idx_min == len(cur) - 1:
            target = len(cur) - 2
        else:
            target = idx_min - 1 if cur[idx_min - 1].weight >= cur[idx_min + 1].weight else idx_min + 1
        lo, hi = sorted([idx_min, target])
        survivor = cur[target]
        merged_note = Note(pitch=survivor.note.pitch,
                            start=min(cur[lo].note.start, cur[hi].note.start),
                            end=max(cur[lo].note.end, cur[hi].note.end),
                            channel=survivor.note.channel)
        new_dur_beats = (merged_note.end - merged_note.start) / tpb
        new_weight = structural_weight(survivor.function, 1.0, new_dur_beats) \
            if new_dur_beats > 0 else survivor.weight
        new_scored = ScoredNote(note=merged_note, function=survivor.function,
                                 weight=max(new_weight, survivor.weight), beat=survivor.beat)
        cur = cur[:lo] + [new_scored] + cur[hi + 1:]
        ladder.append(cur)
    return ladder


@dataclass
class PieceReduction:
    path: str
    tpb: int
    tc: TimeContext
    tonic_pc: int
    mode: str
    spans: List[ChordSpan]
    mel_depths: List[List[ScoredNote]]     # [0]=Ursatz ... [-1]=superficie
    bass_depths: List[List[ScoredNote]]


def reduce_piece_depths(path: str, key: Optional[str] = None, window_beats: Optional[float] = None,
                         max_depth: int = 500, subdiv: int = 4) -> PieceReduction:
    mid, tc, all_notes = load_notes(path)
    num = mid.timesig_map[0][1]
    wb = window_beats if window_beats else float(num)
    tonic_pc, mode, spans = analyze_harmony(all_notes, tc, key=key, window_beats=wb)
    mel_notes, bass_notes = build_skyline(all_notes)
    W = metric_weights(num, subdiv)

    scored_mel = score_voice(mel_notes, tc, spans, num, subdiv, W)
    scored_bass = score_voice(bass_notes, tc, spans, num, subdiv, W)

    # min_notes escala con la duración de la pieza: no reducir por debajo de
    # ~1 nota por compás en promedio, para no colapsar todo a una sola nota.
    last_beat = max((n.note.start_beat(tc) + n.note.duration_beats(tc) for n in scored_mel),
                    default=0.0)
    n_bars = max(1, int(math.ceil(last_beat / num)))
    min_notes_mel = max(2, min(len(scored_mel), n_bars))
    min_notes_bass = max(2, min(len(scored_bass), n_bars))

    ladder_mel = build_reduction_ladder(scored_mel, mid.tpb, min_notes=min_notes_mel, max_depth=max_depth)
    ladder_bass = build_reduction_ladder(scored_bass, mid.tpb, min_notes=min_notes_bass, max_depth=max_depth)

    mel_depths = list(reversed(ladder_mel))
    bass_depths = list(reversed(ladder_bass))
    return PieceReduction(path=path, tpb=mid.tpb, tc=tc, tonic_pc=tonic_pc, mode=mode,
                           spans=spans, mel_depths=mel_depths, bass_depths=bass_depths)


# ══════════════════════════════════════════════════════════════════════════════
#  [5] SELECCIÓN DE "NIVELES PRESENTADOS"
#
#  La escalera de fusión (build_reduction_ladder) da un paso por nota (muy
#  fino). Para el entrenamiento conviene comparar snapshots que difieran en
#  VARIAS notas (una elaboración real: vecinas, notas de paso, arpegios...),
#  no solo una. Elegimos N_LEVELS puntos repartidos a lo largo de la
#  escalera (siempre incluyendo el extremo más reducido y la superficie).
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_LEVELS = 5
LEVEL_BUCKETS = 10   # granularidad del token LEVEL_k (independiente de la pieza)


def select_presented_depths(n_ladder: int, n_levels: int = DEFAULT_N_LEVELS) -> List[int]:
    n_levels = max(2, min(n_levels, n_ladder))
    if n_ladder <= 1:
        return [0] * n_levels
    idxs = sorted(set(round(i * (n_ladder - 1) / (n_levels - 1)) for i in range(n_levels)))
    while len(idxs) < 2:
        idxs.append(n_ladder - 1)
    return idxs


def level_bucket(i: int, n_presented: int) -> int:
    if n_presented <= 1:
        return 0
    return int(round(LEVEL_BUCKETS * i / (n_presented - 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  [6] VOCABULARIO Y TOKENIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

PAD, BOS, EOS = 0, 1, 2
_N_SPECIAL = 3
_N_LEVEL_TOK = LEVEL_BUCKETS + 1                       # LEVEL_0..LEVEL_10
_VOICES = ["mel", "bass"]
_N_VOICE_TOK = len(_VOICES)
_KEYS = [f"{pc_name(pc)}_{m}" for pc in range(12) for m in ("maj", "min")]
_N_KEY_TOK = len(_KEYS)                                 # 24
_ROMANS = sorted(set(list(_MAJ_DEGREES.values()) + list(_MIN_DEGREES.values()) + ["?"]))
_N_CHORD_TOK = len(_ROMANS)
_MAX_DUR_BIN = 64                                        # en unidades de semicorchea (subdiv=4)
_N_DUR_TOK = _MAX_DUR_BIN                                # DUR_1..DUR_64
_MIN_PITCH, _MAX_PITCH = 0, 127
_N_PITCH_TOK = (_MAX_PITCH - _MIN_PITCH + 1) + 1         # + REST

_OFF_LEVEL = _N_SPECIAL
_OFF_VOICE = _OFF_LEVEL + _N_LEVEL_TOK
_OFF_KEY = _OFF_VOICE + _N_VOICE_TOK
_OFF_CHORD = _OFF_KEY + _N_KEY_TOK
_OFF_DUR = _OFF_CHORD + _N_CHORD_TOK
_OFF_PITCH = _OFF_DUR + _N_DUR_TOK
VOCAB_SIZE = _OFF_PITCH + _N_PITCH_TOK
REST_ID = _OFF_PITCH + (_MAX_PITCH - _MIN_PITCH + 1)


def id_level(bucket: int) -> int:
    return _OFF_LEVEL + max(0, min(bucket, _N_LEVEL_TOK - 1))


def id_voice(voice: str) -> int:
    return _OFF_VOICE + (_VOICES.index(voice) if voice in _VOICES else 0)


def id_key(tonic_pc: int, mode: str) -> int:
    name = f"{pc_name(tonic_pc)}_{'maj' if mode == 'maj' else 'min'}"
    return _OFF_KEY + _KEYS.index(name)


def id_chord(roman: str) -> int:
    if roman not in _ROMANS:
        roman = "?"
    return _OFF_CHORD + _ROMANS.index(roman)


def dur_to_bin(duration_beats: float, subdiv: int = 4) -> int:
    units = max(1, int(round(duration_beats * subdiv)))
    return max(1, min(units, _MAX_DUR_BIN))


def bin_to_beats(bin_val: int, subdiv: int = 4) -> float:
    return max(1, bin_val) / subdiv


def id_dur(duration_beats: float, subdiv: int = 4) -> int:
    return _OFF_DUR + (dur_to_bin(duration_beats, subdiv) - 1)


def dur_of_id(tid: int, subdiv: int = 4) -> float:
    return bin_to_beats((tid - _OFF_DUR) + 1, subdiv)


def id_pitch(pitch: Optional[int]) -> int:
    if pitch is None:
        return REST_ID
    p = max(_MIN_PITCH, min(_MAX_PITCH, int(pitch)))
    return _OFF_PITCH + (p - _MIN_PITCH)


def pitch_of_id(tid: int) -> Optional[int]:
    if tid == REST_ID:
        return None
    return _MIN_PITCH + (tid - _OFF_PITCH)


def is_pitch_id(tid: int) -> bool:
    return _OFF_PITCH <= tid < _OFF_PITCH + _N_PITCH_TOK


def is_dur_id(tid: int) -> bool:
    return _OFF_DUR <= tid < _OFF_DUR + _N_DUR_TOK


def tokenize_source_chunk(chunk: List[ScoredNote], level_bkt: int, voice: str, tonic_pc: int, mode: str,
                           tc: TimeContext, spans: List[ChordSpan]) -> List[int]:
    toks = [BOS, id_level(level_bkt), id_voice(voice), id_key(tonic_pc, mode)]
    for sn in chunk:
        ch = chord_at(spans, sn.note.start_beat(tc))
        roman = ch.roman if ch else "?"
        toks.append(id_chord(roman))
        toks.append(id_pitch(sn.note.pitch))
        toks.append(id_dur(sn.note.duration_beats(tc)))
    toks.append(EOS)
    return toks


def tokenize_target_notes(notes: List[Note], tc: TimeContext) -> List[int]:
    toks = [BOS]
    for n in notes:
        toks.append(id_pitch(n.pitch))
        toks.append(id_dur(n.duration_beats(tc)))
    toks.append(EOS)
    return toks


def detokenize_notes(token_ids: List[int], start_tick: int, tpb: int, subdiv: int = 4,
                      channel: int = 0) -> Tuple[List[Note], int]:
    """Convierte una secuencia de tokens PITCH,DUR,PITCH,DUR... (ignora
    especiales/condicionamiento si aparecen) en notas MIDI consecutivas sin
    huecos, empezando en start_tick. Los REST avanzan el cursor sin nota.
    Devuelve (notas, cursor_final) para poder reescalar la duración total."""
    notes: List[Note] = []
    cursor = start_tick
    pending_pitch = "await"
    cur_pitch = None
    for tid in token_ids:
        if tid in (PAD, BOS, EOS):
            continue
        if pending_pitch == "await":
            if is_pitch_id(tid):
                cur_pitch = pitch_of_id(tid)
                pending_pitch = "await_dur"
            # ignora tokens de condicionamiento (LEVEL/KEY/CHORD) intercalados
            continue
        elif pending_pitch == "await_dur":
            if is_dur_id(tid):
                dur_beats = dur_of_id(tid, subdiv)
                dur_ticks = max(1, int(round(dur_beats * tpb)))
                if cur_pitch is not None:
                    notes.append(Note(pitch=cur_pitch, start=cursor, end=cursor + dur_ticks,
                                       channel=channel))
                cursor += dur_ticks
                pending_pitch = "await"
            elif is_pitch_id(tid):
                # duración perdida/errónea: asume corchea por defecto y sigue
                dur_ticks = max(1, int(round(0.5 * tpb)))
                if cur_pitch is not None:
                    notes.append(Note(pitch=cur_pitch, start=cursor, end=cursor + dur_ticks,
                                       channel=channel))
                cursor += dur_ticks
                cur_pitch = pitch_of_id(tid)
                pending_pitch = "await_dur"
    return notes, cursor


# ══════════════════════════════════════════════════════════════════════════════
#  [7] CONSTRUCCIÓN DE PARES DE ENTRENAMIENTO (nivel n → nivel n+1)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_pairs(pr: PieceReduction, voice: str, n_levels: int = DEFAULT_N_LEVELS,
                          max_src_notes: int = 4, max_tgt_tokens: int = 64
                          ) -> List[Tuple[List[int], List[int]]]:
    depths = pr.mel_depths if voice == "mel" else pr.bass_depths
    n_ladder = len(depths)
    presented = select_presented_depths(n_ladder, n_levels)
    pairs: List[Tuple[List[int], List[int]]] = []
    for i in range(len(presented) - 1):
        src_level = depths[presented[i]]
        tgt_level = depths[presented[i + 1]]
        bkt = level_bucket(i, len(presented))
        for start in range(0, len(src_level), max_src_notes):
            chunk = src_level[start:start + max_src_notes]
            if not chunk:
                continue
            t0, t1 = chunk[0].note.start, chunk[-1].note.end
            tgt_notes = [sn.note for sn in tgt_level if t0 <= sn.note.start < t1]
            if not tgt_notes:
                continue
            src_tok = tokenize_source_chunk(chunk, bkt, voice, pr.tonic_pc, pr.mode, pr.tc, pr.spans)
            tgt_tok = tokenize_target_notes(tgt_notes, pr.tc)
            if len(tgt_tok) > max_tgt_tokens:
                continue
            pairs.append((src_tok, tgt_tok))
    return pairs


def _pad(seq: List[int], length: int) -> List[int]:
    seq = seq[:length]
    return seq + [PAD] * (length - len(seq))


def prepare_dataset(corpus_dir: str, out_path: str, n_levels: int = DEFAULT_N_LEVELS,
                     max_src_notes: int = 4, max_reduction_depth: int = 500,
                     window_beats: Optional[float] = None, key: Optional[str] = None,
                     val_frac: float = 0.15, seed: int = 42,
                     max_src_len: int = 24, max_tgt_len: int = 64,
                     verbose: bool = True) -> dict:
    corpus = Path(corpus_dir)
    files = sorted(list(corpus.rglob("*.mid")) + list(corpus.rglob("*.midi")))
    if not files:
        raise ValueError(f"no se encontraron ficheros .mid/.midi en {corpus_dir}")

    rng = random.Random(seed)
    file_list = list(files)
    rng.shuffle(file_list)
    n_val_files = max(1, int(round(len(file_list) * val_frac))) if len(file_list) > 3 else 0
    val_files = set(str(f) for f in file_list[:n_val_files])

    all_src, all_tgt, all_split = [], [], []
    n_ok, n_fail = 0, 0
    for f in files:
        try:
            pr = reduce_piece_depths(str(f), key=key, window_beats=window_beats,
                                      max_depth=max_reduction_depth)
        except Exception as e:
            n_fail += 1
            if verbose:
                print(f"  [aviso] omitido {f.name}: {e}")
            continue
        split = 1 if str(f) in val_files else 0
        got_any = False
        for voice in ("mel", "bass"):
            pairs = build_training_pairs(pr, voice, n_levels=n_levels,
                                          max_src_notes=max_src_notes, max_tgt_tokens=max_tgt_len)
            for src_tok, tgt_tok in pairs:
                if len(src_tok) > max_src_len or len(tgt_tok) > max_tgt_len:
                    continue
                all_src.append(_pad(src_tok, max_src_len))
                all_tgt.append(_pad(tgt_tok, max_tgt_len))
                all_split.append(split)
                got_any = True
        if got_any:
            n_ok += 1
        if verbose:
            print(f"  [ok] {f.name}: {len(pr.mel_depths)} niveles, "
                  f"{sum(1 for s in all_split if True)} pares acumulados")

    if not all_src:
        raise ValueError("no se generó ningún par de entrenamiento válido a partir del corpus")

    src_arr = np.array(all_src, dtype=np.int32)
    tgt_arr = np.array(all_tgt, dtype=np.int32)
    split_arr = np.array(all_split, dtype=np.int8)

    meta = {
        "version": VERSION, "vocab_size": VOCAB_SIZE,
        "max_src_len": max_src_len, "max_tgt_len": max_tgt_len,
        "n_levels": n_levels, "max_src_notes": max_src_notes,
        "n_pairs": len(all_src), "n_files_ok": n_ok, "n_files_failed": n_fail,
        "n_val_pairs": int(split_arr.sum()), "n_train_pairs": int((split_arr == 0).sum()),
    }
    np.savez_compressed(out_path, src=src_arr, tgt=tgt_arr, split=split_arr,
                         meta=json.dumps(meta))
    if verbose:
        print(f"\nDataset guardado en {out_path}")
        print(f"  ficheros OK: {n_ok}   ficheros omitidos: {n_fail}")
        print(f"  pares train: {meta['n_train_pairs']}   pares val: {meta['n_val_pairs']}")
    return meta


# ══════════════════════════════════════════════════════════════════════════════
#  [8] MODELO — Transformer encoder-decoder pequeño (torch, import perezoso)
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(vocab_size: int, d_model: int = 128, nhead: int = 4,
                  num_layers: int = 3, dim_ff: int = 256, dropout: float = 0.1,
                  max_len: int = 96):
    import torch
    import torch.nn as nn
    import warnings
    warnings.filterwarnings("ignore", message=".*nested tensor.*", category=UserWarning)

    class Seq2SeqTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.pos_emb = nn.Embedding(max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                                    batch_first=True)
            dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                                    batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
            self.out_proj = nn.Linear(d_model, vocab_size)
            self.dropout = nn.Dropout(dropout)

        def _embed(self, ids):
            L = ids.size(1)
            pos = torch.arange(L, device=ids.device).unsqueeze(0).clamp(max=max_len - 1)
            return self.dropout(self.tok_emb(ids) * math.sqrt(self.d_model) + self.pos_emb(pos))

        def encode(self, src):
            src_pad_mask = (src == PAD)
            memory = self.encoder(self._embed(src), src_key_padding_mask=src_pad_mask)
            return memory, src_pad_mask

        def decode(self, tgt_in, memory, src_pad_mask):
            L = tgt_in.size(1)
            causal = torch.triu(torch.ones(L, L, device=tgt_in.device, dtype=torch.bool), diagonal=1)
            tgt_pad_mask = (tgt_in == PAD)
            out = self.decoder(self._embed(tgt_in), memory, tgt_mask=causal,
                                tgt_key_padding_mask=tgt_pad_mask,
                                memory_key_padding_mask=src_pad_mask)
            return self.out_proj(out)

        def forward(self, src, tgt_in):
            memory, src_pad_mask = self.encode(src)
            return self.decode(tgt_in, memory, src_pad_mask)

    return Seq2SeqTransformer()


# ══════════════════════════════════════════════════════════════════════════════
#  [9] ENTRENADOR
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    CHECKPOINT_NAME = "checkpoint.pt"
    BEST_NAME = "best_model.pt"
    HISTORY_NAME = "history.json"
    CONFIG_NAME = "model_config.json"

    def __init__(self, model, optimizer, model_dir: Path, patience: int = 20):
        self.model = model
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.patience = patience
        self.history = {"train": [], "val": []}
        self.best_val_loss = float("inf")
        self.no_improve = 0
        self.start_epoch = 0

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        import torch
        state = {"epoch": epoch, "model_state": self.model.state_dict(),
                  "optimizer_state": self.optimizer.state_dict(),
                  "best_val_loss": self.best_val_loss, "no_improve": self.no_improve,
                  "history": self.history}
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        with open(self.model_dir / self.HISTORY_NAME, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self):
        import torch
        path = self.model_dir / self.CHECKPOINT_NAME
        if not path.exists():
            print("[train] Entrenando desde cero.")
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_val_loss = state["best_val_loss"]
        self.no_improve = state["no_improve"]
        self.history = state["history"]
        self.start_epoch = state["epoch"] + 1
        print(f"[train] Reanudando desde época {self.start_epoch} "
              f"(mejor val={self.best_val_loss:.4f})")

    def _run_epoch(self, loader, training: bool):
        import torch
        import torch.nn.functional as F
        self.model.train(training)
        total, n_batches = 0.0, 0
        device = next(self.model.parameters()).device
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                logits = self.model(src, tgt_in)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        tgt_out.reshape(-1), ignore_index=PAD)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total += loss.item()
                n_batches += 1
        return total / max(n_batches, 1)

    def train(self, train_loader, val_loader, n_epochs: int, resume: bool = False):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self.load_checkpoint()

        print(f"\n{'═'*64}")
        print("  SCHENKER COMPOSER — Entrenamiento")
        print(f"  Épocas máx.: {n_epochs}   Early stopping: {self.patience}")
        print(f"  Dispositivo: {device}   Modelo dir: {self.model_dir}")
        print(f"{'═'*64}\n")

        for epoch in range(self.start_epoch, n_epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False) if val_loader is not None else train_loss
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
            print(f"  época {epoch+1:>4}/{n_epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}{'  *mejor*' if is_best else ''}")
            self.save_checkpoint(epoch, val_loss, is_best)
            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {self.patience} épocas sin mejora.")
                break
        print(f"\n  Mejor val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints en: {self.model_dir}/")


def rescale_notes(notes: List[Note], orig_start: int, orig_end: int,
                   target_start: int, target_end: int) -> List[Note]:
    orig_span = max(1, orig_end - orig_start)
    target_span = max(1, target_end - target_start)
    scale = target_span / orig_span
    out = []
    for n in notes:
        ns = target_start + int(round((n.start - orig_start) * scale))
        ne = target_start + int(round((n.end - orig_start) * scale))
        if ne <= ns:
            ne = ns + 1
        out.append(Note(pitch=n.pitch, start=ns, end=ne, channel=n.channel))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  [10] ENTRENAMIENTO — dataset PyTorch + comando CLI
# ══════════════════════════════════════════════════════════════════════════════

def _load_npz_dataset(path: str):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["src"], data["tgt"], data["split"], meta


def cmd_prepare(args):
    prepare_dataset(args.corpus, args.out, n_levels=args.n_levels,
                     max_src_notes=args.max_src_notes,
                     max_reduction_depth=args.max_reduction_depth,
                     window_beats=args.window, key=args.key,
                     val_frac=args.val_frac, seed=args.seed,
                     max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    return 0


def cmd_train(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    src, tgt, split, meta = _load_npz_dataset(args.dataset)
    if meta["vocab_size"] != VOCAB_SIZE:
        print(f"[aviso] el vocab_size del dataset ({meta['vocab_size']}) no coincide con "
              f"el vocabulario actual del programa ({VOCAB_SIZE}); si el dataset se generó "
              f"con otra versión de schenker_composer.py, regenera con 'prepare'.")

    src_t = torch.tensor(src, dtype=torch.long)
    tgt_t = torch.tensor(tgt, dtype=torch.long)
    train_mask = split == 0
    val_mask = split == 1
    if not val_mask.any():
        val_mask = train_mask

    train_ds = TensorDataset(src_t[train_mask], tgt_t[train_mask])
    val_ds = TensorDataset(src_t[val_mask], tgt_t[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    max_len = max(meta["max_src_len"], meta["max_tgt_len"])
    model = _build_model(VOCAB_SIZE, d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.num_layers, dim_ff=args.dim_ff,
                          dropout=args.dropout, max_len=max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "vocab_size": VOCAB_SIZE, "d_model": args.d_model, "nhead": args.nhead,
        "num_layers": args.num_layers, "dim_ff": args.dim_ff, "dropout": args.dropout,
        "max_len": max_len, "max_src_len": meta["max_src_len"], "max_tgt_len": meta["max_tgt_len"],
        "n_levels": meta["n_levels"], "max_src_notes": meta["max_src_notes"], "subdiv": 4,
    }
    with open(model_dir / Trainer.CONFIG_NAME, "w") as f:
        json.dump(config, f, indent=2)

    trainer = Trainer(model, optimizer, model_dir, patience=args.patience)
    trainer.train(train_loader, val_loader, n_epochs=args.epochs, resume=args.resume)
    return 0


def _load_trained_model(model_dir: Path):
    import torch
    with open(model_dir / Trainer.CONFIG_NAME) as f:
        config = json.load(f)
    model = _build_model(config["vocab_size"], d_model=config["d_model"], nhead=config["nhead"],
                          num_layers=config["num_layers"], dim_ff=config["dim_ff"],
                          dropout=config["dropout"], max_len=config["max_len"])
    best_path = model_dir / Trainer.BEST_NAME
    ckpt_path = model_dir / Trainer.CHECKPOINT_NAME
    path = best_path if best_path.exists() else ckpt_path
    if not path.exists():
        raise FileNotFoundError(f"no hay ningún checkpoint en {model_dir}; entrena primero con 'train'")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, config


def generate_tokens(model, src_ids: List[int], max_len: int = 48, temperature: float = 0.9,
                     top_k: int = 0, device: str = "cpu") -> List[int]:
    import torch
    model.to(device)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        memory, src_pad_mask = model.encode(src)
        ys = torch.tensor([[BOS]], dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = model.decode(ys, memory, src_pad_mask)
            next_logits = logits[0, -1].clone()
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / max(temperature, 1e-6)
                if top_k and top_k < next_logits.size(-1):
                    vals, idx = torch.topk(next_logits, top_k)
                    probs = torch.zeros_like(next_logits)
                    probs.scatter_(0, idx, torch.softmax(vals, dim=-1))
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == EOS:
                break
        return ys[0].tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  [11] GENERACIÓN (comando 'generate')
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_list(items: List, size: int) -> List[List]:
    return [items[i:i + size] for i in range(0, len(items), size)] if items else []


def elaborate_voice(notes: List[Note], voice: str, model, config: dict, level_bkt: int,
                     tonic_pc: int, mode: str, tc: TimeContext, spans: List[ChordSpan],
                     tpb: int, temperature: float, top_k: int, device: str) -> List[Note]:
    if not notes:
        return []
    chunks = _chunk_list(notes, config["max_src_notes"])
    out_notes: List[Note] = []
    for chunk in chunks:
        scored_chunk = [ScoredNote(note=n, function="", weight=0.0, beat=n.start / tpb) for n in chunk]
        src_tok = tokenize_source_chunk(scored_chunk, level_bkt, voice, tonic_pc, mode, tc, spans)
        src_tok = _pad(src_tok, config["max_src_len"])
        gen_ids = generate_tokens(model, src_tok, max_len=config["max_tgt_len"],
                                   temperature=temperature, top_k=top_k, device=device)
        notes_rel, cursor = detokenize_notes(gen_ids, start_tick=0, tpb=tpb, subdiv=config["subdiv"],
                                              channel=chunk[0].channel)
        chunk_start, chunk_end = chunk[0].start, chunk[-1].end
        if notes_rel:
            rescaled = rescale_notes(notes_rel, 0, max(cursor, 1), chunk_start, chunk_end)
        else:
            rescaled = [Note(pitch=chunk[0].pitch, start=chunk_start, end=chunk_end,
                              channel=chunk[0].channel)]
        out_notes.extend(rescaled)
    out_notes.sort(key=lambda n: n.start)
    return out_notes


def cmd_generate(args):
    model_dir = Path(args.model_dir)
    model, config = _load_trained_model(model_dir)
    device = "cuda" if _cuda_available() else "cpu"

    mid, tc, all_notes = load_notes(args.midi)
    num = mid.timesig_map[0][1]
    window_beats = args.window if args.window else float(num)
    tonic_pc, mode, spans = analyze_harmony(all_notes, tc, key=args.key, window_beats=window_beats)
    mel_notes, bass_notes = build_skyline(all_notes)

    voices = args.voices.split(",") if args.voices else ["mel", "bass"]
    level_bkt = max(0, min(10, args.level))

    cur_mel, cur_bass = mel_notes, bass_notes
    for it in range(max(1, args.iterations)):
        bkt_this_round = min(10, level_bkt + it)
        new_mel = elaborate_voice(cur_mel, "mel", model, config, bkt_this_round, tonic_pc, mode,
                                   tc, spans, mid.tpb, args.temperature, args.top_k, device) \
            if "mel" in voices else cur_mel
        new_bass = elaborate_voice(cur_bass, "bass", model, config, bkt_this_round, tonic_pc, mode,
                                    tc, spans, mid.tpb, args.temperature, args.top_k, device) \
            if "bass" in voices else cur_bass
        cur_mel, cur_bass = new_mel, new_bass
        print(f"  iteración {it+1}/{args.iterations}: melodía {len(cur_mel)} notas, "
              f"bajo {len(cur_bass)} notas (nivel≈{bkt_this_round})")

    trk_mel = MidiTrackData(name="melodia generada")
    trk_bass = MidiTrackData(name="bajo generado")
    for n in cur_mel:
        trk_mel.events.append(MidiEvent(abs=n.start, kind="note_on", channel=0, pitch=n.pitch, vel=88))
        trk_mel.events.append(MidiEvent(abs=n.end, kind="note_off", channel=0, pitch=n.pitch, vel=0))
    for n in cur_bass:
        trk_bass.events.append(MidiEvent(abs=n.start, kind="note_on", channel=1, pitch=n.pitch, vel=78))
        trk_bass.events.append(MidiEvent(abs=n.end, kind="note_off", channel=1, pitch=n.pitch, vel=0))
    tempo_trk = make_tempo_track(mid.tpb, num=num, den=mid.timesig_map[0][2])
    out_mid = MidiData(fmt=1, tpb=mid.tpb, tracks=[tempo_trk, trk_mel, trk_bass],
                        tempo_map=[(0, 500000)], timesig_map=[(0, num, mid.timesig_map[0][2])])
    write_midi(out_mid, args.output)
    print(f"\n  Tonalidad detectada: {pc_name(tonic_pc)}{'m' if mode=='min' else ''}")
    print(f"  MIDI generado: {args.output}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  [12] INSPECCIÓN (comando 'inspect' — versión ligera de schenkerian_reducer.py)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    pr = reduce_piece_depths(args.midi, key=args.key, window_beats=args.window,
                              max_depth=args.max_reduction_depth)
    keyname = f"{pc_name(pr.tonic_pc)}{'m' if pr.mode == 'min' else ''}"
    print(f"\n{'═'*70}")
    print(f"  SCHENKER COMPOSER — inspección de {args.midi}")
    print(f"{'═'*70}")
    print(f"  Tonalidad detectada: {keyname}")
    print(f"  Profundidad total de la escalera de reducción: {len(pr.mel_depths)} pasos")

    presented = select_presented_depths(len(pr.mel_depths), args.n_levels)
    print(f"\n  Niveles presentados (melodía), {len(presented)} de {len(pr.mel_depths)}:")
    for i, d in enumerate(presented):
        lvl = pr.mel_depths[d]
        bkt = level_bucket(i, len(presented))
        print(f"    nivel presentado {i} (bucket LEVEL_{bkt}, paso escalera {d}): "
              f"{len(lvl)} notas -> {[pc_name(s.note.pitch % 12) + str(s.note.pitch // 12 - 1) for s in lvl[:12]]}"
              f"{' ...' if len(lvl) > 12 else ''}")

    if args.export_levels:
        outdir = Path(args.export_levels)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.midi).stem
        for i, d in enumerate(presented):
            trk_mel = MidiTrackData(name="melodia")
            trk_bass = MidiTrackData(name="bajo")
            for sn in pr.mel_depths[d]:
                trk_mel.events.append(MidiEvent(abs=sn.note.start, kind="note_on", channel=0,
                                                 pitch=sn.note.pitch, vel=90))
                trk_mel.events.append(MidiEvent(abs=sn.note.end, kind="note_off", channel=0,
                                                 pitch=sn.note.pitch, vel=0))
            for sn in pr.bass_depths[min(d, len(pr.bass_depths) - 1)]:
                trk_bass.events.append(MidiEvent(abs=sn.note.start, kind="note_on", channel=1,
                                                  pitch=sn.note.pitch, vel=80))
                trk_bass.events.append(MidiEvent(abs=sn.note.end, kind="note_off", channel=1,
                                                  pitch=sn.note.pitch, vel=0))
            out_mid = MidiData(fmt=1, tpb=pr.tpb, tracks=[trk_mel, trk_bass],
                                tempo_map=[(0, 500000)], timesig_map=[(0, 4, 4)])
            write_midi(out_mid, str(outdir / f"{stem}.presentado{i}.mid"))
        print(f"\n  Niveles exportados en: {outdir}/")
    print(f"{'═'*70}\n")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  [13] CORPUS SINTÉTICO (comando 'synth-corpus' — para pruebas sin corpus real)
# ══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_piece(rng: random.Random, tpb: int = 480) -> MidiData:
    """Genera una melodía tonal simple (progresión I-IV-V-I con ornamentación
    de notas de paso/vecinas) más un bajo por fundamentales, útil solo para
    probar el pipeline end-to-end con datos sintéticos controlados."""
    tonic = rng.choice([60, 62, 64, 65, 67])
    scale = [tonic + iv for iv in MAJOR_SCALE] + [tonic + 12]
    progression = rng.choice([
        [0, 3, 4, 0], [0, 5, 3, 4, 0], [0, 4, 0, 3, 4, 0], [0, 3, 4, 4, 0],
    ])
    chord_tones = {0: [0, 2, 4], 3: [3, 5, 0], 4: [4, 6, 1], 5: [5, 0, 2]}

    trk_mel = MidiTrackData(name="melodia")
    trk_bass = MidiTrackData(name="bajo")
    t = 0
    prev_deg = 0
    for deg in progression:
        for _ in range(rng.choice([2, 4])):
            ct = chord_tones.get(deg, [0, 2, 4])
            if rng.random() < 0.55:
                target_deg = rng.choice(ct)
            else:
                target_deg = (prev_deg + rng.choice([-1, 1])) % 7
            pitch = scale[target_deg % len(scale)]
            dur = rng.choice([tpb, tpb, tpb // 2, tpb // 2, tpb * 2])
            trk_mel.events.append(MidiEvent(abs=t, kind="note_on", pitch=pitch, vel=92, channel=0))
            trk_mel.events.append(MidiEvent(abs=t + dur, kind="note_off", pitch=pitch, vel=0, channel=0))
            t += dur
            prev_deg = target_deg
        bass_pitch = scale[deg % len(scale)] - 12
        bass_dur = 4 * tpb
        b_start = t - bass_dur if t >= bass_dur else 0
    # bajo simple: una fundamental por acorde de la progresión, alineada por compás
    bt = 0
    bar_ticks = 4 * tpb
    for deg in progression:
        bass_pitch = scale[deg % len(scale)] - 12
        trk_bass.events.append(MidiEvent(abs=bt, kind="note_on", pitch=bass_pitch, vel=80, channel=1))
        trk_bass.events.append(MidiEvent(abs=bt + bar_ticks, kind="note_off", pitch=bass_pitch, vel=0, channel=1))
        bt += bar_ticks
    total = max(t, bt)
    tempo_trk = make_tempo_track(tpb, bpm=100, num=4, den=4)
    return MidiData(fmt=1, tpb=tpb, tracks=[tempo_trk, trk_mel, trk_bass],
                     tempo_map=[(0, 500000)], timesig_map=[(0, 4, 4)])


def cmd_synth_corpus(args):
    rng = random.Random(args.seed)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        mid = _make_synthetic_piece(rng)
        write_midi(mid, str(outdir / f"synth_{i:03d}.mid"))
    print(f"  {args.n} MIDIs sintéticos escritos en {outdir}/")
    print("  (uso exclusivo de prueba: melodías tonales I-IV-V-I con vecinas/pasos, "
          "no pretenden tener valor musical propio)")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  [14] CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="schenker_composer.py",
        description="Recomposición neuronal por niveles de reducción schenkeriana.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("synth-corpus", help="Genera MIDIs sintéticos para probar el pipeline")
    p_synth.add_argument("out_dir")
    p_synth.add_argument("--n", type=int, default=30)
    p_synth.add_argument("--seed", type=int, default=42)

    p_insp = sub.add_parser("inspect", help="Muestra la reducción schenkeriana interna de un MIDI")
    p_insp.add_argument("midi")
    p_insp.add_argument("--key")
    p_insp.add_argument("--window", type=float, default=None)
    p_insp.add_argument("--max-reduction-depth", type=int, default=500)
    p_insp.add_argument("--n-levels", type=int, default=DEFAULT_N_LEVELS)
    p_insp.add_argument("--export-levels")

    p_prep = sub.add_parser("prepare", help="Corpus de MIDIs → dataset de pares (nivel n, nivel n+1)")
    p_prep.add_argument("corpus")
    p_prep.add_argument("--out", default="dataset.npz")
    p_prep.add_argument("--key")
    p_prep.add_argument("--window", type=float, default=None)
    p_prep.add_argument("--max-reduction-depth", type=int, default=500)
    p_prep.add_argument("--n-levels", type=int, default=DEFAULT_N_LEVELS)
    p_prep.add_argument("--max-src-notes", type=int, default=4)
    p_prep.add_argument("--max-src-len", type=int, default=24)
    p_prep.add_argument("--max-tgt-len", type=int, default=64)
    p_prep.add_argument("--val-frac", type=float, default=0.15)
    p_prep.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Entrena el Transformer encoder-decoder")
    p_train.add_argument("dataset")
    p_train.add_argument("--model-dir", default="modelo_schenker/")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--d-model", type=int, default=128)
    p_train.add_argument("--nhead", type=int, default=4)
    p_train.add_argument("--num-layers", type=int, default=3)
    p_train.add_argument("--dim-ff", type=int, default=256)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--resume", action="store_true")

    p_gen = sub.add_parser("generate", help="Eleva un MIDI de un nivel n al nivel n+1 (elaboración)")
    p_gen.add_argument("midi")
    p_gen.add_argument("--model-dir", default="modelo_schenker/")
    p_gen.add_argument("--output", default="salida_elaborada.mid")
    p_gen.add_argument("--level", type=int, default=5,
                        help="0=nivel más reducido/Ursatz-like ... 10=superficie (default: 5)")
    p_gen.add_argument("--iterations", type=int, default=1,
                        help="aplica la elaboración N veces seguidas, subiendo de nivel cada vez")
    p_gen.add_argument("--voices", default="mel,bass", help="mel | bass | mel,bass")
    p_gen.add_argument("--key")
    p_gen.add_argument("--window", type=float, default=None)
    p_gen.add_argument("--temperature", type=float, default=0.9)
    p_gen.add_argument("--top-k", type=int, default=0)
    p_gen.add_argument("--seed", type=int, default=42)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
    dispatch = {
        "synth-corpus": cmd_synth_corpus,
        "inspect": cmd_inspect,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "generate": cmd_generate,
    }
    try:
        return dispatch[args.command](args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
