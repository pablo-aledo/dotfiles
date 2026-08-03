#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GRADED VARIATIONS  v1.0  (autocontenido)                  ║
║                                                                              ║
║  Genera variaciones musicales de una pieza de piano simplificada a un grado ║
║  de dificultad pedagógica. Cada variación aísla una dimensión musical       ║
║  (ornamentación, ritmo, textura, registro, armonía, modo, articulación)     ║
║  y produce una versión alternativa de la misma pieza simplificada.          ║
║                                                                              ║
║  El grado de cada variación puede variar ±1 respecto al objetivo para       ║
║  aislar mejor la técnica modificada. Se exporta también la versión           ║
║  simplificada limpia como referencia.                                       ║
║                                                                              ║
║  USO:                                                                        ║
║    python graded_variations.py obra.mid --grade 3 --variations 2            ║
║    python graded_variations.py obra.mid --grade 5 --types ornamental ritmica║
║    python graded_variations.py obra.mid --grade 2 --seed 42 --outdir out/   ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    <nombre>_grade<N>_base.mid              versión simplificada limpia      ║
║    <nombre>_grade<N>_<tipo>_<k>.mid        variación k del tipo             ║
║    + informe por consola con los cambios de cada variación                  ║
║                                                                              ║
║  DEPENDENCIAS: solo numpy (stdlib para todo lo demás). Un único fichero.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sys, os, json, math, random, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal, Set
import numpy as np

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
def _c(k): return _COLORS.get(k, "") if _USE_COLOR else ""

# ══════════════════════════════════════════════════════════════════════════════
#  [0] E/S MIDI AUTOCONTENIDA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class MidiEvent:
    abs: int; kind: str; channel: int = 0; data: bytes = b""
    pitch: int = 0; vel: int = 0; meta_type: int = -1
@dataclass
class MidiTrackData:
    name: str = ""; events: List[MidiEvent] = field(default_factory=list)
@dataclass
class MidiData:
    fmt: int = 1; tpb: int = 480
    tracks: List[MidiTrackData] = field(default_factory=list)
    tempo_map: List[Tuple[int,int]] = field(default_factory=list)
    timesig_map: List[Tuple[int,int,int]] = field(default_factory=list)

def _read_vlq(buf, i):
    val = 0
    while True:
        b = buf[i]; i += 1
        val = (val << 7) | (b & 0x7F)
        if not b & 0x80: return val, i
def _write_vlq(val):
    out = [val & 0x7F]; val >>= 7
    while val: out.append((val & 0x7F) | 0x80); val >>= 7
    return bytes(reversed(out))

def read_midi(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd": raise ValueError(f"{path}: no es un archivo MIDI")
    fmt = int.from_bytes(raw[8:10],"big"); n_tr = int.from_bytes(raw[10:12],"big")
    div = int.from_bytes(raw[12:14],"big")
    if div & 0x8000: raise ValueError(f"{path}: SMPTE no soportado")
    mid = MidiData(fmt=fmt, tpb=div); i = 14
    for _ in range(n_tr):
        if raw[i:i+4] != b"MTrk": raise ValueError("chunk corrupto")
        length = int.from_bytes(raw[i+4:i+8],"big"); chunk = raw[i+8:i+8+length]; i += 8+length
        trk = MidiTrackData(); t = 0; j = 0; status = 0
        while j < len(chunk):
            delta, j = _read_vlq(chunk, j); t += delta; b0 = chunk[j]
            if b0 & 0x80: status = b0; j += 1
            if status == 0xFF:
                mt = chunk[j]; j += 1; ml, j = _read_vlq(chunk, j); md = chunk[j:j+ml]; j += ml
                if mt == 0x03 and not trk.name: trk.name = md.decode("latin-1","replace").strip()
                elif mt == 0x51 and ml == 3: mid.tempo_map.append((t, int.from_bytes(md,"big")))
                elif mt == 0x58 and ml >= 2: mid.timesig_map.append((t, md[0], 2**md[1]))
                if mt != 0x2F: trk.events.append(MidiEvent(abs=t,kind="meta",meta_type=mt,data=bytes([0xFF,mt])+_write_vlq(ml)+md))
            elif status in (0xF0,0xF7):
                sl, j = _read_vlq(chunk, j); j += sl
            else:
                hi, ch = status & 0xF0, status & 0x0F
                if hi in (0xC0,0xD0): j += 1
                else:
                    d1, d2 = chunk[j], chunk[j+1]; j += 2
                    if hi == 0x90 and d2 > 0: trk.events.append(MidiEvent(abs=t,kind="note_on",channel=ch,pitch=d1,vel=d2))
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0): trk.events.append(MidiEvent(abs=t,kind="note_off",channel=ch,pitch=d1,vel=d2))
        mid.tracks.append(trk)
    if not mid.tempo_map: mid.tempo_map = [(0,500000)]
    mid.tempo_map.sort()
    if not mid.timesig_map: mid.timesig_map = [(0,4,4)]
    mid.timesig_map.sort()
    return mid

def write_midi(mid, path):
    chunks = []
    for trk in mid.tracks:
        evs = sorted(trk.events, key=lambda e: (e.abs, 0 if e.kind=="note_off" else 1))
        body = bytearray(); last = 0
        for ev in evs:
            body += _write_vlq(max(0, ev.abs - last)); last = ev.abs
            if ev.kind == "note_on": body += bytes([0x90|(ev.channel&0xF), ev.pitch&0x7F, ev.vel&0x7F])
            elif ev.kind == "note_off": body += bytes([0x80|(ev.channel&0xF), ev.pitch&0x7F, ev.vel&0x7F])
            else: body += ev.data
        body += _write_vlq(0) + bytes([0xFF,0x2F,0x00])
        chunks.append(b"MTrk" + len(body).to_bytes(4,"big") + bytes(body))
    header = (b"MThd" + (6).to_bytes(4,"big") + mid.fmt.to_bytes(2,"big")
              + len(mid.tracks).to_bytes(2,"big") + mid.tpb.to_bytes(2,"big"))
    Path(path).write_bytes(header + b"".join(chunks))

class TimeContext:
    def __init__(self, mid):
        self.tpb = mid.tpb; self.timesig_map = mid.timesig_map
        self._bars = []; bar, prev_tick = 1.0, 0
        prev_tpc = self.tpb * 4 * self.timesig_map[0][1] // self.timesig_map[0][2]
        for tick, num, den in self.timesig_map:
            bar += (tick - prev_tick) / prev_tpc
            tpc = max(1, self.tpb * 4 * num // den)
            self._bars.append((tick, bar, tpc)); prev_tick, prev_tpc = tick, tpc
        if not self._bars or self._bars[0][0] > 0:
            self._bars.insert(0, (0, 1.0, prev_tpc))
        tempo_map = list(mid.tempo_map)
        if not tempo_map or tempo_map[0][0] != 0: tempo_map = [(0,500000)] + tempo_map
        self._tempo_segs = []; cum_sec = 0.0; prev_t, prev_us = 0, tempo_map[0][1]
        for tick, us in tempo_map:
            if tick > prev_t: cum_sec += (tick - prev_t) * prev_us / 1_000_000.0 / self.tpb
            self._tempo_segs.append((tick, cum_sec, us)); prev_t, prev_us = tick, us
    def beat(self, tick): return tick / self.tpb
    def bar(self, tick):
        seg = self._bars[0]
        for s in self._bars:
            if s[0] <= tick: seg = s
            else: break
        return int(seg[1] + (tick - seg[0]) / seg[2])
    def sec(self, tick):
        seg = self._tempo_segs[0]
        for s in self._tempo_segs:
            if s[0] <= tick: seg = s
            else: break
        return seg[1] + (tick - seg[0]) * seg[2] / 1_000_000.0 / self.tpb
    def bar_start_tick(self, bar):
        tick, step = 0, max(1, self.tpb // 8); limit = 20_000_000
        while tick < limit:
            if self.bar(tick) >= bar: return tick
            tick += step
        return limit
    def bar_range_ticks(self, bar): return self.bar_start_tick(bar), self.bar_start_tick(bar+1)

@dataclass
class Note:
    pitch: int; start: int; end: int; vel: int = 90; channel: int = 0
    def start_beat(self, tc): return tc.beat(self.start)
    def duration_beats(self, tc): return (self.end - self.start) / tc.tpb

def extract_notes(trk):
    out = []; stack = {}
    for ev in trk.events:
        if ev.kind == "note_on": stack.setdefault((ev.channel, ev.pitch), []).append(ev.abs)
        elif ev.kind == "note_off":
            key = (ev.channel, ev.pitch)
            if stack.get(key):
                s = stack[key].pop(0)
                if ev.abs > s: out.append(Note(pitch=ev.pitch, start=s, end=ev.abs, channel=ev.channel))
    out.sort(key=lambda n: (n.start, n.pitch))
    return out

def load_notes(path):
    mid = read_midi(path); tc = TimeContext(mid); notes = []
    for trk in mid.tracks: notes.extend(extract_notes(trk))
    if not notes: raise ValueError("el MIDI no contiene notas")
    notes.sort(key=lambda n: (n.start, n.pitch))
    return mid, tc, notes

# ══════════════════════════════════════════════════════════════════════════════
#  [1] ARMONÍA
# ══════════════════════════════════════════════════════════════════════════════
CHORD_TEMPLATES = {"": {0,4,7}, "m": {0,3,7}, "dim": {0,3,6}, "aug": {0,4,8}}
_KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
MAJOR_SCALE = [0,2,4,5,7,9,11]; MINOR_SCALE = [0,2,3,5,7,8,10]

def detect_key(pc_hist):
    best = (0,"maj",-1e9)
    for tonic in range(12):
        cm = float(np.corrcoef(pc_hist, np.roll(_KS_MAJ,tonic))[0,1]) if pc_hist.sum() else 0.0
        cn = float(np.corrcoef(pc_hist, np.roll(_KS_MIN,tonic))[0,1]) if pc_hist.sum() else 0.0
        if cm > best[2]: best = (tonic,"maj",cm)
        if cn > best[2]: best = (tonic,"min",cn)
    return best[0], best[1]

def detect_chord(pcs_weight, bass_pc):
    present = {pc for pc, w in pcs_weight.items() if w > 0}
    if not present:
        return None, "", -1e9   # devolvemos también un valor para el score
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
    return best[0], best[1], best[2]   # ahora devuelve tres valores

@dataclass
class ChordSpan:
    start_beat: float; end_beat: float; root: Optional[int]; quality: str
    bass_pc: Optional[int] = None

def chord_tone_pcs(chord):
    if chord is None or chord.root is None: return set()
    return {(chord.root + iv) % 12 for iv in CHORD_TEMPLATES.get(chord.quality, {0,4,7})}

def analyze_harmony_local(notes, tc, key=None, window_beats=None):
    last_tick = max(n.end for n in notes); n_bars = max(1, tc.bar(last_tick-1))
    pc_hist = np.zeros(12)
    for n in notes: pc_hist[n.pitch % 12] += (n.end - n.start)
    if key:
        parts = key.strip().split(); tonic_pc = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}.get(parts[0][0],0)
        mode = "min" if len(parts) > 1 and parts[1].lower().startswith("m") else "maj"
    else: tonic_pc, mode = detect_key(pc_hist)
    segments = []
    if window_beats:
        step = int(round(window_beats * tc.tpb)); t = 0
        while t < last_tick: segments.append((t, t+step)); t += step
    else:
        for bar in range(1, n_bars+1): segments.append((tc.bar_start_tick(bar), tc.bar_start_tick(bar+1)))
    spans = []
    for (t0,t1) in segments:
        seg_notes = [n for n in notes if n.start < t1 and n.end > t0]
        b0, b1 = tc.beat(t0), tc.beat(t1)
        if not seg_notes: spans.append(ChordSpan(b0,b1,None,"")); continue
        pw = {}
        for n in seg_notes:
            dur = min(n.end,t1) - max(n.start,t0); pw[n.pitch%12] = pw.get(n.pitch%12,0) + dur
        bass_pc = min(seg_notes, key=lambda n: n.pitch).pitch % 12
        root, suffix, _ = detect_chord(pw, bass_pc)
        spans.append(ChordSpan(b0,b1,root,suffix,bass_pc))
    return tonic_pc, mode, spans

def chord_at(spans, beat):
    for sp in spans:
        if sp.start_beat <= beat < sp.end_beat: return sp
    return spans[-1] if spans else None

def scale_degree(pitch, tonic_pc, mode):
    scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
    rel = (pitch - tonic_pc) % 12
    return scale.index(rel) + 1 if rel in scale else None

# ══════════════════════════════════════════════════════════════════════════════
#  [2] PESO MÉTRICO + SKYLINE + CLASIFICACIÓN + PESO ESTRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════
def metric_weights(num, subdiv):
    n = num * subdiv; W = np.zeros(n)
    for i in range(n):
        level, k = 0, i
        while k % 2 == 0 and k != 0: k //= 2; level += 1
        W[i] = level + 1
        if i == 0: W[i] = level + num + 2
    return W

def note_metric_weight(note, tc, num, subdiv, W):
    pos = tc.beat(note.start) % num; idx = int(round(pos * subdiv)) % (num * subdiv)
    return float(W[idx] / W.max()) if W.size else 1.0

def build_skyline(notes):
    if not notes: return [], []
    bounds = sorted(set(n.start for n in notes) | set(n.end for n in notes))
    mel_segs, bass_segs = [], []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        mid_t = (t0+t1)/2.0; active = [n for n in notes if n.start <= mid_t < n.end]
        if not active: continue
        top = max(active, key=lambda n: n.pitch); bot = min(active, key=lambda n: n.pitch)
        mel_segs.append((id(top), Note(top.pitch,t0,t1,top.vel))); bass_segs.append((id(bot), Note(bot.pitch,t0,t1,bot.vel)))
    def _merge(segs):
        out, oids = [], []
        for nid, s in segs:
            if out and oids[-1] == nid and out[-1].end == s.start: out[-1] = Note(out[-1].pitch, out[-1].start, s.end, out[-1].vel)
            else: out.append(Note(s.pitch,s.start,s.end,s.vel)); oids.append(nid)
        return out
    return _merge(mel_segs), _merge(bass_segs)

NoteFunction = Literal["CT","PT","NT","APP","SUS","ANT","ESC","UNCLASSIFIED"]
_STEP_MAX = 2

def classify_note(note, prev, nxt, chord_now, chord_prev, chord_next, mweight):
    pc = note.pitch % 12; ct_now = chord_tone_pcs(chord_now)
    if pc in ct_now: return "CT"
    si = (note.pitch - prev.pitch) if prev else None
    so = (nxt.pitch - note.pitch) if nxt else None
    if prev and note.pitch == prev.pitch and chord_prev and pc in chord_tone_pcs(chord_prev) and so and so < 0 and abs(so) <= _STEP_MAX:
        if nxt and (nxt.pitch%12) in (ct_now | chord_tone_pcs(chord_next)): return "SUS"
    if si and abs(si) > _STEP_MAX and mweight >= 0.5 and so and abs(so) <= _STEP_MAX:
        if nxt and (nxt.pitch%12) in (ct_now | chord_tone_pcs(chord_next)): return "APP"
    if chord_next and mweight < 0.5 and pc in chord_tone_pcs(chord_next) and pc not in ct_now: return "ANT"
    if si and so and abs(si) <= _STEP_MAX and abs(so) > _STEP_MAX and si != 0 and (si>0) != (so>0): return "ESC"
    if si and so and 0 < abs(si) <= _STEP_MAX and 0 < abs(so) <= _STEP_MAX:
        return "PT" if (si>0)==(so>0) else "NT"
    return "UNCLASSIFIED"

DEFAULT_WEIGHTS = {"CT":1.0,"SUS":0.7,"APP":0.5,"ANT":0.3,"PT":0.25,"NT":0.2,"ESC":0.2,"UNCLASSIFIED":0.4}

def structural_weight(function, mweight, dur_beats, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    base = weights.get(function, 0.4)
    df = min(max(dur_beats,0)/1.0, 1.5) if dur_beats > 0 else 1.0
    return base * (0.5 + 0.5*mweight) * max(df, 0.5)

@dataclass
class ScoredNote:
    note: Note; function: NoteFunction; weight: float; beat: float

def _score_voice(voice_notes, tc, spans, num, subdiv, W):
    scored = []
    for i, n in enumerate(voice_notes):
        beat = n.start_beat(tc); ch_now = chord_at(spans, beat)
        ch_prev = chord_at(spans, voice_notes[i-1].start_beat(tc)) if i > 0 else None
        ch_next = chord_at(spans, voice_notes[i+1].start_beat(tc)) if i < len(voice_notes)-1 else None
        mw = note_metric_weight(n, tc, num, subdiv, W)
        func = classify_note(n, voice_notes[i-1] if i>0 else None, voice_notes[i+1] if i<len(voice_notes)-1 else None, ch_now, ch_prev, ch_next, mw)
        scored.append(ScoredNote(n, func, structural_weight(func, mw, n.duration_beats(tc)), beat))
    return scored

# ══════════════════════════════════════════════════════════════════════════════
#  [3] DIFICULTAD PIANÍSTICA (5 factores)
# ══════════════════════════════════════════════════════════════════════════════
def separate_hands(mid, split=60):
    nts = [(i, extract_notes(t)) for i,t in enumerate(mid.tracks)]
    nts = [(i,ns) for i,ns in nts if ns]
    if len(nts) >= 2:
        avg = [(i, sum(n.pitch for n in ns)/len(ns), ns) for i,ns in nts]; avg.sort(key=lambda x: -x[1])
        return list(avg[0][2]), [n for _,_,ns in avg[1:] for n in ns]
    all_n = [n for _,ns in nts for n in ns]
    return [n for n in all_n if n.pitch >= split], [n for n in all_n if n.pitch < split]

def _chords_at_onsets(notes):
    by_start = {}
    for n in notes: by_start.setdefault(n.start, []).append(n)
    return [by_start[s] for s in sorted(by_start)]

def _independence(rh, lh, tc):
    if not rh or not lh: return 0.0
    grid = max(1, tc.tpb // 4)
    sr = {n.start // grid for n in rh}; sl = {n.start // grid for n in lh}
    inter, union = len(sr & sl), len(sr | sl)
    return round(1.0 - inter/union, 3) if union else 0.0

def _peak_nps(notes, tc):
    if not notes: return 0.0
    starts = sorted(tc.sec(n.start) for n in notes); peak, j = 0, 0
    for i in range(len(starts)):
        while starts[i] - starts[j] > 2.0: j += 1
        peak = max(peak, i-j+1)
    return peak / 2.0

def _span_poly_leaps(bar_notes):
    chords = _chords_at_onsets(bar_notes)
    spans = [max(c,key=lambda n:n.pitch).pitch - min(c,key=lambda n:n.pitch).pitch for c in chords if len(c)>1]
    max_span = max(spans) if spans else 0
    max_poly = max((len(c) for c in chords), default=0)
    top = [max(c,key=lambda n:n.pitch).pitch for c in chords]
    leaps = sum(1 for a,b in zip(top[:-1],top[1:]) if abs(b-a) > 9)
    return max_span, max_poly, leaps

def bar_factors(rh_bar, lh_bar, rh_ctx, lh_ctx, tc):
    peak = max(_peak_nps(rh_ctx,tc), _peak_nps(lh_ctx,tc))
    rs, rp, rl = _span_poly_leaps(rh_bar); ls, lp, ll = _span_poly_leaps(lh_bar)
    indep = _independence(rh_bar, lh_bar, tc)
    f = {"velocidad": float(np.clip(peak/12.0,0,1)), "extension": float(np.clip(max(rs,ls)/14.0,0,1)),
         "polifonia": float(np.clip((max(rp,lp)-1)/4.0,0,1)), "saltos": float(np.clip((rl+ll)/2.0,0,1)),
         "independencia": float(np.clip(indep/0.7,0,1))}
    w = {"velocidad":0.30,"extension":0.20,"polifonia":0.20,"saltos":0.15,"independencia":0.15}
    diff = sum(f[k]*w[k] for k in f)
    return int(np.clip(round(diff*7),0,7))+1, f, diff

def whole_piece_grade(rh, lh, tc):
    all_n = rh + lh
    if not all_n: return 1, 0.0
    n_bars = max(1, tc.bar(max(n.end for n in all_n)-1))
    def hm(notes):
        if not notes: return dict(peak=0.0,span=0,poly=0,leaps=0)
        starts = sorted(tc.sec(n.start) for n in notes); peak, j = 0, 0
        for i in range(len(starts)):
            while starts[i]-starts[j] > 2.0: j += 1
            peak = max(peak, i-j+1)
        chords = _chords_at_onsets(notes)
        spans = [max(c,key=lambda n:n.pitch).pitch-min(c,key=lambda n:n.pitch).pitch for c in chords if len(c)>1]
        return dict(peak=peak/2.0, span=max(spans) if spans else 0, poly=max((len(c) for c in chords),default=0),
                    leaps=sum(1 for a,b in zip([max(c,key=lambda n:n.pitch).pitch for c in chords][:-1],
                                               [max(c,key=lambda n:n.pitch).pitch for c in chords][1:]) if abs(b-a)>9))
    mr, ml = hm(rh), hm(lh)
    f = {"velocidad": float(np.clip(max(mr["peak"],ml["peak"])/12.0,0,1)),
         "extension": float(np.clip(max(mr["span"],ml["span"])/14.0,0,1)),
         "polifonia": float(np.clip((max(mr["poly"],ml["poly"])-1)/4.0,0,1)),
         "saltos": float(np.clip((mr["leaps"]+ml["leaps"])/max(1,n_bars)/2.0,0,1)),
         "independencia": float(np.clip(_independence(rh,lh,tc)/0.7,0,1))}
    w = {"velocidad":0.30,"extension":0.20,"polifonia":0.20,"saltos":0.15,"independencia":0.15}
    diff = sum(f[k]*w[k] for k in f)
    return int(np.clip(round(diff*7),0,7))+1, diff

# ══════════════════════════════════════════════════════════════════════════════
#  [4] SIMPLIFICACIÓN AL GRADO OBJETIVO (greedy + rejilla de respaldo)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(eq=False)
class NoteRec:
    note: Note; hand: str; bar: int; voice_role: str; function: NoteFunction
    weight: float; floor_protected: bool; is_octave_dup: bool; candidate: bool; orig_id: int = -1

def simplify_to_grade(midi_path, target_grade, floor_level="ursatz", split=60, threshold=0.35):
    """Simplifica un MIDI a un grado objetivo. Devuelve (records, tc, mid,
    tonic_pc, mode, spans, rh_notes, lh_notes, grade_achieved)."""
    mid, tc, all_notes = load_notes(midi_path)
    tonic_pc, mode, spans = analyze_harmony_local(all_notes, tc)
    rh_notes, lh_notes = separate_hands(mid, split)
    num = mid.timesig_map[0][1]; subdiv = 4; W = metric_weights(num, subdiv)
    # skyline + suelo estructural simplificado
    mel_sk, bass_sk = build_skyline(all_notes)
    scored_mel = _score_voice(mel_sk, tc, spans, num, subdiv, W)
    scored_bass = _score_voice(bass_sk, tc, spans, num, subdiv, W)
    # suelo: notas con peso >= threshold en el nivel más reducido
    def _floor_keys(scored):
        if not scored: return set()
        return {(s.note.start, s.note.pitch) for s in scored if s.weight >= threshold}
    mel_floor = _floor_keys(scored_mel); bass_floor = _floor_keys(scored_bass)
    # construir registros
    rh_set = {(n.start, n.pitch, n.end) for n in rh_notes}
    mel_by_key = {(s.note.start, s.note.pitch): s for s in scored_mel}
    bass_by_key = {(s.note.start, s.note.pitch): s for s in scored_bass}
    by_onset = {}
    for n in all_notes: by_onset.setdefault(n.start, []).append(n)
    dup_ids = set()
    for onset, grp in by_onset.items():
        pcs_seen = {}
        for n in grp: pcs_seen.setdefault(n.pitch%12, []).append(id(n))
        for pc, ids in pcs_seen.items():
            if len(ids) > 1: dup_ids.update(ids)
    records = []
    for orig_id, n in enumerate(all_notes):
        hand = "rh" if (n.start, n.pitch, n.end) in rh_set else "lh"
        bar = tc.bar(n.start); key = (n.start, n.pitch); is_dup = id(n) in dup_ids
        role, func, weight, fp = "inner", "UNCLASSIFIED", 0.4, False
        s_mel, s_bass = mel_by_key.get(key), bass_by_key.get(key)
        if s_mel or s_bass:
            s = s_mel if s_mel else s_bass; role = "mel" if s_mel else "bass"
            func, weight = s.function, s.weight
            fp = key in mel_floor or key in bass_floor
        else:
            beat = n.start_beat(tc); ch = chord_at(spans, beat)
            mw = note_metric_weight(n, tc, num, subdiv, W)
            func = "CT" if (n.pitch%12) in chord_tone_pcs(ch) else "UNCLASSIFIED"
            weight = structural_weight(func, mw, n.duration_beats(tc))
        candidate = False if fp else (False if role in ("mel","bass") and func == "CT" else (True if is_dup or func in ("PT","NT","APP","SUS","ANT","ESC","UNCLASSIFIED") else False))
        records.append(NoteRec(n, hand, bar, role, func, weight, fp, is_dup, candidate, orig_id))
    # greedy por compás
    last_tick = max(n.end for n in all_notes); n_bars = max(1, tc.bar(last_tick-1))
    active = list(records)
    for bar in range(1, n_bars+1):
        bt0, bt1 = tc.bar_range_ticks(bar)
        bar_active = [r for r in active if bt0 <= r.note.start < bt1]
        rh_bar = [r.note for r in bar_active if r.hand == "rh"]; lh_bar = [r.note for r in bar_active if r.hand == "lh"]
        bs0, bs1 = tc.sec(bt0)-2.0, tc.sec(bt1)+2.0
        rh_ctx = [r.note for r in active if r.hand=="rh" and bs0 <= tc.sec(r.note.start) <= bs1]
        lh_ctx = [r.note for r in active if r.hand=="lh" and bs0 <= tc.sec(r.note.start) <= bs1]
        cur_grade, _, _ = bar_factors(rh_bar, lh_bar, rh_ctx, lh_ctx, tc)
        candidates = [r for r in bar_active if r.candidate]
        while cur_grade > target_grade and candidates:
            best, best_delta = None, 0
            for r in candidates:
                if r not in active: continue
                active.remove(r)
                g_after, _, _ = bar_factors(
                    [rr.note for rr in active if rr.hand=="rh" and bt0<=rr.note.start<bt1],
                    [rr.note for rr in active if rr.hand=="lh" and bt0<=rr.note.start<bt1],
                    [rr.note for rr in active if rr.hand=="rh" and bs0<=tc.sec(rr.note.start)<=bs1],
                    [rr.note for rr in active if rr.hand=="lh" and bs0<=tc.sec(rr.note.start)<=bs1], tc)
                active.append(r)
                delta = cur_grade - g_after
                if delta > best_delta: best, best_delta = r, delta
            if best is None or best_delta <= 0: break
            active.remove(best); candidates.remove(best)
            cur_grade, _, _ = bar_factors(
                [rr.note for rr in active if rr.hand=="rh" and bt0<=rr.note.start<bt1],
                [rr.note for rr in active if rr.hand=="lh" and bt0<=rr.note.start<bt1],
                [rr.note for rr in active if rr.hand=="rh" and bs0<=tc.sec(rr.note.start)<=bs1],
                [rr.note for rr in active if rr.hand=="lh" and bs0<=tc.sec(rr.note.start)<=bs1], tc)
    # rejilla de respaldo si no se alcanzó el grado
    rh_final = [r.note for r in active if r.hand == "rh"]; lh_final = [r.note for r in active if r.hand == "lh"]
    achieved, _ = whole_piece_grade(rh_final, lh_final, tc)
    if achieved > target_grade:
        for grid_div, max_v in [(2,3),(2,2),(1,2),(1,1),(2,1),(4,1)]:
            grid_ticks = tc.tpb // grid_div
            for hand_key in ("rh","lh"):
                hand_recs = [r for r in active if r.hand == hand_key]
                protected = [r for r in hand_recs if r.floor_protected]
                rest = [r for r in hand_recs if not r.floor_protected]
                if not rest: continue
                buckets = {}
                for r in rest:
                    slot = round(r.note.start / grid_ticks)
                    if slot not in buckets or r.weight > buckets[slot].weight: buckets[slot] = r
                kept = list(buckets.values())
                for r in rest:
                    if r not in kept and r in active: active.remove(r)
            rh_f = [r.note for r in active if r.hand=="rh"]; lh_f = [r.note for r in active if r.hand=="lh"]
            achieved, _ = whole_piece_grade(rh_f, lh_f, tc)
            if achieved <= target_grade: break
    rh_out = [r.note for r in active if r.hand == "rh"]; lh_out = [r.note for r in active if r.hand == "lh"]
    achieved, _ = whole_piece_grade(rh_out, lh_out, tc)
    return active, tc, mid, tonic_pc, mode, spans, rh_out, lh_out, achieved

# ══════════════════════════════════════════════════════════════════════════════
#  [5] MOTOR DE VARIACIONES
# ══════════════════════════════════════════════════════════════════════════════
ALL_TYPES = ["ornamental", "ritmica", "textura", "registro", "armonica", "modo", "articulacion"]

@dataclass
class VariationResult:
    vtype: str; records: List[NoteRec]; changes: List[str]
    grade_before: int; grade_after: int; seed: int

def _records_to_notes(records):
    """Convierte NoteRec a lista de Note ordenada."""
    return sorted([r.note for r in records], key=lambda n: (n.start, n.pitch))

def _notes_to_records(notes, tc, orig_records):
    """Reconstruye NoteRec desde notas, heredando metadatos del original."""
    orig_by_key = {(r.note.start, r.note.pitch): r for r in orig_records}
    out = []
    for i, n in enumerate(notes):
        key = (n.start, n.pitch)
        if key in orig_by_key:
            o = orig_by_key[key]
            out.append(NoteRec(n, o.hand, tc.bar(n.start), o.voice_role, o.function, o.weight, o.floor_protected, o.is_octave_dup, False, i))
        else:
            hand = "rh" if n.pitch >= 60 else "lh"
            out.append(NoteRec(n, hand, tc.bar(n.start), "inner", "UNCLASSIFIED", 0.5, False, False, False, i))
    return out

def _get_scale_notes(tonic_pc, mode):
    scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
    return set((tonic_pc + iv) % 12 for iv in scale)

def _snap_to_scale(pitch, scale_notes):
    pc = pitch % 12
    if pc in scale_notes: return pitch
    best, best_d = pitch, 99
    for spc in scale_notes:
        for oct in range(pitch // 12 - 1, pitch // 12 + 2):
            candidate = oct * 12 + spc
            d = abs(candidate - pitch)
            if d < best_d: best, best_d = candidate, d
    return best

# --- VARIACIÓN ORNAMENTAL ---
def _var_ornamental(records, tc, tonic_pc, mode, spans, rng, prob=0.35):
    changes = []
    notes = _records_to_notes(records)
    scale_notes = _get_scale_notes(tonic_pc, mode)
    mel_notes = [n for n in notes if n.pitch >= 60]
    mel_notes.sort(key=lambda n: n.start)
    insertions = []
    for i in range(len(mel_notes) - 1):
        n1, n2 = mel_notes[i], mel_notes[i+1]
        interval = n2.pitch - n1.pitch
        if abs(interval) in (2, 3, 4) and rng.random() < prob:
            gap = n2.start - n1.end
            if gap >= tc.tpb // 4:
                mid_pitch = _snap_to_scale(n1.pitch + interval // 2, scale_notes)
                mid_dur = max(tc.tpb // 4, gap // 2)
                insertions.append(Note(mid_pitch, n1.end, n1.end + mid_dur, min(n1.vel, 75), n1.channel))
                changes.append(f"nota de paso {mid_pitch%12} entre compás {tc.bar(n1.start)+1}")
        elif abs(interval) in (1, 2) and rng.random() < prob * 0.6:
            gap = n2.start - n1.end
            if gap >= tc.tpb // 4:
                neighbor = _snap_to_scale(n1.pitch + (1 if interval > 0 else -1), scale_notes)
                if neighbor != n2.pitch:
                    ndur = max(tc.tpb // 4, gap // 2)
                    insertions.append(Note(neighbor, n1.end, n1.end + ndur, min(n1.vel, 70), n1.channel))
                    changes.append(f"vecina superior/inferior en compás {tc.bar(n1.start)+1}")
    all_notes = sorted(notes + insertions, key=lambda n: (n.start, n.pitch))
    return all_notes, changes

# --- VARIACIÓN RÍTMICA ---
def _var_ritmica(records, tc, rng, prob_subdiv=0.3, prob_merge=0.2):
    changes = []
    notes = _records_to_notes(records)
    result = []
    i = 0
    while i < len(notes):
        n = notes[i]
        dur = n.end - n.start
        if dur >= tc.tpb and rng.random() < prob_subdiv:
            half = dur // 2
            result.append(Note(n.pitch, n.start, n.start + half, n.vel, n.channel))
            result.append(Note(n.pitch, n.start + half, n.end, max(n.vel - 10, 30), n.channel))
            changes.append(f"subdivisión en compás {tc.bar(n.start)+1}")
            i += 1
        elif i + 1 < len(notes) and notes[i+1].pitch == n.pitch and notes[i+1].start == n.end and rng.random() < prob_merge:
            nxt = notes[i+1]
            result.append(Note(n.pitch, n.start, nxt.end, n.vel, n.channel))
            changes.append(f"unión rítmica en compás {tc.bar(n.start)+1}")
            i += 2
        elif rng.random() < 0.15 and dur < tc.tpb:
            shift = tc.tpb // 4
            result.append(Note(n.pitch, n.start + shift, n.end + shift, n.vel, n.channel))
            changes.append(f"síncopa suave en compás {tc.bar(n.start)+1}")
            i += 1
        else:
            result.append(Note(n.pitch, n.start, n.end, n.vel, n.channel))
            i += 1
    result.sort(key=lambda n: (n.start, n.pitch))
    return result, changes

# --- VARIACIÓN DE TEXTURA ---
def _var_textura(records, tc, rng, mode="arpegio"):
    changes = []
    notes = _records_to_notes(records)
    lh_notes = [n for n in notes if n.pitch < 60]
    rh_notes = [n for n in notes if n.pitch >= 60]
    if mode == "arpegio":
        chords = _chords_at_onsets(lh_notes)
        new_lh = []
        for chord_notes in chords:
            if len(chord_notes) >= 2 and rng.random() < 0.6:
                sorted_cn = sorted(chord_notes, key=lambda n: n.pitch)
                start = min(n.start for n in sorted_cn)
                total_dur = max(n.end for n in sorted_cn) - start
                step = max(tc.tpb // 4, total_dur // max(len(sorted_cn), 1))
                for j, cn in enumerate(sorted_cn):
                    ns = start + j * step; ne = ns + step
                    new_lh.append(Note(cn.pitch, ns, min(ne, start + total_dur), cn.vel, cn.channel))
                changes.append(f"bloque→arpegio en compás {tc.bar(start)+1}")
            else:
                new_lh.extend(chord_notes)
        new_lh.sort(key=lambda n: (n.start, n.pitch))
        all_notes = sorted(rh_notes + new_lh, key=lambda n: (n.start, n.pitch))
    else:  # "bloque" - simplificar arpegios a bloques
        chords = _chords_at_onsets(lh_notes)
        new_lh = []
        for chord_notes in chords:
            if len(chord_notes) >= 2 and rng.random() < 0.5:
                start = min(n.start for n in chord_notes)
                end = max(n.end for n in chord_notes)
                for cn in chord_notes:
                    new_lh.append(Note(cn.pitch, start, end, cn.vel, cn.channel))
                changes.append(f"arpegio→bloque en compás {tc.bar(start)+1}")
            else:
                new_lh.extend(chord_notes)
        new_lh.sort(key=lambda n: (n.start, n.pitch))
        all_notes = sorted(rh_notes + new_lh, key=lambda n: (n.start, n.pitch))
    return all_notes, changes

# --- VARIACIÓN DE REGISTRO ---
def _var_registro(records, tc, rng, direction=1):
    changes = []
    notes = _records_to_notes(records)
    mel = [n for n in notes if n.pitch >= 60]
    other = [n for n in notes if n.pitch < 60]
    if direction == 1:
        new_mel = [Note(n.pitch + 12, n.start, n.end, n.vel, n.channel) for n in mel]
        changes.append(f"melodía +1 octava ({len(mel)} notas)")
    else:
        new_mel = [Note(max(n.pitch - 12, 21), n.start, n.end, n.vel, n.channel) for n in mel]
        changes.append(f"melodía -1 octava ({len(mel)} notas)")
    all_notes = sorted(other + new_mel, key=lambda n: (n.start, n.pitch))
    return all_notes, changes

# --- VARIACIÓN ARMÓNICA ---
def _var_armonica(records, tc, tonic_pc, mode, spans, rng, prob=0.3):
    changes = []
    notes = _records_to_notes(records)
    # sustituciones diatónicas simples
    subs_maj = {0: 9, 5: 2, 7: 4}   # I→vi, IV→ii, V→iii (grados)
    subs_min = {0: 3, 3: 0, 7: 2}   # i→III, III→i, v→II
    subs = subs_maj if mode == "maj" else subs_min
    new_notes = list(notes)
    for sp in spans:
        if sp.root is None or rng.random() > prob: continue
        rel = (sp.root - tonic_pc) % 12
        if rel in subs:
            new_root_pc = (tonic_pc + subs[rel]) % 12
            shift = new_root_pc - sp.root
            if shift == 0: continue
            t0, t1 = int(sp.start_beat * tc.tpb), int(sp.end_beat * tc.tpb)
            for i, n in enumerate(new_notes):
                if t0 <= n.start < t1 and n.pitch < 60:
                    new_notes[i] = Note(n.pitch + shift, n.start, n.end, n.vel, n.channel)
            changes.append(f"sustitución {rel}→{subs[rel]} en compás {tc.bar(t0)+1}")
    new_notes.sort(key=lambda n: (n.start, n.pitch))
    return new_notes, changes

# --- VARIACIÓN DE MODO ---
def _var_modo(records, tc, tonic_pc, mode, rng):
    changes = []
    notes = _records_to_notes(records)
    new_mode = "min" if mode == "maj" else "maj"
    old_scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
    new_scale = MINOR_SCALE if new_mode == "maj" else MAJOR_SCALE
    # mapeo de grados: cada nota se mueve al grado más cercano del nuevo modo
    degree_map = {}
    for i, old_iv in enumerate(old_scale):
        best_j, best_d = 0, 99
        for j, new_iv in enumerate(new_scale):
            d = abs(new_iv - old_iv)
            if d < best_d: best_j, best_d = j, d
        degree_map[old_iv] = new_scale[best_j]
    new_notes = []
    n_changed = 0
    for n in notes:
        rel = (n.pitch - tonic_pc) % 12
        if rel in degree_map and degree_map[rel] != rel:
            new_pitch = n.pitch - rel + degree_map[rel]
            new_notes.append(Note(new_pitch, n.start, n.end, n.vel, n.channel))
            n_changed += 1
        else:
            new_notes.append(Note(n.pitch, n.start, n.end, n.vel, n.channel))
    changes.append(f"cambio de modo {mode}→{new_mode}: {n_changed} notas ajustadas")
    new_notes.sort(key=lambda n: (n.start, n.pitch))
    return new_notes, changes, new_mode

# --- VARIACIÓN DE ARTICULACIÓN ---
def _var_articulacion(records, tc, rng, style="staccato"):
    changes = []
    notes = _records_to_notes(records)
    if style == "staccato":
        new_notes = []
        for n in notes:
            dur = n.end - n.start
            new_dur = max(dur // 2, tc.tpb // 8)
            new_notes.append(Note(n.pitch, n.start, n.start + new_dur, n.vel, n.channel))
        changes.append(f"staccato global: {len(notes)} notas acortadas al 50%")
    elif style == "legato":
        new_notes = []
        sorted_notes = sorted(notes, key=lambda n: n.start)
        for i, n in enumerate(sorted_notes):
            if i + 1 < len(sorted_notes) and sorted_notes[i+1].start > n.start:
                new_end = min(n.end, sorted_notes[i+1].start)
            else:
                new_end = n.end
            new_notes.append(Note(n.pitch, n.start, new_end, n.vel, n.channel))
        changes.append(f"legato global: {len(notes)} notas extendidas")
    else:  # "dinamica"
        sorted_notes = sorted(notes, key=lambda n: n.start)
        total = max(1, len(sorted_notes))
        new_notes = []
        for i, n in enumerate(sorted_notes):
            progress = i / total
            # crescendo → diminuendo (arco)
            vel_factor = 0.6 + 0.8 * math.sin(progress * math.pi)
            new_vel = int(np.clip(n.vel * vel_factor, 30, 127))
            new_notes.append(Note(n.pitch, n.start, n.end, new_vel, n.channel))
        changes.append(f"arco dinámico (crescendo-diminuendo) sobre {len(notes)} notas")
    return new_notes, changes

# --- GENERADOR PRINCIPAL DE VARIACIONES ---
def generate_all_variations(records, tc, mid, tonic_pc, mode, spans, target_grade,
                            n_variations=2, types=None, seed=42):
    """Genera todas las variaciones. Devuelve lista de VariationResult."""
    types = types or ALL_TYPES
    results = []
    base_notes = _records_to_notes(records)
    base_grade, _ = whole_piece_grade(
        [n for n in base_notes if n.pitch >= 60],
        [n for n in base_notes if n.pitch < 60], tc)
    for vtype in types:
        for k in range(n_variations):
            rng = random.Random(seed + hash(vtype) + k)
            try:
                if vtype == "ornamental":
                    notes, changes = _var_ornamental(records, tc, tonic_pc, mode, spans, rng)
                elif vtype == "ritmica":
                    notes, changes = _var_ritmica(records, tc, rng)
                elif vtype == "textura":
                    sub_mode = "arpegio" if k % 2 == 0 else "bloque"
                    notes, changes = _var_textura(records, tc, rng, mode=sub_mode)
                elif vtype == "registro":
                    direction = 1 if k % 2 == 0 else -1
                    notes, changes = _var_registro(records, tc, rng, direction)
                elif vtype == "armonica":
                    notes, changes = _var_armonica(records, tc, tonic_pc, mode, spans, rng)
                elif vtype == "modo":
                    notes, changes, new_mode = _var_modo(records, tc, tonic_pc, mode, rng)
                elif vtype == "articulacion":
                    styles = ["staccato", "legato", "dinamica"]
                    style = styles[k % len(styles)]
                    notes, changes = _var_articulacion(records, tc, rng, style=style)
                else:
                    continue
                if not notes: continue
                grade_after, _ = whole_piece_grade(
                    [n for n in notes if n.pitch >= 60],
                    [n for n in notes if n.pitch < 60], tc)
                # permitir ±1 grado
                if abs(grade_after - target_grade) > 1:
                    grade_note = f" (descartada: grado {grade_after} fuera de rango ±1)"
                    results.append(VariationResult(vtype, [], changes + [grade_note], base_grade, grade_after, seed))
                    continue
                new_records = _notes_to_records(notes, tc, records)
                results.append(VariationResult(vtype, new_records, changes, base_grade, grade_after, seed))
            except Exception as e:
                results.append(VariationResult(vtype, [], [f"error: {e}"], base_grade, -1, seed))
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  [6] EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════
def _conductor_track(tempo_map, timesig_map):
    trk = MidiTrackData(name="conductor")
    name_b = "conductor".encode("latin-1","replace")
    trk.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x03,
                    data=bytes([0xFF,0x03])+_write_vlq(len(name_b))+name_b))
    for tick, num, den in (timesig_map or [(0,4,4)]):
        dd = int(round(math.log2(max(1,den))))
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x58,
                        data=bytes([0xFF,0x58])+_write_vlq(4)+bytes([num,dd,24,8])))
    for tick, us in (tempo_map or [(0,500000)]):
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x51,
                        data=bytes([0xFF,0x51])+_write_vlq(3)+us.to_bytes(3,"big")))
    trk.events.sort(key=lambda e: e.abs)
    return trk

def export_variation_midi(records, tpb, out_path, tempo_map, timesig_map):
    trk_rh = MidiTrackData(name="MD"); trk_lh = MidiTrackData(name="MI")
    for r in records:
        trk = trk_rh if r.hand == "rh" else trk_lh
        n = r.note
        trk.events.append(MidiEvent(abs=n.start, kind="note_on", channel=0 if r.hand=="rh" else 1, pitch=n.pitch, vel=n.vel))
        trk.events.append(MidiEvent(abs=n.end, kind="note_off", channel=0 if r.hand=="rh" else 1, pitch=n.pitch, vel=0))
    conductor = _conductor_track(tempo_map, timesig_map)
    mid = MidiData(fmt=1, tpb=tpb, tracks=[conductor, trk_rh, trk_lh],
                   tempo_map=list(tempo_map) or [(0,500000)],
                   timesig_map=list(timesig_map) or [(0,4,4)])
    write_midi(mid, out_path)

# ══════════════════════════════════════════════════════════════════════════════
#  [7] INFORME POR CONSOLA
# ══════════════════════════════════════════════════════════════════════════════
def print_variation_report(midi_path, target_grade, base_grade, results, outdir, stem):
    print(f"\n{'═'*78}")
    print(f"  GRADED VARIATIONS v{VERSION} — {midi_path}")
    print(f"{'═'*78}")
    print(f"  Grado objetivo: {target_grade}/8")
    print(f"  Versión simplificada (base): grado {base_grade}/8")
    print(f"  Fichero base: {outdir}/{stem}_grade{target_grade}_base.mid")
    print(f"{'─'*78}")
    n_ok, n_skip = 0, 0
    for vr in results:
        if not vr.records:
            n_skip += 1
            print(f"\n  {_c('RED')}✗{ _c('R')} {vr.vtype}: descartada (grado {vr.grade_after})")
            for ch in vr.changes: print(f"    {_c('G')}·{ _c('R')} {ch}")
            continue
        n_ok += 1
        delta = vr.grade_after - vr.grade_before
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"\n  {_c('GRN')}✓{_c('R')} {vr.vtype}: grado {vr.grade_before}→{vr.grade_after} ({delta_str})")
        print(f"    Cambios ({len(vr.changes)}):")
        for ch in vr.changes[:12]:
            print(f"    {_c('G')}·{_c('R')} {ch}")
        if len(vr.changes) > 12:
            print(f"    {_c('G')}... y {len(vr.changes)-12} más{_c('R')}")
    print(f"\n{'─'*78}")
    print(f"  Total: {n_ok} variaciones generadas, {n_skip} descartadas")
    print(f"{'═'*78}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  [8] CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        prog="graded_variations.py",
        description="Genera variaciones musicales de una pieza simplificada a un grado pedagógico.")
    ap.add_argument("midi")
    ap.add_argument("--grade", type=int, required=True, help="Grado objetivo (1-8)")
    ap.add_argument("--variations", type=int, default=2, help="Variaciones por tipo (def. 2)")
    ap.add_argument("--types", nargs="+", choices=ALL_TYPES, default=None,
                    help="Tipos de variación (def. todos)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=int, default=60)
    ap.add_argument("--outdir")
    args = ap.parse_args()
    if not (1 <= args.grade <= 8):
        print(f"[ERROR] --grade debe estar entre 1 y 8 (recibido: {args.grade})", file=sys.stderr)
        return 1
    # simplificar al grado objetivo
    records, tc, mid, tonic_pc, mode, spans, rh, lh, achieved = simplify_to_grade(
        args.midi, args.grade, split=args.split)
    outdir = Path(args.outdir) if args.outdir else Path(args.midi).parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.midi).stem
    # exportar versión simplificada limpia
    base_path = outdir / f"{stem}_grade{args.grade}_base.mid"
    export_variation_midi(records, mid.tpb, str(base_path), mid.tempo_map, mid.timesig_map)
    # generar variaciones
    results = generate_all_variations(records, tc, mid, tonic_pc, mode, spans,
                                      args.grade, args.variations, args.types, args.seed)
    # exportar variaciones válidas
    for vr in results:
        if not vr.records: continue
        idx = sum(1 for r in results if r.vtype == vr.vtype and r.records and
                  results.index(r) <= results.index(vr))
        out_path = outdir / f"{stem}_grade{args.grade}_{vr.vtype}_{idx}.mid"
        export_variation_midi(vr.records, mid.tpb, str(out_path), mid.tempo_map, mid.timesig_map)
    # informe
    base_grade, _ = whole_piece_grade(rh, lh, tc)
    print_variation_report(args.midi, args.grade, base_grade, results, outdir, stem)
    return 0

if __name__ == "__main__":
    sys.exit(main())
