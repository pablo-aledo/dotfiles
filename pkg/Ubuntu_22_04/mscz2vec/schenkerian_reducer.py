#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCHENKERIAN REDUCER  v1.0                                 ║
║        Reducción analítica inversa: de la superficie al Ursatz              ║
║                                                                              ║
║  El ecosistema ya tiene elaboración schenkeriana como motor GENERATIVO       ║
║  (melody_conditioned.py --engine grammar: parte de un esqueleto de acordes  ║
║  y añade capas de ornamentación). Esta herramienta hace el proceso           ║
║  INVERSO: dado un MIDI ya compuesto (superficie / Vordergrund), produce      ║
║  automáticamente sus reducciones progresivas hasta llegar a una estructura   ║
║  fundamental candidata (Ursatz) — una línea melódica descendente por grados ║
║  conjuntos (Urlinie, 3̂/5̂/8̂ → 1̂) sobre un bajo I-V-I (Bassbrechung).         ║
║                                                                              ║
║  HONESTIDAD EPISTEMOLÓGICA (leer antes de confiar en la salida):             ║
║  El análisis schenkeriano tradicional NO es un algoritmo determinista —      ║
║  distintos analistas humanos proponen reducciones distintas y razonables    ║
║  de la misma pieza, y Schenker nunca formalizó reglas computables            ║
║  completas. Esta herramienta NO pretende dar "la verdad": genera N          ║
║  candidatos puntuados por un score de bondad de forma y muestra el mejor    ║
║  junto con las alternativas, igual que melody_conditioned.py --engine       ║
║  genetic puntúa una población por fitness. El respaldo teórico-             ║
║  computacional real usado aquí no es el Schenker informal, sino su          ║
║  formalización más tratable: la Generative Theory of Tonal Music            ║
║  (Lerdahl & Jackendoff, 1983) — time-span reduction y prolongational        ║
║  reduction mediante reglas de preferencia explícitas y puntuables (peso     ║
║  métrico, proximidad armónica, registro). Inspiración adicional: los        ║
║  trabajos de Alan Marsden sobre análisis schenkeriano computacional         ║
║  (búsqueda sobre reducciones puntuadas por reglas de prolongación).         ║
║                                                                              ║
║  PIPELINE:                                                                   ║
║  [1] PREPROCESADO   — segmentación métrica + armónica, extracción de voces ║
║  [2] CLASIFICACIÓN  — cada nota: CT/PT/NT/APP/SUS/ANT/ESC/UNCLASSIFIED      ║
║  [3] PESO           — peso estructural por nota (armonía+métrica+duración) ║
║  [4] NIVEL 1        — elimina ornamentación local de superficie             ║
║  [5] ÁRBOL          — árbol de prolongación (time-span tree) por acorde     ║
║  [6] NIVELES 2..N   — colapso recursivo de subárboles de bajo peso          ║
║  [7] URSATZ         — 3 hipótesis de Urlinie (3̂/5̂/8̂) puntuadas             ║
║                                                                              ║
║  USO:                                                                        ║
║    python schenkerian_reducer.py obra.mid                                    ║
║    python schenkerian_reducer.py obra.mid --key Cmaj --window 1              ║
║    python schenkerian_reducer.py obra.mid --max-levels 4 --threshold 0.35   ║
║    python schenkerian_reducer.py obra.mid --export-levels reduced/ \\        ║
║           --json obra.schenker.json                                         ║
║    python schenkerian_reducer.py obra.mid --candidates 3 --explain          ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi                  MIDI de entrada                                     ║
║    --key KEY              Fuerza tonalidad (p.ej. Cmaj, Am), si no se        ║
║                           detecta bien automáticamente                       ║
║    --window BEATS         Resolución del análisis armónico (default: 1      ║
║                           compás)                                            ║
║    --max-levels N         Tope de iteraciones de reducción (default: 5)     ║
║    --threshold F          Umbral de structural_weight para nivel 1          ║
║                           (default: 0.35)                                    ║
║    --weights-config FILE  JSON opcional con coeficientes de §4.3            ║
║    --candidates N         Nº de hipótesis de Ursatz a devolver (default: 3) ║
║    --export-levels DIR    Escribe un .mid por nivel de reducción            ║
║    --json FILE            Sidecar JSON con el análisis completo             ║
║    --explain              Explicación en lenguaje llano de cada colapso     ║
║    --no-color             Desactivar colores ANSI                            ║
║                                                                              ║
║  COMO MÓDULO:                                                               ║
║    from schenkerian_reducer import reduce_piece                             ║
║    levels, ursatz_candidates = reduce_piece(midi_path, **options)           ║
║                                                                              ║
║  LIMITACIONES CONOCIDAS (v1):                                               ║
║    · Asume tonalidad funcional razonablemente clara; música muy cromática  ║
║      o post-tonal produce numerales/Ursatz poco fiables.                    ║
║    · Si UNCLASSIFIED supera ~30% de las notas, el informe lo advierte en    ║
║      vez de ofrecer una reducción con falsa confianza (texturas densas o    ║
║      muy contrapuntísticas caen aquí con frecuencia).                       ║
║    · El repertorio de Ursatz se limita a las 3 formas canónicas simples     ║
║      (3̂/5̂/8̂); variantes con interrupción, mixtura, etc. quedan fuera.       ║
║    · El "bajo" se deriva por skyline (nota más grave sonando en cada        ║
║      instante); en piezas monofónicas coincide con la melodía y la          ║
║      confianza de Bassbrechung se apoya entonces solo en la secuencia       ║
║      armónica de números romanos, no en una voz de bajo real.               ║
║                                                                              ║
║  DEPENDENCIAS: numpy (stdlib para todo lo demás — E/S MIDI, teoría armónica ║
║  y peso métrico están reimplementados aquí de forma autocontenida; NO       ║
║  importa playability_auditor.py / harmonic_analyzer.py / metric_analyzer.py ║
║  ni ninguna otra herramienta del ecosistema como dependencia final, aunque  ║
║  reutiliza sus ideas de diseño).                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal

import numpy as np

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


# ══════════════════════════════════════════════════════════════════════════════
#  [0] E/S MIDI AUTOCONTENIDA (SMF 0/1, solo stdlib — sin dependencias del
#      ecosistema; misma idea de diseño que playability_auditor.py pero
#      reimplementada aquí para que este fichero sea independiente)
# ══════════════════════════════════════════════════════════════════════════════

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
                ev = MidiEvent(abs=t, kind="meta", meta_type=mtype,
                               data=bytes([0xFF, mtype]) + _write_vlq(mlen) + mdata)
                if mtype == 0x03 and not trk.name:
                    trk.name = mdata.decode("latin-1", errors="replace").strip()
                elif mtype == 0x51 and mlen == 3:
                    mid.tempo_map.append((t, int.from_bytes(mdata, "big")))
                elif mtype == 0x58 and mlen >= 2:
                    mid.timesig_map.append((t, mdata[0], 2 ** mdata[1]))
                if mtype != 0x2F:
                    trk.events.append(ev)
            elif status in (0xF0, 0xF7):
                slen, j = _read_vlq(chunk, j)
                sdata = chunk[j:j + slen]; j += slen
                trk.events.append(MidiEvent(abs=t, kind="sysex",
                                            data=bytes([status]) + _write_vlq(slen) + sdata))
            else:
                hi, ch = status & 0xF0, status & 0x0F
                if hi in (0xC0, 0xD0):
                    d1 = chunk[j]; j += 1
                    trk.events.append(MidiEvent(abs=t, kind="channel", channel=ch,
                                                data=bytes([status, d1])))
                else:
                    d1, d2 = chunk[j], chunk[j + 1]; j += 2
                    if hi == 0x90 and d2 > 0:
                        trk.events.append(MidiEvent(abs=t, kind="note_on", channel=ch,
                                                    pitch=d1, vel=d2))
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        trk.events.append(MidiEvent(abs=t, kind="note_off", channel=ch,
                                                    pitch=d1, vel=d2))
                    else:
                        trk.events.append(MidiEvent(abs=t, kind="channel", channel=ch,
                                                    data=bytes([status, d1, d2])))
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
            else:
                body += ev.data
        body += _write_vlq(0) + bytes([0xFF, 0x2F, 0x00])
        chunks.append(b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))
    header = (b"MThd" + (6).to_bytes(4, "big") + mid.fmt.to_bytes(2, "big")
              + len(mid.tracks).to_bytes(2, "big") + mid.tpb.to_bytes(2, "big"))
    Path(path).write_bytes(header + b"".join(chunks))


class TimeContext:
    """Conversión ticks <-> beats/compases (versión ligera, solo lo necesario aquí)."""

    def __init__(self, mid: MidiData):
        self.tpb = mid.tpb
        self.timesig_map = mid.timesig_map
        self._bars = []
        bar, prev_tick = 1.0, 0
        prev_tpc = self.tpb * 4 * self.timesig_map[0][1] // self.timesig_map[0][2]
        for tick, num, den in self.timesig_map:
            bar += (tick - prev_tick) / prev_tpc
            tpc = max(1, self.tpb * 4 * num // den)
            self._bars.append((tick, bar, tpc))
            prev_tick, prev_tpc = tick, tpc
        if not self._bars or self._bars[0][0] > 0:
            self._bars.insert(0, (0, 1.0, prev_tpc))

    def beat(self, tick: int) -> float:
        return tick / self.tpb

    def bar(self, tick: int) -> int:
        seg = self._bars[0]
        for s in self._bars:
            if s[0] <= tick:
                seg = s
            else:
                break
        t0, bar0, tpc = seg
        return int(bar0 + (tick - t0) / tpc)

    def beat_in_bar(self, tick: int, num: int) -> float:
        """Posición dentro del compás en beats (0 = tiempo 1)."""
        seg = self._bars[0]
        for s in self._bars:
            if s[0] <= tick:
                seg = s
            else:
                break
        t0, _, tpc = seg
        beats_per_bar = tpc / self.tpb
        offset_ticks = (tick - t0) % tpc
        return (offset_ticks / tpc) * beats_per_bar


@dataclass
class Note:
    pitch: int
    start: int      # ticks
    end: int        # ticks
    vel: int = 90
    channel: int = 0

    def start_beat(self, tc: TimeContext) -> float:
        return tc.beat(self.start)

    def end_beat(self, tc: TimeContext) -> float:
        return tc.beat(self.end)

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
        raise ValueError("el MIDI no contiene notas")
    notes.sort(key=lambda n: (n.start, n.pitch))
    return mid, tc, notes


# ══════════════════════════════════════════════════════════════════════════════
#  [1] ARMONÍA (misma idea de diseño que harmonic_analyzer.py, reimplementada
#      de forma autocontenida)
# ══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    "": {0, 4, 7}, "m": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8},
    "maj7": {0, 4, 7, 11}, "m7": {0, 3, 7, 10}, "7": {0, 4, 7, 10},
    "m7b5": {0, 3, 6, 10}, "dim7": {0, 3, 6, 9},
}
_QUAL_TRIAD = {"": "maj", "maj7": "maj", "7": "maj",
               "m": "min", "m7": "min", "aug": "aug",
               "dim": "dim", "dim7": "dim", "m7b5": "dim"}

_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                    5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                    4.75, 3.98, 2.69, 3.34, 3.17])

_MAJ_DEGREES = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
_MIN_DEGREES = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]      # natural minor (aproximación razonable para v1)


def detect_chord(pcs_weight: Dict[int, float], bass_pc: Optional[int]) -> Tuple[Optional[int], str, float]:
    present = {pc for pc, w in pcs_weight.items() if w > 0}
    if not present:
        return None, "", 0.0
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
    return best


def detect_key(pc_hist: np.ndarray) -> Tuple[int, str]:
    best = (0, "maj", -1e9)
    for tonic in range(12):
        maj_prof = np.roll(_KS_MAJ, tonic)
        min_prof = np.roll(_KS_MIN, tonic)
        cm = float(np.corrcoef(pc_hist, maj_prof)[0, 1]) if pc_hist.sum() else 0.0
        cn = float(np.corrcoef(pc_hist, min_prof)[0, 1]) if pc_hist.sum() else 0.0
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
    if not name:
        raise ValueError(f"tonalidad inválida: {key}")
    letter = name[0].upper()
    if letter not in _PC:
        raise ValueError(f"tonalidad inválida: {key}")
    pc = _PC[letter]
    rest = name[1:]
    if rest.startswith("#"):
        pc += 1
    elif rest.lower().startswith("b"):
        pc -= 1
    return pc % 12, mode


def roman_of(root_pc: int, suffix: str, tonic_pc: int, mode: str) -> Tuple[str, str]:
    rel = (root_pc - tonic_pc) % 12
    degrees = _MAJ_DEGREES if mode == "maj" else _MIN_DEGREES
    triad_q = _QUAL_TRIAD.get(suffix, "maj")
    base = degrees.get(rel)
    if base is None:
        # acorde alterado/cromático: aproximar al grado más cercano
        near = min(degrees.keys(), key=lambda d: min((rel - d) % 12, (d - rel) % 12))
        base = degrees[near] + "(alt)"
    numeral = base if triad_q in ("maj", "aug") else base.lower()
    if triad_q == "dim":
        numeral = base.lower().rstrip("°") + "°"
    is_seventh = suffix in ("7", "maj7", "m7", "dim7", "m7b5")
    if is_seventh:
        numeral += "7"
    func = {0: "T", 2: "S", 4: "T", 5: "S", 7: "D", 9: "T", 11: "D"}.get(rel, "-")
    return numeral, func


@dataclass
class ChordSpan:
    start_beat: float
    end_beat: float
    root: Optional[int]     # pitch class 0-11 (None si sin armonía clara)
    quality: str
    roman_numeral: str
    function: str            # "T" | "S" | "D" | "-"
    bass_pc: Optional[int] = None


def chord_tone_pcs(chord: Optional[ChordSpan]) -> set:
    if chord is None or chord.root is None:
        return set()
    templ = CHORD_TEMPLATES.get(chord.quality, {0, 4, 7})
    return {(chord.root + iv) % 12 for iv in templ}


def analyze_harmony_local(notes: List[Note], tc: TimeContext,
                           key: Optional[str] = None,
                           window_beats: Optional[float] = None) -> Tuple[int, str, List[ChordSpan]]:
    """Segmentación armónica autocontenida: acorde + numeral + función por ventana."""
    last_tick = max(n.end for n in notes)
    n_bars = max(1, tc.bar(last_tick - 1))

    pc_hist = np.zeros(12)
    for n in notes:
        pc_hist[n.pitch % 12] += (n.end - n.start)
    if key:
        tonic_pc, mode = _parse_key(key)
    else:
        tonic_pc, mode = detect_key(pc_hist)

    tpb = tc.tpb
    segments = []
    if window_beats:
        step = int(round(window_beats * tpb))
        t = 0
        while t < last_tick:
            segments.append((t, t + step))
            t += step
    else:
        for bar in range(1, n_bars + 1):
            t0 = _bar_start_tick(tc, bar)
            t1 = _bar_start_tick(tc, bar + 1)
            segments.append((t0, t1))

    spans: List[ChordSpan] = []
    for (t0, t1) in segments:
        seg_notes = [n for n in notes if n.start < t1 and n.end > t0]
        b0, b1 = tc.beat(t0), tc.beat(t1)
        if not seg_notes:
            spans.append(ChordSpan(b0, b1, None, "", "-", "-", None))
            continue
        pcs_weight: Dict[int, float] = {}
        for n in seg_notes:
            dur = min(n.end, t1) - max(n.start, t0)
            pcs_weight[n.pitch % 12] = pcs_weight.get(n.pitch % 12, 0) + dur
        bass_pc = min(seg_notes, key=lambda n: n.pitch).pitch % 12
        root, suffix, score = detect_chord(pcs_weight, bass_pc)
        if root is None:
            spans.append(ChordSpan(b0, b1, None, "", "-", "-", bass_pc))
            continue
        numeral, func = roman_of(root, suffix, tonic_pc, mode)
        spans.append(ChordSpan(b0, b1, root, suffix, numeral, func, bass_pc))

    return tonic_pc, mode, spans


def _bar_start_tick(tc: TimeContext, bar: int) -> int:
    tick, step = 0, max(1, tc.tpb // 8)
    limit = 20_000_000
    while tick < limit:
        if tc.bar(tick) >= bar:
            return tick
        tick += step
    return limit


def chord_at(spans: List[ChordSpan], beat: float) -> Optional[ChordSpan]:
    for sp in spans:
        if sp.start_beat <= beat < sp.end_beat:
            return sp
    return spans[-1] if spans else None


def scale_degree(pitch: int, tonic_pc: int, mode: str) -> Optional[int]:
    scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
    rel = (pitch - tonic_pc) % 12
    if rel in scale:
        return scale.index(rel) + 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  [2] PESO MÉTRICO (misma idea de diseño que metric_analyzer.py, reimplementada
#      de forma autocontenida)
# ══════════════════════════════════════════════════════════════════════════════

def metric_weights(num: int, subdiv: int) -> np.ndarray:
    """Perfil de peso métrico Longuet-Higgins & Lee simplificado, tiempo 1 = máximo."""
    n = num * subdiv
    W = np.zeros(n)
    for i in range(n):
        level, k = 0, i
        while k % 2 == 0 and k != 0:
            k //= 2
            level += 1
        W[i] = level + 1
        if i == 0:
            W[i] = level + num + 2      # tiempo 1 siempre el más fuerte
    return W


def note_metric_weight(note: Note, tc: TimeContext, num: int, subdiv: int,
                        W: np.ndarray) -> float:
    """Peso métrico normalizado 0..1 de la posición de ataque de una nota."""
    beats_per_bar = num
    pos = tc.beat_in_bar(note.start, num)          # 0..beats_per_bar
    idx = int(round(pos * subdiv)) % (num * subdiv)
    w_max = float(W.max()) if W.size else 1.0
    return float(W[idx] / w_max)


# ══════════════════════════════════════════════════════════════════════════════
#  [3] EXTRACCIÓN DE VOZ (skyline monofónico de melodía y bajo a partir de una
#      textura potencialmente polifónica)
# ══════════════════════════════════════════════════════════════════════════════

def build_skyline(notes: List[Note]) -> Tuple[List[Note], List[Note]]:
    """Reduce una textura (posiblemente polifónica) a dos líneas monofónicas:
    la voz superior (skyline agudo) y la voz inferior (skyline grave).
    Si la pieza ya es monofónica, ambas líneas coinciden (limitación documentada
    en la cabecera del módulo: en ese caso Bassbrechung se apoya en la armonía)."""
    if not notes:
        return [], []
    bounds = sorted(set(n.start for n in notes) | set(n.end for n in notes))
    melody_segs: List[Tuple[int, Note]] = []   # (id() de la nota original, segmento)
    bass_segs: List[Tuple[int, Note]] = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        mid = (t0 + t1) / 2.0
        active = [n for n in notes if n.start <= mid < n.end]
        if not active:
            continue
        top = max(active, key=lambda n: n.pitch)
        bot = min(active, key=lambda n: n.pitch)
        melody_segs.append((id(top), Note(pitch=top.pitch, start=t0, end=t1, vel=top.vel)))
        bass_segs.append((id(bot), Note(pitch=bot.pitch, start=t0, end=t1, vel=bot.vel)))

    def _merge(segs: List[Tuple[int, Note]]) -> List[Note]:
        # solo fusiona fragmentos que provienen de LA MISMA nota original partida
        # por el barrido de fronteras temporales — nunca fusiona dos ataques
        # distintos (re-articulaciones), aunque compartan altura, porque eso
        # destruiría información necesaria para detectar suspensiones (§3.1).
        out: List[Note] = []
        out_ids: List[int] = []
        for note_id, s in segs:
            if out and out_ids[-1] == note_id and out[-1].end == s.start:
                out[-1].end = s.end
            else:
                out.append(Note(pitch=s.pitch, start=s.start, end=s.end, vel=s.vel))
                out_ids.append(note_id)
        return out

    return _merge(melody_segs), _merge(bass_segs)


# ══════════════════════════════════════════════════════════════════════════════
#  [4] CLASIFICACIÓN DE NOTAS NO-ARMÓNICAS  — §3.1 de la especificación
# ══════════════════════════════════════════════════════════════════════════════

NoteFunction = Literal["CT", "PT", "NT", "APP", "SUS", "ANT", "ESC", "UNCLASSIFIED"]

_STEP_MAX = 2      # semitonos: hasta un tono entero cuenta como "grado conjunto"


def classify_note(note: Note,
                   prev_note: Optional[Note],
                   next_note: Optional[Note],
                   chord_now: Optional[ChordSpan],
                   chord_prev: Optional[ChordSpan],
                   chord_next: Optional[ChordSpan],
                   metric_weight: float) -> NoteFunction:
    """Clasifica una nota de la superficie en una de las categorías de §3.1.
    UNCLASSIFIED es un resultado legítimo (no forzar cada nota a encajar en el
    vocabulario del s. XVIII); se trata con peso estructural intermedio, no se
    descarta automáticamente (ver structural_weight)."""
    pc = note.pitch % 12
    ct_now = chord_tone_pcs(chord_now)

    if pc in ct_now:
        return "CT"

    step_in = (note.pitch - prev_note.pitch) if prev_note is not None else None
    step_out = (next_note.pitch - note.pitch) if next_note is not None else None

    # SUSPENSIÓN: llega ligada/repetida desde el compás anterior, era CT bajo el
    # acorde previo, y resuelve descendiendo por grado conjunto (mirar chord_prev).
    if (prev_note is not None and note.pitch == prev_note.pitch and chord_prev is not None
            and pc in chord_tone_pcs(chord_prev) and step_out is not None
            and step_out < 0 and abs(step_out) <= _STEP_MAX):
        target_ct = ct_now | chord_tone_pcs(chord_next)
        if next_note is not None and (next_note.pitch % 12) in target_ct:
            return "SUS"

    # APPOGGIATURA: atacada por salto en tiempo fuerte, resuelve por grado
    # conjunto a un chord tone.
    if (step_in is not None and abs(step_in) > _STEP_MAX and metric_weight >= 0.5
            and step_out is not None and abs(step_out) <= _STEP_MAX):
        target_ct = ct_now | chord_tone_pcs(chord_next)
        if next_note is not None and (next_note.pitch % 12) in target_ct:
            return "APP"

    # ANTICIPACIÓN: en posición métrica débil, anticipa un chord tone del acorde
    # siguiente (misma altura que ese chord tone, no del actual).
    if (chord_next is not None and metric_weight < 0.5
            and pc in chord_tone_pcs(chord_next) and pc not in ct_now):
        return "ANT"

    # ESCAPADA: alcanzada por grado conjunto, abandonada por salto en dirección
    # contraria (lo opuesto de la appoggiatura en orden salto/paso).
    if (step_in is not None and step_out is not None
            and abs(step_in) <= _STEP_MAX and abs(step_out) > _STEP_MAX
            and step_in != 0 and (step_in > 0) != (step_out > 0)):
        return "ESC"

    # NOTA DE PASO / BORDADURA: entrada y salida por grado conjunto.
    if (step_in is not None and step_out is not None
            and 0 < abs(step_in) <= _STEP_MAX and 0 < abs(step_out) <= _STEP_MAX):
        if (step_in > 0) == (step_out > 0):
            return "PT"
        return "NT"

    return "UNCLASSIFIED"


# ══════════════════════════════════════════════════════════════════════════════
#  [5] PESO ESTRUCTURAL POR NOTA  — §4.3
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "CT": 1.0, "SUS": 0.7, "APP": 0.5, "ANT": 0.3,
    "PT": 0.25, "NT": 0.2, "ESC": 0.2, "UNCLASSIFIED": 0.4,
}


def load_weights_config(path: Optional[str]) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        w.update({k: float(v) for k, v in data.items()})
    return w


def structural_weight(function: NoteFunction, metric_weight: float,
                       duration_beats: float,
                       weights: Dict[str, float] = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    base = weights.get(function, 0.4)
    duration_factor = min(max(duration_beats, 0.0) / 1.0, 1.5) if duration_beats > 0 else 1.0
    duration_factor = max(duration_factor, 0.5)   # nunca castigar tanto una nota muy corta
    return base * (0.5 + 0.5 * metric_weight) * duration_factor


# ══════════════════════════════════════════════════════════════════════════════
#  [6] ESTRUCTURAS DE DATOS  — §5
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoredNote:
    """Nota + metadatos derivados durante el análisis (no forma parte del §5
    literal, pero se necesita para arrastrar función/peso entre etapas)."""
    note: Note
    function: NoteFunction
    weight: float
    beat: float


@dataclass
class ProlongationNode:
    note: Note
    function: NoteFunction
    weight: float
    children: List["ProlongationNode"] = field(default_factory=list)
    span: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ReductionLevel:
    level: int
    notes: List[Note]
    bass_notes: List[Note]
    tree: Optional[ProlongationNode]
    removed_from_previous: List[Note] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)      # función de cada nota en `notes`


@dataclass
class UrsatzCandidate:
    urlinie_degree_start: int          # 3, 5, u 8
    urlinie_notes: List[Note]
    bassbrechung_notes: List[Note]
    score: float
    bassbrechung_confidence: float


# ══════════════════════════════════════════════════════════════════════════════
#  [7] NIVEL 1 — elimina ornamentación local de superficie  (§4.4)
# ══════════════════════════════════════════════════════════════════════════════

def _score_voice(voice_notes: List[Note], tc: TimeContext, spans: List[ChordSpan],
                  num: int, subdiv: int, W: np.ndarray,
                  weights_cfg: Dict[str, float]) -> List[ScoredNote]:
    scored: List[ScoredNote] = []
    for i, n in enumerate(voice_notes):
        beat = n.start_beat(tc)
        chord_now = chord_at(spans, beat)
        chord_prev = chord_at(spans, voice_notes[i - 1].start_beat(tc)) if i > 0 else None
        chord_next = chord_at(spans, voice_notes[i + 1].start_beat(tc)) if i < len(voice_notes) - 1 else None
        mweight = note_metric_weight(n, tc, num, subdiv, W)
        func = classify_note(
            n,
            voice_notes[i - 1] if i > 0 else None,
            voice_notes[i + 1] if i < len(voice_notes) - 1 else None,
            chord_now, chord_prev, chord_next, mweight)
        w = structural_weight(func, mweight, n.duration_beats(tc), weights_cfg)
        scored.append(ScoredNote(note=n, function=func, weight=w, beat=beat))
    return scored


def reduce_level1(scored: List[ScoredNote], threshold: float) -> Tuple[List[ScoredNote], List[Note]]:
    """Aplica el umbral de peso estructural; extiende notas retenidas para
    cubrir el hueco dejado por las eliminadas (§4.4). APP/SUS: la nota de
    resolución (siguiente) absorbe la eliminación en vez de la anterior."""
    if not scored:
        return [], []
    n = len(scored)
    keep_mask = [s.weight >= threshold for s in scored]
    if not any(keep_mask):
        # nunca dejar una voz vacía: conservar el extremo de mayor peso
        best = max(range(n), key=lambda i: scored[i].weight)
        keep_mask[best] = True
        keep_mask[0] = True
        keep_mask[-1] = True

    kept_idx = [i for i in range(n) if keep_mask[i]]
    starts = {i: scored[i].note.start for i in kept_idx}
    ends = {i: scored[i].note.end for i in kept_idx}

    def _next_kept(j):
        return next((k for k in kept_idx if k > j), None)

    def _prev_kept(j):
        return next((k for k in reversed(kept_idx) if k < j), None)

    removed_notes: List[Note] = []
    for j in range(n):
        if keep_mask[j]:
            continue
        removed_notes.append(scored[j].note)
        pulls_backward = scored[j].function in ("APP", "SUS")
        if pulls_backward:
            nxt = _next_kept(j)
            if nxt is not None:
                starts[nxt] = min(starts[nxt], scored[j].note.start)
                continue
            prv = _prev_kept(j)
            if prv is not None:
                ends[prv] = max(ends[prv], scored[j].note.end)
        else:
            prv = _prev_kept(j)
            if prv is not None:
                ends[prv] = max(ends[prv], scored[j].note.end)
                continue
            nxt = _next_kept(j)
            if nxt is not None:
                starts[nxt] = min(starts[nxt], scored[j].note.start)

    result: List[ScoredNote] = []
    for i in kept_idx:
        s = scored[i]
        new_note = Note(pitch=s.note.pitch, start=starts[i], end=ends[i], vel=s.note.vel)
        result.append(ScoredNote(note=new_note, function=s.function, weight=s.weight, beat=s.beat))

    # evitar solapes introducidos por tirones hacia atrás (APP/SUS)
    for k in range(1, len(result)):
        if result[k].note.start < result[k - 1].note.end:
            result[k - 1].note.end = result[k].note.start

    return result, removed_notes


# ══════════════════════════════════════════════════════════════════════════════
#  [8] ÁRBOL DE PROLONGACIÓN Y REDUCCIÓN RECURSIVA  — §4.5 / §4.6
# ══════════════════════════════════════════════════════════════════════════════

def _is_leading_tone_resolution(prev: ScoredNote, cur: ScoredNote, tonic_pc: int, mode: str) -> bool:
    dp = scale_degree(prev.note.pitch, tonic_pc, mode)
    dc = scale_degree(cur.note.pitch, tonic_pc, mode)
    if dp != 7 or dc != 1:
        return False
    return abs(cur.note.pitch - prev.note.pitch) in (1, 2) and cur.note.pitch > prev.note.pitch


def build_prolongation_spans(scored: List[ScoredNote], spans: List[ChordSpan],
                              tc: TimeContext) -> List[List[int]]:
    """Agrupa índices de `scored` consecutivos que comparten el mismo acorde
    vigente en un span de prolongación."""
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_chord = None
    for i, s in enumerate(scored):
        ch = chord_at(spans, s.beat)
        key = (ch.root, ch.quality) if ch is not None else None
        if cur and key != cur_chord:
            groups.append(cur)
            cur = []
        cur.append(i)
        cur_chord = key
    if cur:
        groups.append(cur)
    return groups


def collapse_iteration(scored: List[ScoredNote], spans: List[ChordSpan], tc: TimeContext,
                        tonic_pc: int, mode: str) -> Tuple[List[ScoredNote], List[Note], bool]:
    """Una iteración de reducción de nivel superior (§4.6). Devuelve la nueva
    lista de notas, las eliminadas en esta pasada, y si hubo algún colapso."""
    groups = build_prolongation_spans(scored, spans, tc)
    protected = set()
    for i in range(1, len(scored)):
        if _is_leading_tone_resolution(scored[i - 1], scored[i], tonic_pc, mode):
            protected.add(i - 1)
            protected.add(i)

    new_scored: List[ScoredNote] = []
    removed: List[Note] = []
    changed = False
    for grp in groups:
        if len(grp) <= 1:
            new_scored.append(scored[grp[0]])
            continue
        if any(idx in protected for idx in grp):
            # no colapsar: regla dura (sensible sin resolver / disonancia protegida)
            for idx in grp:
                new_scored.append(scored[idx])
            continue
        head = max(grp, key=lambda idx: scored[idx].weight)
        span_start = min(scored[idx].note.start for idx in grp)
        span_end = max(scored[idx].note.end for idx in grp)
        head_note = Note(pitch=scored[head].note.pitch, start=span_start, end=span_end,
                          vel=scored[head].note.vel)
        new_scored.append(ScoredNote(note=head_note, function=scored[head].function,
                                     weight=scored[head].weight, beat=scored[head].beat))
        for idx in grp:
            if idx != head:
                removed.append(scored[idx].note)
        changed = True

    return new_scored, removed, changed


# ══════════════════════════════════════════════════════════════════════════════
#  [9] IDENTIFICACIÓN DEL URSATZ CANDIDATO  — §4.7
# ══════════════════════════════════════════════════════════════════════════════

URLINIE_DEGREES = {3: [3, 2, 1], 5: [5, 4, 3, 2, 1], 8: [1, 7, 6, 5, 4, 3, 2, 1]}


def _climax_beat(notes: List[Note], tc: TimeContext) -> Optional[float]:
    if not notes:
        return None
    top = max(notes, key=lambda n: (n.pitch, n.end - n.start))
    return top.start_beat(tc)


def bassbrechung_confidence(spans: List[ChordSpan]) -> float:
    """§4.7.1: el bajo debe terminar en tónica precedida de dominante."""
    seq = [s for s in spans if s.function != "-"]
    if not seq:
        return 0.0
    last = seq[-1]
    if last.function != "T" or not last.roman_numeral.upper().startswith("I"):
        return 0.15
    # buscar una D antes del final, cuanto más cerca mejor
    tail = seq[-6:] if len(seq) >= 6 else seq
    for k in range(len(tail) - 1, -1, -1):
        if tail[k].function == "D":
            proximity = (len(tail) - k) / len(tail)
            return min(1.0, max(0.55, 1.0 - 0.4 * proximity))
    return 0.3


def find_urlinie_candidate(degree_start: int, melody: List[Note], tc: TimeContext,
                            tonic_pc: int, mode: str,
                            climax_beat: Optional[float]) -> Optional[UrsatzCandidate]:
    """Busca la hipótesis de Urlinie recorriendo la melodía HACIA ATRÁS desde
    la última llegada a 1̂. Se ancla en la superficie (nivel 0) — no en el
    nivel más reducido — porque las reducciones de nivel alto colapsan
    precisamente las notas de paso intermedias (p.ej. 2̂) que la propia
    Urlinie necesita como eslabones; el "saltarse notas" del §4.7 se cuenta
    aquí sobre la superficie original, que es donde vive esa información."""
    wanted = URLINIE_DEGREES[degree_start]
    degrees = [scale_degree(n.pitch, tonic_pc, mode) for n in melody]

    tonic_idxs = [i for i, d in enumerate(degrees) if d == 1]
    if not tonic_idxs:
        return None
    end_idx = tonic_idxs[-1]        # última llegada a 1̂: ancla de la resolución final

    cursor = end_idx
    seq_idx = [end_idx]
    skips = 0
    ok = True
    for want_deg in reversed(wanted[:-1]):
        found = None
        j = cursor - 1
        while j >= 0:
            if degrees[j] == want_deg:
                found = j
                break
            j -= 1
        if found is None:
            ok = False
            break
        skips += (cursor - found - 1)
        seq_idx.append(found)
        cursor = found
    if not ok:
        return None
    seq_idx.reverse()

    span_notes = [melody[i] for i in seq_idx]
    total_span = max(1, seq_idx[-1] - seq_idx[0])
    skip_ratio = skips / total_span if total_span else 0.0

    climax_bonus = 0.0
    if climax_beat is not None:
        for n in span_notes:
            if abs(n.start_beat(tc) - climax_beat) <= 4.0:      # dentro de ~1 compás (4/4)
                climax_bonus = 1.0
                break

    # bonus si la resolución final está cerca del final real de la pieza
    reaches_end = end_idx >= len(melody) - max(1, len(melody) // 6)
    end_bonus = 1.0 if reaches_end else 0.5

    score = 0.55 * (1.0 - min(1.0, skip_ratio)) + 0.25 * end_bonus + 0.20 * climax_bonus

    return UrsatzCandidate(
        urlinie_degree_start=degree_start,
        urlinie_notes=span_notes,
        bassbrechung_notes=[],
        score=round(float(score), 4),
        bassbrechung_confidence=0.0,
    )


def identify_ursatz(levels: List[ReductionLevel], tc: TimeContext, tonic_pc: int, mode: str,
                     spans: List[ChordSpan], n_candidates: int) -> List[UrsatzCandidate]:
    surface_melody = levels[0].notes
    climax_beat = _climax_beat(levels[0].notes, tc)
    conf = bassbrechung_confidence(spans)

    cands: List[UrsatzCandidate] = []
    for deg in (3, 5, 8):
        cand = find_urlinie_candidate(deg, surface_melody, tc, tonic_pc, mode, climax_beat)
        if cand is not None:
            cand.bassbrechung_confidence = conf
            cand.score = round(cand.score * (0.7 + 0.3 * conf), 4)
            cands.append(cand)

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:n_candidates]


# ══════════════════════════════════════════════════════════════════════════════
#  [10] ORQUESTACIÓN DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def reduce_piece(midi_path: str,
                  key: Optional[str] = None,
                  window_beats: Optional[float] = None,
                  max_levels: int = 5,
                  threshold: float = 0.35,
                  weights_config: Optional[str] = None,
                  candidates: int = 3,
                  subdiv: int = 4) -> Tuple[List[ReductionLevel], List[UrsatzCandidate]]:
    """API pública. Devuelve (niveles_de_reduccion, hipotesis_de_ursatz)."""
    mid, tc, all_notes = load_notes(midi_path)
    tonic_pc, mode, spans = analyze_harmony_local(all_notes, tc, key=key, window_beats=window_beats)
    melody_notes, bass_notes = build_skyline(all_notes)
    weights_cfg = load_weights_config(weights_config)
    num = mid.timesig_map[0][1]
    W = metric_weights(num, subdiv)

    scored_mel = _score_voice(melody_notes, tc, spans, num, subdiv, W, weights_cfg)
    scored_bass = _score_voice(bass_notes, tc, spans, num, subdiv, W, weights_cfg)

    level0 = ReductionLevel(
        level=0, notes=[s.note for s in scored_mel], bass_notes=[s.note for s in scored_bass],
        tree=None, removed_from_previous=[], functions=[s.function for s in scored_mel])
    levels = [level0]

    mel1, removed1 = reduce_level1(scored_mel, threshold)
    bass1, _ = reduce_level1(scored_bass, threshold)
    levels.append(ReductionLevel(
        level=1, notes=[s.note for s in mel1], bass_notes=[s.note for s in bass1],
        tree=None, removed_from_previous=removed1, functions=[s.function for s in mel1]))

    cur = mel1
    cur_bass = bass1
    lvl = 1
    while lvl < max_levels:
        new_mel, removed, changed_m = collapse_iteration(cur, spans, tc, tonic_pc, mode)
        new_bass, _, changed_b = collapse_iteration(cur_bass, spans, tc, tonic_pc, mode)
        if not changed_m and not changed_b:
            break
        lvl += 1
        cur = new_mel if new_mel else cur
        cur_bass = new_bass if new_bass else cur_bass
        levels.append(ReductionLevel(
            level=lvl, notes=[s.note for s in cur], bass_notes=[s.note for s in cur_bass],
            tree=None, removed_from_previous=removed, functions=[s.function for s in cur]))

    ursatz = identify_ursatz(levels, tc, tonic_pc, mode, spans, candidates)
    return levels, ursatz


# ══════════════════════════════════════════════════════════════════════════════
#  [11] EXPORTACIÓN MIDI POR NIVEL
# ══════════════════════════════════════════════════════════════════════════════

def export_level_midi(level: ReductionLevel, tpb: int, out_path: str, program: int = 0):
    trk_mel = MidiTrackData(name="melodia reducida")
    trk_bass = MidiTrackData(name="bajo reducido")
    for n in level.notes:
        trk_mel.events.append(MidiEvent(abs=n.start, kind="note_on", channel=0, pitch=n.pitch, vel=90))
        trk_mel.events.append(MidiEvent(abs=n.end, kind="note_off", channel=0, pitch=n.pitch, vel=0))
    for n in level.bass_notes:
        trk_bass.events.append(MidiEvent(abs=n.start, kind="note_on", channel=1, pitch=n.pitch, vel=80))
        trk_bass.events.append(MidiEvent(abs=n.end, kind="note_off", channel=1, pitch=n.pitch, vel=0))
    mid = MidiData(fmt=1, tpb=tpb, tracks=[trk_mel, trk_bass],
                    tempo_map=[(0, 500000)], timesig_map=[(0, 4, 4)])
    write_midi(mid, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  [12] INFORME / JSON / CLI
# ══════════════════════════════════════════════════════════════════════════════

def _unclassified_ratio(level0: ReductionLevel) -> float:
    if not level0.functions:
        return 0.0
    return level0.functions.count("UNCLASSIFIED") / len(level0.functions)


def build_json_report(midi_path: str, tonic_pc: int, mode: str,
                       levels: List[ReductionLevel], ursatz: List[UrsatzCandidate],
                       tc: TimeContext) -> dict:
    return {
        "version": VERSION,
        "file": midi_path,
        "key": f"{pc_name(tonic_pc)}{'m' if mode == 'min' else 'maj'}",
        "unclassified_ratio": round(_unclassified_ratio(levels[0]), 4),
        "levels": [
            {
                "level": lv.level,
                "n_notes": len(lv.notes),
                "removed_count": len(lv.removed_from_previous),
                "notes": [
                    {"beat": round(n.start_beat(tc), 3), "pitch": n.pitch,
                     "duration_beats": round(n.duration_beats(tc), 3),
                     "function": (lv.functions[i] if i < len(lv.functions) else None)}
                    for i, n in enumerate(lv.notes)
                ],
                "bass_notes": [
                    {"beat": round(n.start_beat(tc), 3), "pitch": n.pitch,
                     "duration_beats": round(n.duration_beats(tc), 3)}
                    for n in lv.bass_notes
                ],
            }
            for lv in levels
        ],
        "ursatz_candidates": [
            {
                "degree_start": c.urlinie_degree_start,
                "score": c.score,
                "bassbrechung_confidence": round(c.bassbrechung_confidence, 3),
                "urlinie_notes": [
                    {"beat": round(n.start_beat(tc), 3), "pitch": n.pitch,
                     "degree": scale_degree(n.pitch, tonic_pc, mode)}
                    for n in c.urlinie_notes
                ],
            }
            for c in ursatz
        ],
    }


def print_report(midi_path: str, tonic_pc: int, mode: str, levels: List[ReductionLevel],
                  ursatz: List[UrsatzCandidate], explain: bool):
    B, R, G, C, YEL = _c("B"), _c("R"), _c("G"), _c("CYA"), _c("YEL")
    keyname = f"{pc_name(tonic_pc)}{'m' if mode == 'min' else 'maj'}"
    print(f"\n{'═' * 78}")
    print(f"  {B}SCHENKERIAN REDUCER v{VERSION}{R}  ·  {midi_path}")
    print(f"{'═' * 78}")
    print(f"  Tonalidad: {C}{keyname}{R}   niveles calculados: {len(levels)}")

    ur = _unclassified_ratio(levels[0])
    if ur > 0.30:
        print(f"  {YEL}[AVISO] {ur*100:.0f}% de las notas quedaron UNCLASSIFIED — "
              f"textura densa/contrapuntística; la reducción puede ser poco fiable.{R}")

    mel0_pitches = [n.pitch for n in levels[0].notes]
    bass0_pitches = [n.pitch for n in levels[0].bass_notes]
    if mel0_pitches == bass0_pitches:
        print(f"  {YEL}[AVISO] entrada monofónica: no hay una voz de bajo real distinta de "
              f"la melodía — la confianza de Bassbrechung se apoya solo en la secuencia "
              f"armónica de números romanos (ver limitaciones en la cabecera).{R}")

    print(f"\n  {G}nivel  notas  eliminadas{R}")
    for lv in levels:
        print(f"  {lv.level:>5}  {len(lv.notes):>5}  {len(lv.removed_from_previous):>10}")

    if len(levels) >= 2:
        n0 = len(levels[0].notes)
        nf = len(levels[-1].notes)
        pct = 100.0 * (1 - nf / n0) if n0 else 0.0
        print(f"\n  Reducción total voz superior: {n0} → {nf} notas ({pct:.0f}%)")

    print(f"\n  {B}Hipótesis de Ursatz (Urlinie / Bassbrechung):{R}")
    if not ursatz:
        print("  (no se encontró ninguna línea descendente 3̂/5̂/8̂ → 1̂ completa)")
    for c in ursatz:
        degs = "-".join(str(d) for d in URLINIE_DEGREES[c.urlinie_degree_start])
        pitches = " ".join(pc_name(n.pitch) for n in c.urlinie_notes)
        print(f"    {c.urlinie_degree_start}̂ ({degs}) score={c.score:.2f}  "
              f"confianza_bassbrechung={c.bassbrechung_confidence:.2f}   [{pitches}]")
        if explain:
            print(f"      {G}explicación: línea de {len(c.urlinie_notes)} notas identificada "
                  f"en la reducción final; puntuada por continuidad diatónica, cercanía al "
                  f"clímax registral y presencia de cadencia V→I.{R}")
    print(f"{'═' * 78}\n")


def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="schenkerian_reducer.py",
        description="Reducción analítica inversa hacia un Ursatz candidato (GTTM).")
    ap.add_argument("midi")
    ap.add_argument("--key")
    ap.add_argument("--window", type=float, dest="window_beats")
    ap.add_argument("--max-levels", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--weights-config")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--export-levels")
    ap.add_argument("--json")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    try:
        mid, tc, all_notes = load_notes(args.midi)
        tonic_pc, mode, spans = analyze_harmony_local(
            all_notes, tc, key=args.key, window_beats=args.window_beats)
        levels, ursatz = reduce_piece(
            args.midi, key=args.key, window_beats=args.window_beats,
            max_levels=args.max_levels, threshold=args.threshold,
            weights_config=args.weights_config, candidates=args.candidates)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print_report(args.midi, tonic_pc, mode, levels, ursatz, args.explain)

    if args.export_levels:
        outdir = Path(args.export_levels)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.midi).stem
        for lv in levels:
            label = "ursatz" if lv is levels[-1] else f"L{lv.level}"
            out_path = outdir / f"{stem}.{label}.mid"
            export_level_midi(lv, mid.tpb, str(out_path))
        print(f"  Niveles exportados en: {outdir}/")

    if args.json:
        report = build_json_report(args.midi, tonic_pc, mode, levels, ursatz, tc)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
