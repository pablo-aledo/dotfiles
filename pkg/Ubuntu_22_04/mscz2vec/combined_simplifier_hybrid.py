#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              COMBINED SIMPLIFIER HYBRID  v3.0  (autocontenido)               ║
║                                                                              ║
║  PROPOSITO:                                                                 ║
║   Fichero INDEPENDIENTE (no importa combined_simplifier.py ni ningun otro   ║
║   modulo del proyecto) que simplifica una partitura de piano en formato     ║
║   MIDI a distintos niveles de dificultad pedagogica (--target-grade 1..8,   ║
║   de principiante a virtuoso). Es la fusion de dos lineas de trabajo        ║
║   previas sobre el mismo problema:                                         ║
║                                                                              ║
║     · combined_simplifier.py (v1.1/v2.0): motor estructural (suelo         ║
║       schenkeriano + greedy por compas + rejilla de respaldo) y una capa   ║
║       de sustitucion armonica/melodica genuina (no elimina notas).         ║
║     · combined_simplifier_transformed.py (v1.2): un OPTIMIZADOR adaptativo ║
║       de transformaciones musicales (prueba varias tecnicas por pasada y   ║
║       solo acepta la que de verdad mejora el grado medido).                ║
║                                                                              ║
║   Este fichero se queda con lo mejor validado empiricamente de cada uno    ║
║   (ver informe de comparacion en la conversacion que lo origino):          ║
║     - la sustitucion armonica real de v2.0 (simplify_harmony/clarify_bass, ║
║       cambia pitches, NUNCA reduce el numero de notas),                    ║
║     - el bucle de aceptacion adaptativa de v1.2 (prueba-y-acepta-si-       ║
║       mejora, varias pasadas) en vez de la escalera fija de v2.0,          ║
║     - align_hands y merge_repeated_notes de v1.2 (huecos reales de v2.0),  ║
║     - trim_overlaps de v1.2 como red de seguridad tras cualquier tecnica   ║
║       que mueva onsets (huella de un riesgo real detectado en v2.0),       ║
║     - el hallazgo empirico de v2.0 de que, si las transformaciones no      ║
║       bastan, la rejilla de respaldo rinde mas arrancando desde el         ║
║       snapshot ESTRUCTURAL original y no desde el ya transformado.         ║
║   Deliberadamente NO incluye _simplify_chords de v1.2 (recorta voces por   ║
║   posicion): es una eliminacion de notas reempaquetada como "armonica",    ║
║   y esa funcion ya la cubre el adelgazado de voicings del motor            ║
║   estructural (--max-voices-per-chord), con criterio funcional en vez de   ║
║   posicional.                                                              ║
║                                                                              ║
║  TECNICAS (misma taxonomia de 6 categorias usada en todo el proyecto):     ║
║   [ESTRUCTURAL] suelo schenkeriano + greedy por compas (motor original).   ║
║   [REJILLA]     rejilla de respaldo, garantiza alcanzar cualquier grado.   ║
║   [ARMONICA]    simplify_harmony (sustituir extension por tono de acorde   ║
║                 mas cercano), clarify_bass (fundamental en cada cambio     ║
║                 armonico). Pre-pasada, uniforme para todos los grados,     ║
║                 SIEMPRE antes del greedy: nunca cambia el nº de notas.     ║
║   [MELODICA]    resolve_ornaments (funde adornos muy breves en su vecina), ║
║                 reduce_melodic_leaps (saltos grandes -> reubicacion de     ║
║                 octava).                                                   ║
║   [REGISTRO]    close_voicings (comprime voces internas de un bloque       ║
║                 ancho, sin tocar los extremos que definen el contorno).    ║
║   [TEXTURA]     arpeggiate_dense_chords, collapse_octave_doubles.          ║
║   [RITMICA]     merge_repeated_notes, quantize_light, align_hands.         ║
║   [ADITIVA]     add_connective_notes — candidata, no permanente: solo      ║
║                 sobrevive si de verdad mejora el grado medido.             ║
║                                                                              ║
║  OPTIMIZADOR: para cada --target-grade, las tecnicas de REGISTRO/         ║
║  TEXTURA/RITMICA/ADITIVA (la pre-pasada ARMONICA/MELODICA ya corrio antes  ║
║  del greedy) se aplican con un bucle de aceptacion adaptativa: en cada     ║
║  pasada se prueban todas sobre un clon del estado actual y se acepta solo  ║
║  la primera que reduce estrictamente el grado medido (whole_piece_grade);  ║
║  se repite hasta --max-transform-passes o hasta que ninguna mejora ya.     ║
║  Si al final no se alcanzo el target, se recurre a la rejilla de respaldo  ║
║  arrancando desde el snapshot ESTRUCTURAL original (no desde el ya         ║
║  transformado).                                                            ║
║                                                                              ║
║  USO:                                                                       ║
║    python combined_simplifier_hybrid.py obra.mid --target-grade 1 3 5 7    ║
║    python combined_simplifier_hybrid.py obra.mid --target-grade 2 \\        ║
║        --outdir out/ --max-transform-passes 12                             ║
║    python combined_simplifier_hybrid.py obra.mid --target-grade 4 \\        ║
║        --split 60 --no-additive                                            ║
║        (--split: pitch MIDI que separa mano derecha/izquierda, def. 60=C4) ║
║    python combined_simplifier_hybrid.py obra.mid --target-grade 6 \\        ║
║        --no-transforms                                                     ║
║        (desactiva TODA la capa hibrida: replica el motor estructural       ║
║        v1.1 puro, suelo+greedy+rejilla, sin sustituciones ni optimizador)  ║
║                                                                              ║
║  SALIDA: por cada --target-grade se genera <nombre>_grade<N>.mid en        ║
║  --outdir (o junto al fichero de entrada) y un informe por consola con el  ║
║  grado realmente alcanzado, el metodo usado y el porcentaje de notas       ║
║  conservadas.                                                              ║
║                                                                              ║
║  DEPENDENCIAS: solo numpy (stdlib para todo lo demas). Fichero unico y     ║
║  autocontenido: no importa combined_simplifier.py ni                       ║
║  combined_simplifier_transformed.py.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
#  [ESTRUCTURAL] MOTOR DE ANALISIS — E/S MIDI, armonia, clasificacion de notas, suelo
#  schenkeriano, formula de dificultad de 5 factores y greedy por compas.
# ══════════════════════════════════════════════════════════════════════════════
import sys
import json
import bisect
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Literal, Set

import numpy as np

VERSION = "3.0-hybrid"

# ── NOTA SOBRE EL COMPORTAMIENTO ESPERADO ─────────────────────────────────
# En piezas densas en acordes en bloque (poca ornamentacion de superficie,
# mucho tono del acorde estructuralmente relevante), es normal que el greedy
# estructural no consiga alcanzar --target-grade bajos: el suelo protege las
# notas de melodia/bajo estructuralmente centrales y, en ese tipo de piezas,
# apenas queda margen de reduccion antes de llegar a esos grados. Cuando eso
# ocurre entra en juego la rejilla de respaldo (ver seccion [REJILLA]), que
# si garantiza alcanzar cualquier grado. No es un fallo del algoritmo, sino
# el resultado esperable de que esta reduccion nunca inventa simplificaciones
# fuera del vocabulario de funciones tonales CT/PT/NT/APP/SUS/ANT/ESC.
#
# La identidad de cada nota se conserva a traves de los distintos niveles de
# reduccion mediante NoteRec.orig_id (indice estable en la lista de notas de
# superficie), ya que la extension de huecos puede desplazar ligeramente el
# tick de notas vecinas de forma distinta segun el nivel: comparar por
# (pitch, tick) no seria fiable para verificar que un grado mas exigente es
# subconjunto de uno menos exigente.
# ───────────────────────────────────────────────────────────────────────────

_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

LIMITATION_NOTICE = (
    "esta reduccion prioriza coherencia tonal (peso estructural GTTM/Schenker). "
    "No protege motivos ritmicos o gestos de superficie que se repiten pero no "
    "son estructuralmente centrales (ostinatos, sincopas caracteristicas, "
    "riffs). Si la pieza depende fuertemente de un patron reconocible, revisa "
    "manualmente que no se haya eliminado."
)

_LEVELS = ["principiante", "elemental", "elemental-medio", "intermedio",
           "intermedio-alto", "avanzado", "avanzado-alto", "virtuoso"]


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  [0] E/S MIDI AUTOCONTENIDA (lectura/escritura de MThd/MTrk sin librerias)
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
        raise ValueError(f"{path}: division SMPTE no soportada")
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
    """Conversion ticks <-> beats/compases/segundos, incluyendo .sec()
    tempo-aware (necesaria para calcular el peak_nps -notas por segundo
    pico- que usa la formula de dificultad)."""

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

        tempo_map = list(mid.tempo_map)
        if not tempo_map or tempo_map[0][0] != 0:
            tempo_map = [(0, 500000)] + tempo_map
        self._tempo_segs = []
        cum_sec = 0.0
        prev_t, prev_us = 0, tempo_map[0][1]
        for tick, us in tempo_map:
            if tick > prev_t:
                cum_sec += (tick - prev_t) * prev_us / 1_000_000.0 / self.tpb
            self._tempo_segs.append((tick, cum_sec, us))
            prev_t, prev_us = tick, us

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

    def sec(self, tick: int) -> float:
        seg = self._tempo_segs[0]
        for s in self._tempo_segs:
            if s[0] <= tick:
                seg = s
            else:
                break
        t0, sec0, us = seg
        return sec0 + (tick - t0) * us / 1_000_000.0 / self.tpb

    def bar_start_tick(self, bar: int) -> int:
        tick, step = 0, max(1, self.tpb // 8)
        limit = 20_000_000
        while tick < limit:
            if self.bar(tick) >= bar:
                return tick
            tick += step
        return limit

    def bar_range_ticks(self, bar: int) -> Tuple[int, int]:
        return self.bar_start_tick(bar), self.bar_start_tick(bar + 1)


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
#  [1] ARMONIA — solo lo necesario para classify_note / structural_weight /
#      el calculo del suelo estructural (no genera numerales romanos)
# ══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    "": {0, 4, 7}, "m": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8},
    "maj7": {0, 4, 7, 11}, "m7": {0, 3, 7, 10}, "7": {0, 4, 7, 10},
    "m7b5": {0, 3, 6, 10}, "dim7": {0, 3, 6, 9},
}

_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                    5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                    4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


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
        raise ValueError(f"tonalidad invalida: {key}")
    letter = name[0].upper()
    if letter not in _PC:
        raise ValueError(f"tonalidad invalida: {key}")
    pc = _PC[letter]
    rest = name[1:]
    if rest.startswith("#"):
        pc += 1
    elif rest.lower().startswith("b"):
        pc -= 1
    return pc % 12, mode


@dataclass
class ChordSpan:
    start_beat: float
    end_beat: float
    root: Optional[int]
    quality: str
    bass_pc: Optional[int] = None


def chord_tone_pcs(chord: Optional[ChordSpan]) -> set:
    if chord is None or chord.root is None:
        return set()
    templ = CHORD_TEMPLATES.get(chord.quality, {0, 4, 7})
    return {(chord.root + iv) % 12 for iv in templ}


def _bar_start_tick(tc: TimeContext, bar: int) -> int:
    return tc.bar_start_tick(bar)


def analyze_harmony_local(notes: List[Note], tc: TimeContext,
                           key: Optional[str] = None,
                           window_beats: Optional[float] = None) -> Tuple[int, str, List[ChordSpan]]:
    """Segmentacion armonica autocontenida: un acorde por ventana temporal
    (por defecto, un acorde por compas). No genera numerales romanos, solo
    lo necesario para clasificar notas y calcular el suelo estructural."""
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
            spans.append(ChordSpan(b0, b1, None, "", None))
            continue
        pcs_weight: Dict[int, float] = {}
        for n in seg_notes:
            dur = min(n.end, t1) - max(n.start, t0)
            pcs_weight[n.pitch % 12] = pcs_weight.get(n.pitch % 12, 0) + dur
        bass_pc = min(seg_notes, key=lambda n: n.pitch).pitch % 12
        root, suffix, score = detect_chord(pcs_weight, bass_pc)
        if root is None:
            spans.append(ChordSpan(b0, b1, None, "", bass_pc))
            continue
        spans.append(ChordSpan(b0, b1, root, suffix, bass_pc))

    return tonic_pc, mode, spans


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
#  [2] PESO METRICO — cuanto pesa cada posicion dentro del compas segun su
#      nivel de subdivision binaria (el primer tiempo pesa mas, etc.)
# ══════════════════════════════════════════════════════════════════════════════

def metric_weights(num: int, subdiv: int) -> np.ndarray:
    n = num * subdiv
    W = np.zeros(n)
    for i in range(n):
        level, k = 0, i
        while k % 2 == 0 and k != 0:
            k //= 2
            level += 1
        W[i] = level + 1
        if i == 0:
            W[i] = level + num + 2
    return W


def note_metric_weight(note: Note, tc: TimeContext, num: int, subdiv: int,
                        W: np.ndarray) -> float:
    pos = tc.beat_in_bar(note.start, num)
    idx = int(round(pos * subdiv)) % (num * subdiv)
    w_max = float(W.max()) if W.size else 1.0
    return float(W[idx] / w_max)


# ══════════════════════════════════════════════════════════════════════════════
#  [3] SKYLINE
# ══════════════════════════════════════════════════════════════════════════════

def build_skyline(notes: List[Note]) -> Tuple[List[Note], List[Note]]:
    """Reduce la textura a dos lineas monofonicas: voz superior (skyline
    agudo, la nota mas aguda sonando en cada instante) y voz inferior
    (skyline grave, la mas grave). Esta lente de 2 voces es la que se usa
    para clasificar la funcion tonal de cada nota (CT/PT/NT/...); el resto
    de la textura (voces internas) se clasifica despues con una regla mas
    simple basada solo en si la nota pertenece al acorde vigente (ver
    build_note_records)."""
    if not notes:
        return [], []
    bounds = sorted(set(n.start for n in notes) | set(n.end for n in notes))
    melody_segs: List[Tuple[int, Note]] = []
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
#  [4] CLASIFICACION DE NOTAS NO-ARMONICAS
#      Determina si cada nota de una voz melodica es tono del acorde (CT) o
#      una de las figuraciones no-armonicas clasicas: nota de paso (PT),
#      nota de bordado (NT), apoyatura (APP), suspension (SUS), anticipacion
#      (ANT) o escapada (ESC).
# ══════════════════════════════════════════════════════════════════════════════

NoteFunction = Literal["CT", "PT", "NT", "APP", "SUS", "ANT", "ESC", "UNCLASSIFIED"]

_STEP_MAX = 2


def classify_note(note: Note,
                   prev_note: Optional[Note],
                   next_note: Optional[Note],
                   chord_now: Optional[ChordSpan],
                   chord_prev: Optional[ChordSpan],
                   chord_next: Optional[ChordSpan],
                   metric_weight: float) -> NoteFunction:
    pc = note.pitch % 12
    ct_now = chord_tone_pcs(chord_now)

    if pc in ct_now:
        return "CT"

    step_in = (note.pitch - prev_note.pitch) if prev_note is not None else None
    step_out = (next_note.pitch - note.pitch) if next_note is not None else None

    if (prev_note is not None and note.pitch == prev_note.pitch and chord_prev is not None
            and pc in chord_tone_pcs(chord_prev) and step_out is not None
            and step_out < 0 and abs(step_out) <= _STEP_MAX):
        target_ct = ct_now | chord_tone_pcs(chord_next)
        if next_note is not None and (next_note.pitch % 12) in target_ct:
            return "SUS"

    if (step_in is not None and abs(step_in) > _STEP_MAX and metric_weight >= 0.5
            and step_out is not None and abs(step_out) <= _STEP_MAX):
        target_ct = ct_now | chord_tone_pcs(chord_next)
        if next_note is not None and (next_note.pitch % 12) in target_ct:
            return "APP"

    if (chord_next is not None and metric_weight < 0.5
            and pc in chord_tone_pcs(chord_next) and pc not in ct_now):
        return "ANT"

    if (step_in is not None and step_out is not None
            and abs(step_in) <= _STEP_MAX and abs(step_out) > _STEP_MAX
            and step_in != 0 and (step_in > 0) != (step_out > 0)):
        return "ESC"

    if (step_in is not None and step_out is not None
            and 0 < abs(step_in) <= _STEP_MAX and 0 < abs(step_out) <= _STEP_MAX):
        if (step_in > 0) == (step_out > 0):
            return "PT"
        return "NT"

    return "UNCLASSIFIED"


# ══════════════════════════════════════════════════════════════════════════════
#  [5] PESO ESTRUCTURAL — combina funcion tonal, peso metrico y duracion en
#      un unico numero que mide cuan importante es una nota para conservar
#      la identidad de la pieza
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "CT": 1.0, "SUS": 0.7, "APP": 0.5, "ANT": 0.3,
    "PT": 0.25, "NT": 0.2, "ESC": 0.2, "UNCLASSIFIED": 0.4,
}


def structural_weight(function: NoteFunction, metric_weight: float,
                       duration_beats: float,
                       weights: Dict[str, float] = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    base = weights.get(function, 0.4)
    duration_factor = min(max(duration_beats, 0.0) / 1.0, 1.5) if duration_beats > 0 else 1.0
    duration_factor = max(duration_factor, 0.5)
    return base * (0.5 + 0.5 * metric_weight) * duration_factor


@dataclass
class ScoredNote:
    note: Note
    function: NoteFunction
    weight: float
    beat: float


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


# ══════════════════════════════════════════════════════════════════════════════
#  [6] SUELO ESTRUCTURAL — reduccion schenkeriana local por niveles: en cada
#      nivel se eliminan las notas de menor peso estructural (fusionando
#      grupos de notas sobre el mismo acorde en una sola nota representativa)
#      hasta converger. Solo se calcula lo necesario para saber que notas
#      sobreviven en el nivel pedido por --floor-level.
# ══════════════════════════════════════════════════════════════════════════════

def _reduce_level1(scored: List[ScoredNote], threshold: float) -> Tuple[List[ScoredNote], List[Note]]:
    """Primer nivel de reduccion: elimina notas con peso estructural por
    debajo de `threshold` y extiende la nota vecina superviviente para
    cubrir el hueco dejado (extension de huecos)."""
    if not scored:
        return [], []
    n = len(scored)
    keep_mask = [s.weight >= threshold for s in scored]
    if not any(keep_mask):
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

    for k in range(1, len(result)):
        if result[k].note.start < result[k - 1].note.end:
            result[k - 1].note.end = result[k].note.start

    return result, removed_notes


def _is_leading_tone_resolution(prev: ScoredNote, cur: ScoredNote, tonic_pc: int, mode: str) -> bool:
    dp = scale_degree(prev.note.pitch, tonic_pc, mode)
    dc = scale_degree(cur.note.pitch, tonic_pc, mode)
    if dp != 7 or dc != 1:
        return False
    return abs(cur.note.pitch - prev.note.pitch) in (1, 2) and cur.note.pitch > prev.note.pitch


def _build_prolongation_spans(scored: List[ScoredNote], spans: List[ChordSpan]) -> List[List[int]]:
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


def _collapse_iteration(scored: List[ScoredNote], spans: List[ChordSpan],
                         tonic_pc: int, mode: str) -> Tuple[List[ScoredNote], List[Note], bool]:
    """Segundo nivel de reduccion en adelante: agrupa notas consecutivas que
    caen sobre el mismo acorde (prolongaciones) y las funde en una unica
    nota representativa (la de mayor peso estructural del grupo), salvo que
    el grupo contenga una resolucion de sensible protegida."""
    groups = _build_prolongation_spans(scored, spans)
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


@dataclass
class FloorResult:
    floor_index: int              # indice del nivel usado como suelo
    n_levels: int                 # niveles totales calculados hasta convergencia
    mel_keys: Set[Tuple[int, int]]   # (start_tick, pitch) protegidos, melodia
    bass_keys: Set[Tuple[int, int]]  # (start_tick, pitch) protegidos, bajo
    reached_ursatz: bool


def compute_structural_floor(all_notes: List[Note], tc: TimeContext, spans: List[ChordSpan],
                              tonic_pc: int, mode: str, num: int, subdiv: int,
                              W: np.ndarray, weights_cfg: Dict[str, float],
                              floor_level: str, threshold: float = 0.35,
                              max_levels: int = 8) -> Tuple[FloorResult, List[ScoredNote], List[ScoredNote]]:
    """Calcula, para melodia y bajo (skyline), la secuencia de niveles de
    reduccion schenkeriana (cada nivel parte del anterior, hasta converger o
    alcanzar max_levels) y devuelve el conjunto de notas (identificadas por
    start_tick+pitch) que sobreviven en el nivel pedido por --floor-level:
    esas notas quedan protegidas de eliminacion para siempre en el greedy
    de la seccion [9]."""
    melody_notes, bass_notes = build_skyline(all_notes)
    scored_mel0 = _score_voice(melody_notes, tc, spans, num, subdiv, W, weights_cfg)
    scored_bass0 = _score_voice(bass_notes, tc, spans, num, subdiv, W, weights_cfg)

    def _levels_for(scored0: List[ScoredNote]) -> List[List[ScoredNote]]:
        levels = [scored0]
        lvl1, _ = _reduce_level1(scored0, threshold)
        levels.append(lvl1)
        cur = lvl1
        n_iter = 1
        while n_iter < max_levels:
            new_cur, _, changed = _collapse_iteration(cur, spans, tonic_pc, mode)
            if not changed:
                break
            cur = new_cur
            levels.append(cur)
            n_iter += 1
        return levels

    mel_levels = _levels_for(scored_mel0)
    bass_levels = _levels_for(scored_bass0)
    n_levels = max(len(mel_levels), len(bass_levels))

    if floor_level == "ursatz":
        floor_idx = n_levels - 1
        reached_ursatz = True
    else:
        floor_idx = max(0, min(int(floor_level), n_levels - 1))
        reached_ursatz = (floor_idx == n_levels - 1)

    mel_floor = mel_levels[min(floor_idx, len(mel_levels) - 1)]
    bass_floor = bass_levels[min(floor_idx, len(bass_levels) - 1)]

    # `beat` se preserva sin recalcular a traves de _reduce_level1 y
    # _collapse_iteration (ver esos ports), asi que cada ScoredNote del nivel
    # suelo se corresponde exactamente con UNA nota original del skyline
    # (mismo pitch, mismo beat de ataque original) — de ahi que podamos
    # recuperar su (start_tick, pitch) real buscandola en scored_mel0/bass0
    # por (pitch, beat), en vez de usar el start/end ya fusionado del nivel.
    def _orig_keys(level_scored: List[ScoredNote], scored0: List[ScoredNote]) -> Set[Tuple[int, int]]:
        by_beat_pitch = {(round(s.beat, 6), s.note.pitch): s.note.start for s in scored0}
        keys = set()
        for s in level_scored:
            k = (round(s.beat, 6), s.note.pitch)
            if k in by_beat_pitch:
                keys.add((by_beat_pitch[k], s.note.pitch))
        return keys

    mel_keys = _orig_keys(mel_floor, scored_mel0)
    bass_keys = _orig_keys(bass_floor, scored_bass0)

    return (FloorResult(floor_idx, n_levels, mel_keys, bass_keys, reached_ursatz),
            scored_mel0, scored_bass0)


# ══════════════════════════════════════════════════════════════════════════════
#  [7] DIFICULTAD PIANISTICA — separacion de manos e independencia ritmica
#      entre ambas, usadas por la formula de dificultad de 5 factores
# ══════════════════════════════════════════════════════════════════════════════

def separate_hands(mid: MidiData, split: int = 60) -> Tuple[List[Note], List[Note]]:
    """Separa las notas en mano derecha (rh) e izquierda (lh). Si el MIDI
    tiene 2 o mas pistas con notas, usa la pista de pitch medio mas agudo
    como mano derecha y el resto como mano izquierda; si no, separa por
    pitch usando `split` como frontera (por defecto 60 = C4)."""
    note_tracks = [(i, extract_notes(t)) for i, t in enumerate(mid.tracks)]
    note_tracks = [(i, ns) for i, ns in note_tracks if ns]
    if len(note_tracks) >= 2:
        avg = [(i, sum(n.pitch for n in ns) / len(ns), ns) for i, ns in note_tracks]
        avg.sort(key=lambda x: -x[1])
        rh = list(avg[0][2])
        lh = [n for _, _, ns in avg[1:] for n in ns]
        return rh, lh
    all_notes = [n for _, ns in note_tracks for n in ns]
    rh = [n for n in all_notes if n.pitch >= split]
    lh = [n for n in all_notes if n.pitch < split]
    return rh, lh


def _chords_at_onsets(notes: List[Note]) -> List[List[Note]]:
    by_start: Dict[int, List[Note]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    return [by_start[s] for s in sorted(by_start)]


def _independence(rh: List[Note], lh: List[Note], tc: TimeContext) -> float:
    """Mide cuanto se mueven las dos manos de forma independiente entre si
    (proporcion de casillas ritmicas donde una mano tiene onset y la otra
    no), en una rejilla de semicorchea."""
    if not rh or not lh:
        return 0.0
    grid = max(1, tc.tpb // 4)
    sr = {n.start // grid for n in rh}
    sl = {n.start // grid for n in lh}
    if not sr or not sl:
        return 0.0
    inter = len(sr & sl)
    union = len(sr | sl)
    return round(1.0 - inter / union, 3) if union else 0.0


def _peak_nps(ctx_notes: List[Note], tc: TimeContext) -> float:
    if not ctx_notes:
        return 0.0
    starts = sorted(tc.sec(n.start) for n in ctx_notes)
    peak, j = 0, 0
    for i in range(len(starts)):
        while starts[i] - starts[j] > 2.0:
            j += 1
        peak = max(peak, i - j + 1)
    return peak / 2.0


def _span_poly_leaps(bar_notes: List[Note]) -> Tuple[int, int, int]:
    chords = _chords_at_onsets(bar_notes)
    spans = [max(c, key=lambda n: n.pitch).pitch - min(c, key=lambda n: n.pitch).pitch
             for c in chords if len(c) > 1]
    max_span = max(spans) if spans else 0
    max_poly = max((len(c) for c in chords), default=0)
    top = [max(c, key=lambda n: n.pitch).pitch for c in chords]
    leaps = sum(1 for a, b in zip(top[:-1], top[1:]) if abs(b - a) > 9)
    return int(max_span), int(max_poly), int(leaps)


def bar_factors(rh_bar: List[Note], lh_bar: List[Note],
                 rh_ctx: List[Note], lh_ctx: List[Note],
                 tc: TimeContext) -> Tuple[int, Dict[str, float], float]:
    """Aplica la misma formula de dificultad de 5 factores que
    whole_piece_grade(), pero evaluada compas a compas y con normalizacion
    ABSOLUTA contra constantes fijas (no relativa al maximo de la propia
    pieza). Esto es lo que permite comparar el resultado directamente contra
    --target-grade sin que el umbral se desplace con cada eliminacion que
    hace el greedy."""
    peak = max(_peak_nps(rh_ctx, tc), _peak_nps(lh_ctx, tc))
    rh_span, rh_poly, rh_leaps = _span_poly_leaps(rh_bar)
    lh_span, lh_poly, lh_leaps = _span_poly_leaps(lh_bar)
    indep = _independence(rh_bar, lh_bar, tc)

    speed = float(np.clip(peak / 12.0, 0, 1))
    extension = float(np.clip(max(rh_span, lh_span) / 14.0, 0, 1))
    poly = float(np.clip((max(rh_poly, lh_poly) - 1) / 4.0, 0, 1))
    leaps_f = float(np.clip((rh_leaps + lh_leaps) / 2.0, 0, 1))
    independence = float(np.clip(indep / 0.7, 0, 1))

    factors = {"velocidad": speed, "extension": extension, "polifonia": poly,
               "saltos": leaps_f, "independencia": independence}
    weights = {"velocidad": 0.30, "extension": 0.20, "polifonia": 0.20,
               "saltos": 0.15, "independencia": 0.15}
    diff = sum(factors[k] * weights[k] for k in factors)
    grade = int(np.clip(round(diff * 7), 0, 7)) + 1
    return grade, factors, diff


def whole_piece_grade(rh: List[Note], lh: List[Note], tc: TimeContext) -> Tuple[int, float]:
    """Grado global de toda la pieza (para el informe, antes/despues).
    NO reutiliza bar_factors: esa funcion normaliza 'saltos' como cuenta
    ABSOLUTA por compas, lo cual es correcto para UN compas pero saturaria a
    1.0 si se aplicara a la pieza entera sin dividir por el numero de
    compases, asi que aqui se recalcula el factor con esa division."""
    all_notes = rh + lh
    if not all_notes:
        return 1, 0.0
    last = max(n.end for n in all_notes)
    n_bars = max(1, tc.bar(last - 1))

    def hand_metrics(notes):
        if not notes:
            return dict(peak_nps=0.0, max_span=0, max_poly=0, leaps=0)
        starts = sorted(tc.sec(n.start) for n in notes)
        peak, j = 0, 0
        for i in range(len(starts)):
            while starts[i] - starts[j] > 2.0:
                j += 1
            peak = max(peak, i - j + 1)
        chords = _chords_at_onsets(notes)
        spans = [max(c, key=lambda n: n.pitch).pitch - min(c, key=lambda n: n.pitch).pitch
                 for c in chords if len(c) > 1]
        max_span = max(spans) if spans else 0
        max_poly = max((len(c) for c in chords), default=0)
        top = [max(c, key=lambda n: n.pitch).pitch for c in chords]
        leaps = sum(1 for a, b in zip(top[:-1], top[1:]) if abs(b - a) > 9)
        return dict(peak_nps=peak / 2.0, max_span=max_span, max_poly=max_poly, leaps=leaps)

    m_rh, m_lh = hand_metrics(rh), hand_metrics(lh)
    speed = float(np.clip(max(m_rh["peak_nps"], m_lh["peak_nps"]) / 12.0, 0, 1))
    extension = float(np.clip(max(m_rh["max_span"], m_lh["max_span"]) / 14.0, 0, 1))
    poly = float(np.clip((max(m_rh["max_poly"], m_lh["max_poly"]) - 1) / 4.0, 0, 1))
    leaps = float(np.clip((m_rh["leaps"] + m_lh["leaps"]) / max(1, n_bars) / 2.0, 0, 1))
    indep = _independence(rh, lh, tc)
    independence = float(np.clip(indep / 0.7, 0, 1))

    factors = {"velocidad": speed, "extension": extension, "polifonia": poly,
               "saltos": leaps, "independencia": independence}
    weights = {"velocidad": 0.30, "extension": 0.20, "polifonia": 0.20,
               "saltos": 0.15, "independencia": 0.15}
    diff = sum(factors[k] * weights[k] for k in factors)
    grade = int(np.clip(round(diff * 7), 0, 7)) + 1
    return grade, diff


# ══════════════════════════════════════════════════════════════════════════════
#  [8] REGISTROS DE NOTA PARA LA TEXTURA COMPLETA — extiende la clasificacion
#      de funcion tonal a toda la pieza, no solo a las voces del skyline
#      (las voces internas se clasifican con la regla simple de pertenencia
#      al acorde vigente, sin el analisis completo CT/PT/NT/...)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(eq=False)
class NoteRec:
    note: Note
    hand: str                 # "rh" | "lh"
    bar: int
    voice_role: str            # "mel" | "bass" | "inner"
    function: NoteFunction
    weight: float
    floor_protected: bool
    is_octave_dup: bool
    candidate: bool
    orig_id: int = -1          # identidad estable de la nota original de la
                                # superficie (indice en all_notes), preservada
                                # a traves de copias y de la extension de
                                # huecos (que si muta note.start/note.end) —
                                # permite verificar que un grado mas exigente
                                # es subconjunto de uno menos exigente sin que
                                # la comparacion se rompa por ticks distintos
                                # entre niveles


def build_note_records(all_notes: List[Note], rh_notes: List[Note], lh_notes: List[Note],
                        scored_mel0: List[ScoredNote], scored_bass0: List[ScoredNote],
                        spans: List[ChordSpan], tc: TimeContext, num: int, subdiv: int,
                        W: np.ndarray, weights_cfg: Dict[str, float],
                        floor: FloorResult) -> List[NoteRec]:
    mel_by_key = {(s.note.start, s.note.pitch): s for s in scored_mel0}
    bass_by_key = {(s.note.start, s.note.pitch): s for s in scored_bass0}

    rh_set = {(n.start, n.pitch, n.end) for n in rh_notes}

    # duplicacion de octava: mismo pitch-class, mismo onset, en mas de una
    # nota simultanea distinta (comprobacion simple de pitch-class
    # coincidente en el mismo onset entre voces)
    by_onset: Dict[int, List[Note]] = {}
    for n in all_notes:
        by_onset.setdefault(n.start, []).append(n)
    dup_ids: Set[int] = set()
    for onset, group in by_onset.items():
        if len(group) < 2:
            continue
        pcs_seen: Dict[int, List[int]] = {}
        for n in group:
            pcs_seen.setdefault(n.pitch % 12, []).append(id(n))
        for pc, ids in pcs_seen.items():
            if len(ids) > 1:
                dup_ids.update(ids)

    records: List[NoteRec] = []
    for orig_id, n in enumerate(all_notes):
        hand = "rh" if (n.start, n.pitch, n.end) in rh_set else "lh"
        bar = tc.bar(n.start)
        key = (n.start, n.pitch)
        is_dup = id(n) in dup_ids

        role = "inner"
        function: NoteFunction
        weight: float
        floor_protected = False

        s_mel = mel_by_key.get(key)
        s_bass = bass_by_key.get(key)
        if s_mel is not None or s_bass is not None:
            # nota presente en el skyline de melodia y/o bajo: hereda
            # funcion/peso ya calculados por classify_note sobre esa voz.
            # si aparece en ambas (posible en pasajes monofonicos, donde
            # melodia y bajo coinciden), se protege si es CT en cualquiera.
            s = s_mel if s_mel is not None else s_bass
            role = "mel" if s_mel is not None else "bass"
            function = s.function
            weight = s.weight
            in_mel_floor = key in floor.mel_keys
            in_bass_floor = key in floor.bass_keys
            floor_protected = in_mel_floor or in_bass_floor
        else:
            beat = n.start_beat(tc)
            chord_now = chord_at(spans, beat)
            mweight = note_metric_weight(n, tc, num, subdiv, W)
            pc = n.pitch % 12
            if pc in chord_tone_pcs(chord_now):
                function = "CT"
            else:
                function = "UNCLASSIFIED"
            weight = structural_weight(function, mweight, n.duration_beats(tc), weights_cfg)

        # filtro de candidatos a eliminar por el greedy:
        #   - nunca candidatas: notas protegidas por el suelo estructural.
        #   - nunca candidatas: CT de la voz melodica/del bajo (regla dura
        #     adicional, incluso por encima del suelo).
        #   - candidatas: PT/NT/APP/SUS/ANT/ESC, UNCLASSIFIED, o
        #     duplicaciones de octava (incluso si son CT, salvo que ademas
        #     sean CT de la voz melodica/bajo, cubierto por la regla previa).
        if floor_protected:
            candidate = False
        elif role in ("mel", "bass") and function == "CT":
            candidate = False
        elif is_dup:
            candidate = True
        elif function in ("PT", "NT", "APP", "SUS", "ANT", "ESC", "UNCLASSIFIED"):
            candidate = True
        else:  # CT interno, no duplicado: no es candidata a eliminacion
            candidate = False

        records.append(NoteRec(note=n, hand=hand, bar=bar, voice_role=role,
                                function=function, weight=weight,
                                floor_protected=floor_protected,
                                is_octave_dup=is_dup, candidate=candidate,
                                orig_id=orig_id))
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  [8b] ADELGAZADO DE VOICINGS (--max-voices-per-chord)
#
#  La regla de candidatos de la seccion [8] protege TODO tono del acorde (CT)
#  que no sea duplicacion exacta de pitch-class, incluso dentro de un mismo
#  bloque de acorde de una sola mano. Eso deja intocables los factores de
#  "extension" (span de la mano) y "polifonia" del grado de dificultad, que
#  son precisamente los que dominan en piezas con acordes en bloque densos:
#  en esas piezas el greedy por si solo puede quedarse muy por encima del
#  target_grade pedido, porque casi toda la dificultad viene de la anchura y
#  densidad de los bloques de acordes, no de ornamentacion rapida.
#
#  --max-voices-per-chord N activa, como paso PREVIO al greedy de dificultad
#  y de forma UNIFORME para todos los --target-grade pedidos (es una
#  restriccion fija, no ligada a un grado objetivo concreto), un adelgazado
#  de cada bloque simultaneo de notas DENTRO DE UNA MISMA MANO a maximo N
#  voces, eliminando primero las menos esenciales para la identidad del
#  acorde:
#    1º duplicaciones de pitch-class dentro del propio bloque
#    2º la quinta (el tono mas prescindible en voicings reales de piano)
#    3º extensiones/tonos no pertenecientes al acorde
#    4º la sexta/novena si esta presente como color añadido
#    5º la tercera
#    6º la septima
#    7º la fundamental (jamas se elimina si hay alternativa)
#  El extremo agudo y grave de cada bloque (el contorno de esa mano) y toda
#  nota protegida por el suelo estructural NUNCA se tocan aqui, sea cual sea
#  N.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VoicingRemoval:
    bar: int
    hand: str
    pitch: int
    beat: float
    reason: str        # "duplicacion_pc" | "quinta" | "extension" | "sexta_novena" | "tercera" | "septima"


def _voicing_priority_score(pitch: int, chord: Optional[ChordSpan], pcs_in_group: List[int]) -> Tuple[int, str]:
    """Puntuacion de removibilidad: MENOR = se elimina antes. Devuelve
    tambien la razon (para el informe)."""
    pc = pitch % 12
    if pcs_in_group.count(pc) > 1:
        return 0, "duplicacion_pc"
    if chord is None or chord.root is None:
        return 2, "extension"
    rel = (pc - chord.root) % 12
    if rel == 7:
        return 1, "quinta"
    if rel == 9:
        return 3, "sexta_novena"
    if rel in (3, 4):
        return 4, "tercera"
    if rel in (10, 11):
        return 5, "septima"
    if rel == 0:
        return 6, "fundamental"
    return 2, "extension"


def apply_voicing_cap(records: List[NoteRec], tc: TimeContext, spans: List[ChordSpan],
                       max_voices: int) -> Tuple[List[NoteRec], List[VoicingRemoval]]:
    """Adelgaza cada bloque simultaneo (misma mano, mismo onset) a maximo
    `max_voices` notas, usando el orden de prioridad de eliminacion descrito
    en la cabecera de esta seccion (con excepcion de los extremos del bloque
    y de las notas protegidas por el suelo estructural)."""
    by_group: Dict[Tuple[str, int], List[NoteRec]] = {}
    for r in records:
        by_group.setdefault((r.hand, r.note.start), []).append(r)

    removed_ids: Set[int] = set()
    log: List[VoicingRemoval] = []
    for (hand, onset), group in by_group.items():
        if len(group) <= max_voices:
            continue
        group_sorted = sorted(group, key=lambda r: r.note.pitch)
        protected_ids = {id(group_sorted[0]), id(group_sorted[-1])}
        pool = [r for r in group if id(r) not in protected_ids and not r.floor_protected]
        n_to_remove = min(len(group) - max_voices, len(pool))
        if n_to_remove <= 0:
            continue
        pcs_in_group = [r.note.pitch % 12 for r in group]
        beat0 = group[0].note.start_beat(tc)
        chord = chord_at(spans, beat0)
        ranked = sorted(pool, key=lambda r: _voicing_priority_score(r.note.pitch, chord, pcs_in_group)[0])
        for r in ranked[:n_to_remove]:
            _, reason = _voicing_priority_score(r.note.pitch, chord, pcs_in_group)
            removed_ids.add(id(r))
            log.append(VoicingRemoval(bar=r.bar, hand=hand, pitch=r.note.pitch,
                                       beat=round(beat0, 3), reason=reason))

    kept = [r for r in records if id(r) not in removed_ids]
    return kept, log


# ══════════════════════════════════════════════════════════════════════════════
#  [9] ALGORITMO GREEDY — elimina notas candidatas compas a compas, por orden
#      de menor peso estructural, hasta alcanzar el target_grade pedido
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RemovalLogEntry:
    bar: int
    target_grade: int
    pitch: int
    beat: float
    function: str
    weight: float
    delta_difficulty: float


def _copy_bar_state(active: List["NoteRec"], bar_t0: int, bar_t1: int) -> List["NoteRec"]:
    """Copia profunda (de la nota, no de listas globales) del estado activo
    de un compas en el instante en que se llama. Necesario porque el mismo
    objeto Note puede seguir mutando despues (extension de huecos en
    eliminaciones posteriores hacia targets mas agresivos): sin copiar, el
    snapshot de un target mas laxo quedaria corrompido por cambios hechos
    para alcanzar un target mas estricto en el mismo compas."""
    out = []
    for r in active:
        if bar_t0 <= r.note.start < bar_t1:
            n = r.note
            new_note = Note(pitch=n.pitch, start=n.start, end=n.end, vel=n.vel, channel=n.channel)
            out.append(NoteRec(note=new_note, hand=r.hand, bar=r.bar, voice_role=r.voice_role,
                                function=r.function, weight=r.weight,
                                floor_protected=r.floor_protected, is_octave_dup=r.is_octave_dup,
                                candidate=r.candidate, orig_id=r.orig_id))
    return out


@dataclass
class BarOutcome:
    bar: int
    grade_before: int
    grade_after: Dict[int, int]          # target_grade -> grado real alcanzado
    unreachable: Set[int]                # targets que no se pudieron alcanzar
    floor_notes_in_bar: int = 0          # cuantas notas de este compas estan protegidas por el suelo


def _extend_neighbor_on_removal(active: List[NoteRec], removed: NoteRec, bar_t0: int, bar_t1: int):
    """Aplica el relleno de hueco SOLO a notas de voz skyline (melodia/bajo),
    y solo dentro del propio compas (los compases se tratan de forma
    independiente en el greedy). Sigue la misma logica de extension que la
    reduccion schenkeriana del suelo estructural: APP/SUS tiran hacia atras
    (el siguiente absorbe), el resto empuja hacia delante (el anterior
    absorbe)."""
    if removed.voice_role not in ("mel", "bass"):
        return
    same_role = [r for r in active if r.voice_role == removed.voice_role
                 and r.hand == removed.hand and r is not removed
                 and bar_t0 <= r.note.start < bar_t1]
    if not same_role:
        return
    same_role.sort(key=lambda r: r.note.start)
    prevs = [r for r in same_role if r.note.start < removed.note.start]
    nexts = [r for r in same_role if r.note.start > removed.note.start]
    pulls_backward = removed.function in ("APP", "SUS")
    if pulls_backward and nexts:
        nxt = nexts[0]
        nxt.note.start = min(nxt.note.start, removed.note.start)
        return
    if prevs:
        prv = prevs[-1]
        prv.note.end = max(prv.note.end, removed.note.end)
        return
    if nexts:
        nxt = nexts[0]
        nxt.note.start = min(nxt.note.start, removed.note.start)


def reduce_graded(midi_path: str, target_grades: List[int], floor_level: str = "ursatz",
                   split: int = 60, weights_config: Optional[str] = None,
                   threshold: float = 0.35, key: Optional[str] = None,
                   window_beats: Optional[float] = None,
                   max_voices_per_chord: Optional[int] = None,
                   pre_simplify: bool = True) -> dict:
    """API publica. Ejecuta el pipeline completo y devuelve un dict con,
    por cada target_grade pedido, la lista de NoteRec activas (la version
    simplificada) y el informe de decisiones por compas.

    max_voices_per_chord: si se especifica, aplica primero el adelgazado de
    voicings de la seccion [8b] de forma uniforme para todos los
    target_grades pedidos, y luego corre el algoritmo greedy de la seccion
    [9] sobre la textura ya adelgazada."""
    mid, tc, all_notes = load_notes(midi_path)
    tonic_pc, mode, spans = analyze_harmony_local(all_notes, tc, key=key, window_beats=window_beats)
    rh_notes, lh_notes = separate_hands(mid, split)

    weights_cfg = dict(DEFAULT_WEIGHTS)
    if weights_config:
        weights_cfg.update({k: float(v) for k, v in
                            json.loads(Path(weights_config).read_text(encoding="utf-8")).items()})

    num = mid.timesig_map[0][1]
    subdiv = 4
    W = metric_weights(num, subdiv)

    floor, scored_mel0, scored_bass0 = compute_structural_floor(
        all_notes, tc, spans, tonic_pc, mode, num, subdiv, W, weights_cfg,
        floor_level=floor_level, threshold=threshold)

    records = build_note_records(all_notes, rh_notes, lh_notes, scored_mel0, scored_bass0,
                                  spans, tc, num, subdiv, W, weights_cfg, floor)

    voicing_log: List[VoicingRemoval] = []
    if max_voices_per_chord is not None and max_voices_per_chord > 0:
        records, voicing_log = apply_voicing_cap(records, tc, spans, max_voices_per_chord)

    # pre-pasada [ARMONICA]/[MELODICA]: sustituciones que no cambian el
    # numero de notas, uniformes para todos los target_grades, SIEMPRE antes
    # del greedy (para que este ya opere sobre una superficie mas simple).
    pre_simplify_stats = {"armonia_sustituida": 0, "bajo_clarificado": 0, "adornos_resueltos": 0}
    if pre_simplify:
        pre_simplify_stats["armonia_sustituida"] = simplify_harmony(records, spans, tc)
        pre_simplify_stats["bajo_clarificado"] = clarify_bass(records, spans, tc)
        records, n_orn = resolve_ornaments(records, tc)
        pre_simplify_stats["adornos_resueltos"] = n_orn

    last_tick = max(n.end for n in all_notes)
    n_bars = max(1, tc.bar(last_tick - 1))

    targets_sorted = sorted(set(target_grades), reverse=True)
    strictest = targets_sorted[-1]

    # estado activo global (todas las notas presentes en la version actual,
    # se va reduciendo compas a compas)
    active: List[NoteRec] = list(records)

    level_snapshots: Dict[int, List[NoteRec]] = {}
    removal_log: List[RemovalLogEntry] = []
    bar_outcomes: List[BarOutcome] = []

    for bar in range(1, n_bars + 1):
        bar_t0, bar_t1 = tc.bar_range_ticks(bar)
        bar_active = [r for r in active if bar_t0 <= r.note.start < bar_t1]
        rh_bar = [r.note for r in bar_active if r.hand == "rh"]
        lh_bar = [r.note for r in bar_active if r.hand == "lh"]
        bsec0 = tc.sec(bar_t0) - 2.0
        bsec1 = tc.sec(bar_t1) + 2.0
        rh_ctx = [r.note for r in active if r.hand == "rh" and bsec0 <= tc.sec(r.note.start) <= bsec1]
        lh_ctx = [r.note for r in active if r.hand == "lh" and bsec0 <= tc.sec(r.note.start) <= bsec1]
        grade_before, _, _ = bar_factors(rh_bar, lh_bar, rh_ctx, lh_ctx, tc)

        grade_after: Dict[int, int] = {}
        unreachable: Set[int] = set()

        candidates = [r for r in bar_active if r.candidate]
        cur_grade = grade_before
        next_target_idx = 0

        def _recompute_bar_grade():
            nonlocal rh_bar, lh_bar, rh_ctx, lh_ctx
            bar_active_now = [r for r in active if bar_t0 <= r.note.start < bar_t1]
            rh_bar = [r.note for r in bar_active_now if r.hand == "rh"]
            lh_bar = [r.note for r in bar_active_now if r.hand == "lh"]
            rh_ctx2 = [r.note for r in active if r.hand == "rh" and bsec0 <= tc.sec(r.note.start) <= bsec1]
            lh_ctx2 = [r.note for r in active if r.hand == "lh" and bsec0 <= tc.sec(r.note.start) <= bsec1]
            g, _, _ = bar_factors(rh_bar, lh_bar, rh_ctx2, lh_ctx2, tc)
            return g

        bar_snapshots: Dict[int, List[NoteRec]] = {}

        # snapshot inicial (copiado, ver _copy_bar_state) si ya cumple el
        # target mas laxo pedido, antes de que empiece ninguna eliminacion
        while next_target_idx < len(targets_sorted) and cur_grade <= targets_sorted[next_target_idx]:
            t = targets_sorted[next_target_idx]
            grade_after[t] = cur_grade
            bar_snapshots[t] = _copy_bar_state(active, bar_t0, bar_t1)
            next_target_idx += 1

        remaining = list(candidates)
        while remaining and next_target_idx < len(targets_sorted):
            best = None
            best_ratio = -1e18
            best_delta = None
            for r in remaining:
                if r not in active:
                    continue
                # remocion hipotetica: quita r temporalmente, recalcula
                active.remove(r)
                g_after = _recompute_bar_grade()
                active.append(r)  # restaurar (append basta, orden no importa aqui)
                delta = cur_grade - g_after
                ratio = delta / max(r.weight, 1e-6)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = r
                    best_delta = delta
            if best is None or best_delta is None or best_delta <= 0:
                break
            active.remove(best)
            _extend_neighbor_on_removal(active, best, bar_t0, bar_t1)
            removal_log.append(RemovalLogEntry(
                bar=bar, target_grade=targets_sorted[next_target_idx], pitch=best.note.pitch,
                beat=best.note.start_beat(tc), function=best.function, weight=round(best.weight, 4),
                delta_difficulty=round(best_delta, 4)))
            remaining.remove(best)
            cur_grade = _recompute_bar_grade()
            while next_target_idx < len(targets_sorted) and cur_grade <= targets_sorted[next_target_idx]:
                t = targets_sorted[next_target_idx]
                grade_after[t] = cur_grade
                bar_snapshots[t] = _copy_bar_state(active, bar_t0, bar_t1)
                next_target_idx += 1

        # cualquier target no alcanzado: se entrega el suelo (estado actual,
        # ya no se puede reducir mas sin violar el suelo estructural o sin
        # que ninguna eliminacion siga ayudando)
        for t in targets_sorted[next_target_idx:]:
            grade_after[t] = cur_grade
            unreachable.add(t)
            bar_snapshots[t] = _copy_bar_state(active, bar_t0, bar_t1)

        for t in target_grades:
            level_snapshots.setdefault(t, [])
            level_snapshots[t].extend(bar_snapshots[t])

        n_floor_in_bar = sum(1 for r in bar_active if r.floor_protected)
        bar_outcomes.append(BarOutcome(bar=bar, grade_before=grade_before,
                                        grade_after=grade_after, unreachable=unreachable,
                                        floor_notes_in_bar=n_floor_in_bar))

    rh_final, lh_final = rh_notes, lh_notes  # solo para grado global "antes"
    grade_before_piece, _ = whole_piece_grade(rh_final, lh_final, tc)

    return {
        "mid": mid, "tc": tc, "records": records, "floor": floor,
        "n_bars": n_bars, "target_grades": target_grades,
        "level_snapshots": level_snapshots, "removal_log": removal_log,
        "bar_outcomes": bar_outcomes, "grade_before_piece": grade_before_piece,
        "tonic_pc": tonic_pc, "mode": mode, "spans": spans,
        "voicing_log": voicing_log, "max_voices_per_chord": max_voices_per_chord,
        "pre_simplify_stats": pre_simplify_stats,
    }


def grade_bars(midi_path: str, split: int = 60) -> List[Tuple[int, int]]:
    """Utilidad publica: (bar, bar_grade) de la pieza tal cual esta, sin
    reducir nada — util para inspeccionar antes de decidir target grades."""
    mid, tc, all_notes = load_notes(midi_path)
    rh_notes, lh_notes = separate_hands(mid, split)
    last_tick = max(n.end for n in all_notes)
    n_bars = max(1, tc.bar(last_tick - 1))
    out = []
    for bar in range(1, n_bars + 1):
        bar_t0, bar_t1 = tc.bar_range_ticks(bar)
        rh_bar = [n for n in rh_notes if bar_t0 <= n.start < bar_t1]
        lh_bar = [n for n in lh_notes if bar_t0 <= n.start < bar_t1]
        bsec0, bsec1 = tc.sec(bar_t0) - 2.0, tc.sec(bar_t1) + 2.0
        rh_ctx = [n for n in rh_notes if bsec0 <= tc.sec(n.start) <= bsec1]
        lh_ctx = [n for n in lh_notes if bsec0 <= tc.sec(n.start) <= bsec1]
        g, _, _ = bar_factors(rh_bar, lh_bar, rh_ctx, lh_ctx, tc)
        out.append((bar, g))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  [10] EXPORTACION MIDI POR NIVEL
# ══════════════════════════════════════════════════════════════════════════════

def _conductor_track(tempo_map: List[Tuple[int, int]],
                      timesig_map: List[Tuple[int, int, int]]) -> MidiTrackData:
    """Pista de conductor con los eventos de tempo/compas reales. NECESARIA:
    write_midi() serializa unicamente los MidiEvent ya presentes en cada
    pista — los campos MidiData.tempo_map/timesig_map son solo el resultado
    de haber LEIDO un fichero, write_midi() no los vuelve a convertir en
    eventos meta. Sin esta pista, el MIDI exportado perderia el tempo
    original y caeria al default de 120bpm al releerlo, distorsionando
    cualquier analisis de dificultad basado en tiempo real (peak_nps) hecho
    sobre el fichero ya exportado."""
    trk = MidiTrackData(name="conductor")
    name_b = "conductor".encode("latin-1", "replace")
    trk.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x03,
                                data=bytes([0xFF, 0x03]) + _write_vlq(len(name_b)) + name_b))
    for tick, num, den in (timesig_map or [(0, 4, 4)]):
        dd = int(round(math.log2(max(1, den))))
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x58,
                                    data=bytes([0xFF, 0x58]) + _write_vlq(4)
                                    + bytes([num, dd, 24, 8])))
    for tick, us in (tempo_map or [(0, 500000)]):
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x51,
                                    data=bytes([0xFF, 0x51]) + _write_vlq(3)
                                    + us.to_bytes(3, "big")))
    trk.events.sort(key=lambda e: e.abs)
    return trk


def export_grade_midi(records: List[NoteRec], tpb: int, out_path: str,
                       tempo_map: List[Tuple[int, int]], timesig_map: List[Tuple[int, int, int]]):
    trk_rh = MidiTrackData(name="MD (simplificada)")
    trk_lh = MidiTrackData(name="MI (simplificada)")
    for r in records:
        trk = trk_rh if r.hand == "rh" else trk_lh
        n = r.note
        trk.events.append(MidiEvent(abs=n.start, kind="note_on", channel=0 if r.hand == "rh" else 1,
                                    pitch=n.pitch, vel=n.vel))
        trk.events.append(MidiEvent(abs=n.end, kind="note_off", channel=0 if r.hand == "rh" else 1,
                                    pitch=n.pitch, vel=0))
    conductor = _conductor_track(tempo_map, timesig_map)
    mid = MidiData(fmt=1, tpb=tpb, tracks=[conductor, trk_rh, trk_lh],
                    tempo_map=list(tempo_map) or [(0, 500000)],
                    timesig_map=list(timesig_map) or [(0, 4, 4)])
    write_midi(mid, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  [REJILLA] REJILLA PROGRESIVA DE RESPALDO — cuantiza los onsets a una
#  rejilla temporal (en ticks) y recorta la polifonia a un tope de voces.
#  Se activa SOLO cuando el greedy de la seccion [ESTRUCTURAL] no alcanza el
#  target_grade pedido porque el suelo estructural lo bloquea.
# ══════════════════════════════════════════════════════════════════════════════

def _grid_ladder(tpb: int) -> List[Tuple[int, int]]:
    """Escala de (rejilla_en_ticks, max_voces_por_mano), de mas fina/fiel a
    mas gruesa/agresiva, con peldaños intermedios suficientes para poder
    apuntar a un target_grade concreto sin saltos bruscos de fidelidad."""
    q = tpb  # negra
    return [
        (q // 2, 3),   # corchea, hasta triada
        (q // 2, 2),   # corchea, hasta dos voces
        (q, 2),        # negra, hasta dos voces
        (q, 1),        # negra, monofonico
        (q * 2, 1),    # blanca, monofonico
        (q * 4, 1),    # redonda (compas 4/4), monofonico
    ]


@dataclass
class _Event:
    start: int
    end: int
    recs: List["NoteRec"]
    importance: float


def _cluster_events(recs: List["NoteRec"], cluster_eps_ticks: int) -> List[_Event]:
    """Agrupa notas casi-simultaneas (dentro de cluster_eps_ticks) en un
    unico evento, para poder tratarlas como una sola unidad al cuantizar."""
    recs = sorted(recs, key=lambda r: r.note.start)
    events = []
    i = 0
    while i < len(recs):
        start = recs[i].note.start
        group = [recs[i]]
        j = i + 1
        while j < len(recs) and recs[j].note.start - start < cluster_eps_ticks:
            group.append(recs[j])
            j += 1
        end = max(r.note.end for r in group)
        importance = sum(r.weight for r in group) / len(group)
        events.append(_Event(start=start, end=end, recs=group, importance=importance))
        i = j
    return events


def _quantize_hand(recs: List["NoteRec"], tc: "TimeContext", grid_ticks: int,
                    max_voices: int, keep: str) -> List["NoteRec"]:
    """Cuantiza los eventos NO protegidos por el suelo de una mano a
    `grid_ticks`, conserva el evento mas importante por casilla (segun el
    structural_weight ya calculado por el motor estructural), le recorta la
    polifonia a `max_voices` (keep='top' agudo-primero para MD, 'bottom'
    grave-primero para MI) y estira legato hasta el siguiente evento
    conservado. Las notas floor_protected pasan intactas, fuera de la
    competicion por casilla."""
    protected = [r for r in recs if r.floor_protected]
    rest = [r for r in recs if not r.floor_protected]
    if not rest:
        return recs

    cluster_eps = max(1, grid_ticks // 6)
    events = _cluster_events(rest, cluster_eps)

    buckets: Dict[int, List[_Event]] = {}
    for ev in events:
        slot = round(ev.start / grid_ticks)
        buckets.setdefault(slot, []).append(ev)

    kept_recs: List["NoteRec"] = []
    for slot in sorted(buckets):
        best = max(buckets[slot], key=lambda e: e.importance)
        pitches = sorted(set(r.note.pitch for r in best.recs))
        if len(pitches) > max_voices:
            pitches = pitches[-max_voices:] if keep == "top" else pitches[:max_voices]
        template = best.recs[0]
        new_start = slot * grid_ticks
        for p in pitches:
            src = next((r for r in best.recs if r.note.pitch == p), template)
            new_note = Note(pitch=p, start=new_start, end=max(best.end, new_start + 1),
                             vel=src.note.vel, channel=src.note.channel)
            kept_recs.append(NoteRec(
                note=new_note, hand=src.hand, bar=tc.bar(new_start),
                voice_role="grid", function="UNCLASSIFIED", weight=best.importance,
                floor_protected=False, is_octave_dup=False, candidate=True,
                orig_id=src.orig_id))

    # estirar legato: cada evento llega hasta el siguiente onset conservado
    # en la misma mano, para no dejar huecos raros al fundir varias notas
    # en una casilla.
    all_hand = sorted(protected + kept_recs, key=lambda r: r.note.start)
    for i, r in enumerate(all_hand):
        if r.floor_protected:
            continue
        if i + 1 < len(all_hand):
            nxt_start = all_hand[i + 1].note.start
            if nxt_start > r.note.start:
                r.note.end = nxt_start
            else:
                r.note.end = max(r.note.end, r.note.start + 1)
        else:
            r.note.end = max(r.note.end, r.note.start + 1)
    return all_hand


def grid_fallback(records: List["NoteRec"], tc: "TimeContext",
                   grid_ticks: int, max_voices: int) -> List["NoteRec"]:
    rh = [r for r in records if r.hand == "rh"]
    lh = [r for r in records if r.hand == "lh"]
    rh_out = _quantize_hand(rh, tc, grid_ticks, max_voices, keep="top")
    lh_out = _quantize_hand(lh, tc, grid_ticks, max_voices, keep="bottom")
    return sorted(rh_out + lh_out, key=lambda r: r.note.start)


def apply_grid_fallback_until_target(records: List["NoteRec"], tc: "TimeContext",
                                      tpb: int, target_grade: int) -> Tuple[List["NoteRec"], int, bool, int]:
    """Aplica la escala de rejillas progresivamente (cada peldaño parte del
    resultado del anterior, nunca de las notas originales) hasta que
    whole_piece_grade <= target_grade o se agote la escala. Devuelve
    (records, grado_alcanzado, objetivo_cumplido, nº_de_peldaños_usados)."""
    cur = records
    rh = [r.note for r in cur if r.hand == "rh"]
    lh = [r.note for r in cur if r.hand == "lh"]
    grade, _ = whole_piece_grade(rh, lh, tc)
    steps = 0
    for grid_ticks, max_voices in _grid_ladder(tpb):
        if grade <= target_grade:
            break
        cur = grid_fallback(cur, tc, grid_ticks, max_voices)
        rh = [r.note for r in cur if r.hand == "rh"]
        lh = [r.note for r in cur if r.hand == "lh"]
        grade, _ = whole_piece_grade(rh, lh, tc)
        steps += 1
    return cur, grade, grade <= target_grade, steps


# ══════════════════════════════════════════════════════════════════════════════
#  [HIBRIDO] CAPA DE TRANSFORMACIONES QUE NO SE LIMITAN A ELIMINAR NOTAS
#
#  Fusion de combined_simplifier.py v2.0 (sustitucion armonica genuina,
#  hallazgo empirico sobre el punto de arranque de la rejilla) y
#  combined_simplifier_transformed.py v1.2 (bucle de aceptacion adaptativa,
#  align_hands, merge_repeated_notes, trim_overlaps). Ver cabecera del
#  fichero para el detalle de que se tomo de cada uno y por que.
# ══════════════════════════════════════════════════════════════════════════════

def _clone_records(records: List["NoteRec"]) -> List["NoteRec"]:
    out = []
    for r in records:
        n = Note(pitch=r.note.pitch, start=r.note.start, end=r.note.end,
                  vel=r.note.vel, channel=r.note.channel)
        out.append(NoteRec(note=n, hand=r.hand, bar=r.bar, voice_role=r.voice_role,
                            function=r.function, weight=r.weight,
                            floor_protected=r.floor_protected, is_octave_dup=r.is_octave_dup,
                            candidate=r.candidate, orig_id=r.orig_id))
    return out


def _records_grade(records: List["NoteRec"], tc: "TimeContext") -> int:
    rh = [r.note for r in records if r.hand == "rh"]
    lh = [r.note for r in records if r.hand == "lh"]
    g, _ = whole_piece_grade(rh, lh, tc)
    return g


def _sort_records(records: List["NoteRec"]) -> List["NoteRec"]:
    return sorted(records, key=lambda r: (r.note.start, r.note.pitch))


def _trim_overlaps(records: List["NoteRec"]) -> List["NoteRec"]:
    """[de v1.2] Red de seguridad tras cualquier tecnica que mueva onsets:
    recorta solapamientos obvios dentro de cada mano, sin tocar notas
    protegidas por el suelo. quantize_light/align_hands/arpeggiate_dense_chords
    la llaman siempre al terminar."""
    for hand in ("rh", "lh"):
        hs = sorted([r for r in records if r.hand == hand], key=lambda r: (r.note.start, r.note.pitch))
        for i in range(len(hs) - 1):
            cur, nxt = hs[i], hs[i + 1]
            if cur.note.end > nxt.note.start and not cur.floor_protected:
                cur.note.end = max(cur.note.start + 1, nxt.note.start)
    return records


# ── [ARMONICA] pre-pasada, corre ANTES del greedy, uniforme para todo target ─

def _nearest_chord_tone_pitch(pitch: int, chord: Optional[ChordSpan]) -> int:
    ct_pcs = sorted(chord_tone_pcs(chord))
    if not ct_pcs:
        return pitch
    pc = pitch % 12
    best_pc = min(ct_pcs, key=lambda t: min((t - pc) % 12, (pc - t) % 12))
    delta = ((best_pc - pc + 6) % 12) - 6
    return pitch + delta


def simplify_harmony(records: List["NoteRec"], spans: List[ChordSpan], tc: "TimeContext",
                      min_duration_beats: float = 0.4) -> int:
    """[de v2.0] Sustituye (nunca elimina) tonos de extension de voces
    internas por el tono de acorde mas cercano, cuando la nota dura lo
    suficiente como para no ser una figuracion de superficie."""
    count = 0
    for r in records:
        if r.floor_protected or r.voice_role != "inner" or r.function != "UNCLASSIFIED":
            continue
        if r.note.duration_beats(tc) < min_duration_beats:
            continue
        chord = chord_at(spans, r.note.start_beat(tc))
        if chord is None or chord.root is None:
            continue
        if (r.note.pitch % 12) in chord_tone_pcs(chord):
            continue
        new_pitch = _nearest_chord_tone_pitch(r.note.pitch, chord)
        if 0 <= new_pitch <= 127 and new_pitch != r.note.pitch:
            r.note.pitch = new_pitch
            r.function = "CT"
            count += 1
    return count


def clarify_bass(records: List["NoteRec"], spans: List[ChordSpan], tc: "TimeContext") -> int:
    """[de v2.0] En cada cambio de armonia, desplaza (por semitonos, sin
    cambiar de octava) la nota mas grave de la mano izquierda a la
    fundamental del acorde, si no lo era ya. No retira ninguna nota."""
    count = 0
    lh = [r for r in records if r.hand == "lh"]
    for sp in spans:
        if sp.root is None:
            continue
        in_span = [r for r in lh if sp.start_beat <= r.note.start_beat(tc) < sp.end_beat]
        if not in_span:
            continue
        bass_r = min(in_span, key=lambda r: r.note.pitch)
        if bass_r.floor_protected or bass_r.voice_role == "bass":
            continue
        pc = bass_r.note.pitch % 12
        if pc == sp.root:
            continue
        delta = ((sp.root - pc + 6) % 12) - 6
        new_pitch = bass_r.note.pitch + delta
        if 0 <= new_pitch <= 127:
            bass_r.note.pitch = new_pitch
            count += 1
    return count


def resolve_ornaments(records: List["NoteRec"], tc: "TimeContext",
                       grace_beats: float = 0.2) -> Tuple[List["NoteRec"], int]:
    """[de v2.0, MELODICA] Funde adornos muy breves (NT/APP/ESC) en la nota
    vecina de su propia voz (melodia/bajo)."""
    to_remove: Set[int] = set()
    by_voice: Dict[Tuple[str, str], List["NoteRec"]] = defaultdict(list)
    for r in records:
        if r.voice_role in ("mel", "bass"):
            by_voice[(r.hand, r.voice_role)].append(r)
    for lst in by_voice.values():
        lst.sort(key=lambda r: r.note.start)
        for i, r in enumerate(lst):
            if r.floor_protected or r.function not in ("NT", "APP", "ESC"):
                continue
            if r.note.duration_beats(tc) > grace_beats:
                continue
            prevs = [x for x in lst[:i] if id(x) not in to_remove]
            nexts = [x for x in lst[i + 1:] if id(x) not in to_remove]
            if r.function == "APP" and nexts:
                nexts[0].note.start = min(nexts[0].note.start, r.note.start)
            elif prevs:
                prevs[-1].note.end = max(prevs[-1].note.end, r.note.end)
            elif nexts:
                nexts[0].note.start = min(nexts[0].note.start, r.note.start)
            else:
                continue
            to_remove.add(id(r))
    kept = [r for r in records if id(r) not in to_remove]
    return kept, len(to_remove)


# ── [MELODICA/REGISTRO/TEXTURA/RITMICA/ADITIVA] pool del optimizador ────────

def reduce_melodic_leaps(records: List["NoteRec"], tc: "TimeContext", max_leap: int = 9) -> List["NoteRec"]:
    """[de v1.2 — sustituye a la version restringida a mel/bass de v2.0,
    que en la practica casi nunca se activaba] Para cada mano, sigue la nota
    MAS AGUDA de cada onset (no solo la clasificada como melodia/bajo: en
    bloques de acorde el 'contorno' tambien lo marca la nota superior/
    inferior real) y, si el salto respecto al onset anterior supera
    max_leap, la reubica por octavas al valor que mas lo reduce, siempre que
    no aumente el span del bloque mas alla de max(span_original, 14)."""
    for hand in ("rh", "lh"):
        hand_recs = [r for r in records if r.hand == hand]
        onsets = sorted({r.note.start for r in hand_recs})
        groups_by_onset, top_by_onset = {}, {}
        for onset in onsets:
            group = [r for r in hand_recs if r.note.start == onset]
            groups_by_onset[onset] = group
            top_by_onset[onset] = max(group, key=lambda r: r.note.pitch)
        prev_pitch = None
        for onset in onsets:
            top = top_by_onset[onset]
            if prev_pitch is not None and not top.floor_protected:
                leap = abs(top.note.pitch - prev_pitch)
                if leap > max_leap:
                    group = groups_by_onset[onset]
                    old_span = (max(r.note.pitch for r in group) - min(r.note.pitch for r in group)) if group else 0
                    best_pitch, best_diff = top.note.pitch, leap
                    for cand in (top.note.pitch - 12, top.note.pitch + 12,
                                 top.note.pitch - 24, top.note.pitch + 24):
                        if not (0 <= cand <= 127):
                            continue
                        diff = abs(cand - prev_pitch)
                        if diff >= best_diff:
                            continue
                        pitches = [r.note.pitch if r is not top else cand for r in group]
                        new_span = max(pitches) - min(pitches)
                        if new_span <= max(old_span, 14):
                            best_pitch, best_diff = cand, diff
                    if best_pitch != top.note.pitch:
                        top.note.pitch = best_pitch
            prev_pitch = top.note.pitch
    return _sort_records(records)


def close_voicings(records: List["NoteRec"], max_span: int = 14) -> List["NoteRec"]:
    """[de v1.2 — sustituye a la version que solo tocaba voces internas de
    v2.0, mas facil de bloquear] REGISTRO: cuando un bloque simultaneo supera
    max_span semitonos, mueve por octava el extremo (agudo o grave, el que
    no este protegido) que mas reduzca el span total."""
    groups: Dict[Tuple[str, int], List["NoteRec"]] = defaultdict(list)
    for r in records:
        groups[(r.hand, r.note.start)].append(r)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r.note.pitch)
        old_span = group[-1].note.pitch - group[0].note.pitch
        if old_span <= max_span:
            continue
        candidates = []
        top, bottom = group[-1], group[0]
        if not top.floor_protected:
            new_pitch = top.note.pitch - 12
            if 0 <= new_pitch <= 127:
                pitches = [r.note.pitch for r in group[:-1]] + [new_pitch]
                span = max(pitches) - min(pitches)
                if span < old_span:
                    candidates.append((span, top, new_pitch))
        if not bottom.floor_protected:
            new_pitch = bottom.note.pitch + 12
            if 0 <= new_pitch <= 127:
                pitches = [new_pitch] + [r.note.pitch for r in group[1:]]
                span = max(pitches) - min(pitches)
                if span < old_span:
                    candidates.append((span, bottom, new_pitch))
        if candidates:
            _, rec, new_pitch = min(candidates, key=lambda x: x[0])
            rec.note.pitch = new_pitch
    return _sort_records(records)


def collapse_octave_doubles(records: List["NoteRec"]) -> List["NoteRec"]:
    """[de v2.0, TEXTURA] Dentro de un bloque simultaneo, si dos voces
    internas suenan a exactamente una octava, retira la superior."""
    to_remove: Set[int] = set()
    by_group: Dict[Tuple[str, int], List["NoteRec"]] = defaultdict(list)
    for r in records:
        by_group[(r.hand, r.note.start)].append(r)
    for group in by_group.values():
        if len(group) < 3:
            continue
        sg = sorted(group, key=lambda r: r.note.pitch)
        for i in range(len(sg) - 1):
            a, b = sg[i], sg[i + 1]
            if id(a) in to_remove or id(b) in to_remove:
                continue
            if b.note.pitch - a.note.pitch != 12:
                continue
            if a.floor_protected or b.floor_protected:
                continue
            if a.voice_role in ("mel", "bass") or b.voice_role in ("mel", "bass"):
                continue
            to_remove.add(id(b))
    return [r for r in records if id(r) not in to_remove]


def arpeggiate_dense_chords(records: List["NoteRec"], tc: "TimeContext",
                             max_block: int = 3, grid_ticks: Optional[int] = None) -> List["NoteRec"]:
    """[de v1.2 — sustituye a la version de v2.0, que excluia cualquier
    bloque que tocara melodia/bajo y en la practica casi nunca se activaba]
    TEXTURA: en bloques de mas de max_block voces simultaneas, deja un ANCLA
    fija en el extremo que define el contorno (la mas aguda en la mano
    derecha, la mas grave en la izquierda — asi la melodia/bajo no se mueve
    de su ataque) y reparte el resto en un arpegio rapido tras el ancla. No
    retira ninguna nota."""
    if grid_ticks is None:
        grid_ticks = max(1, tc.tpb // 8)
    groups: Dict[Tuple[str, int], List["NoteRec"]] = defaultdict(list)
    for r in records:
        groups[(r.hand, r.note.start)].append(r)
    for (hand, onset), group in groups.items():
        if len(group) <= max_block:
            continue
        group.sort(key=lambda r: r.note.pitch)
        anchor = group[-1] if hand == "rh" else group[0]
        movable = [r for r in group if r is not anchor and not r.floor_protected]
        if not movable:
            continue
        order = sorted(movable, key=lambda r: r.note.pitch)
        for i, r in enumerate(order, start=1):
            dur = max(1, r.note.end - r.note.start)
            new_start = onset + i * grid_ticks
            r.note.start = new_start
            r.note.end = max(new_start + 1, new_start + dur)
            r.bar = tc.bar(new_start)
    _trim_overlaps(records)
    return _sort_records(records)


def merge_repeated_notes(records: List["NoteRec"], tc: "TimeContext") -> List["NoteRec"]:
    """[de v1.2, RITMICA] Funde ataques repetidos contiguos del mismo pitch
    en la misma mano — reduce notas/seg sin tocar el contenido de pitch."""
    out = []
    for hand in ("rh", "lh"):
        hs = sorted([r for r in records if r.hand == hand], key=lambda r: (r.note.start, r.note.pitch))
        merged = []
        for r in hs:
            if (merged and merged[-1].note.pitch == r.note.pitch
                    and merged[-1].note.end >= r.note.start - 1 and not r.floor_protected):
                merged[-1].note.end = max(merged[-1].note.end, r.note.end)
            else:
                merged.append(r)
        out.extend(merged)
    return _sort_records(out)


def quantize_light(records: List["NoteRec"], tc: "TimeContext", grid_ticks: int) -> List["NoteRec"]:
    """[de v2.0, con la red de seguridad trim_overlaps de v1.2, RITMICA]
    Cuantiza ataques no protegidos a `grid_ticks`, conservando la duracion."""
    if grid_ticks <= 1:
        return records
    for r in records:
        if r.floor_protected:
            continue
        dur = max(1, r.note.end - r.note.start)
        q = max(0, int(round(r.note.start / float(grid_ticks))) * grid_ticks)
        r.note.start = q
        r.note.end = max(q + 1, q + dur)
        r.bar = tc.bar(r.note.start)
    _trim_overlaps(records)
    return _sort_records(records)


def align_hands(records: List["NoteRec"], tc: "TimeContext", max_shift_ticks: Optional[int] = None) -> List["NoteRec"]:
    """[de v1.2, RITMICA] Alinea cada onset de la mano izquierda al onset mas
    cercano existente en la mano derecha (dentro de un margen) — ataca el
    factor de independencia directamente, a diferencia de quantize_light."""
    grid = max(1, tc.tpb // 4)
    if max_shift_ticks is None:
        max_shift_ticks = grid
    rh_onsets = sorted({r.note.start for r in records if r.hand == "rh"})
    if not rh_onsets:
        return records
    for r in records:
        if r.hand != "lh" or r.floor_protected:
            continue
        pos = bisect.bisect_left(rh_onsets, r.note.start)
        cands = []
        if pos < len(rh_onsets):
            cands.append(rh_onsets[pos])
        if pos > 0:
            cands.append(rh_onsets[pos - 1])
        if not cands:
            continue
        nearest = min(cands, key=lambda x: abs(x - r.note.start))
        shift = nearest - r.note.start
        if shift != 0 and abs(shift) <= max_shift_ticks:
            dur = max(1, r.note.end - r.note.start)
            r.note.start = nearest
            r.note.end = max(r.note.start + 1, r.note.start + dur)
            r.bar = tc.bar(r.note.start)
    _trim_overlaps(records)
    return _sort_records(records)


def add_connective_notes(records: List["NoteRec"], tc: "TimeContext",
                          gap_threshold_beats: float = 1.0, min_leap: int = 5) -> List["NoteRec"]:
    """[hibrida, ADITIVA] Como candidate=True/floor_protected=False (criterio
    de v1.2: solo sobrevive si el bucle de aceptacion comprueba que de verdad
    mejora el grado), pero limitada a huecos reales de melodia/bajo (criterio
    de v2.0, evita rellenar huecos arbitrarios de voces internas)."""
    new_records = list(records)
    by_voice: Dict[Tuple[str, str], List["NoteRec"]] = defaultdict(list)
    for r in records:
        if r.voice_role in ("mel", "bass"):
            by_voice[(r.hand, r.voice_role)].append(r)
    for (hand, role), lst in by_voice.items():
        lst.sort(key=lambda r: r.note.start)
        for i in range(len(lst) - 1):
            a, b = lst[i], lst[i + 1]
            gap_beats = (b.note.start - a.note.end) / tc.tpb
            leap = abs(b.note.pitch - a.note.pitch)
            if gap_beats < gap_threshold_beats or leap < min_leap:
                continue
            mid_pitch = (a.note.pitch + b.note.pitch) // 2
            start = a.note.end
            dur = min(tc.tpb // 2, max(1, b.note.start - start - 1))
            if dur < max(1, tc.tpb // 8):
                continue
            new_note = Note(pitch=mid_pitch, start=start, end=start + dur,
                             vel=max(30, a.note.vel - 15), channel=a.note.channel)
            new_records.append(NoteRec(note=new_note, hand=hand, bar=tc.bar(start), voice_role=role,
                                        function="PT", weight=0.2, floor_protected=False,
                                        is_octave_dup=False, candidate=True, orig_id=-1))
    _trim_overlaps(new_records)
    return _sort_records(new_records)


# ── [OPTIMIZADOR] bucle de aceptacion adaptativa (de v1.2) ──────────────────

def apply_musical_transformations(records: List["NoteRec"], tc: "TimeContext",
                                   target_grade: int, tpb: int,
                                   allow_added_notes: bool = True,
                                   max_passes: int = 8) -> Tuple[List["NoteRec"], int, List[str]]:
    """Prueba, en cada pasada, todas las tecnicas del pool [REGISTRO] +
    [TEXTURA] + [RITMICA] (+ [ADITIVA] si allow_added_notes) sobre un clon
    del estado actual, y acepta SOLO la primera que reduce estrictamente el
    grado medido. Repite hasta max_passes o hasta que ninguna mejora ya. La
    pre-pasada [ARMONICA]/[MELODICA] (simplify_harmony/clarify_bass/
    resolve_ornaments) NO entra aqui: ya corrio antes del greedy, de forma
    uniforme para todos los targets."""
    cur = _clone_records(records)
    grade = _records_grade(cur, tc)
    accepted_log: List[str] = []
    if grade <= target_grade:
        return cur, grade, accepted_log

    pool = [
        ("ritmica:fusion_repetidas", lambda rs: merge_repeated_notes(rs, tc)),
        ("ritmica:cuantizado_ligero", lambda rs: quantize_light(rs, tc, max(1, tpb // 4))),
        ("ritmica:alineacion_manos", lambda rs: align_hands(rs, tc)),
        ("registro:voicings_cerrados", lambda rs: close_voicings(rs)),
        ("melodica:reduccion_saltos", lambda rs: reduce_melodic_leaps(rs, tc)),
        ("textura:colapso_octavas", lambda rs: collapse_octave_doubles(rs)),
        ("textura:arpegiado", lambda rs: arpeggiate_dense_chords(
            rs, tc, max_block=2 if target_grade <= 2 else 3, grid_ticks=max(1, tpb // 8))),
    ]
    if allow_added_notes:
        pool.append(("aditiva:notas_conectoras", lambda rs: add_connective_notes(rs, tc)))

    for _ in range(max_passes):
        if grade <= target_grade:
            break
        improved = False
        for name, fn in pool:
            cand = _clone_records(cur)
            try:
                cand = fn(cand)
            except Exception:
                continue
            if not cand:
                continue
            g = _records_grade(cand, tc)
            if g < grade:
                cur, grade, improved = cand, g, True
                accepted_log.append(name)
                break
        if not improved:
            break
    return cur, grade, accepted_log


# ══════════════════════════════════════════════════════════════════════════════
#  [ORQUESTACION] COMBINACION DE AMBAS TECNICAS Y CLI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LevelResult:
    target_grade: int
    achieved_grade: int
    method: str
    n_notes: int
    n_notes_original: int
    records: List["NoteRec"]


def combined_simplify(midi_path: str, target_grades: List[int], floor_level: str = "ursatz",
                       split: int = 60, use_transforms: bool = True,
                       allow_added_notes: bool = True, max_transform_passes: int = 8) -> Dict:
    result = reduce_graded(midi_path, target_grades, floor_level=floor_level, split=split,
                            pre_simplify=use_transforms)
    tc, mid = result["tc"], result["mid"]
    n_original = len(result["records"])

    levels: Dict[int, LevelResult] = {}
    for t in sorted(set(target_grades), reverse=True):
        snap = result["level_snapshots"][t]
        rh = [r.note for r in snap if r.hand == "rh"]
        lh = [r.note for r in snap if r.hand == "lh"]
        achieved, _ = whole_piece_grade(rh, lh, tc)

        if achieved <= t:
            method = "estructural"
            chosen, achieved_final = snap, achieved
        elif use_transforms:
            transformed, achieved_t, accepted = apply_musical_transformations(
                snap, tc, t, mid.tpb, allow_added_notes=allow_added_notes,
                max_passes=max_transform_passes)
            if achieved_t <= t:
                method = "estructural+transformaciones (" + " → ".join(accepted) + ")" if accepted else "estructural+transformaciones"
                chosen, achieved_final = transformed, achieved_t
            else:
                # [hallazgo empirico de v2.0] si las transformaciones no
                # bastan, la rejilla rinde mas arrancando desde el snapshot
                # ESTRUCTURAL original, no desde el ya transformado.
                reduced, achieved2, ok, gsteps = apply_grid_fallback_until_target(snap, tc, mid.tpb, t)
                base = "estructural+rejilla" if ok else f"estructural+rejilla (limite tras {gsteps} peldaños)"
                tag = " (transformaciones intentadas: " + ", ".join(accepted) + ")" if accepted else " (transformaciones sin efecto)"
                method = base + tag
                chosen, achieved_final = reduced, achieved2
        else:
            reduced, achieved2, ok, gsteps = apply_grid_fallback_until_target(snap, tc, mid.tpb, t)
            method = "estructural+rejilla" if ok else f"estructural+rejilla (limite tras {gsteps} peldaños)"
            chosen, achieved_final = reduced, achieved2

        levels[t] = LevelResult(t, achieved_final, method, len(chosen), n_original, chosen)

    return {"mid": mid, "tc": tc, "levels": levels, "result": result}


def print_combined_report(midi_path: str, out: Dict):
    mid, tc = out["mid"], out["tc"]
    print(f"\n{'═'*78}\n  COMBINED SIMPLIFIER HYBRID v{VERSION} (autocontenido) — {midi_path}\n{'═'*78}")
    print(f"  grado original estimado: {out['result']['grade_before_piece']}/8")
    pss = out["result"].get("pre_simplify_stats")
    if pss and any(pss.values()):
        print(f"  pre-pasada armonica/melodica (uniforme para todos los grados): "
              f"{pss['armonia_sustituida']} tonos de extension sustituidos, "
              f"{pss['bajo_clarificado']} bajos clarificados, "
              f"{pss['adornos_resueltos']} adornos resueltos")
    for t in sorted(out["levels"], reverse=True):
        lv = out["levels"][t]
        pct = 100.0 * lv.n_notes / max(1, lv.n_notes_original)
        flag = "" if lv.achieved_grade <= t else "  [AVISO: no se alcanzo el target ni con la rejilla completa]"
        print(f"  target-grade {t}: alcanzado {lv.achieved_grade}/8  "
              f"metodo={lv.method}  notas={lv.n_notes}/{lv.n_notes_original} ({pct:.0f}%){flag}")
    print(f"{'═'*78}\n")


def main():
    ap = argparse.ArgumentParser(
        prog="combined_simplifier_hybrid.py",
        description="Simplifica un MIDI de piano a un nivel de dificultad pedagogica pedido: "
                     "sustitucion armonica/melodica + optimizador adaptativo de transformaciones "
                     "(registro/textura/ritmica/aditivas) + rejilla garantizada como ultimo recurso.")
    ap.add_argument("midi")
    ap.add_argument("--target-grade", type=int, nargs="+", required=True)
    ap.add_argument("--floor-level", default="ursatz")
    ap.add_argument("--split", type=int, default=60)
    ap.add_argument("--outdir")
    ap.add_argument("--no-transforms", action="store_true",
                     help="desactiva TODA la capa hibrida (pre-pasada armonica/melodica + "
                          "optimizador de transformaciones): replica el motor estructural puro "
                          "(suelo+greedy+rejilla)")
    ap.add_argument("--no-additive", action="store_true",
                     help="el optimizador no probara add_connective_notes (solo tecnicas que "
                          "nunca añaden notas)")
    ap.add_argument("--max-transform-passes", type=int, default=8,
                     help="numero maximo de pasadas del bucle de aceptacion adaptativa (def. 8)")
    args = ap.parse_args()

    for g in args.target_grade:
        if not (1 <= g <= 8):
            print(f"[ERROR] --target-grade debe estar entre 1 y 8 (recibido: {g})", file=sys.stderr)
            return 1

    out = combined_simplify(args.midi, args.target_grade, floor_level=args.floor_level, split=args.split,
                             use_transforms=not args.no_transforms,
                             allow_added_notes=not args.no_additive,
                             max_transform_passes=args.max_transform_passes)
    print_combined_report(args.midi, out)

    outdir = Path(args.outdir) if args.outdir else Path(args.midi).parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.midi).stem
    mid = out["mid"]
    print("  Ficheros generados:")
    for t in sorted(out["levels"], reverse=True):
        lv = out["levels"][t]
        out_path = outdir / f"{stem}_grade{t}.mid"
        export_grade_midi(lv.records, mid.tpb, str(out_path), mid.tempo_map, mid.timesig_map)
        print(f"    · {out_path}  ({lv.method})")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
