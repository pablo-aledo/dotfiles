#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIDI DOCTOR  v1.0                                   ║
║     Diagnóstico y reparación de MIDIs: red de seguridad del ecosistema       ║
║                                                                              ║
║  Las herramientas generativas pueden dejar un MIDI "casi bien": notas sin    ║
║  cerrar, solapes de la misma altura, notas fuera del registro del            ║
║  instrumento, compás o armadura perdidos al encadenar herramientas…          ║
║  Esta herramienta DETECTA esos problemas, los REPARA sin destruir nada y     ║
║  reinyecta metadatos (compás, armadura, tempo) copiándolos de un original.   ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    check   Diagnostica uno o varios MIDIs (ficheros, globs o carpetas)       ║
║    fix     Repara lo reparable y guarda un MIDI nuevo (nunca pisa el origen) ║
║    meta    Muestra, añade o copia compás / armadura / tempo                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python midi_doctor.py check obra.mid                                      ║
║    python midi_doctor.py check reconstrucciones/*/obra_final.mid --json r.json║
║    python midi_doctor.py check carpeta/ --strict          # rc=3 si hay hallazgos
║    python midi_doctor.py fix obra.mid --output obra_ok.mid                   ║
║    python midi_doctor.py fix obra.mid --instrument-range fold --ensure-meta  ║
║    python midi_doctor.py meta obra.mid                    # solo mostrar     ║
║    python midi_doctor.py meta salida.mid --copy-from original.mid -o ok.mid  ║
║    python midi_doctor.py meta obra.mid --time-sig 6/8 --key Gm -o ok.mid     ║
║                                                                              ║
║  QUÉ DETECTA (código · gravedad):                                            ║
║    E_PARSE     fichero ilegible o truncado                         error     ║
║    E_BYTES     bytes de dato >127 (se recortan al leer)            error     ║
║    W_STUCK     note_on que nunca se cierra                         aviso     ║
║    W_ORPHAN    note_off sin note_on                                aviso     ║
║    W_ZERO      nota de duración 0                                  aviso     ║
║    W_DUP_NOTE  misma nota duplicada en el mismo tick               aviso     ║
║    W_OVERLAP   misma altura y canal re-atacada sin cerrar          aviso     ║
║    W_RANGE     nota fuera del registro del instrumento (GM)        aviso     ║
║    W_MONO      acordes en un instrumento monofónico (viento/metal) aviso     ║
║    W_NO_EOT    pista sin end_of_track                              aviso     ║
║    W_DUP_META  meta idéntico repetido en el mismo tick             aviso     ║
║    W_NO_TS / W_NO_KS / W_NO_TEMPO   falta compás / armadura / tempo aviso    ║
║    W_TS_ALIGN  cambio de compás que no cae en barra de compás      aviso     ║
║    W_EMPTY     pista sin notas ni metadatos útiles                 info      ║
║                                                                              ║
║  OPCIONES DE fix:                                                            ║
║    --output FILE            Salida (default: <entrada>_fixed.mid)            ║
║    --inplace                Sobrescribe la entrada (atómico, deja .bak)      ║
║    --min-dur F              Duración mínima en pulsos para notas de duración 0 ║
║                             (default: 0.0625 = 1/16 de pulso)                ║
║    --drop-zero              Elimina las notas de duración 0 en vez de alargarlas║
║    --instrument-range M     none (default) | fold (plega octavas al registro)║
║                             | drop (elimina lo que cae fuera)                ║
║    --ensure-meta            Añade compás 4/4 y tempo 120 si faltan           ║
║    --time-sig N/D --key K --tempo BPM   Valores a usar con --ensure-meta     ║
║                                                                              ║
║  OPCIONES DE meta:                                                           ║
║    --copy-from ORIG         Copia compás/armadura/tempo (con todos sus       ║
║                             cambios) de ORIG; reescala si cambia el tpb      ║
║    --only LISTA             Qué copiar/tocar: ts,key,tempo (default: los tres)║
║    --time-sig N/D --key K --tempo BPM   Fija valores a mano                  ║
║    --replace                Sustituye los existentes (por defecto NO destruye:║
║                             solo añade lo que falta)                         ║
║                                                                              ║
║  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 fichero ilegible ·             ║
║                     3 hallazgos con --strict                                 ║
║  (Los hallazgos son informativos: sin --strict `check` devuelve 0.)          ║
║                                                                              ║
║  API:  from midi_doctor import check_file, fix_midi, clamp_pitch             ║
║                                                                              ║
║  DEPENDENCIAS: mido                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import mido
    from mido import MetaMessage, MidiFile, MidiTrack
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(k: str) -> str:
    return _COLORS.get(k, "") if _USE_COLOR else ""


SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}
META_TYPES = {"ts": "time_signature", "key": "key_signature", "tempo": "set_tempo"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ══════════════════════════════════════════════════════════════════════════════
#  REGISTROS GM (límites prácticos aproximados; solo avisos)
# ══════════════════════════════════════════════════════════════════════════════

def _gm_ranges() -> Dict[int, Tuple[int, int]]:
    r: Dict[int, Tuple[int, int]] = {}
    for p in range(0, 8):   r[p] = (21, 108)     # pianos
    for p in range(8, 16):  r[p] = (36, 96)      # percusión cromática
    for p in range(16, 24): r[p] = (36, 96)      # órganos
    for p in range(24, 32): r[p] = (40, 88)      # guitarras
    for p in range(32, 40): r[p] = (28, 67)      # bajos
    r[40] = (55, 103); r[41] = (48, 91); r[42] = (36, 76); r[43] = (28, 67)
    r[44] = (36, 103); r[45] = (36, 96); r[46] = (24, 103); r[47] = (36, 65)
    for p in range(48, 56): r[p] = (28, 103)     # conjuntos y coros
    r[56] = (54, 84); r[57] = (40, 72); r[58] = (28, 58); r[59] = (54, 84)
    r[60] = (34, 77)
    for p in range(61, 64): r[p] = (34, 84)
    for p in range(64, 68): r[p] = (49, 93)      # saxos
    r[68] = (58, 91); r[69] = (52, 81); r[70] = (34, 75); r[71] = (50, 94)
    r[72] = (74, 108); r[73] = (60, 96); r[74] = (60, 93)
    for p in range(75, 80): r[p] = (55, 96)      # flautas varias
    return r


GM_RANGES = _gm_ranges()
MONO_PROGRAMS = set([56, 57, 58, 59, 60] + list(range(64, 80)))


def clamp_pitch(p: int, lo: int = 0, hi: int = 127) -> int:
    """Pliega una altura al rango [lo, hi] por octavas (útil como guarda en generadores)."""
    p = int(p)
    if hi - lo < 11:
        return max(lo, min(hi, p))
    while p > hi:
        p -= 12
    while p < lo:
        p += 12
    return max(lo, min(hi, p))


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA TOLERANTE
# ══════════════════════════════════════════════════════════════════════════════

def load_tolerant(path: str) -> Tuple[Optional[MidiFile], List[Dict]]:
    """Carga un MIDI; si falla el modo estricto reintenta con clip=True."""
    issues: List[Dict] = []
    try:
        return MidiFile(path), issues
    except Exception as e_strict:  # noqa: BLE001
        try:
            mid = MidiFile(path, clip=True)
            issues.append(_issue("E_BYTES", "error", None, None, 0,
                                 f"datos fuera de 0..127 (recortados al leer): {str(e_strict) or type(e_strict).__name__}"))
            return mid, issues
        except Exception as e:  # noqa: BLE001
            issues.append(_issue("E_PARSE", "error", None, None, 0, f"no se puede leer ({type(e).__name__}): {str(e) or 'fichero truncado o corrupto'}"))
            return None, issues


def _issue(code: str, sev: str, track, channel, tick: int, msg: str,
           fixable: bool = False, bar: Optional[float] = None) -> Dict:
    d = {"code": code, "severity": sev, "track": track, "channel": channel,
         "tick": tick, "message": msg, "fixable": fixable}
    if bar is not None:
        d["bar"] = round(bar, 2)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  META
# ══════════════════════════════════════════════════════════════════════════════

def abs_events(track) -> List[Tuple[int, object]]:
    t = 0
    out = []
    for m in track:
        t += m.time
        out.append((t, m))
    return out


def collect_meta(mid: MidiFile) -> Dict[str, List[Tuple[int, object]]]:
    res = {k: [] for k in META_TYPES}
    for trk in mid.tracks:
        for tick, m in abs_events(trk):
            for k, tname in META_TYPES.items():
                if m.type == tname:
                    res[k].append((tick, m))
    for k in res:
        res[k].sort(key=lambda e: e[0])
    return res


def fmt_meta(kind: str, m) -> str:
    if kind == "ts":
        return f"{m.numerator}/{m.denominator}"
    if kind == "key":
        return m.key
    return f"{mido.tempo2bpm(m.tempo):.1f} BPM"


def bar_ticks_at(mid: MidiFile, tick: int, ts_list) -> int:
    num, den = 4, 4
    for t, m in ts_list:
        if t <= tick:
            num, den = m.numerator, m.denominator
    return int(mid.ticks_per_beat * num * 4 / den)


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def _extract_track_notes(trk, ti: int):
    """Devuelve (notas, huérfanos, apertura_sin_cierre, ticks_fin).
    nota = dict(start,end,pitch,vel,ch)."""
    notes, orphans, stuck = [], [], []
    active: Dict[Tuple[int, int], dict] = {}
    t = 0
    overlaps, dups = [], []
    for m in trk:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            key = (m.channel, m.note)
            if key in active and active[key]["start"] == t:
                active[key]["dups"] += 1          # nota duplicada en el mismo tick
                dups.append((t, m.channel, m.note))
                continue
            if key in active:
                prev = active.pop(key)
                overlaps.append((t, m.channel, m.note))
                prev["end"] = t
                notes.append(prev)
            active[key] = {"start": t, "end": None, "pitch": m.note, "vel": m.velocity,
                           "ch": m.channel, "dups": 0}
        elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
            key = (m.channel, m.note)
            if key in active:
                if active[key]["dups"] > 0:
                    active[key]["dups"] -= 1
                    continue
                n = active.pop(key)
                n["end"] = t
                notes.append(n)
            else:
                orphans.append((t, m.channel, m.note))
    for n in active.values():
        stuck.append(n)
    return notes, orphans, stuck, overlaps, dups, t


def analyze(mid: MidiFile) -> Dict:
    issues: List[Dict] = []
    meta = collect_meta(mid)
    tpb = mid.ticks_per_beat
    if tpb < 24 or tpb > 4000:
        issues.append(_issue("W_TPB", "warn", None, None, 0,
                             f"ticks_per_beat inusual: {tpb}"))
    bar0 = bar_ticks_at(mid, 0, meta["ts"])
    stats = {"tracks": len(mid.tracks), "notes": 0, "ticks_per_beat": tpb}

    if not meta["ts"]:
        issues.append(_issue("W_NO_TS", "warn", None, None, 0,
                             "sin time_signature (los analizadores asumirán 4/4)", True))
    if not meta["key"]:
        issues.append(_issue("W_NO_KS", "info", None, None, 0,
                             "sin key_signature", True))
    if not meta["tempo"]:
        issues.append(_issue("W_NO_TEMPO", "warn", None, None, 0,
                             "sin set_tempo (se asumirá 120 BPM)", True))

    # compás que cambia fuera de barra
    prev_tick, prev_bar = 0, bar0
    for t, m in meta["ts"]:
        if t > 0:
            if prev_bar and (t - prev_tick) % prev_bar != 0:
                issues.append(_issue("W_TS_ALIGN", "warn", None, None, t,
                                     f"cambio a {m.numerator}/{m.denominator} fuera de barra de compás",
                                     bar=t / bar0))
        prev_tick = t
        prev_bar = int(tpb * m.numerator * 4 / m.denominator)

    for ti, trk in enumerate(mid.tracks):
        evs = abs_events(trk)
        has_eot = any(m.type == "end_of_track" for _, m in evs)
        if not has_eot:
            issues.append(_issue("W_NO_EOT", "warn", ti, None, evs[-1][0] if evs else 0,
                                 "pista sin end_of_track", True))
        seen = set()
        for t, m in evs:
            if m.is_meta and m.type != "end_of_track":
                key = (t, m.type, str({k: v for k, v in m.dict().items() if k != "time"}))
                if key in seen:
                    issues.append(_issue("W_DUP_META", "warn", ti, None, t,
                                         f"meta '{m.type}' duplicado", True))
                seen.add(key)

        notes, orphans, stuck, overlaps, dups, endt = _extract_track_notes(trk, ti)
        stats["notes"] += len(notes) + len(stuck)
        program: Dict[int, int] = {}
        for _, m in evs:
            if m.type == "program_change":
                program.setdefault(m.channel, m.program)
        if not notes and not stuck and not any(not m.is_meta for _, m in evs):
            if not any(m.is_meta and m.type in META_TYPES.values() for _, m in evs):
                issues.append(_issue("W_EMPTY", "info", ti, None, 0, "pista vacía"))

        for (t, ch, p) in orphans:
            issues.append(_issue("W_ORPHAN", "warn", ti, ch, t,
                                 f"note_off sin note_on ({NOTE_NAMES[p % 12]}{p // 12 - 1})",
                                 True, t / bar0))
        for n in stuck:
            issues.append(_issue("W_STUCK", "warn", ti, n["ch"], n["start"],
                                 f"nota {NOTE_NAMES[n['pitch'] % 12]}{n['pitch'] // 12 - 1} nunca se cierra",
                                 True, n["start"] / bar0))
        if dups:
            t0, ch0, p0 = dups[0]
            issues.append(_issue("W_DUP_NOTE", "warn", ti, ch0, t0,
                                 f"{len(dups)} nota(s) duplicada(s) en el mismo tick "
                                 f"(p.ej. {NOTE_NAMES[p0 % 12]}{p0 // 12 - 1})", True, t0 / bar0))
        for (t, ch, p) in overlaps:
            issues.append(_issue("W_OVERLAP", "warn", ti, ch, t,
                                 f"{NOTE_NAMES[p % 12]}{p // 12 - 1} re-atacada sin cerrar", True,
                                 t / bar0))
        for n in notes:
            if n["end"] <= n["start"]:
                issues.append(_issue("W_ZERO", "warn", ti, n["ch"], n["start"],
                                     f"nota de duración 0 ({NOTE_NAMES[n['pitch'] % 12]}"
                                     f"{n['pitch'] // 12 - 1})", True, n["start"] / bar0))

        # registro por instrumento (agrupado: un aviso por pista/canal)
        by_ch = defaultdict(list)
        for n in notes + stuck:
            by_ch[n["ch"]].append(n)
        for ch, lst in by_ch.items():
            if ch == 9:
                continue
            prog = program.get(ch, 0)
            rng = GM_RANGES.get(prog)
            if rng:
                lo, hi = rng
                bad = [n for n in lst if n["pitch"] < lo or n["pitch"] > hi]
                if bad:
                    first = min(bad, key=lambda n: n["start"])
                    issues.append(_issue(
                        "W_RANGE", "warn", ti, ch, first["start"],
                        f"{len(bad)}/{len(lst)} notas fuera del registro del programa {prog} "
                        f"({lo}-{hi}); mín {min(n['pitch'] for n in lst)} "
                        f"máx {max(n['pitch'] for n in lst)}", True, first["start"] / bar0))
            if program.get(ch, 0) in MONO_PROGRAMS:
                ev = sorted([(n["start"], 1) for n in lst] + [(n["end"] or n["start"], -1) for n in lst])
                cur = peak = 0
                for _, d in ev:
                    cur += d
                    peak = max(peak, cur)
                if peak > 1:
                    issues.append(_issue("W_MONO", "warn", ti, ch, 0,
                                         f"{peak} voces simultáneas en un instrumento "
                                         f"monofónico (programa {program[ch]})"))
    issues.sort(key=lambda d: (SEVERITY_ORDER[d["severity"]], d["code"], d["tick"]))
    return {"issues": issues, "stats": stats,
            "meta": {k: [(t, fmt_meta(k, m)) for t, m in v] for k, v in meta.items()}}


def check_file(path: str) -> Dict:
    mid, load_issues = load_tolerant(path)
    res = {"file": path, "issues": list(load_issues), "stats": {}, "meta": {}}
    if mid is None:
        return res
    a = analyze(mid)
    res["issues"] += a["issues"]
    res["stats"], res["meta"] = a["stats"], a["meta"]
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  REPARACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _order_key(m) -> int:
    if m.is_meta:
        return 9 if m.type == "end_of_track" else 0
    if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
        return 1
    if m.type == "note_on":
        return 3
    return 2


def _fold(p: int, lo: int, hi: int) -> int:
    return clamp_pitch(p, lo, hi)


def fix_midi(mid: MidiFile, min_dur: float = 0.0625, drop_zero: bool = False,
             instrument_range: str = "none", ensure_meta: bool = False,
             time_sig: str = "4/4", key: Optional[str] = None,
             tempo_bpm: float = 120.0) -> Tuple[MidiFile, List[str]]:
    log: List[str] = []
    tpb = mid.ticks_per_beat
    min_ticks = max(1, int(round(min_dur * tpb)))
    new = MidiFile(type=mid.type, ticks_per_beat=tpb)
    counters = defaultdict(int)
    file_end = 0
    for trk in mid.tracks:
        t = 0
        for m in trk:
            t += m.time
        file_end = max(file_end, t)

    for ti, trk in enumerate(mid.tracks):
        evs = abs_events(trk)
        program: Dict[int, int] = {}
        for _, m in evs:
            if m.type == "program_change":
                program.setdefault(m.channel, m.program)
        out: List[Tuple[int, int, int, object]] = []   # (tick, prio, seq, msg)
        seq = 0
        active: Dict[Tuple[int, int], dict] = {}
        notes = []
        seen_meta = set()
        track_end = 0
        for t, m in evs:
            track_end = max(track_end, t)
            if m.type == "end_of_track":
                continue
            if m.type == "note_on" and m.velocity > 0:
                key_ = (m.channel, m.note)
                if key_ in active and active[key_]["start"] == t:
                    active[key_]["dups"] += 1
                    counters["dup_notes"] += 1
                    continue
                if key_ in active:
                    prev = active.pop(key_)
                    prev["end"] = t
                    notes.append(prev)
                    counters["overlap"] += 1
                active[key_] = {"start": t, "end": None, "pitch": m.note,
                                "vel": m.velocity, "ch": m.channel, "dups": 0}
            elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
                key_ = (m.channel, m.note)
                if key_ in active:
                    if active[key_]["dups"] > 0:
                        active[key_]["dups"] -= 1
                        continue
                    n = active.pop(key_)
                    n["end"] = t
                    notes.append(n)
                else:
                    counters["orphan"] += 1
            else:
                if m.is_meta:
                    k = (t, m.type, str({a: b for a, b in m.dict().items() if a != "time"}))
                    if k in seen_meta:
                        counters["dup_meta"] += 1
                        continue
                    seen_meta.add(k)
                out.append((t, _order_key(m), seq, m.copy(time=0)))
                seq += 1
        for n in active.values():
            n["end"] = max(file_end, n["start"] + min_ticks)
            counters["stuck"] += 1
            notes.append(n)

        for n in notes:
            if n["end"] <= n["start"]:
                if drop_zero:
                    counters["zero_dropped"] += 1
                    continue
                n["end"] = n["start"] + min_ticks
                counters["zero_extended"] += 1
            p = n["pitch"]
            if instrument_range in ("fold", "drop") and n["ch"] != 9:
                rng = GM_RANGES.get(program.get(n["ch"], 0))
                if rng and not (rng[0] <= p <= rng[1]):
                    if instrument_range == "drop":
                        counters["range_dropped"] += 1
                        continue
                    p = _fold(p, *rng)
                    counters["range_folded"] += 1
            out.append((n["start"], 3, seq, mido.Message(
                "note_on", channel=n["ch"], note=p, velocity=max(1, min(127, n["vel"])), time=0)))
            seq += 1
            out.append((n["end"], 1, seq, mido.Message(
                "note_off", channel=n["ch"], note=p, velocity=0, time=0)))
            seq += 1
            track_end = max(track_end, n["end"])

        out.sort(key=lambda e: (e[0], e[1], e[2]))
        nt = MidiTrack()
        last = 0
        for tick, _, _, msg in out:
            nt.append(msg.copy(time=tick - last))
            last = tick
        nt.append(MetaMessage("end_of_track", time=max(0, track_end - last)))
        if not any(m.type == "end_of_track" for _, m in evs):
            counters["eot_added"] += 1
        new.tracks.append(nt)

    if ensure_meta:
        meta = collect_meta(new)
        n, d = (int(x) for x in time_sig.split("/"))
        add = []
        if not meta["ts"]:
            add.append((0, MetaMessage("time_signature", numerator=n, denominator=d)))
        if not meta["tempo"]:
            add.append((0, MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm))))
        if key and not meta["key"]:
            add.append((0, MetaMessage("key_signature", key=key)))
        if add:
            _insert_meta(new, add)
            counters["meta_added"] += len(add)
    names = {"dup_notes": "notas duplicadas eliminadas", "overlap": "solapes recortados", "orphan": "note_off huérfanos eliminados",
             "dup_meta": "metas duplicados eliminados", "stuck": "notas sin cerrar cerradas",
             "zero_extended": "notas de duración 0 alargadas",
             "zero_dropped": "notas de duración 0 eliminadas",
             "range_folded": "notas plegadas al registro", "range_dropped": "notas fuera de registro eliminadas",
             "eot_added": "end_of_track añadidos", "meta_added": "metadatos añadidos"}
    for k, v in counters.items():
        log.append(f"{names.get(k, k)}: {v}")
    return new, log


def _insert_meta(mid: MidiFile, items: List[Tuple[int, object]]) -> None:
    """Inserta metas (tick abs, msg) en la pista 0 respetando el orden temporal."""
    if not mid.tracks:
        mid.tracks.append(MidiTrack())
    trk = mid.tracks[0]
    evs = [(t, _order_key(m), i, m) for i, (t, m) in enumerate(abs_events(trk))
           if m.type != "end_of_track"]
    end = sum(m.time for m in trk)
    base = len(evs)
    for j, (t, m) in enumerate(items):
        evs.append((t, 0, base + j, m.copy(time=0)))
        end = max(end, t)
    evs.sort(key=lambda e: (e[0], e[1], e[2]))
    nt = MidiTrack()
    last = 0
    for t, _, _, m in evs:
        nt.append(m.copy(time=t - last))
        last = t
    nt.append(MetaMessage("end_of_track", time=max(0, end - last)))
    mid.tracks[0] = nt


def apply_meta(target: MidiFile, items: Dict[str, List[Tuple[int, object]]],
               replace: bool = False) -> List[str]:
    """items: {kind: [(tick, msg)]} ya en los ticks de `target`."""
    log = []
    cur = collect_meta(target)
    to_add: List[Tuple[int, object]] = []
    for kind, lst in items.items():
        if not lst:
            continue
        if cur[kind] and not replace:
            log.append(f"{kind}: ya presente en destino, se respeta (usa --replace para sustituir)")
            continue
        if cur[kind] and replace:
            tname = META_TYPES[kind]
            for i, trk in enumerate(target.tracks):
                evs = [(t, m) for t, m in abs_events(trk) if m.type != tname]
                nt = MidiTrack()
                last = 0
                for t, m in evs:
                    nt.append(m.copy(time=t - last))
                    last = t
                target.tracks[i] = nt
            log.append(f"{kind}: {len(cur[kind])} existente(s) sustituido(s)")
        to_add += lst
        log.append(f"{kind}: añadidos {len(lst)}  ({', '.join(fmt_meta(kind, m) for _, m in lst[:4])}"
                   f"{'…' if len(lst) > 4 else ''})")
    if to_add:
        _insert_meta(target, to_add)
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN / E/S
# ══════════════════════════════════════════════════════════════════════════════

def expand_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [os.path.join(root, f) for f in sorted(fs)
                          if f.lower().endswith((".mid", ".midi"))]
            continue
        hits = sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
        files += [h for h in hits if os.path.isfile(h)]
    return list(dict.fromkeys(files))


def save_atomic(mid: MidiFile, out: str) -> None:
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mid", dir=d)
    os.close(fd)
    try:
        mid.save(tmp)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def default_out(path: str, suffix: str) -> str:
    stem, _ = os.path.splitext(path)
    return f"{stem}{suffix}.mid"


def print_check(res: Dict, verbose: bool) -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    name = os.path.relpath(res["file"]) if os.path.exists(res["file"]) else res["file"]
    iss = res["issues"]
    n_err = sum(i["severity"] == "error" for i in iss)
    n_warn = sum(i["severity"] == "warn" for i in iss)
    st = res.get("stats", {})
    head = (f"{_c('GRN')}✔ limpio{R}" if not iss else
            f"{_c('RED') if n_err else _c('YEL')}{n_err} error(es), {n_warn} aviso(s){R}")
    print(f"\n  {B}{name}{R}  {G}{st.get('tracks', '?')} pistas · {st.get('notes', '?')} notas{R}  {head}")
    meta = res.get("meta", {})
    if meta:
        parts = []
        for k, lab in (("ts", "compás"), ("key", "armadura"), ("tempo", "tempo")):
            vs = meta.get(k, [])
            parts.append(f"{lab}: " + (", ".join(v for _, v in vs[:3]) + ("…" if len(vs) > 3 else "")
                                       if vs else "—"))
        print(f"  {G}{' · '.join(parts)}{R}")
    # agrupa por código
    groups = defaultdict(list)
    for i in iss:
        groups[(i["severity"], i["code"])].append(i)
    for (sev, code), lst in sorted(groups.items(), key=lambda kv: (SEVERITY_ORDER[kv[0][0]], kv[0][1])):
        col = _c("RED") if sev == "error" else (_c("YEL") if sev == "warn" else _c("CYA"))
        shown = lst if verbose else lst[:3]
        for i in shown:
            where = []
            if i["track"] is not None:
                where.append(f"pista {i['track']}")
            if i["channel"] is not None:
                where.append(f"canal {i['channel']}")
            if i.get("bar") is not None:
                where.append(f"compás {i['bar'] + 1:.1f}")
            loc = f" {G}[{', '.join(where)}]{R}" if where else ""
            fx = f" {G}(reparable){R}" if i["fixable"] else ""
            print(f"    {col}{code:<11}{R} {i['message']}{loc}{fx}")
        if len(lst) > len(shown):
            print(f"    {G}{'':<11} … y {len(lst) - len(shown)} más (usa --verbose){R}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_check(a) -> int:
    files = expand_inputs(a.midis)
    if not files:
        print("[ERROR] no se encontró ningún MIDI", file=sys.stderr)
        return 1
    results = [check_file(f) for f in files]
    B, R = _c("B"), _c("R")
    print(f"\n{'═' * 78}\n  {B}MIDI DOCTOR · check{R}  ({len(files)} fichero(s))\n{'═' * 78}")
    for r in results:
        print_check(r, a.verbose)
    n_err = sum(any(i["severity"] == "error" for i in r["issues"]) for r in results)
    n_warn = sum(any(i["severity"] == "warn" for i in r["issues"]) for r in results)
    clean = sum(not r["issues"] for r in results)
    print(f"\n{'─' * 78}\n  {len(results)} ficheros: {_c('GRN')}{clean} limpios{R} · "
          f"{n_warn} con avisos · {_c('RED')}{n_err} con errores{R}\n")
    if a.json:
        with open(_ensure_parent(a.json), "w", encoding="utf-8") as fh:
            json.dump({"version": VERSION, "results": results}, fh, ensure_ascii=False, indent=2)
        print(f"  {_c('CYA')}JSON: {a.json}{R}")
    unreadable = any(any(i["code"] == "E_PARSE" for i in r["issues"]) for r in results)
    if unreadable and len(results) == 1:
        return 2
    if a.strict and any(r["issues"] for r in results):
        return 3
    return 0


def cmd_fix(a) -> int:
    files = expand_inputs(a.midis)
    if not files:
        print("[ERROR] no se encontró ningún MIDI", file=sys.stderr)
        return 1
    if a.output and len(files) > 1:
        print("[ERROR] --output solo vale con un único fichero", file=sys.stderr)
        return 1
    rc = 0
    for f in files:
        mid, load_issues = load_tolerant(f)
        if mid is None:
            print(f"  {_c('RED')}✗ {f}: ilegible{_c('R')}", file=sys.stderr)
            rc = 2
            continue
        new, log = fix_midi(mid, a.min_dur, a.drop_zero, a.instrument_range,
                            a.ensure_meta, a.time_sig, a.key, a.tempo)
        if any(i["code"] == "E_BYTES" for i in load_issues):
            log.insert(0, "bytes de dato fuera de 0..127 recortados al leer")
        out = f if a.inplace else (a.output or default_out(f, "_fixed"))
        if a.inplace:
            shutil.copyfile(f, _ensure_parent(f + ".bak"))
        save_atomic(new, out)
        after = check_file(out)
        left = [i for i in after["issues"] if i["severity"] != "info"]
        print(f"\n  {_c('B')}{os.path.relpath(f)}{_c('R')} → {out}")
        for l in log or ["nada que reparar"]:
            print(f"    · {l}")
        if left:
            print(f"    {_c('YEL')}quedan {len(left)} aviso(s) no reparables automáticamente "
                  f"({', '.join(sorted({i['code'] for i in left}))}){_c('R')}")
        else:
            print(f"    {_c('GRN')}✔ sin avisos tras la reparación{_c('R')}")
    print()
    return rc


def _parse_only(s: Optional[str]) -> List[str]:
    if not s:
        return list(META_TYPES)
    ks = [x.strip() for x in s.split(",") if x.strip()]
    bad = [k for k in ks if k not in META_TYPES]
    if bad:
        raise ValueError(f"--only: valores no válidos {bad} (usa ts,key,tempo)")
    return ks


def cmd_meta(a) -> int:
    mid, issues = load_tolerant(a.midi)
    if mid is None:
        print(f"[ERROR] no se puede leer {a.midi}", file=sys.stderr)
        return 2
    kinds = _parse_only(a.only)
    manual = a.time_sig or a.key or a.tempo
    if not a.copy_from and not manual:
        m = collect_meta(mid)
        print(f"\n  {_c('B')}{a.midi}{_c('R')}  (tpb {mid.ticks_per_beat})")
        for k, lab in (("ts", "Compás"), ("key", "Armadura"), ("tempo", "Tempo")):
            vs = m[k]
            print(f"    {lab:<9}" + ("—" if not vs else
                  "  ".join(f"{fmt_meta(k, x)}@{t}" for t, x in vs)))
        print()
        return 0
    items: Dict[str, List[Tuple[int, object]]] = {k: [] for k in META_TYPES}
    if a.copy_from:
        src, _ = load_tolerant(a.copy_from)
        if src is None:
            print(f"[ERROR] no se puede leer {a.copy_from}", file=sys.stderr)
            return 2
        sm = collect_meta(src)
        scale = mid.ticks_per_beat / src.ticks_per_beat
        for k in kinds:
            items[k] = [(int(round(t * scale)), m) for t, m in sm[k]]
    if a.time_sig:
        n, d = (int(x) for x in a.time_sig.split("/"))
        items["ts"] = [(0, MetaMessage("time_signature", numerator=n, denominator=d))]
    if a.key:
        items["key"] = [(0, MetaMessage("key_signature", key=a.key))]
    if a.tempo:
        items["tempo"] = [(0, MetaMessage("set_tempo", tempo=mido.bpm2tempo(a.tempo)))]
    log = apply_meta(mid, items, replace=a.replace)
    out = a.output or default_out(a.midi, "_meta")
    save_atomic(mid, out)
    print(f"\n  {_c('B')}{a.midi}{_c('R')} → {out}")
    for l in log or ["nada que hacer"]:
        print(f"    · {l}")
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="midi_doctor.py",
        description="Diagnóstico y reparación de MIDIs (check / fix / meta).")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    c = sub.add_parser("check", help="diagnostica MIDIs")
    c.add_argument("midis", nargs="+")
    c.add_argument("--json", default=None)
    c.add_argument("--strict", action="store_true", help="rc=3 si hay hallazgos")
    c.add_argument("--verbose", "-v", action="store_true")
    c.add_argument("--no-color", action="store_true")
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("fix", help="repara y guarda un MIDI nuevo")
    f.add_argument("midis", nargs="+")
    f.add_argument("--output", "-o", default=None)
    f.add_argument("--inplace", action="store_true")
    f.add_argument("--min-dur", type=float, default=0.0625)
    f.add_argument("--drop-zero", action="store_true")
    f.add_argument("--instrument-range", choices=["none", "fold", "drop"], default="none")
    f.add_argument("--ensure-meta", action="store_true")
    f.add_argument("--time-sig", default="4/4")
    f.add_argument("--key", default=None)
    f.add_argument("--tempo", type=float, default=120.0)
    f.add_argument("--no-color", action="store_true")
    f.set_defaults(fn=cmd_fix)

    m = sub.add_parser("meta", help="muestra, añade o copia compás/armadura/tempo")
    m.add_argument("midi")
    m.add_argument("--output", "-o", default=None)
    m.add_argument("--copy-from", default=None)
    m.add_argument("--only", default=None, help="ts,key,tempo")
    m.add_argument("--time-sig", default=None)
    m.add_argument("--key", default=None)
    m.add_argument("--tempo", type=float, default=None)
    m.add_argument("--replace", action="store_true")
    m.add_argument("--no-color", action="store_true")
    m.set_defaults(fn=cmd_meta)
    return ap


def main(argv=None) -> int:
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[interrumpido]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
