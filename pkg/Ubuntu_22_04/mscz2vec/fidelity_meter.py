#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FIDELITY METER  v1.0                                  ║
║        ¿Cuánto se parece una versión a su original? (y por qué)              ║
║                                                                              ║
║  quality_scorer y melody_critic puntúan la CALIDAD de una pieza, no su       ║
║  PARECIDO con otra: un bucle que solo optimice calidad puede alejarse del    ║
║  original sin darse cuenta. Esta herramienta mide fidelidad con métricas     ║
║  robustas al rubato y al acompañamiento añadido, y declara siempre la        ║
║  tolerancia temporal que usa (para no confundir expresividad con pérdida).   ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    compare   Compara un original con una versión (tabla + veredicto)         ║
║    batch     Compara muchas carpetas (original.mid vs obra_final.mid…)       ║
║    rank      Ordena varias versiones de un mismo original por fidelidad      ║
║    gate      Puerta de calidad: rc=0 si cumple los umbrales, rc=3 si no      ║
║                                                                              ║
║  USO:                                                                        ║
║    python fidelity_meter.py compare original.mid version.mid                 ║
║    python fidelity_meter.py compare o.mid v.mid --tol 0.5 --timeline         ║
║    python fidelity_meter.py batch reconstrucciones/*/ --cand obra_final.mid  ║
║    python fidelity_meter.py batch pieza_*/ --cand obra_final.mid \\           ║
║        --cand obra_fiel.mid --md comparacion.md --json comparacion.json      ║
║    python fidelity_meter.py rank original.mid cand_*.mid                     ║
║    python fidelity_meter.py gate original.mid version.mid --min-melody 0.8   ║
║                                                                              ║
║  MÉTRICAS (0-1, mayor = más parecido):                                       ║
║    melody   n-gramas de la línea superior ("skyline": nota más aguda por     ║
║             ventana de 0.5 pulso, sin percusión). Insensible al rubato y al  ║
║             acompañamiento. ES LA MÉTRICA PRINCIPAL DE IDENTIDAD.            ║
║    onset    notas del original (inicio+altura) presentes en la versión       ║
║             dentro de ±TOL pulsos. Se imprime a 3 tolerancias                ║
║             (0.1 / TOL / 0.5) para ver cuánto es rubato y cuánto pérdida.    ║
║    rhythm   posiciones de ataque del original presentes en la versión        ║
║             (ignora la altura) dentro de ±TOL pulsos.                        ║
║    pitch    recall del multiconjunto de alturas (DÉBIL: las escalas se       ║
║             solapan; solo informativo).                                      ║
║    chroma   similitud coseno del perfil de clases de altura (tonalidad).     ║
║    length   cociente de duración versión/original (1.0 = misma longitud).    ║
║    instr    instrumentos GM del original presentes en la versión (0-1).      ║
║    meta     compás y armadura conservados (secuencias iguales).              ║
║    score    0.50·melody + 0.20·rhythm + 0.10·chroma + 0.10·instr + 0.10·meta ║
║                                                                              ║
║  VEREDICTO (según melody): ≥0.85 fiel · ≥0.50 variación · ≥0.20 inspirada ·  ║
║  <0.20 otra pieza                                                            ║
║                                                                              ║
║  OPCIONES COMUNES:                                                           ║
║    --tol F        Tolerancia temporal en pulsos (default 0.25)               ║
║    --window F     Ventana del skyline en pulsos (default 0.5)                ║
║    --ngram N      Tamaño de n-grama de la melodía (default 2)                ║
║    --json FILE    Informe JSON                                               ║
║    --no-color     Sin colores ANSI                                           ║
║                                                                              ║
║  OPCIONES DE compare: --timeline  (fidelidad melódica compás a compás)       ║
║  OPCIONES DE batch:   --cand NOMBRE (repetible; default obra_final.mid)      ║
║                       --orig NOMBRE (default original.mid)  --md FILE        ║
║                       --csv FILE                                             ║
║  OPCIONES DE gate:    --min-melody F  --min-rhythm F  --min-score F          ║
║                       --min-chroma F  --require-meta  --require-instr        ║
║                       --length-range LO HI  (default sin comprobar)          ║
║                                                                              ║
║  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso/ficheros · 3 gate no superado      ║
║                                                                              ║
║  API:  from fidelity_meter import measure   # measure(orig, cand) -> dict    ║
║                                                                              ║
║  DEPENDENCIAS: mido                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import bisect
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
PERC_CH = 9
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

WEIGHTS = {"melody": 0.50, "rhythm": 0.20, "chroma": 0.10, "instr": 0.10, "meta": 0.10}


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


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA
# ══════════════════════════════════════════════════════════════════════════════

class Piece:
    """Resumen de un MIDI: notas en pulsos + metadatos."""

    def __init__(self, path: str):
        self.path = path
        try:
            mid = mido.MidiFile(path)
        except Exception:  # noqa: BLE001  (bytes fuera de rango, etc.)
            mid = mido.MidiFile(path, clip=True)
        tpb = mid.ticks_per_beat or 480
        self.notes: List[Tuple[float, float, int, int]] = []   # (start, end, pitch, ch)
        self.programs = set()
        self.ts: List[Tuple[int, int]] = []
        self.ks: List[str] = []
        self.tempo: Optional[float] = None
        ch_with_notes = set()
        ch_prog: Dict[int, int] = {}
        for trk in mid.tracks:
            t = 0
            active: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for m in trk:
                t += m.time
                if m.type == "time_signature":
                    v = (m.numerator, m.denominator)
                    if not self.ts or self.ts[-1] != v:
                        self.ts.append(v)
                elif m.type == "key_signature":
                    if not self.ks or self.ks[-1] != m.key:
                        self.ks.append(m.key)
                elif m.type == "set_tempo" and self.tempo is None:
                    self.tempo = mido.tempo2bpm(m.tempo)
                elif m.type == "program_change":
                    ch_prog.setdefault(m.channel, m.program)
                elif m.type == "note_on" and m.velocity > 0:
                    active[(m.channel, m.note)].append(t)
                    ch_with_notes.add(m.channel)
                elif m.type in ("note_off", "note_on"):
                    st = active.get((m.channel, m.note))
                    if st:
                        s = st.pop(0)
                        self.notes.append((s / tpb, max(t, s) / tpb, m.note, m.channel))
            for (ch, p), sts in active.items():
                for s in sts:
                    self.notes.append((s / tpb, s / tpb + 0.25, p, ch))
        self.notes.sort()
        self.tpb = tpb
        for ch in ch_with_notes:
            if ch != PERC_CH:
                self.programs.add(ch_prog.get(ch, 0))
        self.melodic = [n for n in self.notes if n[3] != PERC_CH]
        self.length = max((n[1] for n in self.notes), default=0.0)
        num, den = self.ts[0] if self.ts else (4, 4)
        self.beats_per_bar = num * 4 / den

    def skyline(self, window: float = 0.5) -> List[Tuple[int, int]]:
        """[(bin, pitch más agudo)] ordenado por tiempo."""
        d: Dict[int, int] = {}
        for s, _, p, _ in self.melodic:
            k = int(round(s / window))
            if p > d.get(k, -1):
                d[k] = p
        return sorted(d.items())


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def _ngram_recall(a: List[int], b: List[int], n: int) -> Optional[float]:
    if len(a) < n:
        return None
    A = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    B = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    return sum((A & B).values()) / max(1, sum(A.values()))


def melody_recall(a: Piece, b: Piece, window: float, n: int) -> Optional[float]:
    sa = [p for _, p in a.skyline(window)]
    sb = [p for _, p in b.skyline(window)]
    return _ngram_recall(sa, sb, n)


def onset_recall(a: Piece, b: Piece, tol: float, use_pitch: bool = True) -> Optional[float]:
    """Fracción de notas de A con una nota de B (misma altura si use_pitch) a ≤ tol pulsos.
    Emparejamiento voraz uno a uno."""
    if not a.melodic:
        return None
    idx: Dict[int, List[float]] = defaultdict(list)
    for s, _, p, _ in b.melodic:
        idx[p if use_pitch else 0].append(s)
    for v in idx.values():
        v.sort()
    used: Dict[int, set] = defaultdict(set)
    hit = 0
    for s, _, p, _ in a.melodic:
        key = p if use_pitch else 0
        lst = idx.get(key)
        if not lst:
            continue
        i = bisect.bisect_left(lst, s - tol - 1e-9)
        best, bd = None, None
        while i < len(lst) and lst[i] <= s + tol + 1e-9:
            if i not in used[key]:
                d = abs(lst[i] - s)
                if bd is None or d < bd:
                    best, bd = i, d
            i += 1
        if best is not None:
            used[key].add(best)
            hit += 1
    return hit / len(a.melodic)


def pitch_recall(a: Piece, b: Piece) -> Optional[float]:
    if not a.melodic:
        return None
    A = Counter(p for _, _, p, _ in a.melodic)
    B = Counter(p for _, _, p, _ in b.melodic)
    return sum((A & B).values()) / sum(A.values())


def chroma_sim(a: Piece, b: Piece) -> Optional[float]:
    ca, cb = [0.0] * 12, [0.0] * 12
    for s, e, p, _ in a.melodic:
        ca[p % 12] += max(e - s, 0.05)
    for s, e, p, _ in b.melodic:
        cb[p % 12] += max(e - s, 0.05)
    na, nb = math.sqrt(sum(v * v for v in ca)), math.sqrt(sum(v * v for v in cb))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(ca, cb)) / (na * nb)


def timeline(a: Piece, b: Piece, window: float, n: int) -> List[Optional[float]]:
    """Recall melódico compás a compás (según el compás inicial del original)."""
    bar = a.beats_per_bar
    nb = int(math.ceil(a.length / bar)) if bar else 0
    out: List[Optional[float]] = []
    sa, sb = a.skyline(window), b.skyline(window)
    for i in range(nb):
        lo, hi = i * bar / window, (i + 1) * bar / window
        pa = [p for k, p in sa if lo <= k < hi]
        # margen de ±1 ventana en la versión para no penalizar el borde de compás
        pb = [p for k, p in sb if lo - 1 <= k < hi + 1]
        out.append(_ngram_recall(pa, pb, n))
    return out


def measure(orig: str, cand: str, tol: float = 0.25, window: float = 0.5,
            ngram: int = 2) -> Dict:
    """Todas las métricas de fidelidad entre `orig` y `cand`."""
    a, b = Piece(orig), Piece(cand)
    r: Dict = {"original": orig, "candidate": cand, "tol": tol, "window": window,
               "ngram": ngram}
    r["melody"] = melody_recall(a, b, window, ngram)
    r["onset"] = onset_recall(a, b, tol)
    r["onset_tight"] = onset_recall(a, b, 0.1)
    r["onset_loose"] = onset_recall(a, b, 0.5)
    r["rhythm"] = onset_recall(a, b, tol, use_pitch=False)
    r["pitch"] = pitch_recall(a, b)
    r["chroma"] = chroma_sim(a, b)
    r["length"] = (b.length / a.length) if a.length else None
    r["notes_ratio"] = (len(b.melodic) / len(a.melodic)) if a.melodic else None
    if a.programs:
        r["instr"] = len(a.programs & b.programs) / len(a.programs)
    else:
        r["instr"] = 1.0 if not b.programs or b.programs == {0} else 0.0
    ts_ok = a.ts == b.ts
    ks_ok = a.ks == b.ks
    r["ts_kept"], r["ks_kept"] = ts_ok, ks_ok
    r["meta"] = (int(ts_ok) + int(ks_ok)) / 2
    r["tempo_kept"] = (a.tempo is not None and b.tempo is not None
                       and abs(a.tempo - b.tempo) / a.tempo < 0.05)
    r["info"] = {"orig": {"notes": len(a.melodic), "bars": round(a.length / a.beats_per_bar, 1),
                          "programs": sorted(a.programs), "ts": a.ts, "ks": a.ks},
                 "cand": {"notes": len(b.melodic), "bars": round(b.length / b.beats_per_bar, 1),
                          "programs": sorted(b.programs), "ts": b.ts, "ks": b.ks}}
    r["score"] = overall(r)
    r["verdict"] = verdict(r)
    return r


def overall(r: Dict) -> Optional[float]:
    tot, w_sum = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = r.get(k)
        if v is not None:
            tot += w * v
            w_sum += w
    return tot / w_sum if w_sum else None


def verdict(r: Dict) -> str:
    m = r.get("melody")
    if m is None:
        return "indeterminado (melodía demasiado corta)"
    if m >= 0.85:
        return "fiel"
    if m >= 0.50:
        return "variación"
    if m >= 0.20:
        return "inspirada"
    return "otra pieza"


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def pct(v: Optional[float]) -> str:
    return "  —  " if v is None else f"{v * 100:5.1f}%"


def colour_for(v: Optional[float]) -> str:
    if v is None:
        return ""
    return _c("GRN") if v >= 0.8 else (_c("YEL") if v >= 0.4 else _c("RED"))


def print_compare(r: Dict, show_timeline: Optional[List[Optional[float]]] = None) -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}FIDELITY METER{R}\n{'═' * 78}")
    print(f"  original : {r['original']}\n  versión  : {r['candidate']}")
    i = r["info"]
    print(f"  {G}original: {i['orig']['notes']} notas · {i['orig']['bars']} compases · "
          f"programas {i['orig']['programs']}{R}")
    print(f"  {G}versión : {i['cand']['notes']} notas · {i['cand']['bars']} compases · "
          f"programas {i['cand']['programs']}{R}\n")

    def row(label, v, note=""):
        print(f"  {label:<10}{colour_for(v)}{pct(v)}{R}  {G}{note}{R}")

    row("melody", r["melody"], f"{r['ngram']}-gramas del skyline (ventana {r['window']} pulso)")
    row("rhythm", r["rhythm"], f"ataques presentes ±{r['tol']} pulso (sin altura)")
    print(f"  {'onset':<10}{colour_for(r['onset'])}{pct(r['onset'])}{R}  "
          f"{G}notas presentes ±{r['tol']} pulso   "
          f"[±0.1: {pct(r['onset_tight']).strip()} · ±0.5: {pct(r['onset_loose']).strip()}]{R}")
    row("pitch", r["pitch"], "alturas (débil: informativo)")
    row("chroma", r["chroma"], "perfil de clases de altura")
    row("instr", r["instr"], "instrumentos del original presentes")
    row("meta", r["meta"], f"compás {'✔' if r['ts_kept'] else '✘'} · armadura "
        f"{'✔' if r['ks_kept'] else '✘'} · tempo {'✔' if r['tempo_kept'] else '✘'}")
    ln = r["length"]
    print(f"  {'length':<10}{'  —  ' if ln is None else f'{ln:5.2f}x'}  "
          f"{G}duración versión/original · notas {r['notes_ratio']:.2f}x{R}"
          if r["notes_ratio"] is not None else "")
    print(f"\n  {B}score {pct(r['score']).strip()}{R}   veredicto: "
          f"{B}{r['verdict']}{R}")
    if show_timeline is not None:
        blocks = "▁▂▃▄▅▆▇█"
        line = "".join("·" if v is None else ("░" if v == 0 else blocks[min(7, int(v * 7.999))])
                       for v in show_timeline)
        print(f"\n  {G}melodía compás a compás (█=100 %, ░=0 %, ·=sin datos):{R}\n  {line}")
    print(f"{'═' * 78}\n")


def short(path: str) -> str:
    return os.path.relpath(path) if os.path.exists(path) else path


def print_table(rows: List[Dict], title: str, label_key: str = "label") -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}{title}{R}\n{'═' * 78}")
    print(f"  {G}{'#':>3}  {'score':>6} {'melody':>7} {'rhythm':>7} {'onset':>7} {'chroma':>7} "
          f"{'instr':>6} {'meta':>6}  veredicto / {label_key}{R}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>3}  {colour_for(r['score'])}{pct(r['score'])}{R} {pct(r['melody'])} "
              f"{pct(r['rhythm'])} {pct(r['onset'])} {pct(r['chroma'])} {pct(r['instr'])[:6]} "
              f"{pct(r['meta'])}  {r['verdict']:<10} {G}{r.get(label_key, '')}{R}")
    print(f"{'═' * 78}")


def aggregate(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in ("score", "melody", "rhythm", "onset", "pitch", "chroma", "instr", "meta", "length"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {"mean": statistics.mean(vals), "median": statistics.median(vals),
                      "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _need(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no existe el fichero: {path}")


def _dump(path: str, doc: Dict) -> None:
    with open(_ensure_parent(path), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  {_c('CYA')}JSON: {path}{_c('R')}")


def cmd_compare(a) -> int:
    _need(a.original)
    _need(a.candidate)
    r = measure(a.original, a.candidate, a.tol, a.window, a.ngram)
    tl = None
    if a.timeline:
        tl = timeline(Piece(a.original), Piece(a.candidate), a.window, a.ngram)
        r["timeline"] = tl
    print_compare(r, tl)
    if a.json:
        _dump(a.json, r)
    return 0


def _find_dirs(patterns: List[str]) -> List[str]:
    dirs: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or [p]
        dirs += [h.rstrip("/") for h in hits if os.path.isdir(h)]
    return list(dict.fromkeys(dirs))


def cmd_batch(a) -> int:
    dirs = _find_dirs(a.dirs)
    cands = a.cand or ["obra_final.mid"]
    rows: List[Dict] = []
    skipped = 0
    for d in dirs:
        o = os.path.join(d, a.orig)
        if not os.path.isfile(o):
            skipped += 1
            continue
        for cn in cands:
            c = os.path.join(d, cn)
            if not os.path.isfile(c):
                skipped += 1
                continue
            try:
                r = measure(o, c, a.tol, a.window, a.ngram)
            except Exception as e:  # noqa: BLE001
                print(f"  {_c('RED')}✗ {c}: {e}{_c('R')}", file=sys.stderr)
                skipped += 1
                continue
            r["label"] = f"{os.path.basename(d)}/{cn}"
            r["dir"], r["cand_name"] = d, cn
            rows.append(r)
    if not rows:
        print("[ERROR] ninguna pareja original/versión encontrada", file=sys.stderr)
        return 1
    show = rows if len(rows) <= a.top else sorted(rows, key=lambda r: r["score"] or 0)[:a.top]
    print_table(show, f"BATCH · {len(rows)} comparaciones" +
                ("" if show is rows else f" (las {a.top} peores)"))
    agg_by: Dict[str, Dict] = {}
    for cn in cands:
        sub = [r for r in rows if r["cand_name"] == cn]
        if sub:
            agg_by[cn] = aggregate(sub)
    print(f"\n  {_c('B')}Resumen por versión{_c('R')}  (tol ±{a.tol} pulso)")
    for cn, ag in agg_by.items():
        vc = Counter(r["verdict"] for r in rows if r["cand_name"] == cn)
        print(f"  {cn:<22} n={ag['score']['n']:<4} score μ={pct(ag['score']['mean']).strip()}  "
              f"melody μ={pct(ag['melody']['mean']).strip() if 'melody' in ag else '—'}  "
              f"onset μ={pct(ag['onset']['mean']).strip() if 'onset' in ag else '—'}  "
              f"· " + ", ".join(f"{k} {v}" for k, v in vc.most_common()))
    if skipped:
        print(f"  {_c('G')}({skipped} carpetas/ficheros omitidos por falta de original o versión){_c('R')}")
    print()
    if a.json:
        _dump(a.json, {"version": VERSION, "tol": a.tol, "summary": agg_by, "rows": rows})
    if a.csv:
        cols = ["label", "score", "melody", "rhythm", "onset", "onset_tight", "onset_loose",
                "pitch", "chroma", "instr", "meta", "length", "verdict"]
        with open(_ensure_parent(a.csv), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(k) if not isinstance(r.get(k), float) else round(r[k], 4)
                            for k in cols])
        print(f"  {_c('CYA')}CSV: {a.csv}{_c('R')}")
    if a.md:
        L = [f"# Fidelidad: original vs versiones", "",
             f"Tolerancia temporal ±{a.tol} pulso · ventana skyline {a.window} · {a.ngram}-gramas.", "",
             "| Versión | n | score | melody | rhythm | onset | chroma | instr | meta |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cn, ag in agg_by.items():
            g = lambda k: f"{ag[k]['mean'] * 100:.1f} %" if k in ag else "—"
            L.append(f"| `{cn}` | {ag['score']['n']} | {g('score')} | {g('melody')} | {g('rhythm')} | "
                     f"{g('onset')} | {g('chroma')} | {g('instr')} | {g('meta')} |")
        L += ["", "## Detalle", "", "| Pieza | versión | score | melody | onset | veredicto |",
              "|---|---|---:|---:|---:|---|"]
        for r in rows:
            L.append(f"| {os.path.basename(r['dir'])} | `{r['cand_name']}` | {pct(r['score']).strip()} | "
                     f"{pct(r['melody']).strip()} | {pct(r['onset']).strip()} | {r['verdict']} |")
        with open(_ensure_parent(a.md), "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"  {_c('CYA')}Markdown: {a.md}{_c('R')}")
    return 0


def cmd_rank(a) -> int:
    _need(a.original)
    files: List[str] = []
    for p in a.candidates:
        files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    files = [f for f in dict.fromkeys(files) if os.path.isfile(f)]
    if not files:
        print("[ERROR] no se encontró ninguna versión", file=sys.stderr)
        return 1
    rows = []
    for f in files:
        r = measure(a.original, f, a.tol, a.window, a.ngram)
        r["label"] = short(f)
        rows.append(r)
    rows.sort(key=lambda r: (r["score"] or 0, r["melody"] or 0), reverse=True)
    print_table(rows, f"RANKING por fidelidad a {short(a.original)}")
    print(f"  {_c('B')}Más fiel:{_c('R')} {rows[0]['candidate']}  (score {pct(rows[0]['score']).strip()})\n")
    if a.json:
        _dump(a.json, {"version": VERSION, "original": a.original, "winner": rows[0]["candidate"],
                       "rows": rows})
    return 0


def cmd_gate(a) -> int:
    _need(a.original)
    _need(a.candidate)
    r = measure(a.original, a.candidate, a.tol, a.window, a.ngram)
    checks: List[Tuple[str, bool, str]] = []

    def need_min(key, thr):
        if thr is None:
            return
        v = r.get(key)
        ok = v is not None and v >= thr
        checks.append((f"{key} ≥ {thr:g}", ok, "—" if v is None else f"{v:.3f}"))

    need_min("melody", a.min_melody)
    need_min("rhythm", a.min_rhythm)
    need_min("chroma", a.min_chroma)
    need_min("score", a.min_score)
    if a.require_meta:
        checks.append(("compás y armadura conservados", bool(r["ts_kept"] and r["ks_kept"]),
                       f"ts {'✔' if r['ts_kept'] else '✘'} ks {'✔' if r['ks_kept'] else '✘'}"))
    if a.require_instr:
        checks.append(("instrumentación conservada", (r["instr"] or 0) >= 1.0, f"{r['instr']:.2f}"))
    if a.length_range:
        lo, hi = a.length_range
        ln = r["length"]
        checks.append((f"longitud {lo:g}–{hi:g}x", ln is not None and lo <= ln <= hi,
                       "—" if ln is None else f"{ln:.2f}x"))
    if not checks:
        print("[ERROR] indica al menos un umbral (--min-melody, --min-score, --require-meta…)",
              file=sys.stderr)
        return 1
    ok_all = all(ok for _, ok, _ in checks)
    print(f"\n  {_c('B')}GATE{_c('R')}  {short(a.candidate)}  vs  {short(a.original)}")
    for name, ok, val in checks:
        print(f"    {_c('GRN') + '✔' if ok else _c('RED') + '✘'}{_c('R')} {name:<34} {_c('G')}{val}{_c('R')}")
    print(f"  {_c('GRN') + 'PASA' if ok_all else _c('RED') + 'NO PASA'}{_c('R')}\n")
    if a.json:
        r["gate"] = {"passed": ok_all, "checks": [{"check": n, "ok": o, "value": v} for n, o, v in checks]}
        _dump(a.json, r)
    return 0 if ok_all else 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fidelity_meter.py",
        description="Mide cuánto se parece una versión a su original (compare / batch / rank / gate).")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    def common(p):
        p.add_argument("--tol", type=float, default=0.25)
        p.add_argument("--window", type=float, default=0.5)
        p.add_argument("--ngram", type=int, default=2)
        p.add_argument("--json", default=None)
        p.add_argument("--no-color", action="store_true")

    c = sub.add_parser("compare", help="original vs versión")
    c.add_argument("original")
    c.add_argument("candidate")
    c.add_argument("--timeline", action="store_true")
    common(c)
    c.set_defaults(fn=cmd_compare)

    b = sub.add_parser("batch", help="muchas carpetas")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--cand", action="append", default=None)
    b.add_argument("--orig", default="original.mid")
    b.add_argument("--md", default=None)
    b.add_argument("--csv", default=None)
    b.add_argument("--top", type=int, default=25, help="filas a mostrar (las peores si hay más)")
    common(b)
    b.set_defaults(fn=cmd_batch)

    r = sub.add_parser("rank", help="ordena versiones por fidelidad")
    r.add_argument("original")
    r.add_argument("candidates", nargs="+")
    common(r)
    r.set_defaults(fn=cmd_rank)

    g = sub.add_parser("gate", help="puerta de calidad (rc=3 si no pasa)")
    g.add_argument("original")
    g.add_argument("candidate")
    g.add_argument("--min-melody", type=float, default=None)
    g.add_argument("--min-rhythm", type=float, default=None)
    g.add_argument("--min-chroma", type=float, default=None)
    g.add_argument("--min-score", type=float, default=None)
    g.add_argument("--require-meta", action="store_true")
    g.add_argument("--require-instr", action="store_true")
    g.add_argument("--length-range", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    common(g)
    g.set_defaults(fn=cmd_gate)
    return ap


def main(argv=None) -> int:
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[interrumpido]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
